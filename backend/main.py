"""
FastAPI приложение на /backend (Етап 3).

Стартиране:
    uvicorn backend.main:app --reload              # http://127.0.0.1:8000/docs

Ендпойнти (префикс /api/v1):
    GET  /health
    GET  /tickers                         (+ псевдоним GET /assets от архитектурния документ)
    POST /market-data/sync
    GET  /market-data/{ticker}
    POST /simulations/run                 (+ псевдоним POST /backtest)
    GET  /simulations/{simulation_id}
    GET  /simulations/{simulation_id}/audit-logs

Грешки: BackendError -> собственият status_code (DataValidationError / InvalidTickerError /
InvalidDateRangeError -> 422, ExternalDataFetchError -> 502, липсваща симулация -> 404),
нарушено DB ограничение -> 500 (транзакцията е върната), недостъпна БД -> 503.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session, sessionmaker

from backend.config import (
    CRYPTO_TICKERS,
    DATA_END_DATE,
    DATA_START_DATE,
    STOCK_TICKERS,
    TICKER_ASSET_CLASS,
    Settings,
    get_settings,
)
from backend.database import check_connection, get_session_factory, init_db
from backend.exceptions import BackendError
from backend.schemas import (
    AssetInfo,
    AuditLogsResponse,
    Candle,
    ErrorResponse,
    MarketDataResponse,
    MarketDataSyncRequest,
    MarketDataSyncResponse,
    SimulationRequest,
    SimulationResponse,
    SyncResult,
    TickersResponse,
)
from backend.services.data_ingestion import MarketDataService, validate_ticker
from backend.services.simulation_service import SimulationService

logger = logging.getLogger(__name__)

ASSET_NAMES: dict[str, str] = {
    "BTC-USD": "Bitcoin", "ETH-USD": "Ethereum", "BNB-USD": "BNB", "XRP-USD": "XRP",
    "LTC-USD": "Litecoin", "ADA-USD": "Cardano", "DOT-USD": "Polkadot",
    "LINK-USD": "Chainlink", "BCH-USD": "Bitcoin Cash", "XLM-USD": "Stellar",
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corp.", "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc. (Class A)", "META": "Meta Platforms Inc.", "TSLA": "Tesla Inc.",
    "BRK-B": "Berkshire Hathaway Inc. (Class B)", "JPM": "JPMorgan Chase & Co.",
    "JNJ": "Johnson & Johnson", "V": "Visa Inc.",
}

ERROR_RESPONSES = {
    404: {"model": ErrorResponse, "description": "Не е намерено"},
    422: {"model": ErrorResponse, "description": "Невалидни параметри или данни "
                                                  "(напр. > 3 последователни липсващи дни)"},
    502: {"model": ErrorResponse, "description": "Yahoo Finance е недостъпен"},
}


# ==============================================================================
# Dependencies (подменят се в тестовете чрез app.dependency_overrides)
# ==============================================================================


def get_db_session_factory() -> sessionmaker[Session]:
    return get_session_factory()


def get_market_data_service(
    sf: sessionmaker[Session] = Depends(get_db_session_factory),
) -> MarketDataService:
    return MarketDataService(sf)


def get_simulation_service(
    sf: sessionmaker[Session] = Depends(get_db_session_factory),
    market_data: MarketDataService = Depends(get_market_data_service),
) -> SimulationService:
    return SimulationService(sf, market_data)


# ==============================================================================
# Приложение
# ==============================================================================


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        init_db()  # идемпотентно; в продукция – Alembic миграции
    except Exception:  # noqa: BLE001 – API-то стартира и без БД, /health показва статуса
        logger.exception("init_db() неуспешен – провери връзката с PostgreSQL")
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.3.0",
        description="AI-Powered Investment Platform for Simulated Algorithmic Trading",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Execution-Time-Ms"],
    )
    _register_exception_handlers(app)
    app.include_router(router, prefix=settings.api_prefix)
    return app


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(BackendError)
    async def backend_error_handler(_: Request, exc: BackendError) -> JSONResponse:
        if exc.status_code >= 500:
            logger.error("%s: %s", type(exc).__name__, exc.message)
        return JSONResponse(status_code=exc.status_code, content=exc.to_dict())

    @app.exception_handler(IntegrityError)
    async def integrity_error_handler(_: Request, exc: IntegrityError) -> JSONResponse:
        logger.error("DB ограничение нарушено, транзакцията е върната: %s", exc.orig)
        return JSONResponse(status_code=500, content={
            "error": "PersistenceError",
            "message": "Записът е отхвърлен от базата данни; транзакцията е върната "
                       "(не е записан нито SimulationRun, нито AuditLog).",
        })

    @app.exception_handler(OperationalError)
    async def db_unavailable_handler(_: Request, exc: OperationalError) -> JSONResponse:
        logger.error("Базата данни е недостъпна: %s", exc.orig)
        return JSONResponse(status_code=503, content={
            "error": "DatabaseUnavailable", "message": "PostgreSQL е недостъпен.",
        })


router = APIRouter()


# --- Служебни -----------------------------------------------------------------


@router.get("/health", tags=["system"])
def health(sf: sessionmaker[Session] = Depends(get_db_session_factory)) -> dict:
    db_ok = check_connection(sf.kw["bind"])
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}


# --- Тикери -------------------------------------------------------------------


def _asset(ticker: str) -> AssetInfo:
    asset_class = TICKER_ASSET_CLASS[ticker]
    return AssetInfo(ticker=ticker, name=ASSET_NAMES[ticker], asset_type=asset_class,
                     calendar_days=Settings.annualization_factor(ticker))


@router.get("/tickers", response_model=TickersResponse, tags=["assets"])
def list_tickers() -> TickersResponse:
    """20-те фиксирани актива: 10 крипто и 10 акции от S&P 500."""
    crypto = [_asset(t) for t in CRYPTO_TICKERS]
    stocks = [_asset(t) for t in STOCK_TICKERS]
    return TickersResponse(crypto=crypto, stocks=stocks, total=len(crypto) + len(stocks))


@router.get("/assets", response_model=list[AssetInfo], tags=["assets"])
def list_assets() -> list[AssetInfo]:
    """Плосък списък {ticker, name, asset_type, calendar_days} (архитектурен документ, т. 5.2)."""
    return [_asset(t) for t in (*CRYPTO_TICKERS, *STOCK_TICKERS)]


# --- Пазарни данни ------------------------------------------------------------


def _sync_one(service: MarketDataService, ticker: str, force: bool) -> SyncResult:
    if not force and service.is_cached(ticker):
        df = service.get_market_data(ticker)
        status_ = "cached"
    else:
        df = service.sync_ticker(ticker)
        status_ = "synced"
    return SyncResult(ticker=ticker, status=status_, rows=len(df),
                      first_date=df["date"].iloc[0], last_date=df["date"].iloc[-1])


@router.post("/market-data/sync", response_model=MarketDataSyncResponse,
             responses=ERROR_RESPONSES, tags=["market-data"])
def sync_market_data(
    body: MarketDataSyncRequest | None = None,
    service: MarketDataService = Depends(get_market_data_service),
) -> MarketDataSyncResponse:
    """
    Проверява/синхронизира кеша в PostgreSQL. С `ticker` – само този тикер (грешките
    се връщат като HTTP грешка); без `ticker` – всичките 20, като грешка при един не
    спира останалите (всеки тикер е в отделна транзакция).
    """
    body = body or MarketDataSyncRequest()
    if body.ticker is not None:
        results = [_sync_one(service, body.ticker, body.force_refresh)]
    else:
        results = []
        for ticker in (*CRYPTO_TICKERS, *STOCK_TICKERS):
            try:
                results.append(_sync_one(service, ticker, body.force_refresh))
            except BackendError as exc:
                results.append(SyncResult(ticker=ticker, status="failed",
                                          error=f"{type(exc).__name__}: {exc.message}"))
    failed = sum(r.status == "failed" for r in results)
    return MarketDataSyncResponse(results=results, succeeded=len(results) - failed,
                                  failed=failed)


@router.get("/market-data/{ticker}", response_model=MarketDataResponse,
            responses=ERROR_RESPONSES, tags=["market-data"])
def get_market_data(
    ticker: str,
    start_date: date = Query(DATA_START_DATE),
    end_date: date = Query(DATA_END_DATE),
    service: MarketDataService = Depends(get_market_data_service),
) -> MarketDataResponse:
    """Дневни OHLCV свещи за Lightweight Charts (`time` = 'YYYY-MM-DD')."""
    ticker = validate_ticker(ticker.strip().upper())
    df = service.get_market_data(ticker, start_date, end_date)
    candles = [Candle(time=r.date, open=r.open, high=r.high, low=r.low, close=r.close,
                      volume=r.volume) for r in df.itertuples(index=False)]
    return MarketDataResponse(ticker=ticker, asset_type=TICKER_ASSET_CLASS[ticker],
                              start_date=candles[0].time, end_date=candles[-1].time,
                              count=len(candles), candles=candles)


# --- Симулации ----------------------------------------------------------------


@router.post("/simulations/run", response_model=SimulationResponse,
             status_code=status.HTTP_201_CREATED, responses=ERROR_RESPONSES,
             tags=["simulations"])
def run_simulation(
    body: SimulationRequest,
    service: SimulationService = Depends(get_simulation_service),
) -> JSONResponse:
    """
    Пълен цикъл: данни -> AI прогноза -> двуфазна симулация -> атомарен запис
    (SimulationRun + всички AuditLog в една транзакция) -> отчет по договора.
    """
    result = service.run(body)
    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content=result.model_dump(mode="json"),
        headers={"X-Execution-Time-Ms": str(result.execution_time_ms)},
    )


@router.post("/backtest", response_model=SimulationResponse,
             status_code=status.HTTP_201_CREATED, responses=ERROR_RESPONSES,
             tags=["simulations"])
def run_backtest_alias(
    body: SimulationRequest,
    service: SimulationService = Depends(get_simulation_service),
) -> JSONResponse:
    """Псевдоним на POST /simulations/run (архитектурен документ, т. 5.2)."""
    return run_simulation(body, service)


@router.get("/simulations/{simulation_id}", response_model=SimulationResponse,
            responses=ERROR_RESPONSES, tags=["simulations"])
def get_simulation(
    simulation_id: uuid.UUID,
    service: SimulationService = Depends(get_simulation_service),
) -> SimulationResponse:
    return service.get(simulation_id)


@router.get("/simulations/{simulation_id}/audit-logs", response_model=AuditLogsResponse,
            responses=ERROR_RESPONSES, tags=["simulations"])
def get_simulation_audit_logs(
    simulation_id: uuid.UUID,
    status_filter: Literal["EXECUTED", "REJECTED_BY_RISK"] | None = Query(None, alias="status"),
    service: SimulationService = Depends(get_simulation_service),
) -> AuditLogsResponse:
    logs = service.audit_logs(simulation_id, status_filter)
    return AuditLogsResponse(simulation_id=simulation_id, count=len(logs), audit_logs=logs)


app = create_app()

__all__ = ["app", "create_app", "get_db_session_factory", "get_market_data_service",
           "get_simulation_service"]
