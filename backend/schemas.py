"""
Pydantic v2 схеми на REST API (Етап 3).

Изходът на симулацията следва 1:1 `contracts/mock_backtest_response.json`
(request_params, ml_metrics, portfolio_metrics, equity_curve, audit_logs), допълнен
с технически полета на ниво отговор: `simulation_id`, `created_at`,
`execution_time_ms`, `random_state`, `commission`, `engines`.

Единици: `max_position_pct` и `max_drawdown_limit` са дроби (0.25 == 25 %);
метриките с наставка `_pct` са в проценти (42.5 == 42.5 %).
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.config import (
    ALL_TICKERS,
    CRYPTO_TICKERS,
    DATA_END_DATE,
    DATA_START_DATE,
    STOCK_TICKERS,
    AssetClass,
)

# Правило на проекта: максимум 25 % от капитала в една позиция
MAX_POSITION_PCT_CAP: float = 0.25

ActionLiteral = Literal["BUY", "SELL", "FORCE_SELL", "FORCE_SELL_AT_OPEN"]
StatusLiteral = Literal["EXECUTED", "REJECTED_BY_RISK"]


def _check_ticker(value: str) -> str:
    ticker = value.strip().upper()
    if ticker not in ALL_TICKERS:
        raise ValueError(f"Неподдържан тикер '{value}'. Позволени: {', '.join(ALL_TICKERS)}")
    return ticker


def _check_period(start: date, end: date) -> None:
    if start < DATA_START_DATE or end > DATA_END_DATE:
        raise ValueError(
            f"Периодът трябва да е в {DATA_START_DATE} – {DATA_END_DATE} "
            f"(получено {start} – {end})"
        )
    if not start < end:
        raise ValueError(f"start_date ({start}) трябва да е преди end_date ({end})")


# ==============================================================================
# Вход
# ==============================================================================


class SimulationRequest(BaseModel):
    """Тяло на `POST /api/v1/simulations/run` (и псевдонима `POST /api/v1/backtest`)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True, json_schema_extra={
        "example": {
            "ticker": "BTC-USD", "start_date": "2021-01-01", "end_date": "2025-12-31",
            "initial_capital": 10000.0, "t_buy": 0.60, "t_sell": 0.45,
            "max_position_pct": 0.25, "max_drawdown_limit": 0.15, "random_state": 42,
        }
    })

    ticker: str = Field(description="Един от 20-те фиксирани тикера")
    start_date: date = Field(default=DATA_START_DATE)
    end_date: date = Field(default=DATA_END_DATE)
    initial_capital: float = Field(default=10_000.0, gt=0.0, le=1e12)
    t_buy: float = Field(default=0.60, ge=0.50, le=0.99)
    t_sell: float = Field(default=0.45, ge=0.01, le=0.50)
    max_position_pct: float = Field(
        default=0.25, gt=0.0, le=MAX_POSITION_PCT_CAP,
        validation_alias=AliasChoices("max_position_pct", "max_exposure"),
        description="Максимална експозиция (дроб, <= 0.25). Приема се и като `max_exposure`.",
    )
    max_drawdown_limit: float = Field(default=0.15, ge=0.01, le=0.99)
    random_state: int = Field(default=42, ge=0, le=2**32 - 1)

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, v: str) -> str:
        return _check_ticker(v)

    @model_validator(mode="after")
    def _cross_field_rules(self) -> "SimulationRequest":
        _check_period(self.start_date, self.end_date)
        if not self.t_sell < self.t_buy:
            raise ValueError(f"t_sell ({self.t_sell}) трябва да е строго < t_buy ({self.t_buy})")
        return self

    def request_params(self) -> dict:
        """`request_params` точно по договора (без random_state)."""
        return {
            "ticker": self.ticker,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "t_buy": self.t_buy,
            "t_sell": self.t_sell,
            "max_position_pct": self.max_position_pct,
            "max_drawdown_limit": self.max_drawdown_limit,
        }


class MarketDataSyncRequest(BaseModel):
    """Тяло на `POST /api/v1/market-data/sync`; без `ticker` -> всички 20."""

    model_config = ConfigDict(extra="forbid")

    ticker: str | None = None
    force_refresh: bool = False

    @field_validator("ticker")
    @classmethod
    def _ticker(cls, v: str | None) -> str | None:
        return None if v is None else _check_ticker(v)


# ==============================================================================
# Изход – симулация (contracts/mock_backtest_response.json)
# ==============================================================================


class _Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RequestParams(_Out):
    ticker: str
    start_date: str
    end_date: str
    initial_capital: float
    t_buy: float
    t_sell: float
    max_position_pct: float
    max_drawdown_limit: float


class MLMetrics(_Out):
    accuracy: float | None = Field(ge=0.0, le=1.0)
    precision: float | None = Field(ge=0.0, le=1.0)
    recall: float | None = Field(ge=0.0, le=1.0)
    f1_score: float | None = Field(ge=0.0, le=1.0)


class BuyAndHoldMetrics(_Out):
    final_equity: float | None
    net_return_pct: float | None
    max_drawdown_pct: float | None
    sharpe_ratio: float | None


class AIStrategyMetrics(BuyAndHoldMetrics):
    win_rate_pct: float | None
    total_trades: int | None
    circuit_breaker_triggered: bool | None


class PortfolioMetrics(_Out):
    ai_strategy: AIStrategyMetrics
    buy_and_hold: BuyAndHoldMetrics


class EquityPoint(_Out):
    date: str
    close_price: float
    ai_equity_close: float
    bnh_equity_close: float
    drawdown_close_pct: float


class AuditLogEntry(_Out):
    signal_date: str
    execution_date: str | None
    prob_up: float | None
    action: ActionLiteral
    status: StatusLiteral
    execution_price: float | None
    units: float
    fee_paid: float
    reason: str


class EngineInfo(_Out):
    """Кой двигател е изпълнил стъпката: реалният модул или fallback адаптерът."""

    ai_engine: Literal["ai_engine", "fallback"]
    backtester: Literal["backtester", "fallback"]


class SimulationResponse(_Out):
    simulation_id: uuid.UUID
    created_at: datetime | None = None
    execution_time_ms: int | None = Field(default=None, ge=0)
    random_state: int
    commission: float
    engines: EngineInfo | None = None

    request_params: RequestParams
    ml_metrics: MLMetrics
    portfolio_metrics: PortfolioMetrics
    equity_curve: list[EquityPoint]
    audit_logs: list[AuditLogEntry]


class AuditLogsResponse(_Out):
    simulation_id: uuid.UUID
    count: int
    audit_logs: list[AuditLogEntry]


# ==============================================================================
# Изход – тикери и пазарни данни
# ==============================================================================


class AssetInfo(_Out):
    ticker: str
    name: str
    asset_type: AssetClass
    calendar_days: int


class TickersResponse(_Out):
    crypto: list[AssetInfo]
    stocks: list[AssetInfo]
    total: int


class Candle(_Out):
    """Свещ във формата на Lightweight Charts (`time` е 'YYYY-MM-DD')."""

    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class MarketDataResponse(_Out):
    ticker: str
    asset_type: AssetClass
    start_date: str
    end_date: str
    count: int
    candles: list[Candle]


class SyncResult(_Out):
    ticker: str
    status: Literal["cached", "synced", "failed"]
    rows: int | None = None
    first_date: str | None = None
    last_date: str | None = None
    error: str | None = None


class MarketDataSyncResponse(_Out):
    results: list[SyncResult]
    succeeded: int
    failed: int


class ErrorResponse(_Out):
    error: str
    message: str
    ticker: str | None = None
    details: dict | list | None = None


__all__ = [
    "AIStrategyMetrics",
    "AssetInfo",
    "AuditLogEntry",
    "AuditLogsResponse",
    "BuyAndHoldMetrics",
    "Candle",
    "CRYPTO_TICKERS",
    "EngineInfo",
    "EquityPoint",
    "ErrorResponse",
    "MAX_POSITION_PCT_CAP",
    "MLMetrics",
    "MarketDataResponse",
    "MarketDataSyncRequest",
    "MarketDataSyncResponse",
    "PortfolioMetrics",
    "RequestParams",
    "STOCK_TICKERS",
    "SimulationRequest",
    "SimulationResponse",
    "SyncResult",
    "TickersResponse",
]
