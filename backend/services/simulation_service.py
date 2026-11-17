"""
Оркестрация на пълния цикъл (Етап 3):

    SimulationRequest
      -> MarketDataService.get_market_data()          (Договор 1, кеш в PostgreSQL)
      -> AI адаптер:  generate_ai_predictions()       (Договор 2)
      -> Backtester адаптер: run_backtest()           (Договор 3 – mock_backtest_response.json)
      -> SimulationRun + всички AuditLog в ЕДНА транзакция (session.begin())
      -> SimulationResponse

Адаптери към модулите на съотборниците
--------------------------------------
* AI (Любомир):  `ai_engine.predictor.generate_ai_predictions(df_ohlcv) -> dict`
  точно по contracts/CONTRACTS.md §3. Ако функцията приема `random_state`, той се подава.
* Backtester (Дарио): очаква се
      `backtester.simulator.run_backtest(signals_df: pd.DataFrame, params: dict) -> dict`
  с ключове `portfolio_metrics`, `equity_curve`, `audit_logs` (формат от
  contracts/mock_backtest_response.json). `params` съдържа request_params +
  `commission` и `annualization_factor` (365 крипто / 252 акции).

Ако модулът или функцията липсват (ModuleNotFoundError за самия модул / липсващ
атрибут), се ползва детерминираният fallback от `fallback_engine.py`. Грешки ВЪТРЕ в
реален модул не се маскират – те се пропагират.

Изходът на всеки двигател се валидира с Pydantic срещу договора, преди да се запише.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from backend.config import ML_METRIC_KEYS, SIGNALS_COLUMNS, Settings, get_settings
from backend.database import get_session_factory, transactional_session
from backend.exceptions import BackendError
from backend.models import AuditLog, AuditStatus, SimulationRun
from backend.schemas import (
    AuditLogEntry,
    EngineInfo,
    EquityPoint,
    MLMetrics,
    PortfolioMetrics,
    SimulationRequest,
    SimulationResponse,
)
from backend.services.data_ingestion import MarketDataService
from backend.services.fallback_engine import (
    fallback_generate_ai_predictions,
    fallback_run_backtest,
)

logger = logging.getLogger(__name__)

AIFunc = Callable[[pd.DataFrame, int], dict[str, Any]]
BacktestFunc = Callable[[pd.DataFrame, dict[str, Any]], dict[str, Any]]


class EngineContractError(BackendError):
    """Модул от /ai_engine или /backtester върна изход, който нарушава договора."""

    status_code = 500


class SimulationNotFoundError(BackendError):
    status_code = 404


# ==============================================================================
# Адаптери
# ==============================================================================


def _load_attr(module_name: str, attr: str) -> Callable[..., Any] | None:
    """Връща функцията, ако модулът и атрибутът съществуват; иначе None."""
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        # Само липсата на САМИЯ модул (или пакета му) води до fallback
        if exc.name and module_name.startswith(exc.name):
            return None
        raise
    fn = getattr(module, attr, None)
    return fn if callable(fn) else None


@dataclass(frozen=True)
class EngineAdapters:
    ai: AIFunc
    backtest: BacktestFunc
    info: EngineInfo


def resolve_engines() -> EngineAdapters:
    real_ai = _load_attr("ai_engine.predictor", "generate_ai_predictions")
    real_bt = _load_attr("backtester.simulator", "run_backtest")

    if real_ai is not None:
        accepts_seed = "random_state" in inspect.signature(real_ai).parameters

        def ai(df: pd.DataFrame, random_state: int) -> dict[str, Any]:
            return real_ai(df, random_state=random_state) if accepts_seed else real_ai(df)
    else:
        ai = fallback_generate_ai_predictions

    info = EngineInfo(ai_engine="ai_engine" if real_ai else "fallback",
                      backtester="backtester" if real_bt else "fallback")
    if real_ai is None or real_bt is None:
        logger.warning("Използва се fallback двигател: %s", info.model_dump())
    return EngineAdapters(ai=ai, backtest=real_bt or fallback_run_backtest, info=info)


# ==============================================================================
# Проверка на договорите
# ==============================================================================


def _check_ai_output(out: Any) -> tuple[pd.DataFrame, dict[str, float]]:
    if not isinstance(out, dict) or "signals_df" not in out or "ml_metrics" not in out:
        raise EngineContractError("AI изходът трябва да е dict със signals_df и ml_metrics")
    signals = out["signals_df"]
    if not isinstance(signals, pd.DataFrame):
        raise EngineContractError("signals_df трябва да е pandas.DataFrame")
    if isinstance(signals.index, pd.DatetimeIndex) and "date" not in signals.columns:
        signals = signals.rename_axis("date").reset_index()
    missing = [c for c in SIGNALS_COLUMNS if c not in signals.columns]
    if missing:
        raise EngineContractError(f"signals_df няма колони {missing}", missing=missing)
    signals = signals.loc[:, list(SIGNALS_COLUMNS)].copy()
    signals["date"] = pd.to_datetime(signals["date"]).dt.strftime("%Y-%m-%d")
    if signals.empty:
        raise EngineContractError("signals_df е празен (твърде кратък период?)")
    if signals["prob_up"].isna().any() or not signals["prob_up"].between(0, 1).all():
        raise EngineContractError("prob_up трябва да е в [0, 1] без NaN")
    if not signals["date"].is_monotonic_increasing or signals["date"].duplicated().any():
        raise EngineContractError("signals_df трябва да е сортиран по дата без дубликати")

    metrics = out["ml_metrics"]
    try:
        MLMetrics.model_validate({k: metrics.get(k) for k in ML_METRIC_KEYS})
    except (AttributeError, ValidationError) as exc:
        raise EngineContractError(f"Невалидни ml_metrics: {exc}") from exc
    return signals.reset_index(drop=True), {k: float(metrics[k]) for k in ML_METRIC_KEYS}


def _check_backtest_output(out: Any) -> dict[str, Any]:
    if not isinstance(out, dict):
        raise EngineContractError("Изходът на backtester трябва да е dict")
    try:
        PortfolioMetrics.model_validate(out.get("portfolio_metrics"))
        for point in out.get("equity_curve") or []:
            EquityPoint.model_validate(point)
        for entry in out.get("audit_logs") or []:
            AuditLogEntry.model_validate({k: v for k, v in entry.items() if k != "phase"})
    except ValidationError as exc:
        raise EngineContractError(f"Изходът на backtester нарушава договора: {exc}") from exc
    if not out.get("equity_curve"):
        raise EngineContractError("equity_curve е празна")
    return out


# ==============================================================================
# Сервиз
# ==============================================================================


class SimulationService:
    def __init__(self, session_factory: sessionmaker[Session] | None = None,
                 market_data: MarketDataService | None = None,
                 engines: EngineAdapters | None = None,
                 settings: Settings | None = None) -> None:
        self._session_factory = session_factory
        self._market_data = market_data
        self._engines = engines
        self._settings = settings or get_settings()

    @property
    def session_factory(self) -> sessionmaker[Session]:
        return self._session_factory or get_session_factory()

    @property
    def market_data(self) -> MarketDataService:
        return self._market_data or MarketDataService(self.session_factory)

    @property
    def engines(self) -> EngineAdapters:
        if self._engines is None:
            self._engines = resolve_engines()
        return self._engines

    # --- изпълнение -----------------------------------------------------------

    def build_payload(self, req: SimulationRequest) -> dict[str, Any]:
        """Чисто изчисление (без запис): данни -> AI -> backtest -> dict по договора."""
        df = self.market_data.get_market_data(req.ticker, req.start_date, req.end_date)
        signals, ml_metrics = _check_ai_output(self.engines.ai(df, req.random_state))

        params = {
            **req.request_params(),
            "commission": self._settings.default_commission,
            "annualization_factor": Settings.annualization_factor(req.ticker),
        }
        bt = _check_backtest_output(self.engines.backtest(signals, params))
        return {
            "request_params": req.request_params(),
            "ml_metrics": ml_metrics,
            "portfolio_metrics": bt["portfolio_metrics"],
            "equity_curve": bt["equity_curve"],
            "audit_logs": bt["audit_logs"],
        }

    def run(self, req: SimulationRequest) -> SimulationResponse:
        t0 = time.perf_counter()
        payload = self.build_payload(req)
        commission = self._settings.default_commission

        run = SimulationRun.from_contract(payload, commission=commission,
                                          random_state=req.random_state)
        run.id = uuid.uuid4()
        run.execution_time_ms = int(round((time.perf_counter() - t0) * 1000))

        # ACID: SimulationRun и ВСИЧКИ AuditLog в една транзакция (session.begin()).
        # Грешка при който и да е лог -> ROLLBACK на всичко, нищо не остава в БД.
        with transactional_session(self.session_factory) as s:
            s.add(run)
            s.flush()
            created_at = run.created_at  # server_default – зарежда се в транзакцията
            response = self._to_response(run, created_at)

        total_ms = int(round((time.perf_counter() - t0) * 1000))
        if total_ms > self._settings.max_pipeline_seconds_per_asset * 1000:
            logger.warning("%s: симулацията отне %d ms (> лимита от %.0f s)", req.ticker,
                           total_ms, self._settings.max_pipeline_seconds_per_asset)
        response.engines = self.engines.info
        logger.info("Симулация %s (%s) записана: %d лога, %d ms", run.id, req.ticker,
                    len(run.audit_logs), total_ms)
        return response

    # --- четене ---------------------------------------------------------------

    def get(self, simulation_id: uuid.UUID) -> SimulationResponse:
        with self.session_factory() as s:
            run = s.get(SimulationRun, simulation_id)
            if run is None:
                raise SimulationNotFoundError(f"Симулация {simulation_id} не съществува",
                                              simulation_id=str(simulation_id))
            return self._to_response(run, run.created_at)

    def audit_logs(self, simulation_id: uuid.UUID,
                   status: str | None = None) -> list[dict[str, Any]]:
        with self.session_factory() as s:
            if s.get(SimulationRun, simulation_id) is None:
                raise SimulationNotFoundError(f"Симулация {simulation_id} не съществува",
                                              simulation_id=str(simulation_id))
            stmt = (select(AuditLog).where(AuditLog.simulation_run_id == simulation_id)
                    .order_by(AuditLog.sequence))
            if status is not None:
                stmt = stmt.where(AuditLog.status == AuditStatus(status))
            return [log.to_contract() for log in s.scalars(stmt)]

    @staticmethod
    def _to_response(run: SimulationRun, created_at: Any) -> SimulationResponse:
        return SimulationResponse.model_validate({
            "simulation_id": run.id,
            "created_at": created_at,
            "execution_time_ms": run.execution_time_ms,
            "random_state": run.random_state,
            "commission": run.commission,
            **run.to_contract(),
        })


__all__ = [
    "EngineAdapters",
    "EngineContractError",
    "SimulationNotFoundError",
    "SimulationService",
    "resolve_engines",
]
