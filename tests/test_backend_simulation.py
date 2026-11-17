"""Етап 3 – Pydantic схеми, fallback двигател, адаптери и ACID запис на симулацията."""

from __future__ import annotations

import sys
import types
from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from backend.models import AuditLog, SimulationRun
from backend.schemas import SimulationRequest
from backend.services.data_ingestion import MarketDataService
from backend.services.fallback_engine import (
    fallback_generate_ai_predictions,
    fallback_run_backtest,
)
from backend.services.simulation_service import (
    EngineAdapters,
    EngineContractError,
    SimulationNotFoundError,
    SimulationService,
    resolve_engines,
)
from backend.schemas import EngineInfo

# ------------------------------------------------------------------------------
# Синтетични данни
# ------------------------------------------------------------------------------


def random_walk(start: str = "2021-01-01", end: str = "2025-12-31", seed: int = 0,
                freq: str = "D") -> pd.DataFrame:
    """Детерминиран, OHLC-консистентен геометричен random walk."""
    idx = pd.date_range(start, end, freq=freq)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.03, len(idx))))
    open_ = np.concatenate([[100.0], close[:-1]]) * np.exp(rng.normal(0, 0.01, len(idx)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.01, len(idx))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.01, len(idx))))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                         "volume": rng.uniform(1e6, 1e7, len(idx))},
                        index=pd.DatetimeIndex(idx, name="date"))


class FakeDownloader:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame, self.calls = frame, 0

    def __call__(self, ticker, start, end):
        self.calls += 1
        return self.frame.copy()


def signals(rows: list[tuple[float, float, float]], start: str = "2024-01-01") -> pd.DataFrame:
    """rows = [(open, close, prob_up), ...] -> signals_df по Договор 2."""
    dates = pd.date_range(start, periods=len(rows), freq="D").strftime("%Y-%m-%d")
    o = np.array([r[0] for r in rows])
    c = np.array([r[1] for r in rows])
    return pd.DataFrame({"date": dates, "open": o, "high": np.maximum(o, c) * 1.001,
                         "low": np.minimum(o, c) * 0.999, "close": c,
                         "prob_up": [r[2] for r in rows]})


PARAMS = dict(ticker="BTC-USD", start_date="2024-01-01", end_date="2024-12-31",
              initial_capital=10_000.0, t_buy=0.60, t_sell=0.45, max_position_pct=0.25,
              max_drawdown_limit=0.15, commission=0.001, annualization_factor=365)


def by_action(result: dict, action: str, status: str = "EXECUTED") -> list[dict]:
    return [e for e in result["audit_logs"] if e["action"] == action and e["status"] == status]


# ------------------------------------------------------------------------------
# SimulationRequest
# ------------------------------------------------------------------------------


def test_request_defaults():
    req = SimulationRequest(ticker="BTC-USD")
    assert (req.t_buy, req.t_sell, req.max_position_pct, req.max_drawdown_limit,
            req.random_state, req.initial_capital) == (0.60, 0.45, 0.25, 0.15, 42, 10_000.0)
    assert (req.start_date, req.end_date) == (date(2021, 1, 1), date(2025, 12, 31))
    assert set(req.request_params()) == {"ticker", "start_date", "end_date", "initial_capital",
                                         "t_buy", "t_sell", "max_position_pct",
                                         "max_drawdown_limit"}


def test_request_accepts_max_exposure_alias_and_normalizes_ticker():
    req = SimulationRequest.model_validate({"ticker": "brk-b", "max_exposure": 0.2})
    assert req.ticker == "BRK-B" and req.max_position_pct == 0.2


@pytest.mark.parametrize("overrides", [
    {"ticker": "DOGE-USD"},
    {"start_date": "2020-12-31"},
    {"end_date": "2026-01-01"},
    {"start_date": "2023-06-01", "end_date": "2023-06-01"},   # start == end
    {"start_date": "2023-06-02", "end_date": "2023-06-01"},
    {"initial_capital": 0},
    {"t_buy": 0.49},
    {"t_buy": 1.0},
    {"t_sell": 0.0},
    {"t_sell": 0.51},
    {"t_buy": 0.50, "t_sell": 0.50},                          # t_sell не е < t_buy
    {"max_position_pct": 0.26},                               # над 25 %
    {"max_exposure": 0.0},
    {"max_drawdown_limit": 0.0},
    {"max_drawdown_limit": 1.0},
    {"random_state": -1},
    {"unexpected": 1},
])
def test_request_rejects_invalid_input(overrides):
    with pytest.raises(ValidationError):
        SimulationRequest.model_validate({"ticker": "BTC-USD", **overrides})


# ------------------------------------------------------------------------------
# Fallback AI (Договор 2)
# ------------------------------------------------------------------------------


def contract_df(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def test_fallback_ai_matches_contract_and_is_deterministic():
    df = contract_df(random_walk(end="2021-12-31"))
    a = fallback_generate_ai_predictions(df, 42)
    b = fallback_generate_ai_predictions(df, 42)
    c = fallback_generate_ai_predictions(df, 7)

    assert list(a["signals_df"].columns) == ["date", "open", "high", "low", "close", "prob_up"]
    assert a["signals_df"]["prob_up"].between(0, 1).all()
    assert len(a["signals_df"]) == len(df) - 20  # dropna() след прозорците
    assert set(a["ml_metrics"]) == {"accuracy", "precision", "recall", "f1_score"}
    pd.testing.assert_frame_equal(a["signals_df"], b["signals_df"])
    assert a["ml_metrics"] == b["ml_metrics"]
    assert not a["signals_df"]["prob_up"].equals(c["signals_df"]["prob_up"])


def test_fallback_ai_has_no_look_ahead():
    df = contract_df(random_walk(end="2021-06-30"))
    tampered = df.copy()
    tampered.loc[100:, ["open", "high", "low", "close"]] *= 3.0
    a = fallback_generate_ai_predictions(df)["signals_df"]
    b = fallback_generate_ai_predictions(tampered)["signals_df"]
    cut = a.index[a["date"] < df["date"].iloc[100]]
    pd.testing.assert_series_equal(a.loc[cut, "prob_up"], b.loc[cut, "prob_up"])


# ------------------------------------------------------------------------------
# Fallback двуфазен backtester
# ------------------------------------------------------------------------------


def test_buy_signal_at_close_t_executes_at_open_t_plus_1_with_fee():
    result = fallback_run_backtest(signals([(100, 100, 0.70), (105, 106, 0.50),
                                            (107, 108, 0.40), (110, 111, 0.50)]), PARAMS)
    buy, sell = by_action(result, "BUY")[0], by_action(result, "SELL")[0]
    assert (buy["signal_date"], buy["execution_date"]) == ("2024-01-01", "2024-01-02")
    assert buy["execution_price"] == 105.0                     # Open(t+1), не Close(t)
    assert buy["units"] * 105.0 + buy["fee_paid"] == pytest.approx(10_000 * 0.25)
    assert buy["fee_paid"] == pytest.approx(buy["units"] * 105.0 * 0.001)
    assert (sell["signal_date"], sell["execution_date"], sell["execution_price"]) == \
        ("2024-01-03", "2024-01-04", 110.0)
    ai = result["portfolio_metrics"]["ai_strategy"]
    assert ai["total_trades"] == 2 and ai["win_rate_pct"] == 100.0
    assert ai["circuit_breaker_triggered"] is False


def test_overnight_gap_triggers_force_sell_at_open():
    """Гап надолу при Open(t) -> FORCE_SELL_AT_OPEN в същия ден по Open(t)."""
    p = {**PARAMS, "max_position_pct": 1.0 * 0.25, "max_drawdown_limit": 0.05}
    rows = [(100, 100, 0.70), (100, 100, 0.50), (60, 62, 0.50), (62, 63, 0.70)]
    result = fallback_run_backtest(signals(rows), p)
    fs = by_action(result, "FORCE_SELL_AT_OPEN")
    assert len(fs) == 1
    assert fs[0]["signal_date"] == fs[0]["execution_date"] == "2024-01-03"
    assert fs[0]["execution_price"] == 60.0 and fs[0]["prob_up"] is None
    # след прекъсвача новият BUY сигнал е блокиран
    rejected = by_action(result, "BUY", "REJECTED_BY_RISK")
    assert [r["signal_date"] for r in rejected] == ["2024-01-04"]
    assert result["portfolio_metrics"]["ai_strategy"]["circuit_breaker_triggered"] is True


def test_close_breach_queues_force_sell_for_next_open():
    p = {**PARAMS, "max_drawdown_limit": 0.05}
    rows = [(100, 100, 0.70), (100, 70, 0.50), (65, 66, 0.50), (66, 67, 0.50)]
    result = fallback_run_backtest(signals(rows), p)
    fs = by_action(result, "FORCE_SELL")
    assert len(fs) == 1
    assert (fs[0]["signal_date"], fs[0]["execution_date"], fs[0]["execution_price"]) == \
        ("2024-01-02", "2024-01-03", 65.0)
    assert by_action(result, "FORCE_SELL_AT_OPEN") == []


def test_exposure_never_exceeds_max_position_pct():
    df = contract_df(random_walk(end="2022-12-31", seed=3))
    sig = fallback_generate_ai_predictions(df)["signals_df"]
    result = fallback_run_backtest(sig, {**PARAMS, "max_drawdown_limit": 0.99})
    curve = {pt["date"]: pt["ai_equity_close"] for pt in result["equity_curve"]}
    prev = dict(zip(sig["date"], sig["date"].shift(1)))
    for buy in by_action(result, "BUY"):
        spent = buy["units"] * buy["execution_price"] + buy["fee_paid"]
        assert spent <= 0.25 * curve[prev[buy["execution_date"]]] + 0.01  # curve е в центове


def test_buy_and_hold_reference():
    result = fallback_run_backtest(signals([(100, 100, 0.5), (100, 110, 0.5),
                                            (110, 120, 0.5)]), PARAMS)
    bnh = result["portfolio_metrics"]["buy_and_hold"]
    units = 10_000 / (100 * 1.001)
    assert bnh["final_equity"] == pytest.approx(units * 120, abs=0.01)
    assert result["equity_curve"][-1]["bnh_equity_close"] == pytest.approx(units * 120, abs=0.01)


def test_fallback_backtest_is_deterministic():
    df = contract_df(random_walk(end="2022-12-31", seed=5))
    sig = fallback_generate_ai_predictions(df)["signals_df"]
    assert fallback_run_backtest(sig, PARAMS) == fallback_run_backtest(sig, PARAMS)


# ------------------------------------------------------------------------------
# SimulationService: запис, четене, ACID
# ------------------------------------------------------------------------------


@pytest.fixture()
def service(session_factory):
    md = MarketDataService(session_factory, downloader=FakeDownloader(random_walk()))
    return SimulationService(session_factory, md)


def counts(session_factory) -> tuple[int, int]:
    with session_factory() as s:
        return (s.scalar(select(func.count()).select_from(SimulationRun)),
                s.scalar(select(func.count()).select_from(AuditLog)))


def test_run_persists_run_and_all_logs_and_reads_back(service, session_factory):
    res = service.run(SimulationRequest(ticker="BTC-USD"))
    assert res.engines.ai_engine == "fallback" and res.engines.backtester == "fallback"
    assert res.execution_time_ms is not None and res.created_at is not None
    assert counts(session_factory) == (1, len(res.audit_logs))
    assert len(res.audit_logs) > 0

    loaded = service.get(res.simulation_id)
    assert loaded.model_dump(exclude={"engines"}) == res.model_dump(exclude={"engines"})
    assert len(service.audit_logs(res.simulation_id)) == len(res.audit_logs)
    executed = service.audit_logs(res.simulation_id, "EXECUTED")
    assert executed and all(e["status"] == "EXECUTED" for e in executed)


def test_identical_requests_give_identical_results(service):
    req = SimulationRequest(ticker="BTC-USD", start_date="2022-01-01", end_date="2023-12-31")
    a, b = service.run(req), service.run(req)
    assert a.simulation_id != b.simulation_id
    keys = {"request_params", "ml_metrics", "portfolio_metrics", "equity_curve", "audit_logs"}
    assert a.model_dump(include=keys) == b.model_dump(include=keys)


def test_full_cycle_under_five_seconds(service):
    service.market_data.get_market_data("BTC-USD")  # загрява кеша (както seed скрипта)
    res = service.run(SimulationRequest(ticker="BTC-USD"))
    assert len(res.equity_curve) > 1_700
    assert res.execution_time_ms < 5_000


def test_one_invalid_audit_log_rolls_back_whole_simulation(session_factory):
    """ACID: невалиден лог (нарушен CHECK) -> ROLLBACK на SimulationRun и всички логове."""
    def broken_backtest(sig, params):
        out = fallback_run_backtest(sig, params)
        assert len(out["audit_logs"]) >= 2
        out["audit_logs"][-1]["units"] = -1.0          # нарушава non_negative_units
        return out

    md = MarketDataService(session_factory, downloader=FakeDownloader(random_walk()))
    engines = EngineAdapters(ai=fallback_generate_ai_predictions, backtest=broken_backtest,
                             info=EngineInfo(ai_engine="fallback", backtester="fallback"))
    svc = SimulationService(session_factory, md, engines)
    with pytest.raises(IntegrityError):
        svc.run(SimulationRequest(ticker="BTC-USD"))
    assert counts(session_factory) == (0, 0)


def test_engine_contract_violation_is_rejected_before_db(session_factory):
    md = MarketDataService(session_factory, downloader=FakeDownloader(random_walk()))
    engines = EngineAdapters(
        ai=lambda df, seed: {"signals_df": df.assign(prob_up=1.7), "ml_metrics": {}},
        backtest=fallback_run_backtest,
        info=EngineInfo(ai_engine="ai_engine", backtester="fallback"))
    with pytest.raises(EngineContractError):
        SimulationService(session_factory, md, engines).run(SimulationRequest(ticker="BTC-USD"))
    assert counts(session_factory) == (0, 0)


def test_get_unknown_simulation_raises(service):
    import uuid
    with pytest.raises(SimulationNotFoundError):
        service.get(uuid.uuid4())


# ------------------------------------------------------------------------------
# Адаптер: реален модул vs fallback
# ------------------------------------------------------------------------------


def test_resolve_engines_falls_back_when_modules_are_missing():
    engines = resolve_engines()
    assert engines.info.ai_engine == "fallback" and engines.info.backtester == "fallback"


def test_resolve_engines_uses_real_modules_when_present(monkeypatch):
    calls = {}

    def generate_ai_predictions(df_ohlcv):
        calls["ai"] = True
        return fallback_generate_ai_predictions(df_ohlcv)

    def run_backtest(signals_df, params):
        calls["bt"] = params
        return fallback_run_backtest(signals_df, params)

    predictor = types.ModuleType("ai_engine.predictor")
    predictor.generate_ai_predictions = generate_ai_predictions
    simulator = types.ModuleType("backtester.simulator")
    simulator.run_backtest = run_backtest
    monkeypatch.setitem(sys.modules, "ai_engine.predictor", predictor)
    monkeypatch.setitem(sys.modules, "backtester.simulator", simulator)

    engines = resolve_engines()
    assert engines.info.ai_engine == "ai_engine" and engines.info.backtester == "backtester"
    df = contract_df(random_walk(end="2021-12-31"))
    engines.backtest(engines.ai(df, 42)["signals_df"], PARAMS)
    assert calls["ai"] and calls["bt"]["commission"] == 0.001
