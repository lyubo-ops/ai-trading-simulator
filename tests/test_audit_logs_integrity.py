"""
Цялост на одитните логове и атомарност (ACID) на записа на симулация.

1. Броят на събитията в паметта на симулатора (BUY, SELL, FORCE_SELL_AT_OPEN,
   FORCE_SELL, REJECTED_BY_RISK) е ТОЧНО равен на SELECT COUNT(*) в audit_logs
   за тази симулация – общо и по всяка категория.
2. Грешка по средата на записа на логовете -> ROLLBACK: в базата не остават
   нито SimulationRun, нито частични (orphan) AuditLog записи.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select, text
from sqlalchemy.exc import IntegrityError

from backend.config import SIGNALS_COLUMNS
from backend.main import app, get_simulation_service
from backend.models import AuditLog, SimulationRun
from backend.schemas import EngineInfo, SimulationRequest
from backend.services.data_ingestion import MarketDataService
from backend.services.fallback_engine import (
    fallback_generate_ai_predictions,
    fallback_run_backtest,
)
from backend.services.simulation_service import EngineAdapters, SimulationService
from tests.conftest import FakeDownloader, random_walk, run_body

CATEGORIES = {
    "executed_trades": {("BUY", "EXECUTED"), ("SELL", "EXECUTED")},
    "forced_liquidations": {("FORCE_SELL_AT_OPEN", "EXECUTED"), ("FORCE_SELL", "EXECUTED")},
    "rejected_by_risk": {("BUY", "REJECTED_BY_RISK")},
}


# ==============================================================================
# Помощни: двигатели, които запомнят изхода си „в паметта на симулатора“
# ==============================================================================


class RecordingBacktester:
    """Обвива backtester-а и пази точно списъка audit_logs, който е генерирал."""

    def __init__(self, inner=fallback_run_backtest) -> None:
        self.inner = inner
        self.audit_logs: list[dict[str, Any]] = []

    def __call__(self, signals_df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
        out = self.inner(signals_df, params)
        self.audit_logs = [dict(e) for e in out["audit_logs"]]
        return out


def scripted_ai(prob_for_day):
    """AI заместител с предварително зададена вероятност P(t) за всеки ден."""
    def ai(df: pd.DataFrame, random_state: int) -> dict[str, Any]:
        signals = df.assign(prob_up=[prob_for_day(i) for i in range(len(df))])
        return {"signals_df": signals.loc[:, list(SIGNALS_COLUMNS)],
                "ml_metrics": {"accuracy": 0.5, "precision": 0.5, "recall": 0.5,
                               "f1_score": 0.5}}
    return ai


def shock_series(*, gap_at_open: bool, days: int = 60, shock_day: int = 30) -> pd.DataFrame:
    """Плосък ред около 100 с шок в ден `shock_day`: гап при Open(t) или срив до Close(t)."""
    idx = pd.date_range("2024-01-01", periods=days, freq="D")
    base = 100.0 + np.sin(np.arange(days) / 3.0)            # леко движение -> печалби/загуби
    o, c = base.copy(), base.copy() + 0.5
    if gap_at_open:
        o[shock_day:] *= 0.6
        c[shock_day:] *= 0.6
    else:
        c[shock_day:] *= 0.6
        o[shock_day + 1:] *= 0.6
    hi, lo = np.maximum(o, c) * 1.002, np.minimum(o, c) * 0.998
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                         "volume": np.full(days, 1e6)}, index=pd.DatetimeIndex(idx, name="date"))


def alternating_then_bullish(i: int) -> float:
    """Преди шока – редуващи се BUY/SELL сигнали; после – постоянен BUY (блокиран от риска)."""
    if i < 20:
        return {0: 0.70, 2: 0.40}.get(i % 4, 0.50)
    return 0.70


def build_service(session_factory, frame: pd.DataFrame, ai=None,
                  backtester: RecordingBacktester | None = None) -> SimulationService:
    md = MarketDataService(session_factory, downloader=FakeDownloader(frame))
    engines = EngineAdapters(ai=ai or fallback_generate_ai_predictions,
                             backtest=backtester or RecordingBacktester(),
                             info=EngineInfo(ai_engine="fallback", backtester="fallback"))
    return SimulationService(session_factory, md, engines)


def db_count(session_factory, simulation_id) -> int:
    with session_factory() as s:
        return s.execute(text("SELECT COUNT(*) FROM audit_logs WHERE simulation_run_id = :id"),
                         {"id": simulation_id.hex if s.bind.dialect.name == "sqlite"
                          else str(simulation_id)}).scalar_one()


def db_breakdown(session_factory, simulation_id) -> Counter:
    with session_factory() as s:
        rows = s.execute(select(AuditLog.action, AuditLog.status, func.count())
                         .where(AuditLog.simulation_run_id == simulation_id)
                         .group_by(AuditLog.action, AuditLog.status)).all()
    return Counter({(a.value, st.value): n for a, st, n in rows})


def totals(session_factory) -> tuple[int, int]:
    with session_factory() as s:
        return (s.scalar(select(func.count()).select_from(SimulationRun)),
                s.scalar(select(func.count()).select_from(AuditLog)))


# ==============================================================================
# 1. Памет на симулатора == COUNT(*) в audit_logs
# ==============================================================================


SCENARIOS = {
    # гап при Open(t) -> FORCE_SELL_AT_OPEN + блокирани BUY
    "gap_at_open": dict(frame=lambda: shock_series(gap_at_open=True),
                        ai=scripted_ai(alternating_then_bullish), limit=0.05,
                        start="2024-01-01", end="2024-02-29",
                        expect={("FORCE_SELL_AT_OPEN", "EXECUTED"), ("BUY", "REJECTED_BY_RISK"),
                                ("BUY", "EXECUTED"), ("SELL", "EXECUTED")}),
    # срив до Close(t) -> FORCE_SELL по Open(t+1) + блокирани BUY
    "breach_at_close": dict(frame=lambda: shock_series(gap_at_open=False),
                            ai=scripted_ai(alternating_then_bullish), limit=0.05,
                            start="2024-01-01", end="2024-02-29",
                            expect={("FORCE_SELL", "EXECUTED"), ("BUY", "REJECTED_BY_RISK"),
                                    ("BUY", "EXECUTED"), ("SELL", "EXECUTED")}),
    # пълен 5-годишен ред с fallback AI
    "five_year_random_walk": dict(frame=random_walk, ai=None, limit=0.15,
                                  start="2021-01-01", end="2025-12-31",
                                  expect={("BUY", "EXECUTED"), ("SELL", "EXECUTED")}),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_in_memory_audit_events_equal_db_count(session_factory, name):
    sc = SCENARIOS[name]
    recorder = RecordingBacktester()
    service = build_service(session_factory, sc["frame"](), sc["ai"], recorder)
    res = service.run(SimulationRequest(ticker="BTC-USD", start_date=sc["start"],
                                        end_date=sc["end"], max_drawdown_limit=sc["limit"]))

    memory = Counter((e["action"], e["status"]) for e in recorder.audit_logs)
    assert sc["expect"] <= set(memory), f"сценарият не покрива {sc['expect'] - set(memory)}"

    # общо: сделки + принудителни ликвидации + блокирани сигнали == COUNT(*)
    by_category = {cat: sum(memory[k] for k in keys) for cat, keys in CATEGORIES.items()}
    assert sum(by_category.values()) == len(recorder.audit_logs)
    assert db_count(session_factory, res.simulation_id) == len(recorder.audit_logs)

    # по всяка комбинация action/status
    assert db_breakdown(session_factory, res.simulation_id) == memory

    # нищо чуждо и нищо изгубено: редът и съдържанието съвпадат 1:1
    stored = service.audit_logs(res.simulation_id)
    assert stored == [{k: v for k, v in e.items() if k != "phase"}
                      for e in recorder.audit_logs]
    with session_factory() as s:
        seqs = s.scalars(select(AuditLog.sequence)
                         .where(AuditLog.simulation_run_id == res.simulation_id)
                         .order_by(AuditLog.sequence)).all()
    assert seqs == list(range(len(recorder.audit_logs)))

    # total_trades в метриките == изпълнените записи в БД
    executed = sum(n for (_, st), n in memory.items() if st == "EXECUTED")
    assert res.portfolio_metrics.ai_strategy.total_trades == executed


def test_counts_are_isolated_per_simulation(session_factory):
    recorder = RecordingBacktester()
    service = build_service(session_factory, random_walk(), backtester=recorder)
    a = service.run(SimulationRequest(ticker="BTC-USD", start_date="2021-01-01",
                                      end_date="2022-12-31"))
    n_a = len(recorder.audit_logs)
    b = service.run(SimulationRequest(ticker="BTC-USD", start_date="2023-01-01",
                                      end_date="2025-12-31"))
    n_b = len(recorder.audit_logs)
    assert db_count(session_factory, a.simulation_id) == n_a
    assert db_count(session_factory, b.simulation_id) == n_b
    assert totals(session_factory) == (2, n_a + n_b)


def test_api_response_logs_equal_db_count(client, session_factory):
    body = client.post("/api/v1/simulations/run", json=run_body()).json()
    import uuid
    sim_id = uuid.UUID(body["simulation_id"])
    assert db_count(session_factory, sim_id) == len(body["audit_logs"])
    assert client.get(f"/api/v1/simulations/{sim_id}/audit-logs").json()["count"] == \
        len(body["audit_logs"])


# ==============================================================================
# 2. ACID: грешка по средата на записа -> няма частични записи
# ==============================================================================


def test_failure_midway_through_audit_log_writes_leaves_no_orphans(session_factory):
    """
    `after_insert` се задейства СЛЕД като INSERT-ът на лога е изпратен към базата
    (в рамките на транзакцията). Хвърляме грешка на средния лог, след като сме
    проверили, че част от логовете вече е вътре -> целият запис трябва да се върне.
    """
    recorder = RecordingBacktester()
    service = build_service(session_factory, random_walk(), backtester=recorder)
    seen_inside_tx: list[int] = []

    def fail_midway(mapper, connection, target: AuditLog) -> None:
        half = len(recorder.audit_logs) // 2
        if target.sequence == half:
            seen_inside_tx.append(connection.execute(
                select(func.count()).select_from(AuditLog)).scalar_one())
            raise RuntimeError(f"симулиран срив при запис на лог #{half}")

    event.listen(AuditLog, "after_insert", fail_midway)
    try:
        with pytest.raises(RuntimeError, match="симулиран срив"):
            service.run(SimulationRequest(ticker="BTC-USD"))
    finally:
        event.remove(AuditLog, "after_insert", fail_midway)

    assert len(recorder.audit_logs) >= 4
    assert seen_inside_tx and seen_inside_tx[0] > 0, "частичните записи трябва да са били в БД"
    assert totals(session_factory) == (0, 0)
    with session_factory() as s:  # orphan = лог без родителска симулация
        orphans = s.execute(text(
            "SELECT COUNT(*) FROM audit_logs a LEFT JOIN simulation_runs r "
            "ON a.simulation_run_id = r.id WHERE r.id IS NULL")).scalar_one()
    assert orphans == 0


def test_constraint_violation_in_one_log_rolls_back_everything(session_factory):
    """Невалиден лог (нарушен CHECK) -> IntegrityError -> ROLLBACK на всичко."""
    def corrupt_last(sig, params):
        out = fallback_run_backtest(sig, params)
        out["audit_logs"][-1]["units"] = -1.0          # нарушава non_negative_units
        return out

    service = build_service(session_factory, random_walk(),
                            backtester=RecordingBacktester(corrupt_last))
    with pytest.raises(IntegrityError):
        service.run(SimulationRequest(ticker="BTC-USD"))
    assert totals(session_factory) == (0, 0)


def test_rollback_keeps_previous_simulations_intact(session_factory):
    good = build_service(session_factory, random_walk())
    first = good.run(SimulationRequest(ticker="BTC-USD", start_date="2024-01-01"))
    before = totals(session_factory)

    def corrupt_first(sig, params):
        out = fallback_run_backtest(sig, params)
        out["audit_logs"][0]["fee_paid"] = -5.0         # нарушава non_negative_fee
        return out

    bad = build_service(session_factory, random_walk(),
                        backtester=RecordingBacktester(corrupt_first))
    with pytest.raises(IntegrityError):
        bad.run(SimulationRequest(ticker="BTC-USD"))
    assert totals(session_factory) == before
    assert good.get(first.simulation_id).audit_logs == first.audit_logs


def test_api_rollback_returns_500_and_writes_nothing(session_factory):
    def corrupt_last(sig, params):
        out = fallback_run_backtest(sig, params)
        out["audit_logs"][-1]["units"] = -1.0
        return out

    service = build_service(session_factory, random_walk(),
                            backtester=RecordingBacktester(corrupt_last))
    app.dependency_overrides[get_simulation_service] = lambda: service
    try:
        r = TestClient(app).post("/api/v1/simulations/run", json=run_body())
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 500
    assert r.json()["error"] == "PersistenceError"
    assert totals(session_factory) == (0, 0)
