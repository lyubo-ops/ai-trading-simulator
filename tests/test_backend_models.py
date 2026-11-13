"""Етап 1 – схема, атомарност и съответствие с contracts/mock_backtest_response.json."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from backend.database import transactional_session
from backend.models import AuditLog, AuditStatus, MarketData, Phase, SimulationRun
from tests.conftest import make_log, make_run

# ------------------------------------------------------------------------------
# MarketData
# ------------------------------------------------------------------------------


def test_market_data_unique_ticker_date(session_factory):
    row = dict(ticker="AAPL", date=date(2021, 1, 4), open=133.5, high=133.6,
               low=126.8, close=129.4, volume=143_301_900.0)
    with transactional_session(session_factory) as s:
        s.add(MarketData(**row))
    with pytest.raises(IntegrityError):
        with transactional_session(session_factory) as s:
            s.add(MarketData(**row))


def test_market_data_ohlc_consistency(session_factory):
    with pytest.raises(IntegrityError):
        with transactional_session(session_factory) as s:
            s.add(MarketData(ticker="AAPL", date=date(2021, 1, 4), open=10, high=9,
                             low=8, close=9, volume=1.0))


# ------------------------------------------------------------------------------
# SimulationRun
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("overrides", [
    dict(t_buy=0.40, t_sell=0.45),
    dict(max_position_pct=0.0),
    dict(max_position_pct=1.5),
    dict(max_drawdown_limit=1.0),
    dict(start_date=date(2025, 12, 31), end_date=date(2021, 1, 1)),
    dict(ai_max_drawdown_pct=120.0),
    dict(ai_win_rate_pct=-1.0),
    dict(accuracy=1.2),
])
def test_invalid_run_rejected(session_factory, overrides):
    with pytest.raises(IntegrityError):
        with transactional_session(session_factory) as s:
            s.add(make_run(**overrides))


def test_params_hash_deterministic(mock_backtest_response):
    rp = mock_backtest_response["request_params"]
    h1 = SimulationRun.compute_params_hash(rp, commission=0.001, random_state=42)
    h2 = SimulationRun.compute_params_hash(dict(reversed(list(rp.items()))),
                                           commission=0.001, random_state=42)
    assert h1 == h2 and len(h1) == 64
    assert h1 != SimulationRun.compute_params_hash(rp, commission=0.001, random_state=7)


# ------------------------------------------------------------------------------
# Атомарност (ACID)
# ------------------------------------------------------------------------------


def test_invalid_log_rolls_back_whole_run(session_factory):
    """Един невалиден лог -> ROLLBACK на симулацията и всички логове."""
    with pytest.raises(IntegrityError):
        with transactional_session(session_factory) as s:
            run = make_run()
            run.audit_logs = [make_log("buy_executed", 0),
                              make_log("sell_executed", 1, phase=Phase.CLOSE)]
            s.add(run)

    with session_factory() as s:
        assert s.scalars(select(SimulationRun)).all() == []
        assert s.scalars(select(AuditLog)).all() == []


def test_cascade_delete(session_factory):
    with transactional_session(session_factory) as s:
        run = make_run()
        run.audit_logs = [make_log("buy_executed", 0)]
        s.add(run)
        s.flush()
        run_id = run.id
    with transactional_session(session_factory) as s:
        s.delete(s.get(SimulationRun, run_id))
    with session_factory() as s:
        assert s.scalars(select(AuditLog)).all() == []


# ------------------------------------------------------------------------------
# Договорът на Любомир: contracts/mock_backtest_response.json
# ------------------------------------------------------------------------------


def test_contract_mock_round_trip(session_factory, mock_backtest_response):
    """mock -> ORM -> PostgreSQL/SQLite -> ORM -> mock дава ИДЕНТИЧЕН речник."""
    with transactional_session(session_factory) as s:
        run = SimulationRun.from_contract(mock_backtest_response)
        s.add(run)
        s.flush()
        run_id = run.id

    with session_factory() as s:
        loaded = s.get(SimulationRun, run_id)
        assert loaded.to_contract() == mock_backtest_response
        assert s.scalar(select(func.count()).select_from(AuditLog)) == len(
            mock_backtest_response["audit_logs"])


def test_contract_mock_phases_inferred(mock_backtest_response):
    run = SimulationRun.from_contract(mock_backtest_response)
    for log in run.audit_logs:
        if log.status is AuditStatus.EXECUTED:
            assert log.phase is Phase.OPEN
        else:
            assert log.phase is Phase.CLOSE


def test_contract_mock_executions_at_next_open(mock_backtest_response):
    """Всяко изпълнение в mock-а е на ден след сигнала, с комисионна 0.1 %."""
    for entry in mock_backtest_response["audit_logs"]:
        signal = date.fromisoformat(entry["signal_date"])
        if entry["status"] == "EXECUTED":
            assert date.fromisoformat(entry["execution_date"]) > signal
            expected_fee = entry["units"] * entry["execution_price"] * 0.001
            assert entry["fee_paid"] == pytest.approx(expected_fee, abs=0.01)
        else:
            assert entry["execution_price"] is None and entry["units"] == 0.0


def test_contract_mock_contains_no_unknown_fields(mock_backtest_response):
    """to_contract не добавя вътрешни полета (phase, sequence, commission, ...)."""
    out = SimulationRun.from_contract(mock_backtest_response).to_contract()
    assert set(out) == set(mock_backtest_response)
    assert set(out["request_params"]) == set(mock_backtest_response["request_params"])
    for got, exp in zip(out["audit_logs"], mock_backtest_response["audit_logs"]):
        assert set(got) == set(exp)
