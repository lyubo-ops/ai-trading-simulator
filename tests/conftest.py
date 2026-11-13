"""Общи fixtures и фабрики за тестовете на /backend."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from backend.database import create_db_engine, create_session_factory, init_db
from backend.models import AuditAction, AuditLog, AuditStatus, Phase, SimulationRun

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_BACKTEST_RESPONSE = REPO_ROOT / "contracts" / "mock_backtest_response.json"


def load_contract_json(path: Path) -> dict[str, Any]:
    """
    Чете JSON файл от contracts/ БЕЗ да го променя. Ако съдържанието е оградено
    с markdown блок (```json ... ```), оградата се игнорира.
    """
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    if lines and lines[0].lstrip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return json.loads("\n".join(lines))


@pytest.fixture()
def session_factory():
    """Изолирана in-memory SQLite база за всеки тест."""
    engine = create_db_engine("sqlite+pysqlite:///:memory:", echo=False)
    init_db(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture(scope="session")
def mock_backtest_response() -> dict[str, Any]:
    return load_contract_json(MOCK_BACKTEST_RESPONSE)


def make_run(**overrides: Any) -> SimulationRun:
    data: dict[str, Any] = dict(
        ticker="BTC-USD",
        start_date=date(2021, 1, 1),
        end_date=date(2025, 12, 31),
        initial_capital=10_000.0,
        t_buy=0.60,
        t_sell=0.45,
        max_position_pct=0.25,
        max_drawdown_limit=0.15,
        commission=0.001,
        random_state=42,
        params_hash="0" * 64,
        equity_curve=[{"date": "2024-01-01", "close_price": 42200.0, "ai_equity_close": 10000.0,
                       "bnh_equity_close": 10000.0, "drawdown_close_pct": 0.0}],
    )
    data.update(overrides)
    return SimulationRun(**data)


T, T1 = date(2024, 1, 4), date(2024, 1, 5)  # Close(t) -> Open(t+1)

# Валидни шаблони за всяка допустима комбинация (виж AuditLog docstring)
LOG_TEMPLATES: dict[str, dict[str, Any]] = {
    "buy_executed": dict(action=AuditAction.BUY, status=AuditStatus.EXECUTED, phase=Phase.OPEN,
                         signal_date=T, execution_date=T1, prob_up=0.68,
                         execution_price=42300.0, units=0.05904, fee_paid=2.50),
    "sell_executed": dict(action=AuditAction.SELL, status=AuditStatus.EXECUTED, phase=Phase.OPEN,
                          signal_date=T, execution_date=T1, prob_up=0.41,
                          execution_price=44250.0, units=0.05904, fee_paid=2.61),
    "force_sell_executed": dict(action=AuditAction.FORCE_SELL, status=AuditStatus.EXECUTED,
                                phase=Phase.OPEN, signal_date=T, execution_date=T1, prob_up=None,
                                execution_price=36000.0, units=0.05904, fee_paid=2.13),
    "force_sell_at_open": dict(action=AuditAction.FORCE_SELL_AT_OPEN, status=AuditStatus.EXECUTED,
                               phase=Phase.OPEN, signal_date=T1, execution_date=T1, prob_up=None,
                               execution_price=30000.0, units=0.05904, fee_paid=1.77),
    "rejected_close": dict(action=AuditAction.BUY, status=AuditStatus.REJECTED_BY_RISK,
                           phase=Phase.CLOSE, signal_date=T, execution_date=T1, prob_up=0.72,
                           execution_price=None, units=0.0, fee_paid=0.0),
    "rejected_close_no_exec_date": dict(action=AuditAction.BUY,
                                        status=AuditStatus.REJECTED_BY_RISK, phase=Phase.CLOSE,
                                        signal_date=T, execution_date=None, prob_up=0.72,
                                        execution_price=None, units=0.0, fee_paid=0.0),
    "rejected_open": dict(action=AuditAction.BUY, status=AuditStatus.REJECTED_BY_RISK,
                          phase=Phase.OPEN, signal_date=T, execution_date=T1, prob_up=0.66,
                          execution_price=None, units=0.0, fee_paid=0.0),
}


def make_log(kind: str, sequence: int = 0, **overrides: Any) -> AuditLog:
    data = {**LOG_TEMPLATES[kind], "sequence": sequence, "reason": kind}
    data.update(overrides)
    return AuditLog(**data)
