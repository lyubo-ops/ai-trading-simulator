"""
CHECK ограниченията на AuditLog спрямо модела на изпълнение по Open(t+1):

ФАЗА 1 – Open(t):  преоценка; при гап >= max_drawdown_limit -> FORCE_SELL_AT_OPEN
                   (в същия ден) и отмяна на чакащ BUY (REJECTED_BY_RISK, фаза OPEN);
                   иначе изпълнение на чакащите от t-1 BUY / SELL / FORCE_SELL.
ФАЗА 2 – Close(t): преоценка; при пробив -> FORCE_SELL на опашка за Open(t+1);
                   ML сигнал: блокиран BUY -> REJECTED_BY_RISK (фаза CLOSE),
                   одобрени BUY/SELL -> изпълнение по Open(t+1).
"""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy.exc import IntegrityError

from backend.database import transactional_session
from backend.models import AuditAction, AuditStatus, Phase
from tests.conftest import LOG_TEMPLATES, T, T1, make_log, make_run

EXECUTED_KINDS = ("buy_executed", "sell_executed", "force_sell_executed", "force_sell_at_open")
NEXT_OPEN_KINDS = ("buy_executed", "sell_executed", "force_sell_executed")


def _persist(session_factory, *logs) -> None:
    with transactional_session(session_factory) as s:
        run = make_run()
        run.audit_logs = list(logs)
        s.add(run)


# ------------------------------------------------------------------------------
# Допустимите сценарии се приемат
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", sorted(LOG_TEMPLATES))
def test_valid_combination_accepted(session_factory, kind):
    _persist(session_factory, make_log(kind))


def test_close_breach_then_force_sell_at_next_open(session_factory):
    """Пробив при Close(t) -> блокиран BUY в CLOSE -> FORCE_SELL изпълнен по Open(t+1)."""
    _persist(
        session_factory,
        make_log("rejected_close", 0),
        make_log("force_sell_executed", 1),
    )


def test_overnight_gap_scenario(session_factory):
    """Гап при Open(t+1): FORCE_SELL_AT_OPEN + отмяна на чакащия BUY от Close(t), фаза OPEN."""
    _persist(
        session_factory,
        make_log("force_sell_at_open", 0),
        make_log("rejected_open", 1),
    )


# ------------------------------------------------------------------------------
# Никакво изпълнение по Close(t)
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", EXECUTED_KINDS)
def test_execution_in_close_phase_rejected(session_factory, kind):
    with pytest.raises(IntegrityError):
        _persist(session_factory, make_log(kind, phase=Phase.CLOSE))


@pytest.mark.parametrize("kind", NEXT_OPEN_KINDS)
def test_same_day_execution_rejected(session_factory, kind):
    """BUY/SELL/FORCE_SELL с execution_date == signal_date = изпълнение по Close(t)."""
    with pytest.raises(IntegrityError):
        _persist(session_factory, make_log(kind, execution_date=T))


@pytest.mark.parametrize("kind", NEXT_OPEN_KINDS)
def test_execution_before_signal_rejected(session_factory, kind):
    with pytest.raises(IntegrityError):
        _persist(session_factory, make_log(kind, execution_date=date(2024, 1, 3)))


@pytest.mark.parametrize("kind", EXECUTED_KINDS)
def test_execution_requires_price_units_and_date(session_factory, kind):
    for bad in (dict(execution_price=None), dict(execution_price=0.0), dict(units=0.0),
                dict(execution_date=None)):
        with pytest.raises(IntegrityError):
            _persist(session_factory, make_log(kind, **bad))


def test_force_sell_at_open_must_be_same_day(session_factory):
    with pytest.raises(IntegrityError):
        _persist(session_factory, make_log("force_sell_at_open", signal_date=T, execution_date=T1))


# ------------------------------------------------------------------------------
# REJECTED_BY_RISK – само за BUY, без изпълнение, във фаза CLOSE или OPEN
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["rejected_close", "rejected_open"])
def test_rejected_has_no_execution(session_factory, kind):
    for bad in (dict(execution_price=100.0), dict(units=0.1), dict(fee_paid=0.1)):
        with pytest.raises(IntegrityError):
            _persist(session_factory, make_log(kind, **bad))


@pytest.mark.parametrize("action", [AuditAction.SELL, AuditAction.FORCE_SELL,
                                    AuditAction.FORCE_SELL_AT_OPEN])
def test_only_buy_can_be_rejected(session_factory, action):
    prob = 0.4 if action is AuditAction.SELL else None
    with pytest.raises(IntegrityError):
        _persist(session_factory, make_log("rejected_close", action=action, prob_up=prob))


def test_rejected_at_open_needs_next_day_execution_date(session_factory):
    with pytest.raises(IntegrityError):
        _persist(session_factory, make_log("rejected_open", execution_date=None))
    with pytest.raises(IntegrityError):
        _persist(session_factory, make_log("rejected_open", execution_date=T))


def test_rejected_at_close_cannot_point_to_past(session_factory):
    with pytest.raises(IntegrityError):
        _persist(session_factory, make_log("rejected_close", execution_date=T))


# ------------------------------------------------------------------------------
# prob_up – само при ML сигнали
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("kind,prob", [
    ("force_sell_executed", 0.3), ("force_sell_at_open", 0.3),
    ("buy_executed", None), ("sell_executed", None), ("rejected_close", None),
    ("buy_executed", 1.2),
])
def test_prob_up_rules(session_factory, kind, prob):
    with pytest.raises(IntegrityError):
        _persist(session_factory, make_log(kind, prob_up=prob))


def test_status_enum_values_match_contract():
    assert {s.value for s in AuditStatus} == {"EXECUTED", "REJECTED_BY_RISK"}
    assert {a.value for a in AuditAction} == {"BUY", "SELL", "FORCE_SELL", "FORCE_SELL_AT_OPEN"}
