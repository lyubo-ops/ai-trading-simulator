import pandas as pd
import pytest

from backtester.simulator import run_backtest


def test_circuit_breaker_triggers_at_open_on_overnight_gap():
    signals_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "open": [100.0, 100.0, 70.0, 75.0],
            "high": [101.0, 101.0, 72.0, 82.0],
            "low": [99.0, 97.0, 69.0, 74.0],
            "close": [100.0, 98.0, 70.0, 80.0],
            "prob_up": [0.80, 0.50, 0.85, 0.90],
        }
    )

    result = run_backtest(
        signals_df=signals_df,
        initial_capital=10000.0,
        t_buy=0.60,
        t_sell=0.45,
        max_position_pct=0.50,
        max_drawdown_limit=0.10,
        fee_rate=0.001,
    )

    logs = result["audit_logs"]
    statuses = [entry["status"] for entry in logs]
    assert statuses == [
        "EXECUTED",
        "FORCE_SELL_AT_OPEN",
        "REJECTED_BY_RISK",
        "REJECTED_BY_RISK",
    ]

    force_log = logs[1]
    assert force_log["action"] == "SELL"
    assert force_log["execution_date"] == "2024-01-03"
    assert force_log["execution_price"] == pytest.approx(70.0)

    assert logs[2]["action"] == "BUY"
    assert logs[2]["execution_date"] is None
    assert logs[2]["execution_price"] is None

    assert result["portfolio_metrics"]["circuit_breaker_triggered"] is True


def test_circuit_breaker_triggers_at_close_and_executes_force_sell_next_open():
    signals_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [100.0, 100.0, 68.0],
            "high": [101.0, 100.0, 70.0],
            "low": [99.0, 69.0, 67.0],
            "close": [100.0, 70.0, 68.0],
            "prob_up": [0.80, 0.50, 0.80],
        }
    )

    result = run_backtest(
        signals_df=signals_df,
        initial_capital=10000.0,
        t_buy=0.60,
        t_sell=0.45,
        max_position_pct=0.50,
        max_drawdown_limit=0.10,
        fee_rate=0.001,
    )

    logs = result["audit_logs"]
    statuses = [entry["status"] for entry in logs]
    assert statuses == ["EXECUTED", "FORCE_SELL", "REJECTED_BY_RISK"]

    force_log = logs[1]
    assert force_log["signal_date"] == "2024-01-02"
    assert force_log["execution_date"] == "2024-01-03"
    assert force_log["execution_price"] == pytest.approx(68.0)

    assert result["portfolio_metrics"]["circuit_breaker_triggered"] is True