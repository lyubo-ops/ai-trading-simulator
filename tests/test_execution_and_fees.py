import pandas as pd
import pytest

from backtester.simulator import run_backtest


def test_buy_and_sell_execute_at_next_open_with_fee():
    signals_df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "open": [100.0, 110.0, 120.0],
            "high": [102.0, 118.0, 122.0],
            "low": [99.0, 108.0, 118.0],
            "close": [100.0, 115.0, 120.0],
            "prob_up": [0.70, 0.40, 0.50],
        }
    )

    result = run_backtest(
        signals_df=signals_df,
        initial_capital=10000.0,
        t_buy=0.60,
        t_sell=0.45,
        max_position_pct=0.25,
        max_drawdown_limit=0.15,
        fee_rate=0.001,
    )

    logs = result["audit_logs"]
    assert len(logs) == 2

    buy_log = logs[0]
    assert buy_log["action"] == "BUY"
    assert buy_log["status"] == "EXECUTED"
    assert buy_log["signal_date"] == "2024-01-01"
    assert buy_log["execution_date"] == "2024-01-02"
    assert buy_log["execution_price"] == pytest.approx(110.0)
    assert buy_log["fee_paid"] == pytest.approx(2.5)

    expected_units = (2500.0 - 2.5) / 110.0
    assert buy_log["units"] == pytest.approx(expected_units, rel=1e-6)

    sell_log = logs[1]
    assert sell_log["action"] == "SELL"
    assert sell_log["status"] == "EXECUTED"
    assert sell_log["signal_date"] == "2024-01-02"
    assert sell_log["execution_date"] == "2024-01-03"
    assert sell_log["execution_price"] == pytest.approx(120.0)

    expected_gross = expected_units * 120.0
    expected_sell_fee = expected_gross * 0.001
    expected_final_cash = 7500.0 + (expected_gross - expected_sell_fee)

    assert sell_log["fee_paid"] == pytest.approx(expected_sell_fee, rel=1e-5)
    assert result["portfolio_metrics"]["final_capital"] == pytest.approx(
        round(expected_final_cash, 2)
    )
    assert result["portfolio_metrics"]["win_rate_pct"] == 100.0
    assert result["portfolio_metrics"]["circuit_breaker_triggered"] is False