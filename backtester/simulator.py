import pandas as pd

from backtester.metrics import calculate_portfolio_metrics
from backtester.risk_engine import RiskEngine

REQUIRED_SIGNAL_COLUMNS = ["date", "open", "high", "low", "close", "prob_up"]


def run_backtest(
    signals_df: pd.DataFrame,
    initial_capital: float = 10000.0,
    t_buy: float = 0.60,
    t_sell: float = 0.45,
    max_position_pct: float = 0.25,
    max_drawdown_limit: float = 0.15,
    fee_rate: float = 0.001,
    periods_per_year: int = 252,
) -> dict:
    missing = [col for col in REQUIRED_SIGNAL_COLUMNS if col not in signals_df.columns]
    if missing:
        raise ValueError(f"Missing required signal columns: {missing}")
    if signals_df.empty:
        raise ValueError("signals_df cannot be empty.")
    if not (0.0 < t_sell < t_buy < 1.0):
        raise ValueError("Thresholds must satisfy 0 < t_sell < t_buy < 1.")
    if fee_rate < 0.0:
        raise ValueError("fee_rate must be non-negative.")

    df = signals_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    risk_engine = RiskEngine(
        initial_capital=initial_capital,
        max_position_pct=max_position_pct,
        max_drawdown_limit=max_drawdown_limit,
    )

    first_open = float(df.iloc[0]["open"])
    if first_open <= 0.0:
        raise ValueError("Opening prices must be positive.")
    bnh_entry_fee = float(initial_capital) * float(fee_rate)
    bnh_units = (float(initial_capital) - bnh_entry_fee) / first_open

    cash = float(initial_capital)
    units = 0.0
    entry_cost = 0.0
    pending_order: dict | None = None
    last_known_prob = 0.0

    audit_logs: list[dict] = []
    equity_curve: list[dict] = []
    closed_trade_pnls: list[float] = []
    total_executed_orders = 0
    total_fees_paid = 0.0

    for row in df.itertuples(index=False):
        date_str = row.date.strftime("%Y-%m-%d")
        open_price = float(row.open)
        close_price = float(row.close)
        prob_up = float(row.prob_up)

        _, dd_open = risk_engine.evaluate_equity(cash, units, open_price)
        triggered_at_open = risk_engine.check_drawdown_breach(dd_open)

        if triggered_at_open:
            if units > 0.0:
                gross_proceeds = units * open_price
                fee = gross_proceeds * fee_rate
                net_proceeds = gross_proceeds - fee
                sold_units = units
                cash += net_proceeds
                units = 0.0
                closed_trade_pnls.append(net_proceeds - entry_cost)
                entry_cost = 0.0
                total_executed_orders += 1
                total_fees_paid += fee
                audit_logs.append(
                    {
                        "signal_date": date_str,
                        "execution_date": date_str,
                        "prob_up": round(last_known_prob, 4),
                        "action": "SELL",
                        "status": "FORCE_SELL_AT_OPEN",
                        "execution_price": round(open_price, 6),
                        "units": round(sold_units, 8),
                        "fee_paid": round(fee, 6),
                        "reason": "Max drawdown limit breached at market open",
                    }
                )
            if pending_order is not None and pending_order["type"] == "BUY":
                audit_logs.append(
                    {
                        "signal_date": pending_order["signal_date"],
                        "execution_date": None,
                        "prob_up": round(pending_order["prob_up"], 4),
                        "action": "BUY",
                        "status": "REJECTED_BY_RISK",
                        "execution_price": None,
                        "units": 0.0,
                        "fee_paid": 0.0,
                        "reason": "Pending BUY canceled due to circuit breaker at market open",
                    }
                )
            pending_order = None
        elif pending_order is not None:
            if pending_order["type"] == "BUY":
                allocated = min(cash, float(pending_order["allocated_cash"]))
                if allocated > 0.0:
                    fee = allocated * fee_rate
                    net_invested = allocated - fee
                    bought_units = net_invested / open_price
                    cash -= allocated
                    units += bought_units
                    entry_cost = allocated
                    total_executed_orders += 1
                    total_fees_paid += fee
                    audit_logs.append(
                        {
                            "signal_date": pending_order["signal_date"],
                            "execution_date": date_str,
                            "prob_up": round(pending_order["prob_up"], 4),
                            "action": "BUY",
                            "status": "EXECUTED",
                            "execution_price": round(open_price, 6),
                            "units": round(bought_units, 8),
                            "fee_paid": round(fee, 6),
                            "reason": "AI BUY signal executed at next market open",
                        }
                    )
            elif pending_order["type"] in ("SELL", "FORCE_SELL"):
                if units > 0.0:
                    gross_proceeds = units * open_price
                    fee = gross_proceeds * fee_rate
                    net_proceeds = gross_proceeds - fee
                    sold_units = units
                    cash += net_proceeds
                    units = 0.0
                    closed_trade_pnls.append(net_proceeds - entry_cost)
                    entry_cost = 0.0
                    total_executed_orders += 1
                    total_fees_paid += fee
                    status = "FORCE_SELL" if pending_order["type"] == "FORCE_SELL" else "EXECUTED"
                    reason = pending_order.get(
                        "reason", "AI SELL signal executed at next market open"
                    )
                    audit_logs.append(
                        {
                            "signal_date": pending_order["signal_date"],
                            "execution_date": date_str,
                            "prob_up": round(pending_order["prob_up"], 4),
                            "action": "SELL",
                            "status": status,
                            "execution_price": round(open_price, 6),
                            "units": round(sold_units, 8),
                            "fee_paid": round(fee, 6),
                            "reason": reason,
                        }
                    )
            pending_order = None

        equity_close, dd_close = risk_engine.evaluate_equity(cash, units, close_price)
        bnh_equity = bnh_units * close_price
        equity_curve.append(
            {
                "date": date_str,
                "ai_equity": round(equity_close, 2),
                "bnh_equity": round(bnh_equity, 2),
                "ai_drawdown": round(dd_close * 100.0, 2),
            }
        )

        triggered_at_close = risk_engine.check_drawdown_breach(dd_close)
        if triggered_at_close and units > 0.0:
            pending_order = {
                "type": "FORCE_SELL",
                "signal_date": date_str,
                "prob_up": prob_up,
                "reason": "Max drawdown limit breached at market close",
            }

        if prob_up >= t_buy and units == 0.0:
            if risk_engine.circuit_breaker_active:
                audit_logs.append(
                    {
                        "signal_date": date_str,
                        "execution_date": None,
                        "prob_up": round(prob_up, 4),
                        "action": "BUY",
                        "status": "REJECTED_BY_RISK",
                        "execution_price": None,
                        "units": 0.0,
                        "fee_paid": 0.0,
                        "reason": "Circuit breaker active: BUY signal rejected",
                    }
                )
            else:
                allocated_cash = risk_engine.calculate_position_allocation(cash, equity_close)
                if allocated_cash > 0.0:
                    pending_order = {
                        "type": "BUY",
                        "signal_date": date_str,
                        "prob_up": prob_up,
                        "allocated_cash": allocated_cash,
                    }
        elif prob_up <= t_sell and units > 0.0:
            if pending_order is None or pending_order["type"] != "FORCE_SELL":
                pending_order = {
                    "type": "SELL",
                    "signal_date": date_str,
                    "prob_up": prob_up,
                }

        last_known_prob = prob_up

    portfolio_metrics = calculate_portfolio_metrics(
        equity_curve=equity_curve,
        initial_capital=initial_capital,
        closed_trade_pnls=closed_trade_pnls,
        total_executed_orders=total_executed_orders,
        total_fees_paid=total_fees_paid,
        circuit_breaker_triggered=risk_engine.circuit_breaker_active,
        periods_per_year=periods_per_year,
    )

    return {
        "portfolio_metrics": portfolio_metrics,
        "equity_curve": equity_curve,
        "audit_logs": audit_logs,
    }