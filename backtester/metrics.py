import numpy as np


def _compute_max_drawdown_pct(values: list[float], initial_capital: float) -> float:
    if not values:
        return 0.0
    peak = float(initial_capital)
    max_dd = 0.0
    for val in values:
        if val > peak:
            peak = val
        if peak > 0.0:
            dd = (peak - val) / peak
            if dd > max_dd:
                max_dd = dd
    return round(max_dd * 100.0, 2)


def _compute_sharpe_ratio(values: list[float], periods_per_year: int) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=float)
    prev = arr[:-1]
    valid_mask = prev > 0.0
    if not np.any(valid_mask):
        return 0.0
    returns = (arr[1:][valid_mask] - prev[valid_mask]) / prev[valid_mask]
    if len(returns) < 2:
        return 0.0
    std = float(np.std(returns, ddof=1))
    if std == 0.0 or np.isnan(std):
        return 0.0
    sharpe = (float(np.mean(returns)) / std) * float(np.sqrt(periods_per_year))
    return round(sharpe, 4)


def calculate_portfolio_metrics(
    equity_curve: list[dict],
    initial_capital: float,
    closed_trade_pnls: list[float],
    total_executed_orders: int,
    total_fees_paid: float,
    circuit_breaker_triggered: bool,
    periods_per_year: int = 252,
) -> dict:
    ai_values = [float(point["ai_equity"]) for point in equity_curve]
    bnh_values = [float(point["bnh_equity"]) for point in equity_curve]

    ai_final = ai_values[-1] if ai_values else float(initial_capital)
    bnh_final = bnh_values[-1] if bnh_values else float(initial_capital)

    ai_net_return_pct = ((ai_final - initial_capital) / initial_capital) * 100.0
    bnh_net_return_pct = ((bnh_final - initial_capital) / initial_capital) * 100.0

    ai_max_dd_pct = _compute_max_drawdown_pct(ai_values, initial_capital)
    bnh_max_dd_pct = _compute_max_drawdown_pct(bnh_values, initial_capital)

    ai_sharpe = _compute_sharpe_ratio(ai_values, periods_per_year)
    bnh_sharpe = _compute_sharpe_ratio(bnh_values, periods_per_year)

    if closed_trade_pnls:
        winning_trades = sum(1 for pnl in closed_trade_pnls if pnl > 0.0)
        win_rate_pct = (winning_trades / len(closed_trade_pnls)) * 100.0
    else:
        win_rate_pct = 0.0

    return {
        "final_capital": round(ai_final, 2),
        "net_return_pct": round(ai_net_return_pct, 2),
        "max_drawdown_pct": ai_max_dd_pct,
        "sharpe_ratio": ai_sharpe,
        "win_rate_pct": round(win_rate_pct, 2),
        "total_trades": int(total_executed_orders),
        "closed_round_trips": len(closed_trade_pnls),
        "total_fees_paid": round(total_fees_paid, 4),
        "circuit_breaker_triggered": bool(circuit_breaker_triggered),
        "bnh_final_capital": round(bnh_final, 2),
        "bnh_net_return_pct": round(bnh_net_return_pct, 2),
        "bnh_max_drawdown_pct": bnh_max_dd_pct,
        "bnh_sharpe_ratio": bnh_sharpe,
    }