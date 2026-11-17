"""
Детерминирани fallback двигатели за независима работа в `feature/backend`.

Използват се САМО докато `/ai_engine` и `/backtester` не предоставят реалните си
функции (виж `simulation_service.py`). Изходът им е точно във форматите от `/contracts`:

* `fallback_generate_ai_predictions(df_ohlcv, random_state)` – Договор 2:
  {"signals_df": DataFrame[date, open, high, low, close, prob_up], "ml_metrics": {...}}.
  P(t) е прост моментум скор + шум от `np.random.default_rng(random_state)`; използва
  само данни до Close(t) (без look-ahead). Това НЕ е моделът на Любомир – само заместител.

* `fallback_run_backtest(signals_df, params)` – двуфазният дневен цикъл от
  архитектурния документ (docs/Arhitekturen_Proekt.docx, т. 3):

  ФАЗА 1 – Open(t)
    1.1 Equity_Open = Cash + Units * Open(t); Peak = max(Peak, Equity); Drawdown_Open.
    1.2 Drawdown_Open >= limit и прекъсвачът НЕ е активен -> активира се:
        отворена позиция -> FORCE_SELL_AT_OPEN по Open(t);
        чакащ BUY от t-1 -> REJECTED_BY_RISK (фаза OPEN).
    1.3 Иначе се изпълнява чакащата поръчка от t-1 (FORCE_SELL / SELL / BUY) по Open(t).
  ФАЗА 2 – Close(t)
    2.1 Equity_Close = Cash + Units * Close(t) -> точка от equity_curve.
    2.2 Drawdown_Close >= limit и прекъсвачът НЕ е активен -> активира се;
        отворена позиция -> FORCE_SELL на опашка за Open(t+1).
    2.3 ML сигнал P(t):
        P >= t_buy и няма позиция -> при активен прекъсвач REJECTED_BY_RISK (фаза CLOSE),
          иначе BUY на опашка с Cash_allocated = min(Cash, Equity_Close * max_position_pct);
        P <= t_sell и има позиция (без FORCE_SELL на опашка) -> SELL на опашка.

  Комисионна `commission` (0.1 %) върху всяко изпълнение. Buy & Hold: еднократна
  покупка по първия Open с комисионна, оценка по Close.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from backend.config import ML_METRIC_KEYS, SIGNALS_COLUMNS

WARMUP_DAYS = 20  # най-дългият прозорец на индикаторите -> dropna()


# ==============================================================================
# Договор 2 – fallback AI
# ==============================================================================


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = len(y_true)
    accuracy = float(np.mean(y_pred == y_true)) if n else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {k: round(v, 4) for k, v in zip(ML_METRIC_KEYS, (accuracy, precision, recall, f1))}


def fallback_generate_ai_predictions(df_ohlcv: pd.DataFrame,
                                     random_state: int = 42) -> dict[str, Any]:
    df = df_ohlcv.reset_index(drop=True)
    close = df["close"].astype("float64")

    # Индикатори X(t) – само с данни до Close(t) включително
    ret5 = close / close.shift(5) - 1.0
    ret20 = close / close.shift(WARMUP_DAYS) - 1.0
    score = 0.15 * np.tanh(10.0 * ret5) + 0.10 * np.tanh(5.0 * ret20)

    rng = np.random.default_rng(random_state)
    noise = pd.Series(rng.normal(0.0, 0.08, len(df)), index=df.index)
    prob_up = (0.5 + score + noise).clip(0.01, 0.99)

    signals = df.assign(prob_up=prob_up).dropna(subset=["prob_up"]).reset_index(drop=True)
    signals = signals.loc[:, list(SIGNALS_COLUMNS)]
    signals["prob_up"] = signals["prob_up"].round(6)

    # Y(t) = 1, ако Close(t+1) - Close(t) > 0; последният ден няма t+1
    y_true = (signals["close"].shift(-1) - signals["close"] > 0).astype(int).to_numpy()[:-1]
    y_pred = (signals["prob_up"] >= 0.5).astype(int).to_numpy()[:-1]
    return {"signals_df": signals, "ml_metrics": _binary_metrics(y_true, y_pred)}


# ==============================================================================
# Двуфазен backtester
# ==============================================================================


@dataclass
class _Order:
    action: str                 # BUY / SELL / FORCE_SELL
    signal_date: str
    prob_up: float | None
    cash_allocated: float = 0.0
    reason: str = ""


def _sharpe(equity: np.ndarray, periods_per_year: int) -> float:
    if len(equity) < 3:
        return 0.0
    returns = np.diff(equity) / equity[:-1]
    std = float(np.std(returns, ddof=1))
    if std == 0.0 or not math.isfinite(std):
        return 0.0
    return round(float(np.mean(returns)) / std * math.sqrt(periods_per_year), 4)


def _max_drawdown_pct(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    peak = np.maximum.accumulate(values)
    return round(float(np.max((peak - values) / peak)) * 100.0, 4)


def fallback_run_backtest(signals_df: pd.DataFrame, params: dict[str, Any]) -> dict[str, Any]:
    """Връща {"portfolio_metrics", "equity_curve", "audit_logs"} по договора."""
    t_buy, t_sell = float(params["t_buy"]), float(params["t_sell"])
    max_pos = float(params["max_position_pct"])
    limit = float(params["max_drawdown_limit"])
    fee_rate = float(params["commission"])
    capital = float(params["initial_capital"])
    periods = int(params.get("annualization_factor", 252))

    df = signals_df.reset_index(drop=True)
    dates = [str(d)[:10] for d in df["date"]]
    opens = df["open"].to_numpy(dtype="float64")
    closes = df["close"].to_numpy(dtype="float64")
    probs = df["prob_up"].to_numpy(dtype="float64")
    n = len(df)

    cash, units, peak = capital, 0.0, capital
    cb_active = False
    pending: _Order | None = None
    entry_cost = 0.0                      # cash, похарчен за текущата позиция (вкл. такса)
    closed_pnl: list[float] = []
    logs: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []
    all_drawdowns: list[float] = []

    bnh_units = capital / (opens[0] * (1.0 + fee_rate)) if n else 0.0

    def sell(price: float, action: str, signal_date: str, exec_date: str,
             prob: float | None, reason: str) -> None:
        nonlocal cash, units, entry_cost
        gross = units * price
        fee = gross * fee_rate
        cash += gross - fee
        closed_pnl.append(gross - fee - entry_cost)
        logs.append(dict(signal_date=signal_date, execution_date=exec_date, prob_up=prob,
                         action=action, status="EXECUTED", execution_price=price,
                         units=units, fee_paid=fee, reason=reason))
        units, entry_cost = 0.0, 0.0

    for i in range(n):
        t, o, c, p = dates[i], opens[i], closes[i], probs[i]
        next_date = dates[i + 1] if i + 1 < n else None

        # ---------------- ФАЗА 1: Open(t) ----------------
        equity_open = cash + units * o
        peak = max(peak, equity_open)
        dd_open = (peak - equity_open) / peak
        all_drawdowns.append(dd_open)

        if dd_open >= limit and not cb_active:
            cb_active = True
            if units > 0:
                sell(o, "FORCE_SELL_AT_OPEN", t, t, None,
                     f"Overnight gap: Drawdown_Open={dd_open:.2%} >= limit {limit:.2%}; "
                     f"position liquidated at Open(t).")
            # Спецификацията изисква отмяна на чакащ BUY. При long-only стратегия BUY се
            # нарежда само без позиция, а тогава гап не намалява капитала -> клонът е
            # защитен (за паритет със спецификацията и за реалния backtester).
            if pending is not None and pending.action == "BUY":
                logs.append(dict(signal_date=pending.signal_date, execution_date=t,
                                 prob_up=pending.prob_up, action="BUY",
                                 status="REJECTED_BY_RISK", execution_price=None, units=0.0,
                                 fee_paid=0.0, phase="OPEN",
                                 reason=f"Pending BUY cancelled: circuit breaker triggered at "
                                        f"Open(t) (Drawdown={dd_open:.2%})."))
            pending = None
        elif pending is not None:
            if pending.action == "BUY" and units == 0:
                buy_units = pending.cash_allocated / (o * (1.0 + fee_rate))
                if buy_units > 0:
                    fee = buy_units * o * fee_rate
                    cash -= buy_units * o + fee
                    units, entry_cost = buy_units, buy_units * o + fee
                    logs.append(dict(signal_date=pending.signal_date, execution_date=t,
                                     prob_up=pending.prob_up, action="BUY", status="EXECUTED",
                                     execution_price=o, units=buy_units, fee_paid=fee,
                                     reason=pending.reason))
            elif pending.action in ("SELL", "FORCE_SELL") and units > 0:
                sell(o, pending.action, pending.signal_date, t, pending.prob_up, pending.reason)
            pending = None

        # ---------------- ФАЗА 2: Close(t) ----------------
        equity_close = cash + units * c
        peak = max(peak, equity_close)
        dd_close = (peak - equity_close) / peak
        all_drawdowns.append(dd_close)
        curve.append(dict(date=t, close_price=round(c, 6), ai_equity_close=round(equity_close, 2),
                          bnh_equity_close=round(bnh_units * c, 2),
                          drawdown_close_pct=round(dd_close * 100.0, 4)))

        if dd_close >= limit and not cb_active:
            cb_active = True
            if units > 0 and next_date is not None:
                pending = _Order("FORCE_SELL", t, None,
                                 reason=f"Drawdown_Close={dd_close:.2%} >= limit {limit:.2%}; "
                                        f"forced liquidation at Open(t+1).")

        if p >= t_buy and units == 0 and pending is None:
            if cb_active:
                logs.append(dict(signal_date=t, execution_date=next_date, prob_up=p,
                                 action="BUY", status="REJECTED_BY_RISK", execution_price=None,
                                 units=0.0, fee_paid=0.0,
                                 reason=f"Max Drawdown circuit breaker active "
                                        f"(Drawdown >= {limit:.1%})."))
            elif next_date is not None:
                allocated = min(cash, equity_close * max_pos)
                if allocated > 0:
                    pending = _Order("BUY", t, p, allocated,
                                     reason=f"P(t)={p:.2f} >= T_buy({t_buy:.2f}); position size "
                                            f"{max_pos:.0%} approved by Risk Engine.")
        elif p <= t_sell and units > 0 and pending is None and next_date is not None:
            pending = _Order("SELL", t, p,
                             reason=f"P(t)={p:.2f} <= T_sell({t_sell:.2f}); long position "
                                    f"closed at Open(t+1).")

    ai_curve = np.array([pt["ai_equity_close"] for pt in curve], dtype="float64")
    bnh_curve = np.array([pt["bnh_equity_close"] for pt in curve], dtype="float64")
    final_ai = float(cash + units * closes[-1]) if n else capital
    final_bnh = float(bnh_units * closes[-1]) if n else capital
    wins = sum(1 for pnl in closed_pnl if pnl > 0)

    return {
        "portfolio_metrics": {
            "ai_strategy": {
                "final_equity": round(final_ai, 2),
                "net_return_pct": round((final_ai / capital - 1.0) * 100.0, 4),
                "max_drawdown_pct": round(max(all_drawdowns, default=0.0) * 100.0, 4),
                "sharpe_ratio": _sharpe(ai_curve, periods),
                "win_rate_pct": round(wins / len(closed_pnl) * 100.0, 4) if closed_pnl else 0.0,
                "total_trades": sum(1 for e in logs if e["status"] == "EXECUTED"),
                "circuit_breaker_triggered": cb_active,
            },
            "buy_and_hold": {
                "final_equity": round(final_bnh, 2),
                "net_return_pct": round((final_bnh / capital - 1.0) * 100.0, 4),
                "max_drawdown_pct": _max_drawdown_pct(bnh_curve),
                "sharpe_ratio": _sharpe(bnh_curve, periods),
            },
        },
        "equity_curve": curve,
        "audit_logs": logs,
    }


__all__ = ["fallback_generate_ai_predictions", "fallback_run_backtest"]
