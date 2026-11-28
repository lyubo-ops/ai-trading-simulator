import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "sma_10",
    "sma_20",
    "sma_50",
    "sma_ratio_10_50",
    "close_to_sma_20",
    "ema_12",
    "ema_26",
    "macd",
    "macd_signal",
    "macd_hist",
    "rsi_14",
    "volatility_14",
    "high_low_pct",
]

REQUIRED_OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi = rsi.where(avg_loss != 0.0, np.where(avg_gain > 0.0, 100.0, 50.0))
    return rsi


def compute_indicators(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_OHLCV_COLUMNS if col not in df_ohlcv.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    df = df_ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    df["return_1d"] = close.pct_change(periods=1)
    df["return_5d"] = close.pct_change(periods=5)
    df["volatility_14"] = df["return_1d"].rolling(window=14, min_periods=14).std()
    df["high_low_pct"] = (high - low) / close

    df["sma_10"] = close.rolling(window=10, min_periods=10).mean()
    df["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    df["sma_50"] = close.rolling(window=50, min_periods=50).mean()
    df["sma_ratio_10_50"] = df["sma_10"] / df["sma_50"]
    df["close_to_sma_20"] = (close - df["sma_20"]) / df["sma_20"]

    df["ema_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["rsi_14"] = _compute_rsi(close, period=14)

    return df


def prepare_ml_dataset(df_ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = compute_indicators(df_ohlcv)

    next_close = df["close"].shift(-1)
    df["target"] = np.where(
        next_close.isna(),
        np.nan,
        (next_close - df["close"] > 0).astype(float),
    )

    df_clean = df.dropna(subset=FEATURE_COLUMNS + ["target"]).reset_index(drop=True)
    df_clean["target"] = df_clean["target"].astype(int)
    df_clean["date"] = df_clean["date"].dt.strftime("%Y-%m-%d")
    return df_clean