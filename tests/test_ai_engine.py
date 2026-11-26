import numpy as np
import pandas as pd

from ai_engine.features import FEATURE_COLUMNS, compute_indicators, prepare_ml_dataset
from ai_engine.predictor import generate_ai_predictions
from ai_engine.validator import generate_walk_forward_splits


def _build_synthetic_ohlcv(n_days: int = 1800, start: str = "2021-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start=start, periods=n_days, freq="D")
    returns = rng.normal(loc=0.0005, scale=0.02, size=n_days)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_price = close * (1.0 + rng.normal(0.0, 0.005, size=n_days))
    high = np.maximum(open_price, close) * (1.0 + rng.uniform(0.001, 0.02, size=n_days))
    low = np.minimum(open_price, close) * (1.0 - rng.uniform(0.001, 0.02, size=n_days))
    volume = rng.uniform(1000.0, 100000.0, size=n_days)

    return pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_no_lookahead_bias_in_indicators():
    df_original = _build_synthetic_ohlcv(n_days=120, start="2023-01-01")
    res_original = compute_indicators(df_original)

    cutoff = 80
    df_mutated = df_original.copy()
    df_mutated.loc[cutoff + 1 :, ["open", "high", "low", "close"]] *= 5.0
    res_mutated = compute_indicators(df_mutated)

    pd.testing.assert_frame_equal(
        res_original.loc[:cutoff, FEATURE_COLUMNS],
        res_mutated.loc[:cutoff, FEATURE_COLUMNS],
        check_exact=False,
        rtol=1e-12,
    )


def test_walk_forward_no_overlap_and_chronological():
    df = _build_synthetic_ohlcv(n_days=1800, start="2021-01-01")
    dataset = prepare_ml_dataset(df)
    splits = generate_walk_forward_splits(dataset, train_years=3, test_months=6)

    assert len(splits) >= 2

    seen_test_dates: set[str] = set()
    for train_df, test_df in splits:
        assert pd.to_datetime(train_df["date"]).max() < pd.to_datetime(test_df["date"]).min()
        fold_dates = set(test_df["date"].tolist())
        assert seen_test_dates.isdisjoint(fold_dates)
        seen_test_dates.update(fold_dates)


def test_contract_2_output_and_determinism():
    df = _build_synthetic_ohlcv(n_days=1800, start="2021-01-01")

    result_1 = generate_ai_predictions(df)
    result_2 = generate_ai_predictions(df)

    assert set(result_1.keys()) == {"signals_df", "ml_metrics"}

    signals_df = result_1["signals_df"]
    assert list(signals_df.columns) == ["date", "open", "high", "low", "close", "prob_up"]
    assert signals_df.isna().sum().sum() == 0
    assert (signals_df["prob_up"] >= 0.0).all()
    assert (signals_df["prob_up"] <= 1.0).all()

    metrics = result_1["ml_metrics"]
    assert set(metrics.keys()) == {"accuracy", "precision", "recall", "f1_score"}

    pd.testing.assert_frame_equal(result_1["signals_df"], result_2["signals_df"])
    assert result_1["ml_metrics"] == result_2["ml_metrics"]