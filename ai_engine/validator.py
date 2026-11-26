import pandas as pd


def generate_walk_forward_splits(
    df: pd.DataFrame,
    train_years: int = 3,
    test_months: int = 6,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    if df.empty or len(df) < 60:
        raise ValueError("Insufficient rows for walk-forward validation (minimum 60 required).")

    dates = pd.to_datetime(df["date"])
    min_date = dates.min()
    max_date = dates.max()

    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    train_start = min_date
    train_end = train_start + pd.DateOffset(years=train_years)

    if train_end < max_date:
        while train_end <= max_date:
            test_end = train_end + pd.DateOffset(months=test_months)

            train_mask = (dates >= train_start) & (dates < train_end)
            test_mask = (dates >= train_end) & (dates < test_end)

            train_df = df.loc[train_mask].copy()
            test_df = df.loc[test_mask].copy()

            if len(train_df) > 0 and len(test_df) > 0:
                splits.append((train_df, test_df))

            train_start = train_start + pd.DateOffset(months=test_months)
            train_end = train_end + pd.DateOffset(months=test_months)

        return splits

    n = len(df)
    train_size = int(n * 0.60)
    test_size = max(20, int(n * 0.20))

    start_idx = 0
    while start_idx + train_size < n:
        train_end_idx = start_idx + train_size
        test_end_idx = min(n, train_end_idx + test_size)

        train_df = df.iloc[start_idx:train_end_idx].copy()
        test_df = df.iloc[train_end_idx:test_end_idx].copy()

        if len(train_df) > 0 and len(test_df) > 0:
            splits.append((train_df, test_df))

        start_idx += test_size

    return splits