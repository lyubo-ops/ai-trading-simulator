"""Етап 2 – Data Ingestion & Validation (без мрежа: yfinance се подменя)."""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import func, select

from backend.config import OHLCV_COLUMNS, AssetClass
from backend.exceptions import (
    BackendError,
    DataValidationError,
    ExternalDataFetchError,
    InvalidDateRangeError,
    InvalidTickerError,
)
from backend.models import MarketData
from backend.services.data_ingestion import (
    MarketDataService,
    expected_trading_days,
    validate_and_clean_ohlcv,
)

PRICES = ["open", "high", "low", "close"]


# ------------------------------------------------------------------------------
# Синтетични данни
# ------------------------------------------------------------------------------


def make_ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Детерминиран, OHLC-консистентен ред с уникални стойности за всеки ден."""
    base = 100.0 + np.arange(len(index), dtype="float64")
    return pd.DataFrame(
        {"open": base, "high": base + 2.0, "low": base - 2.0, "close": base + 1.0,
         "volume": 1_000.0 + np.arange(len(index), dtype="float64")},
        index=pd.DatetimeIndex(index, name="date"),
    )


def crypto_df(start: str = "2021-01-01", periods: int = 30) -> pd.DataFrame:
    return make_ohlcv(pd.date_range(start, periods=periods, freq="D"))


def stock_df(start: str = "2021-03-01", end: str = "2021-04-30") -> pd.DataFrame:
    return make_ohlcv(expected_trading_days(AssetClass.STOCK, pd.Timestamp(start),
                                            pd.Timestamp(end)))


def with_nan_rows(df: pd.DataFrame, positions: list[int],
                  columns: list[str] | None = None) -> pd.DataFrame:
    out = df.copy()
    out.iloc[positions, [out.columns.get_loc(c) for c in (columns or list(out.columns))]] = np.nan
    return out


def row(clean: pd.DataFrame, day: str) -> pd.Series:
    return clean.set_index("date").loc[day]


# ------------------------------------------------------------------------------
# Формат по contracts/CONTRACTS.md §2
# ------------------------------------------------------------------------------


def assert_contract(clean: pd.DataFrame) -> None:
    assert tuple(clean.columns) == OHLCV_COLUMNS
    for col in OHLCV_COLUMNS[1:]:
        assert clean[col].dtype == np.float64
    assert clean["date"].map(type).eq(str).all()
    assert pd.to_datetime(clean["date"], format="%Y-%m-%d").is_monotonic_increasing
    assert not clean["date"].duplicated().any()
    assert not clean.isna().any().any()
    assert (clean["high"] >= clean[["open", "close", "low"]].max(axis=1)).all()
    assert (clean["low"] <= clean[["open", "close", "high"]].min(axis=1)).all()


def test_clean_series_unchanged_and_matches_contract():
    raw = crypto_df()
    clean = validate_and_clean_ohlcv(raw, "BTC-USD")
    assert_contract(clean)
    assert len(clean) == len(raw)
    np.testing.assert_array_equal(clean[PRICES].to_numpy(), raw[PRICES].to_numpy())


def test_input_frame_is_not_mutated():
    raw = with_nan_rows(crypto_df(), [10, 11])
    snapshot = raw.copy()
    validate_and_clean_ohlcv(raw, "BTC-USD")
    pd.testing.assert_frame_equal(raw, snapshot)


def test_yfinance_multiindex_and_tz_aware_index_are_normalized():
    """Форматът на yf.download: (Price, Ticker) колони, Adj Close, tz-aware индекс."""
    base = stock_df("2021-03-01", "2021-03-31")
    tz_index = pd.DatetimeIndex(base.index).tz_localize("America/New_York")
    col = {c: base[c].to_numpy() for c in base.columns}
    raw = pd.DataFrame(
        {("Adj Close", "AAPL"): col["close"] * 0.98, ("Close", "AAPL"): col["close"],
         ("High", "AAPL"): col["high"], ("Low", "AAPL"): col["low"],
         ("Open", "AAPL"): col["open"], ("Volume", "AAPL"): col["volume"].astype("int64")},
        index=pd.Index(tz_index, name="Date"),
    )
    raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])

    clean = validate_and_clean_ohlcv(raw, "AAPL")
    assert_contract(clean)
    assert clean["date"].iloc[0] == "2021-03-01"
    np.testing.assert_array_equal(clean["close"].to_numpy(), base["close"].to_numpy())


def test_rows_outside_global_period_are_dropped():
    raw = make_ohlcv(pd.date_range("2020-12-25", "2021-01-10", freq="D"))
    clean = validate_and_clean_ohlcv(raw, "ETH-USD")
    assert clean["date"].iloc[0] == "2021-01-01"


# ------------------------------------------------------------------------------
# Правило за липсващи данни – NaN стойности
# ------------------------------------------------------------------------------


def test_scenario_a_two_missing_days_forward_filled_from_t_minus_1():
    """Сценарий (а): 2 последователни NaN дни -> ffill точно със стойностите от t-1."""
    raw = with_nan_rows(crypto_df(), [10, 11])
    clean = validate_and_clean_ohlcv(raw, "BTC-USD")

    assert_contract(clean)
    assert len(clean) == len(raw)
    t_minus_1 = raw.iloc[9]
    for pos in (10, 11):
        filled = clean.iloc[pos]
        for col in PRICES:
            assert filled[col] == t_minus_1[col]
        assert filled["volume"] == 0.0  # contracts/CONTRACTS.md §1
    # Денят след дупката си остава непроменен
    assert clean.iloc[12]["close"] == raw.iloc[12]["close"]


def test_boundary_three_missing_days_forward_filled():
    raw = with_nan_rows(crypto_df(), [10, 11, 12])
    clean = validate_and_clean_ohlcv(raw, "BTC-USD")
    assert len(clean) == len(raw)
    for pos in (10, 11, 12):
        for col in PRICES:
            assert clean.iloc[pos][col] == raw.iloc[9][col]


def test_scenario_b_four_missing_days_raise():
    """Сценарий (б): 4 последователни NaN дни -> DataValidationError."""
    raw = with_nan_rows(crypto_df(), [10, 11, 12, 13])
    with pytest.raises(DataValidationError) as exc_info:
        validate_and_clean_ohlcv(raw, "BTC-USD")
    err = exc_info.value
    assert isinstance(err, ValueError)  # договорът изисква ValueError
    assert err.details["gap_length"] == 4
    assert err.details["gap_start"] == "2021-01-11"


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
def test_four_consecutive_nan_in_single_column_raise(column):
    raw = with_nan_rows(crypto_df(), [5, 6, 7, 8], columns=[column])
    with pytest.raises(DataValidationError):
        validate_and_clean_ohlcv(raw, "ETH-USD")


def test_two_separate_three_day_gaps_are_both_filled():
    raw = with_nan_rows(crypto_df(), [5, 6, 7, 9, 10, 11])
    clean = validate_and_clean_ohlcv(raw, "ETH-USD")
    assert clean.iloc[11]["close"] == raw.iloc[8]["close"]


def test_single_missing_price_copies_whole_candle_from_t_minus_1():
    """Само close е NaN -> цялата свещ е от t-1, за да остане OHLC-консистентна."""
    raw = with_nan_rows(crypto_df(), [10], columns=["close"])
    clean = validate_and_clean_ohlcv(raw, "BTC-USD")
    assert_contract(clean)
    for col in PRICES:
        assert clean.iloc[10][col] == raw.iloc[9][col]


def test_trailing_missing_days_forward_filled():
    raw = with_nan_rows(crypto_df(), [27, 28, 29])
    clean = validate_and_clean_ohlcv(raw, "BTC-USD")
    assert len(clean) == 30
    assert clean.iloc[-1]["close"] == raw.iloc[26]["close"]


# ------------------------------------------------------------------------------
# Look-ahead bias
# ------------------------------------------------------------------------------


def test_no_look_ahead_bias_future_prices_do_not_affect_filled_values():
    """Промяна на t+1, t+2 след дупката не променя попълнените стойности."""
    raw = with_nan_rows(crypto_df(), [10, 11])
    tampered = raw.copy()
    tampered.iloc[12, :4] = [999.0, 1_005.0, 990.0, 1_000.0]
    tampered.iloc[13, :4] = [5.0, 6.0, 4.0, 5.5]

    a = validate_and_clean_ohlcv(raw, "BTC-USD")
    b = validate_and_clean_ohlcv(tampered, "BTC-USD")

    pd.testing.assert_frame_equal(a.iloc[:12], b.iloc[:12])
    for pos in (10, 11):
        for col in PRICES:
            assert b.iloc[pos][col] == raw.iloc[9][col]
            assert b.iloc[pos][col] != tampered.iloc[12][col]


def test_leading_missing_days_are_dropped_not_backfilled():
    raw = with_nan_rows(crypto_df(), [0, 1, 2])
    clean = validate_and_clean_ohlcv(raw, "BTC-USD")
    assert clean["date"].iloc[0] == "2021-01-04"
    assert len(clean) == len(raw) - 3
    assert clean.iloc[0]["close"] == raw.iloc[3]["close"]


def test_four_leading_missing_days_raise():
    raw = with_nan_rows(crypto_df(), [0, 1, 2, 3])
    with pytest.raises(DataValidationError):
        validate_and_clean_ohlcv(raw, "BTC-USD")


# ------------------------------------------------------------------------------
# Липсващи дати в календара
# ------------------------------------------------------------------------------


def test_crypto_two_missing_calendar_dates_are_reinserted_and_filled():
    full = crypto_df()
    raw = full.drop(full.index[[10, 11]])
    clean = validate_and_clean_ohlcv(raw, "BTC-USD")

    assert len(clean) == len(full)  # 365-дневен календар: дните се връщат
    assert list(clean["date"]) == list(full.index.strftime("%Y-%m-%d"))
    for day in ("2021-01-11", "2021-01-12"):
        for col in PRICES:
            assert row(clean, day)[col] == full.iloc[9][col]
        assert row(clean, day)["volume"] == 0.0


def test_crypto_four_missing_calendar_dates_raise():
    full = crypto_df()
    raw = full.drop(full.index[[10, 11, 12, 13]])
    with pytest.raises(DataValidationError):
        validate_and_clean_ohlcv(raw, "BTC-USD")


def test_stock_weekends_and_nyse_holidays_are_not_missing_days():
    """Good Friday (2021-04-02) и уикендите не са липсващи търговски дни."""
    raw = stock_df("2021-03-29", "2021-04-09")
    assert pd.Timestamp("2021-04-02") not in raw.index
    clean = validate_and_clean_ohlcv(raw, "AAPL")
    assert len(clean) == len(raw)
    assert "2021-04-02" not in set(clean["date"])
    assert "2021-04-03" not in set(clean["date"])


def test_stock_missing_trading_day_in_index_is_filled():
    full = stock_df()
    raw = full.drop(full.index[[10, 11]])
    clean = validate_and_clean_ohlcv(raw, "MSFT")
    assert len(clean) == len(full)
    assert row(clean, full.index[11].strftime("%Y-%m-%d"))["close"] == full.iloc[9]["close"]


def test_stock_four_missing_trading_days_in_index_raise():
    full = stock_df()
    raw = full.drop(full.index[[10, 11, 12, 13]])
    with pytest.raises(DataValidationError):
        validate_and_clean_ohlcv(raw, "MSFT")


def test_nyse_calendar_2021_2025_session_counts():
    """~252 търговски дни годишно (реалните бройки на NYSE)."""
    expected = {2021: 252, 2022: 251, 2023: 250, 2024: 252, 2025: 250}
    for year, count in expected.items():
        days = expected_trading_days(AssetClass.STOCK, pd.Timestamp(f"{year}-01-01"),
                                     pd.Timestamp(f"{year}-12-31"))
        assert len(days) == count, year
    assert len(expected_trading_days(AssetClass.CRYPTO, pd.Timestamp("2024-01-01"),
                                     pd.Timestamp("2024-12-31"))) == 366


# ------------------------------------------------------------------------------
# Инварианти
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("column,value", [
    ("high", 50.0),     # high < low
    ("low", 500.0),     # low > high
    ("close", 0.0),     # неположителна цена
    ("open", -1.0),
    ("volume", -5.0),   # отрицателен обем
])
def test_invariant_violations_raise(column, value):
    raw = crypto_df()
    raw.iloc[5, raw.columns.get_loc(column)] = value
    with pytest.raises(DataValidationError):
        validate_and_clean_ohlcv(raw, "BTC-USD")


def test_float_noise_in_ohlc_is_repaired_within_tolerance():
    raw = crypto_df()
    raw.iloc[5, raw.columns.get_loc("close")] = raw.iloc[5]["high"] * (1 + 1e-12)
    clean = validate_and_clean_ohlcv(raw, "BTC-USD")
    assert_contract(clean)


def test_conflicting_duplicate_dates_raise_exact_duplicates_are_dropped():
    raw = crypto_df()
    exact = pd.concat([raw, raw.iloc[[3]]])
    assert len(validate_and_clean_ohlcv(exact, "BTC-USD")) == len(raw)

    conflicting = raw.iloc[[3]].copy()
    conflicting["close"] += 0.5
    with pytest.raises(DataValidationError):
        validate_and_clean_ohlcv(pd.concat([raw, conflicting]), "BTC-USD")


def test_unsorted_input_is_sorted():
    raw = crypto_df().iloc[::-1]
    clean = validate_and_clean_ohlcv(raw, "BTC-USD")
    assert clean["date"].is_monotonic_increasing


def test_missing_column_raises():
    with pytest.raises(DataValidationError):
        validate_and_clean_ohlcv(crypto_df().drop(columns="volume"), "BTC-USD")


def test_validate_rejects_unknown_ticker():
    with pytest.raises(InvalidTickerError):
        validate_and_clean_ohlcv(crypto_df(), "DOGE-USD")


# ------------------------------------------------------------------------------
# Кеширане в market_data (yfinance се подменя)
# ------------------------------------------------------------------------------


class FakeDownloader:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls: list[str] = []

    def __call__(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append(ticker)
        return self.frame.copy()


@pytest.fixture()
def full_crypto_raw() -> pd.DataFrame:
    raw = make_ohlcv(pd.date_range("2021-01-01", "2025-12-31", freq="D"))
    return with_nan_rows(raw, [100, 101])


def count_rows(session_factory, ticker: str) -> int:
    with session_factory() as s:
        return s.scalar(select(func.count()).select_from(MarketData)
                        .where(MarketData.ticker == ticker))


def test_cache_first_call_downloads_and_stores_second_reads_db(session_factory,
                                                               full_crypto_raw):
    fake = FakeDownloader(full_crypto_raw)
    service = MarketDataService(session_factory, downloader=fake)

    first = service.get_market_data("BTC-USD", "2021-01-01", "2025-12-31")
    assert fake.calls == ["BTC-USD"]
    assert count_rows(session_factory, "BTC-USD") == 1826 == len(first)
    assert_contract(first)

    fake.frame = fake.frame * 10  # ако бъде извикан пак, резултатът би се различил
    second = service.get_market_data("BTC-USD", "2021-01-01", "2025-12-31")
    assert fake.calls == ["BTC-USD"]  # без повторно викане на yfinance
    pd.testing.assert_frame_equal(first, second)


def test_cache_returns_requested_subperiod(session_factory, full_crypto_raw):
    service = MarketDataService(session_factory, downloader=FakeDownloader(full_crypto_raw))
    fresh = service.get_market_data("BTC-USD", "2023-02-01", "2023-02-28")
    cached = service.get_market_data("BTC-USD", date(2023, 2, 1), date(2023, 2, 28))

    assert count_rows(session_factory, "BTC-USD") == 1826  # кешира се целият ред
    assert len(fresh) == 28
    assert (fresh["date"].iloc[0], fresh["date"].iloc[-1]) == ("2023-02-01", "2023-02-28")
    pd.testing.assert_frame_equal(fresh, cached)


def test_force_refresh_redownloads_without_duplicates(session_factory, full_crypto_raw):
    fake = FakeDownloader(full_crypto_raw)
    service = MarketDataService(session_factory, downloader=fake)
    service.get_market_data("ETH-USD")
    fake.frame = fake.frame * 2
    refreshed = service.get_market_data("ETH-USD", force_refresh=True)

    assert fake.calls == ["ETH-USD", "ETH-USD"]
    assert count_rows(session_factory, "ETH-USD") == 1826
    assert refreshed["close"].iloc[0] == full_crypto_raw["close"].iloc[0] * 2


def test_cache_is_per_ticker(session_factory, full_crypto_raw):
    fake = FakeDownloader(full_crypto_raw)
    service = MarketDataService(session_factory, downloader=fake)
    service.get_market_data("BTC-USD")
    service.get_market_data("ETH-USD")
    assert fake.calls == ["BTC-USD", "ETH-USD"]


def test_invalid_data_is_not_written(session_factory):
    """DataValidationError -> нищо не се записва в market_data."""
    bad = with_nan_rows(crypto_df(periods=60), [20, 21, 22, 23])
    service = MarketDataService(session_factory, downloader=FakeDownloader(bad))
    with pytest.raises(DataValidationError):
        service.get_market_data("BTC-USD")
    assert count_rows(session_factory, "BTC-USD") == 0


def test_failed_refresh_keeps_existing_cache(session_factory, full_crypto_raw):
    fake = FakeDownloader(full_crypto_raw)
    service = MarketDataService(session_factory, downloader=fake)
    before = service.get_market_data("BTC-USD")
    fake.frame = with_nan_rows(full_crypto_raw, [500, 501, 502, 503])
    with pytest.raises(DataValidationError):
        service.get_market_data("BTC-USD", force_refresh=True)
    pd.testing.assert_frame_equal(before, service.get_market_data("BTC-USD"))


def test_empty_yfinance_response_raises_fetch_error(session_factory):
    def failing(ticker, start, end):
        raise ExternalDataFetchError("празен отговор", ticker=ticker)

    service = MarketDataService(session_factory, downloader=failing)
    with pytest.raises(ExternalDataFetchError):
        service.get_market_data("AAPL")


def test_cached_read_is_fast(session_factory, full_crypto_raw):
    service = MarketDataService(session_factory, downloader=FakeDownloader(full_crypto_raw))
    service.get_market_data("BTC-USD")
    t0 = time.perf_counter()
    service.get_market_data("BTC-USD")
    assert time.perf_counter() - t0 < 1.0  # бюджетът за целия цикъл е 5 s


# ------------------------------------------------------------------------------
# Невалиден тикер / период
# ------------------------------------------------------------------------------


@pytest.mark.parametrize("ticker", ["DOGE-USD", "NVDA", "btc-usd", "", "BRK.B"])
def test_invalid_ticker_rejected_before_download(session_factory, ticker):
    fake = FakeDownloader(crypto_df())
    service = MarketDataService(session_factory, downloader=fake)
    with pytest.raises(InvalidTickerError) as exc_info:
        service.get_market_data(ticker)
    assert isinstance(exc_info.value, BackendError)
    assert fake.calls == []


@pytest.mark.parametrize("start,end", [
    ("2020-12-31", "2021-06-30"),   # преди началото
    ("2025-06-01", "2026-01-01"),   # след края
    ("2019-01-01", "2027-01-01"),
    ("2023-06-01", "2023-01-01"),   # обърнат период
    ("2023-13-01", "2023-12-31"),   # невалидна дата
])
def test_invalid_period_rejected_before_download(session_factory, start, end):
    fake = FakeDownloader(crypto_df())
    service = MarketDataService(session_factory, downloader=fake)
    with pytest.raises(InvalidDateRangeError):
        service.get_market_data("BTC-USD", start, end)
    assert fake.calls == []


def test_period_without_trading_days_raises(session_factory):
    """Уикенд за акция -> няма търговски дни в подпериода."""
    raw = stock_df("2021-01-04", "2021-03-31")
    service = MarketDataService(session_factory, downloader=FakeDownloader(raw))
    with pytest.raises(DataValidationError):
        service.get_market_data("AAPL", "2021-01-09", "2021-01-10")
