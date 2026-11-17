"""
Data Ingestion & Validation (Етап 2).

Поток:
    get_market_data(ticker, start_date, end_date)
        ├─ валидира тикера (20-те от config.py) и периода (2021-01-01 – 2025-12-31)
        ├─ кеширан ли е тикерът в `market_data`?  ── да ──> SELECT за [start, end]
        └─ не (или force_refresh=True)
              ├─ download_raw_ohlcv()      – пълният 5-годишен ред от Yahoo Finance
              ├─ validate_and_clean_ohlcv() – чиста функция (без I/O)
              ├─ DELETE + bulk INSERT в ЕДНА транзакция (transactional_session)
              └─ връща подпериода [start, end]

Изходът следва `contracts/CONTRACTS.md` §2: колони `date, open, high, low, close, volume`
(малки букви), `date` е низ `YYYY-MM-DD`, всички числови колони са `float64`, редовете са
сортирани възходящо по дата и без дубликати.

Правило за липсващи данни (contracts/CONTRACTS.md §1 + заданието):
    * „липсващ ден“ = ред, в който поне една OHLCV стойност е NaN, ИЛИ очакван
      търговски ден, който изобщо липсва в индекса;
    * календар: крипто – всеки календарен ден ('D'); акции – NYSE търговски дни;
    * >= 4 последователни липсващи дни (вкл. в началото) -> DataValidationError;
    * 1–3 начални липсващи дни -> изрязват се (няма t-1, от което да се попълнят);
    * 1–3 вътрешни/крайни липсващи дни -> цените (open/high/low/close) се попълват
      с `ffill(limit=3)` САМО от t-1 (цялата свещ се копира, за да остане
      OHLC-консистентна), а `volume` -> 0. Не се използват bfill/интерполация.

Цени: `auto_adjust=False` -> Open/High/Low/Close на Yahoo са коригирани само за сплитове
(без дивидентна корекция); колоната `Adj Close` се отхвърля.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)
from pandas.tseries.offsets import CustomBusinessDay
from sqlalchemy import delete, func, insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from backend.config import (
    ALL_TICKERS,
    DATA_END_DATE,
    DATA_START_DATE,
    MAX_CONSECUTIVE_MISSING_DAYS,
    OHLCV_COLUMNS,
    OHLCV_PRICE_COLUMNS,
    TICKER_ASSET_CLASS,
    AssetClass,
)
from backend.database import get_session_factory, transactional_session
from backend.exceptions import (
    DataValidationError,
    ExternalDataFetchError,
    InvalidDateRangeError,
    InvalidTickerError,
)
from backend.models import MarketData

logger = logging.getLogger(__name__)

VALUE_COLUMNS: tuple[str, ...] = tuple(c for c in OHLCV_COLUMNS if c != "date")
PRICE_COLUMNS: tuple[str, ...] = OHLCV_PRICE_COLUMNS

# Относителен толеранс за шум с плаваща запетая в данните на Yahoo
# (напр. close = high * (1 + 1e-12)). По-големи нарушения -> DataValidationError.
OHLC_REL_TOLERANCE: float = 1e-6

DateLike = date | datetime | str
Downloader = Callable[[str, date, date], pd.DataFrame]


# ==============================================================================
# Борсови календари
# ==============================================================================


class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """
    Пълните неработни дни на NYSE за периода 2021–2025.

    Нова година, паднала се в събота, НЕ се почива в петък (31.12.2021 е работен ден).
    Juneteenth е борсов празник от 2022 г. 09.01.2025 – национален ден на траур (Дж. Картър).
    """

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
        Holiday("NationalDayOfMourningCarter", year=2025, month=1, day=9),
    ]


_NYSE_SESSION = CustomBusinessDay(calendar=NYSEHolidayCalendar())


def expected_trading_days(asset_class: AssetClass, start: pd.Timestamp,
                          end: pd.Timestamp) -> pd.DatetimeIndex:
    """Очакваните търговски дни на актива в [start, end] (вкл.)."""
    if asset_class is AssetClass.CRYPTO:
        return pd.date_range(start, end, freq="D")
    return pd.date_range(start, end, freq=_NYSE_SESSION)


# ==============================================================================
# Валидиране на входа
# ==============================================================================


def validate_ticker(ticker: str) -> str:
    if not isinstance(ticker, str) or ticker not in TICKER_ASSET_CLASS:
        raise InvalidTickerError(
            f"Неподдържан тикер '{ticker}'. Позволени: {', '.join(ALL_TICKERS)}",
            ticker=ticker if isinstance(ticker, str) else None,
        )
    return ticker


def _to_date(value: DateLike, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise InvalidDateRangeError(
                f"{field}='{value}' не е дата във формат YYYY-MM-DD", **{field: value}
            ) from exc
    raise InvalidDateRangeError(f"{field} трябва да е дата, получено {type(value).__name__}")


def validate_date_range(start_date: DateLike, end_date: DateLike) -> tuple[date, date]:
    start, end = _to_date(start_date, "start_date"), _to_date(end_date, "end_date")
    if start < DATA_START_DATE or end > DATA_END_DATE:
        raise InvalidDateRangeError(
            f"Периодът {start}..{end} излиза извън позволения "
            f"{DATA_START_DATE}..{DATA_END_DATE}",
            start_date=start.isoformat(), end_date=end.isoformat(),
        )
    if start > end:
        raise InvalidDateRangeError(
            f"start_date ({start}) е след end_date ({end})",
            start_date=start.isoformat(), end_date=end.isoformat(),
        )
    return start, end


# ==============================================================================
# Нормализация (yfinance -> OHLCV с DatetimeIndex)
# ==============================================================================


def _flatten_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Премахва MultiIndex колоните на yfinance ((Price, Ticker) или (Ticker, Price))."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    for level in range(df.columns.nlevels):
        if ticker in df.columns.get_level_values(level):
            df = df.xs(ticker, axis=1, level=level, drop_level=True)
            break
    if isinstance(df.columns, pd.MultiIndex):
        # Оставя нивото, което съдържа имената на цените
        for level in range(df.columns.nlevels):
            names = {str(v).strip().lower() for v in df.columns.get_level_values(level)}
            if "close" in names:
                df = df.copy()
                df.columns = df.columns.get_level_values(level)
                break
        else:
            raise DataValidationError("Неразпознати MultiIndex колони", ticker=ticker)
    return df


def normalize_ohlcv_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Привежда суров DataFrame (от yfinance или синтетичен) към:
    наивен (без timezone) нормализиран DatetimeIndex `date`, колони
    `open, high, low, close, volume` (float64), сортиран, в глобалния период.
    NaN стойностите се запазват – обработват се във `validate_and_clean_ohlcv`.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Очаква се pandas.DataFrame", ticker=ticker)

    frame = _flatten_columns(df, ticker).copy()
    frame.columns = [str(c).strip().lower().replace(" ", "_") for c in frame.columns]
    if frame.columns.duplicated().any():
        raise DataValidationError("Дублирани имена на колони", ticker=ticker,
                                  columns=list(frame.columns))

    if "date" in frame.columns:
        frame = frame.set_index("date")
    elif frame.index.name is not None and str(frame.index.name).lower() in {"date", "datetime"}:
        pass
    elif not isinstance(frame.index, pd.DatetimeIndex):
        raise DataValidationError("Липсва колона/индекс 'date'", ticker=ticker)

    missing_cols = [c for c in VALUE_COLUMNS if c not in frame.columns]
    if missing_cols:
        raise DataValidationError(f"Липсват колони: {missing_cols}", ticker=ticker,
                                  missing_columns=missing_cols)
    frame = frame.loc[:, list(VALUE_COLUMNS)].copy()

    try:
        idx = pd.DatetimeIndex(pd.to_datetime(frame.index))
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Неразпознаваеми дати в индекса", ticker=ticker) from exc
    if idx.hasnans:
        raise DataValidationError("Празни (NaT) дати в индекса", ticker=ticker)
    if idx.tz is not None:
        # yfinance: крипто -> UTC, акции -> America/New_York. Търговската дата е
        # локалната (борсова) дата, затова само махаме зоната, без конверсия.
        idx = idx.tz_localize(None)
    frame.index = idx.normalize()
    frame.index.name = "date"

    for col in VALUE_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float64")
    frame = frame.replace([np.inf, -np.inf], np.nan)

    frame = frame.sort_index(kind="mergesort")
    if frame.index.has_duplicates:
        # Еднакви дублирани редове са безвредни; различни стойности за една дата – не.
        dup = frame.reset_index().duplicated(keep="first").to_numpy()
        frame = frame[~dup]
        if frame.index.has_duplicates:
            dates = frame.index[frame.index.duplicated()].strftime("%Y-%m-%d").unique()
            raise DataValidationError(
                f"Противоречиви дублирани дати: {list(dates[:5])}", ticker=ticker,
                duplicate_dates=list(dates),
            )

    start, end = pd.Timestamp(DATA_START_DATE), pd.Timestamp(DATA_END_DATE)
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


# ==============================================================================
# Валидация и почистване (чиста функция)
# ==============================================================================


def _runs(mask: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Последователните True серии в булева маска: [(начало, край, дължина)]."""
    if not mask.any():
        return []
    values = mask.to_numpy(dtype=bool)
    change = np.empty(len(values), dtype=bool)
    change[0] = True
    change[1:] = values[1:] != values[:-1]
    group = np.cumsum(change)
    out = []
    for _, block in mask.groupby(group):
        if block.iloc[0]:
            out.append((block.index[0], block.index[-1], len(block)))
    return out


def _check_ohlc_invariants(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    o, h, l, c, v = (frame[col] for col in VALUE_COLUMNS)

    bad_price = (frame[list(PRICE_COLUMNS)] <= 0).any(axis=1)
    if bad_price.any():
        raise DataValidationError(
            "Неположителни цени", ticker=ticker,
            dates=frame.index[bad_price].strftime("%Y-%m-%d").tolist()[:10],
        )
    if (v < 0).any():
        raise DataValidationError(
            "Отрицателен обем", ticker=ticker,
            dates=frame.index[v < 0].strftime("%Y-%m-%d").tolist()[:10],
        )

    upper = pd.concat([o, c, l], axis=1).max(axis=1)
    lower = pd.concat([o, c, h], axis=1).min(axis=1)
    high_gap = (upper - h).clip(lower=0)
    low_gap = (l - lower).clip(lower=0)
    bad = (high_gap > OHLC_REL_TOLERANCE * upper) | (low_gap > OHLC_REL_TOLERANCE * lower)
    if bad.any():
        raise DataValidationError(
            "Нарушени OHLC инварианти (high >= max(open, close, low), "
            "low <= min(open, close, high))",
            ticker=ticker, dates=frame.index[bad].strftime("%Y-%m-%d").tolist()[:10],
        )
    # Поправка само на шум под толеранса, за да минат CHECK ограниченията в БД
    frame = frame.copy()
    frame["high"] = np.maximum(h, upper)
    frame["low"] = np.minimum(l, lower)
    return frame


def to_contract_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """DatetimeIndex -> колона `date` (YYYY-MM-DD) и точния ред на колоните от договора."""
    out = frame.reset_index()
    out["date"] = pd.DatetimeIndex(out["date"]).strftime("%Y-%m-%d")
    out = out.loc[:, list(OHLCV_COLUMNS)].reset_index(drop=True)
    for col in VALUE_COLUMNS:
        out[col] = out[col].astype("float64")
    return out


def validate_and_clean_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Валидира и почиства суров OHLCV ред. Чиста функция: не променя входа, без I/O.

    Връща DataFrame по contracts/CONTRACTS.md §2.
    Хвърля DataValidationError при > 3 последователни липсващи търговски дни,
    нарушени инварианти или неразпознаваем формат; InvalidTickerError за чужд тикер.
    """
    validate_ticker(ticker)
    asset_class = TICKER_ASSET_CLASS[ticker]
    limit = MAX_CONSECUTIVE_MISSING_DAYS

    frame = normalize_ohlcv_frame(df, ticker)
    if frame.empty:
        raise DataValidationError("Няма данни в периода 2021-01-01 – 2025-12-31",
                                  ticker=ticker)

    # 1) Изваждаме на показ липсващите дати като NaN редове. Обединението с реалния
    #    индекс гарантира, че никой реален ред не се губи заради календара.
    calendar = expected_trading_days(asset_class, frame.index[0], frame.index[-1])
    frame = frame.reindex(calendar.union(frame.index))
    frame.index.name = "date"

    # 2) Ден е „липсващ“, ако поне една OHLCV стойност е NaN
    missing = frame[list(VALUE_COLUMNS)].isna().any(axis=1)
    for run_start, run_end, length in _runs(missing):
        if length > limit:
            raise DataValidationError(
                f"{length} последователни липсващи търговски дни "
                f"({run_start:%Y-%m-%d} – {run_end:%Y-%m-%d}); лимитът е {limit}",
                ticker=ticker, gap_start=f"{run_start:%Y-%m-%d}",
                gap_end=f"{run_end:%Y-%m-%d}", gap_length=length,
            )

    # 3) Начални липсващи дни нямат t-1 -> изрязват се (никакъв bfill)
    first_valid = int(np.argmax(~missing.to_numpy())) if (~missing).any() else len(frame)
    if first_valid >= len(frame):
        raise DataValidationError("Серията не съдържа нито един пълен ред", ticker=ticker)
    frame = frame.iloc[first_valid:].copy()

    # 4) Ако липсва която и да е цена, цялата свещ се взима от t-1 (OHLC-консистентно)
    price_cols = list(PRICE_COLUMNS)
    price_missing = frame[price_cols].isna().any(axis=1)
    frame.loc[price_missing, price_cols] = np.nan
    frame[price_cols] = frame[price_cols].ffill(limit=limit)
    # Договорът: липсващ обем -> 0 (няма реална търговия, която да е наблюдавана)
    frame["volume"] = frame["volume"].fillna(0.0)

    if frame[list(VALUE_COLUMNS)].isna().any().any():  # защитна проверка
        raise DataValidationError("Останали NaN стойности след ffill", ticker=ticker)

    # 5) Инварианти
    frame = _check_ohlc_invariants(frame, ticker)
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise DataValidationError("Датите не са строго възходящи", ticker=ticker)

    filled = int(price_missing.sum())
    if first_valid or filled:
        logger.info("%s: изрязани %d начални, попълнени (ffill) %d дни",
                    ticker, first_valid, filled)
    return to_contract_frame(frame)


# ==============================================================================
# Yahoo Finance
# ==============================================================================


def download_raw_ohlcv(ticker: str, start: date = DATA_START_DATE,
                       end: date = DATA_END_DATE, *, attempts: int = 3) -> pd.DataFrame:
    """Сурови дневни OHLCV от yfinance за [start, end] (вкл.)."""
    import yfinance as yf  # lazy: тестовете подменят downloader-а и не теглят нищо

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            raw = yf.download(
                ticker,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),  # `end` в yfinance е изключващ
                interval="1d",
                auto_adjust=False,   # Open/High/Low/Close: само сплит-корекция
                actions=False,
                progress=False,
                threads=False,
            )
            if raw is not None and not raw.empty:
                return raw
            last_error = None
        except Exception as exc:  # noqa: BLE001 – yfinance хвърля разнородни грешки
            last_error = exc
        logger.warning("yfinance: опит %d/%d за %s неуспешен", attempt, attempts, ticker)
        if attempt < attempts:
            time.sleep(attempt)

    raise ExternalDataFetchError(
        f"Неуспешно теглене на {ticker} от Yahoo Finance"
        + (f": {last_error}" if last_error else " (празен отговор)"),
        ticker=ticker,
    ) from last_error


# ==============================================================================
# Кеширащ сервиз (PostgreSQL / market_data)
# ==============================================================================


class MarketDataService:
    """
    Чете почистени OHLCV от `market_data`, а при липса ги тегли, валидира и записва.

    Кешът е на ниво тикер: записът винаги е целият почистен 5-годишен ред, вкаран
    атомарно (DELETE + INSERT в една транзакция). Затова наличие на редове за тикера
    означава пълен, валидиран ред.
    """

    def __init__(self, session_factory: sessionmaker[Session] | None = None,
                 downloader: Downloader | None = None) -> None:
        self._session_factory = session_factory
        self._downloader = downloader

    @property
    def session_factory(self) -> sessionmaker[Session]:
        return self._session_factory or get_session_factory()

    def _download(self, ticker: str) -> pd.DataFrame:
        downloader = self._downloader or download_raw_ohlcv
        return downloader(ticker, DATA_START_DATE, DATA_END_DATE)

    # --- четене ---------------------------------------------------------------

    def cached_row_count(self, ticker: str) -> int:
        with self.session_factory() as s:
            return int(s.scalar(
                select(func.count()).select_from(MarketData).where(MarketData.ticker == ticker)
            ) or 0)

    def is_cached(self, ticker: str) -> bool:
        return self.cached_row_count(validate_ticker(ticker)) > 0

    def _read_cached(self, ticker: str, start: date, end: date) -> pd.DataFrame:
        stmt = (
            select(MarketData.date, MarketData.open, MarketData.high, MarketData.low,
                   MarketData.close, MarketData.volume)
            .where(MarketData.ticker == ticker, MarketData.date >= start,
                   MarketData.date <= end)
            .order_by(MarketData.date)
        )
        with self.session_factory() as s:
            rows = s.execute(stmt).all()
        frame = pd.DataFrame(rows, columns=list(OHLCV_COLUMNS))
        frame["date"] = [d.isoformat() for d in frame["date"]]
        for col in VALUE_COLUMNS:
            frame[col] = frame[col].astype("float64")
        return frame

    # --- запис ----------------------------------------------------------------

    def sync_ticker(self, ticker: str) -> pd.DataFrame:
        """Тегли, почиства и атомарно презаписва целия ред на тикера. Връща почистения ред."""
        validate_ticker(ticker)
        t0 = time.perf_counter()
        clean = validate_and_clean_ohlcv(self._download(ticker), ticker)

        records: list[dict[str, Any]] = [
            {"ticker": ticker, "date": date.fromisoformat(r.date), "open": r.open,
             "high": r.high, "low": r.low, "close": r.close, "volume": r.volume}
            for r in clean.itertuples(index=False)
        ]
        with transactional_session(self.session_factory) as s:
            s.execute(delete(MarketData).where(MarketData.ticker == ticker))
            s.execute(insert(MarketData), records)
        logger.info("%s: записани %d реда (%s – %s) за %.2f s", ticker, len(records),
                    clean["date"].iloc[0], clean["date"].iloc[-1], time.perf_counter() - t0)
        return clean

    # --- публичен API ---------------------------------------------------------

    def get_market_data(self, ticker: str, start_date: DateLike = DATA_START_DATE,
                        end_date: DateLike = DATA_END_DATE, *,
                        force_refresh: bool = False) -> pd.DataFrame:
        """Почистен OHLCV DataFrame за [start_date, end_date] по contracts/CONTRACTS.md §2."""
        validate_ticker(ticker)
        start, end = validate_date_range(start_date, end_date)

        if not force_refresh and self.is_cached(ticker):
            frame = self._read_cached(ticker, start, end)
        else:
            try:
                clean = self.sync_ticker(ticker)
            except IntegrityError:
                # Паралелна заявка е записала същия тикер преди нас -> ползваме нейния запис
                if not self.is_cached(ticker):
                    raise
                logger.info("%s: кешът е попълнен от паралелна заявка", ticker)
                frame = self._read_cached(ticker, start, end)
            else:
                mask = (clean["date"] >= start.isoformat()) & (clean["date"] <= end.isoformat())
                frame = clean.loc[mask].reset_index(drop=True)

        if frame.empty:
            raise DataValidationError(
                f"Няма търговски дни за {ticker} в периода {start} – {end}",
                ticker=ticker, start_date=start.isoformat(), end_date=end.isoformat(),
            )
        return frame


def get_market_data(ticker: str, start_date: DateLike = DATA_START_DATE,
                    end_date: DateLike = DATA_END_DATE, *, force_refresh: bool = False,
                    session_factory: sessionmaker[Session] | None = None) -> pd.DataFrame:
    """Функцията за зареждане по Договор 1 (/backend -> /ai_engine и /backtester)."""
    return MarketDataService(session_factory).get_market_data(
        ticker, start_date, end_date, force_refresh=force_refresh
    )


__all__ = [
    "MarketDataService",
    "NYSEHolidayCalendar",
    "download_raw_ohlcv",
    "expected_trading_days",
    "get_market_data",
    "normalize_ohlcv_frame",
    "to_contract_frame",
    "validate_and_clean_ohlcv",
    "validate_date_range",
    "validate_ticker",
]
