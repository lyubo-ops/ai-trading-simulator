"""
Централизирана конфигурация на /backend чрез Pydantic Settings.

Всички имена са адаптирани към официалните договори в `contracts/`
(`contracts/CONTRACTS.md` и `contracts/mock_backtest_response.json`):

* параметрите на заявката се казват точно както в `request_params`:
  `t_buy`, `t_sell`, `max_position_pct`, `max_drawdown_limit`, `initial_capital`;
* `max_position_pct` и `max_drawdown_limit` са ДРОБИ (0.25 == 25 %, 0.15 == 15 %),
  докато изходните метрики с наставка `_pct` са в ПРОЦЕНТИ (42.5 == 42.5 %);
* списъкът с 20-те тикера и периодът 2021-01-01 – 2025-12-31 са твърдо зададени
  константи и НЕ могат да се презапишат през environment.

Модел на изпълнение (официалното задание):
    Сигналите BUY/SELL се генерират след Close(t) и се изпълняват във Фаза 1
    на следващия ден по Open(t+1) с комисионна `commission` (0.1 %).
    Max Drawdown прекъсвачът се задейства при Drawdown >= max_drawdown_limit.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from functools import lru_cache
from typing import Final, Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy.engine import URL, make_url

# ==============================================================================
# Фиксиран универсум (contracts/CONTRACTS.md §1)
# ==============================================================================


class AssetClass(str, Enum):
    CRYPTO = "CRYPTO"
    STOCK = "STOCK"


CRYPTO_TICKERS: Final[tuple[str, ...]] = (
    "BTC-USD",
    "ETH-USD",
    "BNB-USD",
    "XRP-USD",
    "LTC-USD",
    "ADA-USD",
    "DOT-USD",
    "LINK-USD",
    "BCH-USD",
    "XLM-USD",
)

STOCK_TICKERS: Final[tuple[str, ...]] = (
    "AAPL",
    "MSFT",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "BRK-B",
    "JPM",
    "JNJ",
    "V",
)

ALL_TICKERS: Final[tuple[str, ...]] = CRYPTO_TICKERS + STOCK_TICKERS

TICKER_ASSET_CLASS: Final[dict[str, AssetClass]] = {
    **{t: AssetClass.CRYPTO for t in CRYPTO_TICKERS},
    **{t: AssetClass.STOCK for t in STOCK_TICKERS},
}

DATA_START_DATE: Final[date] = date(2021, 1, 1)
DATA_END_DATE: Final[date] = date(2025, 12, 31)

# Брой периоди годишно за анюализиране на Sharpe Ratio
ANNUALIZATION_FACTOR: Final[dict[AssetClass, int]] = {
    AssetClass.CRYPTO: 365,
    AssetClass.STOCK: 252,
}

# ==============================================================================
# Формати на данните (contracts/CONTRACTS.md §2 и §3)
# ==============================================================================

# Договор 1: /backend -> /ai_engine и /backtester (всички колони с малки букви)
OHLCV_COLUMNS: Final[tuple[str, ...]] = ("date", "open", "high", "low", "close", "volume")
OHLCV_PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close")

# Договор 2: /ai_engine -> /backtester (`signals_df`)
SIGNALS_COLUMNS: Final[tuple[str, ...]] = ("date", "open", "high", "low", "close", "prob_up")
ML_METRIC_KEYS: Final[tuple[str, ...]] = ("accuracy", "precision", "recall", "f1_score")

# Правило за липсващи данни: до 3 последователни търговски дни -> ffill (цени) / 0 (volume);
# повече -> ValueError и симулацията спира.
MAX_CONSECUTIVE_MISSING_DAYS: Final[int] = 3

assert len(CRYPTO_TICKERS) == 10, "Очакват се точно 10 крипто актива"
assert len(STOCK_TICKERS) == 10, "Очакват се точно 10 акции"
assert len(set(ALL_TICKERS)) == 20, "Тикерите трябва да са 20 уникални"
assert DATA_START_DATE < DATA_END_DATE


# ==============================================================================
# Настройки
# ==============================================================================


class Settings(BaseSettings):
    """Настройки на приложението; стойностите идват от env / .env файл."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    # --- Приложение ------------------------------------------------------------
    app_name: str = "AI-Powered Investment Platform – Backend"
    api_prefix: str = "/api/v1"
    environment: Literal["development", "test", "production"] = "development"
    debug: bool = False

    # --- PostgreSQL (драйвер psycopg2 – виж requirements.txt) -------------------
    postgres_user: str = "trading"
    postgres_password: SecretStr = SecretStr("trading")
    postgres_host: str = "localhost"
    postgres_port: int = Field(default=5432, ge=1, le=65535)
    postgres_db: str = "trading_platform"
    # Ако е зададен, има приоритет пред postgres_* полетата
    database_url: SecretStr | None = None

    db_echo: bool = False
    db_pool_size: int = Field(default=10, ge=1, le=100)
    db_max_overflow: int = Field(default=20, ge=0, le=200)
    db_pool_timeout: int = Field(default=30, ge=1, description="секунди")
    db_pool_recycle: int = Field(default=1800, ge=-1, description="секунди")
    db_statement_timeout_ms: int = Field(default=60_000, ge=0, description="0 = без лимит")

    # --- Параметри по подразбиране (имената следват request_params) ------------
    default_initial_capital: float = Field(default=10_000.0, gt=0.0)
    default_t_buy: float = Field(default=0.60, gt=0.0, le=1.0)
    default_t_sell: float = Field(default=0.45, ge=0.0, lt=1.0)
    default_max_position_pct: float = Field(
        default=0.25, gt=0.0, le=1.0, description="дроб: 0.25 == 25 % от капитала"
    )
    default_max_drawdown_limit: float = Field(
        default=0.15, gt=0.0, lt=1.0, description="дроб: 0.15 == 15 %"
    )
    default_commission: float = Field(
        default=0.001, ge=0.0, lt=0.05, description="0.1 % при изпълнение по Open(t+1)"
    )
    default_random_state: int = Field(default=42, ge=0)

    # --- Нефункционални изисквания ---------------------------------------------
    max_pipeline_seconds_per_asset: float = Field(default=5.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "Settings":
        if not self.default_t_sell < self.default_t_buy:
            raise ValueError(
                f"default_t_sell ({self.default_t_sell}) трябва да е < "
                f"default_t_buy ({self.default_t_buy})"
            )
        return self

    # --- Фиксиран универсум (read-only) ----------------------------------------
    @property
    def tickers(self) -> tuple[str, ...]:
        return ALL_TICKERS

    @property
    def crypto_tickers(self) -> tuple[str, ...]:
        return CRYPTO_TICKERS

    @property
    def stock_tickers(self) -> tuple[str, ...]:
        return STOCK_TICKERS

    @property
    def data_start_date(self) -> date:
        return DATA_START_DATE

    @property
    def data_end_date(self) -> date:
        return DATA_END_DATE

    def default_request_params(self) -> dict[str, float]:
        """Стойности по подразбиране с имената от `request_params` в договора."""
        return {
            "initial_capital": self.default_initial_capital,
            "t_buy": self.default_t_buy,
            "t_sell": self.default_t_sell,
            "max_position_pct": self.default_max_position_pct,
            "max_drawdown_limit": self.default_max_drawdown_limit,
        }

    # --- База данни -------------------------------------------------------------
    @property
    def sqlalchemy_url(self) -> URL:
        if self.database_url is not None:
            return make_url(self.database_url.get_secret_value())
        return URL.create(
            drivername="postgresql+psycopg2",
            username=self.postgres_user,
            password=self.postgres_password.get_secret_value(),
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        )

    # --- Помощни ----------------------------------------------------------------
    @staticmethod
    def is_supported_ticker(ticker: str) -> bool:
        return ticker in TICKER_ASSET_CLASS

    @staticmethod
    def asset_class_of(ticker: str) -> AssetClass:
        try:
            return TICKER_ASSET_CLASS[ticker]
        except KeyError as exc:
            raise ValueError(
                f"Неподдържан тикер '{ticker}'. Позволени: {', '.join(ALL_TICKERS)}"
            ) from exc

    @classmethod
    def annualization_factor(cls, ticker: str) -> int:
        return ANNUALIZATION_FACTOR[cls.asset_class_of(ticker)]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton на настройките (в тестове: get_settings.cache_clear())."""
    return Settings()


__all__ = [
    "ALL_TICKERS",
    "ANNUALIZATION_FACTOR",
    "AssetClass",
    "CRYPTO_TICKERS",
    "DATA_END_DATE",
    "DATA_START_DATE",
    "MAX_CONSECUTIVE_MISSING_DAYS",
    "ML_METRIC_KEYS",
    "OHLCV_COLUMNS",
    "OHLCV_PRICE_COLUMNS",
    "SIGNALS_COLUMNS",
    "STOCK_TICKERS",
    "Settings",
    "TICKER_ASSET_CLASS",
    "get_settings",
]
