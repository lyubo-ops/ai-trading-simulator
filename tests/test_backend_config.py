"""Етап 1 – backend/config.py."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from backend.config import (
    ALL_TICKERS,
    CRYPTO_TICKERS,
    DATA_END_DATE,
    DATA_START_DATE,
    MAX_CONSECUTIVE_MISSING_DAYS,
    ML_METRIC_KEYS,
    OHLCV_COLUMNS,
    SIGNALS_COLUMNS,
    STOCK_TICKERS,
    AssetClass,
    Settings,
)


def test_fixed_universe_matches_contract():
    assert CRYPTO_TICKERS == ("BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "LTC-USD",
                              "ADA-USD", "DOT-USD", "LINK-USD", "BCH-USD", "XLM-USD")
    assert STOCK_TICKERS == ("AAPL", "MSFT", "AMZN", "GOOGL", "META",
                             "TSLA", "BRK-B", "JPM", "JNJ", "V")
    assert len(set(ALL_TICKERS)) == 20
    assert (DATA_START_DATE, DATA_END_DATE) == (date(2021, 1, 1), date(2025, 12, 31))


def test_asset_class_and_annualization():
    s = Settings(_env_file=None)
    assert s.asset_class_of("BTC-USD") is AssetClass.CRYPTO
    assert s.asset_class_of("BRK-B") is AssetClass.STOCK
    assert Settings.annualization_factor("ETH-USD") == 365
    assert Settings.annualization_factor("JPM") == 252
    with pytest.raises(ValueError):
        s.asset_class_of("DOGE-USD")


def test_data_contract_constants():
    assert OHLCV_COLUMNS == ("date", "open", "high", "low", "close", "volume")
    assert SIGNALS_COLUMNS == ("date", "open", "high", "low", "close", "prob_up")
    assert ML_METRIC_KEYS == ("accuracy", "precision", "recall", "f1_score")
    assert MAX_CONSECUTIVE_MISSING_DAYS == 3


def test_defaults_follow_spec():
    s = Settings(_env_file=None)
    assert s.default_t_buy == 0.60
    assert s.default_t_sell == 0.45
    assert s.default_max_position_pct == 0.25
    assert s.default_max_drawdown_limit == 0.15
    assert s.default_commission == 0.001
    assert s.default_random_state == 42


def test_default_request_params_match_contract_keys(mock_backtest_response):
    defaults = Settings(_env_file=None).default_request_params()
    contract_keys = set(mock_backtest_response["request_params"]) - {"ticker", "start_date",
                                                                     "end_date"}
    assert set(defaults) == contract_keys
    for key in ("t_buy", "t_sell", "max_position_pct", "max_drawdown_limit", "initial_capital"):
        assert defaults[key] == mock_backtest_response["request_params"][key]


def test_invalid_thresholds_rejected():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, default_t_buy=0.40, default_t_sell=0.45)


def test_sqlalchemy_url_uses_psycopg2():
    assert Settings(_env_file=None).sqlalchemy_url.drivername == "postgresql+psycopg2"
