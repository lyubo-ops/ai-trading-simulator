"""
FastAPI ендпойнти и Pydantic схеми: валидация на входа и детерминирана възпроизводимост.

* невалиден тикер (извън фиксираните 20) -> HTTP 422 на всеки ендпойнт;
* невалидни прагове (t_sell >= t_buy, max_exposure > 0.25 и др.) и дати извън
  01.01.2021 – 31.12.2025 -> HTTP 422, без запис в базата;
* две последователни извиквания на /api/v1/simulations/run с еднакви параметри ->
  100 % идентични метрики, крива и логове, всяко под 5 секунди.
"""

from __future__ import annotations

import time
from datetime import date

import pytest
from pydantic import ValidationError
from sqlalchemy import func, select

from backend.config import ALL_TICKERS
from backend.models import SimulationRun
from backend.schemas import SimulationRequest, SimulationResponse
from tests.conftest import API, random_walk, run_body

INVALID_TICKERS = ["DOGE-USD", "NVDA", "BRK.B", "BTC", "BTCUSD", "", "  ", "AAPL; DROP TABLE"]
RESULT_KEYS = ("request_params", "ml_metrics", "portfolio_metrics", "equity_curve",
               "audit_logs")


def runs_in_db(session_factory) -> int:
    with session_factory() as s:
        return s.scalar(select(func.count()).select_from(SimulationRun))


def assert_422_on(response, field: str | None = None) -> None:
    assert response.status_code == 422, response.text
    if field is not None:
        detail = response.json()["detail"]
        assert any(field in map(str, err["loc"]) or field in err["msg"] for err in detail), detail


# ==============================================================================
# Pydantic: SimulationRequest
# ==============================================================================


def test_request_defaults():
    req = SimulationRequest(ticker="BTC-USD")
    assert (req.t_buy, req.t_sell, req.max_position_pct, req.max_drawdown_limit,
            req.random_state, req.initial_capital) == (0.60, 0.45, 0.25, 0.15, 42, 10_000.0)
    assert (req.start_date, req.end_date) == (date(2021, 1, 1), date(2025, 12, 31))
    assert set(req.request_params()) == {"ticker", "start_date", "end_date", "initial_capital",
                                         "t_buy", "t_sell", "max_position_pct",
                                         "max_drawdown_limit"}


@pytest.mark.parametrize("ticker", ALL_TICKERS)
def test_request_accepts_each_of_the_20_tickers(ticker):
    assert SimulationRequest(ticker=ticker.lower()).ticker == ticker


def test_request_accepts_max_exposure_alias():
    assert SimulationRequest.model_validate({"ticker": "AAPL", "max_exposure": 0.2}) \
        .max_position_pct == 0.2


def test_request_boundaries_are_inclusive():
    req = SimulationRequest(ticker="BTC-USD", start_date="2021-01-01", end_date="2025-12-31",
                            t_buy=0.99, t_sell=0.50, max_position_pct=0.25,
                            max_drawdown_limit=0.99)
    assert (req.t_buy, req.t_sell, req.max_position_pct) == (0.99, 0.50, 0.25)


@pytest.mark.parametrize("overrides", [
    {"ticker": "DOGE-USD"},
    {"start_date": "2020-12-31"},
    {"end_date": "2026-01-01"},
    {"start_date": "2023-06-01", "end_date": "2023-06-01"},   # start == end
    {"start_date": "2023-06-02", "end_date": "2023-06-01"},
    {"initial_capital": 0},
    {"t_buy": 0.49},
    {"t_buy": 1.0},
    {"t_sell": 0.0},
    {"t_sell": 0.51},
    {"t_buy": 0.50, "t_sell": 0.50},                          # t_sell не е < t_buy
    {"max_position_pct": 0.26},                               # над 25 %
    {"max_exposure": 0.0},
    {"max_drawdown_limit": 0.0},
    {"max_drawdown_limit": 1.0},
    {"random_state": -1},
    {"unexpected": 1},
])
def test_request_schema_rejects_invalid_input(overrides):
    with pytest.raises(ValidationError):
        SimulationRequest.model_validate({"ticker": "BTC-USD", **overrides})


# ==============================================================================
# HTTP 422: невалиден тикер
# ==============================================================================


@pytest.mark.parametrize("ticker", INVALID_TICKERS)
def test_invalid_ticker_rejected_on_simulation_run(client, session_factory, ticker):
    assert_422_on(client.post(f"{API}/simulations/run", json=run_body(ticker=ticker)), "ticker")
    assert runs_in_db(session_factory) == 0


@pytest.mark.parametrize("ticker", ["DOGE-USD", "NVDA", "BRK.B"])
def test_invalid_ticker_rejected_on_backtest_alias(client, ticker):
    assert_422_on(client.post(f"{API}/backtest", json=run_body(ticker=ticker)), "ticker")


@pytest.mark.parametrize("ticker", ["DOGE-USD", "NVDA", "BRK.B"])
def test_invalid_ticker_rejected_on_market_data(client, downloader, ticker):
    r = client.get(f"{API}/market-data/{ticker}")
    assert r.status_code == 422 and r.json()["error"] == "InvalidTickerError"
    assert downloader.calls == []  # yfinance не е извикван


@pytest.mark.parametrize("ticker", ["DOGE-USD", "NVDA"])
def test_invalid_ticker_rejected_on_market_data_sync(client, ticker):
    assert_422_on(client.post(f"{API}/market-data/sync", json={"ticker": ticker}), "ticker")


def test_missing_ticker_rejected(client):
    body = run_body()
    del body["ticker"]
    assert_422_on(client.post(f"{API}/simulations/run", json=body), "ticker")


# ==============================================================================
# HTTP 422: прагове и експозиция
# ==============================================================================


@pytest.mark.parametrize("overrides,field", [
    ({"t_buy": 0.60, "t_sell": 0.60}, "t_sell"),          # t_sell == t_buy
    ({"t_buy": 0.50, "t_sell": 0.50}, "t_sell"),
    ({"t_buy": 0.55, "t_sell": 0.70}, "t_sell"),          # t_sell > t_buy
    ({"t_buy": 0.40}, "t_buy"),                           # под 0.50
    ({"t_buy": 1.20}, "t_buy"),
    ({"t_sell": 0.0}, "t_sell"),
    ({"t_sell": -0.1}, "t_sell"),
    ({"max_exposure": 0.26}, "max_exposure"),            # над 25 %
    ({"max_exposure": 1.0}, "max_exposure"),
    ({"max_position_pct": 0.5}, "max_position_pct"),
    ({"max_position_pct": 0.0}, "max_position_pct"),
    ({"max_drawdown_limit": 0.0}, "max_drawdown_limit"),
    ({"max_drawdown_limit": 1.0}, "max_drawdown_limit"),
    ({"initial_capital": 0}, "initial_capital"),
    ({"initial_capital": -100}, "initial_capital"),
    ({"t_buy": "abc"}, "t_buy"),
    ({"random_state": -1}, "random_state"),
    ({"leverage": 2}, "leverage"),                        # непознато поле
])
def test_invalid_thresholds_rejected(client, session_factory, overrides, field):
    body = run_body(**overrides)
    if "max_exposure" in overrides:
        del body["max_position_pct"]
    assert_422_on(client.post(f"{API}/simulations/run", json=body), field)
    assert runs_in_db(session_factory) == 0


def test_max_exposure_25_percent_is_accepted(client):
    body = run_body(start_date="2025-01-01")
    del body["max_position_pct"]
    r = client.post(f"{API}/simulations/run", json={**body, "max_exposure": 0.25})
    assert r.status_code == 201, r.text
    assert r.json()["request_params"]["max_position_pct"] == 0.25


# ==============================================================================
# HTTP 422: дати
# ==============================================================================


@pytest.mark.parametrize("start,end", [
    ("2020-12-31", "2025-12-31"),   # преди 01.01.2021
    ("2021-01-01", "2026-01-01"),   # след 31.12.2025
    ("2019-01-01", "2027-01-01"),
    ("2024-06-01", "2024-06-01"),   # start == end
    ("2025-12-31", "2021-01-01"),   # обърнат период
    ("2024-02-30", "2024-12-31"),   # несъществуваща дата
    ("01.01.2021", "2025-12-31"),   # грешен формат
])
def test_dates_outside_period_rejected_on_simulation_run(client, session_factory, start, end):
    r = client.post(f"{API}/simulations/run", json=run_body(start_date=start, end_date=end))
    assert r.status_code == 422, r.text
    assert runs_in_db(session_factory) == 0


@pytest.mark.parametrize("params", [
    {"start_date": "2020-12-31"},
    {"end_date": "2026-01-01"},
    {"start_date": "2024-02-01", "end_date": "2024-01-01"},
])
def test_dates_outside_period_rejected_on_market_data(client, downloader, params):
    r = client.get(f"{API}/market-data/BTC-USD", params=params)
    assert r.status_code == 422 and r.json()["error"] == "InvalidDateRangeError"
    assert downloader.calls == []


# ==============================================================================
# Детерминирана възпроизводимост през /api/v1/simulations/run
# ==============================================================================


@pytest.mark.parametrize("ticker,frame", [
    ("BTC-USD", lambda: random_walk()),                                      # 1826 дни
    ("AAPL", lambda: random_walk(start="2021-01-04", freq="B", seed=11)),    # ~1300 дни
])
def test_identical_requests_return_identical_results_under_5s(session_factory, ticker,
                                                               frame, client, downloader):
    downloader.frame = frame()
    body = run_body(ticker=ticker)

    results = []
    for _ in range(2):
        t0 = time.perf_counter()
        r = client.post(f"{API}/simulations/run", json=body)
        wall = time.perf_counter() - t0
        assert r.status_code == 201, r.text
        assert wall < 5.0, f"HTTP заявката отне {wall:.2f} s"
        payload = r.json()
        assert payload["execution_time_ms"] < 5_000
        SimulationResponse.model_validate(payload)      # отговорът спазва договора
        results.append(payload)

    first, second = results
    assert first["simulation_id"] != second["simulation_id"]
    for key in RESULT_KEYS:
        assert first[key] == second[key], f"{key} се различава между двете извиквания"
    assert len(first["equity_curve"]) > 1_000 and first["audit_logs"]
    assert len(downloader.calls) == 1  # второто извикване чете от кеша

    with session_factory() as s:
        hashes = s.scalars(select(SimulationRun.params_hash)).all()
    assert len(hashes) == 2 and hashes[0] == hashes[1]


def test_different_random_state_changes_result(client):
    a = client.post(f"{API}/simulations/run", json=run_body(random_state=42)).json()
    b = client.post(f"{API}/simulations/run", json=run_body(random_state=7)).json()
    assert a["random_state"] == 42 and b["random_state"] == 7
    assert a["audit_logs"] != b["audit_logs"]


def test_stored_simulation_is_identical_to_response(client):
    created = client.post(f"{API}/simulations/run", json=run_body()).json()
    stored = client.get(f"{API}/simulations/{created['simulation_id']}").json()
    for key in (*RESULT_KEYS, "random_state", "commission", "execution_time_ms"):
        assert stored[key] == created[key]
