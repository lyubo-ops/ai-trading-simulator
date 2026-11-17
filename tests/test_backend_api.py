"""Етап 3 – FastAPI REST API (TestClient + SQLite в паметта, без мрежа)."""

from __future__ import annotations

import uuid

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, func, select
from sqlalchemy.pool import StaticPool

from backend.database import create_session_factory, init_db
from backend.main import app, get_db_session_factory, get_market_data_service
from backend.models import AuditLog, SimulationRun
from backend.services.data_ingestion import MarketDataService
from tests.test_backend_simulation import FakeDownloader, random_walk

API = "/api/v1"


@pytest.fixture()
def api_sf():
    """SQLite в паметта, споделена между нишките на TestClient (StaticPool)."""
    engine = create_engine("sqlite://", poolclass=StaticPool,
                           connect_args={"check_same_thread": False})
    event.listen(engine, "connect",
                 lambda conn, _: conn.execute("PRAGMA foreign_keys=ON"))
    init_db(engine)
    yield create_session_factory(engine)
    engine.dispose()


@pytest.fixture()
def downloader():
    return FakeDownloader(random_walk())


@pytest.fixture()
def client(api_sf, downloader):
    app.dependency_overrides[get_db_session_factory] = lambda: api_sf
    app.dependency_overrides[get_market_data_service] = \
        lambda: MarketDataService(api_sf, downloader=downloader)
    yield TestClient(app)  # без `with` -> lifespan (init_db върху PostgreSQL) не се стартира
    app.dependency_overrides.clear()


def run_body(**overrides):
    return {"ticker": "BTC-USD", "start_date": "2021-01-01", "end_date": "2025-12-31",
            "initial_capital": 10000.0, "t_buy": 0.60, "t_sell": 0.45,
            "max_position_pct": 0.25, "max_drawdown_limit": 0.15, **overrides}


# ------------------------------------------------------------------------------
# Тикери / CORS / health
# ------------------------------------------------------------------------------


def test_tickers_split_10_crypto_10_stocks(client):
    body = client.get(f"{API}/tickers").json()
    assert body["total"] == 20
    assert [a["ticker"] for a in body["crypto"]][:2] == ["BTC-USD", "ETH-USD"]
    assert len(body["crypto"]) == 10 and len(body["stocks"]) == 10
    assert {a["calendar_days"] for a in body["crypto"]} == {365}
    assert {a["asset_type"] for a in body["stocks"]} == {"STOCK"}


def test_assets_alias_flat_list(client):
    body = client.get(f"{API}/assets").json()
    assert len(body) == 20 and set(body[0]) == {"ticker", "name", "asset_type", "calendar_days"}


def test_cors_preflight_for_react_dev_server(client):
    r = client.options(f"{API}/simulations/run", headers={
        "Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


def test_cors_rejects_unknown_origin(client):
    r = client.get(f"{API}/tickers", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


def test_health(client):
    assert client.get(f"{API}/health").json() == {"status": "ok", "database": True}


# ------------------------------------------------------------------------------
# Пазарни данни
# ------------------------------------------------------------------------------


def test_market_data_candles_for_lightweight_charts(client, downloader):
    r = client.get(f"{API}/market-data/btc-usd",
                   params={"start_date": "2024-01-01", "end_date": "2024-01-31"})
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "BTC-USD" and body["count"] == 31
    assert set(body["candles"][0]) == {"time", "open", "high", "low", "close", "volume"}
    assert body["candles"][0]["time"] == "2024-01-01"
    client.get(f"{API}/market-data/BTC-USD")
    assert downloader.calls == 1  # второто четене е от кеша


@pytest.mark.parametrize("path,params", [
    ("DOGE-USD", {}),
    ("BTC-USD", {"start_date": "2020-01-01"}),
    ("BTC-USD", {"start_date": "2024-02-01", "end_date": "2024-01-01"}),
])
def test_market_data_invalid_input_is_422(client, path, params):
    r = client.get(f"{API}/market-data/{path}", params=params)
    assert r.status_code == 422
    assert r.json()["error"] in {"InvalidTickerError", "InvalidDateRangeError"}


def test_sync_single_ticker_then_cached(client, downloader):
    first = client.post(f"{API}/market-data/sync", json={"ticker": "ETH-USD"}).json()
    assert first["results"][0]["status"] == "synced" and first["results"][0]["rows"] == 1826
    second = client.post(f"{API}/market-data/sync", json={"ticker": "ETH-USD"}).json()
    assert second["results"][0]["status"] == "cached"
    forced = client.post(f"{API}/market-data/sync",
                         json={"ticker": "ETH-USD", "force_refresh": True}).json()
    assert forced["results"][0]["status"] == "synced" and downloader.calls == 2


def test_sync_all_twenty_reports_per_ticker(client, downloader):
    body = client.post(f"{API}/market-data/sync").json()
    assert len(body["results"]) == 20
    assert body["succeeded"] == 20 and body["failed"] == 0


def test_sync_invalid_ticker_is_422(client):
    assert client.post(f"{API}/market-data/sync", json={"ticker": "NVDA"}).status_code == 422


# ------------------------------------------------------------------------------
# Симулации
# ------------------------------------------------------------------------------


def test_run_simulation_end_to_end(client, api_sf, mock_backtest_response):
    r = client.post(f"{API}/simulations/run", json=run_body())
    assert r.status_code == 201, r.text
    body = r.json()

    # Структура 1:1 с contracts/mock_backtest_response.json (+ технически полета)
    for key, value in mock_backtest_response.items():
        assert key in body
        if isinstance(value, dict):
            assert set(body[key]) == set(value)
    assert set(body["portfolio_metrics"]["ai_strategy"]) == \
        set(mock_backtest_response["portfolio_metrics"]["ai_strategy"])
    assert set(body["audit_logs"][0]) == set(mock_backtest_response["audit_logs"][0])
    assert set(body["equity_curve"][0]) == set(mock_backtest_response["equity_curve"][0])
    assert body["request_params"] == {**run_body(), "initial_capital": 10000.0}

    assert body["execution_time_ms"] < 5_000
    assert r.headers["x-execution-time-ms"] == str(body["execution_time_ms"])
    assert body["engines"] == {"ai_engine": "fallback", "backtester": "fallback"}

    with api_sf() as s:
        assert s.scalar(select(func.count()).select_from(SimulationRun)) == 1
        assert s.scalar(select(func.count()).select_from(AuditLog)) == len(body["audit_logs"])

    sim_id = body["simulation_id"]
    got = client.get(f"{API}/simulations/{sim_id}").json()
    for key in ("request_params", "ml_metrics", "portfolio_metrics", "equity_curve",
                "audit_logs", "execution_time_ms"):
        assert got[key] == body[key]

    logs = client.get(f"{API}/simulations/{sim_id}/audit-logs").json()
    assert logs["count"] == len(body["audit_logs"]) and logs["audit_logs"] == body["audit_logs"]
    executed = client.get(f"{API}/simulations/{sim_id}/audit-logs",
                          params={"status": "EXECUTED"}).json()
    assert all(e["status"] == "EXECUTED" for e in executed["audit_logs"])


def test_backtest_alias_and_max_exposure(client):
    r = client.post(f"{API}/backtest", json={"ticker": "BTC-USD", "max_exposure": 0.1,
                                             "start_date": "2023-01-01"})
    assert r.status_code == 201, r.text
    assert r.json()["request_params"]["max_position_pct"] == 0.1


def test_run_is_reproducible_over_api(client):
    keys = ("ml_metrics", "portfolio_metrics", "equity_curve", "audit_logs")
    a = client.post(f"{API}/simulations/run", json=run_body(start_date="2023-01-01")).json()
    b = client.post(f"{API}/simulations/run", json=run_body(start_date="2023-01-01")).json()
    assert all(a[k] == b[k] for k in keys)


@pytest.mark.parametrize("overrides", [
    {"ticker": "DOGE-USD"}, {"t_buy": 0.40}, {"t_sell": 0.70}, {"max_position_pct": 0.5},
    {"start_date": "2020-01-01"}, {"start_date": "2025-12-31", "end_date": "2025-01-01"},
    {"max_drawdown_limit": 0},
])
def test_run_invalid_params_is_422(client, overrides):
    assert client.post(f"{API}/simulations/run", json=run_body(**overrides)).status_code == 422


def test_data_validation_error_is_422_and_nothing_is_written(api_sf):
    """> 3 последователни липсващи дни -> DataValidationError -> HTTP 422."""
    raw = random_walk()
    raw.iloc[200:204] = np.nan
    app.dependency_overrides[get_db_session_factory] = lambda: api_sf
    app.dependency_overrides[get_market_data_service] = \
        lambda: MarketDataService(api_sf, downloader=FakeDownloader(raw))
    try:
        r = TestClient(app).post(f"{API}/simulations/run", json=run_body())
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "DataValidationError" and body["details"]["gap_length"] == 4
    with api_sf() as s:
        assert s.scalar(select(func.count()).select_from(SimulationRun)) == 0


def test_unknown_simulation_is_404(client):
    missing = uuid.uuid4()
    assert client.get(f"{API}/simulations/{missing}").status_code == 404
    assert client.get(f"{API}/simulations/{missing}/audit-logs").status_code == 404
    assert client.get(f"{API}/simulations/not-a-uuid").status_code == 422


def test_openapi_schema_is_generated(client):
    paths = client.get("/openapi.json").json()["paths"]
    for p in ("/tickers", "/assets", "/market-data/sync", "/market-data/{ticker}",
              "/simulations/run", "/backtest", "/simulations/{simulation_id}",
              "/simulations/{simulation_id}/audit-logs"):
        assert f"{API}{p}" in paths
