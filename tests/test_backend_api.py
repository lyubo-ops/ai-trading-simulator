"""FastAPI ендпойнти – функционални сценарии (валидацията е в test_api_validation.py)."""

from __future__ import annotations

import uuid

from sqlalchemy import func, select

from backend.models import AuditLog, SimulationRun
from tests.conftest import run_body

API = "/api/v1"


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
    assert len(downloader.calls) == 1  # второто четене е от кеша


def test_sync_single_ticker_then_cached(client, downloader):
    first = client.post(f"{API}/market-data/sync", json={"ticker": "ETH-USD"}).json()
    assert first["results"][0]["status"] == "synced" and first["results"][0]["rows"] == 1826
    second = client.post(f"{API}/market-data/sync", json={"ticker": "ETH-USD"}).json()
    assert second["results"][0]["status"] == "cached"
    forced = client.post(f"{API}/market-data/sync",
                         json={"ticker": "ETH-USD", "force_refresh": True}).json()
    assert forced["results"][0]["status"] == "synced" and len(downloader.calls) == 2


def test_sync_all_twenty_reports_per_ticker(client, downloader):
    body = client.post(f"{API}/market-data/sync").json()
    assert len(body["results"]) == 20
    assert body["succeeded"] == 20 and body["failed"] == 0


# ------------------------------------------------------------------------------
# Симулации
# ------------------------------------------------------------------------------


def test_run_simulation_end_to_end(client, session_factory, mock_backtest_response):
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

    with session_factory() as s:
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
