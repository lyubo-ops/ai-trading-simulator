"""
Първоначално синхронизиране (seed) на 20-те тикера в таблицата `market_data`.

    docker compose up -d db
    python -m backend.scripts.seed_market_data                 # само липсващите тикери
    python -m backend.scripts.seed_market_data --force         # презаписва всички
    python -m backend.scripts.seed_market_data --tickers AAPL BTC-USD

Всеки тикер се записва в собствена транзакция: грешка при един (напр.
DataValidationError) не засяга останалите. Изходен код 1, ако поне един е неуспешен.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

from backend.config import ALL_TICKERS
from backend.database import check_connection, init_db
from backend.exceptions import BackendError
from backend.services.data_ingestion import MarketDataService


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=list(ALL_TICKERS),
                        choices=list(ALL_TICKERS), metavar="TICKER",
                        help="подмножество от 20-те тикера (по подразбиране всички)")
    parser.add_argument("--force", action="store_true",
                        help="изтегли наново и презапиши и вече кешираните тикери")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if not check_connection():
        print("Няма връзка с PostgreSQL (провери .env и `docker compose up -d db`).",
              file=sys.stderr)
        return 2
    init_db()  # идемпотентно: създава само липсващите таблици

    service = MarketDataService()
    failures: list[str] = []
    print(f"{'TICKER':<9} {'STATUS':<8} {'ROWS':>5}  {'FIRST':<10}  {'LAST':<10}  {'SEC':>5}")
    for ticker in args.tickers:
        t0 = time.perf_counter()
        try:
            if not args.force and service.is_cached(ticker):
                status, rows = "cached", service.cached_row_count(ticker)
                df = service.get_market_data(ticker)
            else:
                df = service.sync_ticker(ticker)
                status, rows = "synced", len(df)
            first, last = df["date"].iloc[0], df["date"].iloc[-1]
        except BackendError as exc:
            failures.append(ticker)
            print(f"{ticker:<9} {'FAILED':<8} {'-':>5}  {type(exc).__name__}: {exc}")
            continue
        print(f"{ticker:<9} {status:<8} {rows:>5}  {first:<10}  {last:<10}  "
              f"{time.perf_counter() - t0:>5.2f}")

    if failures:
        print(f"\nНеуспешни ({len(failures)}): {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nГотово: {len(args.tickers)} тикера.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
