"""
Run existing ReplayEngine day-by-day and append into databases.

- Reuses ReplayEngine (no duplicate signal path)
- Appends decisions/signals/candles into one shared replay SQLite DB
- Appends OHLCV bars into research_dataset.db
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

from src.core.logger import logger
from src.dataset_builder.schema import DEFAULT_RESEARCH_DATASET_DB
from src.dataset_builder.writer import ResearchDatasetWriter
from src.replay.data_feed import DEFAULT_HISTORY_CSV, HistoricalDataFeed, window_for_day
from src.replay.engine import ReplayEngine, ReplayResult

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNAL_DB = PROJECT_ROOT / "data" / "paper" / "replay_signals.db"
RESOLUTION = "5"


def _iter_days(start: date, end: date):
    current = start
    while current <= end:
        yield current
        current += timedelta(days=1)


def run_repeated_replay(
    *,
    start: date,
    end: date,
    csv_path: Path | str = DEFAULT_HISTORY_CSV,
    signal_db_path: Path | str = DEFAULT_SIGNAL_DB,
    research_db_path: Path | str = DEFAULT_RESEARCH_DATASET_DB,
    symbol: str = "NSE:NIFTY50-INDEX",
) -> list[ReplayResult]:
    """Replay each day in [start, end]; append into signal DB and research_dataset.db."""
    csv_path = Path(csv_path)
    signal_db_path = Path(signal_db_path)
    research_db_path = Path(research_db_path)

    feed = HistoricalDataFeed(csv_path=csv_path, symbol=symbol)
    feed.load()
    writer = ResearchDatasetWriter(research_db_path)
    results: list[ReplayResult] = []

    try:
        for day in _iter_days(start, end):
            window = window_for_day(day)
            candles = feed.replay_candles(window)
            if not candles:
                continue

            writer.upsert_bars(
                [
                    {
                        "symbol": symbol,
                        "resolution": RESOLUTION,
                        "timestamp": candle.timestamp.isoformat(),
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                    }
                    for candle in candles
                ],
            )

            engine = ReplayEngine(
                window=window,
                speed="unlimited",
                csv_path=csv_path,
                signal_db_path=signal_db_path,
                symbol=symbol,
                run_trade_validation=False,
            )
            try:
                result = engine.run()
                results.append(result)
                print(
                    f"[REPEATED REPLAY] {day.isoformat()} "
                    f"candles={result.candles_fed} state={result.state.value}",
                    flush=True,
                )
            finally:
                engine.close()
    finally:
        writer.close()

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ReplayEngine repeatedly; append into databases.")
    parser.add_argument("--from", dest="date_from", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--csv", type=Path, default=DEFAULT_HISTORY_CSV)
    parser.add_argument("--signal-db", type=Path, default=DEFAULT_SIGNAL_DB)
    parser.add_argument("--research-db", type=Path, default=DEFAULT_RESEARCH_DATASET_DB)
    args = parser.parse_args(argv)

    start = date.fromisoformat(args.date_from)
    end = date.fromisoformat(args.date_to)
    try:
        results = run_repeated_replay(
            start=start,
            end=end,
            csv_path=args.csv,
            signal_db_path=args.signal_db,
            research_db_path=args.research_db,
        )
        total_candles = sum(r.candles_fed for r in results)
        print(
            f"[REPEATED REPLAY COMPLETE] days={len(results)} candles={total_candles} "
            f"signal_db={args.signal_db} research_db={args.research_db}",
            flush=True,
        )
        return 0
    except Exception as exc:
        logger.exception("Repeated replay failed.")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
