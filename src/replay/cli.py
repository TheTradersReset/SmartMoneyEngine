"""CLI for Historical Replay Engine."""

from __future__ import annotations

import argparse
import signal
import sys
from datetime import date
from pathlib import Path

from src.core.logger import logger
from src.replay.data_feed import parse_iso_week, parse_year_month
from src.replay.engine import (
    ReplayEngine,
    build_engine_for_day,
    build_engine_for_month,
    build_engine_for_range,
    build_engine_for_week,
)


def _parse_date(text: str) -> date:
    return date.fromisoformat(text.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.replay.cli",
        description=(
            "Historical Replay Engine — feeds OHLCV candles through the "
            "existing Realtime Signal Pipeline (BUY_V3 / SELL_V6 unchanged)."
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--day", type=str, help="Single day YYYY-MM-DD")
    mode.add_argument("--week", type=str, help="ISO week YYYY-Www (e.g. 2026-W11)")
    mode.add_argument("--month", type=str, help="Calendar month YYYY-MM")
    mode.add_argument("--from", dest="date_from", type=str, help="Custom range start YYYY-MM-DD")
    parser.add_argument("--to", dest="date_to", type=str, help="Custom range end YYYY-MM-DD (with --from)")

    parser.add_argument(
        "--speed",
        type=str,
        default="unlimited",
        help="Replay speed: 1x, 5x, 10x, 100x, or unlimited (default: unlimited)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Historical OHLCV CSV (default: outputs/pipeline/NIFTY50_5m_pipeline.csv)",
    )
    parser.add_argument(
        "--signal-db",
        type=Path,
        default=None,
        help="SQLite path for replay signal persistence",
    )
    parser.add_argument(
        "--validation-db",
        type=Path,
        default=None,
        help="SQLite path for trade validation results",
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip Trade Validation Engine after candles",
    )
    return parser


def _build_engine(args: argparse.Namespace) -> ReplayEngine:
    kwargs = {
        "speed": args.speed,
        "csv_path": args.csv,
        "signal_db_path": args.signal_db,
        "validation_db_path": args.validation_db,
        "run_trade_validation": not args.no_validation,
    }
    if args.day:
        return build_engine_for_day(_parse_date(args.day), **kwargs)
    if args.week:
        year, week = parse_iso_week(args.week)
        return build_engine_for_week(year, week, **kwargs)
    if args.month:
        year, month = parse_year_month(args.month)
        return build_engine_for_month(year, month, **kwargs)
    if args.date_from:
        if not args.date_to:
            raise SystemExit("--to is required when using --from")
        return build_engine_for_range(_parse_date(args.date_from), _parse_date(args.date_to), **kwargs)
    raise SystemExit("No replay window specified.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = _build_engine(args)

    def _handle_sigint(_signum: int, _frame: object) -> None:
        print("\n[REPLAY] stop requested — finishing current candle…", flush=True)
        engine.stop()

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        result = engine.run()
        print(f"[REPLAY] state={result.state.value} candles={result.candles_fed} "
              f"signals={result.signals} decisions={result.decisions} "
              f"validations={result.validations}", flush=True)
        return 0 if result.state.value in ("COMPLETED", "STOPPED") else 1
    except KeyboardInterrupt:
        engine.stop()
        return 0
    except Exception as exc:
        logger.exception("Historical replay failed.")
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
