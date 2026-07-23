"""CLI for Dataset Verification Engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.logger import logger
from src.dataset_verification.engine import DEFAULT_OUTPUT_DIR, DatasetVerificationEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.dataset_verification.cli",
        description="Validate historical datasets before Replay (read-only; no repairs).",
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--db", type=Path, help="Path to research_dataset.db")
    src.add_argument("--csv", type=Path, help="Replay-compatible OHLCV CSV")
    parser.add_argument("--symbol", default="NSE:NIFTY50-INDEX")
    parser.add_argument("--resolution", default="5")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start", default=None, help="Optional start timestamp filter (DB only)")
    parser.add_argument("--end", default=None, help="Optional end timestamp filter (DB only)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = DatasetVerificationEngine(
        output_dir=args.out,
        symbol=args.symbol,
        resolution=args.resolution,
    )
    try:
        if args.csv is not None:
            artifacts = engine.verify_csv(args.csv)
        else:
            artifacts = engine.verify_db(
                args.db,
                start_timestamp=args.start,
                end_timestamp=args.end,
            )
        health = artifacts.report["health"]
        print(
            f"[DATASET VERIFY COMPLETE] score={health['health_score']} "
            f"band={health['band']} verdict={health['verdict']}",
            flush=True,
        )
        return 0 if health["health_score"] >= 90 else 2
    except Exception as exc:
        logger.exception("Dataset verification failed.")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
