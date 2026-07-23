"""CLI for Replay Analytics Engine."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.core.logger import logger
from src.replay_analytics.engine import DEFAULT_OUTPUT_DIR, DEFAULT_REPLAY_DB, ReplayAnalyticsEngine


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.replay_analytics.cli",
        description=(
            "Replay Analytics Engine — reads an existing replay SQLite DB and writes "
            "analytics_summary.json, analytics_report.csv, analytics_report.html."
        ),
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_REPLAY_DB,
        help=f"Replay SQLite path (default: {DEFAULT_REPLAY_DB})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    engine = ReplayAnalyticsEngine(db_path=args.db, output_dir=args.out)
    try:
        artifacts = engine.run()
        summary = artifacts.report["replay_summary"]
        print(
            f"[ANALYTICS COMPLETE] window={summary['replay_window_start']}..{summary['replay_window_end']} "
            f"candles={summary['total_candles']} decisions={summary['total_decisions']}",
            flush=True,
        )
        return 0
    except Exception as exc:
        logger.exception("Replay analytics failed.")
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
