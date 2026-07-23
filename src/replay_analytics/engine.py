"""
Replay Analytics Engine.

Reads an existing replay SQLite database and writes JSON / CSV / HTML reports.
Does not modify signal engines, replay engine, or trade validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.logger import logger
from src.replay_analytics.analyzer import analyze_replay
from src.replay_analytics.exporters import export_csv, export_html, export_json
from src.replay_analytics.reader import ReplayAnalyticsReader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "replay_analytics"
DEFAULT_REPLAY_DB = PROJECT_ROOT / "data" / "paper" / "replay_smoke.db"


@dataclass(frozen=True)
class AnalyticsArtifacts:
    summary_json: Path
    report_csv: Path
    report_html: Path
    report: dict[str, Any]


class ReplayAnalyticsEngine:
    """Orchestrate read → analyze → export for one replay database."""

    def __init__(
        self,
        *,
        db_path: Path | str = DEFAULT_REPLAY_DB,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)

    def run(self) -> AnalyticsArtifacts:
        reader = ReplayAnalyticsReader(self.db_path)
        try:
            decisions = reader.fetch_decisions()
            signals = reader.fetch_signals()
            candle_count = reader.fetch_candle_count()
        finally:
            reader.close()

        report = analyze_replay(
            decisions=decisions,
            signals=signals,
            candle_count=candle_count,
            db_path=str(self.db_path),
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = export_json(report, self.output_dir / "analytics_summary.json")
        csv_path = export_csv(report, self.output_dir / "analytics_report.csv")
        html_path = export_html(report, self.output_dir / "analytics_report.html")

        logger.info(
            "Replay analytics written | decisions=%s | json=%s | csv=%s | html=%s",
            report["replay_summary"]["total_decisions"],
            json_path,
            csv_path,
            html_path,
        )
        print(
            f"[ANALYTICS] decisions={report['replay_summary']['total_decisions']} "
            f"BUY={report['replay_summary']['buy_signals']} "
            f"SELL={report['replay_summary']['sell_signals']} "
            f"NO_TRADE={report['replay_summary']['no_trade_count']}",
            flush=True,
        )
        print(f"[ANALYTICS] wrote {json_path}", flush=True)
        print(f"[ANALYTICS] wrote {csv_path}", flush=True)
        print(f"[ANALYTICS] wrote {html_path}", flush=True)

        return AnalyticsArtifacts(
            summary_json=json_path,
            report_csv=csv_path,
            report_html=html_path,
            report=report,
        )
