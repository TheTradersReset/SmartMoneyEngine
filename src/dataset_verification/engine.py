"""
Dataset Verification Engine.

Validates historical datasets before Replay. Never modifies source data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.core.logger import logger
from src.dataset_builder.schema import DEFAULT_RESEARCH_DATASET_DB
from src.dataset_verification.exporters import export_csv, export_html, export_json
from src.dataset_verification.health import score_health
from src.dataset_verification.loader import load_bars_from_csv, load_bars_from_db
from src.dataset_verification.validators import validate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "dataset_verification"

SourceKind = Literal["db", "csv"]


@dataclass(frozen=True)
class VerificationArtifacts:
    report_json: Path
    report_csv: Path
    report_html: Path
    report: dict[str, Any]


class DatasetVerificationEngine:
    """Orchestrate load → validate → score → export (read-only on source)."""

    def __init__(
        self,
        *,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        symbol: str = "NSE:NIFTY50-INDEX",
        resolution: str = "5",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.symbol = symbol
        self.resolution = resolution

    def verify_db(
        self,
        db_path: Path | str = DEFAULT_RESEARCH_DATASET_DB,
        *,
        start_timestamp: str | None = None,
        end_timestamp: str | None = None,
    ) -> VerificationArtifacts:
        frame = load_bars_from_db(
            db_path,
            symbol=self.symbol,
            resolution=self.resolution,
            start_timestamp=start_timestamp,
            end_timestamp=end_timestamp,
        )
        return self._run(frame, source=str(db_path))

    def verify_csv(self, csv_path: Path | str) -> VerificationArtifacts:
        frame = load_bars_from_csv(csv_path)
        return self._run(frame, source=str(csv_path))

    def _run(self, frame, *, source: str) -> VerificationArtifacts:
        report = validate_dataset(
            frame,
            symbol=self.symbol,
            resolution=self.resolution,
            source=source,
        )
        report["health"] = score_health(report)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        json_path = export_json(report, self.output_dir / "dataset_validation.json")
        csv_path = export_csv(report, self.output_dir / "dataset_validation.csv")
        html_path = export_html(report, self.output_dir / "dataset_validation.html")

        health = report["health"]
        logger.info(
            "Dataset verification complete | bars=%s | score=%s | verdict=%s",
            report["meta"]["bar_count"],
            health["health_score"],
            health["verdict"],
        )
        print(
            f"[DATASET VERIFY] bars={report['meta']['bar_count']} "
            f"score={health['health_score']} verdict={health['verdict']}",
            flush=True,
        )
        print(f"[DATASET VERIFY] wrote {json_path}", flush=True)
        print(f"[DATASET VERIFY] wrote {csv_path}", flush=True)
        print(f"[DATASET VERIFY] wrote {html_path}", flush=True)

        return VerificationArtifacts(
            report_json=json_path,
            report_csv=csv_path,
            report_html=html_path,
            report=report,
        )
