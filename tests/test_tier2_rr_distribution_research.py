"""Tests for Tier-2 RR distribution research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.tier2_rr_distribution_research import (
    Tier2RrDistributionError,
    Tier2RrDistributionResearch,
    generate_tier2_rr_distribution_report,
)


def _sample_frame(rows: int = 40) -> pd.DataFrame:
    close = [100.0 + index * 0.5 for index in range(rows)]
    return pd.DataFrame(
        {
            "Date": pd.date_range(
                "2026-01-02 09:15",
                periods=rows,
                freq="5min",
                tz="Asia/Kolkata",
            ).astype(str),
            "Open": close,
            "High": [value + 1 for value in close],
            "Low": [value - 1 for value in close],
            "Close": close,
            "Volume": [1000] * rows,
        }
    )


def test_simulate_rr_profile_reaches_1r() -> None:
    engine = Tier2RrDistributionResearch()
    frame = _sample_frame()
    frame.loc[15, "High"] = 120.0
    profile = engine._simulate_rr_profile(
        frame,
        entry_bar=10,
        entry_price=105.0,
        stop_price=95.0,
        risk=10.0,
        direction="bullish",
        timeframe="5M",
    )
    assert profile["reached"][1]
    assert profile["mfe_points"] >= 15.0


def test_max_r_outcome_label() -> None:
    assert Tier2RrDistributionResearch._max_r_outcome_label(3, 3.2) == "3R"
    assert Tier2RrDistributionResearch._max_r_outcome_label(0, 0.2) == "0R"
    assert Tier2RrDistributionResearch._max_r_outcome_label(5, 6.0) == "5R+"


def test_missing_metadata_raises() -> None:
    with pytest.raises(Tier2RrDistributionError):
        generate_tier2_rr_distribution_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = Tier2RrDistributionResearch(timeframes=("5M",)).run(metadata)
    assert report.total_signals >= 0
    assert "1R" in report.rr_distribution_table
