"""Smoke tests for BUY failure anatomy synthesis."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.buy_failure_anatomy_research import (
    PRECURSOR_EVENTS,
    SOURCE_EXPORTS,
    TAUTOLOGICAL_FEATURES,
    BuyFailureAnatomyResearch,
    _classify_move,
    _precursor_match,
)


def test_precursor_event_universe() -> None:
    assert "Gap Reversal" in PRECURSOR_EVENTS
    assert "Failed Breakdown" in PRECURSOR_EVENTS
    assert "Liquidity Grab" in PRECURSOR_EVENTS


def test_precursor_match_near_support() -> None:
    matched, precursors = _precursor_match(
        first_event="Gap Continuation",
        causal_events=[],
        near_support=True,
        origin_trigger=None,
    )
    assert matched is True
    assert "Near Support" in precursors


def test_classify_real_reversal() -> None:
    label = _classify_move({"move_size_points": 250, "duration_minutes": 120, "context_t60": {}})
    assert label == "Real Reversal"


def test_classify_dead_cat_bounce() -> None:
    label = _classify_move(
        {
            "move_size_points": 80,
            "duration_minutes": 60,
            "context_t60": {"htf_trend": "Strong Bearish"},
            "lde_outcome": "",
        },
    )
    assert label == "Dead Cat Bounce"


def test_generate_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    report_path = tmp_path / "buy_failure_anatomy.json"
    research = BuyFailureAnatomyResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["report_type"] == "BUY Failure Anatomy"
    assert "classification_summary" in payload
    assert "strongest_buy_discriminator" in payload
    assert payload["precursor_filter"]["cohort_size"] >= 1
    assert payload["classification_summary"]["total_classified"] >= 1
    assert len(payload["source_exports"]) == len(SOURCE_EXPORTS)
    assert payload["strongest_buy_discriminator"]["feature"] not in TAUTOLOGICAL_FEATURES
