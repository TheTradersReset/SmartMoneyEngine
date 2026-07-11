"""Tests for institutional blueprint forward validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.institutional_blueprint_forward_validation_research import (
    MIN_SIGNAL_SAMPLES,
    TOP_BLUEPRINT_COUNT,
    BlueprintForwardValidationError,
    InstitutionalBlueprintForwardValidationResearch,
    generate_blueprint_forward_validation_report,
)


def _discovery_report() -> dict:
    return {
        "top_20_bullish_momentum_blueprints": [
            {
                "blueprint": "Displacement:Weak -> BOS -> Zone:Discount",
                "direction": "bullish",
                "reliability_score": 85.0,
                "rank": 1,
            }
        ],
        "top_20_bearish_momentum_blueprints": [
            {
                "blueprint": "BOS -> Displacement:Weak -> FVG",
                "direction": "bearish",
                "reliability_score": 90.0,
                "rank": 1,
            }
        ],
    }


def test_constants() -> None:
    assert TOP_BLUEPRINT_COUNT == 10
    assert MIN_SIGNAL_SAMPLES == 100


def test_classify_blueprint_production_ready() -> None:
    result = InstitutionalBlueprintForwardValidationResearch._classify_blueprint(
        signals=120,
        win_rate_pct=45.0,
        expectancy=80.0,
        profit_factor=2.0,
    )
    assert result == "Production Ready"


def test_classify_blueprint_reject_low_samples() -> None:
    result = InstitutionalBlueprintForwardValidationResearch._classify_blueprint(
        signals=50,
        win_rate_pct=60.0,
        expectancy=80.0,
        profit_factor=2.0,
    )
    assert result == "Reject"


def test_matches_blueprint() -> None:
    required = ("Displacement:Weak", "BOS", "Zone:Discount")
    active = ("Displacement:Weak", "BOS", "Zone:Discount", "Level:Strong")
    assert InstitutionalBlueprintForwardValidationResearch._matches_blueprint(required, active)
    assert not InstitutionalBlueprintForwardValidationResearch._matches_blueprint(
        required,
        ("Displacement:Weak", "BOS"),
    )


def test_load_blueprints(tmp_path: Path) -> None:
    discovery_path = tmp_path / "institutional_expansion_trigger_discovery.json"
    discovery_path.write_text(json.dumps(_discovery_report()), encoding="utf-8")
    engine = InstitutionalBlueprintForwardValidationResearch(
        symbols=("NIFTY50",),
        discovery_report_path=discovery_path,
    )
    bullish, bearish = engine._load_blueprints()
    assert len(bullish) == 1
    assert len(bearish) == 1
    assert bullish[0].blueprint_id == "bullish_bp_01"
    assert bullish[0].signal_side == "BUY"


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-01-01", "end_date": "2026-01-01", "research_window_days": 365}',
        encoding="utf-8",
    )
    discovery_path = tmp_path / "institutional_expansion_trigger_discovery.json"
    discovery_path.write_text(json.dumps(_discovery_report()), encoding="utf-8")
    destination = tmp_path / "institutional_blueprint_forward_validation.json"

    sample_outcome = {
        "symbol": "NIFTY50",
        "timeframe": "5M",
        "timestamp": "2026-01-02 09:15:00+05:30",
        "signal_bar": 120,
        "blueprint_id": "bullish_bp_01",
        "blueprint": "Displacement:Weak -> BOS -> Zone:Discount",
        "blueprint_score": 85.0,
        "signal_side": "BUY",
        "direction": "bullish",
        "entry_price": 100.0,
        "stop_price": 98.0,
        "target_1r": 102.0,
        "target_2r": 104.0,
        "target_3r": 106.0,
        "target_4_opposite_liquidity": 110.0,
        "risk_points": 2.0,
        "hit_1r": True,
        "hit_2r": True,
        "hit_3r": False,
        "hit_4r": False,
        "hit_5r": False,
        "stop_hit": False,
        "hit_opposite_liquidity": True,
        "mfe_points": 8.0,
        "mae_points": 1.0,
        "time_to_1r_bars": 2,
        "time_to_2r_bars": 4,
        "time_to_3r_bars": None,
        "time_to_stop_bars": None,
        "time_to_target_bars": 6,
        "realized_pnl_points": 10.0,
        "realized_rr": 5.0,
        "win": True,
        "is_false_signal": False,
        "filter_context": {"rsi": 35.0, "session": "Opening"},
    }

    def _fake_scan(
        self: InstitutionalBlueprintForwardValidationResearch,
        metadata: dict,
        blueprints: list,
    ) -> list:
        del metadata, blueprints
        from src.research.institutional_blueprint_forward_validation_research import BlueprintSignalOutcome

        return [BlueprintSignalOutcome(**sample_outcome)]

    monkeypatch.setattr(
        InstitutionalBlueprintForwardValidationResearch,
        "_scan_history",
        _fake_scan,
    )

    report = generate_blueprint_forward_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
        discovery_report_path=discovery_path,
        symbols=("NIFTY50",),
    )
    assert destination.exists()
    assert report.trade_construction["entry"] == "Blueprint confirmation candle close"
    assert report.trade_construction["t4"] == "Opposite liquidity pool"
    assert "filter_discovery" in report.as_dict()
