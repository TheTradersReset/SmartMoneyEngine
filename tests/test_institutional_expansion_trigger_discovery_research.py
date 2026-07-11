"""Tests for institutional expansion trigger discovery research."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.research.institutional_expansion_trigger_discovery_research import (
    MIN_BLUEPRINT_SAMPLES,
    MOVE_THRESHOLDS,
    PRE_EXPANSION_LOOKBACK,
    ExpansionTriggerDiscoveryError,
    InstitutionalExpansionTriggerDiscoveryResearch,
    generate_expansion_trigger_discovery_report,
)
from src.research.liquidity_move_reconstruction_research import _CheapMoveCandidate


def _pipeline_frame(length: int = 200) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-01-02 09:15:00+05:30")
    for index in range(length):
        price = 100.0 + index * 0.5
        timestamp = base + pd.Timedelta(minutes=5 * index)
        rows.append(
            {
                "Date": timestamp.isoformat(),
                "Open": price,
                "High": price + 1.0,
                "Low": price - 0.8,
                "Close": price + 0.2,
                "Volume": 100000,
                "Swing_High": pd.NA,
                "Swing_Low": pd.NA,
                "Buy_Side_Liquidity": price + 5,
                "Sell_Side_Liquidity": price - 5,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Bullish_OB_High": pd.NA,
                "Bearish_OB_High": pd.NA,
            }
        )
    frame = pd.DataFrame(rows)
    for index in range(120, length):
        frame.at[index, "High"] = frame.at[119, "Close"] + (index - 119) * 3.0
        frame.at[index, "Low"] = frame.at[index, "High"] - 0.4
        frame.at[index, "Close"] = frame.at[index, "High"] - 0.1
    return frame


def test_constants() -> None:
    assert MOVE_THRESHOLDS == (100, 200, 300, 500)
    assert PRE_EXPANSION_LOOKBACK == 100
    assert MIN_BLUEPRINT_SAMPLES == 100


def test_two_proportion_p_value_identical() -> None:
    p_value = InstitutionalExpansionTriggerDiscoveryResearch._two_proportion_p_value(50, 100, 50, 100)
    assert p_value == 1.0


def test_blueprint_keys_includes_combinations() -> None:
    keys = InstitutionalExpansionTriggerDiscoveryResearch._blueprint_keys(
        ("CHOCH", "BOS", "High Compression"),
    )
    assert "CHOCH" in keys
    assert "BOS -> CHOCH" in keys
    assert "BOS -> High Compression" in keys
    assert "BOS -> CHOCH -> High Compression" in keys


def test_build_blueprint_tags() -> None:
    engine = InstitutionalExpansionTriggerDiscoveryResearch(symbols=("NIFTY50",))
    measurements = {
        "support_resistance": {
            "failed_breakdown_count": 2,
            "failed_breakout_count": 0,
            "number_of_tests": 5,
            "round_number_proximity": True,
            "pdh_interactions": 1,
            "pdl_interactions": 0,
            "pwh_interactions": 0,
            "pwl_interactions": 0,
            "monthly_high_interactions": 0,
            "monthly_low_interactions": 0,
            "level_strength_category": "Strong",
        },
        "absorption": {
            "absorption_candles_at_support": 3,
            "absorption_candles_at_resistance": 0,
            "rejection_wick_count": 6,
        },
        "liquidity": {
            "both_side_sweeps": 1,
            "sell_side_sweeps": 0,
            "buy_side_sweeps": 0,
            "liquidity_grab_count": 1,
            "false_move_count": 4,
        },
        "compression": {
            "volatility_compression_score": 55,
            "consolidation_duration_bars": 35,
            "nr7_count": 3,
        },
        "expansion_trigger_candle": {
            "engulfing": True,
            "marubozu": False,
            "hammer": True,
            "shooting_star": False,
            "morning_star": False,
            "evening_star": False,
            "volume_expansion_ratio": 1.8,
            "displacement_strength": "Strong",
        },
        "structure": {
            "choch_count": 1,
            "bos_count": 1,
            "fvg_count": 1,
            "ob_count": 1,
            "premium_discount": "Discount",
            "htf_alignment": True,
        },
    }
    tags = engine._build_blueprint_tags(measurements, "bullish")
    assert "Failed Breakdown x2+" in tags
    assert "Both-Side Sweep" in tags
    assert "High Compression" in tags


def test_analyze_move_structure() -> None:
    engine = InstitutionalExpansionTriggerDiscoveryResearch(symbols=("NIFTY50",))
    frame = _pipeline_frame()
    enriched = engine.context_builder.enrich(frame)
    calendar = engine.liquidity_map_engine._attach_calendar_levels(frame)
    intel = engine.intelligence_engine.enrich(frame)
    candidate = _CheapMoveCandidate(
        start_bar=120,
        expansion_bar=180,
        direction="bullish",
        magnitude=150.0,
    )
    record = engine._analyze_move("NIFTY50", frame, enriched, calendar, intel, candidate, "5M")
    assert record.hit_100_plus
    assert "support_resistance" in record.measurements
    assert "momentum_outcome" in record.measurements
    assert record.measurements["momentum_outcome"]["move_size_points"] == pytest.approx(150.0)


def test_generate_report(tmp_path: Path) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-01-01", "end_date": "2026-01-01", "research_window_days": 365}',
        encoding="utf-8",
    )
    destination = tmp_path / "institutional_expansion_trigger_discovery.json"
    report = generate_expansion_trigger_discovery_report(
        report_path=destination,
        filter_report_path=filter_report,
        symbols=("NIFTY50",),
    )
    assert destination.exists()
    assert report.pre_expansion_lookback_bars == 100
    assert "measurement_categories" in report.as_dict()
    assert len(report.measurement_categories) == 7
