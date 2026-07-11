"""Tests for institutional setup classification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.signals.setup_classifier import (
    SetupClassificationEngine,
    SetupClassifier,
    SetupClassifierError,
    SetupDirection,
    SetupType,
    generate_setup_classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _empty_frame(length: int = 25) -> pd.DataFrame:
    rows = []
    for index in range(length):
        rows.append(
            {
                "Date": f"2026-01-02 09:{15 + (index % 60):02d}:00+05:30",
                "Open": 100.0 + index * 0.1,
                "High": 100.5 + index * 0.1,
                "Low": 99.5 + index * 0.1,
                "Close": 100.2 + index * 0.1,
                "Trend": "BULLISH",
                "Trend_Strength": 2,
                "Bullish_BOS": pd.NA,
                "Bearish_BOS": pd.NA,
                "Bullish_CHOCH": pd.NA,
                "Bearish_CHOCH": pd.NA,
                "Bullish_FVG_Top": pd.NA,
                "Bullish_FVG_Bottom": pd.NA,
                "Bearish_FVG_Top": pd.NA,
                "Bearish_FVG_Bottom": pd.NA,
                "Bullish_OB_High": pd.NA,
                "Bullish_OB_Low": pd.NA,
                "Bearish_OB_High": pd.NA,
                "Bearish_OB_Low": pd.NA,
                "Bullish_OB_Mitigated": pd.NA,
                "Bearish_OB_Mitigated": pd.NA,
                "Buy_Liquidity_Sweep": pd.NA,
                "Sell_Liquidity_Sweep": pd.NA,
                "Liquidity_Strength": 2,
                "Swing_High": pd.NA,
                "Swing_Low": pd.NA,
            }
        )
    return pd.DataFrame(rows)


def test_detect_liquidity_sweep_bos_bullish() -> None:
    frame = _empty_frame()
    frame.loc[10, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[12, "Bullish_BOS"] = 101.0

    classifier = SetupClassifier()
    setups = classifier.classify_bar(frame, 12)
    types = {setup.setup_type for setup in setups}

    assert SetupType.LIQUIDITY_SWEEP_BOS.value in types
    bullish = [setup for setup in setups if setup.direction == SetupDirection.BULLISH.value]
    assert bullish
    assert bullish[0].quality_score > 0
    assert bullish[0].trigger_bar == 12


def test_detect_choch_fvg_bullish() -> None:
    frame = _empty_frame()
    frame.loc[11, "Bullish_FVG_Top"] = 101.0
    frame.loc[11, "Bullish_FVG_Bottom"] = 100.0
    frame.loc[12, "Bullish_CHOCH"] = 100.5

    classifier = SetupClassifier()
    setups = classifier.classify_bar(frame, 12)
    assert any(setup.setup_type == SetupType.CHOCH_FVG.value for setup in setups)


def test_detect_fresh_ob_retest_bullish() -> None:
    frame = _empty_frame()
    frame.loc[12, "Bullish_OB_High"] = 100.5
    frame.loc[12, "Bullish_OB_Low"] = 99.8
    frame.loc[12, "Low"] = 100.0
    frame.loc[12, "Close"] = 100.2

    classifier = SetupClassifier()
    setups = classifier.classify_bar(frame, 12)
    assert any(setup.setup_type == SetupType.FRESH_OB_RETEST.value for setup in setups)


def test_detect_continuation_bos_bullish() -> None:
    frame = _empty_frame()
    frame.loc[12, "Trend"] = "BULLISH"
    frame.loc[12, "Bullish_BOS"] = 101.0

    classifier = SetupClassifier()
    setups = classifier.classify_bar(frame, 12)
    assert any(setup.setup_type == SetupType.CONTINUATION_BOS.value for setup in setups)


def test_detect_liquidity_grab_fvg_reclaim_bullish() -> None:
    frame = _empty_frame()
    frame.loc[10, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[11, "Bullish_FVG_Top"] = 101.0
    frame.loc[11, "Bullish_FVG_Bottom"] = 100.0
    frame.loc[12, "Close"] = 100.5

    classifier = SetupClassifier()
    setups = classifier.classify_bar(frame, 12)
    assert any(
        setup.setup_type == SetupType.LIQUIDITY_GRAB_FVG_RECLAIM.value for setup in setups
    )


def test_backtest_simulator_wins_on_target() -> None:
    frame = _empty_frame(length=10)
    frame.loc[5, "Low"] = 99.0
    frame.loc[6, "High"] = 105.0

    classifier = SetupClassifier()
    frame.loc[5, "Sell_Liquidity_Sweep"] = 98.5
    frame.loc[5, "Bullish_BOS"] = 101.0
    setups = classifier.classify_bar(frame, 5)
    assert setups

    engine = SetupClassificationEngine()
    report = engine.run(frame)
    assert report.total_setups >= 1
    assert report.setup_metrics
    assert all("frequency" in metric for metric in report.setup_metrics)


def test_analyze_rejects_missing_columns() -> None:
    engine = SetupClassificationEngine()
    with pytest.raises(SetupClassifierError):
        engine.run(pd.DataFrame({"Close": [1.0]}))


def test_generate_report_writes_json(tmp_path: Path) -> None:
    frame = _empty_frame()
    frame.loc[10, "Sell_Liquidity_Sweep"] = 99.0
    frame.loc[12, "Bullish_BOS"] = 101.0

    csv_path = tmp_path / "pipeline.csv"
    report_path = tmp_path / "setup_classification_report.json"
    frame.to_csv(csv_path, index=False)

    report = generate_setup_classification_report(
        pipeline_csv=csv_path,
        report_path=report_path,
        symbol="NIFTY50",
        timeframe="5",
    )

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["symbol"] == "NIFTY50"
    assert len(payload["setup_metrics"]) == 5
    assert report.total_setups >= 1


@pytest.mark.integration
def test_real_pipeline_setup_classification_if_present() -> None:
    pipeline_csv = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
    if not pipeline_csv.exists():
        pytest.skip("Real pipeline CSV not available.")

    engine = SetupClassificationEngine(symbol="NIFTY50", timeframe="5")
    report = engine.run_from_csv(pipeline_csv)

    assert report.total_candles > 1000
    assert report.total_setups > 0
    assert len(report.setup_metrics) == 5
