"""Tests for Tier-2 regime classification research."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.research.tier2_regime_classification_research import (
    Tier2Regime,
    Tier2RegimeClassificationError,
    Tier2RegimeClassificationResearch,
    generate_tier2_regime_classification_report,
)
from src.research.tiered_signal_framework_research import TierSignal


def _sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date": pd.date_range(
                "2026-01-02 09:15",
                periods=30,
                freq="5min",
                tz="Asia/Kolkata",
            ).astype(str),
            "Open": [100.0] * 30,
            "High": [101.0] * 30,
            "Low": [99.0] * 30,
            "Close": [100.5] * 30,
            "Volume": [1000] * 30,
            "Trend": ["BULLISH"] * 30,
            "Trend_Strength": [2] * 30,
            "Bullish_BOS": [pd.NA] * 30,
            "Bearish_BOS": [pd.NA] * 30,
            "Bullish_CHOCH": [pd.NA] * 30,
            "Bearish_CHOCH": [pd.NA] * 30,
            "Buy_Liquidity_Sweep": [pd.NA] * 30,
            "Sell_Liquidity_Sweep": [1.0] + [pd.NA] * 29,
        }
    )


def test_htf_opposes() -> None:
    engine = Tier2RegimeClassificationResearch()
    assert engine._htf_opposes("bullish", "BEARISH")
    assert engine._htf_opposes("bearish", "BULLISH")
    assert not engine._htf_opposes("bullish", "BULLISH")


def test_classify_liquidity_reversal() -> None:
    engine = Tier2RegimeClassificationResearch()
    frame = _sample_frame()
    signal = TierSignal(
        tier="tier_2",
        timeframe="5M",
        direction="bullish",
        bos_bar=10,
        bos_timestamp=str(frame.iloc[10]["Date"]),
        choch_bar=5,
        displacement_bar=3,
    )
    regime = engine.classify_regime(
        frame,
        signal,
        session="Midday",
        htf_1h="BULLISH",
        htf_4h="BULLISH",
        htf_1d="BULLISH",
    )
    assert regime == Tier2Regime.LIQUIDITY_REVERSAL.value


def test_regime_metrics_empty() -> None:
    engine = Tier2RegimeClassificationResearch()
    metrics = engine._regime_metrics(Tier2Regime.TREND_CONTINUATION.value, [])
    assert metrics.signals == 0
    assert metrics.expectancy == 0.0


def test_missing_metadata_raises() -> None:
    with pytest.raises(Tier2RegimeClassificationError):
        generate_tier2_regime_classification_report(filter_report_path=Path("missing.json"))


@pytest.mark.integration
def test_full_report_if_metadata_exists() -> None:
    project_root = Path(__file__).resolve().parents[1]
    filter_report = project_root / "outputs" / "research" / "filter_research_report.json"
    if not filter_report.exists():
        pytest.skip("Filter research report not available.")

    metadata = json.loads(filter_report.read_text(encoding="utf-8"))
    report = Tier2RegimeClassificationResearch(timeframes=("5M",)).run(metadata)
    assert report.total_signals >= 0
    assert len(report.regime_metrics) == 5
