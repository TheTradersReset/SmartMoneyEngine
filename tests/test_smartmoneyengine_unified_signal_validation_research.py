"""Tests for SmartMoneyEngine unified signal validation research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.smartmoneyengine_unified_signal_validation_research import (
    BUY_REQUIRED_TAGS,
    MIN_LEVEL_TESTS,
    SELL_REQUIRED_TAGS,
    SIGNAL_VOLUME_THRESHOLDS,
    SmartMoneyEngineUnifiedSignalValidationResearch,
    UnifiedSignalOutcome,
    UnifiedSignalValidationError,
    generate_unified_signal_validation_report,
)


def _measurements(
    *,
    tests: int = 5,
    engulfing: bool = True,
    direction: str = "bullish",
) -> dict:
    zone = "Discount" if direction == "bullish" else "Premium"
    return {
        "support_resistance": {"number_of_tests": tests},
        "expansion_trigger_candle": {
            "engulfing": engulfing,
            "marubozu": False,
            "hammer": direction == "bullish",
            "shooting_star": direction == "bearish",
            "morning_star": False,
            "evening_star": False,
            "bullish_harami": False,
            "bearish_harami": False,
            "body_pct": 60.0,
            "volume_expansion_ratio": 1.5,
        },
        "structure": {"premium_discount": zone},
        "liquidity": {},
    }


def test_constants() -> None:
    assert MIN_LEVEL_TESTS == 3
    assert len(BUY_REQUIRED_TAGS) == 6
    assert len(SELL_REQUIRED_TAGS) == 6
    assert SIGNAL_VOLUME_THRESHOLDS == (20, 30, 40, 50)


def test_matches_tags() -> None:
    active = ("Liquidity Grab", "Failed Breakdown", "Displacement:Weak", "CHOCH", "BOS", "Zone:Discount")
    assert SmartMoneyEngineUnifiedSignalValidationResearch._matches_tags(BUY_REQUIRED_TAGS, active)
    assert not SmartMoneyEngineUnifiedSignalValidationResearch._matches_tags(
        BUY_REQUIRED_TAGS,
        ("Liquidity Grab", "BOS"),
    )


def test_is_confirmation_candle() -> None:
    trigger = {"engulfing": True, "body_pct": 40.0, "volume_expansion_ratio": 1.0}
    assert SmartMoneyEngineUnifiedSignalValidationResearch._is_confirmation_candle(trigger, "bullish")
    trigger = {"engulfing": False, "body_pct": 60.0, "volume_expansion_ratio": 1.2}
    assert SmartMoneyEngineUnifiedSignalValidationResearch._is_confirmation_candle(trigger, "bearish")


def test_matches_unified_buy_blueprint() -> None:
    engine = SmartMoneyEngineUnifiedSignalValidationResearch(symbols=("NIFTY50",))
    tags = BUY_REQUIRED_TAGS + ("Level Tests",)
    assert engine._matches_unified_blueprint(engine.buy_blueprint, tags, _measurements(tests=4))
    assert not engine._matches_unified_blueprint(engine.buy_blueprint, tags, _measurements(tests=2))
    assert engine._matches_unified_blueprint(
        engine.buy_blueprint,
        tags,
        _measurements(tests=5, engulfing=False),
    )


def test_matches_unified_sell_blueprint() -> None:
    engine = SmartMoneyEngineUnifiedSignalValidationResearch(symbols=("NIFTY50",))
    tags = SELL_REQUIRED_TAGS
    assert engine._matches_unified_blueprint(
        engine.sell_blueprint,
        tags,
        _measurements(tests=1, direction="bearish"),
    )


def test_classify_engine() -> None:
    result = SmartMoneyEngineUnifiedSignalValidationResearch._classify_engine(
        signals=100,
        win_rate_pct=45.0,
        expectancy=80.0,
        profit_factor=2.0,
        hit_1r_rate_pct=70.0,
    )
    assert result == "Production Ready"


def test_assess_signal_volume() -> None:
    engine = SmartMoneyEngineUnifiedSignalValidationResearch(symbols=("NIFTY50",))
    from src.research.smartmoneyengine_unified_signal_validation_research import UnifiedSignalMetrics

    overall = UnifiedSignalMetrics(
        scope="overall",
        signal_side=None,
        total_signals=500,
        signals_per_month=42.0,
        signals_per_week=9.7,
        hit_1r_rate_pct=75.0,
        hit_2r_rate_pct=60.0,
        hit_3r_rate_pct=45.0,
        win_rate_pct=50.0,
        profit_factor=2.0,
        expectancy=80.0,
        average_rr=1.5,
        net_points=4000.0,
        maximum_drawdown_points=200.0,
        high_quality_signals=300,
        high_quality_rate_pct=60.0,
        classification="Production Ready",
    )
    per_symbol = [
        UnifiedSignalMetrics(
            scope="NIFTY50",
            signal_side=None,
            total_signals=200,
            signals_per_month=17.0,
            signals_per_week=4.0,
            hit_1r_rate_pct=70.0,
            hit_2r_rate_pct=55.0,
            hit_3r_rate_pct=40.0,
            win_rate_pct=45.0,
            profit_factor=1.8,
            expectancy=60.0,
            average_rr=1.2,
            net_points=1200.0,
            maximum_drawdown_points=100.0,
            high_quality_signals=120,
            high_quality_rate_pct=60.0,
            classification="Production Ready",
        ),
    ]
    assessment = engine._assess_signal_volume(overall, per_symbol)
    assert assessment["20_plus_per_month"]["realistic"] is True
    assert assessment["50_plus_per_month"]["realistic"] is False


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text(
        '{"start_date": "2025-01-01", "end_date": "2026-01-01", "research_window_days": 365}',
        encoding="utf-8",
    )
    destination = tmp_path / "smartmoneyengine_unified_signal_validation.json"

    sample = UnifiedSignalOutcome(
        symbol="NIFTY50",
        timeframe="5M",
        timestamp="2026-01-02 09:15:00+05:30",
        signal_bar=120,
        blueprint_id="unified_buy",
        blueprint="test",
        signal_side="BUY",
        direction="bullish",
        entry_price=100.0,
        stop_price=98.0,
        target_1r=102.0,
        target_2r=104.0,
        target_3r=106.0,
        risk_points=2.0,
        hit_1r=True,
        hit_2r=True,
        hit_3r=False,
        stop_hit=False,
        realized_pnl_points=10.0,
        realized_rr=5.0,
        win=True,
        high_quality=True,
        filter_context={"session": "Opening"},
    )

    def _fake_scan(
        self: SmartMoneyEngineUnifiedSignalValidationResearch,
        metadata: dict,
    ) -> list[UnifiedSignalOutcome]:
        del metadata
        return [sample]

    monkeypatch.setattr(
        SmartMoneyEngineUnifiedSignalValidationResearch,
        "_scan_history",
        _fake_scan,
    )

    report = generate_unified_signal_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
        symbols=("NIFTY50",),
    )
    assert report.total_signals == 1
    assert destination.exists()
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["buy_blueprint"]["blueprint_id"] == "unified_buy"
    assert "signal_volume_assessment" in payload


def test_generate_report_missing_filter() -> None:
    with pytest.raises(UnifiedSignalValidationError):
        generate_unified_signal_validation_report(filter_report_path=Path("missing.json"))


def test_estimate_runtime_structure() -> None:
    engine = SmartMoneyEngineUnifiedSignalValidationResearch(symbols=("NIFTY50",), timeframes=("5M",))
    metadata = {"start_date": "2025-07-03", "end_date": "2026-07-03", "research_window_days": 365}
    estimate = engine.estimate_runtime(metadata)
    assert estimate.total_seconds > 0
    assert estimate.total_frames == 1
    assert len(estimate.complexity_risks) >= 3
    assert any("O(N)" in risk["pattern"] for risk in estimate.complexity_risks)
    assert len(estimate.export_reuse) >= 1
