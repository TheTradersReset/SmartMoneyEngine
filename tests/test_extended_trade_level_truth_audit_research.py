"""Tests for extended trade level truth audit research."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.research.extended_trade_level_truth_audit_research import (
    PREFERRED_WINDOWS,
    ExtendedTradeLevelTruthAuditResearch,
    _analyze_trade_level_window,
    _resolve_replay_windows,
    _summarize_core_metrics,
    generate_extended_trade_level_truth_audit_report,
)


def _buy_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-06 09:30:00+05:30",
        "bar": 100,
        "direction": "BUY",
        "entry": 23500.0,
        "stop_loss": 23450.0,
        "target_1": 23550.0,
        "target_2": 23600.0,
        "target_3": 23650.0,
        "bars_before_expansion": 10,
        "points_before_expansion": 12.5,
        "mfe_points": 80.0,
        "mae_points": 20.0,
        "trade_duration_bars": 40,
        "hit_1r": True,
        "hit_2r": True,
        "hit_3r": False,
        "win": True,
        "classification": "Real Reversal",
        "realized_pnl_points": 60.0,
        "signal_reason_stack": {
            "layer1": ["Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep"],
            "layer2": {"htf_trend": "Bullish", "vwap": "Reclaimed", "location": "Near Support"},
        },
        "layers": {
            "layer1": {
                "events_detected": ["Failed Breakdown", "Gap Reversal", "Liquidity Grab", "PDL Sweep"],
            },
            "layer2": {
                "htf_trend": "Bullish",
                "vwap_state": "Reclaimed",
                "location": "Near Support",
                "aligned": True,
            },
        },
    }
    base.update(overrides)
    return base


def _sell_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 10:25:00+05:30",
        "bar": 90,
        "direction": "SELL",
        "entry": 23600.0,
        "stop_loss": 23650.0,
        "target_1": 23550.0,
        "target_2": 23500.0,
        "target_3": 23450.0,
        "bars_before_expansion": 5,
        "mfe_points": 120.0,
        "mae_points": 40.0,
        "trade_duration_bars": 35,
        "mfe_capture_tiers": {"40": True, "60": True, "100": True, "200": False},
        "hit_1r": True,
        "hit_2r": True,
        "hit_3r": True,
        "win": True,
        "realized_pnl_points": 80.0,
        "classification": "Winner",
        "layers": {
            "layer1": {"events_detected": ["Failed Breakout"], "primary_event": "Failed Breakout"},
            "layer2": {
                "htf_trend": "Bearish",
                "vwap_state": "Below",
                "vwap_gate_rule": "VWAP Below only",
                "aligned": True,
            },
        },
    }
    base.update(overrides)
    return base


def test_preferred_windows_constant() -> None:
    assert PREFERRED_WINDOWS == (240, 300, 500)


def test_resolve_replay_windows() -> None:
    assert _resolve_replay_windows(600) == (240, 300, 500)
    assert _resolve_replay_windows(280) == (240,)
    assert _resolve_replay_windows(350) == (240, 300)
    assert _resolve_replay_windows(100) == (100,)


def test_analyze_trade_level_window_smoke() -> None:
    buy = [_buy_signal(), _buy_signal(mfe_points=150.0, bars_before_expansion=3)]
    sell = [_sell_signal(), _sell_signal(mfe_points=200.0, bars_before_expansion=8)]
    result = _analyze_trade_level_window(buy, sell, trading_days=240)
    assert result["signal_counts"]["buy_v3"] == 2
    assert result["signal_counts"]["sell_v6"] == 2
    assert "target_achievement_matrix" in result
    assert "conditional_probability" in result
    assert "trade_lifecycle_analysis" in result
    assert "entry_precision_audit" in result
    assert "execution_failure_audit" in result
    assert "runner_optimization_audit" in result
    assert result["buy_v4_sell_v7_potential"]["buy_v4"]["recommendation"] in {"YES", "NO"}


def test_summarize_core_metrics_smoke() -> None:
    buy = [_buy_signal(bar=5)]
    sell = [_sell_signal(bar=3)]
    frame = pd.DataFrame(
        {
            "Date": [f"2026-01-{d:02d} 09:15:00+05:30" for d in range(1, 11)],
            "Open": [23500.0] * 10,
            "High": [23550.0] * 10,
            "Low": [23450.0] * 10,
            "Close": [23520.0] * 10,
            "Volume": [1000] * 10,
        },
    )
    replay_dates = {date(2026, 1, d) for d in range(1, 11)}
    summary = _summarize_core_metrics(
        buy,
        sell,
        frame=frame,
        replay_dates=replay_dates,
        trading_days=240,
        moves=[],
        throttle_maps={"buy_v3": {}, "sell_v6": {}},
    )
    assert summary["buy_v3"]["signals_emitted"] == 1
    assert summary["sell_v6"]["signals_emitted"] == 1
    assert "profit_factor" in summary["combined"]


def test_mocked_full_run_exports_json(tmp_path: Path) -> None:
    metadata = {"end_date": "2026-07-02"}
    filter_path = tmp_path / "filter.json"
    filter_path.write_text(json.dumps(metadata), encoding="utf-8")

    buy = _buy_signal(bar=5)
    sell = _sell_signal(bar=3)
    frame = pd.DataFrame(
        {
            "Date": [
                f"2026-01-{(d % 28) + 1:02d} 09:15:00+05:30"
                for d in range(250)
            ],
            "Open": [23500.0] * 250,
            "High": [23550.0] * 250,
            "Low": [23450.0] * 250,
            "Close": [23520.0] * 250,
            "Volume": [1000] * 250,
        },
    )

    report_path = tmp_path / "extended_trade_level_truth_audit.json"
    research = ExtendedTradeLevelTruthAuditResearch()

    with (
        patch.object(research, "_replay_production", return_value={"buy_v3": [buy], "sell_v6": [sell]}),
        patch(
            "src.research.extended_trade_level_truth_audit_research.FilterResearchEngine",
        ) as mock_filter_cls,
        patch(
            "src.research.extended_trade_level_truth_audit_research._last_n_trading_day_set",
            return_value={date(2026, 1, d) for d in range(1, 11)},
        ),
        patch(
            "src.research.extended_trade_level_truth_audit_research._attach_ema22",
            side_effect=lambda df: df,
        ),
        patch(
            "src.research.extended_trade_level_truth_audit_research.InstitutionalLiquidityMapEngine",
        ) as mock_intel_cls,
    ):
        mock_filter = MagicMock()
        mock_filter._ensure_pipeline.return_value = tmp_path / "data.csv"
        mock_filter_cls.return_value = mock_filter
        mock_intel = MagicMock()
        mock_intel._attach_calendar_levels.return_value = frame
        mock_intel_cls.return_value = mock_intel

        (tmp_path / "data.csv").write_text(frame.to_csv(index=False), encoding="utf-8")
        research.buy_engine.context_builder.enrich = MagicMock(return_value=frame)
        research.sell_engine.context_builder.enrich = MagicMock(return_value=frame)
        research.buy_engine.intelligence.enrich = MagicMock(return_value=frame)
        research.buy_engine._resample_daily = MagicMock(return_value=frame)

        report = research.run(metadata, windows=(240,))
        research.export(report, report_path)

    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Extended Trade Level Truth Audit"
    assert payload["methodology"]["replay_required"] is True
    assert "core_metrics_by_window" in payload
    assert "target_achievement_matrix" in payload
    assert "conditional_probability" in payload
    assert "execution_failure_audit" in payload
    assert "runner_optimization_audit" in payload
    final = payload["final_answer"]
    assert final["buy_v4_recommendation"] in {"YES", "NO"}
    assert final["sell_v7_recommendation"] in {"YES", "NO"}
    assert "stop_loss_validation" in final
    assert "runner_validation" in final
    assert "target_achievement_probability_matrix" in final


def test_generate_helper_raises_without_filter(tmp_path: Path) -> None:
    with pytest.raises(Exception):
        generate_extended_trade_level_truth_audit_report(
            report_path=tmp_path / "out.json",
            filter_report_path=tmp_path / "missing.json",
        )
