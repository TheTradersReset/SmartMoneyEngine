"""Tests for signal monetization & target probability audit research."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.research.signal_monetization_target_probability_audit_research import (
    MONETIZATION_STRUCTURES,
    PATH_LEVELS,
    SignalMonetizationTargetProbabilityAuditError,
    SignalMonetizationTargetProbabilityAuditResearch,
    _reached_before_stop,
    _reward_risk_analysis,
    _structure_pnl,
    _target_path_analysis,
    _target_probability_matrix,
    _target_structure_comparison,
    _time_to_level_minutes,
)


def _buy_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-06 09:30:00+05:30",
        "direction": "BUY",
        "entry": 23500.0,
        "stop_loss": 23490.0,
        "mfe_points": 120.0,
        "mae_points": 8.0,
        "trade_duration_bars": 40,
        "win": True,
        "realized_pnl_points": 60.0,
    }
    base.update(overrides)
    return base


def _sell_signal(**overrides: object) -> dict:
    base = {
        "timestamp": "2026-01-05 10:25:00+05:30",
        "direction": "SELL",
        "entry": 23600.0,
        "stop_loss": 23610.0,
        "mfe_points": 150.0,
        "mae_points": 5.0,
        "trade_duration_bars": 35,
        "mfe_capture_tiers": {"40": True, "60": True, "100": True, "200": False},
        "win": True,
        "realized_pnl_points": 80.0,
    }
    base.update(overrides)
    return base


def test_constants() -> None:
    assert 20 in PATH_LEVELS and 300 in PATH_LEVELS
    assert "60/100/Runner" in MONETIZATION_STRUCTURES
    assert "40 Fixed" in MONETIZATION_STRUCTURES


def test_reached_before_stop_conservative() -> None:
    assert _reached_before_stop(_buy_signal(mfe_points=60, mae_points=8), 60) is True
    assert _reached_before_stop(_buy_signal(mfe_points=60, mae_points=12), 60) is False
    assert _reached_before_stop(_buy_signal(mfe_points=40, mae_points=5), 60) is False


def test_time_to_level_prefers_measured_field() -> None:
    signal = _buy_signal(time_to_60=45.0, mfe_points=80.0, mae_points=5.0)
    minutes, provenance = _time_to_level_minutes(signal, 60)
    assert minutes == 45.0
    assert provenance == "measured"


def test_target_probability_matrix_shape() -> None:
    signals = [
        _buy_signal(mfe_points=100, mae_points=5),
        _buy_signal(mfe_points=30, mae_points=15),
        _buy_signal(mfe_points=80, mae_points=4),
    ]
    result = _target_probability_matrix(
        signals, side="BUY", window_days=240, provenance="test",
    )
    assert result["sample_size"] == 3
    assert "60" in result["by_tier"]
    assert set(result["by_tier"]["60"].keys()) >= {
        "count",
        "frequency",
        "probability_pct",
        "win_pct",
        "loss_pct",
    }
    assert result["by_tier"]["60"]["count"] == 2


def test_target_path_and_rr() -> None:
    signals = [_buy_signal(), _buy_signal(mfe_points=40, mae_points=3)]
    path = _target_path_analysis(signals, side="BUY", window_days=240, provenance="test")
    assert "Signal → 20" in path["path"]
    assert path["nodes"]["Signal"]["probability_pct"] == 100.0
    assert path["nodes"]["60"]["avg_time_to_reach_minutes"] is not None

    rr = _reward_risk_analysis(signals, side="BUY", window_days=240, provenance="test")
    assert "1:1" in rr["by_rr"] and "1:5" in rr["by_rr"]
    assert rr["distribution"]["avg_rr"] is not None


def test_structure_comparison_picks_best() -> None:
    signals = [
        _buy_signal(mfe_points=200, mae_points=4),
        _buy_signal(mfe_points=180, mae_points=6),
        _buy_signal(mfe_points=5, mae_points=12),
    ]
    cmp = _target_structure_comparison(
        signals, side="BUY", window_days=240, provenance="test",
    )
    assert cmp["best_structure"] in MONETIZATION_STRUCTURES
    for label in MONETIZATION_STRUCTURES:
        row = cmp["by_structure"][label]
        assert "expected_wr_pct" in row
        assert "expected_pf" in row
        assert "expected_expectancy" in row
        assert "expected_capture_pct" in row
        assert row["provenance_label"] == "derived_from_path"

    pnl = _structure_pnl(
        _buy_signal(mfe_points=100, mae_points=5),
        MONETIZATION_STRUCTURES["60 Fixed"],
        stop_pts=10.0,
    )
    assert pnl == 60.0


def test_research_run_and_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    research_dir = tmp_path / "outputs" / "research"
    research_dir.mkdir(parents=True)

    buy_signals = [
        _buy_signal(mfe_points=180, mae_points=4),
        _buy_signal(mfe_points=140, mae_points=6),
        _buy_signal(mfe_points=70, mae_points=5),
        _buy_signal(mfe_points=8, mae_points=12),
    ]
    sell_signals = [
        _sell_signal(mfe_points=200, mae_points=3),
        _sell_signal(mfe_points=130, mae_points=4),
        _sell_signal(mfe_points=85, mae_points=5),
        _sell_signal(mfe_points=6, mae_points=14),
    ]

    (research_dir / "extended_trade_level_truth_audit.json").write_text(
        json.dumps(
            {
                "symbol": "NIFTY50",
                "timeframe": "5M",
                "max_replay_window": 240,
                "replay_windows": [240],
                "replay_start_date": "2025-07-11",
                "replay_end_date": "2026-07-02",
                "per_signal_details": {"buy_v3": buy_signals, "sell_v6": sell_signals},
                "core_metrics_by_window": {
                    "240": {
                        "buy_v3": {
                            "signals_emitted": 4,
                            "signals_per_month": 20.0,
                            "expectancy": 40.0,
                            "profit_factor": 1.5,
                            "win_rate_pct": 50.0,
                            "max_drawdown_points": 100.0,
                        },
                        "sell_v6": {
                            "signals_emitted": 4,
                            "signals_per_month": 50.0,
                            "expectancy": 70.0,
                            "profit_factor": 2.0,
                            "win_rate_pct": 60.0,
                            "max_drawdown_points": 80.0,
                        },
                        "combined": {"expectancy": 60.0, "signals_per_month": 70.0},
                    }
                },
                "conditional_probability": {
                    "240": {
                        "buy_v3": {"summary": {"p_60_plus": 50.0}},
                        "sell_v6": {"summary": {"p_60_plus": 75.0}},
                    }
                },
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "trade_level_truth_audit.json").write_text(
        json.dumps({"trading_days_replayed": 120, "per_signal_records": {}}),
        encoding="utf-8",
    )
    (research_dir / "buy_v3_candidate_validation.json").write_text(
        json.dumps(
            {
                "trading_days_replayed": 120,
                "per_signal_details": {"buy_v3": buy_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "sell_v6_replay_validation.json").write_text(
        json.dumps(
            {
                "trading_days_replayed": 120,
                "per_signal_details": {"sell_v6": sell_signals},
            },
        ),
        encoding="utf-8",
    )
    (research_dir / "extended_evidence_validation_real_deployment_audit.json").write_text(
        json.dumps({"report_type": "Extended Evidence", "replay_windows": [120, 250]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "src.research.signal_monetization_target_probability_audit_research.RESEARCH_DIR",
        research_dir,
    )
    monkeypatch.setattr(
        "src.research.signal_monetization_target_probability_audit_research.REQUIRED_EXPORTS",
        {
            "extended_trade_level_truth_audit": research_dir / "extended_trade_level_truth_audit.json",
            "trade_level_truth_audit": research_dir / "trade_level_truth_audit.json",
            "buy_v3_candidate_validation": research_dir / "buy_v3_candidate_validation.json",
            "sell_v6_replay_validation": research_dir / "sell_v6_replay_validation.json",
            "extended_evidence_validation_real_deployment_audit": research_dir
            / "extended_evidence_validation_real_deployment_audit.json",
        },
    )

    out = research_dir / "signal_monetization_target_probability_audit.json"
    research = SignalMonetizationTargetProbabilityAuditResearch(report_path=out)
    report = research.run()
    path = research.export(report)

    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["report_type"] == "Signal Monetization & Target Probability Audit"
    assert "buy_v3" in payload["target_probability_before_stop"]["primary"]
    assert "production_playbook" in payload
    assert payload["final_answer"]["best_target_structure"]["value"]
    assert "Take T1 at" in payload["final_answer"]["production_playbook_one_liners"]["shared"]
    assert payload["final_answer"]["confidence_score"] is not None


def test_missing_export_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    research_dir = tmp_path / "empty"
    research_dir.mkdir()
    monkeypatch.setattr(
        "src.research.signal_monetization_target_probability_audit_research.REQUIRED_EXPORTS",
        {
            "extended_trade_level_truth_audit": research_dir / "missing.json",
            "trade_level_truth_audit": research_dir / "missing2.json",
            "buy_v3_candidate_validation": research_dir / "missing3.json",
            "sell_v6_replay_validation": research_dir / "missing4.json",
            "extended_evidence_validation_real_deployment_audit": research_dir / "missing5.json",
        },
    )
    with pytest.raises(SignalMonetizationTargetProbabilityAuditError):
        SignalMonetizationTargetProbabilityAuditResearch(
            report_path=research_dir / "out.json",
        ).run()
