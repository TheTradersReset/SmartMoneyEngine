"""Tests for SELL_V6 replay validation research."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.research.sell_v6_replay_validation_research import (
    V6_ALLOWED_VWAP_STATES,
    V6_VWAP_GATE_RULE,
    SellV6CandidateEngine,
    SellV6ReplayValidationError,
    SellV6ReplayValidationReport,
    SellV6ReplayValidationResearch,
    _audit_pf_reconciliation,
    _map_removed_classification,
    _removed_trade_analysis,
    _v6_vwap_gate_passes,
    generate_sell_v6_replay_validation_report,
)
from src.research.smartmoneyengine_v4_candidate_validation_research import V4_EMA_BEAR_CONTEXT


def test_v6_vwap_gate_below_only() -> None:
    assert _v6_vwap_gate_passes("Below") is True
    assert _v6_vwap_gate_passes("Rejected") is False
    assert _v6_vwap_gate_passes("Above") is False
    assert _v6_vwap_gate_passes("Reclaimed") is False
    assert _v6_vwap_gate_passes(None) is False


def test_v6_allowed_vwap_states() -> None:
    assert V6_ALLOWED_VWAP_STATES == frozenset({"Below"})


def test_v6_layer2_rejects_rejected_vwap() -> None:
    engine = SellV6CandidateEngine()
    layer2 = engine._layer2_directional_filter(
        {
            "htf_trend": "Bearish",
            "vwap": "Rejected",
            "v4_ema_bearish": "True",
            "v4_ema_structure": V4_EMA_BEAR_CONTEXT,
        },
    )
    assert layer2["vwap_gate_passes"] is False
    assert layer2["vwap_gate_rule"] == V6_VWAP_GATE_RULE
    assert layer2["aligned"] is False
    assert layer2["direction"] == "NO_TRADE"


def test_v6_layer2_accepts_below_vwap() -> None:
    engine = SellV6CandidateEngine()
    layer2 = engine._layer2_directional_filter(
        {
            "htf_trend": "Bearish",
            "vwap": "Below",
            "v4_ema_bearish": "True",
            "v4_ema_structure": V4_EMA_BEAR_CONTEXT,
        },
    )
    assert layer2["vwap_gate_passes"] is True
    assert layer2["aligned"] is True
    assert layer2["direction"] == "SELL"


def test_map_removed_classification_trend_exhaustion() -> None:
    assert _map_removed_classification("Trend Exhaustion") == "Trend Reversal"


def test_removed_trade_analysis() -> None:
    v5 = [
        {"bar": 1, "win": False, "realized_pnl_points": -50, "mfe_points": 10, "mae_points": 50,
         "layers": {"layer2": {"vwap_state": "Rejected"}}, "regime": {}, "classification": "Bear Trap",
         "mfe_capture_tiers": {}},
        {"bar": 2, "win": True, "realized_pnl_points": 100, "mfe_points": 120, "mae_points": 30,
         "layers": {"layer2": {"vwap_state": "Below"}}, "regime": {}, "classification": "Winner",
         "mfe_capture_tiers": {}},
        {"bar": 3, "win": False, "realized_pnl_points": -80, "mfe_points": 15, "mae_points": 80,
         "layers": {"layer2": {"vwap_state": "Rejected"}}, "regime": {}, "classification": "No Expansion",
         "mfe_capture_tiers": {}},
    ]
    v6 = [v5[1]]
    result = _removed_trade_analysis(v5, v6)
    assert result["removed_count"] == 2
    assert result["removed_losers"] == 2
    assert result["removed_winners"] == 0
    assert result["bad_trades_only"] is True


def test_audit_pf_reconciliation_survives() -> None:
    result = _audit_pf_reconciliation(4.05)
    assert result["pf_survives_replay"] is True
    assert result["verdict"] == "YES"


def test_audit_pf_reconciliation_fails() -> None:
    result = _audit_pf_reconciliation(3.0)
    assert result["pf_survives_replay"] is False
    assert result["verdict"] == "NO"


def test_generate_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    filter_report = tmp_path / "filter_research_report.json"
    filter_report.write_text('{"end_date": "2026-07-03"}', encoding="utf-8")
    destination = tmp_path / "sell_v6_replay_validation.json"

    def _fake_run(self, metadata: dict) -> SellV6ReplayValidationReport:
        del metadata
        return SellV6ReplayValidationReport(
            report_type="SELL_V6 Replay Validation",
            engines_compared=["SELL_V5", "SELL_V6"],
            symbol="NIFTY50",
            timeframe="5M",
            trading_days_replayed=120,
            replay_start_date="2026-01-01",
            replay_end_date="2026-07-03",
            v6_change_summary={},
            methodology={},
            comparison_table={
                "sell_v5": {"signals_emitted": 380, "profit_factor": 3.37},
                "sell_v6": {"signals_emitted": 336, "profit_factor": 4.09},
            },
            walk_forward={},
            pf_audit_reconciliation={"verdict": "YES"},
            removed_trade_analysis={"removed_count": 44},
            regime_analysis={},
            trap_and_mae_impact={},
            production_readiness={"score": 85},
            final_verdict={"can_sell_v6_replace_sell_v5": "YES"},
            per_signal_details={"sell_v5": [], "sell_v6": []},
            conclusions=[],
            execution_time_seconds=1.0,
        )

    monkeypatch.setattr(SellV6ReplayValidationResearch, "run", _fake_run)

    report = generate_sell_v6_replay_validation_report(
        report_path=destination,
        filter_report_path=filter_report,
    )
    assert report.comparison_table["sell_v6"]["profit_factor"] == 4.09
    assert destination.exists()


def test_generate_report_missing_filter() -> None:
    with pytest.raises(SellV6ReplayValidationError):
        generate_sell_v6_replay_validation_report(filter_report_path=Path("missing.json"))
