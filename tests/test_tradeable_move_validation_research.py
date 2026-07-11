"""Smoke tests for tradeable move validation research."""

from __future__ import annotations

import json
from pathlib import Path

from src.research.tradeable_move_validation_research import (
    BUY_MODEL_ID,
    SELL_MODEL_ID,
    TRADEABLE_TIERS,
    TradeableMoveValidationResearch,
)


def test_model_ids() -> None:
    assert BUY_MODEL_ID == "LDM-BUY-V1"
    assert SELL_MODEL_ID == "LDM-SELL-V5"
    assert TRADEABLE_TIERS == (40, 60, 80, 100)


def test_generate_report(tmp_path: Path) -> None:
    report_path = tmp_path / "tradeable_move_validation.json"
    research = TradeableMoveValidationResearch(report_path=report_path)
    exported = research.export()
    payload = json.loads(exported.read_text(encoding="utf-8"))

    assert payload["report_type"] == "Tradeable Move Validation"
    assert payload["symbol"] == "NIFTY50"
    assert payload["timeframe"] == "5M"
    assert "methodology" in payload
    assert "source_exports" in payload
    assert "sell_v5_analysis" in payload
    assert "buy_v1_analysis" in payload
    assert "tradeable_tier_metrics" in payload
    assert "lead_time_analysis" in payload
    assert "human_tradeability" in payload
    assert "frequency_classification" in payload
    assert "model_comparison" in payload
    assert "coexistence_verdict" in payload
    assert "forty_sixty_capture_answer" in payload
    assert payload["final_verdict"]["verdict"] in {"YES", "NO", "PARTIAL"}

    buy_signals = payload["buy_v1_analysis"]["per_signal_analysis"]
    assert len(buy_signals) >= 1
    first_buy = buy_signals[0]
    assert "mfe_points" in first_buy
    assert "mae_points" in first_buy
    assert "tradeable_tier_hits" in first_buy
    for tier in TRADEABLE_TIERS:
        assert f"{tier}_plus" in first_buy["tradeable_tier_hits"]

    for tier in TRADEABLE_TIERS:
        assert str(tier) in payload["tradeable_tier_metrics"]["buy_v1"]
        assert str(tier) in payload["tradeable_tier_metrics"]["sell_v5"]

    assert payload["human_tradeability"]["sell_v5"]["verdict"] in {"YES", "NO", "PARTIAL"}
    assert payload["human_tradeability"]["buy_v1"]["verdict"] in {"YES", "NO", "PARTIAL"}
    assert payload["frequency_classification"]["buy_v1"]["classification"] in {"LOW", "MEDIUM", "HIGH"}
    assert payload["frequency_classification"]["sell_v5"]["classification"] in {"LOW", "MEDIUM", "HIGH"}
    assert payload["model_comparison"]["winning_model"] in {SELL_MODEL_ID, BUY_MODEL_ID, None}
    assert payload["coexistence_verdict"]["verdict"] in {"YES", "NO", "PARTIAL"}
    assert payload["forty_sixty_capture_answer"]["answer"] in {"YES", "NO", "PARTIAL"}
    assert len(payload["conclusions"]) >= 5
