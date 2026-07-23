"""Tests for Replay Analytics Engine."""

from __future__ import annotations

import json
from pathlib import Path

from src.replay_analytics.analyzer import analyze_replay
from src.replay_analytics.engine import ReplayAnalyticsEngine
from src.storage.sqlite import PaperSignalDatabase


def _seed_db(path: Path) -> None:
    db = PaperSignalDatabase(path)
    for index, (decision, buy, sell, trend, reasons) in enumerate(
        [
            ("NO_TRADE", 65.0, 40.0, "Neutral", ["FORMULA_INCOMPLETE", "VWAP_MISMATCH"]),
            ("NO_TRADE", 70.0, 45.0, "Bearish", ["FORMULA_INCOMPLETE", "MISSING_PDL_SWEEP"]),
            ("BUY", 100.0, 20.0, "Bullish", []),
            ("SELL", 30.0, 100.0, "Bearish", []),
        ],
        start=1,
    ):
        minute = 10 + index * 5
        db.insert_signal_decision(
            {
                "timestamp": f"2026-03-10T09:{minute:02d}:00+05:30",
                "symbol": "NSE:NIFTY50-INDEX",
                "open": 25000.0,
                "high": 25010.0,
                "low": 24990.0,
                "close": 25005.0,
                "volume": 1000.0,
                "trend": trend,
                "market_regime": "range|unknown_vol|no_gap|mid_range",
                "buy_score": buy,
                "sell_score": sell,
                "final_signal": decision,
                "decision": decision,
                "reason_codes": reasons,
                "evaluation_time_ms": 1.0,
            },
        )
        db.insert_candle(
            symbol="NSE:NIFTY50-INDEX",
            timestamp=f"2026-03-10T09:{minute:02d}:00+05:30",
            open_=25000.0,
            high=25010.0,
            low=24990.0,
            close=25005.0,
            volume=1000.0,
            tick_count=1,
        )
    db.insert_signal(
        {
            "timestamp": "2026-03-10T09:25:00+05:30",
            "direction": "BUY",
            "engine_version": "BUY_V3",
            "entry": 25005.0,
            "stop": 24995.0,
            "target1": 25065.0,
            "target2": 25105.0,
            "target_structure": "60/100/Runner",
            "confidence": 1.0,
            "regime": "range|unknown_vol|no_gap|mid_range",
            "throttle_level": "FULL",
            "accepted": True,
            "rejection_reason": None,
        },
    )
    db.close()


def test_analyze_replay_summary_math() -> None:
    decisions = [
        {
            "timestamp": "2026-03-10T09:15:00+05:30",
            "decision": "NO_TRADE",
            "buy_score": 50.0,
            "sell_score": 25.0,
            "trend": "Neutral",
            "market_regime": "r1",
            "reason_codes": ["FORMULA_INCOMPLETE", "VWAP_MISMATCH"],
        },
        {
            "timestamp": "2026-03-10T09:20:00+05:30",
            "decision": "BUY",
            "buy_score": 100.0,
            "sell_score": 10.0,
            "trend": "Bullish",
            "market_regime": "r2",
            "reason_codes": [],
        },
    ]
    report = analyze_replay(decisions=decisions, signals=[], candle_count=2)
    assert report["replay_summary"]["total_decisions"] == 2
    assert report["replay_summary"]["buy_signals"] == 1
    assert report["replay_summary"]["no_trade_count"] == 1
    assert report["decision_statistics"]["BUY_pct"] == 50.0
    assert report["score_statistics"]["maximum_buy_score"] == 100.0
    assert report["daily_summary"][0]["period"] == "2026-03-10"
    assert report["monthly_summary"][0]["period"] == "2026-03"


def test_engine_writes_artifacts(tmp_path: Path) -> None:
    db_path = tmp_path / "replay.db"
    out_dir = tmp_path / "out"
    _seed_db(db_path)

    artifacts = ReplayAnalyticsEngine(db_path=db_path, output_dir=out_dir).run()
    assert artifacts.summary_json.exists()
    assert artifacts.report_csv.exists()
    assert artifacts.report_html.exists()

    payload = json.loads(artifacts.summary_json.read_text(encoding="utf-8"))
    assert payload["replay_summary"]["total_decisions"] == 4
    assert payload["replay_summary"]["buy_signals"] == 1
    assert payload["replay_summary"]["sell_signals"] == 1
    assert payload["replay_summary"]["no_trade_count"] == 2
    assert payload["score_statistics"]["maximum_buy_score"] == 100.0
    assert "FORMULA_INCOMPLETE" in {
        item["code"] for item in payload["rule_statistics"]["top_rejection_reasons"]
    }
    html = artifacts.report_html.read_text(encoding="utf-8")
    assert "Replay Analytics Report" in html
    assert "Daily Summary" in html
    csv_text = artifacts.report_csv.read_text(encoding="utf-8")
    assert "replay_summary" in csv_text
