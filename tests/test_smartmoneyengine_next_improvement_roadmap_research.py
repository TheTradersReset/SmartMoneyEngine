"""Smoke tests for SmartMoneyEngine next improvement roadmap synthesis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.research.smartmoneyengine_next_improvement_roadmap_research import (
    SOURCE_EXPORTS,
    SmartMoneyEngineNextImprovementRoadmapReport,
    SmartMoneyEngineNextImprovementRoadmapResearch,
    _sell_capture_metrics,
)


class TestNextImprovementRoadmapResearch(unittest.TestCase):
    def test_sell_capture_metrics_proxies_40_and_60_from_50(self) -> None:
        v3 = {"50": {"total_bearish_moves": 10, "signals_before_move": 5, "capture_rate_pct": 50.0}}
        v4 = {"50": {"total_bearish_moves": 10, "signals_before_move": 6, "capture_rate_pct": 60.0}}
        metrics = _sell_capture_metrics(v3_capture=v3, v4_capture=v4)
        self.assertEqual(metrics["v3_baseline_120d"]["40_plus"]["capture_rate_pct"], 50.0)
        self.assertEqual(metrics["v4_candidate_120d"]["60_plus"]["capture_rate_pct"], 60.0)
        self.assertIn("proxy_note", metrics["v3_baseline_120d"]["40_plus"])

    def test_generate_report_from_exports(self) -> None:
        root = Path(__file__).resolve().parents[1]
        export_names = list(SOURCE_EXPORTS.keys())
        path_map = {
            "v4_candidate_validation": "smartmoneyengine_v4_candidate_validation.json",
            "engine_gap_analysis": "smartmoneyengine_engine_gap_analysis.json",
            "v31_validation": "smartmoneyengine_v31_validation.json",
            "final_signal_extraction": "smartmoneyengine_final_signal_extraction.json",
            "buy_formula_reality": "buy_formula_reality_verification.json",
            "buy_side_reality_discovery": "nifty50_buy_side_reality_discovery.json",
            "research_consistency_audit": "research_consistency_audit.json",
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            paths = {}
            for key, name in path_map.items():
                src = root / "outputs" / "research" / name
                dst = tmp / name
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                paths[key] = dst

            report_path = tmp / "smartmoneyengine_next_improvement_roadmap.json"
            research = SmartMoneyEngineNextImprovementRoadmapResearch(
                v4_path=paths["v4_candidate_validation"],
                gap_path=paths["engine_gap_analysis"],
                v31_path=paths["v31_validation"],
                extraction_path=paths["final_signal_extraction"],
                buy_formula_path=paths["buy_formula_reality"],
                buy_discovery_path=paths["buy_side_reality_discovery"],
                consistency_path=paths["research_consistency_audit"],
                report_path=report_path,
            )
            exported = research.export()
            payload = json.loads(exported.read_text(encoding="utf-8"))

            self.assertEqual(payload["report_type"], "SmartMoneyEngine Next Improvement Roadmap")
            self.assertIn("sell_side_analysis", payload)
            self.assertIn("buy_side_analysis", payload)
            self.assertIn("ranked_opportunities", payload)
            self.assertIn("final_recommendation", payload)
            self.assertEqual(len(payload["ranked_opportunities"]), 3)
            self.assertEqual(
                payload["ranked_opportunities"][0]["opportunity"],
                "VWAP Below gate relaxation (V5 research on V4 base)",
            )
            self.assertEqual(payload["buy_side_analysis"]["salvageability"]["verdict"], "PARTIAL")
            self.assertEqual(
                payload["sell_side_analysis"]["current_metrics"]["v4_candidate_latest"]["profit_factor"],
                4.09,
            )

    def test_mocked_run_returns_report(self) -> None:
        mock_report = SmartMoneyEngineNextImprovementRoadmapReport(
            report_type="SmartMoneyEngine Next Improvement Roadmap",
            symbol="NIFTY50",
            timeframe="5M",
            methodology={"research_only": True},
            source_exports=[],
            sell_side_analysis={},
            buy_side_analysis={"salvageability": {"verdict": "PARTIAL"}},
            ranked_opportunities=[{"rank": 1, "opportunity": "mock"}],
            final_recommendation={},
            conclusions=[],
            execution_time_seconds=0.01,
        )
        research = SmartMoneyEngineNextImprovementRoadmapResearch()
        with patch.object(research, "run", return_value=mock_report):
            with tempfile.TemporaryDirectory() as tmp_dir:
                research.report_path = Path(tmp_dir) / "out.json"
                exported = research.export()
                payload = json.loads(exported.read_text(encoding="utf-8"))
                self.assertEqual(payload["report_type"], "SmartMoneyEngine Next Improvement Roadmap")


if __name__ == "__main__":
    unittest.main()
