"""Smoke tests for SmartMoneyEngine engine gap analysis synthesis."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.research.smartmoneyengine_engine_gap_analysis_research import (
    RELAXATION_FILTERS,
    SmartMoneyEngineEngineGapAnalysisResearch,
    _dedupe_moves,
)


class TestEngineGapAnalysisResearch(unittest.TestCase):
    def test_dedupe_moves_by_threshold_and_bar(self) -> None:
        rows = [
            {"threshold_points": 100, "move_start_bar": 1, "captured_by_v3": False},
            {"threshold_points": 100, "move_start_bar": 1, "captured_by_v3": False},
            {"threshold_points": 200, "move_start_bar": 1, "captured_by_v3": True},
        ]
        self.assertEqual(len(_dedupe_moves(rows)), 2)

    def test_generate_report_from_exports(self) -> None:
        root = Path(__file__).resolve().parents[1]
        export_names = [
            "smartmoneyengine_v31_validation.json",
            "smartmoneyengine_v3_implementation_validation.json",
            "nifty50_signal_timing_audit.json",
            "smartmoneyengine_reality_check_validation.json",
            "sell_formula_reality_verification_v2.json",
        ]
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            paths = {}
            for name in export_names:
                src = root / "outputs" / "research" / name
                dst = tmp / name
                dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
                paths[name] = dst

            report_path = tmp / "smartmoneyengine_engine_gap_analysis.json"
            research = SmartMoneyEngineEngineGapAnalysisResearch(
                v31_path=paths["smartmoneyengine_v31_validation.json"],
                v3_path=paths["smartmoneyengine_v3_implementation_validation.json"],
                timing_path=paths["nifty50_signal_timing_audit.json"],
                reality_path=paths["smartmoneyengine_reality_check_validation.json"],
                sell_path=paths["sell_formula_reality_verification_v2.json"],
                report_path=report_path,
            )
            exported = research.export()
            payload = json.loads(exported.read_text(encoding="utf-8"))

            self.assertEqual(payload["report_type"], "SmartMoneyEngine Engine Gap Analysis")
            self.assertEqual(payload["capture_baseline"]["v3_120d"]["200_plus"]["capture_rate_pct"], 50.56)
            self.assertIn("missed_move_analysis", payload)
            self.assertIn("counterfactual_relaxation", payload)
            self.assertIn("final_answer", payload)
            self.assertEqual(
                payload["missed_move_analysis"]["by_threshold"]["200_plus"]["missed_by_v3_aggregate"],
                133,
            )
            self.assertEqual(set(payload["counterfactual_relaxation"]["scenarios"]), set(RELAXATION_FILTERS))
            self.assertIn(
                payload["final_answer"]["single_filter_largest_move_capture_loss"]["filter"],
                RELAXATION_FILTERS,
            )


if __name__ == "__main__":
    unittest.main()
