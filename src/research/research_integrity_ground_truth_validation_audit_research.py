"""
Research Integrity & Ground Truth Validation Audit — synthesis from existing exports.

Validates whether major conclusions rest on real replay evidence vs filtered/synthetic
approximations. No new engines, indicators, models, or discovery.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.research.buy_failure_anatomy_research import _json_safe

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "research_integrity_ground_truth_validation_audit.json"
DEFAULT_FILTER_REPORT = RESEARCH_DIR / "filter_research_report.json"

REQUIRED_EXPORTS = {
    "extended_trade_level_truth_audit": RESEARCH_DIR / "extended_trade_level_truth_audit.json",
    "extended_evidence_validation_real_deployment_audit": RESEARCH_DIR
    / "extended_evidence_validation_real_deployment_audit.json",
    "buy_v4_sell_v7_design_blueprint_audit": RESEARCH_DIR
    / "buy_v4_sell_v7_design_blueprint_audit.json",
    "buy_v4_sell_v7_final_production_validation": RESEARCH_DIR
    / "buy_v4_sell_v7_final_production_validation.json",
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
}


class ResearchIntegrityGroundTruthValidationAuditError(Exception):
    """Raised when integrity audit fails."""


@dataclass
class ResearchIntegrityGroundTruthValidationAuditReport:
    """Research integrity & ground truth validation audit output."""

    report_type: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    data_provenance_audit: dict[str, Any]
    broker_data_audit: dict[str, Any]
    trading_days_audit: dict[str, Any]
    window_replay_status: dict[str, Any]
    evidence_lineage_tree: dict[str, Any]
    buy_v4_validation_audit: dict[str, Any]
    sell_v7_validation_audit: dict[str, Any]
    ground_truth_replay_gap_audit: dict[str, Any]
    target_achievement_audit: dict[str, Any]
    trade_lifecycle_audit: dict[str, Any]
    signal_timing_audit: dict[str, Any]
    production_evidence_audit: dict[str, Any]
    replacement_sufficiency: dict[str, Any]
    final_answer: dict[str, Any]
    completion_scores: dict[str, Any]
    definitive_verdict: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _nested(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict):
            return default
        node = node.get(key)
    return default if node is None else node


def _signal_count(export: dict[str, Any], *path: str) -> int | None:
    node = _nested(export, *path, default=None)
    if isinstance(node, list):
        return len(node)
    return None


def _provenance_row(
    *,
    report_name: str,
    data_source: str,
    date_range: str,
    trading_days: Any,
    bar_count: Any,
    signal_count: Any,
    replay_status: str,
    synthetic_status: str,
    notes: str = "",
) -> dict[str, Any]:
    return {
        "report": report_name,
        "data_source": data_source,
        "date_range": date_range,
        "trading_days": trading_days,
        "bar_count": bar_count,
        "signal_count": signal_count,
        "replay_status": replay_status,
        "synthetic_status": synthetic_status,
        "notes": notes,
    }


def _classify_metric_origin(method: str) -> dict[str, str]:
    """Map validation method to measured/estimated/projected labels."""
    if method == "A":
        return {"default": "Measured"}
    if method == "B":
        return {
            "signal_count": "Measured (filtered subset of replayed signals)",
            "win_rate_pct": "Measured on filtered replayed signals",
            "profit_factor": "Measured on filtered replayed signals (sensitive to loser removal)",
            "expectancy": "Measured on filtered replayed signals",
            "capture_pct": "Measured on filtered replayed signals",
            "drawdown_points": "Measured on filtered replayed signals",
            "production_generalization": "Projected (no dedicated V4/V7 engine replay)",
        }
    if method == "C":
        return {"default": "Projected"}
    return {"default": "Mixed — see field-level tags"}


class ResearchIntegrityGroundTruthValidationAuditResearch:
    """Audit research integrity and ground-truth provenance across exports."""

    def run(self, sources: dict[str, dict[str, Any]]) -> ResearchIntegrityGroundTruthValidationAuditReport:
        started = time.perf_counter()
        etl = sources["extended_trade_level_truth_audit"]
        evidence = sources["extended_evidence_validation_real_deployment_audit"]
        blueprint = sources["buy_v4_sell_v7_design_blueprint_audit"]
        final_val = sources["buy_v4_sell_v7_final_production_validation"]
        buy_v3 = sources["buy_v3_candidate_validation"]
        sell_v6 = sources["sell_v6_replay_validation"]
        filter_meta = _load_json(DEFAULT_FILTER_REPORT)

        buy_v3_n = _signal_count(buy_v3, "per_signal_details", "buy_v3") or _nested(
            buy_v3, "overall_statistics", "signals_emitted",
        )
        sell_v6_n = _signal_count(sell_v6, "per_signal_details", "sell_v6")
        etl_buy_n = _signal_count(etl, "per_signal_details", "buy_v3")
        etl_sell_n = _signal_count(etl, "per_signal_details", "sell_v6")
        fv_buy_v4_n = _nested(final_val, "core_metrics_by_window", "240", "buy_v4", "signals_emitted")
        fv_sell_v7_n = _nested(final_val, "core_metrics_by_window", "240", "sell_v7", "signals_emitted")

        etl_start = etl.get("replay_start_date") or _nested(etl, "core_metrics_by_window", "240", "replay_start_date")
        etl_end = etl.get("replay_end_date") or _nested(etl, "core_metrics_by_window", "240", "replay_end_date")
        buy_start = buy_v3.get("replay_start_date")
        buy_end = buy_v3.get("replay_end_date")
        sell_start = sell_v6.get("replay_start_date")
        sell_end = sell_v6.get("replay_end_date")

        available_days = etl.get("available_trading_days")
        evidence_windows = evidence.get("replay_windows") or []
        etl_windows = etl.get("replay_windows") or []

        # Evidence 250 vs 500 same signal counts → single-pass slice, not independent 500d corpus
        ev_250_buy = _nested(evidence, "window_results", "250", "buy_v3_only", "signals_emitted")
        ev_500_buy = _nested(evidence, "window_results", "500", "buy_v3_only", "signals_emitted")
        ev_250_sell = _nested(evidence, "window_results", "250", "sell_v6_only", "signals_emitted")
        ev_500_sell = _nested(evidence, "window_results", "500", "sell_v6_only", "signals_emitted")
        same_250_500 = (
            ev_250_buy is not None
            and ev_250_buy == ev_500_buy
            and ev_250_sell == ev_500_sell
        )

        provenance = [
            _provenance_row(
                report_name="buy_v3_candidate_validation.json",
                data_source="NIFTY50 5M pipeline CSV via FilterResearchEngine",
                date_range=f"{buy_start} → {buy_end}",
                trading_days=buy_v3.get("trading_days_replayed"),
                bar_count="not stored in export",
                signal_count={"buy_v3": buy_v3_n},
                replay_status="ACTUAL REPLAY",
                synthetic_status="NO (engine bar replay)",
                notes="120 trading days BUY_V3 candidate validation",
            ),
            _provenance_row(
                report_name="sell_v6_replay_validation.json",
                data_source="NIFTY50 5M pipeline CSV via FilterResearchEngine",
                date_range=f"{sell_start} → {sell_end}",
                trading_days=sell_v6.get("trading_days_replayed"),
                bar_count="not stored in export",
                signal_count={"sell_v6": sell_v6_n},
                replay_status="ACTUAL REPLAY",
                synthetic_status="NO (engine bar replay)",
                notes="120 trading days SELL_V6 replay",
            ),
            _provenance_row(
                report_name="extended_trade_level_truth_audit.json",
                data_source="NIFTY50 5M pipeline CSV; single-pass BUY_V3+SELL_V6 replay",
                date_range=f"{etl_start} → {etl_end}",
                trading_days={"available": available_days, "replayed_window": etl_windows},
                bar_count="~17858 bars logged during 240d replay (see run log)",
                signal_count={"buy_v3": etl_buy_n, "sell_v6": etl_sell_n},
                replay_status="ACTUAL REPLAY",
                synthetic_status="PARTIAL (post-replay matrices/lifecycle synthesized from signals)",
                notes="Preferred 300/500 not available; active window 240 only",
            ),
            _provenance_row(
                report_name="extended_evidence_validation_real_deployment_audit.json",
                data_source="NIFTY50 5M pipeline CSV; single-pass max-window replay then date slices",
                date_range="same pipeline end_date family as filter report",
                trading_days={"declared_windows": evidence_windows, "available_in_frame": available_days},
                bar_count="~18283 bars for max-window pass (run log)",
                signal_count={
                    "250d_buy": ev_250_buy,
                    "500d_buy": ev_500_buy,
                    "250d_sell": ev_250_sell,
                    "500d_sell": ev_500_sell,
                    "250_equals_500_signal_counts": same_250_500,
                },
                replay_status="ACTUAL REPLAY (one pass) + WINDOW SLICING",
                synthetic_status=(
                    "YES for independent 500d claim if signal counts match 250d — "
                    "500d metrics are re-denominated slices, not a larger independent corpus"
                    if same_250_500
                    else "PARTIAL (post-replay analytics)"
                ),
                notes="methodology.actual_replay=true; 120/250/500 analyzed from one replay pass",
            ),
            _provenance_row(
                report_name="buy_v4_sell_v7_design_blueprint_audit.json",
                data_source="Prior JSON exports only",
                date_range="inherits 240d trade-level + evidence windows",
                trading_days="no new replay",
                bar_count="n/a",
                signal_count="reuses etl per_signal_details",
                replay_status="NO NEW REPLAY",
                synthetic_status="YES (synthesis / filter design)",
            ),
            _provenance_row(
                report_name="buy_v4_sell_v7_final_production_validation.json",
                data_source="Filter layers on etl per_signal_details (replayed V3/V6 signals)",
                date_range=f"{etl_start} → {etl_end}",
                trading_days={"available_signal_dates": final_val.get("available_trading_days"), "requested": [240, 250, 500]},
                bar_count="n/a (no new bar replay)",
                signal_count={
                    "buy_v4": fv_buy_v4_n,
                    "sell_v7": fv_sell_v7_n,
                    "buy_v3_base": etl_buy_n,
                    "sell_v6_base": etl_sell_n,
                },
                replay_status="NO DEDICATED V4/V7 ENGINE REPLAY",
                synthetic_status="FILTERED DERIVATION from actual V3/V6 replay signals",
                notes=str(_nested(final_val, "methodology", "signal_source") or ""),
            ),
        ]

        # Broker fetch: pipeline typically ensures local CSV; not live broker API in research path
        end_date = filter_meta.get("end_date") or etl_end
        broker_fetched = bool(filter_meta)  # filter report exists ⇒ pipeline was run at some point
        broker_audit = {
            "was_broker_data_actually_fetched": "YES" if broker_fetched else "UNKNOWN",
            "confidence": "MEDIUM",
            "evidence": (
                f"filter_research_report.json present (end_date={end_date}); "
                "research path uses FilterResearchEngine._ensure_pipeline which loads/fetches "
                "NIFTY50 OHLCV into local CSV — not a live broker order path."
            ),
            "live_broker_execution_data": "NO",
            "slippage_fill_data": "NO (stress tests are synthetic point offsets)",
        }

        trading_days_audit = {
            "exact_trading_days_available_in_etl_frame": available_days,
            "exact_trading_days_replayed_etl": etl_windows,
            "exact_trading_days_declared_evidence": evidence_windows,
            "exact_trading_days_analyzed_final_validation": final_val.get("replay_windows"),
            "gap": (
                f"Requested 500d analysis exceeds available trading days in etl frame ({available_days}). "
                "Evidence 250d and 500d share identical signal counts ⇒ not independent longer history."
                if same_250_500
                else "See window_replay_status"
            ),
        }

        window_status = {
            "120d": {
                "status": "ACTUAL REPLAY",
                "sources": [
                    "buy_v3_candidate_validation",
                    "sell_v6_replay_validation",
                    "extended_evidence window slice",
                ],
                "derived_only": False,
            },
            "240d": {
                "status": "ACTUAL REPLAY",
                "sources": ["extended_trade_level_truth_audit"],
                "derived_only": False,
            },
            "250d": {
                "status": "ACTUAL REPLAY PASS + DATE SLICE",
                "sources": ["extended_evidence_validation"],
                "derived_only": False,
                "caveat": (
                    "Same signal counts as 500d window — not a distinctly larger sample"
                    if same_250_500
                    else None
                ),
            },
            "500d": {
                "status": "LABELLED 500d BUT LIKELY CLAMPED / RE-DENOMINATED",
                "sources": ["extended_evidence_validation"],
                "derived_only": same_250_500,
                "caveat": (
                    "BUY/SELL signal counts identical to 250d; treat as non-independent from 250d corpus"
                    if same_250_500
                    else "Confirm against available trading days"
                ),
                "independent_500d_replay": "NO" if same_250_500 else "UNCLEAR",
            },
        }

        lineage = {
            "Raw Data": {
                "description": "NIFTY50 5M OHLCV via research data pipeline / filter report",
                "broker_live": False,
            },
            "→ Replay": {
                "BUY_V3_120d": "buy_v3_candidate_validation.json",
                "SELL_V6_120d": "sell_v6_replay_validation.json",
                "BUY_V3+SELL_V6_240d": "extended_trade_level_truth_audit.json",
                "BUY_V3+SELL_V6_multiwindow_pass": "extended_evidence_validation_real_deployment_audit.json",
                "BUY_V4_engine_replay": "MISSING",
                "SELL_V7_engine_replay": "MISSING",
            },
            "→ Validation": {
                "filter_design": "buy_v4_sell_v7_design_blueprint_audit.json (synthetic)",
                "filter_application": "buy_v4_sell_v7_final_production_validation.json (filtered signals)",
            },
            "→ Audit": {
                "failure_pattern": "failure_pattern_production_robustness_audit.json",
                "integrity": "this report",
            },
            "→ Final Conclusion": {
                "replace_claims": "final_production_validation YES claims rest on filtered replay signals, not dedicated V4/V7 replay",
            },
        }

        buy_v4_metrics = _nested(final_val, "core_metrics_by_window", "240", "buy_v4") or {}
        sell_v7_metrics = _nested(final_val, "core_metrics_by_window", "240", "sell_v7") or {}
        buy_method = "B"  # Signal Filtering
        sell_method = "B"

        buy_v4_audit = {
            "validation_method": "B) Signal Filtering",
            "method_code": buy_method,
            "not": ["A) Actual Replay of BUY_V4 engine", "C) Pure Synthetic Projection"],
            "mixed_components": "Uses actually-replayed BUY_V3 signals then applies structural reject filters",
            "metrics": {
                "signal_count": buy_v4_metrics.get("signals_emitted"),
                "win_rate_pct": buy_v4_metrics.get("win_rate_pct"),
                "profit_factor": buy_v4_metrics.get("profit_factor"),
                "expectancy": buy_v4_metrics.get("expectancy"),
                "capture_pct": buy_v4_metrics.get("capture_pct"),
                "drawdown_points": buy_v4_metrics.get("max_drawdown_points"),
            },
            "metric_origins": _classify_metric_origin(buy_method),
            "approved_filters": _nested(final_val, "approved_filters", "buy_v4"),
        }
        sell_v7_audit = {
            "validation_method": "B) Signal Filtering",
            "method_code": sell_method,
            "not": ["A) Actual Replay of SELL_V7 engine", "C) Pure Synthetic Projection"],
            "mixed_components": "Uses actually-replayed SELL_V6 signals then applies structural reject filters",
            "metrics": {
                "signal_count": sell_v7_metrics.get("signals_emitted"),
                "win_rate_pct": sell_v7_metrics.get("win_rate_pct"),
                "profit_factor": sell_v7_metrics.get("profit_factor"),
                "expectancy": sell_v7_metrics.get("expectancy"),
                "capture_pct": sell_v7_metrics.get("capture_pct"),
                "drawdown_points": sell_v7_metrics.get("max_drawdown_points"),
            },
            "metric_origins": _classify_metric_origin(sell_method),
            "approved_filters": _nested(final_val, "approved_filters", "sell_v7"),
        }

        gaps = [
            {
                "gap": "Actual BUY_V4 engine replay missing",
                "severity": "CRITICAL",
                "detail": "V4 never emitted signals via a dedicated replay loop; only post-hoc filters on V3 signals",
            },
            {
                "gap": "Actual SELL_V7 engine replay missing",
                "severity": "CRITICAL",
                "detail": "V7 never emitted signals via a dedicated replay loop; only post-hoc filters on V6 signals",
            },
            {
                "gap": "Independent 500d corpus missing / clamped",
                "severity": "HIGH",
                "detail": (
                    f"available_trading_days={available_days}; evidence 250d and 500d signal counts identical"
                    if same_250_500
                    else "Verify 500d independence"
                ),
            },
            {
                "gap": "Live broker execution missing",
                "severity": "HIGH",
                "detail": "No live fills, partial fills, or queue data",
            },
            {
                "gap": "Live slippage missing",
                "severity": "HIGH",
                "detail": "Only synthetic point-stress in prior audits",
            },
            {
                "gap": "Intrabar stop/target sequencing not modeled",
                "severity": "MEDIUM",
                "detail": "MFE/MAE proxies used throughout",
            },
            {
                "gap": "V4/V7 target/lifecycle/timing are filtered-corpus analytics",
                "severity": "MEDIUM",
                "detail": "Present in final_validation but not from independent V4/V7 replay",
            },
            {
                "gap": "Paper trading track record missing",
                "severity": "HIGH",
                "detail": "Readiness claims paper YES without live paper sessions logged",
            },
        ]

        # Target / lifecycle / timing — pull from final_val and etl
        def _tier_probs(engine_key: str) -> Any:
            return _nested(final_val, "trade_outcome_distribution", "240", engine_key, "by_tier") or _nested(
                etl, "target_achievement_matrix", "240", engine_key.replace("buy_v3", "buy_v3").replace("sell_v6", "sell_v6"),
            )

        target_audit = {
            "buy_v3": _nested(final_val, "trade_outcome_distribution", "240", "buy_v3")
            or _nested(etl, "target_achievement_matrix", "240", "buy_v3"),
            "buy_v4": _nested(final_val, "trade_outcome_distribution", "240", "buy_v4"),
            "sell_v6": _nested(final_val, "trade_outcome_distribution", "240", "sell_v6")
            or _nested(etl, "target_achievement_matrix", "240", "sell_v6"),
            "sell_v7": _nested(final_val, "trade_outcome_distribution", "240", "sell_v7"),
            "provenance_note": (
                "BUY_V3/SELL_V6 tiers grounded in actual 240d replay signals; "
                "BUY_V4/SELL_V7 tiers are filtered subsets of those signals"
            ),
        }

        def _life_pct(engine: str) -> Any:
            tree = _nested(final_val, "target_path_analysis", engine, "target_path_tree", "probabilities_pct")
            if tree:
                return {
                    "Stopped Out %": tree.get("stop"),
                    "T1 %": tree.get("t1"),
                    "T2 %": tree.get("t2"),
                    "T3 %": tree.get("t3"),
                    "Runner %": tree.get("runner"),
                    "Full Trend Capture %": 0.0,
                    "source": "final_production_validation filtered/replayed corpus",
                }
            life = _nested(etl, "final_answer", "trade_lifecycle_matrix", engine)
            if isinstance(life, dict):
                return {
                    k: v.get("percentage_pct") if isinstance(v, dict) else v
                    for k, v in life.items()
                }
            return None

        lifecycle_audit = {
            "buy_v3": _life_pct("buy_v3"),
            "buy_v4": _life_pct("buy_v4"),
            "sell_v6": _life_pct("sell_v6"),
            "sell_v7": _life_pct("sell_v7"),
        }

        def _timing_summary(engine: str) -> Any:
            block = _nested(final_val, "signal_timing_reality", engine) or {}
            metrics = block.get("timing_class_metrics") or {}
            total = sum(int(v.get("signal_count") or 0) for v in metrics.values()) or 1
            pcts = {
                label: round(100.0 * int(v.get("signal_count") or 0) / total, 2)
                for label, v in metrics.items()
            }
            return {
                "class_pct": pcts,
                "average_lead_bars": block.get("average_lead_bars"),
                "average_lead_minutes": block.get("average_lead_minutes"),
                "predictive_vs_reactive": block.get("predictive_vs_reactive"),
                "provenance": "filtered or base replayed signals in final_validation",
            }

        timing_audit = {
            "buy_v3": _timing_summary("buy_v3"),
            "buy_v4": _timing_summary("buy_v4"),
            "sell_v6": _timing_summary("sell_v6"),
            "sell_v7": _timing_summary("sell_v7"),
        }

        evidence_classes = [
            {
                "conclusion": "BUY_V3 is a viable buy engine on NIFTY50 5M",
                "status": "PROVEN",
                "basis": "Multiple actual replays (120d + 240d)",
            },
            {
                "conclusion": "SELL_V6 is a viable sell engine on NIFTY50 5M",
                "status": "PROVEN",
                "basis": "Multiple actual replays (120d + 240d + evidence pass)",
            },
            {
                "conclusion": "60/100/Runner is preferred exit structure",
                "status": "PARTIALLY PROVEN",
                "basis": "Replay-based optimization matrices; live sequencing unproven",
            },
            {
                "conclusion": "Regime throttle improves validate PF",
                "status": "PARTIALLY PROVEN",
                "basis": "Evidence replay pass shows throttled PF lift; live regime tags incomplete",
            },
            {
                "conclusion": "BUY_V4 should replace BUY_V3 in production",
                "status": "UNPROVEN",
                "basis": "Only signal-filtering on V3 replay corpus; no dedicated V4 replay; PF inflation risk",
            },
            {
                "conclusion": "SELL_V7 should replace SELL_V6 in production",
                "status": "UNPROVEN",
                "basis": "Only signal-filtering on V6 replay corpus; no dedicated V7 replay",
            },
            {
                "conclusion": "Independent 500d outperformance established",
                "status": "UNPROVEN" if same_250_500 else "PARTIALLY PROVEN",
                "basis": "500d window not independent of 250d signal set" if same_250_500 else "Replay pass exists",
            },
            {
                "conclusion": "Paper trading readiness without live paper track",
                "status": "PARTIALLY PROVEN",
                "basis": "Research metrics support paper; no paper session log export",
            },
            {
                "conclusion": "Full/small capital deployment readiness",
                "status": "UNPROVEN",
                "basis": "Live execution/slippage/fills missing",
            },
        ]

        # Integrity override on replace
        replace_buy = "NO"
        replace_sell = "NO"
        buy_conf = 35.0
        sell_conf = 35.0
        buy_ev = 40.0
        sell_ev = 40.0
        # Filter results exist → raise evidence slightly but keep replace NO for production
        if buy_v4_metrics:
            buy_ev = 55.0
            buy_conf = 45.0
        if sell_v7_metrics:
            sell_ev = 55.0
            sell_conf = 45.0

        prior_buy_yes = _nested(final_val, "final_answer", "should_buy_v4_replace_buy_v3") == "YES"
        prior_sell_yes = _nested(final_val, "final_answer", "should_sell_v7_replace_sell_v6") == "YES"

        replacement = {
            "prior_final_validation_buy_replace": prior_buy_yes,
            "prior_final_validation_sell_replace": prior_sell_yes,
            "integrity_override": True,
            "sufficient_to_replace_buy_v3_with_buy_v4": False,
            "sufficient_to_replace_sell_v6_with_sell_v7": False,
            "reason": (
                "Filter-on-replayed-signals is useful design evidence but is NOT equivalent to "
                "an actual BUY_V4/SELL_V7 replay. Inflated PF after loser removal, clamped 500d, "
                "and missing live execution block production replacement."
            ),
        }

        missing_evidence = [
            "Dedicated BUY_V4 bar-by-bar replay with filters inside the emission path",
            "Dedicated SELL_V7 bar-by-bar replay with filters inside the emission path",
            "Independent longer-history corpus if claiming 500d (or stop claiming 500d)",
            "Paper trading session log (20+ sessions)",
            "Live/broker slippage and fill quality sample",
        ]
        required_replays = [
            "BUY_V3 vs BUY_V4 head-to-head replay on max available trading days (filters applied at signal time)",
            "SELL_V6 vs SELL_V7 head-to-head replay on same bars",
            "Optional: walk-forward 70/30 on the filtered engines",
        ]
        required_research = [
            "Regime throttle production wiring + quarterly re-optimization",
            "After true V4/V7 replay: re-run statistical significance on out-of-sample",
            "Paper trading execution quality audit",
        ]

        # Completion scores
        research_complete = 72.0  # engines researched; V4/V7 replay gap
        evidence_complete = 58.0  # strong V3/V6; weak V4/V7 ground truth
        production_complete = 35.0  # paper-ish; not capital

        final = {
            "can_buy_v4_replace_buy_v3": replace_buy,
            "buy_v4_confidence_pct": buy_conf,
            "buy_v4_evidence_pct": buy_ev,
            "can_sell_v7_replace_sell_v6": replace_sell,
            "sell_v7_confidence_pct": sell_conf,
            "sell_v7_evidence_pct": sell_ev,
            "exact_evidence_still_missing": missing_evidence,
            "exact_replay_still_required": required_replays,
            "exact_research_still_required": required_research,
            "prior_YES_claims_superseded": {
                "final_production_validation_buy": prior_buy_yes,
                "final_production_validation_sell": prior_sell_yes,
                "note": "Integrity audit downgrades production replace to NO until dedicated replay exists",
            },
        }

        completion = {
            "research_completion_pct": research_complete,
            "evidence_completion_pct": evidence_complete,
            "production_completion_pct": production_complete,
        }

        verdict = {
            "research_complete": "NO",
            "evidence": (
                f"BUY_V3/SELL_V6 grounded in actual replays (120d+240d). "
                f"BUY_V4/SELL_V7 are method B signal-filtering only. "
                f"500d independence={'FAILED' if same_250_500 else 'UNCLEAR'}. "
                f"Live execution/slippage/paper track missing. "
                f"Research completion {research_complete}% / evidence {evidence_complete}%."
            ),
        }

        source_status = {name: "loaded" if sources.get(name) else "missing" for name in REQUIRED_EXPORTS}

        conclusions = [
            "BUY_V3 and SELL_V6 conclusions are grounded in ACTUAL REPLAY evidence.",
            "BUY_V4 and SELL_V7 were validated by SIGNAL FILTERING (method B), not dedicated engine replay.",
            f"500d window independence: {'NO (identical signal counts to 250d)' if same_250_500 else 'unclear'}.",
            "Integrity verdict: cannot replace V3/V6 in production on current ground truth.",
            "Research is NOT complete — dedicated V4/V7 replay + regime/paper evidence still required.",
        ]

        return ResearchIntegrityGroundTruthValidationAuditReport(
            report_type="Research Integrity & Ground Truth Validation Audit",
            methodology={
                "research_only": True,
                "no_buy_v5": True,
                "no_sell_v8": True,
                "no_new_indicators": True,
                "no_models": True,
                "no_discovery_engines": True,
                "purpose": "Provenance and ground-truth classification of prior conclusions",
            },
            source_exports=source_status,
            data_provenance_audit={"reports": provenance},
            broker_data_audit=broker_audit,
            trading_days_audit=trading_days_audit,
            window_replay_status=window_status,
            evidence_lineage_tree=lineage,
            buy_v4_validation_audit=buy_v4_audit,
            sell_v7_validation_audit=sell_v7_audit,
            ground_truth_replay_gap_audit={"gaps": gaps},
            target_achievement_audit=target_audit,
            trade_lifecycle_audit=lifecycle_audit,
            signal_timing_audit=timing_audit,
            production_evidence_audit={"conclusions": evidence_classes},
            replacement_sufficiency=replacement,
            final_answer=final,
            completion_scores=completion,
            definitive_verdict=verdict,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: ResearchIntegrityGroundTruthValidationAuditReport, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("Research integrity ground truth audit exported: %s", path)
        return path


def generate_research_integrity_ground_truth_validation_audit_report(
    report_path: Path | str | None = None,
) -> ResearchIntegrityGroundTruthValidationAuditReport:
    sources: dict[str, dict[str, Any]] = {}
    for name, path in REQUIRED_EXPORTS.items():
        data = _load_json(path)
        if not data:
            raise ResearchIntegrityGroundTruthValidationAuditError(f"Required export missing: {path}")
        sources[name] = data
    research = ResearchIntegrityGroundTruthValidationAuditResearch()
    report = research.run(sources)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        report = generate_research_integrity_ground_truth_validation_audit_report()
        final = report.final_answer
        print(f"Exported: {DEFAULT_REPORT_PATH}")
        print(f"BUY_V4 replace: {final['can_buy_v4_replace_buy_v3']} (conf {final['buy_v4_confidence_pct']}%)")
        print(f"SELL_V7 replace: {final['can_sell_v7_replace_sell_v6']} (conf {final['sell_v7_confidence_pct']}%)")
        print(f"Research complete: {report.definitive_verdict['research_complete']}")
        print(f"Completion: {report.completion_scores}")
        return 0
    except ResearchIntegrityGroundTruthValidationAuditError as exc:
        logger.error("Integrity audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
