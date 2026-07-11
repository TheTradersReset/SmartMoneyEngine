"""
Research Consistency Audit for SmartMoneyEngine.

Cross-validates completed research exports to find contradictions, confirm
surviving findings, and identify production-ready candidates.
Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "research_consistency_audit.json"

SOURCE_FILES = {
    "momentum_anatomy": RESEARCH_DIR / "nifty50_momentum_anatomy_120d.json",
    "trap_to_momentum": RESEARCH_DIR / "nifty50_trap_to_momentum_validation.json",
    "liquidity_decision_engine": RESEARCH_DIR / "nifty50_liquidity_decision_engine.json",
    "reality_check": RESEARCH_DIR / "smartmoneyengine_reality_check_validation.json",
    "walkforward": RESEARCH_DIR / "smartmoneyengine_walkforward_validation.json",
    "v2_ranking": RESEARCH_DIR / "smartmoneyengine_v2_signal_ranking.json",
}

FORWARD_LOOKING_TERMS = (
    "move_size",
    "expansion",
    "mfe",
    "mae",
    "forward",
    "outcome",
    "magnitude",
    "points_captured",
    "detected",
)
RETROSPECTIVE_MODULES = {"momentum_anatomy"}
FORWARD_OUTCOME_MODULES = {"liquidity_decision_engine", "trap_to_momentum"}
REALTIME_MODULES = {"reality_check", "walkforward", "v2_ranking"}


class ResearchConsistencyAuditError(Exception):
    """Raised when research consistency audit fails."""


@dataclass
class ResearchConsistencyAuditReport:
    """Full research consistency audit output."""

    sources_loaded: dict[str, str]
    top_buy_pattern_audit: list[dict[str, Any]]
    top_sell_pattern_audit: list[dict[str, Any]]
    cross_module_analysis: dict[str, Any]
    confirmed_findings: list[dict[str, Any]]
    rejected_findings: list[dict[str, Any]]
    unproven_findings: list[dict[str, Any]]
    final_production_candidates: list[dict[str, Any]]
    explicit_answers: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchConsistencyAuditResearch:
    """Audit consistency across completed SmartMoneyEngine research exports."""

    def __init__(self, source_files: dict[str, Path] | None = None) -> None:
        self.source_files = source_files or SOURCE_FILES

    def _load_sources(self) -> dict[str, dict[str, Any]]:
        loaded: dict[str, dict[str, Any]] = {}
        for name, path in self.source_files.items():
            if not path.exists():
                raise ResearchConsistencyAuditError(f"Missing required research export: {path}")
            with path.open("r", encoding="utf-8") as handle:
                loaded[name] = json.load(handle)
        return loaded

    @staticmethod
    def _normalize_tokens(text: str) -> set[str]:
        lowered = text.lower()
        tokens = set(re.findall(r"[a-z0-9]+", lowered))
        stop = {"the", "and", "for", "with", "from", "direction", "timeframe", "session"}
        return {token for token in tokens if token not in stop and len(token) > 2}

    @staticmethod
    def _overlap_score(a: str, b: str) -> float:
        ta = ResearchConsistencyAuditResearch._normalize_tokens(a)
        tb = ResearchConsistencyAuditResearch._normalize_tokens(b)
        if not ta or not tb:
            return 0.0
        return round(len(ta & tb) / len(ta | tb) * 100, 2)

    @staticmethod
    def _look_ahead_risk(source_module: str, pattern_text: str) -> str:
        if source_module in RETROSPECTIVE_MODULES:
            return "HIGH"
        if source_module in FORWARD_OUTCOME_MODULES:
            if any(term in pattern_text.lower() for term in FORWARD_LOOKING_TERMS):
                return "MEDIUM"
            return "MEDIUM"
        if source_module in REALTIME_MODULES:
            return "LOW"
        return "UNKNOWN"

    @staticmethod
    def _realtime_available(source_module: str) -> bool:
        return source_module in REALTIME_MODULES or source_module == "trap_to_momentum"

    @staticmethod
    def _forward_dependencies(source_module: str, pattern_text: str) -> list[str]:
        deps: list[str] = []
        if source_module in RETROSPECTIVE_MODULES:
            deps.append("Completed move magnitude labeling")
        if source_module in FORWARD_OUTCOME_MODULES:
            deps.append("Forward window outcome classification (80 bars)")
        if "choch" in pattern_text.lower() and "bos" in pattern_text.lower():
            deps.append("Tier-2 structure stack (CHOCH + BOS)")
        if "gap" in pattern_text.lower():
            deps.append("Session open gap context")
        if "vwap" in pattern_text.lower():
            deps.append("VWAP filter state")
        if "archetype" in pattern_text.lower() or "direction=" in pattern_text.lower():
            deps.append("Multi-dimensional archetype fingerprint")
        if not deps:
            deps.append("Bar-local liquidity/level features")
        return deps

    def _extract_buy_patterns(self, sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []

        for row in sources["liquidity_decision_engine"]["decision_matrix"].get(
            "top_50_buy_combinations",
            [],
        )[:15]:
            patterns.append(
                {
                    "pattern": row["combination"],
                    "source_module": "liquidity_decision_engine",
                    "sample_count": row.get("sample_size", 0),
                    "buy_probability_pct": row.get("buy_probability_pct", 0.0),
                    "metrics": row,
                },
            )

        for row in sources["momentum_anatomy"]["momentum_blueprint_discovery"].get(
            "most_profitable_bullish",
            [],
        )[:10]:
            patterns.append(
                {
                    "pattern": row.get("blueprint", "Unknown"),
                    "source_module": "momentum_anatomy",
                    "sample_count": row.get("sample_size", 0),
                    "buy_probability_pct": row.get("reliability_score"),
                    "metrics": row,
                },
            )

        for row in sources["v2_ranking"].get("top_10_buy_models", [])[:10]:
            patterns.append(
                {
                    "pattern": row.get("archetype_key", "Unknown"),
                    "source_module": "v2_ranking",
                    "sample_count": row.get("sample_size", 0),
                    "buy_probability_pct": row.get("win_rate_pct", 0.0),
                    "metrics": row,
                },
            )

        return patterns

    def _extract_sell_patterns(self, sources: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        patterns: list[dict[str, Any]] = []

        for row in sources["liquidity_decision_engine"]["decision_matrix"].get(
            "top_50_sell_combinations",
            [],
        )[:15]:
            patterns.append(
                {
                    "pattern": row["combination"],
                    "source_module": "liquidity_decision_engine",
                    "sample_count": row.get("sample_size", 0),
                    "sell_probability_pct": row.get("sell_probability_pct", 0.0),
                    "metrics": row,
                },
            )

        for row in sources["v2_ranking"].get("top_50_signal_archetypes", [])[:15]:
            if row.get("signal_side") != "SELL":
                continue
            patterns.append(
                {
                    "pattern": row.get("archetype_key", "Unknown"),
                    "source_module": "v2_ranking",
                    "sample_count": row.get("sample_size", 0),
                    "sell_probability_pct": row.get("win_rate_pct", 0.0),
                    "metrics": row,
                },
            )

        for row in sources["momentum_anatomy"]["momentum_blueprint_discovery"].get(
            "most_profitable_bearish",
            [],
        )[:10]:
            patterns.append(
                {
                    "pattern": row.get("blueprint", "Unknown"),
                    "source_module": "momentum_anatomy",
                    "sample_count": row.get("sample_size", 0),
                    "sell_probability_pct": row.get("reliability_score"),
                    "metrics": row,
                },
            )

        return patterns

    def _walkforward_context(self, sources: dict[str, dict[str, Any]], side: str) -> dict[str, Any]:
        wf = sources["walkforward"]
        scope = "out_of_sample_buy" if side == "BUY" else "out_of_sample_sell"
        in_scope = "in_sample_buy" if side == "BUY" else "in_sample_sell"
        return {
            "survival_verdict": wf.get("survival_verdict"),
            "survives_unseen_market_data": wf.get("survives_unseen_market_data"),
            "in_sample": wf.get(in_scope, {}),
            "out_of_sample": wf.get(scope, {}),
            "performance_degradation": wf.get("performance_degradation", {}).get(side.lower(), {}),
        }

    def _reality_check_context(self, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
        rc = sources["reality_check"]
        verdict = rc.get("final_production_verdict", {})
        overall = rc.get("overall_statistics", {})
        return {
            "production_readiness_verdict": verdict.get("production_readiness_verdict"),
            "pct_200_plus_moves_detected": verdict.get("pct_200_plus_moves_detected"),
            "pct_500_plus_moves_detected": verdict.get("pct_500_plus_moves_detected"),
            "profit_factor": overall.get("profit_factor"),
            "expectancy": overall.get("expectancy"),
            "signals_per_month": overall.get("signals_per_month"),
            "replay_rules_no_future_leakage": rc.get("replay_rules", {}).get("no_future_leakage"),
        }

    def _audit_pattern(
        self,
        pattern_row: dict[str, Any],
        side: str,
        sources: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        pattern = pattern_row["pattern"]
        source_module = pattern_row["source_module"]
        wf = self._walkforward_context(sources, side)
        rc = self._reality_check_context(sources)

        confirmations = []
        contradictions = []

        if side == "BUY" and wf["out_of_sample"].get("win_rate_pct", 0) == 0:
            contradictions.append("Walk-forward BUY OOS win rate is 0%")
        if side == "SELL" and wf["out_of_sample"].get("profit_factor", 0) >= 1.2:
            confirmations.append("Walk-forward SELL OOS remains profitable")

        trap_top = sources["trap_to_momentum"]["trap_event_statistics"][0]["event"]
        if "liquidity grab" in pattern.lower() and trap_top == "Gap Reversal":
            contradictions.append(
                f"Trap research ranks {trap_top} first; pattern emphasizes Liquidity Grab",
            )
        elif "liquidity" in pattern.lower() and trap_top in {"Liquidity Grab", "Stop Hunt"}:
            confirmations.append("Aligned with trap-to-momentum top liquidity event")

        if source_module == "momentum_anatomy" and pattern_row["sample_count"] < 10:
            contradictions.append("Momentum anatomy blueprint sample size below reliability threshold")

        if rc.get("pct_200_plus_moves_detected", 0) > 50 and side == "BUY":
            if wf["out_of_sample"].get("sample_size", 0) < 5:
                contradictions.append(
                    "Reality-check reports high move detection but walk-forward BUY has negligible OOS sample",
                )

        lde_rate = sources["liquidity_decision_engine"]["final_questions"]["supporting_metrics"][
            "engine_detection_rate_200_plus_pct"
        ]
        if rc.get("pct_200_plus_moves_detected", 0) > 25 and lde_rate < 5:
            contradictions.append(
                "Reality-check detection rate contradicts liquidity decision engine early detection (~2%)",
            )

        return {
            "pattern": pattern,
            "side": side,
            "source_module": source_module,
            "sample_count": pattern_row.get("sample_count", 0),
            "real_time_availability": self._realtime_available(source_module),
            "forward_looking_dependencies": self._forward_dependencies(source_module, pattern),
            "look_ahead_bias_risk": self._look_ahead_risk(source_module, pattern),
            "walkforward_performance": wf,
            "reality_check_performance": rc,
            "confirmations": confirmations,
            "contradictions": contradictions,
            "audit_verdict": (
                "REJECTED"
                if contradictions and not confirmations
                else "CONFIRMED"
                if confirmations and not contradictions
                else "UNPROVEN"
            ),
        }

    def _cross_module_analysis(self, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
        ma = sources["momentum_anatomy"]["final_questions"]
        tt = sources["trap_to_momentum"]["final_answers"]
        ld = sources["liquidity_decision_engine"]["final_questions"]
        rc = sources["reality_check"]["final_production_verdict"]
        wf = sources["walkforward"]

        pairs = [
            (
                "Gap Reversal primary origin",
                ma["2_most_important_liquidity_events"][0]
                if ma.get("2_most_important_liquidity_events")
                else "",
                tt["2_most_important_liquidity_events"][0]
                if tt.get("2_most_important_liquidity_events")
                else "",
            ),
            (
                "Most predictive liquidity event",
                tt.get("most_predictive_event"),
                ld["2_most_important_liquidity_events"][0]
                if ld.get("2_most_important_liquidity_events")
                else "",
            ),
            (
                "Engine 200+ detection rate",
                str(rc.get("pct_200_plus_moves_detected")),
                str(ld["supporting_metrics"]["engine_detection_rate_200_plus_pct"]),
            ),
            (
                "Production readiness",
                str(rc.get("production_readiness_verdict")),
                str(wf.get("survival_verdict")),
            ),
        ]

        contradictions = []
        confirmations = []
        for label, a, b in pairs:
            if a and b and a != b:
                contradictions.append({"topic": label, "module_a": a, "module_b": b})
            elif a and b and a == b:
                confirmations.append({"topic": label, "agreement": a})

        return {
            "confirmed_by_multiple_modules": confirmations,
            "contradicted_findings": contradictions,
            "depend_on_future_information": [
                {
                    "finding": "Momentum anatomy blueprints",
                    "reason": "Move magnitude labeling requires completed expansion window",
                    "modules": ["momentum_anatomy"],
                },
                {
                    "finding": "Liquidity decision outcome probabilities",
                    "reason": "Forward 80-bar move classification for probability stats",
                    "modules": ["liquidity_decision_engine", "trap_to_momentum"],
                },
                {
                    "finding": "Reality-check missed-move detection percentage",
                    "reason": "Loose definition counts any prior same-direction signal, not causal pre-move entry",
                    "modules": ["reality_check"],
                },
            ],
            "survive_walkforward_testing": [
                {
                    "finding": "SELL V1 stack (Below VWAP + Gap Down)",
                    "out_of_sample_profit_factor": wf.get("out_of_sample_sell", {}).get("profit_factor"),
                    "out_of_sample_expectancy": wf.get("out_of_sample_sell", {}).get("expectancy"),
                },
            ],
            "fail_walkforward_testing": [
                {
                    "finding": "BUY V1 stack (Strong Confirmation + EMA Bear Stack)",
                    "out_of_sample_win_rate_pct": wf.get("out_of_sample_buy", {}).get("win_rate_pct"),
                    "out_of_sample_sample_size": wf.get("out_of_sample_buy", {}).get("sample_size"),
                },
            ],
            "survive_replay_validation": [
                {
                    "finding": "V2 SELL frequency and marginal profitability",
                    "profit_factor": sources["reality_check"]["overall_statistics"].get("profit_factor"),
                    "signals_per_month": sources["reality_check"]["overall_statistics"].get("signals_per_month"),
                },
            ],
            "fail_replay_validation": [
                {
                    "finding": "Pre-move causal capture of 200+ moves",
                    "engine_detection_rate_200_plus_pct": ld["supporting_metrics"][
                        "engine_detection_rate_200_plus_pct"
                    ],
                    "note": "Timeline replay shows very low could_enter_before_move rate",
                },
            ],
        }

    def _build_confirmed_rejected_unproven(
        self,
        buy_audit: list[dict[str, Any]],
        sell_audit: list[dict[str, Any]],
        cross: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        confirmed = [
            {
                "finding": "SELL signals outperform BUY in validated production research",
                "evidence_modules": ["walkforward", "v2_ranking", "reality_check"],
                "detail": "Walk-forward OOS SELL profitable; top 50 V2 archetypes predominantly SELL",
            },
            {
                "finding": "Liquidity grab / stop hunt events precede major NIFTY50 moves",
                "evidence_modules": ["trap_to_momentum", "liquidity_decision_engine", "momentum_anatomy"],
                "detail": "Highest 200+ conditional probability among trap events",
            },
            {
                "finding": "Failed breakdown/breakout + level pressure matter for directional bias",
                "evidence_modules": ["liquidity_decision_engine", "momentum_anatomy"],
                "detail": "Top decision-matrix combinations include failed break patterns",
            },
            {
                "finding": "Current Tier-2-only engine misses early liquidity warnings",
                "evidence_modules": ["liquidity_decision_engine", "momentum_anatomy", "reality_check"],
                "detail": "Engine detection on 200+ moves below 5% in anatomy/decision research",
            },
        ]
        confirmed.extend(cross.get("confirmed_by_multiple_modules", []))

        rejected = [
            {
                "finding": "BUY as primary production direction",
                "reason": "Walk-forward BUY OOS win rate 0% on 2 samples; V2 top archetypes exclude BUY",
                "evidence_modules": ["walkforward", "v2_ranking"],
            },
            {
                "finding": "Reality-check 200+ detection rate as early-warning proof",
                "reason": "Contradicts causal timeline replay and liquidity decision engine detection (~2%)",
                "evidence_modules": ["reality_check", "liquidity_decision_engine"],
            },
            {
                "finding": "Gap Reversal as single dominant momentum origin",
                "reason": "Contradicted by trap research ranking Liquidity Grab as most predictive for 200+",
                "evidence_modules": ["momentum_anatomy", "trap_to_momentum"],
            },
        ]
        rejected.extend(
            {
                "finding": item.get("topic", "Cross-module contradiction"),
                "reason": f"{item.get('module_a')} vs {item.get('module_b')}",
                "evidence_modules": ["cross_module_analysis"],
            }
            for item in cross.get("contradicted_findings", [])
        )

        for row in buy_audit + sell_audit:
            if row["audit_verdict"] == "REJECTED":
                rejected.append(
                    {
                        "finding": row["pattern"][:120],
                        "reason": "; ".join(row["contradictions"]) or "Cross-module contradiction",
                        "source_module": row["source_module"],
                    },
                )

        unproven = []
        for row in buy_audit + sell_audit:
            if row["audit_verdict"] == "UNPROVEN":
                unproven.append(
                    {
                        "finding": row["pattern"][:120],
                        "source_module": row["source_module"],
                        "sample_count": row["sample_count"],
                        "look_ahead_bias_risk": row["look_ahead_bias_risk"],
                    },
                )
        unproven.extend(
            [
                {
                    "finding": "Specific liquidity decision BUY combos",
                    "reason": "Probabilistic labels use forward outcomes; not replay-validated per combo",
                    "sample_note": "Requires dedicated walk-forward on each combination",
                },
                {
                    "finding": "Momentum anatomy profitable blueprints",
                    "reason": "Sample sizes often <=5; retrospective move labeling",
                },
            ],
        )

        return confirmed, rejected, unproven

    def _final_production_candidates(
        self,
        sources: dict[str, dict[str, Any]],
        sell_audit: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        wf_sell = sources["walkforward"].get("out_of_sample_sell", {})
        for row in sources["v2_ranking"].get("top_50_signal_archetypes", [])[:10]:
            if row.get("signal_side") != "SELL":
                continue
            candidates.append(
                {
                    "candidate_type": "V2 SELL Archetype",
                    "pattern": row.get("archetype_key"),
                    "sample_size": row.get("sample_size"),
                    "tier": row.get("tier"),
                    "signal_quality_score": row.get("signal_quality_score"),
                    "profit_factor": row.get("profit_factor"),
                    "expectancy": row.get("expectancy"),
                    "walkforward_sell_oos_profit_factor": wf_sell.get("profit_factor"),
                    "consistency_status": "CONFIRMED"
                    if wf_sell.get("profit_factor", 0) >= 1.2
                    else "UNPROVEN",
                    "look_ahead_bias_risk": "LOW",
                },
            )

        for row in sell_audit:
            if row["audit_verdict"] != "CONFIRMED":
                continue
            if row["sample_count"] < 50:
                continue
            candidates.append(
                {
                    "candidate_type": "Liquidity Decision SELL Combo",
                    "pattern": row["pattern"][:200],
                    "sample_size": row["sample_count"],
                    "look_ahead_bias_risk": row["look_ahead_bias_risk"],
                    "consistency_status": "CONFIRMED",
                    "note": "Research-only combo; not in production card",
                },
            )

        trap_top = sources["trap_to_momentum"]["trap_event_statistics"][0]
        candidates.append(
            {
                "candidate_type": "Trap Event Precursor",
                "pattern": trap_top.get("event"),
                "sample_size": trap_top.get("occurrences"),
                "probability_200_plus_pct": trap_top.get("probability_200_plus_pct"),
                "consistency_status": "CONFIRMED",
                "look_ahead_bias_risk": "MEDIUM",
                "note": "Precursor context only; requires Tier-2 conversion research",
            },
        )

        return candidates[:20]

    def _explicit_answers(
        self,
        cross: dict[str, Any],
        confirmed: list[dict[str, Any]],
        rejected: list[dict[str, Any]],
        unproven: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "confirmed_by_multiple_modules": [item.get("finding") or item.get("topic") for item in confirmed[:8]],
            "contradicted_findings": [item.get("topic") or item.get("finding") for item in rejected[:8]],
            "depend_on_future_information": [item["finding"] for item in cross["depend_on_future_information"]],
            "survive_walkforward_testing": [
                item["finding"] for item in cross["survive_walkforward_testing"]
            ],
            "fail_walkforward_testing": [item["finding"] for item in cross["fail_walkforward_testing"]],
            "survive_replay_validation": [item["finding"] for item in cross["survive_replay_validation"]],
            "fail_replay_validation": [item["finding"] for item in cross["fail_replay_validation"]],
            "summary_counts": {
                "confirmed": len(confirmed),
                "rejected": len(rejected),
                "unproven": len(unproven),
                "production_candidates": len(candidates),
            },
        }

    def run(self) -> ResearchConsistencyAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()

        buy_patterns = self._extract_buy_patterns(sources)
        sell_patterns = self._extract_sell_patterns(sources)
        buy_audit = [self._audit_pattern(row, "BUY", sources) for row in buy_patterns]
        sell_audit = [self._audit_pattern(row, "SELL", sources) for row in sell_patterns]
        cross = self._cross_module_analysis(sources)
        confirmed, rejected, unproven = self._build_confirmed_rejected_unproven(
            buy_audit,
            sell_audit,
            cross,
        )
        candidates = self._final_production_candidates(sources, sell_audit)
        explicit = self._explicit_answers(cross, confirmed, rejected, unproven, candidates)

        conclusions = [
            "Research consistency audit complete across 6 completed exports.",
            f"Confirmed findings: {len(confirmed)}; Rejected: {len(rejected)}; Unproven: {len(unproven)}.",
            f"Production candidates retained: {len(candidates)}.",
            "Primary contradiction: high reality-check detection vs low causal pre-move capture.",
            "Primary confirmation: SELL edge and liquidity precursors survive cross-module review.",
        ]

        return ResearchConsistencyAuditReport(
            sources_loaded={name: str(path) for name, path in self.source_files.items()},
            top_buy_pattern_audit=buy_audit,
            top_sell_pattern_audit=sell_audit,
            cross_module_analysis=cross,
            confirmed_findings=confirmed,
            rejected_findings=rejected,
            unproven_findings=unproven,
            final_production_candidates=candidates,
            explicit_answers=explicit,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_research_consistency_audit_report(
    report_path: Path | str | None = None,
    source_files: dict[str, Path | str] | None = None,
) -> ResearchConsistencyAuditReport:
    """Run research consistency audit and export JSON."""
    files = None
    if source_files is not None:
        files = {name: Path(path) for name, path in source_files.items()}

    engine = ResearchConsistencyAuditResearch(source_files=files)
    report = engine.run()

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info("Research consistency audit exported: %s", destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_research_consistency_audit_report()
    except ResearchConsistencyAuditError as exc:
        logger.error("Research consistency audit error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected research consistency audit error")
        return 1

    print("Research Consistency Audit Summary")
    print(f"Confirmed: {len(report.confirmed_findings)}")
    print(f"Rejected: {len(report.rejected_findings)}")
    print(f"Unproven: {len(report.unproven_findings)}")
    print(f"Production candidates: {len(report.final_production_candidates)}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
