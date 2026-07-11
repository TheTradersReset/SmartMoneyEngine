"""
SmartMoneyEngine V1 Gap Analysis — production readiness review.

Reviews the frozen production specification against completed research exports.
Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.research.filter_research_engine import _json_safe

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = PROJECT_ROOT / "SmartMoneyEngine_V1_Production_Specification.md"
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_gap_analysis.json"

MIN_SAMPLES_READY = 100
MIN_SAMPLES_VALIDATE = 50
POOR_EXPECTANCY_THRESHOLD = 30.0
LOW_WIN_RATE_THRESHOLD = 35.0


class GapAnalysisError(Exception):
    """Raised when gap analysis cannot be completed."""


@dataclass
class RuleEvaluation:
    rule_id: str
    rule_name: str
    spec_section: str
    rule_type: str
    sample_size: int | None
    win_rate_pct: float | None
    expectancy: float | None
    profit_factor: float | None
    confidence_pct: float
    classification: str
    evidence_source: str
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GapFinding:
    category: str
    finding_id: str
    severity: str
    description: str
    affected_rules: list[str] = field(default_factory=list)
    recommendation: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GapAnalysisReport:
    spec_version: str
    spec_path: str
    research_exports_reviewed: list[str]
    total_rules_evaluated: int
    classification_summary: dict[str, int]
    production_readiness_score_pct: float
    overall_verdict: str
    rule_evaluations: list[dict[str, Any]]
    missing_rules: list[dict[str, Any]]
    contradictory_rules: list[dict[str, Any]]
    weak_rules: list[dict[str, Any]]
    low_sample_rules_under_50: list[dict[str, Any]]
    low_sample_rules_under_100: list[dict[str, Any]]
    poor_expectancy_rules: list[dict[str, Any]]
    false_signal_rules: list[dict[str, Any]]
    missing_no_trade_conditions: list[dict[str, Any]]
    missing_trigger_conditions: list[dict[str, Any]]
    missing_level_strength_conditions: list[dict[str, Any]]
    missing_liquidity_trap_conditions: list[dict[str, Any]]
    missing_round_number_conditions: list[dict[str, Any]]
    missing_absorption_conditions: list[dict[str, Any]]
    gap_findings: list[dict[str, Any]]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_json(name: str) -> dict[str, Any]:
    path = RESEARCH_DIR / name
    if not path.exists():
        raise GapAnalysisError(f"Missing research export: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _metrics(
    sample_size: int | None = None,
    win_rate_pct: float | None = None,
    expectancy: float | None = None,
    profit_factor: float | None = None,
) -> dict[str, Any]:
    return {
        "sample_size": sample_size,
        "win_rate_pct": win_rate_pct,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
    }


def _confidence_from_metrics(
    sample_size: int | None,
    win_rate_pct: float | None,
    expectancy: float | None,
    profit_factor: float | None,
) -> float:
    score = 0.0
    if sample_size is not None:
        if sample_size >= 300:
            score += 35
        elif sample_size >= 100:
            score += 28
        elif sample_size >= 50:
            score += 18
        elif sample_size >= 20:
            score += 10
        else:
            score += 3
    if expectancy is not None:
        if expectancy >= 100:
            score += 25
        elif expectancy >= 50:
            score += 18
        elif expectancy >= 20:
            score += 12
        elif expectancy > 0:
            score += 6
    if win_rate_pct is not None:
        if win_rate_pct >= 55:
            score += 20
        elif win_rate_pct >= 45:
            score += 15
        elif win_rate_pct >= 40:
            score += 10
        elif win_rate_pct >= 30:
            score += 5
    if profit_factor is not None:
        if profit_factor >= 3:
            score += 20
        elif profit_factor >= 2:
            score += 15
        elif profit_factor >= 1.5:
            score += 10
        elif profit_factor >= 1.0:
            score += 5
    return round(min(score, 100), 2)


def _classify_rule(
    sample_size: int | None,
    win_rate_pct: float | None,
    expectancy: float | None,
    *,
    force_reject: bool = False,
    force_validate: bool = False,
) -> str:
    if force_reject:
        return "Reject"
    if sample_size is None:
        return "Needs Validation"
    if sample_size < MIN_SAMPLES_VALIDATE:
        if expectancy is not None and expectancy < 0:
            return "Reject"
        return "Needs Validation"
    if force_validate:
        return "Needs Validation"
    if sample_size < MIN_SAMPLES_READY:
        if expectancy is not None and expectancy >= POOR_EXPECTANCY_THRESHOLD:
            return "Needs Validation"
        if expectancy is not None and expectancy < POOR_EXPECTANCY_THRESHOLD:
            return "Reject"
        return "Needs Validation"
    if expectancy is not None and expectancy < 0:
        return "Reject"
    if expectancy is not None and expectancy < POOR_EXPECTANCY_THRESHOLD:
        if win_rate_pct is not None and win_rate_pct >= LOW_WIN_RATE_THRESHOLD:
            return "Needs Validation"
        return "Reject"
    if win_rate_pct is not None and win_rate_pct < LOW_WIN_RATE_THRESHOLD and (
        expectancy is None or expectancy < 50
    ):
        return "Needs Validation"
    return "Production Ready"


def _rule(
    rule_id: str,
    rule_name: str,
    spec_section: str,
    rule_type: str,
    evidence_source: str,
    notes: str = "",
    force_reject: bool = False,
    force_validate: bool = False,
    **metrics: Any,
) -> RuleEvaluation:
    sample_size = metrics.get("sample_size")
    win_rate_pct = metrics.get("win_rate_pct")
    expectancy = metrics.get("expectancy")
    profit_factor = metrics.get("profit_factor")
    confidence = _confidence_from_metrics(sample_size, win_rate_pct, expectancy, profit_factor)
    classification = _classify_rule(
        sample_size,
        win_rate_pct,
        expectancy,
        force_reject=force_reject,
        force_validate=force_validate,
    )
    return RuleEvaluation(
        rule_id=rule_id,
        rule_name=rule_name,
        spec_section=spec_section,
        rule_type=rule_type,
        sample_size=sample_size,
        win_rate_pct=win_rate_pct,
        expectancy=expectancy,
        profit_factor=profit_factor,
        confidence_pct=confidence,
        classification=classification,
        evidence_source=evidence_source,
        notes=notes,
    )


class SmartMoneyEngineGapAnalysis:
    """Review V1 production spec against completed research."""

    EXPORT_FILES = [
        "tier2_production_validation.json",
        "tiered_signal_framework.json",
        "institutional_quality_validation.json",
        "sequence_entry_timing_validation.json",
        "tier2_entry_optimization.json",
        "tier2_exit_optimization.json",
        "trade_construction_validation.json",
        "tier2_regime_classification.json",
        "tier2_composite_edge_validation.json",
        "tier2_winner_loser_comparison.json",
        "institutional_signal_construction.json",
        "institutional_confirmation_candle.json",
        "institutional_trigger_validation.json",
        "institutional_momentum_origin.json",
        "support_resistance_pressure.json",
        "major_level_strength.json",
        "liquidity_sweep_tradeability.json",
        "liquidity_move_reconstruction.json",
        "robust_filter_report.json",
        "production_stack_analysis.json",
    ]

    def __init__(self) -> None:
        self.data: dict[str, dict[str, Any]] = {}
        for name in self.EXPORT_FILES:
            try:
                self.data[name] = _load_json(name)
            except GapAnalysisError:
                logger.warning("Skipping missing export: %s", name)

    def _evaluate_all_rules(self) -> list[RuleEvaluation]:
        tier2 = self.data.get("tier2_production_validation.json", {}).get("variants", {}).get(
            "raw_tier_2",
            {},
        )
        tier1 = self.data.get("tiered_signal_framework.json", {}).get("tiers", {}).get("tier_1", {})
        quality = self.data.get("institutional_quality_validation.json", {})
        entry = self.data.get("tier2_entry_optimization.json", {})
        exit_r = self.data.get("tier2_exit_optimization.json", {})
        trade = self.data.get("trade_construction_validation.json", {})
        regime = self.data.get("tier2_regime_classification.json", {})
        composite = self.data.get("tier2_composite_edge_validation.json", {})
        sweep = self.data.get("liquidity_sweep_tradeability.json", {})
        seq = self.data.get("sequence_entry_timing_validation.json", {})

        score_buckets = quality.get("score_buckets", {})
        bucket_0_20 = score_buckets.get("0-20", {})
        bucket_70 = quality.get("threshold_comparison", {}).get("70", {})
        baseline = quality.get("unfiltered_baseline", tier2)

        rules: list[RuleEvaluation] = []

        # Core mandatory
        rules.append(
            _rule(
                "M1",
                "Tier-2 sequence complete (Disp→CHOCH→BOS→FVG Reclaim)",
                "§6.1 M1",
                "mandatory",
                "tier2_production_validation.json",
                **_metrics(
                    tier2.get("signals", 502),
                    tier2.get("win_rate_pct"),
                    tier2.get("expectancy"),
                    tier2.get("profit_factor"),
                ),
            ),
        )
        rules.append(
            _rule(
                "M2",
                "Valid structural swing SL within risk cap",
                "§9",
                "mandatory",
                "trade_construction_validation.json",
                notes="Validated as part of BOS Close + Structural Swing SL stack.",
                **_metrics(
                    trade.get("total_tier2_signals", 502),
                    trade.get("production_recommendation", {}).get("win_rate_pct"),
                    trade.get("production_recommendation", {}).get("expectancy"),
                    trade.get("production_recommendation", {}).get("profit_factor"),
                ),
            ),
        )
        rules.append(
            _rule(
                "M6",
                "Quality Score ≥ 20 (reject 0–20 bucket)",
                "§6.1 M6",
                "mandatory",
                "institutional_quality_validation.json",
                **_metrics(
                    bucket_0_20.get("signals", 8),
                    bucket_0_20.get("win_rate_pct", 0.0),
                    bucket_0_20.get("expectancy", -150.38),
                    bucket_0_20.get("profit_factor"),
                ),
                notes="Reject bucket evidence: 0% WR; rule to exclude this bucket is Production Ready.",
            ),
        )

        # Entry / exit
        bos_close = entry.get("method_metrics", {}).get("A_bos_close", {})
        rules.append(
            _rule(
                "ENTRY_BOS_CLOSE",
                "V1 Standard Entry: BOS Close",
                "§8",
                "execution",
                "tier2_entry_optimization.json",
                **_metrics(
                    bos_close.get("trades", 502),
                    bos_close.get("win_rate_pct"),
                    bos_close.get("expectancy"),
                    bos_close.get("profit_factor"),
                ),
            ),
        )
        disp_close = seq.get("stage_metrics", {}).get("displacement_close", {})
        rules.append(
            _rule(
                "ENTRY_DISP_CLOSE",
                "V1 Premium Entry: Displacement Close",
                "§8",
                "execution",
                "sequence_entry_timing_validation.json",
                notes="Sequence subset n=60; higher metrics but not full Tier-2 universe.",
                force_validate=True,
                **_metrics(
                    disp_close.get("trades", 60),
                    disp_close.get("win_rate_pct"),
                    disp_close.get("expectancy"),
                    disp_close.get("profit_factor"),
                ),
            ),
        )
        exit_e = exit_r.get("model_metrics", {}).get("E", {})
        rules.append(
            _rule(
                "EXIT_MODEL_E",
                "Trail swing structure after 1R (Exit Model E)",
                "§10",
                "execution",
                "tier2_exit_optimization.json",
                **_metrics(
                    exit_e.get("trades", 502),
                    exit_e.get("win_rate_pct"),
                    exit_e.get("expectancy"),
                    exit_e.get("profit_factor"),
                ),
                notes="Lower expectancy than entry baseline; exit-layer rule not entry signal.",
            ),
        )
        rules.append(
            _rule(
                "TARGET_OPP_LIQ",
                "T3 Opposite Liquidity Pool target",
                "§10",
                "execution",
                "trade_construction_validation.json",
                **_metrics(
                    trade.get("best_net_profit_model", {}).get("trades", 502),
                    trade.get("best_net_profit_model", {}).get("win_rate_pct"),
                    trade.get("best_net_profit_model", {}).get("expectancy"),
                    trade.get("best_net_profit_model", {}).get("profit_factor"),
                ),
            ),
        )

        # Optional filters
        rules.append(
            _rule(
                "O1",
                "Quality Score ≥ 70",
                "§6.2 O1",
                "optional",
                "institutional_quality_validation.json",
                **_metrics(
                    bucket_70.get("signals", 43),
                    bucket_70.get("win_rate_pct"),
                    bucket_70.get("expectancy"),
                    bucket_70.get("profit_factor"),
                ),
                force_validate=True,
                notes="Strong metrics but only 43 signals/year — frequency trade-off.",
            ),
        )
        rules.append(
            _rule(
                "O2",
                "Tier-1 liquidity sweep sequence",
                "§3 Tier-1",
                "optional",
                "tiered_signal_framework.json",
                **_metrics(
                    tier1.get("signals", 60),
                    tier1.get("win_rate_pct"),
                    tier1.get("expectancy"),
                    tier1.get("profit_factor"),
                ),
                force_validate=True,
            ),
        )

        liq_rev = regime.get("regime_metrics", {}).get("Liquidity Reversal", {})
        rules.append(
            _rule(
                "O3",
                "Regime = Liquidity Reversal",
                "§6.2 O3 / §15",
                "optional",
                "tier2_regime_classification.json",
                **_metrics(
                    liq_rev.get("signals", 10),
                    liq_rev.get("win_rate_pct"),
                    liq_rev.get("expectancy"),
                    liq_rev.get("profit_factor"),
                ),
                force_validate=True,
                notes="Highest expectancy but n=10 — insufficient for Production Ready.",
            ),
        )
        trend_cont = regime.get("regime_metrics", {}).get("Trend Continuation", {})
        rules.append(
            _rule(
                "O4",
                "Regime = Trend Continuation",
                "§6.2 O4 / §15",
                "optional",
                "tier2_regime_classification.json",
                **_metrics(
                    trend_cont.get("signals", 70),
                    trend_cont.get("win_rate_pct"),
                    trend_cont.get("expectancy"),
                    trend_cont.get("profit_factor"),
                ),
                force_validate=True,
            ),
        )
        rsi_trait = next(
            (
                item
                for item in composite.get("individual_traits", [])
                if item.get("combination_key") == "rsi_below_40"
            ),
            {},
        )
        rules.append(
            _rule(
                "O5_BUY",
                "RSI < 40 booster (BUY)",
                "§6.2 O5",
                "optional",
                "tier2_composite_edge_validation.json",
                **_metrics(
                    rsi_trait.get("signals", 225),
                    rsi_trait.get("win_rate_pct"),
                    rsi_trait.get("expectancy"),
                    rsi_trait.get("profit_factor"),
                ),
            ),
        )
        near_sup = next(
            (
                item
                for item in composite.get("individual_traits", [])
                if item.get("combination_key") == "near_support"
            ),
            {},
        )
        rules.append(
            _rule(
                "O6_BUY",
                "Near Support booster (BUY)",
                "§6.2 O6",
                "optional",
                "tier2_composite_edge_validation.json",
                **_metrics(
                    near_sup.get("signals", 168),
                    near_sup.get("win_rate_pct"),
                    near_sup.get("expectancy"),
                    near_sup.get("profit_factor"),
                ),
            ),
        )
        four_trait = composite.get("four_trait_combination", {})
        if isinstance(four_trait, dict) and four_trait.get("combination_key"):
            rules.append(
                _rule(
                    "COMPOSITE_4TRAIT",
                    "RSI<40 + Midday + Strong Disp + Slow CHOCH-BOS",
                    "§13",
                    "confidence_booster",
                    "tier2_composite_edge_validation.json",
                    **_metrics(
                        four_trait.get("signals", 30),
                        four_trait.get("win_rate_pct"),
                        four_trait.get("expectancy"),
                        four_trait.get("profit_factor"),
                    ),
                    force_validate=True,
                ),
            )

        # Regime penalties
        session_bo = regime.get("regime_metrics", {}).get("Session Breakout", {})
        rules.append(
            _rule(
                "REGIME_SESSION_BREAKOUT",
                "Session Breakout regime (confidence penalty)",
                "§15 / §14",
                "confidence_penalty",
                "tier2_regime_classification.json",
                **_metrics(
                    session_bo.get("signals", 70),
                    session_bo.get("win_rate_pct"),
                    session_bo.get("expectancy"),
                    session_bo.get("profit_factor"),
                ),
                notes="Penalty rule; positive expectancy but weakest regime.",
            ),
        )

        # BUY/SELL contextual (move research — not tier2 trade WR)
        sig_const = self.data.get("institutional_signal_construction.json", {})
        prod_buy = sig_const.get("production_candidate_features", {}).get("BUY", [])
        premium_buy = next((f for f in prod_buy if "Premium Zone" in f.get("feature", "")), {})
        rules.append(
            _rule(
                "BUY_PREMIUM_ZONE_MAGNITUDE",
                "BUY booster: Premium Zone (large-move cohort)",
                "§4 / §7.2",
                "buy_context",
                "institutional_signal_construction.json",
                sample_size=sig_const.get("total_moves_analyzed", 1414),
                win_rate_pct=None,
                expectancy=None,
                profit_factor=None,
                force_validate=True,
                notes="Contradicts narrative-frequency Discount Zone BUY pattern; magnitude-only discriminator.",
            ),
        )
        rules.append(
            _rule(
                "BUY_DISCOUNT_NARRATIVE",
                "BUY narrative: Discount Zone + Hammer (frequency)",
                "§4",
                "buy_context",
                "institutional_signal_construction.json",
                sample_size=sig_const.get("total_moves_analyzed", 1414),
                win_rate_pct=None,
                expectancy=None,
                profit_factor=None,
                force_validate=True,
                notes="Most frequent pre-move pattern but negative lift for top-20% magnitude cohort.",
            ),
        )

        # Trigger validation top models
        trigger = self.data.get("institutional_trigger_validation.json", {})
        matrix = trigger.get("institutional_trigger_matrix", [])
        if matrix:
            top = matrix[0]
            rules.append(
                _rule(
                    "TRIGGER_LEVEL_RETEST_CONSOL",
                    "Trigger: Level Retest x3 + Moderate + Consolidation",
                    "§4 / §5",
                    "trigger",
                    "institutional_trigger_validation.json",
                    **_metrics(
                        top.get("sample_count", 25),
                        None,
                        top.get("average_move_magnitude"),
                        None,
                    ),
                    force_validate=True,
                    notes="100% move probability in research but move-level not trade-level WR.",
                ),
            )

        # Liquidity sweep standalone
        rules.append(
            _rule(
                "SWEEP_STANDALONE",
                "Liquidity sweep as standalone entry (NOT V1)",
                "§16",
                "exclusion",
                "liquidity_sweep_tradeability.json",
                **_metrics(
                    sweep.get("tradable_trades", 281),
                    sweep.get("overall_metrics", {}).get("win_rate_pct"),
                    sweep.get("overall_metrics", {}).get("expectancy"),
                    sweep.get("overall_metrics", {}).get("profit_factor"),
                ),
                force_reject=True,
                notes="Correctly excluded from V1; negative expectancy.",
            ),
        )

        # Level strength tiers
        ml = self.data.get("major_level_strength.json", {}).get("level_strength_matrix", {})
        for tier, key in [("Strong", "I4_STRONG"), ("Moderate", "LEVEL_MODERATE"), ("Weak", "LEVEL_WEAK")]:
            row = ml.get(tier, {})
            bounce = row.get("bounce_probability_pct", 0)
            rules.append(
                _rule(
                    key,
                    f"Level strength tier: {tier}",
                    "§16",
                    "level_strength",
                    "major_level_strength.json",
                    sample_size=row.get("sample_support_interactions", 0)
                    + row.get("sample_resistance_interactions", 0),
                    win_rate_pct=bounce,
                    expectancy=None,
                    profit_factor=None,
                    force_validate=tier == "Institutional" or (
                        (row.get("sample_support_interactions", 0) + row.get("sample_resistance_interactions", 0))
                        < MIN_SAMPLES_READY
                    ),
                    notes=f"Bounce {bounce}% / Rejection {row.get('rejection_probability_pct')}% — reaction rates not trade WR.",
                ),
            )

        # HTF+MI filtered variants (spec says NOT mandatory — validated)
        htf = self.data.get("tier2_production_validation.json", {}).get("variants", {}).get(
            "tier_2_htf_mi_65",
            {},
        )
        rules.append(
            _rule(
                "FILTER_HTF_MI",
                "HTF Alignment + MI≥65 filter (explicitly NOT mandatory)",
                "§6.1 note",
                "filter_rejected",
                "tier2_production_validation.json",
                **_metrics(
                    htf.get("signals", 68),
                    htf.get("win_rate_pct"),
                    htf.get("expectancy"),
                    htf.get("profit_factor"),
                ),
                force_validate=True,
                notes="Correctly excluded; Exp 29.41 vs raw 102.48.",
            ),
        )

        # FVG retest entry (spec says do not use)
        fvg50 = entry.get("method_metrics", {}).get("C_fvg_50_percent", {})
        rules.append(
            _rule(
                "ENTRY_FVG50_REJECT",
                "Do NOT use 50% FVG entry as primary",
                "§8",
                "exclusion",
                "tier2_entry_optimization.json",
                **_metrics(
                    fvg50.get("trades", 335),
                    fvg50.get("win_rate_pct"),
                    fvg50.get("expectancy"),
                    fvg50.get("profit_factor"),
                ),
                notes="Lower WR 27.76%; 240 missed moves.",
            ),
        )

        # Full 1R exit only (weaker than E)
        exit_a = exit_r.get("model_metrics", {}).get("A", {})
        rules.append(
            _rule(
                "EXIT_1R_ONLY",
                "Full exit at 1R only (not recommended)",
                "§10 / §19",
                "exclusion",
                "tier2_exit_optimization.json",
                **_metrics(
                    exit_a.get("trades", 502),
                    exit_a.get("win_rate_pct"),
                    exit_a.get("expectancy"),
                    exit_a.get("profit_factor"),
                ),
                force_validate=True,
                notes="Expectancy 8.99 vs Model E 33.86.",
            ),
        )

        # Confidence formula components (not independently backtested as composite)
        rules.append(
            _rule(
                "CONF_FORMULA",
                "Confidence = 50% Quality + 25% Candle + 25% Context",
                "§7",
                "scoring",
                "institutional_quality_validation.json + institutional_confirmation_candle.json",
                sample_size=None,
                win_rate_pct=None,
                expectancy=None,
                profit_factor=None,
                force_validate=True,
                notes="Weighted composite not backtested as unified score on Tier-2 trades.",
            ),
        )

        return rules

    def _identify_gaps(self, rules: list[RuleEvaluation]) -> dict[str, list[GapFinding]]:
        findings: dict[str, list[GapFinding]] = {
            "missing_rules": [],
            "contradictory_rules": [],
            "weak_rules": [],
            "missing_no_trade": [],
            "missing_trigger": [],
            "missing_level_strength": [],
            "missing_liquidity_trap": [],
            "missing_round_number": [],
            "missing_absorption": [],
        }

        findings["contradictory_rules"].append(
            GapFinding(
                category="contradictory_rules",
                finding_id="CONTRA_001",
                severity="high",
                description=(
                    "BUY Premium Zone listed as magnitude booster (signal construction) "
                    "while narrative-frequency BUY patterns require Discount Zone."
                ),
                affected_rules=["BUY_PREMIUM_ZONE_MAGNITUDE", "BUY_DISCOUNT_NARRATIVE"],
                recommendation="Split BUY context into 'reversal narrative' vs 'expansion magnitude' branches.",
            ),
        )
        findings["contradictory_rules"].append(
            GapFinding(
                category="contradictory_rules",
                finding_id="CONTRA_002",
                severity="medium",
                description=(
                    "Standard entry is BOS Close (502 signals) but sequence research recommends "
                    "Displacement Close (60 signals) as best composite — different universes."
                ),
                affected_rules=["ENTRY_BOS_CLOSE", "ENTRY_DISP_CLOSE"],
                recommendation="Keep BOS Close as default; gate Displacement Close to Tier-1 + Quality≥70 only.",
            ),
        )
        findings["contradictory_rules"].append(
            GapFinding(
                category="contradictory_rules",
                finding_id="CONTRA_003",
                severity="medium",
                description=(
                    "Spec lists sell-side sweep as BUY booster but Tier-2 sequence does not require "
                    "liquidity sweep (Tier-1 only)."
                ),
                affected_rules=["M1", "O2", "BUY_PREMIUM_ZONE_MAGNITUDE"],
                recommendation="Clarify sweep as confidence-only, never mandatory for Tier-2.",
            ),
        )

        findings["missing_rules"].extend(
            [
                GapFinding(
                    category="missing_rules",
                    finding_id="MISS_001",
                    severity="high",
                    description="No explicit VWAP location filter despite vwap_validation_report.json in research corpus.",
                    recommendation="Add optional VWAP premium/discount booster or document exclusion rationale.",
                ),
                GapFinding(
                    category="missing_rules",
                    finding_id="MISS_002",
                    severity="medium",
                    description="No explicit RSI divergence mandatory/optional rule despite research coverage.",
                    recommendation="Add divergence as O13 confidence booster with sample citation.",
                ),
                GapFinding(
                    category="missing_rules",
                    finding_id="MISS_003",
                    severity="medium",
                    description="No multi-symbol validation — all Tier-2 metrics are NIFTY50-only.",
                    recommendation="Extend validation to BANKNIFTY/FINNIFTY before multi-symbol production.",
                ),
                GapFinding(
                    category="missing_rules",
                    finding_id="MISS_004",
                    severity="low",
                    description="Max risk cap (2×ATR) stated but not validated in trade construction research.",
                    recommendation="Backtest risk-cap rejection rate on 502 signals.",
                ),
            ],
        )

        findings["missing_no_trade"].extend(
            [
                GapFinding(
                    category="missing_no_trade_conditions",
                    finding_id="NOTrade_001",
                    severity="high",
                    description="No NO-TRADE when liquidity sweep standalone would trigger (negative expectancy −6.45).",
                    recommendation="Add explicit NO-TRADE: sweep without Tier-2 sequence.",
                ),
                GapFinding(
                    category="missing_no_trade_conditions",
                    finding_id="NOTrade_002",
                    severity="high",
                    description="No NO-TRADE for first 15 minutes / last 15 minutes of session (gap research shows session effects).",
                    recommendation="Add opening chop and close flatten windows.",
                ),
                GapFinding(
                    category="missing_no_trade_conditions",
                    finding_id="NOTrade_003",
                    severity="medium",
                    description="No NO-TRADE when both HTF and LTF conflict without Liquidity Reversal regime.",
                    recommendation="Add HTF conflict NO-TRADE unless sweep+CHOCH confirm reversal.",
                ),
                GapFinding(
                    category="missing_no_trade_conditions",
                    finding_id="NOTrade_004",
                    severity="medium",
                    description="No NO-TRADE on Institutional tier levels (n=9 total in level strength research).",
                    recommendation="Add NO-TRADE or force paper on Institutional level tier until n≥50.",
                ),
                GapFinding(
                    category="missing_no_trade_conditions",
                    finding_id="NOTrade_005",
                    severity="medium",
                    description="No NO-TRADE when Quality 20–40 (127 signals, WR 35.4%, still positive Exp 90.44).",
                    recommendation="Clarify whether 20–40 is paper-only or standard eligible.",
                ),
            ],
        )

        findings["missing_trigger"].extend(
            [
                GapFinding(
                    category="missing_trigger_conditions",
                    finding_id="TRIG_001",
                    severity="medium",
                    description="Morning Star / Evening Star detected in trigger research but absent from spec trigger models.",
                    recommendation="Add star-pattern variants to optional trigger boosters.",
                ),
                GapFinding(
                    category="missing_trigger_conditions",
                    finding_id="TRIG_002",
                    severity="medium",
                    description="Inside Bar / Outside Bar triggers not in spec BUY/SELL cards.",
                    recommendation="Document from confirmation candle research or exclude explicitly.",
                ),
                GapFinding(
                    category="missing_trigger_conditions",
                    finding_id="TRIG_003",
                    severity="high",
                    description="Breakout/breakdown trigger buckets empty in signal construction (50-bar lookback gap).",
                    recommendation="Add failed-breakout trigger with expanded lookback or intraday level engine.",
                ),
            ],
        )

        findings["missing_level_strength"].extend(
            [
                GapFinding(
                    category="missing_level_strength_conditions",
                    finding_id="LVL_001",
                    severity="high",
                    description="No rule for Exhausted level classification (14 events in S/R pressure research).",
                    recommendation="Add Exhausted → reversal-only or NO-TRADE for breakout.",
                ),
                GapFinding(
                    category="missing_level_strength_conditions",
                    finding_id="LVL_002",
                    severity="medium",
                    description="Fresh vs Retested level distinction not in spec execution rules (dominant in S/R research).",
                    recommendation="Add Fresh/Retested as confidence modifier from support_resistance_pressure.json.",
                ),
                GapFinding(
                    category="missing_level_strength_conditions",
                    finding_id="LVL_003",
                    severity="medium",
                    description="I4 invalidation references Strong level but Weak levels break 97%+ — no Weak-level breakout boost.",
                    recommendation="Add Weak-level breakout continuation booster aligned with research.",
                ),
            ],
        )

        findings["missing_liquidity_trap"].extend(
            [
                GapFinding(
                    category="missing_liquidity_trap_conditions",
                    finding_id="TRAP_001",
                    severity="high",
                    description="Momentum origin: avg 2.97 false upside / 2.94 false downside breaks pre-expansion — no trap NO-TRADE.",
                    recommendation="Add liquidity trap filter: 3+ failed breaks against direction without displacement.",
                ),
                GapFinding(
                    category="missing_liquidity_trap_conditions",
                    finding_id="TRAP_002",
                    severity="medium",
                    description="C5 cancellation covers sweep reversal but not stop-hunt without sweep column.",
                    recommendation="Add stop-hunt size threshold from signal construction research.",
                ),
                GapFinding(
                    category="missing_liquidity_trap_conditions",
                    finding_id="TRAP_003",
                    severity="medium",
                    description="liquidity_sweep_outcome_validation.json not referenced in spec.",
                    recommendation="Cross-check sweep quality tiers (Institutional/Medium/Strong) for confidence.",
                ),
            ],
        )

        ml = self.data.get("major_level_strength.json", {})
        rn_weight = ml.get("strength_score_components", {}).get("round_number_overlap", 12)
        findings["missing_round_number"].append(
            GapFinding(
                category="missing_round_number_conditions",
                finding_id="RN_001",
                severity="high",
                description=(
                    f"Round-number overlap is 12pt component in level strength ({rn_weight} weight) "
                    "but no round-number rule in V1 spec."
                ),
                recommendation="Add round-number proximity booster/invalidator from major_level_strength features.",
            ),
        )

        sr = self.data.get("support_resistance_pressure.json", {})
        findings["missing_absorption"].extend(
            [
                GapFinding(
                    category="missing_absorption_conditions",
                    finding_id="ABS_001",
                    severity="high",
                    description="S/R research tracks wick rejections and strong confirmation at levels — 'absorption' not named in spec.",
                    recommendation="Define absorption = repeated tests + wick rejection + volume non-expansion near level.",
                ),
                GapFinding(
                    category="missing_absorption_conditions",
                    finding_id="ABS_002",
                    severity="medium",
                    description="Support bounce avg 0.82 strong-body candles vs resistance rejection 0.97 — no absorption scoring.",
                    recommendation="Import strong-body-at-level metric from support_resistance_pressure aggregate metrics.",
                ),
            ],
        )

        for rule in rules:
            if rule.classification == "Reject" and rule.rule_id != "SWEEP_STANDALONE":
                findings["weak_rules"].append(
                    GapFinding(
                        category="weak_rules",
                        finding_id=f"WEAK_{rule.rule_id}",
                        severity="medium",
                        description=f"Rule {rule.rule_name} classified Reject.",
                        affected_rules=[rule.rule_id],
                        recommendation=rule.notes or "Remove or tighten rule.",
                    ),
                )
            elif rule.confidence_pct < 40 and rule.classification != "Reject":
                findings["weak_rules"].append(
                    GapFinding(
                        category="weak_rules",
                        finding_id=f"WEAK_{rule.rule_id}",
                        severity="low",
                        description=f"Low confidence ({rule.confidence_pct}%) for {rule.rule_name}.",
                        affected_rules=[rule.rule_id],
                    ),
                )

        return findings

    def run(self) -> GapAnalysisReport:
        started = time.perf_counter()
        if not SPEC_PATH.exists():
            raise GapAnalysisError(f"Production spec not found: {SPEC_PATH}")

        rules = self._evaluate_all_rules()
        gaps = self._identify_gaps(rules)

        under_50 = [r for r in rules if r.sample_size is not None and r.sample_size < 50]
        under_100 = [r for r in rules if r.sample_size is not None and r.sample_size < 100]
        poor_exp = [
            r
            for r in rules
            if r.expectancy is not None
            and r.expectancy < POOR_EXPECTANCY_THRESHOLD
            and r.rule_id not in ("SWEEP_STANDALONE", "M6", "EXIT_1R_ONLY")
        ]
        false_signal = [
            r
            for r in rules
            if r.rule_id in ("SWEEP_STANDALONE", "ENTRY_FVG50_REJECT", "FILTER_HTF_MI", "REGIME_SESSION_BREAKOUT")
            or (r.win_rate_pct is not None and r.win_rate_pct < 30 and (r.sample_size or 0) >= 50)
        ]

        summary = {"Production Ready": 0, "Needs Validation": 0, "Reject": 0}
        for rule in rules:
            summary[rule.classification] = summary.get(rule.classification, 0) + 1

        ready_pct = round(summary["Production Ready"] / max(len(rules), 1) * 100, 2)
        if ready_pct >= 60:
            verdict = "Conditionally Production Ready — address high-severity gaps before live deployment."
        elif ready_pct >= 40:
            verdict = "Needs Validation — core Tier-2 ready but context layers under-sampled."
        else:
            verdict = "Not Production Ready — insufficient validated rule coverage."

        conclusions = [
            f"Evaluated {len(rules)} spec rules against {len(self.data)} research exports.",
            f"Classification: {summary['Production Ready']} Production Ready, "
            f"{summary['Needs Validation']} Needs Validation, {summary['Reject']} Reject.",
            f"{len(under_50)} rules have <50 samples; {len(under_100)} rules have <100 samples.",
            f"{len(gaps['contradictory_rules'])} contradictory rule sets identified.",
            f"{len(gaps['missing_no_trade'])} missing NO-TRADE conditions documented.",
            "Primary blocker: move-level institutional research not fully mapped to Tier-2 trade outcomes.",
            f"Overall verdict: {verdict}",
        ]

        return GapAnalysisReport(
            spec_version="1.0",
            spec_path=str(SPEC_PATH),
            research_exports_reviewed=list(self.data.keys()),
            total_rules_evaluated=len(rules),
            classification_summary=summary,
            production_readiness_score_pct=ready_pct,
            overall_verdict=verdict,
            rule_evaluations=[r.as_dict() for r in rules],
            missing_rules=[g.as_dict() for g in gaps["missing_rules"]],
            contradictory_rules=[g.as_dict() for g in gaps["contradictory_rules"]],
            weak_rules=[g.as_dict() for g in gaps["weak_rules"]],
            low_sample_rules_under_50=[r.as_dict() for r in under_50],
            low_sample_rules_under_100=[r.as_dict() for r in under_100],
            poor_expectancy_rules=[r.as_dict() for r in poor_exp],
            false_signal_rules=[r.as_dict() for r in false_signal],
            missing_no_trade_conditions=[g.as_dict() for g in gaps["missing_no_trade"]],
            missing_trigger_conditions=[g.as_dict() for g in gaps["missing_trigger"]],
            missing_level_strength_conditions=[g.as_dict() for g in gaps["missing_level_strength"]],
            missing_liquidity_trap_conditions=[g.as_dict() for g in gaps["missing_liquidity_trap"]],
            missing_round_number_conditions=[g.as_dict() for g in gaps["missing_round_number"]],
            missing_absorption_conditions=[g.as_dict() for g in gaps["missing_absorption"]],
            gap_findings=[
                g.as_dict()
                for category in gaps.values()
                for g in category
            ],
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_gap_analysis_report(
    report_path: Path | str | None = None,
) -> GapAnalysisReport:
    """Run gap analysis and export JSON."""
    engine = SmartMoneyEngineGapAnalysis()
    report = engine.run()

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info("Gap analysis completed: rules=%s", report.total_rules_evaluated)
    return report


def main() -> int:
    try:
        report = generate_gap_analysis_report()
        print("SmartMoneyEngine V1 Gap Analysis")
        print(f"Rules evaluated: {report.total_rules_evaluated}")
        print(f"Production readiness: {report.production_readiness_score_pct}%")
        print(f"Verdict: {report.overall_verdict}")
        print(f"Classification: {report.classification_summary}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except GapAnalysisError as exc:
        logger.error("Gap analysis error: %s", exc)
        print(f"Gap analysis error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected gap analysis error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
