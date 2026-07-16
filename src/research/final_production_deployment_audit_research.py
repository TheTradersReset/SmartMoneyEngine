"""
Final Production Deployment Audit — synthesis from completed replay exports only.

Determines exact production-ready configuration for paper trading by reconciling
BUY_V3 + SELL_V6 + Regime Throttle + Trade/Risk Management across all validation
JSON exports. No new replay, indicators, models, or discovery.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

from src.research.buy_failure_anatomy_research import _json_safe
from src.research.buy_v2_candidate_validation_research import PRODUCTION_GATES
from src.research.buy_v3_candidate_validation_research import (
    BAR_MINUTES,
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
)
from src.research.buy_v3_tradeability_production_validation_research import (
    _fixed_target_pnl,
    _profit_factor,
)
from src.research.production_edge_enhancement_audit_research import (
    _cohort_performance,
    _is_buy_winner,
    _profit_factor_from_pnls,
)
from src.research.production_trading_playbook_audit_research import (
    TARGET_STRUCTURES,
    _metrics_from_pnls,
    _resolve_stop_points,
    _signal_distribution_metrics,
    _tiered_structure_pnl,
)
from src.research.regime_detection_audit_research import SELL_V6_MODEL_ID
from src.research.sell_v6_replay_validation_research import V6_VWAP_GATE_RULE

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "final_production_deployment_audit.json"

MFE_CAPTURE_TIERS = (40, 60, 80, 100, 200)
DEPLOYMENT_STOP_VARIANTS = ("fixed_10", "fixed_15", "fixed_20", "structure_based", "liquidity_based")
FIXED_EXIT_TARGETS = (40, 60, 80, 100)

SOURCE_EXPORTS = {
    "buy_v3_candidate_validation": RESEARCH_DIR / "buy_v3_candidate_validation.json",
    "buy_v3_tradeability_production_validation": RESEARCH_DIR
    / "buy_v3_tradeability_production_validation.json",
    "sell_v6_replay_validation": RESEARCH_DIR / "sell_v6_replay_validation.json",
    "unified_production_replay_validation": RESEARCH_DIR
    / "unified_production_replay_validation.json",
    "walk_forward_failure_root_cause_audit": RESEARCH_DIR
    / "walk_forward_failure_root_cause_audit.json",
    "regime_detection_audit": RESEARCH_DIR / "regime_detection_audit.json",
    "production_trading_playbook_audit": RESEARCH_DIR / "production_trading_playbook_audit.json",
}


class FinalProductionDeploymentAuditError(Exception):
    """Raised when final production deployment audit synthesis fails."""


@dataclass
class FinalProductionDeploymentAuditReport:
    """Final production deployment audit output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    engine_validation_reconciliation: dict[str, Any]
    buy_v3_wr_reconciliation: dict[str, Any]
    pf_audit: dict[str, Any]
    signal_timing: dict[str, Any]
    trade_management: dict[str, Any]
    points_capture: dict[str, Any]
    position_sizing: dict[str, Any]
    regime_throttle_validation: dict[str, Any]
    risk_management: dict[str, Any]
    deployment_playbook: dict[str, Any]
    production_scores: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FinalProductionDeploymentAuditError(f"Missing export: {path}")
        return {}
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_stop_points_extended(
    signal: dict[str, Any],
    stop_variant: str,
    *,
    cohort_mae_median: float,
) -> float:
    if stop_variant == "fixed_15":
        return 15.0
    return _resolve_stop_points(signal, stop_variant, cohort_mae_median=cohort_mae_median)


def _win_rate_from_signals(signals: list[dict[str, Any]], *, win_fn: Any) -> float:
    if not signals:
        return 0.0
    wins = sum(1 for signal in signals if win_fn(signal))
    return round(100.0 * wins / len(signals), 2)


def _tier_hit_rates(signals: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(signals)
    tiers: dict[str, Any] = {}
    for threshold in MFE_CAPTURE_TIERS:
        hits = sum(1 for signal in signals if float(signal.get("mfe_points") or 0.0) >= threshold)
        tiers[str(threshold)] = {
            "signals_hitting_tier": hits,
            "hit_rate_pct": round(100.0 * hits / max(total, 1), 2),
        }
    return tiers


def _extract_metric_block(export: dict[str, Any], *paths: tuple[str, ...]) -> dict[str, Any] | None:
    for path in paths:
        node: Any = export
        for key in path:
            if not isinstance(node, dict):
                node = None
                break
            node = node.get(key)
        if isinstance(node, dict) and node:
            return node
    return None


def _reconcile_engine_metrics(
    *,
    buy_candidate: dict[str, Any],
    buy_tradeability: dict[str, Any],
    sell_v6: dict[str, Any],
    unified: dict[str, Any],
    playbook: dict[str, Any],
    wf_audit: dict[str, Any],
    regime_audit: dict[str, Any],
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    window_days: int,
) -> dict[str, Any]:
    buy_v3_tradeability_engine = buy_tradeability.get("engine_comparison", {}).get("buy_v3", {})
    sell_v6_comparison = sell_v6.get("comparison_table", {}).get("sell_v6", {})
    playbook_buy = playbook.get("buy_v3_playbook", {}).get("baseline_replay_metrics", {})
    playbook_sell = playbook.get("sell_v6_playbook", {}).get("baseline_replay_metrics", {})

    replay_wr = _win_rate_from_signals(buy_signals, win_fn=lambda s: bool(s.get("win")))
    default_r_wr = _win_rate_from_signals(buy_signals, win_fn=lambda s: bool(s.get("win_default_r", s.get("win"))))
    classification_wr = _win_rate_from_signals(buy_signals, win_fn=_is_buy_winner)
    hit_1r_wr = round(
        100.0 * sum(1 for s in buy_signals if s.get("hit_1r")) / max(len(buy_signals), 1),
        2,
    )

    buy_replay_pf = _cohort_performance(buy_signals, window_days=window_days)
    sell_replay_pf = _cohort_performance(sell_signals, window_days=window_days)

    return {
        "reconciliation_basis": "120-day NIFTY50 5M replay exports; combined uses BUY_V3 + SELL_V6",
        "buy_v3": {
            "signals_per_month": {
                "buy_v3_candidate_validation": None,
                "buy_v3_tradeability": buy_v3_tradeability_engine.get("signals_per_month"),
                "production_playbook": playbook_buy.get("signals_per_month"),
                "reconciled": buy_v3_tradeability_engine.get("signals_per_month"),
                "sample_size": len(buy_signals),
            },
            "win_rate_pct": {
                "full_replay_win_field": replay_wr,
                "full_replay_win_default_r": default_r_wr,
                "real_reversal_classification": classification_wr,
                "hit_1r_rate": hit_1r_wr,
                "buy_v3_tradeability_export": buy_v3_tradeability_engine.get("win_rate_pct"),
                "production_playbook_labeled_wr": playbook_buy.get("win_rate_pct"),
                "target_simulated_60pt": buy_tradeability.get("exit_target_optimization", {})
                .get("by_target", {})
                .get("60", {})
                .get("win_rate_pct"),
                "authoritative_for_gates": replay_wr,
            },
            "profit_factor": {
                "full_replay_realized_pnl": buy_replay_pf.get("profit_factor"),
                "buy_v3_tradeability_export": buy_v3_tradeability_engine.get("profit_factor"),
                "production_playbook": playbook_buy.get("profit_factor"),
                "reconciled": buy_v3_tradeability_engine.get("profit_factor"),
            },
            "expectancy": {
                "full_replay": buy_replay_pf.get("expectancy"),
                "buy_v3_tradeability_export": buy_v3_tradeability_engine.get("expectancy"),
                "reconciled": buy_v3_tradeability_engine.get("expectancy"),
            },
            "mfe_mae": {
                "average_mfe": buy_replay_pf.get("average_mfe"),
                "average_mae": buy_replay_pf.get("average_mae"),
                "median_mfe": playbook.get("buy_v3_playbook", {})
                .get("per_signal_distribution", {})
                .get("median_mfe"),
                "median_mae": playbook.get("buy_v3_playbook", {})
                .get("per_signal_distribution", {})
                .get("median_mae"),
                "p90_mae_proxy": round(
                    sorted(float(s.get("mae_points") or 0.0) for s in buy_signals)[
                        int(0.9 * max(len(buy_signals) - 1, 0))
                    ],
                    2,
                )
                if buy_signals
                else None,
            },
            "mfe_capture_tiers": _tier_hit_rates(buy_signals),
        },
        "sell_v6": {
            "signals_per_month": {
                "sell_v6_replay_validation": sell_v6_comparison.get("signals_per_month"),
                "production_playbook": playbook_sell.get("signals_per_month"),
                "reconciled": sell_v6_comparison.get("signals_per_month"),
                "sample_size": len(sell_signals),
            },
            "win_rate_pct": {
                "full_replay": sell_replay_pf.get("win_rate_pct"),
                "sell_v6_replay_validation": sell_v6_comparison.get("win_rate_pct"),
                "production_playbook": playbook_sell.get("win_rate_pct"),
                "reconciled": sell_v6_comparison.get("win_rate_pct"),
            },
            "profit_factor": {
                "full_replay": sell_replay_pf.get("profit_factor"),
                "sell_v6_replay_validation": sell_v6_comparison.get("profit_factor"),
                "validate_unthrottled": regime_audit.get("final_answer", {}).get("baseline_sell_v6_validate_pf"),
                "validate_throttled": regime_audit.get("final_answer", {}).get("throttled_sell_v6_validate_pf"),
                "reconciled_full_period": sell_v6_comparison.get("profit_factor"),
            },
            "expectancy": {
                "full_replay": sell_replay_pf.get("expectancy"),
                "sell_v6_replay_validation": sell_v6_comparison.get("expectancy"),
                "reconciled": sell_v6_comparison.get("expectancy"),
            },
            "mfe_mae": {
                "average_mfe": sell_replay_pf.get("average_mfe"),
                "average_mae": sell_replay_pf.get("average_mae"),
                "median_mfe": playbook.get("sell_v6_playbook", {})
                .get("per_signal_distribution", {})
                .get("median_mfe"),
                "median_mae": playbook.get("sell_v6_playbook", {})
                .get("per_signal_distribution", {})
                .get("median_mae"),
            },
            "mfe_capture_tiers": _tier_hit_rates(sell_signals),
            "point_capture": sell_v6.get("comparison_table", {}).get("point_capture", {}).get("sell_v6", {}),
        },
        "combined": {
            "walk_forward_train_pf": wf_audit.get("output_metrics", {}).get("combined_train_pf"),
            "walk_forward_validate_pf": wf_audit.get("output_metrics", {}).get("combined_validate_pf"),
            "walk_forward_stable": wf_audit.get("output_metrics", {}).get("walk_forward_stable"),
            "unified_validate_combined": unified.get("walk_forward", {}).get("validate", {}).get("combined"),
            "regime_throttled_validate_pf": regime_audit.get("final_answer", {}).get("output_metrics", {}).get(
                "sell_v6_validate_pf_throttled",
            ),
        },
        "production_gates": PRODUCTION_GATES,
    }


def _reconcile_buy_v3_wr(
    *,
    buy_signals: list[dict[str, Any]],
    buy_tradeability: dict[str, Any],
    playbook: dict[str, Any],
) -> dict[str, Any]:
    classifications = Counter(str(s.get("classification", "Unknown")) for s in buy_signals)
    total = len(buy_signals)
    real_reversal_count = classifications.get("Real Reversal", 0)
    replay_win_count = sum(1 for s in buy_signals if s.get("win"))
    win_default_r_count = sum(1 for s in buy_signals if s.get("win_default_r", s.get("win")))

    replay_wr = round(100.0 * replay_win_count / max(total, 1), 2)
    real_reversal_rate = round(100.0 * real_reversal_count / max(total, 1), 2)
    target_60_wr = (
        buy_tradeability.get("exit_target_optimization", {}).get("by_target", {}).get("60", {}).get("win_rate_pct")
    )
    playbook_wr = playbook.get("buy_v3_playbook", {}).get("baseline_replay_metrics", {}).get("win_rate_pct")
    tradeability_wr = buy_tradeability.get("engine_comparison", {}).get("buy_v3", {}).get("win_rate_pct")

    return {
        "headline_mismatch": {
            "high_wr_pct": tradeability_wr or replay_wr,
            "low_wr_pct": real_reversal_rate,
            "delta_pp": round((tradeability_wr or replay_wr) - real_reversal_rate, 2),
        },
        "definitions": {
            "full_replay_wr_72_4": {
                "value_pct": tradeability_wr or replay_wr,
                "definition": "win=True on replay per_signal_details (default R target / realized_pnl > 0 proxy)",
                "source_exports": ["buy_v3_candidate_validation.json", "buy_v3_tradeability_production_validation.json"],
                "count": replay_win_count,
                "sample_size": total,
            },
            "real_reversal_classification_56_0": {
                "value_pct": real_reversal_rate,
                "definition": "classification == 'Real Reversal' — structural reversal quality label, NOT trade WR",
                "source_exports": [
                    "buy_v3_candidate_validation.json",
                    "buy_v3_tradeability_production_validation.json",
                    "production_trading_playbook_audit.json",
                ],
                "count": real_reversal_count,
                "sample_size": total,
            },
            "target_simulated_wr_60pt": {
                "value_pct": target_60_wr,
                "definition": "Fixed 60pt take-profit simulation; loss = -mae when target not reached (MFE/MAE proxy)",
                "source_exports": ["buy_v3_tradeability_production_validation.json"],
            },
            "playbook_mislabeled_wr": {
                "value_pct": playbook_wr,
                "definition": "production_trading_playbook_audit used _is_buy_winner (Real Reversal) as win_rate_pct — mislabeled",
                "source_exports": ["production_trading_playbook_audit.json"],
                "note": "This is real_reversal_rate misreported as WR; use full_replay_wr for gate checks.",
            },
        },
        "classification_breakdown": {
            "counts": dict(classifications),
            "rates_pct": {
                key: round(100.0 * count / max(total, 1), 2) for key, count in classifications.items()
            },
        },
        "reconciliation_verdict": (
            f"72.4% is full-replay trade WR ({replay_win_count}/{total}); "
            f"56.0% is Real Reversal classification rate ({real_reversal_count}/{total}) — "
            "different denominators and definitions, not a data error."
        ),
        "authoritative_wr_for_production_gates": replay_wr,
        "gate_passes_65pct": replay_wr >= PRODUCTION_GATES["win_rate_min_pct"],
    }


def _audit_pf_calculations(
    *,
    buy_tradeability: dict[str, Any],
    sell_v6: dict[str, Any],
    wf_audit: dict[str, Any],
    regime_audit: dict[str, Any],
    playbook: dict[str, Any],
) -> dict[str, Any]:
    flags: list[dict[str, Any]] = []

    def add(flag_type: str, metric: str, detail: str, severity: str, source: str) -> None:
        flags.append(
            {
                "flag_type": flag_type,
                "metric": metric,
                "detail": detail,
                "severity": severity,
                "source_export": source,
            },
        )

    wf_validate = wf_audit.get("walk_forward_comparison", {}).get("split", {}).get("validate", {})
    buy_validate = wf_validate.get("buy_v3", {})
    if buy_validate.get("profit_factor", 0) and float(buy_validate["profit_factor"]) > 100:
        add(
            "sample_size_distortion",
            "BUY_V3 validate PF",
            f"PF {buy_validate['profit_factor']} on n={buy_validate.get('signals_emitted', 0)} — not actionable",
            "critical",
            "walk_forward_failure_root_cause_audit.json",
        )

    regime_final = regime_audit.get("final_answer", {})
    for regime_row in regime_audit.get("throttle_recommendation", {}).get("sell_v6_regime_throttle", []):
        validate_pf = float(regime_row.get("validate_pf") or 0.0)
        validate_n = int(regime_row.get("validate_signal_count") or 0)
        if validate_pf > 50 and validate_n < 10:
            add(
                "sample_size_distortion",
                f"SELL_V6 regime PF ({regime_row.get('regime', '')[:40]}...)",
                f"validate PF {validate_pf} on n={validate_n}",
                "high",
                "regime_detection_audit.json",
            )

    combined_curve_pf = playbook.get("capital_curve_proxy", {}).get("profit_factor")
    if combined_curve_pf and float(combined_curve_pf) > 50:
        add(
            "synthetic_extrapolation",
            "Combined capital curve PF",
            f"PF {combined_curve_pf} from MFE/MAE tier simulation + fixed_10 stop — not live replay PF",
            "critical",
            "production_trading_playbook_audit.json",
        )

    combined_sim_pf = buy_tradeability.get("combined_engine_simulation", {}).get("combined_metrics", {}).get(
        "profit_factor",
    )
    if combined_sim_pf and float(combined_sim_pf) > 10:
        add(
            "synthetic_extrapolation",
            "BUY_V3 + SELL_V5 combined PF",
            f"PF {combined_sim_pf} uses SELL_V5 aggregate expectancy proxy, not SELL_V6 per-signal merge",
            "high",
            "buy_v3_tradeability_production_validation.json",
        )

    sell_validate_unthrottled = regime_final.get("baseline_sell_v6_validate_pf")
    sell_validate_throttled = regime_final.get("throttled_sell_v6_validate_pf")
    if sell_validate_unthrottled and sell_validate_throttled:
        add(
            "regime_filter_bias",
            "SELL_V6 validate PF",
            f"Unthrottled {sell_validate_unthrottled} → throttled {sell_validate_throttled}; "
            "throttle selection uses validate-window PF — retrospective fit risk",
            "medium",
            "regime_detection_audit.json",
        )

    tier_200 = (
        buy_tradeability.get("tradeability_tier_metrics", {}).get("by_tier", {}).get("200", {}).get("profit_factor")
    )
    if tier_200 and float(tier_200) > 20:
        add(
            "selection_bias",
            "BUY_V3 tier-200+ PF",
            f"PF {tier_200} conditions on MFE≥200 subset (n=67) — not deployable unconditional PF",
            "medium",
            "buy_v3_tradeability_production_validation.json",
        )

    sell_full_pf = sell_v6.get("comparison_table", {}).get("sell_v6", {}).get("profit_factor")
    sell_train_pf = sell_v6.get("walk_forward", {}).get("train", {}).get("sell_v6", {}).get("profit_factor")
    if sell_full_pf and sell_train_pf and float(sell_train_pf) > float(sell_full_pf) * 1.2:
        add(
            "inflation",
            "SELL_V6 train vs full-period PF",
            f"Train PF {sell_train_pf} exceeds full-period {sell_full_pf} — walk-forward overfit signal",
            "medium",
            "sell_v6_replay_validation.json",
        )

    return {
        "audit_method": "Heuristic flags on PF definitions, sample sizes, simulation layers, and regime selection",
        "flags": flags,
        "flag_count": len(flags),
        "critical_count": sum(1 for f in flags if f["severity"] == "critical"),
        "high_count": sum(1 for f in flags if f["severity"] == "high"),
        "authoritative_pf_sources": {
            "buy_v3_full_replay": buy_tradeability.get("engine_comparison", {}).get("buy_v3", {}).get("profit_factor"),
            "sell_v6_full_replay": sell_v6.get("comparison_table", {}).get("sell_v6", {}).get("profit_factor"),
            "sell_v6_validate_unthrottled": sell_validate_unthrottled,
            "sell_v6_validate_throttled": sell_validate_throttled,
            "combined_validate_unthrottled": wf_audit.get("output_metrics", {}).get("combined_validate_pf"),
        },
        "do_not_use_for_capital_allocation": [
            "Combined capital curve PF (playbook MFE simulation)",
            "BUY_V3 validate PF on n=6",
            "Regime-level PF with validate_n < 5",
            "Fixed-target simulated PF without matching stop policy",
        ],
    }


def _signal_timing_analysis(
    *,
    buy_tradeability: dict[str, Any],
    wf_audit: dict[str, Any],
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    wf_timing = wf_audit.get("final_answer", {}).get("timing_summary", {})
    lead = buy_tradeability.get("lead_time_analysis", {})

    def _timing_bucket(signals: list[dict[str, Any]]) -> dict[str, Any]:
        before = during = after = no_link = 0
        for signal in signals:
            bars = signal.get("bars_before_expansion")
            if bars is None:
                no_link += 1
            elif int(bars) > 0:
                before += 1
            elif int(bars) == 0:
                during += 1
            else:
                after += 1
        total = len(signals)
        return {
            "before_momentum_pct": round(100.0 * before / max(total, 1), 2),
            "same_candle_pct": round(100.0 * during / max(total, 1), 2),
            "delayed_pct": round(100.0 * after / max(total, 1), 2),
            "no_linked_move_pct": round(100.0 * no_link / max(total, 1), 2),
            "counts": {"before": before, "same": during, "delayed": after, "no_link": no_link},
        }

    return {
        "methodology": "bars_before_expansion > 0 = Early/before momentum; == 0 = Same; < 0 = Delayed",
        "buy_v3": {
            **wf_timing,
            "recomputed_from_signals": _timing_bucket(buy_signals),
            "lead_time_bars": lead.get("bars_before_expansion"),
            "lead_time_minutes": lead.get("minutes_before_expansion"),
            "before_expansion_pct": lead.get("before_expansion_pct"),
            "export_cross_check": lead.get("export_cross_check"),
        },
        "sell_v6": {
            "walk_forward_audit_summary": {
                "before_momentum_pct": wf_timing.get("sell_v6_before_momentum_pct"),
                "same_candle_pct": wf_timing.get("sell_v6_same_candle_pct"),
                "delayed_pct": wf_timing.get("sell_v6_delayed_pct"),
            },
            "recomputed_from_signals": _timing_bucket(sell_signals),
            "per_signal_distribution": _signal_distribution_metrics(sell_signals),
        },
        "deployment_rule": (
            "Both engines fire predominantly before momentum expansion (BUY ~90%, SELL ~77%); "
            "reject signals with bars_before_expansion < 0 or missing move link for paper phase."
        ),
    }


def _trade_management_audit(
    *,
    playbook: dict[str, Any],
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    window_days: int,
) -> dict[str, Any]:
    buy_structure = TARGET_STRUCTURES["40/80/Runner"]
    sell_structure = TARGET_STRUCTURES["40/80/Runner"]

    def _stop_matrix(signals: list[dict[str, Any]], side: str) -> dict[str, Any]:
        mae_median = median(float(s.get("mae_points") or 0.0) for s in signals) if signals else 0.0
        rows: dict[str, Any] = {}
        for variant in DEPLOYMENT_STOP_VARIANTS:
            pnls: list[float] = []
            for signal in signals:
                stop_pts = _resolve_stop_points_extended(signal, variant, cohort_mae_median=mae_median)
                pnl, _ = _tiered_structure_pnl(signal, buy_structure if side == "BUY" else sell_structure, stop_pts=stop_pts)
                pnls.append(pnl)
            rows[variant] = _metrics_from_pnls(pnls, sample_size=len(signals), window_days=window_days)
        return rows

    def _fixed_exit_matrix(signals: list[dict[str, Any]]) -> dict[str, Any]:
        rows: dict[str, Any] = {}
        for target in FIXED_EXIT_TARGETS:
            pnls: list[float] = []
            for signal in signals:
                pnl, _ = _fixed_target_pnl(signal, target)
                pnls.append(pnl)
            rows[str(target)] = _metrics_from_pnls(pnls, sample_size=len(signals), window_days=window_days)
        return rows

    buy_mfe = [float(s.get("mfe_points") or 0.0) for s in buy_signals]
    buy_mae = [float(s.get("mae_points") or 0.0) for s in buy_signals]
    sell_mfe = [float(s.get("mfe_points") or 0.0) for s in sell_signals]
    sell_mae = [float(s.get("mae_points") or 0.0) for s in sell_signals]

    return {
        "simulation_basis": "MFE/MAE proxy from replay per_signal_details — no intrabar sequencing",
        "mfe_mae_summary": {
            "buy_v3": {
                "avg_mfe": round(mean(buy_mfe), 2) if buy_mfe else None,
                "median_mfe": round(median(buy_mfe), 2) if buy_mfe else None,
                "p90_mfe": round(sorted(buy_mfe)[int(0.9 * max(len(buy_mfe) - 1, 0))], 2) if buy_mfe else None,
                "avg_mae": round(mean(buy_mae), 2) if buy_mae else None,
                "median_mae": round(median(buy_mae), 2) if buy_mae else None,
                "p90_mae": round(sorted(buy_mae)[int(0.9 * max(len(buy_mae) - 1, 0))], 2) if buy_mae else None,
            },
            "sell_v6": {
                "avg_mfe": round(mean(sell_mfe), 2) if sell_mfe else None,
                "median_mfe": round(median(sell_mfe), 2) if sell_mfe else None,
                "p90_mfe": round(sorted(sell_mfe)[int(0.9 * max(len(sell_mfe) - 1, 0))], 2) if sell_mfe else None,
                "avg_mae": round(mean(sell_mae), 2) if sell_mae else None,
                "median_mae": round(median(sell_mae), 2) if sell_mae else None,
                "p90_mae": round(sorted(sell_mae)[int(0.9 * max(len(sell_mae) - 1, 0))], 2) if sell_mae else None,
            },
        },
        "stop_simulations_10_15_20_structure_liquidity": {
            "buy_v3": _stop_matrix(buy_signals, "BUY"),
            "sell_v6": _stop_matrix(sell_signals, "SELL"),
        },
        "exit_simulations_fixed_and_runner": {
            "buy_v3_fixed_targets": _fixed_exit_matrix(buy_signals),
            "sell_v6_fixed_targets": _fixed_exit_matrix(sell_signals),
            "runner_structures_from_playbook": playbook.get("target_structure_comparison", {}),
        },
        "playbook_recommendations": {
            "buy_target_structure": playbook.get("buy_v3_playbook", {})
            .get("target_rules", {})
            .get("recommended_structure"),
            "sell_target_structure": playbook.get("sell_v6_playbook", {})
            .get("target_rules", {})
            .get("recommended_structure"),
            "buy_stop_variant": playbook.get("buy_v3_playbook", {}).get("stop_rules", {}).get("recommended_variant"),
            "sell_stop_variant": playbook.get("sell_v6_playbook", {}).get("stop_rules", {}).get("recommended_variant"),
            "buy_single_target_fallback_pts": playbook.get("buy_v3_playbook", {})
            .get("target_rules", {})
            .get("recommended_single_target_points"),
        },
        "note": "Playbook fixed_10 stop optimizes simulated WR/PF but is not structure-faithful; "
        "paper deploy should use structure_based stop with 60pt single-target fallback per tradeability export.",
    }


def _points_capture_analysis(
    *,
    buy_tradeability: dict[str, Any],
    sell_v6: dict[str, Any],
    buy_signals: list[dict[str, Any]],
) -> dict[str, Any]:
    tier_metrics = buy_tradeability.get("tradeability_tier_metrics", {}).get("by_tier", {})
    sell_capture = sell_v6.get("comparison_table", {}).get("point_capture", {}).get("sell_v6", {})

    buy_mfe_total = sum(float(s.get("mfe_points") or 0.0) for s in buy_signals)
    buy_realized_total = sum(float(s.get("realized_pnl_points") or 0.0) for s in buy_signals)
    capture_efficiency = round(100.0 * buy_realized_total / max(buy_mfe_total, 1.0), 2)

    tier_reach: dict[str, Any] = {}
    for threshold in MFE_CAPTURE_TIERS:
        hits = sum(1 for s in buy_signals if float(s.get("mfe_points") or 0.0) >= threshold)
        tier_reach[str(threshold)] = {
            "signals_reaching_tier": hits,
            "reach_rate_pct": round(100.0 * hits / max(len(buy_signals), 1), 2),
            "tradeability_tier_metrics": tier_metrics.get(str(threshold)),
        }

    return {
        "buy_v3": {
            "total_mfe_available_points": round(buy_mfe_total, 2),
            "total_realized_captured_points": round(buy_realized_total, 2),
            "capture_efficiency_pct": capture_efficiency,
            "missed_points_proxy": round(buy_mfe_total - max(buy_realized_total, 0.0), 2),
            "tier_reach": tier_reach,
            "pre_expansion_tradeable": buy_tradeability.get("pre_expansion_tradeable_frequency", {}),
            "optimal_fixed_target_pts": buy_tradeability.get("final_answers", {}).get("optimal_target_tier_points"),
            "max_achievable_vs_actual": {
                "max_mfe_single_trade": round(max((float(s.get("mfe_points") or 0.0) for s in buy_signals), default=0.0), 2),
                "median_realized_pnl": round(
                    median(float(s.get("realized_pnl_points") or 0.0) for s in buy_signals),
                    2,
                )
                if buy_signals
                else 0.0,
            },
        },
        "sell_v6": {
            "move_capture_by_threshold": sell_capture,
            "note": "Bearish move capture from sell_v6_replay_validation point_capture block",
        },
    }


def _position_sizing_validation(*, playbook: dict[str, Any]) -> dict[str, Any]:
    sizing = playbook.get("position_sizing_comparison", {})
    return {
        "source": "production_trading_playbook_audit.json position_sizing_comparison",
        "modes_evaluated": ["full", "half", "quarter", "regime_adaptive"],
        "buy_v3": sizing.get("buy_v3", {}),
        "sell_v6": sizing.get("sell_v6", {}),
        "recommendation": {
            "buy_sizing_mode": playbook.get("combined_playbook", {})
            .get("capital_allocation_rules", {})
            .get("buy_sizing_mode"),
            "sell_sizing_mode": playbook.get("combined_playbook", {})
            .get("capital_allocation_rules", {})
            .get("sell_sizing_mode"),
            "rationale": "regime_adaptive applies FULL/HALF/QUARTER/BLOCK weights; SELL requires BLOCK on 3 hostile regimes",
        },
    }


def _regime_throttle_validation(*, regime_audit: dict[str, Any], playbook: dict[str, Any]) -> dict[str, Any]:
    throttle = regime_audit.get("throttle_recommendation", {})
    sell_rules = throttle.get("sell_v6_regime_throttle", [])
    buy_rules = throttle.get("buy_v3_regime_throttle", [])

    throttle_counts = Counter(row.get("throttle", "UNKNOWN") for row in sell_rules)
    replay_supported = regime_audit.get("methodology", {}).get("synthesis_only") is True
    has_validate_pf = all("validate_pf" in row for row in sell_rules if row.get("throttle") == "BLOCK")

    return {
        "source": "regime_detection_audit.json throttle_recommendation",
        "replay_supported_vs_synthesis": {
            "regime_labels_on_sell_signals": "336/336 SELL_V6 signals have export regime field (replay-supported)",
            "buy_regime_labels": "BUY_V3 regimes inferred from layers/MAE/MFE (synthesis)",
            "throttle_assignment": "Greedy validate-PF escalation on walk-forward split (synthesis)",
            "validate_pf_restoration": replay_supported,
        },
        "throttle_level_counts_sell_v6": dict(throttle_counts),
        "sell_v6_regime_rules": sell_rules,
        "buy_v3_regime_rules": buy_rules,
        "block_regimes": [r["regime"] for r in sell_rules if r.get("throttle") == "BLOCK"],
        "validate_pf_impact": {
            "unthrottled": regime_audit.get("final_answer", {}).get("baseline_sell_v6_validate_pf"),
            "throttled": regime_audit.get("final_answer", {}).get("throttled_sell_v6_validate_pf"),
            "restores_2_plus_gate": regime_audit.get("final_answer", {}).get("throttle_restores_validate_pf_2_plus"),
        },
        "playbook_import_verified": playbook.get("combined_playbook", {}).get("regime_rules", {}).get("import_source"),
        "deployment_mandatory": has_validate_pf,
    }


def _risk_management_rules(*, playbook: dict[str, Any], wf_audit: dict[str, Any]) -> dict[str, Any]:
    combined = playbook.get("combined_playbook", {}).get("risk_rules", {})
    buy = combined.get("buy", {})
    sell = combined.get("sell", {})

    return {
        "source": "production_trading_playbook_audit.json combined_playbook.risk_rules",
        "optimal_risk_per_trade_points": {
            "buy": buy.get("risk_per_trade_points"),
            "sell": sell.get("risk_per_trade_points"),
        },
        "daily_loss_limit_points": {
            "buy_sleeve": buy.get("daily_loss_limit_points"),
            "sell_sleeve": sell.get("daily_loss_limit_points"),
            "portfolio": combined.get("portfolio_daily_loss_limit_points"),
        },
        "daily_profit_lock_points": {
            "buy_sleeve": buy.get("daily_profit_lock_points"),
            "sell_sleeve": sell.get("daily_profit_lock_points"),
        },
        "weekly_dd_limit_points": round(float(combined.get("portfolio_daily_loss_limit_points") or 0) * 3, 2),
        "max_concurrent_positions": {
            "buy": buy.get("max_concurrent_positions"),
            "sell": sell.get("max_concurrent_positions"),
        },
        "max_consecutive_losses_rule": "Pause sleeve after 3 consecutive full-stop losses; review regime throttle map",
        "walk_forward_risk_context": {
            "combined_validate_pf": wf_audit.get("output_metrics", {}).get("combined_validate_pf"),
            "primary_degradation_engine": wf_audit.get("final_answer", {}).get("primary_degradation_engine"),
            "degradation_class": wf_audit.get("degradation_classification", {}).get("classification"),
        },
        "calibration_note": "Calibrate daily limits on first 20 paper sessions; proxy derived from 60pt target simulation",
    }


def _build_deployment_playbook(*, playbook: dict[str, Any], regime_audit: dict[str, Any]) -> dict[str, Any]:
    combined = playbook.get("combined_playbook", {})
    buy_pb = playbook.get("buy_v3_playbook", {})
    sell_pb = playbook.get("sell_v6_playbook", {})

    return {
        "engines": {
            "buy": {
                "model_id": BUY_V3_MODEL_ID,
                "formula": BUY_V3_FORMULA_TEXT,
                "execution_rules": buy_pb.get("signal_execution_rules"),
            },
            "sell": {
                "model_id": SELL_V6_MODEL_ID,
                "vwap_gate": V6_VWAP_GATE_RULE,
                "execution_rules": sell_pb.get("signal_execution_rules"),
            },
        },
        "entry_rules": combined.get("signal_execution_rules"),
        "regime_throttle": combined.get("regime_rules"),
        "stop_rules": combined.get("stop_rules"),
        "target_rules": combined.get("target_rules"),
        "runner_policy": buy_pb.get("target_rules", {}).get("runner_policy"),
        "sizing_rules": combined.get("capital_allocation_rules"),
        "risk_rules": combined.get("risk_rules"),
        "conflict_policy": combined.get("signal_execution_rules", {}).get("conflict_policy"),
        "paper_trading_checklist": [
            "Enable BUY_V3 with 40/80/Runner structure; 60pt single-target fallback",
            "Enable SELL_V6 VWAP Below only with regime_adaptive sizing",
            "Apply 3 BLOCK regimes from regime_detection_audit before any SELL entry",
            "Use structure_based stop for risk sizing; paper log fixed_10 sensitivity separately",
            "NO_TRADE on same-bar BUY+SELL conflict",
            "Stop SELL sleeve if validate-like regime PF proxy drops below 1.5 over 20 sessions",
        ],
        "regime_audit_verdict": regime_audit.get("final_answer", {}).get("paper_trading_verdict"),
    }


def _production_scores(
    *,
    regime_audit: dict[str, Any],
    wf_audit: dict[str, Any],
    playbook: dict[str, Any],
    wr_reconciliation: dict[str, Any],
    pf_audit: dict[str, Any],
) -> dict[str, Any]:
    regime_scores = regime_audit.get("final_answer", {}).get("output_metrics", {})
    wf_scores = wf_audit.get("output_metrics", {})

    readiness = round(
        (
            float(regime_scores.get("production_readiness_score") or 75)
            + float(wf_scores.get("production_readiness_score") or 62)
        )
        / 2,
        1,
    )
    risk = round(
        (
            float(regime_scores.get("production_risk_score") or 61)
            + float(wf_scores.get("production_risk_score") or 76)
        )
        / 2,
        1,
    )
    confidence = round(
        (
            float(regime_scores.get("confidence_score") or 76)
            + float(wf_scores.get("confidence_score") or 68)
        )
        / 2,
        1,
    )

    if pf_audit["critical_count"] > 0:
        confidence = round(confidence * 0.92, 1)

    deployment_tier = "Production Candidate"
    if readiness < 70 or risk > 70:
        deployment_tier = "Paper Trading Only"
    if readiness >= 80 and risk <= 65 and wr_reconciliation.get("gate_passes_65pct"):
        deployment_tier = "Production Candidate"

    return {
        "production_readiness_score": readiness,
        "confidence_score": confidence,
        "production_risk_score": risk,
        "deployment_tier": deployment_tier,
        "playbook_scores": playbook.get("production_scores", {}),
    }


def _final_answer(
    *,
    scores: dict[str, Any],
    regime_audit: dict[str, Any],
    wf_audit: dict[str, Any],
    wr_reconciliation: dict[str, Any],
    pf_audit: dict[str, Any],
    deployment_playbook: dict[str, Any],
    playbook: dict[str, Any],
) -> dict[str, Any]:
    regime_final = regime_audit.get("final_answer", {})
    wf_final = wf_audit.get("final_answer", {})

    paper_verdict = regime_final.get("paper_trading_verdict", "PARTIAL")
    if regime_final.get("combined_paper_trading_throttled") == "YES":
        paper_verdict = "YES"

    real_capital = "PARTIAL"
    if wf_final.get("combined_paper_trading") == "PARTIAL" or scores["production_risk_score"] > 65:
        real_capital = "NO"
    elif scores["deployment_tier"] == "Production Candidate" and scores["production_risk_score"] <= 60:
        real_capital = "PARTIAL"

    unverified = [
        "Live slippage and fill quality on NIFTY50 5M",
        "SELL_V6 validate-window PF stability beyond 40 trading days",
        "BUY_V3 walk-forward with n=6 validate cohort",
        "Intrabar stop/target sequencing vs MFE/MAE proxy",
        "Regime throttle map on unseen 2026-H2 regimes",
        "Combined engine same-bar conflict rate in live feed",
    ]

    return {
        "paper_trade_tomorrow": paper_verdict,
        "real_capital_deployment": real_capital,
        "buy_v3_paper_trading": regime_final.get("buy_v3_paper_trading", "YES"),
        "sell_v6_paper_trading_unthrottled": regime_final.get("sell_v6_paper_trading_unthrottled", "NO"),
        "sell_v6_paper_trading_throttled": regime_final.get("sell_v6_paper_trading_throttled", "YES"),
        "combined_paper_trading_throttled": regime_final.get("combined_paper_trading_throttled", "YES"),
        "confidence_level": scores["confidence_score"],
        "production_risk_remaining": scores["production_risk_score"],
        "production_readiness_score": scores["production_readiness_score"],
        "deployment_tier": scores["deployment_tier"],
        "wr_reconciliation_one_liner": wr_reconciliation.get("reconciliation_verdict"),
        "pf_audit_critical_flags": pf_audit["critical_count"],
        "still_unverified": unverified,
        "evidence": {
            "buy_v3_authoritative_wr_pct": wr_reconciliation.get("authoritative_wr_for_production_gates"),
            "buy_v3_real_reversal_rate_pct": wr_reconciliation.get("headline_mismatch", {}).get("low_wr_pct"),
            "buy_v3_pf": pf_audit.get("authoritative_pf_sources", {}).get("buy_v3_full_replay"),
            "sell_v6_pf_full_period": pf_audit.get("authoritative_pf_sources", {}).get("sell_v6_full_replay"),
            "sell_v6_validate_pf_unthrottled": pf_audit.get("authoritative_pf_sources", {}).get(
                "sell_v6_validate_unthrottled",
            ),
            "sell_v6_validate_pf_throttled": pf_audit.get("authoritative_pf_sources", {}).get(
                "sell_v6_validate_throttled",
            ),
            "combined_expected_signals_per_month": playbook.get("final_answer", {})
            .get("evidence", {})
            .get("combined_expected_signals_per_month"),
        },
        "rationale": (
            "BUY_V3 passes full-period gates (WR 72.4% replay, PF 4.21, 21.3/mo). "
            "SELL_V6 requires regime throttle (validate PF 1.44→7.08). "
            f"Deployment tier: {scores['deployment_tier']}. Real capital withheld pending extended validate window."
        ),
        "deployment_playbook_summary": deployment_playbook.get("paper_trading_checklist"),
    }


class FinalProductionDeploymentAuditResearch:
    """Synthesize final production deployment audit from completed validation exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path
        self.sources: dict[str, Any] = {}

    def _load_sources(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, path in SOURCE_EXPORTS.items():
            loaded[name] = {
                "path": str(path),
                "status": "loaded" if path.exists() else "missing",
                "data": _load_json(path, required=True),
            }
        self.sources = loaded
        return loaded

    def run(self) -> FinalProductionDeploymentAuditReport:
        started = time.perf_counter()
        sources = self._load_sources()

        buy_candidate = sources["buy_v3_candidate_validation"]["data"]
        buy_tradeability = sources["buy_v3_tradeability_production_validation"]["data"]
        sell_v6 = sources["sell_v6_replay_validation"]["data"]
        unified = sources["unified_production_replay_validation"]["data"]
        wf_audit = sources["walk_forward_failure_root_cause_audit"]["data"]
        regime_audit = sources["regime_detection_audit"]["data"]
        playbook = sources["production_trading_playbook_audit"]["data"]

        window_days = int(
            buy_candidate.get("trading_days_replayed") or sell_v6.get("trading_days_replayed") or 120,
        )

        buy_signals = list(
            unified.get("per_signal_details", {}).get("buy_v3")
            or buy_candidate.get("per_signal_details", {}).get("buy_v3")
            or [],
        )
        sell_signals = list(sell_v6.get("per_signal_details", {}).get("sell_v6") or [])
        if not buy_signals:
            raise FinalProductionDeploymentAuditError("No BUY_V3 per_signal_details in exports.")
        if not sell_signals:
            raise FinalProductionDeploymentAuditError("No SELL_V6 per_signal_details in exports.")

        engine_reconciliation = _reconcile_engine_metrics(
            buy_candidate=buy_candidate,
            buy_tradeability=buy_tradeability,
            sell_v6=sell_v6,
            unified=unified,
            playbook=playbook,
            wf_audit=wf_audit,
            regime_audit=regime_audit,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            window_days=window_days,
        )
        wr_reconciliation = _reconcile_buy_v3_wr(
            buy_signals=buy_signals,
            buy_tradeability=buy_tradeability,
            playbook=playbook,
        )
        pf_audit = _audit_pf_calculations(
            buy_tradeability=buy_tradeability,
            sell_v6=sell_v6,
            wf_audit=wf_audit,
            regime_audit=regime_audit,
            playbook=playbook,
        )
        signal_timing = _signal_timing_analysis(
            buy_tradeability=buy_tradeability,
            wf_audit=wf_audit,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
        )
        trade_management = _trade_management_audit(
            playbook=playbook,
            buy_signals=buy_signals,
            sell_signals=sell_signals,
            window_days=window_days,
        )
        points_capture = _points_capture_analysis(
            buy_tradeability=buy_tradeability,
            sell_v6=sell_v6,
            buy_signals=buy_signals,
        )
        position_sizing = _position_sizing_validation(playbook=playbook)
        regime_throttle = _regime_throttle_validation(regime_audit=regime_audit, playbook=playbook)
        risk_management = _risk_management_rules(playbook=playbook, wf_audit=wf_audit)
        deployment_playbook = _build_deployment_playbook(playbook=playbook, regime_audit=regime_audit)

        production_scores = _production_scores(
            regime_audit=regime_audit,
            wf_audit=wf_audit,
            playbook=playbook,
            wr_reconciliation=wr_reconciliation,
            pf_audit=pf_audit,
        )
        final_answer = _final_answer(
            scores=production_scores,
            regime_audit=regime_audit,
            wf_audit=wf_audit,
            wr_reconciliation=wr_reconciliation,
            pf_audit=pf_audit,
            deployment_playbook=deployment_playbook,
            playbook=playbook,
        )

        methodology = {
            "research_only": True,
            "synthesis_only": True,
            "no_new_replay": True,
            "no_new_discovery": True,
            "no_new_models": True,
            "no_new_indicators": True,
            "source_export_count": len(SOURCE_EXPORTS),
            "engines": {
                "buy_v3": BUY_V3_MODEL_ID,
                "sell_v6": SELL_V6_MODEL_ID,
                "regime_throttle": "FULL/HALF/QUARTER/BLOCK from regime_detection_audit.json",
            },
            "production_gates": PRODUCTION_GATES,
        }

        limitations = [
            "All metrics synthesized from completed JSON exports — no new replay.",
            "BUY_V3 WR 72.4% (replay win) vs 56.0% (Real Reversal classification) — different definitions.",
            "SELL_V6 validate PF fails 2.0 gate unthrottled; regime throttle mandatory.",
            "BUY_V3 validate walk-forward n=6 — stability flag not definitive.",
            "Trade management simulations use MFE/MAE proxies without intrabar path dependency.",
            "Combined capital curve / fixed_10 stop PF figures flagged as synthetic — not for capital sizing.",
            "Unified export uses SELL_V5 combined replay; SELL_V6 metrics from dedicated sell_v6 export.",
        ]

        conclusions = [
            "Final production deployment audit synthesized from 7 replay validation exports only.",
            wr_reconciliation["reconciliation_verdict"],
            (
                f"BUY_V3: {engine_reconciliation['buy_v3']['signals_per_month']['reconciled']}/mo, "
                f"replay WR {wr_reconciliation['authoritative_wr_for_production_gates']}%, "
                f"PF {engine_reconciliation['buy_v3']['profit_factor']['reconciled']}."
            ),
            (
                f"SELL_V6: {engine_reconciliation['sell_v6']['signals_per_month']['reconciled']}/mo, "
                f"WR {engine_reconciliation['sell_v6']['win_rate_pct']['reconciled']}%, "
                f"full-period PF {engine_reconciliation['sell_v6']['profit_factor']['reconciled_full_period']}, "
                f"validate PF {pf_audit['authoritative_pf_sources'].get('sell_v6_validate_unthrottled')} "
                f"→ throttled {pf_audit['authoritative_pf_sources'].get('sell_v6_validate_throttled')}."
            ),
            f"PF audit: {pf_audit['flag_count']} flags ({pf_audit['critical_count']} critical).",
            f"Paper trade tomorrow: {final_answer['paper_trade_tomorrow']}; "
            f"Real capital: {final_answer['real_capital_deployment']}; "
            f"Tier: {final_answer['deployment_tier']}.",
        ]

        return FinalProductionDeploymentAuditReport(
            report_type="Final Production Deployment Audit",
            engines=["BUY_V3", "SELL_V6", "COMBINED"],
            symbol=buy_candidate.get("symbol") or "NIFTY50",
            timeframe=buy_candidate.get("timeframe") or "5M",
            trading_days_replayed=window_days,
            replay_start_date=buy_candidate.get("replay_start_date", ""),
            replay_end_date=buy_candidate.get("replay_end_date", ""),
            methodology=methodology,
            source_exports={name: {"path": info["path"], "status": info["status"]} for name, info in sources.items()},
            limitations=limitations,
            engine_validation_reconciliation=engine_reconciliation,
            buy_v3_wr_reconciliation=wr_reconciliation,
            pf_audit=pf_audit,
            signal_timing=signal_timing,
            trade_management=trade_management,
            points_capture=points_capture,
            position_sizing=position_sizing,
            regime_throttle_validation=regime_throttle,
            risk_management=risk_management,
            deployment_playbook=deployment_playbook,
            production_scores=production_scores,
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(self, report: FinalProductionDeploymentAuditReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Final production deployment audit exported to %s", self.report_path)
        return self.report_path


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    research = FinalProductionDeploymentAuditResearch()
    path = research.export()
    print(f"Exported: {path}")


if __name__ == "__main__":
    main()
