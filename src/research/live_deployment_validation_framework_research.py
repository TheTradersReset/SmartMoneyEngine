"""
Live Deployment Validation Framework — paper-trading gates for production stack.

Synthesizes a live paper validation playbook for the locked stack only:
BUY_V3 + SELL_V6 + fixed_10 + 60/100/Runner + Regime Throttle.

No BUY_V4 / SELL_V7 promotion, no new indicators/models/discovery, no multi-hour
bar replay. Thresholds are grounded in prior research exports and labeled as
**targets for live validation**, not as already-proven live metrics.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.research.buy_failure_anatomy_research import _json_safe
from src.research.buy_v2_candidate_validation_research import PRODUCTION_GATES
from src.research.buy_v3_candidate_validation_research import BUY_V3_MODEL_ID
from src.research.regime_detection_audit_research import SELL_V6_MODEL_ID

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "live_deployment_validation_framework.json"

LOCKED_STACK = {
    "buy_engine": "BUY_V3",
    "sell_engine": "SELL_V6",
    "buy_model_id": BUY_V3_MODEL_ID,
    "sell_model_id": SELL_V6_MODEL_ID,
    "stop": "fixed_10",
    "targets": "60/100/Runner",
    "t1_points": 60,
    "t2_points": 100,
    "runner": True,
    "leg_weights": [1 / 3, 1 / 3, 1 / 3],
    "regime_throttle": True,
    "conflict_policy": "NO_TRADE on same-bar opposing BUY+SELL",
    "buy_v4_sell_v7_status": "DO_NOT_PROMOTE",
}

# Conservative live targets vs 240d fixed_10 + 60/100/Runner + throttle replay.
# Replay baselines (extended_trade_level_truth_audit 240d):
#   combined_regime_throttle: WR 69.01%, PF 5.58, expectancy 108.05, max_dd 2424.26
REPLAY_BASELINES_240D = {
    "source": "extended_trade_level_truth_audit.json",
    "window_days": 240,
    "stack": "BUY_V3 + SELL_V6 + fixed_10 + 60/100/Runner + Regime Throttle",
    "buy_v3": {"win_rate_pct": 48.29, "profit_factor": 1.51, "expectancy": 38.53, "max_drawdown_points": 10996.65},
    "sell_v6": {"win_rate_pct": 63.85, "profit_factor": 2.47, "expectancy": 73.10, "max_drawdown_points": 7208.70},
    "combined": {"win_rate_pct": 59.63, "profit_factor": 2.13, "expectancy": 63.72, "max_drawdown_points": 17151.40},
    "combined_regime_throttle": {
        "win_rate_pct": 69.01,
        "profit_factor": 5.58,
        "expectancy": 108.05,
        "max_drawdown_points": 2424.26,
        "signals_per_month": 63.89,
    },
    "label": "REPLAY_BASELINE — targets for live validation, NOT proven live",
}

SOURCE_EXPORTS = {
    "deployment_readiness_validation": RESEARCH_DIR / "deployment_readiness_validation.json",
    "production_trading_playbook_audit": RESEARCH_DIR / "production_trading_playbook_audit.json",
    "live_trade_management_execution_efficiency_audit": RESEARCH_DIR
    / "live_trade_management_execution_efficiency_audit.json",
    "production_gap_closure_audit": RESEARCH_DIR / "production_gap_closure_audit.json",
    "extended_trade_level_truth_audit": RESEARCH_DIR / "extended_trade_level_truth_audit.json",
    "extended_evidence_validation_real_deployment_audit": RESEARCH_DIR
    / "extended_evidence_validation_real_deployment_audit.json",
    "ground_truth_production_comparison_audit": RESEARCH_DIR
    / "ground_truth_production_comparison_audit.json",
    "research_integrity_ground_truth_validation_audit": RESEARCH_DIR
    / "research_integrity_ground_truth_validation_audit.json",
    "buy_v4_sell_v7_actual_replay_validation": RESEARCH_DIR
    / "buy_v4_sell_v7_actual_replay_validation.json",
    "production_readiness_closure_audit": RESEARCH_DIR / "production_readiness_closure_audit.json",
    "final_production_deployment_audit": RESEARCH_DIR / "final_production_deployment_audit.json",
}


class LiveDeploymentValidationFrameworkError(Exception):
    """Raised when live deployment validation framework synthesis fails."""


@dataclass
class LiveDeploymentValidationFrameworkReport:
    """Live deployment validation framework output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    methodology: dict[str, Any]
    source_exports: dict[str, Any]
    limitations: list[str]
    stack_locked: dict[str, Any]
    replay_baselines: dict[str, Any]
    current_evidence_status: dict[str, Any]
    measurement_definitions: dict[str, Any]
    session_logging_schema: dict[str, Any]
    checklist_20_session: dict[str, Any]
    checklist_40_session: dict[str, Any]
    promotion_criteria: dict[str, Any]
    risk_controls: dict[str, Any]
    paper_trading_verdict: dict[str, Any]
    capital_tier_readiness: dict[str, Any]
    evidence_required_before_real_capital: list[str]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise LiveDeploymentValidationFrameworkError(f"Missing export: {path}")
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


def _export_meta(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "status": "loaded" if data else ("missing" if not path.exists() else "empty"),
        "loaded": bool(data),
    }


def _extract_replay_baselines(exports: dict[str, Any]) -> dict[str, Any]:
    """Prefer live export numbers; fall back to hardcoded 240d stack baselines."""
    baseline = dict(REPLAY_BASELINES_240D)
    ext = exports.get("extended_trade_level_truth_audit", {}).get("data", {})
    window = _nested(ext, "core_metrics_by_window", "240", default={})
    if window:
        for side in ("buy_v3", "sell_v6", "combined", "combined_regime_throttle"):
            block = window.get(side) or {}
            if block:
                baseline[side] = {
                    "win_rate_pct": block.get("win_rate_pct"),
                    "profit_factor": block.get("profit_factor"),
                    "expectancy": block.get("expectancy"),
                    "max_drawdown_points": block.get("max_drawdown_points"),
                    "signals_per_month": block.get("signals_per_month"),
                }
        baseline["source"] = "extended_trade_level_truth_audit.json (loaded)"
        baseline["replay_start_date"] = ext.get("replay_start_date")
        baseline["replay_end_date"] = ext.get("replay_end_date")

    v4 = exports.get("buy_v4_sell_v7_actual_replay_validation", {}).get("data", {})
    v4_final = v4.get("final_answer", {})
    baseline["v4_v7_promotion"] = {
        "should_buy_v4_replace_buy_v3": v4_final.get("should_buy_v4_replace_buy_v3", "NO"),
        "should_sell_v7_replace_sell_v6": v4_final.get("should_sell_v7_replace_sell_v6", "NO"),
        "best_buy_engine": v4_final.get("best_buy_engine", "BUY_V3"),
        "best_sell_engine": v4_final.get("best_sell_engine", "SELL_V6"),
        "best_stop": v4_final.get("best_stop", "fixed_10"),
        "best_exit_structure": v4_final.get("best_exit_structure", "60/100/Runner"),
        "note": "V4/V7 not production-deployable (lookahead / forward MFE-MAE dependency).",
    }
    return baseline


def _measurement_definitions(*, baselines: dict[str, Any], closure: dict[str, Any]) -> dict[str, Any]:
    slip_max = _nested(
        closure, "part5_live_execution_risk", "slippage_viability_threshold_points", default=10,
    )
    capture_proxy = baselines.get("combined_regime_throttle", {}).get("expectancy")
    return {
        "note": (
            "All formulas are for paper-trading telemetry. Pass thresholds are live-validation "
            "TARGETS grounded in prior replay/stress — not already proven on live fills."
        ),
        "slippage": {
            "definition": (
                "Points difference between intended signal-bar close (engine entry/exit price) "
                "and actual broker fill price, summed across entry + all exit legs."
            ),
            "formula": (
                "slippage_pts = sum(|fill_price - intended_price|) for entry and each exit leg; "
                "session_median = median(slippage_pts across fills); "
                "trade_slippage = entry_slip + exit_leg_slips"
            ),
            "data_source": {
                "engine": "intended_price from signal bar close / target / stop levels",
                "paper_broker": "fill_price, fill_ts, order_id from paper broker logs",
            },
            "pass_threshold": {
                "median_entry_plus_exit_pts_max": 5.0,
                "p90_trade_slippage_pts_max": 8.0,
                "kill_above_pts": float(slip_max),
                "rationale": (
                    f"Prior stress: edge viability threshold {slip_max}pt; "
                    "paper gate uses median ≤5pt (deployment_readiness / gap_closure)."
                ),
            },
            "status": "MISSING_LIVE — simulate/stress only to date",
        },
        "execution_delay": {
            "definition": (
                "Wall-clock lag from signal decision time (bar close) to order submit, "
                "and from submit to fill acknowledgment."
            ),
            "formula": (
                "decision_to_submit_ms = submit_ts - signal_bar_close_ts; "
                "submit_to_fill_ms = fill_ts - submit_ts; "
                "total_delay_ms = fill_ts - signal_bar_close_ts"
            ),
            "data_source": {
                "engine": "signal_bar_close_ts, decision_ts",
                "paper_broker": "submit_ts, ack_ts, fill_ts",
            },
            "pass_threshold": {
                "median_decision_to_submit_ms_max": 2000,
                "median_submit_to_fill_ms_max": 5000,
                "p95_total_delay_ms_max": 15000,
                "rationale": "5M bar stack; delay must stay well inside next-bar open.",
            },
            "status": "MISSING_LIVE",
        },
        "missed_entries": {
            "definition": (
                "Engine emitted BUY_V3 or SELL_V6 (throttle ALLOW/HALF) but no paper position "
                "was opened due to reject, timeout, risk lock, or operator skip."
            ),
            "formula": (
                "missed_entry_rate_pct = 100 * missed_entries / max(eligible_signals, 1); "
                "eligible = non-BLOCK, non-conflict signals"
            ),
            "data_source": {
                "engine": "signal journal (eligible vs blocked)",
                "paper_broker": "reject codes / no-order rows",
            },
            "pass_threshold": {
                "missed_entry_rate_pct_max": 8.0,
                "rationale": "Prior live-risk estimate 3–8% from BUY timing leakage.",
            },
            "status": "PARTIALLY PROVEN (replay timing) / MISSING_LIVE",
        },
        "same_bar_conflicts": {
            "definition": (
                "BUY_V3 and SELL_V6 both fire on the same 5M bar; playbook requires NO_TRADE."
            ),
            "formula": (
                "conflict_rate_pct = 100 * same_bar_conflicts / max(union_signal_bars, 1); "
                "policy_compliance_pct = 100 * no_trade_logged / max(conflicts, 1)"
            ),
            "data_source": {
                "engine": "combined signal clock / conflict detector",
                "paper_broker": "must show zero fills on conflict bars",
            },
            "pass_threshold": {
                "policy_compliance_pct_min": 100.0,
                "conflict_rate_pct_max_alert": 12.0,
                "rationale": "100% NO_TRADE compliance is mandatory; rate is monitored not optimized.",
            },
            "status": "PARTIALLY PROVEN (replay) / MISSING_LIVE",
        },
        "partial_fills": {
            "definition": (
                "Any of the three 60/100/Runner legs fills less than intended size, "
                "or runner leg is truncated by broker/session."
            ),
            "formula": (
                "partial_fill_rate_pct = 100 * partial_leg_events / max(exit_leg_orders, 1); "
                "size_shortfall_pct = 100 * (1 - filled_qty / intended_qty)"
            ),
            "data_source": {
                "engine": "intended leg weights 1/3 each",
                "paper_broker": "filled_qty per leg",
            },
            "pass_threshold": {
                "partial_fill_rate_pct_max": 10.0,
                "mean_size_shortfall_pct_max": 5.0,
                "rationale": "Prior estimate 2–5% expectancy impact; paper must quantify.",
            },
            "status": "MISSING",
        },
        "target_execution_accuracy": {
            "definition": (
                "When MFE reaches T1=60 or T2=100, corresponding exit leg is filled "
                "within slippage tolerance of the target price."
            ),
            "formula": (
                "t1_hit_and_filled_rate = filled_t1 / mfe_ge_60; "
                "t2_hit_and_filled_rate = filled_t2 / mfe_ge_100; "
                "target_price_error_pts = |fill - target|"
            ),
            "data_source": {
                "engine": "MFE path / target levels",
                "paper_broker": "leg fill prices and times",
            },
            "pass_threshold": {
                "t1_fill_when_mfe_ge_60_pct_min": 90.0,
                "t2_fill_when_mfe_ge_100_pct_min": 85.0,
                "median_target_price_error_pts_max": 3.0,
            },
            "status": "MISSING_LIVE (replay uses MFE proxy only)",
        },
        "stop_execution_accuracy": {
            "definition": (
                "When price violates fixed_10 stop, stop order fills; false stops and "
                "missed stops are both counted."
            ),
            "formula": (
                "stop_hit_fill_rate = stop_fills / mae_ge_10_stop_events; "
                "stop_slippage_pts = |fill - stop_price|; "
                "false_stop_rate = stops_without_mae_breach / stop_fills"
            ),
            "data_source": {
                "engine": "stop_price = entry ± 10pts",
                "paper_broker": "stop fill logs",
            },
            "pass_threshold": {
                "stop_hit_fill_rate_pct_min": 95.0,
                "median_stop_slippage_pts_max": 3.0,
                "false_stop_rate_pct_max": 5.0,
            },
            "status": "MISSING_LIVE",
        },
        "capture_efficiency": {
            "definition": (
                "Realized points vs maximum favorable excursion (MFE) available on the trade."
            ),
            "formula": (
                "capture_pct = 100 * realized_pnl_points / max(mfe_points, 1); "
                "session_capture = sum(realized) / sum(mfe)"
            ),
            "data_source": {
                "engine": "MFE from bar path / research proxy",
                "paper_broker": "realized PnL from fills",
            },
            "pass_threshold": {
                "combined_capture_pct_min": 34.0,
                "combined_capture_pct_max": 44.0,
                "replay_proxy_pct": 37.66,
                "tolerance_pp": 3.0,
                "rationale": (
                    f"Replay paper capture ~37.66%; gate ±3pp (34–41% band from gap_closure; "
                    f"extended to 44% upper for runner variance). expectancy proxy={capture_proxy}."
                ),
            },
            "status": "PARTIALLY PROVEN (replay) / MISSING_LIVE",
        },
    }


def _session_logging_schema() -> dict[str, Any]:
    return {
        "session_record": {
            "session_id": "YYYYMMDD-NIFTY50-5M",
            "session_date": "ISO date",
            "stack_fingerprint": "BUY_V3|SELL_V6|fixed_10|60/100/Runner|RegimeThrottle",
            "capital_mode": "paper | inr_50k | inr_1l | inr_2l",
            "lots": "int",
            "signals_emitted": {"buy_v3": "int", "sell_v6": "int", "blocked_sell": "int"},
            "trades_opened": "int",
            "trades_closed": "int",
            "same_bar_conflicts": "int",
            "missed_entries": "int",
            "partial_fill_events": "int",
            "pnl_points": "float",
            "pnl_inr": "float | null",
            "win_rate_pct": "float",
            "profit_factor_session": "float | null",
            "max_adverse_excursion_session_pts": "float",
            "drawdown_pts": "float",
            "median_slippage_pts": "float",
            "median_execution_delay_ms": "float",
            "capture_efficiency_pct": "float",
            "throttle_violations": "int",
            "daily_loss_limit_breached": "bool",
            "kill_switch_fired": "bool | string reason",
            "notes": "string",
        },
        "trade_record": {
            "trade_id": "string",
            "session_id": "string",
            "side": "BUY | SELL",
            "engine": "BUY_V3 | SELL_V6",
            "signal_ts": "ISO datetime",
            "entry_intended": "float",
            "entry_fill": "float",
            "entry_slippage_pts": "float",
            "stop_price": "float (fixed_10)",
            "targets": {"t1": 60, "t2": 100, "runner": True},
            "legs": [
                {
                    "leg": "t1|t2|runner|stop",
                    "intended_qty_pct": 33.33,
                    "fill_price": "float",
                    "fill_qty_pct": "float",
                    "slippage_pts": "float",
                    "fill_ts": "ISO datetime",
                },
            ],
            "mfe_points": "float",
            "mae_points": "float",
            "realized_pnl_points": "float",
            "capture_efficiency_pct": "float",
            "regime_label": "string",
            "throttle_action": "FULL | HALF | BLOCK | N/A",
            "conflict_flag": "bool",
        },
        "daily_rollup_required_fields": [
            "session_date",
            "pnl_points",
            "median_slippage_pts",
            "missed_entry_rate_pct",
            "same_bar_conflict_count",
            "throttle_violations",
            "capture_efficiency_pct",
            "kill_switch_fired",
        ],
    }


def _checklist_20_session() -> dict[str, Any]:
    """Session-by-session and gate checklist for first 20 paper sessions."""
    session_gates = []
    for n in range(1, 21):
        session_gates.append(
            {
                "session": n,
                "gates": [
                    "Stack fingerprint matches LOCKED_STACK (no V4/V7, fixed_10, 60/100/Runner, throttle ON)",
                    "Session log written with required schema fields",
                    "Every SELL in BLOCK regime has throttle_action=BLOCK and zero fills",
                    "Every same-bar conflict logged as NO_TRADE with zero fills",
                    "Median slippage_pts recorded (entry+exit)",
                    "Daily loss limit not breached (portfolio ≤593.79 pts)",
                    "Kill-switch status reviewed (none silent)",
                ],
                "milestone": (
                    "Baseline telemetry week"
                    if n <= 5
                    else "Execution quality week"
                    if n <= 10
                    else "Stability week"
                    if n <= 15
                    else "Promotion readiness week"
                ),
            },
        )

    end_gates = [
        {
            "id": "P20-1",
            "gate": "≥18/20 sessions with non-negative combined PnL proxy (or documented single-day anomaly ≤2)",
            "metric": "sessions_positive_pnl / 20 ≥ 0.90",
        },
        {
            "id": "P20-2",
            "gate": "Median slippage ≤5pt per entry+exit across all fills",
            "metric": "median_slippage_pts ≤ 5.0",
        },
        {
            "id": "P20-3",
            "gate": "SELL throttle BLOCK 100% match vs labeled regimes (shadow)",
            "metric": "throttle_violations == 0",
        },
        {
            "id": "P20-4",
            "gate": "Same-bar conflict NO_TRADE 100% compliance",
            "metric": "conflict_policy_compliance_pct == 100",
        },
        {
            "id": "P20-5",
            "gate": "Daily loss limit never breached",
            "metric": "daily_loss_limit_breaches == 0",
        },
        {
            "id": "P20-6",
            "gate": "Capture efficiency within ±3pp of replay proxy (~37.66%)",
            "metric": "34.0 ≤ capture_efficiency_pct ≤ 41.0",
        },
        {
            "id": "P20-7",
            "gate": "Minimum signal sample: BUY≥15, SELL≥25 (stretch to 30/50 if frequency allows)",
            "metric": "buy_trades≥15 AND sell_trades≥25",
        },
        {
            "id": "P20-8",
            "gate": "Combined rolling PF ≥1.5 AND combined WR ≥55% (live-validation targets)",
            "metric": "pf≥1.5 AND wr≥55 (conservative vs 240d throttled PF 5.58 / WR 69.01)",
        },
        {
            "id": "P20-9",
            "gate": "No kill-switch hard stop fired for stack integrity failure",
            "metric": "integrity_kill_switches == 0",
        },
        {
            "id": "P20-10",
            "gate": "BUY_V4 / SELL_V7 not enabled at any point",
            "metric": "stack_drift_events == 0",
        },
    ]

    return {
        "duration_sessions": 20,
        "capital_mode": "paper",
        "purpose": "Prove execution telemetry and playbook compliance before any real capital",
        "session_by_session": session_gates,
        "end_of_phase_gates": end_gates,
        "gate_to_50k": "All P20-* gates pass + written risk sign-off",
        "source_alignment": "production_gap_closure_audit phase_1_paper + deployment_readiness",
    }


def _checklist_40_session() -> dict[str, Any]:
    """Deeper 40-session validation checklist (promotion toward small capital)."""
    return {
        "duration_sessions": 40,
        "capital_mode": "paper continuing OR INR 50K after P20 pass (not both escalations at once)",
        "purpose": "Prove stability, regime throttle live, and capital-tier promotion readiness",
        "itemized_gates": [
            {
                "id": "P40-1",
                "gate": "40 consecutive logged sessions with complete schema (no gaps)",
                "pass": "session_count_complete == 40",
            },
            {
                "id": "P40-2",
                "gate": "Sample floors: BUY≥30, SELL≥50, combined≥80 trades",
                "pass": "buy≥30 AND sell≥50 AND combined≥80",
            },
            {
                "id": "P40-3",
                "gate": "Combined PF ≥1.8 and WR ≥58% over full 40 sessions (live-validation targets)",
                "pass": "pf≥1.8 AND wr≥58 (still well below 240d throttled PF 5.58 / WR 69.01)",
            },
            {
                "id": "P40-4",
                "gate": "Expectancy ≥40 pts/trade combined (vs replay throttled 108.05 — conservative)",
                "pass": "expectancy_pts ≥ 40",
            },
            {
                "id": "P40-5",
                "gate": "Rolling 20-session SELL PF proxy ≥1.5 at every checkpoint (sessions 20,25,30,35,40)",
                "pass": "min(rolling_sell_pf_20) ≥ 1.5",
            },
            {
                "id": "P40-6",
                "gate": "Zero unthrottled SELL entries in BLOCK regimes across all 40 sessions",
                "pass": "block_regime_fills == 0",
            },
            {
                "id": "P40-7",
                "gate": "Max paper drawdown ≤150% of session-scaled risk budget (see risk_controls)",
                "pass": "max_dd within budget",
            },
            {
                "id": "P40-8",
                "gate": "Median slippage remains ≤5pt; p90 ≤8pt; no session median >10pt",
                "pass": "slippage gates hold",
            },
            {
                "id": "P40-9",
                "gate": "Missed entry rate ≤8%; same-bar conflict compliance 100%",
                "pass": "missed≤8% AND conflict_compliance==100%",
            },
            {
                "id": "P40-10",
                "gate": "Target/stop execution accuracy gates met (T1≥90%, stop fill≥95%)",
                "pass": "execution accuracy gates",
            },
            {
                "id": "P40-11",
                "gate": "Capture efficiency stays in 34–44% band",
                "pass": "capture band",
            },
            {
                "id": "P40-12",
                "gate": "Recovery from worst week within 10 trading days",
                "pass": "recovery_days ≤ 10",
            },
            {
                "id": "P40-13",
                "gate": "Independent review of 40-session ledger (second person or checklist audit)",
                "pass": "audit_signoff == YES",
            },
            {
                "id": "P40-14",
                "gate": "Confidence score proxy: no thesis-invalidating kill-switch in last 20 sessions",
                "pass": "recent_integrity_kills == 0",
            },
        ],
        "gate_to_1l": "P40-* pass while on INR 50K for ≥20 of the 40 sessions OR paper+50K sequential path complete",
        "source_alignment": "production_gap_closure_audit phase_2_small_capital",
    }


def _promotion_criteria(*, baselines: dict[str, Any]) -> dict[str, Any]:
    throttled = baselines.get("combined_regime_throttle", {})
    return {
        "note": (
            "Floors are deliberately conservative vs replay (live-validation TARGETS). "
            f"240d throttled replay: PF={throttled.get('profit_factor')}, "
            f"WR={throttled.get('win_rate_pct')}%, expectancy={throttled.get('expectancy')}."
        ),
        "paper_to_inr_50k": {
            "from": "paper",
            "to": "INR 50,000",
            "min_sessions": 20,
            "min_profit_factor": 1.5,
            "min_win_rate_pct": 55.0,
            "min_expectancy_points": 30.0,
            "max_drawdown_points": 800.0,
            "max_drawdown_pct_of_capital": 8.0,
            "max_median_slippage_pts": 5.0,
            "max_consecutive_losses": 6,
            "max_daily_loss_points": 593.79,
            "lots_max": 1,
            "evidence_required": [
                "Complete 20-session checklist (all P20 gates)",
                "Paper broker fill CSV + engine signal journal joined",
                "Throttle BLOCK compliance report (100%)",
                "Same-bar conflict NO_TRADE log",
                "Slippage distribution (median/p90)",
                "No stack drift (BUY_V3/SELL_V6/fixed_10/60/100/Runner only)",
                "Written risk sign-off for INR 50K",
            ],
            "current_verdict": "NO",
            "current_verdict_rationale": "Paper gate not completed",
        },
        "inr_50k_to_inr_1l": {
            "from": "INR 50,000",
            "to": "INR 100,000",
            "min_sessions": 40,
            "min_sessions_at_50k": 20,
            "min_profit_factor": 1.8,
            "min_win_rate_pct": 58.0,
            "min_expectancy_points": 40.0,
            "max_drawdown_points": 1200.0,
            "max_drawdown_pct_of_capital": 8.0,
            "max_median_slippage_pts": 5.0,
            "max_consecutive_losses": 5,
            "max_daily_loss_points": 593.79,
            "buy_wr_min_pct_on_ge_30_trades": 48.0,
            "sell_rolling_20_pf_min": 1.5,
            "lots_max": 1,
            "evidence_required": [
                "40-session checklist (all P40 gates)",
                "Realized monthly return ≥50% of paper-scaled expectation",
                "Zero BLOCK-regime SELL fills",
                "Live fill quality still within slippage gates at real capital",
                "Structure-stop sensitivity log (paper may keep fixed_10; document DD)",
                "Recovery-from-worst-week ≤10 trading days",
                "Risk sign-off for INR 1L",
            ],
            "current_verdict": "NO",
            "current_verdict_rationale": "Requires INR 50K track record",
        },
        "inr_1l_to_inr_2l": {
            "from": "INR 100,000",
            "to": "INR 200,000",
            "min_sessions": 60,
            "min_sessions_at_1l": 20,
            "min_profit_factor": 2.0,
            "min_win_rate_pct": 60.0,
            "min_expectancy_points": 50.0,
            "max_drawdown_points": 1500.0,
            "max_drawdown_pct_of_capital": 8.0,
            "max_median_slippage_pts": 5.0,
            "max_p90_slippage_pts": 8.0,
            "max_consecutive_losses": 5,
            "max_daily_loss_points": 593.79,
            "execution_risk_material": True,
            "lots_max": 2,
            "evidence_required": [
                "60-session live ledger with independent audit",
                "Slippage stress viability maintained (edge alive at ≤10pt)",
                "Regime throttle map unchanged or improved vs Phase 2",
                "Combined throttled signals/month ≥50",
                "Portfolio DD ≤8% of deployed capital",
                "Confidence score proxy ≥75% (deployment_readiness capital threshold)",
                "Explicit risk committee / owner sign-off for INR 2L",
            ],
            "current_verdict": "NO",
            "current_verdict_rationale": "Execution risk material at INR 2L; gates unmet",
        },
        "production_gate_reference": {
            "win_rate_min_pct_engine_gate": PRODUCTION_GATES["win_rate_min_pct"],
            "profit_factor_min_engine_gate": PRODUCTION_GATES["profit_factor_min"],
            "note": (
                "Engine PRODUCTION_GATES (WR≥65, PF≥2) remain research gates. "
                "Live capital promotions use the more conservative floors above until "
                "live samples reach engine-gate confidence."
            ),
        },
    }


def _risk_controls(*, closure: dict[str, Any], live: dict[str, Any]) -> dict[str, Any]:
    slip_max = float(
        _nested(closure, "part5_live_execution_risk", "slippage_viability_threshold_points", default=10)
        or 10,
    )
    portfolio_daily = float(
        _nested(
            live, "final_answer", "paper_trading_config", "risk_rules", "portfolio_daily_loss_limit_points",
            default=593.79,
        )
        or 593.79,
    )
    return {
        "kill_switch_conditions": [
            {
                "id": "KS-1",
                "trigger": f"Portfolio daily loss exceeds {portfolio_daily} points",
                "action": "FLAT all sleeves; halt new entries for session",
                "severity": "HARD",
            },
            {
                "id": "KS-2",
                "trigger": f"Median trade slippage > {slip_max}pt over any rolling 5 sessions",
                "action": "Halt live/paper promotion; revert to shadow-only; re-calibrate broker",
                "severity": "HARD — edge viability threshold from stress tests",
            },
            {
                "id": "KS-3",
                "trigger": "Any SELL fill in a labeled BLOCK regime (throttle violation)",
                "action": "Immediate halt SELL sleeve; root-cause before resume",
                "severity": "HARD",
            },
            {
                "id": "KS-4",
                "trigger": "Same-bar conflict fill (policy breach — trade opened on conflict bar)",
                "action": "Halt combined engine; fix conflict router",
                "severity": "HARD",
            },
            {
                "id": "KS-5",
                "trigger": "Max consecutive losses ≥7 (paper) or ≥5 (real capital)",
                "action": "Pause new entries 1 full session; review regime + fills",
                "severity": "HARD",
            },
            {
                "id": "KS-6",
                "trigger": "Rolling 20-session combined PF < 1.2",
                "action": "Demote capital tier one step (or paper-only); no scale-up",
                "severity": "HARD",
            },
            {
                "id": "KS-7",
                "trigger": "Rolling 20-session SELL PF proxy < 1.5",
                "action": "Stop SELL sleeve; BUY may continue at half size pending review",
                "severity": "HARD",
            },
            {
                "id": "KS-8",
                "trigger": "Missed entry rate > 15% over 10 sessions",
                "action": "Halt until order path fixed",
                "severity": "HARD",
            },
            {
                "id": "KS-9",
                "trigger": "Same-bar conflict rate > 20% of union signal bars over 10 sessions",
                "action": "Review signal clocks; consider frequency reduction; no capital increase",
                "severity": "SOFT→HARD if also PF degrading",
            },
            {
                "id": "KS-10",
                "trigger": "Stack drift: BUY_V4, SELL_V7, non-fixed_10 paper stop, or non-60/100/Runner enabled",
                "action": "Immediate halt; reset to LOCKED_STACK",
                "severity": "HARD",
            },
            {
                "id": "KS-11",
                "trigger": "Intraday drawdown > 4% of deployed capital (real) or > 400pts (paper proxy)",
                "action": "Flat + session halt",
                "severity": "HARD",
            },
            {
                "id": "KS-12",
                "trigger": "Broker/API disconnect or bar-feed checksum failure mid-session",
                "action": "Cancel working orders; flat if uncertain; no discretionary re-entry",
                "severity": "HARD",
            },
        ],
        "maximum_allowed_drawdown": {
            "paper_points_soft": 800.0,
            "paper_points_hard_kill": 1200.0,
            "real_capital_pct_of_deployed": 8.0,
            "intraday_pct_of_deployed": 4.0,
            "replay_throttled_max_dd_points_reference": 2424.26,
            "note": "Live DD caps are tighter than full-horizon replay DD; scale by capital tier.",
        },
        "maximum_allowed_slippage": {
            "median_pts_pass": 5.0,
            "p90_pts_pass": 8.0,
            "kill_median_pts": slip_max,
            "stress_levels_pts": [0, 2, 5, 10],
        },
        "maximum_consecutive_losses": {
            "paper": 7,
            "inr_50k": 6,
            "inr_1l": 5,
            "inr_2l": 5,
        },
        "other_hard_stops": {
            "same_bar_conflict_fill": "zero tolerance",
            "throttle_block_fill": "zero tolerance",
            "daily_loss_limit_points": portfolio_daily,
            "missed_entry_rate_pct_hard": 15.0,
            "partial_fill_rate_pct_alert": 10.0,
            "capture_efficiency_floor_pct": 30.0,
            "buy_v4_sell_v7": "forbidden in live path",
        },
    }


def _current_evidence_status(exports: dict[str, Any]) -> dict[str, Any]:
    dr = exports.get("deployment_readiness_validation", {}).get("data", {})
    gap = exports.get("production_gap_closure_audit", {}).get("data", {})
    v4 = exports.get("buy_v4_sell_v7_actual_replay_validation", {}).get("data", {})
    gt = exports.get("ground_truth_production_comparison_audit", {}).get("data", {})
    integrity = exports.get("research_integrity_ground_truth_validation_audit", {}).get("data", {})

    still = (
        dr.get("evidence_still_required_before_real_capital")
        or _nested(gap, "final_answer", "missing_evidence_full_capital", default=[])
        or [
            "Live slippage and fill quality on NIFTY50 5M",
            "SELL_V6 validate-window PF stability beyond 40 trading days",
            "BUY_V3 walk-forward with n=6 validate cohort",
            "Intrabar stop/target sequencing vs MFE/MAE proxy",
            "Regime throttle map on unseen regimes",
            "Combined engine same-bar conflict rate in live feed",
        ]
    )

    return {
        "proven_on_replay": [
            "BUY_V3 signal engine passes research gates on historical windows",
            "SELL_V6 + Regime Throttle improves combined PF vs unthrottled (240d PF 5.58 throttled)",
            "fixed_10 + 60/100/Runner is the locked paper stack across live/playbook/extended audits",
            "BUY_V4 / SELL_V7 must NOT replace V3/V6 (ground truth + actual replay: replace=NO)",
            "Slippage stress to 10pt remains viable on replay proxies",
        ],
        "partially_proven": [
            "Execution delay / timing leakage (replay estimates only)",
            "Same-bar conflict rates (replay classification, not live feed)",
            "Capture efficiency ~37–38% (MFE proxy, not broker fills)",
            "Capital tier INR sizing math (points→INR) without live PnL proof",
        ],
        "missing_for_live": still,
        "scores_from_prior_audits": {
            "production_readiness_score": _nested(dr, "production_scores", "production_readiness_score", default=72.0),
            "confidence_score": _nested(dr, "production_scores", "confidence_score", default=66.2),
            "production_risk_score": _nested(dr, "production_scores", "production_risk_score", default=68.5),
            "evidence_score": _nested(dr, "production_scores", "evidence_score", default=84.9),
            "deployment_tier": _nested(dr, "production_scores", "deployment_tier", default="Production Candidate"),
        },
        "v4_v7_status": {
            "ground_truth_replace_buy": _nested(gt, "final_answer", "can_buy_v4_replace_buy_v3", default="NO"),
            "ground_truth_replace_sell": _nested(gt, "final_answer", "can_sell_v7_replace_sell_v6", default="NO"),
            "integrity_replace_buy": _nested(integrity, "final_answer", "can_buy_v4_replace_buy_v3", default="NO"),
            "integrity_replace_sell": _nested(integrity, "final_answer", "can_sell_v7_replace_sell_v6", default="NO"),
            "actual_replay_replace_buy": _nested(v4, "final_answer", "should_buy_v4_replace_buy_v3", default="NO"),
            "actual_replay_replace_sell": _nested(v4, "final_answer", "should_sell_v7_replace_sell_v6", default="NO"),
        },
        "paper_sessions_completed": 0,
        "live_fill_telemetry_present": False,
    }


def _paper_trading_verdict(exports: dict[str, Any]) -> dict[str, Any]:
    dr = exports.get("deployment_readiness_validation", {}).get("data", {})
    gap = exports.get("production_gap_closure_audit", {}).get("data", {})
    live = exports.get("live_trade_management_execution_efficiency_audit", {}).get("data", {})
    answer = (
        _nested(dr, "final_answer", "can_paper_trading_start_now", "answer")
        or _nested(gap, "final_answer", "paper_trading_verdict")
        or _nested(live, "final_answer", "paper_trade_tomorrow")
        or "YES"
    )
    return {
        "verdict": "CONDITIONAL" if answer == "YES" else "NO",
        "paper_start_allowed": answer == "YES",
        "rationale": (
            "Prior audits approve starting paper trading on the locked stack, but real-capital "
            "readiness is NO until 20/40-session gates and live execution telemetry pass. "
            "This framework treats paper as CONDITIONAL: allowed only with kill-switches armed "
            "and LOCKED_STACK fingerprint enforced."
        ),
        "prior_audit_paper_answer": answer,
        "real_capital_ready": "NO",
    }


def _capital_tier_readiness(exports: dict[str, Any]) -> dict[str, Any]:
    dr = exports.get("deployment_readiness_validation", {}).get("data", {})
    fa = dr.get("final_answer", {})
    tiers = _nested(dr, "small_capital_deployment", "tiers", default={})

    def _tier(key: str, label: str, default_answer: str = "NO") -> dict[str, Any]:
        prior = _nested(fa, f"can_{key}_deployment_start_now", default={})
        row = tiers.get(key, {})
        return {
            "tier": label,
            "readiness": row.get("readiness", "CONDITIONAL"),
            "deployment_verdict": row.get("deployment_verdict", default_answer),
            "prior_can_start_now": prior.get("answer", default_answer) if isinstance(prior, dict) else default_answer,
            "rationale": prior.get("evidence") if isinstance(prior, dict) else (
                "No live paper gate completed; confidence below capital threshold."
            ),
            "framework_verdict": "NO",
        }

    return {
        "inr_50k": _tier("inr_50k", "INR 50K"),
        "inr_1l": _tier("inr_1l", "INR 1L"),
        "inr_2l": _tier("inr_2l", "INR 2L"),
        "overall": "NO — paper-only until promotion criteria satisfied",
    }


def _evidence_required_before_real_capital(
    *,
    evidence_status: dict[str, Any],
    promotion: dict[str, Any],
) -> list[str]:
    items = list(evidence_status.get("missing_for_live") or [])
    mandatory = [
        "20-session paper checklist fully passed (P20-1..P20-10) with artifacts archived",
        "Joined paper-broker fill log + engine signal journal for all sessions",
        "Slippage distribution: median ≤5pt, p90 ≤8pt, no kill-threshold breach",
        "Throttle BLOCK compliance = 100% (zero BLOCK-regime SELL fills)",
        "Same-bar conflict NO_TRADE compliance = 100%",
        "Stop and target execution accuracy gates met on paper fills",
        "Capture efficiency in 34–41% band (±3pp of ~37.66% replay proxy)",
        "Combined live-validation PF/WR/expectancy floors met for chosen capital tier",
        "Kill-switch drill documented (at least one simulated daily-loss halt)",
        "Stack lock attestation: BUY_V3 + SELL_V6 + fixed_10 + 60/100/Runner + Regime Throttle only",
        "Written risk sign-off for the specific capital tier",
        "Confidence/readiness: no open thesis-invalidating unknowns from prior audits",
    ]
    # de-dupe preserving order
    seen: set[str] = set()
    out: list[str] = []
    for item in mandatory + items + promotion["paper_to_inr_50k"]["evidence_required"]:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _final_answer(
    *,
    paper: dict[str, Any],
    capital: dict[str, Any],
    evidence_required: list[str],
    risk: dict[str, Any],
    baselines: dict[str, Any],
) -> dict[str, Any]:
    return {
        "stack_locked": (
            "BUY_V3 + SELL_V6 + fixed_10 + 60/100/Runner + Regime Throttle"
        ),
        "buy_v4_sell_v7": "DO_NOT_PROMOTE",
        "paper_trading_verdict": paper["verdict"],
        "real_capital_deployment_ready": "NO",
        "capital_tier_readiness": {
            "inr_50k": capital["inr_50k"]["framework_verdict"],
            "inr_1l": capital["inr_1l"]["framework_verdict"],
            "inr_2l": capital["inr_2l"]["framework_verdict"],
        },
        "evidence_required_before_real_capital": evidence_required,
        "kill_switch_summary": [ks["trigger"] for ks in risk["kill_switch_conditions"][:6]],
        "replay_baseline_throttled_240d": baselines.get("combined_regime_throttle"),
        "one_line_verdict": (
            "NO — real capital not ready; paper-only with CONDITIONAL start on locked stack "
            "until 20/40-session live validation gates pass."
        ),
    }


class LiveDeploymentValidationFrameworkResearch:
    """Build live deployment validation framework from existing research exports."""

    def __init__(self, report_path: Path = DEFAULT_REPORT_PATH) -> None:
        self.report_path = report_path

    def _load_exports(self) -> dict[str, Any]:
        loaded: dict[str, Any] = {}
        for name, path in SOURCE_EXPORTS.items():
            data = _load_json(path, required=False)
            loaded[name] = {"data": data, **_export_meta(path, data)}
        if not any(meta.get("loaded") for meta in loaded.values()):
            logger.warning(
                "No prior research exports found — synthesizing from embedded baselines only.",
            )
        return loaded

    def run(self) -> LiveDeploymentValidationFrameworkReport:
        started = time.perf_counter()
        exports = self._load_exports()

        closure = exports.get("production_readiness_closure_audit", {}).get("data", {})
        live = exports.get("live_trade_management_execution_efficiency_audit", {}).get("data", {})

        baselines = _extract_replay_baselines(exports)
        measurements = _measurement_definitions(baselines=baselines, closure=closure)
        schema = _session_logging_schema()
        c20 = _checklist_20_session()
        c40 = _checklist_40_session()
        promotion = _promotion_criteria(baselines=baselines)
        risk = _risk_controls(closure=closure, live=live)
        evidence_status = _current_evidence_status(exports)
        paper = _paper_trading_verdict(exports)
        capital = _capital_tier_readiness(exports)
        evidence_required = _evidence_required_before_real_capital(
            evidence_status=evidence_status, promotion=promotion,
        )
        final = _final_answer(
            paper=paper,
            capital=capital,
            evidence_required=evidence_required,
            risk=risk,
            baselines=baselines,
        )

        source_summary = {
            name: {"path": meta["path"], "status": meta["status"]}
            for name, meta in exports.items()
        }

        conclusions = [
            final["one_line_verdict"],
            (
                f"Locked stack only: {final['stack_locked']}; "
                "BUY_V4/SELL_V7 remain DO_NOT_PROMOTE."
            ),
            (
                "Paper trading may start CONDITIONAL with kill-switches armed; "
                "INR 50K / 1L / 2L all NO until promotion criteria and live telemetry pass."
            ),
            (
                "Primary live unknowns: slippage, partial fills, intrabar stop/target sequencing, "
                "and throttle compliance on unseen regimes."
            ),
            (
                f"Kill-switch anchors: daily loss {risk['other_hard_stops']['daily_loss_limit_points']}pts, "
                f"slippage kill {risk['maximum_allowed_slippage']['kill_median_pts']}pt, "
                "BLOCK-regime fill, conflict fill, consecutive-loss caps."
            ),
        ]

        return LiveDeploymentValidationFrameworkReport(
            report_type="live_deployment_validation_framework",
            engines=["BUY_V3", "SELL_V6", "COMBINED"],
            symbol="NIFTY50",
            timeframe="5M",
            methodology={
                "research_only": True,
                "no_replay": True,
                "no_new_engines": True,
                "synthesis_from_exports": True,
                "stack_policy": "BUY_V3 + SELL_V6 only; V4/V7 forbidden",
                "threshold_label": "live_validation_TARGETS_not_proven_live",
            },
            source_exports=source_summary,
            limitations=[
                "No live paper sessions have been executed under this framework yet.",
                "Replay PF/WR/expectancy are baselines for targets, not live proof.",
                "MFE/MAE proxies do not establish intrabar stop vs target ordering.",
                "Partial-fill behavior is unmeasured on the paper broker.",
                "Capital INR conversions assume point-value proxies from prior audits.",
            ],
            stack_locked=dict(LOCKED_STACK),
            replay_baselines=baselines,
            current_evidence_status=evidence_status,
            measurement_definitions=measurements,
            session_logging_schema=schema,
            checklist_20_session=c20,
            checklist_40_session=c40,
            promotion_criteria=promotion,
            risk_controls=risk,
            paper_trading_verdict=paper,
            capital_tier_readiness=capital,
            evidence_required_before_real_capital=evidence_required,
            final_answer=final,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )

    def export(self, report: LiveDeploymentValidationFrameworkReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Live deployment validation framework exported to %s", self.report_path)
        return self.report_path


def generate_live_deployment_validation_framework_report(
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export live deployment validation framework JSON."""
    return LiveDeploymentValidationFrameworkResearch(report_path=report_path).export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_live_deployment_validation_framework_report()
    report = json.loads(path.read_text(encoding="utf-8"))
    final = report["final_answer"]
    print(f"Exported: {path}")
    print(f"Verdict: {final['one_line_verdict']}")
    print(f"Paper: {final['paper_trading_verdict']} | Real capital: {final['real_capital_deployment_ready']}")
    print(f"Kill-switches: {len(report['risk_controls']['kill_switch_conditions'])}")
