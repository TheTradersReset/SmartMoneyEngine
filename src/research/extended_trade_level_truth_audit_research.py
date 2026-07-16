"""
Extended Trade Level Truth Audit — multi-window replay + trade-level synthesis.

Runs actual NIFTY50 5M replays for 240/300/500 trading days (largest available in
one pass), then synthesizes target achievement, conditional probability, lifecycle,
entry precision, execution failure, runner optimization, and V4/V7 potential.
Research-only; no production signal logic changes.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import Any

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.buy_failure_anatomy_research import _json_safe
from src.research.buy_v2_candidate_validation_research import _filter_signals_by_dates
from src.research.buy_v3_candidate_validation_research import BUY_V3_MODEL_ID
from src.research.extended_evidence_validation_real_deployment_audit_research import (
    ExtendedEvidenceValidationRealDeploymentAuditResearch,
    _cohort_metrics_block,
    _combined_throttled_metrics,
    _load_json_safe,
    _load_throttle_maps,
)
from src.research.filter_research_engine import FilterResearchEngine
from src.research.nifty50_trap_to_momentum_validation_research import DEFAULT_SYMBOL
from src.research.production_edge_enhancement_audit_research import (
    _is_buy_winner,
    _is_sell_winner,
)
from src.research.production_reality_audit_research import (
    RUNNER_STRATEGIES,
    _extended_metrics,
    _runner_exit_optimization,
)
from src.research.production_trading_playbook_audit_research import _tiered_structure_pnl
from src.research.live_trade_management_execution_efficiency_audit_research import _resolve_stop_extended
from src.research.regime_aware_execution_validation_research import (
    _execution_failure_audit,
)
from src.research.regime_detection_audit_research import SELL_V6_MODEL_ID
from src.research.smartmoneyengine_v3_implementation_validation_research import (
    MOVE_DETECTION_TIMEFRAME,
    PIPELINE_TIMEFRAMES,
)
from src.research.smartmoneyengine_v4_candidate_validation_research import (
    _attach_ema22,
    _last_n_trading_day_set,
)
from src.research.production_edge_enhancement_audit_research import _classify_sell_signal
from src.research.trade_level_truth_audit_research import (
    _build_final_answer,
    _build_per_signal_records,
    _buy_v4_sell_v7_potential,
    _classify_buy_loser,
    _conditional_probability_analysis,
    _entry_precision_audit,
    _trade_level_target_matrix,
    _trade_lifecycle_analysis,
    _uncaptured_edge,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "extended_trade_level_truth_audit.json"
DEFAULT_LOG_PATH = PROJECT_ROOT / "extended_trade_truth_audit_run.log"

PREFERRED_WINDOWS = (240, 300, 500)
CALENDAR_BUFFER = {240: 380, 300: 480, 500: 780}
REGIME_EXPORT_PATH = RESEARCH_DIR / "regime_detection_audit.json"
PRODUCTION_STRUCTURE = RUNNER_STRATEGIES["60_100_runner"]
DEFAULT_STOP_VARIANT = "fixed_10"


class ExtendedTradeLevelTruthAuditError(Exception):
    """Raised when extended trade level truth audit fails."""


@dataclass
class ExtendedTradeLevelTruthAuditReport:
    """Extended trade level truth audit output."""

    report_type: str
    engines: list[str]
    symbol: str
    timeframe: str
    replay_windows: list[int]
    max_replay_window: int
    available_trading_days: int
    replay_start_date: str
    replay_end_date: str
    methodology: dict[str, Any]
    core_metrics_by_window: dict[str, Any]
    per_signal_details: dict[str, Any]
    target_achievement_matrix: dict[str, Any]
    conditional_probability: dict[str, Any]
    trade_lifecycle_analysis: dict[str, Any]
    entry_precision_audit: dict[str, Any]
    execution_failure_audit: dict[str, Any]
    runner_optimization_audit: dict[str, Any]
    buy_v4_sell_v7_potential: dict[str, Any]
    uncaptured_edge: dict[str, Any]
    final_answer: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _count_trading_days(frame: pd.DataFrame) -> int:
    return int(pd.to_datetime(frame["Date"]).dt.date.nunique())


def _resolve_replay_windows(available_days: int) -> tuple[int, ...]:
    """Pick 240/300/500 windows that fit available data; always include largest feasible."""
    if available_days < 240:
        return (available_days,)
    active = [window for window in PREFERRED_WINDOWS if window <= available_days]
    if not active:
        largest = min(PREFERRED_WINDOWS)
        return (min(available_days, largest),)
    return tuple(sorted(active))


def _summarize_core_metrics(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    *,
    frame: pd.DataFrame,
    replay_dates: set[date],
    trading_days: int,
    moves: list[Any],
    throttle_maps: dict[str, dict[str, str]],
) -> dict[str, Any]:
    combined = sorted(buy_signals + sell_signals, key=lambda s: (s.get("timestamp", ""), s.get("bar", 0)))
    buy_block = _cohort_metrics_block(
        buy_signals,
        trading_days=trading_days,
        win_fn=_is_buy_winner,
        moves=moves,
        frame=frame,
        replay_dates=replay_dates,
        direction="BUY",
    )
    sell_block = _cohort_metrics_block(
        sell_signals,
        trading_days=trading_days,
        win_fn=_is_sell_winner,
        moves=moves,
        frame=frame,
        replay_dates=replay_dates,
        direction="SELL",
    )
    combined_block = _cohort_metrics_block(
        combined,
        trading_days=trading_days,
        win_fn=lambda s: _is_buy_winner(s) if s.get("direction") == "BUY" else _is_sell_winner(s),
        moves=moves,
        frame=frame,
        replay_dates=replay_dates,
        direction="COMBINED",
    )
    throttled = _combined_throttled_metrics(
        buy_signals,
        sell_signals,
        throttle_maps,
        trading_days=trading_days,
    )

    def _playbook_capture(signals: list[dict[str, Any]]) -> float | None:
        if not signals:
            return None
        mae_median = median(float(s.get("mae_points") or 0.0) for s in signals)
        pnls: list[float] = []
        for signal in signals:
            stop_pts = _resolve_stop_extended(signal, DEFAULT_STOP_VARIANT, cohort_mae_median=mae_median)
            pnl, _ = _tiered_structure_pnl(signal, PRODUCTION_STRUCTURE, stop_pts=stop_pts)
            pnls.append(pnl)
        return _extended_metrics(pnls, signals=signals, sample_size=len(signals), window_days=trading_days).get(
            "capture_efficiency_pct",
        )

    def _compact(block: dict[str, Any], signals: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "signals_emitted": block.get("signals_emitted"),
            "signals_per_month": block.get("signals_per_month"),
            "win_rate_pct": block.get("win_rate_pct"),
            "profit_factor": block.get("profit_factor"),
            "expectancy": block.get("expectancy"),
            "capture_efficiency_pct": _playbook_capture(signals),
            "max_drawdown_points": block.get("max_drawdown_points"),
            "recovery_factor": block.get("recovery_factor"),
        }

    return {
        "trading_days": trading_days,
        "replay_start_date": min(replay_dates).isoformat() if replay_dates else "",
        "replay_end_date": max(replay_dates).isoformat() if replay_dates else "",
        "buy_v3": _compact(buy_block, buy_signals),
        "sell_v6": _compact(sell_block, sell_signals),
        "combined": _compact(combined_block, combined),
        "combined_regime_throttle": {
            "signals_per_month": throttled.get("signals_per_month"),
            "effective_signals_per_month": throttled.get("effective_signals_per_month"),
            "win_rate_pct": throttled.get("win_rate_pct"),
            "profit_factor": throttled.get("profit_factor"),
            "expectancy": throttled.get("expectancy"),
            "max_drawdown_points": throttled.get("max_drawdown_points"),
            "recovery_factor": throttled.get("recovery_factor"),
        },
    }


def _analyze_trade_level_window(
    buy_signals: list[dict[str, Any]],
    sell_signals: list[dict[str, Any]],
    *,
    trading_days: int,
    stop_variant: str = DEFAULT_STOP_VARIANT,
) -> dict[str, Any]:
    structure = PRODUCTION_STRUCTURE
    return {
        "trading_days": trading_days,
        "signal_counts": {"buy_v3": len(buy_signals), "sell_v6": len(sell_signals)},
        "per_signal_records": {
            "buy_v3": _build_per_signal_records(
                buy_signals,
                side="BUY",
                structure=structure,
                stop_variant=stop_variant,
                win_fn=_is_buy_winner,
                classify_fn=_classify_buy_loser,
            ),
            "sell_v6": _build_per_signal_records(
                sell_signals,
                side="SELL",
                structure=structure,
                stop_variant=stop_variant,
                win_fn=_is_sell_winner,
                classify_fn=_classify_sell_signal,
            ),
        },
        "target_achievement_matrix": {
            "buy_v3": _trade_level_target_matrix(
                buy_signals, side="BUY", structure=structure, stop_variant=stop_variant,
            ),
            "sell_v6": _trade_level_target_matrix(
                sell_signals, side="SELL", structure=structure, stop_variant=stop_variant,
            ),
        },
        "conditional_probability": {
            "buy_v3": _conditional_probability_analysis(
                buy_signals, side="BUY", structure=structure, stop_variant=stop_variant,
            ),
            "sell_v6": _conditional_probability_analysis(
                sell_signals, side="SELL", structure=structure, stop_variant=stop_variant,
            ),
        },
        "trade_lifecycle_analysis": {
            "buy_v3": _trade_lifecycle_analysis(
                buy_signals, side="BUY", structure=structure, stop_variant=stop_variant, win_fn=_is_buy_winner,
            ),
            "sell_v6": _trade_lifecycle_analysis(
                sell_signals, side="SELL", structure=structure, stop_variant=stop_variant, win_fn=_is_sell_winner,
            ),
        },
        "entry_precision_audit": {
            "buy_v3": _entry_precision_audit(buy_signals, side="BUY", win_fn=_is_buy_winner),
            "sell_v6": _entry_precision_audit(sell_signals, side="SELL", win_fn=_is_sell_winner),
        },
        "execution_failure_audit": {
            "buy_v3": _execution_failure_audit(
                buy_signals, structure=structure, win_fn=_is_buy_winner, window_days=trading_days,
            ),
            "sell_v6": _execution_failure_audit(
                sell_signals, structure=structure, win_fn=_is_sell_winner, window_days=trading_days,
            ),
        },
        "runner_optimization_audit": {
            "buy_v3": _runner_exit_optimization(
                buy_signals, side="BUY", stop_variant=stop_variant, window_days=trading_days,
            ),
            "sell_v6": _runner_exit_optimization(
                sell_signals, side="SELL", stop_variant=stop_variant, window_days=trading_days,
            ),
        },
        "buy_v4_sell_v7_potential": _buy_v4_sell_v7_potential(buy_signals, sell_signals),
        "uncaptured_edge": _uncaptured_edge(
            buy_signals,
            sell_signals,
            buy_structure=structure,
            sell_structure=structure,
            buy_stop=stop_variant,
            sell_stop=stop_variant,
            window_days=trading_days,
        ),
    }


def _build_extended_final_answer(
    *,
    max_window_analysis: dict[str, Any],
    core_metrics_by_window: dict[str, Any],
) -> dict[str, Any]:
    base = _build_final_answer(
        buy_cond=max_window_analysis["conditional_probability"]["buy_v3"],
        sell_cond=max_window_analysis["conditional_probability"]["sell_v6"],
        buy_lifecycle=max_window_analysis["trade_lifecycle_analysis"]["buy_v3"],
        sell_lifecycle=max_window_analysis["trade_lifecycle_analysis"]["sell_v6"],
        buy_entry=max_window_analysis["entry_precision_audit"]["buy_v3"],
        sell_entry=max_window_analysis["entry_precision_audit"]["sell_v6"],
        v4_v7=max_window_analysis["buy_v4_sell_v7_potential"],
        uncaptured=max_window_analysis["uncaptured_edge"],
    )

    buy_exec = max_window_analysis["execution_failure_audit"]["buy_v3"]
    sell_exec = max_window_analysis["execution_failure_audit"]["sell_v6"]
    buy_runner = max_window_analysis["runner_optimization_audit"]["buy_v3"]
    sell_runner = max_window_analysis["runner_optimization_audit"]["sell_v6"]

    stop_loss_validation = {
        "playbook_structure": "60/100/Runner",
        "stop_variants_tested": list(buy_exec["by_stop_variant"].keys()),
        "buy_v3": {
            "best_stop_variant": buy_exec["best_stop_variant"],
            "by_stop_variant": buy_exec["by_stop_variant"],
            "min_stop_70pct_winners": buy_exec["by_stop_variant"]
            .get(buy_exec["best_stop_variant"], {})
            .get("min_stop_for_winner_preservation", {})
            .get("70pct"),
        },
        "sell_v6": {
            "best_stop_variant": sell_exec["best_stop_variant"],
            "by_stop_variant": sell_exec["by_stop_variant"],
            "min_stop_70pct_winners": sell_exec["by_stop_variant"]
            .get(sell_exec["best_stop_variant"], {})
            .get("min_stop_for_winner_preservation", {})
            .get("70pct"),
        },
    }

    runner_validation = {
        "strategies_compared": list(RUNNER_STRATEGIES.keys()),
        "production_strategy": "60_100_runner",
        "buy_v3": {
            "best_strategy": buy_runner["best_strategy"],
            "current_vs_best": buy_runner["current_vs_best"],
            "by_strategy": {
                key: {
                    "profit_factor": row.get("profit_factor"),
                    "win_rate_pct": row.get("win_rate_pct"),
                    "expectancy": row.get("expectancy"),
                    "capture_efficiency_pct": row.get("capture_efficiency_pct"),
                    "monthly_points": row.get("monthly_points"),
                }
                for key, row in buy_runner["by_strategy"].items()
            },
        },
        "sell_v6": {
            "best_strategy": sell_runner["best_strategy"],
            "current_vs_best": sell_runner["current_vs_best"],
            "by_strategy": {
                key: {
                    "profit_factor": row.get("profit_factor"),
                    "win_rate_pct": row.get("win_rate_pct"),
                    "expectancy": row.get("expectancy"),
                    "capture_efficiency_pct": row.get("capture_efficiency_pct"),
                    "monthly_points": row.get("monthly_points"),
                }
                for key, row in sell_runner["by_strategy"].items()
            },
        },
    }

    probability_by_window = {
        window: {
            "buy_v3": core_metrics_by_window[window]["conditional_probability"]["buy_v3"]["summary"]
            if window in core_metrics_by_window
            else {},
            "sell_v6": core_metrics_by_window[window]["conditional_probability"]["sell_v6"]["summary"]
            if window in core_metrics_by_window
            else {},
        }
        for window in core_metrics_by_window
        if isinstance(core_metrics_by_window.get(window), dict) and "conditional_probability" in core_metrics_by_window[window]
    }

    return {
        **base,
        "target_achievement_probability_matrix": base["probability_matrix"],
        "trade_lifecycle_matrix": base["trade_lifecycle_matrix"],
        "entry_quality_matrix": base["entry_quality_matrix"],
        "stop_loss_validation": stop_loss_validation,
        "runner_validation": runner_validation,
        "buy_v4_recommendation": base["buy_v4_recommendation"],
        "sell_v7_recommendation": base["sell_v7_recommendation"],
        "maximum_remaining_improvement": base["maximum_theoretical_improvement"],
        "probability_by_window": probability_by_window,
    }


class ExtendedTradeLevelTruthAuditResearch(ExtendedEvidenceValidationRealDeploymentAuditResearch):
    """Multi-window replay with trade-level truth synthesis."""

    def run(
        self,
        metadata: dict[str, Any],
        *,
        windows: tuple[int, ...] | None = None,
    ) -> ExtendedTradeLevelTruthAuditReport:
        started = time.perf_counter()

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=CALENDAR_BUFFER[500],
            timeframes=PIPELINE_TIMEFRAMES,
        )
        end = date.fromisoformat(metadata["end_date"])
        calendar_days = CALENDAR_BUFFER[500]
        start = end - timedelta(days=calendar_days)

        logger.info(
            "Extended trade level truth audit starting: preferred windows=%s, %s 5M",
            PREFERRED_WINDOWS,
            DEFAULT_SYMBOL,
        )

        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        available_days = _count_trading_days(frame)
        active_windows = windows or _resolve_replay_windows(available_days)
        max_window = max(active_windows)

        logger.info(
            "Data: %s trading days available; replay windows=%s (max=%sd)",
            available_days,
            active_windows,
            max_window,
        )

        all_replay_dates = _last_n_trading_day_set(frame, max_window)
        bar_dates = pd.to_datetime(frame["Date"]).dt.date
        replay_bars = [index for index, day in enumerate(bar_dates) if day in all_replay_dates]

        logger.info("Loading enriched context and intel frames...")
        enriched_buy = self.buy_engine.context_builder.enrich(frame)
        enriched_sell = _attach_ema22(self.sell_engine.context_builder.enrich(frame))
        calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        intel_frames: dict[str, pd.DataFrame] = {"5M": self.buy_engine.intelligence.enrich(frame)}
        for tf in ("15M", "1H"):
            tf_path = filter_engine._ensure_pipeline(tf, start, end)
            tf_frame = pd.read_csv(tf_path).reset_index(drop=True)
            intel_frames[tf] = self.buy_engine.intelligence.enrich(tf_frame)
        intel_frames["1D"] = self.buy_engine.intelligence.enrich(
            self.buy_engine._resample_daily(intel_frames["1H"]),
        )

        from src.research.sell_v6_replay_validation_research import _daily_range_lookup

        logger.info("Detecting moves...")
        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, 40),
        )
        daily_ranges = _daily_range_lookup(frame)

        logger.info("Running %sd production replay (BUY_V3 + SELL_V6)...", max_window)
        full_signals = self._replay_production(
            frame=frame,
            enriched_buy=enriched_buy,
            enriched_sell=enriched_sell,
            calendar=calendar,
            intel_frames=intel_frames,
            replay_bars=replay_bars,
            moves=moves,
            daily_ranges=daily_ranges,
            include_ablation=False,
        )

        regime_export = _load_json_safe(REGIME_EXPORT_PATH)
        throttle_maps = _load_throttle_maps(regime_export)

        core_metrics_by_window: dict[str, Any] = {}
        target_achievement_matrix: dict[str, Any] = {}
        conditional_probability: dict[str, Any] = {}
        trade_lifecycle_analysis: dict[str, Any] = {}
        entry_precision_audit: dict[str, Any] = {}
        execution_failure_audit: dict[str, Any] = {}
        runner_optimization_audit: dict[str, Any] = {}
        buy_v4_by_window: dict[str, Any] = {}
        uncaptured_by_window: dict[str, Any] = {}
        window_analyses: dict[str, Any] = {}

        for window in active_windows:
            logger.info("Analyzing trade-level window: %sd trading days", window)
            window_dates = _last_n_trading_day_set(frame, window)
            buy_w = _filter_signals_by_dates(full_signals["buy_v3"], frame, window_dates)
            sell_w = _filter_signals_by_dates(full_signals["sell_v6"], frame, window_dates)

            core_metrics_by_window[str(window)] = _summarize_core_metrics(
                buy_w,
                sell_w,
                frame=frame,
                replay_dates=window_dates,
                trading_days=window,
                moves=moves,
                throttle_maps=throttle_maps,
            )

            analysis = _analyze_trade_level_window(buy_w, sell_w, trading_days=window)
            window_analyses[str(window)] = analysis

            target_achievement_matrix[str(window)] = analysis["target_achievement_matrix"]
            conditional_probability[str(window)] = analysis["conditional_probability"]
            trade_lifecycle_analysis[str(window)] = analysis["trade_lifecycle_analysis"]
            entry_precision_audit[str(window)] = analysis["entry_precision_audit"]
            execution_failure_audit[str(window)] = analysis["execution_failure_audit"]
            runner_optimization_audit[str(window)] = analysis["runner_optimization_audit"]
            buy_v4_by_window[str(window)] = analysis["buy_v4_sell_v7_potential"]
            uncaptured_by_window[str(window)] = analysis["uncaptured_edge"]

            core_metrics_by_window[str(window)]["conditional_probability"] = analysis["conditional_probability"]

            logger.info(
                "Window %sd: BUY=%s SELL=%s combined PF=%s",
                window,
                len(buy_w),
                len(sell_w),
                core_metrics_by_window[str(window)]["combined"].get("profit_factor"),
            )

        max_key = str(max_window)
        max_analysis = window_analyses[max_key]
        max_dates = _last_n_trading_day_set(frame, max_window)

        final_answer = _build_extended_final_answer(
            max_window_analysis=max_analysis,
            core_metrics_by_window={
                k: v for k, v in window_analyses.items() if isinstance(v, dict)
            },
        )

        v4_v7 = max_analysis["buy_v4_sell_v7_potential"]
        uncaptured = max_analysis["uncaptured_edge"]
        buy_cond = max_analysis["conditional_probability"]["buy_v3"]
        sell_cond = max_analysis["conditional_probability"]["sell_v6"]

        conclusions = [
            (
                f"Replay windows: {list(active_windows)}d | max={max_window}d | "
                f"available={available_days} trading days."
            ),
            (
                f"Max window signals: BUY_V3 n={len(full_signals['buy_v3'])} | "
                f"SELL_V6 n={len(full_signals['sell_v6'])}."
            ),
            (
                f"P(40+ before stop) @ {max_window}d: BUY {buy_cond['summary']['p_40_plus']}% | "
                f"SELL {sell_cond['summary']['p_40_plus']}%."
            ),
            (
                f"P(100+ before stop) @ {max_window}d: BUY {buy_cond['summary']['p_100_plus']}% | "
                f"SELL {sell_cond['summary']['p_100_plus']}%."
            ),
            (
                f"Lifecycle capture @ {max_window}d: BUY "
                f"{max_analysis['trade_lifecycle_analysis']['buy_v3']['aggregate']['capture_efficiency_pct']}% | "
                f"SELL {max_analysis['trade_lifecycle_analysis']['sell_v6']['aggregate']['capture_efficiency_pct']}%."
            ),
            (
                f"BUY_V4: {final_answer['buy_v4_recommendation']} | "
                f"SELL_V7: {final_answer['sell_v7_recommendation']}."
            ),
            (
                f"Runner headroom: {uncaptured['combined']['avg_best_strategy_capture_pct']}% best vs "
                f"{uncaptured['combined']['avg_current_capture_pct']}% current capture."
            ),
        ]

        for window in active_windows:
            cm = core_metrics_by_window[str(window)]
            conclusions.append(
                f"{window}d core: BUY PF={cm['buy_v3'].get('profit_factor')} "
                f"SELL PF={cm['sell_v6'].get('profit_factor')} "
                f"combined PF={cm['combined'].get('profit_factor')} "
                f"spm={cm['combined'].get('signals_per_month')}.",
            )

        methodology = {
            "research_only": True,
            "replay_required": True,
            "engines": [BUY_V3_MODEL_ID, SELL_V6_MODEL_ID],
            "exit_playbook": "60/100/Runner",
            "regime_throttle_source": str(REGIME_EXPORT_PATH),
            "stop_variant_default": DEFAULT_STOP_VARIANT,
            "preferred_windows": list(PREFERRED_WINDOWS),
            "active_windows": list(active_windows),
            "single_pass_max_window": max_window,
            "synthesis_modules": [
                "trade_level_truth_audit_research",
                "regime_aware_execution_validation_research",
                "production_reality_audit_research",
            ],
            "excluded": "BUY_V4, SELL_V7 engines, new indicators/models/discovery",
        }

        return ExtendedTradeLevelTruthAuditReport(
            report_type="Extended Trade Level Truth Audit",
            engines=["BUY_V3", "SELL_V6"],
            symbol=DEFAULT_SYMBOL,
            timeframe=MOVE_DETECTION_TIMEFRAME,
            replay_windows=list(active_windows),
            max_replay_window=max_window,
            available_trading_days=available_days,
            replay_start_date=min(max_dates).isoformat() if max_dates else "",
            replay_end_date=max(max_dates).isoformat() if max_dates else "",
            methodology=methodology,
            core_metrics_by_window=core_metrics_by_window,
            per_signal_details={
                "buy_v3": _filter_signals_by_dates(full_signals["buy_v3"], frame, max_dates),
                "sell_v6": _filter_signals_by_dates(full_signals["sell_v6"], frame, max_dates),
            },
            target_achievement_matrix=target_achievement_matrix,
            conditional_probability=conditional_probability,
            trade_lifecycle_analysis=trade_lifecycle_analysis,
            entry_precision_audit=entry_precision_audit,
            execution_failure_audit=execution_failure_audit,
            runner_optimization_audit=runner_optimization_audit,
            buy_v4_sell_v7_potential={
                "by_window": buy_v4_by_window,
                "max_window": v4_v7,
            },
            uncaptured_edge={
                "by_window": uncaptured_by_window,
                "max_window": uncaptured,
            },
            final_answer=final_answer,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )

    def export(
        self,
        report: ExtendedTradeLevelTruthAuditReport,
        report_path: Path = DEFAULT_REPORT_PATH,
    ) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("Extended trade level truth audit exported: %s", report_path)
        return report_path


def _configure_logging(log_path: Path = DEFAULT_LOG_PATH) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_path) for h in root.handlers):
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root.addHandler(stream_handler)


def generate_extended_trade_level_truth_audit_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    *,
    windows: tuple[int, ...] | None = None,
) -> ExtendedTradeLevelTruthAuditReport:
    """Run extended trade level truth replay audit and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise ExtendedTradeLevelTruthAuditError(f"Filter research report not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    research = ExtendedTradeLevelTruthAuditResearch()
    report = research.run(metadata, windows=windows)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    _configure_logging()
    try:
        report = generate_extended_trade_level_truth_audit_report()
        final = report.final_answer
        prob = final["target_achievement_probability_matrix"]
        print(f"Exported: {DEFAULT_REPORT_PATH}")
        print(f"Windows: {report.replay_windows} | max={report.max_replay_window}d")
        print(
            f"BUY signals={len(report.per_signal_details['buy_v3'])} | "
            f"SELL signals={len(report.per_signal_details['sell_v6'])}"
        )
        print(
            f"BUY P(40+/60+/100+): {prob['buy_v3']['p_40_plus']}% / "
            f"{prob['buy_v3']['p_60_plus']}% / {prob['buy_v3']['p_100_plus']}%"
        )
        print(
            f"SELL P(40+/60+/100+): {prob['sell_v6']['p_40_plus']}% / "
            f"{prob['sell_v6']['p_60_plus']}% / {prob['sell_v6']['p_100_plus']}%"
        )
        print(f"BUY_V4: {final['buy_v4_recommendation']} | SELL_V7: {final['sell_v7_recommendation']}")
        return 0
    except ExtendedTradeLevelTruthAuditError as exc:
        logger.error("Extended trade level truth audit failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
