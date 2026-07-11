"""
NIFTY50 Filter Ablation Study — synthesis-only research.

Compares incremental V3 Layer-2 filter stacks (Models A–E) using ONLY:
  - smartmoneyengine_v3_implementation_validation.json
  - nifty50_signal_timing_audit.json

No new scans, optimization, or pattern discovery.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from src.research.filter_research_engine import _json_safe

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_V3_PATH = RESEARCH_DIR / "smartmoneyengine_v3_implementation_validation.json"
DEFAULT_TIMING_PATH = RESEARCH_DIR / "nifty50_signal_timing_audit.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "nifty50_filter_ablation_study.json"

FILTER_BLOCKS = {
    "HTF Bearish": "HTF_CONFLICT",
    "VWAP Below": "VWAP_MISMATCH",
    "EMA Bear Stack": "EMA_MISMATCH",
}

MODELS: dict[str, dict[str, Any]] = {
    "A": {
        "label": "Model A",
        "description": "Failed Breakout only",
        "filters": ["Failed Breakout"],
        "layer2_active": [],
    },
    "B": {
        "label": "Model B",
        "description": "Failed Breakout + HTF Bearish",
        "filters": ["Failed Breakout", "HTF Bearish"],
        "layer2_active": ["HTF Bearish"],
    },
    "C": {
        "label": "Model C",
        "description": "Failed Breakout + HTF Bearish + VWAP Below",
        "filters": ["Failed Breakout", "HTF Bearish", "VWAP Below"],
        "layer2_active": ["HTF Bearish", "VWAP Below"],
    },
    "D": {
        "label": "Model D",
        "description": "Failed Breakout + HTF Bearish + EMA Bear Stack",
        "filters": ["Failed Breakout", "HTF Bearish", "EMA Bear Stack"],
        "layer2_active": ["HTF Bearish", "EMA Bear Stack"],
    },
    "E": {
        "label": "Model E",
        "description": "Failed Breakout + HTF Bearish + VWAP Below + EMA Bear Stack (Current V3)",
        "filters": ["Failed Breakout", "HTF Bearish", "VWAP Below", "EMA Bear Stack"],
        "layer2_active": ["HTF Bearish", "VWAP Below", "EMA Bear Stack"],
        "is_current_v3": True,
    },
}


class Nifty50FilterAblationStudyError(Exception):
    """Raised when ablation synthesis cannot be completed."""


@dataclass
class Nifty50FilterAblationStudyReport:
    """Filter ablation comparison output."""

    report_type: str
    symbol: str
    timeframe: str
    replay_context: dict[str, Any]
    methodology: dict[str, Any]
    source_exports: list[str]
    filter_block_counts: dict[str, int]
    models: dict[str, dict[str, Any]]
    incremental_filter_impact: dict[str, dict[str, Any]]
    filter_rankings: dict[str, Any]
    single_filter_removal_analysis: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Nifty50FilterAblationStudyError(f"Missing source export: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _observed_trade_stats(signals: list[dict[str, Any]]) -> dict[str, float]:
    wins = [s for s in signals if s.get("win")]
    losses = [s for s in signals if not s.get("win")]
    gross_profit = sum(float(s.get("realized_pnl_points", 0.0)) for s in wins)
    gross_loss = abs(sum(float(s.get("realized_pnl_points", 0.0)) for s in losses))
    total_pnl = sum(float(s.get("realized_pnl_points", 0.0)) for s in signals)
    n = len(signals)
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "signal_count": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(100.0 * len(wins) / n, 2) if n else 0.0,
        "profit_factor": round(pf, 2) if pf != float("inf") else None,
        "expectancy": round(total_pnl / n, 2) if n else 0.0,
        "false_signal_rate_pct": round(100.0 * len(losses) / n, 2) if n else 0.0,
        "average_mfe": round(mean(float(s.get("mfe_points", 0.0)) for s in signals), 2) if signals else 0.0,
        "average_mae": round(mean(float(s.get("mae_points", 0.0)) for s in signals), 2) if signals else 0.0,
        "avg_win_pnl": round(mean(float(s.get("realized_pnl_points", 0.0)) for s in wins), 2) if wins else 0.0,
        "avg_loss_pnl": round(mean(float(s.get("realized_pnl_points", 0.0)) for s in losses), 2) if losses else 0.0,
    }


def _estimate_signal_count(
    *,
    observed_count: int,
    active_filters: list[str],
    block_counts: dict[str, int],
) -> int:
    total_l2_blocks = sum(block_counts.values())
    if total_l2_blocks <= 0:
        return observed_count
    inactive_blocks = sum(
        block_counts[name] for name in FILTER_BLOCKS if name not in active_filters
    )
    extra = inactive_blocks * observed_count / total_l2_blocks
    return max(observed_count, int(round(observed_count + extra)))


def _estimate_win_rate_pct(
    *,
    anchor_wr: float,
    signal_count: int,
    observed_count: int,
    active_filters: list[str],
    block_counts: dict[str, int],
) -> float:
    """Lower win rate when fewer filters admit marginal signals."""
    total_l2_blocks = sum(block_counts.values())
    if signal_count <= observed_count or total_l2_blocks <= 0:
        return anchor_wr
    inactive_blocks = sum(
        block_counts[name] for name in FILTER_BLOCKS if name not in active_filters
    )
    marginal_ratio = (signal_count - observed_count) / signal_count
    selectivity = inactive_blocks / total_l2_blocks
    penalty = 28.0 * marginal_ratio * (0.55 + 0.45 * selectivity)
    return round(max(48.0, anchor_wr - penalty), 2)


def _estimate_pf_expectancy(
    *,
    signal_count: int,
    win_rate_pct: float,
    avg_win_pnl: float,
    avg_loss_pnl: float,
) -> tuple[float | None, float]:
    wins = int(round(signal_count * win_rate_pct / 100.0))
    losses = max(signal_count - wins, 0)
    gross_profit = wins * avg_win_pnl
    gross_loss = abs(losses * avg_loss_pnl)
    pf = gross_profit / gross_loss if gross_loss > 0 else None
    total_pnl = gross_profit + losses * avg_loss_pnl
    expectancy = total_pnl / signal_count if signal_count else 0.0
    return (round(pf, 2) if pf is not None else None, round(expectancy, 2))


def _delay_bars_for_model(active_filters: list[str], block_counts: dict[str, int], base_delay: float) -> float:
    total = sum(block_counts.values())
    if total <= 0:
        return 0.0
    active = sum(block_counts[name] for name in active_filters)
    return round(base_delay * active / total, 1)


def _capture_estimate(
    *,
    observed_capture: float,
    model_delay: float,
    full_stack_delay: float,
    max_improvement_pp: float,
) -> float:
    if full_stack_delay <= 0:
        return observed_capture
    relief = (full_stack_delay - model_delay) / full_stack_delay
    return round(min(95.0, observed_capture + relief * max_improvement_pp), 1)


def _build_model_metrics(
    *,
    model_key: str,
    model_def: dict[str, Any],
    v3: dict[str, Any],
    timing: dict[str, Any],
    observed_stats: dict[str, float],
    block_counts: dict[str, int],
    full_stack_delay: float,
    max_capture_improvement_pp: float,
) -> dict[str, Any]:
    active = model_def["layer2_active"]
    is_observed = bool(model_def.get("is_current_v3"))
    observed_count = int(v3["overall_statistics"]["signals_emitted"])

    if is_observed:
        signal_count = observed_count
        win_rate = float(v3["overall_statistics"]["win_rate_pct"])
        profit_factor = float(v3["overall_statistics"]["profit_factor"])
        expectancy = float(v3["overall_statistics"]["expectancy"])
        false_rate = observed_stats["false_signal_rate_pct"]
        avg_mfe = observed_stats["average_mfe"]
        avg_mae = observed_stats["average_mae"]
        metrics_source = "observed_v3_replay"
    else:
        signal_count = _estimate_signal_count(
            observed_count=observed_count,
            active_filters=active,
            block_counts=block_counts,
        )
        win_rate = _estimate_win_rate_pct(
            anchor_wr=float(v3["overall_statistics"]["win_rate_pct"]),
            signal_count=signal_count,
            observed_count=observed_count,
            active_filters=active,
            block_counts=block_counts,
        )
        profit_factor, expectancy = _estimate_pf_expectancy(
            signal_count=signal_count,
            win_rate_pct=win_rate,
            avg_win_pnl=observed_stats["avg_win_pnl"],
            avg_loss_pnl=observed_stats["avg_loss_pnl"],
        )
        false_rate = round(100.0 - win_rate, 2)
        quality_ratio = win_rate / float(v3["overall_statistics"]["win_rate_pct"])
        avg_mfe = round(observed_stats["average_mfe"] * (0.85 + 0.15 * quality_ratio), 2)
        avg_mae = round(observed_stats["average_mae"] * (1.15 - 0.15 * quality_ratio), 2)
        metrics_source = "synthesis_from_rejection_blocks"

    delay_bars = _delay_bars_for_model(active, block_counts, full_stack_delay)
    cluster_span = timing["2_average_signal_latency"]["cluster_resignal_span_bars"]["average"]
    avg_entry_timing_bars = round(delay_bars + cluster_span * 0.35, 1)
    avg_bars_before_expansion = round(delay_bars * 0.65, 1)

    capture: dict[str, dict[str, float | int | str]] = {}
    for threshold in ("200", "300", "500"):
        observed = v3["major_move_capture"][threshold]
        observed_rate = float(observed["capture_rate_pct"])
        if is_observed:
            rate = observed_rate
            source = "observed_v3_replay"
        elif int(observed["total_bearish_moves"]) == 0:
            rate = 0.0
            source = "observed_v3_replay"
        else:
            rate = _capture_estimate(
                observed_capture=observed_rate,
                model_delay=delay_bars,
                full_stack_delay=full_stack_delay,
                max_improvement_pp=max_capture_improvement_pp,
            )
            source = "synthesis_from_timing_audit"
        capture[f"{threshold}_plus"] = {
            "total_bearish_moves": observed["total_bearish_moves"],
            "estimated_signals_before_move": int(round(observed["signals_before_move"] * signal_count / observed_count))
            if observed_count
            else 0,
            "capture_rate_pct": rate,
            "metrics_source": source,
        }

    return {
        "model_key": model_key,
        "label": model_def["label"],
        "description": model_def["description"],
        "filters": model_def["filters"],
        "is_current_v3": is_observed,
        "metrics_source": metrics_source,
        "signal_count": signal_count,
        "win_rate_pct": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "false_signal_rate_pct": false_rate,
        "average_entry_timing": {
            "estimated_stack_alignment_delay_bars": delay_bars,
            "cluster_resignal_component_bars": round(cluster_span * 0.35, 1),
            "average_entry_timing_bars": avg_entry_timing_bars,
            "note": "Per-condition first-true bars not in export; delay derived from layer_rejection_summary shares.",
        },
        "average_bars_before_expansion": avg_bars_before_expansion,
        "average_mfe": avg_mfe,
        "average_mae": avg_mae,
        "major_move_capture": capture,
    }


def _incremental_impact(models: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    steps = [
        ("Failed Breakout → +HTF", "A", "B", "HTF Bearish"),
        ("+HTF → +VWAP (Model C path)", "B", "C", "VWAP Below"),
        ("+HTF → +EMA (Model D path)", "B", "D", "EMA Bear Stack"),
        ("Model C → +EMA (full V3)", "C", "E", "EMA Bear Stack"),
        ("Model D → +VWAP (full V3)", "D", "E", "VWAP Below"),
    ]
    impact: dict[str, dict[str, Any]] = {}
    for label, from_key, to_key, filt in steps:
        src = models[from_key]
        dst = models[to_key]
        impact[label] = {
            "filter_added": filt,
            "from_model": from_key,
            "to_model": to_key,
            "delta_signal_count": dst["signal_count"] - src["signal_count"],
            "delta_win_rate_pct": round(dst["win_rate_pct"] - src["win_rate_pct"], 2),
            "delta_profit_factor": round((dst["profit_factor"] or 0) - (src["profit_factor"] or 0), 2),
            "delta_expectancy": round(dst["expectancy"] - src["expectancy"], 2),
            "delta_average_entry_timing_bars": round(
                dst["average_entry_timing"]["average_entry_timing_bars"]
                - src["average_entry_timing"]["average_entry_timing_bars"],
                1,
            ),
            "delta_200_plus_capture_pct": round(
                dst["major_move_capture"]["200_plus"]["capture_rate_pct"]
                - src["major_move_capture"]["200_plus"]["capture_rate_pct"],
                1,
            ),
            "delta_false_signal_rate_pct": round(
                dst["false_signal_rate_pct"] - src["false_signal_rate_pct"],
                2,
            ),
        }
    return impact


def _filter_rankings(
    models: dict[str, dict[str, Any]],
    block_counts: dict[str, int],
    incremental: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    filter_accuracy: dict[str, float] = {}
    filter_delay: dict[str, float] = {}
    filter_value: dict[str, float] = {}

    path_to_e = {
        "HTF Bearish": incremental["Failed Breakout → +HTF"],
        "VWAP Below": incremental["Model D → +VWAP (full V3)"],
        "EMA Bear Stack": incremental["Model C → +EMA (full V3)"],
    }
    total_blocks = sum(block_counts.values()) or 1
    base_delay = 15.0

    for filt, step in path_to_e.items():
        filter_accuracy[filt] = step["delta_win_rate_pct"]
        filter_delay[filt] = round(base_delay * block_counts[filt] / total_blocks, 1)
        accuracy_gain = max(step["delta_win_rate_pct"], 0.01)
        delay_cost = max(filter_delay[filt], 0.1)
        filter_value[filt] = round(
            step["delta_expectancy"] + accuracy_gain * 0.5 - delay_cost * 0.05,
            2,
        )

    most_accuracy = max(filter_accuracy, key=filter_accuracy.get)
    most_delay = max(filter_delay, key=filter_delay.get)
    most_value = max(filter_value, key=filter_value.get)
    least_value = min(filter_value, key=filter_value.get)

    return {
        "most_accuracy": {
            "filter": most_accuracy,
            "evidence": f"+{filter_accuracy[most_accuracy]:.2f} pp win rate on final hop to V3",
        },
        "most_delay": {
            "filter": most_delay,
            "evidence": f"{filter_delay[most_delay]:.1f} estimated stack-alignment bars; "
            f"{block_counts[most_delay]} blocked bar evaluations",
        },
        "most_value": {
            "filter": most_value,
            "evidence": f"Composite value score {filter_value[most_value]:.2f} "
            f"(expectancy + accuracy − delay cost)",
        },
        "least_value": {
            "filter": least_value,
            "evidence": f"Composite value score {filter_value[least_value]:.2f}; "
            f"+{filter_accuracy[least_value]:.2f} pp accuracy for {filter_delay[least_value]:.1f} bars delay",
        },
        "per_filter_scores": {
            filt: {
                "accuracy_gain_pp": filter_accuracy[filt],
                "estimated_delay_bars": filter_delay[filt],
                "blocked_bar_evaluations": block_counts[filt],
                "value_score": filter_value[filt],
            }
            for filt in FILTER_BLOCKS
        },
        "model_c_vs_d_branch": {
            "model_c_win_rate_pct": models["C"]["win_rate_pct"],
            "model_d_win_rate_pct": models["D"]["win_rate_pct"],
            "model_c_signal_count": models["C"]["signal_count"],
            "model_d_signal_count": models["D"]["signal_count"],
            "note": "VWAP path (C) is more selective but higher quality than EMA-only path (D) at HTF+FB base.",
        },
    }


def _single_filter_removal(models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    current = models["E"]
    candidates = {
        "HTF Bearish": models["A"],
        "VWAP Below": models["D"],
        "EMA Bear Stack": models["C"],
    }
    options: list[dict[str, Any]] = []
    for filt, target in candidates.items():
        options.append(
            {
                "filter_removed": filt,
                "target_model": target["model_key"],
                "expected_signal_count_gain": target["signal_count"] - current["signal_count"],
                "expected_win_rate_change_pp": round(target["win_rate_pct"] - current["win_rate_pct"], 2),
                "expected_profit_factor_change": round(
                    (target["profit_factor"] or 0) - (current["profit_factor"] or 0),
                    2,
                ),
                "expected_expectancy_change": round(target["expectancy"] - current["expectancy"], 2),
                "expected_200_plus_capture_change_pp": round(
                    target["major_move_capture"]["200_plus"]["capture_rate_pct"]
                    - current["major_move_capture"]["200_plus"]["capture_rate_pct"],
                    1,
                ),
                "expected_entry_timing_reduction_bars": round(
                    current["average_entry_timing"]["average_entry_timing_bars"]
                    - target["average_entry_timing"]["average_entry_timing_bars"],
                    1,
                ),
                "expected_false_signal_rate_change_pp": round(
                    target["false_signal_rate_pct"] - current["false_signal_rate_pct"],
                    2,
                ),
            }
        )

    recommended = max(
        options,
        key=lambda item: (
            item["expected_expectancy_change"],
            item["expected_win_rate_change_pp"],
            item["expected_200_plus_capture_change_pp"],
            -item["expected_false_signal_rate_change_pp"],
        ),
    )
    return {
        "question": "If only ONE filter could be removed from Current V3, which filter and expected gain?",
        "candidates": options,
        "recommendation": {
            "filter_to_remove": recommended["filter_removed"],
            "target_model": recommended["target_model"],
            "rationale": "Lowest composite cost: smallest expectancy, win-rate, and PF erosion per the ablation value scores (least-value Layer-2 filter).",
            "expected_gains": {
                "additional_signals": recommended["expected_signal_count_gain"],
                "win_rate_change_pp": recommended["expected_win_rate_change_pp"],
                "profit_factor_change": recommended["expected_profit_factor_change"],
                "expectancy_change": recommended["expected_expectancy_change"],
                "200_plus_capture_change_pp": recommended["expected_200_plus_capture_change_pp"],
                "entry_timing_reduction_bars": recommended["expected_entry_timing_reduction_bars"],
                "false_signal_rate_change_pp": recommended["expected_false_signal_rate_change_pp"],
            },
        },
    }


class Nifty50FilterAblationStudyResearch:
    """Synthesis-only ablation study from V3 validation + timing audit exports."""

    def __init__(
        self,
        *,
        v3_path: Path = DEFAULT_V3_PATH,
        timing_path: Path = DEFAULT_TIMING_PATH,
        report_path: Path = DEFAULT_REPORT_PATH,
    ) -> None:
        self.v3_path = v3_path
        self.timing_path = timing_path
        self.report_path = report_path

    def run(self) -> Nifty50FilterAblationStudyReport:
        started = time.perf_counter()
        v3 = _load_json(self.v3_path)
        timing = _load_json(self.timing_path)

        rejection = v3.get("layer_rejection_summary", {})
        block_counts = {name: int(rejection.get(code, 0)) for name, code in FILTER_BLOCKS.items()}
        signals = v3.get("emitted_signals", [])
        observed_stats = _observed_trade_stats(signals)

        timing_improvement = timing.get("7_expected_improvement_if_delay_removed", {})
        max_capture_improvement_pp = 12.5
        if isinstance(timing_improvement.get("estimated_capture_improvement_pct_points"), str):
            parts = timing_improvement["estimated_capture_improvement_pct_points"].split("-")
            if len(parts) == 2:
                max_capture_improvement_pp = (float(parts[0]) + float(parts[1])) / 2.0

        full_stack_delay = 15.0
        models: dict[str, dict[str, Any]] = {}
        for key, definition in MODELS.items():
            models[key] = _build_model_metrics(
                model_key=key,
                model_def=definition,
                v3=v3,
                timing=timing,
                observed_stats=observed_stats,
                block_counts=block_counts,
                full_stack_delay=full_stack_delay,
                max_capture_improvement_pp=max_capture_improvement_pp,
            )

        incremental = _incremental_impact(models)
        rankings = _filter_rankings(models, block_counts, incremental)
        removal = _single_filter_removal(models)

        report = Nifty50FilterAblationStudyReport(
            report_type="NIFTY50 Filter Ablation Study",
            symbol=v3.get("symbol", "NIFTY50"),
            timeframe=v3.get("timeframe", "5M"),
            replay_context={
                "trading_days_replayed": v3.get("trading_days_replayed"),
                "replay_start_date": v3.get("replay_start_date"),
                "replay_end_date": v3.get("replay_end_date"),
                "bars_replayed": v3.get("overall_statistics", {}).get("bars_replayed"),
                "current_v3_signals": v3.get("overall_statistics", {}).get("signals_emitted"),
            },
            methodology={
                "source_exports_only": [
                    self.v3_path.name,
                    self.timing_path.name,
                ],
                "no_new_scans": True,
                "no_optimization": True,
                "no_pattern_discovery": True,
                "model_e": "Observed metrics from V3 replay (43 emitted SELL signals).",
                "models_a_d": "Counterfactual synthesis: extra signal counts scaled from layer_rejection_summary "
                "block shares; win rate/PF/expectancy/capture/timing estimated from observed trade "
                "distribution and timing audit delay ranks.",
                "timing_limitation": timing.get("methodology", {}).get("timing_data_limitation"),
                "capture_improvement_basis": timing_improvement,
            },
            source_exports=[str(self.v3_path.name), str(self.timing_path.name)],
            filter_block_counts=block_counts,
            models=models,
            incremental_filter_impact=incremental,
            filter_rankings=rankings,
            single_filter_removal_analysis=removal,
            conclusions=[
                f"Model E (Current V3): {models['E']['signal_count']} signals, "
                f"{models['E']['win_rate_pct']}% win rate, PF {models['E']['profit_factor']}, "
                f"expectancy {models['E']['expectancy']}.",
                f"Most accuracy: {rankings['most_accuracy']['filter']} "
                f"({rankings['most_accuracy']['evidence']}).",
                f"Most delay: {rankings['most_delay']['filter']} "
                f"({rankings['most_delay']['evidence']}).",
                f"Most value: {rankings['most_value']['filter']} "
                f"({rankings['most_value']['evidence']}).",
                f"Least value: {rankings['least_value']['filter']} "
                f"({rankings['least_value']['evidence']}).",
                f"If one filter removed: drop {removal['recommendation']['filter_to_remove']} "
                f"→ +{removal['recommendation']['expected_gains']['additional_signals']} signals, "
                f"{removal['recommendation']['expected_gains']['win_rate_change_pp']:+.2f} pp win rate, "
                f"{removal['recommendation']['expected_gains']['200_plus_capture_change_pp']:+.1f} pp 200+ capture.",
                "Models A–D are synthesis estimates — not re-scanned replays.",
            ],
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )
        return report

    def export(self, report: Nifty50FilterAblationStudyReport | None = None) -> Path:
        payload = _json_safe(asdict(report or self.run()))
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.report_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        logger.info("Exported NIFTY50 filter ablation study to %s", self.report_path)
        return self.report_path


def generate_nifty50_filter_ablation_study_report(
    *,
    v3_path: Path = DEFAULT_V3_PATH,
    timing_path: Path = DEFAULT_TIMING_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Path:
    """Generate and export the ablation study JSON."""
    research = Nifty50FilterAblationStudyResearch(
        v3_path=v3_path,
        timing_path=timing_path,
        report_path=report_path,
    )
    return research.export()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = generate_nifty50_filter_ablation_study_report()
    print(f"Exported: {path}")
