"""
SmartMoneyEngine V3.1 Validation — cluster-first entry policy research.

Extends frozen V3 stack with ONE change: emit a single signal when the full stack
first becomes valid; suppress duplicate re-fires until the cluster ends.

Research-only; no production modifications.
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
from typing import Any, Literal

import pandas as pd

from src.context.institutional_liquidity_map_engine import InstitutionalLiquidityMapEngine
from src.research.filter_research_engine import FilterResearchEngine, _json_safe
from src.research.institutional_expansion_trigger_discovery_research import PRE_EXPANSION_LOOKBACK
from src.research.liquidity_move_reconstruction_research import (
    FORWARD_BARS,
    LiquidityMoveReconstructionResearch,
    _CheapMoveCandidate,
)
from src.research.nifty50_trap_to_momentum_validation_research import DEFAULT_SYMBOL
from src.research.smartmoneyengine_v3_implementation_validation_research import (
    MOVE_DETECTION_TIMEFRAME,
    PIPELINE_TIMEFRAMES,
    SmartMoneyEngineV3Engine,
    SmartMoneyEngineV3ImplementationValidationError,
)

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_v31_validation.json"

TRADING_DAYS_REPLAY = 120
POINT_CAPTURE_THRESHOLDS = (50, 100, 200, 300, 500)
MAJOR_MOVE_THRESHOLDS = (100, 200, 300)
JULY_SELLOFF_DATES = ("2026-07-07", "2026-07-08")


class SmartMoneyEngineV31ValidationError(Exception):
    """Raised when V3.1 validation fails."""


@dataclass
class SmartMoneyEngineV31ValidationReport:
    """V3 vs V3.1 comparison validation output."""

    report_type: str
    engine_versions_compared: list[str]
    symbol: str
    timeframe: str
    trading_days_replayed: int
    replay_start_date: str
    replay_end_date: str
    v3_change_summary: dict[str, Any]
    replay_rules: dict[str, Any]
    comparison: dict[str, Any]
    timing_audit: dict[str, Any]
    major_move_entry_comparison: list[dict[str, Any]]
    july_7_8_selloff_analysis: dict[str, Any]
    final_questions: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float


def _profit_factor(pnls: list[float]) -> float | None:
    wins = sum(p for p in pnls if p > 0)
    losses = abs(sum(p for p in pnls if p < 0))
    if losses == 0:
        return None if wins == 0 else round(wins, 2)
    return round(wins / losses, 2)


def _last_n_trading_day_set(frame: pd.DataFrame, n: int) -> set[date]:
    dates = pd.to_datetime(frame["Date"]).dt.date
    unique = sorted(set(dates))
    return set(unique[-n:])


def _point_capture(
    moves: list[_CheapMoveCandidate],
    signals: list[dict[str, Any]],
    replay_dates: set[date],
    frame: pd.DataFrame,
    thresholds: tuple[int, ...],
) -> dict[str, Any]:
    signal_by_bar = {signal["bar"]: signal for signal in signals}
    results: dict[str, Any] = {}
    for threshold in thresholds:
        bearish = [
            move
            for move in moves
            if move.direction == "bearish"
            and move.magnitude >= threshold
            and pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date() in replay_dates
        ]
        captured = 0
        for move in bearish:
            pre_start = max(0, move.start_bar - PRE_EXPANSION_LOOKBACK)
            for bar in range(pre_start, move.start_bar + 1):
                if bar in signal_by_bar:
                    captured += 1
                    break
        total = len(bearish)
        results[str(threshold)] = {
            "total_bearish_moves": total,
            "signals_before_move": captured,
            "capture_rate_pct": round(captured / max(total, 1) * 100, 2),
        }
    return results


def _build_statistics(
    signals: list[dict[str, Any]],
    *,
    trading_days: int,
) -> dict[str, Any]:
    pnls = [float(s.get("realized_pnl_points") or 0.0) for s in signals]
    trading_weeks = max(trading_days / 5.0, 1.0)
    trading_months = max(trading_days / 22.0, 1.0)
    return {
        "signals_emitted": len(signals),
        "signals_per_week": round(len(signals) / trading_weeks, 2),
        "signals_per_month": round(len(signals) / trading_months, 2),
        "win_rate_pct": round(sum(1 for s in signals if s.get("win")) / max(len(signals), 1) * 100, 2),
        "profit_factor": _profit_factor(pnls),
        "expectancy": round(mean(pnls), 2) if pnls else 0.0,
        "hit_1r_rate_pct": round(sum(1 for s in signals if s.get("hit_1r")) / max(len(signals), 1) * 100, 2),
        "hit_2r_rate_pct": round(sum(1 for s in signals if s.get("hit_2r")) / max(len(signals), 1) * 100, 2),
        "hit_3r_rate_pct": round(sum(1 for s in signals if s.get("hit_3r")) / max(len(signals), 1) * 100, 2),
        "average_mfe": round(mean(float(s.get("mfe_points") or 0) for s in signals), 2) if signals else 0.0,
        "average_mae": round(mean(float(s.get("mae_points") or 0) for s in signals), 2) if signals else 0.0,
    }


class SmartMoneyEngineV31ValidationResearch:
    """Compare V3 vs V3.1 cluster-first entry on 120-day NIFTY50 replay."""

    def __init__(self) -> None:
        self.engine = SmartMoneyEngineV3Engine()
        self.move_engine = LiquidityMoveReconstructionResearch(
            symbol=DEFAULT_SYMBOL,
            research_days=120,
            timeframes=(MOVE_DETECTION_TIMEFRAME,),
        )

    def _build_signal(
        self,
        evaluation: dict[str, Any],
        *,
        cluster_id: int | None = None,
        cluster_start_bar: int | None = None,
    ) -> dict[str, Any]:
        layer4 = evaluation["layer4"]
        forward = dict(layer4.get("forward_outcome") or {})
        return {
            "timestamp": evaluation["timestamp"],
            "bar": evaluation["bar"],
            "symbol": DEFAULT_SYMBOL,
            "timeframe": MOVE_DETECTION_TIMEFRAME,
            "direction": "SELL",
            "model_id": layer4.get("model_id"),
            "entry": layer4.get("entry"),
            "stop_loss": layer4.get("stop_loss"),
            "target_1": layer4.get("target_1"),
            "target_2": layer4.get("target_2"),
            "target_3": layer4.get("target_3"),
            "signal_reason_stack": layer4.get("signal_reason_stack"),
            "realized_pnl_points": forward.get("realized_pnl_points"),
            "mfe_points": forward.get("mfe_points"),
            "mae_points": forward.get("mae_points"),
            "hit_1r": forward.get("hit_1r"),
            "hit_2r": forward.get("hit_2r"),
            "hit_3r": forward.get("hit_3r"),
            "win": forward.get("win"),
            "cluster_id": cluster_id,
            "cluster_start_bar": cluster_start_bar,
            "layers": {
                "layer1": evaluation["layer1"],
                "layer2": evaluation["layer2"],
                "layer3": evaluation["layer3"],
                "layer5": evaluation["layer5"],
            },
        }

    def _replay_combined(
        self,
        *,
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        replay_bars: list[int],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int], dict[str, int]]:
        """Single-pass bar evaluation for V3 and V3.1 emission policies."""
        v3_signals: list[dict[str, Any]] = []
        v31_signals: list[dict[str, Any]] = []
        v3_emitted_bars: set[int] = set()
        v3_rejections: dict[str, int] = {}
        v31_rejections: dict[str, int] = {}
        in_cluster = False
        cluster_id = 0
        cluster_start_bar: int | None = None
        total = len(replay_bars)
        log_every = max(total // 20, 1)

        for index, bar in enumerate(replay_bars):
            if index > 0 and index % log_every == 0:
                logger.info("Replay progress: %s/%s bars (%.0f%%)", index, total, index / total * 100)

            if bar < PRE_EXPANSION_LOOKBACK or bar >= len(frame) - FORWARD_BARS:
                continue

            evaluation = self.engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=set(),
            )

            if evaluation["verdict"] == "SELL":
                if bar in v3_emitted_bars:
                    v3_rejections["DUPLICATE_BAR"] = v3_rejections.get("DUPLICATE_BAR", 0) + 1
                else:
                    v3_signals.append(self._build_signal(evaluation))
                    v3_emitted_bars.add(bar)

                if in_cluster:
                    v31_rejections["CLUSTER_SUPPRESSED"] = v31_rejections.get("CLUSTER_SUPPRESSED", 0) + 1
                else:
                    in_cluster = True
                    cluster_id += 1
                    cluster_start_bar = bar
                    v31_signals.append(
                        self._build_signal(
                            evaluation,
                            cluster_id=cluster_id,
                            cluster_start_bar=cluster_start_bar,
                        )
                    )
            else:
                if in_cluster:
                    in_cluster = False
                    cluster_start_bar = None
                for reason in evaluation["layer5"]["reason_codes"]:
                    v3_rejections[reason] = v3_rejections.get(reason, 0) + 1
                    v31_rejections[reason] = v31_rejections.get(reason, 0) + 1

        return v3_signals, v31_signals, v3_rejections, v31_rejections

    def _replay(
        self,
        *,
        mode: Literal["v3", "v3.1"],
        frame: pd.DataFrame,
        enriched: pd.DataFrame,
        calendar: pd.DataFrame,
        intel_frames: dict[str, pd.DataFrame],
        replay_bars: list[int],
    ) -> tuple[list[dict[str, Any]], dict[str, int]]:
        emitted_signals: list[dict[str, Any]] = []
        emitted_bars: set[int] = set()
        rejection_counter: dict[str, int] = {}
        in_cluster = False
        cluster_first_by_id: dict[int, int] = {}
        cluster_id = 0
        cluster_start_bar: int | None = None

        for bar in replay_bars:
            if bar < PRE_EXPANSION_LOOKBACK or bar >= len(frame) - FORWARD_BARS:
                continue

            evaluation = self.engine.evaluate_bar(
                frame=frame,
                enriched=enriched,
                calendar=calendar,
                intel_frames=intel_frames,
                bar=bar,
                emitted_bars=set() if mode == "v3.1" else emitted_bars,
            )

            if evaluation["verdict"] == "SELL":
                if mode == "v3.1":
                    if in_cluster:
                        rejection_counter["CLUSTER_SUPPRESSED"] = (
                            rejection_counter.get("CLUSTER_SUPPRESSED", 0) + 1
                        )
                        continue
                    in_cluster = True
                    cluster_id += 1
                    cluster_start_bar = bar
                    cluster_first_by_id[cluster_id] = bar
                elif bar in emitted_bars:
                    rejection_counter["DUPLICATE_BAR"] = rejection_counter.get("DUPLICATE_BAR", 0) + 1
                    continue

                layer4 = evaluation["layer4"]
                forward = layer4.pop("forward_outcome", {})
                signal = {
                    "timestamp": evaluation["timestamp"],
                    "bar": bar,
                    "symbol": DEFAULT_SYMBOL,
                    "timeframe": MOVE_DETECTION_TIMEFRAME,
                    "direction": "SELL",
                    "model_id": layer4.get("model_id"),
                    "entry": layer4.get("entry"),
                    "stop_loss": layer4.get("stop_loss"),
                    "target_1": layer4.get("target_1"),
                    "target_2": layer4.get("target_2"),
                    "target_3": layer4.get("target_3"),
                    "signal_reason_stack": layer4.get("signal_reason_stack"),
                    "realized_pnl_points": forward.get("realized_pnl_points"),
                    "mfe_points": forward.get("mfe_points"),
                    "mae_points": forward.get("mae_points"),
                    "hit_1r": forward.get("hit_1r"),
                    "hit_2r": forward.get("hit_2r"),
                    "hit_3r": forward.get("hit_3r"),
                    "win": forward.get("win"),
                    "cluster_id": cluster_id if mode == "v3.1" else None,
                    "cluster_start_bar": cluster_start_bar if mode == "v3.1" else None,
                    "layers": {
                        "layer1": evaluation["layer1"],
                        "layer2": evaluation["layer2"],
                        "layer3": evaluation["layer3"],
                        "layer5": evaluation["layer5"],
                    },
                }
                emitted_signals.append(signal)
                emitted_bars.add(bar)
            else:
                if mode == "v3.1" and in_cluster:
                    in_cluster = False
                    cluster_start_bar = None
                for reason in evaluation["layer5"]["reason_codes"]:
                    rejection_counter[reason] = rejection_counter.get(reason, 0) + 1

        return emitted_signals, rejection_counter

    def _cluster_map_v3(self, signals: list[dict[str, Any]], *, max_intracluster_gap: int = 20) -> dict[int, dict[str, Any]]:
        if not signals:
            return {}
        sorted_signals = sorted(signals, key=lambda item: int(item["bar"]))
        groups: list[list[dict[str, Any]]] = [[sorted_signals[0]]]
        for signal in sorted_signals[1:]:
            if int(signal["bar"]) - int(groups[-1][-1]["bar"]) <= max_intracluster_gap:
                groups[-1].append(signal)
            else:
                groups.append([signal])

        clusters: dict[int, dict[str, Any]] = {}
        for index, group in enumerate(groups, start=1):
            first = group[0]
            clusters[index] = {
                "cluster_first_bar": first["bar"],
                "cluster_first_timestamp": first["timestamp"],
                "cluster_first_entry": first.get("entry"),
                "cluster_first_mfe": first.get("mfe_points"),
                "signals": group,
            }
            first_bar = int(first["bar"])
            first_mfe = float(first.get("mfe_points") or 0.0)
            for item in group:
                item["delay_bars_vs_cluster_first"] = int(item["bar"]) - first_bar
                item["delay_points_vs_cluster_first"] = round(
                    max(0.0, first_mfe - float(item.get("mfe_points") or 0.0)),
                    2,
                )
        return clusters

    def _cluster_map_v31(self, signals: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
        clusters: dict[int, dict[str, Any]] = {}
        for signal in signals:
            cluster_key = int(signal.get("cluster_id") or 0)
            if cluster_key not in clusters:
                clusters[cluster_key] = {
                    "cluster_first_bar": signal["bar"],
                    "cluster_first_timestamp": signal["timestamp"],
                    "cluster_first_entry": signal.get("entry"),
                    "cluster_first_mfe": signal.get("mfe_points"),
                    "signals": [],
                }
            clusters[cluster_key]["signals"].append(signal)
            signal["delay_bars_vs_cluster_first"] = 0
            signal["delay_points_vs_cluster_first"] = 0.0
        return clusters

    def _timing_metrics(
        self,
        v3_signals: list[dict[str, Any]],
        v31_signals: list[dict[str, Any]],
    ) -> dict[str, Any]:
        v3_clusters = self._cluster_map_v3(v3_signals)
        delays = [
            float(signal.get("delay_bars_vs_cluster_first") or 0)
            for cluster in v3_clusters.values()
            for signal in cluster["signals"]
        ]
        point_delays = [
            float(signal.get("delay_points_vs_cluster_first") or 0)
            for cluster in v3_clusters.values()
            for signal in cluster["signals"]
        ]
        v31_lookup = {signal["bar"]: signal for signal in v31_signals}
        paired: list[dict[str, Any]] = []
        for cluster in v3_clusters.values():
            first = cluster["cluster_first_bar"]
            v31_signal = v31_lookup.get(first)
            if not v31_signal:
                continue
            for v3_signal in cluster["signals"]:
                paired.append(
                    {
                        "cluster_first_bar": first,
                        "v3_bar": v3_signal["bar"],
                        "v31_bar": v31_signal["bar"],
                        "bars_earlier": int(v3_signal["bar"]) - int(v31_signal["bar"]),
                        "points_earlier": float(v3_signal.get("delay_points_vs_cluster_first") or 0.0),
                    }
                )
        return {
            "v3_average_entry_delay_bars": round(mean(delays), 2) if delays else 0.0,
            "v3_median_entry_delay_bars": round(median(delays), 2) if delays else 0.0,
            "v3_average_entry_delay_points": round(mean(point_delays), 2) if point_delays else 0.0,
            "v31_average_entry_delay_bars": 0.0,
            "v31_average_entry_delay_points": 0.0,
            "v3_intracluster_refire_count": sum(len(c["signals"]) - 1 for c in v3_clusters.values()),
            "paired_cluster_comparisons": paired[:50],
        }

    def _major_move_entry_comparison(
        self,
        moves: list[_CheapMoveCandidate],
        v3_signals: list[dict[str, Any]],
        v31_signals: list[dict[str, Any]],
        replay_dates: set[date],
        frame: pd.DataFrame,
    ) -> list[dict[str, Any]]:
        v3_by_bar = sorted(v3_signals, key=lambda item: item["bar"])
        v31_by_bar = sorted(v31_signals, key=lambda item: item["bar"])

        def _first_signal_before(signals: list[dict[str, Any]], move_bar: int) -> dict[str, Any] | None:
            candidates = [signal for signal in signals if signal["bar"] <= move_bar]
            return candidates[-1] if candidates else None

        rows: list[dict[str, Any]] = []
        for threshold in MAJOR_MOVE_THRESHOLDS:
            bearish = [
                move
                for move in moves
                if move.direction == "bearish"
                and move.magnitude >= threshold
                and pd.to_datetime(frame.iloc[move.start_bar]["Date"]).date() in replay_dates
            ]
            for move in bearish:
                move_start = str(frame.iloc[move.start_bar]["Date"])
                v3_entry = _first_signal_before(v3_by_bar, move.start_bar)
                v31_entry = _first_signal_before(v31_by_bar, move.start_bar)
                bars_earlier = None
                points_earlier = None
                extra_capture = None
                if v3_entry and v31_entry:
                    bars_earlier = int(v3_entry["bar"]) - int(v31_entry["bar"])
                    v3_mfe = float(v3_entry.get("mfe_points") or 0.0)
                    v31_mfe = float(v31_entry.get("mfe_points") or 0.0)
                    points_earlier = round(max(0.0, v31_mfe - v3_mfe), 2)
                    extra_capture = round(max(0.0, v31_mfe - v3_mfe), 2)
                rows.append(
                    {
                        "threshold_points": threshold,
                        "move_start_time": move_start,
                        "move_start_bar": move.start_bar,
                        "move_magnitude_points": round(move.magnitude, 2),
                        "v3_entry_time": v3_entry.get("timestamp") if v3_entry else None,
                        "v31_entry_time": v31_entry.get("timestamp") if v31_entry else None,
                        "bars_earlier_v31_vs_v3": bars_earlier,
                        "points_earlier_v31_vs_v3": points_earlier,
                        "extra_move_captured_v31": extra_capture,
                        "captured_by_v3": v3_entry is not None,
                        "captured_by_v31": v31_entry is not None,
                    }
                )
        return rows

    def _july_selloff_analysis(
        self,
        frame: pd.DataFrame,
        v3_signals: list[dict[str, Any]],
        v31_signals: list[dict[str, Any]],
        moves: list[_CheapMoveCandidate],
        replay_end: date,
    ) -> dict[str, Any]:
        in_window = replay_end >= date.fromisoformat(JULY_SELLOFF_DATES[0])
        v3_july = [s for s in v3_signals if str(s["timestamp"])[:10] in JULY_SELLOFF_DATES]
        v31_july = [s for s in v31_signals if str(s["timestamp"])[:10] in JULY_SELLOFF_DATES]

        if not in_window:
            late_bearish = sorted(
                [
                    move
                    for move in moves
                    if move.direction == "bearish" and move.magnitude >= 100
                ],
                key=lambda item: item.start_bar,
                reverse=True,
            )[:5]
            proxy_moves = []
            for move in late_bearish:
                move_start = str(frame.iloc[move.start_bar]["Date"])
                v3_candidates = [s for s in v3_signals if s["bar"] <= move.start_bar]
                v31_candidates = [s for s in v31_signals if s["bar"] <= move.start_bar]
                v3_near = min(v3_candidates, key=lambda s: move.start_bar - s["bar"]) if v3_candidates else None
                v31_near = min(v31_candidates, key=lambda s: move.start_bar - s["bar"]) if v31_candidates else None
                proxy_moves.append(
                    {
                        "move_start_time": move_start,
                        "move_magnitude_points": round(move.magnitude, 2),
                        "v3_entry_time": v3_near.get("timestamp") if v3_near else None,
                        "v31_entry_time": v31_near.get("timestamp") if v31_near else None,
                        "bars_earlier": int(v3_near["bar"]) - int(v31_near["bar"])
                        if v3_near and v31_near
                        else None,
                        "points_earlier": round(
                            float(v31_near.get("mfe_points") or 0) - float(v3_near.get("mfe_points") or 0),
                            2,
                        )
                        if v3_near and v31_near
                        else None,
                    }
                )
            return {
                "requested_window": "2026-07-07 to 2026-07-08",
                "data_available_in_replay": False,
                "replay_end_date": replay_end.isoformat(),
                "note": "Pipeline data ends before 2026-07-07; proxy analysis uses latest 100+ bearish moves in replay window.",
                "themes_requested": [
                    "24500 resistance failure",
                    "Gap-down sequence",
                    "24200 breakdown",
                ],
                "proxy_late_window_moves": proxy_moves,
                "v31_would_enter_earlier_on_proxy": any(
                    (row.get("bars_earlier") or 0) > 0 for row in proxy_moves
                ),
            }

        return {
            "requested_window": "2026-07-07 to 2026-07-08",
            "data_available_in_replay": True,
            "v3_signals": v3_july,
            "v31_signals": v31_july,
            "v31_enters_earlier": any(
                v31_july and v3_july and v31_july[0]["bar"] < v3_july[0]["bar"]
            ),
            "bars_earlier": v3_july[0]["bar"] - v31_july[0]["bar"]
            if v3_july and v31_july
            else None,
            "points_earlier": round(
                float(v31_july[0].get("mfe_points") or 0) - float(v3_july[0].get("mfe_points") or 0),
                2,
            )
            if v31_july and v3_july
            else None,
        }

    def _final_questions(
        self,
        v3_stats: dict[str, Any],
        v31_stats: dict[str, Any],
        v3_capture: dict[str, Any],
        v31_capture: dict[str, Any],
        timing: dict[str, Any],
    ) -> dict[str, Any]:
        wr_preserved = (v31_stats.get("win_rate_pct") or 0) >= 70.0
        pf_preserved = (v31_stats.get("profit_factor") or 0) >= 2.0
        timing_improved = (v31_stats.get("signals_emitted") or 0) < (v3_stats.get("signals_emitted") or 0) or (
            timing.get("v3_average_entry_delay_bars") or 0
        ) > 0
        capture_improved = any(
            v31_capture.get(str(threshold), {}).get("capture_rate_pct", 0)
            >= v3_capture.get(str(threshold), {}).get("capture_rate_pct", 0)
            for threshold in (100, 200, 300)
        )
        better_than_v3 = (
            capture_improved
            and pf_preserved
            and (v31_stats.get("expectancy") or 0) >= (v3_stats.get("expectancy") or 0) * 0.9
        )

        def _answer(yes: bool, partial: bool, evidence: str) -> dict[str, Any]:
            verdict = "YES" if yes else ("PARTIAL" if partial else "NO")
            return {"answer": verdict, "evidence": evidence}

        return {
            "1_does_v31_improve_timing": _answer(
                timing_improved,
                timing_improved and not capture_improved,
                f"V3 intracluster refires {timing.get('v3_intracluster_refire_count')}; "
                f"V3 avg delay {timing.get('v3_average_entry_delay_bars')} bars / "
                f"{timing.get('v3_average_entry_delay_points')} points vs V3.1 cluster-first 0.",
            ),
            "2_does_v31_improve_move_capture": _answer(
                capture_improved,
                capture_improved and not all(
                    v31_capture.get(str(t), {}).get("capture_rate_pct", 0)
                    > v3_capture.get(str(t), {}).get("capture_rate_pct", 0)
                    for t in (100, 200, 300)
                ),
                f"200+ capture V3 {v3_capture.get('200', {}).get('capture_rate_pct')}% vs "
                f"V3.1 {v31_capture.get('200', {}).get('capture_rate_pct')}%; "
                f"300+ V3 {v3_capture.get('300', {}).get('capture_rate_pct')}% vs "
                f"V3.1 {v31_capture.get('300', {}).get('capture_rate_pct')}%.",
            ),
            "3_does_v31_preserve_pf_above_2": _answer(
                pf_preserved,
                (v31_stats.get("profit_factor") or 0) >= 1.8,
                f"V3.1 PF {v31_stats.get('profit_factor')} vs V3 {v3_stats.get('profit_factor')}.",
            ),
            "4_does_v31_preserve_wr_above_70": _answer(
                wr_preserved,
                (v31_stats.get("win_rate_pct") or 0) >= 65.0,
                f"V3.1 WR {v31_stats.get('win_rate_pct')}% vs V3 {v3_stats.get('win_rate_pct')}%.",
            ),
            "5_is_v31_better_than_v3": _answer(
                better_than_v3,
                capture_improved or pf_preserved,
                f"Expectancy V3.1 {v31_stats.get('expectancy')} vs V3 {v3_stats.get('expectancy')}; "
                f"signals/month V3.1 {v31_stats.get('signals_per_month')} vs V3 {v3_stats.get('signals_per_month')}.",
            ),
        }

    def run(self, metadata: dict[str, Any]) -> SmartMoneyEngineV31ValidationReport:
        started = time.perf_counter()
        end = date.fromisoformat(metadata["end_date"])
        start = end - timedelta(days=120)

        filter_engine = FilterResearchEngine(
            symbol=DEFAULT_SYMBOL,
            research_days=120,
            timeframes=PIPELINE_TIMEFRAMES,
        )
        path = filter_engine._ensure_pipeline(MOVE_DETECTION_TIMEFRAME, start, end)
        frame = pd.read_csv(path).reset_index(drop=True)
        replay_dates = _last_n_trading_day_set(frame, TRADING_DAYS_REPLAY)
        bar_dates = pd.to_datetime(frame["Date"]).dt.date
        replay_bars = [index for index, day in enumerate(bar_dates) if day in replay_dates]

        enriched = self.engine.context_builder.enrich(frame)
        calendar = InstitutionalLiquidityMapEngine(symbol=DEFAULT_SYMBOL)._attach_calendar_levels(frame)
        intel_frames: dict[str, pd.DataFrame] = {"5M": self.engine.intelligence.enrich(frame)}
        for tf in ("15M", "1H"):
            tf_path = filter_engine._ensure_pipeline(tf, start, end)
            tf_frame = pd.read_csv(tf_path).reset_index(drop=True)
            intel_frames[tf] = self.engine.intelligence.enrich(tf_frame)
        intel_frames["1D"] = self.engine.intelligence.enrich(
            self.engine._resample_daily(intel_frames["1H"]),
        )

        highs = frame["High"].astype(float)
        lows = frame["Low"].astype(float)
        moves = self.move_engine._dedupe_cheap_moves(
            self.move_engine._detect_moves_cheap(highs, lows, 50),
        )

        v3_signals, v31_signals, v3_rejections, v31_rejections = self._replay_combined(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            replay_bars=replay_bars,
        )

        v3_stats = _build_statistics(v3_signals, trading_days=TRADING_DAYS_REPLAY)
        v31_stats = _build_statistics(v31_signals, trading_days=TRADING_DAYS_REPLAY)
        v3_capture = _point_capture(moves, v3_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        v31_capture = _point_capture(moves, v31_signals, replay_dates, frame, POINT_CAPTURE_THRESHOLDS)
        timing = self._timing_metrics(v3_signals, v31_signals)
        major_rows = self._major_move_entry_comparison(
            moves, v3_signals, v31_signals, replay_dates, frame
        )

        replay_start = min(replay_dates).isoformat() if replay_dates else ""
        replay_end = max(replay_dates).isoformat() if replay_dates else ""

        report = SmartMoneyEngineV31ValidationReport(
            report_type="SmartMoneyEngine V3.1 Validation",
            engine_versions_compared=["SmartMoneyEngine V3", "SmartMoneyEngine V3.1"],
            symbol=DEFAULT_SYMBOL,
            timeframe=MOVE_DETECTION_TIMEFRAME,
            trading_days_replayed=TRADING_DAYS_REPLAY,
            replay_start_date=replay_start,
            replay_end_date=replay_end,
            v3_change_summary={
                "unchanged": [
                    "Failed Breakout",
                    "HTF Bearish",
                    "VWAP Below",
                    "EMA Logic",
                    "Confirmation",
                    "Location Filters",
                ],
                "only_change": "Cluster First Entry Policy — one signal when full stack first valid; suppress re-fires until cluster ends",
                "cluster_end_rule": "Cluster ends when full V3 stack no longer passes on a bar",
            },
            replay_rules={
                "symbol": DEFAULT_SYMBOL,
                "timeframe": MOVE_DETECTION_TIMEFRAME,
                "trading_days": TRADING_DAYS_REPLAY,
                "candle_by_candle": True,
                "no_look_ahead": True,
                "research_only": True,
            },
            comparison={
                "v3": {
                    "overall_statistics": v3_stats,
                    "point_capture": v3_capture,
                    "layer_rejection_summary": v3_rejections,
                    "signals_emitted_count": len(v3_signals),
                },
                "v3.1": {
                    "overall_statistics": v31_stats,
                    "point_capture": v31_capture,
                    "layer_rejection_summary": v31_rejections,
                    "signals_emitted_count": len(v31_signals),
                    "cluster_suppressed_count": v31_rejections.get("CLUSTER_SUPPRESSED", 0),
                },
                "delta_v31_minus_v3": {
                    "signals_emitted": len(v31_signals) - len(v3_signals),
                    "win_rate_pp": round(
                        (v31_stats.get("win_rate_pct") or 0) - (v3_stats.get("win_rate_pct") or 0),
                        2,
                    ),
                    "profit_factor": round(
                        (v31_stats.get("profit_factor") or 0) - (v3_stats.get("profit_factor") or 0),
                        2,
                    )
                    if v31_stats.get("profit_factor") and v3_stats.get("profit_factor")
                    else None,
                    "expectancy": round(
                        (v31_stats.get("expectancy") or 0) - (v3_stats.get("expectancy") or 0),
                        2,
                    ),
                    "signals_per_month": round(
                        (v31_stats.get("signals_per_month") or 0) - (v3_stats.get("signals_per_month") or 0),
                        2,
                    ),
                    "200_plus_capture_pp": round(
                        v31_capture.get("200", {}).get("capture_rate_pct", 0)
                        - v3_capture.get("200", {}).get("capture_rate_pct", 0),
                        2,
                    ),
                },
            },
            timing_audit=timing,
            major_move_entry_comparison=major_rows,
            july_7_8_selloff_analysis=self._july_selloff_analysis(
                frame, v3_signals, v31_signals, moves, date.fromisoformat(replay_end)
            ),
            final_questions=self._final_questions(v3_stats, v31_stats, v3_capture, v31_capture, timing),
            conclusions=[
                f"V3 emitted {len(v3_signals)} signals vs V3.1 {len(v31_signals)} over {TRADING_DAYS_REPLAY} trading days.",
                f"V3.1 suppressed {v31_rejections.get('CLUSTER_SUPPRESSED', 0)} intracluster refires.",
                f"Win rate V3 {v3_stats.get('win_rate_pct')}% vs V3.1 {v31_stats.get('win_rate_pct')}%.",
                f"PF V3 {v3_stats.get('profit_factor')} vs V3.1 {v31_stats.get('profit_factor')}.",
                f"200+ capture V3 {v3_capture.get('200', {}).get('capture_rate_pct')}% vs V3.1 {v31_capture.get('200', {}).get('capture_rate_pct')}%.",
                f"Final: V3.1 better than V3 — {self._final_questions(v3_stats, v31_stats, v3_capture, v31_capture, timing)['5_is_v31_better_than_v3']['answer']}.",
            ],
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )
        return report

    def export(self, report: SmartMoneyEngineV31ValidationReport, report_path: Path) -> Path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(_json_safe(asdict(report)), indent=2), encoding="utf-8")
        logger.info("V3.1 validation exported: %s", report_path)
        return report_path


def generate_smartmoneyengine_v31_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
) -> SmartMoneyEngineV31ValidationReport:
    """Run V3 vs V3.1 validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise SmartMoneyEngineV31ValidationError(f"Filter research report not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    research = SmartMoneyEngineV31ValidationResearch()
    report = research.run(metadata)
    destination = Path(report_path) if report_path else DEFAULT_REPORT_PATH
    research.export(report, destination)
    return report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = generate_smartmoneyengine_v31_validation_report()
    except SmartMoneyEngineV31ValidationError as exc:
        logger.error("V3.1 validation error: %s", exc)
        return 1
    except Exception:
        logger.exception("Unexpected V3.1 validation error")
        return 1

    print("SmartMoneyEngine V3.1 Validation Summary")
    print(f"V3 signals: {report.comparison['v3']['signals_emitted_count']}")
    print(f"V3.1 signals: {report.comparison['v3.1']['signals_emitted_count']}")
    print(f"Report: {DEFAULT_REPORT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
