"""
SmartMoneyEngine Production Candidate research.

Synthesizes completed research exports to find the smallest feature combinations
that produce the highest-quality BUY and SELL signals. Research-only.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from itertools import combinations
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.liquidity_move_reconstruction_research import FORWARD_BARS
from src.research.major_level_strength_research import MajorLevelStrengthResearch
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tier2_winner_loser_comparison_research import (
    ComparativeTradeRecord,
    Tier2WinnerLoserComparisonResearch,
)
from src.research.tiered_signal_framework_research import TierSignal
from src.research.trade_construction_validation_research import (
    TradeConstructionValidationResearch,
)
from src.signals.decision_engine import DecisionEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_production_candidate.json"

MIN_SAMPLES = 50
MIN_PROFIT_FACTOR = 1.5
MIN_EXPECTANCY = 50.0
TOP_MODEL_COUNT = 20
MAX_COMBO_SIZE = 4
LOOKBACK_BARS = 50

RESEARCH_EXPORTS = (
    "tier2_production_validation.json",
    "institutional_move_dna.json",
    "liquidity_sweep_tradeability.json",
    "institutional_trigger_validation.json",
    "support_resistance_pressure.json",
    "major_level_strength.json",
    "institutional_confirmation_candle.json",
    "tier2_winner_loser_comparison.json",
    "tier2_regime_classification.json",
    "trigger_trade_validation.json",
    "trigger_entry_optimization.json",
    "tier2_composite_edge_validation.json",
    "trade_construction_validation.json",
    "tier2_exit_optimization.json",
)

FEATURE_DEFINITIONS: dict[str, str] = {
    "rsi_below_40": "RSI < 40",
    "rsi_above_60": "RSI > 60",
    "rsi_divergence": "RSI Divergence Present",
    "near_support": "Near Support",
    "near_resistance": "Near Resistance",
    "discount_zone": "Discount Zone",
    "premium_zone": "Premium Zone",
    "liquidity_sweep": "Liquidity Sweep (50-bar)",
    "false_breakout": "False Breakout (50-bar)",
    "false_breakdown": "False Breakdown (50-bar)",
    "choch_present": "CHOCH Present",
    "bos_present": "BOS Present",
    "fvg_reclaim": "FVG Reclaim",
    "order_block_reaction": "Order Block Reaction",
    "strong_confirmation": "Strong Confirmation Candle",
    "ema_bull_stack": "EMA20 > EMA50 > EMA200",
    "ema_bear_stack": "EMA20 < EMA50 < EMA200",
    "above_vwap": "Above VWAP",
    "below_vwap": "Below VWAP",
    "round_number": "Round Number Proximity",
    "level_strong": "Level Strength: Strong",
    "level_moderate": "Level Strength: Moderate",
    "session_morning": "Session: Morning",
    "session_midday": "Session: Midday",
    "session_afternoon": "Session: Afternoon",
    "gap_up": "Gap Up",
    "gap_down": "Gap Down",
    "htf_aligned": "HTF Trend Aligned",
    "strong_displacement": "Strong Displacement",
    "regime_trend_continuation": "Regime: Trend Continuation",
    "regime_liquidity_reversal": "Regime: Liquidity Reversal",
}


class ProductionCandidateError(Exception):
    """Raised when production candidate research fails."""


@dataclass(frozen=True)
class ProductionCandidateTrade:
    """One historical Tier-2 move with full feature matrix and trade metrics."""

    bos_timestamp: str
    timeframe: str
    direction: str
    signal_side: str
    risk_points: float
    realized_pnl_points: float
    realized_rr: float
    win: bool
    hit_1r_before_sl: bool
    hit_2r_before_sl: bool
    hit_3r_before_sl: bool
    feature_flags: dict[str, bool]
    feature_tags: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateModelMetrics:
    """Metrics for one feature combination model."""

    model_key: str
    model_label: str
    signal_side: str
    feature_count: int
    features: list[str]
    trades: int
    trades_per_month: float
    win_rate_pct: float
    hit_1r_rate_pct: float
    hit_2r_rate_pct: float
    hit_3r_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    net_points: float
    maximum_drawdown_points: float
    production_score: float
    rank_1r: int = 0
    rank_2r: int = 0
    rank_3r: int = 0
    rank_expectancy: int = 0
    rank_pf: int = 0
    overall_rank: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionCandidateReport:
    """Full production candidate research output."""

    symbol: str
    research_window_days: int
    start_date: str
    end_date: str
    research_exports_reviewed: list[str]
    minimum_sample_size: int
    rejection_criteria: dict[str, float]
    total_historical_moves: int
    buy_moves: int
    sell_moves: int
    feature_definitions: dict[str, str]
    baseline_buy: dict[str, Any]
    baseline_sell: dict[str, Any]
    eligible_models: list[dict[str, Any]]
    rejected_models: list[dict[str, Any]]
    top_20_buy_models: list[dict[str, Any]]
    top_20_sell_models: list[dict[str, Any]]
    rankings_by_metric: dict[str, list[str]]
    recommended_production_signal_engine: dict[str, Any]
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineProductionCandidateResearch:
    """Find smallest high-quality BUY/SELL feature combinations."""

    def __init__(
        self,
        symbol: str = "NIFTY50",
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
    ) -> None:
        self.symbol = symbol
        self.research_days = research_days
        self.timeframes = timeframes
        self.comparison_engine = Tier2WinnerLoserComparisonResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.trade_engine = TradeConstructionValidationResearch(
            symbol=symbol,
            research_days=research_days,
            timeframes=timeframes,
        )
        self.level_engine = MajorLevelStrengthResearch(research_days=research_days)

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _load_research_export(name: str) -> dict[str, Any]:
        path = RESEARCH_DIR / name
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _simulate_r_hits(
        self,
        frame: pd.DataFrame,
        signal: TierSignal,
    ) -> dict[str, Any]:
        entry_bar = signal.bos_bar
        if entry_bar >= len(frame) - 1:
            return {}

        direction = signal.direction
        entry_price = round(float(frame.iloc[entry_bar]["Close"]), 2)
        stop, risk = self.trade_engine._structural_stop(
            frame,
            entry_bar,
            entry_price,
            direction,
        )
        target = self.trade_engine._opposite_liquidity_target(
            frame,
            entry_bar,
            entry_price,
            direction,
            risk,
        )

        end = min(len(frame) - 1, entry_bar + FORWARD_BARS)
        hit_1r = hit_2r = hit_3r = False
        stop_hit = target_hit = False
        pnl = 0.0
        rr = 0.0

        for index in range(entry_bar + 1, end + 1):
            bar_high = float(frame.iloc[index]["High"])
            bar_low = float(frame.iloc[index]["Low"])

            if direction == "bullish":
                if not stop_hit:
                    if bar_high >= entry_price + risk:
                        hit_1r = True
                    if bar_high >= entry_price + risk * 2:
                        hit_2r = True
                    if bar_high >= entry_price + risk * 3:
                        hit_3r = True
                if bar_low <= stop:
                    stop_hit = True
                    pnl = -risk
                    rr = -1.0
                    break
                if bar_high >= target:
                    target_hit = True
                    pnl = round(target - entry_price, 2)
                    rr = round(pnl / risk, 2) if risk > 0 else 0.0
                    break
            else:
                if not stop_hit:
                    if bar_low <= entry_price - risk:
                        hit_1r = True
                    if bar_low <= entry_price - risk * 2:
                        hit_2r = True
                    if bar_low <= entry_price - risk * 3:
                        hit_3r = True
                if bar_high >= stop:
                    stop_hit = True
                    pnl = -risk
                    rr = -1.0
                    break
                if bar_low <= target:
                    target_hit = True
                    pnl = round(entry_price - target, 2)
                    rr = round(pnl / risk, 2) if risk > 0 else 0.0
                    break

        if not stop_hit and not target_hit:
            close = float(frame.iloc[end]["Close"])
            if direction == "bullish":
                pnl = round(close - entry_price, 2)
            else:
                pnl = round(entry_price - close, 2)
            rr = round(pnl / risk, 2) if risk > 0 else 0.0

        return {
            "risk_points": risk,
            "realized_pnl_points": pnl,
            "realized_rr": rr,
            "win": pnl > 0,
            "hit_1r_before_sl": hit_1r,
            "hit_2r_before_sl": hit_2r,
            "hit_3r_before_sl": hit_3r,
        }

    def _extended_flags(
        self,
        frame: pd.DataFrame,
        filter_frame: pd.DataFrame,
        signal: TierSignal,
        comparative: ComparativeTradeRecord,
    ) -> dict[str, bool]:
        bar = signal.bos_bar
        direction = signal.direction
        filters = self.comparison_engine.filter_context.filter_state(filter_frame, bar)
        start = max(0, bar - LOOKBACK_BARS)
        window = frame.iloc[start : bar + 1]
        row = frame.iloc[bar]

        liquidity_sweep = any(
            self._is_active(window.iloc[index].get("Buy_Liquidity_Sweep"))
            or self._is_active(window.iloc[index].get("Sell_Liquidity_Sweep"))
            for index in range(len(window))
        )

        close = float(row["Close"])
        supports = [
            float(window.iloc[index]["Low"])
            for index in range(len(window))
        ]
        resistances = [
            float(window.iloc[index]["High"])
            for index in range(len(window))
        ]
        prior_high = max(resistances[:-1]) if len(resistances) > 1 else close
        prior_low = min(supports[:-1]) if len(supports) > 1 else close
        false_breakout = false_breakdown = False
        for index in range(start + 1, bar + 1):
            high = float(frame.iloc[index]["High"])
            low = float(frame.iloc[index]["Low"])
            c = float(frame.iloc[index]["Close"])
            if high > prior_high and c < prior_high:
                false_breakout = True
            if low < prior_low and c > prior_low:
                false_breakdown = True

        choch = self._is_active(row.get("Bullish_CHOCH")) or self._is_active(
            row.get("Bearish_CHOCH"),
        )
        bos = self._is_active(row.get("Bullish_BOS")) or self._is_active(row.get("Bearish_BOS"))
        ob = self._is_active(row.get("Bullish_OB_High")) or self._is_active(
            row.get("Bearish_OB_High"),
        )

        open_price = float(row["Open"])
        body = abs(close - open_price)
        candle_range = max(float(row["High"]) - float(row["Low"]), 0.01)
        strong_confirmation = (body / candle_range) >= 0.55

        gap_up = gap_down = False
        if bar >= 1:
            prev_close = float(frame.iloc[bar - 1]["Close"])
            gap = open_price - prev_close
            gap_up = gap > 0.5
            gap_down = gap < -0.5

        round_number = self.level_engine._round_number_overlap(close)

        return {
            "rsi_below_40": comparative.rsi < 40,
            "rsi_above_60": comparative.rsi > 60,
            "rsi_divergence": comparative.rsi_divergence != "No RSI Divergence",
            "near_support": comparative.market_location == "Near Support",
            "near_resistance": comparative.market_location == "Near Resistance",
            "discount_zone": comparative.rsi < 45 and comparative.market_location == "Near Support",
            "premium_zone": comparative.rsi > 55 and comparative.market_location == "Near Resistance",
            "liquidity_sweep": liquidity_sweep,
            "false_breakout": false_breakout,
            "false_breakdown": false_breakdown,
            "choch_present": choch,
            "bos_present": bos,
            "fvg_reclaim": comparative.fvg_retest_count > 0,
            "order_block_reaction": ob,
            "strong_confirmation": strong_confirmation,
            "ema_bull_stack": filters.ema_alignment == "EMA20 > EMA50 > EMA200",
            "ema_bear_stack": filters.ema_alignment == "EMA20 < EMA50 < EMA200",
            "above_vwap": filters.vwap_position == "Above VWAP",
            "below_vwap": filters.vwap_position == "Below VWAP",
            "round_number": round_number,
            "level_strong": comparative.intelligence_score >= 65,
            "level_moderate": 50 <= comparative.intelligence_score < 65,
            "session_morning": comparative.session == "Morning",
            "session_midday": comparative.session == "Midday",
            "session_afternoon": comparative.session == "Afternoon",
            "gap_up": gap_up,
            "gap_down": gap_down,
            "htf_aligned": comparative.regime in {"Trend Continuation", "Liquidity Reversal"},
            "strong_displacement": comparative.displacement_strength == "Strong",
            "regime_trend_continuation": comparative.regime == "Trend Continuation",
            "regime_liquidity_reversal": comparative.regime == "Liquidity Reversal",
        }

    def _collect_candidates(self, metadata: dict[str, Any]) -> list[ProductionCandidateTrade]:
        end = date.fromisoformat(metadata["end_date"])
        start = (
            date.fromisoformat(metadata["start_date"])
            if metadata.get("start_date")
            else end - timedelta(days=self.research_days)
        )

        comparative_records = self.comparison_engine._collect_records(metadata)
        lookup = {(record.timeframe, record.bos_timestamp): record for record in comparative_records}
        candidates: list[ProductionCandidateTrade] = []

        for timeframe_label in self.timeframes:
            path = self.comparison_engine.tier_engine.filter_engine._ensure_pipeline(
                timeframe_label,
                start,
                end,
            )
            frame = pd.read_csv(path).reset_index(drop=True)
            filter_frame = self.comparison_engine.filter_context.enrich(frame)

            for signal in self.comparison_engine.tier_engine._detect_tier2(frame, timeframe_label):
                key = (timeframe_label, signal.bos_timestamp)
                comparative = lookup.get(key)
                if comparative is None:
                    continue

                simulation = self._simulate_r_hits(frame, signal)
                if not simulation:
                    continue

                flags = self._extended_flags(frame, filter_frame, signal, comparative)
                tags = tuple(
                    FEATURE_DEFINITIONS[name]
                    for name, active in sorted(flags.items())
                    if active
                )
                side = "BUY" if signal.direction == "bullish" else "SELL"
                candidates.append(
                    ProductionCandidateTrade(
                        bos_timestamp=signal.bos_timestamp,
                        timeframe=timeframe_label,
                        direction=signal.direction,
                        signal_side=side,
                        risk_points=simulation["risk_points"],
                        realized_pnl_points=simulation["realized_pnl_points"],
                        realized_rr=simulation["realized_rr"],
                        win=simulation["win"],
                        hit_1r_before_sl=simulation["hit_1r_before_sl"],
                        hit_2r_before_sl=simulation["hit_2r_before_sl"],
                        hit_3r_before_sl=simulation["hit_3r_before_sl"],
                        feature_flags=flags,
                        feature_tags=tags,
                    ),
                )

        return candidates

    @staticmethod
    def _matches(trade: ProductionCandidateTrade, feature_keys: tuple[str, ...]) -> bool:
        return all(trade.feature_flags.get(key, False) for key in feature_keys)

    def _aggregate_metrics(
        self,
        feature_keys: tuple[str, ...],
        trades: list[ProductionCandidateTrade],
        research_days: int,
    ) -> CandidateModelMetrics:
        side = trades[0].signal_side if trades else "BUY"
        if feature_keys:
            label = " + ".join(FEATURE_DEFINITIONS[key] for key in feature_keys)
            key = "+".join(feature_keys)
        else:
            label = f"Baseline {side} (Unfiltered Tier-2)"
            key = f"baseline_{side.lower()}"

        pnls = [trade.realized_pnl_points for trade in trades]
        rrs = [trade.realized_rr for trade in trades]
        wins = sum(1 for trade in trades if trade.win)
        total = len(trades)
        pf = self._profit_factor(pnls)
        exp = round(mean(pnls), 2) if pnls else 0.0
        months = max(research_days / 30.44, 1.0)

        metrics = CandidateModelMetrics(
            model_key=key,
            model_label=label,
            signal_side=side,
            feature_count=len(feature_keys),
            features=[FEATURE_DEFINITIONS[k] for k in feature_keys],
            trades=total,
            trades_per_month=round(total / months, 2),
            win_rate_pct=round(wins / total * 100, 2) if total else 0.0,
            hit_1r_rate_pct=round(
                sum(1 for trade in trades if trade.hit_1r_before_sl) / total * 100,
                2,
            )
            if total
            else 0.0,
            hit_2r_rate_pct=round(
                sum(1 for trade in trades if trade.hit_2r_before_sl) / total * 100,
                2,
            )
            if total
            else 0.0,
            hit_3r_rate_pct=round(
                sum(1 for trade in trades if trade.hit_3r_before_sl) / total * 100,
                2,
            )
            if total
            else 0.0,
            profit_factor=pf,
            expectancy=exp,
            average_rr=round(mean(rrs), 2) if rrs else 0.0,
            net_points=round(sum(pnls), 2),
            maximum_drawdown_points=self._maximum_drawdown(pnls),
            production_score=0.0,
        )
        pf_val = pf if pf is not None else 0.0
        if pf_val == float("inf"):
            pf_val = 10.0
        metrics.production_score = round(
            metrics.hit_3r_rate_pct * 0.25
            + metrics.hit_2r_rate_pct * 0.20
            + metrics.hit_1r_rate_pct * 0.15
            + metrics.expectancy * 0.25
            + pf_val * 10.0,
            4,
        )
        return metrics

    def _evaluate_combinations(
        self,
        trades: list[ProductionCandidateTrade],
        research_days: int,
    ) -> tuple[list[CandidateModelMetrics], list[CandidateModelMetrics]]:
        if not trades:
            return [], []

        active_features = [
            key
            for key in FEATURE_DEFINITIONS
            if sum(1 for trade in trades if trade.feature_flags.get(key, False)) >= MIN_SAMPLES
        ]

        eligible: list[CandidateModelMetrics] = []
        rejected: list[CandidateModelMetrics] = []

        for size in range(1, min(MAX_COMBO_SIZE, len(active_features)) + 1):
            for combo in combinations(active_features, size):
                bucket = [trade for trade in trades if self._matches(trade, combo)]
                if len(bucket) < MIN_SAMPLES:
                    continue
                metrics = self._aggregate_metrics(combo, bucket, research_days)
                pf = metrics.profit_factor or 0.0
                if pf < MIN_PROFIT_FACTOR or metrics.expectancy < MIN_EXPECTANCY:
                    rejected.append(metrics)
                else:
                    eligible.append(metrics)

        return eligible, rejected

    @staticmethod
    def _rank_models(models: list[CandidateModelMetrics]) -> list[CandidateModelMetrics]:
        if not models:
            return []

        by_1r = sorted(models, key=lambda item: item.hit_1r_rate_pct, reverse=True)
        by_2r = sorted(models, key=lambda item: item.hit_2r_rate_pct, reverse=True)
        by_3r = sorted(models, key=lambda item: item.hit_3r_rate_pct, reverse=True)
        by_exp = sorted(models, key=lambda item: item.expectancy, reverse=True)
        by_pf = sorted(
            models,
            key=lambda item: item.profit_factor or 0.0,
            reverse=True,
        )
        overall = sorted(models, key=lambda item: item.production_score, reverse=True)

        for index, item in enumerate(by_1r, start=1):
            item.rank_1r = index
        for index, item in enumerate(by_2r, start=1):
            item.rank_2r = index
        for index, item in enumerate(by_3r, start=1):
            item.rank_3r = index
        for index, item in enumerate(by_exp, start=1):
            item.rank_expectancy = index
        for index, item in enumerate(by_pf, start=1):
            item.rank_pf = index
        for index, item in enumerate(overall, start=1):
            item.overall_rank = index

        return overall[:TOP_MODEL_COUNT]

    def _recommended_engine(
        self,
        best_buy: CandidateModelMetrics | None,
        best_sell: CandidateModelMetrics | None,
    ) -> dict[str, Any]:
        trade_export = self._load_research_export("trade_construction_validation.json")
        exit_export = self._load_research_export("tier2_exit_optimization.json")
        production = trade_export.get("production_recommendation", {})
        exit_rec = exit_export.get("recommended_exit_model", "E")

        return {
            "tier_sequence": ["Displacement", "CHOCH", "BOS", "FVG Reclaim"],
            "entry": production.get("entry", "BOS Close"),
            "stop_loss": production.get("stop_loss", "Structural Swing SL"),
            "t1": "1R partial (50%)",
            "t2": "2R partial (33%)",
            "t3": production.get("target", "Opposite Liquidity Pool"),
            "exit_model": exit_rec,
            "exit_label": exit_export.get("exit_models", {}).get(exit_rec, "Trail after 1R"),
            "buy_filter_stack": best_buy.features if best_buy else [],
            "sell_filter_stack": best_sell.features if best_sell else [],
            "buy_model_key": best_buy.model_key if best_buy else None,
            "sell_model_key": best_sell.model_key if best_sell else None,
            "baseline_construction": production,
        }

    def run(self, metadata: dict[str, Any]) -> ProductionCandidateReport:
        started = time.perf_counter()
        research_days = metadata.get("research_window_days", self.research_days)

        candidates = self._collect_candidates(metadata)
        buy_trades = [trade for trade in candidates if trade.signal_side == "BUY"]
        sell_trades = [trade for trade in candidates if trade.signal_side == "SELL"]

        buy_eligible, buy_rejected = self._evaluate_combinations(buy_trades, research_days)
        sell_eligible, sell_rejected = self._evaluate_combinations(sell_trades, research_days)

        top_buy = self._rank_models(buy_eligible)
        top_sell = self._rank_models(sell_eligible)

        baseline_buy = self._aggregate_metrics((), buy_trades, research_days) if buy_trades else None
        baseline_sell = self._aggregate_metrics((), sell_trades, research_days) if sell_trades else None

        recommended = self._recommended_engine(
            top_buy[0] if top_buy else None,
            top_sell[0] if top_sell else None,
        )

        exports_reviewed = [name for name in RESEARCH_EXPORTS if (RESEARCH_DIR / name).exists()]

        conclusions = [
            f"Evaluated {len(candidates)} historical Tier-2 moves with full feature matrix.",
            f"Eligible BUY models: {len(buy_eligible)} | Eligible SELL models: {len(sell_eligible)}.",
            (
                f"Top BUY model: {top_buy[0].model_label[:80]} "
                f"(1R={top_buy[0].hit_1r_rate_pct}%, Exp={top_buy[0].expectancy}, n={top_buy[0].trades})"
                if top_buy
                else "No BUY models passed rejection criteria."
            ),
            (
                f"Top SELL model: {top_sell[0].model_label[:80]} "
                f"(1R={top_sell[0].hit_1r_rate_pct}%, Exp={top_sell[0].expectancy}, n={top_sell[0].trades})"
                if top_sell
                else "No SELL models passed rejection criteria."
            ),
            f"Recommended engine: {recommended['entry']} + {recommended['stop_loss']} + Exit {recommended['exit_model']}.",
        ]

        return ProductionCandidateReport(
            symbol=self.symbol,
            research_window_days=research_days,
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            research_exports_reviewed=exports_reviewed,
            minimum_sample_size=MIN_SAMPLES,
            rejection_criteria={
                "min_samples": MIN_SAMPLES,
                "min_profit_factor": MIN_PROFIT_FACTOR,
                "min_expectancy": MIN_EXPECTANCY,
            },
            total_historical_moves=len(candidates),
            buy_moves=len(buy_trades),
            sell_moves=len(sell_trades),
            feature_definitions=dict(FEATURE_DEFINITIONS),
            baseline_buy=baseline_buy.as_dict() if baseline_buy else {},
            baseline_sell=baseline_sell.as_dict() if baseline_sell else {},
            eligible_models=[item.as_dict() for item in buy_eligible + sell_eligible],
            rejected_models=[item.as_dict() for item in buy_rejected + sell_rejected],
            top_20_buy_models=[item.as_dict() for item in top_buy],
            top_20_sell_models=[item.as_dict() for item in top_sell],
            rankings_by_metric={
                "hit_1r_rate": [item.model_key for item in sorted(buy_eligible + sell_eligible, key=lambda x: x.hit_1r_rate_pct, reverse=True)[:20]],
                "hit_2r_rate": [item.model_key for item in sorted(buy_eligible + sell_eligible, key=lambda x: x.hit_2r_rate_pct, reverse=True)[:20]],
                "hit_3r_rate": [item.model_key for item in sorted(buy_eligible + sell_eligible, key=lambda x: x.hit_3r_rate_pct, reverse=True)[:20]],
                "expectancy": [item.model_key for item in sorted(buy_eligible + sell_eligible, key=lambda x: x.expectancy, reverse=True)[:20]],
                "profit_factor": [item.model_key for item in sorted(buy_eligible + sell_eligible, key=lambda x: x.profit_factor or 0, reverse=True)[:20]],
            },
            recommended_production_signal_engine=recommended,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 3),
        )


def generate_production_candidate_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
) -> ProductionCandidateReport:
    """Run production candidate research and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise ProductionCandidateError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SmartMoneyEngineProductionCandidateResearch(symbol=symbol)
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Production candidate research completed: moves=%s buy_models=%s sell_models=%s",
        report.total_historical_moves,
        len(report.top_20_buy_models),
        len(report.top_20_sell_models),
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_production_candidate_report()
        print("SmartMoneyEngine Production Candidate Research Summary")
        print(f"Historical moves: {report.total_historical_moves}")
        print(f"Eligible BUY models: {len([m for m in report.eligible_models if m.get('signal_side') == 'BUY'])}")
        print(f"Eligible SELL models: {len([m for m in report.eligible_models if m.get('signal_side') == 'SELL'])}")
        if report.top_20_buy_models:
            top = report.top_20_buy_models[0]
            print(
                f"Top BUY: {top['model_label'][:90]} "
                f"(1R={top['hit_1r_rate_pct']}%, Exp={top['expectancy']})",
            )
        if report.top_20_sell_models:
            top = report.top_20_sell_models[0]
            print(
                f"Top SELL: {top['model_label'][:90]} "
                f"(1R={top['hit_1r_rate_pct']}%, Exp={top['expectancy']})",
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except ProductionCandidateError as exc:
        logger.error("Production candidate error: %s", exc)
        print(f"Production candidate error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected production candidate error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
