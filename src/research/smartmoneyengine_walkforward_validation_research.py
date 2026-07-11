"""
SmartMoneyEngine Walk Forward Validation research.

Validates the frozen V1 Production Card on a true 70/30 temporal split.
Research-only; no production modifications.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from src.research.filter_research_engine import RESEARCH_DAYS, _json_safe
from src.research.institutional_quality_validation_research import (
    InstitutionalQualityValidationResearch,
)
from src.research.smartmoneyengine_production_candidate_research import (
    FEATURE_DEFINITIONS,
    SmartMoneyEngineProductionCandidateResearch,
)
from src.research.tier2_production_validation_research import Tier2ProductionValidationResearch
from src.research.tier2_winner_loser_comparison_research import Tier2WinnerLoserComparisonResearch

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_DIR = PROJECT_ROOT / "outputs" / "research"
DEFAULT_FILTER_REPORT_PATH = RESEARCH_DIR / "filter_research_report.json"
DEFAULT_PRODUCTION_CARD_PATH = RESEARCH_DIR / "smartmoneyengine_final_production_validation.json"
DEFAULT_REPORT_PATH = RESEARCH_DIR / "smartmoneyengine_walkforward_validation.json"

TRAIN_FRACTION = 0.70
TEST_FRACTION = 0.30
MIN_OOS_SAMPLES = 10
PREFERRED_EXPECTANCY = 50.0
MANDATORY_CORE = ("Displacement", "CHOCH", "BOS", "FVG Reclaim")

FILTER_LABEL_TO_KEY: dict[str, str] = {
    label: key for key, label in FEATURE_DEFINITIONS.items()
}


class WalkForwardValidationError(Exception):
    """Raised when walk-forward validation fails."""


@dataclass(frozen=True)
class WalkForwardSignal:
    """One V1-filtered Tier-2 signal with walk-forward metadata."""

    symbol: str
    bos_timestamp: str
    timeframe: str
    signal_side: str
    direction: str
    risk_points: float
    realized_pnl_points: float
    realized_rr: float
    win: bool
    hit_1r: bool
    hit_2r: bool
    hit_3r: bool
    trait_tags: tuple[str, ...]
    blocked_by_no_trade: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WalkForwardMetrics:
    """Aggregate metrics for one walk-forward slice."""

    scope: str
    signal_side: str | None
    sample_size: int
    signals_per_month: float
    win_rate_pct: float
    profit_factor: float | None
    expectancy: float
    average_rr: float
    maximum_drawdown_points: float
    hit_1r_rate_pct: float
    hit_2r_rate_pct: float
    hit_3r_rate_pct: float
    net_points: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WalkForwardValidationReport:
    """Full walk-forward validation output."""

    symbols_analyzed: list[str]
    research_window_days: int
    start_date: str
    end_date: str
    train_fraction: float
    test_fraction: float
    train_start_date: str
    train_end_date: str
    test_start_date: str
    test_end_date: str
    frozen_v1_production_card: dict[str, Any]
    mandatory_signal_core: list[str]
    buy_rules: dict[str, Any]
    sell_rules: dict[str, Any]
    no_trade_rules: list[str]
    total_tier2_signals: int
    total_v1_signals: int
    no_trade_blocked_signals: int
    in_sample_metrics: dict[str, Any]
    out_of_sample_metrics: dict[str, Any]
    in_sample_buy: dict[str, Any]
    in_sample_sell: dict[str, Any]
    out_of_sample_buy: dict[str, Any]
    out_of_sample_sell: dict[str, Any]
    performance_degradation: dict[str, Any]
    survives_unseen_market_data: bool
    survival_verdict: str
    conclusions: list[str]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SmartMoneyEngineWalkForwardValidationResearch:
    """Validate frozen V1 production card on temporal train/test split."""

    def __init__(
        self,
        symbols: tuple[str, ...] | None = None,
        research_days: int = RESEARCH_DAYS,
        timeframes: tuple[str, ...] = ("5M", "15M", "1H"),
        production_card_path: Path | str | None = None,
    ) -> None:
        self.symbols = symbols or ("NIFTY50", "BANKNIFTY", "FINNIFTY")
        self.research_days = research_days
        self.timeframes = timeframes
        self.production_card_path = Path(production_card_path or DEFAULT_PRODUCTION_CARD_PATH)

    @staticmethod
    def _profit_factor(pnls: list[float]) -> float | None:
        return InstitutionalQualityValidationResearch._profit_factor(pnls)

    @staticmethod
    def _maximum_drawdown(pnls: list[float]) -> float:
        return Tier2ProductionValidationResearch._maximum_drawdown(pnls)

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        return pd.to_datetime(value).to_pydatetime()

    def _load_production_card(self) -> dict[str, Any]:
        if not self.production_card_path.exists():
            raise WalkForwardValidationError(
                f"Frozen production card not found: {self.production_card_path}",
            )
        with self.production_card_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        card = payload.get("smartmoneyengine_v1_final_production_card")
        if not card:
            raise WalkForwardValidationError("V1 production card missing from final validation export.")
        return card

    @staticmethod
    def _feature_keys_from_labels(labels: list[str]) -> tuple[str, ...]:
        keys: list[str] = []
        for label in labels:
            key = FILTER_LABEL_TO_KEY.get(label)
            if key is None:
                raise WalkForwardValidationError(f"Unknown filter label in V1 card: {label}")
            keys.append(key)
        return tuple(keys)

    @staticmethod
    def _no_trade_blocked(trait_tags: tuple[str, ...], rules: list[str]) -> bool:
        for rule in rules:
            if rule in trait_tags:
                return True
            if any(rule in tag for tag in trait_tags):
                return True
        return False

    def _collect_symbol_signals(
        self,
        symbol: str,
        metadata: dict[str, Any],
        card: dict[str, Any],
    ) -> tuple[list[WalkForwardSignal], int]:
        buy_keys = self._feature_keys_from_labels(card["buy_rules"]["filter_stack"])
        sell_keys = self._feature_keys_from_labels(card["sell_rules"]["filter_stack"])
        no_trade_rules = list(card.get("no_trade_rules", []))

        candidate_engine = SmartMoneyEngineProductionCandidateResearch(
            symbol=symbol,
            research_days=self.research_days,
            timeframes=self.timeframes,
        )
        comparison_engine = Tier2WinnerLoserComparisonResearch(
            symbol=symbol,
            research_days=self.research_days,
            timeframes=self.timeframes,
        )
        comparative_records = comparison_engine._collect_records(metadata)
        trait_lookup = {
            (record.timeframe, record.bos_timestamp): record.trait_tags
            for record in comparative_records
        }

        tier2_total = 0
        signals: list[WalkForwardSignal] = []
        for trade in candidate_engine._collect_candidates(metadata):
            tier2_total += 1
            trait_tags = trait_lookup.get((trade.timeframe, trade.bos_timestamp), ())
            blocked = self._no_trade_blocked(trait_tags, no_trade_rules)
            if blocked:
                signals.append(
                    WalkForwardSignal(
                        symbol=symbol,
                        bos_timestamp=trade.bos_timestamp,
                        timeframe=trade.timeframe,
                        signal_side=trade.signal_side,
                        direction=trade.direction,
                        risk_points=trade.risk_points,
                        realized_pnl_points=trade.realized_pnl_points,
                        realized_rr=trade.realized_rr,
                        win=trade.win,
                        hit_1r=trade.hit_1r_before_sl,
                        hit_2r=trade.hit_2r_before_sl,
                        hit_3r=trade.hit_3r_before_sl,
                        trait_tags=trait_tags,
                        blocked_by_no_trade=True,
                    ),
                )
                continue

            if trade.signal_side == "BUY":
                if not SmartMoneyEngineProductionCandidateResearch._matches(trade, buy_keys):
                    continue
            elif trade.signal_side == "SELL":
                if not SmartMoneyEngineProductionCandidateResearch._matches(trade, sell_keys):
                    continue
            else:
                continue

            signals.append(
                WalkForwardSignal(
                    symbol=symbol,
                    bos_timestamp=trade.bos_timestamp,
                    timeframe=trade.timeframe,
                    signal_side=trade.signal_side,
                    direction=trade.direction,
                    risk_points=trade.risk_points,
                    realized_pnl_points=trade.realized_pnl_points,
                    realized_rr=trade.realized_rr,
                    win=trade.win,
                    hit_1r=trade.hit_1r_before_sl,
                    hit_2r=trade.hit_2r_before_sl,
                    hit_3r=trade.hit_3r_before_sl,
                    trait_tags=trait_tags,
                    blocked_by_no_trade=False,
                ),
            )
        return signals, tier2_total

    def _split_dates(self, metadata: dict[str, Any]) -> tuple[date, date, date, date]:
        start = date.fromisoformat(metadata["start_date"])
        end = date.fromisoformat(metadata["end_date"])
        total_days = max((end - start).days, 1)
        train_days = int(total_days * TRAIN_FRACTION)
        train_end = start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=1)
        return start, train_end, test_start, end

    @staticmethod
    def _in_period(timestamp: str, period_start: date, period_end: date) -> bool:
        trade_date = pd.to_datetime(timestamp).date()
        return period_start <= trade_date <= period_end

    def _aggregate(
        self,
        signals: list[WalkForwardSignal],
        scope: str,
        signal_side: str | None,
        period_days: int,
    ) -> WalkForwardMetrics:
        if signal_side is not None:
            bucket = [
                item
                for item in signals
                if item.signal_side == signal_side and not item.blocked_by_no_trade
            ]
        else:
            bucket = [item for item in signals if not item.blocked_by_no_trade]

        total = len(bucket)
        pnls = [item.realized_pnl_points for item in bucket]
        wins = sum(1 for item in bucket if item.win)
        months = max(period_days / 30.4375, 1.0)
        pf = self._profit_factor(pnls)
        return WalkForwardMetrics(
            scope=scope,
            signal_side=signal_side,
            sample_size=total,
            signals_per_month=round(total / months, 2) if total else 0.0,
            win_rate_pct=round(wins / total * 100, 2) if total else 0.0,
            profit_factor=pf,
            expectancy=round(mean(pnls), 2) if pnls else 0.0,
            average_rr=round(mean(item.realized_rr for item in bucket), 2) if bucket else 0.0,
            maximum_drawdown_points=self._maximum_drawdown(pnls),
            hit_1r_rate_pct=round(sum(1 for item in bucket if item.hit_1r) / total * 100, 2)
            if total
            else 0.0,
            hit_2r_rate_pct=round(sum(1 for item in bucket if item.hit_2r) / total * 100, 2)
            if total
            else 0.0,
            hit_3r_rate_pct=round(sum(1 for item in bucket if item.hit_3r) / total * 100, 2)
            if total
            else 0.0,
            net_points=round(sum(pnls), 2),
        )

    @staticmethod
    def _degradation(in_sample: WalkForwardMetrics, out_sample: WalkForwardMetrics) -> dict[str, Any]:
        def delta(field: str) -> float | None:
            in_val = getattr(in_sample, field)
            out_val = getattr(out_sample, field)
            if in_val in (None, 0):
                return None
            if out_val is None:
                return None
            return round((out_val - in_val) / abs(in_val) * 100, 2)

        return {
            "expectancy_change_pct": delta("expectancy"),
            "profit_factor_change_pct": delta("profit_factor")
            if in_sample.profit_factor not in (None, 0) and out_sample.profit_factor is not None
            else None,
            "win_rate_change_pct": delta("win_rate_pct"),
            "hit_1r_change_pct": delta("hit_1r_rate_pct"),
            "signals_per_month_change_pct": delta("signals_per_month"),
        }

    def _survival_verdict(
        self,
        in_overall: WalkForwardMetrics,
        out_overall: WalkForwardMetrics,
    ) -> tuple[str, bool]:
        if out_overall.sample_size < MIN_OOS_SAMPLES:
            return "INSUFFICIENT_OOS_DATA", False
        if out_overall.expectancy <= 0:
            return "FAIL", False
        pf = out_overall.profit_factor
        if pf is None or pf < 1.0:
            return "FAIL", False
        if (
            out_overall.expectancy >= PREFERRED_EXPECTANCY * 0.5
            and pf >= 1.5
            and out_overall.win_rate_pct >= 40.0
        ):
            survives = out_overall.expectancy >= in_overall.expectancy * 0.4
            return ("SURVIVES" if survives else "DEGRADED"), survives
        if out_overall.expectancy > 0 and pf >= 1.0:
            return "DEGRADED", False
        return "FAIL", False

    @staticmethod
    def _metrics_bundle(
        signals: list[WalkForwardSignal],
        scope: str,
        period_start: date,
        period_end: date,
        engine: SmartMoneyEngineWalkForwardValidationResearch,
    ) -> dict[str, Any]:
        period_signals = [
            item
            for item in signals
            if engine._in_period(item.bos_timestamp, period_start, period_end)
        ]
        period_days = max((period_end - period_start).days + 1, 1)
        overall = engine._aggregate(period_signals, scope, None, period_days)
        buy = engine._aggregate(period_signals, scope, "BUY", period_days)
        sell = engine._aggregate(period_signals, scope, "SELL", period_days)
        return {
            "period_start": period_start.isoformat(),
            "period_end": period_end.isoformat(),
            "period_days": period_days,
            "overall": overall.as_dict(),
            "buy": buy.as_dict(),
            "sell": sell.as_dict(),
        }

    @staticmethod
    def _build_conclusions(
        card: dict[str, Any],
        in_overall: WalkForwardMetrics,
        out_overall: WalkForwardMetrics,
        verdict: str,
        survives: bool,
        degradation: dict[str, Any],
        no_trade_blocked: int,
    ) -> list[str]:
        buy_stack = " + ".join(card["buy_rules"]["filter_stack"])
        sell_stack = " + ".join(card["sell_rules"]["filter_stack"])
        return [
            "Walk-forward validation uses the frozen V1 production card only; no new signal discovery.",
            f"Mandatory core: {' + '.join(MANDATORY_CORE)}.",
            f"BUY rules: {buy_stack}.",
            f"SELL rules: {sell_stack}.",
            f"NO-TRADE exclusions blocked {no_trade_blocked} Tier-2 signals.",
            (
                f"In-sample (first {int(TRAIN_FRACTION * 100)}%): "
                f"{in_overall.sample_size} signals, "
                f"WR {in_overall.win_rate_pct}%, PF {in_overall.profit_factor}, "
                f"expectancy {in_overall.expectancy}, "
                f"1R/2R/3R {in_overall.hit_1r_rate_pct}/{in_overall.hit_2r_rate_pct}/{in_overall.hit_3r_rate_pct}%."
            ),
            (
                f"Out-of-sample (last {int(TEST_FRACTION * 100)}%): "
                f"{out_overall.sample_size} signals, "
                f"WR {out_overall.win_rate_pct}%, PF {out_overall.profit_factor}, "
                f"expectancy {out_overall.expectancy}, "
                f"1R/2R/3R {out_overall.hit_1r_rate_pct}/{out_overall.hit_2r_rate_pct}/{out_overall.hit_3r_rate_pct}%."
            ),
            f"Performance degradation: expectancy {degradation.get('expectancy_change_pct')}%, PF {degradation.get('profit_factor_change_pct')}%.",
            f"V1 survives unseen market data: {'Yes' if survives else 'No'} ({verdict}).",
        ]

    def run(self, metadata: dict[str, Any]) -> WalkForwardValidationReport:
        started = time.perf_counter()
        card = self._load_production_card()
        start, train_end, test_start, end = self._split_dates(metadata)

        all_signals: list[WalkForwardSignal] = []
        tier2_total = 0
        for symbol in self.symbols:
            symbol_signals, symbol_tier2 = self._collect_symbol_signals(symbol, metadata, card)
            all_signals.extend(symbol_signals)
            tier2_total += symbol_tier2

        v1_signals = [item for item in all_signals if not item.blocked_by_no_trade]
        no_trade_blocked = sum(1 for item in all_signals if item.blocked_by_no_trade)

        in_sample_bundle = self._metrics_bundle(all_signals, "in_sample", start, train_end, self)
        out_sample_bundle = self._metrics_bundle(all_signals, "out_of_sample", test_start, end, self)

        in_overall = self._aggregate(
            [item for item in all_signals if self._in_period(item.bos_timestamp, start, train_end)],
            "in_sample",
            None,
            max((train_end - start).days + 1, 1),
        )
        out_overall = self._aggregate(
            [item for item in all_signals if self._in_period(item.bos_timestamp, test_start, end)],
            "out_of_sample",
            None,
            max((end - test_start).days + 1, 1),
        )

        degradation = {
            "overall": self._degradation(in_overall, out_overall),
            "buy": self._degradation(
                self._aggregate(
                    [item for item in all_signals if self._in_period(item.bos_timestamp, start, train_end)],
                    "in_sample",
                    "BUY",
                    max((train_end - start).days + 1, 1),
                ),
                self._aggregate(
                    [item for item in all_signals if self._in_period(item.bos_timestamp, test_start, end)],
                    "out_of_sample",
                    "BUY",
                    max((end - test_start).days + 1, 1),
                ),
            ),
            "sell": self._degradation(
                self._aggregate(
                    [item for item in all_signals if self._in_period(item.bos_timestamp, start, train_end)],
                    "in_sample",
                    "SELL",
                    max((train_end - start).days + 1, 1),
                ),
                self._aggregate(
                    [item for item in all_signals if self._in_period(item.bos_timestamp, test_start, end)],
                    "out_of_sample",
                    "SELL",
                    max((end - test_start).days + 1, 1),
                ),
            ),
        }

        verdict, survives = self._survival_verdict(in_overall, out_overall)
        conclusions = self._build_conclusions(
            card,
            in_overall,
            out_overall,
            verdict,
            survives,
            degradation["overall"],
            no_trade_blocked,
        )

        return WalkForwardValidationReport(
            symbols_analyzed=list(self.symbols),
            research_window_days=metadata.get("research_window_days", self.research_days),
            start_date=metadata.get("start_date", ""),
            end_date=metadata.get("end_date", ""),
            train_fraction=TRAIN_FRACTION,
            test_fraction=TEST_FRACTION,
            train_start_date=start.isoformat(),
            train_end_date=train_end.isoformat(),
            test_start_date=test_start.isoformat(),
            test_end_date=end.isoformat(),
            frozen_v1_production_card=card,
            mandatory_signal_core=list(MANDATORY_CORE),
            buy_rules=card["buy_rules"],
            sell_rules=card["sell_rules"],
            no_trade_rules=list(card.get("no_trade_rules", [])),
            total_tier2_signals=tier2_total,
            total_v1_signals=len(v1_signals),
            no_trade_blocked_signals=no_trade_blocked,
            in_sample_metrics=in_sample_bundle,
            out_of_sample_metrics=out_sample_bundle,
            in_sample_buy=in_sample_bundle["buy"],
            in_sample_sell=in_sample_bundle["sell"],
            out_of_sample_buy=out_sample_bundle["buy"],
            out_of_sample_sell=out_sample_bundle["sell"],
            performance_degradation=degradation,
            survives_unseen_market_data=survives,
            survival_verdict=verdict,
            conclusions=conclusions,
            execution_time_seconds=round(time.perf_counter() - started, 2),
        )


def generate_walkforward_validation_report(
    report_path: Path | str | None = None,
    filter_report_path: Path | str | None = None,
    production_card_path: Path | str | None = None,
    symbols: tuple[str, ...] | None = None,
) -> WalkForwardValidationReport:
    """Run walk-forward validation and export JSON."""
    metadata_path = Path(filter_report_path or DEFAULT_FILTER_REPORT_PATH)
    if not metadata_path.exists():
        raise WalkForwardValidationError(f"Filter research report not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    engine = SmartMoneyEngineWalkForwardValidationResearch(
        symbols=symbols,
        production_card_path=production_card_path,
    )
    report = engine.run(metadata)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(report.as_dict()), handle, indent=2)

    logger.info(
        "Walk-forward validation completed: verdict=%s survives=%s",
        report.survival_verdict,
        report.survives_unseen_market_data,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_walkforward_validation_report()
        print("SmartMoneyEngine Walk Forward Validation Summary")
        print(f"Train: {report.train_start_date} -> {report.train_end_date}")
        print(f"Test: {report.test_start_date} -> {report.test_end_date}")
        print(f"In-sample signals: {report.in_sample_metrics['overall']['sample_size']}")
        print(f"Out-of-sample signals: {report.out_of_sample_metrics['overall']['sample_size']}")
        print(f"In-sample expectancy: {report.in_sample_metrics['overall']['expectancy']}")
        print(f"Out-of-sample expectancy: {report.out_of_sample_metrics['overall']['expectancy']}")
        print(f"Survival verdict: {report.survival_verdict}")
        print(f"V1 survives unseen data: {'Yes' if report.survives_unseen_market_data else 'No'}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except WalkForwardValidationError as exc:
        logger.error("Walk-forward validation error: %s", exc)
        print(f"Walk-forward validation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected walk-forward validation error")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
