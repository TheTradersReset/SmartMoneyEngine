"""
SmartMoneyEngine signal quality engine.

Ranks actionable trade signals before execution using SMC confluence,
multi-timeframe alignment, and trade plan V2 context.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from src.signals.decision_engine import DecisionEngineError, TradeDecision, evaluate_pipeline

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_MTF_REPORT = PROJECT_ROOT / "outputs" / "signals" / "multi_timeframe_report.json"
DEFAULT_TRADE_PLAN_V2 = PROJECT_ROOT / "outputs" / "signals" / "trade_plan_report_v2.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "signals" / "signal_quality_report.json"

FRESH_ZONE_BARS = 20


class SignalQualityEngineError(Exception):
    """Raised when signal quality evaluation fails."""


class SignalGrade(str, Enum):
    """Institutional signal quality grade."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    REJECT = "Reject"


@dataclass
class QualityFactors:
    """Component scores contributing to signal quality."""

    htf_alignment: float
    trend_strength: float
    bos_quality: float
    choch_quality: float
    liquidity_sweep: float
    fresh_fvg: float
    fresh_order_block: float
    structure_quality: float
    trade_plan_quality: float

    def as_dict(self) -> dict[str, float]:
        """Return serializable factor scores."""
        return asdict(self)

    @property
    def total(self) -> float:
        """Sum of all factor scores."""
        return (
            self.htf_alignment
            + self.trend_strength
            + self.bos_quality
            + self.choch_quality
            + self.liquidity_sweep
            + self.fresh_fvg
            + self.fresh_order_block
            + self.structure_quality
            + self.trade_plan_quality
        )


@dataclass
class SignalQuality:
    """Quality assessment for one actionable signal."""

    signal_index: int
    signal_date: str
    decision: str
    quality_score: float
    grade: str
    factors: dict[str, float]
    reasoning: list[str]
    trade_plan_grade: str | None
    risk_reward_t2: float | None
    trade_validity: bool | None

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable signal quality dictionary."""
        return asdict(self)


@dataclass
class SignalQualityReport:
    """Aggregate signal quality report."""

    symbol: str
    timeframe: str
    source_csv: str
    source_mtf_report: str
    source_trade_plan_report: str
    total_signals: int
    average_score: float
    grade_distribution: dict[str, int]
    execution_time_seconds: float
    signals: list[dict[str, Any]] = field(default_factory=list)
    top_signals: list[dict[str, Any]] = field(default_factory=list)
    bottom_signals: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable report dictionary."""
        return asdict(self)


class SignalQualityEngine:
    """
    Score and grade actionable signals from SMC pipeline outputs.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Timeframe label for reporting.
    fresh_zone_bars : int, optional
        Lookback window for fresh FVG and order block detection.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        fresh_zone_bars: int = FRESH_ZONE_BARS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.fresh_zone_bars = fresh_zone_bars
        self._mtf_report: dict[str, Any] | None = None

    @staticmethod
    def _is_active(value: Any) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return not pd.isna(value)
        return bool(str(value).strip())

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if not SignalQualityEngine._is_active(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_trend(value: Any) -> str:
        if not SignalQualityEngine._is_active(value):
            return "Neutral"
        return str(value).strip().capitalize()

    def load_mtf_report(self, path: Path | str | None = None) -> dict[str, Any]:
        """Load multi-timeframe alignment report."""
        report_path = Path(path) if path is not None else DEFAULT_MTF_REPORT
        if not report_path.exists():
            logger.warning("MTF report not found at %s; using neutral defaults.", report_path)
            self._mtf_report = {"overall_bias": "Neutral", "timeframes": []}
            return self._mtf_report
        with report_path.open("r", encoding="utf-8") as handle:
            self._mtf_report = json.load(handle)
        return self._mtf_report

    @staticmethod
    def load_trade_plans(path: Path | str | None = None) -> list[dict[str, Any]]:
        """Load trade plan V2 entries keyed by signal index."""
        plan_path = Path(path) if path is not None else DEFAULT_TRADE_PLAN_V2
        if not plan_path.exists():
            return []
        with plan_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return list(payload.get("trade_plans", []))

    def _htf_alignment_score(self, decision: str) -> tuple[float, list[str]]:
        """Score higher-timeframe alignment (0-20)."""
        notes: list[str] = []
        if self._mtf_report is None:
            return 10.0, ["HTF neutral (no MTF report)"]

        trends = {
            item["timeframe"]: item.get("trend", "Neutral")
            for item in self._mtf_report.get("timeframes", [])
        }
        htf = [trends.get("1D", "Neutral"), trends.get("4H", "Neutral")]
        overall = str(self._mtf_report.get("overall_bias", "Neutral"))

        if decision == TradeDecision.BUY.value:
            aligned = sum(1 for trend in htf if trend == "Bullish")
            opposed = sum(1 for trend in htf if trend == "Bearish")
        else:
            aligned = sum(1 for trend in htf if trend == "Bearish")
            opposed = sum(1 for trend in htf if trend == "Bullish")

        if aligned == 2:
            notes.append("HTF aligned (1D + 4H)")
            return 20.0, notes
        if aligned == 1 and opposed == 0:
            notes.append("Partial HTF alignment")
            return 14.0, notes
        if opposed >= 1 and aligned == 0:
            notes.append("HTF misaligned")
            return 0.0, notes
        if overall in {"Strong Bullish", "Strong Bearish"}:
            notes.append(f"HTF overall bias {overall}")
            return 12.0, notes
        notes.append("HTF neutral")
        return 8.0, notes

    def _trend_strength_score(self, row: pd.Series, decision: str) -> tuple[float, list[str]]:
        """Score trend strength and direction (0-15)."""
        notes: list[str] = []
        trend = self._normalize_trend(row.get("Trend"))
        strength = self._to_float(row.get("Trend_Strength")) or 0.0

        aligned = (
            (decision == TradeDecision.BUY.value and trend == "Bullish")
            or (decision == TradeDecision.SELL.value and trend == "Bearish")
        )
        opposed = (
            (decision == TradeDecision.BUY.value and trend == "Bearish")
            or (decision == TradeDecision.SELL.value and trend == "Bullish")
        )

        if aligned:
            score = min(8.0 + strength * 2.5, 15.0)
            notes.append(f"Trend {trend}")
            notes.append(f"Trend strength {int(strength)}")
            return round(score, 1), notes
        if opposed:
            notes.append(f"Trend opposed ({trend})")
            return 2.0, notes
        notes.append("Trend neutral")
        return 5.0, notes

    def _bos_score(self, row: pd.Series, decision: str) -> tuple[float, list[str]]:
        """Score break-of-structure quality (0-12)."""
        notes: list[str] = []
        if decision == TradeDecision.BUY.value and self._is_active(row.get("Bullish_BOS")):
            notes.append("Bullish BOS")
            return 12.0, notes
        if decision == TradeDecision.SELL.value and self._is_active(row.get("Bearish_BOS")):
            notes.append("Bearish BOS")
            return 12.0, notes
        return 0.0, notes

    def _choch_score(self, row: pd.Series, decision: str) -> tuple[float, list[str]]:
        """Score change-of-character quality (0-10)."""
        notes: list[str] = []
        if decision == TradeDecision.BUY.value and self._is_active(row.get("Bullish_CHOCH")):
            notes.append("Bullish CHOCH")
            return 10.0, notes
        if decision == TradeDecision.SELL.value and self._is_active(row.get("Bearish_CHOCH")):
            notes.append("Bearish CHOCH")
            return 10.0, notes
        return 0.0, notes

    def _liquidity_sweep_score(self, row: pd.Series, decision: str) -> tuple[float, list[str]]:
        """Score liquidity sweep confluence (0-12)."""
        notes: list[str] = []
        if decision == TradeDecision.BUY.value and self._is_active(row.get("Sell_Liquidity_Sweep")):
            notes.append("Sell-side liquidity sweep")
            return 12.0, notes
        if decision == TradeDecision.SELL.value and self._is_active(row.get("Buy_Liquidity_Sweep")):
            notes.append("Buy-side liquidity sweep")
            return 12.0, notes
        return 0.0, notes

    def _fresh_fvg_score(
        self,
        frame: pd.DataFrame,
        index: int,
        decision: str,
    ) -> tuple[float, list[str]]:
        """Score fresh fair value gap presence (0-10)."""
        start = max(0, index - self.fresh_zone_bars)
        window = frame.iloc[start:index + 1]
        notes: list[str] = []

        if decision == TradeDecision.BUY.value:
            has_fvg = window["Bullish_FVG_Top"].notna() & window["Bullish_FVG_Bottom"].notna()
        else:
            has_fvg = window["Bearish_FVG_Top"].notna() & window["Bearish_FVG_Bottom"].notna()

        if has_fvg.any():
            notes.append("Fresh FVG")
            return 10.0, notes
        return 0.0, notes

    def _fresh_ob_score(
        self,
        frame: pd.DataFrame,
        index: int,
        decision: str,
    ) -> tuple[float, list[str]]:
        """Score fresh unmitigated order block (0-12)."""
        start = max(0, index - self.fresh_zone_bars)
        window = frame.iloc[start:index + 1]
        notes: list[str] = []

        if decision == TradeDecision.BUY.value:
            candidates = window[
                window["Bullish_OB_High"].notna() & window["Bullish_OB_Low"].notna()
            ]
            for _, row in candidates.iloc[::-1].iterrows():
                if not self._is_active(row.get("Bullish_OB_Mitigated")):
                    notes.append("Fresh order block")
                    return 12.0, notes
        else:
            candidates = window[
                window["Bearish_OB_High"].notna() & window["Bearish_OB_Low"].notna()
            ]
            for _, row in candidates.iloc[::-1].iterrows():
                if not self._is_active(row.get("Bearish_OB_Mitigated")):
                    notes.append("Fresh order block")
                    return 12.0, notes
        return 0.0, notes

    def _structure_score(self, row: pd.Series, decision: str) -> tuple[float, list[str]]:
        """Score market structure labels (0-9)."""
        notes: list[str] = []
        if decision == TradeDecision.BUY.value:
            if self._is_active(row.get("HH")) and self._is_active(row.get("HL")):
                notes.append("Structure HH-HL")
                return 9.0, notes
            if self._is_active(row.get("HL")):
                notes.append("Structure HL")
                return 6.0, notes
            if self._is_active(row.get("HH")):
                notes.append("Structure HH")
                return 5.0, notes
        else:
            if self._is_active(row.get("LH")) and self._is_active(row.get("LL")):
                notes.append("Structure LH-LL")
                return 9.0, notes
            if self._is_active(row.get("LH")):
                notes.append("Structure LH")
                return 6.0, notes
            if self._is_active(row.get("LL")):
                notes.append("Structure LL")
                return 5.0, notes
        return 0.0, notes

    def _trade_plan_score(self, plan: dict[str, Any] | None) -> tuple[float, list[str]]:
        """Score trade plan V2 quality (0-10)."""
        notes: list[str] = []
        if plan is None:
            return 0.0, notes

        score = 0.0
        if plan.get("trade_validity"):
            score += 4.0
        rr_t2 = self._to_float(plan.get("risk_reward_t2")) or 0.0
        if rr_t2 >= 2.0:
            score += 4.0
            notes.append(f"RR T2 {rr_t2}")
        elif rr_t2 >= 1.5:
            score += 2.0
            notes.append(f"RR T2 {rr_t2}")

        grade = str(plan.get("trade_grade", ""))
        if grade in {SignalGrade.A_PLUS.value, SignalGrade.A.value}:
            score += 2.0
            notes.append(f"Trade plan grade {grade}")
        elif grade == SignalGrade.B.value:
            score += 1.0

        return min(score, 10.0), notes

    @staticmethod
    def _assign_grade(score: float, trade_valid: bool | None) -> SignalGrade:
        if trade_valid is False or score < 40:
            return SignalGrade.REJECT
        if score >= 85:
            return SignalGrade.A_PLUS
        if score >= 70:
            return SignalGrade.A
        if score >= 55:
            return SignalGrade.B
        if score >= 40:
            return SignalGrade.C
        return SignalGrade.REJECT

    def evaluate_signal(
        self,
        frame: pd.DataFrame,
        index: int,
        decision: str,
        signal_date: str,
        plan: dict[str, Any] | None = None,
    ) -> SignalQuality:
        """Compute quality score and grade for one signal."""
        row = frame.iloc[index]
        reasoning: list[str] = []

        htf_score, htf_notes = self._htf_alignment_score(decision)
        reasoning.extend(htf_notes)

        trend_score, trend_notes = self._trend_strength_score(row, decision)
        reasoning.extend(trend_notes)

        bos_score, bos_notes = self._bos_score(row, decision)
        reasoning.extend(bos_notes)

        choch_score, choch_notes = self._choch_score(row, decision)
        reasoning.extend(choch_notes)

        sweep_score, sweep_notes = self._liquidity_sweep_score(row, decision)
        reasoning.extend(sweep_notes)

        fvg_score, fvg_notes = self._fresh_fvg_score(frame, index, decision)
        reasoning.extend(fvg_notes)

        ob_score, ob_notes = self._fresh_ob_score(frame, index, decision)
        reasoning.extend(ob_notes)

        structure_score, structure_notes = self._structure_score(row, decision)
        reasoning.extend(structure_notes)

        plan_score, plan_notes = self._trade_plan_score(plan)
        reasoning.extend(plan_notes)

        factors = QualityFactors(
            htf_alignment=htf_score,
            trend_strength=trend_score,
            bos_quality=bos_score,
            choch_quality=choch_score,
            liquidity_sweep=sweep_score,
            fresh_fvg=fvg_score,
            fresh_order_block=ob_score,
            structure_quality=structure_score,
            trade_plan_quality=plan_score,
        )
        quality_score = round(min(factors.total, 100.0), 1)
        trade_valid = plan.get("trade_validity") if plan else None
        grade = self._assign_grade(quality_score, trade_valid)

        return SignalQuality(
            signal_index=index,
            signal_date=signal_date,
            decision=decision,
            quality_score=quality_score,
            grade=grade.value,
            factors=factors.as_dict(),
            reasoning=reasoning,
            trade_plan_grade=str(plan.get("trade_grade")) if plan else None,
            risk_reward_t2=self._to_float(plan.get("risk_reward_t2")) if plan else None,
            trade_validity=bool(trade_valid) if trade_valid is not None else None,
        )

    def evaluate(
        self,
        frame: pd.DataFrame,
        trade_plans: list[dict[str, Any]] | None = None,
        mtf_report: dict[str, Any] | None = None,
    ) -> SignalQualityReport:
        """
        Evaluate quality for all actionable signals in the pipeline.

        Parameters
        ----------
        frame : pd.DataFrame
            Evaluated pipeline dataframe with decision columns.
        trade_plans : list[dict[str, Any]] | None, optional
            Trade plan V2 entries.
        mtf_report : dict[str, Any] | None, optional
            Multi-timeframe report.

        Returns
        -------
        SignalQualityReport
            Aggregate quality report with ranked signals.
        """
        started = time.perf_counter()
        if mtf_report is not None:
            self._mtf_report = mtf_report
        elif self._mtf_report is None:
            self.load_mtf_report()

        plans = trade_plans if trade_plans is not None else self.load_trade_plans()
        plan_by_index = {int(plan["signal_index"]): plan for plan in plans}

        qualities: list[SignalQuality] = []
        for index, row in frame.iterrows():
            decision = str(row.get("Decision"))
            if decision not in {TradeDecision.BUY.value, TradeDecision.SELL.value}:
                continue
            plan = plan_by_index.get(int(index))
            signal_date = str(plan["signal_date"]) if plan else str(row.get("Date"))
            qualities.append(
                self.evaluate_signal(frame, int(index), decision, signal_date, plan)
            )

        sorted_signals = sorted(qualities, key=lambda item: item.quality_score, reverse=True)
        top_signals = [item.as_dict() for item in sorted_signals[:10]]
        bottom_signals = [item.as_dict() for item in sorted(sorted_signals, key=lambda i: i.quality_score)[:10]]

        grade_distribution = {grade.value: 0 for grade in SignalGrade}
        for item in qualities:
            grade_distribution[item.grade] = grade_distribution.get(item.grade, 0) + 1

        avg_score = round(sum(item.quality_score for item in qualities) / len(qualities), 1) if qualities else 0.0
        elapsed = time.perf_counter() - started

        return SignalQualityReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=str(DEFAULT_PIPELINE_CSV),
            source_mtf_report=str(DEFAULT_MTF_REPORT),
            source_trade_plan_report=str(DEFAULT_TRADE_PLAN_V2),
            total_signals=len(qualities),
            average_score=avg_score,
            grade_distribution=grade_distribution,
            execution_time_seconds=elapsed,
            signals=[item.as_dict() for item in sorted_signals],
            top_signals=top_signals,
            bottom_signals=bottom_signals,
        )


def generate_signal_quality_report(
    pipeline_csv: Path | str | None = None,
    mtf_report_path: Path | str | None = None,
    trade_plan_path: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5",
) -> SignalQualityReport:
    """Run signal quality evaluation and export the JSON report."""
    csv_path = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
    mtf_path = Path(mtf_report_path) if mtf_report_path is not None else DEFAULT_MTF_REPORT
    plan_path = Path(trade_plan_path) if trade_plan_path is not None else DEFAULT_TRADE_PLAN_V2
    json_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH

    evaluated, _ = evaluate_pipeline(pipeline_csv=csv_path)
    engine = SignalQualityEngine(symbol=symbol, timeframe=timeframe)
    mtf_report = engine.load_mtf_report(mtf_path)
    trade_plans = engine.load_trade_plans(plan_path)
    report = engine.evaluate(evaluated, trade_plans=trade_plans, mtf_report=mtf_report)
    report.source_csv = str(csv_path)
    report.source_mtf_report = str(mtf_path)
    report.source_trade_plan_report = str(plan_path)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Signal quality evaluation completed: signals=%s avg_score=%s grades=%s",
        report.total_signals,
        report.average_score,
        report.grade_distribution,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_signal_quality_report()
        print("Signal Quality Engine Summary")
        print(f"Total Signals: {report.total_signals}")
        print(f"Average Score: {report.average_score}")
        print(f"Grade Distribution: {report.grade_distribution}")
        print("Top Signals:")
        for rank, signal in enumerate(report.top_signals[:5], start=1):
            print(
                f"  {rank}. {signal['signal_date']} {signal['decision']} | "
                f"score={signal['quality_score']} grade={signal['grade']}"
            )
        print("Bottom Signals:")
        for rank, signal in enumerate(report.bottom_signals[:5], start=1):
            print(
                f"  {rank}. {signal['signal_date']} {signal['decision']} | "
                f"score={signal['quality_score']} grade={signal['grade']}"
            )
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except (SignalQualityEngineError, DecisionEngineError) as exc:
        logger.error("Signal quality engine error: %s", exc)
        print(f"Signal quality engine error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected signal quality engine failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
