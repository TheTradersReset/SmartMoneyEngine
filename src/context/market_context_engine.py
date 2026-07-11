"""
Market Context Engine for SmartMoneyEngine Phase-1.

Evaluates the location and quality of every BUY/SELL signal before trade
execution using multi-timeframe, location, session, and volatility context.
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

from src.models.market_data import MarketData
from src.pipeline.market_pipeline import prepare_market_dataframe
from src.signals.decision_engine import DecisionEngine, TradeDecision
from src.smc.market_structure import MarketStructure
from src.smc.swing_detector import SwingDetector
from src.smc.trend_engine import TrendEngine

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "context" / "context_report.json"

LOCATION_LOOKBACK_BARS = 200
ATR_PERIOD = 14
ATR_BASELINE_BARS = 100
RANGE_LOOKBACK_BARS = 20

SCORE_WEIGHTS: dict[str, int] = {
    "htf_trend_alignment": 20,
    "market_location": 20,
    "liquidity_context": 20,
    "session_quality": 15,
    "volatility_quality": 15,
    "structure_quality": 10,
}

HTF_RESAMPLE_RULES: tuple[tuple[str, str], ...] = (
    ("1D", "1D"),
    ("4H", "4h"),
    ("1H", "1h"),
    ("15M", "15min"),
)


class MarketContextError(Exception):
    """Raised when market context evaluation fails."""


class ContextGrade(str, Enum):
    """Institutional context grade."""

    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    REJECT = "Reject"


class SessionLabel(str, Enum):
    """Indian equity session bucket."""

    OPENING = "Opening hour"
    MID = "Mid session"
    CLOSING = "Closing hour"
    OUTSIDE = "Outside session"


@dataclass(frozen=True)
class ContextComponents:
    """Component scores contributing to the context score."""

    htf_trend_alignment: float
    market_location: float
    liquidity_context: float
    session_quality: float
    volatility_quality: float
    structure_quality: float

    @property
    def total(self) -> float:
        return (
            self.htf_trend_alignment
            + self.market_location
            + self.liquidity_context
            + self.session_quality
            + self.volatility_quality
            + self.structure_quality
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class SignalContext:
    """Context evaluation for one actionable signal."""

    signal_index: int
    signal_date: str
    decision: str
    context_score: float
    context_grade: str
    components: dict[str, float]
    reasoning: list[str]
    multi_timeframe: dict[str, str]
    market_location: dict[str, Any]
    session: str
    volatility: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MarketContextReport:
    """Aggregate market context report."""

    symbol: str
    timeframe: str
    source_csv: str
    total_candles: int
    total_signals: int
    average_context_score: float
    grade_distribution: dict[str, int]
    execution_time_seconds: float
    signals: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarketContextEngine:
    """
    Evaluate market context for actionable BUY/SELL signals.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Source timeframe label such as ``5``.
    location_lookback_bars : int, optional
        Bars used to derive support/resistance and swing context.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5",
        location_lookback_bars: int = LOCATION_LOOKBACK_BARS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.location_lookback_bars = location_lookback_bars
        self.decision_engine = DecisionEngine(symbol=symbol, timeframe=timeframe)

    @staticmethod
    def _is_active(value: Any) -> bool:
        return DecisionEngine._is_active(value)

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value is None or pd.isna(value):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_trend(value: Any) -> str:
        trend = str(value).strip().upper()
        if trend in {"BULLISH", "BEARISH"}:
            return trend
        if trend in {"SIDEWAYS", "NEUTRAL"}:
            return "NEUTRAL"
        return "NEUTRAL"

    @staticmethod
    def _ensure_ist(series: pd.Series) -> pd.Series:
        timestamps = pd.to_datetime(series, errors="coerce")
        if timestamps.dt.tz is None:
            return timestamps.dt.tz_localize("Asia/Kolkata")
        return timestamps.dt.tz_convert("Asia/Kolkata")

    @staticmethod
    def _normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
        working = frame[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
        working["timestamp"] = MarketContextEngine._ensure_ist(working["Date"])
        working = working.dropna(subset=["timestamp"]).sort_values("timestamp")
        return working.reset_index(drop=True)

    @staticmethod
    def _resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        indexed = frame.set_index("timestamp")
        resampled = (
            indexed.resample(rule)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna(subset=["Open", "High", "Low", "Close"])
            .reset_index()
        )
        return resampled

    def _trend_series_on_resampled(self, resampled: pd.DataFrame) -> pd.DataFrame:
        """Compute per-bar trend labels on a resampled OHLCV series."""
        if len(resampled) < 20:
            empty = resampled[["timestamp"]].copy()
            empty["Trend"] = "NEUTRAL"
            return empty

        input_frame = resampled[["timestamp", "Open", "High", "Low", "Close", "Volume"]].copy()
        input_frame = input_frame.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
            }
        )
        prepared = prepare_market_dataframe(input_frame)
        market = MarketData(prepared.copy())
        SwingDetector().detect(market)
        MarketStructure().detect(market)
        TrendEngine().detect(market)

        result = market.data[["Date", "Trend"]].copy()
        result["timestamp"] = self._ensure_ist(result["Date"])
        result["Trend"] = result["Trend"].apply(self._normalize_trend)
        return result[["timestamp", "Trend"]]

    def _build_htf_trend_lookup(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach per-bar HTF trend context via as-of merge."""
        aligned = frame.copy()
        aligned["timestamp"] = self._ensure_ist(aligned["Date"])
        valid = aligned["timestamp"].notna()
        ohlcv = self._normalize_ohlcv(frame.loc[valid].copy())

        trend_columns: dict[str, pd.Series] = {}
        for label, rule in HTF_RESAMPLE_RULES:
            resampled = self._resample_ohlcv(ohlcv, rule)
            trend_frame = self._trend_series_on_resampled(resampled)
            merged = pd.merge_asof(
                ohlcv[["timestamp"]].sort_values("timestamp"),
                trend_frame.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )
            trend_map = dict(zip(merged["timestamp"], merged["Trend"], strict=False))
            trend_columns[f"HTF_{label}_Trend"] = aligned["timestamp"].map(trend_map).fillna("NEUTRAL")

        lookup = pd.DataFrame(trend_columns)
        if "Trend" in frame.columns:
            lookup["HTF_5M_Trend"] = frame["Trend"].apply(self._normalize_trend).values
        else:
            lookup["HTF_5M_Trend"] = "NEUTRAL"
        return lookup

    @staticmethod
    def _compute_atr(frame: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        high = frame["High"].astype(float)
        low = frame["Low"].astype(float)
        close = frame["Close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def _session_label(timestamp: pd.Timestamp) -> SessionLabel:
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Kolkata")
        else:
            ts = ts.tz_convert("Asia/Kolkata")

        minutes = ts.hour * 60 + ts.minute
        open_start = 9 * 60 + 15
        open_end = 10 * 60 + 30
        close_start = 14 * 60 + 30
        close_end = 15 * 60 + 30

        if open_start <= minutes < open_end:
            return SessionLabel.OPENING
        if open_end <= minutes < close_start:
            return SessionLabel.MID
        if close_start <= minutes <= close_end:
            return SessionLabel.CLOSING
        return SessionLabel.OUTSIDE

    def _collect_levels(self, window: pd.DataFrame, column: str) -> list[float]:
        values: list[float] = []
        if column not in window.columns:
            return values
        for value in window[column]:
            parsed = self._to_float(value)
            if parsed is not None and self._is_active(value):
                values.append(parsed)
        return values

    def _market_levels(self, frame: pd.DataFrame, index: int) -> dict[str, Any]:
        """Derive support, resistance, and swing references near the signal."""
        start = max(0, index - self.location_lookback_bars)
        window = frame.iloc[start : index + 1]
        close = self._to_float(frame.iloc[index]["Close"]) or 0.0

        supports = (
            self._collect_levels(window, "Swing_Low")
            + self._collect_levels(window, "Equal_Low")
            + self._collect_levels(window, "Bullish_OB_Low")
            + self._collect_levels(window, "Sell_Side_Liquidity")
        )
        resistances = (
            self._collect_levels(window, "Swing_High")
            + self._collect_levels(window, "Equal_High")
            + self._collect_levels(window, "Bearish_OB_High")
            + self._collect_levels(window, "Buy_Side_Liquidity")
        )

        major_support = max([level for level in supports if level <= close], default=None)
        major_resistance = min([level for level in resistances if level >= close], default=None)
        if major_support is None and supports:
            major_support = min(supports)
        if major_resistance is None and resistances:
            major_resistance = max(resistances)

        prior_swing_high = max(self._collect_levels(window, "Swing_High"), default=None)
        prior_swing_low = min(self._collect_levels(window, "Swing_Low"), default=None)

        return {
            "price": round(close, 2),
            "major_support": round(major_support, 2) if major_support is not None else None,
            "major_resistance": round(major_resistance, 2) if major_resistance is not None else None,
            "prior_swing_high": round(prior_swing_high, 2) if prior_swing_high is not None else None,
            "prior_swing_low": round(prior_swing_low, 2) if prior_swing_low is not None else None,
            "distance_to_support": round(close - major_support, 2) if major_support is not None else None,
            "distance_to_resistance": round(major_resistance - close, 2)
            if major_resistance is not None
            else None,
        }

    @staticmethod
    def _distance_score(distance: float | None, atr: float, favorable: bool) -> float:
        if distance is None or atr <= 0:
            return 0.5
        ratio = abs(distance) / atr
        if favorable:
            if ratio <= 0.5:
                return 1.0
            if ratio <= 1.0:
                return 0.85
            if ratio <= 2.0:
                return 0.65
            return 0.35
        if ratio <= 0.5:
            return 0.2
        if ratio <= 1.0:
            return 0.45
        return 0.7

    def _score_htf_alignment(
        self,
        decision: str,
        trends: dict[str, str],
        notes: list[str],
    ) -> float:
        wanted = "BULLISH" if decision == TradeDecision.BUY.value else "BEARISH"
        opposed = "BEARISH" if wanted == "BULLISH" else "BULLISH"
        weighted = {"1D": 8, "4H": 7, "1H": 5}
        score = 0.0

        for label, weight in weighted.items():
            trend = trends.get(label, "NEUTRAL")
            if trend == wanted:
                score += weight
                notes.append(f"{label} {wanted.lower()}")
            elif trend == opposed:
                notes.append(f"{label} opposed ({trend.lower()})")
            else:
                score += weight * 0.35
                notes.append(f"{label} neutral")

        for label in ("15M", "5M"):
            trend = trends.get(label, "NEUTRAL")
            if trend == wanted:
                notes.append(f"{label} aligned")
            elif trend == opposed:
                notes.append(f"{label} opposed")

        return round(min(score, SCORE_WEIGHTS["htf_trend_alignment"]), 1)

    def _score_market_location(
        self,
        decision: str,
        location: dict[str, Any],
        atr: float,
        notes: list[str],
    ) -> float:
        price = location["price"]
        support = location["major_support"]
        resistance = location["major_resistance"]

        if decision == TradeDecision.BUY.value:
            support_score = self._distance_score(location["distance_to_support"], atr, favorable=True)
            resistance_score = self._distance_score(location["distance_to_resistance"], atr, favorable=False)
            if support is not None and price >= support:
                notes.append("Price above major support")
            if resistance is not None and price < resistance:
                notes.append("Room to major resistance")
        else:
            support_score = self._distance_score(location["distance_to_support"], atr, favorable=False)
            resistance_score = self._distance_score(location["distance_to_resistance"], atr, favorable=True)
            if resistance is not None and price <= resistance:
                notes.append("Price below major resistance")
            if support is not None and price > support:
                notes.append("Room to major support")

        if location["prior_swing_low"] is not None:
            notes.append(f"Prior swing low at {location['prior_swing_low']}")
        if location["prior_swing_high"] is not None:
            notes.append(f"Prior swing high at {location['prior_swing_high']}")

        combined = (support_score + resistance_score) / 2.0
        return round(combined * SCORE_WEIGHTS["market_location"], 1)

    def _score_liquidity_context(
        self,
        frame: pd.DataFrame,
        index: int,
        decision: str,
        location: dict[str, Any],
        atr: float,
        notes: list[str],
    ) -> float:
        start = max(0, index - 20)
        window = frame.iloc[start : index + 1]
        row = frame.iloc[index]
        score = 0.0

        sell_sweep = self._column_active_in_window(window, "Sell_Liquidity_Sweep")
        buy_sweep = self._column_active_in_window(window, "Buy_Liquidity_Sweep")
        buy_pool = self._to_float(row.get("Buy_Side_Liquidity"))
        sell_pool = self._to_float(row.get("Sell_Side_Liquidity"))

        if decision == TradeDecision.BUY.value:
            if sell_sweep:
                score += 10
                notes.append("Sell-side liquidity already taken")
            if buy_pool is not None:
                dist = abs(location["price"] - buy_pool)
                score += 4 if dist <= atr * 2 else 1
                notes.append("Near HTF buy-side liquidity pool")
        else:
            if buy_sweep:
                score += 10
                notes.append("Buy-side liquidity already taken")
            if sell_pool is not None:
                dist = abs(location["price"] - sell_pool)
                score += 4 if dist <= atr * 2 else 1
                notes.append("Near HTF sell-side liquidity pool")

        if location["major_support"] is not None and decision == TradeDecision.BUY.value:
            if abs(location["price"] - location["major_support"]) <= atr:
                score += 4
                notes.append("Near demand zone")
        if location["major_resistance"] is not None and decision == TradeDecision.SELL.value:
            if abs(location["price"] - location["major_resistance"]) <= atr:
                score += 4
                notes.append("Near supply zone")

        return round(min(score, SCORE_WEIGHTS["liquidity_context"]), 1)

    @staticmethod
    def _column_active_in_window(window: pd.DataFrame, column: str) -> bool:
        if column not in window.columns:
            return False
        return any(MarketContextEngine._is_active(value) for value in window[column])

    def _score_session(self, timestamp: pd.Timestamp, notes: list[str]) -> float:
        session = self._session_label(timestamp)
        notes.append(f"Session: {session.value}")
        if session == SessionLabel.MID:
            return float(SCORE_WEIGHTS["session_quality"])
        if session in {SessionLabel.OPENING, SessionLabel.CLOSING}:
            return round(SCORE_WEIGHTS["session_quality"] * 0.7, 1)
        return round(SCORE_WEIGHTS["session_quality"] * 0.4, 1)

    def _score_volatility(
        self,
        frame: pd.DataFrame,
        index: int,
        atr_series: pd.Series,
        notes: list[str],
    ) -> tuple[float, dict[str, float]]:
        atr = float(atr_series.iloc[index])
        baseline_start = max(0, index - ATR_BASELINE_BARS)
        baseline = float(atr_series.iloc[baseline_start:index + 1].mean()) if index else atr
        atr_ratio = atr / baseline if baseline > 0 else 1.0

        range_start = max(0, index - RANGE_LOOKBACK_BARS)
        ranges = (frame["High"].astype(float) - frame["Low"].astype(float)).iloc[range_start : index + 1]
        current_range = float(frame.iloc[index]["High"] - frame.iloc[index]["Low"])
        avg_range = float(ranges.mean()) if len(ranges) else current_range
        range_ratio = current_range / avg_range if avg_range > 0 else 1.0

        if 0.9 <= atr_ratio <= 1.8:
            atr_score = 1.0
            notes.append("Healthy ATR volatility")
        elif atr_ratio < 0.9:
            atr_score = 0.45
            notes.append("Compressed volatility")
        else:
            atr_score = 0.75
            notes.append("Elevated volatility")

        if range_ratio >= 1.2:
            range_score = 1.0
            notes.append("Strong volatility expansion")
        elif range_ratio >= 0.8:
            range_score = 0.7
        else:
            range_score = 0.4
            notes.append("Weak range expansion")

        combined = (atr_score + range_score) / 2.0
        score = round(combined * SCORE_WEIGHTS["volatility_quality"], 1)
        metrics = {
            "atr": round(atr, 2),
            "atr_ratio": round(atr_ratio, 2),
            "range_ratio": round(range_ratio, 2),
        }
        return score, metrics

    def _score_structure(self, row: pd.Series, decision: str, notes: list[str]) -> float:
        score = 0.0
        trend = self._normalize_trend(row.get("Trend"))
        strength = self._to_float(row.get("Trend_Strength")) or 0.0
        wanted = "BULLISH" if decision == TradeDecision.BUY.value else "BEARISH"

        if trend == wanted:
            score += 4
            notes.append(f"5M trend {trend.lower()}")
        elif trend == "NEUTRAL":
            score += 2
        score += min(strength, 3) * 1.5

        bos_column = "Bullish_BOS" if wanted == "BULLISH" else "Bearish_BOS"
        choch_column = "Bullish_CHOCH" if wanted == "BULLISH" else "Bearish_CHOCH"
        if self._is_active(row.get(bos_column)):
            score += 2
            notes.append("Structure BOS present")
        if self._is_active(row.get(choch_column)):
            score += 1

        return round(min(score, SCORE_WEIGHTS["structure_quality"]), 1)

    @staticmethod
    def _assign_grade(score: float) -> ContextGrade:
        if score >= 90:
            return ContextGrade.A_PLUS
        if score >= 80:
            return ContextGrade.A
        if score >= 70:
            return ContextGrade.B
        if score >= 60:
            return ContextGrade.C
        return ContextGrade.REJECT

    def evaluate_signal(
        self,
        frame: pd.DataFrame,
        index: int,
        htf_lookup: pd.DataFrame,
        atr_series: pd.Series,
    ) -> SignalContext | None:
        """Evaluate market context for one signal bar."""
        row = frame.iloc[index]
        decision = str(row.get("Decision", TradeDecision.WAIT.value))
        if decision not in {TradeDecision.BUY.value, TradeDecision.SELL.value}:
            return None

        notes: list[str] = []
        timestamp = self._ensure_ist(pd.Series([row["Date"]])).iloc[0]
        htf_row = htf_lookup.iloc[index]
        trends = {
            "1D": str(htf_row.get("HTF_1D_Trend", "NEUTRAL")),
            "4H": str(htf_row.get("HTF_4H_Trend", "NEUTRAL")),
            "1H": str(htf_row.get("HTF_1H_Trend", "NEUTRAL")),
            "15M": str(htf_row.get("HTF_15M_Trend", "NEUTRAL")),
            "5M": str(htf_row.get("HTF_5M_Trend", self._normalize_trend(row.get("Trend")))),
        }

        location = self._market_levels(frame, index)
        atr = float(atr_series.iloc[index])

        components = ContextComponents(
            htf_trend_alignment=self._score_htf_alignment(decision, trends, notes),
            market_location=self._score_market_location(decision, location, atr, notes),
            liquidity_context=self._score_liquidity_context(frame, index, decision, location, atr, notes),
            session_quality=self._score_session(timestamp, notes),
            volatility_quality=0.0,
            structure_quality=self._score_structure(row, decision, notes),
        )
        vol_score, vol_metrics = self._score_volatility(frame, index, atr_series, notes)

        components = ContextComponents(
            htf_trend_alignment=components.htf_trend_alignment,
            market_location=components.market_location,
            liquidity_context=components.liquidity_context,
            session_quality=components.session_quality,
            volatility_quality=vol_score,
            structure_quality=components.structure_quality,
        )
        total_score = round(min(components.total, 100.0), 1)
        grade = self._assign_grade(total_score)

        return SignalContext(
            signal_index=index,
            signal_date=str(row.get("Date")),
            decision=decision,
            context_score=total_score,
            context_grade=grade.value,
            components=components.as_dict(),
            reasoning=notes,
            multi_timeframe=trends,
            market_location=location,
            session=self._session_label(timestamp).value,
            volatility=vol_metrics,
        )

    def run(self, frame: pd.DataFrame, source_csv: str = "") -> MarketContextReport:
        """Evaluate context for all actionable signals in a pipeline frame."""
        started = time.perf_counter()
        if frame.empty:
            raise MarketContextError("Pipeline frame is empty.")

        evaluated = (
            frame if "Decision" in frame.columns else self.decision_engine.evaluate(frame.copy())
        )
        working = evaluated.reset_index(drop=True)

        logger.info("Building HTF trend context for %s candles.", len(working))
        htf_lookup = self._build_htf_trend_lookup(working)
        atr_series = self._compute_atr(working)

        contexts: list[SignalContext] = []
        for index in range(len(working)):
            context = self.evaluate_signal(working, index, htf_lookup, atr_series)
            if context is not None:
                contexts.append(context)

        grades = [context.context_grade for context in contexts]
        grade_distribution = {grade.value: grades.count(grade.value) for grade in ContextGrade}
        average_score = (
            round(sum(context.context_score for context in contexts) / len(contexts), 1)
            if contexts
            else 0.0
        )

        return MarketContextReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=source_csv,
            total_candles=len(working),
            total_signals=len(contexts),
            average_context_score=average_score,
            grade_distribution=grade_distribution,
            execution_time_seconds=round(time.perf_counter() - started, 3),
            signals=[context.as_dict() for context in contexts],
        )

    def run_from_csv(self, pipeline_csv: Path | str) -> MarketContextReport:
        """Load pipeline CSV and evaluate market context."""
        csv_path = Path(pipeline_csv)
        frame = self.decision_engine.load_pipeline_csv(csv_path)
        return self.run(frame, source_csv=str(csv_path))


def generate_context_report(
    pipeline_csv: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5",
) -> MarketContextReport:
    """Run market context evaluation and export JSON report."""
    engine = MarketContextEngine(symbol=symbol, timeframe=timeframe)
    csv_path = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
    report = engine.run_from_csv(csv_path)

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Market context evaluation completed: signals=%s avg_score=%s",
        report.total_signals,
        report.average_context_score,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_context_report()
        print("Market Context Evaluation Summary")
        print(f"Symbol: {report.symbol} | Timeframe: {report.timeframe}")
        print(f"Candles: {report.total_candles}")
        print(f"Signals: {report.total_signals}")
        print(f"Average Context Score: {report.average_context_score}")
        print("Grade Distribution:")
        for grade, count in report.grade_distribution.items():
            if count:
                print(f"  - {grade}: {count}")
        if report.signals:
            sample = report.signals[0]
            print("Sample Context:")
            print(f"  Score: {sample['context_score']} | Grade: {sample['context_grade']}")
            print(f"  Reason: {'; '.join(sample['reasoning'][:6])}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except MarketContextError as exc:
        logger.error("Market context evaluation error: %s", exc)
        print(f"Market context evaluation error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected market context evaluation failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
