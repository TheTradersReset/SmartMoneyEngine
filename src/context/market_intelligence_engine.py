"""
Market Intelligence Layer V1 for SmartMoneyEngine.

Evaluates institutional market context for every candle without generating
trades, entries, exits, targets, or stop-loss levels.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "context" / "market_intelligence_report.json"

LOCATION_LOOKBACK_BARS = 200
ATR_PERIOD = 14
ATR_LOOKBACK = 100
RSI_PERIOD = 14
EMA_SLOPE_LOOKBACK = 5
SAMPLE_SUMMARY_COUNT = 12
TOP_EXAMPLE_COUNT = 10

INTELLIGENCE_WEIGHTS: dict[str, int] = {
    "trend": 25,
    "momentum": 25,
    "location": 20,
    "volatility": 15,
    "ema_structure": 15,
}


class MarketIntelligenceError(Exception):
    """Raised when market intelligence evaluation fails."""


class TrendState(str, Enum):
    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


class MomentumState(str, Enum):
    STRONG_BULLISH = "Strong Bullish"
    BULLISH = "Bullish"
    NEUTRAL = "Neutral"
    BEARISH = "Bearish"
    STRONG_BEARISH = "Strong Bearish"


class MarketLocation(str, Enum):
    NEAR_SUPPORT = "Near Support"
    NEAR_RESISTANCE = "Near Resistance"
    MID_RANGE = "Mid Range"


class RsiState(str, Enum):
    OVERSOLD = "Oversold"
    WEAK = "Weak"
    NEUTRAL = "Neutral"
    STRONG = "Strong"
    OVERBOUGHT = "Overbought"


class EmaStructure(str, Enum):
    BULL_STACK = "Bull Stack"
    BEAR_STACK = "Bear Stack"
    MIXED = "Mixed"


class SessionState(str, Enum):
    OPENING = "Opening"
    MIDDAY = "Midday"
    CLOSING = "Closing"
    OUTSIDE = "Outside"


class VolatilityState(str, Enum):
    LOW = "Low"
    NORMAL = "Normal"
    HIGH = "High"


@dataclass(frozen=True)
class IntelligenceComponents:
    """Weighted sub-scores contributing to the intelligence score."""

    trend: float
    momentum: float
    location: float
    volatility: float
    ema_structure: float

    @property
    def total(self) -> float:
        return (
            self.trend * INTELLIGENCE_WEIGHTS["trend"]
            + self.momentum * INTELLIGENCE_WEIGHTS["momentum"]
            + self.location * INTELLIGENCE_WEIGHTS["location"]
            + self.volatility * INTELLIGENCE_WEIGHTS["volatility"]
            + self.ema_structure * INTELLIGENCE_WEIGHTS["ema_structure"]
        ) / 100.0

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CandleIntelligence:
    """Market intelligence evaluation for one candle."""

    index: int
    timestamp: str
    close: float
    trend_state: str
    momentum_state: str
    market_location: str
    rsi_state: str
    ema_structure: str
    session_state: str
    volatility_state: str
    intelligence_score: float
    components: dict[str, float]
    summary: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload


@dataclass
class MarketIntelligenceReport:
    """Aggregate market intelligence report."""

    symbol: str
    timeframe: str
    source_csv: str
    total_candles: int
    average_intelligence_score: float
    score_distribution: dict[str, int]
    state_distribution: dict[str, dict[str, int]]
    top_bullish_examples: list[dict[str, Any]]
    top_bearish_examples: list[dict[str, Any]]
    sample_summaries: list[dict[str, Any]]
    execution_time_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class MarketIntelligenceEngine:
    """
    Evaluate per-candle market intelligence context.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Source timeframe label such as ``5M``.
    location_lookback_bars : int, optional
        Bars used to derive support and resistance context.
    """

    def __init__(
        self,
        symbol: str = "NIFTY50",
        timeframe: str = "5M",
        location_lookback_bars: int = LOCATION_LOOKBACK_BARS,
    ) -> None:
        self.symbol = symbol
        self.timeframe = timeframe
        self.location_lookback_bars = location_lookback_bars

    @staticmethod
    def _is_active(value: Any) -> bool:
        if value is None or pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "false", "0", "nan", "none"}:
                return False
        return True

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
        if trend == "BULLISH":
            return TrendState.BULLISH.value
        if trend == "BEARISH":
            return TrendState.BEARISH.value
        return TrendState.NEUTRAL.value

    @staticmethod
    def _session_state(timestamp: pd.Timestamp) -> SessionState:
        ts = timestamp
        if ts.tzinfo is None:
            ts = ts.tz_localize("Asia/Kolkata")
        else:
            ts = ts.tz_convert("Asia/Kolkata")
        minutes = ts.hour * 60 + ts.minute
        if 9 * 60 + 15 <= minutes < 10 * 60 + 30:
            return SessionState.OPENING
        if 10 * 60 + 30 <= minutes < 14 * 60 + 30:
            return SessionState.MIDDAY
        if 14 * 60 + 30 <= minutes <= 15 * 60 + 30:
            return SessionState.CLOSING
        return SessionState.OUTSIDE

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, pd.NA)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _compute_atr(frame: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
        high = frame["High"].astype(float)
        low = frame["Low"].astype(float)
        close = frame["Close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
            axis=1,
        ).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    @staticmethod
    def _compute_vwap(frame: pd.DataFrame, timestamps: pd.Series) -> pd.Series:
        typical = (
            frame["High"].astype(float)
            + frame["Low"].astype(float)
            + frame["Close"].astype(float)
        ) / 3.0
        volume = frame["Volume"].astype(float).fillna(0.0)
        session_day = timestamps.dt.tz_convert("Asia/Kolkata").dt.date
        cumulative_tpv = (typical * volume).groupby(session_day).cumsum()
        cumulative_volume = volume.groupby(session_day).cumsum().replace(0, pd.NA)
        return cumulative_tpv / cumulative_volume

    def enrich(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Attach indicator columns required for intelligence evaluation."""
        working = frame.reset_index(drop=True).copy()
        close = working["Close"].astype(float)
        working["_timestamp"] = pd.to_datetime(working["Date"], errors="coerce")
        working["EMA20"] = close.ewm(span=20, adjust=False).mean()
        working["EMA50"] = close.ewm(span=50, adjust=False).mean()
        working["EMA200"] = close.ewm(span=200, adjust=False).mean()
        working["RSI"] = self._compute_rsi(close)
        working["_atr"] = self._compute_atr(working)
        working["VWAP"] = self._compute_vwap(working, working["_timestamp"])
        working["_ema20_slope"] = working["EMA20"].diff(EMA_SLOPE_LOOKBACK)
        working["_displacement"] = (working["Close"].astype(float) - working["Open"].astype(float)) / (
            working["_atr"].replace(0, pd.NA)
        )
        return working

    def _collect_levels(self, window: pd.DataFrame, column: str) -> list[float]:
        values: list[float] = []
        if column not in window.columns:
            return values
        for value in window[column]:
            parsed = self._to_float(value)
            if parsed is not None and self._is_active(value):
                values.append(parsed)
        return values

    def _market_levels(self, frame: pd.DataFrame, index: int) -> dict[str, float | None]:
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

        return {
            "close": close,
            "major_support": major_support,
            "major_resistance": major_resistance,
        }

    @staticmethod
    def _ema_structure(ema20: float, ema50: float, ema200: float) -> EmaStructure:
        if ema20 > ema50 > ema200:
            return EmaStructure.BULL_STACK
        if ema20 < ema50 < ema200:
            return EmaStructure.BEAR_STACK
        return EmaStructure.MIXED

    @staticmethod
    def _rsi_state(rsi: float) -> RsiState:
        if rsi < 30:
            return RsiState.OVERSOLD
        if rsi < 40:
            return RsiState.WEAK
        if rsi < 60:
            return RsiState.NEUTRAL
        if rsi < 70:
            return RsiState.STRONG
        return RsiState.OVERBOUGHT

    def _volatility_state(self, atr_series: pd.Series, index: int) -> VolatilityState:
        start = max(0, index - ATR_LOOKBACK + 1)
        window = atr_series.iloc[start : index + 1].dropna()
        if window.empty:
            return VolatilityState.NORMAL
        current = atr_series.iloc[index]
        if pd.isna(current):
            return VolatilityState.NORMAL
        percentile = (window <= current).sum() / len(window) * 100
        if percentile <= 33:
            return VolatilityState.LOW
        if percentile <= 66:
            return VolatilityState.NORMAL
        return VolatilityState.HIGH

    def _trend_state(
        self,
        close: float,
        ema20: float,
        ema50: float,
        ema200: float,
        vwap: float | None,
        pipeline_trend: str,
        ema_structure: EmaStructure,
    ) -> TrendState:
        bull_points = 0
        bear_points = 0

        if close > ema20:
            bull_points += 1
        else:
            bear_points += 1
        if close > ema50:
            bull_points += 1
        else:
            bear_points += 1
        if vwap is not None:
            if close >= vwap:
                bull_points += 1
            else:
                bear_points += 1
        if ema_structure == EmaStructure.BULL_STACK:
            bull_points += 2
        elif ema_structure == EmaStructure.BEAR_STACK:
            bear_points += 2
        if pipeline_trend == TrendState.BULLISH.value:
            bull_points += 1
        elif pipeline_trend == TrendState.BEARISH.value:
            bear_points += 1

        if bull_points >= bear_points + 2:
            return TrendState.BULLISH
        if bear_points >= bull_points + 2:
            return TrendState.BEARISH
        return TrendState.NEUTRAL

    def _momentum_state(
        self,
        rsi: float,
        ema20_slope: float,
        displacement: float,
    ) -> MomentumState:
        score = 0.0
        if rsi >= 70:
            score += 2
        elif rsi >= 58:
            score += 1
        elif rsi <= 30:
            score -= 2
        elif rsi <= 42:
            score -= 1

        if ema20_slope > 0:
            score += 1
        elif ema20_slope < 0:
            score -= 1

        if displacement >= 0.35:
            score += 1
        elif displacement <= -0.35:
            score -= 1

        if score >= 2:
            return MomentumState.STRONG_BULLISH
        if score == 1:
            return MomentumState.BULLISH
        if score <= -2:
            return MomentumState.STRONG_BEARISH
        if score == -1:
            return MomentumState.BEARISH
        return MomentumState.NEUTRAL

    def _market_location(
        self,
        levels: dict[str, float | None],
        atr: float,
    ) -> MarketLocation:
        close = levels["close"]
        support = levels["major_support"]
        resistance = levels["major_resistance"]
        if atr <= 0:
            return MarketLocation.MID_RANGE

        support_distance = abs(close - support) if support is not None else None
        resistance_distance = abs(resistance - close) if resistance is not None else None

        near_support = support_distance is not None and support_distance <= atr * 0.5
        near_resistance = resistance_distance is not None and resistance_distance <= atr * 0.5

        if near_support and not near_resistance:
            return MarketLocation.NEAR_SUPPORT
        if near_resistance and not near_support:
            return MarketLocation.NEAR_RESISTANCE
        if near_support and near_resistance:
            if (support_distance or 0) <= (resistance_distance or 0):
                return MarketLocation.NEAR_SUPPORT
            return MarketLocation.NEAR_RESISTANCE
        return MarketLocation.MID_RANGE

    @staticmethod
    def _component_scores(
        trend_state: TrendState,
        momentum_state: MomentumState,
        market_location: MarketLocation,
        volatility_state: VolatilityState,
        ema_structure: EmaStructure,
    ) -> IntelligenceComponents:
        trend_map = {
            TrendState.BULLISH: 85.0,
            TrendState.NEUTRAL: 50.0,
            TrendState.BEARISH: 15.0,
        }
        momentum_map = {
            MomentumState.STRONG_BULLISH: 95.0,
            MomentumState.BULLISH: 75.0,
            MomentumState.NEUTRAL: 50.0,
            MomentumState.BEARISH: 25.0,
            MomentumState.STRONG_BEARISH: 5.0,
        }
        volatility_map = {
            VolatilityState.LOW: 55.0,
            VolatilityState.NORMAL: 75.0,
            VolatilityState.HIGH: 60.0,
        }
        ema_map = {
            EmaStructure.BULL_STACK: 85.0,
            EmaStructure.MIXED: 50.0,
            EmaStructure.BEAR_STACK: 15.0,
        }

        if trend_state == TrendState.BULLISH:
            location_map = {
                MarketLocation.NEAR_SUPPORT: 85.0,
                MarketLocation.MID_RANGE: 65.0,
                MarketLocation.NEAR_RESISTANCE: 35.0,
            }
        elif trend_state == TrendState.BEARISH:
            location_map = {
                MarketLocation.NEAR_RESISTANCE: 85.0,
                MarketLocation.MID_RANGE: 65.0,
                MarketLocation.NEAR_SUPPORT: 35.0,
            }
        else:
            location_map = {
                MarketLocation.MID_RANGE: 60.0,
                MarketLocation.NEAR_SUPPORT: 55.0,
                MarketLocation.NEAR_RESISTANCE: 55.0,
            }

        return IntelligenceComponents(
            trend=trend_map[trend_state],
            momentum=momentum_map[momentum_state],
            location=location_map[market_location],
            volatility=volatility_map[volatility_state],
            ema_structure=ema_map[ema_structure],
        )

    @staticmethod
    def _build_summary(
        trend_state: TrendState,
        momentum_state: MomentumState,
        market_location: MarketLocation,
        rsi_state: RsiState,
        ema_structure: EmaStructure,
        session_state: SessionState,
        volatility_state: VolatilityState,
        close: float,
        ema20: float,
    ) -> str:
        sentences: list[str] = []

        if momentum_state in {MomentumState.STRONG_BEARISH, MomentumState.BEARISH}:
            sentences.append(f"{momentum_state.value} momentum.")
        elif momentum_state in {MomentumState.STRONG_BULLISH, MomentumState.BULLISH}:
            sentences.append(f"{momentum_state.value} momentum.")
        else:
            sentences.append("Momentum is balanced.")

        if close < ema20 and ema_structure == EmaStructure.BEAR_STACK:
            sentences.append("Price trading below EMA structure.")
        elif close > ema20 and ema_structure == EmaStructure.BULL_STACK:
            sentences.append("Bullish trend aligned across EMA structure.")
        else:
            sentences.append("EMA structure is mixed.")

        if rsi_state in {RsiState.WEAK, RsiState.OVERSOLD}:
            sentences.append("RSI weakening.")
        elif rsi_state in {RsiState.STRONG, RsiState.OVERBOUGHT}:
            sentences.append("RSI strengthening.")
        else:
            sentences.append("RSI neutral.")

        if market_location == MarketLocation.NEAR_SUPPORT:
            sentences.append("Approaching support zone.")
        elif market_location == MarketLocation.NEAR_RESISTANCE:
            sentences.append("Approaching resistance zone.")
        else:
            sentences.append("Price has room within the current range.")

        if trend_state == TrendState.BULLISH and market_location != MarketLocation.NEAR_RESISTANCE:
            sentences.append("Conditions favorable for continuation.")
        elif trend_state == TrendState.BEARISH and market_location != MarketLocation.NEAR_SUPPORT:
            sentences.append("Fresh shorts carry reduced reward.")
        elif trend_state == TrendState.NEUTRAL:
            sentences.append("Market lacks directional conviction.")
        else:
            sentences.append("Reward-to-risk is compressed at current location.")

        if volatility_state == VolatilityState.HIGH:
            sentences.append(f"{session_state.value} session volatility is elevated.")
        elif session_state == SessionState.OPENING:
            sentences.append("Opening session price discovery in progress.")

        return " ".join(sentences[:5])

    def evaluate_bar(self, frame: pd.DataFrame, index: int) -> CandleIntelligence:
        """Evaluate market intelligence for one candle."""
        row = frame.iloc[index]
        close = float(row["Close"])
        ema20 = float(row["EMA20"])
        ema50 = float(row["EMA50"])
        ema200 = float(row["EMA200"])
        rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else 50.0
        atr = float(row["_atr"]) if pd.notna(row["_atr"]) else 1.0
        vwap = self._to_float(row["VWAP"])
        ema20_slope = float(row["_ema20_slope"]) if pd.notna(row["_ema20_slope"]) else 0.0
        displacement = float(row["_displacement"]) if pd.notna(row["_displacement"]) else 0.0
        timestamp = row["_timestamp"]
        pipeline_trend = (
            self._normalize_trend(row["Trend"]) if "Trend" in frame.columns else TrendState.NEUTRAL.value
        )

        ema_structure = self._ema_structure(ema20, ema50, ema200)
        trend_state = self._trend_state(
            close, ema20, ema50, ema200, vwap, pipeline_trend, ema_structure
        )
        momentum_state = self._momentum_state(rsi, ema20_slope, displacement)
        levels = self._market_levels(frame, index)
        market_location = self._market_location(levels, atr)
        rsi_state = self._rsi_state(rsi)
        session_state = (
            self._session_state(timestamp) if pd.notna(timestamp) else SessionState.OUTSIDE
        )
        volatility_state = self._volatility_state(frame["_atr"], index)
        components = self._component_scores(
            trend_state,
            momentum_state,
            market_location,
            volatility_state,
            ema_structure,
        )
        score = round(components.total, 2)
        summary = self._build_summary(
            trend_state,
            momentum_state,
            market_location,
            rsi_state,
            ema_structure,
            session_state,
            volatility_state,
            close,
            ema20,
        )

        return CandleIntelligence(
            index=index,
            timestamp=str(row["Date"]),
            close=round(close, 2),
            trend_state=trend_state.value,
            momentum_state=momentum_state.value,
            market_location=market_location.value,
            rsi_state=rsi_state.value,
            ema_structure=ema_structure.value,
            session_state=session_state.value,
            volatility_state=volatility_state.value,
            intelligence_score=score,
            components=components.as_dict(),
            summary=summary,
        )

    def evaluate(self, frame: pd.DataFrame) -> list[CandleIntelligence]:
        """Evaluate market intelligence for every candle."""
        enriched = self.enrich(frame)
        return [self.evaluate_bar(enriched, index) for index in range(len(enriched))]

    @staticmethod
    def _score_bucket(score: float) -> str:
        if score < 20:
            return "0-19"
        if score < 40:
            return "20-39"
        if score < 60:
            return "40-59"
        if score < 80:
            return "60-79"
        return "80-100"

    def build_report(
        self,
        evaluations: list[CandleIntelligence],
        source_csv: Path | str,
        execution_time_seconds: float,
    ) -> MarketIntelligenceReport:
        """Build aggregate report from per-candle evaluations."""
        score_distribution = Counter(
            self._score_bucket(item.intelligence_score) for item in evaluations
        )
        state_distribution = {
            "trend_state": Counter(item.trend_state for item in evaluations),
            "momentum_state": Counter(item.momentum_state for item in evaluations),
            "market_location": Counter(item.market_location for item in evaluations),
            "rsi_state": Counter(item.rsi_state for item in evaluations),
            "ema_structure": Counter(item.ema_structure for item in evaluations),
            "session_state": Counter(item.session_state for item in evaluations),
            "volatility_state": Counter(item.volatility_state for item in evaluations),
        }

        ranked = sorted(evaluations, key=lambda item: item.intelligence_score, reverse=True)
        top_bullish = [item.as_dict() for item in ranked[:TOP_EXAMPLE_COUNT]]
        top_bearish = [item.as_dict() for item in ranked[-TOP_EXAMPLE_COUNT:][::-1]]

        if evaluations:
            step = max(1, len(evaluations) // SAMPLE_SUMMARY_COUNT)
            sample_indices = list(range(0, len(evaluations), step))[:SAMPLE_SUMMARY_COUNT]
        else:
            sample_indices = []

        sample_summaries = [
            {
                "timestamp": evaluations[index].timestamp,
                "intelligence_score": evaluations[index].intelligence_score,
                "summary": evaluations[index].summary,
            }
            for index in sample_indices
        ]

        average_score = (
            round(sum(item.intelligence_score for item in evaluations) / len(evaluations), 2)
            if evaluations
            else 0.0
        )

        return MarketIntelligenceReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=str(source_csv),
            total_candles=len(evaluations),
            average_intelligence_score=average_score,
            score_distribution=dict(sorted(score_distribution.items())),
            state_distribution={
                key: dict(sorted(counter.items())) for key, counter in state_distribution.items()
            },
            top_bullish_examples=top_bullish,
            top_bearish_examples=top_bearish,
            sample_summaries=sample_summaries,
            execution_time_seconds=round(execution_time_seconds, 3),
        )

    def run(self, frame: pd.DataFrame) -> MarketIntelligenceReport:
        """Evaluate frame and build report metadata."""
        started = time.perf_counter()
        evaluations = self.evaluate(frame)
        return self.build_report(frame, DEFAULT_PIPELINE_CSV, time.perf_counter() - started)


def generate_market_intelligence_report(
    pipeline_csv: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5M",
) -> MarketIntelligenceReport:
    """Evaluate market intelligence and export JSON report."""
    source = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
    if not source.exists():
        raise MarketIntelligenceError(f"Pipeline CSV not found: {source}")

    frame = pd.read_csv(source)
    engine = MarketIntelligenceEngine(symbol=symbol, timeframe=timeframe)
    started = time.perf_counter()
    evaluations = engine.evaluate(frame)
    report = engine.build_report(
        evaluations,
        source_csv=source,
        execution_time_seconds=time.perf_counter() - started,
    )

    destination = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Market intelligence completed: candles=%s avg_score=%s",
        report.total_candles,
        report.average_intelligence_score,
    )
    return report


def main() -> int:
    """CLI entry point."""
    try:
        report = generate_market_intelligence_report()
        print("Market Intelligence Summary")
        print(f"Symbol: {report.symbol} | Timeframe: {report.timeframe}")
        print(f"Candles: {report.total_candles}")
        print(f"Average Score: {report.average_intelligence_score}")
        print("Score Distribution:")
        for bucket, count in report.score_distribution.items():
            print(f"  {bucket}: {count}")
        print("Trend Distribution:")
        for state, count in report.state_distribution["trend_state"].items():
            print(f"  {state}: {count}")
        if report.top_bullish_examples:
            best = report.top_bullish_examples[0]
            print(f"Top Bullish Example: score={best['intelligence_score']} @ {best['timestamp']}")
        if report.top_bearish_examples:
            worst = report.top_bearish_examples[0]
            print(f"Top Bearish Example: score={worst['intelligence_score']} @ {worst['timestamp']}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        return 0
    except MarketIntelligenceError as exc:
        logger.error("Market intelligence error: %s", exc)
        print(f"Market intelligence error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected market intelligence failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
