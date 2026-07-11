"""
SmartMoneyEngine decision layer.

Converts enriched SMC pipeline outputs into per-candle trade decisions using
only structural Smart Money Concepts logic.
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

logger = logging.getLogger("SmartMoneyEngine")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PIPELINE_CSV = PROJECT_ROOT / "outputs" / "pipeline" / "NIFTY50_5m_pipeline.csv"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "outputs" / "signals" / "decision_report.json"

DECISION_THRESHOLD = 55
DECISION_MARGIN = 10
STRONG_SCORE_THRESHOLD = 70
WEAK_SCORE_THRESHOLD = 45


class MarketBias(str, Enum):
    """Directional market bias derived from trend structure."""

    BULLISH = "Bullish"
    BEARISH = "Bearish"
    NEUTRAL = "Neutral"


class InstitutionalBias(str, Enum):
    """Institutional conviction derived from SMC confluence."""

    STRONG_BULLISH = "Strong Bullish"
    WEAK_BULLISH = "Weak Bullish"
    STRONG_BEARISH = "Strong Bearish"
    WEAK_BEARISH = "Weak Bearish"
    NEUTRAL = "Neutral"


class TradeDecision(str, Enum):
    """Actionable trade decision."""

    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"


class DecisionEngineError(Exception):
    """Raised when decision evaluation fails."""


@dataclass(frozen=True)
class DecisionResult:
    """Decision output for a single candle."""

    decision: TradeDecision
    market_bias: MarketBias
    institutional_bias: InstitutionalBias
    setup_quality_score: int
    confidence: float
    reason: str
    bullish_score: float
    bearish_score: float


@dataclass
class DecisionReport:
    """Aggregate decision report."""

    symbol: str
    timeframe: str
    source_csv: str
    rows: int
    execution_time_seconds: float
    decisions: dict[str, int] = field(default_factory=dict)
    market_bias_counts: dict[str, int] = field(default_factory=dict)
    institutional_bias_counts: dict[str, int] = field(default_factory=dict)
    average_setup_quality: float = 0.0
    average_confidence: float = 0.0
    buy_count: int = 0
    sell_count: int = 0
    wait_count: int = 0
    output_columns: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable report dictionary."""
        return asdict(self)


class DecisionEngine:
    """
    Evaluate SMC pipeline outputs and produce trade decisions.

    Parameters
    ----------
    symbol : str, optional
        Symbol label for reporting.
    timeframe : str, optional
        Timeframe label for reporting.
    """

    REQUIRED_PIPELINE_COLUMNS = (
        "Trend",
        "Trend_Strength",
        "Bullish_BOS",
        "Bearish_BOS",
        "Bullish_CHOCH",
        "Bearish_CHOCH",
        "Bullish_FVG_Top",
        "Bearish_FVG_Top",
        "Bullish_OB_High",
        "Bearish_OB_High",
        "Bullish_OB_Mitigated",
        "Bearish_OB_Mitigated",
        "Buy_Liquidity_Sweep",
        "Sell_Liquidity_Sweep",
        "Liquidity_Strength",
    )

    def __init__(self, symbol: str = "NIFTY50", timeframe: str = "5") -> None:
        self.symbol = symbol
        self.timeframe = timeframe

    @staticmethod
    def _is_active(value: Any) -> bool:
        """Return whether a signal value is active on a candle."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)) and not pd.isna(value):
            return True
        if isinstance(value, str) and value.strip():
            return value.strip().lower() not in {"false", "0", "nan", "none"}
        return False

    @staticmethod
    def _trend_market_bias(trend_value: Any) -> MarketBias:
        """Map trend engine output to market bias."""
        trend = str(trend_value).strip().upper()
        if trend == "BULLISH":
            return MarketBias.BULLISH
        if trend == "BEARISH":
            return MarketBias.BEARISH
        return MarketBias.NEUTRAL

    @staticmethod
    def _trend_strength_points(strength: Any) -> float:
        """Convert trend strength into scoring points."""
        if strength is None or pd.isna(strength):
            return 0.0
        try:
            numeric = float(strength)
        except (TypeError, ValueError):
            return 0.0
        return min(max(numeric, 0.0), 3.0) * 5.0

    def _validate_frame(self, frame: pd.DataFrame) -> None:
        """Ensure required SMC columns exist."""
        missing = [
            column
            for column in self.REQUIRED_PIPELINE_COLUMNS
            if column not in frame.columns
        ]
        if missing:
            raise DecisionEngineError(
                f"Pipeline dataframe missing required SMC columns: {missing}"
            )

    def _score_row(self, row: pd.Series) -> tuple[float, float, list[str], list[str]]:
        """
        Score bullish and bearish institutional confluence for one candle.

        Returns
        -------
        tuple[float, float, list[str], list[str]]
            Bullish score, bearish score, bullish reasons, bearish reasons.
        """
        bullish_score = 0.0
        bearish_score = 0.0
        bull_reasons: list[str] = []
        bear_reasons: list[str] = []

        market_bias = self._trend_market_bias(row.get("Trend"))
        if market_bias == MarketBias.BULLISH:
            bullish_score += 15
            bull_reasons.append("Trend bullish")
        elif market_bias == MarketBias.BEARISH:
            bearish_score += 15
            bear_reasons.append("Trend bearish")

        trend_points = self._trend_strength_points(row.get("Trend_Strength"))
        if market_bias == MarketBias.BULLISH:
            bullish_score += trend_points
            if trend_points:
                bull_reasons.append(f"Trend strength {int(row.get('Trend_Strength', 0))}")
        elif market_bias == MarketBias.BEARISH:
            bearish_score += trend_points
            if trend_points:
                bear_reasons.append(f"Trend strength {int(row.get('Trend_Strength', 0))}")

        if self._is_active(row.get("Bullish_BOS")):
            bullish_score += 12
            bull_reasons.append("Bullish BOS")
        if self._is_active(row.get("Bearish_BOS")):
            bearish_score += 12
            bear_reasons.append("Bearish BOS")

        if self._is_active(row.get("Bullish_CHOCH")):
            bullish_score += 10
            bull_reasons.append("Bullish CHOCH")
        if self._is_active(row.get("Bearish_CHOCH")):
            bearish_score += 10
            bear_reasons.append("Bearish CHOCH")

        if self._is_active(row.get("Bullish_FVG_Top")):
            bullish_score += 8
            bull_reasons.append("Bullish FVG")
        if self._is_active(row.get("Bearish_FVG_Top")):
            bearish_score += 8
            bear_reasons.append("Bearish FVG")

        bullish_ob_active = self._is_active(row.get("Bullish_OB_High")) and not self._is_active(
            row.get("Bullish_OB_Mitigated")
        )
        bearish_ob_active = self._is_active(row.get("Bearish_OB_High")) and not self._is_active(
            row.get("Bearish_OB_Mitigated")
        )
        if bullish_ob_active:
            bullish_score += 12
            bull_reasons.append("Active bullish order block")
        if bearish_ob_active:
            bearish_score += 12
            bear_reasons.append("Active bearish order block")

        if self._is_active(row.get("Sell_Liquidity_Sweep")):
            bullish_score += 8
            bull_reasons.append("Sell-side liquidity sweep")
        if self._is_active(row.get("Buy_Liquidity_Sweep")):
            bearish_score += 8
            bear_reasons.append("Buy-side liquidity sweep")

        liquidity_strength = row.get("Liquidity_Strength")
        if liquidity_strength is not None and not pd.isna(liquidity_strength):
            try:
                strength = float(liquidity_strength)
            except (TypeError, ValueError):
                strength = 0.0
            if strength > 0:
                if self._is_active(row.get("Sell_Liquidity_Sweep")):
                    bullish_score += strength * 2
                if self._is_active(row.get("Buy_Liquidity_Sweep")):
                    bearish_score += strength * 2

        return bullish_score, bearish_score, bull_reasons, bear_reasons

    @staticmethod
    def _institutional_bias(
        market_bias: MarketBias,
        bullish_score: float,
        bearish_score: float,
    ) -> InstitutionalBias:
        """Derive institutional bias from directional scores."""
        dominant = max(bullish_score, bearish_score)
        if dominant < WEAK_SCORE_THRESHOLD:
            return InstitutionalBias.NEUTRAL

        if bullish_score > bearish_score:
            if bullish_score >= STRONG_SCORE_THRESHOLD and market_bias != MarketBias.BEARISH:
                return InstitutionalBias.STRONG_BULLISH
            return InstitutionalBias.WEAK_BULLISH

        if bearish_score > bullish_score:
            if bearish_score >= STRONG_SCORE_THRESHOLD and market_bias != MarketBias.BULLISH:
                return InstitutionalBias.STRONG_BEARISH
            return InstitutionalBias.WEAK_BEARISH

        return InstitutionalBias.NEUTRAL

    @staticmethod
    def _setup_quality_score(bullish_score: float, bearish_score: float) -> int:
        """Map directional confluence into a 0-100 setup quality score."""
        dominant = max(bullish_score, bearish_score)
        return int(min(max(round(dominant), 0), 100))

    def _decision_for_scores(
        self,
        market_bias: MarketBias,
        institutional_bias: InstitutionalBias,
        bullish_score: float,
        bearish_score: float,
        setup_quality: int,
    ) -> TradeDecision:
        """Determine BUY/SELL/WAIT from scores and bias."""
        if (
            bullish_score >= DECISION_THRESHOLD
            and bullish_score >= bearish_score + DECISION_MARGIN
            and market_bias != MarketBias.BEARISH
            and institutional_bias in {
                InstitutionalBias.STRONG_BULLISH,
                InstitutionalBias.WEAK_BULLISH,
            }
        ):
            return TradeDecision.BUY

        if (
            bearish_score >= DECISION_THRESHOLD
            and bearish_score >= bullish_score + DECISION_MARGIN
            and market_bias != MarketBias.BULLISH
            and institutional_bias in {
                InstitutionalBias.STRONG_BEARISH,
                InstitutionalBias.WEAK_BEARISH,
            }
        ):
            return TradeDecision.SELL

        if setup_quality < DECISION_THRESHOLD:
            return TradeDecision.WAIT

        return TradeDecision.WAIT

    @staticmethod
    def _confidence(decision: TradeDecision, bullish_score: float, bearish_score: float) -> float:
        """Compute decision confidence from dominant score."""
        dominant = max(bullish_score, bearish_score)
        confidence = dominant / 100.0
        if decision == TradeDecision.WAIT:
            confidence = max(0.0, 1.0 - abs(bullish_score - bearish_score) / 100.0) * 0.5
        return round(min(max(confidence, 0.0), 1.0), 3)

    @staticmethod
    def _build_reason(
        decision: TradeDecision,
        market_bias: MarketBias,
        institutional_bias: InstitutionalBias,
        bull_reasons: list[str],
        bear_reasons: list[str],
    ) -> str:
        """Build a human-readable reason string."""
        if decision == TradeDecision.BUY:
            active = bull_reasons
        elif decision == TradeDecision.SELL:
            active = bear_reasons
        else:
            active = bull_reasons + bear_reasons

        signal_text = ", ".join(active) if active else "Insufficient SMC confluence"
        return (
            f"{decision.value}; Market={market_bias.value}; "
            f"Institutional={institutional_bias.value}; Signals={signal_text}"
        )

    def evaluate_row(self, row: pd.Series) -> DecisionResult:
        """Evaluate one pipeline candle."""
        market_bias = self._trend_market_bias(row.get("Trend"))
        bullish_score, bearish_score, bull_reasons, bear_reasons = self._score_row(row)
        institutional_bias = self._institutional_bias(market_bias, bullish_score, bearish_score)
        setup_quality = self._setup_quality_score(bullish_score, bearish_score)
        decision = self._decision_for_scores(
            market_bias,
            institutional_bias,
            bullish_score,
            bearish_score,
            setup_quality,
        )
        confidence = self._confidence(decision, bullish_score, bearish_score)
        reason = self._build_reason(
            decision,
            market_bias,
            institutional_bias,
            bull_reasons,
            bear_reasons,
        )

        return DecisionResult(
            decision=decision,
            market_bias=market_bias,
            institutional_bias=institutional_bias,
            setup_quality_score=setup_quality,
            confidence=confidence,
            reason=reason,
            bullish_score=round(bullish_score, 2),
            bearish_score=round(bearish_score, 2),
        )

    def evaluate(self, frame: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluate all candles in a pipeline dataframe.

        Returns
        -------
        pd.DataFrame
            Input frame with decision columns appended.
        """
        self._validate_frame(frame)
        working = frame.copy()

        results = [self.evaluate_row(row) for _, row in working.iterrows()]

        working["Decision"] = [result.decision.value for result in results]
        working["Market_Bias"] = [result.market_bias.value for result in results]
        working["Institutional_Bias"] = [result.institutional_bias.value for result in results]
        working["Setup_Quality_Score"] = [result.setup_quality_score for result in results]
        working["Confidence"] = [result.confidence for result in results]
        working["Reason"] = [result.reason for result in results]
        working["Bullish_Score"] = [result.bullish_score for result in results]
        working["Bearish_Score"] = [result.bearish_score for result in results]

        return working

    @staticmethod
    def load_pipeline_csv(path: Path | str) -> pd.DataFrame:
        """Load an enriched pipeline CSV."""
        csv_path = Path(path)
        if not csv_path.exists():
            raise DecisionEngineError(f"Pipeline CSV not found: {csv_path}")
        frame = pd.read_csv(csv_path)
        if frame.empty:
            raise DecisionEngineError(f"Pipeline CSV is empty: {csv_path}")
        return frame

    def build_report(
        self,
        evaluated: pd.DataFrame,
        source_csv: Path | str,
        execution_time_seconds: float,
    ) -> DecisionReport:
        """Build an aggregate decision report."""
        decisions = evaluated["Decision"].value_counts().to_dict()
        market_counts = evaluated["Market_Bias"].value_counts().to_dict()
        institutional_counts = evaluated["Institutional_Bias"].value_counts().to_dict()

        return DecisionReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            source_csv=str(source_csv),
            rows=len(evaluated),
            execution_time_seconds=execution_time_seconds,
            decisions={str(key): int(value) for key, value in decisions.items()},
            market_bias_counts={str(key): int(value) for key, value in market_counts.items()},
            institutional_bias_counts={
                str(key): int(value) for key, value in institutional_counts.items()
            },
            average_setup_quality=round(float(evaluated["Setup_Quality_Score"].mean()), 2),
            average_confidence=round(float(evaluated["Confidence"].mean()), 3),
            buy_count=int((evaluated["Decision"] == TradeDecision.BUY.value).sum()),
            sell_count=int((evaluated["Decision"] == TradeDecision.SELL.value).sum()),
            wait_count=int((evaluated["Decision"] == TradeDecision.WAIT.value).sum()),
            output_columns=list(evaluated.columns),
        )


def evaluate_pipeline(
    pipeline_csv: Path | str | None = None,
    report_path: Path | str | None = None,
    symbol: str = "NIFTY50",
    timeframe: str = "5",
) -> tuple[pd.DataFrame, DecisionReport]:
    """
    Load pipeline output, evaluate decisions, and write the report JSON.

    Returns
    -------
    tuple[pd.DataFrame, DecisionReport]
        Evaluated dataframe and aggregate report.
    """
    csv_path = Path(pipeline_csv) if pipeline_csv is not None else DEFAULT_PIPELINE_CSV
    json_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH

    engine = DecisionEngine(symbol=symbol, timeframe=timeframe)
    started = time.perf_counter()
    frame = engine.load_pipeline_csv(csv_path)
    evaluated = engine.evaluate(frame)
    elapsed = time.perf_counter() - started
    report = engine.build_report(evaluated, csv_path, elapsed)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report.as_dict(), handle, indent=2)

    logger.info(
        "Decision evaluation completed in %.3fs: BUY=%s SELL=%s WAIT=%s",
        elapsed,
        report.buy_count,
        report.sell_count,
        report.wait_count,
    )
    return evaluated, report


def main() -> int:
    """CLI entry point."""
    try:
        evaluated, report = evaluate_pipeline()
        print("Decision Engine Summary")
        print(f"Rows: {report.rows}")
        print(f"BUY: {report.buy_count}")
        print(f"SELL: {report.sell_count}")
        print(f"WAIT: {report.wait_count}")
        print(f"Average Setup Quality: {report.average_setup_quality}")
        print(f"Average Confidence: {report.average_confidence}")
        print(f"Report: {DEFAULT_REPORT_PATH}")
        print(f"Sample Decision: {evaluated['Decision'].iloc[-1]}")
        return 0
    except DecisionEngineError as exc:
        logger.error("Decision engine error: %s", exc)
        print(f"Decision engine error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        logger.exception("Unexpected decision engine failure.")
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
