"""
Production BUY_V3 signal engine wrapper.

Frozen candidate: Failed Breakdown + Gap Reversal + Liquidity Grab + Near Support + PDL Sweep.
Paper signal mode only — delegates to research replay engine logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.research.buy_v3_candidate_validation_research import (
    BUY_V3_FORMULA_TEXT,
    BUY_V3_MODEL_ID,
    BuyV3CandidateEngine,
    _evaluate_buy_bar_fast,
)

ENGINE_VERSION = "BUY_V3"
MODEL_ID = BUY_V3_MODEL_ID
STOP_POINTS = 10.0
TARGET_STRUCTURE = "60/100/Runner"
TARGET1_POINTS = 60.0
TARGET2_POINTS = 100.0


@dataclass(frozen=True)
class BuyV3Signal:
    """Normalized BUY_V3 paper signal."""

    timestamp: str
    direction: str
    entry: float
    stop: float
    target1: float
    target2: float
    target_structure: str
    confidence: float
    engine_version: str
    raw: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "direction": self.direction,
            "entry": self.entry,
            "stop": self.stop,
            "target1": self.target1,
            "target2": self.target2,
            "target_structure": self.target_structure,
            "confidence": self.confidence,
            "engine_version": self.engine_version,
            "formula": BUY_V3_FORMULA_TEXT,
            "raw": self.raw,
        }


class BuyV3Engine:
    """Evaluate BUY_V3 on the latest closed bar."""

    def __init__(self) -> None:
        self._engine = BuyV3CandidateEngine()
        self._emitted_bars: set[int] = set()

    @property
    def model_id(self) -> str:
        return MODEL_ID

    def evaluate_bar(
        self,
        *,
        frame,
        bar: int,
        context: dict[str, str],
        lookback_events: set[str],
        bar_events: set[str],
    ) -> dict[str, Any]:
        return _evaluate_buy_bar_fast(
            self._engine,
            frame=frame,
            bar=bar,
            context=context,
            lookback_events=lookback_events,
            bar_events=bar_events,
            emitted_bars=self._emitted_bars,
        )

    def to_signal(self, evaluation: dict[str, Any]) -> BuyV3Signal | None:
        if evaluation.get("verdict") != "BUY":
            return None
        entry = float(evaluation.get("layer4", {}).get("entry") or evaluation.get("context", {}).get("close") or 0.0)
        if entry <= 0:
            return None
        stop = round(entry - STOP_POINTS, 2)
        target1 = round(entry + TARGET1_POINTS, 2)
        target2 = round(entry + TARGET2_POINTS, 2)
        confidence = 1.0 if evaluation.get("layer5", {}).get("pass") else 0.5
        return BuyV3Signal(
            timestamp=str(evaluation.get("timestamp", "")),
            direction="BUY",
            entry=round(entry, 2),
            stop=stop,
            target1=target1,
            target2=target2,
            target_structure=TARGET_STRUCTURE,
            confidence=confidence,
            engine_version=ENGINE_VERSION,
            raw=evaluation,
        )

    def mark_emitted(self, bar: int) -> None:
        self._emitted_bars.add(bar)
