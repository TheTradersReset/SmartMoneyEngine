"""
Production SELL_V6 signal engine wrapper.

Frozen candidate: SELL_V5 stack with VWAP Below only gate.
Paper signal mode only — delegates to research replay engine logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.research.sell_v6_replay_validation_research import (
    SellV6CandidateEngine,
    V6_VWAP_GATE_RULE,
)

ENGINE_VERSION = "SELL_V6"
MODEL_ID = SellV6CandidateEngine.MODEL_ID
STOP_POINTS = 10.0
TARGET_STRUCTURE = "60/100/Runner"
TARGET1_POINTS = 60.0
TARGET2_POINTS = 100.0


@dataclass(frozen=True)
class SellV6Signal:
    """Normalized SELL_V6 paper signal."""

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
            "vwap_gate_rule": V6_VWAP_GATE_RULE,
            "raw": self.raw,
        }


class SellV6Engine:
    """Evaluate SELL_V6 on the latest closed bar."""

    def __init__(self) -> None:
        self._engine = SellV6CandidateEngine()
        self._emitted_bars: set[int] = set()

    @property
    def model_id(self) -> str:
        return MODEL_ID

    def evaluate_bar(
        self,
        *,
        frame,
        enriched,
        calendar,
        intel_frames: dict[str, Any],
        bar: int,
    ) -> dict[str, Any]:
        return self._engine.evaluate_bar(
            frame=frame,
            enriched=enriched,
            calendar=calendar,
            intel_frames=intel_frames,
            bar=bar,
            emitted_bars=self._emitted_bars,
        )

    def to_signal(self, evaluation: dict[str, Any]) -> SellV6Signal | None:
        if evaluation.get("verdict") != "SELL":
            return None
        entry = float(evaluation.get("layer4", {}).get("entry") or evaluation.get("context", {}).get("close") or 0.0)
        if entry <= 0:
            return None
        stop = round(entry + STOP_POINTS, 2)
        target1 = round(entry - TARGET1_POINTS, 2)
        target2 = round(entry - TARGET2_POINTS, 2)
        confidence = 1.0 if evaluation.get("layer5", {}).get("pass") else 0.5
        return SellV6Signal(
            timestamp=str(evaluation.get("timestamp", "")),
            direction="SELL",
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
