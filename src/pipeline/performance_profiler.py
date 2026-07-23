"""
Per-candle performance profiler for the realtime signal pipeline.

Observability only — does not affect trading logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CandleProfiler:
    """Stage timings for one closed-candle processing cycle (milliseconds)."""

    tick_processing_ms: float = 0.0
    candle_creation_ms: float = 0.0
    indicator_calculation_ms: float = 0.0
    signal_engine_ms: float = 0.0
    decision_persistence_ms: float = 0.0
    sqlite_write_ms: float = 0.0
    total_processing_ms: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)

    def reset(self) -> None:
        self.tick_processing_ms = 0.0
        self.candle_creation_ms = 0.0
        self.indicator_calculation_ms = 0.0
        self.signal_engine_ms = 0.0
        self.decision_persistence_ms = 0.0
        self.sqlite_write_ms = 0.0
        self.total_processing_ms = 0.0
        self.extra.clear()

    def as_dict(self) -> dict[str, float]:
        payload = {
            "tick_processing_ms": round(self.tick_processing_ms, 3),
            "candle_creation_ms": round(self.candle_creation_ms, 3),
            "indicator_calculation_ms": round(self.indicator_calculation_ms, 3),
            "signal_engine_ms": round(self.signal_engine_ms, 3),
            "decision_persistence_ms": round(self.decision_persistence_ms, 3),
            "sqlite_write_ms": round(self.sqlite_write_ms, 3),
            "total_processing_ms": round(self.total_processing_ms, 3),
        }
        for key, value in self.extra.items():
            payload[key] = round(value, 3)
        return payload

    def print_report(self, *, timestamp: str, logger: Any) -> None:
        lines = [
            "",
            "-" * 72,
            f"PERFORMANCE PROFILE | {timestamp}",
            "-" * 72,
            f"Tick Processing       : {self.tick_processing_ms:8.3f} ms",
            f"Candle Creation       : {self.candle_creation_ms:8.3f} ms",
            f"Indicator Calculation : {self.indicator_calculation_ms:8.3f} ms",
            f"Signal Engine         : {self.signal_engine_ms:8.3f} ms",
            f"Decision Persistence  : {self.decision_persistence_ms:8.3f} ms",
            f"SQLite Write (queue)  : {self.sqlite_write_ms:8.3f} ms",
            f"Total Processing      : {self.total_processing_ms:8.3f} ms",
        ]
        for key, value in sorted(self.extra.items()):
            lines.append(f"{key.replace('_', ' ').title():22s}: {value:8.3f} ms")
        lines.append("-" * 72)
        text = "\n".join(lines)
        print(text, flush=True)
        logger.info("Performance profile\n%s", text)
