"""Configuration for post-signal trade validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SIGNAL_DB = PROJECT_ROOT / "data" / "paper" / "realtime_signals.db"
DEFAULT_VALIDATION_DB = PROJECT_ROOT / "data" / "paper" / "trade_validation.db"


@dataclass(frozen=True)
class TradeValidationConfig:
    """
    Evaluation parameters for forward-looking signal validation.

    Target and stop percentages are applied relative to entry price:
      BUY  → target = entry × (1 + target_pct/100), stop = entry × (1 − stop_pct/100)
      SELL → target = entry × (1 − target_pct/100), stop = entry × (1 + stop_pct/100)
    """

    evaluation_window_bars: int = 20
    target_pct: float = 0.24
    stop_pct: float = 0.04
    default_symbol: str = "NSE:NIFTY50-INDEX"
    signal_db_path: Path = DEFAULT_SIGNAL_DB
    validation_db_path: Path = DEFAULT_VALIDATION_DB
    evaluate_rejected_signals: bool = True

    def __post_init__(self) -> None:
        if self.evaluation_window_bars < 1:
            raise ValueError("evaluation_window_bars must be >= 1")
        if self.target_pct <= 0 or self.stop_pct <= 0:
            raise ValueError("target_pct and stop_pct must be positive")
