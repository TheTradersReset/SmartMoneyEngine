"""Post-signal trade validation — independent of the signal engine."""

from src.trade_validation.config import TradeValidationConfig
from src.trade_validation.engine import TradeValidationEngine
from src.trade_validation.evaluator import evaluate_signal
from src.trade_validation.models import CandleBar, SignalRecord, TradeValidationResult
from src.trade_validation.storage import TradeValidationDatabase

__all__ = [
    "CandleBar",
    "SignalRecord",
    "TradeValidationConfig",
    "TradeValidationDatabase",
    "TradeValidationEngine",
    "TradeValidationResult",
    "evaluate_signal",
]
