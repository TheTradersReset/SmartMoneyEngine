"""SmartMoneyEngine trade decision signals."""

from typing import Any

__all__ = [
    "DecisionEngine",
    "DecisionEngineError",
    "DecisionReport",
    "DecisionResult",
    "InstitutionalBias",
    "MarketBias",
    "MultiTimeframeEngine",
    "MultiTimeframeEngineError",
    "MultiTimeframeReport",
    "OverallBias",
    "TradeDecision",
    "TradePlan",
    "TradePlanEngine",
    "TradePlanEngineError",
    "TradePlanReport",
    "evaluate_pipeline",
    "generate_multi_timeframe_report",
    "generate_trade_plans",
]


def __getattr__(name: str) -> Any:
    """Lazy export to avoid import side effects during module execution."""
    if name in __all__:
        if name in {
            "TradePlan",
            "TradePlanEngine",
            "TradePlanEngineError",
            "TradePlanReport",
            "generate_trade_plans",
        }:
            from src.signals import trade_plan_engine as module
        elif name in {
            "MultiTimeframeEngine",
            "MultiTimeframeEngineError",
            "MultiTimeframeReport",
            "OverallBias",
            "generate_multi_timeframe_report",
        }:
            from src.signals import multi_timeframe_engine as module
        else:
            from src.signals import decision_engine as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
