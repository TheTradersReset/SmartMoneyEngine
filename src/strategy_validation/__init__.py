"""Strategy validation campaign — evidence via existing Replay Engine only."""

__all__ = ["run_campaign"]


def __getattr__(name: str):
    if name == "run_campaign":
        from src.strategy_validation.campaign import run_campaign

        return run_campaign
    raise AttributeError(name)
