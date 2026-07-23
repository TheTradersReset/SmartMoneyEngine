"""Dataset Verification Engine — read-only validation before Replay."""

from src.dataset_verification.engine import DatasetVerificationEngine
from src.dataset_verification.health import score_health

__all__ = ["DatasetVerificationEngine", "score_health"]
