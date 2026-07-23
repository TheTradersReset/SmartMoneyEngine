"""
Regime throttle for frozen BUY_V3 / SELL_V6 production stack.

Loads throttle maps from ``regime_detection_audit.json`` when available.
Paper signal mode: BLOCK rejects signal emission; HALF/QUARTER annotate weight.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.core.logger import logger
from src.research.regime_detection_audit_research import THROTTLE_WEIGHT
from src.research.walk_forward_failure_root_cause_audit_research import _infer_regime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGIME_EXPORT = PROJECT_ROOT / "outputs" / "research" / "regime_detection_audit.json"

STACK_FINGERPRINT = "BUY_V3|SELL_V6|fixed_10|60/100/Runner|RegimeThrottle"


@dataclass(frozen=True)
class ThrottleDecision:
    """Throttle outcome for a candidate signal."""

    composite_regime: str
    throttle_level: str
    weight: float
    accepted: bool
    rejection_reason: str | None


class RegimeThrottle:
    """Apply production regime throttle rules to candidate signals."""

    def __init__(
        self,
        *,
        buy_throttle_map: dict[str, str] | None = None,
        sell_throttle_map: dict[str, str] | None = None,
        regime_export_path: Path | str = DEFAULT_REGIME_EXPORT,
    ) -> None:
        if buy_throttle_map is None or sell_throttle_map is None:
            loaded_buy, loaded_sell = self._load_maps(Path(regime_export_path))
            buy_throttle_map = buy_throttle_map if buy_throttle_map is not None else loaded_buy
            sell_throttle_map = sell_throttle_map if sell_throttle_map is not None else loaded_sell
        self.buy_throttle_map = buy_throttle_map
        self.sell_throttle_map = sell_throttle_map

    @staticmethod
    def _load_maps(path: Path) -> tuple[dict[str, str], dict[str, str]]:
        if not path.exists():
            logger.warning("Regime export missing (%s); defaulting to FULL throttle.", path)
            return {}, {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to load regime export %s: %s", path, exc)
            return {}, {}
        throttle = payload.get("throttle_recommendation", {})
        buy_rules = throttle.get("buy_v3_regime_throttle") or []
        sell_rules = throttle.get("sell_v6_regime_throttle") or []
        buy_map = {str(row["regime"]): str(row["throttle"]) for row in buy_rules if row.get("regime")}
        sell_map = {str(row["regime"]): str(row["throttle"]) for row in sell_rules if row.get("regime")}
        logger.info(
            "Loaded regime throttle maps: buy=%s sell=%s rules",
            len(buy_map),
            len(sell_map),
        )
        return buy_map, sell_map

    def composite_key(self, evaluation: dict[str, Any]) -> str:
        inferred = _infer_regime(evaluation)
        parts = [
            inferred.get("trend", "unknown"),
            inferred.get("volatility", "unknown"),
            inferred.get("gap", "unknown"),
            inferred.get("liquidity", "unknown"),
        ]
        composite = "|".join(parts)
        evaluation.setdefault("regime", {})
        if isinstance(evaluation["regime"], dict):
            evaluation["regime"]["composite"] = composite
        return composite

    def apply(self, *, direction: str, evaluation: dict[str, Any]) -> ThrottleDecision:
        composite = self.composite_key(evaluation)
        throttle_map = self.buy_throttle_map if direction == "BUY" else self.sell_throttle_map
        level = throttle_map.get(composite, "FULL")
        weight = float(THROTTLE_WEIGHT.get(level, 1.0))
        if level == "BLOCK":
            return ThrottleDecision(
                composite_regime=composite,
                throttle_level=level,
                weight=weight,
                accepted=False,
                rejection_reason=f"REGIME_BLOCK:{composite}",
            )
        return ThrottleDecision(
            composite_regime=composite,
            throttle_level=level,
            weight=weight,
            accepted=True,
            rejection_reason=None,
        )
