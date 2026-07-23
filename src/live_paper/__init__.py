"""
Live paper trading orchestration for SmartMoneyEngine.

Paper-mode only: websocket ticks -> 5m candles -> frozen BUY_V3 / SELL_V6
signals -> SQLite persistence -> email alerts -> local dashboard.
Never places, modifies, or cancels broker orders.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
