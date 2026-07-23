"""
Compatibility re-export of the FYERS websocket client.

Canonical implementation lives in ``src.brokers.websocket_client``.
"""

from __future__ import annotations

from src.brokers.websocket_client import *  # noqa: F403
from src.brokers.websocket_client import (  # noqa: F401
    DEFAULT_BACKOFF_MULTIPLIER,
    DEFAULT_DATA_TYPE,
    DEFAULT_INITIAL_BACKOFF_SECONDS,
    DEFAULT_MAX_BACKOFF_SECONDS,
    FyersWebsocketClient,
    NIFTY50_SYMBOL,
    WebsocketConnectivityError,
    compute_reconnect_backoff,
    validate_subscribe_symbol,
)
