"""
Thin auth helper re-exporting FYERS token validation.

Canonical implementation: ``src.brokers.fyers.auth.ensure_valid_access_token``.
"""

from __future__ import annotations

from src.brokers.fyers.auth import ensure_valid_access_token

__all__ = ["ensure_valid_access_token", "get_valid_access_token"]


def get_valid_access_token(**kwargs):
    """Alias for ``ensure_valid_access_token``."""
    return ensure_valid_access_token(**kwargs)
