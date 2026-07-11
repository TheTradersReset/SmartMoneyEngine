"""Shared pytest fixtures for SmartMoneyEngine SMC tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.core.data_loader import DataLoader
from src.models.market_data import MarketData
from src.smc.market_structure import MarketStructure
from src.smc.swing_detector import SwingDetector

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "tests" / "sample_data"


@pytest.fixture
def swing_test_market() -> MarketData:
    """Load validated swing_test.csv through the data loader."""
    dataframe = DataLoader().load_csv(SAMPLE_DATA_DIR / "swing_test.csv")
    return MarketData(dataframe)


@pytest.fixture
def swing_test_with_swings(swing_test_market: MarketData) -> MarketData:
    """Swing test dataset with swing columns populated."""
    SwingDetector(lookback=2).detect(swing_test_market)
    return swing_test_market


@pytest.fixture
def swing_test_with_structure(swing_test_with_swings: MarketData) -> MarketData:
    """Swing test dataset with market structure labels populated."""
    MarketStructure().detect(swing_test_with_swings)
    return swing_test_with_swings


def build_structure_market(frame: pd.DataFrame) -> MarketData:
    """Wrap a synthetic OHLC/structure frame in ``MarketData``."""
    return MarketData(frame.copy())


def na_series(length: int) -> list[object]:
    """Return a list of ``pd.NA`` values."""
    return [pd.NA] * length
