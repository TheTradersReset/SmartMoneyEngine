"""Tests for ``BreakOfStructure``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.market_data import MarketData
from src.smc.bos import BreakOfStructure
from src.smc.market_structure import MarketStructure
from src.smc.swing_detector import SwingDetector

from tests.conftest import build_structure_market, na_series


@pytest.fixture
def detector() -> BreakOfStructure:
    """Return a break-of-structure detector instance."""
    return BreakOfStructure()


class TestBreakOfStructureValidation:
    """Validation and error handling tests."""

    def test_missing_hh_column_raises(self, detector: BreakOfStructure) -> None:
        """Missing HH column should raise ``ValueError``."""
        market = MarketData(
            pd.DataFrame(
                {
                    "Close": [100.0],
                    "HL": [pd.NA],
                    "LH": [pd.NA],
                    "LL": [pd.NA],
                }
            )
        )

        with pytest.raises(ValueError, match="HH column not found"):
            detector.detect(market)

    def test_missing_close_column_raises(self, detector: BreakOfStructure) -> None:
        """Missing Close column should raise ``ValueError``."""
        market = MarketData(
            pd.DataFrame(
                {
                    "HH": [pd.NA],
                    "HL": [pd.NA],
                    "LH": [pd.NA],
                    "LL": [pd.NA],
                }
            )
        )

        with pytest.raises(ValueError, match="Close column not found"):
            detector.detect(market)

    def test_missing_hl_column_raises(self, detector: BreakOfStructure) -> None:
        """Missing HL column should raise ``ValueError``."""
        market = MarketData(
            pd.DataFrame(
                {
                    "Close": [100.0],
                    "HH": [pd.NA],
                    "LH": [pd.NA],
                    "LL": [pd.NA],
                }
            )
        )

        with pytest.raises(ValueError, match="HL column not found"):
            detector.detect(market)


class TestBreakOfStructureBullish:
    """Bullish BOS detection tests."""

    def test_bullish_bos_on_first_close_above_hh(
        self,
        detector: BreakOfStructure,
    ) -> None:
        """First close above prior HH should record bullish BOS."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 105.0, 115.0, 112.0],
                    "HH": [pd.NA, 110.0, pd.NA, pd.NA],
                    "HL": na_series(4),
                    "LH": na_series(4),
                    "LL": na_series(4),
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bullish_BOS").iloc[2] == 115.0
        assert pd.isna(market.get_column("Bullish_BOS").iloc[3])
        assert market.get_column("Bullish_BOS").notna().sum() == 1

    def test_no_duplicate_bullish_bos_while_staying_above_hh(
        self,
        detector: BreakOfStructure,
    ) -> None:
        """Already-above-HH closes must not create duplicate BOS events."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 105.0, 115.0, 116.0, 117.0],
                    "HH": [pd.NA, 110.0, pd.NA, pd.NA, pd.NA],
                    "HL": na_series(5),
                    "LH": na_series(5),
                    "LL": na_series(5),
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bullish_BOS").notna().sum() == 1
        assert market.get_column("Bullish_BOS").iloc[2] == 115.0


class TestBreakOfStructureBearish:
    """Bearish BOS detection tests."""

    def test_bearish_bos_on_first_close_below_ll(
        self,
        detector: BreakOfStructure,
    ) -> None:
        """First close below prior LL should record bearish BOS."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 98.0, 90.0, 89.0],
                    "HH": na_series(4),
                    "HL": na_series(4),
                    "LH": na_series(4),
                    "LL": [pd.NA, 95.0, pd.NA, pd.NA],
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bearish_BOS").iloc[2] == 90.0
        assert pd.isna(market.get_column("Bearish_BOS").iloc[3])
        assert market.get_column("Bearish_BOS").notna().sum() == 1


class TestBreakOfStructureNoRepaint:
    """Causal / no same-bar structure break tests."""

    def test_no_bos_on_same_bar_as_new_hh_label(
        self,
        detector: BreakOfStructure,
    ) -> None:
        """HH label on current bar must not be used for BOS on that bar."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 115.0],
                    "HH": [pd.NA, 110.0],
                    "HL": na_series(2),
                    "LH": na_series(2),
                    "LL": na_series(2),
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bullish_BOS").isna().all()

    def test_no_bos_on_same_bar_as_new_ll_label(
        self,
        detector: BreakOfStructure,
    ) -> None:
        """LL label on current bar must not be used for BOS on that bar."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 90.0],
                    "HH": na_series(2),
                    "HL": na_series(2),
                    "LH": na_series(2),
                    "LL": [pd.NA, 95.0],
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bearish_BOS").isna().all()


class TestBreakOfStructureEdgeCases:
    """Edge case and integration tests."""

    def test_all_nan_structure_produces_no_events(
        self,
        detector: BreakOfStructure,
    ) -> None:
        """All-null structure labels should produce no BOS events."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 101.0, 102.0],
                    "HH": na_series(3),
                    "HL": na_series(3),
                    "LH": na_series(3),
                    "LL": na_series(3),
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bullish_BOS").isna().all()
        assert market.get_column("Bearish_BOS").isna().all()

    def test_single_row_dataset(self, detector: BreakOfStructure) -> None:
        """Single-row input should run without error and produce NaN outputs."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0],
                    "HH": [110.0],
                    "HL": [pd.NA],
                    "LH": [pd.NA],
                    "LL": [95.0],
                }
            )
        )

        detector.detect(market)

        assert len(market.data) == 1
        assert market.get_column("Bullish_BOS").isna().all()
        assert market.get_column("Bearish_BOS").isna().all()

    def test_empty_dataset_produces_nan_columns(self, detector: BreakOfStructure) -> None:
        """Empty input should add BOS columns without raising."""
        market = MarketData(
            pd.DataFrame(
                {
                    "Close": pd.Series(dtype=float),
                    "HH": pd.Series(dtype=float),
                    "HL": pd.Series(dtype=float),
                    "LH": pd.Series(dtype=float),
                    "LL": pd.Series(dtype=float),
                }
            )
        )

        detector.detect(market)

        assert len(market.data) == 0
        assert "Bullish_BOS" in market.columns
        assert "Bearish_BOS" in market.columns

    def test_integration_with_swing_test_csv(
        self,
        detector: BreakOfStructure,
        swing_test_with_structure: MarketData,
    ) -> None:
        """Pipeline on swing_test.csv should add BOS columns without error."""
        detector.detect(swing_test_with_structure)

        assert "Bullish_BOS" in swing_test_with_structure.columns
        assert "Bearish_BOS" in swing_test_with_structure.columns
        assert swing_test_with_structure.get_column("Bullish_BOS").dtype.name == "Float64"
