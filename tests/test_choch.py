"""Tests for ``ChangeOfCharacter``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.market_data import MarketData
from src.smc.choch import ChangeOfCharacter

from tests.conftest import build_structure_market, na_series


@pytest.fixture
def detector() -> ChangeOfCharacter:
    """Return a change-of-character detector instance."""
    return ChangeOfCharacter()


class TestChangeOfCharacterValidation:
    """Validation and error handling tests."""

    def test_missing_lh_column_raises(self, detector: ChangeOfCharacter) -> None:
        """Missing LH column should raise ``ValueError``."""
        market = MarketData(
            pd.DataFrame(
                {
                    "Close": [100.0],
                    "HH": [pd.NA],
                    "HL": [pd.NA],
                    "LL": [pd.NA],
                }
            )
        )

        with pytest.raises(ValueError, match="LH column not found"):
            detector.detect(market)

    def test_missing_close_column_raises(self, detector: ChangeOfCharacter) -> None:
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


class TestChangeOfCharacterBullish:
    """Bullish CHOCH detection tests."""

    def test_bullish_choch_after_bearish_structure(
        self,
        detector: ChangeOfCharacter,
    ) -> None:
        """Close above prior LH after bearish bias should record bullish CHOCH."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 98.0, 102.0, 104.0],
                    "HH": na_series(4),
                    "HL": na_series(4),
                    "LH": [pd.NA, 100.0, pd.NA, pd.NA],
                    "LL": [pd.NA, pd.NA, 95.0, pd.NA],
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bullish_CHOCH").iloc[2] == 102.0
        assert market.get_column("Bullish_CHOCH").notna().sum() == 1
        assert market.get_column("Bearish_CHOCH").isna().all()

    def test_no_bullish_choch_without_bearish_bias(
        self,
        detector: ChangeOfCharacter,
    ) -> None:
        """LH break under bullish bias must not produce bullish CHOCH."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 108.0, 112.0],
                    "HH": [pd.NA, 110.0, pd.NA],
                    "HL": na_series(3),
                    "LH": [pd.NA, pd.NA, 105.0],
                    "LL": na_series(3),
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bullish_CHOCH").isna().all()


class TestChangeOfCharacterBearish:
    """Bearish CHOCH detection tests."""

    def test_bearish_choch_after_bullish_structure(
        self,
        detector: ChangeOfCharacter,
    ) -> None:
        """Close below prior HL after bullish bias should record bearish CHOCH."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 108.0, 106.0, 103.0],
                    "HH": [pd.NA, 110.0, pd.NA, pd.NA],
                    "HL": [pd.NA, pd.NA, 105.0, pd.NA],
                    "LH": na_series(4),
                    "LL": na_series(4),
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bearish_CHOCH").iloc[3] == 103.0
        assert market.get_column("Bearish_CHOCH").notna().sum() == 1
        assert market.get_column("Bullish_CHOCH").isna().all()


class TestChangeOfCharacterNoRepaint:
    """Causal structure break tests."""

    def test_no_choch_on_same_bar_as_new_lh_label(
        self,
        detector: ChangeOfCharacter,
    ) -> None:
        """LH label on current bar must not be used for bullish CHOCH same bar."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 98.0, 102.0],
                    "HH": na_series(3),
                    "HL": na_series(3),
                    "LH": [pd.NA, pd.NA, 100.0],
                    "LL": [pd.NA, 95.0, pd.NA],
                }
            )
        )

        detector.detect(market)

        assert pd.isna(market.get_column("Bullish_CHOCH").iloc[2])

    def test_no_choch_on_same_bar_as_new_hl_label(
        self,
        detector: ChangeOfCharacter,
    ) -> None:
        """HL label on current bar must not be used for bearish CHOCH same bar."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 108.0, 103.0],
                    "HH": [pd.NA, 110.0, pd.NA],
                    "HL": [pd.NA, pd.NA, 105.0],
                    "LH": na_series(3),
                    "LL": na_series(3),
                }
            )
        )

        detector.detect(market)

        assert pd.isna(market.get_column("Bearish_CHOCH").iloc[2])

    def test_no_duplicate_bearish_choch(
        self,
        detector: ChangeOfCharacter,
    ) -> None:
        """Subsequent closes below HL must not duplicate CHOCH events."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Close": [100.0, 108.0, 106.0, 103.0, 102.0, 101.0],
                    "HH": [pd.NA, 110.0, pd.NA, pd.NA, pd.NA, pd.NA],
                    "HL": [pd.NA, pd.NA, 105.0, pd.NA, pd.NA, pd.NA],
                    "LH": na_series(6),
                    "LL": na_series(6),
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bearish_CHOCH").notna().sum() == 1
        assert market.get_column("Bearish_CHOCH").iloc[3] == 103.0


class TestChangeOfCharacterEdgeCases:
    """Edge case tests."""

    def test_all_nan_structure_produces_no_events(
        self,
        detector: ChangeOfCharacter,
    ) -> None:
        """Null structure labels should produce no CHOCH events."""
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

        assert market.get_column("Bullish_CHOCH").isna().all()
        assert market.get_column("Bearish_CHOCH").isna().all()

    def test_empty_structure_with_zero_rows_not_valid_for_marketdata(self) -> None:
        """Empty frame is valid for MarketData but detector should handle gracefully."""
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

        ChangeOfCharacter().detect(market)

        assert len(market.data) == 0
