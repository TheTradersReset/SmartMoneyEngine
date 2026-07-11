"""Tests for ``OrderBlockDetector``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.market_data import MarketData
from src.smc.order_block import OrderBlockDetector, OrderBlockDirection

from tests.conftest import build_structure_market, na_series


@pytest.fixture
def detector() -> OrderBlockDetector:
    """Return an order block detector with test-friendly thresholds."""
    return OrderBlockDetector(
        min_body_ratio=0.25,
        min_displacement_body_ratio=0.40,
        min_displacement_multiplier=1.0,
        equal_level_tolerance_ratio=0.10,
    )


class TestOrderBlockDetectorValidation:
    """Validation and constructor tests."""

    def test_missing_bullish_bos_raises(self, detector: OrderBlockDetector) -> None:
        """Missing Bullish_BOS column should raise ``ValueError``."""
        market = MarketData(
            pd.DataFrame(
                {
                    "Open": [100.0],
                    "High": [101.0],
                    "Low": [99.0],
                    "Close": [100.0],
                    "Bearish_BOS": [pd.NA],
                }
            )
        )

        with pytest.raises(ValueError, match="Bullish_BOS column not found"):
            detector.detect(market)

    def test_invalid_min_body_ratio_raises(self) -> None:
        """Invalid min_body_ratio should raise ``ValueError``."""
        with pytest.raises(ValueError, match="min_body_ratio must be between 0 and 1"):
            OrderBlockDetector(min_body_ratio=1.5)

    def test_invalid_overlap_threshold_raises(self) -> None:
        """Invalid overlap_threshold should raise ``ValueError``."""
        with pytest.raises(ValueError, match="overlap_threshold must be between 0 and 1"):
            OrderBlockDetector(overlap_threshold=0.0)

    def test_missing_open_column_raises(self, detector: OrderBlockDetector) -> None:
        """Missing Open column should raise ``ValueError``."""
        market = MarketData(
            pd.DataFrame(
                {
                    "High": [101.0],
                    "Low": [99.0],
                    "Close": [100.0],
                    "Bullish_BOS": [pd.NA],
                    "Bearish_BOS": [pd.NA],
                }
            )
        )

        with pytest.raises(ValueError, match="Open column not found"):
            detector.detect(market)

    def test_invalid_rolling_window_raises(self) -> None:
        """Invalid rolling_window should raise ``ValueError``."""
        with pytest.raises(ValueError, match="rolling_window must be greater than one"):
            OrderBlockDetector(rolling_window=1)


class TestOrderBlockDetectorBullish:
    """Bullish order block detection tests."""

    def test_bullish_order_block_detected_before_bos(
        self,
        detector: OrderBlockDetector,
    ) -> None:
        """Final bearish candle before bullish BOS should become bullish OB."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0, 105.0, 103.0, 104.0, 106.0, 108.0],
                    "High": [101.0, 102.0, 103.0, 106.0, 104.0, 110.0, 112.0, 120.0],
                    "Low": [99.0, 100.0, 101.0, 103.5, 102.0, 103.0, 105.0, 107.0],
                    "Close": [101.0, 102.0, 102.5, 104.0, 103.2, 109.0, 111.0, 118.0],
                    "Bullish_BOS": na_series(7) + [118.0],
                    "Bearish_BOS": na_series(8),
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bullish_OB_High").iloc[3] == 106.0
        assert market.get_column("Bullish_OB_Low").iloc[3] == 103.5
        assert len(detector.order_blocks) == 1
        assert detector.order_blocks[0].direction == OrderBlockDirection.BULLISH
        assert detector.order_blocks[0].position == 3
        assert detector.order_blocks[0].bos_position == 7

    def test_no_bullish_ob_without_bos(self, detector: OrderBlockDetector) -> None:
        """No BOS event should produce no order blocks."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Open": [100.0, 105.0, 103.0, 104.0],
                    "High": [101.0, 106.0, 104.0, 110.0],
                    "Low": [99.0, 103.5, 102.0, 103.0],
                    "Close": [101.0, 104.0, 103.2, 109.0],
                    "Bullish_BOS": na_series(4),
                    "Bearish_BOS": na_series(4),
                }
            )
        )

        detector.detect(market)

        assert len(detector.order_blocks) == 0
        assert market.get_column("Bullish_OB_High").isna().all()


class TestOrderBlockDetectorBearish:
    """Bearish order block detection tests."""

    def test_bearish_order_block_detected_before_bos(
        self,
        detector: OrderBlockDetector,
    ) -> None:
        """Final bullish candle before bearish BOS should become bearish OB."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Open": [100.0, 100.0, 101.0, 95.0, 96.0, 94.0, 92.0, 88.0],
                    "High": [101.0, 101.5, 102.0, 96.0, 97.0, 95.0, 93.0, 89.0],
                    "Low": [99.0, 99.5, 100.0, 94.0, 93.0, 91.0, 87.0, 82.0],
                    "Close": [100.0, 101.0, 95.0, 96.0, 94.0, 92.0, 88.0, 82.0],
                    "Bullish_BOS": na_series(8),
                    "Bearish_BOS": na_series(7) + [82.0],
                }
            )
        )

        detector.detect(market)

        assert len(detector.order_blocks) == 1
        assert detector.order_blocks[0].direction == OrderBlockDirection.BEARISH
        assert detector.order_blocks[0].position == 1
        assert market.get_column("Bearish_OB_High").iloc[1] == 101.5
        assert market.get_column("Bearish_OB_Low").iloc[1] == 99.5


class TestOrderBlockDetectorMitigation:
    """Order block mitigation tests."""

    def test_bullish_order_block_mitigation(
        self,
        detector: OrderBlockDetector,
    ) -> None:
        """Price trading below bullish OB low should mark block mitigated."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0, 105.0, 103.0, 104.0, 106.0, 108.0, 102.0],
                    "High": [101.0, 102.0, 103.0, 106.0, 104.0, 110.0, 112.0, 120.0, 103.0],
                    "Low": [99.0, 100.0, 101.0, 103.5, 102.0, 103.0, 105.0, 107.0, 100.0],
                    "Close": [101.0, 102.0, 102.5, 104.0, 103.2, 109.0, 111.0, 118.0, 101.0],
                    "Bullish_BOS": na_series(7) + [118.0] + [pd.NA],
                    "Bearish_BOS": na_series(9),
                }
            )
        )

        detector.detect(market)

        ob_low = market.get_column("Bullish_OB_Low").dropna().iloc[0]
        assert ob_low == 103.5
        assert market.get_column("Bullish_OB_Mitigated").dropna().iloc[0] == True
        assert detector.order_blocks[0].mitigated is True

    def test_unmitigated_block_stays_false(
        self,
        detector: OrderBlockDetector,
    ) -> None:
        """Bullish OB should remain unmitigated if price never trades below OB low."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0, 105.0, 103.0, 104.0, 106.0, 108.0],
                    "High": [101.0, 102.0, 103.0, 106.0, 104.0, 110.0, 112.0, 120.0],
                    "Low": [99.0, 100.0, 101.0, 103.5, 102.0, 103.0, 105.0, 107.0],
                    "Close": [101.0, 102.0, 102.5, 104.0, 103.2, 109.0, 111.0, 118.0],
                    "Bullish_BOS": na_series(7) + [118.0],
                    "Bearish_BOS": na_series(8),
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bullish_OB_Mitigated").dropna().iloc[0] == False
        assert detector.order_blocks[0].mitigated is False

    def test_bearish_order_block_mitigation(
        self,
        detector: OrderBlockDetector,
    ) -> None:
        """Price trading above bearish OB high should mark block mitigated."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Open": [100.0, 100.0, 101.0, 95.0, 96.0, 94.0, 92.0, 88.0, 102.0],
                    "High": [101.0, 101.5, 102.0, 96.0, 97.0, 95.0, 93.0, 89.0, 103.0],
                    "Low": [99.0, 99.5, 100.0, 94.0, 93.0, 91.0, 87.0, 82.0, 100.0],
                    "Close": [100.0, 101.0, 95.0, 96.0, 94.0, 92.0, 88.0, 82.0, 101.5],
                    "Bullish_BOS": na_series(9),
                    "Bearish_BOS": na_series(7) + [82.0, pd.NA],
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Bearish_OB_Mitigated").dropna().iloc[0] == True
        assert detector.order_blocks[0].mitigated is True


class TestOrderBlockDetectorEdgeCases:
    """Edge cases, duplicates, and invalid structure tests."""

    def test_weak_origin_candle_rejected(self) -> None:
        """Tiny-bodied origin candle should not produce an order block."""
        detector = OrderBlockDetector(
            min_body_ratio=0.80,
            min_displacement_body_ratio=0.40,
            min_displacement_multiplier=1.0,
        )
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0, 105.0, 103.0, 104.0, 106.0, 108.0],
                    "High": [101.0, 102.0, 103.0, 106.0, 104.0, 110.0, 112.0, 120.0],
                    "Low": [99.0, 100.0, 101.0, 103.5, 102.0, 103.0, 105.0, 107.0],
                    "Close": [101.0, 102.0, 102.5, 104.0, 103.2, 109.0, 111.0, 118.0],
                    "Bullish_BOS": na_series(7) + [118.0],
                    "Bearish_BOS": na_series(8),
                }
            )
        )

        detector.detect(market)

        assert len(detector.order_blocks) == 0

    def test_duplicate_overlapping_bullish_obs_suppressed(
        self,
        detector: OrderBlockDetector,
    ) -> None:
        """Second overlapping bullish OB from nearby BOS should be suppressed."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0, 105.0, 103.0, 104.0, 106.0, 108.0, 104.0, 106.0],
                    "High": [101.0, 102.0, 103.0, 106.0, 104.0, 110.0, 112.0, 120.0, 111.0, 125.0],
                    "Low": [99.0, 100.0, 101.0, 103.5, 102.0, 103.0, 105.0, 107.0, 103.0, 105.0],
                    "Close": [101.0, 102.0, 102.5, 104.0, 103.2, 109.0, 111.0, 118.0, 110.0, 122.0],
                    "Bullish_BOS": na_series(7) + [118.0, pd.NA, 122.0],
                    "Bearish_BOS": na_series(10),
                }
            )
        )

        detector.detect(market)

        assert len(detector.order_blocks) == 1

    def test_empty_bos_columns_produce_no_blocks(
        self,
        detector: OrderBlockDetector,
    ) -> None:
        """Dataset with no BOS events should produce no order blocks."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0],
                    "High": [101.0, 102.0, 103.0],
                    "Low": [99.0, 100.0, 101.0],
                    "Close": [100.5, 101.5, 102.5],
                    "Bullish_BOS": na_series(3),
                    "Bearish_BOS": na_series(3),
                }
            )
        )

        detector.detect(market)

        assert len(detector.order_blocks) == 0
        assert market.get_column("Bullish_OB_High").isna().all()
        assert market.get_column("Bearish_OB_High").isna().all()

    def test_nan_bos_values_ignored(self, detector: OrderBlockDetector) -> None:
        """NaN BOS entries must not create spurious order blocks."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Open": [100.0, 101.0, 102.0, 105.0],
                    "High": [101.0, 102.0, 103.0, 106.0],
                    "Low": [99.0, 100.0, 101.0, 103.5],
                    "Close": [101.0, 102.0, 102.5, 104.0],
                    "Bullish_BOS": [pd.NA, pd.NA, pd.NA, pd.NA],
                    "Bearish_BOS": [pd.NA, pd.NA, pd.NA, pd.NA],
                }
            )
        )

        detector.detect(market)

        assert len(detector.order_blocks) == 0
