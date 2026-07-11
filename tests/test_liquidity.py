"""Tests for ``LiquidityDetector``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.market_data import MarketData
from src.smc.liquidity import LiquidityDetector, LiquiditySide

from tests.conftest import build_structure_market


@pytest.fixture
def detector() -> LiquidityDetector:
    """Return a liquidity detector with default tolerance."""
    return LiquidityDetector(tolerance_ratio=0.001)


class TestLiquidityDetectorValidation:
    """Validation and error handling tests."""

    def test_missing_swing_high_raises(self, detector: LiquidityDetector) -> None:
        """Missing Swing_High column should raise ``ValueError``."""
        market = MarketData(
            pd.DataFrame(
                {
                    "Swing_Low": [pd.NA],
                    "High": [100.0],
                    "Low": [99.0],
                    "Close": [100.0],
                }
            )
        )

        with pytest.raises(ValueError, match="Swing_High column not found"):
            detector.detect(market)

    def test_missing_high_column_raises(self, detector: LiquidityDetector) -> None:
        """Missing High column should raise ``ValueError``."""
        market = MarketData(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA],
                    "Swing_Low": [pd.NA],
                    "Low": [99.0],
                    "Close": [100.0],
                }
            )
        )

        with pytest.raises(ValueError, match="High column not found"):
            detector.detect(market)

    def test_invalid_tolerance_raises(self) -> None:
        """Non-positive tolerance should raise ``ValueError``."""
        with pytest.raises(ValueError, match="tolerance_ratio must be greater than zero"):
            LiquidityDetector(tolerance_ratio=0.0)


class TestLiquidityDetectorEqualLevels:
    """Equal high / equal low cluster tests."""

    def test_equal_high_cluster_forms_within_tolerance(
        self,
        detector: LiquidityDetector,
    ) -> None:
        """Two swing highs within tolerance should form equal-high cluster."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA, 100.0, pd.NA, pd.NA, 100.05, pd.NA, pd.NA],
                    "Swing_Low": [pd.NA] * 7,
                    "High": [99, 100, 101, 102, 100.1, 103, 104],
                    "Low": [98, 99, 100, 101, 99.5, 102, 103],
                    "Close": [99, 100, 101, 101.5, 100.0, 102.5, 103.5],
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Equal_High").iloc[1] == 100.05
        assert market.get_column("Equal_High").iloc[4] == 100.05
        assert len(detector.liquidity_pools) == 1
        assert detector.liquidity_pools[0].side == LiquiditySide.BUY
        assert detector.liquidity_pools[0].level == 100.05

    def test_equal_low_cluster_forms_within_tolerance(
        self,
        detector: LiquidityDetector,
    ) -> None:
        """Two swing lows within tolerance should form equal-low cluster."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA] * 6,
                    "Swing_Low": [pd.NA, 50.0, pd.NA, pd.NA, 50.02, pd.NA],
                    "High": [51, 52, 53, 54, 55, 52],
                    "Low": [49, 50.5, 51, 52, 53, 48],
                    "Close": [50, 51, 52, 53, 54, 51],
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Equal_Low").iloc[1] == 50.0
        assert market.get_column("Equal_Low").iloc[4] == 50.0
        assert detector.liquidity_pools[0].side == LiquiditySide.SELL

    def test_no_cluster_when_swings_outside_tolerance(
        self,
        detector: LiquidityDetector,
        swing_test_with_swings: MarketData,
    ) -> None:
        """swing_test.csv highs should not cluster at default tolerance."""
        detector.detect(swing_test_with_swings)

        assert len(detector.liquidity_pools) == 0
        assert swing_test_with_swings.get_column("Equal_High").isna().all()
        assert swing_test_with_swings.get_column("Equal_Low").isna().all()


class TestLiquidityDetectorActivationTiming:
    """Pool activation and projection timing tests."""

    def test_buy_side_liquidity_starts_on_second_touch(
        self,
        detector: LiquidityDetector,
    ) -> None:
        """Buy-side liquidity must not project before cluster confirmation."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA, 100.0, pd.NA, pd.NA, 100.05, pd.NA, pd.NA],
                    "Swing_Low": [pd.NA] * 7,
                    "High": [99, 100, 101, 102, 100.1, 103, 104],
                    "Low": [98, 99, 100, 101, 99.5, 102, 103],
                    "Close": [99, 100, 101, 101.5, 100.0, 102.5, 103.5],
                }
            )
        )

        detector.detect(market)

        assert pd.isna(market.get_column("Buy_Side_Liquidity").iloc[1])
        assert market.get_column("Buy_Side_Liquidity").iloc[4] == 100.05
        assert market.get_column("Buy_Side_Liquidity").iloc[6] == 100.05

    def test_liquidity_strength_two_touches_equals_one(
        self,
        detector: LiquidityDetector,
    ) -> None:
        """Two-touch cluster should produce strength score of 1."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA, 100.0, pd.NA, pd.NA, 100.05, pd.NA],
                    "Swing_Low": [pd.NA] * 6,
                    "High": [99, 100, 101, 102, 100.1, 103],
                    "Low": [98, 99, 100, 101, 99.5, 102],
                    "Close": [99, 100, 101, 101.5, 100.0, 102.5],
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Liquidity_Strength").iloc[4] == 1
        assert detector.liquidity_pools[0].strength == 1

    def test_liquidity_strength_three_touches_equals_two(self) -> None:
        """Three-touch cluster should produce strength score of 2."""
        detector = LiquidityDetector(tolerance_ratio=0.001)
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA, 100.0, pd.NA, 100.03, pd.NA, 100.05, pd.NA],
                    "Swing_Low": [pd.NA] * 7,
                    "High": [99, 100, 101, 100.1, 102, 100.2, 103],
                    "Low": [98, 99, 100, 99.8, 101, 99.7, 102],
                    "Close": [99, 100, 101, 100.0, 101.5, 100.1, 102.5],
                }
            )
        )

        detector.detect(market)

        assert detector.liquidity_pools[0].strength == 2
        assert market.get_column("Liquidity_Strength").iloc[3] == 2

    def test_liquidity_strength_four_touches_equals_three(self) -> None:
        """Four-touch cluster should produce maximum strength score of 3."""
        detector = LiquidityDetector(tolerance_ratio=0.001)
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA, 100.0, pd.NA, 100.03, pd.NA, 100.05, pd.NA, 100.04],
                    "Swing_Low": [pd.NA] * 8,
                    "High": [99, 100, 101, 100.1, 102, 100.2, 103, 100.15],
                    "Low": [98, 99, 100, 99.8, 101, 99.7, 102, 99.75],
                    "Close": [99, 100, 101, 100.0, 101.5, 100.1, 102.5, 100.05],
                }
            )
        )

        detector.detect(market)

        assert detector.liquidity_pools[0].strength == 3
        assert market.get_column("Liquidity_Strength").iloc[3] == 3

    def test_empty_dataset_produces_liquidity_columns(
        self,
        detector: LiquidityDetector,
    ) -> None:
        """Empty input should add liquidity columns without raising."""
        market = MarketData(
            pd.DataFrame(
                {
                    "Swing_High": pd.Series(dtype=float),
                    "Swing_Low": pd.Series(dtype=float),
                    "High": pd.Series(dtype=float),
                    "Low": pd.Series(dtype=float),
                    "Close": pd.Series(dtype=float),
                }
            )
        )

        detector.detect(market)

        assert len(market.data) == 0
        assert len(detector.liquidity_pools) == 0
        assert "Buy_Side_Liquidity" in market.columns

    def test_missing_swing_low_raises(self, detector: LiquidityDetector) -> None:
        """Missing Swing_Low column should raise ``ValueError``."""
        market = MarketData(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA],
                    "High": [100.0],
                    "Low": [99.0],
                    "Close": [100.0],
                }
            )
        )

        with pytest.raises(ValueError, match="Swing_Low column not found"):
            detector.detect(market)


class TestLiquidityDetectorSweeps:
    """Liquidity sweep detection tests."""

    def test_buy_side_liquidity_sweep_detected(
        self,
        detector: LiquidityDetector,
    ) -> None:
        """High above BSL with close back below should record buy-side sweep."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA, 100.0, pd.NA, pd.NA, 100.05, pd.NA, pd.NA],
                    "Swing_Low": [pd.NA] * 7,
                    "High": [99, 100, 101, 102, 100.1, 103, 104],
                    "Low": [98, 99, 100, 101, 99.5, 102, 103],
                    "Close": [99, 100, 101, 101.5, 100.0, 102.5, 103.5],
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Buy_Liquidity_Sweep").iloc[4] == 100.1
        assert detector.liquidity_pools[0].swept is True
        assert detector.liquidity_pools[0].sweep_price == 100.1

    def test_sell_side_liquidity_sweep_detected(
        self,
        detector: LiquidityDetector,
    ) -> None:
        """Low below SSL with close back above should record sell-side sweep."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA] * 6,
                    "Swing_Low": [pd.NA, 50.0, pd.NA, pd.NA, 50.02, pd.NA],
                    "High": [51, 52, 53, 54, 55, 52],
                    "Low": [49, 50.5, 51, 52, 53, 48],
                    "Close": [50, 51, 52, 53, 54, 51],
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Sell_Liquidity_Sweep").iloc[5] == 48.0
        assert detector.liquidity_pools[0].swept is True

    def test_no_sweep_when_close_stays_beyond_liquidity(
        self,
        detector: LiquidityDetector,
    ) -> None:
        """Close remaining below SSL after a wick should not count as sell-side sweep."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA] * 6,
                    "Swing_Low": [pd.NA, 50.0, pd.NA, pd.NA, 50.02, pd.NA],
                    "High": [51, 52, 53, 54, 55, 52],
                    "Low": [49, 50.5, 51, 52, 53, 49],
                    "Close": [50, 51, 52, 53, 54, 49.5],
                }
            )
        )

        detector.detect(market)

        assert market.get_column("Sell_Liquidity_Sweep").isna().all()
        assert detector.liquidity_pools[0].swept is False


class TestLiquidityDetectorEdgeCases:
    """Edge case and NaN handling tests."""

    def test_all_nan_swings_produce_no_pools(
        self,
        detector: LiquidityDetector,
    ) -> None:
        """All-null swing columns should produce no liquidity pools."""
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA, pd.NA, pd.NA],
                    "Swing_Low": [pd.NA, pd.NA, pd.NA],
                    "High": [100.0, 101.0, 102.0],
                    "Low": [99.0, 100.0, 101.0],
                    "Close": [100.0, 101.0, 102.0],
                }
            )
        )

        detector.detect(market)

        assert len(detector.liquidity_pools) == 0
        assert market.get_column("Buy_Side_Liquidity").isna().all()
        assert market.get_column("Sell_Side_Liquidity").isna().all()

    def test_integration_with_swing_test_csv(
        self,
        detector: LiquidityDetector,
        swing_test_with_swings: MarketData,
    ) -> None:
        """Full swing_test.csv pipeline should add all liquidity columns."""
        detector.detect(swing_test_with_swings)

        expected_columns = [
            "Equal_High",
            "Equal_Low",
            "Buy_Side_Liquidity",
            "Sell_Side_Liquidity",
            "Buy_Liquidity_Sweep",
            "Sell_Liquidity_Sweep",
            "Liquidity_Strength",
        ]

        for column in expected_columns:
            assert column in swing_test_with_swings.columns

    def test_duplicate_nearby_clusters_last_projection_wins(self) -> None:
        """Overlapping nearby clusters should not crash; later pool overwrites rows."""
        detector = LiquidityDetector(tolerance_ratio=0.001)
        market = build_structure_market(
            pd.DataFrame(
                {
                    "Swing_High": [pd.NA, 100.0, pd.NA, 100.05, pd.NA, 200.0, pd.NA, 200.1],
                    "Swing_Low": [pd.NA] * 8,
                    "High": [99, 100, 101, 100.1, 101, 200, 201, 200.2],
                    "Low": [98, 99, 100, 99.5, 100, 199, 200, 199.5],
                    "Close": [99, 100, 101, 100.0, 101, 200, 201, 200.0],
                }
            )
        )

        detector.detect(market)

        assert len(detector.liquidity_pools) == 2
        assert market.get_column("Buy_Side_Liquidity").iloc[7] == 200.1
        assert market.get_column("Buy_Side_Liquidity").iloc[4] == 100.05
