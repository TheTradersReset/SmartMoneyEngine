from src.indicators.base_indicator import BaseIndicator
from src.core.logger import logger


class EMA(BaseIndicator):
    """
    Exponential Moving Average (EMA)

    Calculates EMA for any given period and
    stores the result inside MarketData.

    Example:
        EMA(20)
        EMA(50)
        EMA(100)
        EMA(200)
    """

    def __init__(self, period: int):

        if period <= 0:
            raise ValueError(
                "EMA period must be greater than zero."
            )

        super().__init__(f"EMA-{period}")

        self.period = period
        self.column_name = f"EMA_{period}"

    def calculate(self, market):

        self.log_start()

        if not market.has_column("Close"):
            raise ValueError(
                "Close column not found."
            )

        # Prevent duplicate calculation
        if market.has_column(self.column_name):

            logger.warning(
                f"{self.column_name} already exists."
            )

            return market

        ema = (
            market
            .get_column("Close")
            .ewm(
                span=self.period,
                adjust=False
            )
            .mean()
        )

        market.add_column(
            self.column_name,
            ema
        )

        self.log_finish()

        return market