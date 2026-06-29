from src.indicators.base_indicator import BaseIndicator


class EMA(BaseIndicator):

    """
    Exponential Moving Average
    """

    def __init__(self, period):

        super().__init__(f"EMA-{period}")

        self.period = period

    def calculate(self, market_data):

        self.log_start()

        # EMA calculation
        # Next Sprint

        self.log_finish()

        return market_data