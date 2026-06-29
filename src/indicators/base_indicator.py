from abc import ABC, abstractmethod

from src.core.logger import logger


class BaseIndicator(ABC):
    """
    Base class for all indicators.

    Every indicator must inherit from this class.

    Example:
        EMA
        ATR
        VWAP
        RSI
        BOS
        FVG
    """

    def __init__(self, name: str):

        self.name = name

    @abstractmethod
    def calculate(self, market_data):
        """
        Every indicator must implement this method.

        Parameters
        ----------
        market_data : MarketData

        Returns
        -------
        MarketData
        """
        pass

    def log_start(self):

        logger.info(
            f"Calculating {self.name}..."
        )

    def log_finish(self):

        logger.info(
            f"{self.name} calculation completed."
        )