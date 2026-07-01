from abc import ABC, abstractmethod

from src.core.logger import logger


class BaseSMC(ABC):
    """
    Base class for Smart Money Concept detectors.

    Every SMC module must inherit from this class
    and implement the detect() method.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def detect(self, market_data):
        """
        Run SMC detection on market data.

        Parameters
        ----------
        market_data : MarketData

        Returns
        -------
        MarketData
        """
        pass

    def log_start(self) -> None:
        logger.info(f"Starting {self.name}...")

    def log_finish(self) -> None:
        logger.info(f"{self.name} completed.")
