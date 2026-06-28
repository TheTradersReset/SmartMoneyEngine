from pathlib import Path

import pandas as pd

from src.core.logger import logger

from src.core.exceptions import (
    CSVFileNotFoundError,
    InvalidCSVFormatError,
    MissingColumnError,
    EmptyDataFrameError
)


class DataLoader:

    """
    Responsible for loading,
    validating,
    and preparing market data.
    """

    REQUIRED_COLUMNS = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    def __init__(self):

        self.data = None