from pathlib import Path

import pandas as pd

from src.core.logger import logger

from src.core.exceptions import (
    CSVFileNotFoundError,
    InvalidCSVFormatError,
    MissingColumnError,
    EmptyDataFrameError,
)


class DataLoader:
    """
    Responsible for loading and validating market data.
    """

    REQUIRED_COLUMNS = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    def __init__(self):
        self.data = None

    def load_csv(self, file_path: str):
        """
        Load CSV file into a Pandas DataFrame.
        """

        path = Path(file_path)

        logger.info(f"Loading CSV file: {path}")

        # Check if file exists
        if not path.exists():
            logger.error(f"CSV file not found: {path}")
            raise CSVFileNotFoundError(f"{path} does not exist.")

        # Read CSV
        try:
            self.data = pd.read_csv(path)

        except Exception as e:
            logger.error(str(e))
            raise InvalidCSVFormatError(str(e))

        # Check empty dataframe
        if self.data.empty:
            logger.error("CSV file is empty.")
            raise EmptyDataFrameError("CSV file is empty.")

        # Validate required columns
        self.validate_columns()

        logger.info(f"Successfully loaded {len(self.data)} rows.")

        return self.data

    def validate_columns(self):
        """
        Validate all required columns are present.
        """

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.data.columns
        ]

        if missing_columns:
            logger.error(
                f"Missing required columns: {missing_columns}"
            )

            raise MissingColumnError(
                f"Required columns missing: {missing_columns}"
            )

        logger.info("Column validation successful.")