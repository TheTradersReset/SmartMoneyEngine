from pathlib import Path

import pandas as pd

from src.core.logger import logger

from src.core.exceptions import (
    CSVFileNotFoundError,
    InvalidCSVFormatError,
    MissingColumnError,
    EmptyDataFrameError,
    InvalidDataTypeError
)


class DataLoader:
    """
    Responsible for:

    - Loading CSV
    - Validating Columns
    - Validating Datatypes

    Returns a clean pandas DataFrame.
    """

    REQUIRED_COLUMNS = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    NUMERIC_COLUMNS = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    def __init__(self):
        self.data = None

    # -------------------------------------------------
    # Load CSV
    # -------------------------------------------------

    def load_csv(self, file_path):

        logger.info(f"Loading CSV file: {file_path}")

        file_path = Path(file_path)

        if not file_path.exists():
            raise CSVFileNotFoundError(
                f"CSV file not found: {file_path}"
            )

        try:
            self.data = pd.read_csv(file_path)

        except Exception as e:
            raise InvalidCSVFormatError(str(e))

        if self.data.empty:
            raise EmptyDataFrameError(
                "CSV file is empty."
            )

        self.validate_columns()

        self.validate_datatypes()

        logger.info(
            f"Successfully loaded {len(self.data)} rows."
        )

        return self.data

    # -------------------------------------------------
    # Column Validation
    # -------------------------------------------------

    def validate_columns(self):

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in self.data.columns
        ]

        if missing_columns:

            raise MissingColumnError(
                f"Missing columns: {missing_columns}"
            )

        logger.info(
            "Column validation successful."
        )

    # -------------------------------------------------
    # Datatype Validation
    # -------------------------------------------------

    def validate_datatypes(self):

        logger.info(
            "Validating datatypes..."
        )

        # Convert Date column

        try:

            self.data["Date"] = pd.to_datetime(
                self.data["Date"]
            )

        except Exception:

            raise InvalidDataTypeError(
                "Date column contains invalid values."
            )

        logger.info(
            "Date column validated."
        )

        # Convert Numeric Columns

        for column in self.NUMERIC_COLUMNS:

            try:

                self.data[column] = pd.to_numeric(
                    self.data[column]
                )

            except Exception:

                raise InvalidDataTypeError(
                    f"{column} contains invalid numeric values."
                )

        logger.info(
            "Numeric column validation successful."
        )