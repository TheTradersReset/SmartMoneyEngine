from pathlib import Path

import pandas as pd

from src.core.logger import logger

from src.core.exceptions import (
    CSVFileNotFoundError,
    InvalidCSVFormatError,
    MissingColumnError,
    EmptyDataFrameError,
    InvalidDataTypeError,
    MissingValueError,
    DuplicateDataError,
    DuplicateDateError,
    FutureDateError
)


class DataLoader:

    """
    Responsible for:

    - Loading CSV
    - Column Validation
    - Datatype Validation
    - Missing Value Validation
    - Duplicate Row Validation
    - Date Validation

    Returns clean DataFrame.
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

    # =====================================================
    # Load CSV
    # =====================================================

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

        self.validate_missing_values()

        self.validate_duplicate_rows()

        self.validate_dates()

        logger.info(
            f"Successfully loaded {len(self.data)} rows."
        )

        return self.data

    # =====================================================
    # Column Validation
    # =====================================================

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

        logger.info("Column validation successful.")

    # =====================================================
    # Datatype Validation
    # =====================================================

    def validate_datatypes(self):

        logger.info("Validating datatypes...")

        try:
            self.data["Date"] = pd.to_datetime(
                self.data["Date"]
            )

        except Exception:
            raise InvalidDataTypeError(
                "Date column contains invalid values."
            )

        logger.info("Date column validated.")

        for column in self.NUMERIC_COLUMNS:

            try:
                self.data[column] = pd.to_numeric(
                    self.data[column]
                )

            except Exception:
                raise InvalidDataTypeError(
                    f"{column} contains invalid numeric values."
                )

        logger.info("Numeric column validation successful.")

    # =====================================================
    # Missing Value Validation
    # =====================================================

    def validate_missing_values(self):

        logger.info("Checking missing values...")

        missing_summary = self.data.isnull().sum()

        missing_columns = missing_summary[
            missing_summary > 0
        ]

        if not missing_columns.empty:

            error_message = "\nMissing values found:\n"

            for column, count in missing_columns.items():

                error_message += (
                    f"{column}: {count}\n"
                )

            raise MissingValueError(error_message)

        logger.info("No missing values found.")

    # =====================================================
    # Duplicate Row Validation
    # =====================================================

    def validate_duplicate_rows(self):

        logger.info("Checking duplicate rows...")

        duplicate_rows = self.data[
            self.data.duplicated()
        ]

        if not duplicate_rows.empty:

            duplicate_indexes = (
                duplicate_rows.index.tolist()
            )

            raise DuplicateDataError(
                f"Duplicate rows found at indexes: {duplicate_indexes}"
            )

        logger.info("No duplicate rows found.")

    # =====================================================
    # Date Validation
    # =====================================================

    def validate_dates(self):

        logger.info("Validating dates...")

        # Sort dates

        self.data = self.data.sort_values(
            by="Date"
        ).reset_index(drop=True)

        # Duplicate Dates

        duplicate_dates = self.data[
            self.data["Date"].duplicated()
        ]

        if not duplicate_dates.empty:

            raise DuplicateDateError(
                "Duplicate dates found."
            )

        # Future Dates

        today = pd.Timestamp.today()

        future_dates = self.data[
            self.data["Date"] > today
        ]

        if not future_dates.empty:

            raise FutureDateError(
                "Future dates detected."
            )

        logger.info(
            "Date validation successful."
        )