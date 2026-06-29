"""
Custom Exceptions for Smart Money Engine
"""


class SmartMoneyEngineError(Exception):
    """
    Base Exception for Smart Money Engine.
    """
    pass


# ==========================================================
# CSV / File Exceptions
# ==========================================================

class CSVFileNotFoundError(SmartMoneyEngineError):
    """Raised when CSV file is not found."""
    pass


class InvalidCSVFormatError(SmartMoneyEngineError):
    """Raised when CSV format is invalid."""
    pass


class EmptyDataFrameError(SmartMoneyEngineError):
    """Raised when CSV contains no data."""
    pass


# ==========================================================
# Validation Exceptions
# ==========================================================

class MissingColumnError(SmartMoneyEngineError):
    """Raised when required columns are missing."""
    pass


class InvalidDataTypeError(SmartMoneyEngineError):
    """Raised when datatype is invalid."""
    pass


class MissingValueError(SmartMoneyEngineError):
    """Raised when missing values are found."""
    pass


class DuplicateDataError(SmartMoneyEngineError):
    """Raised when duplicate rows are found."""
    pass


class DuplicateDateError(SmartMoneyEngineError):
    """Raised when duplicate dates are found."""
    pass


class InvalidDateError(SmartMoneyEngineError):
    """Raised when invalid dates are found."""
    pass


class FutureDateError(SmartMoneyEngineError):
    """Raised when future dates are found."""
    pass


class InvalidOHLCError(SmartMoneyEngineError):
    """Raised when OHLC values are logically incorrect."""
    pass