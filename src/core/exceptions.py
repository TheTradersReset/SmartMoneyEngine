"""
Custom Exceptions for Smart Money Engine
========================================

This module contains all custom exceptions used
throughout the SmartMoneyEngine project.

Author : Bharat Gupta
Project: SmartMoneyEngine
"""


# =====================================================
# Base Exception
# =====================================================

class SmartMoneyEngineError(Exception):
    """
    Base Exception for SmartMoneyEngine.

    All custom exceptions should inherit from this class.
    """
    pass


# =====================================================
# File Related Exceptions
# =====================================================

class CSVFileNotFoundError(SmartMoneyEngineError):
    """
    Raised when the specified CSV file is not found.
    """
    pass


class InvalidCSVFormatError(SmartMoneyEngineError):
    """
    Raised when the CSV file format is invalid.
    """
    pass


class EmptyDataFrameError(SmartMoneyEngineError):
    """
    Raised when the loaded dataframe is empty.
    """
    pass


# =====================================================
# Validation Exceptions
# =====================================================

class MissingColumnError(SmartMoneyEngineError):
    """
    Raised when one or more required columns are missing.
    """
    pass


class InvalidDataTypeError(SmartMoneyEngineError):
    """
    Raised when a column contains invalid datatype.
    """
    pass


class MissingValueError(SmartMoneyEngineError):
    """
    Raised when required values are missing.
    """
    pass


class DuplicateDataError(SmartMoneyEngineError):
    """
    Raised when duplicate rows are detected.
    """
    pass


class InvalidDateError(SmartMoneyEngineError):
    """
    Raised when the Date column contains invalid values.
    """
    pass


class InvalidOHLCError(SmartMoneyEngineError):
    """
    Raised when OHLC values violate market rules.

    Example:
        High < Low
        Open > High
        Close < Low
    """
    pass