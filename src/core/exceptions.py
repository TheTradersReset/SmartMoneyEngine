"""
Custom Exceptions for Smart Money Engine
"""


class SmartMoneyEngineError(Exception):
    """Base Exception"""
    pass


class CSVFileNotFoundError(SmartMoneyEngineError):
    """Raised when CSV file is not found"""
    pass


class InvalidCSVFormatError(SmartMoneyEngineError):
    """Raised when CSV format is invalid"""
    pass


class MissingColumnError(SmartMoneyEngineError):
    """Raised when required columns are missing"""
    pass


class EmptyDataFrameError(SmartMoneyEngineError):
    """Raised when dataframe is empty"""
    pass