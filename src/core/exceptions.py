class SmartMoneyEngineError(Exception):
    """Base Exception for Smart Money Engine"""
    pass


class DataNotFoundError(SmartMoneyEngineError):
    """Raised when market data is missing"""
    pass


class InvalidDataError(SmartMoneyEngineError):
    """Raised when data format is invalid"""
    pass


class IndicatorCalculationError(SmartMoneyEngineError):
    """Raised when indicator calculation fails"""
    pass