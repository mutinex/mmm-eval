"""
Custom exceptions for data validation and processing.
"""


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class InvalidDateFormatError(Exception):
    """Raised when date parsing fails."""
    pass


class EmptyDataFrameError(Exception):
    """Raised when DataFrame is empty."""
    pass


class ValidationError(Exception):
    """Base class for validation errors."""
    pass