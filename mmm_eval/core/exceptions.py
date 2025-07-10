"""Custom exceptions for MMM validation framework."""


class ValidationError(Exception):
    """Base exception for validation framework errors."""

    pass


class MetricCalculationError(ValidationError):
    """Raised when metric calculation fails."""

    pass


class TestExecutionError(ValidationError):
    """Raised when test execution fails."""

    pass


class InvalidTestNameError(ValidationError):
    """Raised when an invalid test name is provided."""

    pass
