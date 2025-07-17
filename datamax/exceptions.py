"""Custom exceptions for DataMax SDK."""


class DataMaxError(Exception):
    """Base exception for all DataMax errors."""

    pass


class ParseError(DataMaxError):
    """Raised when file parsing fails."""

    pass


class UnsupportedFormatError(DataMaxError):
    """Raised when file format is not supported."""

    pass


class ConfigurationError(DataMaxError):
    """Raised when configuration is invalid."""

    pass


class AuthenticationError(DataMaxError):
    """Raised when authentication fails for cloud storage."""

    pass


class NetworkError(DataMaxError):
    """Raised when network operations fail."""

    pass


class DataCleaningError(DataMaxError):
    """Raised when data cleaning operations fail."""

    pass


class AIAnnotationError(DataMaxError):
    """Raised when AI annotation fails."""

    pass


class CacheError(DataMaxError):
    """Raised when cache operations fail."""

    pass