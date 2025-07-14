"""DataMax - A powerful multi-format file parsing, data cleaning and AI annotation toolkit."""

__version__ = "0.2.0"
__author__ = "ccy"
__email__ = "cy.kron@foxmail.com"

# Main client interface (recommended)
from datamax.client import DataMaxClient

# Core components
from datamax.parser.core import DataMax
from datamax.loader.core import DataLoader
from datamax.config import configure, get_settings, DataMaxSettings

# Exceptions
from datamax.exceptions import (
    DataMaxError,
    ParseError,
    UnsupportedFormatError,
    ConfigurationError,
    AuthenticationError,
    NetworkError,
    DataCleaningError,
    AIAnnotationError,
    CacheError,
)

# Convenience imports
parse_file = DataMaxClient().parse_file
parse_files = DataMaxClient().parse_files
parse_directory = DataMaxClient().parse_directory

__all__ = [
    # Main interface
    "DataMaxClient",
    
    # Core classes
    "DataMax",
    "DataLoader", 
    "DataMaxSettings",
    
    # Configuration
    "configure",
    "get_settings",
    
    # Exceptions
    "DataMaxError",
    "ParseError", 
    "UnsupportedFormatError",
    "ConfigurationError",
    "AuthenticationError",
    "NetworkError",
    "DataCleaningError",
    "AIAnnotationError", 
    "CacheError",
    
    # Convenience functions
    "parse_file",
    "parse_files", 
    "parse_directory",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
]
