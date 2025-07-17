"""
DataMax: A comprehensive data parsing and processing library.

This package provides tools for parsing various document formats and processing data
with support for multiple file types including PDF, DOCX, HTML, CSV, and more.
"""

from .parser import DataMax
from .utils import (
    clean_original_text,
    clean_original_privacy_text,
    setup_environment,
)

# Re-export commonly used utilities
__all__ = [
    "DataMax",
    "clean_original_text",
    "clean_original_privacy_text",
    "setup_environment",
]

# Package metadata
__version__ = "0.1.18rc1"
__author__ = "DataMax Team"
__description__ = "A comprehensive data parsing and processing library"
