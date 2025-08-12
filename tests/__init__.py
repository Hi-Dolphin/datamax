"""DataMax Test Suite

Comprehensive test suite for the DataMax project.

This package contains tests for all DataMax modules:
- Crawler module tests
- Parser module tests  
- CLI module tests
- Integration tests

To run tests:
    pytest tests/
    
To run specific test categories:
    pytest tests/ -m "not slow"  # Skip slow tests
    pytest tests/ --run-integration  # Run integration tests
    pytest tests/ --run-network  # Run network-dependent tests
"""

__version__ = '1.0.0'
__author__ = 'DataMax Development Team'