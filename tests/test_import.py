#!/usr/bin/env python3
"""Test script to reproduce the import issue."""

import sys
import traceback

def test_imports():
    """Test various import scenarios."""
    print("Testing imports...")
    
    # Test 1: Basic package import
    try:
        import datamax
        print("✓ import datamax - SUCCESS")
    except Exception as e:
        print(f"✗ import datamax - FAILED: {e}")
        traceback.print_exc()
    
    # Test 2: Import DataMax from datamax
    try:
        from datamax import DataMax
        print("✓ from datamax import DataMax - SUCCESS")
    except Exception as e:
        print(f"✗ from datamax import DataMax - FAILED: {e}")
        traceback.print_exc()
    
    # Test 3: Import DataMax directly from parser
    try:
        from datamax.parser import DataMax
        print("✓ from datamax.parser import DataMax - SUCCESS")
    except Exception as e:
        print(f"✗ from datamax.parser import DataMax - FAILED: {e}")
        traceback.print_exc()

    # Test 4: Import DataMax directly from core
    try:
        from datamax.parser.core import DataMax
        print("✓ from datamax.parser.core import DataMax - SUCCESS")
    except Exception as e:
        print(f"✗ from datamax.parser.core import DataMax - FAILED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()
