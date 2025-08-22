#!/usr/bin/env python3
"""
One-line Crawler Example

This example demonstrates how to use the one-line crawler interface
to crawl data from various sources with minimal code.

Usage:
    python one_line_crawler_example.py
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import datamax


def main():
    """Demonstrate one-line crawling functionality."""
    print("=== DataMax One-Line Crawler Example ===\n")

    # Example 3: Auto mode - use all engines
    print("1. Auto mode - using all available engines...")
    try:
        result = datamax.crawl("machine learning", engine="auto")
        print(f"   Success! Used {result.get('total_engines', 0)} engines")
        print(f"   Successful engines: {result.get('successful_engines', 0)}")
        print(f"   Failed engines: {result.get('failed_engines', 0)}")
        
        # Show results from successful engines
        results = result.get('results', [])
        for res in results:
            engine = res.get('engine')
            data = res.get('data', {})
            print(res)
            if engine == 'arxiv':
                print(f"   - ArXiv: Found {data.get('total_results', 0)} papers")
            elif engine == 'web':
                print(f"   - Web: Crawled {data.get('url', 'unknown')}")
        print()
    except Exception as e:
        print(f"   Failed: {e}\n")
    
    print("=== Example completed ===")


if __name__ == "__main__":
    main()