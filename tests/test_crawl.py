#!/usr/bin/env python3
"""Test script for the crawl function."""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import datamax


def test_crawl_function():
    """Test the crawl function with different engines."""
    print("Testing crawl function...")
    
    # Test with a web search keyword
    try:
        result = datamax.crawl("latest AI news", engine="web")
        print("Web search successful:")
        print(f"  Type: {result.get('type')}")
        print(f"  Query: {result.get('query')}")
        print(f"  Result count: {result.get('result_count')}")
        if result.get('results'):
            print(f"  First result title: {result['results'][0].get('title')}")
        print()
    except Exception as e:
        print(f"Web search failed: {e}")
        print()
    
    # Test with arxiv
    try:
        result = datamax.crawl("machine learning", engine="arxiv")
        print("ArXiv crawl successful:")
        print(f"  Type: {result.get('type')}")
        print(f"  Target: {result.get('target')}")
        if result.get('type') == 'search_results':
            print(f"  Total results: {result.get('total_results')}")
            data = result.get('data', [])
            if data:
                print(f"  First result title: {data[0].get('title')}")
        print()
    except Exception as e:
        print(f"ArXiv crawl failed: {e}")
        print()
    
    # Test with auto engine using all crawlers
    try:
        result = datamax.crawl("machine learning", engine="auto")
        print("Auto crawl with all engines successful:")
        print(f"  Type: {result.get('type')}")
        print(f"  Total engines: {result.get('total_engines')}")
        print(f"  Successful engines: {result.get('successful_engines')}")
        print(f"  Failed engines: {result.get('failed_engines')}")
        
        # Show results from successful engines
        results = result.get('results', [])
        for res in results:
            engine = res.get('engine')
            data = res.get('data', {})
            if engine == 'arxiv':
                print(f"  - ArXiv: Found {data.get('total_results', 0)} papers")
            elif engine == 'web':
                print(f"  - Web: Found {data.get('result_count', 0)} results")
        print()
    except Exception as e:
        print(f"Auto crawl with all engines failed: {e}")
        print()


if __name__ == "__main__":
    test_crawl_function()