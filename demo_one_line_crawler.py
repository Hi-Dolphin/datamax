#!/usr/bin/env python3
"""
DataMax One-Line Crawler Demo

This script demonstrates the one-line crawler functionality we implemented.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import datamax


def main():
    """Demonstrate the one-line crawler functionality."""
    print("=== DataMax One-Line Crawler Demo ===\n")
    
    # Example 1: Crawl ArXiv papers with a search query
    print("1. Crawling ArXiv papers with search query 'machine learning'...")
    try:
        result = datamax.crawl("machine learning", engine="arxiv")
        print(f"   ✅ Success! Found {result.get('total_results', 0)} papers")
        if result.get('data'):
            first_paper = result['data'][0]
            print(f"   First paper: {first_paper.get('title', 'Unknown')}")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    # Example 2: Crawl a web page
    print("2. Crawling a web page...")
    try:
        result = datamax.crawl("https://httpbin.org/html", engine="web")
        print(f"   ✅ Success! Crawled page: {result.get('url')}")
        title = result.get('metadata', {}).get('title', 'No title')
        print(f"   Page title: {title}")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    # Example 3: Search the web using keywords
    print("3. Searching the web with keywords...")
    try:
        result = datamax.crawl("人工智能发展现状", engine="web")
        if result.get('type') == 'web_search_results':
            print(f"   ✅ Success! Found {result.get('result_count', 0)} search results")
            results = result.get('results', [])
            if results:
                first_result = results[0]
                print(f"   First result: {first_result.get('title', 'No title')}")
                print(f"   URL: {first_result.get('url', 'No URL')}")
        else:
            print(f"   ✅ Success! Crawled page: {result.get('url')}")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    # Example 4: Auto mode - use all engines
    print("4. Auto mode - using all available engines...")
    try:
        result = datamax.crawl("machine learning", engine="auto")
        print(f"   ✅ Success! Used {result.get('total_engines', 0)} engines")
        print(f"   Successful engines: {result.get('successful_engines', 0)}")
        print(f"   Failed engines: {result.get('failed_engines', 0)}")
        
        # Show results from successful engines
        results = result.get('results', [])
        for res in results:
            engine = res.get('engine')
            data = res.get('data', {})
            if engine == 'arxiv':
                print(f"   - ArXiv: Found {data.get('total_results', 0)} papers")
            elif engine == 'web':
                if data.get('type') == 'web_search_results':
                    print(f"   - Web: Found {data.get('result_count', 0)} search results")
                else:
                    print(f"   - Web: Crawled {data.get('url', 'unknown')}")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    # Example 5: Using convenience functions
    print("5. Using convenience functions...")
    try:
        result = datamax.crawler.crawl_arxiv("neural networks")
        print(f"   ✅ Success! Found {result.get('total_results', 0)} papers using crawl_arxiv")
        
        result = datamax.crawler.crawl_web("https://httpbin.org/html")
        print(f"   ✅ Success! Crawled page using crawl_web: {result.get('url')}")
        print()
    except Exception as e:
        print(f"   ❌ Failed: {e}\n")
    
    print("=== Demo completed ===")


if __name__ == "__main__":
    main()