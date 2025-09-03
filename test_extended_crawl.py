#!/usr/bin/env python3
"""
Extended DataMax One-Line Crawler Test

This script tests the extended functionality of the one-line crawler.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

import datamax


def test_single_engine():
    \"\"\"Test single engine crawling.\"\"\"
    print("=== Testing Single Engine Crawling ===")
    
    # Test ArXiv
    try:
        result = datamax.crawl("machine learning", engine="arxiv")
        print(f"✅ ArXiv crawl successful: Found {result.get('total_results', 0)} papers")
    except Exception as e:
        print(f"❌ ArXiv crawl failed: {e}")
    
    # Test Web
    try:
        result = datamax.crawl("https://httpbin.org/html", engine="web")
        print(f"✅ Web crawl successful: Crawled {result.get('url', 'unknown')}")
    except Exception as e:
        print(f"❌ Web crawl failed: {e}")
    
    print()


def test_auto_mode():
    \"\"\"Test auto mode with all engines.\"\"\"
    print("=== Testing Auto Mode (All Engines) ===")
    
    try:
        result = datamax.crawl("machine learning", engine="auto")
        print(f"✅ Auto crawl successful:")
        print(f"   - Total engines: {result.get('total_engines', 0)}")
        print(f"   - Successful engines: {result.get('successful_engines', 0)}")
        print(f"   - Failed engines: {result.get('failed_engines', 0)}")
        
        # Show results from successful engines
        results = result.get('results', [])
        for res in results:
            engine = res.get('engine')
            data = res.get('data', {})
            if engine == 'arxiv':
                print(f"   - ArXiv: Found {data.get('total_results', 0)} papers")
            elif engine == 'web':
                print(f"   - Web: Crawled {data.get('url', 'unknown')}")
                
        # Show errors if any
        errors = result.get('errors', [])
        for error in errors:
            engine = error.get('engine')
            err = error.get('error')
            print(f"   - Error in {engine}: {err}")
            
    except Exception as e:
        print(f"❌ Auto crawl failed: {e}")
    
    print()


def test_convenience_functions():
    \"\"\"Test convenience functions.\"\"\"
    print("=== Testing Convenience Functions ===")
    
    try:
        result = datamax.crawler.crawl_arxiv("neural networks")
        print(f"✅ crawl_arxiv successful: Found {result.get('total_results', 0)} papers")
    except Exception as e:
        print(f"❌ crawl_arxiv failed: {e}")
    
    try:
        result = datamax.crawler.crawl_web("https://httpbin.org/html")
        print(f"✅ crawl_web successful: Crawled {result.get('url', 'unknown')}")
    except Exception as e:
        print(f"❌ crawl_web failed: {e}")
    
    print()


def test_future_extensibility():
    \"\"\"Test that the design supports future extensibility.\"\"\"
    print("=== Testing Future Extensibility ===")
    
    # Show what engines are currently available
    from datamax.crawler.crawler_factory import get_factory
    factory = get_factory()
    crawlers = factory.list_crawlers()
    print(f"Currently registered engines: {crawlers}")
    
    # Show that auto mode will automatically use any new engines
    print("The auto mode will automatically use all registered engines,")
    print("making it easy to add new engines in the future without")
    print("changing the crawl() function interface.")
    print()


def main():
    \"\"\"Run all tests.\"\"\"
    print("=== DataMax One-Line Crawler Extended Test ===\\n")
    
    test_single_engine()
    test_auto_mode()
    test_convenience_functions()
    test_future_extensibility()
    
    print("=== Extended Test Completed ===")


if __name__ == "__main__":
    main()
