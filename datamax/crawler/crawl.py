"""One-line crawler interface for DataMax.

Provides a simple interface for crawling data from various sources.
"""

import asyncio
import json
from typing import Dict, Any, Optional, List
from .crawler_factory import create_crawler, get_factory


async def _async_crawl_single(engine: str, keyword: str, count: int = 10) -> Dict[str, Any]:
    """Asynchronously crawl data using a single engine.
    
    Args:
        engine: Crawler engine to use
        keyword: Search keyword or target URL/ID
        count: Number of results to return (for search engines)
        
    Returns:
        Crawled data dictionary
        
    Raises:
        Exception: If crawling fails
    """
    try:
        # Create crawler and validate target first
        crawler = create_crawler(engine)
        if hasattr(crawler, 'validate_target'):
            if not crawler.validate_target(keyword):
                return {
                    "engine": engine, 
                    "success": False, 
                    "error": f"Target validation failed for {engine} engine"
                }
        
        # Pass count parameter to crawler
        result = await crawler.crawl(keyword, max_results=count)
        return {"engine": engine, "success": True, "data": result}
    except Exception as e:
        return {"engine": engine, "success": False, "error": str(e)}


async def _async_crawl_all(keyword: str, count: int = 10) -> Dict[str, Any]:
    """Asynchronously crawl data using all available engines.
    
    Args:
        keyword: Search keyword or target URL/ID
        count: Number of results to return (for search engines)
        
    Returns:
        Combined crawled data dictionary from all engines
    """
    # Get all registered crawler types
    factory = get_factory()
    crawler_types = factory.list_crawlers()
    
    # Run all crawlers concurrently
    tasks = [_async_crawl_single(engine, keyword, count) for engine in crawler_types]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Process results
    successful_results = []
    failed_results = []
    
    for result in results:
        if isinstance(result, dict):
            if result.get("success"):
                successful_results.append(result)
            else:
                failed_results.append(result)
        else:
            # Handle exceptions
            failed_results.append({
                "engine": "unknown",
                "success": False,
                "error": str(result)
            })
    
    # Combine successful results
    combined_data = {
        "type": "combined_results",
        "keyword": keyword,
        "count": count,
        "engines_used": crawler_types,
        "total_engines": len(crawler_types),
        "successful_engines": len(successful_results),
        "failed_engines": len(failed_results),
        "results": successful_results,
        "errors": failed_results
    }
    
    return combined_data


async def _async_crawl(keyword: str, engine: str = "auto", count: int = 10) -> Dict[str, Any]:
    """Asynchronously crawl data based on keyword and engine.
    
    Args:
        keyword: Search keyword or target URL/ID
        engine: Crawler engine to use ("arxiv", "web", "auto")
        count: Number of results to return (for search engines)
        
    Returns:
        Crawled data dictionary
        
    Raises:
        Exception: If crawling fails
    """
    try:
        if engine == "auto":
            # Use all available engines
            return await _async_crawl_all(keyword, count)
        else:
            # Use specified crawler
            crawler = create_crawler(engine)
            result = await crawler.crawl(keyword, max_results=count)
            return result
        
    except Exception as e:
        raise Exception(f"Crawling failed: {str(e)}") from e


def crawl(keyword: str, engine: str = "auto", count: int = 10) -> Dict[str, Any]:
    """Crawl data based on keyword and engine.
    
    Examples:
        >>> datamax.crawl("航运", engine="arxiv", count=5)
        >>> datamax.crawl("航运", engine="web", count=5)
        >>> datamax.crawl("航运", count=5)  # Uses all engines
        
    Args:
        keyword: Search keyword or target URL/ID
        engine: Crawler engine to use ("arxiv", "web", "auto")
        count: Number of results to return (for search engines)
        
    Returns:
        Crawled data dictionary
        
    Raises:
        Exception: If crawling fails
    """
    try:
        # Run the async crawl function
        return asyncio.run(_async_crawl(keyword, engine, count))
    except Exception as e:
        raise Exception(f"Crawling failed: {str(e)}") from e


# Convenience functions for specific engines
def crawl_arxiv(keyword: str, count: int = 10) -> Dict[str, Any]:
    """Crawl ArXiv data.
    
    Args:
        keyword: ArXiv ID, URL, or search query
        count: Number of results to return (for search queries)
        
    Returns:
        Crawled data dictionary
    """
    return crawl(keyword, engine="arxiv", count=count)


def crawl_web(keyword: str, count: int = 10) -> Dict[str, Any]:
    """Search the web using keywords.
    
    Args:
        keyword: Search keyword
        count: Number of results to return
        
    Returns:
        Search results dictionary
    """
    return crawl(keyword, engine="web", count=count)