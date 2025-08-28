"""Web Crawler Implementation

Provides web search functionality using search engine APIs.
"""

import asyncio
import aiohttp
import os
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse
from datetime import datetime
from .base_crawler import BaseCrawler
from .exceptions import CrawlerException, NetworkException


class WebCrawler(BaseCrawler):
    """Web crawler that performs searches using search engine APIs.
    
    This crawler takes search keywords as input and returns search results
    from a configured search API.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize web crawler.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.config = config or {}
        self.user_agent = self.config.get('user_agent', 'DataMax-Crawler/1.0')
        self.timeout = self.config.get('timeout', 15)
        self.max_retries = self.config.get('max_retries', 2)
        self.rate_limit = self.config.get('rate_limit', 0.5)
        
        # Search API configuration - get from environment variable
        self.search_api_key = os.environ.get('SEARCH_API_KEY')
        self.search_api_url = self.config.get('search_api_url', 'https://api.bochaai.com/v1/web-search')
        
        self.session = None

    def _setup_crawler(self):
        """Setup web crawler specific configurations."""
        # Set attributes from config if not already set
        if not hasattr(self, 'user_agent'):
            self.user_agent = self.config.get('user_agent', 'DataMax-Crawler/1.0')
        if not hasattr(self, 'timeout'):
            self.timeout = self.config.get('timeout', 15)
        if not hasattr(self, 'max_retries'):
            self.max_retries = self.config.get('max_retries', 2)
        if not hasattr(self, 'rate_limit'):
            self.rate_limit = self.config.get('rate_limit', 0.5)
            
        # Set search API attributes
        if not hasattr(self, 'search_api_key'):
            self.search_api_key = self.config.get('search_api_key')
        if not hasattr(self, 'search_api_url'):
            self.search_api_url = self.config.get('search_api_url', 'https://api.bochaai.com/v1/web-search')
            
        # Headers for web requests
        self.headers = {
            'User-Agent': self.user_agent,
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive'
        }

    def validate_target(self, target: str) -> bool:
        """Validate if the target is a valid search keyword.
        
        Args:
            target: Search keyword to validate
            
        Returns:
            True if target is a non-empty string
        """
        # Empty targets are not valid
        if not target or not target.strip():
            return False
            
        # Any non-empty string is valid as a search keyword
        return True

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session.
        
        Returns:
            aiohttp ClientSession
        """
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers=self.headers,
                connector=connector
            )
        return self.session

    async def _close_session(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()

    async def _search_web_api(self, query: str, count: int = 10) -> Dict[str, Any]:
        """Perform web search using search API.
        
        Args:
            query: Search query
            count: Number of results to return
            
        Returns:
            Dictionary containing search results
            
        Raises:
            NetworkException: If search API request fails
        """
        if not self.search_api_key:
            raise NetworkException("Search API key not configured")
            
        session = await self._get_session()
        
        # Prepare search request
        search_data = {
            "query": query,
            "summary": True,
            "count": count
        }
        
        headers = {
            'Authorization': f'Bearer {self.search_api_key}',
            'Content-Type': 'application/json'
        }
        
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.rate_limit)  # Rate limiting
                
                async with session.post(
                    self.search_api_url,
                    json=search_data,
                    headers=headers
                ) as response:
                    if response.status == 200:
                        response_data = await response.json()
                        
                        # Check if response has data
                        if response_data.get('code') == 200 and response_data.get('data'):
                            return response_data['data']
                        else:
                            error_msg = response_data.get('msg', 'Unknown search API error')
                            raise NetworkException(f"Search API error: {error_msg}")
                    else:
                        raise NetworkException(f"Search API HTTP {response.status}: {response.reason}")
                        
            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    raise NetworkException(f"Timeout searching for '{query}'")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise NetworkException(f"Client error searching for '{query}': {str(e)}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise NetworkException(f"Failed to search for '{query}': {str(e)}")
                await asyncio.sleep(2 ** attempt)

    def _format_search_results(self, search_data: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Format search API results into web crawler format.
        
        Args:
            search_data: Raw search API response data
            query: Original search query
            
        Returns:
            Formatted search results
        """
        # Handle the response format from the API
        web_pages = search_data.get('webPages', {})
        results = web_pages.get('value', []) if isinstance(web_pages, dict) else []
        
        # If results is directly in search_data, use that
        if not results and isinstance(search_data.get('results'), list):
            results = search_data['results']
        
        # Format search results as web pages
        formatted_results = []
        for result in results:
            # Handle different result formats
            if isinstance(result, dict):
                formatted_results.append({
                    'url': result.get('url', ''),
                    'title': result.get('title', result.get('name', '')),
                    'description': result.get('description', result.get('snippet', '')),
                    'summary': result.get('summary', ''),
                    'site_name': result.get('siteName', ''),
                    'date_published': result.get('datePublished', result.get('dateLastCrawled', '')),
                    'display_url': result.get('displayUrl', result.get('url', ''))
                })
        
        return {
            'type': 'web_search_results',
            'query': query,
            'original_query': search_data.get('queryContext', {}).get('originalQuery', query),
            'total_estimated_matches': web_pages.get('totalEstimatedMatches', len(formatted_results)) if isinstance(web_pages, dict) else len(formatted_results),
            'results': formatted_results,
            'result_count': len(formatted_results),
            'crawled_at': datetime.now().isoformat(),
            'source': 'web_search'
        }

    async def crawl_async(self, target: str, **kwargs) -> Dict[str, Any]:
        """Async version of crawl method.
        
        Args:
            target: Search keyword
            **kwargs: Additional parameters (including max_results/count)
            
        Returns:
            Crawled data dictionary
        """
        try:
            # Validate target
            if not self.validate_target(target):
                raise ValueError(f"Invalid search keyword: {target}")
            
            # Get count from kwargs, default to 10
            count = kwargs.get('max_results', 10)
            
            # Perform search using web search API
            search_data = await self._search_web_api(target, count)
            return self._format_search_results(search_data, target)
                
        except Exception as e:
            if isinstance(e, (CrawlerException, NetworkException)):
                raise
            raise CrawlerException(f"Web search failed: {str(e)}") from e
        finally:
            await self._close_session()

    async def crawl(self, target: str, **kwargs) -> Dict[str, Any]:
        """Perform web search using keywords.
        
        Args:
            target: Search keyword
            **kwargs: Additional parameters (including max_results/count)
            
        Returns:
            Dictionary containing search results
            
        Raises:
            CrawlerException: If search fails
        """
        try:
            # Validate target
            if not self.validate_target(target):
                raise ValueError(f"Invalid search keyword: {target}")
            
            # Get count from kwargs, default to 10
            count = kwargs.get('max_results', 10)
            
            # Perform search using web search API
            search_data = await self._search_web_api(target, count)
            return self._format_search_results(search_data, target)
                
        except Exception as e:
            if isinstance(e, (CrawlerException, NetworkException)):
                raise
            raise CrawlerException(f"Web search failed: {str(e)}") from e
        finally:
            await self._close_session()