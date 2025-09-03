"""Web Crawler Implementation

Provides general-purpose web crawler for HTML pages and web search API integration.
"""

import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
from urllib.parse import urljoin, urlparse
from datetime import datetime
from .base_crawler import BaseCrawler
from .exceptions import CrawlerException, NetworkException, ParseException


class WebCrawler(BaseCrawler):
    """General-purpose web crawler for HTML pages and web search API.
    
    Can crawl individual web pages or perform web searches using search APIs.
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
        self.follow_redirects = self.config.get('follow_redirects', True)
        self.max_content_length = self.config.get('max_content_length', 10 * 1024 * 1024)  # 10MB
        
        # Search API configuration
        self.search_api_key = self.config.get('search_api_key')
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
        if not hasattr(self, 'follow_redirects'):
            self.follow_redirects = self.config.get('follow_redirects', True)
        if not hasattr(self, 'max_content_length'):
            self.max_content_length = self.config.get('max_content_length', 10 * 1024 * 1024)  # 10MB
            
        # Set search API attributes
        if not hasattr(self, 'search_api_key'):
            self.search_api_key = self.config.get('search_api_key')
        if not hasattr(self, 'search_api_url'):
            self.search_api_url = self.config.get('search_api_url', 'https://api.bochaai.com/v1/web-search')
            
        # Headers for web requests
        self.headers = {
            'User-Agent': self.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }

    def validate_target(self, target: str) -> bool:
        """Validate if the target is a valid URL or search keyword.
        
        Args:
            target: Target URL or search keyword to validate
            
        Returns:
            True if target is valid (URL or non-empty keyword)
        """
        try:
            # Empty targets are not valid
            if not target or not target.strip():
                return False
                
            # Check if it's a valid URL
            parsed = urlparse(target)
            is_url = parsed.scheme in ['http', 'https'] and bool(parsed.netloc)
            
            # For search keywords, ensure they're not just URLs in disguise
            # Any non-empty string is valid as a search keyword
            return True
        except Exception:
            return False

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

    async def _fetch_page(self, url: str) -> Dict[str, Any]:
        """Fetch web page content.
        
        Args:
            url: URL to fetch
            
        Returns:
            Dictionary containing response data
            
        Raises:
            NetworkException: If request fails
        """
        session = await self._get_session()
        
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.rate_limit)  # Rate limiting
                
                async with session.get(
                    url,
                    allow_redirects=self.follow_redirects,
                    max_redirects=5
                ) as response:
                    # Check content length
                    content_length = response.headers.get('content-length')
                    if content_length and int(content_length) > self.max_content_length:
                        raise NetworkException(f"Content too large: {content_length} bytes")
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(ct in content_type for ct in ['text/html', 'application/xhtml', 'text/plain']):
                        raise NetworkException(f"Unsupported content type: {content_type}")
                    
                    if response.status == 200:
                        content = await response.text(encoding='utf-8', errors='ignore')
                        
                        return {
                            'url': str(response.url),
                            'status_code': response.status,
                            'headers': dict(response.headers),
                            'content': content,
                            'content_type': content_type,
                            'encoding': response.charset or 'utf-8'
                        }
                    else:
                        raise NetworkException(f"HTTP {response.status}: {response.reason}")
                        
            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    raise NetworkException(f"Timeout fetching {url}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except aiohttp.ClientError as e:
                if attempt == self.max_retries - 1:
                    raise NetworkException(f"Client error fetching {url}: {str(e)}")
                await asyncio.sleep(2 ** attempt)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise NetworkException(f"Failed to fetch {url}: {str(e)}")
                await asyncio.sleep(2 ** attempt)

    async def _search_web_api(self, query: str) -> Dict[str, Any]:
        """Perform web search using search API.
        
        Args:
            query: Search query
            
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
            "count": 10
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

    def _extract_text_content(self, html_content: str) -> str:
        """Extract clean text content from HTML.
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Cleaned text content
        """
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            # Fallback: simple HTML tag removal using regex
            import re
            text = re.sub(r'<[^>]+>', '', html_content)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        except Exception as e:
            raise ParseException(f"Failed to extract text content: {str(e)}")

    def _extract_metadata(self, html_content: str, url: str) -> Dict[str, Any]:
        """Extract metadata from HTML.
        
        Args:
            html_content: Raw HTML content
            url: Page URL
            
        Returns:
            Dictionary containing extracted metadata
        """
        metadata = {
            'title': '',
            'description': '',
            'keywords': [],
            'author': '',
            'language': '',
            'canonical_url': url,
            'og_title': '',
            'og_description': '',
            'og_image': '',
            'twitter_title': '',
            'twitter_description': ''
        }
        
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Title
            title_tag = soup.find('title')
            if title_tag:
                metadata['title'] = title_tag.get_text().strip()
            
            # Meta tags
            for meta in soup.find_all('meta'):
                name = meta.get('name', '').lower()
                property_attr = meta.get('property', '').lower()
                content = meta.get('content', '')
                
                if name == 'description':
                    metadata['description'] = content
                elif name == 'keywords':
                    metadata['keywords'] = [k.strip() for k in content.split(',')]
                elif name == 'author':
                    metadata['author'] = content
                elif name == 'language' or name == 'lang':
                    metadata['language'] = content
                elif property_attr == 'og:title':
                    metadata['og_title'] = content
                elif property_attr == 'og:description':
                    metadata['og_description'] = content
                elif property_attr == 'og:image':
                    metadata['og_image'] = content
                elif name == 'twitter:title':
                    metadata['twitter_title'] = content
                elif name == 'twitter:description':
                    metadata['twitter_description'] = content
            
            # Canonical URL
            canonical = soup.find('link', rel='canonical')
            if canonical and canonical.get('href'):
                metadata['canonical_url'] = urljoin(url, canonical['href'])
            
            # Language from html tag
            if not metadata['language']:
                html_tag = soup.find('html')
                if html_tag:
                    metadata['language'] = html_tag.get('lang', '')
            
            return metadata
            
        except ImportError:
            # Fallback: basic regex extraction
            import re
            
            title_match = re.search(r'<title[^>]*>([^<]+)</title>', html_content, re.IGNORECASE)
            if title_match:
                metadata['title'] = title_match.group(1).strip()
            
            desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\'>]+)["\']', html_content, re.IGNORECASE)
            if desc_match:
                metadata['description'] = desc_match.group(1)
            
            return metadata
            
        except Exception as e:
            # Return basic metadata on error
            return metadata

    def _extract_links(self, html_content: str, base_url: str) -> List[Dict[str, str]]:
        """Extract links from HTML content.
        
        Args:
            html_content: Raw HTML content
            base_url: Base URL for resolving relative links
            
        Returns:
            List of link dictionaries
        """
        links = []
        
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                text = link.get_text().strip()
                
                # Skip empty links and javascript/mailto links
                if not href or href.startswith(('javascript:', 'mailto:', 'tel:')):
                    continue
                
                # Resolve relative URLs
                absolute_url = urljoin(base_url, href)
                
                # Validate URL
                parsed = urlparse(absolute_url)
                if parsed.scheme in ['http', 'https']:
                    links.append({
                        'url': absolute_url,
                        'text': text,
                        'title': link.get('title', '')
                    })
            
            return links
            
        except ImportError:
            # Fallback: basic regex extraction
            import re
            
            link_pattern = r'<a[^>]*href=["\']([^"\'>]+)["\'][^>]*>([^<]*)</a>'
            matches = re.findall(link_pattern, html_content, re.IGNORECASE)
            
            for href, text in matches:
                if href and not href.startswith(('javascript:', 'mailto:', 'tel:')):
                    absolute_url = urljoin(base_url, href.strip())
                    parsed = urlparse(absolute_url)
                    if parsed.scheme in ['http', 'https']:
                        links.append({
                            'url': absolute_url,
                            'text': text.strip(),
                            'title': ''
                        })
            
            return links
            
        except Exception:
            return links

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
            target: Web page URL or search query
            **kwargs: Additional parameters
            
        Returns:
            Crawled data dictionary
        """
        try:
            # Validate URL
            if not self.validate_target(target):
                raise ValueError(f"Invalid target: {target}")
            
            # Check if target is a URL or search keyword
            parsed = urlparse(target)
            is_url = parsed.scheme in ['http', 'https'] and bool(parsed.netloc)
            
            if is_url:
                # Direct URL crawling
                response_data = await self._fetch_page(target)
                
                # Extract content
                html_content = response_data['content']
                text_content = self._extract_text_content(html_content)
                metadata = self._extract_metadata(html_content, response_data['url'])
                links = self._extract_links(html_content, response_data['url'])
                
                return {
                    'type': 'web_page',
                    'url': response_data['url'],
                    'original_url': target,
                    'status_code': response_data['status_code'],
                    'content_type': response_data['content_type'],
                    'encoding': response_data['encoding'],
                    'metadata': metadata,
                    'text_content': text_content,
                    'links': links,
                    'link_count': len(links),
                    'content_length': len(text_content),
                    'crawled_at': datetime.now().isoformat(),
                    'source': 'web'
                }
            else:
                # Search keyword - use web search API
                search_data = await self._search_web_api(target)
                return self._format_search_results(search_data, target)
                
        except Exception as e:
            if isinstance(e, (CrawlerException, NetworkException, ParseException)):
                raise
            raise CrawlerException(f"Web crawling failed: {str(e)}") from e
        finally:
            await self._close_session()

    async def crawl_multiple_async(self, urls: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Crawl multiple URLs asynchronously.
        
        Args:
            urls: List of URLs to crawl
            **kwargs: Additional parameters
            
        Returns:
            List of crawled data dictionaries
        """
        tasks = []
        for url in urls:
            task = asyncio.create_task(self.crawl_async(url, **kwargs))
            tasks.append(task)
        
        results = []
        for task in asyncio.as_completed(tasks):
            try:
                result = await task
                results.append(result)
            except Exception as e:
                # Log error but continue with other URLs
                results.append({
                    'error': str(e),
                    'url': None,
                    'status': 'failed'
                })
        
        return results

    async def crawl(self, target: str) -> Dict[str, Any]:
        """Crawl web page or perform web search.
        
        Args:
            target: Target URL or search keyword to crawl
            
        Returns:
            Dictionary containing crawled data
            
        Raises:
            CrawlerException: If crawling fails
        """
        try:
            # Check if target is a URL or search keyword
            parsed = urlparse(target)
            is_url = parsed.scheme in ['http', 'https'] and bool(parsed.netloc)
            
            if is_url:
                # Direct URL crawling
                response_data = await self._fetch_page(target)
                
                # Extract content
                html_content = response_data['content']
                text_content = self._extract_text_content(html_content)
                metadata = self._extract_metadata(html_content, response_data['url'])
                links = self._extract_links(html_content, response_data['url'])
                
                return {
                    'type': 'web_page',
                    'url': response_data['url'],
                    'original_target': target,
                    'is_search': False,
                    'status_code': response_data['status_code'],
                    'content_type': response_data['content_type'],
                    'encoding': response_data['encoding'],
                    'metadata': metadata,
                    'text_content': text_content,
                    'links': links,
                    'link_count': len(links),
                    'content_length': len(text_content),
                    'crawled_at': datetime.now().isoformat(),
                    'source': 'web'
                }
            else:
                # Search keyword - use web search API
                search_data = await self._search_web_api(target)
                return self._format_search_results(search_data, target)
                
        except Exception as e:
            if isinstance(e, (CrawlerException, NetworkException, ParseException)):
                raise
            raise CrawlerException(f"Web crawling failed: {str(e)}") from e
        finally:
            await self._close_session()