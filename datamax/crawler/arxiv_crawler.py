"""ArXiv Crawler Implementation

Provides specialized crawler for ArXiv academic papers.
"""

import re
import asyncio
import aiohttp
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, parse_qs
from datetime import datetime
from .base_crawler import BaseCrawler
from .exceptions import CrawlerException, NetworkException, ParseException


class ArxivCrawler(BaseCrawler):
    """Crawler for ArXiv academic papers.
    
    Supports crawling individual papers, search queries, and author pages.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize ArXiv crawler.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.base_url = self.config.get('base_url', 'https://arxiv.org/')
        self.api_url = 'http://export.arxiv.org/api/query'
        self.rate_limit = self.config.get('rate_limit', 1.0)  # seconds between requests
        self.timeout = self.config.get('timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.user_agent = self.config.get('user_agent', 'DataMax-Crawler/1.0')
        self.session = None
    
    def _setup_crawler(self):
        """Setup ArXiv-specific configurations."""
        # ArXiv ID patterns
        self.arxiv_id_pattern = re.compile(
            r'(?:arxiv:)?(?:(?:astro-ph|cond-mat|gr-qc|hep-ex|hep-lat|hep-ph|hep-th|math-ph|nlin|nucl-ex|nucl-th|physics|quant-ph|math|cs|q-bio|q-fin|stat)/\d{7}|\d{4}\.\d{4,5}(?:v\d+)?)',
            re.IGNORECASE
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session.
        
        Returns:
            aiohttp ClientSession
        """
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            headers = {'User-Agent': self.user_agent}
            self.session = aiohttp.ClientSession(timeout=timeout, headers=headers)
        return self.session
    
    async def _close_session(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def validate_target(self, target: str) -> bool:
        """Validate if the target is supported by ArXiv crawler.
        
        Args:
            target: Target URL, ArXiv ID, or search query
            
        Returns:
            True if target is valid for ArXiv crawler
        """
        # Check if it's an ArXiv URL
        if target.startswith(('http://', 'https://')):
            parsed = urlparse(target)
            return 'arxiv.org' in parsed.netloc.lower()
        
        # Check if it's an ArXiv ID
        if self.arxiv_id_pattern.search(target):
            return True
        
        # For search queries, we'll accept any string
        return True
    
    def _extract_arxiv_id(self, target: str) -> Optional[str]:
        """Extract ArXiv ID from various input formats.
        
        Args:
            target: URL, ArXiv ID, or other identifier
            
        Returns:
            Extracted ArXiv ID or None
        """
        # Direct ArXiv ID
        match = self.arxiv_id_pattern.search(target)
        if match:
            return match.group(0).replace('arxiv:', '')
        
        # URL parsing
        if target.startswith(('http://', 'https://')):
            parsed = urlparse(target)
            if 'arxiv.org' in parsed.netloc:
                # Extract from path like /abs/1234.5678 or /pdf/1234.5678.pdf
                path_match = re.search(r'/(abs|pdf)/([^/]+?)(?:\.pdf)?$', parsed.path)
                if path_match:
                    return path_match.group(2)
        
        return None
    
    async def _fetch_paper_metadata(self, arxiv_id: str) -> Dict[str, Any]:
        """Fetch paper metadata from ArXiv API.
        
        Args:
            arxiv_id: ArXiv paper ID
            
        Returns:
            Paper metadata dictionary
            
        Raises:
            NetworkException: If API request fails
            ParseException: If response parsing fails
        """
        session = await self._get_session()
        
        params = {
            'id_list': arxiv_id,
            'max_results': 1
        }
        
        for attempt in range(self.max_retries):
            try:
                await asyncio.sleep(self.rate_limit)  # Rate limiting
                
                async with session.get(self.api_url, params=params) as response:
                    if response.status == 200:
                        content = await response.text()
                        return self._parse_arxiv_response(content)
                    else:
                        raise NetworkException(f"ArXiv API returned status {response.status}")
                        
            except asyncio.TimeoutError:
                if attempt == self.max_retries - 1:
                    raise NetworkException(f"Timeout fetching ArXiv paper {arxiv_id}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise NetworkException(f"Failed to fetch ArXiv paper {arxiv_id}: {str(e)}")
                await asyncio.sleep(2 ** attempt)
    
    def _parse_arxiv_response(self, xml_content: str) -> Dict[str, Any]:
        """Parse ArXiv API XML response.
        
        Args:
            xml_content: XML response from ArXiv API
            
        Returns:
            Parsed paper metadata
            
        Raises:
            ParseException: If XML parsing fails
        """
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(xml_content)
            
            # Define namespaces
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entry = root.find('atom:entry', namespaces)
            if entry is None:
                raise ParseException("No entry found in ArXiv response")
            
            # Extract basic information
            title = entry.find('atom:title', namespaces)
            title_text = title.text.strip() if title is not None else "Unknown Title"
            
            summary = entry.find('atom:summary', namespaces)
            summary_text = summary.text.strip() if summary is not None else ""
            
            # Extract authors
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name_elem = author.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # Extract categories
            categories = []
            for category in entry.findall('atom:category', namespaces):
                term = category.get('term')
                if term:
                    categories.append(term)
            
            # Extract dates
            published = entry.find('atom:published', namespaces)
            published_date = published.text if published is not None else None
            
            updated = entry.find('atom:updated', namespaces)
            updated_date = updated.text if updated is not None else None
            
            # Extract ArXiv ID
            id_elem = entry.find('atom:id', namespaces)
            arxiv_id = None
            if id_elem is not None:
                id_url = id_elem.text
                arxiv_id = id_url.split('/')[-1] if id_url else None
            
            # Extract PDF link
            pdf_url = None
            for link in entry.findall('atom:link', namespaces):
                if link.get('type') == 'application/pdf':
                    pdf_url = link.get('href')
                    break
            
            # Extract DOI if available
            doi = None
            doi_elem = entry.find('arxiv:doi', namespaces)
            if doi_elem is not None:
                doi = doi_elem.text
            
            return {
                'arxiv_id': arxiv_id,
                'title': title_text,
                'authors': authors,
                'summary': summary_text,
                'categories': categories,
                'published_date': published_date,
                'updated_date': updated_date,
                'pdf_url': pdf_url,
                'doi': doi,
                'source': 'arxiv',
                'crawled_at': datetime.now().isoformat()
            }
            
        except ET.ParseError as e:
            raise ParseException(f"Failed to parse ArXiv XML response: {str(e)}")
        except Exception as e:
            raise ParseException(f"Error processing ArXiv response: {str(e)}")
    
    async def _search_papers(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for papers using ArXiv API.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of paper metadata dictionaries
        """
        session = await self._get_session()
        
        params = {
            'search_query': query,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        try:
            await asyncio.sleep(self.rate_limit)
            
            async with session.get(self.api_url, params=params) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._parse_search_response(content)
                else:
                    raise NetworkException(f"ArXiv search API returned status {response.status}")
                    
        except Exception as e:
            raise NetworkException(f"Failed to search ArXiv: {str(e)}")
    
    def _parse_search_response(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse ArXiv search API XML response.
        
        Args:
            xml_content: XML response from ArXiv API
            
        Returns:
            List of parsed paper metadata
        """
        try:
            import xml.etree.ElementTree as ET
            
            root = ET.fromstring(xml_content)
            
            namespaces = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            papers = []
            for entry in root.findall('atom:entry', namespaces):
                # Use the same parsing logic as single paper
                paper_data = self._parse_single_entry(entry, namespaces)
                papers.append(paper_data)
            
            return papers
            
        except Exception as e:
            raise ParseException(f"Error parsing ArXiv search response: {str(e)}")
    
    def _parse_single_entry(self, entry, namespaces: Dict[str, str]) -> Dict[str, Any]:
        """Parse a single entry from ArXiv XML.
        
        Args:
            entry: XML entry element
            namespaces: XML namespaces
            
        Returns:
            Parsed paper metadata
        """
        # This is the same logic as in _parse_arxiv_response but for a single entry
        title = entry.find('atom:title', namespaces)
        title_text = title.text.strip() if title is not None else "Unknown Title"
        
        summary = entry.find('atom:summary', namespaces)
        summary_text = summary.text.strip() if summary is not None else ""
        
        authors = []
        for author in entry.findall('atom:author', namespaces):
            name_elem = author.find('atom:name', namespaces)
            if name_elem is not None:
                authors.append(name_elem.text.strip())
        
        categories = []
        for category in entry.findall('atom:category', namespaces):
            term = category.get('term')
            if term:
                categories.append(term)
        
        published = entry.find('atom:published', namespaces)
        published_date = published.text if published is not None else None
        
        updated = entry.find('atom:updated', namespaces)
        updated_date = updated.text if updated is not None else None
        
        id_elem = entry.find('atom:id', namespaces)
        arxiv_id = None
        if id_elem is not None:
            id_url = id_elem.text
            arxiv_id = id_url.split('/')[-1] if id_url else None
        
        pdf_url = None
        for link in entry.findall('atom:link', namespaces):
            if link.get('type') == 'application/pdf':
                pdf_url = link.get('href')
                break
        
        doi = None
        doi_elem = entry.find('arxiv:doi', namespaces)
        if doi_elem is not None:
            doi = doi_elem.text
        
        return {
            'arxiv_id': arxiv_id,
            'title': title_text,
            'authors': authors,
            'summary': summary_text,
            'categories': categories,
            'published_date': published_date,
            'updated_date': updated_date,
            'pdf_url': pdf_url,
            'doi': doi,
            'source': 'arxiv',
            'crawled_at': datetime.now().isoformat()
        }
    
    async def _make_request(self, url: str, params: Optional[Dict] = None) -> str:
        """Make async HTTP request.
        
        Args:
            url: Request URL
            params: Query parameters
            
        Returns:
            Response text
            
        Raises:
            NetworkException: If request fails
        """
        session = await self._get_session()
        
        try:
            await asyncio.sleep(self.rate_limit)  # Rate limiting
            
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    raise NetworkException(f"HTTP {response.status}: {response.reason}")
        except aiohttp.ClientError as e:
            raise NetworkException(f"Request failed: {str(e)}") from e
    
    async def crawl_async(self, target: str, **kwargs) -> Dict[str, Any]:
        """Async version of crawl method.
        
        Args:
            target: ArXiv ID, URL, or search query
            **kwargs: Additional parameters
            
        Returns:
            Crawled data dictionary
        """
        try:
            # Validate target
            if not self.validate_target(target):
                raise ValueError(f"Invalid ArXiv target: {target}")
            
            # Try to extract ArXiv ID first
            arxiv_id = self._extract_arxiv_id(target)
            
            if arxiv_id:
                # Single paper crawling
                paper_data = await self._fetch_paper_metadata(arxiv_id)
                return {
                    'type': 'single_paper',
                    'target': target,
                    'data': paper_data
                }
            else:
                # Search query
                max_results = kwargs.get('max_results', self.config.get('search_max_results', 10))
                papers = await self._search_papers(target, max_results)
                return {
                    'type': 'search_results',
                    'target': target,
                    'query': target,
                    'total_results': len(papers),
                    'data': papers
                }
                
        except Exception as e:
            if isinstance(e, (CrawlerException, NetworkException, ParseException)):
                raise
            raise CrawlerException(f"ArXiv crawling failed: {str(e)}") from e
        finally:
            await self._close_session()
    
    async def crawl(self, target: str) -> Dict[str, Any]:
        """Crawl ArXiv target.
        
        Args:
            target: ArXiv URL, ID, or search query
            
        Returns:
            Crawled data dictionary
            
        Raises:
            CrawlerException: If crawling fails
        """
        return await self.crawl_async(target)