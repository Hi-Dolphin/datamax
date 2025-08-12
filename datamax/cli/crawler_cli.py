"""Crawler CLI Class

Provides object-oriented interface for crawler CLI operations.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger

from datamax.crawler import (
    CrawlerFactory,
    ArxivCrawler,
    WebCrawler,
    CrawlerConfig,
    create_storage_adapter
)
from datamax.crawler.exceptions import CrawlerException


class CrawlerCLI:
    """Object-oriented interface for crawler operations.
    
    Provides programmatic access to crawler functionality
    that can be used by other applications or scripts.
    """
    
    def __init__(self, config_path: Optional[str] = None, verbose: bool = False):
        """Initialize crawler CLI.
        
        Args:
            config_path: Path to configuration file
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.config = CrawlerConfig(config_path) if config_path else CrawlerConfig()
        self.factory = CrawlerFactory(config=self.config)
        
        # Configure logging
        if verbose:
            logger.remove()
            logger.add(
                lambda msg: print(msg, end=''),
                level="DEBUG",
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>\n"
            )
    
    def crawl(self, 
              target: str, 
              output_path: Optional[str] = None,
              format: str = 'json',
              storage_type: str = 'local',
              async_mode: bool = False,
              **kwargs) -> Dict[str, Any]:
        """Crawl a target using automatic crawler detection.
        
        Args:
            target: URL, ArXiv ID, or search query
            output_path: Output file path (optional)
            format: Output format ('json' or 'yaml')
            storage_type: Storage type ('local' or 'cloud')
            async_mode: Use async crawling
            **kwargs: Additional crawler parameters
            
        Returns:
            Crawled data dictionary
            
        Raises:
            CrawlerException: If crawling fails
        """
        try:
            # Create storage adapter
            storage_config = {
                'type': storage_type,
                'format': format
            }
            if output_path:
                storage_config['base_path'] = str(Path(output_path).parent)
            
            storage_adapter = create_storage_adapter(storage_config)
            
            if self.verbose:
                logger.info(f"Crawling target: {target}")
            
            if async_mode:
                # Async crawling
                result = asyncio.run(self._async_crawl(target, storage_adapter, **kwargs))
            else:
                # Sync crawling
                crawler = self.factory.create_crawler(target)
                crawler.set_storage_adapter(storage_adapter)
                result = crawler.crawl(target, **kwargs)
            
            # Save result if output path specified
            if output_path:
                self._save_result(result, output_path, format)
            
            if self.verbose:
                logger.info("Crawling completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"Crawling failed: {str(e)}")
            raise CrawlerException(f"Crawling failed: {str(e)}") from e
    
    def crawl_arxiv(self,
                    arxiv_input: str,
                    output_path: Optional[str] = None,
                    format: str = 'json',
                    max_results: int = 10,
                    sort_by: str = 'relevance',
                    category: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
        """Crawl ArXiv papers.
        
        Args:
            arxiv_input: ArXiv ID, URL, or search query
            output_path: Output file path (optional)
            format: Output format ('json' or 'yaml')
            max_results: Maximum number of results for search
            sort_by: Sort order ('relevance', 'lastUpdatedDate', 'submittedDate')
            category: ArXiv category filter
            **kwargs: Additional parameters
            
        Returns:
            Crawled ArXiv data
            
        Raises:
            CrawlerException: If crawling fails
        """
        try:
            # Create ArXiv crawler
            crawler = ArxivCrawler(config=self.config.get_crawler_config('arxiv'))
            
            # Set up storage
            storage_config = {'type': 'local', 'format': format}
            storage_adapter = create_storage_adapter(storage_config)
            crawler.set_storage_adapter(storage_adapter)
            
            if self.verbose:
                logger.info(f"Crawling ArXiv: {arxiv_input}")
            
            # Prepare parameters
            crawl_params = {
                'max_results': max_results,
                'sort_by': sort_by,
                **kwargs
            }
            if category:
                crawl_params['category'] = category
            
            # Perform crawling
            result = crawler.crawl(arxiv_input, **crawl_params)
            
            # Save result if output path specified
            if output_path:
                self._save_result(result, output_path, format)
            
            if self.verbose:
                if isinstance(result.get('data'), list):
                    logger.info(f"Found {len(result['data'])} papers")
                else:
                    logger.info("Retrieved 1 paper")
            
            return result
            
        except Exception as e:
            logger.error(f"ArXiv crawling failed: {str(e)}")
            raise CrawlerException(f"ArXiv crawling failed: {str(e)}") from e
    
    def crawl_web(self,
                  url: str,
                  output_path: Optional[str] = None,
                  format: str = 'json',
                  extract_links: bool = False,
                  max_links: int = 100,
                  follow_redirects: bool = True,
                  timeout: int = 30,
                  **kwargs) -> Dict[str, Any]:
        """Crawl a web page.
        
        Args:
            url: Web page URL
            output_path: Output file path (optional)
            format: Output format ('json' or 'yaml')
            extract_links: Extract links from page
            max_links: Maximum number of links to extract
            follow_redirects: Follow HTTP redirects
            timeout: Request timeout in seconds
            **kwargs: Additional parameters
            
        Returns:
            Crawled web page data
            
        Raises:
            CrawlerException: If crawling fails
        """
        try:
            # Create web crawler
            web_config = self.config.get_crawler_config('web')
            web_config.update({
                'timeout': timeout,
                'follow_redirects': follow_redirects,
                'extract_links': extract_links,
                'max_links': max_links,
                **kwargs
            })
            
            crawler = WebCrawler(config=web_config)
            
            # Set up storage
            storage_config = {'type': 'local', 'format': format}
            storage_adapter = create_storage_adapter(storage_config)
            crawler.set_storage_adapter(storage_adapter)
            
            if self.verbose:
                logger.info(f"Crawling web page: {url}")
            
            # Perform crawling
            result = crawler.crawl(url)
            
            # Save result if output path specified
            if output_path:
                self._save_result(result, output_path, format)
            
            if self.verbose:
                text_length = len(result.get('text_content', ''))
                links_count = len(result.get('links', []))
                logger.info(f"Extracted {text_length} characters and {links_count} links")
            
            return result
            
        except Exception as e:
            logger.error(f"Web crawling failed: {str(e)}")
            raise CrawlerException(f"Web crawling failed: {str(e)}") from e
    
    def list_crawlers(self) -> List[str]:
        """List available crawlers.
        
        Returns:
            List of crawler names
        """
        return self.factory.list_crawlers()
    
    def get_crawler_info(self, crawler_name: str) -> Dict[str, Any]:
        """Get information about a specific crawler.
        
        Args:
            crawler_name: Name of the crawler
            
        Returns:
            Crawler information dictionary
        """
        crawlers_info = {
            'arxiv': {
                'name': 'ArXiv Crawler',
                'purpose': 'Academic paper crawling from ArXiv',
                'supports': ['ArXiv IDs', 'URLs', 'search queries'],
                'features': ['Metadata extraction', 'PDF links', 'categories'],
                'config_keys': ['rate_limit', 'max_results', 'timeout']
            },
            'web': {
                'name': 'Web Crawler',
                'purpose': 'General web page content extraction',
                'supports': ['HTTP/HTTPS URLs'],
                'features': ['Text extraction', 'metadata', 'link discovery'],
                'config_keys': ['timeout', 'follow_redirects', 'extract_links', 'max_links']
            }
        }
        
        return crawlers_info.get(crawler_name, {
            'name': f'{crawler_name.title()} Crawler',
            'purpose': 'Custom crawler',
            'supports': ['Custom targets'],
            'features': ['Custom features'],
            'config_keys': []
        })
    
    def validate_target(self, target: str) -> Dict[str, Any]:
        """Validate a crawling target and suggest appropriate crawler.
        
        Args:
            target: Target to validate
            
        Returns:
            Validation result with suggested crawler
        """
        try:
            crawler = self.factory.create_crawler(target)
            crawler_type = type(crawler).__name__.lower().replace('crawler', '')
            
            return {
                'valid': True,
                'suggested_crawler': crawler_type,
                'crawler_class': type(crawler).__name__,
                'target_type': self._detect_target_type(target)
            }
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'suggested_crawler': None,
                'target_type': 'unknown'
            }
    
    async def _async_crawl(self, target: str, storage_adapter, **kwargs) -> Dict[str, Any]:
        """Perform async crawling.
        
        Args:
            target: Target to crawl
            storage_adapter: Storage adapter instance
            **kwargs: Additional parameters
            
        Returns:
            Crawled data
        """
        crawler = self.factory.create_crawler(target)
        crawler.set_storage_adapter(storage_adapter)
        
        if hasattr(crawler, 'crawl_async'):
            return await crawler.crawl_async(target, **kwargs)
        else:
            # Fallback to sync crawling
            return crawler.crawl(target, **kwargs)
    
    def _save_result(self, result: Dict[str, Any], output_path: str, format: str):
        """Save crawling result to file.
        
        Args:
            result: Result data to save
            output_path: Output file path
            format: Output format ('json' or 'yaml')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'yaml':
            import yaml
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
    
    def _detect_target_type(self, target: str) -> str:
        """Detect the type of crawling target.
        
        Args:
            target: Target string
            
        Returns:
            Target type string
        """
        if target.startswith(('http://', 'https://')):
            if 'arxiv.org' in target:
                return 'arxiv_url'
            else:
                return 'web_url'
        elif target.replace('.', '').replace('v', '').isdigit():
            return 'arxiv_id'
        else:
            return 'search_query'