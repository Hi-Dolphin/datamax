"""Crawler Factory

Provides factory pattern implementation for creating crawler instances
based on target URLs or crawler types.
"""

import re
from typing import Dict, Type, Optional, Any
from urllib.parse import urlparse
from .base_crawler import BaseCrawler
from .config_manager import get_config
from .storage_adapter import create_storage_adapter
from .exceptions import CrawlerException, ConfigurationException


class CrawlerFactory:
    """Factory class for creating crawler instances.
    
    Manages crawler registration and provides methods to create
    appropriate crawler instances based on targets or types.
    """
    
    def __init__(self):
        """Initialize the crawler factory."""
        self._crawlers: Dict[str, Type[BaseCrawler]] = {}
        self._url_patterns: Dict[str, str] = {}
        self._config = get_config()
        self._storage_adapter = None
    
    def register_crawler(self, name: str, crawler_class: Type[BaseCrawler], url_patterns: Optional[list] = None):
        """Register a crawler class with the factory.
        
        Args:
            name: Unique name for the crawler
            crawler_class: Crawler class to register
            url_patterns: List of regex patterns that this crawler can handle
        """
        if not issubclass(crawler_class, BaseCrawler):
            raise CrawlerException(f"Crawler class must inherit from BaseCrawler: {crawler_class}")
        
        self._crawlers[name] = crawler_class
        
        if url_patterns:
            for pattern in url_patterns:
                self._url_patterns[pattern] = name
    
    def get_crawler_for_url(self, url: str) -> BaseCrawler:
        """Get appropriate crawler instance for a URL.
        
        Args:
            url: Target URL to crawl
            
        Returns:
            Configured crawler instance
            
        Raises:
            CrawlerException: If no suitable crawler is found
        """
        # Try to match URL patterns
        for pattern, crawler_name in self._url_patterns.items():
            if re.search(pattern, url, re.IGNORECASE):
                return self.create_crawler(crawler_name)
        
        # Fallback to domain-based detection
        parsed_url = urlparse(url)
        domain = parsed_url.netloc.lower()
        
        # ArXiv detection
        if 'arxiv.org' in domain:
            return self.create_crawler('arxiv')
        
        # Default to web crawler for other URLs
        if parsed_url.scheme in ['http', 'https']:
            return self.create_crawler('web')
        
        raise CrawlerException(f"No suitable crawler found for URL: {url}")
    
    def create_crawler(self, crawler_type: str) -> BaseCrawler:
        """Create a crawler instance by type.
        
        Args:
            crawler_type: Type of crawler to create
            
        Returns:
            Configured crawler instance
            
        Raises:
            CrawlerException: If crawler type is not registered
        """
        if crawler_type not in self._crawlers:
            available_types = list(self._crawlers.keys())
            raise CrawlerException(
                f"Unknown crawler type: {crawler_type}. "
                f"Available types: {available_types}"
            )
        
        crawler_class = self._crawlers[crawler_type]
        crawler_config = self._config.get_crawler_config(crawler_type)
        
        # Create crawler instance
        crawler = crawler_class(crawler_config)
        
        # Set up storage adapter
        if self._storage_adapter is None:
            storage_config = self._config.get_storage_config()
            self._storage_adapter = create_storage_adapter(storage_config)
        
        crawler.set_storage_adapter(self._storage_adapter)
        
        return crawler
    
    def list_crawlers(self) -> list:
        """List all registered crawler types.
        
        Returns:
            List of registered crawler type names
        """
        return list(self._crawlers.keys())
    
    def get_supported_patterns(self) -> Dict[str, str]:
        """Get all supported URL patterns.
        
        Returns:
            Dictionary mapping patterns to crawler names
        """
        return self._url_patterns.copy()
    
    def set_storage_adapter(self, adapter):
        """Set custom storage adapter for all created crawlers.
        
        Args:
            adapter: Storage adapter instance
        """
        self._storage_adapter = adapter


# Global factory instance
_factory_instance: Optional[CrawlerFactory] = None


def get_factory() -> CrawlerFactory:
    """Get the global crawler factory instance.
    
    Returns:
        Global CrawlerFactory instance
    """
    global _factory_instance
    if _factory_instance is None:
        _factory_instance = CrawlerFactory()
        _register_default_crawlers(_factory_instance)
    return _factory_instance


def set_factory(factory: CrawlerFactory):
    """Set the global crawler factory instance.
    
    Args:
        factory: CrawlerFactory instance to set as global
    """
    global _factory_instance
    _factory_instance = factory


def _register_default_crawlers(factory: CrawlerFactory):
    """Register default crawlers with the factory.
    
    Args:
        factory: Factory instance to register crawlers with
    """
    # Import crawler classes here to avoid circular imports
    try:
        from .arxiv_crawler import ArxivCrawler
        factory.register_crawler(
            'arxiv', 
            ArxivCrawler,
            [
                r'arxiv\.org',
                r'export\.arxiv\.org'
            ]
        )
    except ImportError:
        # ArxivCrawler not yet implemented
        pass
    
    try:
        from .web_crawler import WebCrawler
        factory.register_crawler(
            'web',
            WebCrawler,
            [
                r'https?://.*'
            ]
        )
    except ImportError:
        # WebCrawler not yet implemented
        pass


def create_crawler_for_url(url: str) -> BaseCrawler:
    """Convenience function to create crawler for a URL.
    
    Args:
        url: Target URL
        
    Returns:
        Configured crawler instance
    """
    factory = get_factory()
    return factory.get_crawler_for_url(url)


def create_crawler(crawler_type: str) -> BaseCrawler:
    """Convenience function to create crawler by type.
    
    Args:
        crawler_type: Type of crawler to create
        
    Returns:
        Configured crawler instance
    """
    factory = get_factory()
    return factory.create_crawler(crawler_type)