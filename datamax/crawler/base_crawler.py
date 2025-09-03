"""Base Crawler Class

Provides the abstract base class for all crawler implementations in DataMax.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datamax.parser.base import BaseLife
from datamax.utils.lifecycle_types import LifeType
from .exceptions import CrawlerException


class BaseCrawler(BaseLife, ABC):
    """Abstract base class for all crawlers.
    
    This class provides the common interface and lifecycle management
    for all crawler implementations in DataMax.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the crawler.
        
        Args:
            config: Configuration dictionary for the crawler
        """
        super().__init__()
        self.config = config or {}
        self.storage_adapter = None
        self._setup_crawler()
    
    def _setup_crawler(self):
        """Setup crawler-specific configurations.
        
        This method can be overridden by subclasses to perform
        crawler-specific initialization.
        """
        pass
    
    @abstractmethod
    async def crawl(self, target: str) -> Dict[str, Any]:
        """Crawl the target resource.
        
        Args:
            target: The target URL, identifier, or query to crawl
            
        Returns:
            Dictionary containing the crawled data
            
        Raises:
            CrawlerException: If crawling fails
        """
        pass
    
    @abstractmethod
    def validate_target(self, target: str) -> bool:
        """Validate if the target is supported by this crawler.
        
        Args:
            target: The target to validate
            
        Returns:
            True if the target is valid for this crawler
        """
        pass
    
    def set_storage_adapter(self, adapter):
        """Set the storage adapter for this crawler.
        
        Args:
            adapter: Storage adapter instance
        """
        self.storage_adapter = adapter
    
    def _set_crawling_status(self):
        """Set the crawler status to crawling."""
        self.set_status(LifeType.DATA_CRAWLING)
    
    def _set_crawled_status(self):
        """Set the crawler status to crawled."""
        self.set_status(LifeType.DATA_CRAWLED)
    
    def _set_crawl_failed_status(self):
        """Set the crawler status to crawl failed."""
        self.set_status(LifeType.DATA_CRAWL_FAILED)
    
    async def safe_crawl(self, target: str) -> Dict[str, Any]:
        """Safely crawl with proper error handling and status management.
        
        Args:
            target: The target to crawl
            
        Returns:
            Dictionary containing the crawled data
            
        Raises:
            CrawlerException: If crawling fails
        """
        if not self.validate_target(target):
            raise CrawlerException(f"Invalid target for this crawler: {target}")
        
        self._set_crawling_status()
        
        try:
            result = await self.crawl(target)
            self._set_crawled_status()
            return result
        except Exception as e:
            self._set_crawl_failed_status()
            if isinstance(e, CrawlerException):
                raise
            raise CrawlerException(f"Crawling failed: {str(e)}") from e