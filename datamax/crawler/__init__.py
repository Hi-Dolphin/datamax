"""DataMax Crawler Module

Provides web crawling capabilities for the DataMax project.
Supports various crawler types including ArXiv papers and general web pages.
"""

from .arxiv_crawler import ArxivCrawler
from .base_crawler import BaseCrawler
from .config_manager import CrawlerConfig, get_config, set_config
from .crawl import crawl, crawl_arxiv, crawl_web
from .crawler_factory import (
    CrawlerFactory,
    create_crawler,
    create_crawler_for_url,
    get_factory,
    set_factory,
)
from .exceptions import (
    AuthenticationException,
    ConfigurationException,
    CrawlerException,
    NetworkException,
    ParseException,
    RateLimitException,
)
from .logging_config import (
    CrawlerLogger,
    CrawlerMetrics,
    get_crawler_logger,
    get_crawler_metrics,
    setup_crawler_logging,
)
from .storage_adapter import (
    CloudStorageAdapter,
    LocalStorageAdapter,
    StorageAdapter,
    create_storage_adapter,
)
from .web_crawler import WebCrawler

__all__ = [
    "BaseCrawler",
    "CrawlerException",
    "NetworkException",
    "ParseException",
    "RateLimitException",
    "AuthenticationException",
    "ConfigurationException",
    "CrawlerConfig",
    "StorageAdapter",
    "LocalStorageAdapter",
    "CloudStorageAdapter",
    "create_storage_adapter",
    "CrawlerFactory",
    "ArxivCrawler",
    "WebCrawler",
    "CrawlerLogger",
    "CrawlerMetrics",
    "setup_crawler_logging",
    "get_crawler_logger",
    "get_crawler_metrics",
    "set_factory",
    "create_crawler_for_url",
    "create_crawler",
    "crawl",
    "crawl_arxiv",
    "crawl_web",
]
