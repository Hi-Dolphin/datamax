"""DataMax Crawler Module

Provides web crawling capabilities for the DataMax project.
Supports various crawler types including ArXiv papers and general web pages.
"""

from .base_crawler import BaseCrawler
from .exceptions import (
    CrawlerException,
    NetworkException,
    ParseException,
    RateLimitException,
    AuthenticationException,
    ConfigurationException
)
from .config_manager import CrawlerConfig, get_config, set_config
from .storage_adapter import StorageAdapter, LocalStorageAdapter, CloudStorageAdapter, create_storage_adapter
from .crawler_factory import CrawlerFactory, get_factory, set_factory, create_crawler_for_url, create_crawler
from .arxiv_crawler import ArxivCrawler
from .web_crawler import WebCrawler
from .logging_config import (
    CrawlerLogger,
    CrawlerMetrics,
    setup_crawler_logging,
    get_crawler_logger,
    get_crawler_metrics
)
from .crawl import crawl, crawl_arxiv, crawl_web

__all__ = [
    'BaseCrawler',
    'CrawlerException',
    'NetworkException',
    'ParseException',
    'RateLimitException',
    'AuthenticationException',
    'ConfigurationException',
    'CrawlerConfig',
    'StorageAdapter',
    'LocalStorageAdapter',
    'CloudStorageAdapter',
    'create_storage_adapter',
    'CrawlerFactory',
    'ArxivCrawler',
    'WebCrawler',
    'CrawlerLogger',
    'CrawlerMetrics',
    'setup_crawler_logging',
    'get_crawler_logger',
    'get_crawler_metrics',
    'set_factory',
    'create_crawler_for_url',
    'create_crawler',
    'crawl',
    'crawl_arxiv',
    'crawl_web'
]