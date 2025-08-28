"""Tests for Crawler Module

Comprehensive test suite for crawler functionality.
"""

import pytest
import asyncio
import json
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

from datamax.crawler import (
    BaseCrawler,
    ArxivCrawler,
    WebCrawler,
    CrawlerFactory,
    CrawlerConfig,
    LocalStorageAdapter,
    create_storage_adapter,
    CrawlerException,
    NetworkException,
    ParseException
)
from datamax.utils.lifecycle_types import LifeType


class TestCrawlerConfig:
    """Test crawler configuration management."""
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = CrawlerConfig()
        assert config.config is not None
        assert 'arxiv' in config.config
        assert 'web' in config.config
        assert 'storage' in config.config
    
    def test_get_crawler_config(self):
        """Test getting specific crawler configuration."""
        config = CrawlerConfig()
        arxiv_config = config.get_crawler_config('arxiv')
        assert isinstance(arxiv_config, dict)
        assert 'base_url' in arxiv_config
    
    def test_get_storage_config(self):
        """Test getting storage configuration."""
        config = CrawlerConfig()
        storage_config = config.get_storage_config()
        assert isinstance(storage_config, dict)
        assert 'type' in storage_config


class TestStorageAdapter:
    """Test storage adapter functionality."""
    
    def test_local_storage_creation(self):
        """Test local storage adapter creation."""
        config = {'type': 'local', 'format': 'json'}
        adapter = create_storage_adapter(config)
        assert isinstance(adapter, LocalStorageAdapter)
    
    def test_local_storage_store(self, tmp_path):
        """Test local storage store functionality."""
        config = {
            'type': 'local',
            'format': 'json',
            'base_path': str(tmp_path)
        }
        adapter = create_storage_adapter(config)
        
        test_data = {
            'type': 'test',
            'data': {'key': 'value'},
            'timestamp': datetime.now().isoformat()
        }
        
        result = adapter.store(test_data)
        assert result is not None
        assert Path(result).exists()
        
        # Verify stored content
        with open(result, 'r', encoding='utf-8') as f:
            stored_data = json.load(f)
        assert stored_data['type'] == 'test'
        assert stored_data['data']['key'] == 'value'


class TestCrawlerFactory:
    """Test crawler factory functionality."""
    
    def test_factory_creation(self):
        """Test factory creation."""
        factory = CrawlerFactory()
        assert factory is not None
    
    def test_arxiv_crawler_creation(self):
        """Test ArXiv crawler creation."""
        factory = CrawlerFactory()
        
        # Test ArXiv ID
        crawler = factory.create_crawler('2301.07041')
        assert isinstance(crawler, ArxivCrawler)
        
        # Test ArXiv URL
        crawler = factory.create_crawler('https://arxiv.org/abs/2301.07041')
        assert isinstance(crawler, ArxivCrawler)
    
    def test_web_crawler_creation(self):
        """Test web crawler creation."""
        factory = CrawlerFactory()
        
        # Test with search keyword
        crawler = factory.create_crawler('latest news')
        assert isinstance(crawler, WebCrawler)
        
        # Test with another search keyword
        crawler = factory.create_crawler('machine learning')
        assert isinstance(crawler, WebCrawler)
    
    def test_list_crawlers(self):
        """Test listing available crawlers."""
        factory = CrawlerFactory()
        crawlers = factory.list_crawlers()
        assert isinstance(crawlers, list)
        assert 'arxiv' in crawlers
        assert 'web' in crawlers


class TestArxivCrawler:
    """Test ArXiv crawler functionality."""
    
    def test_arxiv_crawler_creation(self):
        """Test ArXiv crawler creation."""
        crawler = ArxivCrawler()
        assert crawler is not None
        assert crawler.base_url == 'http://export.arxiv.org/api/'
    
    def test_arxiv_id_validation(self):
        """Test ArXiv ID validation."""
        crawler = ArxivCrawler()
        
        # Valid ArXiv IDs
        assert crawler.validate_target('2301.07041')
        assert crawler.validate_target('1234.5678v2')
        assert crawler.validate_target('https://arxiv.org/abs/2301.07041')
        
        # Invalid targets
        assert crawler.validate_target('invalid-id') == False
    
    def test_arxiv_id_extraction(self):
        """Test ArXiv ID extraction from URLs."""
        crawler = ArxivCrawler()
        
        # Test URL extraction
        arxiv_id = crawler._extract_id_from_url('https://arxiv.org/abs/2301.07041')
        assert arxiv_id == '2301.07041'
        
        # Test PDF URL extraction
        arxiv_id = crawler._extract_id_from_url('https://arxiv.org/pdf/2301.07041.pdf')
        assert arxiv_id == '2301.07041'
    
    @patch('aiohttp.ClientSession.get')
    @pytest.mark.asyncio
    async def test_arxiv_crawl_mock(self, mock_get):
        """Test ArXiv crawling with mocked response."""
        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text.return_value = '''
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/2301.07041v1</id>
                <title>Test Paper Title</title>
                <summary>Test paper summary</summary>
                <author><name>Test Author</name></author>
                <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI"/>
                <published>2023-01-17T18:59:59Z</published>
                <updated>2023-01-17T18:59:59Z</updated>
            </entry>
        </feed>
        '''
        
        mock_get.return_value.__aenter__.return_value = mock_response
        
        crawler = ArxivCrawler()
        result = await crawler.crawl_async('2301.07041')
        
        assert result['type'] == 'single_paper'
        assert result['source'] == 'arxiv'
        assert 'data' in result
        assert result['data']['title'] == 'Test Paper Title'


class TestWebCrawler:
    """Test web crawler functionality."""
    
    def test_web_crawler_creation(self):
        """Test web crawler creation."""
        crawler = WebCrawler()
        assert crawler is not None
    
    def test_url_validation(self):
        """Test search keyword validation."""
        crawler = WebCrawler()
        
        # Valid search keywords
        assert crawler.validate_target('latest news')
        assert crawler.validate_target('machine learning research')
        assert crawler.validate_target('AI developments')
        
        # Invalid keywords
        assert crawler.validate_target('') == False
        assert crawler.validate_target('   ') == False
    
    @patch('aiohttp.ClientSession.post')
    @pytest.mark.asyncio
    async def test_web_crawl_mock(self, mock_post):
        """Test web search with mocked response."""
        # Mock response for search API
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            'code': 200,
            'data': {
                'webPages': {
                    'value': [
                        {
                            'url': 'https://example.com/article1',
                            'title': 'Test Article 1',
                            'description': 'Test article description',
                            'snippet': 'Test article snippet'
                        },
                        {
                            'url': 'https://example.com/article2',
                            'title': 'Test Article 2',
                            'description': 'Another test article',
                            'snippet': 'Another test snippet'
                        }
                    ]
                }
            }
        }
        
        mock_post.return_value.__aenter__.return_value = mock_response
        
        crawler = WebCrawler({
            'search_api_key': 'test-key',
            'search_api_url': 'https://api.test.com/search'
        })
        result = await crawler.crawl_async('test search')
        
        assert result['type'] == 'web_search_results'
        assert result['query'] == 'test search'
        assert result['result_count'] == 2
        assert result['results'][0]['title'] == 'Test Article 1'
        assert result['results'][1]['title'] == 'Test Article 2'


class TestBaseCrawler:
    """Test base crawler functionality."""
    
    def test_base_crawler_lifecycle(self):
        """Test base crawler lifecycle management."""
        class TestCrawler(BaseCrawler):
            def crawl(self, target):
                return {'test': 'data'}
            
            def validate_target(self, target):
                return True
        
        crawler = TestCrawler()
        
        # Test initial status
        assert crawler.status == LifeType.DATA_INIT
        
        # Test status changes
        crawler.set_crawling_status()
        assert crawler.status == LifeType.DATA_CRAWLING
        
        crawler.set_crawled_status()
        assert crawler.status == LifeType.DATA_CRAWLED
        
        crawler.set_crawl_failed_status()
        assert crawler.status == LifeType.DATA_CRAWL_FAILED
    
    def test_storage_adapter_setting(self):
        """Test storage adapter setting."""
        class TestCrawler(BaseCrawler):
            def crawl(self, target):
                return {'test': 'data'}
            
            def validate_target(self, target):
                return True
        
        crawler = TestCrawler()
        storage_adapter = LocalStorageAdapter()
        
        crawler.set_storage_adapter(storage_adapter)
        assert crawler.storage_adapter == storage_adapter


class TestCrawlerExceptions:
    """Test crawler exception handling."""
    
    def test_crawler_exception(self):
        """Test basic crawler exception."""
        with pytest.raises(CrawlerException):
            raise CrawlerException("Test error")
    
    def test_network_exception(self):
        """Test network exception."""
        with pytest.raises(NetworkException):
            raise NetworkException("Network error")
    
    def test_parse_exception(self):
        """Test parse exception."""
        with pytest.raises(ParseException):
            raise ParseException("Parse error")


class TestIntegration:
    """Integration tests for crawler module."""
    
    def test_end_to_end_arxiv(self, tmp_path):
        """Test end-to-end ArXiv crawling workflow."""
        # Create factory and storage
        factory = CrawlerFactory()
        storage_config = {
            'type': 'local',
            'format': 'json',
            'base_path': str(tmp_path)
        }
        storage_adapter = create_storage_adapter(storage_config)
        
        # Create crawler
        crawler = factory.create_crawler('2301.07041')
        crawler.set_storage_adapter(storage_adapter)
        
        # Validate target
        assert crawler.validate_target('2301.07041')
        
        # Test would require actual network call
        # In real test, would mock the network response
    
    def test_end_to_end_web(self, tmp_path):
        """Test end-to-end web search workflow."""
        # Create factory and storage
        factory = CrawlerFactory()
        storage_config = {
            'type': 'local',
            'format': 'json',
            'base_path': str(tmp_path)
        }
        storage_adapter = create_storage_adapter(storage_config)
        
        # Create crawler with search keyword
        crawler = factory.create_crawler('latest news')
        crawler.set_storage_adapter(storage_adapter)
        
        # Validate target
        assert crawler.validate_target('latest news')
        
        # Test would require actual network call
        # In real test, would mock the network response


if __name__ == '__main__':
    pytest.main([__file__])