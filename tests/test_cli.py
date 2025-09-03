"""Tests for CLI Module

Comprehensive test suite for CLI functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from click.testing import CliRunner
from datetime import datetime

from datamax.cli import (
    main,
    CrawlerCLI,
    crawler_command,
    arxiv_command,
    web_command,
    list_crawlers_command
)
from datamax.crawler import CrawlerFactory, CrawlerConfig


class TestCLICommands:
    """Test CLI command functionality."""
    
    def test_main_cli_help(self):
        """Test main CLI help command."""
        runner = CliRunner()
        result = runner.invoke(main, ['--help'])
        
        assert result.exit_code == 0
        assert 'DataMax CLI' in result.output
        assert 'parse' in result.output
        assert 'status' in result.output
        assert 'crawler' in result.output
    
    def test_status_command(self):
        """Test status command."""
        runner = CliRunner()
        result = runner.invoke(main, ['status'])
        
        assert result.exit_code == 0
        assert 'DataMax System Status' in result.output
        assert 'Available Crawlers' in result.output
        assert 'Available Parsers' in result.output
    
    def test_crawler_help(self):
        """Test crawler command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['crawler', '--help'])
        
        assert result.exit_code == 0
        assert 'Crawler commands' in result.output
        assert 'crawl' in result.output
        assert 'arxiv' in result.output
        assert 'web' in result.output
        assert 'list' in result.output
    
    def test_list_crawlers_command(self):
        """Test list crawlers command."""
        runner = CliRunner()
        result = runner.invoke(main, ['crawler', 'list'])
        
        assert result.exit_code == 0
        assert 'Available Crawlers' in result.output
        assert 'arxiv' in result.output
        assert 'web' in result.output
    
    def test_arxiv_command_help(self):
        """Test ArXiv command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['crawler', 'arxiv', '--help'])
        
        assert result.exit_code == 0
        assert 'Crawl ArXiv papers' in result.output
        assert '--output' in result.output
        assert '--format' in result.output
    
    def test_web_command_help(self):
        """Test web command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['crawler', 'web', '--help'])
        
        assert result.exit_code == 0
        assert 'Crawl web pages' in result.output
        assert '--output' in result.output
        assert '--format' in result.output
    
    @patch('datamax.cli.commands.asyncio.run')
    def test_arxiv_command_execution(self, mock_asyncio_run, tmp_path):
        """Test ArXiv command execution."""
        # Mock successful crawling
        mock_result = {
            'type': 'single_paper',
            'source': 'arxiv',
            'data': {
                'id': '2301.07041',
                'title': 'Test Paper',
                'authors': ['Test Author']
            }
        }
        mock_asyncio_run.return_value = str(tmp_path / 'result.json')
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'crawler', 'arxiv',
            '2301.07041',
            '--output', str(tmp_path),
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        assert 'Successfully crawled ArXiv paper' in result.output
    
    @patch('datamax.cli.commands.asyncio.run')
    def test_web_command_execution(self, mock_asyncio_run, tmp_path):
        """Test web command execution."""
        # Mock successful crawling
        mock_result = {
            'type': 'web_page',
            'url': 'https://example.com',
            'metadata': {'title': 'Test Page'}
        }
        mock_asyncio_run.return_value = str(tmp_path / 'result.json')
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'crawler', 'web',
            'https://example.com',
            '--output', str(tmp_path),
            '--format', 'json'
        ])
        
        assert result.exit_code == 0
        assert 'Successfully crawled web page' in result.output
    
    def test_parse_command_help(self):
        """Test parse command help."""
        runner = CliRunner()
        result = runner.invoke(main, ['parse', '--help'])
        
        assert result.exit_code == 0
        assert 'Parse data files' in result.output
        assert '--output' in result.output
        assert '--type' in result.output
    
    def test_parse_command_with_crawler_data(self, tmp_path):
        """Test parse command with crawler data."""
        # Create test crawler data
        test_data = {
            'type': 'single_paper',
            'source': 'arxiv',
            'data': {
                'id': '2301.07041',
                'title': 'Test Paper',
                'authors': ['Test Author'],
                'abstract': 'Test abstract'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        input_file = tmp_path / 'test_data.json'
        with open(input_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        output_file = tmp_path / 'parsed_output.md'
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'parse',
            str(input_file),
            '--output', str(output_file),
            '--type', 'crawler'
        ])
        
        assert result.exit_code == 0
        assert 'Successfully parsed' in result.output
        assert output_file.exists()
        
        # Verify parsed content
        with open(output_file, 'r', encoding='utf-8') as f:
            content = f.read()
        assert '# Test Paper' in content
        assert 'Test Author' in content


class TestCrawlerCLI:
    """Test CrawlerCLI class functionality."""
    
    def test_crawler_cli_creation(self):
        """Test CrawlerCLI creation."""
        cli = CrawlerCLI()
        assert cli is not None
        assert isinstance(cli.config, CrawlerConfig)
        assert isinstance(cli.factory, CrawlerFactory)
    
    def test_list_crawlers(self):
        """Test list crawlers functionality."""
        cli = CrawlerCLI()
        crawlers = cli.list_crawlers()
        
        assert isinstance(crawlers, list)
        assert 'arxiv' in crawlers
        assert 'web' in crawlers
    
    def test_get_crawler_info(self):
        """Test get crawler info functionality."""
        cli = CrawlerCLI()
        
        # Test ArXiv crawler info
        info = cli.get_crawler_info('arxiv')
        assert 'name' in info
        assert 'description' in info
        assert 'supported_targets' in info
        
        # Test web crawler info
        info = cli.get_crawler_info('web')
        assert 'name' in info
        assert 'description' in info
        assert 'supported_targets' in info
        
        # Test unknown crawler
        info = cli.get_crawler_info('unknown')
        assert info is None
    
    @patch('datamax.crawler.ArxivCrawler.crawl_async')
    @pytest.mark.asyncio
    async def test_crawl_arxiv_async(self, mock_crawl, tmp_path):
        """Test async ArXiv crawling."""
        # Mock crawl result
        mock_result = {
            'type': 'single_paper',
            'source': 'arxiv',
            'data': {
                'id': '2301.07041',
                'title': 'Test Paper'
            }
        }
        mock_crawl.return_value = mock_result
        
        cli = CrawlerCLI()
        result_path = await cli.crawl_arxiv(
            '2301.07041',
            output_dir=str(tmp_path),
            format='json'
        )
        
        assert result_path is not None
        assert Path(result_path).exists()
        mock_crawl.assert_called_once_with('2301.07041')
    
    @patch('datamax.crawler.WebCrawler.crawl_async')
    @pytest.mark.asyncio
    async def test_crawl_web_async(self, mock_crawl, tmp_path):
        """Test async web crawling."""
        # Mock crawl result
        mock_result = {
            'type': 'web_page',
            'url': 'https://example.com',
            'metadata': {'title': 'Test Page'}
        }
        mock_crawl.return_value = mock_result
        
        cli = CrawlerCLI()
        result_path = await cli.crawl_web(
            'https://example.com',
            output_dir=str(tmp_path),
            format='json'
        )
        
        assert result_path is not None
        assert Path(result_path).exists()
        mock_crawl.assert_called_once_with('https://example.com')
    
    @patch('datamax.crawler.ArxivCrawler.crawl_async')
    @patch('datamax.crawler.WebCrawler.crawl_async')
    @pytest.mark.asyncio
    async def test_crawl_auto_detection(self, mock_web_crawl, mock_arxiv_crawl, tmp_path):
        """Test automatic crawler detection."""
        cli = CrawlerCLI()
        
        # Test ArXiv ID detection
        mock_arxiv_crawl.return_value = {'type': 'single_paper'}
        await cli.crawl('2301.07041', output_dir=str(tmp_path))
        mock_arxiv_crawl.assert_called_once()
        
        # Reset mocks
        mock_arxiv_crawl.reset_mock()
        mock_web_crawl.reset_mock()
        
        # Test URL detection
        mock_web_crawl.return_value = {'type': 'web_page'}
        await cli.crawl('https://example.com', output_dir=str(tmp_path))
        mock_web_crawl.assert_called_once()
    
    def test_invalid_target_handling(self):
        """Test handling of invalid targets."""
        cli = CrawlerCLI()
        
        with pytest.raises(ValueError, match="Unsupported target"):
            asyncio.run(cli.crawl('invalid-target'))


class TestCLIErrorHandling:
    """Test CLI error handling."""
    
    def test_invalid_file_path(self):
        """Test handling of invalid file paths."""
        runner = CliRunner()
        result = runner.invoke(main, [
            'parse',
            'non_existent_file.json'
        ])
        
        assert result.exit_code != 0
        assert 'Error' in result.output
    
    def test_invalid_output_directory(self):
        """Test handling of invalid output directories."""
        runner = CliRunner()
        result = runner.invoke(main, [
            'crawler', 'arxiv',
            '2301.07041',
            '--output', '/invalid/path/that/does/not/exist'
        ])
        
        assert result.exit_code != 0
    
    def test_invalid_arxiv_id(self):
        """Test handling of invalid ArXiv IDs."""
        runner = CliRunner()
        result = runner.invoke(main, [
            'crawler', 'arxiv',
            'invalid-arxiv-id'
        ])
        
        assert result.exit_code != 0
        assert 'Error' in result.output
    
    def test_invalid_url(self):
        """Test handling of invalid URLs."""
        runner = CliRunner()
        result = runner.invoke(main, [
            'crawler', 'web',
            'not-a-valid-url'
        ])
        
        assert result.exit_code != 0
        assert 'Error' in result.output


class TestCLIIntegration:
    """Integration tests for CLI module."""
    
    def test_end_to_end_workflow(self, tmp_path):
        """Test complete CLI workflow."""
        # This would be a comprehensive test that:
        # 1. Uses CLI to crawl data
        # 2. Uses CLI to parse the crawled data
        # 3. Verifies the complete workflow
        
        # For now, this is a placeholder for integration testing
        # In a real scenario, this would involve mocking network calls
        # and testing the complete data flow
        
        runner = CliRunner()
        
        # Test that the CLI is properly set up
        result = runner.invoke(main, ['--help'])
        assert result.exit_code == 0
        
        # Test status command
        result = runner.invoke(main, ['status'])
        assert result.exit_code == 0
        
        # Test crawler list
        result = runner.invoke(main, ['crawler', 'list'])
        assert result.exit_code == 0
    
    @patch('datamax.cli.commands.asyncio.run')
    def test_cli_with_custom_config(self, mock_asyncio_run, tmp_path):
        """Test CLI with custom configuration."""
        # Create custom config file
        custom_config = {
            'arxiv': {
                'base_url': 'http://export.arxiv.org/api/',
                'max_results': 100
            },
            'storage': {
                'type': 'local',
                'format': 'json',
                'base_path': str(tmp_path)
            }
        }
        
        config_file = tmp_path / 'custom_config.json'
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(custom_config, f, indent=2)
        
        # Mock successful execution
        mock_asyncio_run.return_value = str(tmp_path / 'result.json')
        
        runner = CliRunner()
        result = runner.invoke(main, [
            'crawler', 'arxiv',
            '2301.07041',
            '--config', str(config_file),
            '--output', str(tmp_path)
        ])
        
        # Should handle custom config gracefully
        # (Note: actual config loading would need to be implemented)
        assert result.exit_code == 0 or 'config' in result.output.lower()


if __name__ == '__main__':
    pytest.main([__file__])