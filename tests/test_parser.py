"""Tests for Parser Module

Comprehensive test suite for parser functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

from datamax.parser import CrawlerParser
from datamax.utils.lifecycle_types import LifeType


class TestCrawlerParser:
    """Test crawler parser functionality."""
    
    def test_parser_creation(self):
        """Test parser creation."""
        parser = CrawlerParser()
        assert parser is not None
        assert parser.status == LifeType.DATA_INIT
    
    def test_arxiv_data_parsing(self):
        """Test ArXiv data parsing."""
        # Sample ArXiv data
        arxiv_data = {
            'type': 'single_paper',
            'source': 'arxiv',
            'data': {
                'id': '2301.07041',
                'title': 'Test Paper Title',
                'authors': ['Author One', 'Author Two'],
                'abstract': 'This is a test abstract for the paper.',
                'categories': ['cs.AI', 'cs.LG'],
                'published': '2023-01-17T18:59:59Z',
                'updated': '2023-01-17T18:59:59Z',
                'pdf_url': 'https://arxiv.org/pdf/2301.07041.pdf',
                'abs_url': 'https://arxiv.org/abs/2301.07041'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        parser = CrawlerParser()
        result = parser._parse_arxiv_data(arxiv_data)
        
        assert '# Test Paper Title' in result
        assert '**Authors:** Author One, Author Two' in result
        assert '**ArXiv ID:** 2301.07041' in result
        assert '**Categories:** cs.AI, cs.LG' in result
        assert 'This is a test abstract for the paper.' in result
        assert '[PDF](https://arxiv.org/pdf/2301.07041.pdf)' in result
    
    def test_web_data_parsing(self):
        """Test web data parsing."""
        # Sample web data
        web_data = {
            'type': 'web_page',
            'url': 'https://example.com',
            'metadata': {
                'title': 'Example Page',
                'description': 'This is an example page',
                'keywords': 'example, test, web'
            },
            'text_content': 'This is the main content of the page.',
            'links': [
                {'url': 'https://example.com/page1', 'text': 'Page 1'},
                {'url': 'https://example.com/page2', 'text': 'Page 2'}
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        parser = CrawlerParser()
        result = parser._parse_web_data(web_data)
        
        assert '# Example Page' in result
        assert '**URL:** https://example.com' in result
        assert '**Description:** This is an example page' in result
        assert 'This is the main content of the page.' in result
        assert '[Page 1](https://example.com/page1)' in result
        assert '[Page 2](https://example.com/page2)' in result
    
    def test_search_results_parsing(self):
        """Test search results parsing."""
        # Sample search results data
        search_data = {
            'type': 'search_results',
            'source': 'arxiv',
            'query': 'machine learning',
            'total_results': 2,
            'results': [
                {
                    'id': '2301.07041',
                    'title': 'Paper One',
                    'authors': ['Author A'],
                    'abstract': 'Abstract for paper one.',
                    'published': '2023-01-17T18:59:59Z'
                },
                {
                    'id': '2301.07042',
                    'title': 'Paper Two',
                    'authors': ['Author B', 'Author C'],
                    'abstract': 'Abstract for paper two.',
                    'published': '2023-01-18T18:59:59Z'
                }
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        parser = CrawlerParser()
        result = parser._parse_search_results(search_data)
        
        assert '# Search Results: machine learning' in result
        assert '**Total Results:** 2' in result
        assert '## Paper One' in result
        assert '## Paper Two' in result
        assert 'Author A' in result
        assert 'Author B, Author C' in result
    
    def test_generic_data_parsing(self):
        """Test generic data parsing."""
        # Sample generic data
        generic_data = {
            'type': 'custom_data',
            'title': 'Custom Data Title',
            'content': 'This is custom content.',
            'metadata': {
                'source': 'custom_source',
                'category': 'test'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        parser = CrawlerParser()
        result = parser._parse_generic_data(generic_data)
        
        assert '# Custom Data Title' in result
        assert 'This is custom content.' in result
        assert '**Source:** custom_source' in result
        assert '**Category:** test' in result
    
    def test_file_loading_and_parsing(self, tmp_path):
        """Test loading data from file and parsing."""
        # Create test data file
        test_data = {
            'type': 'single_paper',
            'source': 'arxiv',
            'data': {
                'id': '2301.07041',
                'title': 'Test Paper',
                'authors': ['Test Author'],
                'abstract': 'Test abstract.',
                'categories': ['cs.AI'],
                'published': '2023-01-17T18:59:59Z'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        test_file = tmp_path / 'test_data.json'
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        parser = CrawlerParser()
        parser.load_data(str(test_file))
        
        assert parser.data is not None
        assert parser.data['type'] == 'single_paper'
        assert parser.status == LifeType.DATA_LOADED
        
        # Test parsing
        result = parser.parse()
        assert '# Test Paper' in result
        assert 'Test Author' in result
        assert parser.status == LifeType.DATA_PARSED
    
    def test_invalid_file_handling(self):
        """Test handling of invalid files."""
        parser = CrawlerParser()
        
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            parser.load_data('non_existent_file.json')
        
        # Test invalid JSON
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('invalid json content')
            f.flush()
            
            with pytest.raises(json.JSONDecodeError):
                parser.load_data(f.name)
    
    def test_lifecycle_management(self):
        """Test parser lifecycle management."""
        parser = CrawlerParser()
        
        # Test initial status
        assert parser.status == LifeType.DATA_INIT
        
        # Test status changes
        parser.set_loading_status()
        assert parser.status == LifeType.DATA_LOADING
        
        parser.set_loaded_status()
        assert parser.status == LifeType.DATA_LOADED
        
        parser.set_parsing_status()
        assert parser.status == LifeType.DATA_PARSING
        
        parser.set_parsed_status()
        assert parser.status == LifeType.DATA_PARSED
        
        parser.set_parse_failed_status()
        assert parser.status == LifeType.DATA_PARSE_FAILED
    
    def test_get_data_method(self, tmp_path):
        """Test get_data method."""
        # Create test data file
        test_data = {
            'type': 'web_page',
            'url': 'https://example.com',
            'metadata': {'title': 'Test Page'},
            'text_content': 'Test content',
            'timestamp': datetime.now().isoformat()
        }
        
        test_file = tmp_path / 'test_web_data.json'
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        parser = CrawlerParser()
        parser.load_data(str(test_file))
        
        # Test get_data
        retrieved_data = parser.get_data()
        assert retrieved_data == test_data
    
    def test_parse_without_data(self):
        """Test parsing without loaded data."""
        parser = CrawlerParser()
        
        with pytest.raises(ValueError, match="No data loaded"):
            parser.parse()
    
    def test_unsupported_data_type(self, tmp_path):
        """Test handling of unsupported data types."""
        # Create test data with unsupported type
        test_data = {
            'type': 'unsupported_type',
            'content': 'Some content',
            'timestamp': datetime.now().isoformat()
        }
        
        test_file = tmp_path / 'unsupported_data.json'
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        parser = CrawlerParser()
        parser.load_data(str(test_file))
        
        # Should fall back to generic parsing
        result = parser.parse()
        assert 'Some content' in result
    
    def test_empty_data_handling(self, tmp_path):
        """Test handling of empty or minimal data."""
        # Create minimal test data
        test_data = {
            'type': 'single_paper',
            'source': 'arxiv',
            'data': {},
            'timestamp': datetime.now().isoformat()
        }
        
        test_file = tmp_path / 'minimal_data.json'
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        parser = CrawlerParser()
        parser.load_data(str(test_file))
        
        # Should handle gracefully
        result = parser.parse()
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_markdown_formatting(self):
        """Test markdown formatting in parsed output."""
        # Test data with special characters
        test_data = {
            'type': 'single_paper',
            'source': 'arxiv',
            'data': {
                'title': 'Paper with *special* characters & symbols',
                'authors': ['Author with & symbol'],
                'abstract': 'Abstract with **bold** and *italic* text.',
                'id': '2301.07041'
            },
            'timestamp': datetime.now().isoformat()
        }
        
        parser = CrawlerParser()
        parser.data = test_data
        result = parser._parse_arxiv_data(test_data)
        
        # Check that markdown is properly formatted
        assert '# Paper with *special* characters & symbols' in result
        assert 'Author with & symbol' in result
        assert 'Abstract with **bold** and *italic* text.' in result


class TestParserIntegration:
    """Integration tests for parser module."""
    
    def test_end_to_end_parsing_workflow(self, tmp_path):
        """Test complete parsing workflow."""
        # Create comprehensive test data
        test_data = {
            'type': 'search_results',
            'source': 'arxiv',
            'query': 'artificial intelligence',
            'total_results': 1,
            'results': [
                {
                    'id': '2301.07041',
                    'title': 'Advanced AI Techniques',
                    'authors': ['Dr. AI Researcher', 'Prof. ML Expert'],
                    'abstract': 'This paper presents novel approaches to artificial intelligence.',
                    'categories': ['cs.AI', 'cs.LG', 'stat.ML'],
                    'published': '2023-01-17T18:59:59Z',
                    'updated': '2023-01-17T18:59:59Z',
                    'pdf_url': 'https://arxiv.org/pdf/2301.07041.pdf',
                    'abs_url': 'https://arxiv.org/abs/2301.07041'
                }
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save to file
        test_file = tmp_path / 'search_results.json'
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, indent=2)
        
        # Parse using CrawlerParser
        parser = CrawlerParser()
        
        # Test complete workflow
        parser.load_data(str(test_file))
        assert parser.status == LifeType.DATA_LOADED
        
        result = parser.parse()
        assert parser.status == LifeType.DATA_PARSED
        
        # Verify output quality
        assert '# Search Results: artificial intelligence' in result
        assert '## Advanced AI Techniques' in result
        assert 'Dr. AI Researcher, Prof. ML Expert' in result
        assert 'novel approaches to artificial intelligence' in result
        assert '[PDF](https://arxiv.org/pdf/2301.07041.pdf)' in result
        
        # Test data retrieval
        retrieved_data = parser.get_data()
        assert retrieved_data == test_data


if __name__ == '__main__':
    pytest.main([__file__])