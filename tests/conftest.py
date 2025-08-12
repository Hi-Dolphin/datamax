"""Pytest configuration and fixtures for DataMax tests.

This module provides common fixtures and configuration for all tests.
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, AsyncMock


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def sample_arxiv_data():
    """Sample ArXiv data for testing."""
    return {
        'type': 'single_paper',
        'source': 'arxiv',
        'data': {
            'id': '2301.07041',
            'title': 'Sample ArXiv Paper Title',
            'authors': ['Dr. Sample Author', 'Prof. Test Researcher'],
            'abstract': 'This is a sample abstract for testing purposes. It contains multiple sentences to test text processing.',
            'categories': ['cs.AI', 'cs.LG', 'stat.ML'],
            'published': '2023-01-17T18:59:59Z',
            'updated': '2023-01-17T18:59:59Z',
            'pdf_url': 'https://arxiv.org/pdf/2301.07041.pdf',
            'abs_url': 'https://arxiv.org/abs/2301.07041'
        },
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_web_data():
    """Sample web data for testing."""
    return {
        'type': 'web_page',
        'url': 'https://example.com/test-page',
        'metadata': {
            'title': 'Sample Web Page Title',
            'description': 'This is a sample web page description for testing.',
            'keywords': 'sample, test, web, page',
            'author': 'Test Author',
            'language': 'en'
        },
        'text_content': 'This is the main content of the sample web page. It contains multiple paragraphs and various text elements for testing purposes.',
        'links': [
            {'url': 'https://example.com/page1', 'text': 'Related Page 1'},
            {'url': 'https://example.com/page2', 'text': 'Related Page 2'},
            {'url': '/relative-link', 'text': 'Relative Link'}
        ],
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_search_results():
    """Sample search results data for testing."""
    return {
        'type': 'search_results',
        'source': 'arxiv',
        'query': 'machine learning artificial intelligence',
        'total_results': 3,
        'results': [
            {
                'id': '2301.07041',
                'title': 'First ML Paper',
                'authors': ['Author A', 'Author B'],
                'abstract': 'Abstract for the first machine learning paper.',
                'categories': ['cs.LG'],
                'published': '2023-01-17T18:59:59Z',
                'pdf_url': 'https://arxiv.org/pdf/2301.07041.pdf'
            },
            {
                'id': '2301.07042',
                'title': 'Second AI Paper',
                'authors': ['Author C'],
                'abstract': 'Abstract for the second artificial intelligence paper.',
                'categories': ['cs.AI'],
                'published': '2023-01-18T18:59:59Z',
                'pdf_url': 'https://arxiv.org/pdf/2301.07042.pdf'
            },
            {
                'id': '2301.07043',
                'title': 'Third Combined Paper',
                'authors': ['Author D', 'Author E', 'Author F'],
                'abstract': 'Abstract for the third paper combining ML and AI.',
                'categories': ['cs.AI', 'cs.LG'],
                'published': '2023-01-19T18:59:59Z',
                'pdf_url': 'https://arxiv.org/pdf/2301.07043.pdf'
            }
        ],
        'timestamp': datetime.now().isoformat()
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        'arxiv': {
            'base_url': 'http://export.arxiv.org/api/',
            'max_results': 10,
            'timeout': 30,
            'retry_attempts': 3,
            'retry_delay': 1.0
        },
        'web': {
            'timeout': 30,
            'retry_attempts': 3,
            'retry_delay': 1.0,
            'user_agent': 'DataMax-Crawler/1.0',
            'max_content_length': 10485760  # 10MB
        },
        'storage': {
            'type': 'local',
            'format': 'json',
            'base_path': './data',
            'create_subdirs': True
        },
        'logging': {
            'level': 'INFO',
            'format': 'standard',
            'file_output': False
        }
    }


@pytest.fixture
def mock_aiohttp_response():
    """Mock aiohttp response for testing."""
    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {'Content-Type': 'text/html'}
    return mock_response


@pytest.fixture
def mock_arxiv_xml_response():
    """Mock ArXiv XML response for testing."""
    return '''
    <?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <title>ArXiv Query: search_query=all:test</title>
        <id>http://arxiv.org/api/query?search_query=all:test</id>
        <updated>2023-01-17T00:00:00-05:00</updated>
        <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">1</opensearch:totalResults>
        <opensearch:startIndex xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">0</opensearch:startIndex>
        <opensearch:itemsPerPage xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">1</opensearch:itemsPerPage>
        <entry>
            <id>http://arxiv.org/abs/2301.07041v1</id>
            <updated>2023-01-17T18:59:59Z</updated>
            <published>2023-01-17T18:59:59Z</published>
            <title>Test Paper Title</title>
            <summary>This is a test paper abstract for testing purposes.</summary>
            <author>
                <name>Test Author One</name>
            </author>
            <author>
                <name>Test Author Two</name>
            </author>
            <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
            <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
            <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
            <link href="http://arxiv.org/abs/2301.07041v1" rel="alternate" type="text/html"/>
            <link title="pdf" href="http://arxiv.org/pdf/2301.07041v1.pdf" rel="related" type="application/pdf"/>
        </entry>
    </feed>
    '''


@pytest.fixture
def mock_html_response():
    """Mock HTML response for testing."""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Test Web Page</title>
        <meta name="description" content="This is a test web page for testing purposes.">
        <meta name="keywords" content="test, web, page, sample">
        <meta name="author" content="Test Author">
    </head>
    <body>
        <header>
            <h1>Test Web Page</h1>
            <nav>
                <a href="/home">Home</a>
                <a href="/about">About</a>
                <a href="/contact">Contact</a>
            </nav>
        </header>
        <main>
            <article>
                <h2>Main Article</h2>
                <p>This is the main content of the test web page. It contains multiple paragraphs for testing text extraction.</p>
                <p>This is another paragraph with some <strong>bold text</strong> and <em>italic text</em>.</p>
                <ul>
                    <li>List item 1</li>
                    <li>List item 2</li>
                    <li>List item 3</li>
                </ul>
            </article>
            <aside>
                <h3>Related Links</h3>
                <ul>
                    <li><a href="https://example.com/related1">Related Article 1</a></li>
                    <li><a href="https://example.com/related2">Related Article 2</a></li>
                    <li><a href="/internal-link">Internal Link</a></li>
                </ul>
            </aside>
        </main>
        <footer>
            <p>&copy; 2023 Test Website. All rights reserved.</p>
        </footer>
    </body>
    </html>
    '''


@pytest.fixture
def create_test_file():
    """Factory fixture to create test files with data."""
    def _create_file(file_path, data, format='json'):
        """Create a test file with the given data."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == 'text':
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(str(data))
        
        return file_path
    
    return _create_file


@pytest.fixture
def mock_storage_adapter():
    """Mock storage adapter for testing."""
    mock_adapter = Mock()
    mock_adapter.store.return_value = '/mock/path/to/stored/file.json'
    return mock_adapter


@pytest.fixture
def mock_crawler_config():
    """Mock crawler configuration for testing."""
    mock_config = Mock()
    mock_config.get_crawler_config.return_value = {
        'base_url': 'http://test.example.com',
        'timeout': 30,
        'retry_attempts': 3
    }
    mock_config.get_storage_config.return_value = {
        'type': 'local',
        'format': 'json',
        'base_path': './test_data'
    }
    return mock_config


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "network: marks tests that require network access"
    )


# Custom pytest options
def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests"
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="run integration tests"
    )
    parser.addoption(
        "--run-network",
        action="store_true",
        default=False,
        help="run tests that require network access"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on command line options."""
    if not config.getoption("--run-slow"):
        skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    
    if not config.getoption("--run-integration"):
        skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip_integration)
    
    if not config.getoption("--run-network"):
        skip_network = pytest.mark.skip(reason="need --run-network option to run")
        for item in items:
            if "network" in item.keywords:
                item.add_marker(skip_network)