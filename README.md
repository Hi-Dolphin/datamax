# DataMax - Advanced Data Crawling and Processing Framework

<div align="center">

[中文](README_zh.md) | **English**

[![PyPI version](https://badge.fury.io/py/pydatamax.svg)](https://badge.fury.io/py/pydatamax) [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Tests](https://img.shields.io/badge/tests-pytest-orange.svg)](tests/)

</div>

DataMax is a comprehensive, extensible framework for crawling, parsing, and processing data from various sources. It provides a unified interface for handling different data sources including academic papers from ArXiv, web pages, and more.

## 🚀 Features

### Core Capabilities
- **Multi-Source Crawling**: Support for ArXiv papers, web pages, and extensible to other sources
- **Intelligent Parser**: Automatic data parsing and conversion to structured formats
- **Flexible Storage**: Multiple storage backends (local files, databases)
- **Async Processing**: High-performance asynchronous crawling
- **CLI Interface**: Command-line tools for easy automation
- **Extensible Architecture**: Plugin-based design for easy extension

### Crawler Features
- **ArXiv Integration**: Direct access to ArXiv papers by ID, URL, or search queries
- **Web Crawling**: General-purpose web page crawling with content extraction
- **Rate Limiting**: Built-in rate limiting and retry mechanisms
- **Error Handling**: Comprehensive error handling and logging
- **Configuration Management**: Flexible configuration system

### Parser Features
- **Multi-Format Support**: Parse crawler data into Markdown, JSON, and other formats
- **Metadata Extraction**: Automatic extraction of titles, authors, abstracts, and more
- **Link Processing**: Intelligent handling of internal and external links
- **Content Cleaning**: Text cleaning and normalization

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Install from Source
```bash
# Clone the repository
git clone https://github.com/your-org/datamax.git
cd datamax

# Install in development mode
pip install -e .

# Or install with test dependencies
pip install -e ".[test]"
```

### Install Dependencies
```bash
# Core dependencies
pip install aiohttp beautifulsoup4 click

# Development dependencies
pip install pytest pytest-asyncio pytest-cov
```

## 🎯 Quick Start

### Command Line Usage

#### Crawl ArXiv Papers
```bash
# Crawl a single paper by ArXiv ID
datamax crawler arxiv 2301.07041

# Crawl by ArXiv URL
datamax crawler arxiv https://arxiv.org/abs/2301.07041

# Search and crawl papers with specific count
datamax crawler arxiv "machine learning" --search --max-results 5

# Specify output directory and format
datamax crawler arxiv 2301.07041 --output ./data --format json
```

#### Search the Web
```bash
# Search the web with keywords and specific count
datamax crawler web "latest AI news" --count 5

# Save search results to specific location
datamax crawler web "machine learning research" --output ./web_data --count 10
```

#### Use All Engines
```bash
# Use all engines with specific count
datamax crawler crawl "machine learning" --engine auto --output ./data --count 5
```

### Python API Usage

#### One-Line Crawling
For simple use cases, you can crawl data with a single line of code:

```python
import datamax

# Crawl ArXiv papers with specific count
result = datamax.crawl("machine learning", engine="arxiv", count=5)

# Search the web with keywords and specific count
result = datamax.crawl("latest AI news", engine="web", count=5)

# Use all engines concurrently with specific count
result = datamax.crawl("machine learning", engine="auto", count=5)

# Convenience functions with count parameter
result = datamax.crawl_arxiv("quantum computing", count=3)
result = datamax.crawl_web("latest news", count=7)
```

The `auto` engine mode runs all registered crawlers concurrently and combines their results,
making it easy to gather data from multiple sources with a single call.

#### Basic Crawling
```python
import asyncio
from datamax.crawler import CrawlerFactory, create_storage_adapter

# Create crawler factory
factory = CrawlerFactory()

# Create storage adapter
storage_config = {
    'type': 'local',
    'format': 'json',
    'base_path': './data'
}
storage = create_storage_adapter(storage_config)

# Crawl ArXiv paper
crawler = factory.create_crawler('2301.07041')
crawler.set_storage_adapter(storage)

# Async crawling
async def crawl_paper():
    result = await crawler.crawl_async('2301.07041')
    return result

# Run the crawler
result = asyncio.run(crawl_paper())
print(f"Crawled data saved to: {result}")
```

#### Advanced Usage
```python
from datamax.crawler import ArxivCrawler, WebCrawler, CrawlerConfig
from datamax.parser import CrawlerParser

# Custom configuration
config = {
    'arxiv': {
        'max_results': 50,
        'timeout': 60
    },
    'storage': {
        'type': 'local',
        'base_path': './custom_data'
    }
}

# Create specific crawler
arxiv_crawler = ArxivCrawler(config=config)

# Search for papers with specific count
async def search_papers():
    results = await arxiv_crawler.search_async(
        query="machine learning",
        max_results=5  # Specify count here
    )
    return results

# Parse results
parser = CrawlerParser()
parser.load_data('data/search_results.json')
markdown_content = parser.parse()

with open('results.md', 'w') as f:
    f.write(markdown_content)
```

#### Using the CLI Class
```python
from datamax.cli import CrawlerCLI

# Create CLI instance
cli = CrawlerCLI()

# List available crawlers
crawlers = cli.list_crawlers()
print("Available crawlers:", crawlers)

# Crawl with automatic type detection
async def auto_crawl():
    result = await cli.crawl(
        target='https://arxiv.org/abs/2301.07041',
        output_dir='./data'
    )
    return result

result_path = asyncio.run(auto_crawl())
print(f"Data saved to: {result_path}")
```

## 🏗️ Architecture

### Core Components

```
DataMax/
├── crawler/           # Crawling engine
│   ├── base.py       # Base crawler interface
│   ├── arxiv.py      # ArXiv crawler implementation
│   ├── web.py        # Web crawler implementation
│   ├── crawl.py      # One-line crawler interface
│   ├── factory.py    # Crawler factory
│   ├── config.py     # Configuration management
│   └── storage.py    # Storage adapters
├── parser/           # Data parsing engine
│   ├── base.py       # Base parser interface
│   └── crawler.py    # Crawler data parser
├── cli/              # Command-line interface
│   ├── main.py       # Main CLI entry point
│   ├── commands.py   # CLI commands
│   └── crawler_cli.py # Crawler CLI class
└── utils/            # Utility modules
    └── lifecycle.py  # Lifecycle management
```

### Design Principles
- **Modularity**: Each component is independent and replaceable
- **Extensibility**: Easy to add new crawlers, parsers, and storage backends
- **Async-First**: Built for high-performance asynchronous operations
- **Configuration-Driven**: Behavior controlled through configuration files
- **Error Resilience**: Comprehensive error handling and recovery

## 🔧 Configuration

### Environment Variables
All configuration for DataMax is now handled through environment variables, simplifying setup and deployment.

#### Web Crawler Configuration
- `SEARCH_API_KEY` - API key for the web search API (required for web crawler)
- `WEB_SEARCH_API_URL` - URL for the web search API (default: https://api.bochaai.com/v1/web-search)
- `WEB_USER_AGENT` - User agent string for web requests (default: DataMax-Crawler/1.0)
- `WEB_TIMEOUT` - Timeout for web requests in seconds (default: 15)
- `WEB_MAX_RETRIES` - Maximum number of retry attempts (default: 2)
- `WEB_RATE_LIMIT` - Rate limit between requests in seconds (default: 0.5)

#### ArXiv Crawler Configuration
- `ARXIV_BASE_URL` - Base URL for ArXiv API (default: https://arxiv.org/)
- `ARXIV_USER_AGENT` - User agent string for ArXiv requests (default: DataMax-Crawler/1.0)
- `ARXIV_TIMEOUT` - Timeout for ArXiv requests in seconds (default: 30)
- `ARXIV_MAX_RETRIES` - Maximum number of retry attempts (default: 3)
- `ARXIV_RATE_LIMIT` - Rate limit between requests in seconds (default: 1.0)

#### Storage Configuration
- `STORAGE_DEFAULT_FORMAT` - Default storage format (json or yaml) (default: json)
- `STORAGE_OUTPUT_DIR` - Output directory for stored data (default: ./output)
- `STORAGE_CLOUD_ENABLED` - Enable cloud storage (true/false) (default: false)
- `STORAGE_CLOUD_PROVIDER` - Cloud storage provider (s3, gcs, azure) (default: s3)

#### Logging Configuration
- `LOG_LEVEL` - Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)
- `LOG_FILE` - Path to log file (optional)
- `LOG_ENABLE_JSON` - Enable JSON formatted logging (true/false) (default: false)
- `LOG_ENABLE_CONSOLE` - Enable console logging (true/false) (default: true)

For more details, see [crawler configuration documentation](datamax/crawler/README.md).

### Usage Examples

#### Setting Environment Variables (Linux/Mac)
```bash
export SEARCH_API_KEY="your-search-api-key"
export ARXIV_TIMEOUT=60
export STORAGE_OUTPUT_DIR="/path/to/output"
```

#### Setting Environment Variables (Windows)
```cmd
set SEARCH_API_KEY=your-search-api-key
set ARXIV_TIMEOUT=60
set STORAGE_OUTPUT_DIR=C:\path\to\output
```

#### Using with Docker
```dockerfile
ENV SEARCH_API_KEY=your-search-api-key
ENV ARXIV_TIMEOUT=60
ENV STORAGE_OUTPUT_DIR=/app/output
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
python run_tests.py

# Run specific test categories
python run_tests.py --unit          # Unit tests only
python run_tests.py --integration   # Integration tests
python run_tests.py --coverage      # With coverage report

# Run tests for specific modules
python run_tests.py --module crawler
python run_tests.py --module parser
python run_tests.py --module cli

# Using pytest directly
pytest tests/
pytest tests/test_crawler.py -v
pytest tests/ --cov=datamax
```

### Test Categories
- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Network Tests**: Test actual network operations (optional)
- **CLI Tests**: Test command-line interface functionality

### API Reference

#### Core Functions
```python
# Main crawl function with count parameter
datamax.crawl(keyword, engine="auto", count=10)

# ArXiv specific function with count parameter
datamax.crawl_arxiv(keyword, count=10)

# Web search specific function with count parameter
datamax.crawl_web(keyword, count=10)
```

#### CrawlerFactory
```python
factory = CrawlerFactory(config=None)
crawler = factory.create_crawler(target)  # Auto-detect crawler type
crawlers = factory.list_crawlers()        # List available crawlers
```

#### ArxivCrawler
```python
crawler = ArxivCrawler(config=None)
result = await crawler.crawl_async(arxiv_id, max_results=10)
results = await crawler.search_async(query, max_results=5)
valid = crawler.validate_target(target)
```

#### WebCrawler
```python
crawler = WebCrawler(config=None)
result = await crawler.crawl_async("search keywords", max_results=10)
valid = crawler.validate_target("search keywords")
```

### Parser API

#### CrawlerParser
```python
parser = CrawlerParser()
parser.load_data(file_path)
markdown = parser.parse()
data = parser.get_data()
```

### Storage API

#### LocalStorageAdapter
```python
storage = LocalStorageAdapter(config)
file_path = storage.store(data)
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/your-org/datamax.git
cd datamax
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python run_tests.py
```

### Adding New Crawlers
1. Create a new crawler class inheriting from `BaseCrawler`
2. Implement required methods: `crawl()`, `validate_target()`
3. Register the crawler in `CrawlerFactory`
4. Add tests and documentation

### Adding New Parsers
1. Create a new parser class inheriting from `BaseLife`
2. Implement parsing logic for your data format
3. Add the parser to the parser module
4. Add tests and documentation


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- ArXiv for providing open access to academic papers
- The Python community for excellent libraries
- Contributors and users of the DataMax framework

## 📞 Support

- **Documentation**: [Full documentation](docs/)
- **Issues**: [GitHub Issues](https://github.com/your-org/datamax/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/datamax/discussions)
- 📧 Email: cy.kron@foxmail.com
- 💬 Wechat Group: <br><img src='wechat.jpg' width=300>

---

**DataMax** - Empowering data collection and processing with Python 🐍✨

⭐ If this project helps you, please give us a star!
