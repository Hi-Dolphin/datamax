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

# Search and crawl papers
datamax crawler arxiv "machine learning" --search --max-results 10

# Specify output directory and format
datamax crawler arxiv 2301.07041 --output ./data --format json
```

#### Crawl Web Pages
```bash
# Crawl a web page
datamax crawler web https://example.com

# Crawl multiple pages
datamax crawler web https://example.com https://another-site.com

# Save to specific location
datamax crawler web https://example.com --output ./web_data
```

#### Parse Crawled Data
```bash
# Parse crawler output to Markdown
datamax parse data/arxiv_2301.07041.json --output paper.md

# Parse with specific type
datamax parse data/web_page.json --type crawler --output page.md
```

#### System Status
```bash
# Check system status and available components
datamax status

# List available crawlers
datamax crawler list
```

### Python API Usage

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
config = CrawlerConfig({
    'arxiv': {
        'max_results': 50,
        'timeout': 60
    },
    'storage': {
        'type': 'local',
        'base_path': './custom_data'
    }
})

# Create specific crawler
arxiv_crawler = ArxivCrawler(config=config)

# Search for papers
async def search_papers():
    results = await arxiv_crawler.search_async(
        query="machine learning",
        max_results=10
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

### Default Configuration
DataMax uses a hierarchical configuration system. Create a `config.json` file:

```json
{
  "arxiv": {
    "base_url": "http://export.arxiv.org/api/",
    "max_results": 10,
    "timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 1.0
  },
  "web": {
    "timeout": 30,
    "retry_attempts": 3,
    "retry_delay": 1.0,
    "user_agent": "DataMax-Crawler/1.0",
    "max_content_length": 10485760
  },
  "storage": {
    "type": "local",
    "format": "json",
    "base_path": "./data",
    "create_subdirs": true
  },
  "logging": {
    "level": "INFO",
    "format": "standard",
    "file_output": false
  }
}
```

### Environment Variables
```bash
# Override configuration with environment variables
export DATAMAX_ARXIV_TIMEOUT=60
export DATAMAX_STORAGE_PATH=/custom/data/path
export DATAMAX_LOG_LEVEL=DEBUG
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

## 📚 API Reference

### Crawler API

#### CrawlerFactory
```python
factory = CrawlerFactory(config=None)
crawler = factory.create_crawler(target)  # Auto-detect crawler type
crawlers = factory.list_crawlers()        # List available crawlers
```

#### ArxivCrawler
```python
crawler = ArxivCrawler(config=None)
result = await crawler.crawl_async(arxiv_id)
results = await crawler.search_async(query, max_results=10)
valid = crawler.validate_target(target)
```

#### WebCrawler
```python
crawler = WebCrawler(config=None)
result = await crawler.crawl_async(url)
results = await crawler.crawl_multiple_async(urls)
valid = crawler.validate_target(url)
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
