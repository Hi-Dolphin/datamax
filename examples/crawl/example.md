# DataMax Crawl Module Examples

## Overview

The Crawl module provides powerful web crawling and data collection capabilities. It supports crawling ArXiv papers, general web pages, and can automatically select the best crawler for your target.

## Prerequisites

No API keys required for basic crawling functionality. However, for web search functionality, you may need to set up search API keys.

## CLI Command Examples

### Automatic Crawler Selection

Let DataMax automatically choose the best crawler for your target:

```bash
datamax crawler crawl "https://arxiv.org/abs/2301.07041"
```

This will automatically detect it's an ArXiv URL and use the ArXiv crawler:

```bash
datamax crawler crawl "machine learning transformers" --engine web
```

### ArXiv Paper Crawling

#### Single Paper by ID
```bash
datamax arxiv 2301.07041
```

#### Paper by URL
```bash
datamax arxiv "https://arxiv.org/abs/2301.07041"
```

#### Search with Filters
```bash
datamax arxiv "machine learning" --max-results 5 --category cs.AI
```

#### Custom Output Location
```bash
datamax arxiv 2301.07041 --output my_paper.json --format json
```

### Web Page Crawling

#### Basic Web Page Crawl
```bash
datamax web "https://en.wikipedia.org/wiki/Machine_learning"
```

#### With Link Extraction
```bash
datamax web "https://example.com" --extract-links --max-links 20
```

#### Custom Timeout and Redirect Handling
```bash
datamax web "https://example.com" --timeout 60 --follow-redirects
```

#### Web Search Query
```bash
datamax web "artificial intelligence news" --search
```

### Advanced Options

#### Using Specific Storage Format
```bash
datamax crawler crawl "https://example.com" --format yaml --output result.yaml
```

#### List Available Crawlers
```bash
datamax list-crawlers
```

## Python Code Examples

### Basic ArXiv Crawling

```python
from datamax.crawler import ArxivCrawler, CrawlerConfig

# Create configuration
config = CrawlerConfig()

# Initialize ArXiv crawler
crawler = ArxivCrawler(config=config.get_crawler_config('arxiv'))

# Crawl a single paper
result = crawler.crawl('2301.07041')
print("Paper title:", result.get('title', 'N/A'))
print("Authors:", result.get('authors', []))
print("Abstract:", result.get('abstract', '')[:200] + "...")
```

### Web Page Crawling

```python
from datamax.crawler import WebCrawler, CrawlerConfig

# Create configuration
config = CrawlerConfig()

# Initialize web crawler
web_config = config.get_crawler_config('web')
web_config.update({
    'timeout': 30,
    'follow_redirects': True,
    'extract_links': True,
    'max_links': 10
})

crawler = WebCrawler(config=web_config)

# Crawl a web page
result = crawler.crawl('https://en.wikipedia.org/wiki/Machine_learning')
print("Page title:", result.get('title', 'N/A'))
print("Content length:", len(result.get('text_content', '')))
print("Links found:", len(result.get('links', [])))

# Print first few links
for i, link in enumerate(result.get('links', [])[:5]):
    print(f"Link {i+1}: {link.get('url', 'N/A')}")
```

### Using the Unified Crawl Interface

```python
from datamax.crawler import crawl, CrawlerFactory

# Automatic crawling (DataMax chooses the best crawler)
result = crawl('https://arxiv.org/abs/2301.07041', engine='auto')
print("Auto crawl result type:", result.get('type', 'N/A'))
print("Engines used:", result.get('total_engines', 0))

# List available crawlers
factory = CrawlerFactory()
crawlers = factory.list_crawlers()
print("Available crawlers:", crawlers)

# Create and use specific crawler
crawler = factory.create_crawler('arxiv')
result = crawler.crawl('2301.07041')
print("ArXiv crawl successful:", 'title' in result)
```

### Batch Processing with Storage

```python
from datamax.crawler import CrawlerFactory, create_storage_adapter
from pathlib import Path

# Create storage adapter
storage_config = {
    'type': 'local',
    'format': 'json',
    'base_path': 'examples/crawl/output'
}
storage_adapter = create_storage_adapter(storage_config)

# Create crawlers
factory = CrawlerFactory()
arxiv_crawler = factory.create_crawler('arxiv')
web_crawler = factory.create_crawler('web')

# Set storage adapters
arxiv_crawler.set_storage_adapter(storage_adapter)
web_crawler.set_storage_adapter(storage_adapter)

# Process multiple targets
targets = [
    ('2301.07041', arxiv_crawler),
    ('https://github.com/microsoft/vscode', web_crawler)
]

for target, crawler in targets:
    try:
        result = crawler.crawl(target)
        print(f"Successfully crawled: {target}")

        # Save result
        if hasattr(crawler, 'save_result'):
            output_file = f"result_{Path(target).name or 'output'}.json"
            crawler.save_result(result, output_file)
            print(f"Result saved to: {output_file}")

    except Exception as e:
        print(f"Failed to crawl {target}: {str(e)}")
```

### Advanced Configuration

```python
from datamax.crawler import ArxivCrawler, CrawlerConfig

# Create custom configuration
config = CrawlerConfig()

arxiv_config = config.get_crawler_config('arxiv')
arxiv_config.update({
    'max_results': 3,
    'sort_by': 'relevance',
    'category': 'cs.AI',
    'max_retries': 3,
    'retry_delay': 1.0
})

crawler = ArxivCrawler(config=arxiv_config)

# Search with advanced parameters
result = crawler.crawl(
    'deep learning',
    max_results=3,
    sort_by='relevance',
    category='cs.AI'
)

print("Search results:")
for paper in result.get('data', []):
    print(f"- {paper.get('title', 'N/A')} ({paper.get('published', 'N/A')})")
```

### Async Crawling Example

```python
import asyncio
from datamax.crawler import CrawlerFactory

async def async_crawl_example():
    factory = CrawlerFactory()
    crawler = factory.create_crawler('web')

    # Check if async crawling is supported
    if hasattr(crawler, 'crawl_async'):
        result = await crawler.crawl_async('https://example.com')
        print("Async crawl completed:", bool(result))
    else:
        # Fallback to sync crawling
        result = crawler.crawl('https://example.com')
        print("Sync crawl completed:", bool(result))

# Run the async example
asyncio.run(async_crawl_example())
```

## Expected Output

### ArXiv Crawl Result
```json
{
  "type": "single_paper",
  "title": "Paper Title",
  "authors": ["Author 1", "Author 2"],
  "abstract": "Paper abstract...",
  "published": "2023-01-07",
  "categories": ["cs.AI"],
  "pdf_url": "https://arxiv.org/pdf/2301.07041.pdf"
}
```

### Web Crawl Result
```json
{
  "type": "web_content",
  "title": "Page Title",
  "text_content": "Extracted text content...",
  "links": [
    {"url": "https://example.com/link1", "text": "Link Text"},
    {"url": "https://example.com/link2", "text": "Another Link"}
  ],
  "metadata": {
    "status_code": 200,
    "content_type": "text/html"
  }
}
```

## Best Practices

1. Use automatic crawler selection (`engine=auto`) for mixed content types
2. Set appropriate timeouts for slower websites
3. Enable link extraction when you need to discover related content
4. Use JSON format for structured processing of results
5. Implement retry logic for production crawling scenarios
6. Respect robots.txt and implement rate limiting for ethical crawling
