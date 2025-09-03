"""DataMax CLI Commands

Implements specific commands for crawler functionality.
"""

import sys
import json
import os
import asyncio
from pathlib import Path
from typing import Optional, List

import click
from loguru import logger

from datamax.crawler import (
    CrawlerFactory,
    ArxivCrawler,
    WebCrawler,
    CrawlerConfig,
    create_storage_adapter
)
from datamax.crawler.exceptions import CrawlerException


@click.group()
def crawler():
    """Crawler commands for web scraping and data collection."""
    pass


@crawler.command()
@click.argument('target')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'yaml']), 
              default='json', help='Output format')
@click.option('--storage', '-s', type=click.Choice(['local', 'cloud']), 
              default='local', help='Storage type')
@click.option('--config', '-c', help='Configuration file path')
@click.option('--engine', '-e', type=click.Choice(['auto', 'arxiv', 'web']), 
              default='auto', help='Crawler engine to use')
@click.option('--async-mode', is_flag=True, help='Use async crawling')
@click.pass_context
def crawl(ctx, target, output, format, storage, config, engine, async_mode):
    """Crawl a URL or search query using specified or all crawlers.
    
    TARGET can be a URL, ArXiv ID, or search query.
    With engine=auto, all registered crawlers will be used concurrently.
    """
    try:
        # Load configuration
        if config:
            config_path = Path(config)
            if not config_path.exists():
                click.echo(f"Error: Config file '{config}' not found.", err=True)
                sys.exit(1)
            crawler_config = CrawlerConfig(str(config_path))
        else:
            crawler_config = CrawlerConfig()
        
        # Create storage adapter
        storage_config = {
            'type': storage,
            'format': format
        }
        if output:
            storage_config['base_path'] = str(Path(output).parent)
        
        storage_adapter = create_storage_adapter(storage_config)
        
        if not ctx.obj.get('quiet'):
            click.echo(f"Crawling target: {target}")
            if engine == 'auto':
                click.echo("Using all available engines concurrently")
            else:
                click.echo(f"Using engine: {engine}")
        
        if engine == 'auto':
            # Use the new one-line crawl function with auto mode
            from datamax.crawler import crawl as crawl_function
            result = crawl_function(target, engine='auto')
        elif async_mode:
            # Async crawling with specific engine
            factory = CrawlerFactory()
            result = asyncio.run(_async_crawl(factory, target, storage_adapter, ctx, engine))
        else:
            # Sync crawling with specific engine
            factory = CrawlerFactory()
            crawler = factory.create_crawler(engine)
            crawler.set_storage_adapter(storage_adapter)
            result = crawler.crawl(target)
        
        # Save result
        if output:
            output_path = Path(output)
        else:
            # Generate output filename
            safe_target = "".join(c for c in target if c.isalnum() or c in ('-', '_'))[:50]
            output_path = Path(f"crawl_{safe_target}.{format}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'yaml':
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
        
        if not ctx.obj.get('quiet'):
            click.echo(f"Crawling completed successfully: {output_path}")
            
            # Show summary for auto mode
            if engine == 'auto':
                click.echo(f"Used {result.get('total_engines', 0)} engines:")
                click.echo(f"  Successful: {result.get('successful_engines', 0)}")
                click.echo(f"  Failed: {result.get('failed_engines', 0)}")
            
    except CrawlerException as e:
        logger.error(f"Crawler error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Crawling failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


async def _async_crawl(factory, target, storage_adapter, ctx, engine=None):
    """Perform async crawling."""
    if engine:
        crawler = factory.create_crawler(engine)
    else:
        crawler = factory.create_crawler(target)
    crawler.set_storage_adapter(storage_adapter)
    
    if hasattr(crawler, 'crawl_async'):
        return await crawler.crawl_async(target)
    else:
        # Fallback to sync crawling
        return crawler.crawl(target)


@click.command()
@click.argument('arxiv_input')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'yaml']), 
              default='json', help='Output format')
@click.option('--max-results', '-n', type=int, default=10, 
              help='Maximum number of results for search queries')
@click.option('--sort-by', type=click.Choice(['relevance', 'lastUpdatedDate', 'submittedDate']),
              default='relevance', help='Sort order for search results')
@click.option('--category', help='Filter by ArXiv category (e.g., cs.AI, math.CO)')
@click.pass_context
def arxiv(ctx, arxiv_input, output, format, max_results, sort_by, category):
    """Crawl ArXiv papers by ID, URL, or search query.
    
    ARXIV_INPUT can be:
    - ArXiv ID (e.g., 2301.07041)
    - ArXiv URL (e.g., https://arxiv.org/abs/2301.07041)
    - Search query (e.g., "machine learning transformers")
    """
    try:
        # Create ArXiv crawler
        config = CrawlerConfig()
        crawler = ArxivCrawler(config=config.get_crawler_config('arxiv'))
        
        # Set up storage
        storage_config = {
            'type': 'local',
            'format': format
        }
        storage_adapter = create_storage_adapter(storage_config)
        crawler.set_storage_adapter(storage_adapter)
        
        if not ctx.obj.get('quiet'):
            click.echo(f"Crawling ArXiv: {arxiv_input}")
        
        # Prepare crawl parameters
        crawl_params = {
            'max_results': max_results,
            'sort_by': sort_by
        }
        if category:
            crawl_params['category'] = category
        
        # Perform crawling
        result = crawler.crawl(arxiv_input, **crawl_params)
        
        # Save result
        if output:
            output_path = Path(output)
        else:
            # Generate output filename
            safe_input = "".join(c for c in arxiv_input if c.isalnum() or c in ('-', '_'))[:50]
            output_path = Path(f"arxiv_{safe_input}.{format}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'yaml':
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
        
        if not ctx.obj.get('quiet'):
            click.echo(f"ArXiv crawling completed: {output_path}")
            
            # Show summary
            if isinstance(result.get('data'), list):
                click.echo(f"Found {len(result['data'])} papers")
            elif result.get('type') == 'single_paper':
                click.echo("Retrieved 1 paper")
            
    except CrawlerException as e:
        logger.error(f"ArXiv crawler error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"ArXiv crawling failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command()
@click.argument('target')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'yaml']), 
              default='json', help='Output format')
@click.option('--extract-links', is_flag=True, help='Extract all links from the page')
@click.option('--max-links', type=int, default=100, help='Maximum number of links to extract')
@click.option('--follow-redirects', is_flag=True, help='Follow HTTP redirects')
@click.option('--timeout', type=int, default=30, help='Request timeout in seconds')
@click.option('--search', '-s', is_flag=True, help='Treat target as a search query instead of a URL')
@click.pass_context
def web(ctx, target, output, format, extract_links, max_links, follow_redirects, timeout, search):
    """Crawl a web page or search the web.
    
    TARGET can be a valid HTTP/HTTPS URL or a search query.
    Use --search flag to explicitly treat target as a search query.
    """
    try:
        # Create web crawler
        config = CrawlerConfig()
        web_config = config.get_crawler_config('web')
        web_config.update({
            'timeout': timeout,
            'follow_redirects': follow_redirects,
            'extract_links': extract_links,
            'max_links': max_links
        })
        
        # Add search API configuration if available
        search_api_key = os.environ.get('SEARCH_API_KEY')
        if search_api_key:
            web_config['search_api_key'] = search_api_key
        
        crawler = WebCrawler(config=web_config)
        
        # Set up storage
        storage_config = {
            'type': 'local',
            'format': format
        }
        storage_adapter = create_storage_adapter(storage_config)
        crawler.set_storage_adapter(storage_adapter)
        
        # Determine if target is URL or search query
        from urllib.parse import urlparse
        parsed = urlparse(target)
        is_url = parsed.scheme in ['http', 'https'] and bool(parsed.netloc)
        
        # If search flag is explicitly set, treat as search query
        # If it's not a valid URL, also treat as search query
        is_search = search or (not is_url)
        
        if not ctx.obj.get('quiet'):
            if is_search:
                click.echo(f"Searching web for: {target}")
            else:
                click.echo(f"Crawling web page: {target}")
        
        # Perform crawling
        result = crawler.crawl(target)
        
        # Save result
        if output:
            output_path = Path(output)
        else:
            if is_search:
                # Generate output filename for search
                safe_target = "".join(c for c in target if c.isalnum() or c in ('-', '_'))[:50]
                output_path = Path(f"web_search_{safe_target}.{format}")
            else:
                # Generate output filename from URL
                domain = parsed.netloc.replace('.', '_')
                path_part = parsed.path.replace('/', '_').strip('_')
                safe_name = f"{domain}_{path_part}" if path_part else domain
                safe_name = "".join(c for c in safe_name if c.isalnum() or c in ('-', '_'))[:50]
                output_path = Path(f"web_{safe_name}.{format}")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=str)
        elif format == 'yaml':
            import yaml
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(result, f, default_flow_style=False, allow_unicode=True)
        
        if not ctx.obj.get('quiet'):
            click.echo(f"Web operation completed: {output_path}")
            
            # Show summary based on result type
            if result.get('type') == 'web_search_results':
                result_count = result.get('result_count', 0)
                click.echo(f"Found {result_count} search results")
                if result_count > 0:
                    results = result.get('results', [])
                    for i, res in enumerate(results[:3]):  # Show first 3 results
                        title = res.get('title', 'No title')
                        url = res.get('url', 'No URL')
                        click.echo(f"  {i+1}. {title}")
                        click.echo(f"     {url}")
            else:
                text_length = len(result.get('text_content', ''))
                links_count = len(result.get('links', []))
                click.echo(f"Extracted {text_length} characters of text and {links_count} links")
            
    except CrawlerException as e:
        logger.error(f"Web crawler error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Web operation failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command()
@click.pass_context
def list_crawlers(ctx):
    """List all available crawlers and their capabilities."""
    try:
        factory = CrawlerFactory()
        crawlers = factory.list_crawlers()
        
        click.echo("Available Crawlers:")
        click.echo("=" * 50)
        
        for crawler_name in crawlers:
            click.echo(f"\n📡 {crawler_name.upper()} Crawler")
            
            if crawler_name == 'arxiv':
                click.echo("   Purpose: Academic paper crawling from ArXiv")
                click.echo("   Supports: ArXiv IDs, URLs, search queries")
                click.echo("   Features: Metadata extraction, PDF links, categories")
            elif crawler_name == 'web':
                click.echo("   Purpose: General web page content extraction")
                click.echo("   Supports: HTTP/HTTPS URLs")
                click.echo("   Features: Text extraction, metadata, link discovery")
            else:
                click.echo("   Purpose: Custom crawler")
        
        if not crawlers:
            click.echo("No crawlers registered.")
        
        click.echo("\n💡 Use 'datamax crawler crawl <target>' for automatic crawler selection")
        click.echo("💡 Use 'datamax arxiv <input>' for ArXiv-specific crawling")
        click.echo("💡 Use 'datamax web <url>' for web page crawling")
        
    except Exception as e:
        logger.error(f"Failed to list crawlers: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Export commands for use in main CLI
crawler_command = crawler
arxiv_command = arxiv
web_command = web
list_crawlers_command = list_crawlers