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
from datamax.cleaner import AbnormalCleaner, TextFilter, PrivacyDesensitization


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


@click.command()
@click.argument('input_file', required=False)
@click.option('--output', '-o', help='Output file path')
@click.option('--mode', '-m', type=click.Choice(['full', 'abnormal', 'filter', 'privacy', 'no_html']),
              default='full', help='Cleaning mode')
@click.option('--filter-threshold', type=float, default=0.6,
              help='Word repetition threshold for filtering (0.0-1.0)')
@click.option('--min-chars', type=int, default=30,
              help='Minimum character count for filtering')
@click.option('--max-chars', type=int, default=500000,
              help='Maximum character count for filtering')
@click.option('--numeric-threshold', type=float, default=0.6,
              help='Numeric content threshold for filtering (0.0-1.0)')
@click.option('--stdin', is_flag=True, help='Read from stdin instead of file')
@click.option('--stdout', is_flag=True, help='Write to stdout instead of file')
@click.pass_context
def clean(ctx, input_file, output, mode, filter_threshold, min_chars, max_chars,
          numeric_threshold, stdin, stdout):
    """Clean and process text data.

    Clean text using various modes:
    - full: Complete cleaning (abnormal + filter + privacy)
    - abnormal: Remove abnormal characters, HTML, normalize text
    - filter: Filter by content quality (repetition, length, numeric content)
    - privacy: Remove sensitive information (emails, phones, IDs, etc.)
    - no_html: Basic cleaning without HTML removal

    INPUT_FILE can be a text file path. Use --stdin to read from stdin.
    """
    try:
        # Read input
        if stdin:
            if not ctx.obj.get('quiet'):
                click.echo("Reading from stdin...")
            text = sys.stdin.read()
            if not text.strip():
                click.echo("Error: No input received from stdin.", err=True)
                sys.exit(1)
        else:
            if not input_file:
                input_file = click.prompt("Enter input file path")

            input_path = Path(input_file)
            if not input_path.exists():
                click.echo(f"Error: Input file '{input_file}' not found.", err=True)
                sys.exit(1)

            if not ctx.obj.get('quiet'):
                click.echo(f"Reading input file: {input_file}")

            with open(input_path, 'r', encoding='utf-8') as f:
                text = f.read()

        # Perform cleaning based on mode
        if not ctx.obj.get('quiet'):
            click.echo(f"Cleaning text in '{mode}' mode...")

        result = {}

        if mode == 'full':
            # Complete cleaning pipeline
            cleaner = AbnormalCleaner(text)
            cleaned = cleaner.to_clean()

            if isinstance(cleaned, dict) and 'text' in cleaned and cleaned['text']:
                filter_obj = TextFilter(cleaned['text'])
                filtered = filter_obj.to_filter()

                if isinstance(filtered, dict) and 'text' in filtered and filtered['text']:
                    privacy = PrivacyDesensitization(filtered['text'])
                    result = privacy.to_private()
                else:
                    if not ctx.obj.get('quiet'):
                        click.echo("Warning: Text failed filtering, skipping privacy cleaning")
                    result = cleaned
            else:
                result = cleaned

        elif mode == 'abnormal':
            cleaner = AbnormalCleaner(text)
            result = cleaner.to_clean()

        elif mode == 'filter':
            # Apply abnormal cleaning first, then filter
            cleaner = AbnormalCleaner(text)
            cleaned = cleaner.no_html_clean()

            if isinstance(cleaned, dict) and 'text' in cleaned and cleaned['text']:
                filter_obj = TextFilter(cleaned['text'])
                filter_obj.filter_by_word_repetition(filter_threshold)
                filter_obj.filter_by_char_count(min_chars, max_chars)
                filter_obj.filter_by_numeric_content(numeric_threshold)
                result = filter_obj.to_filter()
            else:
                result = cleaned

        elif mode == 'privacy':
            privacy = PrivacyDesensitization(text)
            result = privacy.to_private()

        elif mode == 'no_html':
            cleaner = AbnormalCleaner(text)
            result = cleaner.no_html_clean()

        # Check if cleaning was successful
        if isinstance(result, dict) and 'text' in result and result['text']:
            cleaned_text = result['text']
            if not ctx.obj.get('quiet'):
                click.echo("Cleaning completed successfully")
                click.echo(f"Original length: {len(text)} characters")
                click.echo(f"Cleaned length: {len(cleaned_text)} characters")

            # Output handling
            if stdout:
                click.echo(cleaned_text)
            else:
                # Determine output path
                if output:
                    output_path = Path(output)
                else:
                    if stdin:
                        output_path = Path("cleaned_output.txt")
                    else:
                        input_path = Path(input_file)
                        output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"

                output_path.parent.mkdir(parents=True, exist_ok=True)

                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_text)

                if not ctx.obj.get('quiet'):
                    click.echo(f"Cleaned text saved to: {output_path}")

        else:
            if isinstance(result, dict):
                click.echo("Warning: Cleaning returned empty result", err=True)
            else:
                click.echo(f"Error: Unexpected result type: {type(result)}", err=True)
            sys.exit(1)

    except Exception as e:
        logger.error(f"Cleaning failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command()
@click.pass_context
def list_cleaners(ctx):
    """List available cleaning modes and their capabilities."""
    try:
        from .cleaner_cli import CleanerCLI

        cleaner_cli = CleanerCLI()
        cleaning_modes = cleaner_cli.list_cleaning_modes()

        click.echo("Available Cleaning Modes:")
        click.echo("=" * 50)

        for mode_name, description in cleaning_modes.items():
            click.echo(f"\n🧹 {mode_name.upper()} Mode")
            click.echo(f"   Description: {description}")

            # Get detailed info
            mode_info = cleaner_cli.get_cleaning_info(mode_name)
            if mode_info.get('steps'):
                click.echo(f"   Steps: {', '.join(mode_info['steps'])}")
            if mode_info.get('parameters'):
                click.echo(f"   Parameters: {', '.join(mode_info['parameters'])}")

        click.echo("\n💡 Use 'datamax clean <input_file> --mode <mode>' to clean text")
        click.echo("💡 Use 'datamax clean --stdin --mode <mode>' to clean from stdin")
        click.echo("💡 Use 'datamax clean <input_file> --stdout' to output to stdout")

    except Exception as e:
        logger.error(f"Failed to list cleaners: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Generator commands
from datamax.cli.generator_cli import GeneratorCLI


@click.command()
@click.argument('input_file')
@click.option('--output', '-o', help='Output file path')
@click.option('--api-key', help='API key')
@click.option('--base-url', help='API base URL')
@click.option('--model', '-m', help='Model name')
@click.option('--chunk-size', type=int, default=500, help='Text chunk size')
@click.option('--chunk-overlap', type=int, default=100, help='Chunk overlap size')
@click.option('--question-number', type=int, default=5, help='Number of questions per chunk')
@click.option('--max-workers', type=int, default=5, help='Maximum number of workers')
@click.pass_context
def qa(ctx, input_file, output, api_key, base_url, model, chunk_size, chunk_overlap,
       question_number, max_workers):
    """Generate QA pairs from text files.

    Generate question-answer pairs from various text formats using LLM.
    Supports PDF, Markdown, and other text documents.
    """
    try:
        # Create output path if specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            input_path = Path(input_file)
            output_path = input_path.parent / f"{input_path.stem}_qa.json"

        # Create generator CLI instance
        generator = GeneratorCLI(verbose=ctx.obj.get('verbose', False))

        if not ctx.obj.get('quiet'):
            click.echo(f"Generating QA pairs from: {input_file}")
            click.echo(f"Output will be saved to: {output_path}")

        # Generate QA pairs
        result = generator.generate_qa(
            input_file=input_file,
            output_file=str(output_path),
            api_key=api_key,
            base_url=base_url,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            question_number=question_number,
            max_workers=max_workers
        )

        # Save result
        saved_path = generator.save_result(result, str(output_path))

        if not ctx.obj.get('quiet'):
            qa_count = len(result.get('qa_pairs', []))
            click.echo(f"QA generation completed successfully!")
            click.echo(f"Generated {qa_count} QA pairs")
            click.echo(f"Results saved to: {saved_path}")

    except Exception as e:
        logger.error(f"QA generation failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command()
@click.argument('input_file')
@click.option('--output', '-o', help='Output file path')
@click.option('--api-key', help='OpenAI API key')
@click.option('--model', '-m', default='gpt-4-vision-preview', help='Model name')
@click.option('--chunk-size', type=int, default=2000, help='Text chunk size')
@click.option('--chunk-overlap', type=int, default=300, help='Chunk overlap size')
@click.option('--question-number', type=int, default=2, help='Number of questions per chunk')
@click.option('--max-workers', type=int, default=5, help='Maximum number of workers')
@click.pass_context
def multimodal(ctx, input_file, output, api_key, model, chunk_size, chunk_overlap,
              question_number, max_workers):
    """Generate multimodal QA pairs from markdown files with images.

    Generate question-answer pairs from markdown files containing images.
    Requires OpenAI API key for vision capabilities.
    """
    try:
        # Create output path if specified
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            input_path = Path(input_file)
            output_path = input_path.parent / f"{input_path.stem}_multimodal_qa.json"

        # Create generator CLI instance
        generator = GeneratorCLI(verbose=ctx.obj.get('verbose', False))

        if not ctx.obj.get('quiet'):
            click.echo(f"Generating multimodal QA pairs from: {input_file}")
            click.echo(f"Output will be saved to: {output_path}")

        # Generate multimodal QA pairs
        result = generator.generate_multimodal_qa(
            input_file=input_file,
            output_file=str(output_path),
            api_key=api_key,
            model=model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            question_number=question_number,
            max_workers=max_workers
        )

        # Save result
        saved_path = generator.save_result(result, str(output_path))

        if not ctx.obj.get('quiet'):
            qa_count = len(result)
            click.echo(f"Multimodal QA generation completed successfully!")
            click.echo(f"Generated {qa_count} QA pairs")
            click.echo(f"Results saved to: {saved_path}")

    except Exception as e:
        logger.error(f"Multimodal QA generation failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command()
@click.pass_context
def list_generators(ctx):
    """List available generators and their descriptions."""
    try:
        generator = GeneratorCLI()
        generators = generator.list_generators()

        click.echo("Available Generators:")
        click.echo("=" * 50)

        for name, description in generators.items():
            click.echo(f"\n🔧 {name.upper()}")
            click.echo(f"   {description}")

        click.echo("\n💡 Usage Examples:")
        click.echo("   datamax generator qa document.pdf")
        click.echo("   datamax generator multimodal document.md --api-key $OPENAI_API_KEY")
        click.echo("   datamax generator list")

    except Exception as e:
        logger.error(f"Failed to list generators: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Parser commands
from datamax.cli.parser_cli import ParserCLI


@click.command()
@click.argument('input_file')
@click.argument('output_file', required=False)
@click.option('--format', '-f', type=click.Choice(['markdown', 'json', 'text']),
              default='markdown', help='Output format')
@click.option('--domain', '-d', default='Technology', help='Document domain')
@click.option('--use-mineru', is_flag=True, help='Use MinerU for PDF parsing')
@click.option('--use-qwen-vl-ocr', is_flag=True, help='Use Qwen-VL OCR for PDF')
@click.option('--use-mllm', is_flag=True, help='Use Vision model for images')
@click.option('--api-key', help='API key for OCR/MLLM services')
@click.option('--base-url', help='Base URL for API services')
@click.option('--model', '-m', help='Model name for API services')
@click.option('--mllm-prompt', help='System prompt for Vision model')
@click.option('--to-markdown', is_flag=True, help='Convert to Markdown format')
@click.pass_context
def parse(ctx, input_file, output_file, format, domain, use_mineru,
          use_qwen_vl_ocr, use_mllm, api_key, base_url, model, mllm_prompt, to_markdown):
    """Parse a single file using DataMax parser.

    INPUT_FILE: Path to the file to parse
    OUTPUT_FILE: Optional output file path (auto-generated if not provided)

    Examples:
        datamax parser parse document.pdf
        datamax parser parse document.pdf output.md --format markdown
        datamax parser parse image.jpg --use-mllm --api-key $OPENAI_API_KEY --mllm-prompt "Describe this image"
        datamax parser parse document.pdf --use-mineru
        datamax parser parse document.docx --to-markdown
    """
    try:
        # Validate options
        if use_mineru and use_qwen_vl_ocr:
            raise click.BadParameter("Cannot use both --use-mineru and --use-qwen-vl-ocr simultaneously")

        if use_mllm and (use_mineru or use_qwen_vl_ocr):
            raise click.BadParameter("Cannot use --use-mllm with PDF options (--use-mineru/--use-qwen-vl-ocr)")

        # Generate output file path if not provided
        if not output_file:
            input_path = Path(input_file)
            extension = {
                'markdown': 'md',
                'json': 'json',
                'text': 'txt'
            }.get(format, 'md')
            output_file = f"{input_path.stem}_parsed.{extension}"

        # Create parser CLI instance
        parser = ParserCLI(verbose=ctx.obj.get('verbose', False))

        if not ctx.obj.get('quiet'):
            click.echo(f"Parsing file: {input_file}")
            click.echo(f"Output will be saved to: {output_file}")

        # Parse the file
        result = parser.parse_file(
            input_file=input_file,
            output_file=output_file,
            format=format,
            domain=domain,
            use_mineru=use_mineru,
            use_qwen_vl_ocr=use_qwen_vl_ocr,
            use_mllm=use_mllm,
            mllm_system_prompt=mllm_prompt or "描述图片内容，包括图片中的文字、图片中的对象、图片中的场景等。输出一份专业的中文markdown报告",
            api_key=api_key,
            base_url=base_url,
            model_name=model,
            to_markdown=to_markdown
        )

        if not ctx.obj.get('quiet'):
            click.echo("✅ Parsing completed successfully!")
            click.echo(f"📄 Results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Parsing failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command()
@click.argument('input_dir')
@click.argument('output_dir')
@click.option('--format', '-f', type=click.Choice(['markdown', 'json', 'text']),
              default='markdown', help='Output format')
@click.option('--pattern', '-p', default='*.*', help='File pattern to match')
@click.option('--recursive', '-r', is_flag=True, help='Process directories recursively')
@click.option('--max-workers', type=int, default=4, help='Maximum concurrent workers')
@click.option('--domain', '-d', default='Technology', help='Document domain')
@click.option('--use-mineru', is_flag=True, help='Use MinerU for PDF parsing')
@click.option('--use-qwen-vl-ocr', is_flag=True, help='Use Qwen-VL OCR for PDF')
@click.option('--use-mllm', is_flag=True, help='Use Vision model for images')
@click.option('--api-key', help='API key for OCR/MLLM services')
@click.option('--base-url', help='Base URL for API services')
@click.option('--model', '-m', help='Model name for API services')
@click.option('--mllm-prompt', help='System prompt for Vision model')
@click.option('--to-markdown', is_flag=True, help='Convert to Markdown format')
@click.pass_context
def batch(ctx, input_dir, output_dir, format, pattern, recursive, max_workers,
          domain, use_mineru, use_qwen_vl_ocr, use_mllm, api_key, base_url,
          model, mllm_prompt, to_markdown):
    """Parse multiple files in batch mode.

    INPUT_DIR: Directory containing files to parse
    OUTPUT_DIR: Directory to save parsed results

    Examples:
        datamax parser batch ./documents ./parsed
        datamax parser batch ./docs ./output --recursive --max-workers 8
        datamax parser batch ./pdfs ./output --use-mineru --pattern "*.pdf"
        datamax parser batch ./images ./output --use-mllm --api-key $OPENAI_API_KEY
    """
    try:
        # Validate options
        if use_mineru and use_qwen_vl_ocr:
            raise click.BadParameter("Cannot use both --use-mineru and --use-qwen-vl-ocr simultaneously")

        # Parse options for passing to parse_file
        parse_options = {
            'domain': domain,
            'use_mineru': use_mineru,
            'use_qwen_vl_ocr': use_qwen_vl_ocr,
            'use_mllm': use_mllm,
            'mllm_system_prompt': mllm_prompt or "描述图片内容，包括图片中的文字、图片中的对象、图片中的场景等。输出一份专业的中文markdown报告",
            'api_key': api_key,
            'base_url': base_url,
            'model_name': model,
            'to_markdown': to_markdown
        }

        # Create parser CLI instance
        parser = ParserCLI(verbose=ctx.obj.get('verbose', False))

        if not ctx.obj.get('quiet'):
            click.echo(f"Starting batch parsing: {input_dir} -> {output_dir}")
            click.echo(f"Pattern: {pattern}, Recursive: {recursive}, Workers: {max_workers}")

        # Parse files in batch
        results = parser.parse_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            format=format,
            pattern=pattern,
            recursive=recursive,
            max_workers=max_workers,
            continue_on_error=True,
            **parse_options
        )

        # Summary is already shown by ParserCLI in verbose mode
        if not ctx.obj.get('verbose') and not ctx.obj.get('quiet'):
            successful = len([r for r in results if r.get('success', False)])
            failed = len(results) - successful
            click.echo(f"\n📊 Batch processing completed!")
            click.echo(f"   ✅ Successful: {successful}")
            click.echo(f"   ❌ Failed: {failed}")
            click.echo(f"   📁 Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Batch parsing failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@click.command()
@click.pass_context
def list_formats(ctx):
    """List all supported file formats and their capabilities."""
    try:
        parser = ParserCLI()
        formats = parser.list_supported_formats()

        click.echo("📄 Supported File Formats:")
        click.echo("=" * 60)

        # Group formats by category
        categories = {
            'Documents': ['.pdf', '.docx', '.doc', '.wps', '.epub', '.md'],
            'Spreadsheets': ['.xlsx', '.xls', '.csv'],
            'Presentations': ['.pptx', '.ppt'],
            'Web': ['.html'],
            'Text': ['.txt'],
            'Images': ['.jpg', '.jpeg', '.png', '.webp'],
            'Code': ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs']
        }

        for category, extensions in categories.items():
            click.echo(f"\n🔧 {category}:")
            for ext in extensions:
                if ext in formats:
                    click.echo(f"   {ext:<8} {formats[ext]}")

        click.echo("\n💡 Usage Examples:")
        click.echo("   datamax parser parse document.pdf")
        click.echo("   datamax parser parse image.jpg --use-mllm --api-key $OPENAI_API_KEY")
        click.echo("   datamax parser batch ./docs ./output --use-mineru")
        click.echo("\n🔗 For advanced options, use:")
        click.echo("   datamax parser parse --help")

    except Exception as e:
        logger.error(f"Failed to list formats: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Export commands for use in main CLI
crawler_command = crawler
arxiv_command = arxiv
web_command = web
list_crawlers_command = list_crawlers
clean_command = clean
list_cleaners_command = list_cleaners
qa_command = qa
multimodal_command = multimodal
list_generators_command = list_generators
parse_command = parse
batch_command = batch
list_formats_command = list_formats
