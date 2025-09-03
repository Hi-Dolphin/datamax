#!/usr/bin/env python3
"""DataMax CLI Main Entry Point

Main command-line interface for DataMax with integrated crawler functionality.
"""

import sys
import click
from pathlib import Path
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from datamax.cli.commands import (
    crawler_command,
    arxiv_command,
    web_command,
    list_crawlers_command
)
from datamax.utils.lifecycle_types import LifeType


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.option('--quiet', '-q', is_flag=True, help='Suppress output except errors')
@click.pass_context
def cli(ctx, verbose, quiet):
    """DataMax - Advanced Data Processing and Crawling Tool
    
    A comprehensive tool for data processing, parsing, and web crawling.
    """
    # Ensure context object exists
    ctx.ensure_object(dict)
    
    # Configure logging
    if quiet:
        logger.remove()
        logger.add(sys.stderr, level="ERROR")
    elif verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG", 
                  format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO",
                  format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
    
    ctx.obj['verbose'] = verbose
    ctx.obj['quiet'] = quiet


@cli.command()
@click.option('--input', '-i', required=True, help='Input file or directory path')
@click.option('--output', '-o', help='Output directory (default: ./output)')
@click.option('--domain', '-d', default='Technology', help='Domain category')
@click.option('--format', '-f', type=click.Choice(['markdown', 'json', 'yaml']), 
              default='markdown', help='Output format')
@click.pass_context
def parse(ctx, input, output, domain, format):
    """Parse data files using DataMax parser.
    
    Parse various data formats including crawler data, documents, and more.
    """
    try:
        from datamax.parser import DataMax, CrawlerParser
        
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"Error: Input path '{input}' does not exist.", err=True)
            sys.exit(1)
        
        output_dir = Path(output) if output else Path('./output')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not ctx.obj.get('quiet'):
            click.echo(f"Parsing {input_path} with domain '{domain}'...")
        
        # Determine parser type based on file content or extension
        if input_path.suffix.lower() == '.json':
            # Try to detect if it's crawler data
            try:
                import json
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if any(key in data for key in ['type', 'source', 'target', 'crawled_at']):
                    # Use crawler parser
                    parser = CrawlerParser(str(input_path), domain=domain)
                    result = parser.parse()
                    
                    output_file = output_dir / f"{input_path.stem}_parsed.md"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.content)
                    
                    if not ctx.obj.get('quiet'):
                        click.echo(f"Crawler data parsed successfully: {output_file}")
                    return
            except Exception:
                pass
        
        # Use default DataMax parser
        datamax = DataMax(str(input_path), domain=domain)
        result = datamax.parse()
        
        # Save result
        if format == 'markdown':
            output_file = output_dir / f"{input_path.stem}_parsed.md"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.content)
        elif format == 'json':
            output_file = output_dir / f"{input_path.stem}_parsed.json"
            import json
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'content': result.content,
                    'extension': result.extension,
                    'lifecycle': [lc.__dict__ for lc in result.lifecycle]
                }, f, indent=2, ensure_ascii=False)
        
        if not ctx.obj.get('quiet'):
            click.echo(f"Parsing completed successfully: {output_file}")
            
    except Exception as e:
        logger.error(f"Parsing failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument('target', required=False)
@click.option('--search', '-s', is_flag=True, help='Search the web for TARGET')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['json', 'yaml']), 
              default='json', help='Output format')
@click.pass_context
def web(ctx, target, search, output, format):
    """Crawl a web page or search the web.
    
    TARGET can be a URL to crawl or a search query.
    If TARGET is not provided, you'll be prompted to enter it.
    Use --search flag to explicitly treat TARGET as a search query.
    """
    # Import here to avoid circular imports
    from .commands import web as web_command
    
    # If no target provided, prompt user
    if not target:
        if search:
            target = click.prompt("Enter your search query")
        else:
            target = click.prompt("Enter URL to crawl or search query")
    
    # Call the web command with all parameters
    ctx.invoke(web_command, target=target, output=output, format=format, 
               extract_links=False, max_links=100, follow_redirects=False, 
               timeout=30, search=search)

@cli.command()
@click.pass_context
def status(ctx):
    """Show DataMax system status and available components."""
    try:
        click.echo("DataMax System Status")
        click.echo("=" * 50)
        
        # Check core components
        try:
            from datamax.parser import DataMax, CrawlerParser
            click.echo("✅ Parser module: Available")
        except ImportError as e:
            click.echo(f"❌ Parser module: Error - {e}")
        
        try:
            from datamax.crawler import CrawlerFactory, ArxivCrawler, WebCrawler
            click.echo("✅ Crawler module: Available")
            
            # Show registered crawlers
            factory = CrawlerFactory()
            crawlers = factory.list_crawlers()
            if crawlers:
                click.echo(f"   Registered crawlers: {', '.join(crawlers)}")
        except ImportError as e:
            click.echo(f"❌ Crawler module: Error - {e}")
        
        # Show lifecycle types
        click.echo("\nAvailable Lifecycle Types:")
        for life_type in LifeType:
            click.echo(f"  - {life_type.value}")
        
        click.echo("\n✅ DataMax is ready to use!")
        
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Add crawler commands
cli.add_command(crawler_command)
cli.add_command(arxiv_command)
cli.add_command(web_command)
cli.add_command(list_crawlers_command)


def main():
    """Main entry point for the CLI."""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nOperation cancelled by user.", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Unexpected error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()