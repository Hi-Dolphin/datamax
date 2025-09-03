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
    list_crawlers_command,
    clean_command,
    list_cleaners_command,
    qa_command,
    multimodal_command,
    list_generators_command,
    parse_command,
    batch_command,
    list_formats_command
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

        try:
            from datamax.cleaner import AbnormalCleaner, TextFilter, PrivacyDesensitization
            click.echo("✅ Cleaner module: Available")
            click.echo("   - Abnormal text cleaning")
            click.echo("   - Content quality filtering")
            click.echo("   - Privacy data desensitization")
        except ImportError as e:
            click.echo(f"❌ Cleaner module: Error - {e}")
        
        # Show lifecycle types
        click.echo("\nAvailable Lifecycle Types:")
        for life_type in LifeType:
            click.echo(f"  - {life_type.value}")
        
        click.echo("\n✅ DataMax is ready to use!")

        # Check generator components
        try:
            from datamax.cli.generator_cli import GeneratorCLI
            click.echo("✅ Generator module: Available")
            generator_cli = GeneratorCLI()
            generators = generator_cli.list_generators()
            if generators:
                click.echo(f"   Available generators: {', '.join(generators.keys())}")
        except ImportError as e:
            click.echo(f"❌ Generator module: Error - {e}")

        # Check parser CLI components
        try:
            from datamax.cli.parser_cli import ParserCLI
            click.echo("✅ Parser CLI: Available")
            parser_cli = ParserCLI()
            formats = parser_cli.list_supported_formats()
            if formats:
                click.echo(f"   Supported formats: {len(formats)} file types")
        except ImportError as e:
            click.echo(f"❌ Parser CLI: Error - {e}")

    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        if ctx.obj.get('verbose'):
            import traceback
            traceback.print_exc()
        sys.exit(1)


# Generator command group
@cli.group()
def generator():
    """Generator commands for QA pair generation."""
    pass


# Parser command group
@cli.group()
def parser():
    """Parser commands for document parsing and conversion."""
    pass


# Add generator subcommands
generator.add_command(qa_command, name='qa')
generator.add_command(multimodal_command, name='multimodal')
generator.add_command(list_generators_command, name='list')

# Add parser subcommands
parser.add_command(parse_command, name='parse')
parser.add_command(batch_command, name='batch')
parser.add_command(list_formats_command, name='list-formats')

# Add crawler commands
cli.add_command(crawler_command)
cli.add_command(arxiv_command)
cli.add_command(web_command)
cli.add_command(list_crawlers_command)

# Add cleaning command
cli.add_command(clean_command)
cli.add_command(list_cleaners_command)

# Add generator command group
cli.add_command(generator)

# Add parser command group
cli.add_command(parser)


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
