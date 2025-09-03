"""DataMax CLI Module

Command-line interface for DataMax with crawler functionality.
"""

from .main import main
from .crawler_cli import CrawlerCLI
from .cleaner_cli import CleanerCLI
from .generator_cli import GeneratorCLI
from .parser_cli import ParserCLI
from .commands import (
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

__all__ = [
    'main',
    'CrawlerCLI',
    'CleanerCLI',
    'GeneratorCLI',
    'ParserCLI',
    'crawler_command',
    'arxiv_command',
    'web_command',
    'list_crawlers_command',
    'clean_command',
    'list_cleaners_command',
    'qa_command',
    'multimodal_command',
    'list_generators_command',
    'parse_command',
    'batch_command',
    'list_formats_command'
]
