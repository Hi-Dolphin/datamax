"""DataMax CLI Module

Command-line interface for DataMax with crawler functionality.
"""

from .cleaner_cli import CleanerCLI
from .commands import (
    arxiv_command,
    batch_command,
    clean_command,
    crawler_command,
    list_cleaners_command,
    list_crawlers_command,
    list_formats_command,
    list_generators_command,
    multimodal_command,
    parse_command,
    qa_command,
    web_command,
)
from .crawler_cli import CrawlerCLI
from .generator_cli import GeneratorCLI
from .main import main
from .parser_cli import ParserCLI

__all__ = [
    "main",
    "CrawlerCLI",
    "CleanerCLI",
    "GeneratorCLI",
    "ParserCLI",
    "crawler_command",
    "arxiv_command",
    "web_command",
    "list_crawlers_command",
    "clean_command",
    "list_cleaners_command",
    "qa_command",
    "multimodal_command",
    "list_generators_command",
    "parse_command",
    "batch_command",
    "list_formats_command",
]
