"""DataMax CLI Module

Command-line interface for DataMax with crawler functionality.
"""

from .main import main
from .crawler_cli import CrawlerCLI
from .commands import (
    crawler_command,
    arxiv_command,
    web_command,
    list_crawlers_command
)

__all__ = [
    'main',
    'CrawlerCLI',
    'crawler_command',
    'arxiv_command', 
    'web_command',
    'list_crawlers_command'
]