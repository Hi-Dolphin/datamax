from __future__ import annotations

import sys

from loguru import logger

from .config import AppSettings


def configure_logging(settings: AppSettings) -> None:
    logger.remove()
    logger.add(
        sys.stdout,
        level=settings.logging.level.upper(),
        format=settings.logging.format,
        serialize=settings.logging.structured,
    )
