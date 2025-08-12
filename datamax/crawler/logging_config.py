"""Logging Configuration for Crawler Module

Provides centralized logging configuration and monitoring utilities.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from loguru import logger


class CrawlerLogger:
    """Centralized logger for crawler operations.
    
    Provides structured logging with different levels and output formats.
    """
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_file: Optional[str] = None,
                 enable_json: bool = False,
                 enable_console: bool = True):
        """Initialize crawler logger.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_file: Path to log file (optional)
            enable_json: Enable JSON formatted logging
            enable_console: Enable console logging
        """
        self.log_level = log_level.upper()
        self.log_file = log_file
        self.enable_json = enable_json
        self.enable_console = enable_console
        
        # Remove default logger
        logger.remove()
        
        # Configure loggers
        self._setup_console_logger()
        self._setup_file_logger()
    
    def _setup_console_logger(self):
        """Setup console logger."""
        if not self.enable_console:
            return
        
        if self.enable_json:
            # JSON format for structured logging
            logger.add(
                sys.stderr,
                level=self.log_level,
                format="{message}",
                serialize=True
            )
        else:
            # Human-readable format
            logger.add(
                sys.stderr,
                level=self.log_level,
                format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                       "<level>{level: <8}</level> | "
                       "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                       "<level>{message}</level>",
                colorize=True
            )
    
    def _setup_file_logger(self):
        """Setup file logger."""
        if not self.log_file:
            return
        
        log_path = Path(self.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.enable_json:
            # JSON format for file logging
            logger.add(
                str(log_path),
                level=self.log_level,
                format="{message}",
                serialize=True,
                rotation="10 MB",
                retention="30 days",
                compression="gz"
            )
        else:
            # Standard format for file logging
            logger.add(
                str(log_path),
                level=self.log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation="10 MB",
                retention="30 days",
                compression="gz"
            )
    
    @staticmethod
    def log_crawler_start(crawler_type: str, target: str, **kwargs):
        """Log crawler start event.
        
        Args:
            crawler_type: Type of crawler
            target: Crawling target
            **kwargs: Additional parameters
        """
        logger.info(
            "Crawler started",
            extra={
                "event": "crawler_start",
                "crawler_type": crawler_type,
                "target": target,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
        )
    
    @staticmethod
    def log_crawler_success(crawler_type: str, target: str, result_size: int, duration: float, **kwargs):
        """Log successful crawler completion.
        
        Args:
            crawler_type: Type of crawler
            target: Crawling target
            result_size: Size of crawled data
            duration: Crawling duration in seconds
            **kwargs: Additional parameters
        """
        logger.info(
            "Crawler completed successfully",
            extra={
                "event": "crawler_success",
                "crawler_type": crawler_type,
                "target": target,
                "result_size": result_size,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
        )
    
    @staticmethod
    def log_crawler_error(crawler_type: str, target: str, error: str, duration: float, **kwargs):
        """Log crawler error.
        
        Args:
            crawler_type: Type of crawler
            target: Crawling target
            error: Error message
            duration: Duration before error in seconds
            **kwargs: Additional parameters
        """
        logger.error(
            "Crawler failed",
            extra={
                "event": "crawler_error",
                "crawler_type": crawler_type,
                "target": target,
                "error": error,
                "duration_seconds": duration,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
        )
    
    @staticmethod
    def log_rate_limit(crawler_type: str, target: str, delay: float, **kwargs):
        """Log rate limiting event.
        
        Args:
            crawler_type: Type of crawler
            target: Crawling target
            delay: Rate limit delay in seconds
            **kwargs: Additional parameters
        """
        logger.debug(
            "Rate limit applied",
            extra={
                "event": "rate_limit",
                "crawler_type": crawler_type,
                "target": target,
                "delay_seconds": delay,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
        )
    
    @staticmethod
    def log_retry_attempt(crawler_type: str, target: str, attempt: int, max_attempts: int, error: str, **kwargs):
        """Log retry attempt.
        
        Args:
            crawler_type: Type of crawler
            target: Crawling target
            attempt: Current attempt number
            max_attempts: Maximum number of attempts
            error: Error that triggered retry
            **kwargs: Additional parameters
        """
        logger.warning(
            "Retrying crawler operation",
            extra={
                "event": "retry_attempt",
                "crawler_type": crawler_type,
                "target": target,
                "attempt": attempt,
                "max_attempts": max_attempts,
                "error": error,
                "timestamp": datetime.now().isoformat(),
                **kwargs
            }
        )


class CrawlerMetrics:
    """Metrics collection for crawler operations.
    
    Tracks performance and usage statistics.
    """
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics = {
            'total_crawls': 0,
            'successful_crawls': 0,
            'failed_crawls': 0,
            'total_duration': 0.0,
            'avg_duration': 0.0,
            'crawlers_used': {},
            'error_types': {},
            'targets_crawled': set(),
            'start_time': datetime.now().isoformat()
        }
    
    def record_crawl_start(self, crawler_type: str, target: str):
        """Record crawl start.
        
        Args:
            crawler_type: Type of crawler
            target: Crawling target
        """
        self.metrics['total_crawls'] += 1
        self.metrics['crawlers_used'][crawler_type] = self.metrics['crawlers_used'].get(crawler_type, 0) + 1
        self.metrics['targets_crawled'].add(target)
    
    def record_crawl_success(self, crawler_type: str, target: str, duration: float):
        """Record successful crawl.
        
        Args:
            crawler_type: Type of crawler
            target: Crawling target
            duration: Crawling duration in seconds
        """
        self.metrics['successful_crawls'] += 1
        self.metrics['total_duration'] += duration
        self._update_avg_duration()
    
    def record_crawl_failure(self, crawler_type: str, target: str, error_type: str, duration: float):
        """Record failed crawl.
        
        Args:
            crawler_type: Type of crawler
            target: Crawling target
            error_type: Type of error
            duration: Duration before failure in seconds
        """
        self.metrics['failed_crawls'] += 1
        self.metrics['total_duration'] += duration
        self.metrics['error_types'][error_type] = self.metrics['error_types'].get(error_type, 0) + 1
        self._update_avg_duration()
    
    def _update_avg_duration(self):
        """Update average duration."""
        if self.metrics['total_crawls'] > 0:
            self.metrics['avg_duration'] = self.metrics['total_duration'] / self.metrics['total_crawls']
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Metrics dictionary
        """
        # Convert set to list for JSON serialization
        metrics_copy = self.metrics.copy()
        metrics_copy['targets_crawled'] = list(self.metrics['targets_crawled'])
        metrics_copy['unique_targets'] = len(self.metrics['targets_crawled'])
        
        # Calculate success rate
        if self.metrics['total_crawls'] > 0:
            metrics_copy['success_rate'] = self.metrics['successful_crawls'] / self.metrics['total_crawls']
        else:
            metrics_copy['success_rate'] = 0.0
        
        return metrics_copy
    
    def save_metrics(self, file_path: str):
        """Save metrics to file.
        
        Args:
            file_path: Path to save metrics
        """
        metrics_data = self.get_metrics()
        metrics_data['saved_at'] = datetime.now().isoformat()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.__init__()


# Global instances
_crawler_logger = None
_crawler_metrics = CrawlerMetrics()


def setup_crawler_logging(log_level: str = "INFO",
                         log_file: Optional[str] = None,
                         enable_json: bool = False,
                         enable_console: bool = True) -> CrawlerLogger:
    """Setup global crawler logging.
    
    Args:
        log_level: Logging level
        log_file: Path to log file
        enable_json: Enable JSON logging
        enable_console: Enable console logging
        
    Returns:
        Configured CrawlerLogger instance
    """
    global _crawler_logger
    _crawler_logger = CrawlerLogger(
        log_level=log_level,
        log_file=log_file,
        enable_json=enable_json,
        enable_console=enable_console
    )
    return _crawler_logger


def get_crawler_logger() -> Optional[CrawlerLogger]:
    """Get global crawler logger.
    
    Returns:
        CrawlerLogger instance or None if not setup
    """
    return _crawler_logger


def get_crawler_metrics() -> CrawlerMetrics:
    """Get global crawler metrics.
    
    Returns:
        CrawlerMetrics instance
    """
    return _crawler_metrics