"""Crawler Configuration Manager

Provides configuration management for DataMax crawlers.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from .exceptions import ConfigurationException


class CrawlerConfig:
    """Configuration manager for crawlers.
    
    Handles loading and accessing crawler configurations from YAML files.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default locations.
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            ConfigurationException: If configuration cannot be loaded
        """
        config_paths = self._get_config_paths()
        
        for path in config_paths:
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        return yaml.safe_load(f) or {}
                except Exception as e:
                    raise ConfigurationException(
                        f"Failed to load configuration from {path}: {str(e)}"
                    ) from e
        
        # Return default configuration if no config file found
        return self._get_default_config()
    
    def _get_config_paths(self) -> list[Path]:
        """Get list of possible configuration file paths.
        
        Returns:
            List of Path objects to check for configuration files
        """
        if self.config_path:
            return [Path(self.config_path)]
        
        # Default configuration file locations
        current_dir = Path.cwd()
        return [
            current_dir / "config" / "crawler_config.yaml",
            current_dir / "crawler_config.yaml",
            Path.home() / ".datamax" / "crawler_config.yaml",
        ]
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration.
        
        Returns:
            Default configuration dictionary
        """
        return {
            "crawlers": {
                "arxiv": {
                    "base_url": "https://arxiv.org/",
                    "rate_limit": 1.0,
                    "timeout": 30,
                    "max_retries": 3,
                    "user_agent": "DataMax-Crawler/1.0"
                },
                "web": {
                    "user_agent": "DataMax-Crawler/1.0",
                    "timeout": 15,
                    "max_retries": 2,
                    "rate_limit": 0.5
                }
            },
            "storage": {
                "default_format": "json",
                "output_dir": "./output",
                "cloud_storage": {
                    "enabled": False,
                    "provider": "s3"
                }
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        }
    
    def get_crawler_config(self, crawler_type: str) -> Dict[str, Any]:
        """Get configuration for a specific crawler type.
        
        Args:
            crawler_type: Type of crawler (e.g., 'arxiv', 'web')
            
        Returns:
            Configuration dictionary for the specified crawler
        """
        crawlers_config = self.config.get('crawlers', {})
        return crawlers_config.get(crawler_type, {})
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration.
        
        Returns:
            Storage configuration dictionary
        """
        return self.config.get('storage', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration.
        
        Returns:
            Logging configuration dictionary
        """
        return self.config.get('logging', {})
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by key.
        
        Args:
            key: Configuration key (supports dot notation, e.g., 'crawlers.arxiv.timeout')
            default: Default value if key is not found
            
        Returns:
            Configuration value or default
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update_config(self, key: str, value: Any):
        """Update a configuration value.
        
        Args:
            key: Configuration key (supports dot notation)
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self, path: Optional[str] = None):
        """Save current configuration to file.
        
        Args:
            path: Path to save the configuration. If None, uses the original config path.
            
        Raises:
            ConfigurationException: If configuration cannot be saved
        """
        save_path = Path(path) if path else self._get_config_paths()[0]
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
        except Exception as e:
            raise ConfigurationException(
                f"Failed to save configuration to {save_path}: {str(e)}"
            ) from e


# Global configuration instance
_config_instance: Optional[CrawlerConfig] = None


def get_config() -> CrawlerConfig:
    """Get the global configuration instance.
    
    Returns:
        Global CrawlerConfig instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = CrawlerConfig()
    return _config_instance


def set_config(config: CrawlerConfig):
    """Set the global configuration instance.
    
    Args:
        config: CrawlerConfig instance to set as global
    """
    global _config_instance
    _config_instance = config