"""Configuration management for DataMax SDK."""

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from datamax.exceptions import ConfigurationError


class ParseConfig(BaseModel):
    """Configuration for parsing operations."""

    use_mineru: bool = Field(default=False, description="Whether to use MinerU for PDF parsing")
    to_markdown: bool = Field(default=False, description="Convert to markdown format")
    ttl_cache: int = Field(default=3600, description="Cache TTL in seconds")
    chunk_size: int = Field(default=500, description="Text chunking size")
    chunk_overlap: int = Field(default=100, description="Text chunking overlap")


class AIConfig(BaseModel):
    """Configuration for AI operations."""

    api_key: str = Field(description="OpenAI-compatible API key")
    base_url: str = Field(description="API base URL")
    model_name: str = Field(default="gpt-3.5-turbo", description="Model name")
    max_workers: int = Field(default=5, description="Max concurrent workers")
    question_number: int = Field(default=5, description="Questions per chunk")
    language: str = Field(default="zh", description="Language for QA generation")


class StorageConfig(BaseModel):
    """Configuration for cloud storage."""

    provider: str = Field(description="Storage provider (oss/minio)")
    endpoint: str = Field(description="Storage endpoint")
    access_key: str = Field(description="Access key")
    secret_key: str = Field(description="Secret key")
    bucket_name: str = Field(description="Bucket name")


class DataMaxSettings(BaseSettings):
    """Main settings for DataMax SDK."""

    model_config = SettingsConfigDict(
        env_prefix="DATAMAX_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Global settings
    domain: str = Field(default="Technology", description="Default domain")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Parse settings
    parse: ParseConfig = Field(default_factory=ParseConfig)
    
    # AI settings (optional)
    ai: AIConfig | None = None
    
    # Storage settings (optional)
    storage: StorageConfig | None = None

    @classmethod
    def from_file(cls, config_path: str | Path) -> "DataMaxSettings":
        """Load settings from a config file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        if config_path.suffix == ".json":
            import json
            with open(config_path, encoding="utf-8") as f:
                data = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            import yaml
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        else:
            raise ConfigurationError(f"Unsupported config file format: {config_path.suffix}")
        
        return cls(**data)

    def validate_ai_config(self) -> None:
        """Validate AI configuration."""
        if self.ai is None:
            raise ConfigurationError("AI configuration is required for annotation operations")
        
        if not self.ai.api_key:
            raise ConfigurationError("AI API key is required")
        
        if not self.ai.base_url:
            raise ConfigurationError("AI base URL is required")

    def validate_storage_config(self) -> None:
        """Validate storage configuration."""
        if self.storage is None:
            raise ConfigurationError("Storage configuration is required for cloud operations")
        
        required_fields = ["provider", "endpoint", "access_key", "secret_key", "bucket_name"]
        for field in required_fields:
            if not getattr(self.storage, field):
                raise ConfigurationError(f"Storage {field} is required")


# Global settings instance
_settings: DataMaxSettings | None = None


def get_settings() -> DataMaxSettings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = DataMaxSettings()
    return _settings


def configure(
    config_path: str | Path | None = None,
    **kwargs: Any,
) -> DataMaxSettings:
    """Configure DataMax with settings."""
    global _settings
    
    if config_path:
        _settings = DataMaxSettings.from_file(config_path)
    else:
        _settings = DataMaxSettings(**kwargs)
    
    return _settings


def reset_config() -> None:
    """Reset configuration to default."""
    global _settings
    _settings = None