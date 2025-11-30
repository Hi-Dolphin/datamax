from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve project root so settings always load the shared .env/config files.
_CONFIG_RESOLVED_PATH = Path(__file__).resolve()
ROOT_PATH = (
    _CONFIG_RESOLVED_PATH.parents[3]
    if len(_CONFIG_RESOLVED_PATH.parents) >= 4
    else Path.cwd()
)
DEFAULT_CONFIG_PATH = ROOT_PATH / "config" / "config.json"
ENV_FILE_PATH = ROOT_PATH / ".env"


class LoggingConfig(BaseModel):
    level: str = "INFO"
    structured: bool = False
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DATAMAX_",
        env_file=ENV_FILE_PATH,
        extra="ignore",
    )

    project_name: str = "Data Factory Operations API"
    environment: str = "development"
    api_v1_prefix: str = "/api/v1"
    allow_origins: list[str] = Field(default_factory=lambda: ["*"])
    postgres_dsn: str | None = "sqlite+aiosqlite:///./operations.db"
    redis_url: str | None = None
    mock_mode: bool = False
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_file(cls, path: Path | None = None) -> "AppSettings":
        path = path or DEFAULT_CONFIG_PATH
        if path and path.exists():
            data = cls.model_validate_json(path.read_text(encoding="utf-8"))
            return data
        return cls()


@lru_cache
def get_settings(config_path: str | None = None) -> AppSettings:
    """Load settings from env or optional config file."""
    if config_path:
        path = Path(config_path)
        if path.exists():
            return AppSettings.from_file(path)
    if DEFAULT_CONFIG_PATH.exists():
        return AppSettings.from_file(DEFAULT_CONFIG_PATH)
    return AppSettings()
