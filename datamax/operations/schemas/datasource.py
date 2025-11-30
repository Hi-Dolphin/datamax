from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from .base import APIModel, ResourceBase


class DataSourceConnection(APIModel):
    type: Literal["postgres", "mysql", "s3", "http", "local", "other"] = "postgres"
    host: Optional[str] = None
    port: Optional[int] = None
    database: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = Field(default=None, repr=False)
    extra: dict[str, Any] | None = None


class DataSourceBase(APIModel):
    name: str
    kind: Literal["database", "object_storage", "file", "api", "other"] = "database"
    description: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    connection: DataSourceConnection | dict[str, Any]
    is_active: bool = True


class DataSourceCreate(DataSourceBase):
    pass


class DataSourceUpdate(APIModel):
    name: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    connection: Optional[DataSourceConnection | dict[str, Any]] = None
    is_active: Optional[bool] = None


class DataSource(ResourceBase, DataSourceBase):
    status: Literal["active", "inactive", "error", "unknown"] = "active"
    health_message: Optional[str] = None
