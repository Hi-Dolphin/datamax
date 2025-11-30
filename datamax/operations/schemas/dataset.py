from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import Field

from .base import APIModel, ResourceBase


class DatasetBase(APIModel):
    name: str
    datasource_id: UUID
    description: Optional[str] = None
    schema_version: str = "v1"
    record_count: int = 0
    format: Literal["parquet", "csv", "jsonl", "delta", "custom"] = "parquet"
    last_refreshed_at: Optional[datetime] = None
    tags: list[str] = Field(default_factory=list)


class DatasetCreate(DatasetBase):
    pass


class DatasetUpdate(APIModel):
    name: Optional[str] = None
    description: Optional[str] = None
    schema_version: Optional[str] = None
    record_count: Optional[int] = None
    format: Optional[Literal["parquet", "csv", "jsonl", "delta", "custom"]] = None
    tags: Optional[list[str]] = None
    last_refreshed_at: Optional[datetime] = None


class Dataset(ResourceBase, DatasetBase):
    status: Literal["ready", "processing", "failed", "archived"] = "ready"
