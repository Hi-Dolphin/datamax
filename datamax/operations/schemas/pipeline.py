from __future__ import annotations

from typing import Literal, Optional
from uuid import UUID

from pydantic import Field

from .base import APIModel, ResourceBase


class PipelineStage(APIModel):
    stage_id: UUID | None = None
    name: str
    kind: str
    order: int
    config: dict[str, object] = Field(default_factory=dict)
    is_optional: bool = False
    stats: dict[str, object] = Field(default_factory=dict)


class PipelineBase(APIModel):
    name: str
    description: Optional[str] = None
    project: Optional[str] = None
    stages: list[PipelineStage] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class PipelineCreate(PipelineBase):
    pass


class PipelineUpdate(APIModel):
    name: Optional[str] = None
    description: Optional[str] = None
    project: Optional[str] = None
    stages: Optional[list[PipelineStage]] = None
    tags: Optional[list[str]] = None


class Pipeline(ResourceBase, PipelineBase):
    status: Literal["draft", "active", "disabled", "archived"] = "draft"
