from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import Field

from .base import APIModel, ResourceBase


TaskStatus = Literal["pending", "running", "succeeded", "failed", "cancelled"]


class TaskBase(APIModel):
    pipeline_id: UUID
    dataset_id: Optional[UUID] = None
    name: str
    description: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    parameters: dict[str, object] = Field(default_factory=dict)


class TaskCreate(TaskBase):
    pass


class TaskUpdate(APIModel):
    name: Optional[str] = None
    description: Optional[str] = None
    scheduled_at: Optional[datetime] = None
    parameters: Optional[dict[str, object]] = None
    status: Optional[TaskStatus] = None


class Task(ResourceBase, TaskBase):
    status: TaskStatus = "pending"
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    owner: Optional[str] = None


class TaskRun(ResourceBase):
    task_id: UUID
    status: TaskStatus = "pending"
    duration_seconds: Optional[float] = None
    tokens_total: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    requests: int = 0
    metrics: dict[str, float] = Field(default_factory=dict)
    error_message: Optional[str] = None
    triggered_by: Optional[str] = None
