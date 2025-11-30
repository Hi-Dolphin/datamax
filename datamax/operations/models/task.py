from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text
from sqlalchemy.ext.mutable import MutableDict
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from .base import Base, TimestampedMixin
from .dataset import Dataset
from .pipeline import Pipeline

_JSON = JSON


class Task(TimestampedMixin, Base):
    __tablename__ = "tasks"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipelines.id", ondelete="SET NULL"))
    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("datasets.id", ondelete="SET NULL")
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    scheduled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    parameters: Mapped[dict] = mapped_column(MutableDict.as_mutable(_JSON), default=dict)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    owner: Mapped[str | None] = mapped_column(String(64))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    runs: Mapped[list["TaskRun"]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="TaskRun.created_at.desc()"
    )
    pipeline: Mapped[Pipeline | None] = relationship("Pipeline", backref="tasks")
    dataset: Mapped[Dataset | None] = relationship("Dataset", backref="tasks")


class TaskRun(TimestampedMixin, Base):
    __tablename__ = "task_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    task_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("tasks.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(32), default="pending")
    duration_seconds: Mapped[float | None] = mapped_column()
    tokens_total: Mapped[int] = mapped_column(default=0)
    tokens_prompt: Mapped[int] = mapped_column(default=0)
    tokens_completion: Mapped[int] = mapped_column(default=0)
    requests: Mapped[int] = mapped_column(default=0)
    metrics: Mapped[dict] = mapped_column(MutableDict.as_mutable(_JSON), default=dict)
    error_message: Mapped[str | None] = mapped_column(Text)
    triggered_by: Mapped[str | None] = mapped_column(String(64))

    task: Mapped[Task] = relationship(back_populates="runs")
