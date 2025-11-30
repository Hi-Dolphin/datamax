from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from .base import Base, TimestampedMixin

_JSON = JSON


class Pipeline(TimestampedMixin, Base):
    __tablename__ = "pipelines"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    project: Mapped[str | None] = mapped_column(String(64))
    tags: Mapped[list[str]] = mapped_column(MutableList.as_mutable(_JSON), default=list)
    status: Mapped[str] = mapped_column(String(32), default="draft")

    stages: Mapped[list["PipelineStage"]] = relationship(
        back_populates="pipeline", cascade="all, delete-orphan", order_by="PipelineStage.order"
    )


class PipelineStage(TimestampedMixin, Base):
    __tablename__ = "pipeline_stages"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    pipeline_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("pipelines.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    kind: Mapped[str] = mapped_column(String(64), nullable=False)
    order: Mapped[int] = mapped_column(nullable=False)
    config: Mapped[dict] = mapped_column(MutableDict.as_mutable(_JSON), default=dict)
    is_optional: Mapped[bool] = mapped_column(default=False)
    stats: Mapped[dict] = mapped_column(MutableDict.as_mutable(_JSON), default=dict)

    pipeline: Mapped[Pipeline] = relationship(back_populates="stages")
