from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.types import JSON
from sqlalchemy.ext.mutable import MutableList
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampedMixin
from .datasource import DataSource


class Dataset(TimestampedMixin, Base):
    __tablename__ = "datasets"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    datasource_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    schema_version: Mapped[str] = mapped_column(String(32), default="v1")
    record_count: Mapped[int] = mapped_column(Integer, default=0)
    format: Mapped[str] = mapped_column(String(32), default="parquet")
    tags: Mapped[list[str]] = mapped_column(MutableList.as_mutable(JSON), default=list)
    last_refreshed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[str] = mapped_column(String(32), default="ready")

    datasource: Mapped[DataSource] = relationship(backref="datasets", lazy="joined")
