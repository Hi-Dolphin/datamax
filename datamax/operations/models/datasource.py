from __future__ import annotations

import uuid

from sqlalchemy import String, Text
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .base import Base, TimestampedMixin


_JSON = JSON


class DataSource(TimestampedMixin, Base):
    __tablename__ = "data_sources"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="database")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    tags: Mapped[list[str]] = mapped_column(MutableList.as_mutable(_JSON), default=list)
    connection: Mapped[dict] = mapped_column(MutableDict.as_mutable(_JSON), default=dict)
    is_active: Mapped[bool] = mapped_column(default=True)
    status: Mapped[str] = mapped_column(String(32), default="unknown")
    health_message: Mapped[str | None] = mapped_column(Text, nullable=True)
