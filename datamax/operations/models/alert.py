from __future__ import annotations

import uuid

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampedMixin


class Alert(TimestampedMixin, Base):
    __tablename__ = "alerts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    severity: Mapped[str] = mapped_column(String(32), default="info")
    title: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(String(32), default="system")
    source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("pipelines.id", ondelete="SET NULL"))
    acknowledged: Mapped[bool] = mapped_column(default=False)
