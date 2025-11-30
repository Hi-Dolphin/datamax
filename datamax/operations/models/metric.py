from __future__ import annotations

import uuid

from sqlalchemy import String
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from .base import Base, TimestampedMixin

_JSON = JSON


class MetricSummary(TimestampedMixin, Base):
    __tablename__ = "metric_summaries"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    scope: Mapped[str] = mapped_column(String(64), nullable=False)
    scope_id: Mapped[uuid.UUID | None] = mapped_column()
    window: Mapped[str] = mapped_column(String(16), default="1h")
    totals: Mapped[dict] = mapped_column(MutableDict.as_mutable(_JSON), default=dict)
    series: Mapped[list[dict]] = mapped_column(MutableList.as_mutable(_JSON), default=list)
