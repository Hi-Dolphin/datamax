from __future__ import annotations

from typing import Literal, Optional
from uuid import UUID

from .base import ResourceBase


class Alert(ResourceBase):
    severity: Literal["info", "warning", "critical"] = "info"
    title: str
    description: Optional[str] = None
    source: Literal["pipeline", "task", "system"] = "system"
    source_id: Optional[UUID] = None
    acknowledged: bool = False
