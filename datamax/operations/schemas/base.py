from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=timezone.utc)


class APIModel(BaseModel):
    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
        "json_encoders": {datetime: lambda dt: dt.isoformat()},
    }


class ResourceBase(APIModel):
    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        object.__setattr__(self, "updated_at", utc_now())


class PaginatedResponse(APIModel):
    items: list[Any]
    total: int
    page: int = 1
    size: int = 50
