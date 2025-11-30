from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import Field

from .base import APIModel, ResourceBase


class MetricPoint(APIModel):
    timestamp: str
    value: float


class MetricSeries(APIModel):
    name: str
    points: list[MetricPoint] = Field(default_factory=list)


class MetricSummary(ResourceBase):
    scope: str
    scope_id: Optional[UUID] = None
    window: str = "1h"
    totals: dict[str, float] = Field(default_factory=dict)
    series: list[MetricSeries] = Field(default_factory=list)


class DashboardTotals(APIModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_requests: int = 0
    average_duration: float = 0.0


class StageMetrics(APIModel):
    stage: str
    runs: int = 0
    items: int = 0
    duration: float = 0.0
    reqs: int = 0
    qpm: float = 0.0
    total_tokens: int = 0
    tokens_per_sec: float = 0.0
    tokens_per_req: float = 0.0


class SeriesPoint(APIModel):
    label: str | int
    value: float


class TokenSplit(APIModel):
    label: str
    value: int
    percentage: float


class DashboardPayload(APIModel):
    totals: DashboardTotals
    stages: list[StageMetrics] = Field(default_factory=list)
    throughput_series: list[SeriesPoint] = Field(default_factory=list)
    history_series: list[SeriesPoint] = Field(default_factory=list)
    token_split: list[TokenSplit] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.utcnow)
