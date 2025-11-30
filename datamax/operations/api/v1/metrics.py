from __future__ import annotations

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from ...api.deps import get_db_session
from ...schemas.metrics import DashboardPayload, MetricSummary
from ...services.metrics import compute_dashboard_payload

router = APIRouter(prefix="/metrics")


@router.get("", response_model=List[MetricSummary])
async def list_metrics(session: AsyncSession = Depends(get_db_session)) -> List[MetricSummary]:
    payload = await compute_dashboard_payload(session)
    summary = MetricSummary(
        scope="dashboard",
        totals={
            "tokens_total": payload.totals.total_tokens,
            "tokens_prompt": payload.totals.prompt_tokens,
            "tokens_completion": payload.totals.completion_tokens,
            "requests": payload.totals.total_requests,
            "avg_duration": payload.totals.average_duration,
        },
        series=[],
    )
    return [summary]


@router.get(
    "/dashboard",
    response_model=DashboardPayload,
    status_code=status.HTTP_200_OK,
)
async def dashboard_metrics(session: AsyncSession = Depends(get_db_session)) -> DashboardPayload:
    return await compute_dashboard_payload(session)


@router.get("/{metric_key}", response_model=MetricSummary)
async def get_metric(metric_key: str, session: AsyncSession = Depends(get_db_session)) -> MetricSummary:
    if metric_key != "dashboard":
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metric summary not found")
    payload = await compute_dashboard_payload(session)
    return MetricSummary(
        scope="dashboard",
        totals={
            "tokens_total": payload.totals.total_tokens,
            "tokens_prompt": payload.totals.prompt_tokens,
            "tokens_completion": payload.totals.completion_tokens,
            "requests": payload.totals.total_requests,
            "avg_duration": payload.totals.average_duration,
        },
        series=[],
    )
