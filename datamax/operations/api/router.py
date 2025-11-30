from __future__ import annotations

from fastapi import APIRouter

from .v1 import datasources, datasets, pipelines, tasks, metrics, alerts, system


def get_api_router() -> APIRouter:
    router = APIRouter()
    router.include_router(datasources.router, tags=["datasources"])
    router.include_router(datasets.router, tags=["datasets"])
    router.include_router(pipelines.router, tags=["pipelines"])
    router.include_router(tasks.router, tags=["tasks"])
    router.include_router(metrics.router, tags=["metrics"])
    router.include_router(alerts.router, tags=["alerts"])
    router.include_router(system.router, tags=["system"])
    return router
