from __future__ import annotations

from typing import Optional

from pydantic import Field

from .base import APIModel


class NotificationChannel(APIModel):
    kind: str
    target: str
    enabled: bool = True
    config: dict[str, object] = Field(default_factory=dict)


class SystemConfig(APIModel):
    realtime_refresh_seconds: int = 60
    history_refresh_minutes: int = 5
    grpc_timeout_seconds: int = 30
    enable_websocket: bool = False
    default_time_window: str = "7d"
    alert_thresholds: dict[str, float] = Field(default_factory=dict)
    notification_channels: list[NotificationChannel] = Field(default_factory=list)
