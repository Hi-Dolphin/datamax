from __future__ import annotations

from fastapi import APIRouter, Depends

from ...core.config import AppSettings
from ...api.deps import get_settings_dep
from ...schemas.system_config import SystemConfig

router = APIRouter(prefix="/system")

_system_config_state: SystemConfig | None = None


@router.get("/config", response_model=SystemConfig)
def get_system_config(settings: AppSettings = Depends(get_settings_dep)) -> SystemConfig:
    global _system_config_state
    if _system_config_state is None:
        _system_config_state = SystemConfig(
            realtime_refresh_seconds=60,
            history_refresh_minutes=5,
            grpc_timeout_seconds=30,
            enable_websocket=settings.mock_mode is False,
        )
    return _system_config_state


@router.put("/config", response_model=SystemConfig)
def update_system_config(payload: SystemConfig) -> SystemConfig:
    global _system_config_state
    _system_config_state = payload
    return payload
