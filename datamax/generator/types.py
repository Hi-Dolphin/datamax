from __future__ import annotations

import dataclasses
import datetime as dt
import os
import urllib.parse
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence


@dataclasses.dataclass(slots=True)
class QAGenerationResult:
    """Structured return for QA generation results."""

    qa_pairs: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    performance: Optional[Dict[str, Any]]
    raw_content: str
    source_file: Optional[str]
    domain_tree: Optional[Dict[str, Any]]
    call_records: List[Dict[str, Any]] = dataclasses.field(default_factory=list)

    def total_pairs(self) -> int:
        return len(self.qa_pairs)


@dataclasses.dataclass(slots=True)
class PersistenceToggles:
    save_pairs: bool = True
    save_raw_payload: bool = True
    save_pipeline_events: bool = True
    save_hourly_metrics: bool = True
    save_daily_metrics: bool = True


@dataclasses.dataclass(slots=True)
class PersistenceConfig:
    backend: str
    dsn: str
    source_key: str
    source_name: Optional[str] = None
    owner_team: Optional[str] = None
    created_by: Optional[str] = None
    run_name: Optional[str] = None
    trigger_type: str = "manual"
    model_key: Optional[str] = None
    model_provider: Optional[str] = None
    model_version: Optional[str] = None
    question_number: Optional[int] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    max_qps: Optional[float] = None
    tags: Dict[str, Any] = dataclasses.field(default_factory=dict)
    toggles: PersistenceToggles = dataclasses.field(default_factory=PersistenceToggles)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PersistenceConfig":
        backend = data.get("backend") or os.getenv("QA_PERSIST_BACKEND")
        if not backend:
            raise ValueError("persistence backend must be specified")

        backend = backend.lower()
        if backend != "postgres":
            raise ValueError(f"Unsupported persistence backend: {backend}")

        dsn = (
            data.get("dsn")
            or os.getenv("QA_DB_DSN")
            or os.getenv("DATABASE_URL")
            or cls._build_dsn_from_env(data)
        )
        if not dsn:
            raise ValueError("Postgres persistence requires a DSN string.")

        source_key = data.get("source_key") or os.getenv("QA_SOURCE_KEY")
        if not source_key:
            raise ValueError("Postgres persistence requires a source_key.")

        toggles = data.get("toggles")
        if isinstance(toggles, Mapping):
            toggle_obj = PersistenceToggles(
                save_pairs=bool(toggles.get("save_pairs", True)),
                save_raw_payload=bool(toggles.get("save_raw_payload", True)),
                save_pipeline_events=bool(toggles.get("save_pipeline_events", True)),
                save_hourly_metrics=bool(toggles.get("save_hourly_metrics", True)),
                save_daily_metrics=bool(toggles.get("save_daily_metrics", True)),
            )
        else:
            toggle_obj = PersistenceToggles()

        cfg = cls(
            backend=backend,
            dsn=str(dsn),
            source_key=str(source_key),
            source_name=data.get("source_name") or os.getenv("QA_SOURCE_NAME"),
            owner_team=data.get("owner_team") or os.getenv("QA_OWNER_TEAM"),
            created_by=data.get("created_by") or os.getenv("QA_CREATED_BY"),
            run_name=data.get("run_name")
            or os.getenv("QA_RUN_NAME")
            or f"qa_run_{dt.datetime.utcnow():%Y%m%d_%H%M%S}",
            trigger_type=data.get("trigger_type") or os.getenv("QA_TRIGGER_TYPE", "manual"),
            model_key=data.get("model_key") or os.getenv("QA_MODEL"),
            model_provider=data.get("model_provider") or os.getenv("QA_MODEL_PROVIDER"),
            model_version=data.get("model_version") or os.getenv("QA_MODEL_VERSION"),
            question_number=_coerce_int(data.get("question_number") or os.getenv("QA_QUESTION_NUMBER")),
            chunk_size=_coerce_int(data.get("chunk_size") or os.getenv("QA_CHUNK_SIZE")),
            chunk_overlap=_coerce_int(data.get("chunk_overlap") or os.getenv("QA_CHUNK_OVERLAP")),
            max_qps=_coerce_float(data.get("max_qps") or os.getenv("QA_MAX_QPS")),
            tags=dict(data.get("tags") or {}),
            toggles=toggle_obj,
        )
        return cfg

    @staticmethod
    def _build_dsn_from_env(data: Mapping[str, Any]) -> Optional[str]:
        prefix = data.get("env_prefix") or os.getenv("QA_DB_PREFIX", "POSTGRES_PROD_DIFY")
        if not prefix:
            return None
        host = os.getenv(f"{prefix}_HOST")
        port = os.getenv(f"{prefix}_PORT")
        user = os.getenv(f"{prefix}_USERNAME")
        password = os.getenv(f"{prefix}_PASSWORD")
        database = os.getenv(f"{prefix}_DB")
        schema = os.getenv(f"{prefix}_SCHEMA")
        protocol = os.getenv(f"{prefix}_CONNECT", "postgresql")
        if not host or not user or not database:
            return None
        port_part = f":{port}" if port else ""
        schema_param = f"?options=-c%20search_path%3D{schema}" if schema else ""
        username = urllib.parse.quote(user, safe="")
        password_part = ""
        if password is not None:
            encoded_password = urllib.parse.quote(password, safe="")
            password_part = f":{encoded_password}"
        return f"{protocol}://{username}{password_part}@{host}{port_part}/{database}{schema_param}"


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
