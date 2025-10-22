from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import urljoin

from loguru import logger
import requests

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - optional dependency
    END = "__langgraph_end__"
    StateGraph = None

from datamax.utils.performance_monitor import PerformanceMonitor

from .qa_generator import (
    DEFAULT_MAX_RETRIES,
    MIN_REQUEST_INTERVAL_SECONDS,
    QAProgressTracker,
    extract_json_from_llm_output,
    llm_generator,
)
from .auth import AuthManager, AuthContext


@dataclass
class ToolSpec:
    name: str
    method: str
    path: str
    description: str
    operation_id: Optional[str]
    input_schema: Optional[dict] = None
    output_schema: Optional[dict] = None
    tags: List[str] = field(default_factory=list)
    source_spec: Optional[str] = None
    parameters: List[dict] = field(default_factory=list)
    servers: List[str] = field(default_factory=list)
    security: List[dict] = field(default_factory=list)
    security_schemes: Dict[str, dict] = field(default_factory=dict)

    def to_prompt_block(self) -> str:
        request_schema = ""
        if self.input_schema:
            request_schema = json.dumps(self.input_schema, ensure_ascii=False)[:2000]
        response_schema = ""
        if self.output_schema:
            response_schema = json.dumps(self.output_schema, ensure_ascii=False)[:2000]
        parts = [
            f"Tool Name: {self.name}",
            f"HTTP: {self.method.upper()} {self.path}",
            f"Description: {self.description or 'N/A'}",
        ]
        if self.tags:
            parts.append(f"Tags: {', '.join(self.tags)}")
        if request_schema:
            parts.append(f"Input JSON Schema: {request_schema}")
        if response_schema:
            parts.append(f"Output JSON Schema: {response_schema}")
        if self.operation_id:
            parts.append(f"operationId: {self.operation_id}")
        if self.security:
            schemes = ", ".join(sorted({key for entry in self.security for key in entry.keys()}))
            parts.append(f"Auth Required: {schemes or 'unspecified scheme'}")
        return "\n".join(parts)


@dataclass
class ApiEndpoint:
    identifier: str
    method: str
    path: str
    summary: str
    description: str
    tags: List[str]
    parameters: List[dict]
    request_body: Optional[dict]
    responses: Dict[str, Any]
    security: List[dict]
    servers: List[str]
    source_spec: str
    dependencies: List[str] = field(default_factory=list)
    request_schema: Optional[dict] = None
    response_schemas: Dict[str, dict] = field(default_factory=dict)
    security_schemes: Dict[str, dict] = field(default_factory=dict)

    def tool_name(self) -> str:
        if self.identifier:
            return self.identifier
        clean_path = self.path.strip("/").replace("/", "_").replace("{", "").replace("}", "")
        return f"{self.method.lower()}_{clean_path or 'root'}"

    def to_tool_spec(self) -> ToolSpec:
        success_response = None
        success_status = None
        for status, resp in self.responses.items():
            if str(status).startswith("2"):
                success_status = str(status)
                success_response = resp
                break
        return ToolSpec(
            name=self.tool_name(),
            method=self.method,
            path=self.path,
            description=self.summary or self.description or "",
            operation_id=self.identifier if self.identifier and self.identifier != self.tool_name() else None,
            input_schema=self._extract_request_schema(),
            output_schema=self._extract_response_schema(success_status, success_response),
            tags=list(self.tags),
            source_spec=self.source_spec,
            servers=list(self.servers),
            security=list(self.security),
            security_schemes=dict(self.security_schemes),
            parameters=[param for param in self.parameters],
        )

    def _extract_request_schema(self) -> Optional[dict]:
        if self.request_schema:
            return self.request_schema
        if not self.request_body:
            return None
        content = self.request_body.get("content") or {}
        for media_type in ("application/json", "application/*+json"):
            if media_type in content:
                schema = content[media_type].get("schema")
                if isinstance(schema, dict):
                    return schema
        for item in content.values():
            if isinstance(item, dict) and isinstance(item.get("schema"), dict):
                return item["schema"]
        return None

    def _extract_response_schema(self, status: Optional[str], response: Optional[dict]) -> Optional[dict]:
        if status and status in self.response_schemas:
            return self.response_schemas[status]
        if self.response_schemas:
            for key, schema in self.response_schemas.items():
                if key.startswith("2"):
                    return schema
            return next(iter(self.response_schemas.values()))
        if not response:
            return None
        content = response.get("content") or {}
        for media_type in ("application/json", "application/*+json"):
            if media_type in content and isinstance(content[media_type], dict):
                schema = content[media_type].get("schema")
                if isinstance(schema, dict):
                    return schema
        for item in content.values():
            if isinstance(item, dict) and isinstance(item.get("schema"), dict):
                return item["schema"]
        return None


@dataclass
class ApiSpec:
    source: str
    raw: dict
    title: str
    version: str
    servers: List[str]
    endpoints: List[ApiEndpoint]
    security_schemes: Dict[str, dict]


@dataclass
class PromptContext:
    name: str
    endpoints: List[ApiEndpoint]
    context_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentQuestion:
    question_id: str
    prompt: str
    target_tools: List[str]
    scenario_type: str
    difficulty: str
    prerequisites: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ToolCandidate:
    tool_name: str
    score: Optional[float]
    rationale: str

    def to_dict(self) -> dict:
        return {
            "tool_name": self.tool_name,
            "score": self.score,
            "rationale": self.rationale,
        }


@dataclass
class ToolCall:
    sequence: int
    tool_name: str
    input: Dict[str, Any]
    observation: str
    success: bool
    latency_seconds: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sequence": self.sequence,
            "tool_name": self.tool_name,
            "input": self.input,
            "observation": self.observation,
            "success": self.success,
            "latency_seconds": self.latency_seconds,
            "error": self.error,
            "metadata": self.metadata,
        }


@dataclass
class AgentTurn:
    role: str
    content: str
    tool_name: Optional[str] = None
    tool_input: Optional[Dict[str, Any]] = None
    observation: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "observation": self.observation,
        }


@dataclass
class ReviewResult:
    success: bool
    score: Optional[float]
    issues: List[str]
    suggestions: List[str]
    raw: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "score": self.score,
            "issues": list(self.issues),
            "suggestions": list(self.suggestions),
            "raw": self.raw,
        }


@dataclass
class AgentEpisode:
    episode_id: str
    question: AgentQuestion
    tool_candidates: List[ToolCandidate]
    tool_calls: List[ToolCall]
    turns: List[AgentTurn]
    final_answer: str
    agent_name: str
    review: Optional[ReviewResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "question": self.question.to_dict(),
            "tool_candidates": [candidate.to_dict() for candidate in self.tool_candidates],
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "turns": [turn.to_dict() for turn in self.turns],
            "final_answer": self.final_answer,
            "agent_name": self.agent_name,
            "review": self.review.to_dict() if self.review else None,
            "metadata": self.metadata,
        }


@dataclass
class AgentGenerationConfig:
    api_key: str
    base_url: str
    agent_question_generate_model: str
    classify_model: str
    core_agent_answer_generate_model: str
    review_model: str
    question_count: int = 20
    max_questions_per_context: int = 4
    top_k_tools: int = 5
    max_turns: int = 8
    langgraph_retry: int = 1
    checkpoint_path: Optional[str] = None
    resume_from_checkpoint: bool = True
    max_retries: int = DEFAULT_MAX_RETRIES
    min_request_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS
    question_temperature: float = 0.5
    classify_temperature: float = 0.3
    agent_temperature: float = 0.7
    review_temperature: float = 0.2
    max_workers: int = 4
    debug: bool = False
    agent_backend: str = "langgraph"
    auth: Optional[Dict[str, Any]] = None
    default_tool_server: Optional[str] = None
    tool_request_timeout: float = 30.0
    require_auth_for_protected_tools: bool = True


# ---------------------------------------------------------------------------
# API specification loading and normalisation
# ---------------------------------------------------------------------------


class ApiSpecLoader:
    """Load OpenAPI / Swagger specifications and convert to internal structures."""

    def load(self, spec_sources: Sequence[Union[str, dict]]) -> List[ApiSpec]:
        specs: List[ApiSpec] = []
        seen_keys: set[str] = set()
        for source in spec_sources:
            key = self._build_source_key(source)
            if key in seen_keys:
                logger.debug(f"Skipping duplicate specification source: {key}")
                continue
            seen_keys.add(key)
            try:
                specs.append(self._load_single(source))
            except Exception as exc:
                logger.error(f"Failed to load specification {source}: {exc}")
        return specs

    def _load_single(self, source: Union[str, dict]) -> ApiSpec:
        if isinstance(source, dict):
            raw = source
            title = raw.get("info", {}).get("title", "Unnamed API")
            version = raw.get("info", {}).get("version", "0.0.0")
            return self._build_spec(raw, title, version, "<memory>")

        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Specification file {source} does not exist")
        text = path.read_text(encoding="utf-8")

        raw: Optional[dict] = None
        try:
            raw = json.loads(text)
        except json.JSONDecodeError:
            pass

        if raw is None:
            if yaml is None:
                raise RuntimeError(
                    "PyYAML is required to parse YAML specifications. Install pyyaml to continue."
                )
            raw = yaml.safe_load(text)

        if not isinstance(raw, dict):
            raise ValueError(f"Specification {source} is not a valid OpenAPI document")

        title = raw.get("info", {}).get("title", path.stem)
        version = raw.get("info", {}).get("version", "0.0.0")
        return self._build_spec(raw, title, version, str(path))

    def _build_spec(self, raw: dict, title: str, version: str, source: str) -> ApiSpec:
        paths = raw.get("paths") or {}
        servers = [srv.get("url", "") for srv in raw.get("servers", []) if isinstance(srv, dict)]
        components = raw.get("components") or {}
        security_schemes = components.get("securitySchemes") if isinstance(components, dict) else {}
        if not isinstance(security_schemes, dict):
            security_schemes = {}
        endpoints: List[ApiEndpoint] = []
        for path, operations in paths.items():
            if not isinstance(operations, dict):
                continue
            for method, operation in operations.items():
                method_lower = str(method).lower()
                if method_lower not in {"get", "post", "put", "delete", "patch", "options", "head"}:
                    continue
                if not isinstance(operation, dict):
                    continue
                identifier = operation.get("operationId") or f"{method_upper(method_lower)} {path}"
                summary = operation.get("summary") or ""
                description = operation.get("description") or ""
                tags = operation.get("tags") or []
                parameters = self._normalise_parameters(operation.get("parameters"), raw)
                request_body = operation.get("requestBody")
                responses = operation.get("responses") or {}
                security = operation.get("security") or raw.get("security") or []
                op_servers = operation.get("servers") or []
                endpoint_servers = [srv.get("url", "") for srv in op_servers if isinstance(srv, dict)]
                resolved_request_schema = (
                    self._resolve_request_schema(request_body, raw) if isinstance(request_body, dict) else None
                )
                resolved_responses = (
                    self._resolve_response_schemas(responses, raw) if isinstance(responses, dict) else {}
                )
                endpoint = ApiEndpoint(
                    identifier=identifier,
                    method=method_lower,
                    path=path,
                    summary=summary,
                    description=description,
                    tags=tags if isinstance(tags, list) else [],
                    parameters=parameters,
                    request_body=request_body if isinstance(request_body, dict) else None,
                    responses=responses if isinstance(responses, dict) else {},
                    security=security if isinstance(security, list) else [],
                    servers=endpoint_servers or servers,
                    source_spec=source,
                    dependencies=self._extract_dependencies(operation),
                    request_schema=resolved_request_schema,
                    response_schemas=resolved_responses,
                    security_schemes=security_schemes,
                )
                endpoints.append(endpoint)
        self._infer_structural_dependencies(endpoints)
        return ApiSpec(
            source=source,
            raw=raw,
            title=title,
            version=version,
            servers=servers,
            endpoints=endpoints,
            security_schemes=security_schemes,
        )

    def _build_source_key(self, source: Union[str, dict]) -> str:
        if isinstance(source, dict):
            if "info" in source:
                info = source.get("info") or {}
                title = info.get("title") or "<unnamed>"
                version = info.get("version") or "<unknown>"
                return f"dict::{title}::{version}"
            return f"dict::{id(source)}"
        try:
            path = Path(source)
            return f"path::{path.resolve()}"
        except Exception:
            return f"raw::{str(source)}"

    def _normalise_parameters(self, params: Any, raw: dict) -> List[dict]:
        if not isinstance(params, list):
            return []
        normalised: List[dict] = []
        for param in params:
            resolved = self._resolve_parameter(param, raw)
            if isinstance(resolved, dict):
                normalised.append(resolved)
        return normalised

    def _resolve_parameter(self, param: Any, raw: dict, trail: Optional[set[str]] = None) -> Optional[dict]:
        if not isinstance(param, dict):
            return None
        param_copy = dict(param)
        ref = param_copy.get("$ref")
        if isinstance(ref, str):
            if trail is None:
                trail = set()
            if ref in trail:
                param_copy.pop("$ref", None)
            else:
                target = self._lookup_ref(ref, raw)
                if isinstance(target, dict):
                    merged = self._merge_dicts(target, {k: v for k, v in param_copy.items() if k != "$ref"})
                    trail = set(trail)
                    trail.add(ref)
                    return self._resolve_parameter(merged, raw, trail)
        schema = param_copy.get("schema")
        if isinstance(schema, dict):
            param_copy["schema"] = self._resolve_schema(schema, raw)
        return param_copy

    def _resolve_request_schema(self, request_body: dict, raw: dict) -> Optional[dict]:
        content = request_body.get("content")
        if not isinstance(content, dict):
            return None
        media_preferences = (
            "application/json",
            "application/*+json",
            "application/x-www-form-urlencoded",
            "*/*",
        )
        for media_type in media_preferences:
            body = content.get(media_type)
            if isinstance(body, dict) and isinstance(body.get("schema"), dict):
                return self._resolve_schema(body["schema"], raw)
        for body in content.values():
            if isinstance(body, dict) and isinstance(body.get("schema"), dict):
                return self._resolve_schema(body["schema"], raw)
        return None

    def _resolve_response_schemas(self, responses: dict, raw: dict) -> Dict[str, dict]:
        resolved: Dict[str, dict] = {}
        for status, response in responses.items():
            if not isinstance(response, dict):
                continue
            content = response.get("content")
            if not isinstance(content, dict):
                continue
            schema: Optional[dict] = None
            for media_type in ("application/json", "application/*+json"):
                body = content.get(media_type)
                if isinstance(body, dict) and isinstance(body.get("schema"), dict):
                    schema = self._resolve_schema(body["schema"], raw)
                    break
            if schema is None:
                for body in content.values():
                    if isinstance(body, dict) and isinstance(body.get("schema"), dict):
                        schema = self._resolve_schema(body["schema"], raw)
                        break
            if schema is not None:
                resolved[str(status)] = schema
        return resolved

    def _resolve_schema(self, schema: Any, raw: dict, trail: Optional[set[str]] = None) -> Any:
        if not isinstance(schema, dict):
            return schema
        schema_copy = dict(schema)
        ref = schema_copy.get("$ref")
        if isinstance(ref, str):
            if trail is None:
                trail = set()
            if ref not in trail:
                target = self._lookup_ref(ref, raw)
                if isinstance(target, dict):
                    merged = self._merge_dicts(target, {k: v for k, v in schema_copy.items() if k != "$ref"})
                    new_trail = set(trail)
                    new_trail.add(ref)
                    return self._resolve_schema(merged, raw, new_trail)
            schema_copy.pop("$ref", None)
        for key, value in list(schema_copy.items()):
            if isinstance(value, dict):
                schema_copy[key] = self._resolve_schema(value, raw, trail)
            elif isinstance(value, list):
                schema_copy[key] = [self._resolve_schema(item, raw, trail) for item in value]
        return schema_copy

    @staticmethod
    def _merge_dicts(base: dict, override: dict) -> dict:
        merged = dict(base)
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ApiSpecLoader._merge_dicts(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _lookup_ref(ref: str, raw: dict) -> Optional[dict]:
        if not ref.startswith("#/"):
            return None
        parts = ref.lstrip("#/").split("/")
        node: Any = raw
        for part in parts:
            if not isinstance(node, dict):
                return None
            node = node.get(part)
            if node is None:
                return None
        if isinstance(node, dict):
            return node
        return None

    @staticmethod
    def _extract_dependencies(operation: dict) -> List[str]:
        deps: List[str] = []
        for key in ("x-depends-on", "x-dependencies", "xSequence"):
            value = operation.get(key)
            if isinstance(value, list):
                deps.extend([str(item) for item in value if isinstance(item, (str, int))])
            elif isinstance(value, str):
                deps.append(value)
        return deps

    @staticmethod
    def _infer_structural_dependencies(endpoints: List[ApiEndpoint]) -> None:
        by_base_path: Dict[str, List[ApiEndpoint]] = {}
        for endpoint in endpoints:
            base = endpoint.path.split("/{")[0]
            by_base_path.setdefault(base, []).append(endpoint)

        for base, group in by_base_path.items():
            creators = [ep for ep in group if ep.method in {"post", "put"}]
            readers = [ep for ep in group if ep.method in {"get", "patch", "delete"} and "{" in ep.path]
            for reader in readers:
                for creator in creators:
                    if creator.identifier not in reader.dependencies:
                        reader.dependencies.append(creator.identifier)


def method_upper(method: str) -> str:
    return method.upper()

# ---------------------------------------------------------------------------
# API graph and helper views
# ---------------------------------------------------------------------------


class ApiGraph:
    def __init__(self, specs: Sequence[ApiSpec]):
        self.specs = list(specs)
        self._endpoints: List[ApiEndpoint] = []
        for spec in specs:
            self._endpoints.extend(spec.endpoints)
        self._endpoint_by_id: Dict[str, ApiEndpoint] = {
            endpoint.identifier: endpoint for endpoint in self._endpoints
        }
        self._tool_catalog: Optional[List[ToolSpec]] = None

    def endpoints(self) -> List[ApiEndpoint]:
        return list(self._endpoints)

    def get(self, identifier: str) -> Optional[ApiEndpoint]:
        return self._endpoint_by_id.get(identifier)

    def dependencies_for(self, identifier: str) -> List[str]:
        endpoint = self.get(identifier)
        if not endpoint:
            return []
        return list(endpoint.dependencies)

    def build_prompt_contexts(self) -> List[PromptContext]:
        contexts: List[PromptContext] = []
        by_tag: Dict[str, List[ApiEndpoint]] = {}
        for endpoint in self._endpoints:
            if endpoint.tags:
                for tag in endpoint.tags:
                    by_tag.setdefault(tag, []).append(endpoint)
            else:
                by_tag.setdefault("untagged", []).append(endpoint)

        for tag, tagged_endpoints in by_tag.items():
            contexts.append(
                PromptContext(
                    name=f"Tag:{tag}",
                    endpoints=tagged_endpoints,
                    context_type="tag",
                    metadata={"tag": tag},
                )
            )

        for endpoint in self._endpoints:
            deps = [self.get(dep_id) for dep_id in endpoint.dependencies if self.get(dep_id)]
            if deps:
                chain = deps + [endpoint]
                contexts.append(
                    PromptContext(
                        name=f"Chain:{endpoint.identifier}",
                        endpoints=[ep for ep in chain if ep],
                        context_type="dependency_chain",
                        metadata={"target": endpoint.identifier},
                    )
                )

        return contexts

    def tool_catalog(self) -> List[ToolSpec]:
        if self._tool_catalog is None:
            catalog: List[ToolSpec] = []
            seen: set[str] = set()
            for endpoint in self._endpoints:
                tool = endpoint.to_tool_spec()
                if tool.name not in seen:
                    catalog.append(tool)
                    seen.add(tool.name)
            self._tool_catalog = catalog
        return list(self._tool_catalog)

    def describe_endpoints(self, endpoints: Sequence[ApiEndpoint], limit: int = 8) -> str:
        lines: List[str] = []
        for endpoint in list(endpoints)[:limit]:
            params = self._format_parameters(endpoint.parameters)
            request_info = self._summarize_schema(endpoint.request_schema)
            summary = endpoint.summary or endpoint.description or "No description"
            dependency_text = ""
            if endpoint.dependencies:
                dependency_text = f" Depends on: {', '.join(endpoint.dependencies)}."
            lines.append(
                f"- {endpoint.identifier} [{endpoint.method.upper()} {endpoint.path}] :: {summary}. "
                f"Params: {params}. Body: {request_info}.{dependency_text}"
            )
        if len(endpoints) > limit:
            lines.append(f"... ({len(endpoints) - limit} more endpoints omitted)")
        return "\n".join(lines)

    @staticmethod
    def _format_parameters(parameters: Sequence[dict], limit: int = 6) -> str:
        formatted: List[str] = []
        for param in parameters:
            if not isinstance(param, dict):
                continue
            name = str(param.get("name") or "_")
            location = param.get("in")
            schema = param.get("schema") if isinstance(param.get("schema"), dict) else {}
            type_name = ApiGraph._extract_type_name(schema) or "any"
            desc_source = param.get("description") or schema.get("description") or ""
            description = ApiGraph._truncate(desc_source, 60)
            required = bool(param.get("required"))
            parts = [name]
            if location:
                parts.append(f"[{location}]")
            parts.append(f"({type_name})")
            if required:
                parts.append("!")
            entry = "".join(parts)
            if description:
                entry += f": {description}"
            formatted.append(entry)
        if not formatted:
            return "None"
        if len(formatted) > limit:
            extra = len(formatted) - limit
            formatted = formatted[:limit] + [f"... +{extra} more"]
        return "; ".join(formatted)

    @staticmethod
    def _summarize_schema(schema: Optional[dict], limit: int = 6) -> str:
        if not schema:
            return "None"
        schema_type = schema.get("type") or ApiGraph._extract_type_name(schema) or "object"
        if schema_type.startswith("array<"):
            return schema_type
        if schema_type == "array":
            item_type = ApiGraph._extract_type_name(schema.get("items"))
            return f"array<{item_type or 'object'}>"
        properties = schema.get("properties")
        if isinstance(properties, dict) and properties:
            required = {str(name) for name in schema.get("required", []) if isinstance(name, str)}
            items: List[str] = []
            for key, value in list(properties.items())[:limit]:
                entry = ApiGraph._summarize_property(key, value, required)
                items.append(entry)
            if len(properties) > limit:
                items.append(f"... +{len(properties) - limit} more")
            return f"{schema_type}{{{'; '.join(items)}}}"
        if "enum" in schema and isinstance(schema["enum"], list):
            enum_preview = ", ".join(str(item) for item in schema["enum"][:limit])
            if len(schema["enum"]) > limit:
                enum_preview += ", ..."
            return f"enum[{enum_preview}]"
        description = schema.get("description")
        if description:
            return f"{schema_type}: {ApiGraph._truncate(description, 80)}"
        return schema_type

    @staticmethod
    def _summarize_property(name: str, prop: Any, required: set[str]) -> str:
        if not isinstance(prop, dict):
            return name
        type_name = ApiGraph._extract_type_name(prop) or prop.get("type") or "object"
        description = prop.get("description") or ""
        description = ApiGraph._truncate(description, 60)
        entry = f"{name}({type_name})"
        if name in required:
            entry += "!"
        if description:
            entry += f": {description}"
        return entry

    @staticmethod
    def _extract_type_name(schema: Any) -> str:
        if not isinstance(schema, dict):
            return ""
        schema_type = schema.get("type")
        if schema_type == "array":
            item_type = ApiGraph._extract_type_name(schema.get("items"))
            return f"array<{item_type or 'object'}>"
        if schema_type:
            return str(schema_type)
        ref = schema.get("$ref")
        if isinstance(ref, str) and ref:
            return ref.split("/")[-1]
        if "enum" in schema and isinstance(schema["enum"], list):
            return "enum"
        for composite_key in ("oneOf", "anyOf", "allOf"):
            composite = schema.get(composite_key)
            if isinstance(composite, list) and composite:
                delimiter = " | " if composite_key != "allOf" else " & "
                parts = [
                    part for part in (ApiGraph._extract_type_name(item) for item in composite if isinstance(item, dict))
                    if part
                ]
                if parts:
                    return delimiter.join(parts)
        return ""

    @staticmethod
    def _truncate(text: str, length: int = 80) -> str:
        if not text:
            return ""
        if len(text) <= length:
            return text
        return text[: length - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Question generation
# ---------------------------------------------------------------------------


class AgentQuestionGenerator:
    def __init__(self, config: AgentGenerationConfig):
        self.config = config

    def generate(self, api_graph: ApiGraph, monitor: PerformanceMonitor) -> List[AgentQuestion]:
        questions: List[AgentQuestion] = []
        contexts = api_graph.build_prompt_contexts()
        if not contexts:
            logger.warning("No prompt contexts derived from API specifications")
            return questions

        with monitor.stage("agent_question_generation"):
            for context in contexts:
                if len(questions) >= self.config.question_count:
                    break
                batch_limit = min(
                    self.config.max_questions_per_context,
                    self.config.question_count - len(questions),
                )
                prompt = self._build_prompt(context, batch_limit, api_graph)
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an API integration expert responsible for designing"
                            " realistic, high-coverage agent tasks for tool-augmented assistants."
                            " Create diverse user questions grounded in the provided interfaces."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
                logger.debug(f"Generating questions for context {context.name}")
                raw = llm_generator(
                    api_key=self.config.api_key,
                    model=self.config.agent_question_generate_model,
                    base_url=self.config.base_url,
                    message=messages,
                    temperature=self.config.question_temperature,
                    top_p=0.9,
                    type="agent_question",
                    debug=self.config.debug,
                    max_retries=self.config.max_retries,
                    min_interval_seconds=self.config.min_request_interval_seconds,
                    perf_monitor=monitor,
                    perf_stage="agent_question_generation",
                )
                if not raw:
                    continue
                parsed = extract_json_from_llm_output(raw[0])
                if not isinstance(parsed, list):
                    logger.warning("Question generation output not a list; skipping context")
                    continue
                for entry in parsed:
                    if len(questions) >= self.config.question_count:
                        break
                    question = self._normalise_question(entry, context, api_graph)
                    if question:
                        questions.append(question)
            monitor.add_stage_items("agent_question_generation", len(questions))
        return questions

    def _build_prompt(self, context: PromptContext, batch_limit: int, api_graph: ApiGraph) -> str:
        endpoint_summary = api_graph.describe_endpoints(context.endpoints, limit=12)
        instructions = (
            "Generate realistic end-user questions/goals that require interacting with the provided APIs. "
            "Ensure coverage of single-tool and multi-tool chains (including failure handling when relevant). "
            "Return a JSON array where each item has keys: "
            "`question` (string), `required_tools` (list of tool names), `scenario_type` "
            "(one of ['single_call','multi_step','error_recovery','batch','analytics']), "
            "`difficulty` (easy|medium|hard), `prerequisites` (list of preconditions), "
            "`metadata` (object with helpful notes). "
            "Required tools must reference the tool names listed below. "
            "Prioritise questions that highlight dependencies or typical workflows described."
        )
        return (
            f"Context: {context.name}\n"
            f"Endpoints:\n{endpoint_summary}\n\n"
            f"Please generate up to {batch_limit} questions.\n"
            f"{instructions}\n"
            "Respond with JSON only."
        )

    def _normalise_question(
        self,
        entry: Any,
        context: PromptContext,
        api_graph: ApiGraph,
    ) -> Optional[AgentQuestion]:
        if not isinstance(entry, dict):
            return None
        question_text = entry.get("question") or entry.get("prompt")
        if not question_text or not isinstance(question_text, str):
            return None
        required_tools = entry.get("required_tools")
        if not isinstance(required_tools, list):
            required_tools = []
        normalised_tools: List[str] = []
        for tool in required_tools:
            if not isinstance(tool, str):
                continue
            tool = tool.strip()
            if tool in api_graph._endpoint_by_id or any(
                tool == ep.tool_name() for ep in context.endpoints
            ):
                normalised_tools.append(tool)
        if not normalised_tools:
            normalised_tools = [context.endpoints[0].tool_name()]
        scenario_type = entry.get("scenario_type") or "multi_step"
        difficulty = entry.get("difficulty") or "medium"
        prerequisites = entry.get("prerequisites") or []
        metadata = entry.get("metadata") or {}
        question_id = str(uuid.uuid4())
        return AgentQuestion(
            question_id=question_id,
            prompt=question_text.strip(),
            target_tools=normalised_tools,
            scenario_type=scenario_type,
            difficulty=difficulty,
            prerequisites=prerequisites if isinstance(prerequisites, list) else [],
            metadata=metadata if isinstance(metadata, dict) else {},
        )


# ---------------------------------------------------------------------------
# Tool classification
# ---------------------------------------------------------------------------


class ToolClassifier:
    def __init__(self, config: AgentGenerationConfig):
        self.config = config

    def classify(
        self,
        question: AgentQuestion,
        tool_catalog: Sequence[ToolSpec],
        monitor: PerformanceMonitor,
    ) -> List[ToolCandidate]:
        tool_blocks = []
        for tool in tool_catalog:
            tool_blocks.append(tool.to_prompt_block())
        prompt = (
            "You are selecting the best tools for an autonomous agent. "
            "Given the question and tool catalog, return a JSON array with each item containing "
            "`tool_name`, `score` (0-1 float), and `rationale`. "
            f"Select at most {self.config.top_k_tools} tools sorted by descending score."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question.prompt}\n\n"
                    f"Candidate tools:\n" + "\n\n".join(tool_blocks)
                ),
            },
        ]
        raw = llm_generator(
            api_key=self.config.api_key,
            model=self.config.classify_model,
            base_url=self.config.base_url,
            message=messages,
            temperature=self.config.classify_temperature,
            top_p=0.9,
            type="tool_classification",
            debug=self.config.debug,
            max_retries=self.config.max_retries,
            min_interval_seconds=self.config.min_request_interval_seconds,
            perf_monitor=monitor,
            perf_stage="tool_classification",
        )
        if not raw:
            return []
        parsed = extract_json_from_llm_output(raw[0])
        candidates: List[ToolCandidate] = []
        if isinstance(parsed, list):
            for item in parsed[: self.config.top_k_tools]:
                if not isinstance(item, dict):
                    continue
                name = item.get("tool_name")
                if not name or not isinstance(name, str):
                    continue
                score = item.get("score")
                try:
                    score_value = float(score) if score is not None else None
                except (TypeError, ValueError):
                    score_value = None
                rationale = item.get("rationale") or ""
                candidates.append(
                    ToolCandidate(tool_name=name.strip(), score=score_value, rationale=str(rationale)),
                )
        monitor.add_stage_items("tool_classification", len(candidates))
        return candidates


# ---------------------------------------------------------------------------
# LangGraph-based agent execution
# ---------------------------------------------------------------------------


class ToolRegistry:
    def __init__(
        self,
        tool_specs: Sequence[ToolSpec],
        *,
        auth_manager: Optional[AuthManager] = None,
        default_server: Optional[str] = None,
        timeout: float = 30.0,
        require_auth: bool = True,
    ):
        self._by_name: Dict[str, ToolSpec] = {spec.name: spec for spec in tool_specs}
        self._auth_manager = auth_manager or AuthManager(None)
        self._default_server = default_server
        self._timeout = timeout
        self._require_auth = require_auth

    def resolve(self, name: str) -> Optional[ToolSpec]:
        return self._by_name.get(name)

    def list_tool_names(self) -> List[str]:
        return list(self._by_name.keys())

    def specs(self) -> List[ToolSpec]:
        return list(self._by_name.values())

    def simulate_invocation(self, tool: ToolSpec, params: Dict[str, Any]) -> str:
        fields = []
        schema = tool.output_schema or {}
        properties = schema.get("properties") if isinstance(schema, dict) else None
        if isinstance(properties, dict):
            fields = list(properties.keys())
        dominant_fields = ", ".join(fields[:6]) if fields else "unspecified fields"
        param_desc = json.dumps(params, ensure_ascii=False) if params else "{}"
        return (
            f"Simulated response from {tool.method.upper()} {tool.path}. "
            f"Parameters: {param_desc}. Likely returns {dominant_fields}."
        )

    def invoke(
        self,
        tool: ToolSpec,
        params: Optional[Dict[str, Any]],
        monitor: Optional[PerformanceMonitor] = None,
    ) -> Tuple[str, bool, Optional[str], Optional[float]]:
        observation, success, error, latency = self._execute_request(tool, params or {})
        if monitor and latency is not None:
            monitor.record_request(stage="tool_invocation", duration_seconds=latency)
        return observation, success, error, latency

    def _execute_request(
        self,
        tool: ToolSpec,
        params: Dict[str, Any],
    ) -> Tuple[str, bool, Optional[str], Optional[float]]:
        if not isinstance(params, dict):
            raise TypeError(
                f"Tool invocation expects parameters as an object, received {type(params).__name__}."
            )
        method, url, headers, query_params, json_payload, data_payload = self._prepare_request(tool, params)
        start = time.perf_counter()
        try:
            response = requests.request(
                method=method,
                url=url,
                headers=headers or None,
                params=query_params or None,
                json=json_payload if json_payload is not None else None,
                data=data_payload,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:
            latency = time.perf_counter() - start
            message = f"Request failure invoking {tool.name}: {exc}"
            logger.error(message)
            return message, False, str(exc), latency

        latency = time.perf_counter() - start
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            observation = self._format_error_response(response)
            logger.warning(
                "Tool %s responded with HTTP %s: %s",
                tool.name,
                response.status_code,
                response.reason,
            )
            return observation, False, str(exc), latency

        observation = self._summarise_response(response)
        return observation, True, None, latency

    def _prepare_request(
        self,
        tool: ToolSpec,
        params: Dict[str, Any],
    ) -> Tuple[str, str, Dict[str, str], Dict[str, Any], Optional[Any], Optional[Any]]:
        payload = {key: value for key, value in params.items() if value is not None}
        path_values, query_params, header_params, remaining = self._extract_parameter_values(tool, payload)
        url, remaining_after_path = self._resolve_url(tool, path_values, remaining)
        method = (tool.method or "get").lower()

        body_json: Optional[Any] = None
        body_data: Optional[Any] = None
        if method in {"get", "delete", "head", "options"}:
            for key, value in remaining_after_path.items():
                query_params.setdefault(key, value)
        else:
            body_candidate = None
            if "body" in remaining_after_path:
                body_candidate = remaining_after_path.pop("body")
            elif "json" in remaining_after_path:
                body_candidate = remaining_after_path.pop("json")
            elif remaining_after_path:
                body_candidate = remaining_after_path

            if isinstance(body_candidate, (str, bytes, bytearray)):
                body_data = body_candidate
            elif body_candidate is not None:
                body_json = body_candidate

        auth_context = self._resolve_auth_context(tool)

        headers: Dict[str, str] = {k: str(v) for k, v in header_params.items()}
        query_merged: Dict[str, Any] = dict(query_params)
        if auth_context.headers:
            headers.update(auth_context.headers)
        if auth_context.query_params:
            query_merged.update(auth_context.query_params)

        if body_json is not None:
            header_names = {key.lower(): key for key in headers.keys()}
            if "content-type" not in header_names:
                headers["Content-Type"] = "application/json"

        return method, url, headers, query_merged, body_json, body_data

    def _extract_parameter_values(
        self,
        tool: ToolSpec,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        working = dict(payload)
        path_values: Dict[str, Any] = {}
        query_params: Dict[str, Any] = {}
        header_params: Dict[str, Any] = {}

        for parameter in tool.parameters or []:
            if not isinstance(parameter, dict):
                continue
            name = parameter.get("name")
            if not name:
                continue
            location = (parameter.get("in") or "").lower()
            required = bool(parameter.get("required"))
            lookup_keys = [name]
            if "-" in name:
                lookup_keys.append(name.replace("-", "_"))

            value_found = False
            value = None
            for key in lookup_keys:
                if key in working:
                    value = working.pop(key)
                    value_found = True
                    break

            if location == "path":
                if not value_found:
                    if required:
                        raise RuntimeError(
                            f"Missing required path parameter '{name}' for tool '{tool.name}'."
                        )
                    continue
                path_values[name] = value
            elif value_found:
                if location == "query":
                    query_params[name] = value
                elif location == "header":
                    header_params[name] = value

        return path_values, query_params, header_params, working

    def _resolve_url(
        self,
        tool: ToolSpec,
        path_values: Dict[str, Any],
        payload: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        path_template = tool.path or "/"
        remaining = dict(payload)
        placeholders = re.findall(r"{([^}]+)}", path_template)

        for placeholder in placeholders:
            key = placeholder.split(":")[0]
            if key in path_values:
                value = path_values.pop(key)
            elif key in remaining:
                value = remaining.pop(key)
            elif key.replace("-", "_") in remaining:
                alt_key = key.replace("-", "_")
                value = remaining.pop(alt_key)
            else:
                raise RuntimeError(
                    f"Missing value for path placeholder '{key}' in tool '{tool.name}'."
                )
            path_template = path_template.replace(f"{{{placeholder}}}", str(value))

        base_url = self._select_base_url(tool)
        full_url = urljoin(base_url.rstrip("/") + "/", path_template.lstrip("/"))
        return full_url, remaining

    def _select_base_url(self, tool: ToolSpec) -> str:
        for candidate in tool.servers:
            if not isinstance(candidate, str):
                continue
            cleaned = candidate.strip()
            if not cleaned:
                continue
            if cleaned.startswith("http://") or cleaned.startswith("https://"):
                return cleaned
            if self._default_server:
                return urljoin(self._default_server.rstrip("/") + "/", cleaned.lstrip("/"))
        if self._default_server:
            return self._default_server
        raise RuntimeError(
            f"No server URL available for tool '{tool.name}'. Provide `default_tool_server` in the config."
        )

    def _resolve_auth_context(self, tool: ToolSpec) -> AuthContext:
        try:
            return self._auth_manager.get_context(tool)
        except RuntimeError:
            if self._require_auth:
                raise
            logger.warning("Auth disabled yet required for tool '%s'; continuing without credentials.", tool.name)
            return AuthContext()

    def _summarise_response(self, response: requests.Response) -> str:
        body_preview = self._extract_body_preview(response)
        return f"HTTP {response.status_code}: {body_preview}"

    def _format_error_response(self, response: requests.Response) -> str:
        body_preview = self._extract_body_preview(response)
        reason = response.reason or "error"
        return f"HTTP {response.status_code} {reason}: {body_preview}"

    @staticmethod
    def _extract_body_preview(response: requests.Response, limit: int = 2000) -> str:
        content_type = (response.headers.get("Content-Type") or "").lower()
        if "application/json" in content_type:
            try:
                payload = response.json()
                text = json.dumps(payload, ensure_ascii=False)
            except ValueError:
                text = response.text or ""
        else:
            text = response.text or ""
        return ToolRegistry._truncate(text, limit=limit)

    @staticmethod
    def _truncate(value: str, limit: int = 2000) -> str:
        if value is None:
            return ""
        if len(value) <= limit:
            return value
        return value[: limit - 3] + "..."


def build_agent_plan(
    config: AgentGenerationConfig,
    tool_registry: ToolRegistry,
    question: AgentQuestion,
    tool_candidates: Sequence[str],
    monitor: Optional[PerformanceMonitor],
) -> List[dict]:
    catalog_lines: List[str] = []
    for name in tool_candidates:
        tool = tool_registry.resolve(name)
        if tool:
            catalog_lines.append(tool.to_prompt_block())
    if not catalog_lines:
        catalog_lines = [spec.to_prompt_block() for spec in tool_registry.specs()]

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert planner for an autonomous API agent. "
                "Given the question and available tools, produce a JSON array plan. "
                "Each item must include `tool`, `reason`, `inputs` (object), "
                "and optional `assistant_thought` summarising internal reasoning."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question.prompt}\n\nAvailable tools:\n"
                + "\n\n".join(catalog_lines)
            ),
        },
    ]

    raw = llm_generator(
        api_key=config.api_key,
        model=config.core_agent_answer_generate_model,
        base_url=config.base_url,
        message=messages,
        temperature=config.agent_temperature,
        top_p=0.9,
        type="agent_plan",
        debug=config.debug,
        max_retries=config.max_retries,
        min_interval_seconds=config.min_request_interval_seconds,
        perf_monitor=monitor,
        perf_stage="agent_answer_generation",
    )
    parsed = extract_json_from_llm_output(raw[0]) if raw else None
    tool_list = list(tool_candidates)
    fallback_tools = tool_list or tool_registry.list_tool_names()
    if not isinstance(parsed, list):
        if not fallback_tools:
            return []
        return [{"tool": fallback_tools[0], "reason": "Default fallback", "inputs": {}}]
    steps: List[dict] = []
    for entry in parsed[: config.max_turns]:
        if not isinstance(entry, dict):
            continue
        tool_name = entry.get("tool")
        if not tool_name or not isinstance(tool_name, str):
            continue
        steps.append(
            {
                "tool": tool_name.strip(),
                "reason": entry.get("reason") or "",
                "inputs": entry.get("inputs") if isinstance(entry.get("inputs"), dict) else {},
                "assistant_thought": entry.get("assistant_thought"),
            }
        )
    if not steps and fallback_tools:
        steps.append({"tool": fallback_tools[0], "reason": "Default fallback", "inputs": {}})
    return steps


def generate_agent_final_answer(
    config: AgentGenerationConfig,
    question: AgentQuestion,
    tool_calls: Sequence[ToolCall],
    monitor: Optional[PerformanceMonitor],
) -> str:
    call_summaries = []
    for call in tool_calls:
        call_summaries.append(
            f"Tool {call.tool_name} with input {json.dumps(call.input, ensure_ascii=False)} "
            f"produced observation: {call.observation}"
        )
    messages = [
        {
            "role": "system",
            "content": (
                "You are the final responding agent. Summarise the findings, "
                "provide the requested output, and offer any follow-up actions. "
                "Incorporate reflection on tool reliability when helpful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question.prompt}\n\n"
                f"Tool call summary:\n" + "\n".join(call_summaries)
            ),
        },
    ]
    raw = llm_generator(
        api_key=config.api_key,
        model=config.core_agent_answer_generate_model,
        base_url=config.base_url,
        message=messages,
        temperature=config.agent_temperature,
        top_p=0.9,
        type="agent_final_answer",
        debug=config.debug,
        max_retries=config.max_retries,
        min_interval_seconds=config.min_request_interval_seconds,
        perf_monitor=monitor,
        perf_stage="agent_answer_generation",
    )
    if raw:
        return raw[0]
    return "Unable to produce final answer due to generation failure."


def ensure_langgraph_available() -> None:
    if StateGraph is None:
        raise ImportError(
            "LangGraph is not installed. Install langgraph>=0.0.30 to run the agent pipeline."
        )


class LangGraphAgent:
    def __init__(self, config: AgentGenerationConfig, tool_registry: ToolRegistry):
        ensure_langgraph_available()
        self.config = config
        self.tool_registry = tool_registry
        self.agent_name = "langgraph_agent"
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(dict)

        def plan_node(state: dict) -> dict:
            state["plan"] = self._generate_plan(state)
            state["step_index"] = 0
            turns: List[AgentTurn] = state.get("turns") or []
            plan_explanation = "\n".join(
                f"{idx + 1}. {step.get('tool')} - {step.get('reason')}"
                for idx, step in enumerate(state["plan"])
            )
            turns.append(
                AgentTurn(
                    role="assistant",
                    content=f"Planning steps:\n{plan_explanation}",
                )
            )
            state["turns"] = turns
            return state

        def act_node(state: dict) -> dict:
            plan: List[dict] = state.get("plan") or []
            index = state.get("step_index", 0)
            if index >= len(plan):
                state["complete"] = True
                return state
            step = plan[index]
            tool_name = step.get("tool")
            if not tool_name:
                state.setdefault("issues", []).append("Plan step missing tool name")
                state["step_index"] = index + 1
                return state
            tool_spec = self.tool_registry.resolve(tool_name)
            if not tool_spec:
                state.setdefault("issues", []).append(f"Tool {tool_name} not found")
                state["step_index"] = index + 1
                return state
            params = step.get("inputs") or {}
            observation, success, error, latency = self.tool_registry.invoke(
                tool_spec, params, monitor
            )
            tool_calls: List[ToolCall] = state.get("tool_calls") or []
            sequence = len(tool_calls) + 1
            tool_calls.append(
                ToolCall(
                    sequence=sequence,
                    tool_name=tool_spec.name,
                    input=params if isinstance(params, dict) else {},
                    observation=observation,
                    success=success,
                    latency_seconds=latency,
                    error=error,
                    metadata={"reason": step.get("reason")},
                )
            )
            if not success and error:
                state.setdefault("issues", []).append(
                    f"Tool {tool_spec.name} invocation failed: {error}"
                )
            turns: List[AgentTurn] = state.get("turns") or []
            turns.append(
                AgentTurn(
                    role="assistant",
                    content=step.get("assistant_thought") or f"Calling tool {tool_spec.name}",
                    tool_name=tool_spec.name,
                    tool_input=params if isinstance(params, dict) else {},
                )
            )
            turns.append(
                AgentTurn(
                    role="tool",
                    content=observation,
                    tool_name=tool_spec.name,
                    observation=observation,
                )
            )
            state["turns"] = turns
            state["tool_calls"] = tool_calls
            state["step_index"] = index + 1
            return state

        def should_continue(state: dict) -> str:
            plan: List[dict] = state.get("plan") or []
            index = state.get("step_index", 0)
            if index >= len(plan) or index >= self.config.max_turns:
                return "finalise"
            return "act"

        def finalise_node(state: dict) -> dict:
            final_answer = self._generate_final_answer(state)
            state["final_answer"] = final_answer
            turns: List[AgentTurn] = state.get("turns") or []
            turns.append(AgentTurn(role="assistant", content=final_answer))
            state["turns"] = turns
            return state

        graph.add_node("plan", plan_node)
        graph.add_node("act", act_node)
        graph.add_node("finalise", finalise_node)
        graph.set_entry_point("plan")
        graph.add_edge("plan", "act")
        graph.add_conditional_edges(
            "act",
            should_continue,
            {"act": "act", "finalise": "finalise"},
        )
        graph.add_edge("finalise", END)
        return graph.compile()

    def run(
        self,
        question: AgentQuestion,
        tool_candidates: Sequence[ToolCandidate],
        monitor: PerformanceMonitor,
    ) -> Tuple[List[AgentTurn], List[ToolCall], str]:
        tool_names = [candidate.tool_name for candidate in tool_candidates if candidate.tool_name]
        initial_state = {
            "question": question,
            "tool_candidates": tool_names,
            "turns": [AgentTurn(role="user", content=question.prompt)],
            "tool_calls": [],
            "monitor": monitor,
        }
        with monitor.stage("agent_answer_generation"):
            final_state = self.graph.invoke(initial_state)
        tool_calls = final_state.get("tool_calls") or []
        turns = final_state.get("turns") or []
        final_answer = final_state.get("final_answer") or ""
        monitor.add_stage_items("agent_answer_generation", 1)
        return turns, tool_calls, final_answer

    def _generate_plan(self, state: dict) -> List[dict]:
        question: AgentQuestion = state["question"]
        tool_candidates: Sequence[str] = state.get("tool_candidates") or []
        monitor: Optional[PerformanceMonitor] = state.get("monitor")
        return build_agent_plan(
            config=self.config,
            tool_registry=self.tool_registry,
            question=question,
            tool_candidates=tool_candidates,
            monitor=monitor,
        )

    def _generate_final_answer(self, state: dict) -> str:
        question: AgentQuestion = state["question"]
        tool_calls: List[ToolCall] = state.get("tool_calls") or []
        monitor: Optional[PerformanceMonitor] = state.get("monitor")
        return generate_agent_final_answer(
            config=self.config,
            question=question,
            tool_calls=tool_calls,
            monitor=monitor,
        )


# ---------------------------------------------------------------------------
# Native OpenAI-style agent
# ---------------------------------------------------------------------------


class OpenAIAgent:
    def __init__(self, config: AgentGenerationConfig, tool_registry: ToolRegistry):
        self.config = config
        self.tool_registry = tool_registry
        self.agent_name = "openai_native_agent"

    def run(
        self,
        question: AgentQuestion,
        tool_candidates: Sequence[ToolCandidate],
        monitor: PerformanceMonitor,
    ) -> Tuple[List[AgentTurn], List[ToolCall], str]:
        tool_names = [candidate.tool_name for candidate in tool_candidates if candidate.tool_name]
        turns: List[AgentTurn] = [AgentTurn(role="user", content=question.prompt)]
        tool_calls: List[ToolCall] = []

        with monitor.stage("agent_answer_generation"):
            plan = build_agent_plan(
                config=self.config,
                tool_registry=self.tool_registry,
                question=question,
                tool_candidates=tool_names,
                monitor=monitor,
            )
            if plan:
                plan_explanation = "\n".join(
                    f"{idx + 1}. {step.get('tool')} - {step.get('reason')}"
                    for idx, step in enumerate(plan)
                )
                turns.append(
                    AgentTurn(
                        role="assistant",
                        content=f"Planning steps:\n{plan_explanation}",
                    )
                )
            for step_index, step in enumerate(plan, 1):
                tool_name = step.get("tool")
                if not tool_name:
                    continue
                tool_spec = self.tool_registry.resolve(tool_name)
                if not tool_spec:
                    turns.append(
                        AgentTurn(
                            role="assistant",
                            content=f"Step {step_index}: tool '{tool_name}' unavailable, skipping.",
                        )
                    )
                    continue
                params = step.get("inputs") if isinstance(step.get("inputs"), dict) else {}
                assistant_thought = step.get("assistant_thought") or f"Calling tool {tool_spec.name}"
                turns.append(
                    AgentTurn(
                        role="assistant",
                        content=assistant_thought,
                        tool_name=tool_spec.name,
                        tool_input=params,
                    )
                )
                observation, success, error, latency = self.tool_registry.invoke(
                    tool_spec, params, monitor
                )
                tool_call = ToolCall(
                    sequence=len(tool_calls) + 1,
                    tool_name=tool_spec.name,
                    input=params,
                    observation=observation,
                    success=success,
                    error=error,
                    latency_seconds=latency,
                    metadata={"reason": step.get("reason")},
                )
                tool_calls.append(tool_call)
                if not success and error:
                    turns.append(
                        AgentTurn(
                            role="assistant",
                            content=f"Tool {tool_spec.name} failed: {error}",
                        )
                    )
                turns.append(
                    AgentTurn(
                        role="tool",
                        content=observation,
                        tool_name=tool_spec.name,
                        observation=observation,
                    )
                )
                if len(tool_calls) >= self.config.max_turns:
                    break

            final_answer = generate_agent_final_answer(
                config=self.config,
                question=question,
                tool_calls=tool_calls,
                monitor=monitor,
            )
            turns.append(AgentTurn(role="assistant", content=final_answer))

        monitor.add_stage_items("agent_answer_generation", 1)
        return turns, tool_calls, final_answer


# ---------------------------------------------------------------------------
# Review pipeline
# ---------------------------------------------------------------------------


class AgentReviewPipeline:
    def __init__(self, config: AgentGenerationConfig):
        self.config = config

    def review(self, episode: AgentEpisode, monitor: PerformanceMonitor) -> ReviewResult:
        issues: List[str] = []
        suggestions: List[str] = []
        messages = [
            {
                "role": "system",
                "content": (
                    "You are auditing agent-generated training data. "
                    "Check for correctness, completeness, and tool alignment. "
                    "Return JSON with fields: success (bool), score (0-1), issues (list), suggestions (list)."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "question": episode.question.to_dict(),
                        "tool_calls": [call.to_dict() for call in episode.tool_calls],
                        "final_answer": episode.final_answer,
                        "turns": [turn.to_dict() for turn in episode.turns],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            },
        ]
        raw = llm_generator(
            api_key=self.config.api_key,
            model=self.config.review_model,
            base_url=self.config.base_url,
            message=messages,
            temperature=self.config.review_temperature,
            top_p=0.1,
            type="agent_review",
            debug=self.config.debug,
            max_retries=self.config.max_retries,
            min_interval_seconds=self.config.min_request_interval_seconds,
            perf_monitor=monitor,
            perf_stage="agent_review",
        )
        parsed = extract_json_from_llm_output(raw[0]) if raw else None
        success = False
        score = None
        if isinstance(parsed, dict):
            success = bool(parsed.get("success"))
            score = parsed.get("score")
            issues = parsed.get("issues") if isinstance(parsed.get("issues"), list) else []
            suggestions = (
                parsed.get("suggestions") if isinstance(parsed.get("suggestions"), list) else []
            )
        monitor.add_stage_items("agent_review", 1)
        return ReviewResult(
            success=success,
            score=float(score) if isinstance(score, (int, float)) else None,
            issues=[str(item) for item in issues],
            suggestions=[str(item) for item in suggestions],
            raw=parsed if isinstance(parsed, dict) else None,
        )


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


class AgentProgressTracker(QAProgressTracker):
    def _make_key(self, entry: dict) -> Optional[str]:
        if "episode_id" in entry:
            return entry["episode_id"]
        if "question" in entry and isinstance(entry["question"], dict):
            question_id = entry["question"].get("question_id")
            if question_id:
                return str(question_id)
        return super()._make_key(entry)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


class AgentTrainingDataGenerator:
    def __init__(self, config: AgentGenerationConfig):
        self.config = config
        self.loader = ApiSpecLoader()
        self.monitor = PerformanceMonitor()
        self.question_generator = AgentQuestionGenerator(config)
        self.classifier = ToolClassifier(config)
        self.review_pipeline = AgentReviewPipeline(config)

    def run(self, spec_sources: Sequence[Union[str, dict]]) -> dict:
        specs = self.loader.load(spec_sources)
        if not specs:
            raise ValueError("No API specifications could be loaded; aborting generation.")

        api_graph = ApiGraph(specs)
        tool_catalog = api_graph.tool_catalog()
        if self.config.require_auth_for_protected_tools and not self.config.auth:
            protected_schemes = sorted(
                {
                    scheme
                    for tool in tool_catalog
                    for requirement in tool.security
                    if isinstance(requirement, dict)
                    for scheme in requirement.keys()
                }
            )
            if protected_schemes:
                raise RuntimeError(
                    "Authentication is required for the loaded API specifications. "
                    "Provide credentials for the following security schemes via AgentGenerationConfig.auth: "
                    + ", ".join(protected_schemes)
                )
        auth_manager = AuthManager(self.config.auth)
        tool_registry = ToolRegistry(
            tool_catalog,
            auth_manager=auth_manager,
            default_server=self.config.default_tool_server,
            timeout=self.config.tool_request_timeout,
            require_auth=self.config.require_auth_for_protected_tools,
        )
        backend = (self.config.agent_backend or "langgraph").lower()
        if backend == "langgraph":
            agent = LangGraphAgent(self.config, tool_registry)
        elif backend == "openai":
            agent = OpenAIAgent(self.config, tool_registry)
        else:
            raise ValueError(f"Unsupported agent backend: {self.config.agent_backend}")

        tracker: Optional[AgentProgressTracker] = None
        if self.config.checkpoint_path:
            tracker = AgentProgressTracker(
                self.config.checkpoint_path, resume=self.config.resume_from_checkpoint
            )

        questions = self.question_generator.generate(api_graph, self.monitor)
        episodes: List[AgentEpisode] = []
        for question in questions:
            episode_id = question.question_id
            if tracker and episode_id in tracker.entries_by_key:
                continue
            candidates = self.classifier.classify(question, tool_catalog, self.monitor)
            turns, tool_calls, final_answer = agent.run(question, candidates, self.monitor)
            episode = AgentEpisode(
                episode_id=episode_id,
                question=question,
                tool_candidates=candidates,
                tool_calls=tool_calls,
                turns=turns,
                final_answer=final_answer,
                agent_name=agent.agent_name,
                metadata={
                    "difficulty": question.difficulty,
                    "agent_backend": backend,
                },
            )
            review = self.review_pipeline.review(episode, self.monitor)
            episode.review = review
            episodes.append(episode)
            if tracker:
                tracker.record(episode.to_dict())

        result = {
            "episodes": [episode.to_dict() for episode in episodes],
            "tool_catalog": [asdict(tool) for tool in tool_catalog],
            "questions": [question.to_dict() for question in questions],
            "metadata": {
                "spec_count": len(specs),
                "question_count": len(questions),
                "episode_count": len(episodes),
                "max_turns": self.config.max_turns,
                "agent_backend": backend,
            },
            "performance": self.monitor.build_report(),
        }
        return result


def generate_agent_training_data(
    spec_sources: Sequence[Union[str, dict]],
    config: AgentGenerationConfig,
) -> dict:
    generator = AgentTrainingDataGenerator(config)
    return generator.run(spec_sources)
