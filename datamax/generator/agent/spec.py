from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from loguru import logger

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from .models import ApiEndpoint, ApiSpec, PromptContext, ToolSpec

SPEC_EXTENSIONS: Tuple[str, ...] = ('.json', '.yaml', '.yml')

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


def discover_spec_files(directory: Path, extensions: Sequence[str] = SPEC_EXTENSIONS) -> List[Path]:
    if not directory.exists():
        return []

    files: Dict[Path, Path] = {}
    for ext in extensions:
        for path in directory.rglob(f"*{ext}"):
            if path.is_file():
                relative = try_relative(path, directory)
                files.setdefault(relative, path)

    return [files[key] for key in sorted(files)]


def try_relative(path: Path, base: Path) -> Path:
    try:
        return path.relative_to(base)
    except ValueError:
        return Path(path.name)


def remove_all_suffixes(path: Path) -> Path:
    result = path
    while result.suffix:
        try:
            result = result.with_suffix("")
        except ValueError:
            break
    return result


def split_relative_path(spec_path: Path, base_dir: Path) -> tuple[Path, str]:
    relative = try_relative(spec_path, base_dir)
    base = remove_all_suffixes(relative)
    return base.parent, base.name


def make_agent_output_stem(spec_path: Path, spec_root: Path, output_root: Path) -> Path:
    relative_dir, stem = split_relative_path(spec_path, spec_root)
    output_dir = output_root / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / stem


def make_agent_checkpoint_path(spec_path: Path, spec_root: Path, checkpoint_root: Path) -> Path:
    relative_dir, stem = split_relative_path(spec_path, spec_root)
    checkpoint_dir = checkpoint_root / relative_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir / f"{stem}.jsonl"


__all__ = [
    "SPEC_EXTENSIONS",
    "ApiGraph",
    "ApiSpec",
    "ApiSpecLoader",
    "PromptContext",
    "ToolSpec",
    "discover_spec_files",
    "make_agent_checkpoint_path",
    "make_agent_output_stem",
    "method_upper",
    "remove_all_suffixes",
    "split_relative_path",
    "try_relative",
]

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


