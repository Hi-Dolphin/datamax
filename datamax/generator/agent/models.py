from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


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


__all__ = [
    "AgentEpisode",
    "AgentQuestion",
    "AgentTurn",
    "ApiEndpoint",
    "ApiSpec",
    "PromptContext",
    "ReviewResult",
    "ToolCandidate",
    "ToolCall",
    "ToolSpec",
]
