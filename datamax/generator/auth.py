from __future__ import annotations

import json
import os
import base64
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import requests
from loguru import logger


@dataclass
class AuthContext:
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)

    def merge(self, other: "AuthContext") -> "AuthContext":
        if other.headers:
            self.headers.update(other.headers)
        if other.query_params:
            self.query_params.update(other.query_params)
        return self


class AuthProvider(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_context(
        self,
        scopes: Optional[Sequence[str]] = None,
        scheme: Optional[dict] = None,
    ) -> AuthContext:
        """Produce headers/query params for the request."""


class OAuthClientCredentialsProvider(AuthProvider):
    def __init__(
        self,
        name: str,
        client_id: str,
        client_secret: str,
        token_url: str,
        *,
        scopes: Optional[Sequence[str]] = None,
        audience: Optional[str] = None,
        tenant_id: Optional[str] = None,
        extra_params: Optional[Dict[str, str]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        token_field: str = "access_token",
        timeout: float = 30.0,
    ):
        super().__init__(name)
        self.client_id = client_id
        self.client_secret = client_secret
        self.token_url = token_url
        self.default_scopes = tuple(str(scope) for scope in scopes or [])
        self.audience = audience
        self.tenant_id = tenant_id
        self.extra_params = dict(extra_params or {})
        self.extra_headers = dict(extra_headers or {})
        self.token_field = token_field
        self.timeout = timeout
        self._token_cache: Dict[Tuple[str, ...], Tuple[str, float]] = {}
        self._lock = threading.Lock()

    def get_context(
        self,
        scopes: Optional[Sequence[str]] = None,
        scheme: Optional[dict] = None,
    ) -> AuthContext:
        desired_scopes = set(self.default_scopes)
        if scopes:
            desired_scopes.update(str(scope) for scope in scopes if scope is not None)
        scope_key = tuple(sorted(desired_scopes))
        token = self._get_token(scope_key)
        return AuthContext(headers={"Authorization": f"Bearer {token}"})

    def _get_token(self, scope_key: Tuple[str, ...]) -> str:
        now = time.time()
        cached = self._token_cache.get(scope_key)
        if cached and cached[1] > now:
            return cached[0]

        with self._lock:
            cached = self._token_cache.get(scope_key)
            if cached and cached[1] > time.time():
                return cached[0]
            token, expires_at = self._fetch_token(scope_key)
            self._token_cache[scope_key] = (token, expires_at)
            return token

    def _fetch_token(self, scope_key: Tuple[str, ...]) -> Tuple[str, float]:
        token_url = self._resolve_token_url()
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if scope_key:
            data["scope"] = " ".join(scope_key)
        if self.audience:
            data["audience"] = self.audience
        if self.tenant_id:
            data.setdefault("tenant_id", self.tenant_id)
        if self.extra_params:
            data.update(self.extra_params)

        headers = {"Accept": "application/json"}
        if self.extra_headers:
            headers.update(self.extra_headers)

        response = requests.post(token_url, data=data, headers=headers, timeout=self.timeout)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:  # pragma: no cover - defensive logging
            logger.error(f"OAuth token request failed for provider '{self.name}': {exc}")
            raise

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("OAuth token response is not valid JSON") from exc

        token = payload.get(self.token_field)
        if not token:
            raise RuntimeError(
                f"OAuth token response missing '{self.token_field}' field for provider '{self.name}'"
            )
        expires_in = payload.get("expires_in")
        if isinstance(expires_in, (int, float)):
            expires_at = time.time() + float(expires_in) - 30.0
        else:
            expires_at = time.time() + 300.0
        return str(token), expires_at

    def _resolve_token_url(self) -> str:
        if "{tenant_id}" in self.token_url and self.tenant_id:
            return self.token_url.replace("{tenant_id}", self.tenant_id)
        return self.token_url


class UrlKeyAuthProvider(AuthProvider):
    def __init__(
        self,
        name: str,
        value: str,
        *,
        param_name: Optional[str] = None,
        location: Optional[str] = None,
        header_name: Optional[str] = None,
    ):
        super().__init__(name)
        self.value = value
        self.param_name = param_name
        self.location = (location or "query").lower()
        self.header_name = header_name

    def get_context(
        self,
        scopes: Optional[Sequence[str]] = None,
        scheme: Optional[dict] = None,
    ) -> AuthContext:
        location = self.location
        name = self.param_name
        header_name = self.header_name

        if scheme and not name:
            name = scheme.get("name") or name
        if scheme and header_name is None and (scheme.get("in") or "").lower() == "header":
            header_name = scheme.get("name")
        if not name and not header_name:
            raise RuntimeError(
                f"URL auth provider '{self.name}' requires a parameter or header name."
            )

        if location == "header" or header_name:
            header = header_name or name
            if not header:
                raise RuntimeError(
                    f"URL auth provider '{self.name}' has no header name resolved."
                )
            return AuthContext(headers={header: self.value})
        return AuthContext(query_params={name or "auth_key": self.value})


class BasicAuthProvider(AuthProvider):
    def __init__(self, name: str, username: str, password: str):
        super().__init__(name)
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("utf-8")
        self._header = {"Authorization": f"Basic {token}"}

    def get_context(
        self,
        scopes: Optional[Sequence[str]] = None,
        scheme: Optional[dict] = None,
    ) -> AuthContext:
        return AuthContext(headers=dict(self._header))


class AuthManager:
    def __init__(self, config: Optional[dict]):
        self._providers: Dict[str, AuthProvider] = {}
        self._scheme_to_provider: Dict[str, str] = {}
        self._default_provider_name: Optional[str] = None
        self._fail_on_missing: bool = True

        if not config:
            return

        options = {}
        if isinstance(config, dict):
            options = config.get("options") if isinstance(config.get("options"), dict) else {}
            providers_block = config.get("providers")
            if isinstance(providers_block, dict):
                items = providers_block.items()
            else:
                reserved = {"default", "options", "providers"}
                items = ((name, cfg) for name, cfg in config.items() if name not in reserved)
            self._default_provider_name = config.get("default")
        else:
            items = []

        fail_flag = options.get("fail_on_missing")
        if isinstance(fail_flag, bool):
            self._fail_on_missing = fail_flag

        for name, provider_cfg in items:
            provider, scheme_names = self._build_provider(name, provider_cfg)
            if provider:
                self._providers[name] = provider
                for scheme_name in scheme_names:
                    if scheme_name in self._scheme_to_provider:
                        logger.warning(
                            "Security scheme '%s' already mapped; overriding with provider '%s'.",
                            scheme_name,
                            name,
                        )
                    self._scheme_to_provider[scheme_name] = name

        if self._default_provider_name and self._default_provider_name not in self._providers:
            logger.warning(
                "Default auth provider '%s' is not defined; ignoring default.",
                self._default_provider_name,
            )
            self._default_provider_name = None

    def _build_provider(
        self,
        name: str,
        provider_cfg: Optional[dict],
    ) -> Tuple[Optional[AuthProvider], Iterable[str]]:
        if not isinstance(provider_cfg, dict):
            logger.error("Auth provider '%s' configuration must be a mapping.", name)
            return None, []
        provider_type = provider_cfg.get("type")
        timeout = float(provider_cfg.get("timeout", 30.0))
        scheme_names = provider_cfg.get("schemes")
        scheme_names = (
            [str(item) for item in scheme_names]
            if isinstance(scheme_names, (list, tuple, set))
            else [name]
        )

        try:
            if provider_type == "oauth_client_credentials":
                provider = OAuthClientCredentialsProvider(
                    name=name,
                    client_id=provider_cfg["client_id"],
                    client_secret=provider_cfg["client_secret"],
                    token_url=provider_cfg["token_url"],
                    scopes=provider_cfg.get("scopes"),
                    audience=provider_cfg.get("audience"),
                    tenant_id=provider_cfg.get("tenant_id"),
                    extra_params=provider_cfg.get("extra_params"),
                    extra_headers=provider_cfg.get("extra_headers"),
                    token_field=provider_cfg.get("token_field", "access_token"),
                    timeout=timeout,
                )
            elif provider_type == "url_auth_key":
                provider = UrlKeyAuthProvider(
                    name=name,
                    value=provider_cfg["value"],
                    param_name=provider_cfg.get("param_name"),
                    location=provider_cfg.get("location"),
                    header_name=provider_cfg.get("header_name"),
                )
            elif provider_type == "basic_auth":
                provider = BasicAuthProvider(
                    name=name,
                    username=provider_cfg["username"],
                    password=provider_cfg["password"],
                )
            else:
                logger.error(
                    "Unsupported auth provider type '%s' for provider '%s'.",
                    provider_type,
                    name,
                )
                return None, []
        except KeyError as exc:
            missing_key = exc.args[0]
            logger.error(
                "Auth provider '%s' missing required configuration key '%s'.",
                name,
                missing_key,
            )
            return None, []

        return provider, scheme_names

    def get_context(self, tool_spec) -> AuthContext:
        security = getattr(tool_spec, "security", None) or []
        security_schemes = getattr(tool_spec, "security_schemes", None) or {}

        if security:
            missing_for_requirements = []
            for requirement in security:
                if not isinstance(requirement, dict):
                    continue
                requirement_context = AuthContext()
                requirement_supported = True
                missing_schemes = []
                for scheme_name, scopes in requirement.items():
                    provider = self._provider_for_scheme(scheme_name)
                    if not provider:
                        requirement_supported = False
                        missing_schemes.append(scheme_name)
                        continue
                    scheme_def = security_schemes.get(scheme_name) if security_schemes else None
                    context = provider.get_context(scopes=scopes, scheme=scheme_def)
                    requirement_context.merge(context)
                if requirement_supported:
                    return requirement_context
                if missing_schemes:
                    missing_for_requirements.extend(missing_schemes)
            if missing_for_requirements and self._fail_on_missing:
                unique_missing = ", ".join(sorted(set(missing_for_requirements)))
                raise RuntimeError(
                    f"No auth provider configured for required security schemes: {unique_missing}."
                )

        default_provider = self._providers.get(self._default_provider_name) if self._default_provider_name else None
        if default_provider:
            return default_provider.get_context()
        return AuthContext()

    def _provider_for_scheme(self, scheme_name: str) -> Optional[AuthProvider]:
        provider_name = self._scheme_to_provider.get(scheme_name)
        if not provider_name and scheme_name in self._providers:
            provider_name = scheme_name
        if provider_name:
            return self._providers.get(provider_name)
        return None


def load_auth_configuration_from_env(
    *,
    config_env_var: str = "AGENT_AUTH_CONFIG",
    path_env_var: str = "AGENT_AUTH_CONFIG_PATH",
) -> Optional[dict]:
    """Load auth configuration for agent generators from environment variables."""
    raw_json = os.getenv(config_env_var)
    config_path = os.getenv(path_env_var)

    if raw_json:
        try:
            return json.loads(raw_json)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Invalid JSON provided via {config_env_var}: {exc}"
            ) from exc

    if config_path:
        path = Path(config_path).expanduser()
        if not path.exists():
            raise RuntimeError(f"Auth configuration file {path} not found.")
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(
                f"Invalid JSON inside auth configuration file {path}: {exc}"
            ) from exc

    return None
