from __future__ import annotations

from typing import Any, Iterable

class Response:
    status_code: int

    def raise_for_status(self) -> None: ...

    def iter_content(self, chunk_size: int = ...) -> Iterable[bytes]: ...

    def json(self) -> Any: ...

class RequestException(Exception): ...

class HTTPError(RequestException): ...

class _Exceptions:
    RequestException: type[RequestException]
    HTTPError: type[HTTPError]

exceptions: _Exceptions


def get(
    url: str,
    params: Any = ...,
    data: Any = ...,
    headers: Any = ...,
    timeout: Any = ...,
    stream: bool = ...,
) -> Response: ...


def post(
    url: str,
    data: Any = ...,
    json: Any = ...,
    headers: Any = ...,
    timeout: Any = ...,
) -> Response: ...

