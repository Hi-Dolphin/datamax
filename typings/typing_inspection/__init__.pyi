from __future__ import annotations

from typing import Any, Tuple

TypeInfo = Any


def is_protocol(tp: Any) -> bool: ...

def get_generic_type(obj: Any) -> Any: ...

def get_generic_parameters(obj: Any) -> Tuple[Any, ...]: ...

