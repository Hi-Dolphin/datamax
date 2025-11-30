from __future__ import annotations

import warnings

try:
    from .crawler import crawl  # type: ignore[attr-defined]
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    warnings.warn(f"Optional crawler dependencies not available: {exc}", RuntimeWarning)
    crawl = None  # type: ignore[assignment]

try:
    from .parser import DataMax  # type: ignore[attr-defined]
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    warnings.warn(f"Optional parser dependencies not available: {exc}", RuntimeWarning)
    DataMax = None  # type: ignore[assignment]

__all__ = ["crawl", "DataMax"]
