"""Minimal Hypothesis compatibility layer for deterministic local tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import strategies as st


class HealthCheck:
    function_scoped_fixture = "function_scoped_fixture"
    differing_executors = "differing_executors"


class Verbosity:
    quiet = "quiet"
    normal = "normal"
    verbose = "verbose"


class Phase:
    generate = "generate"


@dataclass
class _SettingsManager:
    """No-op settings manager API compatible with a subset of Hypothesis."""

    def __call__(self, **kwargs):
        def decorator(fn):
            return fn

        return decorator

    def register_profile(self, name: str, **kwargs: Any) -> None:
        return None

    def load_profile(self, name: str) -> None:
        return None


settings = _SettingsManager()


def _mark_hypothesis_test(fn):
    setattr(fn, "is_hypothesis_test", True)
    return fn


def is_hypothesis_test(test) -> bool:
    return bool(getattr(test, "is_hypothesis_test", False))


def given(**kwargs):
    def decorator(fn):
        def wrapper(*args, **inner_kwargs):
            generated = {k: v.example() for k, v in kwargs.items()}
            return fn(*args, **generated, **inner_kwargs)

        wrapper.__name__ = getattr(fn, "__name__", "wrapper")
        wrapper.__doc__ = getattr(fn, "__doc__")
        wrapper.__module__ = getattr(fn, "__module__")
        return _mark_hypothesis_test(wrapper)

    return decorator
