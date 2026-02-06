"""Minimal Hypothesis compatibility layer for deterministic local tests."""

from . import strategies as st


def given(**kwargs):
    def decorator(fn):
        def wrapper(*args, **inner_kwargs):
            generated = {k: v.example() for k, v in kwargs.items()}
            return fn(*args, **generated, **inner_kwargs)

        return wrapper

    return decorator


def settings(**kwargs):
    def decorator(fn):
        return fn

    return decorator
