"""Subset of provider helpers required by pytest-hypothesis plugin."""

from __future__ import annotations


def _get_local_constants():
    """Return empty constants map for compatibility with plugin hooks."""
    return {}
