from __future__ import annotations

import json
from typing import Any


class NotebookNode(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def __setattr__(self, key, value):
        self[key] = value


def _convert(value: Any) -> Any:
    if isinstance(value, dict):
        n = NotebookNode()
        for k, v in value.items():
            n[k] = _convert(v)
        return n
    if isinstance(value, list):
        return [_convert(v) for v in value]
    return value


def read(fp, as_version: int = 4):
    return _convert(json.load(fp))


def write(nb: NotebookNode, fp):
    json.dump(nb, fp, indent=1)


class v4:
    @staticmethod
    def new_notebook(cells=None, metadata=None):
        return _convert(
            {
                "cells": cells or [],
                "metadata": metadata or {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )

    @staticmethod
    def new_code_cell(source=""):
        return _convert(
            {
                "cell_type": "code",
                "metadata": {},
                "outputs": [],
                "execution_count": None,
                "source": source,
            }
        )


def new_notebook():
    return v4.new_notebook()


def new_code_cell(source=""):
    return v4.new_code_cell(source)
