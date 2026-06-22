from __future__ import annotations
import json
from typing import Any

class NotebookNode(dict):
    def __getattr__(self, item):
        try: return self[item]
        except KeyError as e: raise AttributeError(item) from e
    def __setattr__(self,key,value):
        self[key]=value

def _convert(value: Any) -> Any:
    if isinstance(value, dict):
        n=NotebookNode()
        for k,v in value.items(): n[k]=_convert(v)
        return n
    if isinstance(value, list):
        return [_convert(v) for v in value]
    return value

def read(fp, as_version: int = 4):
    content = fp.read()
    if not content: return NotebookNode(cells=[], metadata={}, nbformat=4, nbformat_minor=5)
    return _convert(json.loads(content))

def write(nb, fp):
    json.dump(nb, fp, indent=1)
