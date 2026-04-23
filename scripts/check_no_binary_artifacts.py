#!/usr/bin/env python3
"""Block binary artifacts and prohibited model/data blob extensions."""

from __future__ import annotations

import mimetypes
import os
import sys
from pathlib import Path

BLOCKED_EXTS = {".pkl", ".pt", ".h5", ".npy", ".bin", ".ckpt", ".pdb", ".npz"}


def is_binary_file(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            chunk = f.read(2048)
        if b"\x00" in chunk:
            return True
        textchars = bytearray({7, 8, 9, 10, 12, 13, 27} | set(range(0x20, 0x100)))
        return bool(chunk.translate(None, textchars))
    except OSError:
        return False


def main(argv: list[str]) -> int:
    failed = []
    for raw in argv[1:]:
        path = Path(raw)
        if not path.exists() or not path.is_file():
            continue

        if path.suffix.lower() in BLOCKED_EXTS:
            failed.append(f"{path} (blocked extension)")
            continue

        mime, _ = mimetypes.guess_type(str(path))
        if mime is not None and not mime.startswith("text/") and not mime.endswith(("json", "xml", "yaml", "x-python")):
            if mime.startswith("application/") or mime.startswith("image/") or mime.startswith("audio/") or mime.startswith("video/"):
                failed.append(f"{path} (mime={mime})")
                continue

        if is_binary_file(path):
            failed.append(f"{path} (binary content detected)")

    if failed:
        print("Blocked potential binary artifact(s):")
        for item in failed:
            print(f" - {item}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
