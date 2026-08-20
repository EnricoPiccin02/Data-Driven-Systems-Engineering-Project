"""
Lightweight content-hash dataset versioning.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path


def file_sha256(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(paths: Iterable[Path], manifest_path: Path) -> dict:
    """Write (and return) a version manifest for the given dataset files."""
    entries = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        entries[str(p.name)] = {
            "sha256": file_sha256(p),
            "size_bytes": p.stat().st_size,
        }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def verify_manifest(manifest_path: Path) -> bool:
    """Return True iff every file's current hash matches the recorded hash."""
    if not manifest_path.exists():
        return False
    with open(manifest_path) as f:
        manifest = json.load(f)
    base_dir = manifest_path.parent
    for fname, meta in manifest["files"].items():
        fpath = base_dir / fname
        if not fpath.exists() or file_sha256(fpath) != meta["sha256"]:
            return False
    return True
