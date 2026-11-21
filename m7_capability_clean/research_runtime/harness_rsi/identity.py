"""M6 -- immutable content identity and artifact freezing.

Every candidate, manifest, result and gate in M6 is identified by the SHA-256
of its COMPLETE immutable bytes. A name, a partial dictionary or a caller
assertion is never an identity. This module is framework-agnostic and has no
dependency on the research engines.

Design rules:
  * ``sha256_json`` canonicalises with sorted keys and stable separators; the
    same object always yields the same digest.
  * ``freeze_json`` writes an artifact to disk and RETURNS the digest recomputed
    from the bytes actually on disk (not from the in-memory object), so a
    subsequent disk re-read is meaningful.
  * ``verify_frozen`` re-reads the file and fails closed on any difference,
    which is the mechanism that detects "modified candidate after freeze"
    and "modified result after evaluation".
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

_SHA64 = "0123456789abcdef"


class IdentityError(ValueError):
    """Raised when content identity cannot be established or verified."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    if not p.is_file():
        raise IdentityError(f"artifact does not exist: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), default=str).encode("utf-8")


def sha256_json(obj: Any) -> str:
    """Full-artifact SHA of any JSON-serialisable object."""
    return sha256_bytes(_canonical_bytes(obj))


def freeze_json(obj: Any, path: str | Path) -> dict[str, Any]:
    """Write an immutable artifact and return its ON-DISK identity.

    Writes atomically via a temporary file plus rename so a crash cannot leave
    a half-written artifact that later looks frozen. Re-reads the bytes after
    the rename and binds the returned digest to what is actually on disk.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(_canonical_bytes(obj))
    os.replace(tmp, p)
    digest = sha256_file(p)
    return {"path": str(p), "sha256": digest, "n_bytes": p.stat().st_size}


def freeze_artifact(obj: Any, path: str | Path) -> dict[str, Any]:
    """Freeze a standalone artifact (e.g. sealed gold) and return its identity.

    Unlike ``freeze_json`` this does NOT embed the digest in the object, so the
    object's own content SHA can be referenced by another manifest without
    becoming self-referential.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_bytes(_canonical_bytes(obj))
    os.replace(tmp, p)
    return {"path": str(p), "sha256": sha256_file(p),
            "n_bytes": p.stat().st_size}


def verify_frozen(path: str | Path, expected_sha256: str) -> dict[str, Any]:
    """Re-read a frozen artifact and fail closed if its bytes changed."""
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "reasons": ["artifact_missing"], "path": str(p)}
    actual = sha256_file(p)
    if actual != expected_sha256:
        return {"ok": False, "reasons": ["artifact_sha_mismatch"],
                "path": str(p), "expected": expected_sha256, "actual": actual}
    return {"ok": True, "reasons": [], "path": str(p), "sha256": actual}


def is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(c in _SHA64 for c in value.lower()))


def git_fingerprint(repo: str | Path) -> dict[str, Any]:
    """Record an exact code version (HEAD + dirty state) for evidence binding."""
    p = Path(repo)
    if not p.is_dir():
        return {"head": "", "dirty": None, "branch": "", "error": "repo_missing"}
    try:
        head = subprocess.run(["git", "-C", str(p), "rev-parse", "HEAD"],
                              capture_output=True, text=True, timeout=15).stdout.strip()
        branch = subprocess.run(["git", "-C", str(p), "rev-parse",
                                 "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True, timeout=15).stdout.strip()
        status = subprocess.run(["git", "-C", str(p), "status", "--porcelain"],
                                capture_output=True, text=True, timeout=15).stdout.strip()
        return {"head": head, "branch": branch, "dirty": bool(status),
                "error": ""}
    except Exception as exc:  # noqa: BLE001 - evidence must be honest
        return {"head": "", "dirty": None, "branch": "",
                "error": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "IdentityError", "sha256_bytes", "sha256_file", "sha256_json",
    "freeze_json", "freeze_artifact", "verify_frozen", "is_sha256",
    "git_fingerprint",
]
