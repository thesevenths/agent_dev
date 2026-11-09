"""P1-05: immutable, hash-traceable comparison report storage.

Layout (under the campaign root, which is already ignored by git because every
campaign_id is unique and reproducible)::

    <campaign_root>/<campaign_id>/comparisons/
        <candidate_id>__<comparison_id>.json   # never overwritten, ever
        index.json                           # metadata only, atomically replaced

Guarantees
---------
* A report file is created with ``xb`` (exclusive create). If the same
  comparison_id already exists, the EXISTING bytes are returned untouched.
* ``comparison_id`` is derived from the campaign, candidate, algorithm version
  and the canonical evidence fingerprint. Changing the algorithm version (or
  the evidence) therefore yields a NEW file, never a silent overwrite.
* Both the canonical content digest (stored inside the report as
  ``report_sha256``) and the on-disk file digest are recorded in ``index.json``.
* The campaign's own manifest / event log are never modified here; the caller
  may append a ``comparison_completed`` event to the CampaignStore.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..schemas import ComparisonReport
from .exceptions import ComparisonStoreError
from .schemas import COMPARISON_ALGORITHM_VERSION

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_report_payload(report: ComparisonReport) -> dict[str, Any]:
    """Report content WITHOUT the digest field (so the digest is computable)."""
    payload = report.model_dump(mode="json")
    payload.pop("report_sha256", None)
    return payload


def content_sha256(report: ComparisonReport) -> str:
    return sha256_bytes(json.dumps(canonical_report_payload(report),
                                   sort_keys=True, ensure_ascii=False,
                                   separators=(",", ":")).encode("utf-8"))


def _safe_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))
    if cleaned in ("", ".", "..") or ".." in cleaned:
        raise ComparisonStoreError(f"unsafe report identifier segment: {value!r}")
    return cleaned


@dataclass(frozen=True)
class ReportArtifact:
    path: str
    comparison_id: str
    content_sha256: str
    file_sha256: str
    created_new: bool


class ComparisonReportStore:
    def __init__(self, campaign_root: str | Path):
        self.campaign_root = Path(campaign_root)

    def _comparisons_dir(self, campaign_id: str, *, create: bool) -> Path:
        if not _SAFE_ID_RE.match(campaign_id) or ".." in campaign_id:
            raise ComparisonStoreError(f"unsafe campaign_id {campaign_id!r}")
        directory = self.campaign_root / campaign_id / "comparisons"
        if create:
            directory.mkdir(parents=True, exist_ok=True)
        return directory

    @staticmethod
    def _report_filename(report: ComparisonReport) -> str:
        return (f"{_safe_segment(report.candidate_id)}__"
                f"{_safe_segment(report.comparison_id)}.json")

    @staticmethod
    def _file_bytes(report: ComparisonReport) -> bytes:
        return (json.dumps(report.model_dump(mode="json"), indent=2,
                           ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")

    # ------------------------------------------------------------- writes
    def save(self, report: ComparisonReport, *,
             campaign_id: str | None = None) -> ReportArtifact:
        cid = campaign_id or report.campaign_id
        if not cid:
            raise ComparisonStoreError("campaign_id is required to store a report")
        digest = content_sha256(report)
        if report.report_sha256 and report.report_sha256 != digest:
            raise ComparisonStoreError(
                "report_sha256 does not match the canonical report content")
        final = (report if report.report_sha256 == digest
                else report.model_copy(update={"report_sha256": digest}))
        directory = self._comparisons_dir(cid, create=True)
        path = directory / self._report_filename(final)
        payload = self._file_bytes(final)

        created_new = False
        try:
            with path.open("xb") as handle:
                handle.write(payload)
            created_new = True
        except FileExistsError:
            # Immutable: never rewrite. Validate that the existing file really
            # is the same comparison before returning it.
            existing_bytes = path.read_bytes()
            try:
                existing = ComparisonReport.model_validate_json(existing_bytes)
            except (ValueError, TypeError) as exc:
                raise ComparisonStoreError(
                    f"existing report file is corrupt: {path}") from exc
            if existing.comparison_id != final.comparison_id:
                raise ComparisonStoreError(
                    f"comparison_id collision at {path}: "
                    f"{existing.comparison_id} != {final.comparison_id}")
            payload = existing_bytes
        except OSError as exc:
            raise ComparisonStoreError(f"cannot write report {path}: {exc}") from exc

        artifact = ReportArtifact(
            path=str(path), comparison_id=final.comparison_id,
            content_sha256=digest, file_sha256=sha256_bytes(payload),
            created_new=created_new,
        )
        self._update_index(directory, cid, final, artifact)
        return artifact

    def _update_index(self, directory: Path, campaign_id: str,
                      report: ComparisonReport,
                      artifact: ReportArtifact) -> None:
        index_path = directory / "index.json"
        index: dict[str, Any] = {"campaign_id": campaign_id, "reports": {}}
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                raise ComparisonStoreError(f"unreadable comparison index: {exc}") from exc
        reports = index.setdefault("reports", {})
        bucket = reports.setdefault(report.candidate_id, [])
        if not any(item.get("comparison_id") == artifact.comparison_id
                   for item in bucket):
            bucket.append({
                "comparison_id": artifact.comparison_id,
                "algorithm_version": (report.comparison_algorithm_version
                                      or COMPARISON_ALGORITHM_VERSION),
                "decision": report.decision,
                "created_at": report.created_at,
                "file": Path(artifact.path).name,
                "content_sha256": artifact.content_sha256,
                "file_sha256": artifact.file_sha256,
            })
        versions = sorted({str(v) for v in index.get("algorithm_versions_seen", [])}
                         | {str(report.comparison_algorithm_version
                                or COMPARISON_ALGORITHM_VERSION)})
        index["algorithm_versions_seen"] = versions
        tmp = directory / (index_path.name + ".tmp")
        try:
            tmp.write_text(json.dumps(index, indent=2, ensure_ascii=False,
                                      sort_keys=True), encoding="utf-8")
            tmp.replace(index_path)
        except OSError as exc:
            raise ComparisonStoreError(f"cannot update comparison index: {exc}") from exc

    # -------------------------------------------------------------- reads
    def list_reports(self, campaign_id: str,
                     candidate_id: str | None = None) -> list[dict[str, Any]]:
        directory = self._comparisons_dir(campaign_id, create=False)
        index_path = directory / "index.json"
        if not index_path.is_file():
            return []
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise ComparisonStoreError(f"unreadable comparison index: {exc}") from exc
        reports = index.get("reports") or {}
        if candidate_id is not None:
            return list(reports.get(candidate_id) or [])
        return [item for bucket in reports.values() for item in bucket]

    def latest(self, campaign_id: str, candidate_id: str) -> dict[str, Any] | None:
        items = self.list_reports(campaign_id, candidate_id)
        return items[-1] if items else None

    def load(self, campaign_id: str,
             comparison_id: str) -> ComparisonReport | None:
        for item in self.list_reports(campaign_id):
            if item.get("comparison_id") != comparison_id:
                continue
            path = (self._comparisons_dir(campaign_id, create=False)
                    / str(item["file"]))
            if not path.is_file():
                raise ComparisonStoreError(f"indexed report missing: {path}")
            data = path.read_bytes()
            if sha256_bytes(data) != item.get("file_sha256"):
                raise ComparisonStoreError(
                    f"indexed report {path} failed its SHA-256 check")
            return ComparisonReport.model_validate_json(data)
        return None
