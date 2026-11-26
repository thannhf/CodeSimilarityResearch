from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Sequence

from src.data.source_compile_audit_config import SourceCompileAuditConfig
from src.data.source_compile_audit_models import (
    SourceCompileAuditResult,
    SourceCompileAuditSummary,
    SourceCompileStatus,
)
from src.data.source_compile_auditor import SourceCompileAuditor


class SourceCompileAuditRunner:
    def __init__(
        self,
        config: SourceCompileAuditConfig | None = None,
        auditor: SourceCompileAuditor | None = None,
    ) -> None:
        self.config = config or SourceCompileAuditConfig()
        self.auditor = auditor or SourceCompileAuditor(self.config)

    @staticmethod
    def _read_csv(path: Path) -> list[dict[str, str]]:
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")

        with path.open("r", encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {path}")
            return list(reader)

    def run(
        self,
        limit: int | None = None,
        sample_ids: Sequence[str] | None = None,
    ) -> tuple[
        list[SourceCompileAuditResult],
        SourceCompileAuditSummary,
    ]:
        if limit is not None and (
            type(limit) is not int or limit <= 0
        ):
            raise ValueError("limit must be a positive integer or None")

        if limit is not None and sample_ids is not None:
            raise ValueError("limit and sample_ids cannot be used together")

        requested_ids: set[str] | None = None
        if sample_ids is not None:
            if isinstance(sample_ids, (str, bytes)):
                raise TypeError("sample_ids must be a sequence of IDs")
            requested_list = list(sample_ids)
            if not requested_list:
                raise ValueError("sample_ids cannot be empty")
            if any(not isinstance(item, str) or not item.strip()
                   for item in requested_list):
                raise ValueError("sample_ids must contain non-empty strings")
            if len(requested_list) != len(set(requested_list)):
                raise ValueError("Duplicate sample_id in requested sample_ids")
            requested_ids = set(requested_list)

        metadata = self._read_csv(self.config.metadata_path)
        splits = self._read_csv(self.config.splits_path)

        metadata_required = {
            "sample_id", "problem_id", "source_path", "status"
        }
        split_required = {"sample_id", "split"}

        if not metadata_required.issubset(
            set(metadata[0].keys()) if metadata else set()
        ):
            raise ValueError(
                "Source metadata is empty or missing required columns"
            )

        if not split_required.issubset(
            set(splits[0].keys()) if splits else set()
        ):
            raise ValueError(
                "Split metadata is empty or missing required columns"
            )

        metadata_ids = [row["sample_id"] for row in metadata]
        split_ids = [row["sample_id"] for row in splits]

        if len(metadata_ids) != len(set(metadata_ids)):
            raise ValueError("Duplicate sample_id in source metadata")

        if len(split_ids) != len(set(split_ids)):
            raise ValueError("Duplicate sample_id in split metadata")

        valid_metadata = {
            row["sample_id"]: row
            for row in metadata
            if row["status"].strip().lower() == "success"
        }
        split_by_id = {
            row["sample_id"]: row["split"].strip()
            for row in splits
        }

        if set(valid_metadata) != set(split_by_id):
            raise ValueError(
                "Metadata/split sample IDs do not match: "
                f"missing_splits={len(set(valid_metadata) - set(split_by_id))}, "
                f"extra_splits={len(set(split_by_id) - set(valid_metadata))}"
            )

        if requested_ids is not None:
            missing_ids = requested_ids - set(valid_metadata)
            if missing_ids:
                preview = sorted(missing_ids)[:10]
                raise ValueError(
                    f"Requested sample_ids not found in successful metadata: "
                    f"{preview}"
                )

        ordered_metadata = sorted(
            valid_metadata.items(),
            key=lambda item: (item[1]["problem_id"], item[0]),
        )

        if requested_ids is not None:
            ordered_metadata = [
                item for item in ordered_metadata
                if item[0] in requested_ids
            ]
        elif limit is not None:
            ordered_metadata = ordered_metadata[:limit]

        results: list[SourceCompileAuditResult] = []

        for sample_id, row in ordered_metadata:
            split = split_by_id[sample_id]
            source_path = (
                self.config.source_root / row["source_path"]
            ).resolve()

            if not source_path.is_relative_to(
                self.config.source_root.resolve()
            ):
                raise ValueError(
                    f"Source path escapes root: {row['source_path']}"
                )

            if split not in {"train", "validation", "test"}:
                raise ValueError(
                    f"Invalid split for {sample_id}: {split!r}"
                )

            try:
                source_text = source_path.read_text(encoding="utf-8")
            except (OSError, UnicodeError) as exc:
                results.append(
                    SourceCompileAuditResult(
                        sample_id=sample_id,
                        problem_id=row["problem_id"],
                        split=split,
                        source_path=str(source_path),
                        status=SourceCompileStatus.ERROR,
                        reason_code="SOURCE_READ_ERROR",
                        message=str(exc)[:4000],
                        source_sha256="",
                    )
                )
                continue

            results.append(
                self.auditor.audit_source(
                    sample_id=sample_id,
                    problem_id=row["problem_id"],
                    split=split,
                    source_path=source_path,
                    source_text=source_text,
                )
            )

        results.sort(key=lambda item: (item.problem_id, item.sample_id))

        summary = SourceCompileAuditSummary(
            total=len(results),
            passed=sum(
                r.status == SourceCompileStatus.PASS for r in results
            ),
            failed=sum(
                r.status == SourceCompileStatus.FAIL for r in results
            ),
            errors=sum(
                r.status == SourceCompileStatus.ERROR for r in results
            ),
        )
        return results, summary
