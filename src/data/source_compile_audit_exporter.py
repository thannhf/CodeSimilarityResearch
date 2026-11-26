from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

from src.data.source_compile_audit_models import (
    SourceCompileAuditResult,
    SourceCompileStatus,
)


class SourceCompileAuditExporter:
    AUDIT_FIELDS = (
        "sample_id",
        "problem_id",
        "split",
        "source_path",
        "status",
        "reason_code",
        "message",
        "source_sha256",
    )

    SUBSET_FIELDS = (
        "sample_id",
        "problem_id",
        "filename",
        "language",
        "source_path",
        "original_path",
        "file_size_bytes",
        "status",
        "split",
        "compile_status",
        "compile_reason_code",
        "source_sha256",
    )

    @staticmethod
    def _atomic_write_csv(
        path: Path,
        fieldnames: tuple[str, ...],
        rows: list[dict],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                writer = csv.DictWriter(
                    temp_file,
                    fieldnames=fieldnames,
                    extrasaction="raise",
                )
                writer.writeheader()
                writer.writerows(rows)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            os.replace(temp_path, path)
            temp_path = None
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def export(
        self,
        results: list[SourceCompileAuditResult],
        metadata_path: Path | str,
        splits_path: Path | str,
        audit_output_path: Path | str,
        valid_subset_output_path: Path | str,
    ) -> tuple[int, int]:
        metadata_path = Path(metadata_path)
        splits_path = Path(splits_path)

        with metadata_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            metadata_rows = list(csv.DictReader(file))

        with splits_path.open(
            "r", encoding="utf-8-sig", newline=""
        ) as file:
            split_rows = list(csv.DictReader(file))

        metadata_by_id = {
            row["sample_id"]: row for row in metadata_rows
            if row.get("status", "").strip().lower() == "success"
        }
        splits_by_id = {
            row["sample_id"]: row["split"]
            for row in split_rows
        }
        results_by_id = {result.sample_id: result for result in results}

        if len(results_by_id) != len(results):
            raise ValueError("Duplicate sample IDs in audit results")

        if set(results_by_id) != set(metadata_by_id):
            raise ValueError("Audit results do not match successful metadata IDs")

        if set(metadata_by_id) != set(splits_by_id):
            raise ValueError("Metadata and split IDs do not match")

        audit_rows = [
            {
                field: getattr(result, field)
                for field in self.AUDIT_FIELDS
            }
            for result in sorted(
                results, key=lambda r: (r.problem_id, r.sample_id)
            )
        ]

        subset_rows = []
        for result in results:
            if result.status != SourceCompileStatus.PASS:
                continue

            metadata = metadata_by_id[result.sample_id]
            split = splits_by_id[result.sample_id]

            if split != result.split:
                raise ValueError(
                    f"Split mismatch for {result.sample_id}"
                )

            subset_rows.append({
                "sample_id": result.sample_id,
                "problem_id": metadata["problem_id"],
                "filename": metadata["filename"],
                "language": metadata["language"],
                "source_path": metadata["source_path"],
                "original_path": metadata.get("original_path", ""),
                "file_size_bytes": metadata["file_size_bytes"],
                "status": metadata["status"],
                "split": split,
                "compile_status": result.status.value,
                "compile_reason_code": result.reason_code,
                "source_sha256": result.source_sha256,
            })

        subset_rows.sort(
            key=lambda row: (row["problem_id"], row["sample_id"])
        )

        self._atomic_write_csv(
            Path(audit_output_path),
            self.AUDIT_FIELDS,
            audit_rows,
        )
        self._atomic_write_csv(
            Path(valid_subset_output_path),
            self.SUBSET_FIELDS,
            subset_rows,
        )

        return len(audit_rows), len(subset_rows)

    def export_audit_only(
        self,
        results: list[SourceCompileAuditResult],
        audit_output_path: Path | str,
    ) -> int:
        results_by_id = {
            result.sample_id: result for result in results
        }

        if len(results_by_id) != len(results):
            raise ValueError("Duplicate sample IDs in audit results")

        audit_rows = [
            {
                field: getattr(result, field)
                for field in self.AUDIT_FIELDS
            }
            for result in sorted(
                results, key=lambda r: (r.problem_id, r.sample_id)
            )
        ]

        self._atomic_write_csv(
            Path(audit_output_path),
            self.AUDIT_FIELDS,
            audit_rows,
        )

        return len(audit_rows)
