import csv

import pytest

from src.data.source_compile_audit_exporter import SourceCompileAuditExporter
from src.data.source_compile_audit_models import (
    SourceCompileAuditResult,
    SourceCompileStatus,
)


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_result(sample_id, split, status):
    return SourceCompileAuditResult(
        sample_id=sample_id,
        problem_id="1",
        split=split,
        source_path=f"source/{sample_id}.cpp",
        status=status,
        reason_code=status.value,
        message=f"{status.value} message",
        source_sha256=f"hash_{sample_id}",
    )


def setup_metadata(tmp_path):
    metadata_path = tmp_path / "samples.csv"
    splits_path = tmp_path / "splits.csv"

    write_csv(metadata_path, [
        {
            "sample_id": "s1", "problem_id": "1",
            "filename": "1.cpp", "language": "cpp",
            "source_path": "source/1.cpp",
            "original_path": "ProgramData/1.cpp",
            "file_size_bytes": "10", "status": "success",
        },
        {
            "sample_id": "s2", "problem_id": "1",
            "filename": "2.cpp", "language": "cpp",
            "source_path": "source/2.cpp",
            "original_path": "ProgramData/2.cpp",
            "file_size_bytes": "20", "status": "success",
        },
        {
            "sample_id": "s3", "problem_id": "1",
            "filename": "3.cpp", "language": "cpp",
            "source_path": "source/3.cpp",
            "original_path": "ProgramData/3.cpp",
            "file_size_bytes": "30", "status": "success",
        },
    ])
    write_csv(splits_path, [
        {"sample_id": "s1", "split": "train"},
        {"sample_id": "s2", "split": "validation"},
        {"sample_id": "s3", "split": "test"},
    ])
    return metadata_path, splits_path


def test_export_pass_only_subset_preserves_metadata_and_splits(tmp_path):
    metadata_path, splits_path = setup_metadata(tmp_path)

    audit_path = tmp_path / "nested" / "audit.csv"
    subset_path = tmp_path / "nested" / "valid.csv"

    results = [
        make_result("s1", "train", SourceCompileStatus.PASS),
        make_result("s2", "validation", SourceCompileStatus.FAIL),
        make_result("s3", "test", SourceCompileStatus.ERROR),
    ]

    audit_count, subset_count = SourceCompileAuditExporter().export(
        results, metadata_path, splits_path, audit_path, subset_path
    )

    audit_rows = read_csv(audit_path)
    subset_rows = read_csv(subset_path)

    assert audit_count == 3
    assert subset_count == 1
    assert {row["sample_id"] for row in audit_rows} == {"s1", "s2", "s3"}
    assert len(subset_rows) == 1

    row = subset_rows[0]
    assert row["sample_id"] == "s1"
    assert row["split"] == "train"
    assert row["compile_status"] == "PASS"
    assert row["source_sha256"] == "hash_s1"
    assert row["original_path"] == "ProgramData/1.cpp"

    assert audit_rows[1]["status"] == "FAIL"
    assert audit_rows[2]["status"] == "ERROR"


def test_export_rejects_duplicate_audit_ids(tmp_path):
    metadata_path, splits_path = setup_metadata(tmp_path)
    results = [
        make_result("s1", "train", SourceCompileStatus.PASS),
        make_result("s1", "train", SourceCompileStatus.PASS),
        make_result("s2", "validation", SourceCompileStatus.FAIL),
        make_result("s3", "test", SourceCompileStatus.ERROR),
    ]

    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        SourceCompileAuditExporter().export(
            results,
            metadata_path,
            splits_path,
            tmp_path / "audit.csv",
            tmp_path / "valid.csv",
        )


def test_export_rejects_split_mismatch(tmp_path):
    metadata_path, splits_path = setup_metadata(tmp_path)
    results = [
        make_result("s1", "test", SourceCompileStatus.PASS),
        make_result("s2", "validation", SourceCompileStatus.FAIL),
        make_result("s3", "test", SourceCompileStatus.ERROR),
    ]

    with pytest.raises(ValueError, match="Split mismatch"):
        SourceCompileAuditExporter().export(
            results,
            metadata_path,
            splits_path,
            tmp_path / "audit.csv",
            tmp_path / "valid.csv",
        )

def test_export_audit_only_does_not_create_valid_subset(tmp_path):
    audit_path = tmp_path / "partial_audit.csv"
    subset_path = tmp_path / "compile_valid_samples.csv"

    results = [
        make_result("s2", "validation", SourceCompileStatus.FAIL),
        make_result("s1", "train", SourceCompileStatus.PASS),
    ]

    count = SourceCompileAuditExporter().export_audit_only(
        results,
        audit_path,
    )

    rows = read_csv(audit_path)

    assert count == 2
    assert [row["sample_id"] for row in rows] == ["s1", "s2"]
    assert rows[0]["status"] == "PASS"
    assert rows[1]["status"] == "FAIL"
    assert not subset_path.exists()
