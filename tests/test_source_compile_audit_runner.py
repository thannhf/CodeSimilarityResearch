import csv

from src.data.source_compile_audit_runner import SourceCompileAuditRunner
from src.data.source_compile_audit_config import SourceCompileAuditConfig
from src.data.source_compile_audit_models import (
    SourceCompileAuditResult,
    SourceCompileStatus,
    SourceCompileAuditSummary,
)


class FakeAuditor:
    def audit_source(self, sample_id, problem_id, split, source_path, source_text):
        status = (
            SourceCompileStatus.PASS
            if "PASS" in source_text
            else SourceCompileStatus.FAIL
        )
        return SourceCompileAuditResult(
            sample_id=sample_id,
            problem_id=problem_id,
            split=split,
            source_path=str(source_path),
            status=status,
            reason_code=status.value,
            message="fake result",
            source_sha256="fakehash",
        )


def write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_runner_streams_samples_and_preserves_splits(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    (root / "a.cpp").write_text("PASS", encoding="utf-8")
    (root / "b.cpp").write_text("FAIL", encoding="utf-8")
    # c.cpp deliberately missing to exercise source-read ERROR.

    metadata_path = tmp_path / "samples.csv"
    splits_path = tmp_path / "splits.csv"

    write_csv(metadata_path, [
        {"sample_id": "s1", "problem_id": "1", "source_path": "a.cpp", "status": "success"},
        {"sample_id": "s2", "problem_id": "2", "source_path": "b.cpp", "status": "success"},
        {"sample_id": "s3", "problem_id": "3", "source_path": "c.cpp", "status": "success"},
    ])
    write_csv(splits_path, [
        {"sample_id": "s1", "split": "train"},
        {"sample_id": "s2", "split": "validation"},
        {"sample_id": "s3", "split": "test"},
    ])

    config = SourceCompileAuditConfig(
        metadata_path=metadata_path,
        splits_path=splits_path,
        source_root=root,
    )
    runner = SourceCompileAuditRunner(config, FakeAuditor())
    results, summary = runner.run()

    assert len(results) == 3
    assert {r.sample_id: r.split for r in results} == {
        "s1": "train",
        "s2": "validation",
        "s3": "test",
    }
    assert {r.sample_id: r.status for r in results} == {
        "s1": SourceCompileStatus.PASS,
        "s2": SourceCompileStatus.FAIL,
        "s3": SourceCompileStatus.ERROR,
    }
    assert summary == SourceCompileAuditSummary(
        total=3, passed=1, failed=1, errors=1
    )


def test_runner_rejects_split_mismatch(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "a.cpp").write_text("PASS", encoding="utf-8")

    metadata_path = tmp_path / "samples.csv"
    splits_path = tmp_path / "splits.csv"

    write_csv(metadata_path, [
        {"sample_id": "s1", "problem_id": "1", "source_path": "a.cpp", "status": "success"},
    ])
    write_csv(splits_path, [
        {"sample_id": "unknown", "split": "train"},
    ])

    config = SourceCompileAuditConfig(
        metadata_path=metadata_path,
        splits_path=splits_path,
        source_root=root,
    )

    runner = SourceCompileAuditRunner(config, FakeAuditor())

    try:
        runner.run()
    except ValueError as exc:
        assert "sample IDs do not match" in str(exc)
    else:
        raise AssertionError("Expected split mismatch to raise ValueError")

def test_runner_limit_is_deterministic_and_respects_limit(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    for name in ("a.cpp", "b.cpp", "c.cpp"):
        (root / name).write_text("PASS", encoding="utf-8")

    metadata_path = tmp_path / "samples.csv"
    splits_path = tmp_path / "splits.csv"

    write_csv(metadata_path, [
        {"sample_id": "s3", "problem_id": "3", "source_path": "c.cpp", "status": "success"},
        {"sample_id": "s1", "problem_id": "1", "source_path": "a.cpp", "status": "success"},
        {"sample_id": "s2", "problem_id": "2", "source_path": "b.cpp", "status": "success"},
    ])
    write_csv(splits_path, [
        {"sample_id": "s3", "split": "test"},
        {"sample_id": "s1", "split": "train"},
        {"sample_id": "s2", "split": "validation"},
    ])

    config = SourceCompileAuditConfig(
        metadata_path=metadata_path,
        splits_path=splits_path,
        source_root=root,
    )
    runner = SourceCompileAuditRunner(config, FakeAuditor())

    results, summary = runner.run(limit=2)

    assert [r.sample_id for r in results] == ["s1", "s2"]
    assert summary == SourceCompileAuditSummary(
        total=2, passed=2, failed=0, errors=0
    )


def test_runner_rejects_invalid_limit(tmp_path):
    runner = SourceCompileAuditRunner(
        SourceCompileAuditConfig(),
        FakeAuditor(),
    )

    for invalid_limit in (0, -1, 1.5, True):
        try:
            runner.run(limit=invalid_limit)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected invalid limit {invalid_limit!r} to raise ValueError"
            )

def test_runner_selects_exact_sample_ids(tmp_path):
    root = tmp_path / "dataset"
    root.mkdir()

    for name in ("a.cpp", "b.cpp", "c.cpp"):
        (root / name).write_text("PASS", encoding="utf-8")

    metadata_path = tmp_path / "samples.csv"
    splits_path = tmp_path / "splits.csv"

    write_csv(metadata_path, [
        {"sample_id": "s1", "problem_id": "1", "source_path": "a.cpp", "status": "success"},
        {"sample_id": "s2", "problem_id": "2", "source_path": "b.cpp", "status": "success"},
        {"sample_id": "s3", "problem_id": "3", "source_path": "c.cpp", "status": "success"},
    ])
    write_csv(splits_path, [
        {"sample_id": "s1", "split": "train"},
        {"sample_id": "s2", "split": "validation"},
        {"sample_id": "s3", "split": "test"},
    ])

    runner = SourceCompileAuditRunner(
        SourceCompileAuditConfig(
            metadata_path=metadata_path,
            splits_path=splits_path,
            source_root=root,
        ),
        FakeAuditor(),
    )

    results, summary = runner.run(sample_ids=["s3", "s1"])

    assert [r.sample_id for r in results] == ["s1", "s3"]
    assert {r.sample_id: r.split for r in results} == {
        "s1": "train",
        "s3": "test",
    }
    assert summary.total == 2


def test_runner_rejects_unknown_sample_ids(tmp_path):
    runner = SourceCompileAuditRunner(
        SourceCompileAuditConfig(),
        FakeAuditor(),
    )

    try:
        runner.run(sample_ids=["unknown"])
    except FileNotFoundError:
        pass
    except ValueError as exc:
        assert "not found" in str(exc)
    else:
        raise AssertionError("Expected unknown sample ID to be rejected")


def test_runner_rejects_duplicate_sample_ids():
    runner = SourceCompileAuditRunner(
        SourceCompileAuditConfig(),
        FakeAuditor(),
    )

    try:
        runner.run(sample_ids=["s1", "s1"])
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Expected duplicate sample IDs to be rejected")


def test_runner_rejects_limit_and_sample_ids_together():
    runner = SourceCompileAuditRunner(
        SourceCompileAuditConfig(),
        FakeAuditor(),
    )

    try:
        runner.run(limit=2, sample_ids=["s1"])
    except ValueError as exc:
        assert "cannot be used together" in str(exc)
    else:
        raise AssertionError("Expected mutually exclusive arguments")


def test_runner_rejects_empty_sample_ids():
    runner = SourceCompileAuditRunner(
        SourceCompileAuditConfig(),
        FakeAuditor(),
    )

    try:
        runner.run(sample_ids=[])
    except ValueError as exc:
        assert "cannot be empty" in str(exc)
    else:
        raise AssertionError("Expected empty sample IDs to be rejected")
