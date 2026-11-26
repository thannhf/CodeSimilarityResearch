import csv

import pytest

from src.data.source_compile_diagnostic_exporter import (
    SourceCompileDiagnosticExporter,
)


FIELDS = [
    "sample_id",
    "problem_id",
    "split",
    "source_path",
    "status",
    "reason_code",
    "message",
    "source_sha256",
]


def write_audit(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_row(sample_id, status, message, problem_id="1", split="train"):
    return {
        "sample_id": sample_id,
        "problem_id": problem_id,
        "split": split,
        "source_path": f"source/{sample_id}.cpp",
        "status": status,
        "reason_code": "SYNTAX_ERROR" if status == "FAIL" else status,
        "message": message,
        "source_sha256": "dummy-hash",
    }


def test_exports_fail_diagnostics_only(tmp_path):
    audit = tmp_path / "audit.csv"
    output = tmp_path / "diagnostics.csv"

    write_audit(
        audit,
        [
            make_row(
                "s1",
                "FAIL",
                "'printf' was not declared in this scope",
            ),
            make_row("s2", "PASS", ""),
            make_row("s3", "ERROR", "Compiler unavailable"),
        ],
    )

    count = SourceCompileDiagnosticExporter().export(audit, output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert count == 1
    assert len(rows) == 1
    assert rows[0]["sample_id"] == "s1"
    assert rows[0]["diagnostic_category"] == "MISSING_DECLARATION"


def test_multi_category_sample_exports_multiple_rows(tmp_path):
    audit = tmp_path / "audit.csv"
    output = tmp_path / "diagnostics.csv"

    write_audit(
        audit,
        [
            make_row(
                "s1",
                "FAIL",
                "'::main' must return 'int'\n"
                "'scanf' was not declared in this scope",
            )
        ],
    )

    count = SourceCompileDiagnosticExporter().export(audit, output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert count == 2
    assert {row["diagnostic_category"] for row in rows} == {
        "INVALID_MAIN_SIGNATURE",
        "MISSING_DECLARATION",
    }
    assert {row["sample_id"] for row in rows} == {"s1"}


def test_empty_fail_diagnostics_are_exported_as_no_diagnostic(tmp_path):
    audit = tmp_path / "audit.csv"
    output = tmp_path / "diagnostics.csv"

    write_audit(audit, [make_row("s1", "FAIL", "")])

    count = SourceCompileDiagnosticExporter().export(audit, output)

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert count == 1
    assert rows[0]["diagnostic_category"] == "NO_DIAGNOSTIC"


def test_duplicate_sample_ids_are_rejected(tmp_path):
    audit = tmp_path / "audit.csv"
    output = tmp_path / "diagnostics.csv"

    write_audit(
        audit,
        [
            make_row("s1", "FAIL", "error A"),
            make_row("s1", "FAIL", "error B"),
        ],
    )

    with pytest.raises(ValueError, match="Duplicate audit sample_id"):
        SourceCompileDiagnosticExporter().export(audit, output)


def test_missing_input_file_is_rejected(tmp_path):
    output = tmp_path / "diagnostics.csv"

    with pytest.raises(FileNotFoundError):
        SourceCompileDiagnosticExporter().export(
            tmp_path / "missing.csv",
            output,
        )


def test_export_is_deterministic(tmp_path):
    audit = tmp_path / "audit.csv"
    output = tmp_path / "diagnostics.csv"

    write_audit(
        audit,
        [
            make_row("s2", "FAIL", "'cout' was not declared in this scope"),
            make_row("s1", "FAIL", "'printf' was not declared in this scope"),
        ],
    )

    exporter = SourceCompileDiagnosticExporter()
    exporter.export(audit, output)
    first_content = output.read_text(encoding="utf-8")

    exporter.export(audit, output)
    second_content = output.read_text(encoding="utf-8")

    assert first_content == second_content
