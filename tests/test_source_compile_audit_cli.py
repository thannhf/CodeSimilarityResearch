from src.data import source_compile_audit_cli as cli
from src.data.source_compile_audit_models import SourceCompileAuditSummary


class FakeRunner:
    calls = []
    sample_id_calls = []

    def __init__(self, config):
        self.config = config

    def run(self, limit=None, sample_ids=None):
        self.calls.append(limit)
        if sample_ids is not None:
            self.sample_id_calls.append(list(sample_ids))
        return [], SourceCompileAuditSummary(
            total=0, passed=0, failed=0, errors=0
        )


class FakeExporter:
    audit_only_calls = []
    full_calls = []

    def export_audit_only(self, results, audit_output_path):
        self.audit_only_calls.append(audit_output_path)
        return len(results)

    def export(self, **kwargs):
        self.full_calls.append(kwargs)
        return 0, 0


def reset_fakes():
    FakeRunner.calls = []
    FakeRunner.sample_id_calls = []
    FakeRunner.sample_id_calls = []
    FakeExporter.audit_only_calls = []
    FakeExporter.full_calls = []


def test_cli_without_mode_does_not_run_audit(monkeypatch, capsys):
    reset_fakes()
    monkeypatch.setattr(cli, "SourceCompileAuditRunner", FakeRunner)
    monkeypatch.setattr(cli, "SourceCompileAuditExporter", FakeExporter)

    code = cli.main([])

    assert code == 2
    assert FakeRunner.calls == []
    assert "choose" in capsys.readouterr().out.lower()


def test_cli_partial_exports_audit_only(monkeypatch):
    reset_fakes()
    monkeypatch.setattr(cli, "SourceCompileAuditRunner", FakeRunner)
    monkeypatch.setattr(cli, "SourceCompileAuditExporter", FakeExporter)

    code = cli.main(["--limit", "2"])

    assert code == 0
    assert FakeRunner.calls == [2]
    assert len(FakeExporter.audit_only_calls) == 1
    assert FakeExporter.full_calls == []


def test_cli_full_requires_explicit_flag_and_exports_subset(monkeypatch):
    reset_fakes()
    monkeypatch.setattr(cli, "SourceCompileAuditRunner", FakeRunner)
    monkeypatch.setattr(cli, "SourceCompileAuditExporter", FakeExporter)

    code = cli.main(["--full"])

    assert code == 0
    assert FakeRunner.calls == [None]
    assert FakeExporter.audit_only_calls == []
    assert len(FakeExporter.full_calls) == 1

def test_cli_rejects_zero_limit_without_running(monkeypatch):
    reset_fakes()
    monkeypatch.setattr(cli, "SourceCompileAuditRunner", FakeRunner)

    code = cli.main(["--limit", "0"])

    assert code == 2
    assert FakeRunner.calls == []


def test_cli_full_with_errors_blocks_subset_export(monkeypatch):
    reset_fakes()
    monkeypatch.setattr(cli, "SourceCompileAuditRunner", FakeRunner)
    monkeypatch.setattr(cli, "SourceCompileAuditExporter", FakeExporter)

    original_run = FakeRunner.run

    def run_with_error(self, limit=None):
        FakeRunner.calls.append(limit)
        return [], SourceCompileAuditSummary(
            total=1, passed=0, failed=0, errors=1
        )

    monkeypatch.setattr(FakeRunner, "run", run_with_error)

    code = cli.main(["--full"])

    assert code == 1
    assert FakeRunner.calls == [None]
    assert len(FakeExporter.audit_only_calls) == 1
    assert FakeExporter.full_calls == []

    monkeypatch.setattr(FakeRunner, "run", original_run)


def test_cli_rejects_conflicting_modes(monkeypatch):
    reset_fakes()
    monkeypatch.setattr(cli, "SourceCompileAuditRunner", FakeRunner)

    try:
        cli.main(["--limit", "2", "--full"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected conflicting modes to be rejected")

    assert FakeRunner.calls == []
from pathlib import Path


def test_cli_manifest_uses_exact_ids_and_separate_report(monkeypatch, tmp_path):
    reset_fakes()
    monkeypatch.setattr(cli, "SourceCompileAuditRunner", FakeRunner)
    monkeypatch.setattr(cli, "SourceCompileAuditExporter", FakeExporter)
    monkeypatch.setattr(
        cli,
        "load_manifest_sample_ids",
        lambda *args: ["s1", "s2"],
    )

    manifest = tmp_path / "manifest.csv"
    manifest.write_text("sample_id,problem_id,split\ns1,1,train\ns2,1,test\n")

    code = cli.main(["--manifest", str(manifest)])

    assert code == 0
    assert FakeRunner.sample_id_calls == [["s1", "s2"]]
    assert FakeRunner.calls == [None]
    assert FakeExporter.full_calls == []
    assert FakeExporter.audit_only_calls == [
        cli.MANIFEST_AUDIT_OUTPUT
    ]


def test_manifest_rejects_duplicate_ids(tmp_path):
    manifest = tmp_path / "manifest.csv"
    metadata = tmp_path / "metadata.csv"
    splits = tmp_path / "splits.csv"

    manifest.write_text(
        "sample_id,problem_id,split\ns1,1,train\ns1,1,train\n"
    )
    metadata.write_text("sample_id,problem_id\ns1,1\n")
    splits.write_text("sample_id,problem_id,split\ns1,1,train\n")

    try:
        cli.load_manifest_sample_ids(manifest, metadata, splits)
    except ValueError as exc:
        assert "Duplicate" in str(exc)
    else:
        raise AssertionError("Expected duplicate manifest ID rejection")


def test_manifest_rejects_problem_or_split_mismatch(tmp_path):
    manifest = tmp_path / "manifest.csv"
    metadata = tmp_path / "metadata.csv"
    splits = tmp_path / "splits.csv"

    manifest.write_text(
        "sample_id,problem_id,split\ns1,2,test\n"
    )
    metadata.write_text("sample_id,problem_id\ns1,1\n")
    splits.write_text("sample_id,problem_id,split\ns1,1,train\n")

    try:
        cli.load_manifest_sample_ids(manifest, metadata, splits)
    except ValueError as exc:
        assert "mismatch" in str(exc).lower()
    else:
        raise AssertionError("Expected manifest mismatch rejection")


def test_manifest_rejects_missing_sample_id(tmp_path):
    manifest = tmp_path / "manifest.csv"
    metadata = tmp_path / "metadata.csv"
    splits = tmp_path / "splits.csv"

    manifest.write_text(
        "sample_id,problem_id,split\nunknown,1,train\n"
    )
    metadata.write_text("sample_id,problem_id\n")
    splits.write_text("sample_id,problem_id,split\n")

    try:
        cli.load_manifest_sample_ids(manifest, metadata, splits)
    except ValueError as exc:
        assert "missing" in str(exc).lower()
    else:
        raise AssertionError("Expected unknown ID rejection")

