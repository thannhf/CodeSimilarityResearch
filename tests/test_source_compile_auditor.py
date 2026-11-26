from pathlib import Path
from subprocess import CompletedProcess, TimeoutExpired

from src.data.source_compile_auditor import SourceCompileAuditor
from src.data.source_compile_audit_models import SourceCompileStatus


SOURCE = "int main() { return 0; }\n"


def test_compile_pass(monkeypatch, tmp_path):
    source_path = tmp_path / "source.cpp"
    source_path.write_text(SOURCE, encoding="utf-8")

    monkeypatch.setattr(
        "src.data.source_compile_auditor.shutil.which",
        lambda _: "fake-g++",
    )

    auditor = SourceCompileAuditor()
    monkeypatch.setattr(
        "src.data.source_compile_auditor.subprocess.run",
        lambda *a, **kw: CompletedProcess(a[0], 0, "", ""),
    )

    result = auditor.audit_source(
        "s1", "1", "train", source_path, SOURCE
    )

    assert result.status == SourceCompileStatus.PASS
    assert result.reason_code == "COMPILE_PASS"
    assert source_path.read_text(encoding="utf-8") == SOURCE


def test_compile_fail(monkeypatch, tmp_path):
    source_path = tmp_path / "source.cpp"

    monkeypatch.setattr(
        "src.data.source_compile_auditor.shutil.which",
        lambda _: "fake-g++",
    )
    monkeypatch.setattr(
        "src.data.source_compile_auditor.subprocess.run",
        lambda *a, **kw: CompletedProcess(
            a[0], 1, "", "syntax error"
        ),
    )

    result = SourceCompileAuditor().audit_source(
        "s2", "1", "train", source_path, SOURCE
    )

    assert result.status == SourceCompileStatus.FAIL
    assert result.reason_code == "SYNTAX_ERROR"
    assert "syntax error" in result.message


def test_compiler_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.data.source_compile_auditor.shutil.which",
        lambda _: None,
    )

    result = SourceCompileAuditor().audit_source(
        "s3", "1", "test", tmp_path / "source.cpp", SOURCE
    )

    assert result.status == SourceCompileStatus.ERROR
    assert result.reason_code == "COMPILER_UNAVAILABLE"


def test_compile_timeout(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "src.data.source_compile_auditor.shutil.which",
        lambda _: "fake-g++",
    )
    monkeypatch.setattr(
        "src.data.source_compile_auditor.subprocess.run",
        lambda *a, **kw: (_ for _ in ()).throw(
            TimeoutExpired("fake-g++", 10)
        ),
    )

    result = SourceCompileAuditor().audit_source(
        "s4", "1", "validation", tmp_path / "source.cpp", SOURCE
    )

    assert result.status == SourceCompileStatus.ERROR
    assert result.reason_code == "COMPILER_TIMEOUT"


def test_source_hash_and_original_unchanged(monkeypatch, tmp_path):
    import hashlib

    source_path = tmp_path / "original.cpp"
    source_path.write_text(SOURCE, encoding="utf-8")
    original_bytes = source_path.read_bytes()

    monkeypatch.setattr(
        "src.data.source_compile_auditor.shutil.which",
        lambda _: "fake-g++",
    )
    monkeypatch.setattr(
        "src.data.source_compile_auditor.subprocess.run",
        lambda *a, **kw: CompletedProcess(a[0], 0, "", ""),
    )

    result = SourceCompileAuditor().audit_source(
        "s5", "1", "test", source_path, SOURCE
    )

    assert result.source_sha256 == hashlib.sha256(
        SOURCE.encode("utf-8")
    ).hexdigest()
    assert source_path.read_bytes() == original_bytes
