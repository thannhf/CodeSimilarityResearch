from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

from src.data.source_compile_audit_config import (
    SourceCompileAuditConfig,
)
from src.data.source_compile_audit_models import (
    SourceCompileAuditResult,
    SourceCompileStatus,
)


class SourceCompileAuditor:
    def __init__(
        self,
        config: SourceCompileAuditConfig | None = None,
    ) -> None:
        self.config = config or SourceCompileAuditConfig()
        self.compiler_path = shutil.which(self.config.compiler)

    def audit_source(
        self,
        sample_id: str,
        problem_id: str,
        split: str,
        source_path: Path,
        source_text: str,
    ) -> SourceCompileAuditResult:
        digest = hashlib.sha256(
            source_text.encode("utf-8")
        ).hexdigest()

        if self.compiler_path is None:
            return self._result(
                sample_id, problem_id, split, source_path,
                SourceCompileStatus.ERROR,
                "COMPILER_UNAVAILABLE",
                f"Compiler not found: {self.config.compiler}",
                digest,
            )

        try:
            with tempfile.TemporaryDirectory(
                prefix="source_compile_audit_"
            ) as temp_dir:
                candidate = Path(temp_dir) / "source.cpp"
                candidate.write_text(
                    source_text,
                    encoding="utf-8",
                )

                completed = subprocess.run(
                    [
                        self.compiler_path,
                        f"-std={self.config.standard}",
                        "-fsyntax-only",
                        "-x", "c++",
                        str(candidate),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout_seconds,
                    check=False,
                )

                if completed.returncode == 0:
                    return self._result(
                        sample_id, problem_id, split, source_path,
                        SourceCompileStatus.PASS,
                        "COMPILE_PASS",
                        "Source compiled successfully.",
                        digest,
                    )

                diagnostic = (
                    completed.stderr.strip()
                    or completed.stdout.strip()
                    or f"Compiler exited with code {completed.returncode}"
                )

                return self._result(
                    sample_id, problem_id, split, source_path,
                    SourceCompileStatus.FAIL,
                    "SYNTAX_ERROR",
                    diagnostic[:4000],
                    digest,
                )

        except subprocess.TimeoutExpired:
            return self._result(
                sample_id, problem_id, split, source_path,
                SourceCompileStatus.ERROR,
                "COMPILER_TIMEOUT",
                f"Compilation timed out after "
                f"{self.config.timeout_seconds} seconds.",
                digest,
            )
        except OSError as exc:
            return self._result(
                sample_id, problem_id, split, source_path,
                SourceCompileStatus.ERROR,
                "COMPILER_EXECUTION_ERROR",
                str(exc)[:4000],
                digest,
            )

    @staticmethod
    def _result(
        sample_id: str,
        problem_id: str,
        split: str,
        source_path: Path,
        status: SourceCompileStatus,
        reason_code: str,
        message: str,
        digest: str,
    ) -> SourceCompileAuditResult:
        return SourceCompileAuditResult(
            sample_id=sample_id,
            problem_id=problem_id,
            split=split,
            source_path=str(source_path),
            status=status,
            reason_code=reason_code,
            message=message,
            source_sha256=digest,
        )
