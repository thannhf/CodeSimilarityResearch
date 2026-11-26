from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from src.transformations.validation.models import (
    ValidationResult,
    ValidationStatus,
)


class BaselineAwareSyntaxValidator:
    """Conservative baseline-aware C++ syntax validation."""

    def __init__(
        self,
        compiler: str = "clang++",
        standard: str = "c++17",
        timeout_seconds: int = 10,
    ) -> None:
        self.compiler = compiler
        self.standard = standard
        self.timeout_seconds = timeout_seconds

    def _compile(self, source: str) -> bool:
        compiler_path = shutil.which(self.compiler)
        if compiler_path is None:
            raise FileNotFoundError(
                f"Compiler unavailable: {self.compiler}"
            )

        with tempfile.TemporaryDirectory(
            prefix="baseline_aware_v3_"
        ) as temp_dir:
            source_path = Path(temp_dir) / "candidate.cpp"
            source_path.write_text(
                source,
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    compiler_path,
                    f"-std={self.standard}",
                    "-fsyntax-only",
                    "-x",
                    "c++",
                    str(source_path),
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )

            return completed.returncode == 0

    def validate(
        self,
        original: str,
        candidate: str,
    ) -> ValidationResult:
        try:
            baseline_pass = self._compile(original)
            candidate_pass = self._compile(candidate)

        except FileNotFoundError as exc:
            return ValidationResult(
                validator_name="v3_baseline_aware",
                status=ValidationStatus.ERROR,
                reason_codes=("COMPILER_UNAVAILABLE",),
                message=str(exc),
                details={},
            )

        except subprocess.TimeoutExpired:
            return ValidationResult(
                validator_name="v3_baseline_aware",
                status=ValidationStatus.ERROR,
                reason_codes=("VALIDATION_TIMEOUT",),
                message="Baseline-aware compilation timed out.",
                details={},
            )

        except OSError as exc:
            return ValidationResult(
                validator_name="v3_baseline_aware",
                status=ValidationStatus.ERROR,
                reason_codes=("COMPILER_EXECUTION_ERROR",),
                message=str(exc),
                details={},
            )

        details = {
            "baseline_pass": baseline_pass,
            "candidate_pass": candidate_pass,
            "compiler": self.compiler,
            "standard": self.standard,
        }

        if baseline_pass and candidate_pass:
            return ValidationResult(
                validator_name="v3_baseline_aware",
                status=ValidationStatus.PASS,
                reason_codes=(),
                message="Baseline and candidate compile successfully.",
                details=details,
            )

        if baseline_pass and not candidate_pass:
            return ValidationResult(
                validator_name="v3_baseline_aware",
                status=ValidationStatus.FAIL,
                reason_codes=("CANDIDATE_SYNTAX_ERROR",),
                message="Candidate introduces a syntax regression.",
                details=details,
            )

        if not baseline_pass and candidate_pass:
            return ValidationResult(
                validator_name="v3_baseline_aware",
                status=ValidationStatus.PASS,
                reason_codes=("CANDIDATE_IMPROVES_COMPILABILITY",),
                message="Candidate compiles while baseline does not.",
                details=details,
            )

        return ValidationResult(
            validator_name="v3_baseline_aware",
            status=ValidationStatus.ERROR,
            reason_codes=("BASELINE_INVALID_CANDIDATE_INVALID",),
            message=(
                "Both baseline and candidate fail compilation; "
                "preservation is inconclusive."
            ),
            details=details,
        )
