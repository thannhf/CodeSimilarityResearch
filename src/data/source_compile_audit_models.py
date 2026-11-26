from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SourceCompileStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SourceCompileAuditResult:
    sample_id: str
    problem_id: str
    split: str
    source_path: str
    status: SourceCompileStatus
    reason_code: str
    message: str
    source_sha256: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, SourceCompileStatus):
            raise TypeError("status must be SourceCompileStatus")


@dataclass(frozen=True)
class SourceCompileAuditSummary:
    total: int
    passed: int
    failed: int
    errors: int

    def __post_init__(self) -> None:
        counts = (self.total, self.passed, self.failed, self.errors)

        if any(type(value) is not int or value < 0 for value in counts):
            raise ValueError("Counts must be non-negative integers")

        if self.passed + self.failed + self.errors != self.total:
            raise ValueError("Audit counts must sum to total")
