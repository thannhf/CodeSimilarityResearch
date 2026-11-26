from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceCompileAuditConfig:
    compiler: str = "g++"
    standard: str = "c++17"
    timeout_seconds: int = 10

    metadata_path: Path = Path(
        "data/processed/poj104/metadata/samples.csv"
    )
    splits_path: Path = Path(
        "data/processed/poj104/metadata/source_splits.csv"
    )
    source_root: Path = Path("data/processed/poj104")

    audit_output_path: Path = Path(
        "data/processed/poj104/metadata/source_compile_audit.csv"
    )
    valid_subset_output_path: Path = Path(
        "data/processed/poj104/metadata/compile_valid_samples.csv"
    )

    def __post_init__(self) -> None:
        if not self.compiler.strip():
            raise ValueError("compiler must not be empty")

        if not self.standard.strip():
            raise ValueError("standard must not be empty")

        if (
            type(self.timeout_seconds) is not int
            or self.timeout_seconds <= 0
        ):
            raise ValueError(
                "timeout_seconds must be a positive integer"
            )

        for field_name in (
            "metadata_path",
            "splits_path",
            "source_root",
            "audit_output_path",
            "valid_subset_output_path",
        ):
            value = getattr(self, field_name)

            if not isinstance(value, Path):
                raise TypeError(
                    f"{field_name} must be a Path"
                )
