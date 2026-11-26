from __future__ import annotations

import csv
import json
import os
import tempfile
from pathlib import Path

from src.data.source_compile_diagnostic_classifier import (
    SourceCompileDiagnosticClassifier,
)


class SourceCompileDiagnosticExporter:
    INPUT_FIELDS = {
        "sample_id",
        "problem_id",
        "split",
        "status",
        "message",
    }

    OUTPUT_FIELDS = (
        "sample_id",
        "problem_id",
        "split",
        "diagnostic_category",
        "matched_patterns",
    )

    def __init__(
        self,
        classifier: SourceCompileDiagnosticClassifier | None = None,
    ) -> None:
        self.classifier = (
            classifier or SourceCompileDiagnosticClassifier()
        )

    def export(
        self,
        audit_csv_path: Path,
        output_csv_path: Path,
    ) -> int:
        audit_csv_path = Path(audit_csv_path)
        output_csv_path = Path(output_csv_path)

        if not audit_csv_path.is_file():
            raise FileNotFoundError(audit_csv_path)

        output_csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path: Path | None = None
        exported_count = 0
        seen_sample_ids: set[str] = set()

        try:
            with audit_csv_path.open(
                "r",
                encoding="utf-8-sig",
                newline="",
            ) as source:
                reader = csv.DictReader(source)

                if reader.fieldnames is None:
                    raise ValueError("Audit CSV has no header.")

                missing_fields = (
                    self.INPUT_FIELDS - set(reader.fieldnames)
                )

                if missing_fields:
                    raise ValueError(
                        "Audit CSV missing fields: "
                        + ", ".join(sorted(missing_fields))
                    )

                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    newline="",
                    dir=output_csv_path.parent,
                    prefix=f".{output_csv_path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temp_file:
                    temp_path = Path(temp_file.name)
                    writer = csv.DictWriter(
                        temp_file,
                        fieldnames=self.OUTPUT_FIELDS,
                    )
                    writer.writeheader()

                    for row in reader:
                        sample_id = row["sample_id"]

                        if sample_id in seen_sample_ids:
                            raise ValueError(
                                f"Duplicate audit sample_id: {sample_id}"
                            )

                        seen_sample_ids.add(sample_id)

                        if row["status"] != "FAIL":
                            continue

                        diagnostics = self.classifier.classify(
                            row["message"] or ""
                        )

                        for diagnostic in diagnostics:
                            writer.writerow(
                                {
                                    "sample_id": sample_id,
                                    "problem_id": row["problem_id"],
                                    "split": row["split"],
                                    "diagnostic_category": (
                                        diagnostic.category
                                    ),
                                    "matched_patterns": json.dumps(
                                        diagnostic.matched_patterns,
                                        ensure_ascii=False,
                                    ),
                                }
                            )
                            exported_count += 1

                    temp_file.flush()
                    os.fsync(temp_file.fileno())

            os.replace(temp_path, output_csv_path)
            temp_path = None

            return exported_count

        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
