from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.data.source_compile_audit_config import SourceCompileAuditConfig
from src.data.source_compile_audit_runner import SourceCompileAuditRunner
from src.data.source_compile_audit_exporter import SourceCompileAuditExporter


MANIFEST_AUDIT_OUTPUT = Path(
    "data/processed/poj104/metadata/source_compile_audit_pilot_624.csv"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit POJ104 source compile validity."
    )
    parser.add_argument("--compiler", default="g++")
    parser.add_argument("--standard", default="c++17")
    parser.add_argument("--timeout", type=int, default=10)

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--limit",
        type=int,
        help="Partial audit limit. Only audit report is exported.",
    )
    mode_group.add_argument(
        "--manifest",
        type=Path,
        help="Audit exactly the sample IDs listed in a validated manifest.",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Explicitly authorize full dataset audit and subset export.",
    )
    return parser


def load_manifest_sample_ids(
    manifest_path: Path,
    metadata_path: Path,
    splits_path: Path,
) -> list[str]:
    with manifest_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "problem_id", "split"}

        if reader.fieldnames is None or not required.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError(
                "Manifest must contain sample_id, problem_id, split."
            )

        manifest_rows = list(reader)

    if not manifest_rows:
        raise ValueError("Manifest is empty.")

    metadata_by_id = {}
    with metadata_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "problem_id"}

        if reader.fieldnames is None or not required.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError("Metadata CSV has an invalid schema.")

        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in metadata_by_id:
                raise ValueError(
                    f"Duplicate sample_id in metadata: {sample_id}"
                )
            metadata_by_id[sample_id] = row

    splits_by_id = {}
    with splits_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "problem_id", "split"}

        if reader.fieldnames is None or not required.issubset(
            set(reader.fieldnames)
        ):
            raise ValueError("Splits CSV has an invalid schema.")

        for row in reader:
            sample_id = row["sample_id"]
            if sample_id in splits_by_id:
                raise ValueError(
                    f"Duplicate sample_id in splits: {sample_id}"
                )
            splits_by_id[sample_id] = row

    sample_ids = []
    seen = set()

    for row in manifest_rows:
        sample_id = (row.get("sample_id") or "").strip()
        problem_id = (row.get("problem_id") or "").strip()
        split = (row.get("split") or "").strip()

        if not sample_id or not problem_id or not split:
            raise ValueError(
                "Manifest contains an empty sample_id, problem_id, or split."
            )

        if sample_id in seen:
            raise ValueError(
                f"Duplicate sample_id in manifest: {sample_id}"
            )
        seen.add(sample_id)

        metadata_row = metadata_by_id.get(sample_id)
        split_row = splits_by_id.get(sample_id)

        if metadata_row is None or split_row is None:
            raise ValueError(
                f"Manifest sample_id is missing from source metadata: {sample_id}"
            )

        if metadata_row["problem_id"] != problem_id:
            raise ValueError(
                f"Manifest problem_id mismatch for {sample_id}."
            )

        if (
            split_row["problem_id"] != problem_id
            or split_row["split"] != split
        ):
            raise ValueError(
                f"Manifest split/problem mismatch for {sample_id}."
            )

        sample_ids.append(sample_id)

    return sample_ids


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.full and args.limit is None and args.manifest is None:
        print(
            "ERROR: choose --limit N, --manifest PATH, "
            "or explicitly authorize with --full."
        )
        return 2

    if args.limit is not None and args.limit <= 0:
        print("ERROR: --limit must be positive.")
        return 2

    try:
        config = SourceCompileAuditConfig(
            compiler=args.compiler,
            standard=args.standard,
            timeout_seconds=args.timeout,
        )
    except (TypeError, ValueError) as exc:
        print(f"ERROR: Invalid configuration: {exc}")
        return 2

    is_manifest = args.manifest is not None

    if is_manifest:
        try:
            sample_ids = load_manifest_sample_ids(
                args.manifest,
                config.metadata_path,
                config.splits_path,
            )
        except (OSError, csv.Error, ValueError) as exc:
            print(f"ERROR: Invalid manifest: {exc}")
            return 2

        print(f"Manifest samples validated: {len(sample_ids)}")
        results, summary = SourceCompileAuditRunner(config).run(
            sample_ids=sample_ids
        )
    else:
        results, summary = SourceCompileAuditRunner(config).run(
            limit=args.limit
        )

    is_full = args.full

    print("===== SOURCE COMPILE AUDIT =====")
    if is_full:
        mode = "FULL"
    elif is_manifest:
        mode = "MANIFEST"
    else:
        mode = "PARTIAL"

    print(f"Mode: {mode}")
    print(f"Audited: {summary.total}")
    print(f"PASS: {summary.passed}")
    print(f"FAIL: {summary.failed}")
    print(f"ERROR: {summary.errors}")

    exporter = SourceCompileAuditExporter()

    if not is_full:
        audit_output_path = (
            MANIFEST_AUDIT_OUTPUT if is_manifest
            else config.audit_output_path
        )
        audit_count = exporter.export_audit_only(
            results,
            audit_output_path,
        )
        print(f"Audit CSV records: {audit_count}")
        print(f"Audit report: {audit_output_path}")

        if is_manifest:
            print("Manifest audit: official subset export skipped.")
        else:
            print("Partial audit: official subset export skipped.")

        return 0 if summary.errors == 0 else 1

    if summary.errors > 0:
        audit_count = exporter.export_audit_only(
            results,
            config.audit_output_path,
        )
        print(f"Audit CSV records: {audit_count}")
        print(f"Audit report: {config.audit_output_path}")
        print("Full audit has ERROR results; official subset export blocked.")
        return 1

    audit_count, subset_count = exporter.export(
        results=results,
        metadata_path=config.metadata_path,
        splits_path=config.splits_path,
        audit_output_path=config.audit_output_path,
        valid_subset_output_path=config.valid_subset_output_path,
    )

    print(f"Audit CSV records: {audit_count}")
    print(f"Compile-valid subset: {subset_count}")
    print(f"Audit report: {config.audit_output_path}")
    print(f"Valid subset: {config.valid_subset_output_path}")
    print("FULL AUDIT COMPLETE")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
