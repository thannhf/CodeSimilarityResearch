
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Sequence

from datetime import datetime, timezone
from src.data.variant_generation_bootstrap import (
    build_default_loader,
    build_default_runner,
)

from src.data.variant_generation_run_summary import (
    VariantGenerationRunSummaryBuilder,
    VariantGenerationRunSummaryExporter,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate code variants for the POJ104 dataset."
    )

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/processed/poj104"),
        help="Root directory of the processed dataset.",
    )

    parser.add_argument(
        "--split",
        choices=["train", "validation", "test", "all"],
        default="train",
        help="Dataset split to process.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed.",
    )

    parser.add_argument(
        "--variants-root",
        type=Path,
        default=Path("data/processed/poj104/variants"),
        help="Directory for generated variant source files.",
    )

    parser.add_argument(
        "--registry-path",
        type=Path,
        default=Path("data/processed/poj104/metadata/variant_registry.csv"),
        help="Path to the variant registry CSV.",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=Path("data/pairs/reports/variant_generation_report.json"),
        help="Path for the generation report JSON.",
    )

    return parser


def validate_cli_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be >= 0.")

    if not args.dataset_root.exists():
        raise ValueError(
            f"Dataset root does not exist: {args.dataset_root}"
        )

    if not args.dataset_root.is_dir():
        raise ValueError(
            f"Dataset root is not a directory: {args.dataset_root}"
        )


def run_cli(
    args: argparse.Namespace,
    runner_factory: Callable[..., Any],
    loader_factory: Callable[..., Any],
) -> Any:
    validate_cli_args(args)

    metadata_path = args.dataset_root / "metadata" / "samples.csv"
    splits_path = args.dataset_root / "metadata" / "source_splits.csv"
    source_root = args.dataset_root

    loader = loader_factory(
        metadata_path=metadata_path,
        splits_path=splits_path,
        source_root=source_root,
    )

    samples = loader.load()

    if args.split != "all":
        samples = [
            sample for sample in samples
            if sample.split == args.split
        ]

    runner = runner_factory(
        master_seed=args.seed,
        variants_root=args.variants_root,
        registry_path=args.registry_path,
        report_path=args.report_path,
    )

    result = runner.run(samples)
    return result



def main(
    argv: Sequence[str] | None = None,
    *,
    runner_factory: Callable[..., Any] | None = None,
    loader_factory: Callable[..., Any] | None = None,
) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if runner_factory is None:
        runner_factory = build_default_runner

    if loader_factory is None:
        loader_factory = build_default_loader

    started_at = datetime.now(timezone.utc).isoformat()

    try:
        result = run_cli(
            args=args,
            runner_factory=runner_factory,
            loader_factory=loader_factory,
        )

        finished_at = datetime.now(timezone.utc).isoformat()

        summary = VariantGenerationRunSummaryBuilder.build(
            result,
            dataset=args.dataset_root.name,
            split=args.split,
            started_at=started_at,
            finished_at=finished_at,
        )

        VariantGenerationRunSummaryExporter.export_json(
            summary,
            args.report_path,
        )

    except Exception as exc:
        print(f"Variant generation run failed: {exc}")
        return 1

    print("=" * 60)
    print("VARIANT GENERATION RUN SUMMARY")
    print("=" * 60)
    print(f"Dataset: {summary.dataset}")
    print(f"Split: {summary.split}")
    print(f"Master seed: {summary.master_seed}")
    print(f"Status: {summary.status}")
    print(f"Samples processed: {summary.samples_processed}")
    print(f"Samples succeeded: {summary.samples_succeeded}")
    print(f"Samples failed: {summary.samples_failed}")
    print(f"Success rate: {summary.success_rate:.2%}")
    print(f"Variants written: {summary.variants_written}")
    print(f"Variants registered: {summary.variants_registered}")
    print(f"Errors: {summary.error_count}")
    print(f"Started at: {summary.started_at}")
    print(f"Finished at: {summary.finished_at}")
    print(f"Report: {args.report_path}")
    print("=" * 60)

    if summary.errors:
        print("SAMPLE ERRORS:")
        for error in summary.errors[:10]:
            print(
                f"- {error['sample_id']} | "
                f"{error['error_type']} | "
                f"{error['message']}"
            )

        if summary.error_count > 10:
            print(
                f"... and {summary.error_count - 10} more errors."
            )

    return 0 if summary.status == "COMPLETED" else 1
