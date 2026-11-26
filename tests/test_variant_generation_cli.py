
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.data.variant_generation_cli import (
    build_parser,
    validate_cli_args,
    run_cli,
)


def test_parser_defaults():
    parser = build_parser()
    args = parser.parse_args([])

    assert args.dataset_root == Path("data/processed/poj104")
    assert args.split == "train"
    assert args.seed == 42


def test_parser_accepts_custom_arguments():
    parser = build_parser()

    args = parser.parse_args([
        "--dataset-root", "custom_dataset",
        "--split", "validation",
        "--seed", "123",
        "--variants-root", "custom_variants",
        "--registry-path", "custom_registry.csv",
        "--report-path", "custom_report.json",
    ])

    assert args.dataset_root == Path("custom_dataset")
    assert args.split == "validation"
    assert args.seed == 123
    assert args.variants_root == Path("custom_variants")
    assert args.registry_path == Path("custom_registry.csv")
    assert args.report_path == Path("custom_report.json")


def test_parser_accepts_all_split():
    args = build_parser().parse_args(["--split", "all"])
    assert args.split == "all"


def test_parser_rejects_invalid_split():
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--split", "invalid"])


def test_validate_rejects_negative_seed(tmp_path):
    args = build_parser().parse_args([
        "--dataset-root", str(tmp_path),
        "--seed", "-1",
    ])

    with pytest.raises(ValueError, match="seed"):
        validate_cli_args(args)


def test_validate_rejects_missing_dataset_root(tmp_path):
    missing = tmp_path / "missing"

    args = build_parser().parse_args([
        "--dataset-root", str(missing),
    ])

    with pytest.raises(ValueError, match="does not exist"):
        validate_cli_args(args)


def test_validate_rejects_dataset_root_that_is_file(tmp_path):
    file_path = tmp_path / "dataset.csv"
    file_path.write_text("test", encoding="utf-8")

    args = build_parser().parse_args([
        "--dataset-root", str(file_path),
    ])

    with pytest.raises(ValueError, match="not a directory"):
        validate_cli_args(args)


def test_run_cli_filters_requested_split(tmp_path):
    train_sample = SimpleNamespace(
        sample_id="train_1",
        problem_id="1",
        split="train",
        source_text="int main(){}",
        source_path=tmp_path / "train.cpp",
    )

    validation_sample = SimpleNamespace(
        sample_id="val_1",
        problem_id="1",
        split="validation",
        source_text="int main(){}",
        source_path=tmp_path / "val.cpp",
    )

    loader = Mock()
    loader.load.return_value = [
        train_sample,
        validation_sample,
    ]

    loader_factory = Mock(return_value=loader)

    expected_result = SimpleNamespace(
        status="COMPLETED",
        samples_processed=1,
        samples_succeeded=1,
        samples_failed=0,
        variants_written=2,
        variants_registered=2,
    )

    runner = Mock()
    runner.run.return_value = expected_result

    runner_factory = Mock(return_value=runner)

    args = build_parser().parse_args([
        "--dataset-root", str(tmp_path),
        "--split", "train",
        "--seed", "123",
    ])

    result = run_cli(
        args=args,
        runner_factory=runner_factory,
        loader_factory=loader_factory,
    )

    assert result is expected_result
    loader_factory.assert_called_once_with(
        metadata_path=tmp_path / "metadata" / "samples.csv",
        splits_path=tmp_path / "metadata" / "source_splits.csv",
        source_root=tmp_path,
    )
    loader.load.assert_called_once()

    passed_samples = runner.run.call_args.args[0]
    assert len(passed_samples) == 1
    assert passed_samples[0].sample_id == "train_1"

    runner_factory.assert_called_once_with(
        master_seed=123,
        variants_root=args.variants_root,
        registry_path=args.registry_path,
        report_path=args.report_path,
    )


def test_run_cli_all_keeps_all_samples(tmp_path):
    samples = [
        SimpleNamespace(split="train", sample_id="a"),
        SimpleNamespace(split="validation", sample_id="b"),
        SimpleNamespace(split="test", sample_id="c"),
    ]

    loader = Mock()
    loader.load.return_value = samples

    runner = Mock()
    runner.run.return_value = "batch_result"

    args = build_parser().parse_args([
        "--dataset-root", str(tmp_path),
        "--split", "all",
    ])

    result = run_cli(
        args,
        runner_factory=Mock(return_value=runner),
        loader_factory=Mock(return_value=loader),
    )

    assert result == "batch_result"
    assert runner.run.call_args.args[0] == samples

