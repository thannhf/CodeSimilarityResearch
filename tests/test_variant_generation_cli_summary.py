
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.data.variant_generation_cli import main


def test_main_exports_summary_json(tmp_path):
    dataset_root = tmp_path / "poj104"
    dataset_root.mkdir()

    batch_result = SimpleNamespace(
        master_seed=42,
        status="COMPLETED",
        samples_processed=2,
        samples_succeeded=2,
        samples_failed=0,
        variants_written=4,
        variants_registered=4,
        success_rate=1.0,
        errors=[],
    )

    runner = Mock()
    runner.run.return_value = batch_result

    runner_factory = Mock(return_value=runner)

    loader = Mock()
    loader.load.return_value = []

    loader_factory = Mock(return_value=loader)

    report_path = tmp_path / "reports" / "summary.json"

    exit_code = main(
        [
            "--dataset-root", str(dataset_root),
            "--split", "all",
            "--report-path", str(report_path),
        ],
        runner_factory=runner_factory,
        loader_factory=loader_factory,
    )

    assert exit_code == 0
    assert report_path.exists()


def test_main_returns_nonzero_for_partial_run(tmp_path):
    dataset_root = tmp_path / "poj104"
    dataset_root.mkdir()

    batch_result = SimpleNamespace(
        master_seed=42,
        status="PARTIAL",
        samples_processed=2,
        samples_succeeded=1,
        samples_failed=1,
        variants_written=2,
        variants_registered=2,
        success_rate=0.5,
        errors=[
            SimpleNamespace(
                sample_id="bad_sample",
                error_type="ValueError",
                message="Invalid sample",
            )
        ],
    )

    runner = Mock()
    runner.run.return_value = batch_result

    loader = Mock()
    loader.load.return_value = []

    report_path = tmp_path / "summary.json"

    exit_code = main(
        [
            "--dataset-root", str(dataset_root),
            "--report-path", str(report_path),
        ],
        runner_factory=Mock(return_value=runner),
        loader_factory=Mock(return_value=loader),
    )

    assert exit_code == 1
    assert report_path.exists()


def test_main_returns_nonzero_for_failed_run(tmp_path):
    dataset_root = tmp_path / "poj104"
    dataset_root.mkdir()

    batch_result = SimpleNamespace(
        master_seed=42,
        status="FAILED",
        samples_processed=2,
        samples_succeeded=0,
        samples_failed=2,
        variants_written=0,
        variants_registered=0,
        success_rate=0.0,
        errors=[],
    )

    runner = Mock()
    runner.run.return_value = batch_result

    loader = Mock()
    loader.load.return_value = []

    report_path = tmp_path / "summary.json"

    exit_code = main(
        [
            "--dataset-root", str(dataset_root),
            "--report-path", str(report_path),
        ],
        runner_factory=Mock(return_value=runner),
        loader_factory=Mock(return_value=loader),
    )

    assert exit_code == 1
    assert report_path.exists()


def test_main_handles_runner_exception(tmp_path):
    dataset_root = tmp_path / "poj104"
    dataset_root.mkdir()

    runner = Mock()
    runner.run.side_effect = RuntimeError("runner crashed")

    loader = Mock()
    loader.load.return_value = []

    exit_code = main(
        [
            "--dataset-root", str(dataset_root),
            "--report-path", str(tmp_path / "summary.json"),
        ],
        runner_factory=Mock(return_value=runner),
        loader_factory=Mock(return_value=loader),
    )

    assert exit_code == 1
