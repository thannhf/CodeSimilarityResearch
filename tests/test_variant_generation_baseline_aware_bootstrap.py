from src.data.variant_generation_bootstrap import (
    build_baseline_aware_runner,
    build_default_runner,
)


def test_baseline_aware_runner_uses_distinct_protocol_label(tmp_path):
    runner = build_baseline_aware_runner(
        master_seed=42,
        variants_root=tmp_path / "variants",
        registry_path=tmp_path / "registry.csv",
        report_path=tmp_path / "report.json",
    )

    assert runner.master_seed == 42
    assert (
        runner.orchestrator.validator_version
        == "V1+V2+V3_BASELINE_AWARE"
    )


def test_default_runner_keeps_strict_protocol_label(tmp_path):
    runner = build_default_runner(
        master_seed=42,
        variants_root=tmp_path / "variants",
        registry_path=tmp_path / "registry.csv",
        report_path=tmp_path / "report.json",
    )

    assert runner.master_seed == 42
    assert runner.orchestrator.validator_version == "V1+V2+V3"

