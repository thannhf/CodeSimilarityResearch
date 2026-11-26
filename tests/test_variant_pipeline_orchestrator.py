from pathlib import Path
from unittest.mock import Mock

from src.data.variant_pipeline_orchestrator import (
    VariantPipelineOrchestrator,
    VariantPersistenceResult,
)


def test_persistence_result_success():
    result = VariantPersistenceResult(
        variant_id="sample_v001",
        written=True,
        registered=True,
    )

    assert result.success is True


def test_persistence_result_written_but_not_registered():
    result = VariantPersistenceResult(
        variant_id="sample_v001",
        written=True,
        registered=False,
        error="Registry error",
    )

    assert result.success is False
    assert result.written is True
    assert result.registered is False


def test_persistence_result_failed_write():
    result = VariantPersistenceResult(
        variant_id="sample_v001",
        written=False,
        registered=False,
        error="Write error",
    )

    assert result.success is False
    assert result.written is False


def test_orchestrator_calls_generator_and_report():
    generator = Mock()
    generation_result = Mock()
    generation_result.accepted_variants = []
    generator.generate.return_value = generation_result

    writer = Mock()
    registry = Mock()
    report = Mock()

    orchestrator = VariantPipelineOrchestrator(
        generator=generator,
        writer=writer,
        registry=registry,
        report=report,
        validator_version="v1",
    )

    result = orchestrator.process_sample(
        sample_id="sample001",
        problem_id="100",
        split="train",
        original_source="print(1);",
        original_source_path=Path("originals/sample001.cpp"),
        seed=42,
    )

    generator.generate.assert_called_once_with(
        original_sample_id="sample001",
        original_source="print(1);",
        seed=42,
    )

    report.add_result.assert_called_once_with(
        generation_result,
        split="train",
    )

    assert result.generation_result is generation_result
    assert result.persistence_results == []
    writer.write_variant.assert_not_called()
    registry.register.assert_not_called()


def test_orchestrator_rejects_empty_validator_version():
    try:
        VariantPipelineOrchestrator(
            generator=Mock(),
            writer=Mock(),
            registry=Mock(),
            report=Mock(),
            validator_version="",
        )
    except ValueError as exc:
        assert "validator_version" in str(exc)
    else:
        raise AssertionError("Expected ValueError")
def test_persisted_registry_record_preserves_validator_version(tmp_path):
    from src.transformations.variant_generator import GeneratedVariant

    generated_path = tmp_path / "variant.cpp"
    generated_path.write_text("int main() { return 0; }", encoding="utf-8")

    generator = Mock()
    variant = GeneratedVariant(
        variant_id="sample001_v001",
        original_sample_id="sample001",
        variant_index=1,
        transformation_type="whitespace",
        generation_seed=42,
        source_text="int main(){return 0;}",
        transformed_text="int main() { return 0; }",
    )

    generation_result = Mock()
    generation_result.accepted_variants = [variant]
    generator.generate.return_value = generation_result

    writer = Mock()
    writer.write_variant.return_value = generated_path

    registry = Mock()
    registry.compute_sha256.return_value = "test_sha256"
    registry.utc_now_iso.return_value = "2026-01-01T00:00:00Z"

    report = Mock()

    orchestrator = VariantPipelineOrchestrator(
        generator=generator,
        writer=writer,
        registry=registry,
        report=report,
        validator_version="V1+V2+V3_BASELINE_AWARE",
    )

    result = orchestrator.process_sample(
        sample_id="sample001",
        problem_id="100",
        split="train",
        original_source="int main(){return 0;}",
        original_source_path=tmp_path / "original.cpp",
        seed=42,
    )

    assert result.persistence_results[0].success is True

    record = registry.register.call_args.args[0]
    assert record.validator_version == "V1+V2+V3_BASELINE_AWARE"
    assert record.validation_status == "PASS"
    assert record.source_hash == "test_sha256"
def test_e2e_real_generator_writer_registry(tmp_path):
    import csv

    from src.data.variant_output_writer import VariantOutputWriter
    from src.data.variant_registry import (
        VariantRegistry,
        VARIANT_REGISTRY_COLUMNS,
    )
    from src.transformations.base import Transformation
    from src.transformations.whitespace import WhitespaceTransformation
    from src.transformations.variant_generator import (
        VariantGenerator,
        VariantValidationOutcome,
    )

    original_source = "int main() {\n\treturn 0;\n}\n"
    original_path = tmp_path / "original.cpp"
    original_path.write_text(original_source, encoding="utf-8")

    generator = VariantGenerator(
        transformations=[WhitespaceTransformation()],
        probabilities={"whitespace": 1.0},
        validator=lambda original, candidate, transformation: (
            VariantValidationOutcome(passed=True)
        ),
        target_variants=1,
        max_attempts_per_variant=1,
    )

    variants_root = tmp_path / "variants"
    registry_path = tmp_path / "registry.csv"

    writer = VariantOutputWriter(variants_root)
    registry = VariantRegistry(registry_path)

    class Report:
        def __init__(self):
            self.results = []

        def add_result(self, result, *, split):
            self.results.append((result, split))

    report = Report()

    orchestrator = VariantPipelineOrchestrator(
        generator=generator,
        writer=writer,
        registry=registry,
        report=report,
        validator_version="E2E_TEST_V1",
    )

    result = orchestrator.process_sample(
        sample_id="e2e_sample",
        problem_id="100",
        split="train",
        original_source=original_source,
        original_source_path=original_path,
        seed=42,
    )

    assert result.generation_result.accepted_count == 1
    assert result.written_count == 1
    assert result.registered_count == 1
    assert result.failed_count == 0

    variant = result.generation_result.accepted_variants[0]
    output_path = Path(result.persistence_results[0].source_path)

    assert output_path.is_file()
    assert output_path.read_text(encoding="utf-8") == variant.transformed_text
    assert output_path.parent == (variants_root / "100").resolve()

    with registry_path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    row = rows[0]

    assert list(row.keys()) == VARIANT_REGISTRY_COLUMNS
    assert row["variant_id"] == "e2e_sample_v001"
    assert row["original_sample_id"] == "e2e_sample"
    assert row["problem_id"] == "100"
    assert row["split"] == "train"
    assert row["transformation_type"] == "whitespace"
    assert row["validator_version"] == "E2E_TEST_V1"
    assert row["validation_status"] == "PASS"
    assert row["source_hash"] == registry.compute_sha256(output_path)
    assert result.persistence_results[0].success is True
