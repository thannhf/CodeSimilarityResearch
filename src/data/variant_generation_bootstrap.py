from __future__ import annotations

from pathlib import Path
from typing import Any

from src.transformations.validation.baseline_aware_variant_validator import (
    BaselineAwareVariantValidator,
)
from src.transformations.validation.v3_baseline_aware import (
    BaselineAwareSyntaxValidator,
)

from src.data.variant_generation_report import VariantGenerationReport
from src.data.variant_generation_runner import VariantGenerationRunner
from src.data.variant_pipeline_orchestrator import VariantPipelineOrchestrator
from src.data.variant_output_writer import VariantOutputWriter
from src.data.variant_registry import VariantRegistry
from src.data.variant_source_loader import VariantSourceLoader

from src.transformations import (
    CommentsTransformation,
    FormattingTransformation,
    WhitespaceTransformation,
)
from src.transformations.variant_generator import (
    VariantGenerator,
    VariantValidationOutcome,
)
from src.transformations.validation.validator import VariantValidator


TARGET_VARIANTS = 2
MAX_ATTEMPTS_PER_VARIANT = 5

TRANSFORMATION_WEIGHTS = {
    "whitespace": 1 / 3,
    "formatting": 1 / 3,
    "comments": 1 / 3,
}


def build_default_loader(
    *,
    metadata_path: Path,
    splits_path: Path,
    source_root: Path,
) -> VariantSourceLoader:
    return VariantSourceLoader(
        metadata_path=metadata_path,
        splits_path=splits_path,
        source_root=source_root,
    )


def build_default_runner(
    *,
    master_seed: int,
    variants_root: Path,
    registry_path: Path,
    report_path: Path,
) -> VariantGenerationRunner:
    transformations = [
        WhitespaceTransformation(),
        FormattingTransformation(),
        CommentsTransformation(),
    ]

    transformation_by_name = {
        transformation.name: transformation
        for transformation in transformations
    }

    validator = VariantValidator()

    def validation_adapter(
        original: str,
        candidate: str,
        transformation_name: str,
    ) -> VariantValidationOutcome:
        transformation = transformation_by_name.get(
            transformation_name
        )

        if transformation is None:
            return VariantValidationOutcome(
                passed=False,
                reason="UNKNOWN_TRANSFORMATION",
            )

        validation_report = validator.validate(
            variant_id="bootstrap_validation_candidate",
            original_sample_id="bootstrap_validation_sample",
            original=original,
            candidate=candidate,
            transformation=transformation,
            seed=None,
        )

        if validation_report.accepted:
            return VariantValidationOutcome(passed=True)

        return VariantValidationOutcome(
            passed=False,
            reason="VALIDATION_FAILED",
        )

    generator = VariantGenerator(
        transformations=transformations,
        probabilities=TRANSFORMATION_WEIGHTS,
        validator=validation_adapter,
        target_variants=TARGET_VARIANTS,
        max_attempts_per_variant=MAX_ATTEMPTS_PER_VARIANT,
    )

    writer = VariantOutputWriter(variants_root=variants_root)
    registry = VariantRegistry(registry_path=registry_path)

    generation_report = VariantGenerationReport(
        dataset=variants_root.parent.name,
        quota_target=TARGET_VARIANTS,
    )

    orchestrator = VariantPipelineOrchestrator(
        generator=generator,
        writer=writer,
        registry=registry,
        report=generation_report,
        validator_version="V1+V2+V3",
    )

    return VariantGenerationRunner(
        orchestrator=orchestrator,
        master_seed=master_seed,
    )


def build_baseline_aware_runner(
    *,
    master_seed: int,
    variants_root: Path,
    registry_path: Path,
    report_path: Path,
) -> VariantGenerationRunner:
    transformations = [
        WhitespaceTransformation(),
        FormattingTransformation(),
        CommentsTransformation(),
    ]

    transformation_by_name = {
        transformation.name: transformation
        for transformation in transformations
    }

    validator = BaselineAwareVariantValidator(
        v3=BaselineAwareSyntaxValidator(
            compiler="g++",
            standard="c++17",
        )
    )

    def validation_adapter(
        original: str,
        candidate: str,
        transformation_name: str,
    ) -> VariantValidationOutcome:
        transformation = transformation_by_name.get(
            transformation_name
        )

        if transformation is None:
            return VariantValidationOutcome(
                passed=False,
                reason="UNKNOWN_TRANSFORMATION",
            )

        validation_report = validator.validate(
            variant_id="bootstrap_validation_candidate",
            original_sample_id="bootstrap_validation_sample",
            original=original,
            candidate=candidate,
            transformation=transformation,
            seed=None,
        )

        if validation_report.accepted:
            return VariantValidationOutcome(passed=True)

        diagnostics = []

        for label, component in (
            ("V1", validation_report.v1),
            ("V2", validation_report.v2),
            ("V3", validation_report.v3),
        ):
            status = getattr(
                component.status,
                "value",
                str(component.status),
            )

            codes = ",".join(component.reason_codes) or "-"
            message = component.message.strip().replace(
                "`r", " "
            ).replace("`n", " ")

            if len(message) > 160:
                message = message[:160] + "..."

            diagnostics.append(
                f"{label}={status} codes={codes} msg={message or '-'}"
            )

        return VariantValidationOutcome(
            passed=False,
            reason="VALIDATION_FAILED [" + " | ".join(diagnostics) + "]",
        )

    generator = VariantGenerator(
        transformations=transformations,
        probabilities=TRANSFORMATION_WEIGHTS,
        validator=validation_adapter,
        target_variants=TARGET_VARIANTS,
        max_attempts_per_variant=MAX_ATTEMPTS_PER_VARIANT,
    )

    writer = VariantOutputWriter(variants_root=variants_root)
    registry = VariantRegistry(registry_path=registry_path)

    generation_report = VariantGenerationReport(
        dataset=variants_root.parent.name,
        quota_target=TARGET_VARIANTS,
    )

    orchestrator = VariantPipelineOrchestrator(
        generator=generator,
        writer=writer,
        registry=registry,
        report=generation_report,
        validator_version="V1+V2+V3_BASELINE_AWARE",
    )

    return VariantGenerationRunner(
        orchestrator=orchestrator,
        master_seed=master_seed,
    )


