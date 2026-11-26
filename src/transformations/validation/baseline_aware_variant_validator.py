from __future__ import annotations

from src.transformations.base import Transformation
from src.transformations.validation.models import (
    ValidationStatus,
    VariantValidationReport,
)
from src.transformations.validation.v1_structural import (
    StructuralValidator,
)
from src.transformations.validation.v2_transformation import (
    TransformationValidator,
)
from src.transformations.validation.v3_baseline_aware import (
    BaselineAwareSyntaxValidator,
)


class BaselineAwareVariantValidator:
    """Variant validator using baseline-aware V3 syntax validation."""

    def __init__(
        self,
        v1: StructuralValidator | None = None,
        v2: TransformationValidator | None = None,
        v3: BaselineAwareSyntaxValidator | None = None,
    ) -> None:
        self.v1 = v1 or StructuralValidator()
        self.v2 = v2 or TransformationValidator()
        self.v3 = v3 or BaselineAwareSyntaxValidator()

    def validate(
        self,
        variant_id: str,
        original_sample_id: str,
        original: str,
        candidate: str,
        transformation: Transformation,
        seed: int | None = None,
    ) -> VariantValidationReport:
        v1_result = self.v1.validate(
            original=original,
            candidate=candidate,
            variant_id=variant_id,
            original_sample_id=original_sample_id,
            transformation_type=transformation.name,
        )

        v2_result = self.v2.validate(
            original=original,
            candidate=candidate,
            transformation=transformation,
            seed=seed,
        )

        v3_result = self.v3.validate(
            original=original,
            candidate=candidate,
        )

        results = (v1_result, v2_result, v3_result)

        overall_status = (
            ValidationStatus.PASS
            if all(
                result.status == ValidationStatus.PASS
                for result in results
            )
            else ValidationStatus.FAIL
        )

        return VariantValidationReport(
            variant_id=variant_id,
            overall_status=overall_status,
            v1=v1_result,
            v2=v2_result,
            v3=v3_result,
        )
