from src.transformations.validation.models import (
    ValidationResult,
    ValidationStatus,
)

class FakeTransformation:
    name = "fake_transformation"


class FakeV1:
    def __init__(self, status=ValidationStatus.PASS):
        self.status = status

    def validate(self, **kwargs):
        return ValidationResult(
            validator_name="v1",
            status=self.status,
        )


class FakeV2:
    def __init__(self, status=ValidationStatus.PASS):
        self.status = status

    def validate(self, **kwargs):
        return ValidationResult(
            validator_name="v2",
            status=self.status,
        )


class FakeV3:
    def __init__(self, status=ValidationStatus.PASS):
        self.status = status
        self.received = None

    def validate(self, original, candidate):
        self.received = (original, candidate)
        return ValidationResult(
            validator_name="v3_baseline_aware",
            status=self.status,
        )


def test_all_validators_pass_returns_accepted_report():
    from src.transformations.validation.baseline_aware_variant_validator import (
        BaselineAwareVariantValidator,
    )

    v3 = FakeV3()
    validator = BaselineAwareVariantValidator(
        v1=FakeV1(),
        v2=FakeV2(),
        v3=v3,
    )

    report = validator.validate(
        variant_id="variant_1",
        original_sample_id="sample_1",
        original="int main(){}",
        candidate="int main(){return 0;}",
        transformation=FakeTransformation(),
        seed=42,
    )

    assert report.accepted
    assert report.overall_status == ValidationStatus.PASS
    assert v3.received == (
        "int main(){}",
        "int main(){return 0;}",
    )


def test_v3_fail_rejects_report():
    from src.transformations.validation.baseline_aware_variant_validator import (
        BaselineAwareVariantValidator,
    )

    validator = BaselineAwareVariantValidator(
        v1=FakeV1(),
        v2=FakeV2(),
        v3=FakeV3(ValidationStatus.FAIL),
    )

    report = validator.validate(
        variant_id="variant_2",
        original_sample_id="sample_1",
        original="int main(){}",
        candidate="broken",
        transformation=FakeTransformation(),
    )

    assert not report.accepted
    assert report.overall_status == ValidationStatus.FAIL
    assert report.v3.status == ValidationStatus.FAIL


def test_v3_error_rejects_report():
    from src.transformations.validation.baseline_aware_variant_validator import (
        BaselineAwareVariantValidator,
    )

    validator = BaselineAwareVariantValidator(
        v1=FakeV1(),
        v2=FakeV2(),
        v3=FakeV3(ValidationStatus.ERROR),
    )

    report = validator.validate(
        variant_id="variant_3",
        original_sample_id="sample_1",
        original="int main(){}",
        candidate="int main(){}",
        transformation=FakeTransformation(),
    )

    assert not report.accepted
    assert report.overall_status == ValidationStatus.FAIL
    assert report.v3.status == ValidationStatus.ERROR

