from unittest.mock import Mock

from src.transformations.validation.models import ValidationStatus


def test_baseline_and_candidate_both_compile_pass(monkeypatch):
    from src.transformations.validation.v3_baseline_aware import (
        BaselineAwareSyntaxValidator,
    )
    validator = BaselineAwareSyntaxValidator(compiler="g++")
    monkeypatch.setattr(validator, "_compile", lambda source: True)
    result = validator.validate("int main(){}", "int main(){ return 0; }")
    assert result.status == ValidationStatus.PASS


def test_valid_baseline_broken_candidate_fails(monkeypatch):
    from src.transformations.validation.v3_baseline_aware import (
        BaselineAwareSyntaxValidator,
    )
    validator = BaselineAwareSyntaxValidator(compiler="g++")
    outcomes = iter([True, False])
    monkeypatch.setattr(validator, "_compile", lambda source: next(outcomes))
    result = validator.validate("int main(){}", "int main(){ broken }")
    assert result.status == ValidationStatus.FAIL
    assert "CANDIDATE_SYNTAX_ERROR" in result.reason_codes


def test_invalid_baseline_valid_candidate_passes(monkeypatch):
    from src.transformations.validation.v3_baseline_aware import (
        BaselineAwareSyntaxValidator,
    )
    validator = BaselineAwareSyntaxValidator(compiler="g++")
    outcomes = iter([False, True])
    monkeypatch.setattr(validator, "_compile", lambda source: next(outcomes))
    result = validator.validate("void main(){}", "int main(){ return 0; }")
    assert result.status == ValidationStatus.PASS


def test_invalid_baseline_and_candidate_is_inconclusive(monkeypatch):
    from src.transformations.validation.v3_baseline_aware import (
        BaselineAwareSyntaxValidator,
    )
    validator = BaselineAwareSyntaxValidator(compiler="g++")
    monkeypatch.setattr(validator, "_compile", lambda source: False)
    result = validator.validate("void main(){}", "void main(){}")
    assert result.status == ValidationStatus.ERROR
    assert "BASELINE_INVALID_CANDIDATE_INVALID" in result.reason_codes

def test_compiler_unavailable_returns_error(monkeypatch):
    from src.transformations.validation.v3_baseline_aware import (
        BaselineAwareSyntaxValidator,
    )
    validator = BaselineAwareSyntaxValidator(compiler="missing-compiler")
    monkeypatch.setattr(
        validator,
        "_compile",
        lambda source: (_ for _ in ()).throw(
            FileNotFoundError("compiler unavailable")
        ),
    )
    result = validator.validate("int main(){}", "int main(){}")
    assert result.status == ValidationStatus.ERROR
    assert "COMPILER_UNAVAILABLE" in result.reason_codes


def test_validation_timeout_returns_error(monkeypatch):
    import subprocess
    from src.transformations.validation.v3_baseline_aware import (
        BaselineAwareSyntaxValidator,
    )
    validator = BaselineAwareSyntaxValidator(compiler="g++")
    monkeypatch.setattr(
        validator,
        "_compile",
        lambda source: (_ for _ in ()).throw(
            subprocess.TimeoutExpired("g++", 10)
        ),
    )
    result = validator.validate("int main(){}", "int main(){}")
    assert result.status == ValidationStatus.ERROR
    assert "VALIDATION_TIMEOUT" in result.reason_codes


def test_compiler_execution_error_returns_error(monkeypatch):
    from src.transformations.validation.v3_baseline_aware import (
        BaselineAwareSyntaxValidator,
    )
    validator = BaselineAwareSyntaxValidator(compiler="g++")
    monkeypatch.setattr(
        validator,
        "_compile",
        lambda source: (_ for _ in ()).throw(
            OSError("compiler execution failed")
        ),
    )
    result = validator.validate("int main(){}", "int main(){}")
    assert result.status == ValidationStatus.ERROR
    assert "COMPILER_EXECUTION_ERROR" in result.reason_codes
