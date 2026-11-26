import pytest

from src.data.source_compile_diagnostic_classifier import (
    SourceCompileDiagnosticClassifier,
)


@pytest.fixture
def classifier():
    return SourceCompileDiagnosticClassifier()


def test_classifies_invalid_main_signature(classifier):
    text = "error: '::main' must return 'int'"

    results = classifier.classify(text)

    assert [item.category for item in results] == [
        "INVALID_MAIN_SIGNATURE"
    ]


def test_classifies_missing_declaration(classifier):
    text = "'printf' was not declared in this scope"

    results = classifier.classify(text)

    assert [item.category for item in results] == [
        "MISSING_DECLARATION"
    ]


def test_classifies_multiple_diagnostic_categories(classifier):
    text = (
        "error: '::main' must return 'int'\n"
        "error: 'scanf' was not declared in this scope\n"
        "error: 'printf' was not declared in this scope"
    )

    results = classifier.classify(text)

    categories = {item.category for item in results}

    assert categories == {
        "INVALID_MAIN_SIGNATURE",
        "MISSING_DECLARATION",
    }


def test_classifies_other_compile_error(classifier):
    text = "error: expected ';' before '}' token"

    results = classifier.classify(text)

    assert [item.category for item in results] == [
        "OTHER_COMPILE_ERROR"
    ]


def test_classifies_empty_diagnostic(classifier):
    results = classifier.classify("")

    assert [item.category for item in results] == [
        "NO_DIAGNOSTIC"
    ]


def test_rejects_non_string_diagnostic(classifier):
    with pytest.raises(TypeError):
        classifier.classify(None)


def test_patterns_are_preserved(classifier):
    text = (
        "'::main' must return 'int'; "
        "'cin' was not declared in this scope"
    )

    results = classifier.classify(text)

    assert all(item.matched_patterns for item in results)
