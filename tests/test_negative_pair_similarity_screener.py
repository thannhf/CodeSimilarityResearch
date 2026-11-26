
from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.data.negative_pair_similarity_screener import (
    CppTokenizer,
    SimilarityScreeningStatus,
    TokenJaccardScreener,
)


@dataclass
class Sample:
    sample_id: str
    source_text: str


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------


def test_tokenizer_basic_cpp_tokens():
    tokenizer = CppTokenizer()

    tokens = tokenizer.tokenize(
        "int main(){return a+b;}"
    )

    assert tokens == [
        "int",
        "main",
        "(",
        ")",
        "{",
        "return",
        "a",
        "+",
        "b",
        ";",
        "}",
    ]


def test_tokenizer_uses_longest_match_for_operators():
    tokenizer = CppTokenizer()

    tokens = tokenizer.tokenize(
        "a->b; x >>= 2; y .* z; a::b; ++x;"
    )

    assert "->" in tokens
    assert ">>=" in tokens
    assert ".*" in tokens
    assert "::" in tokens
    assert "++" in tokens


def test_tokenizer_preserves_numeric_literals():
    tokenizer = CppTokenizer()

    tokens = tokenizer.tokenize(
        "1.25e-3 0xFF 42ULL 0b1010"
    )

    assert tokens == [
        "1.25e-3",
        "0xFF",
        "42ULL",
        "0b1010",
    ]


def test_tokenizer_preserves_string_literal():
    tokenizer = CppTokenizer()

    tokens = tokenizer.tokenize(
        'const char* s="//not comment";'
    )

    assert '"//not comment"' in tokens


def test_tokenizer_preserves_character_literal():
    tokenizer = CppTokenizer()

    tokens = tokenizer.tokenize(
        r"char c='\\n';"
    )

    assert r"'\\n'" in tokens


@pytest.mark.parametrize(
    "source",
    [
        'R"tag(/* raw comment */)tag"',
        'u8R"tag(// raw comment)tag"',
        'uR"tag(// raw comment)tag"',
        'UR"tag(/* raw comment */)tag"',
        'LR"tag(/* raw comment */)tag"',
    ],
)
def test_tokenizer_preserves_raw_string_literals(source):
    tokenizer = CppTokenizer()

    tokens = tokenizer.tokenize(source)

    assert len(tokens) == 1
    assert tokens[0] == source


def test_tokenizer_ignores_whitespace():
    tokenizer = CppTokenizer()

    assert tokenizer.tokenize("a+b") == tokenizer.tokenize(
        " a   +   b "
    )


def test_tokenizer_rejects_unterminated_string():
    tokenizer = CppTokenizer()

    with pytest.raises(ValueError, match="unterminated"):
        tokenizer.tokenize('"unfinished')


def test_tokenizer_rejects_unterminated_char_literal():
    tokenizer = CppTokenizer()

    with pytest.raises(ValueError, match="unterminated"):
        tokenizer.tokenize("'x")


def test_tokenizer_rejects_unterminated_raw_string():
    tokenizer = CppTokenizer()

    with pytest.raises(
        ValueError,
        match="unterminated_raw_string_literal",
    ):
        tokenizer.tokenize('R"tag(unfinished')


def test_tokenizer_rejects_unsupported_non_ascii_identifier():
    tokenizer = CppTokenizer()

    with pytest.raises(
        ValueError,
        match="unsupported_character",
    ):
        tokenizer.tokenize("int café = 1;")


# ---------------------------------------------------------------------------
# Screener tests
# ---------------------------------------------------------------------------


def make_sample(
    sample_id: str,
    source: str,
) -> Sample:
    return Sample(
        sample_id=sample_id,
        source_text=source,
    )


def make_screener(
    threshold: float = 0.85,
) -> TokenJaccardScreener:
    return TokenJaccardScreener(
        threshold=threshold,
    )


def test_identical_code_is_rejected():
    screener = make_screener()

    source = "int add(int a,int b){return a+b;}"

    result = screener.screen(
        make_sample("A", source),
        make_sample("B", source),
    )

    assert result.status is SimilarityScreeningStatus.REJECT
    assert result.passed is False
    assert result.score == 1.0


def test_formatting_only_difference_is_rejected():
    screener = make_screener()

    source_a = "int add(int a,int b){return a+b;}"
    source_b = "int add ( int a , int b ) { return a + b ; }"

    result = screener.screen(
        make_sample("A", source_a),
        make_sample("B", source_b),
    )

    assert result.status is SimilarityScreeningStatus.REJECT
    assert result.score == 1.0


def test_comments_only_difference_is_rejected():
    screener = make_screener()

    source_a = "int add(int a,int b){return a+b;}"
    source_b = "int add(int a,int b){/*comment*/return a+b;}"

    result = screener.screen(
        make_sample("A", source_a),
        make_sample("B", source_b),
    )

    assert result.status is SimilarityScreeningStatus.REJECT
    assert result.score == 1.0


def test_different_code_passes_screening():
    screener = make_screener()

    source_a = "int add(int a,int b){return a+b;}"
    source_b = "long multiply(long x,long y){return x*y;}"

    result = screener.screen(
        make_sample("A", source_a),
        make_sample("B", source_b),
    )

    assert result.status is SimilarityScreeningStatus.PASS
    assert result.passed is True
    assert result.score < 0.85


def test_threshold_boundary_is_rejected():
    screener = make_screener(threshold=0.85)

    shared = [f"t{i}" for i in range(17)]

    source_a = " ".join(
        shared + ["uniqueA1", "uniqueA2", "uniqueA3"]
    )
    source_b = " ".join(shared)

    result = screener.screen(
        make_sample("A", source_a),
        make_sample("B", source_b),
    )

    assert result.score == 0.85
    assert result.status is SimilarityScreeningStatus.REJECT
    assert result.passed is False
    assert result.intersection_count == 17
    assert result.union_count == 20


def test_score_below_threshold_passes():
    screener = make_screener(threshold=0.85)

    source_a = "alpha beta gamma delta"
    source_b = "alpha beta"

    result = screener.screen(
        make_sample("A", source_a),
        make_sample("B", source_b),
    )

    assert result.score == 0.5
    assert result.status is SimilarityScreeningStatus.PASS
    assert result.passed is True


def test_comment_markers_inside_string_do_not_change_score():
    screener = make_screener()

    source_a = 'int main(){const char* s="//text";return 0;}'
    source_b = 'int main(){const char* s="//other";return 0;}'

    result = screener.screen(
        make_sample("A", source_a),
        make_sample("B", source_b),
    )

    assert result.status is SimilarityScreeningStatus.REJECT
    assert result.score == pytest.approx(0.875)


def test_comment_markers_inside_prefixed_raw_string_are_preserved():
    screener = make_screener()

    source_a = 'int main(){auto s=u8R"tag(/*text*/)tag";return 0;}'
    source_b = 'int main(){auto s=u8R"tag(/*other*/)tag";return 0;}'

    result = screener.screen(
        make_sample("A", source_a),
        make_sample("B", source_b),
    )

    assert result.status is SimilarityScreeningStatus.REJECT
    assert result.score == pytest.approx(0.8571428571428571)


def test_malformed_literal_fails_closed():
    screener = make_screener()

    result = screener.screen(
        make_sample("A", "int a = 1;"),
        make_sample("B", 'char x = "unterminated;'),
    )

    assert result.status is SimilarityScreeningStatus.ERROR
    assert result.passed is False
    assert result.score is None
    assert result.error_reason is not None


def test_empty_token_set_fails_closed():
    screener = make_screener()

    result = screener.screen(
        make_sample("A", "/* only comment */"),
        make_sample("B", "int x = 1;"),
    )

    assert result.status is SimilarityScreeningStatus.ERROR
    assert result.passed is False
    assert result.score is None
    assert result.error_reason == "empty_token_set"


def test_line_splicing_comment_removal_fails_closed():
    screener = make_screener()

    source_a = "int x = 1;\\\nint y = 2;"
    source_b = "int x = 1; int y = 2;"

    result = screener.screen(
        make_sample("A", source_a),
        make_sample("B", source_b),
    )

    assert result.status is SimilarityScreeningStatus.ERROR
    assert result.passed is False
    assert result.score is None
    assert "line_splicing_detected" in result.error_reason


def test_screening_is_deterministic():
    screener = make_screener()

    sample_a = make_sample(
        "A",
        "int add(int a,int b){return a+b;}",
    )
    sample_b = make_sample(
        "B",
        "long multiply(long x,long y){return x*y;}",
    )

    result1 = screener.screen(sample_a, sample_b)
    result2 = screener.screen(sample_a, sample_b)

    assert result1 == result2


def test_screening_does_not_mutate_inputs():
    screener = make_screener()

    source_a = "int add(int a,int b){return a+b;}"
    source_b = "long multiply(long x,long y){return x*y;}"

    sample_a = make_sample("A", source_a)
    sample_b = make_sample("B", source_b)

    screener.screen(sample_a, sample_b)

    assert sample_a.source_text == source_a
    assert sample_b.source_text == source_b


def test_evidence_contains_correct_counts_and_metadata():
    screener = make_screener()

    result = screener.screen(
        make_sample("sample-a", "int x = 1;"),
        make_sample("sample-b", "int y = 2;"),
    )

    assert result.sample_a_id == "sample-a"
    assert result.sample_b_id == "sample-b"
    assert result.metric == "token_jaccard"
    assert result.token_mode == "cpp_lexical_unique_tokens"
    assert result.method == "cpp_lexical_unique_token_jaccard_v1"
    assert result.threshold == 0.85
    assert result.unique_tokens_a == 5
    assert result.unique_tokens_b == 5
    assert result.intersection_count == 3
    assert result.union_count == 7
    assert result.score == pytest.approx(3 / 7)


def test_boolean_adapter_only_passes_screened_candidates():
    screener = make_screener()

    identical_a = make_sample("A", "int x = 1;")
    identical_b = make_sample("B", "int x = 1;")

    different_a = make_sample("C", "int x = 1;")
    different_b = make_sample("D", "long y = 999;")

    assert screener(identical_a, identical_b) is False
    assert screener(different_a, different_b) is True


def test_invalid_threshold_is_rejected():
    with pytest.raises(ValueError, match="threshold_out_of_range"):
        TokenJaccardScreener(threshold=1.5)


def test_boolean_threshold_is_rejected():
    with pytest.raises(TypeError, match="threshold_must_be_numeric"):
        TokenJaccardScreener(threshold=True)

def test_tokenizer_recognizes_spaceship_operator():
    tokenizer = CppTokenizer()

    assert tokenizer.tokenize("a <=> b") == [
        "a",
        "<=>",
        "b",
    ]


def test_tokenizer_recognizes_cpp_attribute_delimiters():
    tokenizer = CppTokenizer()

    assert tokenizer.tokenize("[[nodiscard]] int f();") == [
        "[[",
        "nodiscard",
        "]]",
        "int",
        "f",
        "(",
        ")",
        ";",
    ]


@pytest.mark.parametrize(
    "source,expected",
    [
        ('"hello"_tag', ['"hello"', "_tag"]),
        ("123_km", ["123_km"]),
    ],
)
def test_tokenizer_preserves_user_defined_literal_suffixes(
    source,
    expected,
):
    tokenizer = CppTokenizer()

    assert tokenizer.tokenize(source) == expected
