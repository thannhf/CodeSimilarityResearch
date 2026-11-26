
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from src.transformations.comments import CommentsTransformation


class SimilarityScreeningStatus(str, Enum):
    PASS = "PASS"
    REJECT = "REJECT"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SimilarityScreeningEvidence:
    sample_a_id: str | None
    sample_b_id: str | None
    metric: str
    token_mode: str
    score: float | None
    threshold: float
    passed: bool
    status: SimilarityScreeningStatus
    method: str
    unique_tokens_a: int
    unique_tokens_b: int
    intersection_count: int
    union_count: int
    error_reason: str | None = None


class CppTokenizer:
    """Conservative C++ lexical-token tokenizer for similarity screening."""

    _OPERATORS = tuple(
        sorted(
            {
                "%:%:",
                ">>=",
                "<<=",
                "->*",
                "...",
                "<=>",
                "[[",
                "]]",
                ".*",
                "##",
                "::",
                "->",
                "++",
                "--",
                "<<",
                ">>",
                "<=",
                ">=",
                "==",
                "!=",
                "&&",
                "||",
                "*=",
                "/=",
                "%=",
                "+=",
                "-=",
                "&=",
                "^=",
                "|=",
                "<:",
                ":>",
                "<%",
                "%>",
                "%:",
            },
            key=len,
            reverse=True,
        )
    )

    _SINGLE_CHAR_TOKENS = set(
        "{}[]()<>+-*/%&|^~!=?:;,.#"
    )

    _PREFIXES = ("u8", "u", "U", "L")

    def tokenize(self, source: str) -> list[str]:
        if not isinstance(source, str):
            raise TypeError("source_must_be_string")

        tokens: list[str] = []
        i = 0
        n = len(source)

        while i < n:
            char = source[i]

            if char.isspace():
                i += 1
                continue

            raw_end = self._raw_string_end(source, i)
            if raw_end is not None:
                tokens.append(source[i:raw_end])
                i = raw_end
                continue

            literal_end = self._quoted_literal_end(source, i)
            if literal_end is not None:
                tokens.append(source[i:literal_end])
                i = literal_end
                continue

            if self._is_identifier_start(char):
                end = self._identifier_end(source, i)
                tokens.append(source[i:end])
                i = end
                continue

            if char.isdigit() or (
                char == "."
                and i + 1 < n
                and source[i + 1].isdigit()
            ):
                end = self._number_end(source, i)
                tokens.append(source[i:end])
                i = end
                continue

            operator = self._match_operator(source, i)
            if operator is not None:
                tokens.append(operator)
                i += len(operator)
                continue

            if char in self._SINGLE_CHAR_TOKENS:
                tokens.append(char)
                i += 1
                continue

            raise ValueError(
                f"unsupported_character_at_offset_{i}"
            )

        return tokens

    @staticmethod
    def _is_identifier_start(char: str) -> bool:
        return char == "_" or char.isascii() and char.isalpha()

    @staticmethod
    def _is_identifier_continue(char: str) -> bool:
        return (
            char == "_"
            or char.isascii() and char.isalnum()
        )

    @classmethod
    def _identifier_end(cls, source: str, start: int) -> int:
        i = start + 1

        while (
            i < len(source)
            and cls._is_identifier_continue(source[i])
        ):
            i += 1

        return i

    @classmethod
    def _number_end(cls, source: str, start: int) -> int:
        """Conservatively consume preprocessing-number-like tokens."""
        i = start
        n = len(source)

        while i < n:
            char = source[i]

            if char.isascii() and (
                char.isalnum() or char == "_" or char == "."
            ):
                i += 1
                continue

            if char in "+-" and i > start:
                previous = source[i - 1]

                if previous in "eEpP":
                    i += 1
                    continue

            if char == "'" and i > start:
                # C++ digit separator.
                i += 1
                continue

            break

        return i

    @classmethod
    def _match_operator(
        cls,
        source: str,
        start: int,
    ) -> str | None:
        for operator in cls._OPERATORS:
            if source.startswith(operator, start):
                return operator

        return None

    @classmethod
    def _quoted_literal_end(
        cls,
        source: str,
        start: int,
    ) -> int | None:
        n = len(source)

        for prefix in ("u8", "u", "U", "L", ""):
            candidate = start + len(prefix)

            if candidate >= n:
                continue

            quote = source[candidate]

            if quote not in ('"', "'"):
                continue

            if prefix and source[start:candidate] != prefix:
                continue

            i = candidate + 1

            while i < n:
                if source[i] == "\\":
                    i += 1

                    if i >= n:
                        raise ValueError(
                            "unterminated_escape_sequence"
                        )

                    i += 1
                    continue

                if source[i] == quote:
                    return i + 1

                if source[i] in "\r\n":
                    raise ValueError(
                        "newline_in_quoted_literal"
                    )

                i += 1

            raise ValueError("unterminated_quoted_literal")

        return None

    @staticmethod
    def _raw_string_end(
        source: str,
        start: int,
    ) -> int | None:
        prefixes = ("u8R", "uR", "UR", "LR", "R")

        prefix = next(
            (
                item
                for item in prefixes
                if source.startswith(item + '"', start)
            ),
            None,
        )

        if prefix is None:
            return None

        delimiter_start = start + len(prefix) + 1
        open_paren = source.find("(", delimiter_start)

        if open_paren == -1:
            raise ValueError("malformed_raw_string_literal")

        delimiter = source[delimiter_start:open_paren]

        if (
            len(delimiter) > 16
            or any(
                char in " ()\\\t\r\n"
                for char in delimiter
            )
        ):
            raise ValueError("invalid_raw_string_delimiter")

        closing = ")" + delimiter + '"'
        close_start = source.find(
            closing,
            open_paren + 1,
        )

        if close_start == -1:
            raise ValueError("unterminated_raw_string_literal")

        return close_start + len(closing)


class TokenJaccardScreener:
    """Fail-closed unique-token Jaccard similarity screener."""

    METRIC = "token_jaccard"
    TOKEN_MODE = "cpp_lexical_unique_tokens"
    METHOD = "cpp_lexical_unique_token_jaccard_v1"

    def __init__(
        self,
        threshold: float = 0.85,
    ) -> None:
        if isinstance(threshold, bool) or not isinstance(
            threshold, (int, float)
        ):
            raise TypeError("threshold_must_be_numeric")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold_out_of_range")

        self.threshold = float(threshold)
        self._tokenizer = CppTokenizer()
        self._comments = CommentsTransformation()

    def screen(
        self,
        sample_a: Any,
        sample_b: Any,
    ) -> SimilarityScreeningEvidence:
        sample_a_id = self._sample_id(sample_a)
        sample_b_id = self._sample_id(sample_b)

        try:
            source_a = self._source(sample_a)
            source_b = self._source(sample_b)

            normalized_a = self._remove_comments(source_a)
            normalized_b = self._remove_comments(source_b)

            tokens_a = self._tokenizer.tokenize(normalized_a)
            tokens_b = self._tokenizer.tokenize(normalized_b)

            unique_a = set(tokens_a)
            unique_b = set(tokens_b)

            if not unique_a or not unique_b:
                return self._error(
                    sample_a_id,
                    sample_b_id,
                    "empty_token_set",
                    len(unique_a),
                    len(unique_b),
                )

            intersection = unique_a & unique_b
            union = unique_a | unique_b

            score = len(intersection) / len(union)

            passed = score < self.threshold

            return SimilarityScreeningEvidence(
                sample_a_id=sample_a_id,
                sample_b_id=sample_b_id,
                metric=self.METRIC,
                token_mode=self.TOKEN_MODE,
                score=score,
                threshold=self.threshold,
                passed=passed,
                status=(
                    SimilarityScreeningStatus.PASS
                    if passed
                    else SimilarityScreeningStatus.REJECT
                ),
                method=self.METHOD,
                unique_tokens_a=len(unique_a),
                unique_tokens_b=len(unique_b),
                intersection_count=len(intersection),
                union_count=len(union),
            )

        except Exception as exc:
            return self._error(
                sample_a_id,
                sample_b_id,
                type(exc).__name__ + ":" + str(exc),
            )

    def __call__(
        self,
        sample_a: Any,
        sample_b: Any,
    ) -> bool:
        """Boolean-compatible adapter for NegativePairGenerator."""
        evidence = self.screen(sample_a, sample_b)
        return evidence.status is SimilarityScreeningStatus.PASS

    @staticmethod
    def _sample_id(sample: Any) -> str | None:
        value = getattr(sample, "sample_id", None)

        if value is None:
            return None

        return str(value)

    @staticmethod
    def _source(sample: Any) -> str:
        for attribute in (
            "source_text",
            "code",
            "content",
            "text",
        ):
            value = getattr(sample, attribute, None)

            if isinstance(value, str):
                return value

        raise ValueError("sample_source_text_not_found")

    def _remove_comments(self, source: str) -> str:
        result = self._comments.transform(source)

        skipped_reason = result.config.get("skipped_reason")

        if skipped_reason is not None:
            raise ValueError(
                "comment_removal_skipped:" + str(skipped_reason)
            )

        return result.transformed_text

    def _error(
        self,
        sample_a_id: str | None,
        sample_b_id: str | None,
        reason: str,
        unique_tokens_a: int = 0,
        unique_tokens_b: int = 0,
    ) -> SimilarityScreeningEvidence:
        return SimilarityScreeningEvidence(
            sample_a_id=sample_a_id,
            sample_b_id=sample_b_id,
            metric=self.METRIC,
            token_mode=self.TOKEN_MODE,
            score=None,
            threshold=self.threshold,
            passed=False,
            status=SimilarityScreeningStatus.ERROR,
            method=self.METHOD,
            unique_tokens_a=unique_tokens_a,
            unique_tokens_b=unique_tokens_b,
            intersection_count=0,
            union_count=0,
            error_reason=reason,
        )
