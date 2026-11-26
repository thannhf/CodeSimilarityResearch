
from __future__ import annotations

from src.transformations.base import (
    Transformation,
    TransformationResult,
)


class CommentsTransformation(Transformation):
    """Safely remove C++ comments while preserving literals."""

    @property
    def name(self) -> str:
        return "comments"

    def transform(
        self,
        source: str,
        seed: int | None = None,
    ) -> TransformationResult:
        # Conservative handling of C++ line splicing.
        # We skip transformation rather than risk changing comment semantics.
        if "\\\n" in source or "\\\r\n" in source:
            return TransformationResult(
                transformation_type=self.name,
                source_text=source,
                transformed_text=source,
                changed=False,
                config={
                    "strategy": "lexical_comment_removal",
                    "seed": seed,
                    "skipped_reason": "line_splicing_detected",
                },
            )

        output: list[str] = []
        i = 0
        n = len(source)

        while i < n:
            # Raw string literal: R"delimiter(... )delimiter"
            try:
                raw_end = self._raw_string_end(source, i)
            except ValueError:
                return TransformationResult(
                    transformation_type=self.name,
                    source_text=source,
                    transformed_text=source,
                    changed=False,
                    config={
                        "strategy": "lexical_comment_removal",
                        "seed": seed,
                        "skipped_reason": "malformed_raw_string_literal",
                    },
                )
            if raw_end is not None:
                output.append(source[i:raw_end])
                i = raw_end
                continue

            # Normal string literal.
            if source[i] == '"':
                end = self._quoted_end(source, i, '"')
                output.append(source[i:end])
                i = end
                continue

            # Character literal.
            if source[i] == "'":
                end = self._quoted_end(source, i, "'")
                output.append(source[i:end])
                i = end
                continue

            # Single-line comment.
            if source.startswith("//", i):
                end = source.find("\n", i)

                if end == -1:
                    end = n

                output.append(self._replace_comment_with_spaces(source[i:end]))
                i = end
                continue

            # Block comment.
            if source.startswith("/*", i):
                end = source.find("*/", i + 2)

                if end == -1:
                    # Fail closed for an unterminated block comment.
                    return TransformationResult(
                        transformation_type=self.name,
                        source_text=source,
                        transformed_text=source,
                        changed=False,
                        config={
                            "strategy": "lexical_comment_removal",
                            "seed": seed,
                            "skipped_reason": "unterminated_block_comment",
                        },
                    )

                end += 2
                comment = source[i:end]
                output.append(self._replace_comment_with_spaces(comment))
                i = end
                continue

            output.append(source[i])
            i += 1

        transformed = "".join(output)

        return TransformationResult(
            transformation_type=self.name,
            source_text=source,
            transformed_text=transformed,
            changed=(source != transformed),
            config={
                "strategy": "lexical_comment_removal",
                "removed_comment_style": [
                    "line",
                    "block",
                ],
                "preserve_newlines": True,
                "seed": seed,
            },
        )

    @staticmethod
    def _replace_comment_with_spaces(comment: str) -> str:
        """Replace comment content with spaces while preserving newlines."""
        return "".join(
            char if char in "\r\n" else " "
            for char in comment
        )

    @staticmethod
    def _quoted_end(
        source: str,
        start: int,
        quote: str,
    ) -> int:
        """Return position after a quoted literal, honoring escapes."""
        i = start + 1
        n = len(source)

        while i < n:
            if source[i] == "\\":
                i += 2
                continue

            if source[i] == quote:
                return i + 1

            i += 1

        # Unterminated literal: preserve remaining source unchanged.
        return n

    @staticmethod
    def _raw_string_end(
        source: str,
        start: int,
    ) -> int | None:
        """Detect prefixed and unprefixed C++ raw string literals."""
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

        if len(delimiter) > 16:
            raise ValueError("malformed_raw_string_literal")

        if any(
            char in " ()\\\t\r\n" for char in delimiter
        ):
            raise ValueError("malformed_raw_string_literal")

        closing = ")" + delimiter + '"'
        close_start = source.find(closing, open_paren + 1)

        if close_start == -1:
            raise ValueError("malformed_raw_string_literal")

        return close_start + len(closing)
