from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceCompileDiagnostic:
    category: str
    matched_patterns: tuple[str, ...]


class SourceCompileDiagnosticClassifier:
    _RULES = (
        (
            "INVALID_MAIN_SIGNATURE",
            (
                r"'::main' must return 'int'",
                r"main.*must return.*int",
            ),
        ),
        (
            "MISSING_DECLARATION",
            (
                r"\b(?:was not declared in this scope|"
                r"not declared in this scope)\b",
                r"\bundeclared\b",
            ),
        ),
    )

    def classify(self, diagnostic_text: str) -> tuple[SourceCompileDiagnostic, ...]:
        if not isinstance(diagnostic_text, str):
            raise TypeError("diagnostic_text must be a string")

        results: list[SourceCompileDiagnostic] = []

        for category, patterns in self._RULES:
            matched = tuple(
                pattern
                for pattern in patterns
                if re.search(pattern, diagnostic_text, re.IGNORECASE)
            )

            if matched:
                results.append(
                    SourceCompileDiagnostic(
                        category=category,
                        matched_patterns=matched,
                    )
                )

        if not results and diagnostic_text.strip():
            results.append(
                SourceCompileDiagnostic(
                    category="OTHER_COMPILE_ERROR",
                matched_patterns=(),
                )
            )

        if not diagnostic_text.strip():
            results.append(
                SourceCompileDiagnostic(
                    category="NO_DIAGNOSTIC",
                    matched_patterns=(),
                )
            )

        return tuple(results)
