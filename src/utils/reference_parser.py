"""Utilities to extract reference snippets from agent responses."""

from __future__ import annotations

import re
from typing import Iterable, List

_REFERENCE_PATTERN = re.compile(r"\[(.+?)\]")


def _clean_reference(reference: str) -> str:
    """Normalise whitespace inside a single reference block."""
    collapsed = re.sub(r"\s+", " ", reference.strip())
    # Normalise commas to have a single trailing space when present.
    collapsed = re.sub(r"\s*,\s*", ", ", collapsed)
    return collapsed


def _iter_references(raw_content: str) -> Iterable[str]:
    """Yield cleaned reference strings that appear inside square brackets."""
    if not raw_content:
        return []
    matches = _REFERENCE_PATTERN.findall(raw_content)
    cleaned: List[str] = []
    for match in matches:
        cleaned_ref = _clean_reference(match)
        if cleaned_ref:
            cleaned.append(cleaned_ref)
    return cleaned


def parse_references(raw_content: str, *, deduplicate: bool = True) -> List[str]:
    """Extract references from ``raw_content`` and return them as a list of strings.

    Parameters
    ----------
    raw_content:
        Full text produced by the agent that may contain references in square brackets
        (for example ``"[Reglamento de Admisión, p8]"``).
    deduplicate:
        When ``True`` (default) repeated references are collapsed preserving their first
        appearance order.

    Returns
    -------
    List[str]
        The extracted references. Returns an empty list when none are found.
    """

    references = _iter_references(raw_content)
    if not references:
        return []

    if not deduplicate:
        return list(references)

    seen = set()
    ordered_unique: List[str] = []
    for ref in references:
        token = ref.casefold()
        if token in seen:
            continue
        seen.add(token)
        ordered_unique.append(ref)

    return ordered_unique


__all__ = ["parse_references"]
