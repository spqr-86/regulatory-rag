"""Immutable structural norm and lexical tokens shared by offline alignment."""

# ANCHOR: source/kind/number/ordinal → stable parent identity and cached word set.
# Ported from eval/chunking_ab/norms.py; the evaluation copy stays unchanged.

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import cached_property


def words(text: str) -> frozenset[str]:
    return frozenset(re.findall(r"[а-яёa-z0-9]{4,}", text.lower()))


@dataclass(frozen=True)
class Norm:
    doc: str
    kind: str
    number: str
    paragraphs: tuple[str, ...]
    ordinal: int

    @property
    def id(self) -> str:
        return f"{self.doc}::{self.kind}:{self.number}:{self.ordinal}"

    @cached_property
    def text(self) -> str:
        return "\n".join(self.paragraphs)

    @cached_property
    def words(self) -> frozenset[str]:
        return words(self.text)
