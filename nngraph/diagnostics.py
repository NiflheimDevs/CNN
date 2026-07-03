"""
nngraph/diagnostics.py

Phase 5: the diagnostic types shared across the whole pipeline.

Deliberately dependency-free -- no ANTLR imports, no nngraph.ast_nodes,
no nngraph.semantic_analyzer. Same reasoning as ast_nodes.py back in
Phase 2: this is a leaf module that everything else depends on, so it
must not depend on anything else in the project. If Diagnostic lived
inside semantic_analyzer.py instead, error_listener.py (Phase 1's
syntax errors) would have to import Phase 3 just to report a parse
failure -- backwards, and it would make semantic_analyzer.py a
dependency of the earliest phase in the pipeline instead of a later
one.
"""

from dataclasses import dataclass
from enum import Enum


class Severity(Enum):
    ERROR = "Error"
    WARNING = "Warning"


@dataclass
class Diagnostic:
    severity: Severity
    line: int | None
    message: str
    hint: str | None = None

    def format(self) -> str:
        """Renders in the exact style Section 7 of the spec uses:
            Error [line 14]: Edge references undefined node 'fc99'.
             Hint: Declared nodes are: [x, fc1, relu1, fc2, out]
        """
        location = f" [line {self.line}]" if self.line is not None else ""
        text = f"{self.severity.value}{location}: {self.message}"
        if self.hint:
            text += f"\n Hint: {self.hint}"
        return text


def has_errors(diagnostics: list[Diagnostic]) -> bool:
    """The one gating rule for the whole pipeline: ERROR blocks
    codegen, WARNING never does. Section 5 of the spec has exactly one
    documented Warning case (orphan nodes) and treats every other rule
    as an Error -- that distinction only means anything if something
    downstream actually branches on it, which is what this is for."""
    return any(d.severity is Severity.ERROR for d in diagnostics)


def sort_diagnostics(diagnostics: list[Diagnostic]) -> list[Diagnostic]:
    """Stable sort by line number, so output reads top-to-bottom the
    way the source file does. Deliberately NOT `key=lambda d: (d.line
    is None, d.line)` -- that tuple form looks natural but breaks: if
    two diagnostics both have line=None, Python has to compare the
    second tuple elements (None < None) to break the tie, which raises
    TypeError. A plain int sentinel sidesteps the issue entirely."""
    return sorted(diagnostics, key=lambda d: -1 if d.line is None else d.line)
