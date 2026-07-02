"""
AST node definitions for the NNGraph DSL.
Intended location in the project layout: src/nngraph/ast_nodes.py

These are plain dataclasses, deliberately decoupled from ANTLR's
generated ParserRuleContext types. Every phase after ast_builder.py
(semantic analysis, codegen) operates exclusively on these types and
never touches an ANTLR context object again. That boundary means
Phase 3+ can be unit-tested by hand-constructing these dataclasses
directly, with zero dependency on the parser ever running.

Literal DSL values (int, float, bool, string, shape-tuple, None) are
represented with their native Python equivalents rather than a custom
Value wrapper class. Python's own types already disambiguate all five
DSL data types from Section 3.2 unambiguously -- a shape is a tuple,
a flag is a bool, an absent value is None -- so a wrapper class would
just be ceremony around information Python already carries.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

# A node parameter or config entry value is always exactly one of these.
Value = Union[int, float, bool, str, tuple[int, ...], None]


@dataclass
class InputDecl:
    name: str
    shape: tuple[int, ...]
    line: int


@dataclass
class Model:
    name: str
    input: InputDecl
    output: str  # id of the output node, resolved against Graph.nodes later
    line: int


@dataclass
class GraphNode:
    id: str
    layer_type: str
    params: dict[str, Value] = field(default_factory=dict)
    line: int = 0


@dataclass
class Edge:
    src: str
    dst: str
    label: Optional[str]
    line: int


@dataclass
class Graph:
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    line: int = 0


@dataclass
class Config:
    entries: dict[str, Value] = field(default_factory=dict)
    line: int = 0


@dataclass
class Program:
    model: Model
    graph: Graph
    config: Optional[Config]
