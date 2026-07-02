"""
nngraph/semantic_analyzer.py

Phase 3: semantic analysis over the AST from ast_builder.py.
Operates only on ast_nodes.py dataclasses (Program, Model, InputDecl,
Graph, GraphNode, Edge, Config) — never touches ANTLR context objects.
"""

from dataclasses import dataclass
from nngraph.ast_nodes import Program, GraphNode, Edge


@dataclass
class SemanticError:
    message: str
    node_id: str | None = None
    line: int | None = None


class SemanticAnalyzer:
    def __init__(self, program: Program):
        self.program = program
        self.errors: list[SemanticError] = []

        self.nodes: dict[str, GraphNode] = {}   # graph.nodes only (no input)
        self.valid_ids: set[str] = set()        # nodes + input.name
        self.adjacency: dict[str, list[str]] = {}
        self.incoming_count: dict[str, int] = {}

    def analyze(self) -> list[SemanticError]:
        self._collect_nodes()
        self._validate_edges()
        self._check_output_resolves()
        self._check_orphans()
        self._check_residual_arity()
        self._check_param_types()
        self._check_reachability()
        self._check_acyclic()
        return self.errors

    # ---- Unique node IDs --------------------------------------------------
    def _collect_nodes(self):
        input_name = self.program.model.input.name
        self.valid_ids.add(input_name)
        self.adjacency[input_name] = []

        for n in self.program.graph.nodes:
            if n.id in self.nodes or n.id == input_name:
                self.errors.append(SemanticError(
                    f"Duplicate node ID '{n.id}'", n.id, n.line))
                continue
            self.nodes[n.id] = n
            self.valid_ids.add(n.id)
            self.adjacency[n.id] = []
            self.incoming_count[n.id] = 0

    # ---- Undefined references + adjacency build --------------------------
    def _validate_edges(self):
        for e in self.program.graph.edges:
            for endpoint in (e.src, e.dst):
                if endpoint not in self.valid_ids:
                    self.errors.append(SemanticError(
                        f"Edge references undefined node '{endpoint}'",
                        line=e.line))
            if e.src in self.adjacency:
                self.adjacency[e.src].append(e.dst)
            if e.dst in self.incoming_count:
                self.incoming_count[e.dst] += 1

    # ---- output resolves against graph.nodes ------------------------------
    def _check_output_resolves(self):
        output = self.program.model.output
        if output not in self.valid_ids:
            self.errors.append(SemanticError(
                f"Output '{output}' does not resolve to any declared node"))

    # ---- Orphan nodes ------------------------------------------------------
    def _check_orphans(self):
        endpoints = set()
        for e in self.program.graph.edges:
            endpoints.add(e.src)
            endpoints.add(e.dst)

        output = self.program.model.output
        input_name = self.program.model.input.name

        for node_id in self.nodes:
            if node_id in (input_name, output):
                continue
            if node_id not in endpoints:
                self.errors.append(SemanticError(
                    f"Node '{node_id}' is orphaned (not part of any edge)",
                    node_id, self.nodes[node_id].line))

    # ---- Residual arity ------------------------------------------------
    def _check_residual_arity(self):
        for node_id, n in self.nodes.items():
            if n.layer_type == "Residual":
                count = self.incoming_count.get(node_id, 0)
                if count != 2:
                    self.errors.append(SemanticError(
                        f"Residual node '{node_id}' must have exactly "
                        f"2 incoming edges, found {count}", node_id, n.line))

    # ---- Parameter type checking ----------------------------------------
    EXPECTED_PARAM_TYPES: dict[str, dict[str, type]] = {
        "Linear": {"in_features": int, "out_features": int, "bias": bool},
        "Conv2d": {"in_channels": int, "out_channels": int,
                   "kernel_size": int, "stride": int, "padding": int},
        "Dropout": {"p": float},
        # extend per the spec's layer catalogue
    }

    def _check_param_types(self):
        for node_id, n in self.nodes.items():
            expected = self.EXPECTED_PARAM_TYPES.get(n.layer_type)
            if not expected:
                continue
            for name, value in n.params.items():
                exp_type = expected.get(name)
                if exp_type is None:
                    continue
                # bool is a subclass of int in Python -- guard against
                # a bool literal silently passing an `int` param check.
                if exp_type is int and isinstance(value, bool):
                    ok = False
                else:
                    ok = isinstance(value, exp_type)
                if not ok:
                    self.errors.append(SemanticError(
                        f"Node '{node_id}': param '{name}' expected "
                        f"{exp_type.__name__}, got {type(value).__name__}",
                        node_id, n.line))

    # ---- Reachability ----------------------------------------------------
    def _check_reachability(self):
        input_name = self.program.model.input.name
        output = self.program.model.output

        reachable = self._bfs(self.adjacency, input_name)

        for node_id in self.nodes:
            if node_id not in reachable:
                self.errors.append(SemanticError(
                    f"Node '{node_id}' is not reachable from input "
                    f"'{input_name}'", node_id, self.nodes[node_id].line))

        if output not in reachable and output != input_name:
            self.errors.append(SemanticError(
                f"Output '{output}' is not reachable from input"))

    @staticmethod
    def _bfs(adjacency: dict, start: str) -> set:
        visited = {start}
        stack = [start]
        while stack:
            current = stack.pop()
            for neighbor in adjacency.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)
        return visited

    # ---- Cycle detection (DAG check) --------------------------------------
    def _check_acyclic(self):
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {n: WHITE for n in self.adjacency}

        def visit(node_id):
            color[node_id] = GRAY
            for neighbor in self.adjacency.get(node_id, []):
                if neighbor not in color:
                    continue
                if color[neighbor] == GRAY:
                    self.errors.append(SemanticError(
                        f"Cycle detected involving node '{neighbor}'",
                        neighbor))
                elif color[neighbor] == WHITE:
                    visit(neighbor)
            color[node_id] = BLACK

        for node_id in self.adjacency:
            if color[node_id] == WHITE:
                visit(node_id)