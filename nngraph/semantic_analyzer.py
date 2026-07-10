"""
nngraph/semantic_analyzer.py

Phase 3: semantic analysis over the AST from ast_builder.py.
Operates only on ast_nodes.py dataclasses (Program, Model, InputDecl,
Graph, GraphNode, Edge, Config) -- never touches ANTLR context objects.

CONTRACT WITH PHASE 4 (codegen.py): after analyze() returns an empty
error list, every GraphNode.params dict in the program has already
been rewritten to use PyTorch's own keyword-argument names, not the
DSL's names. Renaming happens in _normalize_param_names(), the last
step of analyze(). Codegen no longer needs to know that `in_ch` and
`in_channels` are the same thing -- by the time it sees the AST, that
distinction no longer exists. This is a deliberate front-end/back-end
split: semantic analysis validates AND canonicalizes; codegen just
translates a trusted, already-canonical AST.
"""

from dataclasses import dataclass
from nngraph.ast_nodes import Program, GraphNode, Edge
from nngraph.diagnostics import Diagnostic, Severity
from nngraph.layer_catalogue import STRUCTURAL_OPS
from nngraph.graph_utils import topological_order, GraphCycleError
from nngraph.shape_inference import infer_shapes, Shape


@dataclass
class SemanticError:
    message: str
    node_id: str | None = None
    line: int | None = None
    severity: Severity = Severity.ERROR


class SemanticAnalyzer:
    def __init__(self, program: Program):
        self.program = program
        self.errors: list[SemanticError] = []

        self.nodes: dict[str, GraphNode] = {}   # graph.nodes only (no input)
        self.valid_ids: set[str] = set()        # nodes + input.name
        self.adjacency: dict[str, list[str]] = {}
        self.incoming_count: dict[str, int] = {}

        # Edge-keyed (not just node-id-keyed) adjacency, preserving
        # declaration order -- needed by topological_order() and by
        # shape inference, neither of which the plain `adjacency`
        # dict above is sufficient for: topological_order needs actual
        # Edge objects to walk (see graph_utils.py), and shape
        # inference needs to know, for a node with multiple incoming
        # edges, which shapes are arriving in which declared order.
        self.incoming_edges: dict[str, list[Edge]] = {}
        self.outgoing_edges: dict[str, list[Edge]] = {}

        # node_id -> inferred shape, or None if unknown/undetermined.
        # Populated by _infer_shapes(); stays empty if shape inference
        # was skipped because an earlier structural error made the
        # graph unsafe to walk (see _infer_shapes() itself).
        self.shapes: dict[str, Shape | None] = {}

    def analyze(self) -> list[SemanticError]:
        self._collect_nodes()
        self._validate_edges()
        self._check_output_resolves()
        self._check_orphans()
        self._check_residual_arity()
        self._check_implicit_sum_fanin()
        self._check_param_types()
        # Reachability and acyclicity now run BEFORE the required-param
        # check (they used to run after it) -- shape inference needs a
        # confirmed-acyclic, confirmed-reference-valid graph to safely
        # topologically sort and walk, and it has to run before the
        # required-param check so that params it successfully infers
        # (in_features, in_ch, ...) are already present by the time
        # that check looks for them. Moving reachability/acyclic
        # earlier was the smallest reordering that achieves that,
        # since shape inference depends on both anyway.
        self._check_reachability()
        self._check_acyclic()
        self._infer_shapes()
        self._check_required_params()
        # Normalization runs last and unconditionally. Even if a node
        # had an error (e.g. an unknown extra param), whatever known
        # params it DOES have -- including ones _infer_shapes just
        # filled in -- are still safe to rename -- the unknown ones
        # are left untouched and harmless, since a caller is expected
        # to gate codegen behind an empty error list anyway.
        self._normalize_param_names()
        return self.errors

    def to_diagnostics(self) -> list[Diagnostic]:
        """Converts accumulated SemanticError objects into Diagnostic
        objects, for unified reporting alongside Phase 1's syntax
        errors -- same type, same sort_diagnostics(), same
        has_errors() gate.

        Best-effort location fallback: a few call sites above don't
        currently thread a line through --
        _check_output_resolves()'s error (the output id doesn't
        resolve to anything, so there's no node to point at) and
        _check_acyclic()'s cycle error (has a node_id but the
        SemanticError(msg, neighbor) call only fills in message and
        node_id, not line) both leave line=None on the SemanticError.
        Rather than let those diagnostics render with no location at
        all -- which the spec's Section 7 format never does, every
        example shows "[line N]" -- this falls back to the referenced
        node's own declared line when node_id is known, and to the
        model block's line as a last resort when it isn't. This is
        diagnostics doing best-effort reporting on data it didn't
        create, not a substitute for threading the real line through
        at the point each error is raised -- that's still the more
        correct fix if you want to tighten it up in semantic_analyzer.py
        itself later.
        """
        result = []
        for err in self.errors:
            line = err.line
            if line is None and err.node_id is not None:
                node = self.nodes.get(err.node_id)
                if node is not None:
                    line = node.line
            if line is None:
                line = self.program.model.line
            result.append(Diagnostic(err.severity, line, err.message))
        return result

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
            # Built regardless of whether src/dst resolved -- an edge
            # referencing an undefined node already produced an error
            # above, and topological_order()/shape inference are both
            # gated on zero errors before they ever run (see
            # _infer_shapes), so a stray entry keyed on a bad id here
            # is harmless dead data, not a correctness risk.
            self.outgoing_edges.setdefault(e.src, []).append(e)
            self.incoming_edges.setdefault(e.dst, []).append(e)

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
                    node_id, self.nodes[node_id].line,
                    severity=Severity.WARNING))

    # ---- Residual arity ------------------------------------------------
    def _check_residual_arity(self):
        for node_id, n in self.nodes.items():
            if n.layer_type == "Residual":
                count = self.incoming_count.get(node_id, 0)
                if count != 2:
                    self.errors.append(SemanticError(
                        f"Residual node '{node_id}' must have exactly "
                        f"2 incoming edges, found {count}", node_id, n.line))

    # ---- Implicit-sum multi-input warning ----------------------------------
    def _check_implicit_sum_fanin(self):
        """A 'regular' (non-structural) layer with more than one
        incoming edge is valid DSL, not a mistake by construction --
        Section 4's Transformer Encoder example wires both
        `drop1 -> norm1` and `x -> norm1` (a residual connection) into
        a plain LayerNorm node, and its generated code sums them
        before calling the layer: `x = self.norm1(x + attn_out)`. This
        isn't stated as an explicit rule anywhere in Section 3/5/6's
        prose -- it only shows up in that one worked example -- but
        it's clearly intentional, and codegen.py treats it exactly
        like Residual: sum every incoming edge, then apply the layer
        to the sum.

        It's also an easy thing to trigger by accident -- wiring a
        second edge into a node you only meant to feed once -- so it's
        surfaced as a WARNING rather than silently accepted. The DSL
        author should know their layer is about to receive the
        elementwise sum of two tensors, not "whichever edge got
        declared last" or anything else that might be the more
        intuitive assumption.

        Shape/dimension compatibility between the summed tensors is
        NOT checked here -- that requires actually tracking tensor
        shapes through the graph, which this method doesn't do. See
        the shape-inference design notes for where that's headed.
        """
        for node_id, n in self.nodes.items():
            if n.layer_type in STRUCTURAL_OPS:
                continue  # Add/Concat/Residual/Split already expect multiple inputs
            count = self.incoming_count.get(node_id, 0)
            if count > 1:
                self.errors.append(SemanticError(
                    f"Node '{node_id}' ({n.layer_type}) has {count} "
                    f"incoming edges; they will be summed elementwise "
                    f"before being passed to {n.layer_type}, the same "
                    f"way a Residual node works. If that's not what you "
                    f"intended, remove the extra edge or route it "
                    f"through an explicit Add() or Concat() node instead.",
                    node_id, n.line, severity=Severity.WARNING))

    # ---- Layer catalogue ---------------------------------------------------
    # Keys and param names here match the DSL exactly as documented in
    # Section 3.3-3.5 of the spec -- e.g. Conv2d's 'in_ch', not
    # PyTorch's 'in_channels'. Validation happens in these terms
    # deliberately, so error messages reference what the user actually
    # typed. The DSL -> PyTorch rename only happens afterward, in
    # _normalize_param_names().
    EXPECTED_PARAM_TYPES: dict[str, dict[str, type]] = {
        # -- Layers (3.3) --
        "Linear": {"in_features": int, "out_features": int, "bias": bool},
        "Conv2d": {"in_ch": int, "out_ch": int, "kernel": int,
                   "stride": int, "padding": int},
        "Conv1d": {"in_ch": int, "out_ch": int, "kernel": int, "stride": int},
        "BatchNorm2d": {"num_features": int},
        # normalized_shape is technically int-or-tuple in real PyTorch,
        # but every worked example in the spec writes it as a shape
        # literal -- (128) -- which parses to a 1-tuple in the AST, so
        # `tuple` is what's actually checked here. A bare
        # normalized_shape=128 would fail this check even though the
        # grammar permits it; a known, narrow gap, not fixed here.
        "LayerNorm": {"normalized_shape": tuple},
        "MaxPool2d": {"kernel": int, "stride": int},
        "AvgPool2d": {"kernel": int, "stride": int},
        "Dropout": {"p": float},
        "Flatten": {"start_dim": int, "end_dim": int},
        "Embedding": {"num_embeddings": int, "embedding_dim": int},
        "MultiHeadAttn": {"embed_dim": int, "num_heads": int},
        "LSTM": {"input_size": int, "hidden_size": int, "num_layers": int},
        "GRU": {"input_size": int, "hidden_size": int},
        # -- Activations (3.4) --
        "ReLU": {},
        "Sigmoid": {},
        "Tanh": {},
        "GELU": {},
        "Softmax": {"dim": int},
        "LeakyReLU": {"negative_slope": float},
        "ELU": {"alpha": float},
        # -- Special ops (3.5) --
        "Add": {},
        "Concat": {"dim": int},
        "Residual": {},
        "Split": {"chunks": int, "dim": int},
    }

    # Params listed in EXPECTED_PARAM_TYPES that are NOT required --
    # anything for a given layer_type that's absent from this set (but
    # present in EXPECTED_PARAM_TYPES) is required.
    #
    # Note what's deliberately NOT here: in_features, in_ch,
    # num_features, normalized_shape, embed_dim, input_size (see
    # INFERABLE_PARAM in layer_catalogue.py) are all omittable in
    # valid DSL now that _infer_shapes() exists, but they're still
    # untouched here. That's not an oversight -- _infer_shapes() runs
    # BEFORE _check_required_params() in analyze() and, when it
    # successfully infers one of these, writes it directly into
    # node.params. By the time _check_required_params looks for it,
    # it's simply already there. If inference *couldn't* fill it
    # (e.g. shape was lost downstream of a Flatten), it's genuinely
    # still missing and this check is right to flag it -- so no
    # separate "optional if inferable" concept needs to exist here at
    # all; the ordering alone makes it work.
    OPTIONAL_PARAMS: dict[str, set[str]] = {
        "Linear": {"bias"},
        "Flatten": {"start_dim", "end_dim"},
        "LeakyReLU": {"negative_slope"},
        "ELU": {"alpha"},
    }

    # DSL param name -> PyTorch kwarg name, only where they differ.
    # Applied by _normalize_param_names() after validation succeeds.
    PARAM_RENAME: dict[str, dict[str, str]] = {
        "Conv2d": {"in_ch": "in_channels", "out_ch": "out_channels", "kernel": "kernel_size"},
        "Conv1d": {"in_ch": "in_channels", "out_ch": "out_channels", "kernel": "kernel_size"},
        "MaxPool2d": {"kernel": "kernel_size"},
        "AvgPool2d": {"kernel": "kernel_size"},
    }

    # ---- Shape inference ----------------------------------------------------
    def _infer_shapes(self):
        """Runs shape propagation (nngraph/shape_inference.py) over the
        graph: auto-fills inferable params (see INFERABLE_PARAM in
        layer_catalogue.py) directly into each node's params dict, and
        reports genuine shape mismatches as errors through the same
        self.errors list everything else uses.

        Skipped entirely if any ERROR-severity diagnostic already
        exists. Undefined references or a cycle make the graph unsafe
        to topologically sort at all -- there's no point computing
        shapes over a graph that's already going to be rejected, and
        attempting it could produce a second, confusing wave of
        errors on top of the real problem.

        shape_inference.py deliberately doesn't know about
        SemanticError -- it reports through a plain callback so it
        never has to import this module (which would create a cycle,
        since this method is what imports and calls it). The lambda
        below is the only place that translates between the two.
        """
        if any(e.severity is Severity.ERROR for e in self.errors):
            return
        try:
            order = topological_order(self.program, self.outgoing_edges)
        except GraphCycleError:
            # Should be unreachable -- _check_acyclic() already ran
            # (see the ordering in analyze()) and found nothing,
            # which is exactly the condition the guard above checks.
            # Defensive only, same invariant-assertion philosophy as
            # codegen.py's own defensive checks.
            return
        self.shapes = infer_shapes(
            self.program, self.nodes, order, self.incoming_edges,
            report_error=lambda msg, node_id, line, severity: self.errors.append(
                SemanticError(msg, node_id, line, severity=severity)),
        )

    # ---- Parameter type checking ----------------------------------------
    def _check_param_types(self):
        for node_id, n in self.nodes.items():
            expected = self.EXPECTED_PARAM_TYPES.get(n.layer_type)
            # Must be `is None`, not `not expected` -- ReLU, Add,
            # Residual etc. are known layer types that legitimately
            # map to an empty dict `{}`, which is also falsy in
            # Python. `not expected` would incorrectly flag every
            # zero-parameter layer type as unknown.
            if expected is None:
                self.errors.append(SemanticError(
                    f"Node '{node_id}' has undefined layer_type '{n.layer_type}'", node_id, n.line
                ))
                continue
            for name, value in n.params.items():
                exp_type = expected.get(name)
                if exp_type is None:
                    self.errors.append(SemanticError(
                        f"Node '{node_id}' has undefined parameter '{name}' for layer_type '{n.layer_type}'", node_id, n.line
                    ))
                    continue
                # bool is a subclass of int in Python -- guard against
                # a bool literal silently passing an `int` param check.
                if exp_type is int and isinstance(value, bool):
                    ok = False
                # A float-typed param written as a bare whole number
                # (p=0, alpha=1) lexes as INT under the grammar, not
                # FLOAT -- there's no decimal point to trigger the
                # FLOAT token. That's completely normal PyTorch usage,
                # so widen int -> float here instead of rejecting it.
                # (isinstance(True, int) is also True in Python, so
                # this must explicitly exclude bool too, or a bool
                # would silently satisfy a float-typed param as well.)
                elif exp_type is float and isinstance(value, int) and not isinstance(value, bool):
                    ok = True
                else:
                    ok = isinstance(value, exp_type)
                if not ok:
                    self.errors.append(SemanticError(
                        f"Node '{node_id}': param '{name}' expected "
                        f"{exp_type.__name__}, got {type(value).__name__}",
                        node_id, n.line))

    # ---- Required parameter presence ---------------------------------------
    def _check_required_params(self):
        for node_id, n in self.nodes.items():
            expected = self.EXPECTED_PARAM_TYPES.get(n.layer_type)
            if expected is None:
                continue  # already reported by _check_param_types
            optional = self.OPTIONAL_PARAMS.get(n.layer_type, set())
            for param_name in expected:
                if param_name in optional:
                    continue
                if param_name not in n.params:
                    self.errors.append(SemanticError(
                        f"Node '{node_id}' ({n.layer_type}) is missing "
                        f"required parameter '{param_name}'", node_id, n.line))

    # ---- DSL -> PyTorch parameter name normalization ------------------------
    def _normalize_param_names(self):
        """Rewrites every node's params dict in place so keys match
        PyTorch's actual constructor kwargs. Runs last, after all
        validation, so error messages above always reference the DSL
        names the user actually wrote -- never the renamed ones."""
        for n in self.nodes.values():
            rename = self.PARAM_RENAME.get(n.layer_type)
            if not rename:
                continue
            n.params = {rename.get(k, k): v for k, v in n.params.items()}

    # ---- Reachability ----------------------------------------------------
    def _check_reachability(self):
        input_name = self.program.model.input.name
        output = self.program.model.output

        reachable = self._bfs(self.adjacency, input_name)

        for node_id in self.nodes:
            if node_id == output:
                continue  # reported below with a more specific message
            if node_id not in reachable:
                self.errors.append(SemanticError(
                    f"Node '{node_id}' is not reachable from input "
                    f"'{input_name}'", node_id, self.nodes[node_id].line))

        if (output not in reachable and output != input_name
                and output in self.valid_ids):
            # If output isn't even a valid id, _check_output_resolves()
            # already reported that -- don't pile on a second, confusing
            # "not reachable" error for a node that doesn't exist.
            self.errors.append(SemanticError(
                f"Output '{output}' is not reachable from input "
                f"'{input_name}'", output,
                self.nodes[output].line if output in self.nodes else None))

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
