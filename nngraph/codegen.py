"""
nngraph/codegen.py

Phase 4: code generation. Consumes a Program (ast_nodes.py) that is
assumed to have already passed SemanticAnalyzer.analyze() with an
empty error list.

PRECONDITION (guaranteed by semantic_analyzer.py, not re-checked
here): every GraphNode.params dict already uses PyTorch's own
keyword-argument names -- SemanticAnalyzer._normalize_param_names()
rewrites DSL names (in_ch, kernel, ...) to PyTorch names (in_channels,
kernel_size, ...) as the last step of analyze(). Codegen does NOT
rename anything; there is no PARAM_RENAME table here anymore. If
you're looking for where 'in_ch' becomes 'in_channels', it's in
Phase 3, not Phase 4 -- that's a deliberate front-end/back-end split
(semantic analysis validates AND canonicalizes; codegen just
translates an already-canonical AST).

This module does not re-check the structural rules Phase 3 already
owns (cycles, undefined references, residual arity, required
parameters, ...) -- it assumes those are proven true. It DOES still
raise CodegenError defensively at a few spots (see inline notes) as
invariant checks rather than blind trust -- cheap insurance against
codegen ever being invoked on an AST that didn't actually go through
semantic analysis first (e.g. a hand-built AST in a unit test), so a
violated assumption surfaces as a clear message instead of a
confusing KeyError three lines later.
"""

import keyword

from nngraph.ast_nodes import Edge, GraphNode, Program
from nngraph.graph_utils import GraphCycleError, topological_order
from nngraph.layer_catalogue import STRUCTURAL_OPS


class CodegenError(Exception):
    """Raised when the AST violates an invariant codegen depends on.
    Under the documented contract with semantic_analyzer.py, none of
    these should be reachable in practice -- see each raise site."""


# DSL layer type -> torch.nn class (Section 3.3/3.4 of the spec).
PYTORCH_CLASS_MAP = {
    "Linear": "nn.Linear",
    "Conv2d": "nn.Conv2d",
    "Conv1d": "nn.Conv1d",
    "BatchNorm2d": "nn.BatchNorm2d",
    "LayerNorm": "nn.LayerNorm",
    "MaxPool2d": "nn.MaxPool2d",
    "AvgPool2d": "nn.AvgPool2d",
    "Dropout": "nn.Dropout",
    "Flatten": "nn.Flatten",
    "Embedding": "nn.Embedding",
    "MultiHeadAttn": "nn.MultiheadAttention",
    "LSTM": "nn.LSTM",
    "GRU": "nn.GRU",
    "ReLU": "nn.ReLU",
    "Sigmoid": "nn.Sigmoid",
    "Tanh": "nn.Tanh",
    "GELU": "nn.GELU",
    "Softmax": "nn.Softmax",
    "LeakyReLU": "nn.LeakyReLU",
    "ELU": "nn.ELU",
}

# Params the generator injects itself -- not sourced from the DSL.
# MultiheadAttention defaults to batch_first=False in PyTorch, but
# every use of it elsewhere in this codebase assumes (batch, seq,
# dim) tensors, so we pin it explicitly rather than silently
# inheriting PyTorch's default.
INJECTED_PARAMS = {
    "MultiHeadAttn": {"batch_first": True},
}


def _format_value(value) -> str:
    """Render an AST literal (int/float/str/bool/None/shape-tuple) as
    Python source text.

    Single-element shape tuples are unwrapped to a bare number --
    normalized_shape=(128) in the DSL becomes normalized_shape=128 in
    the generated call. PyTorch's LayerNorm accepts either form, and
    the bare form is what the spec's own worked example emits, so
    this keeps generated output matching hand-written PyTorch style
    rather than carrying a visible DSL-ism into the output.
    """
    if isinstance(value, tuple):
        return str(value[0]) if len(value) == 1 else str(tuple(value))
    return repr(value)  # correct by construction for int/float/str/bool/None


def _safe_identifier(name: str) -> str:
    """Node, model, and input IDs come straight from the grammar's ID
    token, which matches Python's own identifier syntax character-
    for-character -- so no sanitization is needed for *shape*. But
    the ID token also happily accepts Python keywords ('for', 'class',
    'def', ...), which are syntactically valid DSL identifiers yet
    illegal as Python variable/class/attribute names. `node for :
    ReLU()` would otherwise emit `for = self.for(x)`, a syntax error.

    Nothing in the current SemanticAnalyzer rejects keyword-shaped
    IDs -- recommend adding that check to Phase 3, since catching it
    there gives a proper line-numbered diagnostic instead of a
    generated-but-broken .py file. Until then, this is the belt-and-
    suspenders fallback: append the standard PEP 8 trailing
    underscore (`class_`) used for exactly this collision.

    IMPORTANT: this is a rendering-time transform only. It must never
    be applied to a string used as a dict key or compared against
    edge.src/edge.dst -- those all stay in terms of the raw AST id,
    or every lookup keyed on an escaped id silently stops matching
    edges that still reference the raw one.
    """
    return f"{name}_" if keyword.iskeyword(name) else name


class CodeGenerator:
    def __init__(self, program: Program):
        self.program = program
        self.input_id = program.model.input.name  # RAW id -- used for all lookups/comparisons
        self.nodes_by_id: dict[str, GraphNode] = {n.id: n for n in program.graph.nodes}

        # Declaration-order-preserving edge lists, keyed by raw id.
        # Order matters for Concat (channel order is semantically
        # meaningful) and Split (see _predecessor_expr) even though
        # it's purely cosmetic for Add/Residual, since elementwise
        # addition is commutative -- "which summand is the shortcut"
        # doesn't change the computed result, only readability.
        self.outgoing: dict[str, list[Edge]] = {}
        self.incoming: dict[str, list[Edge]] = {}
        for edge in program.graph.edges:
            self.outgoing.setdefault(edge.src, []).append(edge)
            self.incoming.setdefault(edge.dst, []).append(edge)

    # ------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------
    def generate(self) -> str:
        class_name = _safe_identifier(self.program.model.name)
        input_param = _safe_identifier(self.input_id)

        order = self._topological_order()
        init_body = self._emit_init()
        forward_body = self._emit_forward(order)

        lines = [
            "import torch",
            "import torch.nn as nn",
            "",
            f"class {class_name}(nn.Module):",
            "    def __init__(self):",
            f"        super({class_name}, self).__init__()",
            *[f"        {line}" for line in init_body],
            "",
            f"    def forward(self, {input_param}):",
            *[f"        {line}" for line in forward_body],
        ]
        main_block = self._emit_main_block(class_name)
        if main_block:
            lines += ["", ""] + main_block
        return "\n".join(lines) + "\n"

    # ------------------------------------------------------------
    # Topological sort -- delegates to graph_utils.topological_order,
    # the same function shape_inference.py now uses. Kept as a thin
    # method here (rather than calling the shared function directly at
    # the one call site in generate()) only so CodegenError stays the
    # single exception type this module's public surface raises --
    # GraphCycleError is an implementation detail of graph_utils, not
    # something a caller of CodeGenerator should need to know exists.
    # ------------------------------------------------------------
    def _topological_order(self) -> list[str]:
        try:
            return topological_order(self.program, self.outgoing)
        except GraphCycleError as e:
            # Should already be unreachable -- SemanticAnalyzer.
            # _check_acyclic() ran during Phase 3 and found nothing.
            # Defensive backstop, not the primary detection path.
            raise CodegenError(str(e)) from e

    # ------------------------------------------------------------
    # __init__ body -- declaration order, skipping structural ops
    # ------------------------------------------------------------
    def _emit_init(self) -> list[str]:
        lines = []
        for node in self.program.graph.nodes:
            if node.layer_type in STRUCTURAL_OPS:
                continue
            if node.layer_type not in PYTORCH_CLASS_MAP:
                raise CodegenError(
                    f"Unknown layer type '{node.layer_type}' on node "
                    f"'{node.id}' (line {node.line}). SemanticAnalyzer's "
                    f"EXPECTED_PARAM_TYPES now covers the full layer "
                    f"catalogue and should already reject this -- if "
                    f"you're seeing this, either codegen ran without "
                    f"semantic analysis first, or PYTORCH_CLASS_MAP here "
                    f"and EXPECTED_PARAM_TYPES in Phase 3 have drifted "
                    f"out of sync (they list the same ~19 layer types "
                    f"independently; keeping them in lockstep is a "
                    f"manual invariant across the two files, not "
                    f"something enforced by the type system)."
                )
            pytorch_class = PYTORCH_CLASS_MAP[node.layer_type]
            # No renaming here -- node.params keys are already PyTorch
            # kwarg names by the time this runs (see module docstring:
            # SemanticAnalyzer._normalize_param_names() did that already).
            params = dict(node.params)
            params.update(INJECTED_PARAMS.get(node.layer_type, {}))

            kwargs = ", ".join(
                f"{k}={_format_value(v)}" for k, v in params.items()
            )
            var = _safe_identifier(node.id)
            lines.append(f"self.{var} = {pytorch_class}({kwargs})")
        return lines

    # ------------------------------------------------------------
    # forward() body -- topological order
    # ------------------------------------------------------------
    def _emit_forward(self, order: list[str]) -> list[str]:
        lines = []
        for node_id in order:
            if node_id == self.input_id:
                continue  # already bound as the forward() parameter
            node = self.nodes_by_id[node_id]
            var = _safe_identifier(node_id)
            preds = self.incoming.get(node_id, [])

            if node.layer_type == "Add":
                if not preds:
                    raise CodegenError(
                        f"Add node '{node_id}' (line {node.line}) has no "
                        f"incoming edges; Add requires at least one input "
                        f"to sum. The orphan check in SemanticAnalyzer "
                        f"only catches nodes with NO edge involvement at "
                        f"all, not nodes with outgoing-but-no-incoming "
                        f"edges, so this slips through today."
                    )
                exprs = [self._predecessor_expr(e) for e in preds]
                lines.append(f"{var} = {' + '.join(exprs)}")

            elif node.layer_type == "Concat":
                if not preds:
                    raise CodegenError(
                        f"Concat node '{node_id}' (line {node.line}) has "
                        f"no incoming edges. Same gap as the Add case above."
                    )
                if "dim" not in node.params:
                    raise CodegenError(
                        f"Concat node '{node_id}' (line {node.line}) is "
                        f"missing required parameter 'dim'. "
                        f"SemanticAnalyzer._check_required_params() "
                        f"should already have caught this -- this is a "
                        f"defensive backstop, not the primary detection "
                        f"path. Deliberately erroring here rather than "
                        f"defaulting dim=0, since a silent default would "
                        f"hide a real authoring mistake if this path is "
                        f"ever reached."
                    )
                exprs = [self._predecessor_expr(e) for e in preds]
                dim = node.params["dim"]
                lines.append(f"{var} = torch.cat([{', '.join(exprs)}], dim={dim})")

            elif node.layer_type == "Residual":
                # Arity == 2 is already guaranteed by
                # SemanticAnalyzer._check_residual_arity. Which of the
                # two predecessors is the "shortcut" vs "main path" is
                # cosmetic only -- elementwise addition is commutative
                # -- so edge-declaration order is used as a
                # deterministic, readable default rather than trying
                # to infer intent from edge labels.
                a, b = (self._predecessor_expr(e) for e in preds)
                lines.append(f"{var} = {a} + {b}")

            elif node.layer_type == "Split":
                if len(preds) != 1:
                    raise CodegenError(
                        f"Split node '{node_id}' (line {node.line}) has "
                        f"{len(preds)} incoming edges; Split takes exactly "
                        f"one input tensor to chunk. SemanticAnalyzer has "
                        f"an arity check for Residual but not for Split -- "
                        f"recommend adding one alongside it."
                    )
                missing = [p for p in ("chunks", "dim") if p not in node.params]
                if missing:
                    raise CodegenError(
                        f"Split node '{node_id}' (line {node.line}) is "
                        f"missing required parameter(s): {', '.join(missing)}. "
                        f"Same defensive backstop as the Concat case above."
                    )
                src_expr = self._predecessor_expr(preds[0])
                chunks = node.params["chunks"]
                dim = node.params["dim"]
                lines.append(f"{var} = torch.chunk({src_expr}, {chunks}, dim={dim})")

            else:
                # Regular stateful layer. Fan-in of exactly 1 is the
                # common case, but fan-in > 1 is valid DSL too --
                # Section 4's Transformer Encoder example wires two
                # edges into a plain LayerNorm node (a residual
                # connection) and its generated code sums them before
                # the call: `x = self.norm1(x + attn_out)`.
                # SemanticAnalyzer emits a WARNING for this case (not
                # an error, since it's valid), see
                # _check_implicit_sum_fanin there. Summing is handled
                # the same way regardless of fan-in count: joining a
                # single-element list with ' + ' just yields that one
                # element unchanged, so this one code path covers both.
                #
                # Fan-in == 0 is NOT possible here and isn't checked:
                # SemanticAnalyzer's reachability check already
                # guarantees every non-input node reachable from input
                # has at least one incoming edge (the last hop of
                # whatever path proved it reachable), and codegen only
                # ever runs on a program that passed that check with
                # zero errors.
                exprs = [self._predecessor_expr(e) for e in preds]
                arg = " + ".join(exprs)

                if node.layer_type == "MultiHeadAttn":
                    # Self-attention per Section 6.3: query=key=value.
                    # Real nn.MultiheadAttention returns (output,
                    # attn_weights) -- the weights are discarded here,
                    # matching the spec's own generated pattern
                    # (`attn_out, _ = self.attn(x, x, x)`). Missing
                    # this unpack was a pre-existing bug: without it,
                    # `var` would be bound to the whole 2-tuple instead
                    # of just the output tensor, silently breaking
                    # every layer downstream of it.
                    lines.append(f"{var}, _ = self.{var}({arg}, {arg}, {arg})")
                elif node.layer_type in ("LSTM", "GRU"):
                    # Real nn.LSTM/nn.GRU also return (output,
                    # hidden_state) -- hidden state discarded per
                    # Section 6.3 ("hidden state discarded by
                    # default"). Same pre-existing bug as MultiHeadAttn.
                    lines.append(f"{var}, _ = self.{var}({arg})")
                else:
                    lines.append(f"{var} = self.{var}({arg})")

        output_var = _safe_identifier(self.program.model.output)
        lines.append(f"return {output_var}")
        return lines

    def _predecessor_expr(self, edge: Edge) -> str:
        """The Python expression an edge's source contributes to its
        destination. Ordinarily this is just the source node's own
        variable name -- var-name == node-id is the convention this
        whole generator follows, so downstream code can reference a
        predecessor's output by simply reusing its declared id.

        The one exception is Split: it produces a tuple of chunks, not
        a single tensor, so each of ITS outgoing edges must index into
        that tuple. Which index each edge gets is not specified
        anywhere in the DSL (there's no chunk-index syntax on edges) --
        this is an interpretive extension, not something the spec
        defines: the Nth edge declared out of a given Split node maps
        to chunk N, in source order. If that's not the intended
        semantics, the grammar would need an explicit annotation, e.g.
        `edge split -> convA1 [chunk=0]`, analogous to the existing
        `[label=...]` syntax on edges.
        """
        src_var = _safe_identifier(edge.src)
        src_node = self.nodes_by_id.get(edge.src)
        if src_node is not None and src_node.layer_type == "Split":
            index = self.outgoing[edge.src].index(edge)
            return f"{src_var}[{index}]"
        return src_var

    # ------------------------------------------------------------
    # Optional __main__ scaffold
    # ------------------------------------------------------------
    def _emit_main_block(self, class_name: str) -> list[str]:
        # Every worked example in the spec that has a config block
        # also has a __main__ scaffold, and every example without a
        # config block omits it entirely -- config presence is used
        # as the trigger here, matching that pattern rather than
        # inventing a separate unstated flag.
        config = self.program.config
        if config is None:
            return []

        device = config.entries.get("device", "cpu")
        batch_size = config.entries.get("batch_size", 1)
        shape = ", ".join(str(d) for d in self.program.model.input.shape)
        input_var = _safe_identifier(self.input_id)

        return [
            "if __name__ == '__main__':",
            f"    device = torch.device({_format_value(device)})",
            f"    model = {class_name}().to(device)",
            f"    {input_var} = torch.randn({batch_size}, {shape}).to(device)",
            f"    print(model({input_var}).shape)",
            # Deliberately no inferred-shape trailing comment here
            # (the spec's own example has "# -> torch.Size([64, 10])").
            # SemanticAnalyzer now DOES compute this (self.shapes,
            # populated by _infer_shapes()) -- CodeGenerator just isn't
            # wired up to consume it yet. That's a small, mechanical
            # follow-up (thread `shapes: dict[str, Shape | None]`
            # through CodeGenerator.__init__, look up
            # shapes[program.model.output] here), not a fundamental
            # gap the way it was before this session.
        ]


def generate(program: Program) -> str:
    return CodeGenerator(program).generate()
