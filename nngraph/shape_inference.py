"""
nngraph/shape_inference.py

Phase 3 extension: propagates tensor shapes through the graph in
topological order, validates that shapes agree wherever tensors are
combined (implicit-sum on multi-input regular layers, Add, Concat,
Residual, Split), and auto-fills the one class of parameter that's
genuinely derivable from context: "how many features/channels flow IN
to this layer" (see INFERABLE_PARAM in layer_catalogue.py). The
symmetric "how many flow OUT" parameters (out_features, out_ch,
hidden_size, embedding_dim, ...) are never touched -- they're the DSL
author's own design choice, not something derivable from upstream
shape.

Shapes here never include the batch dimension, matching
InputDecl.shape's existing convention -- config.batch_size is a
wholly separate, independently-specified concern applied only in the
generated __main__ scaffold.

IMPORTANT -- dim values are batch-inclusive, shapes are not: every
`dim` parameter in the DSL (Concat's, Split's) is written the way
you'd write it against the REAL runtime tensor, which PyTorch always
gives an implicit batch axis at index 0. `Concat(dim=1)` on a real
(batch, channels, H, W) tensor concatenates channels -- dim=1 IS the
channel axis specifically because batch occupies dim=0. But this
module's tracked shapes are (channels, H, W), with no batch axis at
all, so dim=1 read directly against that tuple lands on H, not
channels. _translate_dim() below converts a batch-inclusive dim into
the correct index for a batch-less shape tuple (shift non-negative
dims down by one; negative dims like dim=-1 need no shift, since
"count from the end" is unaffected by however many leading axes
exist). Every place that indexes a tracked shape using a DSL-provided
dim value MUST go through _translate_dim() -- reading node.params["dim"]
directly against a shape tuple is exactly the bug this paragraph
exists to prevent someone from reintroducing.

Flatten and Embedding are treated as shape-UNKNOWN rather than
computed precisely. The spec's own worked examples are ambiguous
about whether Flatten's start_dim/end_dim are meant to index the
batch-inclusive runtime tensor or the batch-less DSL shape this
module tracks -- guessing wrong here would silently produce a
confidently-incorrect inferred shape instead of an honest "don't
know". UNKNOWN propagates forward: it disables further inference (and
shape *validation*) for anything downstream, but does NOT disable
required-parameter checking -- a node past an UNKNOWN simply falls
back to needing every "in_" param spelled out explicitly in the DSL,
with the ordinary "missing required parameter" diagnostic from
SemanticAnalyzer._check_required_params() rather than anything
shape-specific.

DEPENDENCY NOTE: this module has no idea what error type its caller
wants and does not import SemanticError. Errors are reported through
a plain callback (`report_error`) instead of a return value, so
nothing here needs to import semantic_analyzer.py -- which would
create a circular import, since semantic_analyzer.py is what calls
infer_shapes().
"""

from typing import Callable, Optional

from nngraph.ast_nodes import Edge, GraphNode, Program
from nngraph.diagnostics import Severity
from nngraph.layer_catalogue import INFERABLE_PARAM

Shape = tuple[int, ...]
UNKNOWN: Optional[Shape] = None  # see module docstring for why this exists

# (message, node_id, line, severity) -- shaped closely enough to
# SemanticError's own fields that a caller can wrap each call directly
# into `SemanticError(message, node_id, line, severity=severity)`.
ReportError = Callable[[str, Optional[str], Optional[int], Severity], None]


def infer_shapes(
    program: Program,
    nodes: dict[str, GraphNode],
    order: list[str],
    incoming: dict[str, list[Edge]],
    report_error: ReportError,
) -> dict[str, Optional[Shape]]:
    """Returns node_id -> inferred output shape (or None if unknown or
    lost downstream of an unknown). `order` must already be a valid
    topological order over the same graph (see
    graph_utils.topological_order) -- this function does not
    re-derive or re-validate it.
    """
    input_id = program.model.input.name
    shapes: dict[str, Optional[Shape]] = {input_id: tuple(program.model.input.shape)}

    for node_id in order:
        if node_id == input_id:
            continue
        node = nodes[node_id]
        preds = incoming.get(node_id, [])
        pred_shapes = [shapes.get(e.src) for e in preds]

        if node.layer_type == "Concat":
            shapes[node_id] = _concat_shape(node, pred_shapes, report_error)
            continue

        if node.layer_type == "Split":
            input_shape = pred_shapes[0] if pred_shapes else None
            shapes[node_id] = _split_shape(node, input_shape, report_error)
            continue

        # Everything else -- Add, Residual, and any regular layer with
        # fan-in > 1 (the implicit-sum case from
        # SemanticAnalyzer._check_implicit_sum_fanin) -- combines
        # multiple inputs by requiring them to share an identical
        # shape. A single predecessor trivially "combines" to just
        # that one shape unchanged, so fan-in == 1 and fan-in > 1 share
        # one code path.
        combined = _sum_shape(node, pred_shapes, report_error)

        if node.layer_type in ("Add", "Residual"):
            shapes[node_id] = combined  # the sum IS the output
        else:
            shapes[node_id] = _apply_layer_rule(node, combined, report_error)

    return shapes


def _sum_shape(
    node: GraphNode, pred_shapes: list[Optional[Shape]], report_error: ReportError
) -> Optional[Shape]:
    if not pred_shapes:
        return UNKNOWN  # unreachable in practice -- see codegen.py's note on why
    if any(s is UNKNOWN for s in pred_shapes):
        return UNKNOWN
    first = pred_shapes[0]
    for s in pred_shapes[1:]:
        if s != first:
            report_error(
                f"Node '{node.id}': incoming tensors have mismatched "
                f"shapes {first} and {s}; they are summed elementwise "
                f"and must be identical.",
                node.id, node.line, Severity.ERROR,
            )
            return UNKNOWN  # can't meaningfully continue inference from here
    return first


def _check_or_fill(node: GraphNode, param_name: str, computed_value, report_error: ReportError) -> None:
    """If the DSL author omitted `param_name`, fill it in with the
    value implied by the incoming shape. If they provided it
    explicitly and it disagrees with what the shape implies, report a
    mismatch -- but leave their original value in place rather than
    silently overwriting it. Codegen won't run on a program with
    errors anyway; leaving the original value means anyone inspecting
    the AST after a failed compile sees exactly what they wrote, not a
    value this pass quietly substituted."""
    if param_name in node.params:
        provided = node.params[param_name]
        if provided != computed_value:
            report_error(
                f"Node '{node.id}': parameter '{param_name}' is set to "
                f"{provided!r}, but the incoming tensor shape implies "
                f"it should be {computed_value!r}.",
                node.id, node.line, Severity.ERROR,
            )
    else:
        node.params[param_name] = computed_value


# ------------------------------------------------------------
# Per-layer-type shape rules
# ------------------------------------------------------------
# Each rule receives the node and its (already-resolved, already
# shape-checked-if-multi-input) single effective input shape, and
# returns the output shape -- or UNKNOWN if a non-inferable required
# param is still missing (SemanticAnalyzer._check_required_params
# will report that separately; no need to duplicate it here) or a
# dimension constraint is violated (reported directly, since THAT is
# this module's actual job).

def _passthrough(node: GraphNode, input_shape: Shape, report_error: ReportError) -> Shape:
    return input_shape  # Dropout, ReLU, Sigmoid, Tanh, GELU, Softmax, LeakyReLU, ELU


def _rule_linear(node: GraphNode, input_shape: Shape, report_error: ReportError) -> Optional[Shape]:
    _check_or_fill(node, INFERABLE_PARAM["Linear"], input_shape[-1], report_error)
    if "out_features" not in node.params:
        return UNKNOWN
    return input_shape[:-1] + (node.params["out_features"],)


def _rule_conv2d(node: GraphNode, input_shape: Shape, report_error: ReportError) -> Optional[Shape]:
    if len(input_shape) != 3:
        report_error(
            f"Node '{node.id}': Conv2d expects a 3D (channels, height, "
            f"width) input shape, got {input_shape}.",
            node.id, node.line, Severity.ERROR,
        )
        return UNKNOWN
    _check_or_fill(node, INFERABLE_PARAM["Conv2d"], input_shape[0], report_error)
    if any(p not in node.params for p in ("out_ch", "kernel", "stride", "padding")):
        return UNKNOWN
    c, h, w = input_shape
    k, s, p = node.params["kernel"], node.params["stride"], node.params["padding"]
    h2 = (h + 2 * p - k) // s + 1
    w2 = (w + 2 * p - k) // s + 1
    if h2 <= 0 or w2 <= 0:
        report_error(
            f"Node '{node.id}': Conv2d parameters produce a "
            f"non-positive output spatial size ({h2}x{w2}) from input "
            f"{input_shape}.", node.id, node.line, Severity.ERROR,
        )
        return UNKNOWN
    return (node.params["out_ch"], h2, w2)


def _rule_conv1d(node: GraphNode, input_shape: Shape, report_error: ReportError) -> Optional[Shape]:
    if len(input_shape) != 2:
        report_error(
            f"Node '{node.id}': Conv1d expects a 2D (channels, length) "
            f"input shape, got {input_shape}.",
            node.id, node.line, Severity.ERROR,
        )
        return UNKNOWN
    _check_or_fill(node, INFERABLE_PARAM["Conv1d"], input_shape[0], report_error)
    if any(p not in node.params for p in ("out_ch", "kernel", "stride")):
        return UNKNOWN
    _c, length = input_shape
    k, s = node.params["kernel"], node.params["stride"]
    length2 = (length - k) // s + 1
    if length2 <= 0:
        report_error(
            f"Node '{node.id}': Conv1d parameters produce a "
            f"non-positive output length ({length2}) from input "
            f"{input_shape}.", node.id, node.line, Severity.ERROR,
        )
        return UNKNOWN
    return (node.params["out_ch"], length2)


def _rule_batchnorm2d(node: GraphNode, input_shape: Shape, report_error: ReportError) -> Optional[Shape]:
    if len(input_shape) != 3:
        report_error(
            f"Node '{node.id}': BatchNorm2d expects a 3D (channels, "
            f"height, width) input shape, got {input_shape}.",
            node.id, node.line, Severity.ERROR,
        )
        return UNKNOWN
    _check_or_fill(node, INFERABLE_PARAM["BatchNorm2d"], input_shape[0], report_error)
    return input_shape


def _rule_layernorm(node: GraphNode, input_shape: Shape, report_error: ReportError) -> Shape:
    _check_or_fill(node, INFERABLE_PARAM["LayerNorm"], (input_shape[-1],), report_error)
    return input_shape


def _rule_pool2d(node: GraphNode, input_shape: Shape, report_error: ReportError) -> Optional[Shape]:
    if len(input_shape) != 3:
        report_error(
            f"Node '{node.id}': {node.layer_type} expects a 3D "
            f"(channels, height, width) input shape, got {input_shape}.",
            node.id, node.line, Severity.ERROR,
        )
        return UNKNOWN
    if any(p not in node.params for p in ("kernel", "stride")):
        return UNKNOWN
    c, h, w = input_shape
    k, s = node.params["kernel"], node.params["stride"]
    h2 = (h - k) // s + 1
    w2 = (w - k) // s + 1
    if h2 <= 0 or w2 <= 0:
        report_error(
            f"Node '{node.id}': {node.layer_type} parameters produce a "
            f"non-positive output spatial size ({h2}x{w2}) from input "
            f"{input_shape}.", node.id, node.line, Severity.ERROR,
        )
        return UNKNOWN
    return (c, h2, w2)


def _rule_multihead_attn(node: GraphNode, input_shape: Shape, report_error: ReportError) -> Shape:
    _check_or_fill(node, INFERABLE_PARAM["MultiHeadAttn"], input_shape[-1], report_error)
    return input_shape


def _rule_rnn(node: GraphNode, input_shape: Shape, report_error: ReportError) -> Optional[Shape]:
    _check_or_fill(node, INFERABLE_PARAM[node.layer_type], input_shape[-1], report_error)
    if "hidden_size" not in node.params:
        return UNKNOWN
    return input_shape[:-1] + (node.params["hidden_size"],)


def _rule_unknown(node: GraphNode, input_shape: Shape, report_error: ReportError) -> None:
    return UNKNOWN  # Flatten, Embedding -- see module docstring


_LAYER_RULES: dict[str, Callable[[GraphNode, Shape, ReportError], Optional[Shape]]] = {
    "Linear": _rule_linear,
    "Conv2d": _rule_conv2d,
    "Conv1d": _rule_conv1d,
    "BatchNorm2d": _rule_batchnorm2d,
    "LayerNorm": _rule_layernorm,
    "MaxPool2d": _rule_pool2d,
    "AvgPool2d": _rule_pool2d,
    "MultiHeadAttn": _rule_multihead_attn,
    "LSTM": _rule_rnn,
    "GRU": _rule_rnn,
    "Flatten": _rule_unknown,
    "Embedding": _rule_unknown,
}


def _apply_layer_rule(node: GraphNode, input_shape: Optional[Shape], report_error: ReportError) -> Optional[Shape]:
    if input_shape is UNKNOWN:
        return UNKNOWN
    try:
        return _LAYER_RULES.get(node.layer_type, _passthrough)(node, input_shape, report_error)
    except (TypeError, ValueError, KeyError, ZeroDivisionError):
        # A malformed param (wrong type, zero stride, ...) would
        # already have been reported by _check_param_types --
        # inference degrades to UNKNOWN silently here rather than
        # raising a second, redundant diagnostic or crashing the
        # whole analyze() call over a problem that's already being
        # reported elsewhere.
        return UNKNOWN


# ------------------------------------------------------------
# Multi-input structural ops with their own shape rules (not a simple
# "must match" sum -- Concat merges along one dim, Split divides one)
# ------------------------------------------------------------

def _translate_dim(
    dim: int, rank: int, node: GraphNode, op_name: str, report_error: ReportError
) -> Optional[int]:
    """Converts a DSL `dim` value (written assuming the batch-included
    runtime tensor) into the correct index into a batch-less shape
    tuple of the given rank. See the module docstring's "dim values
    are batch-inclusive" section for the full reasoning -- summary:
    dim=1 in the DSL means channel-axis-of-a-4D-tensor, which is index
    0 of our 3D (C, H, W) tuple, so non-negative dims shift down by
    one; dim=-1 ("last axis") needs no shift at all.

    Returns None (after reporting an error) if dim==0 -- the batch
    axis itself, not a valid target for a per-sample operation in a
    DSL whose tracked shapes don't model batch at all -- or if the
    translated index is out of range for `rank`.
    """
    if dim == 0:
        report_error(
            f"Node '{node.id}': {op_name} dim=0 refers to the batch "
            f"dimension, which this DSL's shapes don't model (shapes "
            f"are per-sample; batch only exists via config.batch_size "
            f"in the generated __main__ scaffold). Did you mean dim=1, "
            f"the first per-sample dimension?",
            node.id, node.line, Severity.ERROR,
        )
        return None
    adjusted = dim - 1 if dim > 0 else dim
    if not (-rank <= adjusted < rank):
        report_error(
            f"Node '{node.id}': {op_name} dim={dim} is out of range. "
            f"DSL dim values count the batch dimension (dim=1 is the "
            f"first per-sample axis, dim=2 the second, ...), so "
            f"dim={dim} refers to per-sample axis {adjusted}, but the "
            f"per-sample shape here only has {rank} dimension(s).",
            node.id, node.line, Severity.ERROR,
        )
        return None
    return adjusted if adjusted >= 0 else adjusted + rank


def _concat_shape(
    node: GraphNode, pred_shapes: list[Optional[Shape]], report_error: ReportError
) -> Optional[Shape]:
    if not pred_shapes or any(s is UNKNOWN for s in pred_shapes):
        return UNKNOWN
    if "dim" not in node.params:
        return UNKNOWN
    dim = node.params["dim"]
    first = pred_shapes[0]
    rank = len(first)
    norm_dim = _translate_dim(dim, rank, node, "Concat", report_error)
    if norm_dim is None:
        return UNKNOWN
    for s in pred_shapes[1:]:
        if len(s) != rank:
            report_error(
                f"Node '{node.id}': Concat inputs have different ranks "
                f"({first} vs {s}); all inputs must have the same "
                f"number of dimensions.", node.id, node.line, Severity.ERROR,
            )
            return UNKNOWN
        for i, (a, b) in enumerate(zip(first, s)):
            if i != norm_dim and a != b:
                report_error(
                    f"Node '{node.id}': Concat inputs have mismatched "
                    f"shapes {first} and {s} outside the concat "
                    f"dimension (dim={dim}).",
                    node.id, node.line, Severity.ERROR,
                )
                return UNKNOWN
    concat_size = sum(s[norm_dim] for s in pred_shapes)
    return first[:norm_dim] + (concat_size,) + first[norm_dim + 1:]


def _split_shape(
    node: GraphNode, input_shape: Optional[Shape], report_error: ReportError
) -> Optional[Shape]:
    if input_shape is UNKNOWN:
        return UNKNOWN
    if any(p not in node.params for p in ("chunks", "dim")):
        return UNKNOWN
    chunks, dim = node.params["chunks"], node.params["dim"]
    rank = len(input_shape)
    norm_dim = _translate_dim(dim, rank, node, "Split", report_error)
    if norm_dim is None:
        return UNKNOWN
    size = input_shape[norm_dim]
    if chunks <= 0 or size % chunks != 0:
        report_error(
            f"Node '{node.id}': Split cannot divide dimension size "
            f"{size} evenly into {chunks} chunks.",
            node.id, node.line, Severity.ERROR,
        )
        return UNKNOWN
    chunk_size = size // chunks
    # Every chunk has this same shape -- the DSL has no syntax for
    # requesting unevenly-sized chunks (torch.chunk itself would
    # produce a smaller final chunk if the size didn't divide evenly,
    # but that case is rejected above rather than modeled precisely).
    # This is also why storing ONE shape per node works uniformly even
    # for Split: whichever downstream edge consumes shapes[split_id],
    # it's getting an identical-shaped chunk either way.
    return input_shape[:norm_dim] + (chunk_size,) + input_shape[norm_dim + 1:]
