"""
nngraph/dot_visualizer.py

Node graph visualization (the last unchecked box in TODO.md).

Turns a Program (ast_nodes.py) into Graphviz DOT source and, if the
`dot` binary is available on PATH, renders it to an image. Nothing in
this file imports semantic_analyzer.py or codegen.py, and nothing here
calls SemanticAnalyzer.analyze() -- on purpose:

    - This is meant to be usable as a *debugging* tool for a graph
      that doesn't compile yet -- an undefined edge target, a cycle, a
      disconnected node. Those are exactly the situations someone
      reaches for a picture to understand, and semantic analysis
      would refuse to hand back a usable AST in most of them (Gate 2
      in nngraph_compiler.py stops the pipeline on any ERROR
      diagnostic). Visualization has no such gate: it draws whatever
      the parser was able to build, correct or not.
    - Concretely, this means _build_dot() never calls
      graph_utils.topological_order() (which raises GraphCycleError on
      a cyclic graph) and never validates that an edge's src/dst is a
      known node id. An edge naming a node that was never declared
      with `node ...` still gets drawn -- Graphviz itself will just
      synthesize an unstyled, empty-label node for that id, which
      _is_ the correct picture of "this edge points at something that
      doesn't exist."

Only syntax has to have succeeded (i.e. ASTBuilder ran without
throwing) for this module to work -- see the module docstring on
nngraph_visualize.py for how the two-stage "parse, then maybe don't
even try to analyze" pipeline is wired up.
"""

import shutil
import subprocess
from pathlib import Path

from nngraph.ast_nodes import Program
from nngraph.layer_catalogue import STRUCTURAL_OPS

# Layer types with no backing nn.Module (Add, Concat, Residual, Split)
# are drawn as diamonds -- visually distinct from the rectangular
# "real layer" boxes, mirroring how codegen.py already treats them as
# a different kind of thing (no self.<id> registration).
_STRUCTURAL_SHAPE = "diamond"
_LAYER_SHAPE = "box"
_STRUCTURAL_FILL = "#fdebd0"
_LAYER_FILL = "#eaf2f8"
_INPUT_FILL = "#d5f5e3"
_OUTPUT_PERIPHERIES = "2"


class DotVisualizerError(Exception):
    """Raised only for things outside the AST's own content -- e.g.
    asking to render an image when the `dot` executable isn't
    installed. Never raised for anything about the *shape* of the
    graph itself (see the module docstring: that's deliberate)."""


def _escape(text: str) -> str:
    """Escape text for use inside a double-quoted DOT string. DOT's
    quoting rules only require '\"' and '\\' to be escaped -- this is
    intentionally not a generic sanitizer, just enough to keep a
    layer_type, param value, or edge label from breaking out of its
    quotes and corrupting the generated syntax."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _format_value(value) -> str:
    """Render one AST literal (see ast_nodes.Value) the way a human
    reads it in the .nng source, for use inside a node's label --
    deliberately separate from codegen._format_value, which renders
    Python *source* (e.g. wraps strings in repr()). A label is prose,
    not code: True stays "True", a shape stays "(3, 224, 224)", a
    string doesn't grow an extra layer of escaped quotes.
    """
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, tuple):
        return "(" + ", ".join(str(v) for v in value) + ")"
    return str(value)


def _node_label(node) -> str:
    """Builds the label text with a literal DOT line-break (a single
    backslash followed by 'n', which Graphviz's label renderer treats
    specially) between the node id and its layer/params -- NOT a
    Python '\\n'. Each dynamic piece (id, layer_type, param values) is
    escaped individually and the '\\n' separator is spliced in
    afterwards, so a later _escape() pass over the whole label can
    never double-escape that separator into a literal backslash-n.
    """
    params = ", ".join(f"{_escape(str(k))}={_escape(_format_value(v))}" for k, v in node.params.items())
    return f"{_escape(node.id)}\\n{_escape(node.layer_type)}({params})"


def generate_dot(program: Program) -> str:
    """Builds Graphviz DOT source text for `program`'s graph block.
    Pure string assembly -- no Graphviz library, no `dot` subprocess,
    so this half of the module works even in an environment that
    doesn't have Graphviz installed at all (render_dot() is the only
    part that needs the binary).
    """
    lines = [f'digraph "{_escape(program.model.name)}" {{', "  rankdir=LR;", '  node [fontname="Helvetica"];']

    # Virtual input node -- not part of program.graph.nodes (it lives
    # on Model/InputDecl instead), but it's the one true source of the
    # graph and every real worked example starts an edge from it, so
    # it earns its own node statement rather than being left implicit.
    input_decl = program.model.input
    input_label = f"{_escape(input_decl.name)}\\ninput {_escape(_format_value(input_decl.shape))}"
    lines.append(
        f'  "{_escape(input_decl.name)}" '
        f'[label="{input_label}", shape=ellipse, style=filled, fillcolor="{_INPUT_FILL}"];'
    )

    output_id = program.model.output
    for node in program.graph.nodes:
        shape = _STRUCTURAL_SHAPE if node.layer_type in STRUCTURAL_OPS else _LAYER_SHAPE
        fill = _STRUCTURAL_FILL if node.layer_type in STRUCTURAL_OPS else _LAYER_FILL
        extra = f", peripheries={_OUTPUT_PERIPHERIES}" if node.id == output_id else ""
        lines.append(
            f'  "{_escape(node.id)}" '
            f'[label="{_node_label(node)}", shape={shape}, style=filled, fillcolor="{fill}"{extra}];'
        )

    for edge in program.graph.edges:
        attrs = f' [label="{_escape(edge.label)}"]' if edge.label else ""
        lines.append(f'  "{_escape(edge.src)}" -> "{_escape(edge.dst)}"{attrs};')

    lines.append("}")
    return "\n".join(lines) + "\n"


def render_dot(dot_source: str, output_path: str, fmt: str = "png") -> None:
    """Shells out to the `dot` CLI (part of any Graphviz install) to
    render `dot_source` to `output_path`. Deliberately a subprocess
    call rather than a dependency on the `graphviz` PyPI package --
    that package is itself just a thin subprocess wrapper around this
    same binary, so taking on the dependency wouldn't remove the one
    real external requirement (Graphviz has to be installed), it would
    just add a second, unnecessary one for no benefit here.
    """
    if shutil.which("dot") is None:
        raise DotVisualizerError(
            "the 'dot' executable was not found on PATH -- install Graphviz "
            "(e.g. `apt install graphviz` / `brew install graphviz`) to render "
            "images. The .dot source itself can still be generated and opened "
            "with any Graphviz viewer, or pasted into https://dreampuf.github.io/GraphvizOnline/."
        )

    try:
        subprocess.run(
            ["dot", f"-T{fmt}", "-o", str(output_path)],
            input=dot_source,
            text=True,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        # dot's own stderr (e.g. "syntax error in line N") is the most
        # useful diagnostic available here -- passed through as-is,
        # same philosophy as CollectingErrorListener passing ANTLR's
        # own syntax-error text through untouched.
        raise DotVisualizerError(f"'dot' failed to render {output_path}: {e.stderr.strip()}") from e


def visualize(program: Program, output_path: str, fmt: str | None = None) -> str:
    """Convenience wrapper: generate DOT source, write it next to
    `output_path` (same stem, .dot extension) so the raw source is
    always available even if rendering fails or Graphviz isn't
    installed, then render to `output_path`. Returns the DOT source
    (callers that only want the text -- e.g. nngraph_visualize.py's
    --dot-only mode -- can skip calling render_dot() entirely).
    """
    dot_source = generate_dot(program)

    out = Path(output_path)
    dot_path = out.with_suffix(".dot")
    dot_path.write_text(dot_source, encoding="utf-8")

    resolved_fmt = fmt or (out.suffix[1:] if out.suffix else "png")
    render_dot(dot_source, str(out), fmt=resolved_fmt)
    return dot_source
