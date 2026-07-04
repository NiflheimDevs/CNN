#!/usr/bin/env python3
"""
nngraph_visualize.py

Node graph visualization CLI -- the sibling entry point to
nngraph_compiler.py that TODO.md's last unchecked box asks for:

    python nngraph_visualize.py my_model.nng --output my_model.png

Pipeline, deliberately shorter than nngraph_compiler.py's:

    1. Lex + parse (ANTLR), same CollectingErrorListener wiring as the
       compiler.
    2. GATE 1 -- syntax only. Same reasoning as nngraph_compiler.py:
       ASTBuilder must never run on a parse tree produced with error
       recovery. This is the ONLY gate here.
    3. Build the AST (ASTBuilder) and hand it straight to
       dot_visualizer -- SemanticAnalyzer is never constructed, never
       imported. A graph with an undefined edge target, a cycle, or an
       orphan node still has syntax; it will still draw. See
       dot_visualizer.py's module docstring for why that's the point,
       not an oversight -- this tool exists to let someone SEE what's
       wrong with a graph that doesn't compile yet, not just to
       decorate a graph that already does.

Input can therefore be a .nng file in any state: fully valid, valid
syntax with semantic errors, or -- if the user hands this tool
something that was never NNGraph source to begin with -- a syntax
error, in which case GATE 1 reports it exactly the way the compiler
does and stops, since there is no tree to draw from at all.

Assumes the ANTLR-generated lexer/parser already exist (see gen/), the
same precondition nngraph_compiler.py documents.
"""

import argparse
import sys
from pathlib import Path

from antlr4 import CommonTokenStream, FileStream

from gen.NNGraphLexer import NNGraphLexer
from gen.NNGraphParser import NNGraphParser

from nngraph.ast_builder import ASTBuilder
from nngraph.diagnostics import has_errors, sort_diagnostics
from nngraph.dot_visualizer import DotVisualizerError, generate_dot, render_dot
from nngraph.error_listener import CollectingErrorListener


def visualize_file(input_path: str, output_path: str, fmt: str, dot_only: bool) -> int:
    """Mirrors compile_file()'s shape in nngraph_compiler.py (return an
    exit code, never call sys.exit() directly) so this is equally
    callable from tests or other tooling.
    """
    try:
        char_stream = FileStream(input_path, encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"error: could not read '{input_path}': {e}", file=sys.stderr)
        return 1

    listener = CollectingErrorListener()

    lexer = NNGraphLexer(char_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)

    token_stream = CommonTokenStream(lexer)
    parser = NNGraphParser(token_stream)
    parser.removeErrorListeners()
    parser.addErrorListener(listener)

    tree = parser.program()

    # GATE 1 -- the only gate. No Gate 2: see the module docstring for
    # why semantic analysis is deliberately never run here.
    if has_errors(listener.diagnostics):
        for d in sort_diagnostics(listener.diagnostics):
            print(d.format(), file=sys.stderr)
        return 1

    program = ASTBuilder().visit(tree)
    dot_source = generate_dot(program)

    dot_path = Path(output_path).with_suffix(".dot")
    try:
        dot_path.write_text(dot_source, encoding="utf-8")
    except OSError as e:
        print(f"error: could not write '{dot_path}': {e}", file=sys.stderr)
        return 1
    print(f"Wrote DOT source to '{dot_path}'")

    if dot_only:
        return 0

    try:
        render_dot(dot_source, output_path, fmt=fmt)
    except DotVisualizerError as e:
        # The .dot file above was already written successfully, so a
        # missing/failing `dot` binary is reported but not fatal to
        # the overall run -- there's still a usable artifact on disk.
        print(f"warning: {e}", file=sys.stderr)
        return 1

    print(f"Rendered '{input_path}' -> '{output_path}'")
    return 0


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        prog="nngraph_visualize.py",
        description=(
            "Render an NNGraph DSL (.nng) file's node graph as a Graphviz "
            "diagram. Parses syntax only -- no semantic analysis -- so this "
            "also works on graphs that don't compile yet."
        ),
    )
    arg_parser.add_argument("input", help="Path to the .nng source file")
    arg_parser.add_argument(
        "--output", "-o",
        help="Path to write the rendered image (default: input path with .png extension)",
    )
    arg_parser.add_argument(
        "--format", "-T",
        default=None,
        help="Graphviz output format (png, svg, pdf, ...). Default: inferred from --output's extension, else png.",
    )
    arg_parser.add_argument(
        "--dot-only",
        action="store_true",
        help="Only write the .dot source file; skip invoking Graphviz to render an image.",
    )
    args = arg_parser.parse_args()

    output_path = args.output or str(Path(args.input).with_suffix(".png"))
    fmt = args.format or (Path(output_path).suffix[1:] if Path(output_path).suffix else "png")
    sys.exit(visualize_file(args.input, output_path, fmt, args.dot_only))


if __name__ == "__main__":
    main()
