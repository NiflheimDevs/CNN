#!/usr/bin/env python3
"""
compiler.py

Unified command-line entry point for the NNGraph DSL compiler,
tying all phases together as specified in Section 8.3 of the spec.

Two modes:

1. COMPILE MODE (default):
    python nngraph_compiler.py my_model.nng --output my_model.py

   Pipeline:
    1. Lex + parse (ANTLR) with CollectingErrorListener
    2. GATE 1: syntax errors → print and stop
    3. Build AST
    4. Run semantic analysis (including shape inference)
    5. GATE 2: semantic errors → print all diagnostics and stop
    6. Generate code and write to output file

2. VISUALIZE MODE (--dot):
    python nngraph_compiler.py my_model.nng --dot --output my_model.png

   Pipeline (deliberately shorter):
    1. Lex + parse with CollectingErrorListener
    2. GATE 1: syntax errors → print and stop (ONLY gate)
    3. Build AST
    4. Generate DOT source and render image

   No semantic analysis -- graphs with undefined edges, cycles, or
   orphan nodes still have syntax and will still draw, letting you
   SEE what's wrong before it compiles.

Assumes ANTLR-generated lexer/parser exist in gen/:
    java -jar antlr-4.13.1-complete.jar -Dlanguage=Python3 -visitor \
         -o gen/ grammar/NNGraph.g4
"""

import argparse
import sys
from pathlib import Path

from antlr4 import CommonTokenStream, FileStream

from gen.NNGraphLexer import NNGraphLexer
from gen.NNGraphParser import NNGraphParser

from nngraph.ast_builder import ASTBuilder
from nngraph.codegen import CodegenError, generate
from nngraph.diagnostics import Diagnostic, has_errors, sort_diagnostics
from nngraph.dot_visualizer import DotVisualizerError, generate_dot, render_dot
from nngraph.error_listener import CollectingErrorListener
from nngraph.semantic_analyzer import SemanticAnalyzer


def compile_file(input_path: str, output_path: str) -> int:
    """Runs the full compilation pipeline. Returns exit code
    (0 = success, 1 = failure).
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

    # GATE 1 -- syntax
    if has_errors(listener.diagnostics):
        _print_diagnostics(listener.diagnostics)
        return 1

    program = ASTBuilder().visit(tree)

    analyzer = SemanticAnalyzer(program)
    analyzer.analyze()
    diagnostics = analyzer.to_diagnostics()

    # GATE 2 -- semantics
    _print_diagnostics(diagnostics)
    if has_errors(diagnostics):
        return 1

    try:
        source = generate(program)
    except CodegenError as e:
        print(f"error: code generation failed: {e}", file=sys.stderr)
        return 1

    try:
        Path(output_path).write_text(source, encoding="utf-8")
    except OSError as e:
        print(f"error: could not write '{output_path}': {e}", file=sys.stderr)
        return 1

    print(f"Compiled '{input_path}' -> '{output_path}'")
    return 0


def visualize_file(input_path: str, output_path: str, fmt: str, dot_only: bool) -> int:
    """Runs the visualization pipeline. Returns exit code
    (0 = success, 1 = failure).
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

    # GATE 1 -- syntax only (no semantic analysis)
    if has_errors(listener.diagnostics):
        _print_diagnostics(listener.diagnostics)
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
        print(f"warning: {e}", file=sys.stderr)
        return 1

    print(f"Rendered '{input_path}' -> '{output_path}'")
    return 0


def _print_diagnostics(diagnostics: list[Diagnostic]) -> None:
    for d in sort_diagnostics(diagnostics):
        print(d.format(), file=sys.stderr)


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        prog="compiler.py",
        description=(
            "NNGraph DSL compiler. Compiles .nng files into PyTorch nn.Module "
            "code (.py), or renders node graph diagrams with --dot."
        ),
    )
    arg_parser.add_argument("input", help="Path to the .nng source file")
    arg_parser.add_argument(
        "--output", "-o",
        help=(
            "Output path. Default: input path with .py extension (compile mode) "
            "or .png extension (--dot mode)."
        ),
    )
    arg_parser.add_argument(
        "--dot",
        action="store_true",
        help=(
            "Visualize mode: render the node graph as a Graphviz diagram. "
            "Parses syntax only -- no semantic analysis -- so this also works "
            "on graphs that don't compile yet."
        ),
    )
    arg_parser.add_argument(
        "--format", "-T",
        default=None,
        help=(
            "Graphviz output format in --dot mode (png, svg, pdf, ...). "
            "Default: inferred from --output's extension, else png."
        ),
    )
    arg_parser.add_argument(
        "--dot-only",
        action="store_true",
        help="In --dot mode: only write the .dot source file; skip rendering an image.",
    )
    args = arg_parser.parse_args()

    if args.dot:
        # Visualization mode
        default_ext = ".png"
        output_path = args.output or str(Path(args.input).with_suffix(default_ext))
        fmt = args.format or (
            Path(output_path).suffix[1:] if Path(output_path).suffix else "png"
        )
        sys.exit(visualize_file(args.input, output_path, fmt, args.dot_only))
    else:
        # Compile mode
        output_path = args.output or str(Path(args.input).with_suffix(".py"))
        sys.exit(compile_file(args.input, output_path))


if __name__ == "__main__":
    main()
