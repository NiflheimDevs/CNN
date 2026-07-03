#!/usr/bin/env python3
"""
nngraph_compiler.py

Phase 6: the actual command-line entry point tying every previous
phase together, matching Section 8.3 of the spec exactly:

    python nngraph_compiler.py my_model.nng --output my_model.py

Pipeline, in order:
    1. Lex + parse (ANTLR), with syntax errors collected via
       CollectingErrorListener instead of printed to stderr by
       ANTLR's own default listener.
    2. GATE 1: if there are syntax errors, print them and stop. Do NOT
       build an AST from a parse tree that used error recovery -- see
       error_listener.py's module docstring for why that's unsafe.
    3. Build the AST (ASTBuilder).
    4. Run semantic analysis (SemanticAnalyzer.analyze()), which also
       runs shape inference internally and auto-fills what it can.
    5. GATE 2: if semantic analysis found any ERROR-severity
       diagnostic, print all diagnostics (errors AND warnings) and
       stop. Warnings alone never block.
    6. Generate code (codegen.generate()) and write it to the output
       path.

Two separate gates, not one "any diagnostics means stop" check --
a syntax error and a semantic error are unsafe for different reasons
at different points in the pipeline. See the inline comment at each
gate below.

Assumes the ANTLR-generated lexer/parser already exist:
    java -jar antlr-4.13.1-complete.jar -Dlanguage=Python3 -visitor \
         -o generated/ grammar/NNGraph.g4
and that this script is run from the project root, with `generated/`
and `nngraph/` both importable as top-level packages from there.
"""

import argparse
import sys
from pathlib import Path

from antlr4 import CommonTokenStream, FileStream

from generated.NNGraphLexer import NNGraphLexer
from generated.NNGraphParser import NNGraphParser

from nngraph.ast_builder import ASTBuilder
from nngraph.codegen import CodegenError, generate
from nngraph.diagnostics import Diagnostic, has_errors, sort_diagnostics
from nngraph.error_listener import CollectingErrorListener
from nngraph.semantic_analyzer import SemanticAnalyzer


def compile_file(input_path: str, output_path: str) -> int:
    """Runs the full pipeline once on a single file. Returns a process
    exit code (0 = success, 1 = failure) rather than calling
    sys.exit() directly -- that keeps this function callable from
    tests or other tooling without killing the interpreter. main()
    below is the only place in this file that actually exits.
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

    # GATE 1 -- syntax. A parse tree produced after a syntax error used
    # ANTLR's error-recovery strategy to keep going, which can contain
    # synthesized or mismatched subtrees that don't correspond to
    # anything the user actually wrote. Building an AST from that is
    # unsafe (crash, or worse, a silently-wrong Program) -- stop here,
    # unconditionally, before ASTBuilder ever runs.
    if has_errors(listener.diagnostics):
        _print_diagnostics(listener.diagnostics)
        return 1

    program = ASTBuilder().visit(tree)

    analyzer = SemanticAnalyzer(program)
    analyzer.analyze()
    diagnostics = analyzer.to_diagnostics()

    # GATE 2 -- semantics. Unlike Gate 1, warnings here do NOT stop the
    # pipeline -- an orphan node or an implicit-sum fan-in is valid,
    # just worth flagging -- only has_errors() (ERROR severity) does.
    # Diagnostics are printed either way, so a successful compile that
    # happens to have warnings still surfaces them instead of quietly
    # discarding them.
    _print_diagnostics(diagnostics)
    if has_errors(diagnostics):
        return 1

    try:
        source = generate(program)
    except CodegenError as e:
        # Should be unreachable if Gate 2 did its job -- every
        # CodegenError raise site in codegen.py names the specific
        # SemanticAnalyzer check that's supposed to prevent it.
        # Printed plainly rather than as a raw Python traceback if it
        # somehow still happens.
        print(f"error: code generation failed: {e}", file=sys.stderr)
        return 1

    try:
        Path(output_path).write_text(source, encoding="utf-8")
    except OSError as e:
        print(f"error: could not write '{output_path}': {e}", file=sys.stderr)
        return 1

    print(f"Compiled '{input_path}' -> '{output_path}'")
    return 0


def _print_diagnostics(diagnostics: list[Diagnostic]) -> None:
    # Diagnostics go to stderr, the success message above goes to
    # stdout (via plain print()) -- the standard Unix convention so
    # `nngraph_compiler.py model.nng 2>errors.log` or piping stdout
    # elsewhere both do the obviously-intended thing.
    for d in sort_diagnostics(diagnostics):
        print(d.format(), file=sys.stderr)


def main() -> None:
    arg_parser = argparse.ArgumentParser(
        prog="nngraph_compiler.py",
        description="Compile an NNGraph DSL (.nng) file into a PyTorch nn.Module (.py file).",
    )
    arg_parser.add_argument("input", help="Path to the .nng source file")
    arg_parser.add_argument(
        "--output", "-o",
        help="Path to write the generated .py file (default: input path with .py extension)",
    )
    args = arg_parser.parse_args()

    output_path = args.output or str(Path(args.input).with_suffix(".py"))
    sys.exit(compile_file(args.input, output_path))


if __name__ == "__main__":
    main()
