"""
Manual smoke test for Phases 1-2 together.
Intended location: scripts/parse_demo.py (run from the project root)

Usage, after generating the parser:
    java -jar antlr-4.13.1-complete.jar -Dlanguage=Python3 -visitor \
         -o generated/ grammar/NNGraph.g4
    pip install antlr4-python3-runtime --break-system-packages
    python3 scripts/parse_demo.py tests/fixtures/mlp.nng

Prints the resulting Program dataclass tree. If this runs cleanly on
all four worked examples from the spec (mlp, resblock, inception,
transformer), Phases 1-2 are solid and Phase 3 (semantic analysis)
can build on top with confidence.
"""

import sys

from antlr4 import CommonTokenStream, FileStream
from gen.NNGraphLexer import NNGraphLexer
from gen.NNGraphParser import NNGraphParser

from nngraph.ast_builder import ASTBuilder


def main(path: str) -> None:
    char_stream = FileStream(path, encoding="utf-8")
    lexer = NNGraphLexer(char_stream)
    token_stream = CommonTokenStream(lexer)
    parser = NNGraphParser(token_stream)

    tree = parser.program()  # entry rule -- triggers the full parse
    if parser.getNumberOfSyntaxErrors() > 0:
        # Phase 5 will replace this with structured Diagnostic objects;
        # for now the default ConsoleErrorListener has already printed
        # ANTLR's own syntax error messages to stderr.
        sys.exit(1)

    program = ASTBuilder().visit(tree)
    print(program)


if __name__ == "__main__":
    path = "./mlp.nng"
    main(path)
