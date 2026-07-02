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

def show_ast(ast_root_node):
    import networkx as nx
    from matplotlib import pyplot as plt
    from networkx.drawing.nx_pydot import graphviz_layout

    graph = transform_ast_to_networkx(ast_root_node)
    pos = graphviz_layout(graph, prog="dot")
    nx.draw(graph, pos, node_size=500, labels=nx.get_node_attributes(graph, "label"),
            alpha=0.5, node_color="cyan", with_labels=True)
    ax = plt.gca()
    ax.margins(0.20)
    plt.axis("off")
    plt.show()


def transform_ast_to_networkx(node):
    """Convert AST nodes to NetworkX graph for visualization."""
    import networkx as nx
    from dataclasses import is_dataclass, fields

    G = nx.DiGraph()
    node_counter = [0]

    def add_node_recursive(obj, parent_id=None):
        current_id = node_counter[0]
        node_counter[0] += 1

        # Determine label
        if is_dataclass(obj):
            label = type(obj).__name__
        elif isinstance(obj, list):
            label = f"List[{len(obj)}]"
        else:
            label = str(obj)

        G.add_node(current_id, label=label)

        if parent_id is not None:
            G.add_edge(parent_id, current_id)

        # Recurse into structure
        if is_dataclass(obj):
            for field in fields(obj):
                child = getattr(obj, field.name)
                if child is not None:
                    add_node_recursive(child, current_id)
        elif isinstance(obj, list):
            for item in obj:
                add_node_recursive(item, current_id)

        return current_id

    add_node_recursive(node)
    return G


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

    show_ast(program)


if __name__ == "__main__":
    path = "./mlp.nng"
    main(path)
