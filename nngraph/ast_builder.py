"""
Phase 2 -- ANTLR4 parse-tree visitor that builds the dataclass AST
defined in ast_nodes.py.
Intended location in the project layout: src/nngraph/ast_builder.py

This is the ONLY file in the compiler that imports from `generated/`
(the ANTLR-generated lexer/parser/visitor). Everything downstream --
semantic analysis, codegen -- depends only on ast_nodes.py types.

Requires the parser to already be generated:
    java -jar antlr-4.13.1-complete.jar -Dlanguage=Python3 -visitor \
         -o generated/ grammar/NNGraph.g4
"""

import re

from gen.NNGraphParser import NNGraphParser
from gen.NNGraphVisitor import NNGraphVisitor

from nngraph.ast_nodes import Config, Edge, Graph, GraphNode, InputDecl, Model, Program

_ESCAPES = {'"': '"', "\\": "\\", "n": "\n", "t": "\t", "r": "\r"}


def _unquote(raw_string_token: str) -> str:
    """Strip the surrounding quotes from a STRING token's raw text and
    resolve backslash escapes. `raw_string_token` still has its opening
    and closing '"' characters -- ANTLR terminal nodes always hand back
    the untouched source text via getText(), lexer rule and all."""
    inner = raw_string_token[1:-1]
    return re.sub(r"\\(.)", lambda m: _ESCAPES.get(m.group(1), m.group(1)), inner)


class ASTBuilder(NNGraphVisitor):
    """Walks a parse tree produced by NNGraphParser.program() and
    returns a Program.

    Every method below explicitly calls self.visit(...) on the exact
    sub-contexts it needs and assembles a dataclass from the results.
    None of them fall back on the base class's default visitChildren()
    traversal. That default only keeps the LAST child's result unless
    you override aggregateResult() -- relying on it here would silently
    drop every graph node but the final one. Writing each visitX
    explicitly sidesteps that trap entirely, at the cost of one method
    per rule you actually care about.

    Rules that exist purely for grammar structure -- graphStmt (a bare
    nodeDecl | edgeDecl alternative) and layerType (a bare ID) -- have
    no corresponding visit method here, because nothing ever calls
    self.visit() on a context of that type; call sites drill straight
    into the matched child instead. You only need to implement visitX
    for rules you actually dispatch through.
    """

    # ---- Top level ---------------------------------------------------

    def visitProgram(self, ctx: NNGraphParser.ProgramContext) -> Program:
        model = self.visit(ctx.modelDecl())
        graph = self.visit(ctx.graphBlock())
        config = self.visit(ctx.configBlock()) if ctx.configBlock() else None
        return Program(model=model, graph=graph, config=config)

    # ---- Model block ---------------------------------------------------

    def visitModelDecl(self, ctx: NNGraphParser.ModelDeclContext) -> Model:
        # Exactly one ID token appears in 'model' ID '{' ... '}', so
        # ctx.ID() returns a single TerminalNode -- contrast with
        # visitEdgeDecl below, where ID appears twice in one alternative.
        name = ctx.ID().getText()
        input_decl = self.visit(ctx.inputDecl())
        output_name = ctx.outputDecl().ID().getText()
        return Model(name=name, input=input_decl, output=output_name, line=ctx.start.line)

    def visitInputDecl(self, ctx: NNGraphParser.InputDeclContext) -> InputDecl:
        name = ctx.ID().getText()
        shape = self.visit(ctx.shapeArgs())
        return InputDecl(name=name, shape=shape, line=ctx.start.line)

    def visitShapeArgs(self, ctx: NNGraphParser.ShapeArgsContext) -> tuple:
        # INT appears via 'INT (',' INT)*' -- a repeated token -- so
        # ctx.INT() with NO index returns a list of every INT terminal
        # matched, in source order. ctx.INT(i) would return just the i-th.
        return tuple(int(tok.getText()) for tok in ctx.INT())

    # ---- Graph block ---------------------------------------------------

    def visitGraphBlock(self, ctx: NNGraphParser.GraphBlockContext) -> Graph:
        nodes: list[GraphNode] = []
        edges: list[Edge] = []
        for stmt in ctx.graphStmt():
            # graphStmt : nodeDecl | edgeDecl -- exactly one accessor
            # below is non-None per stmt, since it's a pure alternative.
            if stmt.nodeDecl():
                nodes.append(self.visit(stmt.nodeDecl()))
            else:
                edges.append(self.visit(stmt.edgeDecl()))
        return Graph(nodes=nodes, edges=edges, line=ctx.start.line)

    def visitNodeDecl(self, ctx: NNGraphParser.NodeDeclContext) -> GraphNode:
        node_id = ctx.ID().getText()
        layer_type = ctx.layerType().getText()
        params = self.visit(ctx.paramList()) if ctx.paramList() else {}
        return GraphNode(id=node_id, layer_type=layer_type, params=params, line=ctx.start.line)

    def visitParamList(self, ctx: NNGraphParser.ParamListContext) -> dict:
        params: dict = {}
        for param_ctx in ctx.param():
            key, value = self.visit(param_ctx)
            params[key] = value
        return params

    def visitParam(self, ctx: NNGraphParser.ParamContext) -> tuple:
        key = ctx.ID().getText()
        value = self.visit(ctx.value())
        return key, value

    def visitValue(self, ctx: NNGraphParser.ValueContext):
        # value : INT | FLOAT | STRING | BOOL | NONE | shapeLiteral --
        # another pure alternative, so exactly one accessor is non-None.
        if ctx.INT():
            return int(ctx.INT().getText())
        if ctx.FLOAT():
            return float(ctx.FLOAT().getText())
        if ctx.STRING():
            return _unquote(ctx.STRING().getText())
        if ctx.BOOL():
            return ctx.BOOL().getText() == "true"
        if ctx.NONE():
            return None
        if ctx.shapeLiteral():
            return self.visit(ctx.shapeLiteral())
        raise AssertionError(f"unreachable: no value alternative matched at line {ctx.start.line}")

    def visitShapeLiteral(self, ctx: NNGraphParser.ShapeLiteralContext) -> tuple:
        return tuple(int(tok.getText()) for tok in ctx.INT())

    def visitEdgeDecl(self, ctx: NNGraphParser.EdgeDeclContext) -> Edge:
        # ID appears twice in 'edge' ID '->' ID, so ctx.ID() with no
        # index returns [srcNode, dstNode] in source order. Indexing
        # explicitly (rather than unpacking) keeps 0=src, 1=dst legible
        # at the call site.
        src = ctx.ID(0).getText()
        dst = ctx.ID(1).getText()
        label = _unquote(ctx.edgeLabel().STRING().getText()) if ctx.edgeLabel() else None
        return Edge(src=src, dst=dst, label=label, line=ctx.start.line)

    # ---- Config block ---------------------------------------------------

    def visitConfigBlock(self, ctx: NNGraphParser.ConfigBlockContext) -> Config:
        entries: dict = {}
        for entry_ctx in ctx.configEntry():
            key, value = self.visit(entry_ctx)
            entries[key] = value
        return Config(entries=entries, line=ctx.start.line)

    def visitConfigEntry(self, ctx: NNGraphParser.ConfigEntryContext) -> tuple:
        key = ctx.ID().getText()
        value = self.visit(ctx.value())
        return key, value
