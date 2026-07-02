# Generated from C:/Users/Kiarash/Documents/college/compiler/CNN/NNGraph.g4 by ANTLR 4.13.2
from antlr4 import *
if "." in __name__:
    from .NNGraphParser import NNGraphParser
else:
    from NNGraphParser import NNGraphParser

# This class defines a complete generic visitor for a parse tree produced by NNGraphParser.

class NNGraphVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by NNGraphParser#program.
    def visitProgram(self, ctx:NNGraphParser.ProgramContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#modelDecl.
    def visitModelDecl(self, ctx:NNGraphParser.ModelDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#inputDecl.
    def visitInputDecl(self, ctx:NNGraphParser.InputDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#outputDecl.
    def visitOutputDecl(self, ctx:NNGraphParser.OutputDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#shapeArgs.
    def visitShapeArgs(self, ctx:NNGraphParser.ShapeArgsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#graphBlock.
    def visitGraphBlock(self, ctx:NNGraphParser.GraphBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#graphStmt.
    def visitGraphStmt(self, ctx:NNGraphParser.GraphStmtContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#nodeDecl.
    def visitNodeDecl(self, ctx:NNGraphParser.NodeDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#layerType.
    def visitLayerType(self, ctx:NNGraphParser.LayerTypeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#paramList.
    def visitParamList(self, ctx:NNGraphParser.ParamListContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#param.
    def visitParam(self, ctx:NNGraphParser.ParamContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#value.
    def visitValue(self, ctx:NNGraphParser.ValueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#shapeLiteral.
    def visitShapeLiteral(self, ctx:NNGraphParser.ShapeLiteralContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#edgeDecl.
    def visitEdgeDecl(self, ctx:NNGraphParser.EdgeDeclContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#edgeLabel.
    def visitEdgeLabel(self, ctx:NNGraphParser.EdgeLabelContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#configBlock.
    def visitConfigBlock(self, ctx:NNGraphParser.ConfigBlockContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by NNGraphParser#configEntry.
    def visitConfigEntry(self, ctx:NNGraphParser.ConfigEntryContext):
        return self.visitChildren(ctx)



del NNGraphParser