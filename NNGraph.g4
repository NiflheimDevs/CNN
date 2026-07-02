grammar NNGraph;

/* ============================================================
 * NNGraph DSL — ANTLR4 Combined Grammar
 *
 * Source: NNGraph DSL Specification, Section 3 (Language Spec)
 * Target: Python3 (per Section 8.2 toolchain — antlr4-python3-runtime)
 *
 * Generate with:
 *   java -jar antlr-4.13.1-complete.jar -Dlanguage=Python3 -visitor \
 *        -o generated/ NNGraph.g4
 * ============================================================ */

// ------------------------------------------------------------
// PARSER RULES
// ------------------------------------------------------------

program
    : modelDecl graphBlock configBlock? EOF
    ;

// --- 3.1 Model declaration -----------------------------------
modelDecl
    : 'model' ID '{' inputDecl outputDecl '}'
    ;

inputDecl
    : 'input' ID ':' 'tensor' '(' shapeArgs ')'
    ;

outputDecl
    : 'output' ID
    ;

shapeArgs
    : INT (',' INT)*
    ;

// --- 3.1 Graph block -------------------------------------------
graphBlock
    : 'graph' '{' graphStmt* '}'
    ;

graphStmt
    : nodeDecl
    | edgeDecl
    ;

// --- Node declarations (3.3 layers, 3.4 activations, 3.5 ops) --
// layerType is deliberately a bare ID, not a closed set of
// keywords — the known-layer registry lives in the semantic
// analyzer, not the grammar. See summary for rationale.
nodeDecl
    : 'node' ID ':' layerType '(' paramList? ')'
    ;

layerType
    : ID
    ;

paramList
    : param (',' param)*
    ;

param
    : ID '=' value
    ;

// --- 3.2 Data types ---------------------------------------------
value
    : INT
    | FLOAT
    | STRING
    | BOOL
    | NONE
    | shapeLiteral
    ;

shapeLiteral
    : '(' INT (',' INT)* ')'
    ;

// --- 3.6 Edge syntax ----------------------------------------------
edgeDecl
    : 'edge' ID '->' ID edgeLabel?
    ;

edgeLabel
    : '[' 'label' '=' STRING ']'
    ;

// --- 3.1 Config block ----------------------------------------------
configBlock
    : 'config' '{' configEntry* '}'
    ;

configEntry
    : ID '=' value
    ;

// ------------------------------------------------------------
// LEXER RULES
// ------------------------------------------------------------
// Structural keywords ('model', 'input', 'output', 'tensor',
// 'graph', 'node', 'edge', 'label', 'config', '->') are NOT
// declared explicitly below. ANTLR4 auto-generates an implicit
// token for every quoted literal used in a parser rule, and in
// a combined grammar those implicit tokens are registered in
// the order they're first encountered while the file is read
// top-to-bottom. Since every keyword literal above appears
// before this lexer section, they all take priority over ID on
// tie-break — see the written summary for the full mechanics.

BOOL   : 'true' | 'false' ;
NONE   : 'None' ;

FLOAT  : '-'? [0-9]+ '.' [0-9]+ ;
INT    : '-'? [0-9]+ ;

ID     : [a-zA-Z_][a-zA-Z_0-9]* ;

STRING
    : '"' (ESC | ~["\\\r\n])* '"'
    ;
fragment ESC : '\\' [\\"ntr] ;

LINE_COMMENT
    : '//' ~[\r\n]* -> skip
    ;

BLOCK_COMMENT
    : '/*' .*? '*/' -> skip
    ;

WS
    : [ \t\r\n]+ -> skip
    ;
