"""
nngraph/error_listener.py

Phase 5: bridges ANTLR4's own syntax-error reporting into the same
Diagnostic type Phase 3's semantic errors get converted to, so a
caller has exactly one list to sort, format, and gate on -- never two
independently-shaped error paths (one being raw print()s to stderr,
the other structured objects) that have to be kept in sync by hand.

USAGE (both the lexer AND the parser need this wired up separately --
see the class docstring below for why):

    listener = CollectingErrorListener()

    lexer = NNGraphLexer(char_stream)
    lexer.removeErrorListeners()
    lexer.addErrorListener(listener)

    token_stream = CommonTokenStream(lexer)
    parser = NNGraphParser(token_stream)
    parser.removeErrorListeners()
    parser.addErrorListener(listener)

    tree = parser.program()
    if has_errors(listener.diagnostics):
        # do NOT proceed to ASTBuilder -- a parse tree produced after
        # a syntax error used ANTLR's error-recovery strategy to keep
        # going, which can insert synthetic/missing tokens and
        # mismatched subtrees. ASTBuilder has no idea those are
        # synthetic and will either crash on an unexpected shape or,
        # worse, silently build a Program that doesn't represent what
        # the user actually wrote.
        ...
"""

from antlr4.error.ErrorListener import ErrorListener

from nngraph.diagnostics import Diagnostic, Severity


class CollectingErrorListener(ErrorListener):
    """Every ANTLR Lexer and Parser is constructed with exactly one
    error listener already attached: ConsoleErrorListener.INSTANCE,
    whose entire job is `print(..., file=sys.stderr)`. Calling
    addErrorListener() WITHOUT first calling removeErrorListeners()
    doesn't replace that default -- it adds a second listener
    alongside it, so every syntax error fires both: this one (silently
    collecting) and the built-in one (printing to stderr regardless of
    what the caller wants). removeErrorListeners() must be called on
    the lexer and the parser SEPARATELY -- they maintain independent
    listener lists, because they report different classes of error.
    An unterminated string or a stray character outside the lexer's
    alphabet is a lexer-level error (nothing after it even tokenizes
    correctly); a syntactically-valid token sequence that violates the
    grammar's structure (e.g. `edge x ->` with no destination) is a
    parser-level error. Registering this listener on only one of the
    two silently leaves the other one still printing to stderr.
    """

    def __init__(self):
        super().__init__()
        self.diagnostics: list[Diagnostic] = []

    def syntaxError(self, recognizer, offendingSymbol, line, column, msg, e):
        # ANTLR's own `msg` (e.g. "mismatched input '}' expecting
        # ID") is derived directly from the grammar's actual
        # expected-token set at the exact point parsing failed --
        # passed through as-is rather than reworded into something
        # "friendlier", since a hand-written rewording risks silently
        # being wrong (or misleading) in cases the rewriter didn't
        # anticipate, where ANTLR's own message is always accurate to
        # what the parser's state machine actually expected.
        #
        # `column` (0-indexed) isn't threaded into Diagnostic today --
        # Diagnostic.line matches the granularity Section 7's spec
        # examples use ("Error [line 14]: ..."), not line:column. Easy
        # to add later if finer-grained location ever matters.
        self.diagnostics.append(Diagnostic(Severity.ERROR, line, msg))
