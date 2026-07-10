# NNGraph LSP

The server is `lsp/server.py`. It communicates over standard input/output using the Language Server Protocol and reuses `compiler.py` for diagnostics.

Run it directly with the Windows Python executable that has `antlr4-python3-runtime` installed:

```powershell
python lsp\server.py
```

The VS Code extension in `vscode-extension/` launches it automatically.
