#!/usr/bin/env python3
"""Minimal stdio LSP server for NNGraph diagnostics."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "compiler.py"


def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.decode("ascii").strip()
        if not line:
            break
        key, value = line.split(":", 1)
        headers[key.lower()] = value.strip()
    length = int(headers["content-length"])
    body = sys.stdin.buffer.read(length)
    return json.loads(body.decode("utf-8"))


def send_message(payload):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def response(request_id, result):
    send_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def notification(method, params):
    send_message({"jsonrpc": "2.0", "method": method, "params": params})


def parse_diagnostics(stderr, text):
    diagnostics = []
    lines = text.splitlines() or [""]
    for raw in stderr.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        severity = 1 if raw.startswith("Error") else 2 if raw.startswith("Warning") else None
        if severity is None:
            continue
        line = 0
        message = raw
        marker = " [line "
        if marker in raw:
            prefix, suffix = raw.split(marker, 1)
            number, _, rest = suffix.partition("]: ")
            try:
                line = max(0, int(number) - 1)
                message = rest or prefix
            except ValueError:
                pass
        line = min(line, max(0, len(lines) - 1))
        end_character = max(1, len(lines[line]))
        diagnostics.append({
            "range": {"start": {"line": line, "character": 0},
                      "end": {"line": line, "character": end_character}},
            "severity": severity,
            "source": "NNGraph",
            "message": message,
        })
    return diagnostics


def publish(uri, text):
    with tempfile.TemporaryDirectory(prefix="nngraph-lsp-") as directory:
        source = Path(directory) / "document.nng"
        source.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(COMPILER), str(source), "--output", str(Path(directory) / "generated.py")],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    notification("textDocument/publishDiagnostics", {
        "uri": uri,
        "diagnostics": parse_diagnostics(result.stderr, text),
    })


def main():
    documents = {}
    while True:
        request = read_message()
        if request is None:
            return
        method = request.get("method")
        params = request.get("params", {})
        if method == "initialize":
            response(request.get("id"), {"capabilities": {
                "textDocumentSync": {"openClose": True, "change": 1},
            }})
        elif method == "shutdown":
            response(request.get("id"), None)
        elif method == "exit":
            return
        elif method == "textDocument/didOpen":
            document = params["textDocument"]
            documents[document["uri"]] = document["text"]
            publish(document["uri"], document["text"])
        elif method == "textDocument/didChange":
            uri = params["textDocument"]["uri"]
            text = params["contentChanges"][-1]["text"]
            documents[uri] = text
            publish(uri, text)
        elif method == "textDocument/didClose":
            uri = params["textDocument"]["uri"]
            documents.pop(uri, None)
            notification("textDocument/publishDiagnostics", {"uri": uri, "diagnostics": []})


if __name__ == "__main__":
    main()
