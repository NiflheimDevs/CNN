#!/usr/bin/env python3
"""Minimal stdio LSP server for NNGraph diagnostics."""
import json
import logging
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER = ROOT / "compiler.py"
LOG_FILE = ROOT / "lsp" / "server.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    filemode="a",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("nngraph-lsp")


def read_message():
    try:
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
    except Exception:
        log.error("failed to read LSP message:\n%s", traceback.format_exc())
        return None


def send_message(payload):
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii"))
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.flush()


def response(request_id, result):
    send_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def notification(method, params):
    send_message({"jsonrpc": "2.0", "method": method, "params": params})


def parse_diagnostics(output_text, text):
    diagnostics = []
    lines = text.splitlines() or [""]
    for raw in output_text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        lowered = raw.lower()
        if lowered.startswith("error"):
            severity = 1
        elif lowered.startswith("warning"):
            severity = 2
        else:
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
    diagnostics = []
    try:
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
        log.debug(
            "compiler run: returncode=%s stdout=%r stderr=%r",
            result.returncode, result.stdout, result.stderr,
        )

        # Check both streams -- some tools print diagnostics to stdout.
        diagnostics = parse_diagnostics(result.stderr, text) + parse_diagnostics(result.stdout, text)

        # If the compiler failed but we couldn't extract any structured
        # diagnostic from either stream, still surface *something* rather
        # than silently doing nothing.
        if result.returncode != 0 and not diagnostics:
            fallback_message = (result.stderr or result.stdout or
                                 f"compiler exited with code {result.returncode}").strip()
            diagnostics = [{
                "range": {"start": {"line": 0, "character": 0},
                          "end": {"line": 0, "character": 1}},
                "severity": 1,
                "source": "NNGraph",
                "message": fallback_message,
            }]
    except Exception:
        log.error("failed to run compiler for %s:\n%s", uri, traceback.format_exc())
        diagnostics = [{
            "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 1}},
            "severity": 1,
            "source": "NNGraph",
            "message": f"NNGraph language server error: {traceback.format_exc(limit=1)}",
        }]

    notification("textDocument/publishDiagnostics", {"uri": uri, "diagnostics": diagnostics})


def main():
    log.info("server starting, COMPILER=%s", COMPILER)
    documents = {}
    while True:
        try:
            request = read_message()
            if request is None:
                log.info("stdin closed or unreadable message, exiting")
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
        except Exception:
            log.error("unhandled error in main loop:\n%s", traceback.format_exc())
            # keep the server alive rather than crashing on a single bad message


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log.error("server crashed:\n%s", traceback.format_exc())
        raise
