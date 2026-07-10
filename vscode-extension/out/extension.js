"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const path = require("path");
const vscode = require("vscode");
const node_1 = require("vscode-languageclient/node");
let client;
let output;
function activate(context) {
    output = vscode.window.createOutputChannel('NNGraph Language Server');
    context.subscriptions.push(output);
    const python = vscode.workspace.getConfiguration('nngraph').get('pythonPath', 'python');
    // Always resolve the server relative to the extension's own install location,
    // not the user's workspace folder (which may not contain lsp/server.py at all).
    const server = path.join(context.extensionPath, '..', 'lsp', 'server.py');
    const cwd = context.extensionPath;
    output.appendLine(`[nngraph] extensionPath: ${context.extensionPath}`);
    output.appendLine(`[nngraph] python: ${python}`);
    output.appendLine(`[nngraph] server script: ${server}`);
    output.appendLine(`[nngraph] cwd: ${cwd}`);
    if (!require('fs').existsSync(server)) {
        output.appendLine(`[nngraph] ERROR: server script not found at ${server}`);
        vscode.window.showErrorMessage(`NNGraph: language server script not found at ${server}. See "NNGraph Language Server" output channel.`);
        return;
    }
    const serverOptions = {
        run: { command: python, args: [server], options: { cwd } },
        debug: { command: python, args: [server], options: { cwd } },
    };
    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'nngraph' }],
        outputChannel: output,
    };
    client = new node_1.LanguageClient('nngraphLanguageServer', 'NNGraph Language Server', serverOptions, clientOptions);
    client.start().then(() => output.appendLine('[nngraph] language client started successfully'), (err) => {
        output.appendLine(`[nngraph] ERROR: failed to start language client: ${err}`);
        vscode.window.showErrorMessage(`NNGraph: failed to start language server. See "NNGraph Language Server" output channel for details.`);
    });
    context.subscriptions.push({
        dispose: () => {
            client?.stop().then(() => output.appendLine('[nngraph] language client stopped'), (err) => output.appendLine(`[nngraph] error stopping language client: ${err}`));
        },
    });
}
function deactivate() {
    return client?.stop();
}
//# sourceMappingURL=extension.js.map