"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.activate = activate;
exports.deactivate = deactivate;
const path = require("path");
const vscode = require("vscode");
const node_1 = require("vscode-languageclient/node");
let client;
function activate(context) {
    const python = vscode.workspace.getConfiguration('nngraph').get('pythonPath', 'python');
    const workspace = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? path.dirname(context.extensionPath);
    const server = path.join(workspace, 'lsp', 'server.py');
    const serverOptions = {
        run: { command: python, args: [server], options: { cwd: workspace } },
        debug: { command: python, args: [server], options: { cwd: workspace } },
    };
    const clientOptions = {
        documentSelector: [{ scheme: 'file', language: 'nngraph' }],
    };
    client = new node_1.LanguageClient('nngraphLanguageServer', 'NNGraph Language Server', serverOptions, clientOptions);
    client.start();
    context.subscriptions.push({ dispose: () => client.stop() });
}
function deactivate() {
    return client?.stop();
}
//# sourceMappingURL=extension.js.map