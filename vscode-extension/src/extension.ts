import * as path from 'path';
import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient/node';

let client: LanguageClient;

export function activate(context: vscode.ExtensionContext) {
  const python = vscode.workspace.getConfiguration('nngraph').get<string>('pythonPath', 'python');
  const workspace = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? path.dirname(context.extensionPath);
  const server = path.join(workspace, 'lsp', 'server.py');
  const serverOptions: ServerOptions = {
    run: { command: python, args: [server], options: { cwd: workspace } },
    debug: { command: python, args: [server], options: { cwd: workspace } },
  };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'nngraph' }],
  };
  client = new LanguageClient('nngraphLanguageServer', 'NNGraph Language Server', serverOptions, clientOptions);
  client.start();
  context.subscriptions.push({ dispose: () => client.stop() });
}

export function deactivate(): Thenable<void> | undefined {
  return client?.stop();
}
