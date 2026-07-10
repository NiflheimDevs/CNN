import * as path from 'path';
import * as vscode from 'vscode';
import { LanguageClient, LanguageClientOptions, ServerOptions } from 'vscode-languageclient/node';

let client: LanguageClient;
let output: vscode.OutputChannel;

export function activate(context: vscode.ExtensionContext) {
  output = vscode.window.createOutputChannel('NNGraph Language Server');
  context.subscriptions.push(output);

  const python = vscode.workspace.getConfiguration('nngraph').get<string>('pythonPath', 'python');

  // Always resolve the server relative to the extension's own install location,
  // not the user's workspace folder (which may not contain lsp/server.py at all).
  const server = path.join(context.extensionPath, '..' ,'lsp', 'server.py');
  const cwd = context.extensionPath;

  output.appendLine(`[nngraph] extensionPath: ${context.extensionPath}`);
  output.appendLine(`[nngraph] python: ${python}`);
  output.appendLine(`[nngraph] server script: ${server}`);
  output.appendLine(`[nngraph] cwd: ${cwd}`);

  if (!require('fs').existsSync(server)) {
    output.appendLine(`[nngraph] ERROR: server script not found at ${server}`);
    vscode.window.showErrorMessage(
      `NNGraph: language server script not found at ${server}. See "NNGraph Language Server" output channel.`
    );
    return;
  }

  const serverOptions: ServerOptions = {
    run: { command: python, args: [server], options: { cwd } },
    debug: { command: python, args: [server], options: { cwd } },
  };
  const clientOptions: LanguageClientOptions = {
    documentSelector: [{ scheme: 'file', language: 'nngraph' }],
    outputChannel: output,
  };

  client = new LanguageClient('nngraphLanguageServer', 'NNGraph Language Server', serverOptions, clientOptions);

  client.start().then(
    () => output.appendLine('[nngraph] language client started successfully'),
    (err: unknown) => {
      output.appendLine(`[nngraph] ERROR: failed to start language client: ${err}`);
      vscode.window.showErrorMessage(
        `NNGraph: failed to start language server. See "NNGraph Language Server" output channel for details.`
      );
    }
  );

  context.subscriptions.push({
    dispose: () => {
      client?.stop().then(
        () => output.appendLine('[nngraph] language client stopped'),
        (err: unknown) => output.appendLine(`[nngraph] error stopping language client: ${err}`)
      );
    },
  });
}

export function deactivate(): Thenable<void> | undefined {
  return client?.stop();
}
