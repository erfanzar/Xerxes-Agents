// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Xerxes Agents for VS Code. The view is a chat built from the desktop
 * renderer's conversation pieces (ui/VscodeApp.tsx), over the same bridge
 * (desktop/bridgeApi.ts) and the same runtime connection
 * (desktop/main/daemon.ts), working in the window's folder. Files, diffs,
 * Git and pickers use VS Code's own. The Bun runtime ships inside the
 * extension, so nothing else needs installing.
 */

import { existsSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { isAbsolute, join, relative } from 'node:path'

import * as vscode from 'vscode'

import { DaemonRpc } from '../desktop/main/daemon.js'
import type { PushFrame, ResultFrame } from './messages.js'
import { otherCopies } from './otherCopies.js'
import { PanelHost, type PanelOptions, type PanelPorts } from './panelHost.js'

const VIEW = 'xerxes.chat'
const PANEL = 'xerxes.panel'
let context: vscode.ExtensionContext
const hosts = new Set<{ host: PanelHost; webview: vscode.Webview }>()
/** Overflow-menu surfaces; keep in step with SHEETS in scripts/buildVscode.ts. */
const SHEETS = ['activity', 'usage'] as const

export function activate(extension: vscode.ExtensionContext): void {
  context = extension
  useBundledRuntime(extension.extensionPath)
  void offerToRemoveOtherCopies(extension.extension.id)
  extension.subscriptions.push(
    vscode.window.registerWebviewViewProvider(VIEW, { resolveWebviewView: view => mount(view.webview, () => view.visible, view.onDidDispose) }, { webviewOptions: { retainContextWhenHidden: true } }),
    vscode.commands.registerCommand('xerxes.newTask', () => menu('new-task')),
    vscode.commands.registerCommand('xerxes.history', () => menu('history')),
    ...SHEETS.map(name => vscode.commands.registerCommand(`xerxes.panel.${name}`, () => menu(`panel:${name}`))),
    vscode.commands.registerCommand('xerxes.settings', () => menu('settings')),
    vscode.commands.registerCommand('xerxes.search', () => menu('search')),
    vscode.commands.registerCommand('xerxes.palette', () => menu('palette')),
    vscode.commands.registerCommand('xerxes.exportTranscript', () => menu('export')),
    vscode.commands.registerCommand('xerxes.sendSelection', sendSelection),
  )
}

export function deactivate(): void {
  for (const entry of hosts) entry.host.dispose()
  hosts.clear()
}

/**
 * Another installed copy of this extension (an earlier publisher ID, a
 * side-loaded .vsix) contributes the same view, so every title-bar button
 * shows twice. Say which one it is and offer to remove it.
 */
async function offerToRemoveOtherCopies(self: string): Promise<void> {
  for (const id of otherCopies(self, VIEW, vscode.extensions.all)) {
    const choice = await vscode.window.showWarningMessage(`Another copy of Xerxes Agents (${id}) is installed, so its buttons appear twice.`, 'Uninstall it')
    if (choice !== 'Uninstall it') continue
    await vscode.commands.executeCommand('workbench.extensions.uninstallExtension', id)
    await vscode.commands.executeCommand('workbench.action.reloadWindow')
    return
  }
}

/** Point the runtime launcher at the Bun and CLI inside the extension, unless the person set their own. */
function useBundledRuntime(root: string): void {
  const bun = join(root, 'runtime', process.platform === 'win32' ? 'bun.exe' : 'bun')
  const cli = join(root, 'runtime', 'cli.js')
  if (!process.env.XERXES_BUN && !process.env.XERXES_TUI_BUN && existsSync(bun)) process.env.XERXES_BUN = bun
  if (!process.env.XERXES_BUN_DAEMON && !process.env.XERXES_TUI_BUN_DAEMON && existsSync(cli)) process.env.XERXES_BUN_DAEMON = cli
}

function openPanel(options: PanelOptions): void {
  const panel = vscode.window.createWebviewPanel(PANEL, 'Xerxes', vscode.ViewColumn.Beside, { enableScripts: true, retainContextWhenHidden: true, localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')] })
  panel.iconPath = vscode.Uri.joinPath(context.extensionUri, 'media', 'icon.png')
  mount(panel.webview, () => panel.visible, panel.onDidDispose, options, 'panel')
}

function mount(webview: vscode.Webview, visible: () => boolean, onDispose: vscode.Event<void>, options: PanelOptions = {}, surface: 'view' | 'panel' = 'view'): void {
  webview.options = { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')] }
  const folder = vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? null
  const ports: PanelPorts = {
    workspace: folder,
    workspaceFolders: () => (vscode.workspace.workspaceFolders ?? []).map(entry => entry.uri.fsPath),
    remoteName: vscode.env.remoteName,
    async chooseFolder() {
      const picked = await vscode.window.showOpenDialog({ canSelectFolders: true, canSelectFiles: false, canSelectMany: false, openLabel: 'Open with Xerxes' })
      return picked?.[0]?.fsPath ?? null
    },
    async openFolder(dir) { await vscode.commands.executeCommand('vscode.openFolder', vscode.Uri.file(dir), { forceNewWindow: true }) },
    openPanel,
    async revealPath(path) {
      // Only paths inside this workspace or ~/.xerxes, which the runtime resolves.
      const roots = [folder, process.env.XERXES_HOME, join(process.env.HOME ?? '', '.xerxes')].filter((root): root is string => Boolean(root))
      if (!roots.some(root => { const rel = relative(root, path); return rel === '' || (!rel.startsWith('..') && !rel.startsWith('/')) })) return false
      await vscode.commands.executeCommand('revealInExplorer', vscode.Uri.file(path)).then(undefined, () => vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(path)))
      return true
    },
    async backgroundData(name) {
      const bytes = await readFile(join(context.extensionPath, 'media', 'renderer', name))
      return `data:image/jpeg;base64,${bytes.toString('base64')}`
    },
    memory: {
      get: key => context.workspaceState.get<string>(key),
      set: (key, value) => { void context.workspaceState.update(key, value) },
    },
    notify: message => { void vscode.window.showInformationMessage(message, 'Show').then(choice => { if (choice) void vscode.commands.executeCommand('xerxes.chat.focus') }) },
    visible,
    async openFile(path, line) {
      const uri = vscode.Uri.file(isAbsolute(path) || !folder ? path : join(folder, path))
      const at = line ? new vscode.Position(line - 1, 0) : undefined
      await vscode.window.showTextDocument(uri, { preview: true, ...(at ? { selection: new vscode.Range(at, at) } : {}) })
      return true
    },
    async openDiff(path) {
      const uri = vscode.Uri.file(isAbsolute(path) || !folder ? path : join(folder, path))
      // VS Code's own working-tree diff, as in Source Control; a file outside git just opens.
      try { await vscode.commands.executeCommand('git.openChange', uri) }
      catch { await vscode.window.showTextDocument(uri, { preview: true }) }
      return true
    },
    async pickFiles() {
      const files = await vscode.workspace.findFiles('**/*', undefined, 20_000)
      const items = files.map(uri => ({ label: vscode.workspace.asRelativePath(uri), uri }))
        .sort((a, b) => a.label.localeCompare(b.label))
      const picked = await vscode.window.showQuickPick(items, { canPickMany: true, matchOnDescription: true, placeHolder: 'Add files as context for Xerxes' })
      return (picked ?? []).map(item => item.label)
    },
    async showSourceControl() { await vscode.commands.executeCommand('workbench.view.scm'); return true },
  }
  const post = (frame: ResultFrame | PushFrame): void => { void webview.postMessage(frame) }
  const host = new PanelHost(ports, post, projectDir => new DaemonRpc({ projectDir }), options)
  const entry = { host, webview }
  hosts.add(entry)
  const received = webview.onDidReceiveMessage(message => { void host.receive(message) })
  onDispose(() => { received.dispose(); host.dispose(); hosts.delete(entry) })
  webview.html = html(webview, surface)
}

/** The chat view's page, with VS Code's content-security policy. An editor tab sits on the editor background, the view on the sidebar's. */
function html(webview: vscode.Webview, surface: 'view' | 'panel'): string {
  const media = vscode.Uri.joinPath(context.extensionUri, 'media')
  const asset = (path: string) => webview.asWebviewUri(vscode.Uri.joinPath(media, path)).toString()
  const base = asset('renderer/')
  const source = webview.cspSource
  return `<!doctype html>
<html lang="en" data-layout="claude" data-host="vscode" data-surface="${surface}">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src ${source}; style-src ${source} 'unsafe-inline'; font-src ${source}; img-src ${source} data:;" />
    <base href="${base}" />
    <title>Xerxes</title>
    <link rel="stylesheet" href="${asset('renderer/theme.css')}" />
    <link rel="stylesheet" href="${asset('renderer/xterm.css')}" />
    <link rel="stylesheet" href="${asset('renderer/app.css')}" />
    <link rel="stylesheet" href="${asset('renderer/atelier.css')}" />
    <link rel="stylesheet" href="${asset('vscode.css')}" />
    <!-- VS Code pads every webview body (0 20px); the view is edge to edge. -->
    <style>html, body { margin: 0; padding: 0; overflow: hidden; }</style>
  </head>
  <body>
    <div id="root"></div>
    <script src="${asset('webview.js')}"></script>
    <script type="module" src="${asset('chat.js')}"></script>
  </body>
</html>`
}

/** Push a menu command to every open Xerxes panel (the desktop's menu bar equivalent). */
function menu(command: string): void {
  if (!hosts.size) { void vscode.commands.executeCommand(`${VIEW}.focus`).then(() => setTimeout(() => menu(command), 500)); return }
  for (const entry of hosts) void entry.webview.postMessage({ kind: 'push', channel: 'desktop:menu', value: command } satisfies PushFrame)
}

/** The editor selection, with its file and lines, into the Xerxes composer. */
async function sendSelection(): Promise<void> {
  const editor = vscode.window.activeTextEditor
  if (!editor) return
  const selection = editor.selection
  const code = editor.document.getText(selection.isEmpty ? undefined : selection)
  const file = vscode.workspace.asRelativePath(editor.document.uri)
  const lines = selection.isEmpty ? '' : `:${selection.start.line + 1}-${selection.end.line + 1}`
  const text = `\n${file}${lines}\n\`\`\`${editor.document.languageId}\n${code}\n\`\`\`\n`
  await vscode.commands.executeCommand(`${VIEW}.focus`)
  const deliver = () => { for (const entry of hosts) void entry.webview.postMessage({ kind: 'push', channel: 'desktop:compose', value: text } satisfies PushFrame) }
  if (hosts.size) deliver(); else setTimeout(deliver, 800)
}
