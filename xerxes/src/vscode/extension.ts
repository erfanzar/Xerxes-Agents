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

import { existsSync, statSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { isAbsolute, join } from 'node:path'

import * as vscode from 'vscode'

import { DaemonRpc } from '../desktop/main/daemon.js'
import { pickTargetHost, type TargetHostCandidate } from './hostTarget.js'
import type { PushFrame, ResultFrame } from './messages.js'
import { otherCopies } from './otherCopies.js'
import { PanelHost, isAllowedRevealPath, type PanelOptions, type PanelPorts } from './panelHost.js'

const VIEW = 'xerxes.chat'
const PANEL = 'xerxes.panel'
const MENU_FOCUS_MAX_RETRIES = 6
let context: vscode.ExtensionContext
let activeSeq = 0
interface MountedHost extends TargetHostCandidate {
  readonly host: PanelHost
  readonly webview: vscode.Webview
  lastActiveAt: number
}
const hosts = new Set<MountedHost>()
const pendingTimers = new Set<ReturnType<typeof setTimeout>>()
/** Overflow-menu surfaces; keep in step with SHEETS in scripts/buildVscode.ts. */
const SHEETS = ['activity', 'usage'] as const

export function activate(extension: vscode.ExtensionContext): void {
  context = extension
  useBundledRuntime(extension.extensionPath)
  void offerToRemoveOtherCopies(extension.extension.id)
  extension.subscriptions.push(
    vscode.window.registerWebviewViewProvider(
      VIEW,
      {
        resolveWebviewView: view =>
          mount(
            view.webview,
            () => view.visible,
            () => view.visible,
            view.onDidDispose,
            {},
            'view',
            onActivate => view.onDidChangeVisibility(() => { if (view.visible) onActivate() }),
          ),
      },
      { webviewOptions: { retainContextWhenHidden: true } },
    ),
    vscode.commands.registerCommand('xerxes.newTask', () => menu('new-task')),
    vscode.commands.registerCommand('xerxes.history', () => menu('history')),
    ...SHEETS.map(name => vscode.commands.registerCommand(`xerxes.panel.${name}`, () => menu(`panel:${name}`))),
    vscode.commands.registerCommand('xerxes.settings', () => menu('settings')),
    vscode.commands.registerCommand('xerxes.search', () => menu('search')),
    vscode.commands.registerCommand('xerxes.palette', () => menu('palette')),
    vscode.commands.registerCommand('xerxes.exportTranscript', () => menu('export')),
    vscode.commands.registerCommand('xerxes.sendSelection', sendSelection),
    // Switches between Vibes and Neo.
    vscode.commands.registerCommand('xerxes.toggleStyle', () => setStyle(nextStyle(currentStyle()))),
    // Settings UI, settings.json, the palette or another window: every open view follows.
    vscode.workspace.onDidChangeConfiguration(event => {
      if (!event.affectsConfiguration('xerxes.style')) return
      const style = currentStyle()
      for (const entry of hosts) void entry.webview.postMessage({ kind: 'push', channel: 'desktop:style', value: style } satisfies PushFrame)
    }),
  )
}

type VscodeStyle = 'vibes' | 'neo'

/** Xerxes: Style. Vibes unless the person chose Neo; a saved Classic or Shading (retired) opens as Vibes. */
function currentStyle(): VscodeStyle {
  const value = vscode.workspace.getConfiguration('xerxes').get<string>('style')
  return value === 'neo' ? 'neo' : 'vibes'
}

function nextStyle(style: VscodeStyle): VscodeStyle {
  return style === 'neo' ? 'vibes' : 'neo'
}

/** Save the style in the user's settings; the configuration listener repaints every view. */
async function setStyle(style: VscodeStyle): Promise<boolean> {
  await vscode.workspace.getConfiguration('xerxes').update('style', style, vscode.ConfigurationTarget.Global)
  return true
}

export function deactivate(): void {
  for (const timer of pendingTimers) clearTimeout(timer)
  pendingTimers.clear()
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
  mount(
    panel.webview,
    () => panel.visible,
    () => panel.active,
    panel.onDidDispose,
    options,
    'panel',
    onActivate => panel.onDidChangeViewState(() => { if (panel.active || panel.visible) onActivate() }),
  )
}

function currentFolder(): string | null {
  return vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ?? null
}

function scheduleExtensionTimer(callback: () => void, delayMs: number): void {
  const timer = setTimeout(() => {
    pendingTimers.delete(timer)
    callback()
  }, delayMs)
  pendingTimers.add(timer)
}

function mount(
  webview: vscode.Webview,
  visible: () => boolean,
  active: () => boolean,
  onDispose: vscode.Event<void>,
  options: PanelOptions = {},
  surface: 'view' | 'panel' = 'view',
  subscribeActivate?: (onActivate: () => void) => vscode.Disposable,
): void {
  webview.options = { enableScripts: true, localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'media')] }
  const ports: PanelPorts = {
    get workspace() { return currentFolder() },
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
      const home = homedir()
      const roots = [currentFolder(), process.env.XERXES_HOME, home ? join(home, '.xerxes') : undefined]
      if (!isAllowedRevealPath(path, roots)) return false
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
      const folder = currentFolder()
      const uri = vscode.Uri.file(isAbsolute(path) || !folder ? path : join(folder, path))
      const at = typeof line === 'number' && Number.isSafeInteger(line) && line >= 1 ? new vscode.Position(line - 1, 0) : undefined
      await vscode.window.showTextDocument(uri, { preview: true, ...(at ? { selection: new vscode.Range(at, at) } : {}) })
      return true
    },
    async openDiff(path) {
      const folder = currentFolder()
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
    // PanelHost has already limited the link to http, https and mailto.
    async openExternal(url) { return vscode.env.openExternal(vscode.Uri.parse(url, true)) },
    // PanelHost has already resolved the path inside the task folder and refused launchable files.
    async openWithSystem(path) { return await vscode.env.openExternal(vscode.Uri.file(path)) ? '' : 'the system did not open it' },
    async revealInOS(path) { await vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(path)) },
    setStyle,
  }
  const post = (frame: ResultFrame | PushFrame): void => { void webview.postMessage(frame) }
  const host = new PanelHost(ports, post, projectDir => new DaemonRpc({ projectDir }), options)
  const entry: MountedHost = {
    host,
    webview,
    surface,
    isVisible: visible,
    isActive: active,
    lastActiveAt: ++activeSeq,
  }
  hosts.add(entry)
  const markActive = (): void => { entry.lastActiveAt = ++activeSeq }
  const activationSub = subscribeActivate?.(markActive)
  const received = webview.onDidReceiveMessage(message => { markActive(); void host.receive(message) })
  onDispose(() => { activationSub?.dispose(); received.dispose(); host.dispose(); hosts.delete(entry) })
  webview.html = html(webview, surface)
}

/** The chat view's page, with VS Code's content-security policy. An editor tab sits on the editor background, the view on the sidebar's. */
function html(webview: vscode.Webview, surface: 'view' | 'panel'): string {
  const media = vscode.Uri.joinPath(context.extensionUri, 'media')
  // Stamped with the file's build time: VS Code's webview keeps a resource
  // cache across restarts keyed by URL, so a rebuilt file at the same path
  // could be served stale.
  const asset = (path: string) => {
    const file = vscode.Uri.joinPath(media, path)
    const stamp = existsSync(file.fsPath) ? Math.round(statSync(file.fsPath).mtimeMs).toString(36) : ''
    return webview.asWebviewUri(file).toString() + (stamp ? `?v=${stamp}` : '')
  }
  const base = webview.asWebviewUri(vscode.Uri.joinPath(media, 'renderer/')).toString()
  const source = webview.cspSource
  // Stamped before first paint so the view never flashes an unstyled page.
  const style = currentStyle()
  return `<!doctype html>
<html lang="en" data-layout="claude" data-host="vscode" data-surface="${surface}" data-style-choice="${style}" data-style="${style}">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src ${source} data:; style-src ${source} 'unsafe-inline'; font-src ${source}; img-src ${source} data:; frame-src blob:;" />
    <base href="${base}" />
    <title>Xerxes</title>
    <link rel="stylesheet" href="${asset('renderer/theme.css')}" />
    <link rel="stylesheet" href="${asset('renderer/xterm.css')}" />
    <link rel="stylesheet" href="${asset('renderer/app.css')}" />
    <link rel="stylesheet" href="${asset('renderer/atelier.css')}" />
    <!-- The one layout every style shares. -->
    <link rel="stylesheet" href="${asset('renderer/layout.css')}" />
    <!-- Claude Code's layout, shared by Vibes and Neo; Vibes colours and type. -->
    <link rel="stylesheet" href="${asset('renderer/vibes.css')}" />
    <!-- The Neo look over the same layout; inert unless data-style="neo". -->
    <link rel="stylesheet" href="${asset('renderer/neo.css')}" />
    <!-- Loaded LAST so VS Code theme tokens, High Contrast, flat sheets, and 1-box composer win. -->
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

/** Push a menu command to the active/visible Xerxes host (never broadcast to every open panel). */
function menu(command: string, attempt = 0): void {
  const target = pickTargetHost(hosts)
  if (!target) {
    if (attempt >= MENU_FOCUS_MAX_RETRIES) return
    void vscode.commands.executeCommand(`${VIEW}.focus`).then(
      () => scheduleExtensionTimer(() => menu(command, attempt + 1), 500),
      () => undefined,
    )
    return
  }
  void target.webview.postMessage({ kind: 'push', channel: 'desktop:menu', value: command } satisfies PushFrame)
}

/** The editor selection, with its file and lines, into the active/visible Xerxes composer. */
async function sendSelection(): Promise<void> {
  const editor = vscode.window.activeTextEditor
  if (!editor) return
  const selection = editor.selection
  const code = editor.document.getText(selection.isEmpty ? undefined : selection)
  const file = vscode.workspace.asRelativePath(editor.document.uri)
  const lines = selection.isEmpty ? '' : `:${selection.start.line + 1}-${selection.end.line + 1}`
  const text = `\n${file}${lines}\n\`\`\`${editor.document.languageId}\n${code}\n\`\`\`\n`
  const hadVisible = [...hosts].some(entry => entry.isVisible())
  if (!hadVisible) await vscode.commands.executeCommand(`${VIEW}.focus`)
  const deliver = () => {
    const target = pickTargetHost(hosts)
    if (target) void target.webview.postMessage({ kind: 'push', channel: 'desktop:compose', value: text } satisfies PushFrame)
  }
  if (hosts.size) deliver(); else scheduleExtensionTimer(deliver, 800)
}
