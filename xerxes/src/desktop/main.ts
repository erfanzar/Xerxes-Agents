// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Electron entry. Owns the window and the one daemon connection; everything
 * renderer-side crosses the preload bridge (`daemon:call` in, `daemon:event`
 * out). The renderer is sandboxed with no Node access and a self-only CSP —
 * those properties are the design, not configuration.
 */

import { app, BrowserWindow, dialog, ipcMain, nativeImage, Notification, shell } from 'electron'
import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { dirname, isAbsolute, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { desktopMachineCommand } from './main/machines.js'
import { xerxesHome } from '../daemon/paths.js'
import { DaemonRpc } from './main/daemon.js'
import { registerDaemonBridge, setDaemonEventObserver } from './main/ipc.js'
import { dictationPort, transcribeDictation } from './main/voice.js'
import { saveDesktopWorkspace } from './main/workspaceSettings.js'
import {
  openRemote,
  remoteTarget,
  type RemoteConnection,
  type RemoteTarget,
} from './main/remote.js'
import { notificationFor } from './main/notify.js'

const APP_NAME = 'Xerxes Agents'
const here = dirname(fileURLToPath(import.meta.url))
// Prefer the packaged runtime while preserving explicit developer overrides.
const bundledBun = join(here, '..', 'bun')
const bundledRuntime = join(here, '..', 'runtime', 'cli.js')
if (!process.env.XERXES_TUI_BUN_DAEMON && !process.env.XERXES_BUN_DAEMON && existsSync(bundledRuntime)) process.env.XERXES_BUN_DAEMON = bundledRuntime
if (!process.env.XERXES_TUI_BUN && !process.env.XERXES_BUN && existsSync(bundledBun)) process.env.XERXES_BUN = bundledBun

// A source launch runs inside Electron's stock executable, whose fallback
// identity is literally "Electron" unless the application claims its own name
// before ready. Keep native menus, notifications, window titles, and the macOS
// dock label aligned with the product instead of exposing the host runtime.
app.setName(APP_NAME)
process.title = APP_NAME

/**
 * The phoenix mark lives at the checkout root (assets/logo.png). dist/desktop
 * is three levels down, so resolve up from the bundle — and fall back to cwd
 * for the odd direct launch. Absent file = stock Electron icon, not a crash.
 */
function appIcon(): ReturnType<typeof nativeImage.createFromPath> | undefined {
  const candidates = [
    join(here, '..', '..', '..', 'assets', 'logo.png'),
    join(process.cwd(), 'assets', 'logo.png'),
  ]
  for (const candidate of candidates) {
    if (!existsSync(candidate)) continue
    const image = nativeImage.createFromPath(candidate)
    if (!image.isEmpty()) return image
  }
  return undefined
}

let daemon: DaemonRpc | null = null
let remote: RemoteConnection | null = null
let remoteMachine: RemoteTarget | null = null
let remoteAttempt: AbortController | null = null
let remoteError = ''
let selectedWorkspace: string | null = null
let resumeSession: string | null = null

// ── Native notifications + launch at login ──────────────────────────────
// Needs-input (approval, question) and task-finished moments deserve a ping
// only when the user is NOT already looking at the app — the preference and
// the focus gate both live here in the main process, next to the event pipe.

let notificationsEnabled = true

function maybeNotify(type: string, payload: Record<string, unknown>): void {
  if (!notificationsEnabled || !Notification.isSupported()) return
  if (BrowserWindow.getAllWindows().some((window) => !window.isDestroyed() && window.isFocused()))
    return
  const decision = notificationFor({ type, payload })
  if (!decision) return
  const ping = new Notification({ title: decision.title, body: decision.body })
  ping.on('click', () => {
    const window = BrowserWindow.getAllWindows().find((w) => !w.isDestroyed())
    if (!window) return
    if (window.isMinimized()) window.restore()
    window.show()
    window.focus()
  })
  ping.show()
}

// ── Workspace selection ─────────────────────────────────────────────────
// A workspace is a folder: its daemon owns that project's sessions, and the
// sidebar groups every chat under its folder name. The chosen folder is the
// app's only daemon target, persisted across launches.

const workspaceFile = (): string => join(xerxesHome(), 'desktop.json')

function loadWorkspace(): string | null {
  // Sync read keeps startup deterministic; the file is a one-field object.
  try {
    const raw = readFileSync(workspaceFile(), 'utf8')
    const parsed = JSON.parse(raw) as { workspace?: unknown }
    return typeof parsed.workspace === 'string' && parsed.workspace ? parsed.workspace : null
  } catch {
    return null
  }
}

async function pickWorkspace(): Promise<string | null> {
  const result = await dialog.showOpenDialog({
    title: 'Choose a workspace folder',
    defaultPath: homedir(),
    properties: ['openDirectory', 'createDirectory'],
  })
  return result.canceled || !result.filePaths[0] ? null : result.filePaths[0]
}

/** Point the shell at a new workspace daemon and give the renderer a clean boot. */
function useProject(directory: string, sessionId: string | null = null): void {
  saveDesktopWorkspace(workspaceFile(), directory)
  remoteAttempt?.abort()
  remoteAttempt = null
  void remote?.close()
  remote = null
  remoteMachine = null
  remoteError = ''
  selectedWorkspace = directory
  resumeSession = sessionId
  const next = new DaemonRpc({ projectDir: directory })
  const previous = daemon
  daemon = next
  registerDaemonBridge(next)
  previous?.dispose()
  for (const window of BrowserWindow.getAllWindows()) {
    if (!window.isDestroyed()) window.webContents.reload()
  }
}

function createWindow(): BrowserWindow {
  // Windows/Linux window icon (macOS reads the dock tile set at boot).
  const windowIcon = process.platform === 'darwin' ? undefined : appIcon()
  const window = new BrowserWindow({
    title: APP_NAME,
    width: 1560,
    height: 980,
    minWidth: 760,
    minHeight: 560,
    // Native backdrop material is visible through navigation only. Content
    // remains opaque for legibility; the traffic lights stay system-owned.
    titleBarStyle: process.platform === 'darwin' ? 'hiddenInset' : 'default',
    // Center the 14px native buttons in the renderer's 44px top bar.
    trafficLightPosition: { x: 20, y: 15 },
    ...(windowIcon ? { icon: windowIcon } : {}),
    backgroundColor: process.platform === 'darwin' ? '#00000000' : '#202124',
    ...(process.platform === 'darwin' ? { vibrancy: 'under-window' as const, visualEffectState: 'followWindow' as const } : {}),
    show: false,
    webPreferences: {
      contextIsolation: true,
      // Keep daemon progress and completed IPC updates live while the window is unfocused.
      backgroundThrottling: false,
      nodeIntegration: false,
      sandbox: true,
      preload: join(here, 'preload.js'),
    },
  })

  window.once('ready-to-show', () => window.show())
  void window.loadFile(join(here, 'renderer', 'index.html'))

  // Agent output must not navigate the shell away.
  window.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: 'deny' }
  })
  window.webContents.on('will-navigate', (event) => event.preventDefault())

  return window
}

void app.whenReady().then(async () => {
  app.setAboutPanelOptions({ applicationName: APP_NAME })
  // The dock/taskbar icon is the phoenix mark; on macOS the running app's
  // dock tile only changes through app.dock.
  const icon = appIcon()
  if (icon && process.platform === 'darwin') app.dock?.setIcon(icon)
  // Workspace gate: with no saved workspace the shell boots WITHOUT a daemon
  // and the renderer shows the create-workspace screen. A daemon target only
  // exists once the user picks a folder — a silent cwd fallback would open
  // the app on a workspace the user never chose.
  let workspace = loadWorkspace()
  selectedWorkspace = workspace
  registerDaemonBridge()
  if (workspace) {
    daemon = new DaemonRpc({ projectDir: workspace })
    registerDaemonBridge(daemon)
    // Warm the connection without gating first paint on it.
    void daemon.call('runtime.status', {}).catch(() => {})
  }
  // The renderer's workspace gate lands here: pick a folder (or enter one
  // from the sidebar), move the bridge to that project's daemon, reload for
  // a clean session view.
  ipcMain.handle('desktop:choose-workspace', async () => {
    const picked = await pickWorkspace()
    if (!picked) return null
    useProject(picked)
    return picked
  })
  ipcMain.handle('desktop:use-workspace', (_event, dir: unknown, sessionId?: unknown) => {
    if (typeof dir !== 'string' || !dir) throw new TypeError('Invalid workspace directory')
    if (sessionId !== undefined && (typeof sessionId !== 'string' || !sessionId || sessionId.length > 256 || /[\x00-\x1f]/.test(sessionId))) throw new TypeError('Invalid resume session id')
    if (remote && sessionId) throw new Error('Open this session from its saved SSH workspace first.')
    if (dir !== selectedWorkspace || remote || sessionId) useProject(dir, typeof sessionId === 'string' ? sessionId : null)
    return dir
  })
  ipcMain.handle('desktop:workspace', () => selectedWorkspace)
  ipcMain.handle('desktop:resume', () => {
    const selected = resumeSession
    resumeSession = null
    return selected
  })
  let voiceRequest: AbortController | null = null
  ipcMain.handle('desktop:voice', async (_event, action: unknown, value: unknown) => {
    if (action === 'cancel') {
      voiceRequest?.abort()
      return true
    }
    const port = dictationPort(process.env)
    if (action === 'check') return true
    if (action !== 'transcribe') throw new Error('Unknown dictation action')
    if (voiceRequest) throw new Error('Dictation is already transcribing')
    const controller = new AbortController()
    voiceRequest = controller
    const timeout = setTimeout(() => controller.abort(), 90000)
    try {
      return await transcribeDictation(value, port, controller.signal)
    } finally {
      clearTimeout(timeout)
      if (voiceRequest === controller) voiceRequest = null
    }
  })
  ipcMain.handle(
    'desktop:remote',
    async (_event, action: unknown, params: Record<string, unknown>) => {
      if (!params || typeof params !== 'object' || Array.isArray(params))
        throw new Error('Invalid remote parameters')
      const argument = (value: unknown) => {
        if (typeof value !== 'string' || /['\x00-\x1f\x7f]/.test(value))
          throw new Error('Invalid remote argument')
        return "'" + value + "'"
      }
      if (action === 'status')
        return {
          ok: true,
          machine: remoteMachine,
          connected: Boolean(remote && daemon?.online),
          error: remoteError,
        }
      if (action === 'cancel') {
        remoteAttempt?.abort()
        return { ok: true }
      }
      if (action === 'connect') {
        if (remoteAttempt) throw new Error('A connection is already being prepared')
        const machine = remoteTarget(params.machine),
          controller = new AbortController()
        remoteAttempt = controller
        remoteError = ''
        try {
          const next = await openRemote(machine, controller.signal, (error) => {
            remoteError = error.message
          })
          const rpc = new DaemonRpc({ projectDir: next.projectDir, socketPath: next.socketPath })
          try {
            await rpc.call('runtime.status')
            controller.signal.throwIfAborted()
          } catch (error) {
            rpc.dispose()
            await next.close()
            throw error
          }
          resumeSession =
            remoteMachine?.target === machine.target &&
            remoteMachine?.workspacePath === machine.workspacePath &&
            typeof params.resume_session_id === 'string' &&
            /^[a-zA-Z0-9_-]{1,128}$/.test(params.resume_session_id)
              ? params.resume_session_id
              : null
          const previous = remote
          daemon?.dispose()
          daemon = rpc
          registerDaemonBridge(rpc)
          remote = next
          remoteMachine = machine
          selectedWorkspace = next.projectDir
          await previous?.close()
          for (const window of BrowserWindow.getAllWindows())
            if (!window.isDestroyed()) window.webContents.reload()
          return { ok: true }
        } catch (error) {
          remoteError = error instanceof Error ? error.message : String(error)
          throw error
        } finally {
          if (remoteAttempt === controller) remoteAttempt = null
        }
      }
      let command: string
      if (action === 'hosts' || action === 'list') command = action
      else if (action === 'browse')
        command =
          'browse ' +
          argument(params.target) +
          ' ' +
          argument(
            Buffer.from(typeof params.path === 'string' ? params.path : '').toString('base64url'),
          )
      else if (action === 'save') {
        const m = remoteTarget(params.machine)
        command = 'add ' + [m.alias, m.target, m.workspacePath].map(argument).join(' ')
      } else if (action === 'remove') command = 'remove ' + argument(params.alias)
      else throw new Error('Unknown remote action')
      return desktopMachineCommand(join(xerxesHome(), 'machines.json'), command)
    },
  )

  // Native capabilities behind validated, narrow channels (repo law: every
  // preload capability has a bridge contract, a default wrapper, and types).
  setDaemonEventObserver((type, payload) => maybeNotify(type, payload))
  ipcMain.handle('native:notifications:set', (_event, on: unknown) => {
    if (typeof on !== 'boolean') throw new TypeError('notifications expects a boolean')
    notificationsEnabled = on
    return notificationsEnabled
  })
  ipcMain.handle('native:login-item:get', () => app.getLoginItemSettings().openAtLogin)
  ipcMain.handle('native:login-item:set', (_event, on: unknown) => {
    if (typeof on !== 'boolean') throw new TypeError('login item expects a boolean')
    app.setLoginItemSettings({ openAtLogin: on })
    return app.getLoginItemSettings().openAtLogin
  })
  ipcMain.handle('native:preset:open-path', async (_event, value: unknown) => {
    if (typeof value !== 'string' || !value) throw new TypeError('invalid preset path')
    const root = resolve(xerxesHome(), 'agents')
    const candidate = resolve(value)
    const fromRoot = relative(root, candidate)
    if (!fromRoot || fromRoot.startsWith('..') || isAbsolute(fromRoot)) {
      throw new Error('preset path must be a child of the Xerxes user agent directory')
    }
    return (await shell.openPath(candidate)) === ''
  })
  createWindow()

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })
})

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit()
})

app.on('will-quit', () => {
  // Ours drops the socket; a launched daemon keeps serving other surfaces.
  daemon?.dispose()
  remoteAttempt?.abort()
  void remote?.close()
})
