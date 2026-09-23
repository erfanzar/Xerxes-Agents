// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Electron entry. Owns independent workspace windows and their daemon connections; everything
 * renderer-side crosses the preload bridge (`daemon:call` in, `daemon:event`
 * out). The renderer is sandboxed with no Node access and a self-only CSP —
 * those properties are the design, not configuration.
 */

import { app, BrowserWindow, WebContentsView, dialog, ipcMain, nativeImage, Notification, shell, Menu, screen, type IpcMainInvokeEvent } from 'electron'
import { existsSync, readFileSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { activateWorkspaceView } from './main/workspaceNavigation.js'
import { contextScope, contextSessions, sameRemote, type WorkspaceContext } from './main/contextNavigation.js'
import { loadWindowLayout, saveWindowLayout, visibleWindowBounds, type SavedWindow } from './main/windowState.js'
import { WindowRoutes } from './main/windowRoutes.js'
import { windowRecovery } from './main/windowRecovery.js'
import { desktopMachineCommand } from './main/machines.js'
import { xerxesHome } from '../daemon/paths.js'
import { DaemonRpc } from './main/daemon.js'
import { DesktopProviderForwarding } from './main/providerForwarding.js'
import { registerDaemonBridge, detachDaemon } from './main/ipc.js'
import { dictationPort, transcribeDictation } from './main/voice.js'
import { loadDesktopWorkspaces, saveDesktopWorkspace } from './main/workspaceSettings.js'
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

// Electron scopes this lock to userData, so isolated profiles remain independent.
// Exit before registering persistence or runtime handlers: a duplicate must not
// overwrite the first process's window layout or attach another daemon client.
if (!app.requestSingleInstanceLock()) app.exit(0)
app.on('second-instance', () => {
  void app.whenReady().then(() => {
    const window = BrowserWindow.getFocusedWindow()
      ?? BrowserWindow.getAllWindows().find(candidate => !candidate.isDestroyed())
      ?? createWorkspaceWindow(loadWorkspace())
    if (window.isMinimized()) window.restore()
    window.show()
    window.focus()
  })
})

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

// ── Native notifications + launch at login ──────────────────────────────
// Needs-input (approval, question) and task-finished moments deserve a ping
// only when the user is NOT already looking at the app — the preference and
// the focus gate both live here in the main process, next to the event pipe.

let notificationsEnabled = true

function maybeNotify(window: BrowserWindow, type: string, payload: Record<string, unknown>): void {
  if (!notificationsEnabled || !Notification.isSupported()) return
  if (window.isDestroyed() || window.isFocused())
    return
  const decision = notificationFor({ type, payload })
  if (!decision) return
  const ping = new Notification({ title: decision.title, body: decision.body })
  ping.on('click', () => {
    if (window.isDestroyed()) return
    if (window.isMinimized()) window.restore()
    window.show()
    window.focus()
  })
  ping.show()
}

// ── Workspace selection ─────────────────────────────────────────────────
// A workspace is a folder: the shared daemon isolates its sessions, and the
// sidebar groups every chat under its folder name. The chosen folder is the
// window's daemon target, persisted across launches.

const workspaceFile = (): string => join(xerxesHome(), 'desktop.json')

function loadWorkspace(): string | null {
  // Legacy fallback for installations without a saved window layout.
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

function createWindow(saved?: SavedWindow): BrowserWindow {
  // Windows/Linux window icon (macOS reads the dock tile set at boot).
  const windowIcon = process.platform === 'darwin' ? undefined : appIcon()
  const window = new BrowserWindow({
    title: APP_NAME,
    width: 1560,
    height: 980,
    ...(saved ? visibleWindowBounds(saved.bounds, screen.getAllDisplays().map(display => display.workArea)) : {}),
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

  window.once('ready-to-show', () => {
    if (saved?.maximized) window.maximize()
    if (saved?.fullscreen) window.setFullScreen(true)
    window.show()
  })
  void window.loadFile(join(here, 'renderer', 'index.html')).catch(error => console.error('Desktop page load failed:', error))

  // Agent output must not navigate the shell away.
  window.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url)
    return { action: 'deny' }
  })
  window.on('page-title-updated', event => event.preventDefault())
  window.webContents.on('will-navigate', (event) => event.preventDefault())

  return window
}

const windowRoutes = new WindowRoutes<IpcMainInvokeEvent>()
const registeredChannels = new Set<string>()
const cleanupWindows = new Set<() => void>()

const workspaceSurfaces = new Map<number, { host: BrowserWindow; view: WebContentsView | null }>()
const activeSurfaces = new Map<number, number>()
const contextReaders = new Map<number, () => Promise<WorkspaceContext>>()
function activateSurface(id: number): void {
  const surface = workspaceSurfaces.get(id)
  if (!surface || surface.host.isDestroyed()) return
  for (const [otherId, other] of workspaceSurfaces) {
    if (other.host === surface.host && other.view) other.view.setVisible(otherId === id)
  }
  // An already attached view keeps its old stacking position. Activation must
  // bring it above the other retained workspaces as well as make it visible.
  if (surface.view) surface.host.contentView.addChildView(surface.view)
  activeSurfaces.set(surface.host.id, id)
  const contents = surface.view?.webContents ?? surface.host.webContents
  contents.focus()
  const state = windowStates.get(id)?.()
  surface.host.setTitle(state?.remote
    ? `${state.remote.alias} · ${basename(state.remote.workspacePath)} — ${APP_NAME}`
    : state?.workspace ? `${basename(state.workspace)} — ${APP_NAME}` : APP_NAME)
  scheduleWindowSave()
}

const windowStates = new Map<number, () => SavedWindow>()
let quitting = false
let saveTimer: ReturnType<typeof setTimeout> | undefined
const layoutFile = () => join(xerxesHome(), 'desktop-windows.json')
function persistWindows(): void {
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = undefined
  try { saveWindowLayout(layoutFile(), [...windowStates.values()].map(read => read())) }
  catch (error) { console.error('Could not save desktop windows:', error) }
}
function scheduleWindowSave(): void {
  if (quitting) return
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(persistWindows, 250)
}

function openWorkspaceView(host: BrowserWindow, directory: string, sessionId?: string): void {
  const views = [...workspaceSurfaces].flatMap(([id, surface]) => {
    const state = windowStates.get(id)?.()
    return state && surface.host === host ? [{ ...state, isDestroyed: () => host.isDestroyed(),
      isMinimized: () => host.isMinimized(), restore: () => host.restore(),
      show: () => host.show(), focus: () => { host.focus(); activateSurface(id) } }] : []
  })
  if (!activateWorkspaceView(views, directory, sessionId)) createWorkspaceWindow(directory, undefined, sessionId, host)
}

function openRemoteView(host: BrowserWindow, machine: RemoteTarget, sessionId?: string): void {
  for (const [id, surface] of workspaceSurfaces) {
    const state = windowStates.get(id)?.()
    if (surface.host === host && sameRemote(state?.remote, machine) && (!sessionId || state?.sessionId === sessionId)) {
      activateSurface(id)
      return
    }
  }
  createWorkspaceWindow(null, { workspace: null, remote: machine, sessionId: sessionId ?? null,
    bounds: host.getNormalBounds(), maximized: host.isMaximized(), fullscreen: host.isFullScreen(),
  }, undefined, host)
}

function createWorkspaceWindow(initialWorkspace: string | null = null, saved?: SavedWindow, initialSessionId?: string, host?: BrowserWindow): BrowserWindow {
  let daemon: DaemonRpc | null = null
  let remote: RemoteConnection | null = null
  let remoteMachine: RemoteTarget | null = null
  let remoteAttempt: AbortController | null = null
  let remoteError = ''
  let providerForwarding: DesktopProviderForwarding | null = null
  let localProviderDaemon: DaemonRpc | null = null
  const closeProviderForwarding = () => {
    void providerForwarding?.disconnect()
    providerForwarding = null
    localProviderDaemon?.dispose()
    localProviderDaemon = null
  }
  let selectedWorkspace: string | null = null
  let resumeSession: string | null = initialSessionId ?? saved?.sessionId ?? null
  let currentSession: string | null = resumeSession


  // Recording the workspace in the recents list is bookkeeping, not a
  // precondition for showing a window. This runs inside the whenReady restore
  // loop, before the window exists, so letting an unwritable or unparseable
  // desktop.json throw here left the user with no windows at all.
  if (initialWorkspace) {
    try {
      saveDesktopWorkspace(workspaceFile(), initialWorkspace)
    } catch (error) {
      console.error('Could not record the workspace in recents:', error instanceof Error ? error.message : error)
    }
  }
  const window = host ?? createWindow(saved)
  const view = host ? new WebContentsView({ webPreferences: {
    contextIsolation: true, backgroundThrottling: false, nodeIntegration: false,
    sandbox: true, preload: join(here, 'preload.js'),
  } }) : null
  const contents = view?.webContents ?? window.webContents
  if (view) {
    view.setBackgroundColor('#181b23')
    const fit = () => { if (!window.isDestroyed()) { const [width, height] = window.getContentSize(); view.setBounds({ x: 0, y: 0, width: width!, height: height! }) } }
    window.contentView.addChildView(view)
    fit()
    window.on('resize', fit)
    contents.once('destroyed', () => window.removeListener('resize', fit))
    contents.setWindowOpenHandler(({ url }) => { void shell.openExternal(url); return { action: 'deny' } })
    contents.on('will-navigate', event => event.preventDefault())
    void contents.loadFile(join(here, 'renderer', 'index.html')).catch(error => console.error('Workspace view failed:', error))
  }
  const recover = windowRecovery({
    closed: () => window.isDestroyed(),
    prompt: async detail => {
      window.show()
      const result = await dialog.showMessageBox(window, {
        type: 'error', message: 'The workspace view stopped responding',
        detail: `${detail}\n\nReload this view to reconnect to your session. Running work stays in the daemon.`,
        buttons: ['Reload view', 'Cancel'], defaultId: 0, cancelId: 1,
      })
      return result.response === 0
    },
    reload: async () => {
      resumeSession = currentSession
      await contents.loadFile(join(here, 'renderer', 'index.html'))
      if (!window.isDestroyed()) {
        window.show()
        window.focus()
        contents.focus()
      }
    },
    report: error => console.error('Desktop recovery failed:', error),
  })
  contents.on('render-process-gone', (_event, details) => {
    if (details.reason !== 'clean-exit') void recover(`Renderer exited: ${details.reason} (code ${details.exitCode}).`)
  })
  contents.on('did-fail-load', (_event, code, description, _url, isMainFrame) => {
    // ERR_ABORTED is expected when a workspace switch replaces a navigation.
    if (isMainFrame && code !== -3) void recover(`Page load failed: ${description} (${code}).`)
  })
  const id = contents.id
  workspaceSurfaces.set(id, { host: window, view })
  activateSurface(id)
  const attach = (next?: DaemonRpc) => registerDaemonBridge(contents, next, (type, payload) => maybeNotify(window, type, payload), (method, result) => {
    if (method !== 'initialize' && method !== 'session.open') return
    const session = result.session as Record<string, unknown> | undefined
    if (session && typeof session.id === 'string' && /^[a-zA-Z0-9_-]{1,256}$/.test(session.id)) {
      currentSession = session.id
      scheduleWindowSave()
    }
  })
  const handle = <Args extends unknown[], Result>(channel: string, handler: (event: IpcMainInvokeEvent, ...args: Args) => Result): void => {
    windowRoutes.bind(id, channel, handler)
    if (!registeredChannels.has(channel)) {
      registeredChannels.add(channel)
      ipcMain.handle(channel, (event, ...args: unknown[]) => windowRoutes.invoke(channel, event, args))
    }
  }
  const chromeState = () => ({ trafficLights: process.platform === 'darwin' && !window.isFullScreen() })
  const sendChrome = () => { if (!contents.isDestroyed()) contents.send('desktop:window-chrome', chromeState()) }
  handle('desktop:window-chrome', () => chromeState())
  window.on('enter-full-screen', sendChrome)
  window.on('leave-full-screen', sendChrome)
  contents.once('destroyed', () => { window.removeListener('enter-full-screen', sendChrome); window.removeListener('leave-full-screen', sendChrome) })
  /** Bind a fresh connection to the selected workspace on the shared daemon. */
  function useProject(directory: string, sessionId: string | null = null): void {
    if (window.isDestroyed()) throw new Error('Workspace window is closed')
    saveDesktopWorkspace(workspaceFile(), directory)
    remoteAttempt?.abort()
    remoteAttempt = null
    closeProviderForwarding()
    void remote?.close()
    remote = null
    remoteMachine = null
    remoteError = ''
    selectedWorkspace = directory
    window.setTitle(`${basename(directory)} — ${APP_NAME}`)
    resumeSession = sessionId
    currentSession = sessionId
    scheduleWindowSave()
    const next = new DaemonRpc({ projectDir: directory })
    const previous = daemon
    daemon = next
    attach(next)
    previous?.dispose()
    if (!window.isDestroyed()) contents.reload()
  }


  // Workspace gate: with no saved workspace the shell boots WITHOUT a daemon
  // and the renderer shows the create-workspace screen. A daemon target only
  // exists once the user picks a folder — a silent cwd fallback would open
  // the app on a workspace the user never chose.
  const workspace = initialWorkspace
  selectedWorkspace = workspace
  if (workspace) window.setTitle(`${basename(workspace)} — ${APP_NAME}`)
  attach()
  if (workspace) {
    daemon = new DaemonRpc({ projectDir: workspace })
    attach(daemon)
    // Warm the connection without gating first paint on it.
    void daemon.call('runtime.status', {}).catch(() => {})
  }
  // The renderer's workspace gate lands here: pick a folder (or enter one
  // from the sidebar), bind the shared-daemon bridge to that workspace, reload for
  // a clean session view.
  handle('desktop:choose-workspace', async () => {
    const picked = await pickWorkspace()
    if (!picked) return null
    if (selectedWorkspace || remote) openWorkspaceView(window, picked)
    else useProject(picked)
    return picked
  })
  handle('desktop:use-workspace', (_event, dir: unknown, sessionId?: unknown) => {
    if (typeof dir !== 'string' || !isAbsolute(dir) || /[\x00-\x1f]/.test(dir)) throw new TypeError('Invalid workspace directory')
    if (sessionId !== undefined && (typeof sessionId !== 'string' || !sessionId || sessionId.length > 256 || /[\x00-\x1f]/.test(sessionId))) throw new TypeError('Invalid resume session id')
    if (remoteMachine) {
      openRemoteView(window, { ...remoteMachine, workspacePath: dir }, typeof sessionId === 'string' ? sessionId : undefined)
      return dir
    }
    if (selectedWorkspace || remote) {
      if (dir !== selectedWorkspace || remote || sessionId) openWorkspaceView(window, dir, typeof sessionId === 'string' ? sessionId : undefined)
    } else useProject(dir, typeof sessionId === 'string' ? sessionId : null)
    return dir
  })
  handle('desktop:workspaces', () => remoteMachine || saved?.remote ? [] : loadDesktopWorkspaces(workspaceFile()))
  handle('desktop:context-scope', () => contextScope(remoteMachine ?? saved?.remote))
  handle('desktop:contexts', async () => {
    const currentScope = contextScope(remoteMachine ?? saved?.remote)
    const scopes = new Set([currentScope])
    const readers: Array<Promise<WorkspaceContext>> = []
    for (const [otherId, surface] of workspaceSurfaces) {
      if (surface.host !== window) continue
      const scope = contextScope(windowStates.get(otherId)?.().remote)
      const read = contextReaders.get(otherId)
      if (scopes.has(scope) || !read) continue
      scopes.add(scope)
      readers.push(read())
    }
    return Promise.all(readers)
  })
  handle('desktop:activate-context', async (_event, contextId: unknown, sessionId?: unknown) => {
    if (typeof contextId !== 'number' || workspaceSurfaces.get(contextId)?.host !== window) throw new Error('Workspace context is unavailable')
    const state = windowStates.get(contextId)?.()
    if (!state) throw new Error('Workspace context is closed')
    if (sessionId === undefined) { activateSurface(contextId); return }
    const row = (await contextReaders.get(contextId)?.())?.sessions.find(row => row.id === sessionId)
    if (!row) throw new Error('Session is no longer available in this workspace')
    if (state.remote) openRemoteView(window, { ...state.remote, workspacePath: row.cwd }, row.id)
    else openWorkspaceView(window, row.cwd, row.id)
  })
  handle('desktop:workspace', () => selectedWorkspace)
  handle('desktop:resume', () => {
    const selected = resumeSession
    resumeSession = null
    return selected
  })
  let voiceRequest: AbortController | null = null
  handle('desktop:voice', async (_event, action: unknown, value: unknown) => {
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
  async function connectRemote(params: Record<string, unknown>) {
    if (remoteAttempt) throw new Error('A connection is already being prepared')
    const machine = remoteTarget(params.machine),
      controller = new AbortController()
    remoteAttempt = controller
    remoteError = ''
    try {
      let next = await openRemote(machine, controller.signal, (error) => {
        remoteError = error.message
      })
      const localProviders = new DaemonRpc({projectDir:homedir()})
      let forwarding: DesktopProviderForwarding
      const rpc = new DaemonRpc({ projectDir: next.projectDir, socketPath: next.socketPath,
        providerRelay: (binding, frame, signal) => forwarding.relay(binding, frame, signal),
        remoteUpdate: () => next.update(), expectedRemoteBuildId: () => next.expectedBuildId,
        reconnectRemote: async signal => {
          const replacement = await next.reconnect(signal, error => { remoteError = error.message })
          if (signal.aborted || replacement.projectDir !== next.projectDir) {
            await replacement.close()
            signal.throwIfAborted()
            throw new Error('Remote workspace changed during reconnect.')
          }
          const previous = next
          next = replacement
          if (remote === previous) remote = replacement
          remoteError = ''
          await previous.close()
          return replacement.socketPath
        },
      })
      forwarding = new DesktopProviderForwarding((method, args) => localProviders.call(method,args),
        (method,args) => rpc.call(method,args), machine.target, next.projectDir)
      rpc.onConnection(online => { if (!online) void forwarding.disconnect() })
      localProviders.onConnection(online => { if (!online) void forwarding.disconnect() })
      try {
        await rpc.call('runtime.status')
        controller.signal.throwIfAborted()
      } catch (error) {
        void forwarding.disconnect()
        localProviders.dispose()
        rpc.dispose()
        await next.close()
        throw error
      }
      resumeSession =
        (remoteMachine ?? saved?.remote)?.target === machine.target &&
        (remoteMachine ?? saved?.remote)?.workspacePath === machine.workspacePath &&
        typeof params.resume_session_id === 'string' &&
        /^[a-zA-Z0-9_-]{1,128}$/.test(params.resume_session_id)
          ? params.resume_session_id
          : null
      const previous = remote
      closeProviderForwarding()
      providerForwarding = forwarding
      localProviderDaemon = localProviders
      daemon?.dispose()
      daemon = rpc
      attach(rpc)
      remote = next
      remoteMachine = machine
      selectedWorkspace = next.projectDir
      currentSession = resumeSession
      scheduleWindowSave()
      if (activeSurfaces.get(window.id) === id) window.setTitle(`${machine.alias} · ${basename(next.projectDir)} — ${APP_NAME}`)
      await previous?.close()
      if (!window.isDestroyed()) contents.reload()
      return { ok: true }
    } catch (error) {
      remoteError = error instanceof Error ? error.message : String(error)
      throw error
    } finally {
      if (remoteAttempt === controller) remoteAttempt = null
    }

  }
  handle(
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
          machine: remoteMachine ?? saved?.remote ?? null,
          connected: Boolean(remote && daemon?.online),
          connecting: Boolean(remoteAttempt),
          resume_session_id: resumeSession ?? currentSession,
          error: remoteError,
        }
      if (action === 'cancel') {
        remoteAttempt?.abort()
        return { ok: true }
      }
      if (action === 'provider-review' || action === 'provider-share' || action === 'provider-revoke') {
        if (!remote || !daemon?.online || !providerForwarding) throw new Error('Reconnect the SSH workspace before reviewing local provider access.')
        if (action === 'provider-review') return providerForwarding.inspect()
        if (action === 'provider-share') return providerForwarding.authorize(params)
        return providerForwarding.revoke(params.sessionKey)
      }
      if (action === 'connect') {
        const machine = remoteTarget(params.machine)
        // A connection belongs to a retained conversation context. Never replace
        // a local binding or another host just because the user selected SSH.
        const bound = remoteMachine ?? saved?.remote
        if ((selectedWorkspace || bound) && !sameRemote(bound, machine)) {
          openRemoteView(window, machine)
          return { ok: true }
        }
        return connectRemote(params)
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
  handle('native:notifications:set', (_event, on: unknown) => {
    if (typeof on !== 'boolean') throw new TypeError('notifications expects a boolean')
    notificationsEnabled = on
    return notificationsEnabled
  })
  handle('native:login-item:get', () => app.getLoginItemSettings().openAtLogin)
  handle('native:login-item:set', (_event, on: unknown) => {
    if (typeof on !== 'boolean') throw new TypeError('login item expects a boolean')
    app.setLoginItemSettings({ openAtLogin: on })
    return app.getLoginItemSettings().openAtLogin
  })
  handle('native:preset:open-path', async (_event, value: unknown) => {
    if (typeof value !== 'string' || !value) throw new TypeError('invalid preset path')
    const root = resolve(xerxesHome(), 'agents')
    const candidate = resolve(value)
    const fromRoot = relative(root, candidate)
    if (!fromRoot || fromRoot.startsWith('..') || isAbsolute(fromRoot)) {
      throw new Error('preset path must be a child of the Xerxes user agent directory')
    }
    return (await shell.openPath(candidate)) === ''
  })
  handle('desktop:new-window', async (_event, directory?: unknown, sessionId?: unknown) => {
    if (sessionId !== undefined && (typeof sessionId !== 'string' || !sessionId || sessionId.length > 256 || /[\x00-\x1f]/.test(sessionId))) throw new TypeError('Invalid session identity')
    if (directory !== undefined && (typeof directory !== 'string' || !isAbsolute(directory) || /[\x00-\x1f]/.test(directory))) throw new TypeError('Invalid workspace directory')
    const picked = typeof directory === 'string' ? directory : await pickWorkspace()
    if (!picked || window.isDestroyed()) return null
    if (remoteMachine && picked === selectedWorkspace) {
      const savedWindow = windowStates.get(id)?.()
      if (!savedWindow) throw new Error('Remote workspace is not ready')
      createWorkspaceWindow(null, { ...savedWindow, sessionId: typeof sessionId === 'string' ? sessionId : null }, undefined, window)
    } else if (sessionId) openWorkspaceView(window, picked, sessionId as string)
    else createWorkspaceWindow(picked)
    return picked
  })
  const cleanup = () => {
    contextReaders.delete(id)
    workspaceSurfaces.delete(id)
    if (activeSurfaces.get(window.id) === id) activeSurfaces.delete(window.id)
    if (view && !contents.isDestroyed()) contents.close()
    windowStates.delete(id)
    if (!quitting) scheduleWindowSave()
    windowRoutes.remove(id)
    detachDaemon(id)
    closeProviderForwarding()
    daemon?.dispose()
    remoteAttempt?.abort()
    voiceRequest?.abort()
    void remote?.close()
    cleanupWindows.delete(cleanup)
  }
  cleanupWindows.add(cleanup)
  window.once('closed', cleanup)
  windowStates.set(id, () => ({
    workspace: remoteMachine || saved?.remote && !selectedWorkspace ? null : selectedWorkspace,
    remote: remoteMachine ?? (selectedWorkspace ? null : saved?.remote ?? null),
    windowGroup: String(window.id), active: activeSurfaces.get(window.id) === id,
    sessionId: currentSession, bounds: window.getNormalBounds(), maximized: window.isMaximized(), fullscreen: window.isFullScreen(),
  }))
  let cachedSessions: WorkspaceContext['sessions'] = []
  let pendingSessions: Promise<void> | undefined
  contextReaders.set(id, async () => {
    if (daemon?.online && !pendingSessions) {
      pendingSessions = (async () => { try {
        const [savedRows, liveRows] = await Promise.all([
          daemon.call<Record<string, unknown>>('session.list', { kind: 'main', scope: 'global', limit: 60 }),
          daemon.call<Record<string, unknown>>('session.active_list', { history_limit: 0 }),
        ])
        cachedSessions = contextSessions(savedRows.sessions, liveRows.sessions)
      } catch { /* A disconnected endpoint keeps its last known navigation. */ }
      finally { pendingSessions = undefined }
      })()
    }
    const machine = remoteMachine ?? saved?.remote
    return { id, label: machine ? `${machine.alias} · SSH` : 'This Mac',
      workspace: selectedWorkspace ?? machine?.workspacePath ?? '', remote: Boolean(machine), sessions: cachedSessions }
  })
  window.on('move', scheduleWindowSave)
  window.on('resize', scheduleWindowSave)
  window.on('maximize', scheduleWindowSave)
  window.on('unmaximize', scheduleWindowSave)
  window.on('enter-full-screen', scheduleWindowSave)
  window.on('leave-full-screen', scheduleWindowSave)
  scheduleWindowSave()
  // The retained SSH view reads status and offers retry/cancel inline. A modal
  // on the host can interrupt a different, healthy workspace during restoration.
  if (saved?.remote) void connectRemote({ machine: saved.remote, resume_session_id: saved.sessionId }).catch(error => {
    console.error('Could not reopen SSH workspace:', error)
  })
  return window
}

void app.whenReady().then(async () => {
  app.setAboutPanelOptions({ applicationName: APP_NAME })
  // The dock/taskbar icon is the phoenix mark; on macOS the running app's
  // dock tile only changes through app.dock.
  const icon = appIcon()
  if (icon && process.platform === 'darwin') app.dock?.setIcon(icon)

  // Commands the menu can only express by asking the focused renderer:
  // the shell owns sessions, the palette and the find bar.
  const toFocused = (command: string) => () => {
    BrowserWindow.getFocusedWindow()?.webContents.send('desktop:menu', command)
  }
  // `role: 'appMenu'` ships no Preferences item, so ⌘, was the one binding
  // the app honoured and never advertised; there was also no Help menu and
  // no Find, in an app whose main surface is thousands of lines of output.
  const appMenu = {
    label: APP_NAME,
    submenu: [
      { role: 'about' as const },
      { type: 'separator' as const },
      { label: 'Settings…', accelerator: 'CmdOrCtrl+,', click: toFocused('settings') },
      { type: 'separator' as const },
      { role: 'services' as const },
      { type: 'separator' as const },
      { role: 'hide' as const }, { role: 'hideOthers' as const }, { role: 'unhide' as const },
      { type: 'separator' as const },
      { role: 'quit' as const },
    ],
  }
  Menu.setApplicationMenu(Menu.buildFromTemplate([
    ...(process.platform === 'darwin' ? [appMenu] : []),
    { label: 'File', submenu: [
      { label: 'New Task', accelerator: 'CmdOrCtrl+N', click: toFocused('new-task') },
      { label: 'New Task from Template…', accelerator: 'CmdOrCtrl+Alt+N', click: toFocused('new-task-wizard') },
      { type: 'separator' as const },
      { label: 'New Window', accelerator: 'CmdOrCtrl+Shift+N', click: () => { createWorkspaceWindow() } },
      { label: 'Open Workspace in New Window…', accelerator: 'CmdOrCtrl+Shift+O', click: () => { void pickWorkspace().then(directory => { if (directory) createWorkspaceWindow(directory) }) } },
      { type: 'separator' as const },
      { label: 'Export Transcript…', accelerator: 'CmdOrCtrl+E', click: toFocused('export') },
      { type: 'separator' }, { role: 'close' },
    ] },
    { label: 'Edit', submenu: [
      { role: 'undo' as const }, { role: 'redo' as const },
      { type: 'separator' as const },
      { role: 'cut' as const }, { role: 'copy' as const }, { role: 'paste' as const }, { role: 'selectAll' as const },
      { type: 'separator' as const },
      { label: 'Find in Conversation…', accelerator: 'CmdOrCtrl+F', click: toFocused('find') },
      { label: 'Search All Tasks…', accelerator: 'CmdOrCtrl+Shift+F', click: toFocused('search') },
      { label: 'Command Palette…', accelerator: 'CmdOrCtrl+K', click: toFocused('palette') },
    ] },
    { role: 'viewMenu' }, { role: 'windowMenu' },
    { role: 'help', submenu: [
      { label: 'Keyboard Shortcuts', accelerator: 'CmdOrCtrl+/', click: toFocused('shortcuts') },
    ] },
  ]))
  const saved = loadWindowLayout(layoutFile())
  if (saved?.length) {
    const hosts = new Map<string, BrowserWindow>()
    const selected: number[] = []
    for (const state of saved) {
      const group = state.windowGroup ?? 'legacy-workspaces'
      const host = hosts.get(group)
      const window = createWorkspaceWindow(state.remote ? null : state.workspace, state, undefined, host)
      hosts.set(group, window)
      if (state.active) selected.push(activeSurfaces.get(window.id)!)
    }
    for (const id of selected) activateSurface(id)
  }
  else createWorkspaceWindow(loadWorkspace())
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWorkspaceWindow(loadWorkspace()) })
})
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })
app.on('before-quit', () => { persistWindows(); quitting = true })
app.on('will-quit', () => { for (const cleanup of [...cleanupWindows]) cleanup() })
