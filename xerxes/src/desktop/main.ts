// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Electron entry. Owns independent workspace windows and their daemon connections; everything
 * renderer-side crosses the preload bridge (`daemon:call` in, `daemon:event`
 * out). The renderer is sandboxed with no Node access and a self-only CSP —
 * those properties are the design, not configuration.
 */

import { app, BrowserWindow, WebContentsView, clipboard, dialog, ipcMain, nativeImage, Notification, powerMonitor, safeStorage, session as electronSession, shell, Menu, screen, webContents, type IpcMainInvokeEvent } from 'electron'
import { existsSync, readFileSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { basename, dirname, isAbsolute, join, relative, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

import { activateWorkspaceView } from './main/workspaceNavigation.js'
import { contextKey, contextLabel, contextScope, contextSessions, latestSessionIn, sameRemote, sessionsFor, type WorkspaceContext } from './main/contextNavigation.js'
import { loadWindowLayout, saveWindowLayout, visibleWindowBounds, WindowLayoutRecorder, type SavedWindow } from './main/windowState.js'
import {
  applyChromiumPerformanceSwitches,
  createIdleCacheTrimmer,
  pickEvictableBackgroundViews,
  planStartupSurfaces,
  setBoundedMapEntry,
} from './main/perfHygiene.js'
import { SurfaceSession } from './main/surfaceSession.js'
import { WindowRoutes } from './main/windowRoutes.js'
import { windowRecovery } from './main/windowRecovery.js'
import { desktopMachineCommand } from './main/machines.js'
import { SavedHostContexts } from './main/savedHosts.js'
import { xerxesHome } from '../daemon/paths.js'
import { DaemonRpc } from './main/daemon.js'
import { DesktopProviderForwarding } from './main/providerForwarding.js'
import { localActiveProfile } from './main/profileSync.js'
import { ProviderAccountDirectory } from './main/providerAccounts.js'
import { spawnSecretWriter } from './main/remoteCredentials.js'
import { ensureRemoteProviderClis } from './main/remoteProviderSetup.js'
import { ensureLocalProviderClis } from './main/localProviderSetup.js'
import { ProviderAccountRegistry } from '../auth/providerAccountRegistry.js'
import { GlobalProviderAccounts } from './main/globalProviderAccounts.js'
import { runBunProviderAccountLogin } from './main/providerAccountLogin.js'
import { RemoteNetwork } from './main/remoteNetwork.js'
import { BrowserTab } from './main/browserTab.js'
import { startAppUpdates } from './main/appUpdateController.js'
import { registerDaemonBridge, detachDaemon } from './main/ipc.js'
import { dictationPort, transcribeDictation } from './main/voice.js'
import { localDesktopWorkspaces, rememberDesktopWorkspace, removeDesktopWorkspace, saveDesktopWorkspace } from './main/workspaceSettings.js'
import {
  openRemote,
  remoteTarget,
  type RemoteConnection,
  type RemoteTarget,
} from './main/remote.js'
import { shouldNotify } from './main/notify.js'
import { createCliRunner } from './main/hubCli.js'
import { createHubCockpit } from './main/hubCockpit.js'
import { registerHubCockpitIpc } from './main/hubCockpitIpc.js'
import { nodeHubHttp } from './main/hubHttp.js'
import { connectHubSocket } from './main/hubSocket.js'
import { createHubStorage } from './main/hubStorage.js'
import { createNodeShareDeps, createShareRegistry, shareRemoteWorkspace, type HubShareDeps, type HubShareHandle } from './main/hubShare.js'
import { createMacPermissions, registerMacPermissionsIpc } from './main/macPermissions.js'
import { electronMacPermissionsPort } from './main/macPermissionsPort.js'
import { MAC_PERMISSIONS_STATE_EVENT } from './macPermissionsShared.js'
import { createHubHealth, createNodeHubHealthPort, registerHubHealthIpc } from './main/hubHealth.js'
import { HUB_HEALTH_STATE_EVENT } from './hubHealthShared.js'
import { IMAGE_DATA_CHANNEL, OPEN_EXTERNAL_CHANNEL, PROJECT_ICON_CHANNEL, chatImageRequestOf, externalUrlOf } from './chatMediaShared.js'
import { readLocalChatImage, readRemoteChatImage } from './main/chatMedia.js'
import { resolveProjectIconDataUrl } from './main/projectFavicon.js'
import { FILE_ACTION_CHANNEL, fileActionRequestOf } from './fileActionShared.js'
import { performFileAction } from './main/fileActions.js'

/** Keep an SSH conversation on this computer's sign-in provider; a failure is logged, never thrown. */
async function followLocalProviders(forwarding: DesktopProviderForwarding): Promise<void> {
  const result = await forwarding.follow()
  if (result.status === 'failed') console.warn(`Could not use this computer's sign-in provider on the SSH host: ${result.reason}`)
}

const APP_NAME = 'Xerxes Agents'
/** Longer than the runtime's 30 s connection-lease grace, so a reattach always wins. */
const REATTACH_GRACE_MS = 35_000
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
// Disable unused Chromium background networking/sync/component services before ready.
applyChromiumPerformanceSwitches(app.commandLine)
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
// Local and SSH workspace events share the same native display policy.

let notificationsEnabled = true

/** Last goal phase seen per session, so a repeated status_update never pings twice. */
const goalPhases = new Map<string, string>()
const MAX_GOAL_PHASE_ENTRIES = 256

function maybeNotify(window: BrowserWindow, contents: Electron.WebContents, type: string, payload: Record<string, unknown>, stillBound: () => boolean): void {
  // Track transitions even when notifications are off, so enabling them never replays one.
  let previousGoalPhase: string | undefined
  if (type === 'status_update' && typeof payload.goal_phase === 'string') {
    const key = typeof payload.session_id === 'string' ? payload.session_id : ''
    previousGoalPhase = goalPhases.get(key)
    setBoundedMapEntry(goalPhases, key, payload.goal_phase, MAX_GOAL_PHASE_ENTRIES)
  }
  if (window.isDestroyed() || !Notification.isSupported()) return
  const decision = shouldNotify({ type, payload }, { enabled: notificationsEnabled, anyWindowFocused: window.isFocused() }, previousGoalPhase)
  if (!decision) return
  const ping = new Notification({ title: decision.title, body: decision.body })
  ping.on('click', () => {
    if (window.isDestroyed() || contents.isDestroyed() || !stillBound()) return
    if (window.isMinimized()) window.restore()
    window.show()
    window.focus()
    // The host may still be showing another retained local or SSH workspace.
    activateSurface(contents.id)
    // Open what the notification was about: its task, and an artifact when there is one.
    if (decision.target && !contents.isDestroyed()) contents.send('desktop:navigate', decision.target)
  })
  ping.show()
}

// ── Workspace selection ─────────────────────────────────────────────────
// A workspace is a folder: the shared daemon isolates its sessions, and the
// sidebar groups every chat under its folder name. The chosen folder is the
// window's daemon target, persisted across launches.

const workspaceFile = (): string => join(xerxesHome(), 'desktop.json')
// Shares connected SSH workspaces with the linked Hub as a last-resort route. Created on first use.
let hubShareDeps: HubShareDeps | null = null
const hubShareDepsOf = (): HubShareDeps => (hubShareDeps ??= createNodeShareDeps({ home: xerxesHome(), cli: createCliRunner(process.env, homedir()) }))
// One relay per host label across every window and view (see createShareRegistry).
const shareWithHub = createShareRegistry(input => shareRemoteWorkspace(hubShareDepsOf(), input))

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
    // Center the 14px native buttons in the renderer's 52px T3 top bar.
    trafficLightPosition: { x: 16, y: 19 },
    ...(windowIcon ? { icon: windowIcon } : {}),
    backgroundColor: process.platform === 'darwin' ? '#00000000' : '#000000',
    // Always active: with 'followWindow' the material went inactive whenever
    // another app had focus and stopped sampling what is behind the window,
    // so a transparent background kept showing a frozen copy of whatever was
    // there (another app's window) instead of the live view.
    ...(process.platform === 'darwin' ? { vibrancy: 'under-window' as const, visualEffectState: 'active' as const } : {}),
    show: false,
    webPreferences: {
      contextIsolation: true,
      // Keep the active window's GPU compositor and IPC updates live behind
      // other apps / macOS Mission Control; covered workspace views are
      // explicitly throttled via setBackgroundThrottling(true) in syncWindowVisibility.
      backgroundThrottling: false,
      nodeIntegration: false,
      sandbox: true,
      spellcheck: false,
      enableWebSQL: false,
      webviewTag: false,
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

interface WorkspaceSurface {
  readonly host: BrowserWindow
  readonly view: WebContentsView | null
  lastActiveAt: number
  hasActiveTurn: boolean
  readonly isBusy: () => boolean
  readonly onHostHidden: (hidden: boolean) => void
  readonly suspendBrowser: () => void
}

const workspaceSurfaces = new Map<number, WorkspaceSurface>()
const activeSurfaces = new Map<number, number>()
const contextReaders = new Map<number, () => Promise<WorkspaceContext>>()
const savedHostContexts = new SavedHostContexts({ list: () => desktopMachineCommand(join(xerxesHome(), 'machines.json'), 'list') })
function openContextRemotes(host?: BrowserWindow): RemoteTarget[] {
  const remotes: RemoteTarget[] = []
  for (const [id, surface] of workspaceSurfaces) {
    if (host && surface.host !== host) continue
    const machine = windowStates.get(id)?.().remote
    if (machine) remotes.push(machine)
  }
  return remotes
}
const surfaceSessions = new Map<number, SurfaceSession>()
/** Views that can be closed on their own; a window's own page closes with the window. */
const viewClosers = new Map<number, () => void>()
/**
 * Surfaces (both base pages and WebContentsViews) currently unseen because they
 * are covered by another workspace view, or their host window is minimized,
 * hidden, or the system is locked/suspended. Kept here as well as pushed so a
 * surface that finishes loading after the push asks and gets the same answer.
 */
const occludedSurfaces = new Set<number>()
/** Windows explicitly hidden or minimized by the user (avoids marking a booting pre-ready-to-show window occluded). */
const hiddenWindows = new WeakSet<BrowserWindow>()
let systemSuspendedOrLocked = false

const isHostWindowHidden = (host: BrowserWindow): boolean =>
  systemSuspendedOrLocked || host.isDestroyed() || host.isMinimized() || hiddenWindows.has(host)

const isAppInteractive = (): boolean =>
  !systemSuspendedOrLocked
  && BrowserWindow.getAllWindows().some(win => !win.isDestroyed() && !win.isMinimized() && !hiddenWindows.has(win))

const hasFocusedWindow = (): boolean =>
  !systemSuspendedOrLocked
  && BrowserWindow.getAllWindows().some(win => !win.isDestroyed() && !win.isMinimized() && !hiddenWindows.has(win) && win.isFocused())

function workspaceWebContents(): Electron.WebContents[] {
  const list: Electron.WebContents[] = []
  for (const surface of workspaceSurfaces.values()) {
    const target = surface.view?.webContents ?? surface.host.webContents
    if (!target.isDestroyed()) list.push(target)
  }
  return list
}

const cacheTrimmer = createIdleCacheTrimmer({
  trim: async () => {
    // Trim only Chromium HTTP + compiled code caches; NEVER clear cookies, localStorage, or IndexedDB.
    await Promise.allSettled([
      electronSession.defaultSession.clearCache(),
      electronSession.defaultSession.clearCodeCaches({}),
    ])
  },
})

function syncCacheTrimState(): void {
  if (isAppInteractive()) cacheTrimmer.markForeground()
  else cacheTrimmer.markBackground()
}

function syncWindowVisibility(host: BrowserWindow, forceNotify = false): void {
  if (host.isDestroyed()) return
  const activeId = activeSurfaces.get(host.id)
  const hostHidden = isHostWindowHidden(host)
  for (const [otherId, other] of workspaceSurfaces) {
    if (other.host !== host) continue
    const isActive = otherId === activeId
    // `occluded` means covered by another workspace view inside THIS OS window.
    // Never hide (`visibility:hidden`) or background-throttle the active surface
    // on window hide/blur/space switch: on macOS with vibrancy, Chromium evicts
    // the active GPU IOSurface (solid-black Mission Control thumbnail and a
    // 4-5s repaint stall when switching back).
    const occluded = !isActive
    const wasOccluded = occludedSurfaces.has(otherId)
    if (occluded) occludedSurfaces.add(otherId)
    else occludedSurfaces.delete(otherId)
    if (other.view) {
      other.view.setVisible(isActive)
      if (!other.view.webContents.isDestroyed()) {
        other.view.webContents.setBackgroundThrottling(occluded)
        if (forceNotify || occluded !== wasOccluded) {
          other.view.webContents.send('desktop:occluded', occluded)
        }
      }
    } else {
      // Base page: throttle and hide only while covered by a foreground view.
      if (!other.host.webContents.isDestroyed()) {
        other.host.webContents.setBackgroundThrottling(occluded)
        if (forceNotify || occluded !== wasOccluded) {
          other.host.webContents.send('desktop:occluded', occluded)
        }
      }
    }
    other.onHostHidden(hostHidden || occluded)
  }
  syncCacheTrimState()
}

function syncAllWindowVisibility(): void {
  for (const win of BrowserWindow.getAllWindows()) {
    if (!win.isDestroyed()) syncWindowVisibility(win)
  }
  syncCacheTrimState()
}

function evictBackgroundViews(host?: BrowserWindow): void {
  if (quitting) return
  const hosts = host ? [host] : BrowserWindow.getAllWindows()
  const now = Date.now()
  for (const win of hosts) {
    if (win.isDestroyed()) continue
    const activeId = activeSurfaces.get(win.id)
    const candidates = []
    for (const [surfaceId, surface] of workspaceSurfaces) {
      if (surface.host !== win) continue
      candidates.push({
        id: surfaceId,
        isBasePage: surface.view === null,
        isActive: surfaceId === activeId,
        hasActiveTurn: surface.hasActiveTurn,
        isBusy: surface.isBusy(),
        lastActiveAt: surface.lastActiveAt,
      })
    }
    for (const victimId of pickEvictableBackgroundViews(candidates, now)) {
      viewClosers.get(victimId)?.()
    }
  }
}

function activateSurface(id: number): void {
  const surface = workspaceSurfaces.get(id)
  if (!surface || surface.host.isDestroyed()) return
  surface.lastActiveAt = Date.now()
  activeSurfaces.set(surface.host.id, id)
  windowStates.used(id)
  syncWindowVisibility(surface.host, true)
  // An already attached view keeps its old stacking position. Activation must
  // bring it above the other retained workspaces as well as make it visible.
  if (surface.view) surface.host.contentView.addChildView(surface.view)
  const contents = surface.view?.webContents ?? surface.host.webContents
  if (!contents.isDestroyed()) contents.focus()
  const state = windowStates.get(id)?.()
  surface.host.setTitle(state?.remote
    ? `${state.remote.alias} · ${basename(state.remote.workspacePath)} — ${APP_NAME}`
    : state?.workspace ? `${basename(state.workspace)} — ${APP_NAME}` : APP_NAME)
  evictBackgroundViews(surface.host)
  scheduleWindowSave()
}

const windowStates = new WindowLayoutRecorder()
let quitting = false
let saveTimer: ReturnType<typeof setTimeout> | undefined
const layoutFile = () => join(xerxesHome(), 'desktop-windows.json')
function persistWindows(): void {
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = undefined
  try { saveWindowLayout(layoutFile(), windowStates.rows()) }
  catch (error) { console.error('Could not save desktop windows:', error) }
}
function scheduleWindowSave(): void {
  if (quitting) return
  if (saveTimer) clearTimeout(saveTimer)
  saveTimer = setTimeout(persistWindows, 250)
}

/**
 * Tell a retained view that one of its tasks was clicked. Activating a view
 * only shows it as it was left, so a view last on XERXES (or another page)
 * came back on that page instead of the chat that was clicked.
 */
function showTask(id: number, sessionId: string): void {
  const surface = workspaceSurfaces.get(id)
  const contents = surface?.view?.webContents ?? surface?.host.webContents
  if (contents && !contents.isDestroyed()) contents.send('desktop:navigate', { sessionId })
}

function openWorkspaceView(host: BrowserWindow, directory: string, sessionId?: string): void {
  const views = [...workspaceSurfaces].flatMap(([id, surface]) => {
    const state = windowStates.get(id)?.()
    return state && surface.host === host ? [{ ...state, cwd: surfaceSessions.get(id)?.cwd ?? null, isDestroyed: () => host.isDestroyed(),
      isMinimized: () => host.isMinimized(), restore: () => host.restore(),
      show: () => host.show(), focus: () => { host.focus(); activateSurface(id); if (sessionId) showTask(id, sessionId) } }] : []
  })
  if (!activateWorkspaceView(views, directory, sessionId)) createWorkspaceWindow(directory, undefined, sessionId, host)
}

/**
 * File > Close. Navigation and mid-turn tasks open views, and nothing else
 * ever closed one: each kept a renderer and a daemon connection for the life
 * of the window and was restored on every launch. Closing the shown view
 * leaves its session in the daemon and the sidebar; the window itself closes
 * only when its own page is what is shown.
 */
function closeActiveSurface(): void {
  const window = BrowserWindow.getFocusedWindow()
  if (!window) return
  const active = activeSurfaces.get(window.id)
  const close = active === undefined ? undefined : viewClosers.get(active)
  if (!close) { window.close(); return }
  close()
  const remaining = [...workspaceSurfaces].filter(([, surface]) => surface.host === window).map(([id]) => id)
  const next = windowStates.mostRecent(remaining) ?? remaining[0]
  if (next !== undefined) activateSurface(next)
}

/** The view in this window already showing that SSH workspace (and session, when given). */
function remoteViewOf(host: BrowserWindow, machine: RemoteTarget, sessionId?: string): number | undefined {
  for (const [id, surface] of workspaceSurfaces) {
    const state = windowStates.get(id)?.()
    if (surface.host === host && sameRemote(state?.remote, machine) && (!sessionId || state?.sessionId === sessionId)) return id
  }
  return undefined
}

function openRemoteView(host: BrowserWindow, machine: RemoteTarget, sessionId?: string): void {
  const existing = remoteViewOf(host, machine, sessionId)
  if (existing !== undefined) { activateSurface(existing); if (sessionId) showTask(existing, sessionId); return }
  createWorkspaceWindow(null, { workspace: null, remote: machine, sessionId: sessionId ?? null,
    bounds: host.getNormalBounds(), maximized: host.isMaximized(), fullscreen: host.isFullScreen(),
  }, undefined, host)
}

function createWorkspaceWindow(initialWorkspace: string | null = null, saved?: SavedWindow, initialSessionId?: string, host?: BrowserWindow, startFresh = false, startPreset?: string): BrowserWindow {
  let daemon: DaemonRpc | null = null
  let remote: RemoteConnection | null = null
  let remoteMachine: RemoteTarget | null = null
  let remoteAttempt: AbortController | null = null
  let remoteError = ''
  // The connected SSH workspace, relayed to the Hub while this window holds it.
  let hubShare: HubShareHandle | null = null
  const stopHubShare = () => { void hubShare?.close(); hubShare = null }
  const startHubShare = (label: string, socketPath: string) => {
    stopHubShare()
    hubShare = shareWithHub({ label, localSocket: socketPath })
  }
  let providerForwarding: DesktopProviderForwarding | null = null
  let localProviderDaemon: DaemonRpc | null = null
  // Explicit per-host provider binding; null when local or no host is attached.
  let accountDirectory: ProviderAccountDirectory | null = null
  let globalAccounts: GlobalProviderAccounts | null = null
  let globalAccountsTarget: string | undefined
  // The SSH workspace's network (browser-tab SOCKS + port forwards) and the Browser tab's page.
  let remoteNetwork: RemoteNetwork | null = null
  let browserTab: BrowserTab | null = null
  const networkOf = (): RemoteNetwork | null => remoteMachine ? (remoteNetwork ??= new RemoteNetwork(remoteMachine)) : null
  const closeProviderForwarding = () => {
    // A workspace change or close ends its tunnels and drops the page that used them.
    browserTab?.reset()
    remoteNetwork?.close()
    remoteNetwork = null
    accountDirectory = null
    globalAccounts = null
    globalAccountsTarget = undefined
    void providerForwarding?.disconnect()
    providerForwarding = null
    localProviderDaemon?.dispose()
    localProviderDaemon = null
  }
  const accountCenter = () => {
    const target = remoteMachine?.target
    if (!globalAccounts || globalAccountsTarget !== target) {
      globalAccountsTarget = target
      globalAccounts = new GlobalProviderAccounts(
        new ProviderAccountRegistry(),
        remoteMachine ?? undefined,
        (method, args) => {
          if (!daemon) throw new Error('The workspace is not connected')
          return daemon.call(method, args)
        },
        async (_method, args, onAuthorizationUrl) => {
          const accountId = args?.account_id
          if (typeof accountId !== 'string' || !accountId) throw new Error('An account id is required')
          // Login is a standalone Bun helper: an app relaunch may leave an older
          // daemon running which does not know provider_account.login yet.
          await runBunProviderAccountLogin(accountId, { onAuthorizationUrl })
          return { ok: true }
        },
      )
    }
    return globalAccounts
  }
  let selectedWorkspace: string | null = null
  const session = new SurfaceSession(initialSessionId ?? saved?.sessionId ?? null)


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
    sandbox: true, spellcheck: false, enableWebSQL: false, webviewTag: false,
    preload: join(here, 'preload.js'),
  } }) : null
  const contents = view?.webContents ?? window.webContents
  if (view) {
    // Clear on macOS so a transparent background shows the window's native
    // material, like the window's own contents do.
    view.setBackgroundColor(process.platform === 'darwin' ? '#00000000' : '#181b23')
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
  let voiceRequest: AbortController | null = null
  const surfaceRecord: WorkspaceSurface = {
    host: window,
    view,
    lastActiveAt: Date.now(),
    hasActiveTurn: false,
    isBusy: () => Boolean(remoteAttempt || voiceRequest),
    onHostHidden: hidden => {
      if (hidden) browserTab?.setBounds(null)
    },
    suspendBrowser: () => {
      browserTab?.suspend()
    },
  }
  workspaceSurfaces.set(id, surfaceRecord)
  surfaceSessions.set(id, session)
  activateSurface(id)
  const attach = (next?: DaemonRpc) => {
    surfaceRecord.hasActiveTurn = false
    next?.onConnection(online => { if (!online) surfaceRecord.hasActiveTurn = false })
    registerDaemonBridge(contents, next, (type, payload) => {
      if (type === 'turn_begin') surfaceRecord.hasActiveTurn = true
      else if (type === 'turn_end') surfaceRecord.hasActiveTurn = false
      // Some session-bound events omit the id. Capture the owning task now,
      // before a later session.open changes what this surface shows.
      const current = session.current
      maybeNotify(window, contents, type, current && payload.session_id === undefined ? { ...payload, session_id: current } : payload,
        () => workspaceSurfaces.has(id) && daemon === next)
      // A turn that ended (perhaps because access was lost while it ran, after
      // an app restart) leaves the conversation idle: bind it again now, so a
      // Retry works instead of failing the same way.
      if (type === 'turn_end' && remote && providerForwarding) void followLocalProviders(providerForwarding)
    }, (method, result) => {
      if (session.observe(method, result)) scheduleWindowSave()
      // Opening or starting a conversation on an SSH host: keep it on this
      // computer's sign-in provider (ChatGPT, Claude Code), whose login stays here.
      if ((method === 'initialize' || method === 'session.open') && remote && providerForwarding) void followLocalProviders(providerForwarding)
    })
  }
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
  handle('desktop:browser', async (_event, action: unknown, params: Record<string, unknown> = {}) => {
    if (window.isDestroyed()) throw new Error('Workspace window is closed')
    const tab = browserTab ??= new BrowserTab(window, networkOf, state => { if (!contents.isDestroyed()) contents.send('desktop:browser-state', state) })
    const args = params && typeof params === 'object' ? params : {}
    switch (action) {
      case 'state': return tab.state()
      case 'open': return tab.open(args.url)
      case 'back': case 'forward': case 'reload': case 'stop': return tab.navigate(action)
      case 'bounds': tab.setBounds(args.rect ?? null); return { ok: true }
      case 'external': await tab.openExternal(); return { ok: true }
      case 'reset': tab.reset(); return { ok: true }
      case 'ports': return { via: remoteMachine?.alias ?? null, ports: remoteNetwork?.list() ?? [] }
      case 'port-forward': {
        const network = networkOf()
        if (!network) throw new Error('Port forwarding needs an SSH workspace; local ports are already on this computer')
        const local = await network.forward(args.remote)
        return { ok: true, remote: Number(args.remote), local, url: `http://localhost:${local}/` }
      }
      case 'unforward': remoteNetwork?.stop(args.remote); return { ok: true }
      default: throw new Error('Unknown browser action')
    }
  })
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
    stopHubShare()
    remote = null
    remoteMachine = null
    remoteError = ''
    selectedWorkspace = directory
    window.setTitle(`${basename(directory)} — ${APP_NAME}`)
    session.bind(sessionId)
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
      // Entering another folder of this host resumes its newest task; a blank
      // chat made a project with history look empty.
      // A view already open on that folder is brought forward as it is.
      const target = { ...remoteMachine, workspacePath: dir }
      if (typeof sessionId !== 'string') {
        const existing = remoteViewOf(window, target)
        if (existing !== undefined) { activateSurface(existing); return dir }
      }
      const resume = typeof sessionId === 'string' ? sessionId : latestSessionIn(cachedSessions, dir)
      openRemoteView(window, target, resume)
      return dir
    }
    if (selectedWorkspace || remote) {
      if (dir !== selectedWorkspace || remote || sessionId) openWorkspaceView(window, dir, typeof sessionId === 'string' ? sessionId : undefined)
    } else useProject(dir, typeof sessionId === 'string' ? sessionId : null)
    return dir
  })
  handle('desktop:workspaces', () => remoteMachine || saved?.remote ? [] : localDesktopWorkspaces(workspaceFile()))
  handle('desktop:remove-workspace', (_event, dir: unknown) => {
    if (typeof dir !== 'string' || !isAbsolute(dir) || /[\x00-\x1f]/.test(dir)) throw new TypeError('Invalid workspace directory')
    if (remoteMachine || saved?.remote) return []
    return removeDesktopWorkspace(workspaceFile(), dir)
  })
  handle('desktop:restore-workspace', (_event, dir: unknown) => {
    if (typeof dir !== 'string' || !isAbsolute(dir) || /[\x00-\x1f]/.test(dir)) throw new TypeError('Invalid workspace directory')
    if (remoteMachine || saved?.remote) return []
    return rememberDesktopWorkspace(workspaceFile(), dir)
  })
  handle('desktop:context-scope', () => contextScope(remoteMachine ?? saved?.remote))
  // The name the sidebar gives this view's host: the SSH alias, or null for this Mac.
  handle('desktop:context-host', () => (remoteMachine ?? saved?.remote)?.alias ?? null)
  handle('desktop:contexts', async () => {
    // Keyed by host AND folder: two projects on one SSH host are two contexts.
    const currentScope = contextKey(remoteMachine ?? saved?.remote)
    const scopes = new Set([currentScope])
    const readers: Array<Promise<WorkspaceContext>> = []
    for (const [otherId, surface] of workspaceSurfaces) {
      if (surface.host !== window) continue
      const scope = contextKey(windowStates.get(otherId)?.().remote)
      const read = contextReaders.get(otherId)
      if (scopes.has(scope) || !read) continue
      scopes.add(scope)
      readers.push(read())
    }
    const open = openContextRemotes(window)
    void savedHostContexts.refresh(open)
    return [...await Promise.all(readers), ...savedHostContexts.contexts(open)]
  })
  handle('desktop:activate-context', async (_event, contextId: unknown, sessionId?: unknown) => {
    if (typeof contextId !== 'number' || !Number.isSafeInteger(contextId) || sessionId !== undefined && (typeof sessionId !== 'string' || !/^[a-zA-Z0-9_-]{1,256}$/.test(sessionId))) throw new Error('Invalid workspace context')
    if (contextId < 0) {
      const selection = savedHostContexts.selection(contextId, sessionId)
      openRemoteView(window, selection.machine, selection.sessionId)
      return
    }
    if (workspaceSurfaces.get(contextId)?.host !== window) throw new Error('Workspace context is unavailable')
    const state = windowStates.get(contextId)?.()
    if (!state) throw new Error('Workspace context is closed')
    if (sessionId === undefined) { activateSurface(contextId); return }
    const row = (await contextReaders.get(contextId)?.())?.sessions.find(row => row.id === sessionId)
    if (!row) throw new Error('Session is no longer available in this workspace')
    if (state.remote) openRemoteView(window, { ...state.remote, workspacePath: row.cwd }, row.id)
    else openWorkspaceView(window, row.cwd, row.id)
  })
  handle('desktop:workspace', () => selectedWorkspace)
  handle('desktop:occluded', () => occludedSurfaces.has(id))
  // A bundled background as a data URL. Pixels read from a file:// image
  // taint a canvas, so the renderer could not bake its blur; only the
  // shipped backgrounds folder is readable, by exact name.
  handle('desktop:background-data', async (_event, name: unknown) => {
    if (typeof name !== 'string' || !/^backgrounds\/background-\d+\.jpg$/.test(name)) throw new TypeError('Invalid background')
    const bytes = await readFile(join(here, 'renderer', name))
    return `data:image/jpeg;base64,${bytes.toString('base64')}`
  })
  // Transparent background with blur off: remove the native material so the
  // window is clear glass; any other choice puts it back. Verified on
  // Electron 44: removing it leaves the window clear, not black.
  handle('desktop:window-blur', (_event, on: unknown) => {
    if (typeof on !== 'boolean') throw new TypeError('Invalid window blur')
    if (process.platform !== 'darwin' || window.isDestroyed()) return
    window.setVibrancy(on ? 'under-window' : null, { animationDuration: 0 })
  })
  // Every load asks, not only the first: a reload the renderer starts itself
  // (the error screen, View > Reload) must come back to this surface's own
  // session, not the workspace-wide preference another view last wrote.
  handle('desktop:resume', () => session.current)
  // Asked once at startup: a window opened for a fresh task must not resume
  // the workspace's remembered chat (that is the one still running a turn).
  let startsFresh = startFresh
  handle('desktop:starts-fresh', () => {
    const fresh = startsFresh
    startsFresh = false
    return fresh
  })
  // The preset a fresh view starts with (Creator mode opened while a turn ran), asked once.
  let freshPreset = startFresh ? startPreset ?? null : null
  handle('desktop:fresh-preset', () => {
    const preset = freshPreset
    freshPreset = null
    return preset
  })
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
      }, undefined, {
        confirmHostKey: async (target, prompt, signal) => {
          if (window.isDestroyed() || contents.isDestroyed() || signal.aborted) return false
          const result = await dialog.showMessageBox(window, {
            type: 'warning', title: 'Verify SSH server',
            message: `Trust the SSH host key for ${target}?`,
            detail: `${prompt}\n\nCompare this fingerprint with one from the server administrator or another trusted connection. Trusting it saves the key in your SSH known_hosts file. A changed key is never accepted here.`,
            buttons: ['Cancel', 'Trust and connect'], defaultId: 0, cancelId: 0,
            noLink: true, signal,
          })
          return result.response === 1 && !signal.aborted && !window.isDestroyed() && !contents.isDestroyed()
        },
      })
      const localProviders = new DaemonRpc({projectDir:homedir()})
      let forwarding: DesktopProviderForwarding
      const rpc = new DaemonRpc({ projectDir: next.projectDir, socketPath: next.socketPath,
        providerRelay: (binding, frame, signal) => forwarding.relay(binding, frame, signal),
        remoteUpdate: force => next.update(force), expectedRemoteBuildId: () => next.expectedBuildId,
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
          // The forward socket moved: relay the new one.
          if (remote === replacement && remoteMachine) startHubShare(remoteMachine.alias, replacement.socketPath)
          remoteError = ''
          await previous.close()
          return replacement.socketPath
        },
      })
      forwarding = new DesktopProviderForwarding((method, args) => localProviders.call(method,args),
        (method,args) => rpc.call(method,args), machine.target, next.projectDir,
        () => localActiveProfile(join(xerxesHome(), 'profiles.json')))
      // Access ends with the transport; the open conversation follows this
      // computer's sign-in provider again once it is back.
      // A brief drop must not end access: the runtime keeps this window's
      // bindings while it reattaches, so the grants here wait as long. Only a
      // link that stays down past that window revokes them.
      let dropped: ReturnType<typeof setTimeout> | undefined
      rpc.onConnection(online => {
        clearTimeout(dropped)
        if (!online) dropped = setTimeout(() => { void forwarding.disconnect() }, REATTACH_GRACE_MS)
        else void followLocalProviders(forwarding)
      })
      // The window shows when this conversation's prompts run on this computer.
      forwarding.onChange(() => { if (!contents.isDestroyed()) contents.send('desktop:provider-relay') })
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
      session.bind(
        (remoteMachine ?? saved?.remote)?.target === machine.target &&
        (remoteMachine ?? saved?.remote)?.workspacePath === machine.workspacePath &&
        typeof params.resume_session_id === 'string' &&
        /^[a-zA-Z0-9_-]{1,128}$/.test(params.resume_session_id)
          ? params.resume_session_id
          : null)
      const previous = remote
      closeProviderForwarding()
      providerForwarding = forwarding
      localProviderDaemon = localProviders
      daemon?.dispose()
      daemon = rpc
      attach(rpc)
      // Explicit per-host provider accounts. No profile or key is copied until
      // the person reviews and confirms one named account for this host.
      accountDirectory = new ProviderAccountDirectory(
        machine,
        join(xerxesHome(), 'profiles.json'),
        (method, args) => rpc.call(method, args),
        (profile, signal) => spawnSecretWriter(machine, profile, signal),
      )
      remote = next
      remoteMachine = machine
      startHubShare(machine.alias, next.socketPath)
      selectedWorkspace = next.projectDir
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
          resume_session_id: session.current,
          error: remoteError,
        }
      if (action === 'cancel') {
        remoteAttempt?.abort()
        return { ok: true }
      }
      if (action === 'provider-use-local') {
        if (!remote || !daemon?.online || !providerForwarding) throw new Error('Reconnect the SSH workspace before choosing a provider on this computer.')
        return providerForwarding.useLocal(params.profile)
      }
      if (action === 'provider-use-host') {
        if (!remote || !daemon?.online || !providerForwarding) throw new Error('Reconnect the SSH workspace before choosing a provider on the host.')
        return providerForwarding.useHost(params.profile)
      }
      if (action === 'provider-activity') {
        if (!remote || !daemon?.online || !providerForwarding) return { ok: true, bound: false }
        return providerForwarding.current()
      }
      if (action === 'provider-review' || action === 'provider-share' || action === 'provider-revoke') {
        if (!remote || !daemon?.online || !providerForwarding) throw new Error('Reconnect the SSH workspace before reviewing local provider access.')
        if (action === 'provider-review') return providerForwarding.inspect()
        if (action === 'provider-share') return providerForwarding.authorize(params)
        return providerForwarding.revoke(params.sessionKey)
      }
      if (action === 'provider-accounts' || action === 'provider-bind-review' || action === 'provider-bind' || action === 'provider-activate' || action === 'provider-quota') {
        if (!remote || !daemon?.online || !accountDirectory) throw new Error('Reconnect the SSH workspace before managing provider accounts.')
        if (action === 'provider-accounts') {
          const accounts = await accountDirectory.list()
          try { return { ...accounts, provider_setup: await ensureRemoteProviderClis(remoteMachine!, new AbortController().signal) } }
          catch (error) { return { ...accounts, provider_setup_error: error instanceof Error ? error.message : String(error) } }
        }
        if (action === 'provider-bind-review') return accountDirectory.reviewBind(String(params.profile ?? ''))
        if (action === 'provider-bind') return accountDirectory.confirmBind(String(params.review ?? ''), params.confirm === true, new AbortController().signal)
        if (action === 'provider-activate') return accountDirectory.activate(String(params.sessionKey ?? ''))
        return accountDirectory.quota()
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
      } else if (action === 'remove') {
        command = 'remove ' + argument(params.alias)
        const result = await desktopMachineCommand(join(xerxesHome(), 'machines.json'), command)
        if (typeof params.alias === 'string' && params.alias) savedHostContexts.dropAlias(params.alias)
        return result
      } else throw new Error('Unknown remote action')
      return desktopMachineCommand(join(xerxesHome(), 'machines.json'), command)
    },
  )

  handle('desktop:accounts', async (event, action: unknown, params: Record<string, unknown>) => {
    if (typeof action !== 'string') throw new TypeError('account action is required')
    const accounts = accountCenter()
    if (action === 'list') return accounts.list()
    if (action === 'create-gemini') {
      if (typeof params.label !== 'string' || typeof params.apiKey !== 'string' || typeof params.model !== 'string') throw new Error('Gemini account label, API key and model id are required')
      return accounts.createGemini(params.label, params.apiKey, params.model)
    }
    if (action === 'login') {
      const kind = params.kind
      if (kind !== 'claude' && kind !== 'codex') throw new Error('Choose Claude or Codex')
      const loginId = String(params.loginId ?? '')
      if (!/^[a-f0-9-]{36}$/.test(loginId)) throw new Error('A login id is required')
      return accounts.login(kind, String(params.label ?? ''), url => {
        if (!event.sender.isDestroyed()) event.sender.send('desktop:account-login', { type: 'authorization_url', login_id: loginId, url })
      })
    }
    if (action === 'relogin') {
      const loginId = String(params.loginId ?? '')
      if (!/^[a-f0-9-]{36}$/.test(loginId)) throw new Error('A login id is required')
      return accounts.relogin(String(params.accountId ?? ''), url => {
        if (!event.sender.isDestroyed()) event.sender.send('desktop:account-login', { type: 'authorization_url', login_id: loginId, url })
      })
    }
    if (action === 'copy-login-link') {
      const url = String(params.url ?? '')
      if (new URL(url).protocol !== 'https:') throw new Error('Only an HTTPS sign-in link can be copied')
      clipboard.writeText(url)
      return { ok: true }
    }
    if (action === 'activate') return accounts.activate(String(params.accountId ?? ''), String(params.sessionKey ?? ''))
    if (action === 'usage') return accounts.usage(String(params.accountId ?? ''))
    if (action === 'sync-review') return accounts.reviewSync(String(params.accountId ?? ''))
    if (action === 'sync') return accounts.confirmSync(String(params.review ?? ''), params.confirm === true)
    if (action === 'remove') return accounts.remove(String(params.accountId ?? ''))
    throw new Error('Unknown provider account action')
  })

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
  // Chat links: only http, https and mailto reach the system browser.
  handle(OPEN_EXTERNAL_CHANNEL, async (_event, value: unknown) => {
    const url = externalUrlOf(value)
    if (!url) throw new TypeError('Only http, https and mailto links can be opened')
    await shell.openExternal(url)
    return true
  })
  // Chat images as data URLs (the page's CSP has no file: or web origin).
  // An SSH workspace's files are read by its runtime, never from this disk.
  handle(IMAGE_DATA_CHANNEL, async (_event, value: unknown) => {
    const request = chatImageRequestOf(value)
    if ('error' in request) return { ok: false, error: request.error }
    if (remoteMachine) {
      const rpc = daemon
      if (!rpc) return { ok: false, error: 'the SSH workspace is not connected' }
      return readRemoteChatImage(request, (method, params) => rpc.call(method, params))
    }
    return readLocalChatImage(request.path, request.cwd ?? selectedWorkspace)
  })
  // Real project logo / favicon discovery for local workspaces (T3 Code parity).
  handle(PROJECT_ICON_CHANNEL, async (_event, cwd: unknown) => {
    if (typeof cwd !== 'string' || !cwd || !isAbsolute(cwd) || /[\x00-\x1f\x7f]/.test(cwd)) return null
    return resolveProjectIconDataUrl(cwd, bytes => {
      const image = nativeImage.createFromBuffer(bytes)
      if (image.isEmpty()) return null
      return image.resize({ width: 64, quality: 'good' }).toPNG()
    })
  })
  // Files panel: open a workspace file in its default app, or show it in Finder.
  handle(FILE_ACTION_CHANNEL, async (_event, value: unknown) => {
    const request = fileActionRequestOf(value)
    if ('error' in request) return { ok: false, error: request.error }
    if (remoteMachine) return { ok: false, error: 'This file is on the SSH host, so it cannot be opened on this computer.' }
    return performFileAction(request, { openPath: path => shell.openPath(path), showItemInFolder: path => shell.showItemInFolder(path) })
  })
  handle('desktop:new-window', async (_event, directory?: unknown, sessionId?: unknown, fresh?: unknown, existingOnly?: unknown, agentPreset?: unknown) => {
    if (sessionId !== undefined && (typeof sessionId !== 'string' || !sessionId || sessionId.length > 256 || /[\x00-\x1f]/.test(sessionId))) throw new TypeError('Invalid session identity')
    if (fresh !== undefined && typeof fresh !== 'boolean') throw new TypeError('Invalid new-window option')
    if (agentPreset !== undefined && (typeof agentPreset !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$/.test(agentPreset) || fresh !== true)) throw new TypeError('Invalid agent preset')
    const preset = typeof agentPreset === 'string' ? agentPreset : undefined
    if (fresh === true && sessionId !== undefined) throw new TypeError('A fresh task window cannot also resume a session')
    if (directory !== undefined && (typeof directory !== 'string' || !isAbsolute(directory) || /[\x00-\x1f]/.test(directory))) throw new TypeError('Invalid workspace directory')
    if (existingOnly === true) {
      // Only bring forward a view that already shows this session; the caller
      // attaches in place when there is none (no new view, connection or reload).
      if (!remoteMachine || typeof directory !== 'string' || directory !== selectedWorkspace || typeof sessionId !== 'string') return null
      const existing = remoteViewOf(window, { ...remoteMachine, workspacePath: directory }, sessionId)
      if (existing === undefined || existing === id) return null
      activateSurface(existing)
      return directory
    }
    const picked = typeof directory === 'string' ? directory : await pickWorkspace()
    if (!picked || window.isDestroyed()) return null
    if (remoteMachine && picked === selectedWorkspace && typeof sessionId === 'string' && fresh !== true) {
      // Opening a task of this SSH workspace while another one runs: bring
      // forward a view that already shows it. Building a new view each time
      // meant a new SSH connection and a full reload on every click, and the
      // views piled up on the running task.
      openRemoteView(window, { ...remoteMachine, workspacePath: picked }, sessionId)
    } else if (remoteMachine && picked === selectedWorkspace) {
      const savedWindow = windowStates.get(id)?.()
      if (!savedWindow) throw new Error('Remote workspace is not ready')
      createWorkspaceWindow(null, { ...savedWindow, sessionId: typeof sessionId === 'string' ? sessionId : null }, undefined, window, fresh === true, preset)
    } else if (sessionId) openWorkspaceView(window, picked, sessionId as string)
    // A fresh task opens as another view in THIS window (like opening a
    // second chat mid-turn does), never as a new OS window; the running task
    // keeps its own view.
    else if (fresh === true) createWorkspaceWindow(picked, undefined, undefined, window, true, preset)
    else createWorkspaceWindow(picked)
    return picked
  })
  const layoutEvents = ['move', 'resize', 'maximize', 'unmaximize', 'enter-full-screen', 'leave-full-screen'] as const
  const onHostHide = () => { hiddenWindows.add(window); syncWindowVisibility(window) }
  const onHostShow = () => { hiddenWindows.delete(window); syncWindowVisibility(window) }
  const onHostFocus = () => {
    if (hiddenWindows.delete(window)) syncWindowVisibility(window)
    syncCacheTrimState()
  }
  const onHostBlur = () => syncCacheTrimState()
  let cleaned = false
  const cleanup = () => {
    if (cleaned) return
    cleaned = true
    occludedSurfaces.delete(id)
    contextReaders.delete(id)
    surfaceSessions.delete(id)
    viewClosers.delete(id)
    workspaceSurfaces.delete(id)
    if (activeSurfaces.get(window.id) === id) activeSurfaces.delete(window.id)
    if (view && !window.isDestroyed()) {
      // A view closed on its own leaves a live window behind.
      window.contentView.removeChildView(view)
    } else if (!view && !window.isDestroyed()) {
      hiddenWindows.delete(window)
      for (const event of layoutEvents) window.removeListener(event as 'move', scheduleWindowSave)
      window.removeListener('minimize', onHostHide)
      window.removeListener('hide', onHostHide)
      window.removeListener('restore', onHostShow)
      window.removeListener('show', onHostShow)
      window.removeListener('focus', onHostFocus)
      window.removeListener('blur', onHostBlur)
    }
    if (view && !contents.isDestroyed()) contents.close()
    windowStates.delete(id)
    if (!quitting) scheduleWindowSave()
    windowRoutes.remove(id)
    detachDaemon(id)
    closeProviderForwarding()
    browserTab?.close()
    browserTab = null
    daemon?.dispose()
    remoteAttempt?.abort()
    voiceRequest?.abort()
    void remote?.close()
    stopHubShare()
    cleanupWindows.delete(cleanup)
    syncCacheTrimState()
  }
  cleanupWindows.add(cleanup)
  window.once('closed', cleanup)
  if (view) viewClosers.set(id, () => {
    window.removeListener('closed', cleanup)
    cleanup()
  })
  // Snapshot while the window and its views still exist: the 'closed'
  // teardown below empties the layout before the quit that follows the last
  // window closing, and saving that wiped every window from the next launch.
  else window.once('close', () => windowStates.closing(String(window.id)))
  windowStates.set(id, () => ({
    workspace: remoteMachine || saved?.remote && !selectedWorkspace ? null : selectedWorkspace,
    remote: remoteMachine ?? (selectedWorkspace ? null : saved?.remote ?? null),
    windowGroup: String(window.id), active: activeSurfaces.get(window.id) === id,
    sessionId: session.current, bounds: window.getNormalBounds(), maximized: window.isMaximized(), fullscreen: window.isFullScreen(),
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
    if (!machine) return { id, label: 'This Mac', workspace: selectedWorkspace ?? '', remote: false, sessions: cachedSessions }
    // The host's daemon lists every project on it; keep this folder's tasks
    // when another folder of the same host is saved or open.
    const folders = savedHostContexts.folders(machine.target, openContextRemotes())
    return { id, label: contextLabel(machine.alias, machine.workspacePath),
      workspace: selectedWorkspace ?? machine.workspacePath, workspacePath: machine.workspacePath, alias: machine.alias, remote: true,
      sessions: sessionsFor(cachedSessions, machine.workspacePath, folders) }
  })
  if (!view) {
    // Attach host-level layout & visibility listeners once per BrowserWindow, not once per child view.
    for (const event of layoutEvents) window.on(event as 'move', scheduleWindowSave)
    window.on('minimize', onHostHide)
    window.on('hide', onHostHide)
    window.on('restore', onHostShow)
    window.on('show', onHostShow)
    window.on('focus', onHostFocus)
    window.on('blur', onHostBlur)
  }
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
  // A Claude Code / Codex profile cannot run on this computer without its CLI,
  // and the desktop never inherited ~/.bun/bin. Install only what is missing,
  // in the background: a failure here must not stop the app from opening.
  void ensureLocalProviderClis({ onProgress: message => console.log(message) })
    .then(status => console.log(`Provider CLIs ready: claude=${status.claude} codex=${status.codex}`))
    .catch(error => console.warn('Provider CLI setup failed:', error instanceof Error ? error.message : error))
  // Offers new releases from GitHub; installs only when the person says so.
  const appUpdates = startAppUpdates({
    settingsDirectory: xerxesHome(),
    version: app.getVersion(),
    shouldRunBackgroundCheck: isAppInteractive,
    targets: workspaceWebContents,
  })
  // Hub cockpit: link this Mac, pair a phone and sign in to the XERXES tab.
  // The CLI, the Hub cookies and the Hub socket stay in this process.
  const hubCockpit = createHubCockpit({
    runner: createCliRunner(process.env, homedir()),
    http: nodeHubHttp,
    connect: (target, handlers) => connectHubSocket(target, handlers),
    // The asynchronous keychain API only: the synchronous one waits on the
    // main thread, and an unanswered keychain prompt kept the app from ever
    // opening a window.
    storage: createHubStorage(xerxesHome(), {
      available: () => safeStorage.isAsyncEncryptionAvailable(),
      encrypt: text => safeStorage.encryptStringAsync(text),
      decrypt: async bytes => {
        const opened = await safeStorage.decryptStringAsync(Buffer.from(bytes))
        return { text: opened.result, stale: opened.shouldReEncrypt }
      },
    }),
    emit: (senderId, channel, payload) => {
      const contents = webContents.fromId(senderId)
      if (contents && !contents.isDestroyed()) contents.send(channel, payload)
    },
    log: message => console.warn(message),
  })
  registerHubCockpitIpc(ipcMain, hubCockpit, {
    onDestroyed: (senderId, callback) => {
      const contents = webContents.fromId(senderId)
      if (contents) contents.once('destroyed', callback)
      else callback()
    },
  })
  app.on('will-quit', () => { appUpdates.dispose(); hubCockpit.dispose(); cacheTrimmer.dispose() })
  // macOS privacy permissions for computer use and dictation. Read silently at
  // start (no prompt), again when a window regains focus (throttled, so coming
  // back from System Settings updates the notice), and pushed to every workspace page.
  const macPermissions = createMacPermissions(electronMacPermissionsPort({ settingsDirectory: xerxesHome(), appName: APP_NAME }))
  registerMacPermissionsIpc(ipcMain, macPermissions)
  macPermissions.onChange(state => {
    for (const contents of workspaceWebContents()) {
      contents.send(MAC_PERMISSIONS_STATE_EVENT, state)
    }
  })
  if (process.platform === 'darwin') {
    macPermissions.check()
    app.on('browser-window-focus', () => { if (!systemSuspendedOrLocked) macPermissions.recheckOnFocus() })
  }
  // Hub health: a notice when no Hub is set up or the saved one does not
  // answer its /healthz. Checked at start and again when a window regains focus.
  const hubHealth = createHubHealth(createNodeHubHealthPort({ home: xerxesHome() }))
  registerHubHealthIpc(ipcMain, hubHealth)
  hubHealth.onChange(state => {
    for (const contents of workspaceWebContents()) {
      contents.send(HUB_HEALTH_STATE_EVENT, state)
    }
  })
  void hubHealth.check()
  app.on('browser-window-focus', () => {
    if (systemSuspendedOrLocked) return
    hubHealth.recheckOnFocus()
    void savedHostContexts.refresh(openContextRemotes())
  })
  // Track macOS app hide/show and OS suspend/lock transitions so renderers,
  // BrowserTabs, and background pollers pause cleanly while unseen.
  const onSystemPause = () => {
    systemSuspendedOrLocked = true
    syncAllWindowVisibility()
    for (const surface of workspaceSurfaces.values()) surface.suspendBrowser()
    cacheTrimmer.triggerNow()
  }
  const onSystemResume = () => {
    systemSuspendedOrLocked = false
    syncAllWindowVisibility()
  }
  powerMonitor.on('suspend', onSystemPause)
  powerMonitor.on('lock-screen', onSystemPause)
  powerMonitor.on('resume', onSystemResume)
  powerMonitor.on('unlock-screen', onSystemResume)
  app.on('did-become-active', () => {
    for (const win of BrowserWindow.getAllWindows()) {
      if (!win.isDestroyed() && !win.isMinimized() && win.isVisible()) hiddenWindows.delete(win)
    }
    syncAllWindowVisibility()
  })
  app.on('did-resign-active', () => syncCacheTrimState())
  // The dock/taskbar icon is the phoenix mark; on macOS the running app's
  // dock tile only changes through app.dock.
  const icon = appIcon()
  if (icon && process.platform === 'darwin') app.dock?.setIcon(icon)

  // Commands the menu can only express by asking the focused renderer:
  // the shell owns sessions, the palette and the find bar.
  const toFocused = (command: string) => () => {
    const win = BrowserWindow.getFocusedWindow()
    if (!win) return
    const activeId = activeSurfaces.get(win.id)
    const activeSurface = activeId === undefined ? undefined : workspaceSurfaces.get(activeId)
    const target = activeSurface?.view?.webContents ?? win.webContents
    if (!target.isDestroyed()) target.send('desktop:menu', command)
  }
  // `role: 'appMenu'` ships no Preferences item, so ⌘, was the one binding
  // the app honoured and never advertised; there was also no Help menu and
  // no Find, in an app whose main surface is thousands of lines of output.
  const appMenu = {
    label: APP_NAME,
    submenu: [
      { role: 'about' as const },
      { label: 'Check for Updates…', click: () => { void appUpdates.check() } },
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
      { type: 'separator' }, { label: 'Close', accelerator: 'CmdOrCtrl+W', click: closeActiveSurface },
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
    // Materialize only the single active surface per OS window group on startup.
    // Inactive accumulated views stay out of RAM (avoiding N Chromium renderer
    // processes & N concurrent daemon session hydrations); every session remains
    // one click away in the sidebar and saved SSH hosts list.
    const plans = planStartupSurfaces(saved)
    for (const { primary } of plans) {
      const win = createWorkspaceWindow(primary.remote ? null : primary.workspace, primary)
      const activeId = activeSurfaces.get(win.id)
      if (activeId !== undefined) activateSurface(activeId)
    }
  }
  else createWorkspaceWindow(loadWorkspace())
  // Probe saved endpoints without opening views or running setup. Results are
  // cached globally and each host has its own timeout and single-flight guard.
  void savedHostContexts.refresh(openContextRemotes())
  const savedHostTimer = setInterval(() => {
    if (!hasFocusedWindow()) return
    void savedHostContexts.refresh(openContextRemotes())
  }, 120_000)
  savedHostTimer.unref()
  const idleViewTimer = setInterval(() => {
    if (!systemSuspendedOrLocked) evictBackgroundViews()
  }, 120_000)
  idleViewTimer.unref()
  app.once('will-quit', () => {
    clearInterval(savedHostTimer)
    clearInterval(idleViewTimer)
    powerMonitor.removeListener('suspend', onSystemPause)
    powerMonitor.removeListener('lock-screen', onSystemPause)
    powerMonitor.removeListener('resume', onSystemResume)
    powerMonitor.removeListener('unlock-screen', onSystemResume)
    savedHostContexts.dispose()
  })
  app.on('activate', () => { if (BrowserWindow.getAllWindows().length === 0) createWorkspaceWindow(loadWorkspace()) })
})
app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit() })
app.on('before-quit', () => { persistWindows(); quitting = true })
app.on('will-quit', () => { for (const cleanup of [...cleanupWindows]) cleanup() })
