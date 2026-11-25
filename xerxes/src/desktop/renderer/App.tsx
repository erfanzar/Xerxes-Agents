// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { GoalInspectorDisclosure } from "./GoalInspector.js"
import { createPortal } from 'react-dom'
import { Fragment, createContext, memo, useCallback, useContext, useLayoutEffect, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactElement } from 'react'

import { CommandPalette, PickerLayer, DelegationMenu, ModelMenu, ModelPicker, ReasoningPicker, SettingsModal, bareModelName } from './Overlays.js'
import { SessionSearch } from './SearchPanel.js'
import { captureHistoryScrollAnchor, restoreHistoryScrollAnchor, useTranscriptScroll, type HistoryScrollAnchor } from './transcriptScroll.js'
import { resolveExternalAcpCatalog, showsImmortalsForSession, store, type CompletionHintItem, type ComposerImage, type Snapshot, isPlanReview } from './store.js'
import { connectionFailureKind } from './connectionFailure.js'
import { failureView } from './turnFailure.js'
import type { AgentMember, ApprovalResponse, Block } from './types.js'
import { applyCompletion, wantsHints } from './hints.js'
import { SIDEBAR_SECTION_ROWS, groupByWorkspace, hostOfScope, limitRows, loadCollapsedSections, saveCollapsedSections, sidebarOrder, sidebarSections } from './workspaceGroups.js'
import { CopyButton } from './CopyButton.js'
import { currentSessionDensity, subscribeSessionDensity } from './displayPrefs.js'
import { ChangesTab, LogTab, PlanTab } from './Workspaces.js'
import { Markdown } from './markdown.js'
import { WidgetOpenArtifactContext, WidgetSessionContext } from './WidgetFrame.js'
import { CodeRunContext } from './codeRun.js'
import { Dictation } from './Dictation.js'
import { draftKey, readDraft, transitionDraft, writeDraft, acceptedDraft } from './drafts.js'
import { PANEL_SIZES, PanelDivider, panelWidthOf, usePanelLayout, vibesPanelLayout } from './layout.js'
import { handleDialogEscape, useDialogFocus } from './dialogFocus.js'
import { keyedActivityGroups, isDisclosedActivity } from "./activityGroups.js"
import { approvalTitle, thinkingHeadingOf } from './activityPhrase.js'
import { ToolCallRow } from "./Execution.js"
import { lineDiff } from './FileToolRow.js'
import { trimmedChangedLines } from './blocks.js'
import { ToolTimelineGroup } from './ToolTimeline.js'
import { ArtifactToast, latestArtifactId } from './ArtifactCard.js'
import {
  AcpActiveBanner,
  AcpExtraConfigPickers,
  AcpModePicker,
  AcpModelPicker,
  AcpRuntimePicker,
  AcpThinkingPicker,
  activeAcpModelLabel,
  activeAcpSummaryParts,
  isExternalAcpBinding,
  transportBadgeLabel,
} from './AcpRuntimePicker.js'
import { RuntimeBrandIcon, XerxesLogoMark } from './BrandMark.js'
import { closeTab, openTab, panelOfKey, retitleTab, tabKey, type RailTab } from './railTabs.js'
import { activityFleetRows, agentState, agentKindLabel } from './AgentRoster.js'
import { RailAgents, RailFiles, RailWorkflows } from './RailLists.js'
import { AgentDetailsCard, AgentInspector } from './AgentInspector.js'
import { AgentMenu } from './AgentSwitcher.js'
import { OutputViewer } from './OutputViewer.js'
import { Icon } from './Icon.js'
import { RelayBadge } from './RelayBadge.js'
import { isVscodeHost } from './host.js'
import { pageIsInactive } from './pageVisibility.js'
import { RailStatus, currentActionOf, orbStateOf } from './RailStatus.js'
import { AgentOrb } from './AgentOrb.js'
import { AgentsCard } from './AgentsCard.js'
import { SupervisorPage } from './SupervisorPage.js'
import { SectionsMenu } from './SectionsMenu.js'
import { parseGoal } from './goalText.js'
import { AppUpdatePrompt } from './AppUpdatePrompt.js'
import { MacPermissionsNotice } from './MacPermissionsPanel.js'
import { HubHealthNotice } from './HubHealthNotice.js'
// Re-exported: it moved into its own module so the rail's diagnostics
// drawer (DesktopPanels) can import it without a cycle through App.
export { SessionDiagnostics } from './SessionDiagnostics.js'
import { ErrorBoundary } from './ErrorBoundary.js'
import { FindBar } from './FindBar.js'
import { ElapsedClock } from './ElapsedClock.js'
import { Shortcuts } from './Shortcuts.js'
import { FirstRunSetup } from './Setup.js'
import { ModelSplitChip } from './TaskModels.js'
import { RemoteWorkspaceGate } from './RemoteWorkspaceGate.js'
import { DesktopNavigation, DesktopSheet, DesktopPage, DesktopRail, PAGE_TITLES, RAIL_PANELS, useDesktopNavigation, useVibesStyle, workspaceFilePath, type DesktopPageName, type DesktopPanel, type RailPanel } from './DesktopPanels.js'
import { desktopCall } from './desktopRpc.js'
import { ConnectedReflectionCard, LearnedBadge } from './LearnedPanel.js'
import {
  ANTIGRAVITY_PULL_REQUEST_PROMPT,
  isAntigravityAcpSession,
  prefillComposerPrompt,
  pullRequestPrompt,
} from './GitPanel.js'
import { elapsedOf } from './duration.js'
import { swarmMemberCounts } from './swarmState.js'
import type { ScmStatus } from '../../workspace/gitScm.js'
import { T3Sidebar, type T3SidebarRow } from './T3Sidebar.js'
import { ThreadDetailsCard } from './ThreadDetailsCard.js'
import { TerminalDrawer, loadTerminalDrawerHeight } from './TerminalDrawer.js'
import { RightPanelLauncher } from './RightPanelLauncher.js'
import { T3ModelPicker } from './T3ModelPicker.js'
import { ProjectMonogram } from './ProjectMonogram.js'
import { T3Icon } from './T3Icons.js'
import {
  isSnoozedActive,
  setSettledInState,
  setSnoozedInState,
  setThreadInboxState,
  snoozePresets,
  togglePinInState,
  useThreadInbox,
} from './threadInbox.js'

const ActivityVisible = createContext(false)

/** Every palette read goes through the generated custom properties. */
const C = {
  working: 'var(--x-working)',
  activity: 'var(--x-activity)',
  done: 'var(--x-done)',
  failed: 'var(--x-failed)',
  needs: 'var(--x-needs)',
} as const

function statusColor(status: string): string {
  if (status === 'working' || status === 'running') return C.activity
  if (status === 'failed' || status === 'error') return C.failed
  if (status === 'waiting') return C.needs
  return C.done
}

function workspaceLabel(cwd: string): string {
  const base = cwd.replaceAll('\\', '/').split('/').filter(Boolean).at(-1)
  return base || 'workspace'
}

/**
 * One persistent polite live region for the whole shell.
 *
 * Every status in this renderer was a conditionally-mounted `role="status"`
 * node that arrives already containing its text — assistive tech announces
 * changes to a live region, not the insertion of one, so turn completion,
 * failure and a pending approval were announced by nothing at all. This
 * node is always mounted and only its text changes.
 */
function Announcer({ snap }: { snap: Snapshot }): ReactElement {
  const message = snap.approval
    ? `Approval required for ${snap.approval.toolName || 'a tool call'}`
    : snap.question ? 'The agent is asking a question'
    : snap.turnActive ? 'Working'
    : snap.submissionPending ? 'Starting'
    : snap.failed ? `Turn failed: ${snap.failed.error}`
    : snap.connection === 'offline' ? 'Disconnected from the workspace runtime'
    : snap.turnCount > 0 ? 'Finished' : ''
  return <p className="sr-only" role="status" aria-live="polite" aria-atomic="true">{message}</p>
}

/** Names the rail in a boundary fallback: "Files could not be shown". */
function railLabel(rail: RailPanel): string {
  return rail === 'subagents' ? 'Subagents' : rail === 'swarm' ? 'Immortals Board' : rail === 'btw' ? 'Side question' : rail === 'docs' ? 'Artifacts' : rail === 'files' ? 'Project files' : rail === 'outputs' ? 'Outputs' : rail === 'review' ? 'Source control' : rail === 'terminal' ? 'Terminal' : rail === 'background' ? 'Background' : rail === 'browser' ? 'Browser' : rail === 'usage' ? 'Usage' : 'Activity'
}

/** Blinking tail shown on streaming blocks. */
const CARET = <span className="caret" />

/** Live thinking preview: latest non-empty line, clipped from its left. */
function thinkingTailOf(text: string, max = 110): string {
  const clean = (text.split('\n').findLast(line => line.trim()) ?? '').trim()
  return clean.length > max ? `…${clean.slice(-(max - 1))}` : clean
}

/** Tool verb → row label: `exec_command` → `Exec command`, `read` → `Read`. */
function toolLabelOf(verb: string): string {
  const spaced = verb.replace(/[_-]+/g, ' ').trim()
  return spaced ? spaced[0]!.toUpperCase() + spaced.slice(1) : ''
}

/** Compact turn clock for the feed status line: 43 → "43s", 255 → "4m 15s". */
const turnDurOf = (seconds: number): string => elapsedOf(seconds)

/** The composer's one-line status: the running tool, else what the model is doing. */
function composerPhraseOf(snap: Snapshot): string {
  if (snap.networkRetrying) return 'Retrying connection…'
  if (snap.approval || snap.question) return 'Waiting for you'
  const liveFocus = snap.liveFocus?.trim()
  if (liveFocus) return liveFocus
  const action = currentActionOf(snap)?.verb
  if (action) return action
  // While the model reasons, its latest **Heading** says what it is working on.
  const last = snap.blocks.at(-1)
  const heading = last?.kind === 'thinking' && last.streaming ? thinkingHeadingOf(last.text) : undefined
  return heading ? `${heading}…` : 'Thinking…'
}

function ttftOf(milliseconds: number): string {
  return milliseconds < 1_000 ? `${Math.round(milliseconds)}ms` : `${(milliseconds / 1_000).toFixed(1)}s`
}


function compactTokensOf(tokens: number): string {
  if (tokens < 1_000) return String(Math.round(tokens))
  if (tokens < 1_000_000) return `${(tokens / 1_000).toFixed(tokens < 10_000 ? 1 : 0)}K`
  return `${(tokens / 1_000_000).toFixed(1)}M`
}

export { parseGoal }

/** Width of the centered conversation column plus minimum symmetric side gutter for inline ThreadDetailsCard. */
export const CONVERSATION_COLUMN_WIDTH = 728
export const THREAD_DETAILS_MIN_GUTTER = 276

/**
 * True only when the window has enough real gutter room on BOTH sides of the
 * centered 728px conversation column to float `ThreadDetailsCard` inline
 * without overlapping chat prose, and no rail/launcher/sub-agent view is open.
 */
export function isDetailsInlineEligible(params: {
  readonly vibesStyle: boolean
  readonly isVscode: boolean
  readonly noWorkspace: boolean
  readonly railOpen: boolean
  readonly railLauncherOpen: boolean
  readonly agentOpen: boolean
  readonly focused: boolean
  readonly sidebarWidth: number
  readonly windowWidth: number
}): boolean {
  if (!params.vibesStyle || params.isVscode || params.noWorkspace || params.railOpen || params.railLauncherOpen || params.agentOpen) {
    return false
  }
  const requiredWidth = (params.focused ? 0 : params.sidebarWidth) + CONVERSATION_COLUMN_WIDTH + 2 * THREAD_DETAILS_MIN_GUTTER
  return params.windowWidth >= requiredWidth
}

/** Prune workspace-scoped `files:*` tabs on cwd change and session-scoped `docs:*` tabs on thread change. */
export function pruneRailTabsForContext(
  tabs: readonly RailTab[],
  activeKey: string | null | undefined,
  opts: { readonly cwdChanged: boolean; readonly sessionChanged: boolean },
): { readonly tabs: readonly RailTab[]; readonly activeKey: string | null | undefined } {
  const isStaleTab = (tab: Pick<RailTab, 'panel' | 'item'>): boolean =>
    (opts.cwdChanged && tab.panel === 'files' && Boolean(tab.item)) ||
    (opts.sessionChanged && tab.panel === 'docs' && Boolean(tab.item))
  const isStaleKey = (key: string | null | undefined): boolean =>
    Boolean(
      key &&
      ((opts.cwdChanged && key.startsWith('files:')) ||
        (opts.sessionChanged && key.startsWith('docs:'))),
    )
  const nextTabs = tabs.filter(tab => !isStaleTab(tab))
  const activeDropped = isStaleKey(activeKey)
  if (nextTabs.length === tabs.length && !activeDropped) {
    return { tabs, activeKey }
  }
  const nextActive = activeDropped
    ? (nextTabs.length > 0 ? tabKey(nextTabs[nextTabs.length - 1]!) : null)
    : activeKey
  return { tabs: nextTabs, activeKey: nextActive }
}

export interface NotificationToastItem {
  readonly id: string
  readonly title: string
  readonly body: string
  readonly kind?: 'artifact' | 'attention' | 'finished' | 'stopped' | 'failed'
  readonly surfaceId?: number
  readonly target?: { readonly sessionId?: string; readonly panel?: 'docs'; readonly artifactId?: string }
}

function toastIconName(kind: NotificationToastItem['kind']): 'file' | 'warning' | 'check' | 'stop' | 'error' {
  if (kind === 'artifact') return 'file'
  if (kind === 'attention') return 'warning'
  if (kind === 'failed') return 'error'
  if (kind === 'stopped') return 'stop'
  return 'check'
}

export function NotificationToastStack({
  toasts,
  onActivate,
  onDismiss,
  onPause,
  onResume,
}: {
  toasts: readonly NotificationToastItem[]
  onActivate: (toast: NotificationToastItem) => void
  onDismiss: (id: string) => void
  onPause?: (id: string) => void
  onResume?: (toast: NotificationToastItem) => void
}): ReactElement | null {
  if (toasts.length === 0) return null
  return (
    <div className="notify-stack" role="region" aria-label="Notifications" aria-live="polite">
      {toasts.map(toast => {
        const clickable = Boolean(toast.target?.sessionId || toast.target?.artifactId || toast.surfaceId !== undefined)
        const actionLabel = toast.kind === 'artifact' ? 'Open' : clickable ? 'View' : ''
        return (
          <div
            key={toast.id}
            className="notify-toast"
            data-kind={toast.kind ?? 'finished'}
            role={toast.kind === 'failed' || toast.kind === 'attention' ? 'alert' : 'status'}
            onMouseEnter={() => onPause?.(toast.id)}
            onMouseLeave={() => onResume?.(toast)}
          >
            <button
              type="button"
              className="notify-toast__main"
              disabled={!clickable}
              onClick={() => onActivate(toast)}
            >
              <span className="notify-toast__icon" aria-hidden="true">
                <Icon name={toastIconName(toast.kind)} size={14} />
              </span>
              <span className="notify-toast__copy">
                <strong className="notify-toast__title">{toast.title}</strong>
                {toast.body && <span className="notify-toast__body">{toast.body}</span>}
              </span>
              {actionLabel && (
                <span className="notify-toast__action" aria-hidden="true">
                  {actionLabel} →
                </span>
              )}
            </button>
            <button
              type="button"
              className="notify-toast__close"
              aria-label={`Dismiss ${toast.title}`}
              onClick={() => onDismiss(toast.id)}
            >
              <Icon name="close" size={12} />
            </button>
          </div>
        )
      })}
    </div>
  )
}

export function App(): ReactElement {
  const snap = useSyncExternalStore(store.subscribe, store.getSnapshot)

  useEffect(() => {
    store.start()
  }, [])

  return <Shell snap={snap} />
}

/** Presentational shell — pure over the snapshot, SSR-friendly. */
export function Shell({ snap }: { snap: Snapshot }): ReactElement {
  const [trafficLights, setTrafficLights] = useState(true)
  useEffect(() => {
    let current = true
    let received = false
    const update = (state: { trafficLights: boolean }) => { if (current) setTrafficLights(state.trafficLights) }
    const unsubscribe = window.xerxes.onWindowChrome?.(state => { received = true; update(state) })
    void window.xerxes.getWindowChrome?.().then(state => { if (!received) update(state) }).catch(() => {})
    try {
      const enabled = (typeof localStorage === 'undefined' ? null : localStorage.getItem('xerxes.notifications')) !== '0'
      void window.xerxes.setNotifications?.(enabled)
    } catch {
      // Storage unavailable; keep main's default enabled state.
    }
    return () => { current = false; unsubscribe?.() }
  }, [])
  const [panel, setPanel] = useState<DesktopPanel>(null)
  const { layout: storedLayout, setLayout } = usePanelLayout()
  const vibesStyle = useVibesStyle()
  const layout = vibesStyle ? vibesPanelLayout(storedLayout) : storedLayout
  const inbox = useThreadInbox()
  // null = auto (open inline in the top-right gutter on wide windows when right panel is closed).
  const [detailsChoice, setDetailsChoice] = useState<boolean | null>(null)
  const [terminalDrawerOpen, setTerminalDrawerOpen] = useState(false)
  const [terminalDrawerHeight, setTerminalDrawerHeight] = useState<number>(loadTerminalDrawerHeight)
  const lastRailRef = useRef<RailPanel | null>(null)
  const [railLauncherOpen, setRailLauncherOpen] = useState(false)
  const [narrowNavigation, setNarrowNavigation] = useState(false)
  const [windowWidth, setWindowWidth] = useState(() => typeof window === 'undefined' ? 1440 : window.innerWidth)
  useEffect(() => { const resize = () => setWindowWidth(window.innerWidth); window.addEventListener('resize', resize); return () => window.removeEventListener('resize', resize) }, [])
  // Keep the workspace Git branch current across embedded terminal checkouts,
  // external editors, and agent turns without re-initializing the session.
  useEffect(() => {
    if (snap.connection !== 'online' || !snap.cwd) return
    const sync = (): void => { if (!pageIsInactive()) void store.refreshBranch() }
    const onFocus = (): void => { void store.refreshBranch() }
    const onVis = (): void => { if (typeof document !== 'undefined' && document.visibilityState === 'visible') void store.refreshBranch() }
    window.addEventListener('focus', onFocus)
    document.addEventListener('visibilitychange', onVis)
    const timer = setInterval(sync, 4_000)
    return () => {
      window.removeEventListener('focus', onFocus)
      document.removeEventListener('visibilitychange', onVis)
      clearInterval(timer)
    }
  }, [snap.connection, snap.cwd, snap.currentId])
  useEffect(() => {
    if (snap.connection === 'online' && snap.cwd && !snap.turnActive) {
      void store.refreshBranch()
    }
  }, [snap.connection, snap.cwd, snap.turnActive])
  // The sidebar stays at 820 px; it only turns into a drawer below 640 px.
  const narrow = windowWidth < 640
  const focused = narrow ? !narrowNavigation : layout.sidebarHidden
  /** The active right-panel tab's key (railTabs.ts); undefined until the user opens or closes one. */
  const [railChoice, setRailChoice] = useState<string | null | undefined>(undefined)
  const [railTabs, setRailTabs] = useState<readonly RailTab[]>([])
  const [dismissedActiveSwarmId, setDismissedActiveSwarmId] = useState('')
  useEffect(() => { if (!snap.swarm?.active) setDismissedActiveSwarmId('') }, [snap.swarm?.active, snap.currentId])
  const [filesExpanded, setFilesExpanded] = useState(false)
  const [reviewPath, setReviewPath] = useState('')
  /** A file to open in Files (a tool row's path); the count re-opens the same one. */
  const [filesTarget, setFilesTarget] = useState({ path: '', count: 0 })
  /** An artifact to open in Artifacts (a transcript card's id); the count re-opens the same one. */
  const [artifactTarget, setArtifactTarget] = useState({ id: '', count: 0 })
  const [selectedAgent, setSelectedAgent] = useState('')
  const [page, setPage] = useState<DesktopPageName | null>(null)
  // Each side panel has its own default and minimum width (layout.tsx PANEL_SIZES);
  // the chat always keeps 320px, so a wide default narrows to fit instead of
  // taking the window. Only when even the panel's minimum does not fit does
  // the panel go full width.
  const shownPanel: RailPanel = railChoice ? panelOfKey(railChoice) : 'activity'
  const panelMin = PANEL_SIZES[shownPanel].min
  const inspectorMax = Math.max(panelMin, windowWidth - (focused ? 0 : layout.sidebarWidth) - 320)
  const panelWidth = Math.max(panelMin, Math.min(inspectorMax, panelWidthOf(layout, shownPanel)))
  const contextRequiresFullWidth = windowWidth < (focused ? 0 : layout.sidebarWidth) + panelWidth + 320
  // In Vibes/Neo, the right panel starts closed and the floating ThreadDetailsCard
  // occupies the top-right gutter on wide windows. Classic keeps Activity open on entry.
  const railKey = railChoice === undefined ? (vibesStyle ? null : (!snap.noWorkspace && !contextRequiresFullWidth ? 'activity' : null)) : railChoice
  const rail: RailPanel | null = railKey ? panelOfKey(railKey) : null
  if (rail) lastRailRef.current = rail
  const agentOpen = page === null && Boolean(selectedAgent)
  const detailsInlineEligible = isDetailsInlineEligible({
    vibesStyle,
    isVscode: isVscodeHost(),
    noWorkspace: Boolean(snap.noWorkspace),
    railOpen: rail !== null,
    railLauncherOpen,
    agentOpen,
    focused,
    sidebarWidth: layout.sidebarWidth,
    windowWidth,
  })
  const detailsMode: 'inline' | 'popover' = detailsInlineEligible ? 'inline' : 'popover'
  const detailsOpen = detailsChoice === null ? detailsInlineEligible : detailsChoice
  // The entry tab (Activity) is shown before anything was opened explicitly; active swarms appear in the tab bar automatically unless closed.
  const entryTabs = rail && railKey && !railTabs.some(tab => tabKey(tab) === railKey) ? openTab(railTabs, { panel: rail }, railKey) : railTabs
  const activeSwarmIdentity = snap.swarm?.active ? (snap.swarm.session_id || snap.currentId || 'active') : ''
  const shownTabs = activeSwarmIdentity && dismissedActiveSwarmId !== activeSwarmIdentity && !entryTabs.some(tab => tab.panel === 'swarm')
    ? openTab(entryTabs, { panel: 'swarm' }, railKey ?? undefined)
    : entryTabs
  const railTabsRef = useRef(shownTabs)
  railTabsRef.current = shownTabs
  const railChoiceRef = useRef(railChoice)
  railChoiceRef.current = railChoice
  const prevCwdRef = useRef(snap.cwd)
  const prevSessionIdRef = useRef(snap.currentId)
  /** Open (or focus) a tab: a section, or one item inside it such as an artifact. Null closes the panel. */
  const setRail = (next: RailPanel | null, item?: string, title?: string): void => {
    if (next === null) {
      setFilesExpanded(false)
      setRailLauncherOpen(false)
      setRailChoice(null)
      return
    }
    if (next === 'swarm') setDismissedActiveSwarmId('')
    if (next === 'btw' && !store.getSnapshot().sideQuestionOpen) store.openSideQuestion()
    setRailLauncherOpen(false)
    lastRailRef.current = next
    const tab: RailTab = item ? { panel: next, item, ...(title ? { title } : {}) } : { panel: next }
    const key = tabKey(tab)
    setRailTabs(openTab(shownTabs, tab, key))
    setRailChoice(key)
  }
  const toggleRightPanel = (): void => {
    if (rail !== null || railLauncherOpen) {
      setRail(null)
      store.closeSwarmBoard()
      store.closeSideQuestion()
    } else if (shownTabs.length > 0 || lastRailRef.current !== null) {
      const target = lastRailRef.current ?? shownTabs[shownTabs.length - 1]!.panel
      setRail(target)
    } else if (vibesStyle) {
      setRailLauncherOpen(true)
    } else {
      setRail('activity')
    }
  }
  const toggleTerminalDrawer = (): void => setTerminalDrawerOpen(value => !value)
  const toggleDetails = (): void => setDetailsChoice(current => !(current === null ? detailsInlineEligible : current))
  const toggleSidebar = () => narrow ? setNarrowNavigation(value => !value) : setLayout({ sidebarHidden: !focused })
  // In a narrow window (the VS Code sidebar, a small app window) the task
  // list is a drawer over the conversation: picking a task closes it.
  useEffect(() => { setNarrowNavigation(false) }, [snap.currentId])
  // Back / Forward navigation history across opened threads and pages (sidebar chrome).
  type NavEntry = { readonly sessionId: string; readonly page: DesktopPageName | null }
  const [navHistory, setNavHistory] = useState<{ readonly entries: readonly NavEntry[]; readonly index: number }>(() => ({
    entries: snap.currentId || page ? [{ sessionId: snap.currentId, page }] : [],
    index: snap.currentId || page ? 0 : -1,
  }))
  const navReplaying = useRef(false)
  useEffect(() => {
    if (!snap.currentId && page === null) return
    if (navReplaying.current) {
      navReplaying.current = false
      return
    }
    setNavHistory(current => {
      const top = current.entries[current.index]
      if (top && top.sessionId === snap.currentId && top.page === page) return current
      const nextEntries = [...current.entries.slice(0, current.index + 1), { sessionId: snap.currentId, page }].slice(-40)
      return { entries: nextEntries, index: nextEntries.length - 1 }
    })
  }, [snap.currentId, page])
  const canGoBack = navHistory.index > 0
  const canGoForward = navHistory.index >= 0 && navHistory.index < navHistory.entries.length - 1
  const visitNavEntry = (entry: NavEntry): void => {
    navReplaying.current = true
    setPage(entry.page)
    setPanel(null)
    setSelectedAgent('')
    if (entry.sessionId && entry.sessionId !== store.getSnapshot().currentId) {
      void store.openSession(entry.sessionId)
    }
  }
  const goBack = (): void => {
    if (!canGoBack) return
    const nextIndex = navHistory.index - 1
    const target = navHistory.entries[nextIndex]
    if (!target) return
    setNavHistory(current => ({ ...current, index: nextIndex }))
    visitNavEntry(target)
  }
  const goForward = (): void => {
    if (!canGoForward) return
    const nextIndex = navHistory.index + 1
    const target = navHistory.entries[nextIndex]
    if (!target) return
    setNavHistory(current => ({ ...current, index: nextIndex }))
    visitNavEntry(target)
  }
  // Command Palette layout actions.
  const t3ActionRef = useRef({ toggleTerminalDrawer, toggleDetails, toggleRightPanel, inbox })
  t3ActionRef.current = { toggleTerminalDrawer, toggleDetails, toggleRightPanel, inbox }
  useEffect(() => {
    const onAction = (event: Event): void => {
      const detail = (event as CustomEvent<unknown>).detail
      if (detail === 'toggle-terminal-drawer') t3ActionRef.current.toggleTerminalDrawer()
      else if (detail === 'toggle-thread-details') t3ActionRef.current.toggleDetails()
      else if (detail === 'toggle-right-panel') t3ActionRef.current.toggleRightPanel()
      else if (detail === 'toggle-sidebar-view') {
        const cur = t3ActionRef.current.inbox.state.viewMode
        t3ActionRef.current.inbox.setViewMode(cur === 'inbox' ? 'folders' : 'inbox')
      }
    }
    window.addEventListener('xerxes:t3-action', onAction)
    return () => window.removeEventListener('xerxes:t3-action', onAction)
  }, [])
  const chrome: WindowChromeState = {
    sidebarVisible: !focused,
    inspectorOpen: rail !== null || railLauncherOpen,
    toggleSidebar,
    detailsOpen,
    detailsMode,
    toggleDetails,
    terminalDrawerOpen,
    terminalDrawerHeight,
    setTerminalDrawerHeight,
    toggleTerminalDrawer,
    toggleRightPanel,
    canGoBack,
    canGoForward,
    goBack,
    goForward,
    inbox,
  }
  // XERXES supervisor is its own full view, so the task's side panel steps aside while it is shown.
  const xerxesMode = vibesStyle && page === 'xerxes'
  // Which stylesheet takeovers are in effect, named once so the decision
  // dock below and the CSS cannot disagree about when .chat is gone.
  // Git sits in the rail like the others; only its expanded diff view takes the window.
  const contextFull = Boolean((rail || railLauncherOpen) && (filesExpanded || contextRequiresFullWidth))
  // An open sub-agent takes the conversation's place, composer and cards included.
  const chatHidden = { contextFull, any: contextFull || agentOpen }
  // The divider never drags past the width that hides the divider itself
  // (inspectorMax above): the chat always keeps 320px.
  const navigate = (next: DesktopPanel, filePath?: string): void => {
    if (next === 'review') setReviewPath(filePath ?? '')
    if (next === 'files' && filePath) setFilesTarget(target => ({ path: filePath, count: target.count + 1 }))
    if (next === 'docs' && filePath) {
      setArtifactTarget(target => ({ id: filePath, count: target.count + 1 }))
      setFilesExpanded(false)
    }
    // A link to one file's changes opens straight into the diff view.
    if (next === 'review' && filePath && !contextRequiresFullWidth) setFilesExpanded(true)
    // An agent opens in the conversation's place (DeepSeek Harness's layout);
    // the rail stays as it was.
    if (next === 'activity' && filePath) { setSelectedAgent(filePath); setPage(null); setPanel(null); return }
    if (next === 'activity') requestAnimationFrame(() => document.querySelector('.desktop-rail .studio-sheet-content')?.scrollTo({ top: 0 }))
    // Every file gets its own tab, so two files stay open side by side.
    if (next === 'files' && filePath) { const path = workspaceFilePath(filePath, snap.cwd); setRail('files', path, path.split('/').filter(Boolean).at(-1) ?? path) }
    // Every right-panel view opens as a tab; the list is the one source, so a new view cannot fall through to a dialog.
    else if (next !== null && (RAIL_PANELS as readonly string[]).includes(next)) setRail(next as RailPanel, next === 'docs' ? filePath : undefined)
    else if (next === 'agents' || next === 'xerxes' || next === 'extensions' || next === 'artifacts' || next === 'schedules') { setPage(next); setPanel(null); setSelectedAgent('') }
    else if (next === null) { setPage(null); setPanel(null); setSelectedAgent(''); store.closeLearning(); store.closeSwarmBoard() }
    else setPanel(next)
  }
  // Tasks, Agents and XERXES are independent destinations. Every explicit
  // session open returns to Tasks, including reopening the current session;
  // the desktop connection's sessionKey does not change between sessions.
  // A clicked notification: open its task, then the artifact it names. The open
  // above resets the panels, so the artifact waits until that reset has run.
  const pendingArtifact = useRef<{ sessionId?: string; artifactId: string } | null>(null)
  const handleExternalNavigate = (target: { sessionId?: string; panel?: 'docs'; artifactId?: string }): void => {
    const current = store.getSnapshot().currentId
    if (target.artifactId) pendingArtifact.current = { artifactId: target.artifactId, ...(target.sessionId ? { sessionId: target.sessionId } : {}) }
    if (target.sessionId && target.sessionId !== current) { void store.openSession(target.sessionId); return }
    if (target.artifactId) { pendingArtifact.current = null; navigate('docs', target.artifactId); return }
    // This view's own task was clicked (from another view, or a notification):
    // show its chat, never the page (XERXES, Customize…) the view was left on.
    if (target.sessionId) navigate(null)
  }
  useEffect(() => window.xerxes.onNavigate?.(handleExternalNavigate), [])
  useEffect(() => {
    setPanel(null); setPage(null); setSelectedAgent(''); setFilesExpanded(false)
    const cwdChanged = prevCwdRef.current !== snap.cwd
    const sessionChanged = prevSessionIdRef.current !== snap.currentId
    prevCwdRef.current = snap.cwd
    prevSessionIdRef.current = snap.currentId
    if (cwdChanged || sessionChanged) {
      const pruned = pruneRailTabsForContext(railTabsRef.current, railChoiceRef.current, { cwdChanged, sessionChanged })
      if (pruned.tabs !== railTabsRef.current) setRailTabs(pruned.tabs)
      if (pruned.activeKey !== railChoiceRef.current) {
        setRailChoice(pruned.activeKey)
        if (pruned.activeKey === null) lastRailRef.current = null
      }
      if (cwdChanged) {
        setFilesTarget({ path: '', count: 0 })
        setReviewPath('')
      }
      if (sessionChanged) {
        setArtifactTarget({ id: '', count: 0 })
      }
    }
    const pending = pendingArtifact.current
    if (pending && (!pending.sessionId || pending.sessionId === snap.currentId)) { pendingArtifact.current = null; navigate('docs', pending.artifactId) }
  }, [snap.cwd, snap.currentId, snap.sessionOpenRevision])
  useEffect(() => { if (snap.learningOpen) { setPage('extensions'); setPanel(null); setFilesExpanded(false); if (contextRequiresFullWidth) setRail(null) } }, [snap.learningOpenRevision])
  useEffect(() => { if (snap.swarmOpen) { setPage(null); setPanel(null); setSelectedAgent(''); setRail('swarm') } }, [snap.swarmOpenRevision])
  useEffect(() => {
    if (snap.sideQuestionOpen) {
      setPage(null); setPanel(null); setSelectedAgent(''); setRail('btw')
    } else if (rail === 'btw' || railTabsRef.current.some(tab => tab.panel === 'btw')) {
      const next = closeTab(railTabsRef.current, 'btw', railKey ?? 'btw')
      setRailTabs(next.tabs)
      setRailChoice(next.active)
      if (next.active === null) {
        setFilesExpanded(false)
        lastRailRef.current = null
      }
    }
  }, [snap.sideQuestionOpen, snap.sideQuestionOpenRevision])
  return (
    <DesktopNavigation.Provider value={navigate}><WidgetSessionContext.Provider value={snap.sessionKey}><WidgetOpenArtifactContext.Provider value={artifactId => navigate('docs', artifactId)}><ActivityVisible.Provider value={rail === 'activity'}><WindowChrome.Provider value={chrome}>
    <div className={`app atelier${trafficLights ? '' : ' app--no-traffic-lights'}${focused ? ' atelier--focus' : ''}${narrow ? ' app--narrow' : ''}`} style={{ '--sidebar-width': `${layout.sidebarWidth}px`, '--inspector-width': `${panelWidth}px` } as React.CSSProperties}>
      {/* No title bar: its controls live in the sidebar's top row and the
          conversation header — one row, as in Claude's app. */}
      <FirstRunSetup snap={snap} />
      <div className="app__body" data-context-full={chatHidden.contextFull || undefined} data-review={(rail === "review" && contextFull) || undefined}>
        <Sidebar snap={snap} page={page} />
        {narrow && narrowNavigation && <div className="side-scrim" aria-hidden="true" onClick={() => setNarrowNavigation(false)} />}
        {!focused && <PanelDivider label="Resize sessions" value={layout.sidebarWidth} min={180} max={360} onChange={sidebarWidth => setLayout({ sidebarWidth })} />}
        {snap.noWorkspace ? snap.storageScope?.startsWith('ssh:') ? <RemoteWorkspaceGate /> : <WorkspaceGate /> : <ErrorBoundary label="This conversation"><Chat snap={snap} page={page} agent={agentOpen ? selectedAgent : ''} openAgent={setSelectedAgent} /></ErrorBoundary>}
        {!rail && !railLauncherOpen && !snap.noWorkspace && !xerxesMode && !vibesStyle && <FloatingRail open={navigate} swarmActive={Boolean(snap.swarm?.active)} externalAcp={isExternalAcpBinding(snap.acpBinding)} showImmortals={showsImmortalsForSession(snap)} />}
        {(rail || railLauncherOpen) && !xerxesMode && <PanelDivider label="Resize inspector" value={panelWidth} min={panelMin} max={inspectorMax} reverse onChange={width => setLayout({ panelWidths: { ...layout.panelWidths, [rail ?? 'activity']: width } })} />}
        {rail && !xerxesMode && <ErrorBoundary label={railLabel(rail)}><DesktopRail panel={rail} snap={snap} close={() => { setRail(null); store.closeSwarmBoard(); store.closeSideQuestion() }} filesExpanded={filesExpanded} reviewPath={reviewPath} filesTarget={filesTarget} artifactTarget={artifactTarget} tabs={shownTabs} activeKey={railKey ?? rail}
          onSelectTab={key => setRailChoice(key)}
          onCloseTab={key => {
            if (key === 'swarm') { store.closeSwarmBoard(); if (activeSwarmIdentity) setDismissedActiveSwarmId(activeSwarmIdentity) }
            if (key === 'btw') store.closeSideQuestion()
            const next = closeTab(shownTabs, key, railKey ?? rail)
            setRailTabs(next.tabs)
            setRailChoice(next.active)
            if (next.active === null) {
              setFilesExpanded(false)
              lastRailRef.current = null
              if (vibesStyle) setRailLauncherOpen(true)
            }
          }}
          onRetitle={(key, title) => setRailTabs(tabs => retitleTab(tabs, key, title))} {...(!contextRequiresFullWidth ? { toggleFilesExpanded: () => setFilesExpanded(value => !value), setExpanded: setFilesExpanded } : {})} activityDetails={<ActivityDetails snap={snap} selectedAgent={agentOpen ? selectedAgent : ''} />} /></ErrorBoundary>}
        {!rail && railLauncherOpen && !xerxesMode && (
          <RightPanelLauncher
            snap={snap}
            onPick={surface => navigate(surface)}
            onClose={() => { setFilesExpanded(false); setRailLauncherOpen(false) }}
            filesExpanded={filesExpanded}
            {...(!contextRequiresFullWidth ? { toggleFilesExpanded: () => setFilesExpanded(value => !value) } : {})}
          />
        )}
        {!snap.noWorkspace && (typeof window === 'undefined' || typeof window.xerxes?.onToast !== 'function') && (
          <ArtifactToast sessionId={snap.currentId} showingArtifacts={rail === 'docs'} />
        )}

      </div>
      {/* The review takeover and an expanded rail both hide .chat, which
          took the composer, the approval card and the question card with
          it — a decision that arrived mid-review was invisible and
          unanswerable. Float it above the takeover instead. */}
      {chatHidden.any && (snap.approval || snap.question) && (
        <div className="decision-dock">
          {/* The chat header (and its badge) is hidden here too, so a hidden
              card keeps a way back in the dock. */}
          {snap.interactionHidden
            ? <button className="badge badge--need" title="Show the request waiting for you" onClick={() => store.showInteraction()}>needs input{snap.turnActive ? ' · acting paused' : ''}</button>
            : snap.approval
            ? <ApprovalCard key={`dock:${snap.approval.id}`} approval={snap.approval} policy={snap.permissionMode} />
            : snap.question ? <QuestionCard key={`dock:${snap.question.requestId}`} question={snap.question} plan={snap.plan} /> : null}
        </div>
      )}
      <FindBar />
      <Shortcuts />
      <Announcer snap={snap} />
      <SettingsModal snap={snap} />
      <AppUpdatePrompt update={snap.appUpdate} />
      {/* Startup notices share one corner and stack instead of overlapping. */}
      <div className="notice-stack"><HubHealthNotice /><MacPermissionsNotice wanted={snap.computerUseWanted} /></div>
      {snap.taskModalOpen && <TaskModal snap={snap} />}
      {/* Mounted only while open: the palette's hooks (needle, cursor,
          focus effect) must never share a fiber with a closed render. */}
      {snap.paletteOpen && <CommandPalette snap={snap} />}
      {snap.searchOpen && <SessionSearch snap={snap} />}
      {snap.wsMenuOpen && <WorkspaceMenu snap={snap} />}
      {snap.sessionMenu && <SessionMenu menu={snap.sessionMenu} />}
      {panel && <DesktopSheet panel={panel} snap={snap} close={() => setPanel(null)} activityDetails={<ActivityDetails snap={snap} />} />}
      {(snap.workspaceBusy || snap.workspaceError) && <div className="workspace-notice" role={snap.workspaceError ? 'alert' : 'status'}>
        <span>{snap.workspaceError || 'Opening workspace…'}</span>
        {snap.workspaceError && <button aria-label="Dismiss workspace error" onClick={() => store.clearWorkspaceError()}><Icon name="close" size={13} /></button>}
      </div>}
      <GlobalKeys
        snap={snap}
        closeSurface={detailsOpen && detailsMode === 'popover' ? () => setDetailsChoice(false) : panel || page ? () => { setPanel(null); setPage(null) } : null}
        toggleTerminalDrawer={toggleTerminalDrawer}
        toggleRightPanel={toggleRightPanel}
        undoInbox={inbox.undo ? inbox.applyUndo : null}
        toggleSwarm={() => {
          if (rail === 'swarm') {
            store.closeSwarmBoard()
            if (activeSwarmIdentity) setDismissedActiveSwarmId(activeSwarmIdentity)
            const next = closeTab(shownTabs, 'swarm', railKey ?? 'swarm')
            setRailTabs(next.tabs)
            setRailChoice(next.active)
            if (next.active === null) setFilesExpanded(false)
          } else {
            store.openSwarmBoard()
          }
        }}
      />
    </div>
    </WindowChrome.Provider></ActivityVisible.Provider></WidgetOpenArtifactContext.Provider></WidgetSessionContext.Provider></DesktopNavigation.Provider>
  )
}

// ── Top bar ─────────────────────────────────────────────────────────────

/** What the window's top row needs from the shell (it has no title bar). */
interface WindowChromeState {
  readonly sidebarVisible: boolean
  readonly inspectorOpen: boolean
  readonly toggleSidebar: () => void
  readonly detailsOpen?: boolean
  readonly detailsMode?: 'inline' | 'popover'
  readonly toggleDetails?: () => void
  readonly terminalDrawerOpen?: boolean
  readonly terminalDrawerHeight?: number
  readonly setTerminalDrawerHeight?: (height: number) => void
  readonly toggleTerminalDrawer?: () => void
  readonly toggleRightPanel?: () => void
  readonly canGoBack?: boolean
  readonly canGoForward?: boolean
  readonly goBack?: () => void
  readonly goForward?: () => void
  readonly inbox?: ReturnType<typeof useThreadInbox>
}

const WindowChrome = createContext<WindowChromeState>({ sidebarVisible: true, inspectorOpen: false, toggleSidebar: () => {} })

function SidebarToggle(): ReactElement {
  const chrome = useContext(WindowChrome)
  const vibes = useVibesStyle()
  return <button type="button" className="studio-icon window-toggle" title="Toggle sidebar" aria-label="Toggle sidebar" aria-expanded={chrome.sidebarVisible} aria-controls="session-sidebar" onClick={chrome.toggleSidebar}>{vibes ? <T3Icon name="panelLeft" size={15} /> : <Icon name="sidebar" />}</button>
}

/**
 * While the inspector is closed, a slim column of icons opens one of its
 * sections. Vibes and Neo show it as a row of icons in the header (vibes.css).
 */
export function FloatingRail({
  open,
  swarmActive = false,
  externalAcp = false,
  showImmortals,
}: {
  open: (panel: DesktopPanel) => void
  swarmActive?: boolean
  externalAcp?: boolean
  showImmortals?: boolean
}): ReactElement {
  const vibes = useVibesStyle()
  // The rail: Subagents, Immortals (shown on native Xerxes, Antigravity / swarm-capable ACP, or active swarm), Side question, Artifacts, Changes, Files, Terminal (then Activity).
  const allItems = [
    { panel: 'subagents', icon: 'agents', label: 'Subagents' },
    { panel: 'swarm', icon: 'tree', label: swarmActive ? 'Immortals (active)' : 'Immortals' },
    { panel: 'btw', icon: 'chat', label: 'Side question' },
    { panel: 'docs', icon: 'stack', label: 'Artifacts' },
    { panel: 'review', icon: 'changes', label: 'Changes' },
    { panel: 'files', icon: 'folder', label: 'Files' },
    { panel: 'terminal', icon: 'terminal', label: 'Terminal' },
    { panel: 'activity', icon: vibes ? 'listBullets' : 'activity', label: 'Activity' },
  ] as const
  const includeSwarm = showImmortals ?? (!externalAcp || swarmActive)
  const items = includeSwarm
    ? allItems
    : allItems.filter(item => item.panel !== 'swarm')
  return <nav className="floatrail" aria-label="Open a side panel">
    <button type="button" title="Search sessions" aria-label="Search sessions" onClick={() => store.openSessionSearch()}><Icon name="search" size={16} /></button>
    {items.map(item => <button key={item.panel} type="button" data-live={(item.panel === 'swarm' && swarmActive) || undefined} title={item.label} aria-label={item.label} onClick={() => open(item.panel)}><Icon name={item.icon} size={16} /></button>)}
  </nav>
}

/** Search only below 761px; Activity, Changes and Files live in the floating rail and the Sections menu. */
function WorkspaceTools(_props: { snap: Snapshot; inspectorOpen: boolean }): ReactElement {
  return <div className="top__actions" role="group" aria-label="Workspace tools">
    {/* Below 761px the floating rail is hidden, so Search falls back to the top bar there (layout.css). */}
    <button type="button" className="top__search" title="Search sessions" aria-label="Search sessions" onClick={() => store.openSessionSearch()}><Icon name="search" /></button>
  </div>
}


/** One workspace-level home for connection, version mismatch and recovery. */
function RuntimeStatus({ snap, compact = false, iconOnly = false }: { snap: Snapshot; compact?: boolean; iconOnly?: boolean }): ReactElement {
  const navigate = useDesktopNavigation()
  const [expanded, setExpanded] = useState(false)
  const trigger = useRef<HTMLButtonElement>(null)
  const popup = useRef<HTMLDivElement>(null)
  const busy = snap.runtimeUpdate === 'checking' || snap.runtimeUpdate === 'restarting'
  const label = snap.noWorkspace ? snap.storageScope?.startsWith('ssh:') ? 'SSH connection' : 'Choose workspace' : busy ? 'Updating runtime…' : snap.runtimeUpdate === 'waiting' ? snap.runtimeUpdateMessage?.includes('goal round') ? 'Update after this goal round' : 'Update waiting for idle' : snap.connection === 'offline' ? connectionFailureKind(snap.error) === 'transport' ? 'Runtime offline' : 'Workspace needs attention' : snap.connection === 'connecting' ? 'Connecting…' : snap.daemonWarning ? 'Runtime update available' : 'Connected'
  useLayoutEffect(() => {
    if (!expanded) return
    const place = () => {
      const anchor = trigger.current?.getBoundingClientRect(), panel = popup.current
      if (!anchor || !panel) return
      const left = Math.max(8, Math.min(anchor.left, window.innerWidth - panel.offsetWidth - 8))
      const top = anchor.top > panel.offsetHeight + 16 ? anchor.top - panel.offsetHeight - 8 : anchor.bottom + 8
      panel.style.left = left + 'px'
      panel.style.top = Math.max(8, Math.min(top, window.innerHeight - panel.offsetHeight - 8)) + 'px'
    }
    place()
    popup.current?.focus()
    const dismiss = (event: MouseEvent) => {
      if (event.target instanceof Node && !popup.current?.contains(event.target) && !trigger.current?.contains(event.target)) setExpanded(false)
    }
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); setExpanded(false); trigger.current?.focus() }
    }
    window.addEventListener('resize', place)
    document.addEventListener('mousedown', dismiss)
    document.addEventListener('keydown', escape)
    return () => { window.removeEventListener('resize', place); document.removeEventListener('mousedown', dismiss); document.removeEventListener('keydown', escape) }
  }, [expanded, label, snap.daemonWarning, snap.runtimeUpdateMessage])
  return <>
    {iconOnly ? (
      <button
        ref={trigger}
        type="button"
        className={`t3-icon-btn t3-runtime-icon${snap.daemonWarning ? ' has-update' : ''}${snap.connection === 'offline' ? ' is-offline' : ''}`}
        title={label}
        aria-label={'Workspace runtime: ' + label}
        aria-expanded={expanded}
        aria-haspopup="dialog"
        onClick={() => setExpanded(value => !value)}
      >
        <T3Icon name="circleArrowUp" size={15} />
      </button>
    ) : (
      <button ref={trigger} type="button" className={'runtime-status' + (compact ? ' runtime-status--compact' : '')} aria-label={'Workspace runtime: ' + label} aria-expanded={expanded} aria-haspopup="dialog" onClick={() => setExpanded(value => !value)}>
        <span className="runtime-status__dot" data-state={snap.connection} data-warning={Boolean(snap.daemonWarning)} aria-hidden="true" />
        <span>{label}</span>
      </button>
    )}
    {expanded && createPortal(<div ref={popup} className="runtime-popover" role="dialog" aria-label="Workspace runtime" tabIndex={-1}>
      <header><strong>Workspace runtime</strong><button type="button" aria-label="Close runtime status" onClick={() => { setExpanded(false); trigger.current?.focus() }}><Icon name="close" size={13} /></button></header>
      <p role="status" aria-live="polite">{snap.connection === 'online' ? 'Connected to this workspace' : label}</p>
      {snap.daemonWarning && <><strong>App and runtime versions differ</strong><p>{snap.daemonWarning}</p></>}
      {snap.runtimeUpdateMessage && <p role="status">{snap.runtimeUpdateMessage}</p>}
      {snap.daemonWarning && snap.runtimeUpdate === 'waiting' ? <>
        {(snap.runtimeBlockers?.length ?? 0) > 0 && <><p>Still running:</p><ul className="runtime-popover__blockers">{snap.runtimeBlockers!.map(item => <li key={item}>{item}</li>)}</ul></>}
        <p>Restarting now stops that work. Goals pick up again in the new runtime.</p>
        <div className="runtime-popover__actions">
          <button type="button" onClick={() => { setExpanded(false); navigate('activity') }}>Review activity</button>
          <button type="button" className="runtime-popover__danger" disabled={busy} onClick={() => void store.restartRuntimeNow()}>Restart now</button>
        </div>
      </> : snap.daemonWarning?.startsWith('The app is older')
        // The runtime is newer than this app: restarting it cannot help (the
        // app would refuse, or reinstall the same newer runtime). Update the app.
        ? <><p>{isVscodeHost() ? 'Update the Xerxes extension, then reload the window. Your tasks keep running meanwhile.' : 'Update this app to match the runtime. Your tasks keep running meanwhile.'}</p>{!isVscodeHost() && <div className="runtime-popover__actions"><button type="button" disabled={snap.appUpdate?.phase === 'checking' || snap.appUpdate?.phase === 'downloading'} onClick={() => { setExpanded(false); void store.appUpdateAction('check') }}>Update the app</button></div>}</>
        : snap.daemonWarning && <><p>The current runtime keeps your tasks running. Review activity before restarting.</p><div className="runtime-popover__actions"><button type="button" onClick={() => { setExpanded(false); navigate('activity') }}>Review activity</button><button type="button" disabled={busy} onClick={() => void store.restartDaemon()}>Restart workspace runtime</button></div></>}
      {!snap.daemonWarning && snap.connection !== 'online' && <button type="button" onClick={() => { setExpanded(false); navigate('workspace') }}>Open workspace settings</button>}
    </div>, document.body)}
  </>
}

// ── Workspace switcher menu (mockup 16) ─────────────────────────────────

/**
 * The topbar chip's dropdown: every folder that holds chats, current first
 * and marked, then the folder picker. Other workspaces open independently
 * so the current session keeps its connection and draft.
 */
function WorkspaceMenu({ snap }: { snap: Snapshot }): ReactElement {
  const chrome = useContext(WindowChrome)
  const hiddenSet = new Set((chrome.inbox?.state.hiddenWorkspaces ?? []).map(dir => dir.replace(/\/+$/, '') || '/'))
  const groups = groupByWorkspace(snap.sessions, snap.cwd, snap.workspaceDirectories).filter(group => {
    const clean = group.cwd.replace(/\/+$/, '') || '/'
    return clean === (snap.cwd.replace(/\/+$/, '') || '/') || !hiddenSet.has(clean)
  })
  const statusFor = (cwd: string): string => {
    if (cwd === snap.cwd) return 'current'
    const count = groups.find(g => g.cwd === cwd)?.rows.length ?? 0
    return `${count} ${count === 1 ? 'task' : 'tasks'}`
  }
  return (
    <>
      <div className="backdrop backdrop--clear" onClick={() => store.closeWorkspaceMenu()} />
      {/* role="menu" owns only menuitem children; the plain buttons and
          captions here made it an invalid tree that screen readers report
          as an empty menu. A labelled group describes what is actually
          rendered. */}
      <div className="wsmenu" role="group" aria-label="Workspaces">
        <div className="cap wsmenu__cap">Workspaces</div>
        {groups.map(group => (
          <button
            key={group.cwd || group.name}
            type="button"
            aria-current={group.cwd === snap.cwd ? 'true' : undefined}
            className={`wsrow${group.cwd === snap.cwd ? ' is-cur' : ''}`}
            title={group.cwd === snap.cwd ? `${group.cwd} · current workspace` : `Switch to ${group.cwd}`}
            onClick={() => { if (group.cwd !== snap.cwd) store.enterWorkspace(group.cwd) }}
          >
            <span className={`dot ${group.cwd === snap.cwd ? 'dot--live' : 'dot--idle'}`} aria-hidden="true" />
            <span className="wsrow__main">
              <span className="wsrow__t">{group.name}</span>
              <span className="wsrow__s">{group.cwd || group.name} · {statusFor(group.cwd)}</span>
            </span>
            {group.cwd === snap.cwd && <span className="kbd"><Icon name="check" size={11} /></span>}
          </button>
        ))}
        {groups.length === 0 && (
          <div className="wsrow is-cur">
            <span className="dot dot--live" aria-hidden="true" />
            <span className="wsrow__main">
              <span className="wsrow__t">{workspaceLabel(snap.cwd) || 'No workspace'}</span>
              <span className="wsrow__s">{snap.cwd || 'Choose a folder to begin'}</span>
            </span>
            <span className="kbd"><Icon name="check" size={11} /></span>
          </div>
        )}
        <div className="menu__sep" />
        <button type="button" className="menu__item" onClick={() => { store.closeWorkspaceMenu(); store.chooseWorkspace() }}>
          <Icon name="plus" size={13} /> Add workspace…
        </button>
      </div>
    </>
  )
}

// ── New-task modal (mockup 18) ──────────────────────────────────────────

/**
 * ⌘N: describe the outcome, optionally arm the plan ceiling, start. The
 * workspace seg shows the current folder plus the folder picker (worktree
 * slots await a daemon capability); the model row is informational — the
 * live picker stays anchored to the composer chip.
 */
function TaskModal({ snap }: { snap: Snapshot }): ReactElement | null {
  const [objectiveText, setObjectiveText] = useState('')
  const [planFirst, setPlanFirst] = useState(true)
  const [namingWorktree, setNamingWorktree] = useState(false)
  const [worktreeName, setWorktreeName] = useState('')
  const [agentPreset, setAgentPreset] = useState('')
  const [model, setModel] = useState(snap.model)
  const [starting, setStarting] = useState(false)
  // Every other aria-modal dialog in the renderer traps and restores focus;
  // this one shipped without it, so Tab walked out into the shell behind.
  const dialogRef = useRef<HTMLDivElement>(null)
  const objectiveField = useRef<HTMLTextAreaElement>(null)
  useDialogFocus(objectiveField)
  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      if (namingWorktree) {
        event.preventDefault()
        event.stopImmediatePropagation()
        setNamingWorktree(false)
        setWorktreeName('')
        return
      }
      if (starting) {
        event.preventDefault()
        event.stopImmediatePropagation()
        return
      }
      const root = dialogRef.current
      const ownsFocus = !root || root.contains(document.activeElement) || document.activeElement === document.body
      handleDialogEscape(event, () => store.closeTaskModal(), ownsFocus)
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [namingWorktree, starting])
  const models = [...new Set([snap.model, ...snap.models.map(choice => choice.id)].filter(Boolean))]
  const presets = snap.agentPresets ?? []
  const selectedPreset = agentPreset || presets.find(row => row.isDefault && !row.broken)?.id || 'default'
  const start = (): void => {
    if (starting || !objectiveText.trim() || !model) return
    setStarting(true)
    void store.startTask(objectiveText, planFirst, selectedPreset, model).finally(() => setStarting(false))
  }
  const submitWorktree = (): void => {
    if (!worktreeName.trim()) { setNamingWorktree(false); return }
    void store.createWorktree(worktreeName)
  }
  return (
    <div
      className="backdrop"
      onMouseDown={event => {
        if (event.target === event.currentTarget && !starting) store.closeTaskModal()
      }}
    >
      <div
        ref={dialogRef}
        className="modal taskmodal"
        role="dialog"
        aria-modal="true"
        aria-label="New task"
        onKeyDown={event => {
          if (event.key === 'Escape' && !event.defaultPrevented && !starting) {
            event.preventDefault()
            event.stopPropagation()
            store.closeTaskModal()
          }
        }}
      >
        <div className="modal__main">
          <h2 className="modal__title">New task</h2>
          <p className="modal__sub">Choose how you want to work, then describe your task.</p>

          <div className="field">
            <label htmlFor="task-workspace-btn">Workspace</label>
            <div className="seg" style={{ flexWrap: 'wrap' }}>
              <button id="task-workspace-btn" type="button" className="is-on" title={snap.cwd}><Icon name="folder" size={13} /> {workspaceLabel(snap.cwd) || 'no workspace'}</button>
              <button type="button" onClick={() => store.chooseWorkspace()} title="Open a different folder as the workspace"><Icon name="plus" size={13} /> different folder…</button>
              <button
                type="button"
                aria-expanded={namingWorktree}
                onClick={() => setNamingWorktree(value => !value)}
                title="Create an isolated git worktree next to the repo and switch into it"
              >
                <Icon name="plus" size={12} /> new worktree…
              </button>
              {namingWorktree && (
                <>
                  <input
                    className="taskmodal__wt"
                    autoFocus
                    spellCheck={false}
                    aria-label="New worktree name"
                    placeholder="Worktree name (branch + folder)"
                    value={worktreeName}
                    onChange={e => setWorktreeName(e.target.value)}
                    onKeyDown={e => {
                      if (e.key === 'Enter') { e.preventDefault(); submitWorktree() }
                      if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); setNamingWorktree(false); setWorktreeName('') }
                    }}
                  />
                  <button type="button" className="btn btn--solid" disabled={!worktreeName.trim()} onClick={submitWorktree}>Create</button>
                  <button type="button" className="btn btn--ghost" onClick={() => { setNamingWorktree(false); setWorktreeName('') }}>Cancel</button>
                </>
              )}
            </div>
          </div>

          <div className="field">
            <label htmlFor="task-agent">Agent preset</label>
            <select
              id="task-agent"
              disabled={starting}
              value={selectedPreset}
              onChange={event => setAgentPreset(event.target.value)}
              title="Fixed after this session starts"
            >
              {presets.filter(row => !row.broken).map(row => (
                <option key={row.id} value={row.id}>{row.name}{row.isDefault && row.name !== 'default' ? ' · default' : ''}</option>
              ))}
              {presets.length === 0 && <option value="default">default</option>}
            </select>
            <div className="row__s" style={{ marginTop: 4 }}>Instructions and tools for this task.</div>
          </div>

          <div className="field">
            <label htmlFor="task-objective">Objective</label>
            <textarea
              id="task-objective"
              ref={objectiveField}
              className="composer__input taskmodal__objective"
              rows={4}
              value={objectiveText}
              placeholder="What would you like to get done?"
              onChange={e => setObjectiveText(e.target.value)}
              onKeyDown={e => {
                if (e.nativeEvent.isComposing) return
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  start()
                }
              }}
            />
          </div>

          <div className="row">
            <div className="row__main">
              <div className="row__t">Review plan before changes</div>
              <div className="row__s">The agent proposes a checklist; you approve it before anything runs.</div>
            </div>
            <button
              type="button"
              className={`switch${planFirst ? ' is-on' : ''}`}
              role="switch"
              aria-checked={planFirst}
              aria-label="Review plan before changes"
              onClick={() => setPlanFirst(value => !value)}
            />
          </div>

          <div className="field">
            <label htmlFor="task-model">Model</label>
            <select aria-label="Model" id="task-model" value={model} disabled={starting} onChange={event => setModel(event.target.value)}>
              {!model && <option value="">Choose a model</option>}
              {models.map(id => <option key={id} value={id}>{id}</option>)}
            </select>
          </div>

          {/* The mode is pinned per session, not per workspace — saying
              "in this workspace" promised the next task would inherit it.
              And the affordance was a bare <u>: not focusable, not a
              button, the only one of its kind in the renderer. */}
          <div className="fieldnote" style={{ marginBottom: 16 }}>
            approvals for this task: {snap.permissionMode || 'daemon policy'} ·{' '}
            <button className="linkish" onClick={() => { store.closeTaskModal(); store.openSettings('permissions') }}>change</button>
          </div>

          {snap.error && <div role="alert" className="taskmodal__error">{snap.error}</div>}
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <button type="button" className="btn btn--ghost" disabled={starting} onClick={() => store.closeTaskModal()}>Cancel</button>
            <button type="button" className="btn btn--solid" disabled={starting || !objectiveText.trim() || !model} onClick={start}>{starting ? 'Starting…' : 'Start task'}</button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Session context menu (mockup 08) ────────────────────────────────────

/**
 * Copy to the clipboard: the async Clipboard API first (Electron and VS Code
 * webviews), then the execCommand fallback where that API is missing or refused.
 */
function copyText(text: string): void {
  const clipboard = typeof navigator === 'object' ? navigator.clipboard : undefined
  if (clipboard && typeof clipboard.writeText === 'function') {
    clipboard.writeText(text).catch(() => copyTextFallback(text))
    return
  }
  copyTextFallback(text)
}

function copyTextFallback(text: string): void {
  const ta = document.createElement('textarea')
  ta.value = text
  ta.setAttribute('readonly', '')
  ta.style.position = 'fixed'
  ta.style.opacity = '0'
  document.body.appendChild(ta)
  ta.select()
  document.execCommand('copy')
  ta.remove()
}

/**
 * Right-click menu on a sidebar session: Open, Rename… (through the
 * daemon's `session.title`), Copy id, Export as markdown. Items the mockup
 * shows but no wire capability backs (move-to-worktree, delete) are
 * deliberately absent — no dead switches.
 */
function SessionMenu({ menu }: { menu: Snapshot['sessionMenu'] }): ReactElement | null {
  const chrome = useContext(WindowChrome)
  const [renaming, setRenaming] = useState(menu?.renaming ?? false)
  // Delete asks in the menu itself: VS Code webviews block modal browser
  // dialogs (a confirm prompt returns false without showing anything).
  const [confirmingDelete, setConfirmingDelete] = useState(false)
  const [draft, setDraft] = useState(menu?.title ?? '')
  const menuRef = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const element = menuRef.current
    if (!element || !menu) return
    const previous = document.activeElement
    element.querySelector<HTMLElement>('button,input')?.focus()
    return () => { if (previous instanceof HTMLElement && previous.isConnected) previous.focus() }
  }, [menu?.key])
  useLayoutEffect(() => {
    const element = menuRef.current
    if (!element || !menu) return
    const place = (): void => {
      const rect = element.getBoundingClientRect()
      element.style.left = `${Math.max(8, Math.min(menu.x, window.innerWidth - rect.width - 8))}px`
      element.style.top = `${Math.max(8, Math.min(menu.y, window.innerHeight - rect.height - 8))}px`
    }
    place()
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(place) : null
    observer?.observe(element)
    window.addEventListener('resize', place)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', place)
    }
  }, [menu?.key, menu?.x, menu?.y, renaming, confirmingDelete, menu?.pending, menu?.error])
  const inputRef = useRef<HTMLInputElement>(null)
  useEffect(() => {
    if (renaming) inputRef.current?.select()
  }, [renaming])
  if (!menu) return null
  const submitRename = (): void => {
    void store.renameSession(menu.key, draft)
  }
  return (
    <>
      <div
        className="backdrop backdrop--clear"
        onClick={() => store.closeSessionMenu()}
        onContextMenu={event => { event.preventDefault(); store.closeSessionMenu() }}
      />
      <div ref={menuRef} onKeyDown={event => {
        if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); store.closeSessionMenu(); return }
        if (renaming) return
        if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Home' || event.key === 'End') {
          event.preventDefault()
          const buttons = Array.from(menuRef.current?.querySelectorAll<HTMLButtonElement>('button') ?? [])
          const current = buttons.indexOf(document.activeElement as HTMLButtonElement)
          const next = event.key === 'Home' ? 0 : event.key === 'End' ? buttons.length - 1 : (current + (event.key === 'ArrowDown' ? 1 : buttons.length - 1)) % buttons.length
          buttons[next]?.focus()
        }
      }} className={'menu' + (renaming ? ' menu--renaming' : '')} role={renaming ? 'form' : confirmingDelete ? 'dialog' : 'menu'} aria-label={renaming ? 'Rename session' : confirmingDelete ? 'Confirm delete session' : 'Session actions'} style={{ left: menu.x, top: menu.y }}>
        {renaming ? (
          <div className="menu__rename">
            <input
              ref={inputRef}
              value={draft}
              disabled={menu.pending}
              onChange={e => setDraft(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter' || e.key === 'Escape') {
                  e.preventDefault()
                  e.stopPropagation()
                  if (e.key === 'Enter') submitRename()
                  else store.closeSessionMenu()
                }
              }}
              aria-label="Session title"
              spellCheck={false}
            />
            <button type="button" className="btn btn--solid" onClick={submitRename} disabled={menu.pending || !draft.trim()} aria-label="Save session title">{menu.pending ? 'Saving…' : 'Save'}</button>
          </div>
        ) : (
          <>
            <button type="button" role="menuitem" className="menu__item" onClick={() => { store.closeSessionMenu(); void store.openSession(menu.id) }}>
              <Icon name="arrow" size={13} /> Open <kbd className="kbd">⏎</kbd>
            </button>
            <button type="button" role="menuitem" className="menu__item" onClick={() => setRenaming(true)}>
              <Icon name="note" size={13} /> Rename…
            </button>
            <div className="menu__sep" />
            {(() => {
              const inboxState = chrome.inbox?.state
              const pinned = inboxState?.pinnedIds.includes(menu.id) ?? false
              const settled = inboxState?.settledIds.includes(menu.id) ?? false
              const snoozed = inboxState ? isSnoozedActive(inboxState.snoozed, menu.id) : false
              const title = menu.title || 'Thread'
              return (
                <>
                  <button
                    type="button"
                    role="menuitem"
                    className="menu__item"
                    onClick={() => {
                      if (chrome.inbox) chrome.inbox.togglePin(menu.id, title)
                      else setThreadInboxState(current => togglePinInState(current, menu.id))
                      store.closeSessionMenu()
                    }}
                  >
                    <T3Icon name={pinned ? 'pinOff' : 'pin'} size={13} /> {pinned ? 'Unpin' : 'Pin'}
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    className="menu__item"
                    onClick={() => {
                      const wakeAt = snoozed ? null : snoozePresets()[0]!.wakeAt
                      if (chrome.inbox) chrome.inbox.snooze(menu.id, title, wakeAt)
                      else setThreadInboxState(current => setSnoozedInState(current, menu.id, wakeAt))
                      store.closeSessionMenu()
                    }}
                  >
                    <T3Icon name="alarmClock" size={13} /> {snoozed ? 'Wake now' : 'Snooze 1 hour'}
                  </button>
                  <button
                    type="button"
                    role="menuitem"
                    className="menu__item"
                    onClick={() => {
                      if (chrome.inbox) chrome.inbox.settle(menu.id, title, !settled)
                      else setThreadInboxState(current => setSettledInState(current, menu.id, !settled))
                      store.closeSessionMenu()
                    }}
                  >
                    <T3Icon name={settled ? 'undo' : 'check'} size={13} /> {settled ? 'Restore to Inbox' : 'Settle'}
                  </button>
                  <div className="menu__sep" />
                </>
              )
            })()}
            <button type="button" role="menuitem" className="menu__item" onClick={() => { copyText(menu.id); store.closeSessionMenu() }}>
              <Icon name="copy" size={13} /> Copy ID
            </button>
            <button type="button" role="menuitem" className="menu__item" onClick={() => { void store.exportSessionTranscript(menu.key) }}>
              <Icon name="download" size={13} /> Export as Markdown
            </button>
            <div className="menu__sep" />
            {confirmingDelete || menu.pending ? (
              <>
                <p className="session-menu-confirm" role="alert">Delete “{menu.title || 'this chat'}”? Its saved history is removed permanently.</p>
                <button type="button" className="menu__item menu__item--danger" disabled={menu.pending} onClick={() => { void store.deleteSession(menu.id) }}>
                  <Icon name="trash" size={13} /> {menu.pending ? 'Deleting…' : 'Delete permanently'}
                </button>
                <button type="button" className="menu__item" disabled={menu.pending} onClick={() => setConfirmingDelete(false)}>
                  Cancel
                </button>
              </>
            ) : (
              <button type="button" role="menuitem" className="menu__item menu__item--danger" onClick={() => setConfirmingDelete(true)}>
                <Icon name="trash" size={13} /> Delete…
              </button>
            )}
          </>
        )}
        {menu.error && <p className="session-menu-error" role="alert">{menu.error}</p>}
      </div>
    </>
  )
}

// ── Statusline ──────────────────────────────────────────────────────────

// ── Sidebar ─────────────────────────────────────────────────────────────

/** Tasks, Agents and XERXES: the three sidebar destinations. */
export function SidebarViews({ page, open }: { page: DesktopPageName | null; open: (panel: DesktopPanel) => void }): ReactElement {
  return <div className="studio-segment" role="group" aria-label="Sidebar view"><button type="button" aria-pressed={page === null} className={page === null ? 'is-selected' : ''} onClick={() => open(null)}>Tasks</button><button type="button" aria-pressed={page === 'agents'} className={page === 'agents' ? 'is-selected' : ''} onClick={() => open('agents')}>Agents</button><button type="button" aria-pressed={page === 'xerxes'} className={page === 'xerxes' ? 'is-selected' : ''} onClick={() => open('xerxes')}>XERXES</button></div>
}

export function Sidebar({ snap, page }: { snap: Snapshot; page: DesktopPageName | null }): ReactElement {
  const open = useDesktopNavigation()
  const chrome = useContext(WindowChrome)
  const [filter, setFilter] = useState('')
  const needle = filter.trim().toLowerCase()
  // Folded sections show only their name (remembered); expanded ones list every task, not just the last five.
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(loadCollapsedSections)
  const [showAll, setShowAll] = useState<ReadonlySet<string>>(() => new Set())
  const toggleCollapsed = (key: string): void => setCollapsed(previous => {
    const next = new Set(previous)
    if (!next.delete(key)) next.add(key)
    saveCollapsedSections(next)
    return next
  })
  const toggleShowAll = (key: string): void => setShowAll(previous => {
    const next = new Set(previous)
    if (!next.delete(key)) next.add(key)
    return next
  })
  // Match what the row DISPLAYS: enriched snippets replace raw titles, and
  // filtering on the hidden '#shortid' would drop the row the eye found.
  const shownTitle = (row: { id: string }): string =>
    snap.snippets[row.id] ?? (snap.live.find(l => l.id === row.id) ?? snap.sessions.find(s => s.id === row.id))?.title ?? ''
  const match = (row: { title: string; id: string }): boolean =>
    !needle
    || shownTitle(row).toLowerCase().includes(needle)
    || row.title.toLowerCase().includes(needle)
    || row.id.includes(needle)
  // Mockup 07: the CURRENT task stays in its workspace group, marked and
  // carrying live status — derived from the snapshot so it updates the
  // instant a turn starts/ends instead of lagging behind list refreshes.
  const currentRow: Snapshot['live'][number] | null = snap.currentId
    ? {
        id: snap.currentId,
        // Key-scoped RPCs (rename) must address the connection's bound key,
        // not the session id — the daemon binds resumed sessions by key.
        key: snap.sessionKey || snap.currentId,
        title: snap.currentTitle || 'New task',
        status: snap.turnActive ? 'working' : snap.turnFailed ? 'failed' : 'idle',
        age: snap.turnActive ? elapsedOf(snap.turnSeconds, true) : '',
        current: true,
        kind: 'main',
        turns: snap.turnCount,
        messages: 0,
        cwd: snap.cwd,
        untitled: false,
      }
    : null
  // One order, by latest message. Opening a chat is not activity, so a click
  // never moves anything: the open chat used to be appended at the end, chats
  // the daemon had open were listed ahead of the rest, and the open chat's
  // folder was hoisted to the top — one click could reshuffle the sidebar.
  // A chat with no message yet (a fresh New task) sorts first.
  const rows = sidebarOrder([
    ...snap.live.filter(row => row.id !== snap.currentId),
    ...snap.sessions.filter(row => row.id !== snap.currentId),
    ...(currentRow ? [currentRow] : []),
  ], snap.sessionActivity ?? {}).filter(match)
  const groups = groupByWorkspace(rows, '', snap.workspaceDirectories)
  // One list for every host and folder, in an order no click changes.
  const sections = sidebarSections(groups, snap.contexts ?? [], snap.contextHost ?? hostOfScope(snap.storageScope))
  const online = snap.connection === 'online'
  const vibes = useVibesStyle()
  const fallbackInbox = useThreadInbox()
  const inbox = chrome.inbox ?? fallbackInbox

  if (vibes) {
    const activeExternal = isExternalAcpBinding(snap.acpBinding)
      ? snap.acpBinding
      : snap.savedAcpBinding && snap.savedAcpBinding.transport !== 'native'
        ? snap.savedAcpBinding
        : null
    const activeProvider = activeExternal ? activeExternal.provider : 'xerxes' as const
    const t3Rows: T3SidebarRow[] = [
      ...rows.map(row => ({
        id: row.id,
        key: row.key,
        title: row.title,
        ...(snap.snippets[row.id] !== undefined ? { snippet: snap.snippets[row.id] } : {}),
        status: row.status,
        age: row.age,
        current: row.current,
        turns: row.turns,
        cwd: row.cwd || snap.cwd,
        ...(row.cwd === snap.cwd && snap.branch ? { branch: snap.branch } : {}),
        provider: row.current ? activeProvider : 'xerxes',
        opensWindow: snap.turnActive && row.id !== snap.currentId,
      })),
      ...sections
        .filter((section): section is Extract<typeof section, { kind: 'context' }> => section.kind === 'context')
        .flatMap(section =>
          section.rows
            .filter(row => !needle || row.title.toLowerCase().includes(needle))
            .map(row => ({
              id: row.id,
              key: row.id,
              title: row.title,
              status: row.status,
              age: '',
              current: false,
              turns: 0,
              cwd: row.cwd || section.cwd,
              host: section.host,
              contextId: section.context.id,
            })),
        ),
    ]
    return (
      <T3Sidebar
        snap={snap}
        page={page}
        sections={sections}
        allRows={t3Rows}
        filter={filter}
        setFilter={setFilter}
        inbox={inbox.state}
        undo={inbox.undo}
        onUndo={() => {
          const entry = inbox.undo
          inbox.applyUndo()
          if (entry?.verb === 'Removed' && entry.threadId) {
            void store.restoreWorkspace(entry.threadId)
          }
        }}
        onDismissUndo={inbox.dismissUndo}
        onSetViewMode={inbox.setViewMode}
        onToggleProjectScope={inbox.toggleProjectScope}
        onClearProjectScope={inbox.clearProjectScope}
        onTogglePin={inbox.togglePin}
        onSettle={inbox.settle}
        onSnooze={inbox.snooze}
        onToggleShelf={inbox.toggleShelf}
        onRemoveWorkspace={(cwd, label) => {
          inbox.hideWorkspace(cwd, label)
          void store.removeWorkspace(cwd)
        }}
        onRemoveRemoteHost={alias => {
          void store.removeRemoteHost(alias)
        }}
        canGoBack={chrome.canGoBack ?? false}
        canGoForward={chrome.canGoForward ?? false}
        onGoBack={chrome.goBack ?? (() => {})}
        onGoForward={chrome.goForward ?? (() => {})}
        onOpen={open}
        sidebarToggle={<SidebarToggle />}
        runtimeStatus={<RuntimeStatus snap={snap} />}
      />
    )
  }

  return (
    <aside id="session-sidebar" className="side" aria-label="Sessions">
      <div className="side__head"><SidebarToggle />{vibes && <VibesModeSwitch page={page} />}</div>
      {vibes ? <div className="side__pad vibes-side"><VibesSideNav snap={snap} page={page} filter={filter} setFilter={setFilter} /></div> : <div className="side__pad">
        <div className="sidebar-wordmark" aria-label="Xerxes">XERXES</div>
        {/* `page !== 'agents'` lit Sessions while Skills & tools or
            Artifacts was open, so two destinations claimed to be current. */}
        <SidebarViews page={page} open={open} />
        <button
          className="newchat"
          disabled={!online}
          onClick={() => { if (!snap.turnActive) open(null); store.newChat() }}
          title={homeLabel(snap) ? `Start a new task in ${homeLabel(snap)} (⌘N, or ⇧⌘N to choose a preset and worktree)` : 'Start a new task (⌘N)'}
        ><Icon name="plus" /><span>New task</span><kbd>⌘ N</kbd></button>
        <button className={`studio-nav${page === "extensions" ? " is-selected" : ""}`} aria-pressed={page === 'extensions'} disabled={!online} onClick={() => open('extensions')}><Icon name="tools" /><span>Skills & tools</span><LearnedBadge snap={snap} /></button>
        <button className={`studio-nav${page === "schedules" ? " is-selected" : ""}`} aria-pressed={page === 'schedules'} disabled={!online} onClick={() => open('schedules')}><Icon name="clock" /><span>Scheduled jobs</span></button>
        <input
          className="side__search"
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder="Filter tasks…"
          disabled={!online}
          aria-label="Filter tasks in this list"
          spellCheck={false}
        />
        <button
          className="side__find"
          disabled={!online}
          title="Full-text search across the messages of every saved task"
          onClick={() => store.openSessionSearch()}
        >Search message history <Icon name="arrow" size={12} /></button>
      </div>}
      <nav className="side__list">
        {sections.map(section => {
          // A filter searches every task, folded or not.
          const folded = !needle && collapsed.has(section.key)
          const all = Boolean(needle) || showAll.has(section.key)
          const availability = section.kind === 'context' && section.context.availability && section.context.availability !== 'online'
            ? section.context.availability === 'no-runtime' ? 'no runtime' : section.context.availability
            : ''
          const reason = section.kind === 'context' ? section.context.reason ?? '' : ''
          const head = (
            <div className="wgroup__head">
              <button
                className="wgroup__cap"
                aria-expanded={!folded}
                title={`${[section.cwd, reason].filter(Boolean).join(' · ')} — ${folded ? 'show' : 'hide'} its tasks`}
                onClick={() => toggleCollapsed(section.key)}
              >
                {section.kind === 'here' && <span className="wgroup__mark" data-here={section.cwd === snap.cwd || undefined} aria-hidden="true" />}
                {section.label}
                {availability && <span className="sess__s"> · {availability}</span>}
                {section.kind === 'here' && section.cwd === snap.cwd && <span className="wgroup__cur">current</span>}
                <span className="wgroup__caret" aria-hidden="true"><Icon name="caretDown" size={11} /></span>
              </button>
              {section.kind === 'context'
                ? <button className="wgroup__add" title={`Open ${section.label}`} aria-label={`Open ${section.label}`} onClick={() => void store.activateContext(section.context.id)}><Icon name="arrow" size={14} /></button>
                : <button className="wgroup__add" disabled={!online} title={`New task in ${section.label}`} aria-label={`New task in ${section.label}`} onClick={() => store.newChatIn(section.cwd)}><Icon name="plus" size={14} /></button>}
            </div>
          )
          const more = (hidden: number, total: number): ReactElement | null => !needle && (hidden > 0 || all && total > SIDEBAR_SECTION_ROWS)
            ? <button className="wgroup__more" onClick={() => toggleShowAll(section.key)}>{hidden > 0 ? `Show ${hidden} more` : 'Show less'}</button>
            : null
          if (section.kind === 'context') {
            const listed = section.rows.filter(row => !needle || row.title.toLowerCase().includes(needle))
            const { shown, hidden } = limitRows(listed, SIDEBAR_SECTION_ROWS, all)
            return (
              <div className="wgroup" key={section.key} data-collapsed={folded || undefined}>
                {head}
                {!folded && <>
                  {reason && <div className="sess__s">{reason}</div>}
                  {shown.map(row => (
                    <button className="sess" key={row.id} title={`${section.label} · ${row.cwd}`} onClick={() => void store.activateContext(section.context.id, row.id)}>
                      <span className="sess__dot" data-status={row.status} style={{ background: statusColor(row.status) }} />
                      <span className="sess__body"><span className="sess__t">{row.title}</span><span className="sess__s">{workspaceLabel(row.cwd)}{row.status === 'working' ? ' · working' : ''}</span></span>
                    </button>
                  ))}
                  {more(hidden, listed.length)}
                </>}
              </div>
            )
          }
          const { shown, hidden } = limitRows(section.rows, SIDEBAR_SECTION_ROWS, all, snap.currentId)
          return (
            <div key={section.key} className={`wgroup${section.cwd === snap.cwd ? ' is-home' : ''}`} data-collapsed={folded || undefined}>
              {head}
              {!folded && <>
                {shown.map(row => {
                  const snippet = snap.snippets[row.id]
                  return snippet === undefined
                    ? <SessionCell key={row.id} row={row} opensWindow={snap.turnActive && row.id !== snap.currentId} />
                    : <SessionCell key={row.id} row={row} snippet={snippet} opensWindow={snap.turnActive && row.id !== snap.currentId} />
                })}
                {more(hidden, section.rows.length)}
              </>}
            </div>
          )
        })}
        {/* A zero-match filter used to fall through to "No tasks yet",
            which reads as "this workspace is empty" rather than "your
            query matched nothing". */}
        {groups.length === 0 && (
          <div className="side__empty">{needle ? <>No task matches “{filter.trim()}”. <button className="linkish" onClick={() => setFilter('')}>Clear filter</button></> : snap.noWorkspace ? snap.storageScope?.startsWith('ssh:') ? 'Remote tasks appear after connecting' : 'Your tasks will appear here' : online ? 'No tasks yet — they are stored inside the workspace folder' : connectionFailureKind(snap.error) === 'transport' ? 'Connecting to the runtime…' : 'Workspace needs attention'}</div>
        )}
        <button className="addws" onClick={() => store.chooseWorkspace()} title="Choose another folder to open as a workspace">
          <Icon name="plus" size={12} /> Add folder…
        </button>
      </nav>
      {vibes
        ? <div className="studio-side-bottom vibes-foot"><RuntimeStatus snap={snap} /><button className="vibes-foot__settings" title="Settings" aria-label="Settings" onClick={() => store.openSettings()}><Icon name="settings" /></button></div>
        : <div className="studio-side-bottom"><RuntimeStatus snap={snap} /><button onClick={() => store.openSettings()}><Icon name="settings" /><span>Settings</span></button><button onClick={() => open('workspace')}><Icon name="folder" /><span>{homeLabel(snap) && snap.cwd ? homeLabel(snap) : 'Choose a folder…'}</span></button></div>}
    </aside>
  )
}

/**
 * Vibes only: Claude's Chat | Code switch at the top right of the sidebar,
 * as XERXES | Code. XERXES is its own full view (the supervisor), Code is the
 * task workspace. Both keep the existing navigation handlers.
 */
export function VibesModeSwitch({ page }: { page: DesktopPageName | null }): ReactElement {
  const open = useDesktopNavigation()
  const xerxes = page === 'xerxes'
  return <div className="vibes-mode" role="group" aria-label="Mode">
    <button type="button" aria-pressed={xerxes} className={xerxes ? 'is-selected' : ''} title="XERXES, the supervisor across every workspace" aria-label="XERXES" onClick={() => open('xerxes')}><Icon name="spark" size={16} /></button>
    <button type="button" aria-pressed={!xerxes} className={xerxes ? '' : 'is-selected'} title="Code: tasks in this workspace" aria-label="Code" onClick={() => open(null)}><Icon name="code" size={16} /></button>
  </div>
}

/**
 * Vibes only: Claude Code's five sidebar rows (New, Search, Artifacts,
 * Customize, More). Xerxes-only destinations keep their handlers and move
 * under More; the task filter opens from there.
 */
export function VibesSideNav({ snap, page, filter, setFilter, defaultMoreOpen = false }: {
  snap: Snapshot
  page: DesktopPageName | null
  filter: string
  setFilter: (value: string) => void
  /** Tests render the menu open; the app always starts closed. */
  defaultMoreOpen?: boolean
}): ReactElement {
  const open = useDesktopNavigation()
  const online = snap.connection === 'online'
  const [moreOpen, setMoreOpen] = useState(defaultMoreOpen)
  const [filtering, setFiltering] = useState(false)
  const menu = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  useEffect(() => {
    if (!moreOpen) return
    const dismiss = (event: MouseEvent) => {
      if (event.target instanceof Node && !menu.current?.contains(event.target) && !trigger.current?.contains(event.target)) setMoreOpen(false)
    }
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') { event.preventDefault(); setMoreOpen(false); trigger.current?.focus() }
    }
    document.addEventListener('mousedown', dismiss)
    document.addEventListener('keydown', escape)
    return () => { document.removeEventListener('mousedown', dismiss); document.removeEventListener('keydown', escape) }
  }, [moreOpen])
  const pick = (run: () => void) => () => { setMoreOpen(false); run() }
  const row = (selected: boolean): string => `vibes-nav__row${selected ? ' is-selected' : ''}`
  return (
    <nav className="vibes-nav" aria-label="Navigation">
      <button
        className="vibes-nav__row"
        disabled={!online}
        onClick={() => { if (!snap.turnActive) open(null); store.newChat() }}
        title={homeLabel(snap) ? `Start a new task in ${homeLabel(snap)} (⌘N)` : 'Start a new task (⌘N)'}
      ><Icon name="plus" /><span>New</span></button>
      <button className="vibes-nav__row" disabled={!online} title="Search the messages of every saved task" onClick={() => store.openSessionSearch()}><Icon name="search" /><span>Search</span></button>
      <button className={row(page === 'artifacts')} aria-pressed={page === 'artifacts'} disabled={!online} onClick={() => open('artifacts')}><Icon name="stack" /><span>Artifacts</span></button>
      <button className={row(page === 'extensions')} aria-pressed={page === 'extensions'} disabled={!online} title="Skills & tools" onClick={() => open('extensions')}><Icon name="tools" /><span>Customize</span><LearnedBadge snap={snap} /></button>
      <div className="vibes-nav__anchor">
        <button ref={trigger} className={row(page === 'agents' || page === 'schedules')} aria-haspopup="menu" aria-expanded={moreOpen} onClick={() => setMoreOpen(value => !value)}><Icon name="dots" /><span>More</span></button>
        {moreOpen && <div ref={menu} className="vibes-nav__menu" role="menu" aria-label="More">
          <button role="menuitem" aria-current={page === null || undefined} onClick={pick(() => open(null))}><Icon name="chat" /><span>Tasks</span></button>
          <button role="menuitem" aria-current={page === 'agents' || undefined} onClick={pick(() => open('agents'))}><Icon name="agents" /><span>Agents</span></button>
          {showsImmortalsForSession(snap) && <button role="menuitem" disabled={!online} onClick={pick(() => open('swarm'))}><Icon name="tree" /><span>Immortals</span></button>}
          <button role="menuitem" disabled={!online} onClick={pick(() => open('btw'))}><Icon name="chat" /><span>Side question</span></button>
          <button role="menuitem" aria-current={page === 'schedules' || undefined} disabled={!online} onClick={pick(() => open('schedules'))}><Icon name="clock" /><span>Scheduled jobs</span></button>
          <hr />
          <button role="menuitem" disabled={!online} onClick={pick(() => setFiltering(true))}><Icon name="search" /><span>Filter tasks…</span></button>
          <button role="menuitem" onClick={pick(() => open('workspace'))}><Icon name="folder" /><span>{homeLabel(snap) && snap.cwd ? `Workspace · ${homeLabel(snap)}` : 'Choose a folder…'}</span></button>
        </div>}
      </div>
      {(filtering || filter) && <input
        className="side__search"
        value={filter}
        autoFocus
        onChange={e => setFilter(e.target.value)}
        onKeyDown={e => { if (e.key === 'Escape') { e.preventDefault(); setFilter(''); setFiltering(false) } }}
        placeholder="Filter tasks…"
        disabled={!online}
        aria-label="Filter tasks in this list"
        spellCheck={false}
      />}
    </nav>
  )
}

const homeLabel = (snap: Snapshot): string => workspaceLabel(snap.cwd)

function SessionCell({
  row,
  snippet,
  opensWindow = false,
}: {
  row: Snapshot['live'][number]
  snippet?: string
  /** Inspect another session in its own window while this turn continues. */
  opensWindow?: boolean
}): ReactElement {
  const title = snippet ?? row.title
  const density = useSyncExternalStore(subscribeSessionDensity, currentSessionDensity, () => 'comfortable' as const)
  // dsh row grammar: title + right-aligned age on the first line, a status
  // subline only when it says something the age doesn't.
  const sub = [
    row.status === 'working' ? 'acting' : row.status === 'failed' ? 'failed' : '',
  ].filter(Boolean).join(' · ')
  return (
    <button
      className={`sess${row.current ? ' is-current' : ''}`}
      title={opensWindow ? 'Open this session while the current task continues' : undefined}
      aria-haspopup="menu"
      onClick={() => store.openSession(row.id)}
      onKeyDown={event => {
        if (event.key === 'ContextMenu' || event.key === 'F10' && event.shiftKey) {
          event.preventDefault()
          const rect = event.currentTarget.getBoundingClientRect()
          store.openSessionMenu({ id: row.id, key: row.key, title: row.title }, rect.left, rect.bottom)
        }
      }}
      onContextMenu={event => {
        event.preventDefault()
        store.openSessionMenu(
          { id: row.id, key: row.key, title: row.title },
          event.clientX,
          event.clientY,
        )
      }}
    >
      <span className="sess__dot" data-status={row.status} style={{ background: statusColor(row.status) }} />
      <span className="sess__body">
        <span className="sess__t">{title}</span>
        {sub && <span className="sess__s">{sub}</span>}
        {/* Only at the Detailed session-list density (Settings): the
            default row stays free of idle turn counts. */}
        {density === 'detailed' && row.turns > 0 && <span className="sess__detail">{row.turns} turn{row.turns === 1 ? '' : 's'}{row.cwd ? ` · ${workspaceLabel(row.cwd)}` : ''}</span>}
      </span>
      {row.age && <span className="sess__age">{row.age}</span>}
    </button>
  )
}
// ── Chat column ─────────────────────────────────────────────────────────



function Chat({ snap, page, agent = '', openAgent = () => {} }: { snap: Snapshot; page: DesktopPageName | null; agent?: string; openAgent?: (id: string) => void }): ReactElement {
  const open = useDesktopNavigation()
  const chrome = useContext(WindowChrome)
  const vibes = useVibesStyle()
  // DeepSeek Harness's layout: the header counts the task's sub-agents and
  // switches between them; an open one replaces the conversation below.
  const fleet = useMemo(() => activityFleetRows(snap.fleet, snap.blocks), [snap.fleet, snap.blocks])
  const agentRow = agent ? fleet.find(row => row.id === agent || row.agentDetails?.requestKey === agent) : undefined
  const working = fleet.filter(row => agentState(row.status).priority < 2).length
  const taskTitle = snap.currentTitle || (snap.connection === 'online' ? 'New thread' : 'Not connected')
  const projectLabel = workspaceLabel(snap.cwd)
  const [renamingTitle, setRenamingTitle] = useState(false)
  const [titleDraft, setTitleDraft] = useState(taskTitle)
  useEffect(() => { setRenamingTitle(false); setTitleDraft(taskTitle) }, [snap.currentId, taskTitle])
  const commitTitleRename = (): void => {
    const trimmed = titleDraft.trim()
    setRenamingTitle(false)
    if (!trimmed || trimmed === taskTitle || !snap.currentId) return
    void store.renameSession(snap.sessionKey || snap.currentId, trimmed)
  }
  const changes = snap.changes
  const totals = useMemo(() => ({
    adds: changes.reduce((sum, file) => sum + file.adds, 0),
    dels: changes.reduce((sum, file) => sum + file.dels, 0),
  }), [changes])
  const needsInput = snap.approval !== null || snap.question !== null
  // A page (XERXES, Agents, Skills & tools, Artifacts, Scheduled jobs) is
  // its own place: the task's title, status line and steer box under it read
  // as part of the page, so the header names the page and the composer hides.
  const onPage = page !== null

  return (
    <main className="chat">
      <header className={`chat__head chat__head--minimal${vibes ? ' t3-topbar' : ''}`} data-workspace-header={vibes ? 'true' : undefined}>
        {!chrome.sidebarVisible && <SidebarToggle />}
        {vibes ? (
          agent && !onPage ? (
            <nav className="t3-breadcrumb agent-crumb" aria-label="Sub-agent">
              {snap.cwd && (
                <>
                  <button type="button" className="t3-breadcrumb__project" title={snap.cwd} onClick={() => store.openWorkspaceMenu()}>
                    <ProjectMonogram name={projectLabel} seed={snap.cwd} size={16} />
                    <span>{projectLabel}</span>
                  </button>
                  <span className="t3-breadcrumb__sep" aria-hidden="true">/</span>
                </>
              )}
              <button type="button" className="agent-crumb__parent" title="Back to the task's conversation" onClick={() => openAgent('')}>{taskTitle}</button>
              <span className="t3-breadcrumb__sep" aria-hidden="true">/</span>
              <AgentMenu rows={fleet} {...(agentRow ? { currentId: agentRow.id } : {})} label={agentRow?.title ?? 'Agent'} title="Switch agent" onPick={openAgent} />
            </nav>
          ) : onPage ? (
            <nav className="t3-breadcrumb" aria-label="Page breadcrumb">
              {snap.cwd && (
                <>
                  <button
                    type="button"
                    className="t3-breadcrumb__project"
                    title={`Back to ${projectLabel}`}
                    onClick={() => open(null)}
                  >
                    <ProjectMonogram name={projectLabel} seed={snap.cwd} size={16} />
                    <span>{projectLabel}</span>
                  </button>
                  <span className="t3-breadcrumb__sep" aria-hidden="true">/</span>
                </>
              )}
              <span className="chat__title t3-breadcrumb__title">{page === 'extensions' && isExternalAcpBinding(snap.acpBinding) ? 'Customize' : PAGE_TITLES[page]}</span>
            </nav>
          ) : (
            <nav className="t3-breadcrumb" aria-label="Workspace breadcrumb">
              {snap.cwd && (
                <>
                  <button
                    type="button"
                    className="t3-breadcrumb__project"
                    title={`${snap.cwd} — switch project`}
                    onClick={() => store.openWorkspaceMenu()}
                  >
                    <ProjectMonogram name={projectLabel} seed={snap.cwd} size={16} />
                    <span>{projectLabel}</span>
                  </button>
                  <span className="t3-breadcrumb__sep" aria-hidden="true">/</span>
                </>
              )}
              {renamingTitle ? (
                <input
                  className="t3-breadcrumb__rename"
                  autoFocus
                  spellCheck={false}
                  aria-label="Rename thread"
                  value={titleDraft}
                  onChange={event => setTitleDraft(event.target.value)}
                  onBlur={commitTitleRename}
                  onKeyDown={event => {
                    if (event.key === 'Enter') { event.preventDefault(); commitTitleRename() }
                    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); setRenamingTitle(false); setTitleDraft(taskTitle) }
                  }}
                />
              ) : (
                <>
                  <button
                    type="button"
                    className="chat__title t3-breadcrumb__thread"
                    title={snap.currentId ? 'Double-click to rename thread' : undefined}
                    onDoubleClick={() => { if (snap.currentId) { setTitleDraft(taskTitle); setRenamingTitle(true) } }}
                    onContextMenu={event => {
                      if (!snap.currentId) return
                      event.preventDefault()
                      store.openSessionMenu({ id: snap.currentId, key: snap.sessionKey || snap.currentId, title: taskTitle }, event.clientX, event.clientY)
                    }}
                  >
                    <span className="t3-breadcrumb__thread-text">{taskTitle}</span>
                  </button>
                  {snap.currentId && (
                    <button
                      type="button"
                      className="t3-icon-btn t3-breadcrumb__actions"
                      aria-label="Thread actions"
                      aria-haspopup="menu"
                      title="Thread actions"
                      onClick={event => {
                        const rect = event.currentTarget.getBoundingClientRect()
                        store.openSessionMenu({ id: snap.currentId, key: snap.sessionKey || snap.currentId, title: taskTitle }, rect.left, rect.bottom + 4)
                      }}
                    >
                      <T3Icon name="chevronDown" size={12} />
                    </button>
                  )}
                </>
              )}
            </nav>
          )
        ) : (
          agent && !onPage ? <nav className="agent-crumb" aria-label="Sub-agent">
            <button className="agent-crumb__parent" title="Back to the task's conversation" onClick={() => openAgent('')}>{taskTitle}</button>
            <span className="agent-crumb__sep" aria-hidden="true">/</span>
            <AgentMenu rows={fleet} {...(agentRow ? { currentId: agentRow.id } : {})} label={agentRow?.title ?? 'Agent'} title="Switch agent" onPick={openAgent} />
          </nav> : <span className="chat__title">{page ? (page === 'extensions' && isExternalAcpBinding(snap.acpBinding) ? 'Customize' : PAGE_TITLES[page]) : taskTitle}</span>
        )}
        {!onPage && !agent && fleet.length > 0 && <AgentMenu variant="count" rows={fleet} label={<>{fleet.length} agent{fleet.length === 1 ? '' : 's'}{working > 0 && <span className="agent-menu__live">{working} working</span>}</>} title="Open a sub-agent" onPick={openAgent} />}
        {!onPage && !agent && (isExternalAcpBinding(snap.acpBinding) ? (() => {
          const summary = activeAcpSummaryParts(snap.acpBinding).join(' · ')
          return (
            <span
              className="hchip chat__kind chat__acp"
              data-provider={snap.acpBinding.provider}
              title={`External runtime: ${snap.acpBinding.name} (${transportBadgeLabel(snap.acpBinding.transport)})${summary ? ` · ${summary}` : ''}`}
            >
              <RuntimeBrandIcon provider={snap.acpBinding.provider} size={12} />
              <span>{snap.acpBinding.name} · {transportBadgeLabel(snap.acpBinding.transport)}</span>
            </span>
          )
        })() : (
          !vibes && <span className="hchip chat__kind" title="Agent preset for this session · fixed after its first turn">{snap.currentAgentPreset === 'creator' ? 'Creator mode' : (snap.currentAgentPreset || 'default')}</span>
        ))}
        <span className="composer__flex" />
        {/* On Session edits or the plan, the approval card is off screen: this is the signal. */}
        {needsInput && <button className="badge badge--need" title={snap.interactionHidden ? 'Show the request waiting for you' : undefined} onClick={() => { store.showInteraction(); store.setTab('activity'); if (page) open(null); if (agent) openAgent('') }}>needs input{snap.turnActive ? ' · acting paused' : ''}</button>}
        {page === null && snap.tab !== 'activity' && (
          <button
            type="button"
            className="chat__view chat__back"
            aria-label="Back to conversation"
            title="Back to conversation (Esc)"
            onClick={() => store.setTab('activity')}
          >
            <span aria-hidden="true">←</span> Conversation
          </button>
        )}
        {!onPage && !agent && snap.swarm?.active && (() => {
          const swarmCounts = swarmMemberCounts(snap.swarm.members)
          return (
            <button
              type="button"
              className="chat__view chat__swarm"
              data-working={swarmCounts.working > 0 ? 'true' : undefined}
              title="Open the Immortals Board"
              aria-label="Open Immortals Board"
              onClick={() => open('swarm')}
            >
              <Icon name="tree" size={13} /> Immortals{' '}
              <span className="pillcount">
                {swarmCounts.working > 0 ? `${swarmCounts.working} working` : `${snap.swarm.members.length}`}
              </span>
            </button>
          )
        })()}
        {page === null && changes.length > 0 && <button className={`chat__view${snap.tab === 'changes' ? ' is-on' : ''}`} aria-pressed={snap.tab === 'changes'} onClick={() => store.setTab(snap.tab === 'changes' ? 'activity' : 'changes')}>
          Session edits <span className="pillcount"><span className="add">+{totals.adds}</span> <span className="del">−{totals.dels}</span></span>
        </button>}
        {vibes ? (
          !onPage && !snap.noWorkspace && (
            <div className="t3-panel-controls" data-panel-layout-controls="true" role="group" aria-label="Panel layout controls">
              <button
                type="button"
                className={`t3-icon-btn${chrome.detailsOpen ? ' is-selected' : ''}`}
                data-thread-details-toggle="true"
                aria-pressed={Boolean(chrome.detailsOpen)}
                aria-label="Toggle thread details"
                title="Thread details"
                onClick={() => chrome.toggleDetails?.()}
              >
                <T3Icon name="squareMenu" size={15} />
              </button>
              <button
                type="button"
                className={`t3-icon-btn${chrome.terminalDrawerOpen ? ' is-selected' : ''}`}
                aria-pressed={Boolean(chrome.terminalDrawerOpen)}
                aria-label="Toggle terminal drawer"
                title="Toggle terminal drawer (⌘J)"
                onClick={() => chrome.toggleTerminalDrawer?.()}
              >
                <T3Icon name="panelBottom" size={15} />
              </button>
              <button
                type="button"
                className={`t3-icon-btn${chrome.inspectorOpen ? ' is-selected' : ''}`}
                aria-pressed={chrome.inspectorOpen}
                aria-label="Toggle right panel"
                title="Toggle right panel (⌘\)"
                onClick={() => chrome.toggleRightPanel?.()}
              >
                <T3Icon name="panelRight" size={15} />
              </button>
            </div>
          )
        ) : (
          <>
            {!onPage && !agent && !snap.noWorkspace && <SectionsMenu snap={snap} />}
            {!chrome.sidebarVisible && <RuntimeStatus snap={snap} compact />}
            {!chrome.inspectorOpen && <WorkspaceTools snap={snap} inspectorOpen={false} />}
          </>
        )}
      </header>

      {!onPage && !agent && snap.connection !== 'online' && snap.blocks.length > 0 && <ConnectionBanner snap={snap} />}

      <div className="chat__body">
        {vibes && !onPage && !agent && chrome.detailsOpen && (
          <ThreadDetailsCard
            snap={snap}
            mode={chrome.detailsMode ?? 'popover'}
            onClose={() => chrome.toggleDetails?.()}
            onNavigate={open}
            onOpenTerminal={() => chrome.toggleTerminalDrawer?.()}
          />
        )}

        {page === 'xerxes' && <SupervisorPage />}
        {page && page !== 'xerxes' && <DesktopPage panel={page} snap={snap} />}
        {agent && !onPage && (agentRow
          ? <AgentInspector key={snap.sessionKey + ':' + agentRow.id} row={agentRow} rows={fleet} sessionKey={snap.sessionKey} online={snap.connection === 'online'} />
          : <section className="agent-inspector agent-pane" aria-label="Agent inspector"><div className="stream"><div className="stream__col">
              <p role="status">This agent is no longer available in this session.</p>
              <button className="btn" onClick={() => openAgent('')}>Back to the conversation</button>
            </div></div></section>)}
        <div className="conversation-surface" hidden={page !== null || Boolean(agent)}>
        {vibes && <div className="t3-topbar-fade" aria-hidden="true" />}
        {snap.tab === 'activity' && <Stream snap={snap} />}
        {snap.tab === 'changes' && <div className="workspace"><ChangesTab snap={snap} /></div>}
        {snap.tab === 'plan' && <div className="workspace"><PlanTab snap={snap} /></div>}
        {snap.tab === 'log' && <div className="workspace workspace--log"><LogTab snap={snap} /></div>}

        </div>
        {/* Hidden, not unmounted, so a half-typed task message survives a visit to a page. */}
        <div className="chat__composer-host" hidden={onPage || Boolean(agent)}><Composer snap={snap} /></div>
      </div>
      {vibes && !onPage && !agent && chrome.terminalDrawerOpen && (
        <TerminalDrawer
          key={`${snap.cwd}:${snap.sessionKey}`}
          snap={snap}
          height={chrome.terminalDrawerHeight ?? 280}
          onResize={height => chrome.setTerminalDrawerHeight?.(height)}
          onMoveToRightPanel={() => {
            chrome.toggleTerminalDrawer?.()
            open('terminal')
          }}
          onClose={() => chrome.toggleTerminalDrawer?.()}
        />
      )}
    </main>
  )
}

/**
 * The one connection surface that used to hardcode "Connection lost.
 * Retrying…". `beat()` refuses to retry configuration and session
 * rejections, so for those the banner promised a recovery that was never
 * coming — and the offer of "Retry now" made it worse.
 */
function ConnectionBanner({ snap }: { snap: Snapshot }): ReactElement {
  const kind = connectionFailureKind(snap.error)
  const retrying = snap.connection === 'connecting' || kind === 'transport'
  return (
    <div className="connection-status" role={retrying ? 'status' : 'alert'} aria-live="polite">
      <strong>
        {snap.navigating === 'fresh' ? 'Starting a new task…'
          : snap.navigating === 'session' ? 'Opening that task…'
          : snap.connection === 'connecting' ? 'Reconnecting…'
          : kind === 'transport' ? 'Connection lost. Retrying…'
          : kind === 'session' ? 'This session belongs to another workspace'
          : 'This workspace needs attention'}
      </strong>
      {snap.error && !snap.navigating && <span>{snap.error}</span>}
      {/* retryConnection() is inert during a deliberate navigation, so the
          button would have been a lie exactly when it looked most useful. */}
      {snap.navigating ? null : retrying
        ? <button onClick={() => store.retryConnection()}>Retry now</button>
        : <button onClick={() => store.openSettings()}>Open settings</button>}
    </div>
  )
}

// ── Activity stream ─────────────────────────────────────────────────────

/** Number of trailing transcript activity groups rendered eagerly before progressive reveal. */
export const TRANSCRIPT_TAIL_GROUPS = 50

/**
 * Subscribe a tail-windowed transcript container to top-scroll reveal and
 * in-page Find triggers (both DOM `Cmd/Ctrl+F` and native macOS `Edit > Find`,
 * which dispatches the custom `xerxes:find` event without a DOM `keydown`).
 */
export function attachTranscriptExpandListeners(
  el: Pick<HTMLElement, 'scrollTop' | 'addEventListener' | 'removeEventListener'> | null,
  win: Pick<Window, 'addEventListener' | 'removeEventListener'> | null,
  onExpandMore: () => void,
  onExpandAll: () => void,
): () => void {
  const onScroll = (): void => {
    if (el && el.scrollTop < 160) onExpandMore()
  }
  const onKeyDown = (event: KeyboardEvent): void => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'f') {
      onExpandAll()
    }
  }
  el?.addEventListener('scroll', onScroll, { passive: true })
  win?.addEventListener('keydown', onKeyDown)
  win?.addEventListener('xerxes:find', onExpandAll)
  return () => {
    el?.removeEventListener('scroll', onScroll)
    win?.removeEventListener('keydown', onKeyDown)
    win?.removeEventListener('xerxes:find', onExpandAll)
  }
}

const StreamGroupRow = memo(function StreamGroupRow({
  group,
  disclosed,
  active,
  turnActive,
  inlineApproval,
  approval,
  permissionMode,
  changes,
  cwd,
}: {
  readonly group: readonly Block[]
  readonly disclosed: boolean
  readonly active: boolean
  readonly turnActive: boolean
  readonly inlineApproval: boolean
  readonly approval: Snapshot['approval']
  readonly permissionMode: string
  readonly changes: Snapshot['changes']
  readonly cwd?: string
}): ReactElement {
  return (
    <div className="stream__group" data-history-anchor={group[0]!.id}>
      {disclosed
        ? <ActivityGroup blocks={group} active={active} turnActive={turnActive} />
        : group.map(block => <BlockView key={block.id} block={block} />)}
      {inlineApproval && approval && <ApprovalCard key={approval.id} approval={approval} policy={permissionMode} inline />}
      <FilesChangedCard group={group} changes={changes} {...(cwd ? { cwd } : {})} />
    </div>
  )
})

function sameBlockList(a: readonly Block[] | undefined, b: readonly Block[]): boolean {
  if (!a || a.length !== b.length) return false
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) return false
  }
  return true
}

function Stream({ snap }: { snap: Snapshot }): ReactElement {
  const sessionScope = `${snap.currentId}:${snap.sessionOpenRevision}`
  const [windowLimit, setWindowLimit] = useState(TRANSCRIPT_TAIL_GROUPS)
  useEffect(() => {
    setWindowLimit(TRANSCRIPT_TAIL_GROUPS)
  }, [sessionScope])

  // 'Stream thinking' off hides reasoning trails from the feed — the daemon
  // still streams them; this is a display choice, not a policy change.
  const visible = useMemo(
    () => (snap.streamThinking ? snap.blocks : snap.blocks.filter(b => b.kind !== 'thinking')),
    [snap.streamThinking, snap.blocks],
  )
  const blocks = useMemo(
    () => (snap.failed && !snap.turnActive ? withoutDuplicateFailure(visible, snap.failed.error) : visible),
    [snap.failed, snap.turnActive, visible],
  )
  const offline = snap.connection === 'offline'
  const existingWork = Boolean(parseGoal(snap.goal) || snap.fleet.length || snap.todos?.length || snap.plan)
  const empty = snap.connection === 'online' && blocks.every(block => block.kind === 'notice' && !block.error) && !snap.turnActive
  const ends = useMemo(() => replyEnds(blocks, snap.turnActive), [blocks, snap.turnActive])

  // dsh: the approval attaches to the tool call it is about — render it
  // directly under the trail holding that call, not floating elsewhere.
  // A hidden card is still pending: the header badge reopens it.
  const approval = snap.interactionHidden ? null : snap.approval
  const question = snap.interactionHidden ? null : snap.question
  let approvalIndex = -1
  if (approval?.toolCallId) {
    approvalIndex = blocks.findIndex(
      b => b.kind === 'tools' && b.items.some(i => i.id === approval.toolCallId),
    )
  }
  const inlineApproval = approval !== null && approvalIndex !== -1
  const floatApproval = approval !== null && !inlineApproval
  const failedCard = snap.failed && !snap.turnActive ? <FailedCard failed={snap.failed} /> : null

  // Preserve `===` group array references for settled historical groups so
  // streaming deltas at the bottom never re-render older ToolTimelineGroups.
  const groupCacheRef = useRef<Map<string, readonly Block[]>>(new Map())
  const allGroups = useMemo(() => {
    const raw = keyedActivityGroups(blocks, approval?.toolCallId)
    const nextCache = new Map<string, readonly Block[]>()
    const stabilized = raw.map(entry => {
      const prev = groupCacheRef.current.get(entry.key)
      const stableBlocks = sameBlockList(prev, entry.blocks) ? prev! : entry.blocks
      nextCache.set(entry.key, stableBlocks)
      return { key: entry.key, blocks: stableBlocks }
    })
    groupCacheRef.current = nextCache
    return stabilized
  }, [blocks, approval?.toolCallId])

  const hiddenGroupCount = Math.max(0, allGroups.length - windowLimit)
  const shownGroups = hiddenGroupCount > 0 ? allGroups.slice(hiddenGroupCount) : allGroups
  // Gate automatic daemon paging on `hiddenGroupCount === 0` so scrolling near the top
  // reveals local tail-windowed groups first instead of firing duplicate daemon loads.
  const { ref, loadOlder, showJumpLatest, scrollToLatest, unpin } = useTranscriptScroll(sessionScope, {
    more: Boolean(snap.historyMore),
    loading: Boolean(snap.historyLoading),
    automatic: !snap.historyError && hiddenGroupCount === 0,
    load: () => store.loadOlderHistory(),
  })
  // Re-pin to the bottom whenever the user sends a new message in the current session.
  const lastUserBlockId = useMemo(() => {
    for (let i = blocks.length - 1; i >= 0; i -= 1) {
      if (blocks[i]?.kind === 'user') return blocks[i]!.id
    }
    return null
  }, [blocks])
  const seenUserScopeRef = useRef<{ readonly scope: string; readonly userId: number | null }>({
    scope: sessionScope,
    userId: lastUserBlockId,
  })
  useLayoutEffect(() => {
    const prev = seenUserScopeRef.current
    seenUserScopeRef.current = { scope: sessionScope, userId: lastUserBlockId }
    if (prev.scope === sessionScope && lastUserBlockId !== null && lastUserBlockId !== prev.userId) {
      scrollToLatest()
    }
  }, [sessionScope, lastUserBlockId, scrollToLatest])
  const windowExpandAnchorRef = useRef<HistoryScrollAnchor | null>(null)
  const expandWindow = useCallback((nextLimit: (current: number) => number, force = false): void => {
    if (!force && windowExpandAnchorRef.current) return
    const el = ref.current
    if (el && !windowExpandAnchorRef.current) {
      windowExpandAnchorRef.current = captureHistoryScrollAnchor(el)
    }
    unpin()
    setWindowLimit(nextLimit)
  }, [ref, unpin])

  // Preserve `[data-history-anchor]` viewport offset when prepending earlier tail-windowed
  // groups so `.stream { overflow-anchor: none }` never jumps the user to the top of the slice.
  useLayoutEffect(() => {
    const anchor = windowExpandAnchorRef.current
    if (!anchor) return
    windowExpandAnchorRef.current = null
    const el = ref.current
    if (!el) return
    el.scrollTop = restoreHistoryScrollAnchor(el, anchor)
  }, [windowLimit, hiddenGroupCount, ref])

  // Auto-expand earlier transcript groups when scrolling near top or opening in-page Find
  // (both DOM Cmd/Ctrl+F and native macOS Edit > Find via `xerxes:find`).
  useEffect(() => {
    if (hiddenGroupCount <= 0) return
    return attachTranscriptExpandListeners(
      ref.current,
      typeof window !== 'undefined' ? window : null,
      () => expandWindow(current => current + TRANSCRIPT_TAIL_GROUPS),
      () => expandWindow(current => Math.max(current, allGroups.length), true),
    )
  }, [hiddenGroupCount, allGroups.length, ref, expandWindow])

  if (offline && blocks.length === 0 && !question) {
    return <div className="stream" ref={ref}><Offline cwd={snap.cwd} error={snap.error} /></div>
  }

  // The failed card and pending question own the tail of the feed even when
  // the transcript itself is still empty — an empty welcome must not bury
  // the thing asking for a decision.
  return (
    // Focusable and named: the transcript is the main content of the
    // window and had no way to reach it by keyboard or name it to a screen
    // reader. Deliberately not role="log" — a polite live region over
    // token-level streaming announces continuously; the Announcer below
    // reports the transitions that actually matter instead.
    <ReplyEnds.Provider value={ends}>
    <div className={`stream${empty && !failedCard && !question && !approval ? ' stream--welcome' : ''}`} ref={ref} tabIndex={0} role="region" aria-label="Conversation">
      {empty && !failedCard && !question && !approval ? <>
        {blocks.length > 0 && <div className="welcome-notices" aria-live="polite">{blocks.map(block => <BlockView key={block.id} block={block} />)}</div>}
        {existingWork ? <TaskContinuation snap={snap} /> : <Welcome snap={snap} />}
      </> : (
        <div className="stream__col">
          {(snap.historyMore || snap.historyError) && hiddenGroupCount === 0 && <div className="history-pager">
            <button type="button" className="btn" disabled={snap.historyLoading} onClick={() => void loadOlder()}>{snap.historyLoading ? 'Loading earlier conversation…' : snap.historyError ? 'Retry loading earlier conversation' : 'Show earlier conversation'}</button>
            {snap.historyError && <p role="alert">{snap.historyError}</p>}
          </div>}
          {hiddenGroupCount > 0 && (
            <div className="history-pager history-pager--window">
              <button
                type="button"
                className="btn"
                onClick={() => expandWindow(current => current + TRANSCRIPT_TAIL_GROUPS)}
              >
                Show earlier messages ({hiddenGroupCount})
              </button>
            </div>
          )}
          {shownGroups.map(({ key, blocks: group }, index) => {
            const isLast = index === shownGroups.length - 1
            const hasInlineApproval = inlineApproval && group.some(block => block === blocks[approvalIndex])
            return (
              <StreamGroupRow
                key={`${snap.currentId}:${key}`}
                group={group}
                disclosed={isDisclosedActivity(group, approval?.toolCallId)}
                active={snap.turnActive && isLast}
                turnActive={snap.turnActive}
                inlineApproval={hasInlineApproval}
                approval={hasInlineApproval ? approval : null}
                permissionMode={snap.permissionMode}
                changes={snap.changes}
                {...(snap.cwd ? { cwd: snap.cwd } : {})}
              />
            )
          })}
          {failedCard}
        </div>
      )}
      {floatApproval && <div className="stream__col"><ApprovalCard key={approval.id} approval={approval} policy={snap.permissionMode} /></div>}
      {question && (
        <div className="stream__col">
          <QuestionCard key={`${snap.currentId}:${question.requestId}`} question={question} plan={snap.plan} />
        </div>
      )}
      {/* Zero-flow-height sticky dock so toggling .jump-latest never alters
          .stream.scrollHeight or triggers a bottom-overscroll oscillation loop. */}
      {showJumpLatest && blocks.length > 0 && (
        <div className="jump-latest-dock" data-scroll-ignore="true">
          <button type="button" className="jump-latest" data-working={snap.turnActive || undefined} title={snap.turnActive ? 'Still working · scroll to end' : 'Scroll to end'} onClick={scrollToLatest}>
            <span className="jump-latest__label">Scroll to end</span> <Icon name="chevron" size={11} />
          </button>
        </div>
      )}
    </div>
    </ReplyEnds.Provider>
  )
}

export { AgentsCard }
/** The conversation pieces the VS Code chat view composes around its own header. */
export { Stream, Composer, ConnectionBanner, Announcer, GlobalKeys, TaskModal }

function TaskContinuation({ snap }: { snap: Snapshot }): ReactElement {
  const open = useDesktopNavigation()
  return <section className="task-continuation"><Icon name="activity" size={24}/><h1>Continue this task</h1><p>This session has saved work. Review its progress or send your next instruction below.</p><div><button type="button" className="btn" onClick={() => open('activity')}>Review activity</button>{(snap.todos?.length || snap.plan) ? <button type="button" className="btn" onClick={() => store.setTab('plan')}>View plan &amp; to-dos</button> : null}</div></section>
}

/** Most recent first; the daemon already returns them in that order. */
const RESUMABLE = 4

/**
 * The home screen.
 *
 * This used to be a wordmark, a tagline and three verbs — "Research a
 * question", "Make a plan", "Build something" — that prefilled the composer
 * with "Help me research ". It was the same screen whether you had opened a
 * fresh folder or had eleven sessions and a dirty tree, which meant the one
 * surface you see on every launch knew nothing about your work.
 *
 * Everything below comes from the snapshot the store already holds. No new
 * wire calls: the sidebar is rendering these same sessions a few hundred
 * pixels to the left.
 */
function Welcome({ snap }: { snap: Snapshot }): ReactElement {
  const here = snap.cwd
  // Sessions from other folders belong to those folders' home screens.
  const resumable = snap.sessions
    .filter(row => row.kind === 'main' && !row.current && (!here || row.cwd === here))
    .slice(0, RESUMABLE)
  const dirty = snap.changes.length
  const adds = snap.changes.reduce((sum, file) => sum + file.adds, 0)
  const dels = snap.changes.reduce((sum, file) => sum + file.dels, 0)
  const place = workspaceLabel(here)

  // VS Code: a compact start, like its other chat views; no wordmark.
  if (isVscodeHost()) return <div className="welcome welcome--compact">
    <XerxesLogoMark size={44} className="welcome__logo welcome__logo--compact" alt="Xerxes Agents" />
    <h1 className="welcome__compact-title">What should Xerxes do{place ? <> in <b>{place}</b></> : ''}?</h1>
    {dirty > 0 && <p className="welcome__dirty">{dirty} uncommitted {dirty === 1 ? 'file' : 'files'} <b>+{adds}</b> <i>−{dels}</i></p>}
    {resumable.length > 0 && <section className="welcome__resume" aria-label="Recent tasks here">
      <h2>Recent</h2>
      {resumable.map(row => <button type="button" className="resumerow" key={row.id} onClick={() => void store.openSession(row.id)}>
        <span className="resumerow__dot" data-state={agentState(row.status).tone} aria-hidden="true" />
        <span className="resumerow__name">{row.title}</span>
        <span className="resumerow__age">{row.age}</span>
      </button>)}
    </section>}
  </div>

  return <div className="welcome">
    <XerxesLogoMark size={64} className="welcome__logo" alt="Xerxes Agents" />
    <h1 className="welcome__wordmark">XERXES</h1>

    {place ? (
      <p className="welcome__where">
        <Icon name="folder" size={14} />{place}
        {snap.branch && <><span className="welcome__sep" aria-hidden="true" /><Icon name="branch" size={14} />{snap.branch}</>}
        {dirty > 0 && <><span className="welcome__sep" aria-hidden="true" />
          <span className="welcome__dirty">{dirty} uncommitted {dirty === 1 ? 'file' : 'files'} <b>+{adds}</b> <i>−{dels}</i></span>
        </>}
      </p>
    ) : <p>A place to think, build, and finish.</p>}

    {/* Your own work outranks our suggestions — but only if you have any.
        A genuinely fresh folder still needs somewhere to start, so the
        starters remain the fallback rather than being deleted outright. */}
    {resumable.length > 0 ? (
      <section className="welcome__resume" aria-label="Recent tasks here">
        <h2>Pick up where you left off</h2>
        {resumable.map(row => {
          const state = agentState(row.status)
          return (
            <button type="button" className="resumerow" key={row.id} onClick={() => void store.openSession(row.id)}>
              <span className="resumerow__dot" data-state={state.tone} aria-hidden="true" />
              <span className="resumerow__name">{row.title}</span>
              <span className="resumerow__age">{row.age}</span>
              <Icon name="chevron" size={12} />
            </button>
          )
        })}
      </section>
    ) : (
      <div className="welcome__ideas">{([
        ['Research a question', 'Help me research '],
        ['Make a plan', 'Help me plan '],
        ['Build something', 'Help me implement '],
      ] as const).map(([label, prompt]) => (
        <button type="button" className="idea" key={label} onClick={() => window.dispatchEvent(new CustomEvent('xerxes:add-context', { detail: prompt }))}>
          <span>{label}</span><Icon name="arrow" size={14} />
        </button>
      ))}</div>
    )}

    {resumable.length > 0 && <p className="welcome__start">Or describe something new below.</p>}
  </div>
}

/**
 * Workspace gate — a fresh shell has no folder and therefore no daemon.
 * The composer is unavailable by design until a folder is chosen; the
 * pick feeds useProject, which binds the shared daemon to the folder and reloads
 * the shell into it.
 */
function WorkspaceGate(): ReactElement {
  return (
    <main className="chat">
      <div className="wsgate">
        <div className="wsgate__mark">
          <XerxesLogoMark size={48} className="welcome__logo" alt="Xerxes" />
        </div>
        <h1>Welcome to Xerxes.</h1>
        <p>
          Choose the folder you want to work in. Next, connect a model provider and start your first task. Your existing files stay in place.
        </p>
        <button type="button" className="btn btn--solid wsgate__btn" onClick={() => store.chooseWorkspace()}>
          Choose your first folder…
        </button>
        <p className="studio-muted">Bun is bundled on macOS. Git and SSH are optional and only needed for Git or remote workspaces.</p>
      </div>
    </main>
  )
}

function Offline({ cwd, error }: { cwd: string; error: string | null }): ReactElement {
  const kind = connectionFailureKind(error)
  return (
    <div className="offline">
      <div className="offline__mark"><AgentOrb size={32} state={kind === 'transport' ? 'connecting' : 'breathing'} live={kind === 'transport'} /></div>
      <h1>{kind === 'transport' ? 'Connecting to the shared daemon' : kind === 'session' ? 'Session belongs to another workspace' : 'Could not open this workspace'}</h1>
      {error && <p className="connection-error" role="alert">{error}</p>}
      <p>
        The terminal, TUI and desktop app share a daemon across workspaces. Each task keeps its own workspace and session. The app connects automatically when the daemon is available.
      </p>
      <button type="button" className="btn" onClick={() => store.retryConnection()}><Icon name="retry" size={13} /> Retry now</button>
      {cwd && <div className="cmd">Workspace: {cwd}</div>}
    </div>
  )
}

export const ActivityGroup = memo(function ActivityGroup({ blocks, active, turnActive }: { blocks: Snapshot['blocks']; active: boolean; turnActive: boolean }): ReactElement {
  return <ToolTimelineGroup blocks={blocks} active={active} turnActive={turnActive} />
})

/**
 * One action row per finished reply, under its last paragraph (Claude's):
 * a reply streams as several agent blocks between tool calls, and a copy
 * button under every one of them read as clutter. Maps the last agent
 * block of each settled turn to the whole reply's text.
 */
export function replyEnds(blocks: Snapshot['blocks'], turnActive: boolean): ReadonlyMap<number, { readonly text: string; readonly latest: boolean }> {
  const ends = new Map<number, { text: string; latest: boolean }>()
  let parts: string[] = []
  let last: number | undefined
  const close = () => {
    if (last !== undefined && parts.length) ends.set(last, { text: parts.join('\n\n'), latest: false })
    parts = []
    last = undefined
  }
  for (const block of blocks) {
    if (block.kind === 'user' && !block.contextSummary) close()
    else if (block.kind === 'agent' && !block.streaming && block.text.trim()) { parts.push(block.text.trim()); last = block.id }
  }
  // The running turn's reply is not finished yet.
  if (!turnActive) close()
  const newest = [...ends.keys()].at(-1)
  if (newest) ends.get(newest)!.latest = true
  return ends
}

const ReplyEnds = createContext<ReadonlyMap<number, { readonly text: string; readonly latest: boolean }>>(new Map())

function ReplyActionRow({ blockId }: { blockId: number }): ReactElement | null {
  const end = useContext(ReplyEnds).get(blockId)
  if (!end) return null
  return <div className={`msg__actions msg__actions--reply${end.latest ? ' is-latest' : ''}`}><CopyButton icon text={end.text} label="Copy reply" /></div>
}

const UserMessage = memo(function UserMessage({ text, images = [] }: { text: string; images?: readonly string[] }): ReactElement {
  const [expanded, setExpanded] = useState(false)
  const long = text.length > 1600 || text.split('\n').length > 16
  // The wrapper carries the bubble's right alignment so the copy row can sit
  // under the bubble rather than inside it.
  // Attached images sit above the bubble, whole and readable, as the person
  // pasted them; the bubble holds only what they typed.
  return <div className="msg-turn msg-turn--user">
    {images.length > 0 && <div className="msg__images">{images.map((url, index) => <a key={index} href={url} target="_blank" rel="noreferrer" title="Open image"><img src={url} alt={`Attached image ${index + 1}`} /></a>)}</div>}
    {text && <div className="msg msg--user">
      <div className={`msg__text${long && !expanded ? ' msg__text--preview' : ''}`}>{text}</div>
      {long && <button type="button" className="message-expand" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>{expanded ? 'Show less' : 'Show full message'}</button>}
    </div>}
    <div className="msg__actions"><CopyButton icon text={text} label="Copy message" /></div>
  </div>
})

/** Agent prose whose shell code blocks offer Run: an idle Terminal shell (or a new one) runs it, shown in the rail. */
const RunnableReply = memo(function RunnableReply({ text, streaming }: { text: string; streaming: boolean }): ReactElement {
  const navigate = useDesktopNavigation()
  const run = useCallback(async (command: string) => {
    await store.runInTerminal(command)
    navigate('terminal')
  }, [navigate])
  return (
    <CodeRunContext.Provider value={isVscodeHost() ? undefined : run}>
      <Markdown text={text} className="md--agent" tail={streaming ? CARET : undefined} />
    </CodeRunContext.Provider>
  )
})

// Memoized: the turn clock re-renders the conversation every second, and an
// unchanged block (same object) has nothing new to draw.
export const BlockView = memo(function BlockView({ block }: { block: Snapshot['blocks'][number] }): ReactElement {
  if (block.kind === 'reflection') return <ConnectedReflectionCard reflection={block.reflection} />
  if (block.kind === 'agents') {
    return <AgentsCard members={block.members} />
  }
  if (block.kind === 'user') {
    if (block.contextSummary) return <details className="context-summary"><summary>Earlier conversation summary</summary><Markdown text={block.text} /></details>
    return <UserMessage text={block.text} {...(block.images ? { images: block.images } : {})} />
  }
  if (block.kind === 'agent') {
    return (
      <section className="msg msg--agent">
        {/* Agent prose IS markdown — code fences, tables, links, lists. The
            renderer is React-elements-only, so model output cannot inject
            markup; the streaming caret trails only the live tail block. */}
        <div className="msg__text">
          <RunnableReply text={block.text} streaming={block.streaming} />
        </div>
        <ReplyActionRow blockId={block.id} />
      </section>
    )
  }
  if (block.kind === 'thinking') {
    // Flat one-liner: the excerpt tracks the stream live; click to read the
    // full trail. No `open` prop — the user's toggle survives re-renders.
    return (
      <details className="thinkrow">
        <summary>
          <span className="frow__icon"><Icon name="activity" size={14} /></span>
          <span className="frow__label">Reasoning</span>
          <span className="frow__sep">·</span>
          <span className="frow__excerpt">{thinkingTailOf(block.text)}{block.streaming ? ' …' : ''}</span>
        </summary>
        <div className="thinkrow__body">{block.text}</div>
      </details>
    )
  }
  if (block.kind === 'tools') {
    // One flat row per call — no fold, no summary header. State rides the
    // icon; duration and diff stats trail the row.
    return (
      <div className="tlist">
        {block.items.map(item => (
          <ToolCallRow key={item.id} item={item} label={toolLabelOf(item.verb)} />
        ))}
      </div>
    )
  }
  if (block.kind === 'checkpoint') {
    return (
      <div className="frow frow--sys">
        <span className="frow__icon"><Icon name="clock" size={13} /></span>
        <span className="frow__label">Checkpoint</span>
        <span className="frow__sep">·</span>
        <span className="frow__excerpt">turn {block.turn} end · <b>+{block.adds} −{block.dels}</b> cumulative</span>
      </div>
    )
  }
  if (block.text.length > 320 || block.text.split('\n').length > 4) return <details className="activity-notice"><summary><Icon name="activity" size={14}/><span>{block.text.split('\n')[0]!.slice(0,160)}</span><Icon name="chevron" size={12}/></summary><OutputViewer text={block.text} label="Activity details" /></details>
  return (
    <div className={`frow frow--sys${block.error ? ' frow--err' : ''}`}>
      <span className="frow__icon"><Icon name={block.error ? "warning" : "info"} size={13} /></span>
      <span className="frow__excerpt frow__excerpt--wrap" title={block.text.length > 160 ? block.text : undefined}>{block.text}</span>
    </div>
  )
})

/** Command-shaped tools read as a command; everything else as its fields. */
const COMMAND_KEYS = ['command', 'cmd', 'CommandLine', 'script', 'shell'] as const
const COMMAND_KEY_LOWER = new Set(COMMAND_KEYS.map(key => key.toLowerCase()))
const APPROVAL_PLUMBING_KEYS = new Set([
  'waitmsbeforeasync',
  'toolsummary',
  'tool_summary',
  'safetoautorun',
  'blocking',
  'isdaemon',
])
const APPROVAL_DIFF_OMIT_KEYS = new Set([
  'path', 'Path', 'file_path',
  'Target', 'Replacement', 'old_string', 'new_string', 'old_text', 'new_text', 'oldText', 'newText',
  'content', 'Content',
])
const APPROVAL_DIFF_PREVIEW_LINES = 16

function approvalToolSummaryOf(inputs: Readonly<Record<string, unknown>> | undefined): string | null {
  if (!inputs) return null
  const raw = inputs.toolSummary ?? inputs.tool_summary
  return typeof raw === 'string' && raw.trim() ? raw.trim() : null
}

/** Map digit shortcuts (`1`/`2`/`3`) on an approval card, respecting `allowPermanent === false`. */
export function approvalKeyAction(key: string, allowPermanent?: boolean): ApprovalResponse | null {
  if (key === '1') return 'allow_once'
  if (key === '2') return allowPermanent === false ? 'deny' : 'allow_session'
  if (key === '3') return 'deny'
  return null
}

function approvalOptionDecision(kind: string): 'approve' | 'approve_for_session' | 'reject' {
  if (kind === 'allow_always') return 'approve_for_session'
  if (kind === 'reject_once' || kind === 'reject_always') return 'reject'
  return 'approve'
}

function approvalOptionButtonClass(kind: string, vibes: boolean): string {
  const isReject = kind === 'reject_once' || kind === 'reject_always'
  const isPrimary = kind === 'allow_once'
  if (vibes) {
    if (isPrimary) return 'vq__submit'
    return isReject ? 'vq__skip vq__skip--danger' : 'vq__skip'
  }
  if (isPrimary) return 'btn btn--solid'
  if (isReject) return 'btn btn--danger'
  return 'btn'
}

function normalizeStructuredApprovalDiff(
  diff: { readonly path: string; readonly oldText?: string; readonly newText?: string },
  fallbackPath: string,
): { readonly path: string; readonly before: string; readonly after: string } | null {
  if (diff.oldText === undefined && diff.newText === undefined) return null
  const rawBefore = diff.oldText ?? ''
  const rawAfter = diff.newText ?? ''
  if (rawBefore && rawAfter) {
    const span = trimmedChangedLines(rawBefore, rawAfter, 3)
    return {
      path: diff.path || fallbackPath,
      before: span.before.join('\n'),
      after: span.after.join('\n'),
    }
  }
  return { path: diff.path || fallbackPath, before: rawBefore, after: rawAfter }
}

function approvalDiffsOf(
  approval: NonNullable<Snapshot['approval']>,
): readonly { readonly path: string; readonly before: string; readonly after: string }[] {
  const inputs = approval.inputs
  const inputPath = inputs
    ? (typeof inputs.file_path === 'string' && inputs.file_path.trim())
      || (typeof inputs.path === 'string' && inputs.path.trim())
      || (typeof inputs.Path === 'string' && inputs.Path.trim())
      || ''
    : ''
  if (approval.diffs && approval.diffs.length > 0) {
    const normalized = approval.diffs
      .map(entry => normalizeStructuredApprovalDiff(entry, inputPath))
      .filter((entry): entry is { readonly path: string; readonly before: string; readonly after: string } => entry !== null)
    if (normalized.length > 0) return normalized
  }
  if (approval.diff) {
    const single = normalizeStructuredApprovalDiff(approval.diff, inputPath)
    if (single) return [single]
  }
  if (!inputs) return []
  const before =
    typeof inputs.old_string === 'string'
      ? inputs.old_string
      : typeof inputs.Target === 'string'
        ? inputs.Target
        : typeof inputs.old_text === 'string'
          ? inputs.old_text
          : typeof inputs.oldText === 'string'
            ? inputs.oldText
            : undefined
  const after =
    typeof inputs.new_string === 'string'
      ? inputs.new_string
      : typeof inputs.Replacement === 'string'
        ? inputs.Replacement
        : typeof inputs.new_text === 'string'
          ? inputs.new_text
          : typeof inputs.newText === 'string'
            ? inputs.newText
            : typeof inputs.content === 'string'
              ? inputs.content
              : typeof inputs.Content === 'string'
                ? inputs.Content
                : undefined
  if (before === undefined && after === undefined) return []
  return [{ path: inputPath, before: before ?? '', after: after ?? '' }]
}

function ApprovalDiffPreview({ diff }: { readonly diff: { readonly path: string; readonly before: string; readonly after: string } }): ReactElement {
  const [expanded, setExpanded] = useState(false)
  const lines = useMemo(() => lineDiff(diff.before, diff.after), [diff.before, diff.after])
  const adds = useMemo(() => lines.filter(line => line.kind === 'add').length, [lines])
  const dels = useMemo(() => lines.filter(line => line.kind === 'del').length, [lines])
  const shown = expanded ? lines : lines.slice(0, APPROVAL_DIFF_PREVIEW_LINES)
  const hiddenCount = lines.length - shown.length

  return (
    <div className="approval__diff" role="region" aria-label={diff.path ? `Proposed diff for ${diff.path}` : 'Proposed diff'}>
      <div className="approval__diff-head">
        <span className="approval__diff-path" title={diff.path || undefined}>{diff.path || 'File changes'}</span>
        <span className="approval__diff-stats">
          <span className="add">+{adds}</span>
          {dels > 0 && <span className="del">−{dels}</span>}
        </span>
      </div>
      <pre className="approval__diff-code">
        {shown.map((line, index) => (
          <span key={index} className={`approval__diff-line approval__diff-line--${line.kind}`}>
            <span className="approval__diff-sign" aria-hidden="true">
              {line.kind === 'add' ? '+' : line.kind === 'del' ? '−' : ' '}
            </span>
            <code>{line.text || ' '}</code>
          </span>
        ))}
      </pre>
      {lines.length > APPROVAL_DIFF_PREVIEW_LINES && (
        <button
          type="button"
          className="approval__diff-more"
          onClick={() => setExpanded(value => !value)}
        >
          {expanded ? 'Show less' : `Show ${hiddenCount} more line${hiddenCount === 1 ? '' : 's'}`}
        </button>
      )}
    </div>
  )
}

/**
 * What the decision is actually about. The daemon ships the tool's real
 * arguments; rendering them per shape — the command for a shell call, the
 * path + diff for an edit, labelled fields otherwise — is the difference
 * between approving `send_message(telegram)` and approving a message you
 * have read.
 */
function ApprovalBody({
  approval,
  hideToolSummary = false,
}: {
  approval: NonNullable<Snapshot['approval']>
  hideToolSummary?: boolean
}): ReactElement | null {
  const diffs = approvalDiffsOf(approval)
  const inputs = approval.inputs
  const toolSummary = approvalToolSummaryOf(inputs)
  const showInlineSummary = Boolean(toolSummary && !hideToolSummary && toolSummary !== approval.reason)
  const entries = inputs
    ? Object.entries(inputs).filter(([, value]) => value !== null && value !== undefined && value !== '')
    : []
  if (diffs.length > 0) {
    const rest = entries.filter(
      ([key]) => !APPROVAL_DIFF_OMIT_KEYS.has(key) && !APPROVAL_PLUMBING_KEYS.has(key.toLowerCase()),
    )
    return <>
      {showInlineSummary && <p className="approval__reason">{toolSummary}</p>}
      <div className="approval__diffs">
        {diffs.map((diff, index) => (
          <ApprovalDiffPreview key={`${diff.path}:${index}`} diff={diff} />
        ))}
      </div>
      {rest.length > 0 && <ApprovalFields entries={rest} />}
    </>
  }
  if (entries.length === 0) return approval.description ? <pre className="approval__desc">{approval.description}</pre> : null
  const command = entries.find(([key]) => COMMAND_KEY_LOWER.has(key.toLowerCase()))
  if (command && typeof command[1] === 'string') {
    const rest = entries.filter(
      ([key]) => !COMMAND_KEY_LOWER.has(key.toLowerCase()) && !APPROVAL_PLUMBING_KEYS.has(key.toLowerCase()),
    )
    return <>
      {showInlineSummary && <p className="approval__reason">{toolSummary}</p>}
      <pre className="approval__desc">{command[1]}</pre>
      {rest.length > 0 && <ApprovalFields entries={rest} />}
    </>
  }
  const visibleEntries = entries.filter(([key]) => !APPROVAL_PLUMBING_KEYS.has(key.toLowerCase()))
  if (visibleEntries.length === 0) {
    if (showInlineSummary) return <p className="approval__reason">{toolSummary}</p>
    return approval.description ? <pre className="approval__desc">{approval.description}</pre> : null
  }
  return <>
    {showInlineSummary && <p className="approval__reason">{toolSummary}</p>}
    <ApprovalFields entries={visibleEntries} />
  </>
}

function ApprovalFields({ entries }: { entries: readonly [string, unknown][] }): ReactElement {
  return (
    <dl className="approval__fields">
      {entries.map(([key, value]) => (
        <div key={key}>
          <dt>{key.replace(/[_-]+/g, ' ')}</dt>
          <dd>{typeof value === 'string' ? value : JSON.stringify(value, null, 1)}</dd>
        </div>
      ))}
    </dl>
  )
}

/** Leaving plan mode is a plan decision, not a tool permission: it gets the plan card. */
export function isPlanExitApproval(approval: Pick<NonNullable<Snapshot['approval']>, 'toolName' | 'action'>): boolean {
  return /^ExitPlanMode(Tool)?$/.test(approval.toolName || approval.action || '')
}

export function ApprovalCard(props: { approval: NonNullable<Snapshot['approval']>; policy: string; inline?: boolean; blocks?: readonly Block[] }): ReactElement {
  const { blocks, ...rest } = props
  return isPlanExitApproval(props.approval) ? <PlanApprovalCard approval={props.approval} planId={latestArtifactId(blocks ?? store.getSnapshot().blocks)} /> : <ToolApprovalCard {...rest} />
}

/**
 * The plan is ready: continue with it in one click, or send a note that keeps
 * planning. Refining rejects the exit (the session stays in plan mode) and
 * queues the note as a steer, so the agent revises the same plan.
 */
type PlanChoice = 'accept-all' | 'manual' | 'keep'

/** Claude Code's exit-plan choices (2.1.329): proceed auto-accepting edits, proceed asking for each, or keep planning. */
export const PLAN_CHOICES: readonly { readonly value: PlanChoice; readonly label: string; readonly detail: string }[] = [
  { value: 'accept-all', label: 'Yes, and auto-accept edits', detail: 'Edits and commands run without asking' },
  { value: 'manual', label: 'Yes, manually approve edits', detail: 'Each change asks you first' },
  { value: 'keep', label: 'No, keep planning', detail: 'Tell the agent what to change' },
]

/**
 * The plan is ready, shown the way Claude Code shows it: the plan itself in
 * the card (folded to a preview, unfolds in place, opens beside the chat),
 * then "Would you like to proceed?" with its three answers. Proceeding sets
 * the permission mode the person picked; keeping planning rejects the exit
 * (the session stays in plan mode) and queues the note as a steer.
 */
export function PlanApprovalCard({ approval, planId }: { approval: NonNullable<Snapshot['approval']>; planId?: string | undefined }): ReactElement {
  const open = useDesktopNavigation()
  const [choice, setChoice] = useState<PlanChoice>('accept-all')
  const [note, setNote] = useState('')
  const [plan, setPlan] = useState<{ readonly title: string; readonly content: string } | null>(null)
  const [expanded, setExpanded] = useState(false)
  const card = useRef<HTMLDivElement>(null)
  useEffect(() => { card.current?.focus() }, [approval.id])
  useEffect(() => {
    setPlan(null)
    if (!planId) return
    let live = true
    void desktopCall(window.xerxes, store.getSnapshot().sessionKey, 'artifacts.read', { id: planId })
      .then(result => {
        const art = result.artifact
        if (!live || !art || typeof art !== 'object') return
        const row = art as Record<string, unknown>
        if (typeof row.content === 'string' && row.content.trim()) setPlan({ title: typeof row.title === 'string' ? row.title : 'Plan', content: row.content })
      })
      // Unreadable plan: the card still offers "Open the plan".
      .catch(() => {})
    return () => { live = false }
  }, [planId])
  const submit = (): void => {
    if (choice === 'keep') {
      const text = note.trim()
      if (!text) return
      store.approve(approval.id, 'deny')
      const externalAcp = isExternalAcpBinding(store.getSnapshot().acpBinding)
      void store.steer(externalAcp ? text : `Keep planning; do not execute yet. Revise the plan with this feedback: ${text}`)
      setNote('')
      return
    }
    store.approvePlan(approval.id, choice)
  }
  const canSubmit = choice !== 'keep' || note.trim() !== ''
  return (
    <div ref={card} tabIndex={-1} className="plancard" role="alertdialog" aria-modal="false" aria-label="Plan ready"
      onKeyDown={event => {
        if (event.defaultPrevented) return
        if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) { event.preventDefault(); submit(); return }
        if (event.target instanceof HTMLTextAreaElement) return
        const index = ['1', '2', '3'].indexOf(event.key)
        if (index >= 0) { event.preventDefault(); setChoice(PLAN_CHOICES[index]!.value) }
      }}>
      <div className="plancard__head">
        <span className="plancard__dot" aria-hidden="true" />
        <span className="plancard__title">Ready to code?</span>
        <span className="plancard__tag">Plan ready</span>
        <button type="button" className="plancard__hide" title="Hide this card without stopping the plan timer" aria-label="Hide this request" onClick={() => store.dismissInteraction()}><Icon name="close" size={13} /></button>
      </div>
      {plan
        ? <section className="plancard__plan" aria-label={plan.title}>
            <div className={`plancard__doc${expanded ? ' is-open' : ''}`}><Markdown text={plan.content} className="md--compact" /></div>
            <div className="plancard__tools">
              <button type="button" onClick={() => setExpanded(value => !value)}>{expanded ? 'Show less' : 'Show the whole plan'}</button>
              <button type="button" onClick={() => open('docs', planId)}><Icon name="file" size={12} /> Open beside the chat</button>
            </div>
          </section>
        // Opens the plan itself (the newest plan artifact of this round), not just the list.
        : <button type="button" className="plancard__open" onClick={() => open('docs', planId)}><Icon name="file" size={13} /> Open the plan</button>}
      <p className="plancard__ask">Would you like to proceed?</p>
      {approval.autoAcceptAt !== undefined && <p>If you don’t respond by {new Date(approval.autoAcceptAt).toLocaleTimeString()}, the plan will proceed with edits and commands automatically approved (3-minute timeout).</p>}
      <div className="plancard__opts" role="radiogroup" aria-label="How to proceed">
        {PLAN_CHOICES.map((option, index) => (
          <button key={option.value} type="button" role="radio" aria-checked={choice === option.value} className="plancard__opt" onClick={() => setChoice(option.value)}>
            <kbd>{index + 1}</kbd>
            <span className="plancard__text"><span className="plancard__label">{option.label}</span><span className="plancard__detail">{option.detail}</span></span>
          </button>
        ))}
      </div>
      {choice === 'keep' && <textarea className="plancard__note" aria-label="Refine the plan" placeholder="Tell the agent what to change, add or drop" rows={2} value={note} autoFocus
        onChange={event => setNote(event.target.value)}
        onKeyDown={event => {
          if (event.nativeEvent.isComposing) return
          if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); event.stopPropagation(); submit() }
        }} />}
      <div className="plancard__actions">
        <button type="button" className="plancard__submit" disabled={!canSubmit} onClick={submit}>{choice === 'keep' ? 'Send feedback' : 'Continue with the plan'} <kbd>⌘</kbd><kbd>↵</kbd></button>
      </div>
    </div>
  )
}

function ToolApprovalCard({ approval, policy, inline }: { approval: NonNullable<Snapshot['approval']>; policy: string; inline?: boolean }): ReactElement {
  const card = useRef<HTMLDivElement>(null)
  const vibes = useVibesStyle()
  const dynamicOptions = approval.options && approval.options.length > 0 ? approval.options : null
  const canAllowSession = approval.allowPermanent !== false
  const toolSummary = approvalToolSummaryOf(approval.inputs)
  const subtitle = approval.reason || toolSummary
  const hideBodySummary = Boolean(toolSummary && subtitle === toolSummary)
  const optionWarnings = dynamicOptions
    ? dynamicOptions.filter(opt => Boolean(opt.warning)).map(opt => `${opt.name}: ${opt.warning}`)
    : []
  const primaryOption = dynamicOptions
    ? (dynamicOptions.find(opt => opt.kind === 'allow_once')
      ?? dynamicOptions.find(opt => opt.kind === 'allow_always')
      ?? dynamicOptions[0])
    : undefined
  // role="alertdialog" that never takes focus announces nothing, and the
  // 1/2/3 keys it advertises are dropped while the composer (auto-focused
  // on entry) holds focus. Moving focus here fixes both at once — and it
  // is what makes the keys safe to bind at all.
  useEffect(() => {
    const previous = document.activeElement
    // The card itself, not its first button: landing on a control would
    // make Enter activate whatever happens to be first in the DOM.
    card.current?.focus()
    return () => { if (previous instanceof HTMLElement && previous.isConnected) previous.focus() }
  }, [approval.id])
  if (vibes) {
    // Same card as the Vibes question: the "Needs your input" label, then what is asked
    // on the header band, the exact command or fields, and the answers at the right.
    // Keys 1..N still answer while the card has focus (the window handler accepts .vq--approval).
    const title = approvalTitle(approval.toolName || approval.action || '')
    return (
      <div ref={card} tabIndex={-1} className="vq vq--approval" role="alertdialog" aria-modal="false" aria-label={`Approve ${approval.toolName || approval.action || 'tool call'}`}
        onKeyDown={event => {
          if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
            event.preventDefault()
            if (primaryOption) {
              store.approveWithOption(approval.id, primaryOption.optionId, approvalOptionDecision(primaryOption.kind))
            } else {
              store.approve(approval.id, 'allow_once')
            }
          }
        }}>
        <span className="vq__pill"><span className="vq__ring" aria-hidden="true" />Needs your input</span>
        <div className="vq__card">
          <div className="vq__q vq__q--approval">
            <span className="vq__qtitle" title={approval.toolName || approval.action || undefined}>{title}</span>
            <button type="button" className="vq__hide" title="Hide until you decide; the request stays pending" aria-label="Hide this request" onClick={() => store.dismissInteraction()}><Icon name="close" size={13} /></button>
          </div>
          {subtitle && <p className="vq__reason">{subtitle}</p>}
          <div className="vq__body"><ApprovalBody approval={approval} hideToolSummary={hideBodySummary} /></div>
          {approval.cwd && <p className="vq__cwd" title={approval.cwd}>in {approval.cwd}</p>}
          {optionWarnings.map((warn, idx) => (
            <p key={idx} className="vq__reason" role="note">{warn}</p>
          ))}
          <div className="vq__actions">
            {dynamicOptions ? (
              dynamicOptions.map((opt, index) => (
                <button
                  key={opt.optionId}
                  type="button"
                  className={approvalOptionButtonClass(opt.kind, true)}
                  data-option-id={opt.optionId}
                  data-option-kind={opt.kind}
                  title={opt.warning || undefined}
                  onClick={() => store.approveWithOption(approval.id, opt.optionId, approvalOptionDecision(opt.kind))}
                >
                  {opt.name}
                  {index < 9 && <> <kbd>{index + 1}</kbd></>}
                </button>
              ))
            ) : (
              <>
                <button type="button" className="vq__skip" onClick={() => store.approve(approval.id, 'deny')}>Deny <kbd>{canAllowSession ? '3' : '2'}</kbd></button>
                {canAllowSession && <button type="button" className="vq__skip" onClick={() => store.approve(approval.id, 'allow_session')}>Allow for this session <kbd>2</kbd></button>}
                <button type="button" className="vq__submit" onClick={() => store.approve(approval.id, 'allow_once')}>Allow once <kbd>⌘</kbd><kbd>↵</kbd></button>
              </>
            )}
          </div>
        </div>
      </div>
    )
  }
  return (
    <div ref={card} tabIndex={-1} className={`approval${inline ? ' approval--inline' : ''}`} role="alertdialog" aria-modal="false" aria-label={`Approve ${approval.toolName || approval.action || 'tool call'}`}>
      <div className="approval__head">
        <span className="approval__dot" aria-hidden="true" />
        <span className="approval__title" title={approval.toolName || approval.action || undefined}>{approvalTitle(approval.toolName || approval.action || '')}</span>
        <button className="approval__dismiss" title="Hide until you decide — the request stays pending" aria-label="Hide this request" onClick={() => store.dismissInteraction()}><Icon name="close" size={13} /></button>
      </div>
      {subtitle && <p className="approval__reason">{subtitle}</p>}
      <ApprovalBody approval={approval} hideToolSummary={hideBodySummary} />
      {optionWarnings.map((warn, idx) => (
        <p key={idx} className="approval__reason" role="note">{warn}</p>
      ))}
      <div className="approval__row">
        {dynamicOptions ? (
          dynamicOptions.map((opt, index) => (
            <button
              key={opt.optionId}
              type="button"
              className={approvalOptionButtonClass(opt.kind, false)}
              data-option-id={opt.optionId}
              data-option-kind={opt.kind}
              title={opt.warning || undefined}
              onClick={() => store.approveWithOption(approval.id, opt.optionId, approvalOptionDecision(opt.kind))}
            >
              {opt.name}
              {index < 9 && <> <kbd>{index + 1}</kbd></>}
            </button>
          ))
        ) : (
          <>
            <button className="btn btn--solid" onClick={() => store.approve(approval.id, 'allow_once')}>Allow once <kbd>1</kbd></button>
            {canAllowSession && <button className="btn" onClick={() => store.approve(approval.id, 'allow_session')}>This session <kbd>2</kbd></button>}
            <button className="btn btn--danger" onClick={() => store.approve(approval.id, 'deny')}>Deny <kbd>{canAllowSession ? '3' : '2'}</kbd></button>
          </>
        )}
      </div>
      <p className="approval__cwd" title={approval.cwd || undefined}>
        <span className="appr-policy">Policy: {(approval.permissionMode ?? policy) || 'runtime default'}</span>{approval.cwd && <> · in {approval.cwd}</>}
      </p>
    </div>
  )
}

/** Failed turn: the error, a retry from the last instruction, or resolve. */
export function withoutDuplicateFailure(blocks: readonly Block[], error: string): readonly Block[] {
  const lastUser = blocks.findLastIndex(block => block.kind === 'user')
  const matches = (text: string) => text === error || (
    error.startsWith('Automatic context compaction failed:') && text.startsWith('Automatic context compaction failed:')
  )
  return blocks.filter((block, index) => index <= lastUser || !(
    (block.kind === 'notice' && block.error && matches(block.text)) ||
    (block.kind === 'agent' && block.text.trim().startsWith('[Error: ') && block.text.trim().endsWith(']') && matches(block.text.trim().slice(8, -1)))
  ))
}

export function FailedCard({ failed }: { failed: NonNullable<Snapshot['failed']> }): ReactElement {
  const [compacting, setCompacting] = useState(false)
  // The old check only knew about credentials, so the single most common
  // recoverable failure — a provider usage limit — rendered as raw JSON
  // with a unix timestamp and offered a Retry that could only fail again.
  const view = failureView(failed.error, Date.now())
  const compaction = view.kind === 'compaction'
  return (
    <div className="terr" role="alert">
      <div className="terr__head"><Icon name="error" size={14} /> Turn {failed.turn} failed</div>
      <div className="terr__body">{view.summary || failed.error}</div>
      {view.summary && <details><summary>What the provider said</summary><p>{failed.error}</p></details>}
      <div className="terr__row">
        {view.offerProviderSettings && <button className="btn btn--solid" onClick={() => store.openSettings('models')}>{view.kind === 'rate-limit' || view.kind === 'context-length' ? 'Switch model or provider' : 'Provider settings'}</button>}
        {compaction
          ? <button className="btn" disabled={compacting} onClick={async () => { setCompacting(true); try { await store.retryCompaction() } finally { setCompacting(false) } }}>{compacting ? 'Compacting…' : 'Retry compaction'}</button>
          : <button
              className="btn"
              // Still offered when futile — the limit may have reset, and
              // hiding it would strand a user who knows it has — but never
              // as the obvious next thing to click.
              title={view.retryIsFutile ? 'This will fail again until the cause above is resolved' : 'Continue from where the turn stopped'}
              onClick={() => store.retryFailed()}
            >
              <Icon name="retry" size={13} /> Retry
            </button>}
        <button className="btn btn--ghost" onClick={() => store.resolveFailure()}>Dismiss</button>
      </div>
    </div>
  )
}

/** dsh user-questions card: batched questions, options, Other free-text. */
/** "Label — description" shows as a label over a grey line. Display only: the answer is the whole option. */
function optionParts(option: string): { label: string; detail: string } {
  const at = option.indexOf(' — ')
  return at > 0 ? { label: option.slice(0, at), detail: option.slice(at + 3) } : { label: option, detail: '' }
}

/** Toggle a choice in single-select or multi-select question state. */
export function toggleQuestionOption(
  current: readonly string[],
  option: string,
  multiSelect = false,
): string[] {
  if (multiSelect) {
    return current.includes(option) ? current.filter(item => item !== option) : [...current, option]
  }
  return current.includes(option) ? [] : [option]
}

/**
 * Format QuestionCard answers into wire-compatible `Record<string, string>`.
 * Multi-select answers join chosen option titles (and any custom Other entry)
 * with `\n` so option titles containing commas stay unambiguous across ACP & TUI.
 */
export function formatQuestionAnswers(
  items: NonNullable<Snapshot['question']>['items'],
  selections: Readonly<Record<string, readonly string[]>>,
  others: Readonly<Record<string, string>>,
): Record<string, string> {
  const answers: Record<string, string> = {}
  for (const item of items) {
    const custom = others[item.id]?.trim() ?? ''
    const picked = selections[item.id] ?? []
    if (item.multiSelect) {
      const combined = custom ? [...picked, custom] : [...picked]
      if (combined.length > 0) answers[item.id] = combined.join('\n')
    } else if (custom) {
      answers[item.id] = custom
    } else if (picked.length > 0) {
      answers[item.id] = picked.join(', ')
    }
  }
  return answers
}

/**
 * ACP `elicitation/create` URL-mode consent card: always displays the full
 * destination URL visibly before opening, lets the user launch it via the
 * host's external URL bridge, and sends an explicit accept or decline answer.
 */
export function UrlConsentCard({
  question,
}: {
  readonly question: NonNullable<Snapshot['question']>
}): ReactElement {
  const vibes = useVibesStyle()
  const [opened, setOpened] = useState(false)
  const item = question.items[0]
  const consentKey = item?.id || 'url_consent'
  const url = question.url || ''
  const promptText = question.message || item?.question || 'The agent requests opening an external link to continue.'
  const acceptValue =
    item?.options.find(option => /open|continue|accept|done|yes/i.test(option))
    || item?.options[0]
    || 'accept'
  const declineValue =
    item?.options.find(option => /decline|reject|cancel|no/i.test(option))
    || 'Decline'

  const openLink = (): void => {
    if (!url) return
    setOpened(true)
    if (typeof window !== 'undefined' && window.xerxes?.openExternal) {
      void window.xerxes.openExternal(url)
    } else if (typeof window !== 'undefined' && typeof window.open === 'function') {
      window.open(url, '_blank', 'noopener,noreferrer')
    }
  }

  const accept = (): void => {
    store.answerQuestion(question.requestId, { [consentKey]: acceptValue })
  }

  const decline = (): void => {
    store.answerQuestion(question.requestId, { [consentKey]: declineValue })
  }

  if (vibes) {
    return (
      <div className="vq vq--url" role="alertdialog" aria-modal="false" aria-label="External link confirmation">
        <span className="vq__pill"><span className="vq__ring" aria-hidden="true" />External link requested</span>
        <div className="vq__card">
          <div className="vq__q">
            <Markdown text={promptText} className="md--compact" />
          </div>
          {url && (
            <div className="qcard__url-wrap">
              <code className="qcard__url" data-elicitation-url={url}>{url}</code>
            </div>
          )}
          <div className="vq__actions">
            <button type="button" className="vq__skip vq__skip--danger" onClick={decline}>Decline</button>
            {url && (
              <button type="button" className="vq__skip" onClick={openLink}>
                {opened ? 'Open link again' : 'Open link'}
              </button>
            )}
            <button type="button" className="vq__submit" onClick={accept}>Continue / Done</button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="qcard qcard--url" role="alertdialog" aria-modal="false" aria-label="External link confirmation">
      <div className="qcard__head">
        <span className="approval__dot" aria-hidden="true" />
        <span className="qcard__title">External link confirmation</span>
      </div>
      <div className="qcard__q">
        <Markdown text={promptText} className="md--compact" />
      </div>
      {url && (
        <div className="qcard__url-wrap">
          <code className="qcard__url" data-elicitation-url={url}>{url}</code>
        </div>
      )}
      <div className="approval__row">
        {url && (
          <button type="button" className="btn" onClick={openLink}>
            {opened ? 'Open link again' : 'Open link'}
          </button>
        )}
        <button type="button" className="btn btn--solid" onClick={accept}>Continue / Done</button>
        <button type="button" className="btn btn--danger" onClick={decline}>Decline</button>
      </div>
    </div>
  )
}

export function QuestionCard({
  question,
  plan,
}: {
  question: NonNullable<Snapshot['question']>
  plan: Snapshot['plan']
}): ReactElement {
  const [selections, setSelections] = useState<Record<string, string[]>>({})
  const [others, setOthers] = useState<Record<string, string>>({})
  const cardRef = useRef<HTMLDivElement>(null)
  const isUrlMode = question.mode === 'url'
  const review = !isUrlMode && isPlanReview(question)
  const vibes = useVibesStyle()
  const canSubmit = question.items.length > 0 && question.items.every(item =>
    (item.allowFreeform && !!others[item.id]?.trim()) || (selections[item.id]?.length ?? 0) > 0)
  const submit = (): void => {
    if (!canSubmit) return
    store.answerQuestion(question.requestId, formatQuestionAnswers(question.items, selections, others))
  }
  const handleCardKeyDown = (event: React.KeyboardEvent<HTMLDivElement>): void => {
    if (event.defaultPrevented || event.key !== 'Enter') return
    if (event.metaKey || event.ctrlKey) {
      const target = event.target as HTMLElement | null
      if (target?.closest('.vq__skip')) return
      if (!canSubmit) return
      event.preventDefault()
      event.stopPropagation()
      submit()
    }
  }
  useEffect(() => {
    if (review || isUrlMode) return
    const previous = document.activeElement
    cardRef.current?.focus()
    return () => { if (previous instanceof HTMLElement && previous.isConnected) previous.focus() }
  }, [question.requestId, review, isUrlMode])
  // Number keys choose options while the card is up (not while typing); the
  // hints are only honest for the FIRST question — later ones have no keys.
  useEffect(() => {
    if (review || isUrlMode) return // PlanReviewCard & UrlConsentCard own their responses.
    const onKey = (event: KeyboardEvent): void => {
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) return
      if (typeof document !== 'undefined' && document.querySelector('[role="dialog"][aria-modal="true"]')) return
      const target = event.target as HTMLElement | null
      const insideCard = Boolean(target && cardRef.current?.contains(target))
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable || (target.tagName === 'BUTTON' && !insideCard))) return
      const number = Number.parseInt(event.key, 10)
      if (!Number.isFinite(number) || number < 1) return
      const item = question.items[0]
      if (!item) return
      const option = item.options[number - 1]
      if (!option) return
      event.preventDefault()
      setSelections(prev => ({
        ...prev,
        [item.id]: toggleQuestionOption(prev[item.id] ?? [], option, item.multiSelect === true),
      }))
      if (!item.multiSelect) setOthers(prev => ({ ...prev, [item.id]: '' }))
    }
    const onEnter = (event: KeyboardEvent): void => {
      if (event.defaultPrevented || event.key !== 'Enter') return
      if (typeof document !== 'undefined' && document.querySelector('[role="dialog"][aria-modal="true"]')) return
      const target = event.target as HTMLElement | null
      const modEnter = event.metaKey || event.ctrlKey
      const insideCard = Boolean(target && cardRef.current?.contains(target))
      const isEditable = Boolean(target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable))
      if (modEnter) {
        if (target?.closest('.vq__skip')) return
        // Allow ⌘↵ / Ctrl+Enter after clicking any radio/option button inside the card,
        // without hijacking ⌘↵ steer inside an unrelated textarea (e.g. Composer).
        if (!insideCard && isEditable) return
        if (!canSubmit) return
        event.preventDefault()
        event.stopPropagation()
        submit()
        return
      }
      if (target && (isEditable || target.tagName === 'BUTTON')) return
      event.preventDefault()
      submit()
    }
    window.addEventListener('keydown', onKey)
    window.addEventListener('keydown', onEnter)
    return () => {
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('keydown', onEnter)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [question, selections, others, review, isUrlMode])

  if (isUrlMode) return <UrlConsentCard question={question} />
  if (review) return <PlanReviewCard question={question} plan={plan} />
  if (vibes) {
    // Claude Code's question card: a "Needs your input" pill, the question on a
    // header band, radio/checkbox rows (label over a grey description), an Other
    // row, then Skip and Submit (⌘↵) at the right.
    const skippable = question.items.every(item => item.allowFreeform)
    const skip = (): void => store.answerQuestion(question.requestId, Object.fromEntries(question.items.map(item => [item.id, 'Skipped'])))
    return (
      <div ref={cardRef} tabIndex={-1} className="vq" role="form" aria-label="Agent questions" onKeyDown={handleCardKeyDown}>
        <span className="vq__pill"><span className="vq__ring" aria-hidden="true" />Needs your input</span>
        <div className="vq__card">
          {question.items.map((item, qi) => {
            const picked = selections[item.id] ?? []
            const other = others[item.id] ?? ''
            const multi = item.multiSelect === true
            return (
              <div key={item.id} className="vq__item" data-multi={multi ? 'true' : undefined}>
                <div className="vq__q">
                  <Markdown text={`${question.items.length > 1 ? `${qi + 1}. ` : ''}${item.question}`} className="md--compact" />
                  {multi && <span className="vq__multi-hint">Select all that apply</span>}
                </div>
                <div className="vq__opts" role={multi ? 'group' : 'radiogroup'} aria-label={item.question}>
                  {item.options.map(option => {
                    const on = picked.includes(option)
                    const { label, detail } = optionParts(option)
                    return (
                      <button
                        key={option}
                        type="button"
                        role={multi ? 'checkbox' : 'radio'}
                        aria-checked={on}
                        className="vq__opt"
                        onClick={() => {
                          if (!multi) setOthers(prev => ({ ...prev, [item.id]: '' }))
                          setSelections(prev => ({
                            ...prev,
                            [item.id]: toggleQuestionOption(prev[item.id] ?? [], option, multi),
                          }))
                        }}
                      >
                        {multi
                          ? <span className="vq__check" aria-hidden="true">{on ? <Icon name="check" size={12} /> : null}</span>
                          : <span className="vq__radio" aria-hidden="true" />}
                        <span className="vq__text"><span className="vq__label">{label}</span>{detail && <span className="vq__detail">{detail}</span>}</span>
                      </button>
                    )
                  })}
                  {item.allowFreeform && (
                    <div
                      className="vq__opt vq__opt--other"
                      data-checked={other.trim() !== '' ? 'true' : 'false'}
                    >
                      <button
                        type="button"
                        role={multi ? 'checkbox' : 'radio'}
                        aria-checked={other.trim() !== ''}
                        aria-label={item.placeholder || 'Other'}
                        className="vq__other-toggle"
                        onClick={event => {
                          const input = event.currentTarget.parentElement?.querySelector<HTMLInputElement>('input')
                          input?.focus()
                          if (!multi) setSelections(prev => ({ ...prev, [item.id]: [] }))
                        }}
                      >
                        {multi
                          ? <span className="vq__check" aria-hidden="true">{other.trim() !== '' ? <Icon name="check" size={12} /> : null}</span>
                          : <span className="vq__radio" aria-hidden="true" />}
                      </button>
                      <input
                        aria-label="Other answer"
                        placeholder={item.placeholder || 'Other'}
                        spellCheck={false}
                        value={other}
                        onChange={e => {
                          const value = e.target.value
                          setOthers(prev => ({ ...prev, [item.id]: value }))
                          if (!multi) setSelections(prev => ({ ...prev, [item.id]: [] }))
                        }}
                        onKeyDown={e => { if (!e.nativeEvent.isComposing && e.key === 'Enter' && (e.metaKey || e.ctrlKey || !e.shiftKey)) { e.preventDefault(); e.stopPropagation(); submit() } }}
                      />
                    </div>
                  )}
                </div>
              </div>
            )
          })}
          <div className="vq__actions">
            {skippable && <button type="button" className="vq__skip" title="Answer “Skipped” and let the agent continue" onClick={skip}>Skip</button>}
            <button type="button" className="vq__submit" disabled={!canSubmit} onClick={submit}>Submit <kbd>⌘</kbd><kbd>↵</kbd></button>
          </div>
        </div>
      </div>
    )
  }
  return (
    <div ref={cardRef} tabIndex={-1} className="qcard" role="form" aria-label="Agent questions" onKeyDown={handleCardKeyDown}>
      <div className="qcard__head">
        <span className="approval__dot" aria-hidden="true" />
        <span className="qcard__title">{question.items.length === 1 ? 'The agent has a question' : `The agent has ${question.items.length} questions`}</span>
      </div>
      {question.items.map((item, qi) => {
        const picked = selections[item.id] ?? []
        const multi = item.multiSelect === true
        return (
          <div key={item.id} style={{ display: 'grid', gap: 6 }}>
            {/* The question is the agent's prose: paths in backticks and short
                lists read as they were written, not as raw markup. */}
            <div className="qcard__q">
              <Markdown text={`${question.items.length > 1 ? `${qi + 1}. ` : ''}${item.question}`} className="md--compact" />
              {multi && <span className="vq__multi-hint">Select all that apply</span>}
            </div>
            {item.options.length > 0 && (
              <div className="optlist" role={multi ? 'group' : 'radiogroup'} aria-label={item.question}>
                {item.options.map((option, oi) => {
                  const on = picked.includes(option)
                  return (
                    <button
                      key={option}
                      type="button"
                      role={multi ? 'checkbox' : 'radio'}
                      aria-checked={on}
                      className={`opt${on ? ' is-approve' : ''}`}
                      onClick={() => {
                        if (!multi) setOthers(prev => ({ ...prev, [item.id]: '' }))
                        setSelections(prev => ({
                          ...prev,
                          [item.id]: toggleQuestionOption(prev[item.id] ?? [], option, multi),
                        }))
                      }}
                    >
                      <span className="opt__label">{option}</span>
                      {/* Keys drive the first question only — no kbd badge
                          on later questions would advertise a dead key. */}
                      <span className="opt__kbd">{on ? <Icon name="check" size={12} /> : qi === 0 && oi < 9 ? <kbd>{oi + 1}</kbd> : null}</span>
                    </button>
                  )
                })}
              </div>
            )}
            {item.allowFreeform && (
              <div className="otherbox">
                <input
                  aria-label="Other answer"
                  placeholder={item.placeholder || (item.options.length ? 'Or type your own answer…' : 'Type your answer…')}
                  spellCheck={false}
                  value={others[item.id] ?? ''}
                  onChange={e => {
                    setOthers(prev => ({ ...prev, [item.id]: e.target.value }))
                    if (!multi) setSelections(prev => ({ ...prev, [item.id]: [] }))
                  }}
                  onKeyDown={e => { if (!e.nativeEvent.isComposing && e.key === 'Enter' && (e.metaKey || e.ctrlKey || !e.shiftKey)) { e.preventDefault(); e.stopPropagation(); submit() } }}
                />
              </div>
            )}
          </div>
        )
      })}
      <div className="approval__row">
        <button type="button" className="btn btn--solid" disabled={!canSubmit} onClick={submit}>{question.items.length === 1 ? 'Send answer' : 'Send answers'} <kbd>⏎</kbd></button>
      </div>
    </div>
  )
}

/**
 * A plan review IS a batched question wearing its approval clothes: the
 * markdown proposal above it, one named approve option, everything else (or
 * custom text) keeps planning with feedback.
 */
function PlanReviewCard({
  question,
  plan,
}: {
  question: NonNullable<Snapshot['question']>
  plan: Snapshot['plan']
}): ReactElement {
  const [feedback, setFeedback] = useState('')
  const [picked, setPicked] = useState<string | null>(null)
  // Mirrors for the window-level Enter handler (closures over stale state
  // would send an empty answer).
  const feedbackRef = useRef('')
  const pickedRef = useRef<string | null>(null)
  const readyToSendRef = useRef(false)
  feedbackRef.current = feedback
  pickedRef.current = picked
  const item = question.items[0]
  // No `?? options[0]` fallback: it promoted whatever came first — possibly
  // "Cancel" — to the solid primary button bound to `1`. When nothing reads
  // like approval, every option is just an option.
  const approveOption = item?.options.find(option => /approve|accept|start|proceed|go ahead|yes/i.test(option)) ?? ''
  const otherOptions = item ? item.options.filter(option => option !== approveOption) : []
  /** `1` belongs to the approve action only when there is one. */
  const firstOtherKey = approveOption ? 2 : 1
  const answer = (value: string): void => {
    if (!item) return
    store.answerQuestion(question.requestId, { [item.id]: value })
  }
  const readyToSend = feedback.trim() || picked
  readyToSendRef.current = Boolean(readyToSend)

  useEffect(() => {
    if (!item) return
    const onKey = (event: KeyboardEvent): void => {
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
      if (event.key === '1' && approveOption) {
        event.preventDefault()
        answer(approveOption)
      } else if (/^[1-9]$/.test(event.key) && otherOptions[Number(event.key) - firstOtherKey]) {
        // Only 1 and 2 were bound, so a third option ("Cancel") was
        // mouse-only while its siblings advertised keys.
        event.preventDefault()
        setPicked(otherOptions[Number(event.key) - firstOtherKey]!)
      } else if (event.key === 'Enter' && readyToSendRef.current) {
        // The Send hint says ⏎; honor it wherever focus sits, not only in
        // the feedback input.
        event.preventDefault()
        answer(feedbackRef.current.trim() || pickedRef.current!)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [item, question, approveOption, otherOptions, firstOtherKey])

  if (!item) return <></>

  return (
    <div className="qcard qcard--plan" role="form" aria-label="Plan review">
      <div className="qcard__head">
        <span className="qcard__header-tag">plan review</span>
        <span className="appr-policy">Exit plan · execution waits for your decision</span>
      </div>
      <div className="qcard__q">{item.question}</div>
      {plan && <Markdown text={plan.markdown} />}
      <div className="optlist">
        {approveOption && (
          <button type="button" className="opt is-approve" onClick={() => answer(approveOption)}>
            <span className="opt__label">{approveOption}</span>
            <span className="opt__kbd"><kbd>1</kbd></span>
          </button>
        )}
        {otherOptions.map((option, index) => (
          <button
            key={option}
            type="button"
            className={`opt${picked === option ? ' is-approve' : ''}`}
            onClick={() => setPicked(option)}
          >
            <span className="opt__label">{option}</span>
            {/* Was pinned to index 0, so a daemon sending
                ["Approve", "Cancel", "Revise"] labelled Cancel as the
                revise action. Key off what the option says instead. */}
            {/revis|edit|change|feedback/i.test(option) ? <span className="opt__desc">· tell the agent what to revise</span> : null}
            <span className="opt__kbd">{index + firstOtherKey < 10 ? <kbd>{index + firstOtherKey}</kbd> : null}</span>
          </button>
        ))}
      </div>
      <div className="otherbox">
        <span>or</span>
        <input
          aria-label="Plan revision feedback"
          placeholder="feedback for the next revision…"
          value={feedback}
          onChange={e => setFeedback(e.target.value)}
          onKeyDown={e => {
            if (e.nativeEvent.isComposing) return
            if (e.key === 'Enter' && readyToSend) {
              e.preventDefault()
              answer(feedback.trim() || picked!)
            }
          }}
        />
        {readyToSend ? (
          <button type="button" className="btn btn--solid" onClick={() => answer(feedback.trim() || picked!)}>Send <kbd>⏎</kbd></button>
        ) : null}
      </div>
    </div>
  )
}

// ── Composer ────────────────────────────────────────────────────────────

/**
 * Longest side an attached image keeps. Claude Code resizes anything larger
 * itself, and a resized image stops its prompt cache from extending: every
 * later round of the task re-wrote the whole conversation (measured: a
 * 2400px image read no cache, 2000px read all of it). The API scales images
 * to 1568px anyway, so nothing visible is lost.
 */
export const MAX_IMAGE_EDGE = 2000

/**
 * Largest image the Anthropic API accepts: 5 MB of base64, which is 3.75 MB of
 * bytes. A pasted 3024px render was 7.6 MB, and the model never saw it.
 */
export const MAX_IMAGE_BYTES = 3_750_000

/** `file` at most MAX_IMAGE_EDGE on its longest side; unchanged when it fits or cannot be decoded here. */
async function fittedImage(file: File): Promise<Blob> {
  if (typeof createImageBitmap !== 'function' || typeof document === 'undefined') return file
  let bitmap: ImageBitmap
  try { bitmap = await createImageBitmap(file) } catch { return file }
  try {
    const scale = Math.min(1, MAX_IMAGE_EDGE / Math.max(bitmap.width, bitmap.height))
    if (scale >= 1 && file.size <= MAX_IMAGE_BYTES) return file
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(bitmap.width * scale))
    canvas.height = Math.max(1, Math.round(bitmap.height * scale))
    const context = canvas.getContext('2d')
    if (!context) return file
    context.imageSmoothingQuality = 'high'
    context.drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    const encode = (type: string, quality?: number) => new Promise<Blob | null>(resolve => canvas.toBlob(resolve, type, quality))
    // PNG keeps screenshot text sharp; a photo stays a JPEG. A PNG still over
    // the API's limit (a detailed render) becomes a JPEG, at falling quality.
    const first = await encode(file.type === 'image/jpeg' ? 'image/jpeg' : 'image/png', 0.92)
    if (first && first.size <= MAX_IMAGE_BYTES) return first
    for (const quality of [0.9, 0.8, 0.7, 0.6]) {
      const jpeg = await encode('image/jpeg', quality)
      if (jpeg && jpeg.size <= MAX_IMAGE_BYTES) return jpeg
    }
    return first ?? file
  } finally { bitmap.close() }
}

/** Image files from a paste or drop, read as base64 for `turn.submit`. */
export async function composerImages(files: Iterable<File>): Promise<ComposerImage[]> {
  const images: ComposerImage[] = []
  for (const file of files) {
    if (!file.type.startsWith('image/')) continue
    const fitted = await fittedImage(file)
    const bytes = new Uint8Array(await fitted.arrayBuffer())
    let binary = ''
    for (let offset = 0; offset < bytes.length; offset += 0x8000) binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000))
    images.push({ id: crypto.randomUUID(), mediaType: fitted.type || file.type, data: btoa(binary), name: file.name || 'Pasted image' })
  }
  return images
}

/** What a message with only images says, since a turn needs text. */
export const IMAGE_ONLY_MESSAGE = 'See the attached image.'

function Composer({ snap }: { snap: Snapshot }): ReactElement {
  const open = useDesktopNavigation()
  // The RPC binding key may change on resume; drafts belong to the durable session.
  const workspace = (snap.storageScope ?? '') + snap.cwd
  const key = draftKey(workspace, snap.currentId || snap.sessionKey)
  const [draft, setDraft] = useState(() => readDraft(key))
  const [hints, setHints] = useState<{ items: CompletionHintItem[]; index: number } | null>(null)
  const [images, setImages] = useState<ComposerImage[]>([])
  const attach = (files: Iterable<File>): void => {
    void composerImages(files).then(added => { if (added.length) setImages(current => [...current, ...added]) })
  }
  const ref = useRef<HTMLTextAreaElement>(null)
  const hintSeq = useRef(0)
  const draftSession = useRef({ key, workspace, sessionId: snap.currentId })
  const latestDraft = useRef({ identity: { key, workspace, sessionId: snap.currentId }, text: draft })
  latestDraft.current = { identity: { key, workspace, sessionId: snap.currentId }, text: draft }
  const sendingDraft = useRef(false)
  useEffect(() => {
    const next = { key, workspace, sessionId: snap.currentId }
    const changed = draftSession.current.key !== key
    const value = transitionDraft(draftSession.current, next, draft)
    draftSession.current = next
    if (!changed) return
    setDraft(value)
    setHints(null)
    ref.current?.focus()
  }, [key, workspace, snap.currentId, draft])
  useEffect(() => {
    const add = (event: Event) => { const detail = (event as CustomEvent<unknown>).detail; if (typeof detail === 'string') { setDraft(value => value ? value + ' ' + detail : detail); ref.current?.focus() } }
    const insert = (event: Event) => { const detail = (event as CustomEvent<unknown>).detail; if (typeof detail === 'string') { setDraft(value => applyCompletion(detail) + value); setHints(null); ref.current?.focus() } }
    const prefill = (event: Event) => {
      const detail = (event as CustomEvent<unknown>).detail
      if (typeof detail === 'string') {
        setDraft(detail)
        setHints(null)
        ref.current?.focus()
        if (typeof requestAnimationFrame === 'function') {
          requestAnimationFrame(() => {
            const el = ref.current
            if (!el) return
            el.focus()
            el.setSelectionRange(detail.length, detail.length)
          })
        }
      }
    }
    window.addEventListener('xerxes:add-context', add)
    window.addEventListener('xerxes:insert-command', insert)
    window.addEventListener('xerxes:prefill-prompt', prefill)
    return () => {
      window.removeEventListener('xerxes:add-context', add)
      window.removeEventListener('xerxes:insert-command', insert)
      window.removeEventListener('xerxes:prefill-prompt', prefill)
    }
  }, [])
  const grow = (): void => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    const limit = Math.max(96, Math.min(280, window.innerHeight * .32))
    el.style.height = `${Math.min(el.scrollHeight, limit)}px`
    el.style.overflowY = el.scrollHeight > limit ? 'auto' : 'hidden'
  }
  // Re-measure on every keystroke…
  useLayoutEffect(grow, [draft])
  // …but subscribe once. With [draft] on the subscription effect, every
  // character tore down and rebuilt a ResizeObserver and a window listener.
  useLayoutEffect(() => {
    const observer = new ResizeObserver(grow)
    if (ref.current?.parentElement) observer.observe(ref.current.parentElement)
    window.addEventListener('resize', grow)
    return () => { observer.disconnect(); window.removeEventListener('resize', grow) }
  }, [])
  const send = async (mode: 'default' | 'steer' = 'default'): Promise<void> => {
    if (hints && hints.items.length > 0 && wantsHints(draft)) {
      const item = hints.items[hints.index]
      if (item) {
        pickHint(item)
        return
      }
    }
    if ((!draft.trim() && !images.length) || snap.connection !== 'online' || Boolean(snap.compacting) || store.getSnapshot().submissionPending || sendingDraft.current) return
    open(null)
    const origin = latestDraft.current.identity
    const sent = draft
    const attached = images
    writeDraft(origin.key, sent)
    sendingDraft.current = true
    try {
      const payload = sent.trim() || IMAGE_ONLY_MESSAGE
      const delivered = mode === 'steer' && snap.turnActive
        ? await store.steerNow(payload, attached)
        : await store.submit(payload, undefined, attached)
      if (delivered) {
        setImages(current => current.filter(image => !attached.includes(image)))
        const current = latestDraft.current
        const next = acceptedDraft(origin, current.identity, sent, current.text)
        if (next !== current.text) setDraft(next)
      }
    } finally {
      sendingDraft.current = false
      requestAnimationFrame(grow)
    }
  }

  // Live slash/skill hints: debounced daemon completions while the draft is a
  // bare `/tok` or a `/skill <ref>`; latest request wins, stale ones drop.
  useEffect(() => {
    const seq = ++hintSeq.current
    if (!wantsHints(draft) || snap.connection !== 'online') {
      setHints(null)
      return
    }
    const timer = setTimeout(() => {
      void store
        .completeText(draft)
        .then(items => {
          if (hintSeq.current === seq) setHints({ items, index: 0 })
        })
        .catch(() => { if (hintSeq.current === seq) setHints({items: [], index: 0}) })
    }, 90)
    return () => clearTimeout(timer)
  }, [draft, snap.connection, snap.acpBinding?.connected, snap.acpBinding?.agent_id, snap.acpBinding?.available_commands, snap.acpBinding?.discovered_skills])

  const pickHint = (item: CompletionHintItem): void => {
    hintSeq.current += 1
    setDraft(applyCompletion(item.value))
    setHints(null)
    requestAnimationFrame(grow)
    ref.current?.focus()
  }

  // Keep the keyboard-picked row visible inside the scrollable strip.
  const hintsRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    hintsRef.current
      ?.querySelector('.hints__row.is-on')
      ?.scrollIntoView({ block: 'nearest' })
  }, [hints?.index, hints?.items.length])

  const externalAcp = isExternalAcpBinding(snap.acpBinding)
  const hasExternalAcpSession = externalAcp || Boolean(snap.savedAcpBinding && snap.savedAcpBinding.transport !== 'native')
  const acpSupportsImages = !externalAcp || snap.acpBinding?.capabilities?.prompt.image !== false
  const hasAcpPlanMode = externalAcp && Boolean(snap.acpBinding?.modes?.some(mode => /^(plan|planning)$/i.test(mode.id)))
  const inPlanMode = externalAcp
    ? /^(plan|planning)$/i.test(snap.acpBinding?.current_mode ?? '')
    : snap.planMode
  const ready = snap.connection === 'online' && !snap.compacting
  const placeholder =
    snap.connection !== 'online'
      ? 'Waiting for workspace runtime…'
      : snap.turnActive
        ? 'Queue follow-up (⏎) or steer now (⌘⏎)'
        : snap.compacting
          ? 'Compacting the conversation. You can send when it finishes.'
          : inPlanMode
          ? 'Planning · describe the outcome, or /plan <msg> to steer'
          : 'Describe what you need'
  // VS Code has Source Control for the repo line, and no dictation in a webview.
  const vscode = isVscodeHost()
  // Vibes draws Claude's prompt box: a return-arrow send and plain-text meta.
  const vibes = useVibesStyle()

  return (
    <div
      className="composer-wrap"
      onMouseDown={event => {
        // The whole card is the click target: a press on its padding, the
        // controls row's gaps or the folder row focuses the message box.
        const target = event.target as HTMLElement
        if (target.closest('button,a,input,textarea,select,[role="menu"],[role="dialog"],.hints,.composer__images')) return
        event.preventDefault()
        ref.current?.focus()
      }}
    >
      {hints && (
        <div className="hints">
          <div className="hints__header">
            <strong>{externalAcp ? `${snap.acpBinding!.name} commands & skills` : 'Commands & skills'}</strong>
            <span>{hints.items.length} available</span>
          </div>
          <div id="composer-hints-list" className="hints__list" role="listbox" aria-label="Command and skill hints" ref={hintsRef}>
          {hints.items.length === 0 && <div className="hints__empty">No matches. Try another name or description.</div>}
          {hints.items.map((item, index) => (
            <button
              key={`${item.runtimeProvider ?? item.kind}:${item.value}`}
              id={`composer-hint-${index}`}
              type="button"
              role="option"
              aria-selected={index === hints.index}
              className={`hints__row${index === hints.index ? ' is-on' : ''}`}
              onMouseEnter={() => setHints({ ...hints, index })}
              onClick={() => pickHint(item)}
            >
              <span className="hints__label">
                {item.label}
                <small
                  className={item.runtimeProvider ? 'hints__badge hints__badge--acp' : undefined}
                  data-provider={item.runtimeProvider}
                >
                  {item.badge ?? (item.kind === 'skill' ? 'Skill' : 'Command')}
                </small>
              </span>
              <span className="hints__meta">{item.meta}</span>
            </button>
          ))}
          </div>
          <div className="hints__keys"><span>Type to filter</span><kbd>↵</kbd> / <kbd>Tab</kbd> complete <kbd>↑↓</kbd> pick <kbd>Esc</kbd> dismiss</div>
        </div>
      )}
      {snap.compacting && !snap.turnActive && <div className="streamstatus composer-status" role="status" aria-live="polite"><AgentOrb size={20} state="working" /><span className="streamstatus__phrase">{snap.compacting}</span></div>}
      {/* One status row at a time: every .composer-status shares the composer grid's row, so a
          "Sending…" shown during compaction was drawn over "Compacting…". The message is sent after it. */}
      {snap.submissionPending && !snap.turnActive && !snap.compacting && <div className="streamstatus composer-status" role="status"><AgentOrb size={20} state="connecting" /><span className="streamstatus__phrase">Sending…</span></div>}
      {snap.turnActive && <div className={`streamstatus composer-status${snap.stopArmed ? ' streamstatus--armed' : ''}`}><AgentOrb size={20} state={orbStateOf(snap)} /><span role="status" aria-live="polite">{snap.stopArmed
        ? <span className="streamstatus__phrase">Press <kbd>Esc</kbd> again to stop this task and its agents</span>
        : <span className="streamstatus__phrase">{composerPhraseOf(snap)}</span>}</span><ElapsedClock active={snap.turnActive} since={snap.turnStartedAt} baseSeconds={snap.turnSeconds} format={turnDurOf} className="streamstatus__clock" /></div>}
      <ComposerTaskSummary snap={snap} />
      {!vscode && !vibes && <RepoBar snap={snap} />}
      <AcpActiveBanner snap={snap} />
      <div className="composer-dock">
      <div className="composer" onDragOver={e => { if (acpSupportsImages && Array.from(e.dataTransfer.items).some(item => item.type.startsWith('image/'))) e.preventDefault() }} onDrop={e => { if (!acpSupportsImages) return; const files = Array.from(e.dataTransfer.files).filter(file => file.type.startsWith('image/')); if (files.length) { e.preventDefault(); attach(files) } }}>
        {images.length > 0 && <div className="composer__images" aria-label="Attached images">
          {images.map(image => <div key={image.id} className="composer__image">
            <img src={`data:${image.mediaType};base64,${image.data}`} alt={image.name} />
            <button type="button" aria-label={`Remove ${image.name}`} title="Remove" onClick={() => setImages(current => current.filter(item => item.id !== image.id))}><Icon name="close" size={11} /></button>
          </div>)}
        </div>}
        <textarea
          ref={ref}
          className="composer__input"
          role="combobox"
          aria-autocomplete="list"
          aria-expanded={Boolean(hints)}
          {...(hints ? { 'aria-controls': 'composer-hints-list' } : {})}
          {...(hints && hints.items.length > 0 ? { 'aria-activedescendant': `composer-hint-${hints.index}` } : {})}
          aria-label="Message"
          rows={1}
          value={draft}
          placeholder={placeholder}
          onChange={e => { hintSeq.current += 1; setHints(null); setDraft(e.target.value); grow() }}
          onPaste={e => {
            if (!acpSupportsImages) return
            // A screenshot arrives as a file; text in the same paste still pastes.
            const files = Array.from(e.clipboardData.files).filter(file => file.type.startsWith('image/'))
            if (!files.length) return
            if (!e.clipboardData.getData('text/plain')) e.preventDefault()
            attach(files)
          }}
          onKeyDown={e => {
            if (e.nativeEvent.isComposing) return
            // Shift+Tab switches between plan mode and normal mode, as in Claude Code.
            if ((!externalAcp || hasAcpPlanMode) && e.key === 'Tab' && e.shiftKey && !e.altKey && !e.metaKey && !e.ctrlKey) { e.preventDefault(); store.togglePlanMode(); return }
            if (e.key === 'Enter' && (e.metaKey || e.ctrlKey) && snap.turnActive) { e.preventDefault(); void send('steer') }
            else if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send('default') }
            else if (hints && hints.items.length > 0 && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
              // Hints own the arrows while open; the caret keeps them when closed.
              e.preventDefault()
              const delta = e.key === 'ArrowDown' ? 1 : hints.items.length - 1
              setHints({ ...hints, index: (hints.index + delta) % hints.items.length })
            }
            else if (e.key === 'Tab' && hints && hints.items[hints.index]) {
              e.preventDefault()
              pickHint(hints.items[hints.index]!)
            }
            else if (e.key === 'Escape') {
              if (hints) {
                e.preventDefault()
                e.stopPropagation()
                hintSeq.current += 1
                setHints(null)
              }
              // Global Escape may stop a running turn, but never erases the draft.
            }
          }}
        />
        {/* Stop while a turn runs and nothing is typed; otherwise send (or queue). */}
        {(snap.turnActive || snap.submissionPending) && !draft.trim() && !images.length
          ? <button type="button" className="composer__send composer__send--stop" title="Stop (Esc Esc)" aria-label="Stop" onClick={() => store.cancel()}>{vibes ? <T3Icon name="stopSquare" size={14} /> : <Icon name="stop" size={18} />}</button>
          : <button
              type="button"
              className="composer__send"
              disabled={!ready || (!draft.trim() && !images.length) || snap.submissionPending}
              title={snap.turnActive ? 'Queue for next step (⏎ · ⌘⏎ to steer now)' : 'Send (⏎ · ⇧⏎ newline)'}
              onClick={() => void send('default')}
              aria-label={snap.turnActive ? 'Queue message' : 'Send message'}
            >{vibes ? <T3Icon name="arrowUp" size={16} /> : <Icon name="arrowUp" size={16} />}</button>}
      </div>
      </div>
      {vibes && !vscode ? (
        <div className="composer__toolbar t3-composer-toolbar">
          <button
            type="button"
            className="cchip composer__icon t3-attach-btn"
            data-images-supported={acpSupportsImages ? 'true' : 'false'}
            title={
              acpSupportsImages
                ? 'Add files or folders as context'
                : `Add files or folders as context · ${snap.acpBinding!.name} does not support image attachments`
            }
            aria-label="Add context"
            onClick={() => open('files')}
          >
            <T3Icon name="paperclip" size={16} />
          </button>
          {snap.sideQuestionStreaming && (
            <button
              type="button"
              className="cchip composer__text composer__aside is-on"
              title="Open side question (⌘;)"
              onClick={() => open('btw')}
            >
              Ask aside
            </button>
          )}
          <span className="composer__flex" />
          {snap.turnActive && (draft.trim() || images.length > 0) && (
            <button
              type="button"
              className="cchip composer__text composer__steer"
              disabled={!ready || snap.submissionPending}
              title="Stop the running step and send this message now (⌘⏎)"
              onClick={() => void send('steer')}
            >
              Steer now
            </button>
          )}
          {!hasExternalAcpSession && snap.providerRelay && <RelayBadge relay={snap.providerRelay} />}
          <T3ModelPicker snap={snap} />
          <Dictation compact sessionKey={snap.sessionKey} onText={text => { setDraft(value => (value ? value + ' ' : '') + text) }} />
        </div>
      ) : (
        <div className="composer__toolbar">
          <button
            type="button"
            className="cchip composer__icon"
            data-images-supported={acpSupportsImages ? 'true' : 'false'}
            title={
              acpSupportsImages
                ? 'Add files or folders as context'
                : `Add files or folders as context · ${snap.acpBinding!.name} does not support image attachments`
            }
            aria-label="Add context"
            onClick={() => open('files')}
          ><Icon name="plus" size={16} /></button>
          {snap.turnActive && (draft.trim() || images.length > 0) && (
            <button
              type="button"
              className="cchip composer__text composer__steer"
              disabled={!ready || snap.submissionPending}
              title="Stop the running step and send this message now (⌘⏎)"
              onClick={() => void send('steer')}
            >
              Steer now
            </button>
          )}
          {!vscode && <Dictation compact sessionKey={snap.sessionKey} onText={text=>{setDraft(value=>(value ? value+" " : "")+text)}} />}
          {!externalAcp && (
            <>
              <button
                type="button"
                className="cchip composer__text composer__perm"
                title={`Approval policy for this task: ${snap.permissionMode || 'not configured'} · open settings`}
                onClick={() => store.openSettings('permissions')}
              >{permissionLabel(snap.permissionMode)}</button>
              <button
                type="button"
                className={`cchip composer__text composer__plan${snap.planMode ? ' is-on' : ''}`}
                aria-pressed={snap.planMode}
                title={snap.planMode ? 'Planning first. Switch to work directly.' : 'Plan before making changes.'}
                onClick={() => store.togglePlanMode()}
              >{snap.planMode ? 'Plan first' : 'Plan'}</button>
              <div className="chipanchor">
                <button
                  type="button"
                  className={`cchip composer__text delegation-chip${snap.delegationMode === 'auto' ? '' : ' is-on'}`}
                  aria-haspopup="dialog"
                  aria-expanded={snap.delegationMenuOpen}
                  title="Choose how eagerly this task uses agents and workflows"
                  onClick={() => store.toggleDelegationMenu()}
                >{!vibes && <Icon name="agents" size={14} />}Agents · {snap.delegationMode === 'off' ? 'Off' : snap.delegationMode === 'eager' ? 'Eager' : 'Auto'}</button>
                <PickerLayer>
                {snap.delegationMenuOpen && <DelegationMenu snap={snap} onClose={() => store.closeDelegationMenu()} />}
                </PickerLayer>
              </div>
            </>
          )}
          <button
            type="button"
            className={`cchip composer__text composer__aside${snap.sideQuestionStreaming ? ' is-on' : ''}`}
            disabled={snap.connection !== 'online'}
            title="Ask an ephemeral side question without interrupting or polluting this chat (⌘; or /btw)"
            onClick={() => open('btw')}
          >{!vibes && <Icon name="chat" size={14} />}Ask aside</button>
          <span className="composer__flex" />
          {!hasExternalAcpSession && snap.providerRelay && <RelayBadge relay={snap.providerRelay} />}
          <AcpRuntimePicker snap={snap} compact={vscode} />
          {externalAcp ? (
            <>
              <AcpModelPicker snap={snap} compact={vscode} />
              <AcpThinkingPicker snap={snap} compact={vscode} />
              <AcpModePicker snap={snap} compact={vscode} />
              <AcpExtraConfigPickers snap={snap} compact={vscode} />
            </>
          ) : (
            <>
              <div className="chipanchor composer__model">
                <button
                  type="button"
                  className={`cchip composer__text${snap.model ? '' : ' is-custom'}`}
                  title="Choose model"
                  onClick={() => store.toggleModelMenu()}
                >{snap.model ? bareModelName(snap.model) : 'Choose model'}</button>
                <PickerLayer>
                {snap.modelMenuOpen && <ModelMenu snap={snap} onClose={() => store.closeModelMenu()} />}
                {snap.pickerOpen && <ModelPicker snap={snap} onClose={() => store.closePicker()} />}
                {snap.reasoningPickerOpen && <ReasoningPicker snap={snap} onClose={() => store.closeReasoningPicker()} />}
                </PickerLayer>
              </div>
              {snap.reasoningLevels.length > 0 && <button
                className="cchip composer__text composer__effort"
                title="Reasoning effort — click to change"
                onClick={() => store.toggleReasoningPicker()}
              >{capitalizeWord(snap.reasoningEffort || 'default')}</button>}
              <ModelSplitChip snap={snap} onOpen={() => open('usage')} />
            </>
          )}
          <ContextRing snap={snap} onOpen={() => open('usage')} />
        </div>
      )}
    </div>
  )
}

const capitalizeWord = (word: string): string => word ? word[0]!.toUpperCase() + word.slice(1) : word

/** The approval policy as one short word, like Claude's "Auto". */
function permissionLabel(mode: string): string {
  if (!mode) return 'Permissions'
  const words: Record<string, string> = { 'accept-all': 'Accept all', auto: 'Auto', manual: 'Ask', plan: 'Plan only' }
  return words[mode] ?? capitalizeWord(mode.replace(/[-_]/g, ' '))
}

/** How full the context window is, as a ring; opens Usage. */
function ContextRing({ snap, onOpen }: { snap: Snapshot; onOpen: () => void }): ReactElement {
  const used = snap.contextTokens ?? 0
  const max = snap.contextMax ?? 0
  const share = max > 0 ? Math.min(1, used / max) : 0
  const r = 7, circumference = 2 * Math.PI * r
  const label = max > 0 ? `Context ${Math.round(share * 100)}% · ${used.toLocaleString()} of ${max.toLocaleString()} tokens — open usage` : 'Context usage — open usage'
  return <button className={`cchip composer__icon composer__ring${share >= 0.8 ? ' is-high' : ''}`} title={label} aria-label={label} onClick={onOpen}>
    <svg width="18" height="18" viewBox="0 0 18 18" aria-hidden="true">
      <circle cx="9" cy="9" r={r} className="composer__ring-track" />
      <circle cx="9" cy="9" r={r} className="composer__ring-fill" strokeDasharray={`${circumference * share} ${circumference}`} transform="rotate(-90 9 9)" />
    </svg>
  </button>
}

/** `lines` is null on a runtime that predates line totals. */
interface RepoSummary { readonly branch: string | null; readonly lines: { readonly added: number; readonly removed: number } | null; readonly files: number }

/**
 * The repository line above the composer (Claude's): folder, branch, lines
 * changed. The totals open the Git panel; × hides the line for this folder
 * until the app restarts. Outside a repository, or on a runtime without Git
 * support, it shows the folder alone.
 */
function RepoBar({ snap }: { snap: Snapshot }): ReactElement | null {
  const open = useDesktopNavigation()
  const vibes = useVibesStyle()
  const [repo, setRepo] = useState<RepoSummary | null>(null)
  const [hidden, setHidden] = useState(() => hiddenRepoBars.has(snap.cwd))
  const [creatingWorktree, setCreatingWorktree] = useState(false)
  const [worktreeName, setWorktreeName] = useState('')
  const reqRef = useRef(0)
  useEffect(() => { reqRef.current += 1; setHidden(hiddenRepoBars.has(snap.cwd)); setRepo(null); setCreatingWorktree(false) }, [snap.cwd])
  const refresh = useCallback(async () => {
    if (!snap.cwd || snap.connection !== 'online') return
    const req = ++reqRef.current
    try {
      const result = await desktopCall(window.xerxes, snap.sessionKey, 'git.status', {})
      if (req !== reqRef.current) return
      const status = result.repository as ScmStatus | null | undefined
      if (status !== undefined) store.syncBranch(status?.branch ?? '')
      setRepo(status ? {
        branch: status.branch,
        lines: status.lines ?? null,
        files: status.counts.staged + status.counts.unstaged + status.counts.untracked + status.counts.conflicts,
      } : null)
    } catch {
      // A runtime without git.status: the line shows the folder alone.
      if (req === reqRef.current) setRepo(null)
    }
  }, [snap.cwd, snap.sessionKey, snap.connection])
  // After each turn and each file the agent changes; on focus for outside edits.
  useEffect(() => { if (!snap.turnActive) void refresh() }, [refresh, snap.turnActive, snap.changes.length])
  useEffect(() => {
    const onFocus = () => { void refresh() }
    window.addEventListener('focus', onFocus)
    return () => window.removeEventListener('focus', onFocus)
  }, [refresh])
  if (hidden || !snap.cwd) return null
  const changed = repo !== null && repo.files > 0
  const branchName = repo?.branch ?? snap.branch
  const folder = homeLabel(snap)
  // Line totals when there are any; otherwise (only new files, or an older
  // runtime without totals) the file count — never a misleading "+0 −0".
  const lines = repo?.lines && repo.lines.added + repo.lines.removed > 0 ? repo.lines : null
  const submitWorktree = (): void => {
    const trimmed = worktreeName.trim()
    setCreatingWorktree(false)
    setWorktreeName('')
    if (trimmed) void store.createWorktree(trimmed)
  }
  return <div className={`repobar${vibes ? ' t3-context-strip' : ''}`} data-context-strip={vibes ? 'true' : undefined}>
    <button type="button" className="repobar__where" title={`${snap.cwd} — switch project`} onClick={() => vibes ? store.openWorkspaceMenu() : open('workspace')}>
      {vibes && <ProjectMonogram name={folder} seed={snap.cwd} size={14} />}
      <span className="repobar__folder">{folder}</span>
      {branchName && (
        <span className="repobar__branch">
          {vibes && <Icon name="branch" size={11} />}
          {branchName}
        </span>
      )}
    </button>
    {vibes && (
      <div className="t3-wt-anchor">
        <button
          type="button"
          className="t3-strip-btn"
          title="Create a new isolated git worktree"
          onClick={() => setCreatingWorktree(value => !value)}
        >
          <T3Icon name="worktree" size={12} />
          <span>New worktree</span>
        </button>
        {creatingWorktree && (
          <div className="t3-wt-pop" role="dialog" aria-label="New worktree">
            <input
              type="text"
              autoFocus
              spellCheck={false}
              placeholder="branch / worktree name…"
              aria-label="Worktree branch name"
              value={worktreeName}
              onChange={event => setWorktreeName(event.target.value)}
              onKeyDown={event => {
                if (event.key === 'Enter') { event.preventDefault(); submitWorktree() }
                else if (event.key === 'Escape') { event.preventDefault(); setCreatingWorktree(false); setWorktreeName('') }
              }}
            />
            <button type="button" className="t3-wt-pop__go" disabled={!worktreeName.trim()} onClick={submitWorktree}>Create</button>
          </div>
        )}
      </div>
    )}
    <span className="composer__flex" />
    {changed && <button type="button" className="repobar__diff" title={`${repo.files} changed file${repo.files === 1 ? '' : 's'} — open Git`} onClick={() => open('review')}>
      {lines
        ? <><span className="repobar__added">+{lines.added}</span><span className="repobar__removed">−{lines.removed}</span></>
        : <span>{repo.files} file{repo.files === 1 ? '' : 's'}</span>}
    </button>}
    {repo && <button type="button" className="repobar__action" onClick={() => open('review')}>{changed ? 'Review' : 'Git'}</button>}
    {repo && <button
      type="button"
      className="repobar__action repobar__pr"
      disabled={snap.connection !== 'online' || snap.turnActive || snap.submissionPending}
      title={
        snap.turnActive
          ? 'Available when the current turn finishes'
          : isAntigravityAcpSession(snap)
            ? 'Pre-fill a pull request prompt in the composer'
            : 'Ask the agent to commit, push and open a pull request with gh'
      }
      onClick={() => {
        if (isAntigravityAcpSession(snap)) {
          prefillComposerPrompt(ANTIGRAVITY_PULL_REQUEST_PROMPT)
          return
        }
        void store.submit(pullRequestPrompt(repo.branch ?? snap.branch ?? null), 'Create a pull request')
      }}
    >{vibes && <T3Icon name="gitPr" size={11} />}Create PR</button>}
    {!vibes && <button type="button" className="repobar__close" aria-label="Hide repository line" title="Hide until restart" onClick={() => { hiddenRepoBars.add(snap.cwd); setHidden(true) }}><Icon name="close" size={14} /></button>}
  </div>
}

const hiddenRepoBars = new Set<string>()

interface ChangedTreeFile {
  readonly file: Snapshot['changes'][number]
  readonly name: string
  readonly relPath: string
}

interface ChangedTreeDir {
  readonly key: string
  readonly name: string
  readonly dirs: readonly ChangedTreeDir[]
  readonly files: readonly ChangedTreeFile[]
  readonly adds: number
  readonly dels: number
}

interface MutableChangedDir {
  readonly key: string
  readonly name: string
  readonly dirs: Map<string, MutableChangedDir>
  readonly files: ChangedTreeFile[]
}

/** Relativize an edited path against the workspace cwd so ACP absolute paths fold into clean trees. */
export function relativizeChangedPath(rawPath: string, cwd?: string): string {
  let normalized = rawPath.replaceAll('\\', '/').replace(/\/{2,}/g, '/').replace(/^\.\//, '')
  if (cwd) {
    const normCwd = cwd.replaceAll('\\', '/').replace(/\/{2,}/g, '/').replace(/\/+$/, '')
    if (normCwd) {
      if (normalized === normCwd) {
        const idx = normalized.lastIndexOf('/')
        return idx >= 0 ? normalized.slice(idx + 1) : normalized
      }
      if (normalized.startsWith(`${normCwd}/`)) {
        normalized = normalized.slice(normCwd.length + 1)
      }
    }
  }
  const brainMatch = /(?:^|\/)\.gemini\/antigravity\/brain\/[^/]+\/(.+)$/.exec(normalized)
  if (brainMatch?.[1]) {
    return `artifacts/${brainMatch[1]}`
  }
  return normalized.replace(/^\/+/, '').replace(/\/+$/, '') || rawPath
}

function finalizeChangedDir(node: MutableChangedDir, depth: number): ChangedTreeDir {
  const finalizedDirs = [...node.dirs.values()]
    .map(child => finalizeChangedDir(child, depth + 1))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }))
  const finalizedFiles = [...node.files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }),
  )
  // Compact single-child directory chains below depth 0 (e.g. `src/desktop/renderer`),
  // keeping top-level root directories (`xerxes`, `docs`) collapsed cleanly at depth 0.
  if (depth >= 1 && finalizedFiles.length === 0 && finalizedDirs.length === 1) {
    const sole = finalizedDirs[0]!
    return {
      key: sole.key,
      name: `${node.name}/${sole.name}`,
      dirs: sole.dirs,
      files: sole.files,
      adds: sole.adds,
      dels: sole.dels,
    }
  }
  const adds =
    finalizedDirs.reduce((sum, child) => sum + child.adds, 0) +
    finalizedFiles.reduce((sum, entry) => sum + entry.file.adds, 0)
  const dels =
    finalizedDirs.reduce((sum, child) => sum + child.dels, 0) +
    finalizedFiles.reduce((sum, entry) => sum + entry.file.dels, 0)
  return {
    key: node.key,
    name: node.name,
    dirs: finalizedDirs,
    files: finalizedFiles,
    adds,
    dels,
  }
}

function buildChangedFileTree(
  files: readonly Snapshot['changes'][number][],
  cwd?: string,
): { readonly dirs: readonly ChangedTreeDir[]; readonly files: readonly ChangedTreeFile[]; readonly allDirKeys: readonly string[] } {
  const root: MutableChangedDir = { key: '', name: '', dirs: new Map(), files: [] }
  for (const file of files) {
    const relPath = relativizeChangedPath(file.path, cwd)
    const segments = relPath.split('/').filter(Boolean)
    if (segments.length <= 1) {
      root.files.push({ file, name: segments[0] ?? relPath, relPath })
      continue
    }
    let cursor = root
    let keyPrefix = ''
    for (let i = 0; i < segments.length - 1; i += 1) {
      const seg = segments[i]!
      keyPrefix = keyPrefix ? `${keyPrefix}/${seg}` : seg
      let child = cursor.dirs.get(seg)
      if (!child) {
        child = { key: keyPrefix, name: seg, dirs: new Map(), files: [] }
        cursor.dirs.set(seg, child)
      }
      cursor = child
    }
    cursor.files.push({ file, name: segments[segments.length - 1]!, relPath })
  }
  const dirs = [...root.dirs.values()]
    .map(child => finalizeChangedDir(child, 0))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }))
  const rootFiles = [...root.files].sort((a, b) =>
    a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }),
  )
  const allDirKeys: string[] = []
  const collectKeys = (list: readonly ChangedTreeDir[]): void => {
    for (const dir of list) {
      allDirKeys.push(dir.key)
      collectKeys(dir.dirs)
    }
  }
  collectKeys(dirs)
  return { dirs, files: rootFiles, allDirKeys }
}

function ChangedFileIcon({ name }: { readonly name: string }): ReactElement {
  const dot = name.lastIndexOf('.')
  const ext = dot >= 0 ? name.slice(dot + 1).toLowerCase() : ''
  if (ext === 'tsx' || ext === 'jsx') {
    return (
      <span className="t3-file-ico t3-file-ico--tsx" aria-hidden="true">
        <svg viewBox="0 0 16 16" width="14" height="14" fill="none">
          <circle cx="8" cy="8" r="1.4" fill="currentColor" />
          <ellipse cx="8" cy="8" rx="6.2" ry="2.4" stroke="currentColor" strokeWidth="1.15" />
          <ellipse cx="8" cy="8" rx="6.2" ry="2.4" stroke="currentColor" strokeWidth="1.15" transform="rotate(60 8 8)" />
          <ellipse cx="8" cy="8" rx="6.2" ry="2.4" stroke="currentColor" strokeWidth="1.15" transform="rotate(120 8 8)" />
        </svg>
      </span>
    )
  }
  if (ext === 'css' || ext === 'scss' || ext === 'less') return <span className="t3-file-badge t3-file-badge--css" aria-hidden="true">css</span>
  if (ext === 'ts' || ext === 'mts' || ext === 'cts') return <span className="t3-file-badge t3-file-badge--ts" aria-hidden="true">ts</span>
  if (ext === 'js' || ext === 'mjs' || ext === 'cjs') return <span className="t3-file-badge t3-file-badge--js" aria-hidden="true">js</span>
  if (ext === 'json' || ext === 'jsonc') return <span className="t3-file-badge t3-file-badge--json" aria-hidden="true">{'{}'}</span>
  if (ext === 'md' || ext === 'mdx') return <span className="t3-file-badge t3-file-badge--md" aria-hidden="true">md</span>
  if (ext === 'py' || ext === 'pyi') return <span className="t3-file-badge t3-file-badge--py" aria-hidden="true">py</span>
  if (ext === 'sh' || ext === 'zsh' || ext === 'bash') return <span className="t3-file-badge t3-file-badge--sh" aria-hidden="true">sh</span>
  if (ext === 'yml' || ext === 'yaml' || ext === 'toml' || ext === 'ini') return <span className="t3-file-badge t3-file-badge--cfg" aria-hidden="true">cfg</span>
  return <span className="t3-file-ico" aria-hidden="true"><Icon name="file" size={13} /></span>
}

/**
 * The end-of-turn summary of edited files. A turn that changed something
 * ends in a checkpoint block; the files listed are the ones whose latest
 * edit happened in that turn. Each row's counts are the file's session
 * totals (what the review panel shows), so the header says so when a file
 * was also edited earlier.
 */
export function FilesChangedCard({
  group,
  changes,
  cwd,
}: {
  group: readonly Snapshot['blocks'][number][]
  changes: Snapshot['changes']
  cwd?: string
}): ReactElement | null {
  const vibes = useVibesStyle()
  const open = useDesktopNavigation()
  const [expandedDirs, setExpandedDirs] = useState<ReadonlySet<string>>(() => new Set())
  const checkpoint = group.find(block => block.kind === 'checkpoint')
  if (!checkpoint || checkpoint.kind !== 'checkpoint') return null
  const files = changes.filter(file => file.turn === checkpoint.turn || (file.turns?.includes(checkpoint.turn) ?? false))
  if (!files.length) return null
  const adds = files.reduce((sum, file) => sum + file.adds, 0)
  const dels = files.reduce((sum, file) => sum + file.dels, 0)

  if (vibes) {
    const tree = buildChangedFileTree(files, cwd)
    const allOpen = tree.allDirKeys.length > 0 && tree.allDirKeys.every(key => expandedDirs.has(key))
    const toggleAll = (): void => {
      setExpandedDirs(allOpen ? new Set() : new Set(tree.allDirKeys))
    }
    const toggleDir = (key: string): void => {
      setExpandedDirs(current => {
        const next = new Set(current)
        if (!next.delete(key)) next.add(key)
        return next
      })
    }
    const fmtCompact = (n: number): string => (n >= 1000 ? `${(n / 1000).toFixed(1).replace(/\.0$/, '')}k` : String(n))

    const renderFileRow = (entry: ChangedTreeFile, depth: number): ReactElement => {
      const indentPx = 8 + depth * 18 + (depth > 0 ? 18 : 0)
      return (
        <button
          key={entry.file.path}
          type="button"
          className="t3-changed-file__row"
          style={{ paddingLeft: `${indentPx}px` }}
          title={`${entry.relPath} — open the diff`}
          onClick={() => store.openChange(entry.file.path)}
        >
          <ChangedFileIcon name={entry.name} />
          <span className="t3-changed-file__name">{entry.name}</span>
          {entry.file.isNew && <span className="fileschanged__new">new</span>}
          <span className="t3-changed-file__stats">
            <span className="fileschanged__add">+{fmtCompact(entry.file.adds)}</span>
            <span className="fileschanged__del">-{fmtCompact(entry.file.dels)}</span>
          </span>
        </button>
      )
    }

    const renderDirNode = (dir: ChangedTreeDir, depth: number): ReactElement => {
      const isOpen = expandedDirs.has(dir.key)
      const indentPx = 8 + depth * 18
      return (
        <div key={dir.key} className="t3-changed-dir">
          <button
            type="button"
            className="t3-changed-dir__row"
            style={{ paddingLeft: `${indentPx}px` }}
            aria-expanded={isOpen}
            onClick={() => toggleDir(dir.key)}
          >
            <T3Icon name={isOpen ? 'chevronDown' : 'chevronRight'} size={12} className="t3-changed-dir__chev" />
            <T3Icon name="folder" size={14} className="t3-changed-dir__ico" />
            <span className="t3-changed-dir__name">{dir.name}</span>
            <span className="t3-changed-dir__stats">
              <span className="fileschanged__add">+{fmtCompact(dir.adds)}</span>
              <span className="fileschanged__del">-{fmtCompact(dir.dels)}</span>
            </span>
          </button>
          {isOpen && (
            <div className="t3-changed-tree">
              {dir.dirs.map(child => renderDirNode(child, depth + 1))}
              {dir.files.map(fileEntry => renderFileRow(fileEntry, depth + 1))}
            </div>
          )}
        </div>
      )
    }

    return (
      <section className="fileschanged t3-changed-card" aria-label={`Files changed in turn ${checkpoint.turn}`}>
        <header className="t3-changed-card__head">
          <span className="t3-changed-card__title">{files.length} changed {files.length === 1 ? 'file' : 'files'}</span>
          <span className="t3-changed-card__stats">
            <span className="fileschanged__add">+{fmtCompact(adds)}</span>
            <span className="fileschanged__del">-{fmtCompact(dels)}</span>
          </span>
          <span className="composer__flex" />
          {tree.allDirKeys.length > 0 && (
            <button
              type="button"
              className="t3-changed-card__btn t3-changed-card__btn--ico"
              aria-label={allOpen ? 'Collapse all folders' : 'Expand all folders'}
              title={allOpen ? 'Collapse all folders' : 'Expand all folders'}
              onClick={toggleAll}
            >
              <T3Icon name="chevronsUpDown" size={13} />
            </button>
          )}
          <button type="button" className="t3-changed-card__btn t3-changed-card__open" onClick={() => open('review')}>
            <T3Icon name="fileDiff" size={13} />
            <span>Open diff</span>
          </button>
        </header>
        <div className="t3-changed-card__body">
          {tree.dirs.map(dir => renderDirNode(dir, 0))}
          {tree.files.map(fileEntry => renderFileRow(fileEntry, 0))}
        </div>
      </section>
    )
  }

  return <section className="fileschanged" aria-label={`Files changed in turn ${checkpoint.turn}`}>
    <header className="fileschanged__head">
      <span>Files changed</span>
      <span className="fileschanged__count">{files.length}</span>
      <span className="fileschanged__add">+{adds}</span>
      <span className="fileschanged__del">−{dels}</span>
    </header>
    <ul className="fileschanged__list">
      {files.map(file => <li key={file.path}>
        <button type="button" className="fileschanged__row" title={`${file.path} — open the diff`} onClick={() => store.openChange(file.path)}>
          <Icon name="file" size={13} />
          <span className="fileschanged__path">{relativizeChangedPath(file.path, cwd)}</span>
          {file.isNew && <span className="fileschanged__new">new</span>}
          <span className="fileschanged__add">+{file.adds}</span>
          <span className="fileschanged__del">−{file.dels}</span>
        </button>
      </li>)}
    </ul>
  </section>
}

// ── Right rail ──────────────────────────────────────────────────────────

export function ComposerTaskSummary({ snap }: { snap: Snapshot }): ReactElement | null {
  const open = useDesktopNavigation()
  const goal = parseGoal(snap.goal)
  const activityVisible = useContext(ActivityVisible)
  const visibleGoal = goal && ['active', 'paused', 'blocked'].includes(goal.phase) ? goal : null
  const todos = snap.todos ?? snap.plan?.items.map((item, index) => ({ id: String(index), content: item.text, status: item.done ? 'completed' : 'pending' })) ?? []
  const current = todos.find(item => item.status === 'in_progress')
  const next = current ?? todos.find(item => item.status === 'pending')
  const viewingPlan = snap.tab === 'plan'
  const [expandedPref, setExpandedPref] = useState<boolean | null>(null)
  useEffect(() => { setExpandedPref(null) }, [snap.sessionKey])
  if (!viewingPlan && !visibleGoal && !todos.length && !snap.queue.length) return null
  const done = todos.filter(item => item.status === 'completed').length
  const remaining = Math.max(0, todos.length - done)
  const stepsExpanded = !viewingPlan && todos.length > 0 && (expandedPref ?? remaining > 0)
  const statusSummary = remaining > 0 ? `${remaining} left · ${done}/${todos.length}` : `${done}/${todos.length}`
  return <section className="composer-task-summary" aria-label="Current task">
    {visibleGoal && <button onClick={() => open('activity')} title={visibleGoal.objective}><span className="composer-task-summary__label">Goal</span><span className="composer-task-summary__text">{activityVisible ? 'View goal details' : visibleGoal.objective}</span><span className="composer-task-summary__status">{visibleGoal.phase}</span></button>}
    {viewingPlan && <button onClick={() => store.setTab('activity')} title="Return to the latest messages">
      <Icon name="chat" size={13} /><span className="composer-task-summary__text">Back to conversation</span>
    </button>}
    {!viewingPlan && todos.length > 0 && (
      <div className="composer-task-summary__todos">
        <div className="composer-task-summary__row">
          <button
            type="button"
            className="composer-task-summary__main"
            onClick={() => store.setTab('plan')}
            title={next?.content ?? 'View completed tasks'}
          >
            <span className="composer-task-summary__label">
              {current && <AgentOrb size={12} state={snap.turnActive ? orbStateOf(snap) : 'breathing'} live={snap.turnActive} />}
              {current ? 'Doing' : next ? 'Next' : 'To-dos'}
            </span>
            <span className="composer-task-summary__text">{next?.content ?? 'All tasks completed'}</span>
            <span className="composer-task-summary__status">{statusSummary}</span>
          </button>
          <button
            type="button"
            className={`composer-task-summary__toggle${stepsExpanded ? ' is-open' : ''}`}
            aria-expanded={stepsExpanded}
            aria-label={stepsExpanded ? 'Hide steps' : 'Show steps'}
            title={stepsExpanded ? 'Hide steps' : 'Show steps'}
            onClick={() => setExpandedPref(!(expandedPref ?? remaining > 0))}
          >
            <Icon name="chevron" size={12} />
          </button>
        </div>
        {stepsExpanded && (
          <ol className="composer-task-summary__steps" aria-label="Steps">
            {todos.map((item, index) => (
              <li key={`${item.id}:${index}`} data-state={item.status}>
                <span className="composer-task-summary__mark" aria-hidden="true">
                  {item.status === 'completed'
                    ? <Icon name="check" size={11} />
                    : item.status === 'in_progress'
                      ? <AgentOrb size={16} state={snap.turnActive ? orbStateOf(snap) : 'breathing'} live={snap.turnActive} />
                      : null}
                </span>
                <span className="composer-task-summary__step-text">{item.content}</span>
                <span className="sr-only">
                  {item.status === 'completed' ? 'completed' : item.status === 'in_progress' ? 'in progress' : 'pending'}
                </span>
              </li>
            ))}
          </ol>
        )}
      </div>
    )}
    {snap.queue.length > 0 && <div className="composer-queue"><div className="composer-queue__heading">Queued messages <span>{snap.queue.length}</span></div><div className="composer-queue__list">{snap.queue.map(item => <div className="composer-queue__message" key={item.id}><p>{item.text}</p><div className="composer-queue__actions">{snap.turnActive && <button type="button" className="composer-queue__steer" title="Stop the running step and send this message now" onClick={() => void store.promoteQueued(item.id)}>Steer now</button>}<button type="button" className="composer-queue__drop" aria-label="Hide queued message from view" title="Hide from view only — this does not cancel the queued message" onClick={() => store.dropQueued(item.id)}><Icon name="close" size={14} /></button></div></div>)}</div></div>}
  </section>
}

/** The session's to-dos as an Activity card (Claude layout: no Plan tab). */
function RailTodos({ snap }: { snap: Snapshot }): ReactElement | null {
  const todos = snap.todos ?? snap.plan?.items.map((item, index) => ({ id: String(index), content: item.text, status: item.done ? 'completed' : 'pending' })) ?? []
  if (!todos.length) return null
  const done = todos.filter(item => item.status === 'completed').length
  return <section className="railcard railcard--todos" aria-label="To-dos">
    <header className="railcard__head"><span className="railcard__title">To-dos</span><span className="railcard__meta">{done}/{todos.length}</span></header>
    <ol className="railtodos">
      {todos.map((item, index) => <li key={`${item.id}:${index}`} data-state={item.status}>
        <span className="railtodos__mark" aria-hidden="true">{item.status === 'completed' ? <Icon name="check" size={12} /> : item.status === 'in_progress' ? <AgentOrb size={20} state={snap.turnActive ? orbStateOf(snap) : 'breathing'} live={snap.turnActive} /> : null}</span>
        <span className="railtodos__text">{item.content}</span>
        <span className="sr-only">{item.status === 'completed' ? 'completed' : item.status === 'in_progress' ? 'in progress' : 'pending'}</span>
      </li>)}
    </ol>
    {snap.plan && <button className="railtodos__plan" onClick={() => store.setTab('plan')}>Open the plan</button>}
  </section>
}

function GoalObjective({ text }: { text: string }): ReactElement {
  const [expanded, setExpanded] = useState(false)
  return <><p className={`goal-objective${expanded ? " is-expanded" : ""}`}>{text}</p>{text.length > 180 && <button className="goal-expand" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>{expanded ? "Show less" : "Show full goal"}</button>}</>
}

export function ActivityDetails({ snap, selectedAgent = '' }: { snap: Snapshot; selectedAgent?: string }): ReactElement | null {
  const navigate = useDesktopNavigation()
  const fleet = activityFleetRows(snap.fleet, snap.blocks)
  const goal = parseGoal(snap.goal)
  // The agent open in the conversation's place: its details live here, not in its chat.
  const openAgent = selectedAgent ? fleet.find(row => row.id === selectedAgent || row.agentDetails?.requestKey === selectedAgent) : undefined
  const externalBinding = isExternalAcpBinding(snap.acpBinding) ? snap.acpBinding : null
  const displayExternalBinding =
    externalBinding ?? (snap.savedAcpBinding && snap.savedAcpBinding.transport !== 'native' ? snap.savedAcpBinding : null)
  const railSnap = displayExternalBinding
    ? { ...snap, model: activeAcpModelLabel(displayExternalBinding) }
    : snap
  const acpCatalog = externalBinding ? resolveExternalAcpCatalog(externalBinding) : null
  const acpCaps = externalBinding?.capabilities
  const capabilityBadges = acpCaps
    ? [
        acpCaps.prompt.image ? 'Images' : null,
        acpCaps.prompt.embedded_context ? 'Context' : null,
        acpCaps.prompt.audio ? 'Audio' : null,
        acpCaps.session.resume ? 'Resume' : null,
        acpCaps.session.load ? 'Load' : null,
        acpCaps.mcp.http || acpCaps.mcp.sse ? 'MCP' : null,
      ].filter((label): label is string => Boolean(label))
    : []
  return (
    <aside className="activity-details">
      {openAgent && <AgentDetailsCard key={openAgent.id} row={openAgent} rows={fleet} sessionKey={snap.sessionKey} />}
      {/* Zone 1 — always populated, always first. The rail used to open on
          ten collapsed sections, so it looked identical whether the agent
          was idle, running three subagents, or had failed. */}
      <RailStatus snap={railSnap} />

      {externalBinding && (
        <section className="railcard railcard--environment" aria-label="Environment" data-provider={externalBinding.provider}>
          <header className="railcard__head">
            <span className="railcard__title">Environment</span>
            <span className="railcard__meta">
              {transportBadgeLabel(externalBinding.transport)}
              {externalBinding.agent_version
                ? ` · ${externalBinding.agent_version.startsWith('v') ? externalBinding.agent_version : `v${externalBinding.agent_version}`}`
                : ''}
            </span>
          </header>
          <div className="rail-env">
            <div className="rail-env__row">
              <RuntimeBrandIcon provider={externalBinding.provider} size={14} />
              <span className="rail-env__label">{externalBinding.name}</span>
              {externalBinding.external_session_id && (
                <span className="rail-env__meta" title={externalBinding.external_session_id}>
                  {externalBinding.external_session_id.slice(0, 8)}
                </span>
              )}
            </div>
            <button
              type="button"
              className="rail-env__row rail-env__row--btn"
              title={`Inspect ${externalBinding.name} skills, commands and capabilities`}
              onClick={() => navigate('extensions')}
            >
              <Icon name="tools" size={14} />
              <span className="rail-env__label">
                {acpCatalog && (acpCatalog.skills.length > 0 || acpCatalog.commands.length > 0)
                  ? `${acpCatalog.skills.length} ${acpCatalog.skills.length === 1 ? 'skill' : 'skills'} · ${acpCatalog.commands.length} ${acpCatalog.commands.length === 1 ? 'command' : 'commands'}`
                  : 'Runtime features'}
              </span>
              <span className="rail-env__meta">Customize</span>
            </button>
            {capabilityBadges.length > 0 && (
              <div className="rail-env__caps" aria-label="Runtime capabilities">
                {capabilityBadges.map(badge => (
                  <span key={badge} className="rail-env__cap">{badge}</span>
                ))}
              </div>
            )}
          </div>
        </section>
      )}

      {/* Zone 2 — only what is true right now. Each of these is absent
          unless it has something to say. */}
      <RailTodos snap={snap} />
      {goal ? (
        <section className="railcard railcard--goal" aria-label="Goal">
          {/* A card like Agents and Touched. As a bare caption over loose
              text it had no boundary, and its buttons ran into the next card. */}
          <header className="railcard__head"><span className="railcard__title">Goal</span><span className="railcard__meta" data-phase={goal.phase}>{goal.phase}{goal.activation === 'armed' ? ' · armed' : ''}</span></header>
          <div className="goalcard">
            <GoalObjective text={goal.objective} />
            <GoalInspectorDisclosure snap={snap} />
            {goal.rounds ? <div className="goalcard__rounds">Rounds: {goal.rounds}</div> : null}
            <div className="goalcard__btns">
              {/* The daemon only pauses an active goal and only resumes a
                  paused/blocked one — phase-gate the button instead of
                  offering a verb that always fails. */}
              {goal.phase === 'active' && (
                <button className="chipbtn" onClick={() => void store.submit('/goal pause')}>pause</button>
              )}
              {(goal.phase === 'paused' || goal.phase === 'blocked') && (
                <button className="chipbtn" onClick={() => void store.submit('/goal resume')}>resume</button>
              )}
              <button className="chipbtn" onClick={() => void store.submit('/goal clear')}>clear</button>
            </div>
          </div>
        </section>
      ) : null}

      {snap.swarm?.active ? (() => {
        const swarmCounts = swarmMemberCounts(snap.swarm.members)
        return (
          <section className="railcard railcard--swarm" aria-label="Immortals">
            <header className="railcard__head">
              <span className="railcard__title">{snap.swarm.title || 'Immortals'}</span>
              <span className="railcard__meta" data-tone={swarmCounts.working > 0 ? 'working' : undefined}>
                {swarmCounts.working} working · {swarmCounts.waiting} waiting
              </span>
            </header>
            <div className="railswarm">
              <button type="button" className="chipbtn" onClick={() => navigate('swarm')}>
                Open Immortals Board
              </button>
            </div>
          </section>
        )
      })() : null}

      <RailWorkflows rows={fleet} onInspect={id => navigate('activity', id)} />
      <RailAgents rows={fleet.filter(row => !row.agentDetails?.group)} onInspect={id => navigate('activity', id)} />

      {/* The rows, not a summary line. A rail that only says "5 files" is
          a number you still have to go and decode; the diffs and the undo
          stay in the session-edits tab, one click away. */}
      <RailFiles files={snap.changes} onOpen={path => store.openChange(path)} />
    </aside>
  )
}

// ── Global keys ─────────────────────────────────────────────────────────

/** True for `Alt+S` / macOS `Option+S` (matches `code === 'KeyS'` because macOS reports `key === 'ß'`). */
export function isImmortalsShortcutEvent(
  event: Pick<KeyboardEvent, 'altKey' | 'metaKey' | 'ctrlKey' | 'shiftKey' | 'code'>,
): boolean {
  return event.altKey && !event.metaKey && !event.ctrlKey && !event.shiftKey && event.code === 'KeyS'
}

/** ⌘K palette · ⌘N new task · ⌘, settings · ⌘J terminal drawer · ⌘\ right panel · ⌥S Immortals · Esc stop · 1/2/3 approvals. */
function GlobalKeys({
  snap,
  closeSurface,
  toggleSwarm,
  toggleTerminalDrawer,
  toggleRightPanel,
  undoInbox,
}: {
  snap: Snapshot
  closeSurface: (() => void) | null
  toggleSwarm?: () => void
  toggleTerminalDrawer?: () => void
  toggleRightPanel?: () => void
  undoInbox?: (() => void) | null
}): ReactElement | null {
  const toggleSwarmRef = useRef(toggleSwarm)
  toggleSwarmRef.current = toggleSwarm
  const t3KeysRef = useRef({ toggleTerminalDrawer, toggleRightPanel, undoInbox })
  t3KeysRef.current = { toggleTerminalDrawer, toggleRightPanel, undoInbox }
  // Native menu items are the discoverable half of the keyboard contract;
  // routing them here keeps one implementation per command.
  useEffect(() => window.xerxes.onMenuCommand?.(command => {
    if (command === 'settings') {
      if (store.getSnapshot().settingsOpen) store.closeSettings()
      else store.openSettings()
    }
    else if (command === 'new-task') store.newChat()
    else if (command === 'new-task-wizard') store.openTaskModal()
    else if (command === 'export') void store.exportSessionTranscript(store.getSnapshot().sessionKey)
    else if (command === 'find') window.dispatchEvent(new CustomEvent('xerxes:find'))
    else if (command === 'search') store.openSessionSearch()
    else if (command === 'palette') store.togglePalette()
    else if (command === 'shortcuts') window.dispatchEvent(new CustomEvent('xerxes:shortcuts'))
  }), [])
  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      // A native modal owns keyboard focus, including Escape. Reading output
      // must never cancel the running task underneath it.
      if (event.defaultPrevented || document.querySelector('dialog[open], [role="dialog"][aria-modal="true"]')) return
      if (isImmortalsShortcutEvent(event)) {
        event.preventDefault()
        if (toggleSwarmRef.current) toggleSwarmRef.current()
        else store.openSwarmBoard()
        return
      }
      const meta = event.metaKey || event.ctrlKey
      if (meta && event.key === ',') {
        event.preventDefault()
        if (store.getSnapshot().settingsOpen) store.closeSettings()
        else store.openSettings()
        return
      }
      if (meta && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        store.togglePalette()
        return
      }
      // ⌘N / ⌥⌘N / ⌘, / ⌘F / ⌘⇧F also arrive as menu commands below; these
      // branches keep them working if the native menu is ever unavailable.
      // ⇧⌘N belongs to File ▸ New Window and must not be taken here.
      if (meta && !event.shiftKey && event.key.toLowerCase() === 'n') {
        event.preventDefault()
        // ⌥⌘N is the wizard (workspace, worktree, preset, plan ceiling,
        // model); ⌘N stays the one-keystroke blank task the sidebar button
        // offers. Both now match the labels that advertise them.
        if (event.altKey) store.openTaskModal()
        else store.newChat()
        return
      }
      // Find in the transcript. The primary surface of this app is
      // thousands of lines of tool output and there was no find at all.
      if (meta && event.key.toLowerCase() === 'f') {
        event.preventDefault()
        if (event.shiftKey) store.openSessionSearch()
        else window.dispatchEvent(new CustomEvent('xerxes:find'))
        return
      }
      if (meta && event.key === '/') {
        event.preventDefault()
        window.dispatchEvent(new CustomEvent('xerxes:shortcuts'))
        return
      }
      if (meta && event.key === ';') {
        event.preventDefault()
        store.toggleSideQuestion()
        return
      }
      if ((meta && !event.shiftKey && !event.altKey && event.key.toLowerCase() === 'j') || (event.ctrlKey && !event.metaKey && event.key === '`')) {
        if (t3KeysRef.current.toggleTerminalDrawer) {
          event.preventDefault()
          t3KeysRef.current.toggleTerminalDrawer()
          return
        }
      }
      if (meta && !event.shiftKey && !event.altKey && event.key === '\\') {
        if (t3KeysRef.current.toggleRightPanel) {
          event.preventDefault()
          t3KeysRef.current.toggleRightPanel()
          return
        }
      }
      if (meta && !event.shiftKey && !event.altKey && event.key.toLowerCase() === 'z' && t3KeysRef.current.undoInbox) {
        const target = event.target as HTMLElement | null
        const editing = Boolean(target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable))
        if (!editing) {
          event.preventDefault()
          t3KeysRef.current.undoInbox()
          return
        }
      }
      if (event.key === 'Escape') {
        if (snap.paletteOpen) {
          event.preventDefault()
          store.closePalette()
          return
        }
        if (snap.searchOpen) {
          event.preventDefault()
          store.closeSessionSearch()
          return
        }
        if (snap.taskModalOpen) {
          event.preventDefault()
          store.closeTaskModal()
          return
        }
        if (snap.settingsOpen) {
          event.preventDefault()
          store.closeSettings()
          return
        }
        if (snap.pickerOpen) {
          event.preventDefault()
          store.closePicker()
          return
        }
        if (snap.reasoningPickerOpen) {
          event.preventDefault()
          store.closeReasoningPicker()
          return
        }
        if (snap.delegationMenuOpen) {
          event.preventDefault()
          store.closeDelegationMenu()
          return
        }
        if (snap.modelMenuOpen) {
          event.preventDefault()
          store.closeModelMenu()
          return
        }
        if (snap.contextMenuOpen) {
          event.preventDefault()
          store.closeContextMenu()
          return
        }
        if (snap.wsMenuOpen) {
          event.preventDefault()
          store.closeWorkspaceMenu()
          return
        }
        if (snap.sessionMenu) {
          event.preventDefault()
          store.closeSessionMenu()
          return
        }
        // A full-page destination (Agents / Skills & tools / Artifacts) and
        // the bottom sheets live in Shell state, so the ladder could not
        // see them: Escape fell straight through to cancelling the turn,
        // and DesktopPage has no close button to try instead.
        if (closeSurface) {
          event.preventDefault()
          closeSurface()
          return
        }
        if (snap.tab !== 'activity') {
          event.preventDefault()
          store.setTab('activity')
          return
        }
        if (snap.turnActive) {
          event.preventDefault()
          // Two presses: one stray Escape must not cancel the turn and every
          // agent it spawned. The Stop button stays a single click.
          store.escapeStop()
        }
        return
      }
      // Approval keys. The old guard only excluded INPUT/TEXTAREA, so with
      // focus on any button — where it lands after clicking almost anything
      // in the chrome — a single unmodified digit granted a pending tool
      // call. Require the card itself to own focus instead: it takes focus
      // on arrival, so the advertised keys work exactly when the card is
      // the thing you are looking at, and never by accident.
      if (!snap.approval || event.metaKey || event.ctrlKey || event.altKey) return
      const active = document.activeElement
      if (!(active instanceof HTMLElement) || !active.closest('.approval, .vq--approval')) return
      if (snap.approval.options && snap.approval.options.length > 0) {
        if (/^[1-9]$/.test(event.key)) {
          const picked = snap.approval.options[Number(event.key) - 1]
          if (picked) {
            event.preventDefault()
            store.approveWithOption(snap.approval.id, picked.optionId, approvalOptionDecision(picked.kind))
          }
        }
        return
      }
      const action = approvalKeyAction(event.key, snap.approval.allowPermanent)
      if (action) {
        event.preventDefault()
        store.approve(snap.approval.id, action)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // Every overlay flag the handler branches on must be a dep — a stale
    // closure here swallowed Escape after the task modal closed (the old
    // snap still claimed taskModalOpen, so settings could never dismiss).
  }, [snap.approval, snap.paletteOpen, snap.searchOpen, snap.taskModalOpen, snap.settingsOpen, snap.pickerOpen, snap.reasoningPickerOpen, snap.modelMenuOpen, snap.delegationMenuOpen, snap.contextMenuOpen, snap.wsMenuOpen, snap.sessionMenu, snap.tab, snap.turnActive, closeSurface])
  return null
}
