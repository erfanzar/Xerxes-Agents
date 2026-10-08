// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { GoalInspectorDisclosure } from "./GoalInspector.js"
import { createPortal } from 'react-dom'
import { Fragment, createContext, memo, useCallback, useContext, useLayoutEffect, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactElement } from 'react'

import { CommandPalette, PickerLayer, DelegationMenu, ModelMenu, ModelPicker, ReasoningPicker, SettingsModal, bareModelName } from './Overlays.js'
import { SessionSearch } from './SearchPanel.js'
import { useTranscriptScroll } from './transcriptScroll.js'
import { store, type Snapshot, isPlanReview } from './store.js'
import { connectionFailureKind } from './connectionFailure.js'
import { failureView } from './turnFailure.js'
import type { AgentMember, Block } from './types.js'
import { applyCompletion, wantsHints, type HintItem } from './hints.js'
import { groupByWorkspace, sidebarOrder } from './workspaceGroups.js'
import { CopyButton } from './CopyButton.js'
import { currentSessionDensity, subscribeSessionDensity } from './displayPrefs.js'
import { ChangesTab, LogTab, PlanTab } from './Workspaces.js'
import { Markdown } from './markdown.js'
import { Dictation } from './Dictation.js'
import { draftKey, readDraft, transitionDraft, writeDraft, acceptedDraft } from './drafts.js'
import { PanelDivider, usePanelLayout } from './layout.js'
import { useDialogFocus } from './dialogFocus.js'
import { keyedActivityGroups, isDisclosedActivity } from "./activityGroups.js"
import { activitySummary, approvalTitle, liveActivityPhrase } from './activityPhrase.js'
import { ToolCallRow, toolHasFailed } from "./Execution.js"
import { activityFleetRows, agentState, agentKindLabel } from './AgentRoster.js'
import { RailAgents, RailFiles, RailWorkflows } from './RailLists.js'
import { AgentInspector } from './AgentInspector.js'
import { OutputViewer } from './OutputViewer.js'
import { Icon } from './Icon.js'
import { RelayBadge } from './RelayBadge.js'
import { isVscodeHost } from './host.js'
import { RailStatus, currentActionOf, orbStateOf } from './RailStatus.js'
import { AgentOrb } from './AgentOrb.js'
import { AgentsCard } from './AgentsCard.js'
import { parseGoal } from './goalText.js'
import { AppUpdatePrompt } from './AppUpdatePrompt.js'
// Re-exported: it moved into its own module so the rail's diagnostics
// drawer (DesktopPanels) can import it without a cycle through App.
export { SessionDiagnostics } from './SessionDiagnostics.js'
import { ErrorBoundary } from './ErrorBoundary.js'
import { FindBar } from './FindBar.js'
import { Shortcuts } from './Shortcuts.js'
import { FirstRunSetup } from './Setup.js'
import { RemoteWorkspaceGate } from './RemoteWorkspaceGate.js'
import { BackgroundIndicator, DesktopNavigation, DesktopSheet, DesktopPage, DesktopRail, useDesktopNavigation, type DesktopPanel, type RailPanel } from './DesktopPanels.js'
import { desktopCall } from './desktopRpc.js'
import { pullRequestPrompt } from './GitPanel.js'
import { elapsedOf } from './duration.js'
import type { ScmStatus } from '../../workspace/gitScm.js'

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
  return rail === 'files' ? 'Project files' : rail === 'review' ? 'Source control' : rail === 'terminal' ? 'Terminal' : rail === 'usage' ? 'Usage' : 'Activity'
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
  return currentActionOf(snap)?.verb ?? 'Thinking…'
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
    return () => { current = false; unsubscribe?.() }
  }, [])
  const [panel, setPanel] = useState<DesktopPanel>(null)
  const { layout, setLayout } = usePanelLayout()
  const [narrowNavigation, setNarrowNavigation] = useState(false)
  const [windowWidth, setWindowWidth] = useState(() => typeof window === 'undefined' ? 1440 : window.innerWidth)
  useEffect(() => { const resize = () => setWindowWidth(window.innerWidth); window.addEventListener('resize', resize); return () => window.removeEventListener('resize', resize) }, [])
  const narrow = windowWidth < 850
  const focused = narrow ? !narrowNavigation : layout.sidebarHidden
  const [railChoice, setRail] = useState<RailPanel | null | undefined>(undefined)
  const [filesExpanded, setFilesExpanded] = useState(false)
  const [reviewPath, setReviewPath] = useState('')
  /** A file to open in Files (a tool row's path); the count re-opens the same one. */
  const [filesTarget, setFilesTarget] = useState({ path: '', count: 0 })
  const [selectedAgent, setSelectedAgent] = useState('')
  const contextRequiresFullWidth = windowWidth < (focused ? 0 : layout.sidebarWidth) + layout.inspectorWidth + 320
  // Show activity on entry without covering the conversation in a narrow window.
  // An explicit open/close choice survives session changes and resizing.
  const rail = railChoice === undefined ? (!snap.noWorkspace && !contextRequiresFullWidth ? 'activity' : null) : railChoice
  const toggleSidebar = () => narrow ? setNarrowNavigation(value => !value) : setLayout({ sidebarHidden: !focused })
  // In a narrow window (the VS Code sidebar, a small app window) the task
  // list is a drawer over the conversation: picking a task closes it.
  useEffect(() => { setNarrowNavigation(false) }, [snap.currentId])
  const chrome: WindowChromeState = { sidebarVisible: !focused, inspectorOpen: rail !== null, toggleSidebar }
  const [page, setPage] = useState<'agents' | 'extensions' | 'artifacts' | null>(null)
  // Which stylesheet takeovers are in effect, named once so the decision
  // dock below and the CSS cannot disagree about when .chat is gone.
  // Git sits in the rail like the others; only its expanded diff view takes the window.
  const contextFull = Boolean(rail && (filesExpanded || contextRequiresFullWidth))
  const chatHidden = { contextFull, any: contextFull }
  // The divider must never be draggable past the width that hides the
  // divider itself — the old max let a persisted value strand the rail
  // over the whole window with no control able to shrink it again.
  // Wide windows can give the rail real room; the chat always keeps 320px.
  const inspectorMax = Math.max(260, windowWidth - (focused ? 0 : layout.sidebarWidth) - 320)
  const navigate = (next: DesktopPanel, filePath?: string): void => {
    if (next === 'review') setReviewPath(filePath ?? '')
    if (next === 'files' && filePath) setFilesTarget(target => ({ path: filePath, count: target.count + 1 }))
    // A link to one file's changes opens straight into the diff view.
    if (next === 'review' && filePath && !contextRequiresFullWidth) setFilesExpanded(true)
    if (next === 'activity') setSelectedAgent(filePath ?? '')
    if (next === 'activity') requestAnimationFrame(() => document.querySelector('.desktop-rail .studio-sheet-content')?.scrollTo({ top: 0 }))
    if (next === 'files' || next === 'review' || next === 'terminal' || next === 'activity' || next === 'usage') setRail(next)
    else if (next === 'agents' || next === 'extensions' || next === 'artifacts') { setPage(next); setPanel(null) }
    else if (next === null) { setPage(null); setPanel(null) }
    else setPanel(next)
  }
  useEffect(() => { setPanel(null); setPage(null); setSelectedAgent('') }, [snap.cwd, snap.sessionKey])
  return (
    <DesktopNavigation.Provider value={navigate}><ActivityVisible.Provider value={rail === 'activity'}><WindowChrome.Provider value={chrome}>
    <div className={`app atelier${trafficLights ? '' : ' app--no-traffic-lights'}${focused ? ' atelier--focus' : ''}${narrow ? ' app--narrow' : ''}`} style={{ '--sidebar-width': `${layout.sidebarWidth}px`, '--inspector-width': `${layout.inspectorWidth}px` } as React.CSSProperties}>
      {/* No title bar: its controls live in the sidebar's top row and the
          conversation header — one row, as in Claude's app. */}
      <FirstRunSetup snap={snap} />
      <div className="app__body" data-context-full={chatHidden.contextFull || undefined} data-review={(rail === "review" && contextFull) || undefined}>
        <Sidebar snap={snap} page={page} />
        {narrow && narrowNavigation && <div className="side-scrim" aria-hidden="true" onClick={() => setNarrowNavigation(false)} />}
        {!focused && <PanelDivider label="Resize sessions" value={layout.sidebarWidth} min={180} max={360} onChange={sidebarWidth => setLayout({ sidebarWidth })} />}
        {snap.noWorkspace ? snap.storageScope?.startsWith('ssh:') ? <RemoteWorkspaceGate /> : <WorkspaceGate /> : <ErrorBoundary label="This conversation"><Chat snap={snap} page={page} /></ErrorBoundary>}
        {rail && <PanelDivider label="Resize inspector" value={layout.inspectorWidth} min={260} max={inspectorMax} reverse onChange={inspectorWidth => setLayout({ inspectorWidth })} />}
        {rail && <ErrorBoundary label={railLabel(rail)}><DesktopRail activityFocused={Boolean(selectedAgent)} panel={rail} snap={snap} close={() => setRail(null)} filesExpanded={filesExpanded} reviewPath={reviewPath} filesTarget={filesTarget} {...(!contextRequiresFullWidth ? { toggleFilesExpanded: () => setFilesExpanded(value => !value), setExpanded: setFilesExpanded } : {})} activityDetails={<ActivityDetails snap={snap} selectedAgent={selectedAgent} />} /></ErrorBoundary>}

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
            ? <ApprovalCard approval={snap.approval} policy={snap.permissionMode} />
            : snap.question ? <QuestionCard key={`dock:${snap.question.requestId}`} question={snap.question} plan={snap.plan} /> : null}
        </div>
      )}
      <FindBar />
      <Shortcuts />
      <Announcer snap={snap} />
      <SettingsModal snap={snap} />
      <AppUpdatePrompt update={snap.appUpdate} />
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
      <GlobalKeys snap={snap} closeSurface={panel || page ? () => { setPanel(null); setPage(null) } : null} />
    </div>
    </WindowChrome.Provider></ActivityVisible.Provider></DesktopNavigation.Provider>
  )
}

// ── Top bar ─────────────────────────────────────────────────────────────

/** What the window's top row needs from the shell (it has no title bar). */
interface WindowChromeState {
  readonly sidebarVisible: boolean
  readonly inspectorOpen: boolean
  readonly toggleSidebar: () => void
}

const WindowChrome = createContext<WindowChromeState>({ sidebarVisible: true, inspectorOpen: false, toggleSidebar: () => {} })

function SidebarToggle(): ReactElement {
  const chrome = useContext(WindowChrome)
  return <button className="studio-icon window-toggle" title="Toggle sidebar" aria-label="Toggle sidebar" aria-expanded={chrome.sidebarVisible} aria-controls="session-sidebar" onClick={chrome.toggleSidebar}><Icon name="sidebar" /></button>
}

/** Search, background work and — while the rail is closed — Git and Files. */
function WorkspaceTools({ snap, inspectorOpen }: { snap: Snapshot; inspectorOpen: boolean }): ReactElement {
  const open = useDesktopNavigation()
  return <div className="top__actions" role="group" aria-label="Workspace tools">
    <button title="Search sessions" aria-label="Search sessions" onClick={() => store.openSessionSearch()}><Icon name="search" /></button>
    <BackgroundIndicator snap={snap} />
    {!inspectorOpen && <>
    <button title="Source control" onClick={() => open('review')}><Icon name="branch" /><span>Git</span></button>
    <button title="Project files" aria-label="Project files" onClick={() => open('files')}><Icon name="sidebar" /></button></>}
  </div>
}


/** One workspace-level home for connection, version mismatch and recovery. */
function RuntimeStatus({ snap, compact = false }: { snap: Snapshot; compact?: boolean }): ReactElement {
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
    <button ref={trigger} className={'runtime-status' + (compact ? ' runtime-status--compact' : '')} aria-label={'Workspace runtime: ' + label} aria-expanded={expanded} aria-haspopup="dialog" onClick={() => setExpanded(value => !value)}>
      <span className="runtime-status__dot" data-state={snap.connection} data-warning={Boolean(snap.daemonWarning)} />
      <span>{label}</span>
    </button>
    {expanded && createPortal(<div ref={popup} className="runtime-popover" role="dialog" aria-label="Workspace runtime" tabIndex={-1}>
      <header><strong>Workspace runtime</strong><button aria-label="Close runtime status" onClick={() => { setExpanded(false); trigger.current?.focus() }}><Icon name="close" size={13} /></button></header>
      <p role="status" aria-live="polite">{snap.connection === 'online' ? 'Connected to this workspace' : label}</p>
      {snap.daemonWarning && <><strong>App and runtime versions differ</strong><p>{snap.daemonWarning}</p></>}
      {snap.runtimeUpdateMessage && <p role="status">{snap.runtimeUpdateMessage}</p>}
      {snap.daemonWarning && snap.runtimeUpdate === 'waiting' ? <>
        {(snap.runtimeBlockers?.length ?? 0) > 0 && <><p>Still running:</p><ul className="runtime-popover__blockers">{snap.runtimeBlockers!.map(item => <li key={item}>{item}</li>)}</ul></>}
        <p>Restarting now stops that work. Goals pick up again in the new runtime.</p>
        <div className="runtime-popover__actions">
          <button onClick={() => { setExpanded(false); navigate('activity') }}>Review activity</button>
          <button className="runtime-popover__danger" disabled={busy} onClick={() => void store.restartRuntimeNow()}>Restart now</button>
        </div>
      </> : snap.daemonWarning && <><p>The current runtime keeps your tasks running. Review activity before restarting.</p><div className="runtime-popover__actions"><button onClick={() => { setExpanded(false); navigate('activity') }}>Review activity</button><button disabled={busy} onClick={() => void store.restartDaemon()}>Restart workspace runtime</button></div></>}
      {!snap.daemonWarning && snap.connection !== 'online' && <button onClick={() => { setExpanded(false); navigate('workspace') }}>Open workspace settings</button>}
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
  const groups = groupByWorkspace(snap.sessions, snap.cwd, snap.workspaceDirectories)
  const statusFor = (cwd: string): string => (cwd === snap.cwd ? 'current' : `${groups.find(g => g.cwd === cwd)?.rows.length ?? 0} tasks`)
  return (
    <>
      <div className="backdrop backdrop--clear" onClick={() => store.closeWorkspaceMenu()} />
      {/* role="menu" owns only menuitem children; the plain buttons and
          captions here made it an invalid tree that screen readers report
          as an empty menu. A labelled group describes what is actually
          rendered. */}
      <div className="wsmenu" role="group" aria-label="Workspaces">
        <div className="cap" style={{ padding: '6px 10px 2px' }}>Workspaces</div>
        {groups.map(group => (
          <button
            key={group.cwd || group.name}
            aria-current={group.cwd === snap.cwd ? 'true' : undefined}
            className={`wsrow${group.cwd === snap.cwd ? ' is-cur' : ''}`}
            title={group.cwd === snap.cwd ? `${group.cwd} — you are here` : `Switch to ${group.cwd}`}
            onClick={() => { if (group.cwd !== snap.cwd) store.enterWorkspace(group.cwd) }}
          >
            <span className={`dot ${group.cwd === snap.cwd ? 'dot--live' : 'dot--idle'}`} />
            <span className="wsrow__main">
              <span className="wsrow__t">{group.name}</span>
              <span className="wsrow__s">{group.cwd || group.name} · {statusFor(group.cwd)}</span>
            </span>
            {group.cwd === snap.cwd && <span className="kbd"><Icon name="check" size={11} /></span>}
          </button>
        ))}
        {groups.length === 0 && (
          <div className="wsrow is-cur">
            <span className="dot dot--live" />
            <span className="wsrow__main">
              <span className="wsrow__t">{workspaceLabel(snap.cwd) || 'No workspace'}</span>
              <span className="wsrow__s">{snap.cwd || 'choose a folder to begin'}</span>
            </span>
            <span className="kbd"><Icon name="check" size={11} /></span>
          </div>
        )}
        <div className="menu__sep" />
        <button className="menu__item" onClick={() => { store.closeWorkspaceMenu(); store.chooseWorkspace() }}>
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
  const objectiveField = useRef<HTMLTextAreaElement>(null)
  useDialogFocus(objectiveField)
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
    <div className="backdrop">
      <div className="modal taskmodal" role="dialog" aria-modal="true" aria-label="New task">
        <div className="modal__main">
          <h2 className="modal__title">New task</h2>
          <p className="modal__sub">Choose how you want to work, then describe your task.</p>

          <div className="field">
            <label>Workspace</label>
            <div className="seg" style={{ flexWrap: 'wrap' }}>
              <button className="is-on" title={snap.cwd}>▣ {workspaceLabel(snap.cwd) || 'no workspace'}</button>
              <button onClick={() => store.chooseWorkspace()} title="Open a different folder as the workspace"><Icon name="plus" size={13} /> different folder…</button>
              <button
                onClick={() => setNamingWorktree(value => !value)}
                title="Create an isolated git worktree next to the repo and switch into it"
              >
                <Icon name="plus" size={12} /> new worktree…
              </button>
              {namingWorktree && (
                <input
                  className="taskmodal__wt"
                  autoFocus
                  spellCheck={false}
                  placeholder="worktree name (branch + folder)"
                  value={worktreeName}
                  onChange={e => setWorktreeName(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter') { e.preventDefault(); submitWorktree() }
                    if (e.key === 'Escape') { e.preventDefault(); setNamingWorktree(false); setWorktreeName('') }
                  }}
                />
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
            <label>Objective</label>
            <textarea
              ref={objectiveField}
              className="composer__input taskmodal__objective"
              rows={4}
              value={objectiveText}
              spellCheck={false}
              placeholder="What would you like to get done?"
              onChange={e => setObjectiveText(e.target.value)}
              onKeyDown={e => {
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
              className={`switch${planFirst ? ' is-on' : ''}`}
              role="switch"
              aria-checked={planFirst}
              aria-label="Review plan before changes"
              onClick={() => setPlanFirst(value => !value)}
            />
          </div>

          <div className="field">
            <label>Model</label>
            <select aria-label="Model" value={model} disabled={starting} onChange={event => setModel(event.target.value)}>
              {!model && <option value="">Choose a model</option>}
              {models.map(id => <option key={id} value={id}>{id}</option>)}
            </select>
          </div>

          {/* The mode is pinned per session, not per workspace — saying
              "in this workspace" promised the next task would inherit it.
              And the affordance was a bare <u>: not focusable, not a
              button, the only one of its kind in the renderer. */}
          <div className="fieldnote" style={{ marginBottom: 16 }}>
            approvals for this task: {snap.permissionMode || 'daemon policy'} —{' '}
            <button className="linkish" onClick={() => { store.closeTaskModal(); store.openSettings('permissions') }}>change</button>
          </div>

          <div role="status" className="taskmodal__error">{snap.error}</div>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <button className="btn btn--ghost" disabled={starting} onClick={() => store.closeTaskModal()}>Cancel</button>
            <button className="btn" disabled={starting || !objectiveText.trim() || !model} onClick={start}>{starting ? 'Starting…' : 'Start task'}</button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Session context menu (mockup 08) ────────────────────────────────────

/** execCommand copy — the renderer has no clipboard capability on file:// and none is exposed. */
function copyText(text: string): void {
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
  const [renaming, setRenaming] = useState(menu?.renaming ?? false)
  const [draft, setDraft] = useState(menu?.title ?? '')
  const menuRef = useRef<HTMLDivElement>(null)
  useLayoutEffect(() => {
    const element = menuRef.current
    if (!element || !menu) return
    const previous = document.activeElement
    const place = () => { const rect = element.getBoundingClientRect(); element.style.left = `${Math.max(8, Math.min(menu.x, window.innerWidth - rect.width - 8))}px`; element.style.top = `${Math.max(8, Math.min(menu.y, window.innerHeight - rect.height - 8))}px` }
    place()
    element.querySelector<HTMLElement>('button,input')?.focus()
    window.addEventListener('resize', place)
    return () => { window.removeEventListener('resize', place); if(previous instanceof HTMLElement && previous.isConnected) previous.focus() }
  }, [menu?.key, menu?.x, menu?.y, renaming])
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
      }} className={'menu' + (renaming ? ' menu--renaming' : '')} role={renaming ? 'form' : 'menu'} aria-label={renaming ? 'Rename session' : 'Session actions'} style={{ left: menu.x, top: menu.y }}>
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
            <button className="menu__item" onClick={submitRename} disabled={menu.pending || !draft.trim()} aria-label="Save session title">{menu.pending ? 'Saving…' : 'Save'}</button>
          </div>
        ) : (
          <>
            <button role="menuitem" className="menu__item" onClick={() => { store.closeSessionMenu(); void store.openSession(menu.id) }}>
              <Icon name="arrow" size={13} /> Open <kbd className="kbd">⏎</kbd>
            </button>
            <button role="menuitem" className="menu__item" onClick={() => setRenaming(true)}>
              <Icon name="note" size={13} /> Rename…
            </button>
            <div className="menu__sep" />
            <button role="menuitem" className="menu__item" onClick={() => { copyText(menu.id); store.closeSessionMenu() }}>
              <Icon name="copy" size={13} /> Copy ID
            </button>
            <button role="menuitem" className="menu__item" onClick={() => { void store.exportSessionTranscript(menu.key) }}>
              <Icon name="download" size={13} /> Export as Markdown
            </button>
            <div className="menu__sep" />
            <button role="menuitem" className="menu__item menu__item--danger" disabled={menu.pending} onClick={() => {
              // Permanent: the saved transcript is removed, not archived.
              if (window.confirm(`Delete “${menu.title || 'this chat'}”? Its saved history is removed permanently.`)) void store.deleteSession(menu.id)
            }}>
              <Icon name="trash" size={13} /> {menu.pending ? 'Deleting…' : 'Delete…'}
            </button>
          </>
        )}
        {menu.error && <p className="session-menu-error" role="alert">{menu.error}</p>}
      </div>
    </>
  )
}

// ── Statusline ──────────────────────────────────────────────────────────

// ── Sidebar ─────────────────────────────────────────────────────────────

function Sidebar({ snap, page }: { snap: Snapshot; page: 'agents' | 'extensions' | 'artifacts' | null }): ReactElement {
  const open = useDesktopNavigation()
  const chrome = useContext(WindowChrome)
  const [filter, setFilter] = useState('')
  const needle = filter.trim().toLowerCase()
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
  const online = snap.connection === 'online'

  return (
    <aside id="session-sidebar" className="side" aria-label="Sessions">
      <div className="side__head"><SidebarToggle /></div>
      <div className="side__pad">
        <div className="sidebar-wordmark" aria-label="Xerxes">XERXES</div>
        {/* `page !== 'agents'` lit Sessions while Skills & tools or
            Artifacts was open, so two destinations claimed to be current. */}
        <div className="studio-segment" role="group" aria-label="Sidebar view"><button aria-pressed={page === null} className={page === null ? 'is-selected' : ''} onClick={() => open(null)}>Tasks</button><button aria-pressed={page === 'agents'} className={page === 'agents' ? 'is-selected' : ''} onClick={() => open('agents')}>Agents</button></div>
        <button
          className="newchat"
          disabled={!online}
          onClick={() => { if (!snap.turnActive) open(null); store.newChat() }}
          title={homeLabel(snap) ? `Start a new task in ${homeLabel(snap)} (⌘N, or ⇧⌘N to choose a preset and worktree)` : 'Start a new task (⌘N)'}
        ><Icon name="plus" /><span>New task</span><kbd>⌘ N</kbd></button>
        <button className={`studio-nav${page === "extensions" ? " is-selected" : ""}`} aria-pressed={page === 'extensions'} disabled={!online} onClick={() => open('extensions')}><Icon name="tools" /><span>Skills & tools</span></button>
        <button className={`studio-nav${page === "artifacts" ? " is-selected" : ""}`} aria-pressed={page === 'artifacts'} disabled={!online} onClick={() => open('artifacts')}><Icon name="folder" /><span>Artifacts</span></button>
        <button className="studio-nav" disabled={!online} onClick={() => open('schedules')}><Icon name="clock" /><span>Scheduled jobs</span></button>
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
      </div>
      <nav className="side__list">
        {snap.contexts?.map(context => (
          <div className="wgroup" key={`context-${context.id}`}>
            <button className="wgroup__cap" title={context.workspace} onClick={() => void store.activateContext(context.id)}>
              <Icon name="folder" size={14} /> {context.label}
            </button>
            {context.sessions.filter(row => !filter || row.title.toLowerCase().includes(filter.toLowerCase())).map(row => (
              <button className="sess" key={row.id} title={`${context.label} · ${row.cwd}`} onClick={() => void store.activateContext(context.id, row.id)}>
                <span className="sess__dot" style={{ background: statusColor(row.status) }} />
                <span className="sess__body"><span className="sess__t">{row.title}</span><span className="sess__s">{workspaceLabel(row.cwd)}{row.status === 'working' ? ' · working' : ''}</span></span>
              </button>
            ))}
          </div>
        ))}
        {groups.map(group => (
          <div key={group.cwd} className={`wgroup${group.cwd === snap.cwd ? ' is-home' : ''}`}>
            <div className="wgroup__head">
              <button
                className="wgroup__cap"
                title={group.cwd === snap.cwd ? `${group.cwd} — you are here` : `Switch to ${group.cwd}`}
                onClick={() => { if (group.cwd !== snap.cwd) store.enterWorkspace(group.cwd) }}
              >
                <span className="wgroup__mark" data-here={group.cwd === snap.cwd || undefined} aria-hidden="true" />
                {group.name}
                {group.cwd === snap.cwd && <span className="wgroup__cur">current</span>}
              </button>
              <button className="wgroup__add" disabled={!online} title={`New task in ${group.name}`} aria-label={`New task in ${group.name}`} onClick={() => store.newChatIn(group.cwd)}><Icon name="plus" size={14} /></button>
            </div>
            {group.rows.map(row => {
              const snippet = snap.snippets[row.id]
              return snippet === undefined
                ? <SessionCell key={row.id} row={row} opensWindow={snap.turnActive && row.id !== snap.currentId} />
                : <SessionCell key={row.id} row={row} snippet={snippet} opensWindow={snap.turnActive && row.id !== snap.currentId} />
            })}
          </div>
        ))}
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
      <div className="studio-side-bottom"><RuntimeStatus snap={snap} /><button onClick={() => store.openSettings()}><Icon name="settings" /><span>Settings</span></button><button onClick={() => open('workspace')}><Icon name="folder" /><span>{homeLabel(snap) && snap.cwd ? homeLabel(snap) : 'Choose a folder…'}</span></button></div>
    </aside>
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



function Chat({ snap, page }: { snap: Snapshot; page: 'agents' | 'extensions' | 'artifacts' | null }): ReactElement {
  const open = useDesktopNavigation()
  const chrome = useContext(WindowChrome)
  const changes = snap.changes
  const totals = useMemo(() => ({
    adds: changes.reduce((sum, file) => sum + file.adds, 0),
    dels: changes.reduce((sum, file) => sum + file.dels, 0),
  }), [changes])
  const needsInput = snap.approval !== null || snap.question !== null

  return (
    <main className="chat">
      <header className="chat__head chat__head--minimal">
        {!chrome.sidebarVisible && <SidebarToggle />}
        <span className="chat__title">{snap.currentTitle || (snap.connection === 'online' ? 'New task' : 'Not connected')}</span>
        <span className="hchip chat__kind" title="Agent preset for this session · fixed after its first turn">{snap.currentAgentPreset === 'creator' ? 'Creator mode' : (snap.currentAgentPreset || 'default')}</span>
        <span className="composer__flex" />
        {/* On Session edits or the plan, the approval card is off screen: this is the signal. */}
        {needsInput && <button className="badge badge--need" title={snap.interactionHidden ? 'Show the request waiting for you' : undefined} onClick={() => { store.showInteraction(); store.setTab('activity') }}>needs input{snap.turnActive ? ' · acting paused' : ''}</button>}
        {page === null && snap.tab !== 'activity' && <button className="chat__view" onClick={() => store.setTab('activity')}><Icon name="chat" size={13} /> Conversation</button>}
        {page === null && changes.length > 0 && <button className={`chat__view${snap.tab === 'changes' ? ' is-on' : ''}`} aria-pressed={snap.tab === 'changes'} onClick={() => store.setTab(snap.tab === 'changes' ? 'activity' : 'changes')}>
          Session edits <span className="pillcount"><span className="add">+{totals.adds}</span> <span className="del">−{totals.dels}</span></span>
        </button>}
        {!chrome.sidebarVisible && <RuntimeStatus snap={snap} compact />}
        {!chrome.inspectorOpen && <WorkspaceTools snap={snap} inspectorOpen={false} />}
      </header>

      {page && <DesktopPage panel={page} snap={snap} />}
      <div className="conversation-surface" hidden={page !== null}>
      {snap.connection !== 'online' && snap.blocks.length > 0 && <ConnectionBanner snap={snap} />}
      {snap.tab === 'activity' && <Stream snap={snap} />}
      {snap.tab === 'changes' && <div className="workspace"><ChangesTab snap={snap} /></div>}
      {snap.tab === 'plan' && <div className="workspace"><PlanTab snap={snap} /></div>}
      {snap.tab === 'log' && <div className="workspace workspace--log"><LogTab snap={snap} /></div>}

      </div>
      <Composer snap={snap} />
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

function Stream({ snap }: { snap: Snapshot }): ReactElement {
  const { ref, loadOlder, following, scrollToLatest } = useTranscriptScroll(`${snap.currentId}:${snap.sessionOpenRevision}`, { more: Boolean(snap.historyMore), loading: Boolean(snap.historyLoading), automatic: !snap.historyError, load: () => store.loadOlderHistory() })
  // 'Stream thinking' off hides reasoning trails from the feed — the daemon
  // still streams them; this is a display choice, not a policy change.
  const visible = snap.streamThinking ? snap.blocks : snap.blocks.filter(b => b.kind !== 'thinking')
  const blocks = snap.failed && !snap.turnActive ? withoutDuplicateFailure(visible, snap.failed.error) : visible
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
          {(snap.historyMore || snap.historyError) && <div className="history-pager">
            <button className="btn" disabled={snap.historyLoading} onClick={() => void loadOlder()}>{snap.historyLoading ? 'Loading earlier conversation…' : snap.historyError ? 'Retry loading earlier conversation' : 'Show earlier conversation'}</button>
            {snap.historyError && <p role="alert">{snap.historyError}</p>}
          </div>}
          {keyedActivityGroups(blocks, approval?.toolCallId).map(({ key, blocks: group }, index, groups) => (
            <div key={`${snap.currentId}:${key}`} data-history-anchor={group[0]!.id}>
              {isDisclosedActivity(group, approval?.toolCallId) ? <ActivityGroup blocks={group} active={snap.turnActive && index === groups.length - 1} turnActive={snap.turnActive} /> : group.map(block => <BlockView key={block.id} block={block} />)}
              {inlineApproval && group.some(block => block === blocks[approvalIndex]) && <ApprovalCard approval={approval} policy={snap.permissionMode} inline />}
            </div>
          ))}
          {failedCard}
        </div>
      )}
      {floatApproval && <div className="stream__col"><ApprovalCard approval={approval} policy={snap.permissionMode} /></div>}
      {question && (
        <div className="stream__col">
          <QuestionCard key={`${snap.currentId}:${question.requestId}`} question={question} plan={snap.plan} />
        </div>
      )}
      {/* Scrolling up silently unpins the tail; without this the only way
          back was to drag into a 64px band by hand, and a turn that kept
          streaming gave no sign it had moved on without you. */}
      {!following && blocks.length > 0 && (
        <button className="jump-latest" onClick={scrollToLatest}>
          {snap.turnActive ? 'Still working — jump to latest' : 'Jump to latest'} <Icon name="chevron" size={12} />
        </button>
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
  return <section className="task-continuation"><Icon name="activity" size={24}/><h1>Continue this task</h1><p>This session has saved work. Review its progress or send your next instruction below.</p><div><button className="btn" onClick={() => open('activity')}>Review activity</button>{(snap.todos?.length || snap.plan) ? <button className="btn" onClick={() => store.setTab('plan')}>View plan &amp; todos</button> : null}</div></section>
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
    <AgentOrb className="welcome__orb" size={32} state="breathing" />
    <h1 className="welcome__compact-title">What should Xerxes do{place ? <> in <b>{place}</b></> : ''}?</h1>
    {dirty > 0 && <p className="welcome__dirty">{dirty} uncommitted {dirty === 1 ? 'file' : 'files'} <b>+{adds}</b> <i>−{dels}</i></p>}
    {resumable.length > 0 && <section className="welcome__resume" aria-label="Recent tasks here">
      <h2>Recent</h2>
      {resumable.map(row => <button className="resumerow" key={row.id} onClick={() => void store.openSession(row.id)}>
        <span className="resumerow__dot" data-state={agentState(row.status).tone} aria-hidden="true" />
        <span className="resumerow__name">{row.title}</span>
        <span className="resumerow__age">{row.age}</span>
      </button>)}
    </section>}
  </div>

  return <div className="welcome">
    {/* No settle time: it stopped after nine seconds and read as frozen. The
        ticker still pauses it while the window is hidden, unfocused or covered. */}
    <AgentOrb className="welcome__orb" size={64} state="breathing" />
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
            <button className="resumerow" key={row.id} onClick={() => void store.openSession(row.id)}>
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
        <button className="idea" key={label} onClick={() => window.dispatchEvent(new CustomEvent('xerxes:add-context', { detail: prompt }))}>
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
        <div className="wsgate__mark"><Icon name="folder" size={26} /></div>
        <h1>Welcome to Xerxes.</h1>
        <p>
          Choose the folder you want to work in. Next, connect a model provider and start your first task. Your existing files stay in place.
        </p>
        <button className="btn btn--solid wsgate__btn" onClick={() => store.chooseWorkspace()}>
          Choose your first folder…
        </button>
        <p className="studio-muted">The macOS app includes Bun. Git and SSH are needed only for their respective features.</p>
      </div>
    </main>
  )
}

function Offline({ cwd, error }: { cwd: string; error: string | null }): ReactElement {
  const kind = connectionFailureKind(error)
  return (
    <div className="offline">
      <div className="offline__dot" />
      <h1>{kind === 'transport' ? 'Connecting to the shared daemon' : kind === 'session' ? 'Session belongs to another workspace' : 'Could not open this workspace'}</h1>
      {error && <p className="connection-error" role="alert">{error}</p>}
      <p>
        The terminal, TUI and desktop app share a daemon across workspaces. Each task keeps its own workspace and session. The app connects automatically when the daemon is available.
      </p>
      <button className="btn" onClick={() => store.retryConnection()}><Icon name="retry" size={13} /> Retry now</button>
      {cwd && <div className="cmd">Workspace: {cwd}</div>}
    </div>
  )
}

function ActivityGroup({ blocks, active, turnActive }: { blocks: Snapshot['blocks']; active: boolean; turnActive: boolean }): ReactElement {
  const tools = blocks.flatMap(block => block.kind === 'tools' ? block.items : [])
  const failures = tools.filter(toolHasFailed).length + blocks.filter(block => block.kind === 'notice' && block.error).length + blocks.reduce((total, block) => total + (block.kind === 'agents' ? block.members.filter(member => member.status === 'failed').length : 0), 0)
  const liveAgents = blocks.reduce((total, block) => total + (block.kind === 'agents' ? block.members.filter(member => member.status === 'working' || member.status === 'running').length : 0), 0)
  // Subagents may outlive the turn. That is not the turn still "Working" —
  // the group reads as finished and notes who is still out.
  const running = active || tools.some(item => item.state === 'working') || blocks.some(block => block.kind === 'thinking' && block.streaming) || turnActive && liveAgents > 0
  const [expanded, setExpanded] = useState(false)
  const [inspected, setInspected] = useState(false)
  return <details className={`activity-group${running ? ' is-running' : ''}`} open={expanded} onToggle={event => { setExpanded(event.currentTarget.open); if (event.currentTarget.open) setInspected(true) }}>
    {/* One quiet line, the chevron trailing the text: "Ran 3 commands, read a file ›". */}
    <summary>{running
      ? <span className="activity-group__live"><span className="activity-group__working">Working</span><LivePhrase text={liveActivityPhrase(blocks)} /></span>
      : <span className="activity-group__done">{activitySummary(blocks)}{liveAgents > 0 && <span className="activity-group__pending"> · {liveAgents} agent{liveAgents === 1 ? '' : 's'} still running</span>}</span>}{failures > 0 && <strong>{failures} failed</strong>}<Icon name="chevron" size={13} /></summary>
    <div className="activity-group__body">{(expanded || inspected) && blocks.map((block, index) => <BlockView key={block.id} block={block} />)}</div>
  </details>
}

/**
 * One status phrase that crossfades when it changes: the outgoing phrase
 * lifts away while the incoming one settles in the same grid cell, so the
 * header never jumps width mid-swap. Reduced motion swaps instantly (CSS).
 */
function LivePhrase({ text }: { text: string }): ReactElement {
  const [phrase, setPhrase] = useState<{ text: string; key: number; leaving: { text: string; key: number } | null }>({ text, key: 0, leaving: null })
  useEffect(() => {
    setPhrase(prev => prev.text === text ? prev : { text, key: prev.key + 1, leaving: { text: prev.text, key: prev.key } })
  }, [text])
  useEffect(() => {
    if (!phrase.leaving) return
    const timer = setTimeout(() => setPhrase(prev => ({ ...prev, leaving: null })), 320)
    return () => clearTimeout(timer)
  }, [phrase.leaving])
  return <span className="live-phrase">
    {phrase.leaving && <span key={phrase.leaving.key} className="live-phrase__item is-leaving" aria-hidden="true">{phrase.leaving.text}</span>}
    <span key={phrase.key} className={`live-phrase__item${phrase.key ? ' is-entering' : ''}`} title={phrase.text}>{phrase.text}</span>
  </span>
}

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

function UserMessage({ text }: { text: string }): ReactElement {
  const [expanded, setExpanded] = useState(false)
  const long = text.length > 1600 || text.split('\n').length > 16
  // The wrapper carries the bubble's right alignment so the copy row can sit
  // under the bubble rather than inside it.
  return <div className="msg-turn msg-turn--user">
    <div className="msg msg--user">
      <div className={`msg__text${long && !expanded ? ' msg__text--preview' : ''}`}>{text}</div>
      {long && <button className="message-expand" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>{expanded ? 'Show less' : 'Show full message'}</button>}
    </div>
    <div className="msg__actions"><CopyButton icon text={text} label="Copy message" /></div>
  </div>
}

// Memoized: the turn clock re-renders the conversation every second, and an
// unchanged block (same object) has nothing new to draw.
const BlockView = memo(function BlockView({ block }: { block: Snapshot['blocks'][number] }): ReactElement {
  if (block.kind === 'agents') {
    return <AgentsCard members={block.members} />
  }
  if (block.kind === 'user') {
    if (block.contextSummary) return <details className="context-summary"><summary>Earlier conversation summary</summary><Markdown text={block.text} /></details>
    return <UserMessage text={block.text} />
  }
  if (block.kind === 'agent') {
    return (
      <section className="msg msg--agent">
        {/* Agent prose IS markdown — code fences, tables, links, lists. The
            renderer is React-elements-only, so model output cannot inject
            markup; the streaming caret trails only the live tail block. */}
        <div className="msg__text">
          <Markdown text={block.text} className="md--agent" tail={block.streaming ? CARET : undefined} />
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
const COMMAND_KEYS = ['command', 'cmd', 'script', 'shell'] as const

/**
 * What the decision is actually about. The daemon ships the tool's real
 * arguments; rendering them per shape — the command for a shell call, the
 * path + diff for an edit, labelled fields otherwise — is the difference
 * between approving `send_message(telegram)` and approving a message you
 * have read.
 */
function ApprovalBody({ approval }: { approval: NonNullable<Snapshot['approval']> }): ReactElement | null {
  const inputs = approval.inputs
  if (!inputs) return approval.description ? <pre className="approval__desc">{approval.description}</pre> : null
  const entries = Object.entries(inputs).filter(([, value]) => value !== null && value !== undefined && value !== '')
  if (entries.length === 0) return approval.description ? <pre className="approval__desc">{approval.description}</pre> : null
  const command = entries.find(([key]) => (COMMAND_KEYS as readonly string[]).includes(key))
  if (command && typeof command[1] === 'string') {
    const rest = entries.filter(entry => entry !== command)
    return <>
      <pre className="approval__desc">{command[1]}</pre>
      {rest.length > 0 && <ApprovalFields entries={rest} />}
    </>
  }
  return <ApprovalFields entries={entries} />
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

function ApprovalCard({ approval, policy, inline }: { approval: NonNullable<Snapshot['approval']>; policy: string; inline?: boolean }): ReactElement {
  const card = useRef<HTMLDivElement>(null)
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
  return (
    <div ref={card} tabIndex={-1} className={`approval${inline ? ' approval--inline' : ''}`} role="alertdialog" aria-modal="false" aria-label={`Approve ${approval.toolName || approval.action || 'tool call'}`}>
      <div className="approval__head">
        <span className="approval__dot" aria-hidden="true" />
        <span className="approval__title" title={approval.toolName || approval.action || undefined}>{approvalTitle(approval.toolName || approval.action || '')}</span>
        <button className="approval__dismiss" title="Hide until you decide — the request stays pending" aria-label="Hide this request" onClick={() => store.dismissInteraction()}><Icon name="close" size={13} /></button>
      </div>
      {approval.reason && <p className="approval__reason">{approval.reason}</p>}
      <ApprovalBody approval={approval} />
      <div className="approval__row">
        <button className="btn btn--solid" onClick={() => store.approve(approval.id, 'allow_once')}>Allow once <kbd>1</kbd></button>
        <button className="btn" onClick={() => store.approve(approval.id, 'allow_session')}>This session <kbd>2</kbd></button>
        <button className="btn btn--danger" onClick={() => store.approve(approval.id, 'deny')}>Deny <kbd>3</kbd></button>
      </div>
      <p className="approval__cwd" title={approval.cwd || undefined}>
        <span className="appr-policy">Policy: {policy || 'runtime default'}</span>{approval.cwd && <> · in {approval.cwd}</>}
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
              disabled={!failed.lastUser}
              // Still offered when futile — the limit may have reset, and
              // hiding it would strand a user who knows it has — but never
              // as the obvious next thing to click.
              title={view.retryIsFutile ? 'This will fail again until the cause above is resolved' : failed.lastUser ? 'Continue from where the turn stopped' : undefined}
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
function QuestionCard({
  question,
  plan,
}: {
  question: NonNullable<Snapshot['question']>
  plan: Snapshot['plan']
}): ReactElement {
  const [selections, setSelections] = useState<Record<string, string[]>>({})
  const [others, setOthers] = useState<Record<string, string>>({})
  const review = isPlanReview(question)
  const canSubmit = question.items.length > 0 && question.items.every(item =>
    (item.allowFreeform && !!others[item.id]?.trim()) || (selections[item.id]?.length ?? 0) > 0)
  const submit = (): void => {
    if (!canSubmit) return
    const answers: Record<string, string> = {}
    for (const item of question.items) {
      const custom = others[item.id]?.trim()
      const picked = selections[item.id] ?? []
      if (custom) answers[item.id] = custom
      else if (picked.length) answers[item.id] = picked.join(', ')
    }
    store.answerQuestion(question.requestId, answers)
  }
  // Number keys choose options while the card is up (not while typing); the
  // hints are only honest for the FIRST question — later ones have no keys.
  useEffect(() => {
    if (review) return // PlanReviewCard owns its keyboard responses.
    const onKey = (event: KeyboardEvent): void => {
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'BUTTON' || target.isContentEditable)) return
      const number = Number.parseInt(event.key, 10)
      if (!Number.isFinite(number) || number < 1) return
      const item = question.items[0]
      if (!item) return
      const option = item.options[number - 1]
      if (!option) return
      event.preventDefault()
      setSelections(prev => ({ ...prev, [item.id]: [option] }))
      setOthers(prev => ({ ...prev, [item.id]: '' }))
    }
    const onEnter = (event: KeyboardEvent): void => {
      if (event.key !== 'Enter') return
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'BUTTON' || target.isContentEditable)) return
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
  }, [question, selections, others, review])

  if (review) return <PlanReviewCard question={question} plan={plan} />
  return (
    <div className="qcard" role="form" aria-label="Agent questions">
      <div className="qcard__head">
        <span className="approval__dot" aria-hidden="true" />
        <span className="qcard__title">{question.items.length === 1 ? 'The agent has a question' : `The agent has ${question.items.length} questions`}</span>
      </div>
      {question.items.map((item, qi) => {
        const picked = selections[item.id] ?? []
        return (
          <div key={item.id} style={{ display: 'grid', gap: 6 }}>
            {/* The question is the agent's prose: paths in backticks and short
                lists read as they were written, not as raw markup. */}
            <div className="qcard__q"><Markdown text={`${question.items.length > 1 ? `${qi + 1}. ` : ''}${item.question}`} className="md--compact" /></div>
            {item.options.length > 0 && (
              <div className="optlist">
                {item.options.map((option, oi) => {
                  const on = picked.includes(option)
                  return (
                    <button
                      key={option}
                      className={`opt${on ? ' is-approve' : ''}`}
                      aria-pressed={on}
                      onClick={() => {
                        setOthers(prev => ({ ...prev, [item.id]: '' }))
                        // One answer per question: a second click moves the
                        // choice instead of sending "A, B".
                        setSelections(prev => ({ ...prev, [item.id]: on ? [] : [option] }))
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
                    setSelections(prev => ({ ...prev, [item.id]: [] }))
                  }}
                />
              </div>
            )}
          </div>
        )
      })}
      <div className="approval__row">
        <button className="btn btn--solid" disabled={!canSubmit} onClick={submit}>{question.items.length === 1 ? 'Send answer' : 'Send answers'} <kbd>⏎</kbd></button>
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
  if (!item) return <></>
  // No `?? options[0]` fallback: it promoted whatever came first — possibly
  // "Cancel" — to the solid primary button bound to `1`. When nothing reads
  // like approval, every option is just an option.
  const approveOption = item.options.find(option => /approve|accept|start|proceed|go ahead|yes/i.test(option)) ?? ''
  const otherOptions = item.options.filter(option => option !== approveOption)
  /** `1` belongs to the approve action only when there is one. */
  const firstOtherKey = approveOption ? 2 : 1
  const answer = (value: string): void => store.answerQuestion(question.requestId, { [item.id]: value })
  const readyToSend = feedback.trim() || picked
  readyToSendRef.current = Boolean(readyToSend)

  useEffect(() => {
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
  }, [question, approveOption, otherOptions, firstOtherKey])

  return (
    <div className="qcard qcard--plan" role="form" aria-label="Plan review">
      <div className="qcard__head">
        <span className="qcard__header-tag">plan review</span>
        <span className="appr-policy">exit plan — execution waits for this</span>
      </div>
      <div className="qcard__q">{item.question}</div>
      {plan && <Markdown text={plan.markdown} />}
      <div className="optlist">
        {approveOption && (
          <button className="opt is-approve" onClick={() => answer(approveOption)}>
            <span className="opt__label">{approveOption}</span>
            <span className="opt__kbd"><kbd>1</kbd></span>
          </button>
        )}
        {otherOptions.map((option, index) => (
          <button
            key={option}
            className={`opt${picked === option ? ' is-approve' : ''}`}
            onClick={() => setPicked(option)}
          >
            <span className="opt__label">{option}</span>
            {/* Was pinned to index 0, so a daemon sending
                ["Approve", "Cancel", "Revise"] labelled Cancel as the
                revise action. Key off what the option says instead. */}
            {/revis|edit|change|feedback/i.test(option) ? <span className="opt__desc">— tell the agent what to revise</span> : null}
            <span className="opt__kbd">{index + firstOtherKey < 10 ? <kbd>{index + firstOtherKey}</kbd> : null}</span>
          </button>
        ))}
      </div>
      <div className="otherbox">
        <span>or</span>
        <input
          placeholder="feedback for the next revision…"
          spellCheck={false}
          value={feedback}
          onChange={e => setFeedback(e.target.value)}
          onKeyDown={e => {
            if (e.key === 'Enter' && readyToSend) {
              e.preventDefault()
              answer(feedback.trim() || picked!)
            }
          }}
        />
        {readyToSend ? (
          <button className="btn btn--solid" onClick={() => answer(feedback.trim() || picked!)}>Send <kbd>⏎</kbd></button>
        ) : null}
      </div>
    </div>
  )
}

// ── Composer ────────────────────────────────────────────────────────────

function Composer({ snap }: { snap: Snapshot }): ReactElement {
  const open = useDesktopNavigation()
  // The RPC binding key may change on resume; drafts belong to the durable session.
  const workspace = (snap.storageScope ?? '') + snap.cwd
  const key = draftKey(workspace, snap.currentId || snap.sessionKey)
  const [draft, setDraft] = useState(() => readDraft(key))
  const [hints, setHints] = useState<{ items: HintItem[]; index: number } | null>(null)
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
    window.addEventListener('xerxes:add-context', add)
    window.addEventListener('xerxes:insert-command', insert)
    return () => { window.removeEventListener('xerxes:add-context', add); window.removeEventListener('xerxes:insert-command', insert) }
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
  const send = async (): Promise<void> => {
    if (wantsHints(draft)) {
      const seq = hintSeq.current
      const items = hints?.items ?? await store.completeText(draft).catch(() => [])
      if (seq !== hintSeq.current) return
      const item = items[hints?.index ?? 0]
      if (item) pickHint(item)
      else setHints({items: [], index: 0})
      return
    }
    if (!draft.trim() || snap.connection !== 'online' || store.getSnapshot().submissionPending || sendingDraft.current) return
    open(null)
    const origin = latestDraft.current.identity
    const sent = draft
    writeDraft(origin.key, sent)
    sendingDraft.current = true
    try {
      if (await store.submit(sent)) {
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
  }, [draft, snap.connection])

  const pickHint = (item: HintItem): void => {
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

  const ready = snap.connection === 'online' && !snap.compacting
  const placeholder =
    snap.connection !== 'online'
      ? 'Connect to a daemon first…'
      : snap.turnActive
        ? 'Steer now — queued until this step settles'
        : snap.compacting
          ? 'Compacting the conversation — you can send when it finishes'
          : snap.planMode
          ? 'Planning — describe the outcome, or /plan <msg> to steer'
          : 'Describe what you need'
  // VS Code has Source Control for the repo line, and no dictation in a webview.
  const vscode = isVscodeHost()

  return (
    <div className="composer-wrap">
      {hints && (
        <div className="hints">
          <div className="hints__header">
            <strong>Commands & skills</strong>
            <span>{hints.items.length} available</span>
          </div>
          <div className="hints__list" role="listbox" aria-label="Command and skill hints" ref={hintsRef}>
          {hints.items.length === 0 && <div className="hints__empty">No matches. Try another name or description.</div>}
          {hints.items.map((item, index) => (
            <button
              key={item.value}
              type="button"
              role="option"
              aria-selected={index === hints.index}
              className={`hints__row${index === hints.index ? ' is-on' : ''}`}
              onMouseEnter={() => setHints({ ...hints, index })}
              onClick={() => pickHint(item)}
            >
              <span className="hints__label">{item.label}<small>{item.kind === "skill" ? "Skill" : "Command"}</small></span>
              <span className="hints__meta">{item.meta}</span>
            </button>
          ))}
          </div>
          <div className="hints__keys"><span>Type to filter</span><kbd>↵</kbd> / <kbd>tab</kbd> complete <kbd>↑↓</kbd> pick <kbd>esc</kbd> dismiss</div>
        </div>
      )}
      {snap.compacting && !snap.turnActive && <div className="streamstatus composer-status" role="status" aria-live="polite"><AgentOrb size={20} state="working" /><span className="streamstatus__phrase">{snap.compacting}</span></div>}
      {snap.submissionPending && !snap.turnActive && <div className="streamstatus composer-status" role="status"><AgentOrb size={20} state="connecting" /><span className="streamstatus__phrase">Sending…</span></div>}
      {snap.turnActive && <div className={`streamstatus composer-status${snap.stopArmed ? ' streamstatus--armed' : ''}`} role="status" aria-live="polite"><AgentOrb size={20} state={orbStateOf(snap)} />{snap.stopArmed
        ? <span className="streamstatus__phrase">Press <kbd>esc</kbd> again to stop this task and its agents</span>
        : <span className="streamstatus__phrase">{composerPhraseOf(snap)}</span>}<span className="streamstatus__clock">{turnDurOf(snap.turnSeconds)}</span></div>}
      <ComposerTaskSummary snap={snap} />
      {!vscode && <RepoBar snap={snap} />}
      <div className="composer-dock">
      <div className="composer">
        <textarea
          ref={ref}
          className="composer__input"
          aria-label="Message"
          rows={1}
          value={draft}
          placeholder={placeholder}
          spellCheck={false}
          onChange={e => { hintSeq.current += 1; setHints(null); setDraft(e.target.value); grow() }}
          onKeyDown={e => {
            if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send() }
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
        {(snap.turnActive || snap.submissionPending) && !draft.trim()
          ? <button className="composer__send composer__send--stop" title="Stop (esc esc)" aria-label="Stop" onClick={() => store.cancel()}><Icon name="stop" size={18} /></button>
          : <button
              className="composer__send"
              disabled={!ready || !draft.trim() || snap.submissionPending}
              title={snap.turnActive ? 'Queue — runs when this step settles (⏎)' : 'Send (⏎ · ⇧⏎ newline)'}
              onClick={send}
              aria-label={snap.turnActive ? 'Queue message' : 'Send message'}
            ><Icon name="arrowUp" size={16} /></button>}
      </div>
      </div>
      <div className="composer__toolbar">
        <button className="cchip composer__icon" title="Add files or folders as context" aria-label="Add context" onClick={() => open('files')}><Icon name="plus" size={16} /></button>
        {!vscode && <Dictation compact sessionKey={snap.sessionKey} onText={text=>{setDraft(value=>(value ? value+" " : "")+text)}} />}
        <button
          className="cchip composer__text"
          title={`Approval policy for this task: ${snap.permissionMode || 'not configured'} — open settings`}
          onClick={() => store.openSettings('permissions')}
        >{permissionLabel(snap.permissionMode)}</button>
        <button
          className={`cchip composer__text${snap.planMode ? ' is-on' : ''}`}
          aria-pressed={snap.planMode}
          title={snap.planMode ? 'Planning first. Click to work directly.' : 'Click to plan before making changes.'}
          onClick={() => store.togglePlanMode()}
        >{snap.planMode ? 'Plan first' : 'Plan'}</button>
        <div className="chipanchor">
          <button
            className={`cchip composer__text delegation-chip${snap.delegationMode === 'auto' ? '' : ' is-on'}`}
            aria-haspopup="dialog"
            aria-expanded={snap.delegationMenuOpen}
            title="How eagerly this task uses agents and workflows — click to change"
            onClick={() => store.toggleDelegationMenu()}
          ><Icon name="agents" size={14} />Agents · {snap.delegationMode === 'off' ? 'Off' : snap.delegationMode === 'eager' ? 'Eager' : 'Auto'}</button>
          <PickerLayer>
          {snap.delegationMenuOpen && <DelegationMenu snap={snap} onClose={() => store.closeDelegationMenu()} />}
          </PickerLayer>
        </div>
        <span className="composer__flex" />
        {snap.providerRelay && <RelayBadge relay={snap.providerRelay} />}
        <div className="chipanchor">
          <button
            className={`cchip composer__text${snap.model ? '' : ' is-custom'}`}
            title="Model — click to change"
            onClick={() => store.toggleModelMenu()}
          >{snap.model ? bareModelName(snap.model) : 'Choose model'}</button>
          <PickerLayer>
          {snap.modelMenuOpen && <ModelMenu snap={snap} onClose={() => store.closeModelMenu()} />}
          {snap.pickerOpen && <ModelPicker snap={snap} onClose={() => store.closePicker()} />}
          {snap.reasoningPickerOpen && <ReasoningPicker snap={snap} onClose={() => store.closeReasoningPicker()} />}
          </PickerLayer>
        </div>
        {snap.reasoningLevels.length > 0 && <button
          className="cchip composer__text"
          title="Reasoning effort — click to change"
          onClick={() => store.toggleReasoningPicker()}
        >{capitalizeWord(snap.reasoningEffort || 'default')}</button>}
        <ContextRing snap={snap} onOpen={() => open('usage')} />
      </div>
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
  const [repo, setRepo] = useState<RepoSummary | null>(null)
  const [hidden, setHidden] = useState(() => hiddenRepoBars.has(snap.cwd))
  useEffect(() => { setHidden(hiddenRepoBars.has(snap.cwd)); setRepo(null) }, [snap.cwd])
  const refresh = useCallback(async () => {
    if (!snap.cwd || snap.connection !== 'online') return
    try {
      const result = await desktopCall(window.xerxes, snap.sessionKey, 'git.status', {})
      const status = result.repository as ScmStatus | null | undefined
      setRepo(status ? {
        branch: status.branch,
        lines: status.lines ?? null,
        files: status.counts.staged + status.counts.unstaged + status.counts.untracked + status.counts.conflicts,
      } : null)
    } catch {
      // A runtime without git.status: the line shows the folder alone.
      setRepo(null)
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
  // Line totals when there are any; otherwise (only new files, or an older
  // runtime without totals) the file count — never a misleading "+0 −0".
  const lines = repo?.lines && repo.lines.added + repo.lines.removed > 0 ? repo.lines : null
  return <div className="repobar">
    <button className="repobar__where" title={`${snap.cwd} — change folder`} onClick={() => open('workspace')}>
      <span className="repobar__folder">{homeLabel(snap)}</span>
      {(repo?.branch ?? snap.branch) && <span className="repobar__branch">{repo?.branch ?? snap.branch}</span>}
    </button>
    <span className="composer__flex" />
    {changed && <button className="repobar__diff" title={`${repo.files} changed file${repo.files === 1 ? '' : 's'} — open Git`} onClick={() => open('review')}>
      {lines
        ? <><span className="repobar__added">+{lines.added}</span><span className="repobar__removed">−{lines.removed}</span></>
        : <span>{repo.files} file{repo.files === 1 ? '' : 's'}</span>}
    </button>}
    {repo && <button className="repobar__action" onClick={() => open('review')}>{changed ? 'Review' : 'Git'}</button>}
    {repo && <button
      className="repobar__action repobar__pr"
      disabled={snap.connection !== 'online' || snap.turnActive || snap.submissionPending}
      title={snap.turnActive ? 'Available when the current turn finishes' : 'Ask the agent to commit, push and open a pull request with gh'}
      onClick={() => void store.submit(pullRequestPrompt(repo.branch ?? snap.branch ?? null), 'Create a pull request')}
    >Create PR</button>}
    <button className="repobar__close" aria-label="Hide repository line" title="Hide until restart" onClick={() => { hiddenRepoBars.add(snap.cwd); setHidden(true) }}><Icon name="close" size={14} /></button>
  </div>
}

const hiddenRepoBars = new Set<string>()

// ── Right rail ──────────────────────────────────────────────────────────

export function ComposerTaskSummary({ snap }: { snap: Snapshot }): ReactElement | null {
  const open = useDesktopNavigation()
  const goal = parseGoal(snap.goal)
  const activityVisible = useContext(ActivityVisible)
  const visibleGoal = goal && ['active', 'paused', 'blocked'].includes(goal.phase) ? goal : null
  const todos = snap.todos ?? snap.plan?.items.map((item, index) => ({ id: String(index), content: item.text, status: item.done ? 'completed' : 'pending' })) ?? []
  const current = todos.find(item => item.status === 'in_progress')
  const next = current ?? todos.find(item => item.status === 'pending')
  if (!visibleGoal && !todos.length && !snap.queue.length) return null
  const done = todos.filter(item => item.status === 'completed').length
  return <section className="composer-task-summary" aria-label="Current task">
    {visibleGoal && <button onClick={() => open('activity')} title={visibleGoal.objective}><span className="composer-task-summary__label">Goal</span><span className="composer-task-summary__text">{activityVisible ? 'View goal details' : visibleGoal.objective}</span><span className="composer-task-summary__status">{visibleGoal.phase}</span></button>}
    {todos.length > 0 && <button onClick={() => store.setTab('plan')} title={next?.content ?? 'View completed tasks'}><span className="composer-task-summary__label">{current ? 'Doing' : next ? 'Next' : 'To-dos'}</span><span className="composer-task-summary__text">{next?.content ?? 'All tasks completed'}</span><span className="composer-task-summary__status">{done}/{todos.length}</span></button>}
    {snap.queue.length > 0 && <div className="composer-queue"><div className="composer-queue__heading">Queued messages <span>{snap.queue.length}</span></div><div className="composer-queue__list">{snap.queue.map(item => <div className="composer-queue__message" key={item.id}><p>{item.text}</p><button aria-label="Hide queued message from view" title="Hide from view only — this does not cancel the queued message" onClick={() => store.dropQueued(item.id)}><Icon name="close" size={14} /></button></div>)}</div></div>}
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
  if (selectedAgent) {
    const row = fleet.find(agent => agent.id === selectedAgent || agent.agentDetails?.requestKey === selectedAgent)
    return <section className="agent-inspector"><button className="agent-inspector__back" onClick={() => navigate('activity')}><Icon name="arrow" size={12} /> All activity</button>{row ? <AgentInspector key={snap.sessionKey + ':' + row.id} row={row} rows={fleet} sessionKey={snap.sessionKey} online={snap.connection === 'online'} /> : <p role="status">This agent is no longer available in this session. Return to activity to refresh its status.</p>}</section>
  }
  const goal = parseGoal(snap.goal)
  return (
    <aside className="activity-details">
      {/* Zone 1 — always populated, always first. The rail used to open on
          ten collapsed sections, so it looked identical whether the agent
          was idle, running three subagents, or had failed. */}
      <RailStatus snap={snap} />

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

/** ⌘K palette · ⌘N new task · ⌘, settings · Esc stop · 1/2/3 approvals. */
function GlobalKeys({ snap, closeSurface }: { snap: Snapshot; closeSurface: (() => void) | null }): ReactElement | null {
  // Native menu items are the discoverable half of the keyboard contract;
  // routing them here keeps one implementation per command.
  useEffect(() => window.xerxes.onMenuCommand?.(command => {
    if (command === 'settings') store.openSettings()
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
      if (event.defaultPrevented || document.querySelector('dialog[open]')) return
      const meta = event.metaKey || event.ctrlKey
      if (meta && event.key === ',') {
        event.preventDefault()
        store.openSettings()
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
      if (!(active instanceof HTMLElement) || !active.closest('.approval')) return
      if (event.key === '1') { event.preventDefault(); store.approve(snap.approval.id, 'allow_once') }
      else if (event.key === '2') { event.preventDefault(); store.approve(snap.approval.id, 'allow_session') }
      else if (event.key === '3') { event.preventDefault(); store.approve(snap.approval.id, 'deny') }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // Every overlay flag the handler branches on must be a dep — a stale
    // closure here swallowed Escape after the task modal closed (the old
    // snap still claimed taskModalOpen, so settings could never dismiss).
  }, [snap.approval, snap.paletteOpen, snap.searchOpen, snap.taskModalOpen, snap.settingsOpen, snap.pickerOpen, snap.reasoningPickerOpen, snap.modelMenuOpen, snap.delegationMenuOpen, snap.contextMenuOpen, snap.wsMenuOpen, snap.sessionMenu, snap.turnActive, closeSurface])
  return null
}
