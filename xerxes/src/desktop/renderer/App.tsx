// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createPortal } from 'react-dom'
import { Fragment, useLayoutEffect, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactElement } from 'react'

import { CommandPalette, ModelMenu, ModelPicker, ReasoningPicker, SettingsModal, bareModelName } from './Overlays.js'
import { SessionSearch } from './SearchPanel.js'
import { store, type Snapshot, isPlanReview } from './store.js'
import type { AgentMember } from './types.js'
import { applyCompletion, wantsHints, type HintItem } from './hints.js'
import { groupByWorkspace } from './workspaceGroups.js'
import { ChangesTab, LogTab, PlanTab } from './Workspaces.js'
import { Markdown } from './markdown.js'
import { Dictation } from './Dictation.js'
import { draftKey, readDraft, writeDraft } from './drafts.js'
import { PanelDivider, usePanelLayout } from './layout.js'
import { groupActivity } from "./activityGroups.js"
import { ExecutionDetails } from "./Execution.js"
import { Icon } from './Icon.js'
import { FirstRunSetup } from './Setup.js'
import { BackgroundIndicator, DesktopNavigation, DesktopSheet, DesktopPage, DesktopRail, useDesktopNavigation, type DesktopPanel } from './DesktopPanels.js'

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
function turnDurOf(seconds: number): string {
  return seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`
}

function ttftOf(milliseconds: number): string {
  return milliseconds < 1_000 ? `${Math.round(milliseconds)}ms` : `${(milliseconds / 1_000).toFixed(1)}s`
}

function metricDurationOf(milliseconds: number): string {
  const seconds = Math.max(0, Math.round(milliseconds / 1_000))
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return `${minutes}m${remainder ? `${remainder}s` : ''}`
}

function compactTokensOf(tokens: number): string {
  if (tokens < 1_000) return String(Math.round(tokens))
  if (tokens < 1_000_000) return `${(tokens / 1_000).toFixed(tokens < 10_000 ? 1 : 0)}K`
  return `${(tokens / 1_000_000).toFixed(1)}M`
}

/** Parse the daemon's rendered goal text into card fields. */
export function parseGoal(
  text: string,
): { objective: string; phase: string; rounds: string; activation: string } | null {
  if (!text || text.startsWith('No goal')) return null
  const pick = (prefix: string): string => {
    for (const line of text.split('\n')) {
      const trimmed = line.trim()
      if (trimmed.startsWith(prefix)) return trimmed.slice(prefix.length).trim()
    }
    return ''
  }
  const objective = pick('Objective:')
  if (!objective) return null
  return {
    objective,
    phase: pick('Status:'),
    rounds: pick('Rounds:'),
    activation: pick('Activation:'),
  }
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
  const [panel, setPanel] = useState<DesktopPanel>(null)
  const { layout, setLayout } = usePanelLayout()
  const focused = layout.sidebarHidden
  const [windowWidth, setWindowWidth] = useState(() => typeof window === 'undefined' ? 1440 : window.innerWidth)
  useEffect(() => { const resize = () => setWindowWidth(window.innerWidth); window.addEventListener('resize', resize); return () => window.removeEventListener('resize', resize) }, [])
  const [rail, setRail] = useState<'files' | 'review' | 'activity' | null>(null)
  const [filesExpanded, setFilesExpanded] = useState(false)
  const [reviewPath, setReviewPath] = useState('')
  const contextRequiresFullWidth = windowWidth < (focused ? 0 : layout.sidebarWidth) + layout.inspectorWidth + 320
  const [page, setPage] = useState<'agents' | 'extensions' | 'artifacts' | null>(null)
  const navigate = (next: DesktopPanel, filePath?: string): void => {
    if (next === 'review') setReviewPath(filePath ?? '')
    if (next === 'files' || next === 'review' || next === 'activity') setRail(next)
    else if (next === 'agents' || next === 'extensions' || next === 'artifacts') { setPage(next); setPanel(null) }
    else if (next === null) { setPage(null); setPanel(null) }
    else setPanel(next)
  }
  useEffect(() => { setPanel(null); setPage(null) }, [snap.cwd, snap.sessionKey])
  return (
    <DesktopNavigation.Provider value={navigate}>
    <div className={`app atelier${focused ? ' atelier--focus' : ''}`} style={{ '--sidebar-width': `${layout.sidebarWidth}px`, '--inspector-width': `${layout.inspectorWidth}px` } as React.CSSProperties}>
      <Topbar snap={snap} inspectorOpen={rail !== null} sidebarVisible={!focused} onFocus={() => setLayout({ sidebarHidden: !focused })} />
      <FirstRunSetup snap={snap} />
      <div className="app__body" data-context-full={rail && rail !== 'review' && (rail === 'files' && filesExpanded || contextRequiresFullWidth) || undefined} data-review={rail === "review" || undefined}>
        <Sidebar snap={snap} page={page} />
        {!focused && <PanelDivider label="Resize sessions" value={layout.sidebarWidth} min={180} max={360} onChange={sidebarWidth => setLayout({ sidebarWidth })} />}
        {snap.noWorkspace ? <WorkspaceGate /> : <Chat snap={snap} page={page} />}
        {rail && rail !== 'review' && <PanelDivider label="Resize inspector" value={layout.inspectorWidth} min={260} max={520} reverse onChange={inspectorWidth => setLayout({ inspectorWidth })} />}
        {rail && <DesktopRail panel={rail} snap={snap} close={() => setRail(null)} filesExpanded={filesExpanded} reviewPath={reviewPath} {...(!contextRequiresFullWidth ? { toggleFilesExpanded: () => setFilesExpanded(value => !value) } : {})} activityDetails={<><ActivityDetails snap={snap} /><SessionDiagnostics snap={snap} /></>} />}

      </div>
      <SettingsModal snap={snap} />
      {snap.taskModalOpen && <TaskModal snap={snap} />}
      {/* Mounted only while open: the palette's hooks (needle, cursor,
          focus effect) must never share a fiber with a closed render. */}
      {snap.paletteOpen && <CommandPalette snap={snap} />}
      {snap.searchOpen && <SessionSearch snap={snap} />}
      {snap.wsMenuOpen && <WorkspaceMenu snap={snap} />}
      {snap.sessionMenu && <SessionMenu menu={snap.sessionMenu} />}
      {panel && <DesktopSheet panel={panel} snap={snap} close={() => setPanel(null)} activityDetails={<><ActivityDetails snap={snap} /><SessionDiagnostics snap={snap} /></>} />}
      {(snap.workspaceBusy || snap.workspaceError) && <div className="workspace-notice" role={snap.workspaceError ? 'alert' : 'status'}>
        <span>{snap.workspaceError || 'Opening workspace…'}</span>
        {snap.workspaceError && <button aria-label="Dismiss workspace error" onClick={() => store.clearWorkspaceError()}>×</button>}
      </div>}
      <GlobalKeys snap={snap} />
    </div>
    </DesktopNavigation.Provider>
  )
}

// ── Top bar ─────────────────────────────────────────────────────────────

function Topbar({ snap, inspectorOpen, sidebarVisible, onFocus }: { snap: Snapshot; inspectorOpen: boolean; sidebarVisible: boolean; onFocus: () => void }): ReactElement {
  const open = useDesktopNavigation()
  return <div className="top">
    <button className="studio-icon" title="Toggle sidebar" aria-label="Toggle sidebar" aria-expanded={sidebarVisible} aria-controls="session-sidebar" onClick={onFocus}><Icon name="sidebar" /></button>
    <span className="top__name">{snap.currentTitle || 'New session'}</span>
    {!sidebarVisible && <RuntimeStatus snap={snap} compact />}
    <div className="top__actions" role="group" aria-label="Workspace tools">
    <button title="Search sessions" aria-label="Search sessions" onClick={() => store.openSessionSearch()}><Icon name="search" /></button>
    <BackgroundIndicator snap={snap} />
    {!inspectorOpen && <>
    <button title="Working tree changes" onClick={() => open('review')}><Icon name="changes" /><span>Changes</span></button>
    <button title="Project files" aria-label="Project files" onClick={() => open('files')}><Icon name="sidebar" /></button></>}
    </div>
  </div>
}

/** One workspace-level home for connection, version mismatch and recovery. */
function RuntimeStatus({ snap, compact = false }: { snap: Snapshot; compact?: boolean }): ReactElement {
  const navigate = useDesktopNavigation()
  const [expanded, setExpanded] = useState(false)
  const trigger = useRef<HTMLButtonElement>(null)
  const popup = useRef<HTMLDivElement>(null)
  const busy = snap.runtimeUpdate === 'checking' || snap.runtimeUpdate === 'restarting'
  const label = snap.noWorkspace ? 'Choose workspace' : busy ? 'Updating runtime…' : snap.runtimeUpdate === 'waiting' ? 'Update waiting for idle' : snap.connection === 'offline' ? 'Runtime offline' : snap.connection === 'connecting' ? 'Connecting…' : snap.daemonWarning ? 'Runtime update available' : 'Connected'
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
      <header><strong>Workspace runtime</strong><button aria-label="Close runtime status" onClick={() => { setExpanded(false); trigger.current?.focus() }}>×</button></header>
      <p role="status" aria-live="polite">{snap.connection === 'online' ? 'Connected to this workspace' : label}</p>
      {snap.daemonWarning && <><strong>App and runtime versions differ</strong><p>{snap.daemonWarning}</p></>}
      {snap.runtimeUpdateMessage && <p role="status">{snap.runtimeUpdateMessage}</p>}
      {snap.daemonWarning && <><p>The current runtime keeps your tasks running. Review activity before restarting.</p><div className="runtime-popover__actions"><button onClick={() => { setExpanded(false); navigate('activity') }}>Review activity</button><button disabled={snap.turnActive || busy} onClick={() => void store.restartDaemon()}>Restart workspace runtime</button></div></>}
      {!snap.daemonWarning && snap.connection !== 'online' && <button onClick={() => { setExpanded(false); navigate('workspace') }}>Open workspace settings</button>}
    </div>, document.body)}
  </>
}

// ── Workspace switcher menu (mockup 16) ─────────────────────────────────

/**
 * The topbar chip's dropdown: every folder that holds chats, current first
 * and marked, then the folder picker. Rows are entrances — clicking one
 * retargets the shell to that folder's daemon, exactly like the sidebar
 * workspace headers.
 */
function WorkspaceMenu({ snap }: { snap: Snapshot }): ReactElement {
  const home = workspaceLabel(snap.cwd)
  const groups = groupByWorkspace(snap.sessions, snap.cwd)
  const statusFor = (name: string): string => (name === home ? '● current' : `${groups.find(g => g.name === name)?.rows.length ?? 0} task${(groups.find(g => g.name === name)?.rows.length ?? 0) === 1 ? '' : 's'}`)
  return (
    <>
      <div className="backdrop backdrop--clear" onClick={() => store.closeWorkspaceMenu()} />
      <div className="wsmenu" role="menu" aria-label="Workspaces">
        <div className="cap" style={{ padding: '6px 10px 2px' }}>Workspaces</div>
        {groups.map(group => (
          <button
            key={group.cwd || group.name}
            className={`wsrow${group.name === home ? ' is-cur' : ''}`}
            title={group.name === home ? `${group.cwd} — you are here` : `Switch to ${group.cwd}`}
            onClick={() => { if (group.name !== home) store.enterWorkspace(group.cwd) }}
          >
            <span className={`dot ${group.name === home ? 'dot--live' : 'dot--idle'}`} />
            <span className="wsrow__main">
              <span className="wsrow__t">{group.name}</span>
              <span className="wsrow__s">{group.cwd || group.name} · {statusFor(group.name)}</span>
            </span>
            {group.name === home && <span className="kbd">✓</span>}
          </button>
        ))}
        {groups.length === 0 && (
          <div className="wsrow is-cur">
            <span className="dot dot--live" />
            <span className="wsrow__main">
              <span className="wsrow__t">{home || 'No workspace'}</span>
              <span className="wsrow__s">{snap.cwd || 'choose a folder to begin'}</span>
            </span>
            <span className="kbd">✓</span>
          </div>
        )}
        <div className="menu__sep" />
        <button className="menu__item" onClick={() => { store.closeWorkspaceMenu(); store.chooseWorkspace() }}>
          <span className="ico">＋</span> Add workspace…
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
  const [objective, setObjective] = useState('')
  const [planFirst, setPlanFirst] = useState(true)
  const [namingWorktree, setNamingWorktree] = useState(false)
  const [worktreeName, setWorktreeName] = useState('')
  const [agentPreset, setAgentPreset] = useState('')
  const [model, setModel] = useState(snap.model)
  const [starting, setStarting] = useState(false)
  const models = [...new Set([snap.model, ...snap.models.map(choice => choice.id)].filter(Boolean))]
  const presets = snap.agentPresets ?? []
  const selectedPreset = agentPreset || presets.find(row => row.isDefault && !row.broken)?.id || 'default'
  const start = (): void => {
    if (starting || !objective.trim() || !model) return
    setStarting(true)
    void store.startTask(objective, planFirst, selectedPreset, model).finally(() => setStarting(false))
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
              <button onClick={() => store.chooseWorkspace()} title="Open a different folder as the workspace">＋ different folder…</button>
              <button
                onClick={() => setNamingWorktree(value => !value)}
                title="Create an isolated git worktree next to the repo and switch into it"
              >
                ＋ new worktree…
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
              className="composer__input taskmodal__objective"
              rows={4}
              value={objective}
              autoFocus
              spellCheck={false}
              placeholder="What would you like to get done?"
              onChange={e => setObjective(e.target.value)}
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
              <div className="row__s">agent proposes a checklist; you approve before anything runs</div>
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

          <div className="fieldnote" style={{ marginBottom: 16 }}>
            approvals in this workspace: {snap.permissionMode || 'daemon policy'} —{' '}
            <u style={{ cursor: 'pointer' }} onClick={() => { store.closeTaskModal(); store.openSettings('permissions') }}>change</u>
          </div>

          <div role="status" className="taskmodal__error">{snap.error}</div>
          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <button className="btn btn--ghost" disabled={starting} onClick={() => store.closeTaskModal()}>Cancel</button>
            <button className="btn" disabled={starting || !objective.trim() || !model} onClick={start}>{starting ? 'Starting…' : 'Start task ↵'}</button>
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
              <span className="ico">↵</span> Open <span className="kbd">⏎</span>
            </button>
            <button role="menuitem" className="menu__item" onClick={() => setRenaming(true)}>
              <span className="ico">✎</span> Rename…
            </button>
            <div className="menu__sep" />
            <button role="menuitem" className="menu__item" onClick={() => { copyText(menu.id); store.closeSessionMenu() }}>
              <span className="ico">⧉</span> Copy id
            </button>
            <button role="menuitem" className="menu__item" onClick={() => { void store.exportSessionTranscript(menu.key) }}>
              <span className="ico">⬇</span> Export md
            </button>
          </>
        )}
        {menu.error && <p className="session-menu-error" role="alert">{menu.error}</p>}
      </div>
    </>
  )
}

// ── Statusline ──────────────────────────────────────────────────────────

export function SessionDiagnostics({ snap }: { snap: Snapshot }): ReactElement {
  const livePhaseMs = snap.turnActive && snap.metricPhaseStartedAt != null ? Math.max(0, Date.now() - snap.metricPhaseStartedAt) : 0
  const metrics: [string, string][] = [
    ['Turns', String(snap.turnCount)],
    ['Steps', String(snap.llmSteps + snap.toolSteps)],
    ['Model time', metricDurationOf(snap.llmDurationMs + (snap.metricPhase === 'llm' ? livePhaseMs : 0))],
    ['Tool time', metricDurationOf(snap.toolDurationMs + (snap.metricPhase === 'tool' ? livePhaseMs : 0))],
    ['First response', snap.ttftMs == null ? 'Unavailable' : ttftOf(snap.ttftMs)],
    ['Generation', snap.tokensPerSecond == null ? 'Unavailable' : snap.tokensPerSecond.toFixed(1) + ' tokens/s'],
    ['Cache hit', snap.cacheHitRate == null ? 'Unavailable' : Math.round(snap.cacheHitRate * 100) + '%'],
    ['Input tokens', compactTokensOf(snap.inputTokens)],
  ]
  if (snap.costUsd != null && snap.costUsd > 0) metrics.push(['Cost', '$' + snap.costUsd.toFixed(snap.costUsd < 0.01 ? 4 : 2)])
  if (snap.model) metrics.push(['Model', snap.model])
  if (snap.branch) metrics.push(['Branch', snap.branch])
  return <details className="session-diagnostics"><summary>Session statistics</summary>
    <dl className="session-diagnostics__values">{metrics.map(([label, value]) => <Fragment key={label}><dt>{label}</dt><dd>{value}</dd></Fragment>)}</dl>
  </details>
}

// ── Sidebar ─────────────────────────────────────────────────────────────

function Sidebar({ snap, page }: { snap: Snapshot; page: 'agents' | 'extensions' | 'artifacts' | null }): ReactElement {
  const open = useDesktopNavigation()
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
        age: snap.turnActive ? `${snap.turnSeconds}s` : '',
        current: true,
        kind: 'main',
        turns: snap.turnCount,
        messages: 0,
        cwd: snap.cwd,
        untitled: false,
      }
    : null
  const rows = [
    ...snap.live.filter(row => row.id !== snap.currentId),
    ...snap.sessions.filter(row => row.id !== snap.currentId),
    ...(currentRow ? [currentRow] : []),
  ].filter(match)
  const groups = groupByWorkspace(rows, snap.cwd)
  const online = snap.connection === 'online'

  return (
    <aside id="session-sidebar" className="side" aria-label="Sessions">
      <div className="side__pad">
        <div className="studio-segment"><button className={page !== 'agents' ? 'is-selected' : ''} onClick={() => open(null)}>Sessions</button><button className={page === 'agents' ? 'is-selected' : ''} onClick={() => open('agents')}>Agents</button></div>
        <button
          className="newchat"
          disabled={!online || snap.turnActive}
          onClick={() => { open(null); store.newChat() }}
          title={homeLabel(snap) ? `Start a new task in ${homeLabel(snap)} (⌘N)` : 'Start a fresh session (⌘N)'}
        ><Icon name="plus" /><span>New session</span><kbd>⌘ N</kbd></button>
        <button className={`studio-nav${page === "extensions" ? " is-selected" : ""}`} disabled={!online} onClick={() => open('extensions')}><Icon name="tools" /><span>Skills & tools</span></button>
        <button className={`studio-nav${page === "artifacts" ? " is-selected" : ""}`} disabled={!online} onClick={() => open('artifacts')}><Icon name="folder" /><span>Artifacts</span></button>
        <button className="studio-nav" disabled={!online} onClick={() => open('schedules')}><Icon name="clock" /><span>Scheduled jobs</span></button>
        <input
          className="side__search"
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder={online ? 'Search sessions…' : 'Offline'}
          aria-label="Filter sessions"
          spellCheck={false}
        />
        <button
          className="side__find"
          disabled={!online}
          title="Full-text search across every saved session (daemon transcript index)"
          onClick={() => store.openSessionSearch()}
        >Search message history →</button>
      </div>
      <nav className="side__list">
        {groups.map(group => (
          <div key={group.name} className={`wgroup${group.name === homeLabel(snap) ? ' is-home' : ''}`}>
            <button
              className="wgroup__cap"
              title={group.name === homeLabel(snap) ? `${group.cwd} — you are here` : `Switch to ${group.cwd}`}
              onClick={() => { if (group.name !== homeLabel(snap)) store.enterWorkspace(group.cwd) }}
            >
              <span className="wgroup__mark">{group.name === homeLabel(snap) ? '●' : '⌂'}</span>
              {group.name}
              {group.name === homeLabel(snap) && <span className="wgroup__cur">current</span>}
            </button>
            {group.rows.map(row => {
              const snippet = snap.snippets[row.id]
              return snippet === undefined
                ? <SessionCell key={row.id} row={row} locked={snap.turnActive && row.id !== snap.currentId} />
                : <SessionCell key={row.id} row={row} snippet={snippet} locked={snap.turnActive && row.id !== snap.currentId} />
            })}
          </div>
        ))}
        {groups.length === 0 && (
          <div className="side__empty">{snap.noWorkspace ? 'Your sessions will appear here' : online ? 'No tasks yet — your chats live inside the workspace folder' : 'Daemon offline — retrying automatically'}</div>
        )}
        <button className="addws" onClick={() => store.chooseWorkspace()} title="Choose another folder to open as a workspace">
          ＋ Add folder…
        </button>
      </nav>
      <div className="studio-side-bottom"><RuntimeStatus snap={snap} /><button onClick={() => store.openSettings()}><Icon name="settings" /><span>Settings</span></button><button onClick={() => open('workspace')}><Icon name="folder" /><span>{homeLabel(snap) || 'Workspace'}</span></button></div>
    </aside>
  )
}

const homeLabel = (snap: Snapshot): string => workspaceLabel(snap.cwd)

function SessionCell({
  row,
  snippet,
  locked = false,
  displayOnly = false,
}: {
  row: Snapshot['live'][number]
  snippet?: string
  /** A running turn owns the connection's session slot — switching now would silently no-op. */
  locked?: boolean
  /** Status-only rows (fleet snapshots): the daemon refuses resuming them while owned. */
  displayOnly?: boolean
}): ReactElement {
  const title = snippet ?? row.title
  // dsh row grammar: title + right-aligned age on the first line, a status
  // subline only when it says something the age doesn't.
  const sub = [
    row.status === 'working' ? 'acting' : row.status === 'failed' ? 'failed' : '',
  ].filter(Boolean).join(' · ')
  if (displayOnly) {
    return (
      <details className="agent-row"><summary><span className="sess__dot" style={{ background: statusColor(row.status) }} /><span>{title}</span><small>{row.status}</small></summary><p>{title}</p>{row.turns > 0 && <small>{row.turns} turns</small>}</details>
    )
  }
  return (
    <button
      className={`sess${row.current ? ' is-current' : ''}`}
      disabled={locked}
      title={locked ? 'Finish or stop the current task before switching' : undefined}
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
      <span className="sess__dot" style={{ background: statusColor(row.status) }} />
      <span className="sess__body">
        <span className="sess__t">{title}</span>
        {sub && <span className="sess__s">{sub}</span>}
      </span>
      {row.age && <span className="sess__age">{row.age}</span>}
    </button>
  )
}
// ── Chat column ─────────────────────────────────────────────────────────

/**
 * Header fleet chip (dsh "N subagents ⌄"): the count is live from the
 * snapshot; clicking opens the roster — name, status dot, state label.
 * Rows are read-only status; subagents are supervised from the Fleet rail.
 */
function FleetChip({ snap }: { snap: Snapshot }): ReactElement {
  const [open, setOpen] = useState(false)
  const fleet = snap.fleet
  if (fleet.length === 0) return <></>
  return (
    <span className="chipanchor">
      <button
        className={`hchip hchip--btn${open ? ' is-on' : ''}`}
        title="Subagents spawned by this task"
        aria-expanded={open}
        onClick={() => setOpen(value => !value)}
      >
        ⚇ {fleet.length} subagent{fleet.length === 1 ? '' : 's'} <span className="c">▾</span>
      </button>
      {open && (
        <>
          <div className="backdrop backdrop--clear" onClick={() => setOpen(false)} />
          <div className="fleetpop" role="menu" aria-label="Subagents">
            <div className="cap fleetpop__cap">Subagents · {fleet.length}</div>
            {fleet.map(row => (
              <div key={row.id} className="fleetpop__row">
                <span className="sess__dot" style={{ background: statusColor(row.status) }} />
                <span className="fleetpop__t">{row.title}</span>
                <span className="fleetpop__s">{row.status}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </span>
  )
}

/**
 * Header background-jobs chip (dsh "N background jobs ⌄"): daemon-
 * backgrounded turns working right now, from the event pipe + active_list.
 */
function JobsChip({ snap }: { snap: Snapshot }): ReactElement {
  const [open, setOpen] = useState(false)
  const jobs = snap.backgroundJobs
  if (jobs.length === 0) return <></>
  return (
    <span className="chipanchor">
      <button
        className={`hchip hchip--btn${open ? ' is-on' : ''}`}
        title="Daemon-backgrounded turns running now"
        aria-expanded={open}
        onClick={() => setOpen(value => !value)}
      >
        ⧉ {jobs.length} background job{jobs.length === 1 ? '' : 's'} running <span className="c">▾</span>
      </button>
      {open && (
        <>
          <div className="backdrop backdrop--clear" onClick={() => setOpen(false)} />
          <div className="fleetpop" role="menu" aria-label="Background jobs">
            <div className="cap fleetpop__cap">Background jobs · {jobs.length}</div>
            {jobs.map(job => (
              <div key={job.id} className="fleetpop__row">
                <span className="sess__dot" style={{ background: statusColor(job.status) }} />
                <span className="fleetpop__t">{job.title}</span>
                <span className="fleetpop__s">{job.status}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </span>
  )
}

function Chat({ snap, page }: { snap: Snapshot; page: 'agents' | 'extensions' | 'artifacts' | null }): ReactElement {
  const open = useDesktopNavigation()
  const changes = snap.changes
  const totals = useMemo(() => ({
    adds: changes.reduce((sum, file) => sum + file.adds, 0),
    dels: changes.reduce((sum, file) => sum + file.dels, 0),
  }), [changes])
  const planDone = snap.plan?.items.filter(item => item.done).length ?? 0
  const planTotal = snap.plan?.items.length ?? 0
  const needsInput = snap.approval !== null || snap.question !== null

  return (
    <main className="chat">
      <header className="chat__head">
        <button className="chat__workspace" onClick={() => open('workspace')}>{homeLabel(snap)}</button>
        <span className="chat__title">{snap.currentTitle || (snap.connection === 'online' ? 'New task' : 'Not connected')}</span>
        {snap.currentId && <span className="chat__id">{snap.currentId.slice(0, 8)}</span>}
        <span className="hchip" title="Agent preset for this session · fixed after its first turn">◈ {snap.currentAgentPreset === 'creator' ? 'Creator mode' : (snap.currentAgentPreset || 'default')}</span>
        {/* Fleet + mode ride the header as chips (dsh grammar); the fleet
            chip opens the live subagent list, the mode chip toggles plan. */}
        <FleetChip snap={snap} />
        <JobsChip snap={snap} />
        <button
          className={`hchip hchip--btn${snap.planMode ? ' is-plan' : ''}`}
          title={snap.planMode ? 'Plan mode on — click to act' : 'Standard mode — click to plan first'}
          onClick={() => store.togglePlanMode()}
        >
          {snap.planMode ? '⏸ Plan mode' : 'Standard mode'}
        </button>
        <div className="chat__state">
          {/* Needs-input outranks acting: an approval can land mid-turn, and
              on any other tab the stream (and its card) is not mounted —
              the header is the only place the signal survives. */}
          {needsInput ? (
            <>
              <span className="badge badge--need">needs input{snap.turnActive ? ' · acting paused' : ''}</span>
              {snap.turnActive && <button className="stop" onClick={() => store.cancel()}>Stop</button>}
            </>
          ) : snap.turnActive ? (
            <>
              <span className="badge badge--live">Working</span>
              <button className="stop" onClick={() => store.cancel()}>Stop</button>
            </>
          ) : snap.planMode ? (
            <span className="badge badge--plan">⏸ plan mode</span>
          ) : snap.failed ? (
            <span className="badge badge--fail">failed</span>
          ) : (
            <span className="badge">idle</span>
          )}
        </div>
        <button
          className="hchip hchip--btn chat__log"
          disabled={!snap.currentId}
          title="Export this session's transcript as markdown"
          onClick={() => void store.exportSessionTranscript(snap.sessionKey)}
        >
          ⤓ Session log
        </button>
      <div hidden={page !== null} className={`workspace__tabs${snap.turnCount === 0 && !snap.blocks.some(block => block.kind !== "notice") ? ' workspace__tabs--empty' : ''}`}>
        <div className="tabs">
          <button className={`tab${snap.tab === 'activity' ? ' is-on' : ''}`} onClick={() => store.setTab('activity')}>Conversation</button>
          <button className={`tab${snap.tab === 'plan' ? ' is-on' : ''}`} onClick={() => store.setTab('plan')}>
            Plan{planTotal ? <> <span className="pillcount">{planDone}/{planTotal}</span></> : null}
          </button>
          <button className={`tab${snap.tab === 'log' ? ' is-on' : ''}`} onClick={() => store.setTab('log')}>Log</button>
        </div>
      </div>
      </header>

      {page && <DesktopPage panel={page} snap={snap} />}
      <div className="conversation-surface" hidden={page !== null}>
      {snap.connection !== 'online' && snap.blocks.length > 0 && <div className="connection-status" role="status" aria-live="polite"><strong>{snap.connection === 'connecting' ? 'Reconnecting…' : 'Connection lost. Retrying…'}</strong>{snap.error && <span>{snap.error}</span>}<button onClick={() => store.retryConnection()}>Retry now</button></div>}
      {snap.tab === 'activity' && <Stream snap={snap} />}
      {snap.tab === 'changes' && <div className="workspace"><ChangesTab snap={snap} /></div>}
      {snap.tab === 'plan' && <div className="workspace"><PlanTab snap={snap} /></div>}
      {snap.tab === 'log' && <div className="workspace workspace--log"><LogTab snap={snap} /></div>}

      </div>
      <Composer snap={snap} />
    </main>
  )
}

// ── Activity stream ─────────────────────────────────────────────────────

function Stream({ snap }: { snap: Snapshot }): ReactElement {
  const ref = useRef<HTMLDivElement>(null)
  // Stick to the newest content unless the human scrolled up to read.
  useEffect(() => {
    const el = ref.current
    if (el && el.scrollHeight - el.scrollTop - el.clientHeight < 180) el.scrollTop = el.scrollHeight
  })
  // 'Stream thinking' off hides reasoning trails from the feed — the daemon
  // still streams them; this is a display choice, not a policy change.
  const blocks = snap.streamThinking ? snap.blocks : snap.blocks.filter(b => b.kind !== 'thinking')
  const offline = snap.connection === 'offline'
  const empty = snap.connection === 'online' && blocks.every(block => block.kind === 'notice' && !block.error) && !snap.turnActive

  // dsh: the approval attaches to the tool call it is about — render it
  // directly under the trail holding that call, not floating elsewhere.
  const approval = snap.approval
  let approvalIndex = -1
  if (approval?.toolCallId) {
    approvalIndex = blocks.findIndex(
      b => b.kind === 'tools' && b.items.some(i => i.id === approval.toolCallId),
    )
  }
  const inlineApproval = approval !== null && approvalIndex !== -1
  const floatApproval = approval !== null && !inlineApproval
  const failedCard = snap.failed && !snap.turnActive ? <FailedCard failed={snap.failed} /> : null

  if (offline && blocks.length === 0 && !snap.question) {
    return <div className="stream" ref={ref}><Offline cwd={snap.cwd} error={snap.error} /></div>
  }

  // The failed card and pending question own the tail of the feed even when
  // the transcript itself is still empty — an empty welcome must not bury
  // the thing asking for a decision.
  return (
    <div className={`stream${empty && !failedCard && !snap.question ? ' stream--welcome' : ''}`} ref={ref}>
      {empty && !failedCard && !snap.question ? <>
        {blocks.length > 0 && <div className="welcome-notices" aria-live="polite">{blocks.map(block => <BlockView key={block.id} block={block} />)}</div>}
        <Welcome />
      </> : (
        <div className="stream__col">
          {groupActivity(blocks, approval?.toolCallId).map(group => (
            <Fragment key={group[0]!.id}>
              {group.length > 1 ? <ActivityGroup blocks={group} /> : <BlockView block={group[0]!} />}
              {inlineApproval && group.some(block => block === blocks[approvalIndex]) && <ApprovalCard approval={approval} inline />}
            </Fragment>
          ))}
          {failedCard}
          {/* The plan card lives in the feed where the work happens — the
              rail stays for goal + fleet supervision. */}
          {snap.plan && <TodosCard plan={snap.plan} turnActive={snap.turnActive} />}
          {/* Live tail of the feed: the turn clock the header badge shows,
              repeated where the eye actually is while scrolling. */}
          {snap.turnActive && <div className="streamstatus" role="status" aria-live="polite">{snap.networkRetrying ? 'Retrying connection…' : 'Acting…'} {turnDurOf(snap.turnSeconds)}</div>}
        </div>
      )}
      {floatApproval && <ApprovalCard approval={approval} />}
      {snap.question && (
        <div className="stream__col">
          <QuestionCard question={snap.question} plan={snap.plan} />
        </div>
      )}
    </div>
  )
}

/**
 * The plan as a live to-dos card inside the feed (dsh grammar): header
 * counts by state, a spinner on the item the turn is chewing, green checks
 * behind it, dashed circles ahead. Collapse is the user's — local state.
 */
function TodosCard({ plan, turnActive }: { plan: NonNullable<Snapshot['plan']>; turnActive: boolean }): ReactElement | null {
  const [open, setOpen] = useState(true)
  const items = plan.items
  if (items.length === 0) return null
  const done = items.filter(item => item.done).length
  const current = items.findIndex(item => !item.done)
  const inProgress = turnActive && current !== -1 ? 1 : 0
  const counts = [
    `${done} completed`,
    inProgress ? '1 in progress' : '',
    `${items.length - done - inProgress} pending`,
  ].filter(Boolean).join(' · ')
  return (
    <div className="todos">
      <button className="todos__head" onClick={() => setOpen(value => !value)} aria-expanded={open}>
        <span className="todos__ico">☷</span>
        <span className="todos__title">To-dos</span>
        <span className="todos__counts">{counts}</span>
        <span className={`todos__chev${open ? ' is-open' : ''}`}>▾</span>
      </button>
      {open && (
        <div className="todos__list">
          {items.map((item, index) => {
            const state = item.done ? 'done' : index === current && turnActive ? 'cur' : 'todo'
            return (
              <div key={index} className={`todo todo--${state}`}>
                <span className="todo__icon" data-state={state}>{state === 'done' ? '✓' : ''}</span>
                <span className="todo__t">{item.text}</span>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}

/**
 * The turn's spawned subagents as an in-chat card (dsh batch grammar):
 * header counts by state, one row per member with a status dot and label.
 * Event-driven from the spawn call itself, so a batch that dies inside the
 * daemon still shows — then the daemon snapshots land terminal states.
 */
function AgentsCard({ members }: { members: readonly AgentMember[] }): ReactElement {
  const [open, setOpen] = useState(true)
  const working = members.filter(m => m.status === 'working').length
  const failed = members.filter(m => m.status === 'failed' || m.status === 'cancelled').length
  const counts = [
    `${members.length} agent${members.length === 1 ? '' : 's'}`,
    working ? `${working} working` : '',
    failed ? `${failed} failed` : '',
    !working && !failed ? 'completed' : '',
  ].filter(Boolean).join(' · ')
  return (
    <div className="acard">
      <button className="acard__head" onClick={() => setOpen(value => !value)} aria-expanded={open}>
        <span className="acard__ico">⚇</span>
        <span className="acard__title">Subagents</span>
        <span className="acard__counts">{counts}</span>
        {working > 0 && <span className="acard__spin" />}
        <span className={`acard__chev${open ? ' is-open' : ''}`}>▾</span>
      </button>
      {open && (
        <div className="acard__list">
          {members.map(member => (
            <div key={member.key} className="acard__row">
              <span className="sess__dot" style={{ background: statusColor(member.status) }} />
              <span className="acard__t">{member.title}</span>
              <span className="acard__s" data-state={member.status}>{member.status}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Welcome(): ReactElement {
  return <div className="welcome">
    <h1 className="welcome__wordmark">XERXES</h1>
    <p>A place to think, build, and finish.</p>
    <div className="welcome__ideas">{([
      ['Understand this project', 'Explain the structure of this project and where to start.'],
      ['Review recent changes', 'Review recent changes for bugs and explain the findings.'],
      ['Build something', 'Help me implement '],
    ] as const).map(([label, prompt]) => <button className="idea" key={label} onClick={() => window.dispatchEvent(new CustomEvent('xerxes:add-context', { detail: prompt }))}><span>{label}</span><Icon name="arrow" size={14} /></button>)}</div>
  </div>
}

/**
 * Workspace gate — a fresh shell has no folder and therefore no daemon.
 * The composer is unavailable by design until a folder is chosen; the
 * pick feeds useProject, which spawns that project's daemon and reboots
 * the shell into it.
 */
function WorkspaceGate(): ReactElement {
  return (
    <main className="chat">
      <div className="wsgate">
        <div className="wsgate__mark">⌂</div>
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
  return (
    <div className="offline">
      <div className="offline__dot" />
      <h1>Daemon offline</h1>
      {error && <p className="connection-error" role="alert">{error}</p>}
      <p>
        Every task runs against a per-project daemon — the terminal, TUI and this app share the same sessions through it. The app launches one automatically and reconnects on its own.
      </p>
      <button className="btn" onClick={() => store.retryConnection()}>↻ Retry now</button>
      <div className="cmd">bun xerxes daemon --project-dir {cwd || '<this project>'}</div>
    </div>
  )
}

function ActivityGroup({ blocks }: { blocks: Snapshot['blocks'] }): ReactElement {
  const tools = blocks.flatMap(block => block.kind === 'tools' ? block.items : [])
  const failures = tools.filter(item => item.state === 'failed').length
  const running = tools.some(item => item.state === 'working') || blocks.some(block => block.kind === 'thinking' && block.streaming)
  const [expanded, setExpanded] = useState(failures > 0)
  useEffect(() => { if (failures) setExpanded(true) }, [failures])
  return <details className="activity-group" open={expanded} onToggle={event => setExpanded(event.currentTarget.open)}>
    <summary><Icon name="chevron" size={14} /><span>{running ? 'Working' : 'Work details'}{tools.length ? ' · ' + tools.length + ' operation' + (tools.length === 1 ? '' : 's') : ''}</span>{failures > 0 && <strong>{failures} failed</strong>}</summary>
    <div className="activity-group__body">{blocks.map(block => <BlockView key={block.id} block={block} />)}</div>
  </details>
}

function UserMessage({ text }: { text: string }): ReactElement {
  const [expanded, setExpanded] = useState(false)
  const long = text.length > 1600 || text.split('\n').length > 16
  return <div className="msg msg--user">
    <div className={`msg__text${long && !expanded ? ' msg__text--preview' : ''}`}>{text}</div>
    {long && <button className="message-expand" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>{expanded ? 'Show less' : 'Show full message'}</button>}
  </div>
}

function BlockView({ block }: { block: Snapshot['blocks'][number] }): ReactElement {
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
          <details key={item.id} className="toolrow">
            <summary className="frow frow--tool">
              <span className="frow__icon" data-state={item.state}><Icon name={item.verb.includes("exec") ? "terminal" : "tools"} size={14} /></span>
              <span className="frow__label">{toolLabelOf(item.verb)}</span>
              {item.arg ? (
                <>
                  <span className="frow__sep">·</span>
                  <span className="frow__excerpt">{item.arg}</span>
                </>
              ) : null}
              {item.diff && (
                <span className="frow__diff"><span className="add">+{item.diff.adds}</span> <span className="del">−{item.diff.dels}</span></span>
              )}
              <span className="frow__dur">{item.state === 'working' ? 'Running' : item.state === 'failed' ? 'Failed' : item.dur && item.dur !== '0.0s' ? item.dur : ''}</span>
            </summary>
            <ExecutionDetails item={item} />
          </details>
        ))}
      </div>
    )
  }
  if (block.kind === 'checkpoint') {
    return (
      <div className="frow frow--sys">
        <span className="frow__icon">⏱</span>
        <span className="frow__label">Checkpoint</span>
        <span className="frow__sep">·</span>
        <span className="frow__excerpt">turn {block.turn} end · <b>+{block.adds} −{block.dels}</b> cumulative</span>
      </div>
    )
  }
  return (
    <div className={`frow frow--sys${block.error ? ' frow--err' : ''}`}>
      <span className="frow__icon">▤</span>
      <span className="frow__excerpt frow__excerpt--wrap">{block.text}</span>
    </div>
  )
}

function ApprovalCard({ approval, inline }: { approval: NonNullable<Snapshot['approval']>; inline?: boolean }): ReactElement {
  return (
    <div className={`approval${inline ? ' approval--inline' : ''}`} role="alertdialog" aria-label="Tool approval">
      <div className="approval__head">
        <span className="approval__dot">●</span>
        <span className="approval__title">{approval.toolName || approval.action || 'tool'} — approval required</span>
        <span className="approval__keys">1 / 2 / 3</span>
      </div>
      {approval.description && <pre className="approval__desc">{approval.description}</pre>}
      <div className="approval__row">
        <button className="btn btn--solid" onClick={() => store.approve(approval.id, 'allow_once')}>Allow once <kbd>1</kbd></button>
        <button className="btn" onClick={() => store.approve(approval.id, 'allow_session')}>This session <kbd>2</kbd></button>
        <button className="btn btn--danger" onClick={() => store.approve(approval.id, 'deny')}>Deny <kbd>3</kbd></button>
        <span className="appr-policy" style={{ marginLeft: 'auto', alignSelf: 'center' }}>session policy: ask</span>
      </div>
    </div>
  )
}

/** Failed turn: the error, a retry from the last instruction, or resolve. */
function FailedCard({ failed }: { failed: NonNullable<Snapshot['failed']> }): ReactElement {
  // Navigation hint only: transport/security code remains responsible for retry policy.
  const providerSettingsRelevant = /authenticat|credential|api[ _-]?key|certificate|self.signed|provider.*config/i.test(failed.error)
  return (
    <div className="terr" role="alert">
      <div className="terr__head"><span>✕</span> Turn {failed.turn} failed</div>
      <div className="terr__body">{failed.error}</div>
      <div className="terr__row">
        {providerSettingsRelevant && <button className="btn" onClick={() => store.openSettings('models')}>Provider settings</button>}
        <button className="btn" disabled={!failed.lastUser} onClick={() => store.retryFailed()}>
          ↺ Retry{failed.lastUser ? ' — resubmit the instruction' : ''}
        </button>
        <button className="btn btn--ghost" onClick={() => store.resolveFailure()}>Mark resolved</button>
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
  const submit = (): void => {
    const answers: Record<string, string> = {}
    for (const item of question.items) {
      const custom = others[item.id]?.trim()
      const picked = selections[item.id] ?? []
      if (custom) answers[item.id] = custom
      else if (picked.length) answers[item.id] = picked.join(', ')
      else if (!item.allowFreeform && item.options.length) answers[item.id] = item.options[0] ?? ''
    }
    store.answerQuestion(question.requestId, answers)
  }
  // Number keys choose options while the card is up (not while typing); the
  // hints are only honest for the FIRST question — later ones have no keys.
  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
      const number = Number.parseInt(event.key, 10)
      if (!Number.isFinite(number) || number < 1) return
      const item = question.items[0]
      if (!item) return
      const option = item.options[number - 1]
      if (!option) return
      event.preventDefault()
      setSelections(prev => ({ ...prev, [item.id]: [option] }))
    }
    const onEnter = (event: KeyboardEvent): void => {
      if (event.key !== 'Enter') return
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
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
  }, [question, selections, others])

  if (review) return <PlanReviewCard question={question} plan={plan} />
  return (
    <div className="qcard" role="form" aria-label="Agent questions">
      <div className="qcard__head">
        <span className="qcard__header-tag">questions</span>
        <span className="appr-policy">{question.items.length} · answers return to the agent</span>
      </div>
      {question.items.map((item, qi) => {
        const picked = selections[item.id] ?? []
        return (
          <div key={item.id} style={{ display: 'grid', gap: 6 }}>
            <div className="qcard__q">{qi + 1} · {item.question}</div>
            {item.options.length > 0 && (
              <div className="optlist">
                {item.options.map((option, oi) => {
                  const on = picked.includes(option)
                  return (
                    <button
                      key={option}
                      className={`opt${on ? ' is-approve' : ''}`}
                      onClick={() => {
                        setSelections(prev => {
                          const current = prev[item.id] ?? []
                          return {
                            ...prev,
                            [item.id]: on ? current.filter(x => x !== option) : [...current, option],
                          }
                        })
                      }}
                    >
                      <span className="opt__label">{option}</span>
                      {/* Keys drive the first question only — no kbd badge
                          on later questions would advertise a dead key. */}
                      <span className="opt__kbd">{on ? '✓' : qi === 0 && oi < 9 ? <kbd>{oi + 1}</kbd> : null}</span>
                    </button>
                  )
                })}
              </div>
            )}
            {item.allowFreeform && (
              <div className="otherbox">
                <span>Other</span>
                <input
                  placeholder={item.placeholder || 'type a custom answer…'}
                  spellCheck={false}
                  value={others[item.id] ?? ''}
                  onChange={e => setOthers(prev => ({ ...prev, [item.id]: e.target.value }))}
                />
              </div>
            )}
          </div>
        )
      })}
      <div className="approval__row">
        <button className="btn btn--solid" onClick={submit}>Submit answers ⏎</button>
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
  const approveOption = item.options.find(option => /approve|accept|start/i.test(option)) ?? item.options[0] ?? ''
  const otherOptions = item.options.filter(option => option !== approveOption)
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
      } else if (event.key === '2' && otherOptions[0]) {
        event.preventDefault()
        setPicked(otherOptions[0])
      } else if (event.key === 'Enter' && readyToSendRef.current) {
        // The Send hint says ⏎; honor it wherever focus sits, not only in
        // the feedback input.
        event.preventDefault()
        answer(feedbackRef.current.trim() || pickedRef.current!)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [question, approveOption, otherOptions])

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
            {index === 0 ? <span className="opt__desc">— tell the agent what to revise</span> : null}
            <span className="opt__kbd">{index === 0 ? <kbd>2</kbd> : null}</span>
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
  const key = draftKey(snap.cwd, snap.sessionKey)
  const [draft, setDraft] = useState(() => readDraft(key))
  const [hints, setHints] = useState<{ items: HintItem[]; index: number } | null>(null)
  const ref = useRef<HTMLTextAreaElement>(null)
  const hintSeq = useRef(0)
  const draftSession = useRef(key)
  useEffect(() => {
    if (draftSession.current === key) {
      writeDraft(key, draft)
      return
    }
    writeDraft(draftSession.current, draft)
    draftSession.current = key
    setDraft(readDraft(key))
    setHints(null)
    ref.current?.focus()
  }, [key, draft])
  useEffect(() => {
    const add = (event: Event) => { const detail = (event as CustomEvent<unknown>).detail; if (typeof detail === 'string') { setDraft(value => value + ' ' + detail); ref.current?.focus() } }
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
  useLayoutEffect(() => {
    grow()
    const observer = new ResizeObserver(grow)
    if (ref.current?.parentElement) observer.observe(ref.current.parentElement)
    window.addEventListener('resize', grow)
    return () => { observer.disconnect(); window.removeEventListener('resize', grow) }
  }, [draft])
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
    if (!draft.trim() || snap.connection !== 'online') return
    open(null)
    void store.submit(draft)
    setDraft('')
    requestAnimationFrame(grow)
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

  const ready = snap.connection === 'online'
  const placeholder =
    snap.connection !== 'online'
      ? 'Connect to a daemon first…'
      : snap.turnActive
        ? 'Steer now — queued until this step settles'
        : snap.planMode
          ? 'Planning — describe the outcome, or /plan <msg> to steer'
          : 'Describe what you need'

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
      {snap.queue.length > 0 && (
        <div className="queue">
          {snap.queue.map((item, index) => (
            <div key={item.id} className="qmsg">
              <span className="q-tag">queued {index + 1}</span> {item.text}
              <button className="q-x" title="Hide from view — the daemon already holds it" onClick={() => store.dropQueued(item.id)}>✕</button>
            </div>
          ))}
        </div>
      )}
      <div className="composer">
        <textarea
          ref={ref}
          className="composer__input"
          aria-label="Message"
          rows={2}
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
        <div className="composer__bar">
          <button className="composer__context" aria-label="Add context" onClick={() => open('files')}><Icon name="folder" size={16} /></button>
          <Dictation sessionKey={snap.sessionKey} onText={text=>{setDraft(value=>(value ? value+" " : "")+text)}} />

          <span className="composer__flex" />
          <div className="chipanchor">
            <button
              className={`cchip${snap.model ? '' : ' is-custom'}`}
              title="Model and reasoning effort — click to change"
              onClick={() => store.toggleModelMenu()}
            >
              <span className="star">✳</span> {snap.model ? bareModelName(snap.model) : 'model'}
              {snap.reasoningEffort && snap.reasoningEffort !== 'off' ? ` ${snap.reasoningEffort}` : ''}
              {' '}<span className="c">▾</span>
            </button>
            {snap.modelMenuOpen && <ModelMenu snap={snap} onClose={() => store.closeModelMenu()} />}
            {snap.pickerOpen && <ModelPicker snap={snap} onClose={() => store.closePicker()} />}
            {snap.reasoningPickerOpen && <ReasoningPicker snap={snap} onClose={() => store.closeReasoningPicker()} />}
          </div>
          <button
            className="composer__send"
            disabled={!ready || !draft.trim()}
            title={snap.turnActive ? 'Queue — runs when this step settles (⏎)' : 'Send (⏎)'}
            onClick={send}
          >↑</button>
        </div>
      </div>
      <div className="composer__hints">          <button
            className="cchip"
            title={`Permissions: ${snap.permissionMode || 'not configured'} — open settings`}
            onClick={() => store.openSettings('permissions')}
          >
            <Icon name="shield" size={14} /> Permissions <span className="c">▾</span>
          </button>
          <button
            className={`cchip${snap.planMode ? ' is-on' : ''}`}
            aria-pressed={snap.planMode}
            title={snap.planMode ? 'Planning enabled. Click to work directly.' : 'Click to plan before making changes.'}
            onClick={() => store.togglePlanMode()}
          >
            <Icon name="plan" size={14} /> {snap.planMode ? 'Plan first' : 'Work directly'}
          </button>

        <button className="cchip composer__workspace" title="Change workspace" onClick={() => open('workspace')}><Icon name="folder" size={14} />{homeLabel(snap)}</button>
        <span><kbd>⏎</kbd> {snap.turnActive ? 'queue' : 'send'}</span>
        <span><kbd>⇧⏎</kbd> newline</span>
        <span><kbd>esc</kbd> stop / clear</span>
        <span><kbd>⌘K</kbd> palette</span>
        {snap.goal && parseGoal(snap.goal) && <span className="composer__goal" title={snap.goal}>◎ goal set</span>}
      </div>
    </div>
  )
}

// ── Right rail ──────────────────────────────────────────────────────────

function GoalObjective({ text }: { text: string }): ReactElement {
  const [expanded, setExpanded] = useState(false)
  return <><p className={`goal-objective${expanded ? " is-expanded" : ""}`}>{text}</p>{text.length > 180 && <button className="goal-expand" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>{expanded ? "Show less" : "Show full goal"}</button>}</>
}

export function ActivityDetails({ snap }: { snap: Snapshot }): ReactElement | null {
  const fleet = snap.fleet
  const goal = parseGoal(snap.goal)
  if (!goal && !fleet.length && !snap.skillSuggestions.length && !snap.creatorTrace.length) return null
  return (
    <aside className="activity-details">
      {goal ? (
        <>
          <div className="rail__cap goal-heading">Goal <span>{goal.phase}{goal.activation === 'armed' ? ' · armed' : ''}</span></div>
          <div className="goalcard">
            <GoalObjective text={goal.objective} />
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
        </>
      ) : null}
      {fleet.length > 0 && <><div className="rail__cap">Agents <span>{fleet.length}</span></div>
      <div className="rail__fleet">{fleet.map(row => <SessionCell key={row.id} row={row} displayOnly />)}</div></>}
      {snap.skillSuggestions.length > 0 && (
        <>
          <div className="rail__cap">Skill suggestions · {snap.skillSuggestions.length}</div>
          <div className="rail__skills">
            {snap.skillSuggestions.slice(-3).reverse().map(suggestion => (
              <div className="skillcard" key={suggestion.skillName}>
                <div className="skillcard__head">
                  <span>{suggestion.skillName}</span>
                  {suggestion.version && <span>v{suggestion.version}</span>}
                </div>
                {suggestion.description && <div className="skillcard__desc">{suggestion.description}</div>}
                <div className="skillcard__meta">
                  {suggestion.toolCount} tool call{suggestion.toolCount === 1 ? '' : 's'}
                  {suggestion.uniqueTools.length ? ` · ${suggestion.uniqueTools.join(', ')}` : ''}
                </div>
              </div>
            ))}
          </div>
        </>
      )}
      {snap.creatorTrace.length > 0 && (
        <>
          <div className="rail__cap">Template forge · legacy</div>
          <div className="rail__creator">
            {snap.creatorTrace.slice(-4).reverse().map((trace, index) => (
              <div className="creatorrow" key={`${trace.at}:${trace.action}:${index}`}>
                <span className="creatorrow__state" data-state={trace.status}>{trace.status === 'ok' ? '✓' : '!'}</span>
                <span className="creatorrow__body">
                  <span>{trace.action} · {trace.name || 'forge'}{trace.version ? `@${trace.version}` : ''}</span>
                  {trace.detail && <span>{trace.detail}</span>}
                </span>
              </div>
            ))}
          </div>
        </>
      )}
    </aside>
  )
}

// ── Global keys ─────────────────────────────────────────────────────────

/** ⌘K palette · ⌘N new task · Esc stop · 1/2/3 approvals. */
function GlobalKeys({ snap }: { snap: Snapshot }): ReactElement | null {
  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      const meta = event.metaKey || event.ctrlKey
      if (meta && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        store.togglePalette()
        return
      }
      if (meta && event.key.toLowerCase() === 'n') {
        event.preventDefault()
        store.newChat()
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
        if (snap.turnActive) {
          event.preventDefault()
          store.cancel()
        }
        return
      }
      // Approval keys work unless a field has the focus.
      if (!snap.approval) return
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA')) return
      if (event.key === '1') store.approve(snap.approval.id, 'allow_once')
      else if (event.key === '2') store.approve(snap.approval.id, 'allow_session')
      else if (event.key === '3') store.approve(snap.approval.id, 'deny')
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
    // Every overlay flag the handler branches on must be a dep — a stale
    // closure here swallowed Escape after the task modal closed (the old
    // snap still claimed taskModalOpen, so settings could never dismiss).
  }, [snap.approval, snap.paletteOpen, snap.searchOpen, snap.taskModalOpen, snap.settingsOpen, snap.pickerOpen, snap.reasoningPickerOpen, snap.modelMenuOpen, snap.contextMenuOpen, snap.wsMenuOpen, snap.sessionMenu, snap.turnActive])
  return null
}
