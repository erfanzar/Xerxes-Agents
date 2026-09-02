// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Fragment, useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactElement } from 'react'

import { CommandPalette, ContextMenu, ModelMenu, ModelPicker, ReasoningPicker, SettingsModal, bareModelName } from './Overlays.js'
import { SessionSearch } from './SearchPanel.js'
import { store, type Snapshot, isPlanReview } from './store.js'
import type { AgentMember } from './types.js'
import { applyCompletion, wantsHints, HINT_LIMIT, type HintItem } from './hints.js'
import { groupByWorkspace } from './workspaceGroups.js'
import { ChangesTab, LogTab, PlanTab } from './Workspaces.js'
import { Markdown } from './markdown.js'

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
  return (
    <div className="app">
      <Topbar snap={snap} />
      <div className="app__body">
        <Sidebar snap={snap} />
        {snap.noWorkspace ? <WorkspaceGate /> : <Chat snap={snap} />}
        {!snap.noWorkspace && <Rail snap={snap} />}
      </div>
      <Statusbar snap={snap} />
      <SettingsModal snap={snap} />
      {snap.taskModalOpen && <TaskModal snap={snap} />}
      {/* Mounted only while open: the palette's hooks (needle, cursor,
          focus effect) must never share a fiber with a closed render. */}
      {snap.paletteOpen && <CommandPalette snap={snap} />}
      {snap.searchOpen && <SessionSearch snap={snap} />}
      {snap.wsMenuOpen && <WorkspaceMenu snap={snap} />}
      {snap.sessionMenu && <SessionMenu menu={snap.sessionMenu} />}
      <GlobalKeys snap={snap} />
    </div>
  )
}

// ── Top bar ─────────────────────────────────────────────────────────────

function Topbar({ snap }: { snap: Snapshot }): ReactElement {
  const online = snap.connection === 'online'
  return (
    <div className="top">
      <img className="top__brand top__logo" src="./logo-128.png" alt="Xerxes" />
      <span className="top__name">Xerxes</span>
      <span className="top__sep">·</span>
      {/* Mockup 01/16: the workspace is a switcher chip, not a static label —
          one click lists every folder the shell knows and the gate to add one. */}
      <button className="wschip" title="Switch workspace" onClick={() => store.toggleWorkspaceMenu()}>
        <span className="wschip__ico">▣</span> {workspaceLabel(snap.cwd) || 'no workspace'} <span className="wschip__c">▾</span>
      </button>
      {snap.daemonWarning && (
        <button className="top__warn" title="Restart the project daemon" onClick={() => store.restartDaemon()}>
          ⚠ {snap.daemonWarning}
        </button>
      )}
      <div className="top__conn">
        <span className="dot" data-state={snap.connection} />
        {online ? 'daemon connected' : snap.connection === 'connecting' ? 'connecting…' : 'daemon offline'}
      </div>
    </div>
  )
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
  const presets = snap.agentPresets ?? []
  const selectedPreset = agentPreset || presets.find(row => row.isDefault && !row.broken)?.id || 'default'
  const start = (): void => { void store.startTask(objective, planFirst, selectedPreset) }
  const submitWorktree = (): void => {
    if (!worktreeName.trim()) { setNamingWorktree(false); return }
    void store.createWorktree(worktreeName)
  }
  return (
    <div className="backdrop">
      <div className="modal taskmodal" role="dialog" aria-label="New task">
        <div className="modal__main">
          <h2 className="modal__title">New task</h2>
          <p className="modal__sub"><kbd>⌘N</kbd> · describe the outcome — the planner turns it into a checklist first.</p>

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
            <label>Agent preset</label>
            <select
              value={selectedPreset}
              onChange={event => setAgentPreset(event.target.value)}
              title="Fixed after this session starts"
            >
              {presets.filter(row => !row.broken).map(row => (
                <option key={row.id} value={row.id}>{row.name}{row.isDefault ? ' · default' : ''}</option>
              ))}
              {presets.length === 0 && <option value="default">default</option>}
            </select>
            <div className="row__s" style={{ marginTop: 4 }}>chooses this session’s tools, prompt, and subagents · fixed after the first turn</div>
          </div>

          <div className="field">
            <label>Objective</label>
            <textarea
              className="composer__input taskmodal__objective"
              rows={4}
              value={objective}
              autoFocus
              spellCheck={false}
              placeholder="e.g. Add rate limiting to the public API — 429s with a Retry-After header, config-driven limits. Verify with the new integration test."
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
            <span className="mchip is-custom" title="Change the model from the composer chip once the task starts">
              <span className="star">✳</span> {snap.model || 'unset'}
            </span>
          </div>

          <div className="fieldnote" style={{ marginBottom: 16 }}>
            approvals in this workspace: {snap.permissionMode || 'daemon policy'} —{' '}
            <u style={{ cursor: 'pointer' }} onClick={() => { store.closeTaskModal(); store.openSettings('permissions') }}>change</u>
          </div>

          <div style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
            <button className="btn btn--ghost" onClick={() => store.closeTaskModal()}>Cancel</button>
            <button className="btn" onClick={start}>Start task ⏎</button>
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
  const [renaming, setRenaming] = useState(false)
  const [draft, setDraft] = useState(menu?.title ?? '')
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
      <div className="menu" role="menu" aria-label="Session actions" style={{ left: menu.x, top: menu.y }}>
        {renaming ? (
          <div className="menu__rename">
            <input
              ref={inputRef}
              value={draft}
              onChange={e => setDraft(e.target.value)}
              onKeyDown={e => {
                if (e.key === 'Enter') submitRename()
                if (e.key === 'Escape') store.closeSessionMenu()
              }}
              aria-label="Session title"
              spellCheck={false}
            />
            <button className="menu__item" onClick={submitRename} title="Rename">⏎</button>
          </div>
        ) : (
          <>
            <button className="menu__item" onClick={() => { store.closeSessionMenu(); void store.openSession(menu.id) }}>
              <span className="ico">↵</span> Open <span className="kbd">⏎</span>
            </button>
            <button className="menu__item" onClick={() => setRenaming(true)}>
              <span className="ico">✎</span> Rename…
            </button>
            <div className="menu__sep" />
            <button className="menu__item" onClick={() => { copyText(menu.id); store.closeSessionMenu() }}>
              <span className="ico">⧉</span> Copy id
            </button>
            <button className="menu__item" onClick={() => { void store.exportSessionTranscript(menu.key) }}>
              <span className="ico">⬇</span> Export md
            </button>
          </>
        )}
      </div>
    </>
  )
}

// ── Statusline ──────────────────────────────────────────────────────────

function Statusbar({ snap }: { snap: Snapshot }): ReactElement {
  const online = snap.connection === 'online'
  const livePhaseMs = snap.turnActive && snap.metricPhaseStartedAt != null
    ? Math.max(0, Date.now() - snap.metricPhaseStartedAt)
    : 0
  const llmDurationMs = snap.llmDurationMs + (snap.metricPhase === 'llm' ? livePhaseMs : 0)
  const toolDurationMs = snap.toolDurationMs + (snap.metricPhase === 'tool' ? livePhaseMs : 0)
  const steps = snap.llmSteps + snap.toolSteps
  return (
    <div className="status">
      <span className="st-mode">{snap.planMode ? '⏸ plan' : snap.turnActive ? '▶ act' : online ? 'idle' : '—'}</span>
      <span>{snap.turnCount} turn{snap.turnCount === 1 ? '' : 's'} · {steps} step{steps === 1 ? '' : 's'}</span>
      <span>LLM {metricDurationOf(llmDurationMs)} · Tool calls {metricDurationOf(toolDurationMs)}</span>
      {snap.ttftMs != null && <span>TTFT avg {ttftOf(snap.ttftMs)}</span>}
      {snap.tokensPerSecond != null && <span>{snap.tokensPerSecond.toFixed(1)} tok/s</span>}
      <span>Cache hit {snap.cacheHitRate == null ? '—' : `${Math.round(snap.cacheHitRate * 100)}%`}</span>
      <span>Input {compactTokensOf(snap.inputTokens)}</span>
      <span className="chipanchor">
        <button
          className="st-ctx"
          title="Context usage — click for the token split"
          onClick={() => store.toggleContextMenu()}
        >
          {snap.contextMax == null
            ? 'ctx unknown'
            : `ctx ${Math.round((snap.contextTokens ?? 0) / 1000)}k/${Math.round(snap.contextMax / 1000)}k`}
        </button>
        {snap.contextMenuOpen && <ContextMenu snap={snap} onClose={() => store.closeContextMenu()} />}
      </span>
      {snap.costUsd != null && snap.costUsd > 0 && (
        <span>${snap.costUsd < 0.01 ? snap.costUsd.toFixed(4) : snap.costUsd.toFixed(2)}</span>
      )}
      {snap.model && <span className="st-model">{snap.model}</span>}
      {snap.branch && <span>⎇ {snap.branch}</span>}
      <span className="st-conn">
        <span className="st-dot" data-state={snap.connection} />
        {online ? 'connected' : snap.connection === 'connecting' ? 'connecting' : 'offline'}
      </span>
    </div>
  )
}

// ── Sidebar ─────────────────────────────────────────────────────────────

function Sidebar({ snap }: { snap: Snapshot }): ReactElement {
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
    <aside className="side">
      <div className="side__pad">
        <button
          className="newchat"
          disabled={!online || snap.turnActive}
          onClick={() => store.openTaskModal()}
          title={homeLabel(snap) ? `Start a new task in ${homeLabel(snap)} (⌘N)` : 'Start a fresh session (⌘N)'}
        >＋ New task</button>
        <button
          className="creatorchat"
          disabled={!online || snap.turnActive}
          onClick={() => store.startCreatorMode()}
          title="Start a fresh DSH-style Creator mode session"
        >◈ Creator mode</button>
        <input
          className="side__search"
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder={online ? 'Filter tasks' : 'Offline'}
          spellCheck={false}
        />
        <button
          className="side__find"
          disabled={!online}
          title="Full-text search across every saved session (daemon transcript index)"
          onClick={() => store.openSessionSearch()}
        >⌕ Search sessions & messages…</button>
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
          <div className="side__empty">{online ? 'No tasks yet — your chats live inside the workspace folder' : 'Daemon offline — retrying automatically'}</div>
        )}
        <button className="addws" onClick={() => store.chooseWorkspace()} title="Choose another folder to open as a workspace">
          ＋ Add folder…
        </button>
      </nav>
      <Foot snap={snap} />
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
    // Untitled rows carry their turn count as identity; the snapshot-derived
    // current row (empty age) shows it as progress. Daemon rows keep their age.
    row.untitled || !row.age ? `${row.turns} turn${row.turns === 1 ? '' : 's'}` : '',
  ].filter(Boolean).join(' · ')
  if (displayOnly) {
    return (
      <div className="sess sess--status" title="subagent snapshot — read-only while it runs">
        <span className="sess__dot" style={{ background: statusColor(row.status) }} />
        <span className="sess__body">
          <span className="sess__t">{title}</span>
          {sub && <span className="sess__s">{sub}</span>}
        </span>
        {row.age && <span className="sess__age">{row.age}</span>}
      </div>
    )
  }
  return (
    <button
      className={`sess${row.current ? ' is-current' : ''}`}
      disabled={locked}
      title={locked ? 'Finish or stop the current task before switching' : undefined}
      onClick={() => store.openSession(row.id)}
      onContextMenu={event => {
        event.preventDefault()
        store.openSessionMenu(
          { id: row.id, key: row.key, title: snippet ?? title },
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
function Foot({ snap }: { snap: Snapshot }): ReactElement {
  return (
    <div className="side__foot">
      <div className="meter" title={snap.contextMax == null ? 'Context window unknown' : 'Context usage'}>
        <div className="meter__bar">
          {snap.contextMax != null && (
            <div className="meter__fill" style={{ width: `${Math.min(100, ((snap.contextTokens ?? 0) / snap.contextMax) * 100)}%` }} />
          )}
        </div>
        <span className="meter__label">
          {snap.contextMax == null
            ? 'ctx unknown'
            : `ctx ${Math.round((snap.contextTokens ?? 0) / 1000)}k / ${Math.round(snap.contextMax / 1000)}k`}
        </span>
      </div>
      <button className="footlink" title="Settings" onClick={() => store.openSettings()}>
        <span className="footlink__ico">⚙</span> Settings
      </button>
    </div>
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

function Chat({ snap }: { snap: Snapshot }): ReactElement {
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
              <span className="badge badge--live">▶ acting · {snap.turnSeconds}s</span>
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
      </header>

      <div className="workspace__tabs">
        <div className="tabs">
          <button className={`tab${snap.tab === 'activity' ? ' is-on' : ''}`} onClick={() => store.setTab('activity')}>Activity</button>
          <button className={`tab${snap.tab === 'changes' ? ' is-on' : ''}`} onClick={() => store.setTab('changes')}>
            Changes
            {totals.adds || totals.dels ? (
              <> <span className="pillcount add">+{totals.adds}</span> <span className="pillcount del">−{totals.dels}</span></>
            ) : null}
          </button>
          <button className={`tab${snap.tab === 'plan' ? ' is-on' : ''}`} onClick={() => store.setTab('plan')}>
            Plan{planTotal ? <> <span className="pillcount">{planDone}/{planTotal}</span></> : null}
          </button>
          <button className={`tab${snap.tab === 'log' ? ' is-on' : ''}`} onClick={() => store.setTab('log')}>Log</button>
          <span className="cost">
            {snap.turnActive ? `${snap.turnSeconds}s` : planTotal ? `${planDone}/${planTotal} steps` : `${snap.turnCount} turn${snap.turnCount === 1 ? '' : 's'}`}
          </span>
        </div>
      </div>

      {snap.tab === 'activity' && <Stream snap={snap} />}
      {snap.tab === 'changes' && <div className="workspace"><ChangesTab snap={snap} /></div>}
      {snap.tab === 'plan' && <div className="workspace"><PlanTab snap={snap} /></div>}
      {snap.tab === 'log' && <div className="workspace workspace--log"><LogTab snap={snap} /></div>}

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
  const empty = snap.connection === 'online' && blocks.length === 0 && !snap.turnActive

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
    return <div className="stream" ref={ref}><Offline cwd={snap.cwd} /></div>
  }

  // The failed card and pending question own the tail of the feed even when
  // the transcript itself is still empty — an empty welcome must not bury
  // the thing asking for a decision.
  return (
    <div className="stream" ref={ref}>
      {empty && !failedCard && !snap.question ? <Welcome /> : (
        <div className="stream__col">
          {blocks.map((block, index) => (
            <Fragment key={block.id}>
              <BlockView block={block} />
              {inlineApproval && index === approvalIndex && <ApprovalCard approval={approval} inline />}
            </Fragment>
          ))}
          {failedCard}
          {/* The plan card lives in the feed where the work happens — the
              rail stays for goal + fleet supervision. */}
          {snap.plan && <TodosCard plan={snap.plan} turnActive={snap.turnActive} />}
          {/* Live tail of the feed: the turn clock the header badge shows,
              repeated where the eye actually is while scrolling. */}
          {snap.turnActive && <div className="streamstatus">Acting… {turnDurOf(snap.turnSeconds)}</div>}
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
  const ideas: Array<[string, string]> = [
    ['Map this repo', 'Map this repository: entry points, hot paths, and anything dead.'],
    ['Explain the streaming loop', 'Explain how the streaming loop normalizes provider deltas into events.'],
    ['Audit TODOs', 'Find every TODO in src/ and group them by subsystem.'],
  ]
  return (
    <div className="welcome">
      <img className="welcome__mark welcome__logo" src="./logo-128.png" alt="Xerxes" />
      <h1>Describe the outcome, not the steps</h1>
      <div className="welcome__ideas">
        {ideas.map(([label, text]) => (
          <button key={label} className="idea" onClick={() => void store.submit(text)}>
            <span className="idea__label">{label}</span>
            <span className="idea__go">↵</span>
          </button>
        ))}
      </div>
    </div>
  )
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
        <h1>Create a workspace</h1>
        <p>
          A workspace is a folder that holds your chats. Pick one — Xerxes runs
          a dedicated daemon inside it, and every task you start stays in that
          folder's context.
        </p>
        <button className="btn btn--solid wsgate__btn" onClick={() => store.chooseWorkspace()}>
          Choose folder…
        </button>
      </div>
    </main>
  )
}

function Offline({ cwd }: { cwd: string }): ReactElement {
  return (
    <div className="offline">
      <div className="offline__dot" />
      <h1>Daemon offline</h1>
      <p>
        Every task runs against a per-project daemon — the terminal, TUI and this app share the same sessions through it. The app launches one automatically and reconnects on its own.
      </p>
      <button className="btn" onClick={() => store.retryConnection()}>↻ Retry now</button>
      <div className="cmd">bun xerxes daemon --project-dir {cwd || '<this project>'}</div>
    </div>
  )
}

function BlockView({ block }: { block: Snapshot['blocks'][number] }): ReactElement {
  if (block.kind === 'agents') {
    return <AgentsCard members={block.members} />
  }
  if (block.kind === 'user') {
    return <div className="msg msg--user"><div className="msg__text">{block.text}</div></div>
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
          <span className="frow__icon">⚛</span>
          <span className="frow__label">Think</span>
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
              <span className="frow__icon" data-state={item.state}>⌘</span>
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
              <span className="frow__dur">{item.dur || (item.state === 'working' ? 'running' : '')}</span>
            </summary>
            <div className="toolrow__body">
              <div className="toolrow__meta"><b>{item.name || toolLabelOf(item.verb)}</b><code>{item.id}</code></div>
              <div className="toolrow__section"><span>Input</span><pre>{item.input || '(no arguments)'}</pre></div>
              {item.error
                ? <div className="toolrow__section toolrow__section--error"><span>Error</span><pre>{item.error}</pre></div>
                : <div className="toolrow__section"><span>Output</span><pre>{item.output || (item.state === 'working' ? '(waiting)' : '(no output)')}</pre></div>}
            </div>
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
  return (
    <div className="terr" role="alert">
      <div className="terr__head"><span>✕</span> Turn {failed.turn} failed</div>
      <div className="terr__body">{failed.error}</div>
      <div className="terr__row">
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
  const [draft, setDraft] = useState('')
  const [hints, setHints] = useState<{ items: HintItem[]; index: number } | null>(null)
  const ref = useRef<HTMLTextAreaElement>(null)
  const hintSeq = useRef(0)
  const grow = (): void => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }
  const send = (): void => {
    if (!draft.trim() || snap.connection !== 'online') return
    void store.submit(draft)
    setDraft('')
    requestAnimationFrame(grow)
  }

  // Live slash/skill hints: debounced daemon completions while the draft is a
  // bare `/tok` or a `/skill <ref>`; latest request wins, stale ones drop.
  useEffect(() => {
    if (!wantsHints(draft) || snap.connection !== 'online') {
      setHints(null)
      return
    }
    const seq = ++hintSeq.current
    const timer = setTimeout(() => {
      void store
        .completeText(draft)
        .then(items => {
          if (hintSeq.current === seq) setHints({ items: items.slice(0, HINT_LIMIT), index: 0 })
        })
        .catch(() => {})
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
          : 'Describe the outcome — the agents plan and execute'

  return (
    <div className="composer-wrap">
      {hints && hints.items.length > 0 && (
        <div className="hints" role="listbox" aria-label="Command and skill hints" ref={hintsRef}>
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
              <span className="hints__label">{item.label}</span>
              <span className="hints__meta">{item.meta}</span>
            </button>
          ))}
          <div className="hints__keys"><kbd>tab</kbd> complete <kbd>↑↓</kbd> pick <kbd>esc</kbd> dismiss</div>
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
          rows={1}
          value={draft}
          placeholder={placeholder}
          spellCheck={false}
          onChange={e => { setDraft(e.target.value); grow() }}
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
              if (hints) { hintSeq.current += 1; setHints(null) }
              else if (draft) { setDraft(''); grow() }
              else if (snap.turnActive) store.cancel()
            }
          }}
        />
        <div className="composer__bar">
          <button
            className="cchip"
            title="Tool approvals — change in Settings → Permissions"
            onClick={() => store.openSettings('permissions')}
          >
            <span className="ico">⛨</span> {snap.permissionMode || 'policy'} <span className="c">▾</span>
          </button>
          <button
            className={`cchip${snap.planMode ? ' is-on' : ''}`}
            title={snap.planMode ? 'Plan mode ON — nothing touches disk' : 'Plan mode — propose before touching disk'}
            onClick={() => store.togglePlanMode()}
          >
            <span className="ico">⏸</span> plan{snap.planMode ? ' on' : ''}
          </button>
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
      <div className="composer__hints">
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

function Rail({ snap }: { snap: Snapshot }): ReactElement {
  const wide = useWindowWide()
  if (!wide) return <></>
  const fleet = snap.fleet
  const goal = parseGoal(snap.goal)
  return (
    <aside className="rail">
      {goal ? (
        <>
          <div className="rail__cap">Goal</div>
          <div className="goalcard">
            <span className="ph">◎ {goal.phase}{goal.activation === 'armed' ? ' · armed' : ''}</span>
            {goal.objective}
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
      ) : (
        <>
          <div className="rail__cap">Goal</div>
          <div className="rail__hint">
            No goal set — <code>/goal {'<objective>'}</code> in the composer arms autonomous rounds.
          </div>
        </>
      )}
      <div className="rail__cap">Fleet · {fleet.length}</div>
      <div className="rail__fleet">
        {fleet.length === 0
          ? <div className="rail__hint">{snap.turnActive ? 'No subagents running.' : 'Nothing running.'}</div>
          : fleet.map(row => <SessionCell key={row.id} row={row} displayOnly />)}
      </div>
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

function useWindowWide(): boolean {
  const [wide, setWide] = useState(() => (typeof window === 'undefined' ? true : window.innerWidth >= 1240))
  useEffect(() => {
    const onResize = (): void => setWide(window.innerWidth >= 1240)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])
  return useMemo(() => wide, [wide])
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
        store.openTaskModal()
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
