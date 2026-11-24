// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The VS Code chat view: one conversation, its composer and a slim header,
 * the way the Claude Code and Codex panels work, rather than the desktop
 * app's sidebar, rails and window chrome. Everything with a native home
 * goes there: files open in the editor, changes in VS Code's diff, Git in
 * Source Control, the picker for context is a quick pick. New task, history
 * and settings are the view's own title-bar actions; activity and usage are
 * in its overflow menu. Nothing else from the desktop window comes along.
 */

import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactElement } from 'react'

import { ActivityDetails, Announcer, Composer, ConnectionBanner, GlobalKeys, Stream, TaskModal } from '../../desktop/renderer/App.js'
import {
  activeAcpConfigOptionLabel,
  activeAcpModeLabel,
  activeAcpSummaryParts,
  activeAcpThoughtLevelLabel,
  extraAcpConfigOptions,
  isExternalAcpBinding,
  transportBadgeLabel,
} from '../../desktop/renderer/AcpRuntimePicker.js'
import { ActivityPanel, DesktopNavigation, DesktopSheet, type DesktopPanel } from '../../desktop/renderer/DesktopPanels.js'
import { ErrorBoundary } from '../../desktop/renderer/ErrorBoundary.js'
import { FindBar } from '../../desktop/renderer/FindBar.js'
import { Icon } from '../../desktop/renderer/Icon.js'
import { CommandPalette, SettingsModal } from '../../desktop/renderer/Overlays.js'
import { SessionSearch } from '../../desktop/renderer/SearchPanel.js'
import { FirstRunSetup } from '../../desktop/renderer/Setup.js'
import { Shortcuts } from '../../desktop/renderer/Shortcuts.js'
import { store, type Snapshot } from '../../desktop/renderer/store.js'
import { swarmMemberCounts } from '../../desktop/renderer/swarmState.js'
import type { SessionRow } from '../../desktop/renderer/types.js'
import { sidebarOrder } from '../../desktop/renderer/workspaceGroups.js'
import { ChangesTab, PlanTab } from '../../desktop/renderer/Workspaces.js'
import { AgentBrandGlyph } from '../../desktop/renderer/XerxesSprites.js'
import { SideQuestionCard } from './SideQuestionCard.js'

type Sheet = Exclude<DesktopPanel, null>

/** Wide enough for the conversation column and the activity column side by side. */
const ACTIVITY_COLUMN_MIN_WIDTH = 1_320

/** True while the view is at least `min` pixels wide, following resizes. */
function useWideView(min: number): boolean {
  const [wide, setWide] = useState(() => typeof window !== 'undefined' && window.innerWidth >= min)
  useEffect(() => {
    const update = () => setWide(window.innerWidth >= min)
    window.addEventListener('resize', update)
    return () => window.removeEventListener('resize', update)
  }, [min])
  return wide
}

/** Overflow-menu commands the extension pushes (`panel:<name>`) → the sheet they open. */
export const SHEETS: ReadonlySet<string> = new Set(['activity', 'usage', 'swarm', 'extensions'])

/** Next highlighted index for the task history list; stays at 0 when empty (never -1). */
export function nextHistoryCursor(current: number, rowCount: number, delta: 1 | -1): number {
  if (rowCount <= 0) return 0
  const clamped = Math.max(0, Math.min(rowCount - 1, current))
  return Math.max(0, Math.min(rowCount - 1, clamped + delta))
}

export function VscodeApp(): ReactElement {
  const snap = useSyncExternalStore(store.subscribe, store.getSnapshot)
  useEffect(() => { store.start() }, [])
  return <VscodeChat snap={snap} />
}

/**
 * A runtime from an older install keeps serving after the extension updates:
 * the runtime is shared, so a window reload reconnects to it. The store
 * replaces it once nothing is running; this says that is pending, what it
 * waits for, and lets the person restart now. Without it a busy or failed
 * update was invisible, and fixes in the new extension never took effect.
 */
function RuntimeUpdate({ snap }: { snap: Snapshot }): ReactElement {
  const working = snap.runtimeUpdate === 'checking' || snap.runtimeUpdate === 'restarting'
  const appOlder = snap.daemonWarning?.startsWith('The app is older') === true
  const blockers = snap.runtimeUpdate === 'waiting' ? snap.runtimeBlockers ?? [] : []
  const text = snap.runtimeUpdateMessage || (appOlder
    ? 'The running runtime is newer than this extension. Update the extension, then reload the window.'
    : 'The running runtime is older than this extension. It is replaced when nothing is running.')
  return <div className="xv-runtime" role="status" aria-live="polite">
    <span className="xv-runtime__text">{text}{blockers.length > 0 && <> Waiting for: {blockers.join(', ')}. Restarting now stops that work.</>}</span>
    {!appOlder && <button disabled={working} onClick={() => void store.restartRuntimeNow()}>{working ? 'Restarting…' : 'Restart now'}</button>}
  </div>
}

/** Presentational over the snapshot, like the desktop Shell. */
export function VscodeChat({ snap, initialSheet = null }: { snap: Snapshot; initialSheet?: Sheet | null }): ReactElement {
  const [sheet, setSheet] = useState<Sheet | null>(initialSheet)
  const [history, setHistory] = useState(false)
  const btwOpen = Boolean(snap.sideQuestionOpen)

  // A full window or a wide editor tab has room to keep activity in view.
  const activityColumn = useWideView(ACTIVITY_COLUMN_MIN_WIDTH) && !snap.noWorkspace
  const navigate = (panel: DesktopPanel, path?: string): void => {
    if ((panel as string | null) === 'btw') {
      setHistory(false)
      store.openSideQuestion()
      return
    }
    if (panel === 'activity' && activityColumn) { document.querySelector('.xv-activity')?.scrollTo({ top: 0 }); return }
    void navigateInVscode(window.xerxes, setSheet, panel, path)
  }
  useEffect(() => { setSheet(null); setHistory(false) }, [snap.cwd, snap.sessionKey])
  useEffect(() => { if (snap.swarmOpen) { setHistory(false); setSheet('swarm') } }, [snap.swarmOpenRevision])
  useEffect(() => { if (snap.sideQuestionOpen) setHistory(false) }, [snap.sideQuestionOpen, snap.sideQuestionOpenRevision])
  useEffect(() => window.xerxes.onMenuCommand?.(command => {
    if (command === 'history') setHistory(open => !open)
    else if (command === 'btw' || command === 'panel:btw') {
      if (store.getSnapshot().sideQuestionOpen) store.closeSideQuestion()
      else { setHistory(false); store.openSideQuestion() }
    }
    else if (command.startsWith('panel:') && SHEETS.has(command.slice(6))) setSheet(command.slice(6) as Sheet)
  }), [])
  return (
    <DesktopNavigation.Provider value={navigate}>
      <div className="app atelier xv">
        <FirstRunSetup snap={snap} />
        <Header
          snap={snap}
          history={history}
          setHistory={setHistory}
          openSwarm={() => setSheet('swarm')}
          openAcpSheet={() => { setHistory(false); setSheet('extensions') }}
        />
        <div className="xv-body">
          {snap.noWorkspace ? <NoFolder /> : <ErrorBoundary label="This conversation"><main className="chat xv-chat">
            {snap.connection !== 'online' && snap.blocks.length > 0 && <ConnectionBanner snap={snap} />}
            {snap.connection === 'online' && snap.daemonWarning && <RuntimeUpdate snap={snap} />}
            {snap.tab === 'changes' ? <div className="workspace"><ChangesTab snap={snap} /></div>
              : snap.tab === 'plan' ? <div className="workspace"><PlanTab snap={snap} /></div>
              : <Stream snap={snap} />}
            {btwOpen && <SideQuestionCard snap={snap} />}
            <Composer snap={snap} />
          </main></ErrorBoundary>}
          {activityColumn && <ErrorBoundary label="Activity"><aside className="xv-activity desktop-rail" aria-label="Activity">
            <ActivityDetails snap={snap} />
            <ActivityPanel snap={snap} />
          </aside></ErrorBoundary>}
        </div>
        <FindBar />
        <Shortcuts />
        <Announcer snap={snap} />
        <SettingsModal snap={snap} />
        {snap.taskModalOpen && <TaskModal snap={snap} />}
        {snap.paletteOpen && <CommandPalette snap={snap} />}
        {snap.searchOpen && <SessionSearch snap={snap} />}
        {sheet && <DesktopSheet panel={sheet} snap={snap} close={() => { setSheet(null); store.closeSwarmBoard() }} activityDetails={<ActivityDetails snap={snap} />} />}
        {(snap.workspaceBusy || snap.workspaceError) && <div className="workspace-notice" role={snap.workspaceError ? 'alert' : 'status'}>
          <span>{snap.workspaceError || 'Opening workspace…'}</span>
          {snap.workspaceError && <button aria-label="Dismiss workspace error" onClick={() => store.clearWorkspaceError()}><Icon name="close" size={13} /></button>}
        </div>}
        <GlobalKeys snap={snap} closeSurface={btwOpen ? () => store.closeSideQuestion() : sheet || history ? () => { setSheet(null); setHistory(false); store.closeSwarmBoard() } : null} />
      </div>
    </DesktopNavigation.Provider>
  )
}

/**
 * Where the renderer's links go in VS Code: a file opens in the editor, its
 * changes in VS Code's diff, Git in Source Control, "add context" in a quick
 * pick whose files land in the message as `@"path"` mentions (as the
 * desktop's file viewer adds them). Anything else opens as a sheet.
 */
export async function navigateInVscode(
  bridge: Pick<Window['xerxes'], 'openInEditor' | 'openDiff' | 'pickFiles' | 'showSourceControl'>,
  openSheet: (sheet: Sheet | null) => void,
  panel: DesktopPanel,
  path?: string,
): Promise<void> {
  if (panel === null) openSheet(null)
  else if (panel === 'files' && path) await bridge.openInEditor?.(path)
  else if (panel === 'files') {
    const paths = await bridge.pickFiles?.() ?? []
    if (paths.length) window.dispatchEvent(new CustomEvent('xerxes:add-context', { detail: paths.map(file => '@' + JSON.stringify(file)).join(' ') }))
  }
  else if (panel === 'review') await (path ? bridge.openDiff?.(path) : bridge.showSourceControl?.())
  else openSheet(panel)
}

/**
 * The title is the history menu, as in Claude Code: click it to switch to
 * an earlier task in this folder. Session edits and a pending decision are
 * the only other things worth the room.
 */
function Header({
  snap,
  history,
  setHistory,
  openSwarm,
  openAcpSheet,
}: {
  snap: Snapshot
  history: boolean
  setHistory: (open: boolean) => void
  openSwarm: () => void
  openAcpSheet: () => void
}): ReactElement {
  const titleRef = useRef<HTMLButtonElement>(null)
  const totals = useMemo(() => snap.changes.reduce((sum, file) => ({ adds: sum.adds + file.adds, dels: sum.dels + file.dels }), { adds: 0, dels: 0 }), [snap.changes])
  const title = snap.currentTitle || (snap.connection === 'online' ? 'New task' : snap.connection === 'connecting' ? 'Connecting…' : 'Not connected')
  const needsInput = snap.approval !== null || snap.question !== null
  const swarmCounts = snap.swarm?.active ? swarmMemberCounts(snap.swarm.members) : null
  return <header className="xv-head">
    <button ref={titleRef} className="xv-title" aria-haspopup="dialog" aria-expanded={history} title="Switch task" onClick={() => setHistory(!history)}>
      <span className="xv-title__text">{title}</span>
      <Icon name="caretDown" size={12} />
    </button>
    <span className="xv-head__flex" />
    {!snap.noWorkspace && isExternalAcpBinding(snap.acpBinding) && (() => {
      const summary = activeAcpSummaryParts(snap.acpBinding).join(' · ')
      const externalThinking = activeAcpThoughtLevelLabel(snap.acpBinding)
      const externalMode = snap.acpBinding.modes?.length ? activeAcpModeLabel(snap.acpBinding) : null
      const extraMeta = extraAcpConfigOptions(snap.acpBinding).map(activeAcpConfigOptionLabel).filter(Boolean)
      const pillMeta = [externalThinking, externalMode, ...extraMeta].filter(Boolean).join(' · ')
      return (
        <button
          type="button"
          className="xv-pill xv-pill--acp"
          data-provider={snap.acpBinding.provider}
          aria-haspopup="dialog"
          title={`${snap.acpBinding.name} (${transportBadgeLabel(snap.acpBinding.transport)})${summary ? ` · ${summary}` : ''} — click to inspect runtime & customize`}
          aria-label={`External runtime: ${snap.acpBinding.name}${summary ? ` · ${summary}` : ''}`}
          onClick={openAcpSheet}
        >
          <AgentBrandGlyph provider={snap.acpBinding.provider} size={11} />
          <span className="xv-pill__label">{snap.acpBinding.name}{pillMeta ? ` · ${pillMeta}` : ''}</span>
        </button>
      )
    })()}
    {swarmCounts && (
      <button
        type="button"
        className="xv-pill xv-pill--swarm"
        data-working={swarmCounts.working > 0 ? 'true' : undefined}
        title="Open Immortals Board"
        aria-label="Open Immortals Board"
        onClick={openSwarm}
      >
        <Icon name="tree" size={11} />
        <span className="xv-pill__label">Immortals{swarmCounts.working > 0 ? ` · ${swarmCounts.working}` : ''}</span>
      </button>
    )}
    {needsInput && snap.interactionHidden && <button className="xv-pill xv-pill--need" title="Show the request waiting for you" onClick={() => store.showInteraction()}><span className="xv-pill__label">needs input</span></button>}
    {snap.changes.length > 0 && <button className="xv-pill xv-pill--diff" title="Files this task changed — open Source Control" onClick={() => void window.xerxes.showSourceControl?.()}>
      <span className="add">+{totals.adds}</span> <span className="del">−{totals.dels}</span>
    </button>}
    {snap.tab !== 'activity' && <button className="xv-icon" title="Back to the conversation" aria-label="Back to the conversation" onClick={() => store.setTab('activity')}><Icon name="chat" size={15} /></button>}
    <span className="xv-dot" data-state={snap.connection} title={snap.connection === 'online' ? 'Runtime connected' : snap.connection === 'connecting' ? 'Connecting to the runtime' : snap.error || 'Runtime offline'} />
    {history && <History snap={snap} close={() => setHistory(false)} returnFocus={() => titleRef.current?.focus()} />}
  </header>
}

/** Earlier tasks in this folder, newest first, filterable. */
function History({ snap, close, returnFocus }: { snap: Snapshot; close: () => void; returnFocus?: () => void }): ReactElement {
  const [needle, setNeedle] = useState('')
  const [cursor, setCursor] = useState(0)
  const field = useRef<HTMLInputElement>(null)
  const panel = useRef<HTMLDivElement>(null)
  const closeWithFocus = (): void => { close(); returnFocus?.() }
  useEffect(() => { field.current?.focus() }, [])
  useEffect(() => {
    const dismiss = (event: MouseEvent) => { if (event.target instanceof Node && !panel.current?.contains(event.target) && !(event.target as Element).closest?.('.xv-title')) close() }
    document.addEventListener('mousedown', dismiss)
    return () => document.removeEventListener('mousedown', dismiss)
  }, [close])
  const titleOf = (row: SessionRow): string => historyTitle(snap, row)
  const rows = useMemo(() => historyRows(snap, needle), [snap.live, snap.sessions, snap.cwd, snap.snippets, snap.currentId, snap.currentTitle, snap.sessionKey, snap.turnActive, snap.turnCount, needle])
  const pick = (row: SessionRow | undefined): void => {
    if (!row) return
    close()
    if (row.id !== snap.currentId) void store.openSession(row.id, row)
  }
  return <div
    className="xv-history"
    ref={panel}
    role="dialog"
    aria-label="Task history"
    onKeyDown={event => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        closeWithFocus()
      }
    }}
  >
    <div className="xv-history__search">
      <Icon name="search" size={13} />
      <input ref={field} value={needle} placeholder="Search tasks" aria-label="Search tasks" spellCheck={false}
        onChange={event => { setNeedle(event.target.value); setCursor(0) }}
        onKeyDown={event => {
          if (event.key === 'ArrowDown') { event.preventDefault(); setCursor(index => nextHistoryCursor(index, rows.length, 1)) }
          else if (event.key === 'ArrowUp') { event.preventDefault(); setCursor(index => nextHistoryCursor(index, rows.length, -1)) }
          else if (event.key === 'Enter') { event.preventDefault(); if (rows.length > 0) pick(rows[cursor]) }
          else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); closeWithFocus() }
        }} />
    </div>
    <button className="xv-history__new" onClick={() => { close(); store.newChat() }}><Icon name="plus" size={13} /> New task</button>
    <div className="xv-history__list" role="listbox" aria-label="Tasks">
      {rows.length === 0 && <p className="xv-history__empty">{needle ? 'No tasks match.' : 'No earlier tasks in this folder.'}</p>}
      {rows.map((row, index) => <button key={row.id} role="option" aria-selected={index === cursor} aria-current={row.id === snap.currentId || undefined}
        className={`xv-history__row${index === cursor ? ' is-on' : ''}`} onMouseEnter={() => setCursor(index)} onClick={() => pick(row)}>
        <span className="xv-history__title">{titleOf(row)}</span>
        {row.status === 'working' || row.status === 'running' ? <span className="xv-history__live">working</span> : <span className="xv-history__age">{row.age}</span>}
      </button>)}
    </div>
  </div>
}

const historyTitle = (snap: Snapshot, row: SessionRow): string => snap.snippets[row.id] ?? (row.title || 'Untitled task')

/** The history list: main tasks in this folder, the open one included, newest first, matching the filter. */
export function historyRows(snap: Snapshot, needle: string): SessionRow[] {
  const seen = new Set<string>()
  // The open task, from the snapshot so its status is live; the lists can lag or leave it out.
  const current: SessionRow[] = snap.currentId ? [{
    id: snap.currentId, key: snap.sessionKey || snap.currentId, title: snap.currentTitle || 'New task',
    status: snap.turnActive ? 'working' : 'idle', age: '', current: true, kind: 'main',
    turns: snap.turnCount, messages: 0, cwd: snap.cwd, untitled: false,
    activeAt: [...snap.live, ...snap.sessions].find(row => row.id === snap.currentId)?.activeAt ?? Number.POSITIVE_INFINITY,
  }] : []
  const all = [...current, ...snap.live, ...snap.sessions].filter(row => {
    if (row.kind !== 'main' || seen.has(row.id) || (snap.cwd && row.cwd && row.cwd !== snap.cwd)) return false
    seen.add(row.id)
    return true
  })
  const query = needle.trim().toLowerCase()
  return sidebarOrder(all).filter(row => !query || historyTitle(snap, row).toLowerCase().includes(query) || row.id.includes(query))
}

/** VS Code without a folder: the runtime works on a folder, so ask for one. */
function NoFolder(): ReactElement {
  return <div className="xv-empty">
    <p>Open a folder to work on it with Xerxes.</p>
    <button className="xv-button" onClick={() => store.chooseWorkspace()}>Open Folder</button>
  </div>
}
