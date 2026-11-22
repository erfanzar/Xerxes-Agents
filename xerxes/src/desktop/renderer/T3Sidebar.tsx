// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * T3 Code Sidebar V2 with Machine + Project Folder Grouping:
 * - 52px top chrome: sidebar toggle + `Xerxes Code` wordmark + adjacent Back/Forward arrows.
 * - Borderless inline `Search` + 3 right-aligned icons (Filter/View, Add Project, New Thread).
 * - Projects mode (default): Machine groups (`This Mac`, SSH hosts) → T3 Project Folders
 *   (monogram, name, collapsed status dot + count swapping on hover to New Thread, rotating
 *   trailing chevron) → clean vertical-guide nested thread rows (rich active row with branch
 *   & runtime emblem; single-line calm inactive rows with hover Pin/Snooze/Settle).
 * - Inbox mode (optional via filter menu): T3 flat chronological 3-line cards.
 * - Pinned threads at top, collapsible `Snoozed` & `Settled  N >` bottom shelves,
 *   Undo banner (`⌘Z`), and minimal icon footer.
 */

import { useEffect, useLayoutEffect, useMemo, useRef, useState, type ReactElement, type ReactNode } from 'react'
import { store, type Snapshot } from './store.js'
import { ProjectMonogram } from './ProjectMonogram.js'
import { T3Icon } from './T3Icons.js'
import { copyToClipboard } from './CopyButton.js'
import { LearnedBadge } from './LearnedPanel.js'
import { isExternalAcpBinding } from './AcpRuntimePicker.js'
import { RuntimeBrandIcon } from './T3ModelPicker.js'
import type { AgentBrandProvider } from './XerxesSprites.js'
import {
  formatSnoozeRemaining,
  isSnoozedActive,
  normalizeWorkspacePath,
  partitionInboxThreads,
  snoozePresets,
  type InboxUndoEntry,
  type SidebarViewMode,
  type ThreadInboxState,
} from './threadInbox.js'
import {
  SIDEBAR_SECTION_ROWS,
  THIS_MAC,
  limitRows,
  loadCollapsedSections,
  saveCollapsedSections,
  workspaceName,
  type SidebarSection,
} from './workspaceGroups.js'
import type { DesktopPageName, DesktopPanel } from './DesktopPanels.js'

export interface T3SidebarRow {
  readonly id: string
  readonly key: string
  readonly title: string
  readonly snippet?: string
  readonly status: string
  readonly age: string
  readonly current: boolean
  readonly turns: number
  readonly cwd: string
  readonly branch?: string
  readonly provider?: AgentBrandProvider
  readonly host?: string
  readonly contextId?: number
  readonly opensWindow?: boolean
}

export interface MachineSectionGroup<T> {
  readonly host: string
  readonly isLocal: boolean
  readonly sections: readonly SidebarSection<T>[]
}

/**
 * Groups `SidebarSection`s by machine (`This Mac` first, then remote SSH hosts),
 * preserving `sidebarSections()`'s stable host and folder ordering.
 */
export function groupSectionsByMachine<T>(
  sections: readonly SidebarSection<T>[],
): readonly MachineSectionGroup<T>[] {
  const order: string[] = []
  const byHost = new Map<string, SidebarSection<T>[]>()
  for (const section of sections) {
    const host = section.host || THIS_MAC
    const list = byHost.get(host)
    if (list) list.push(section)
    else {
      order.push(host)
      byHost.set(host, [section])
    }
  }
  return order.map(host => ({
    host,
    isLocal: host === THIS_MAC,
    sections: byHost.get(host)!,
  }))
}

function statusDotTone(status: string): 'working' | 'failed' | 'waiting' | 'idle' {
  if (status === 'working' || status === 'running') return 'working'
  if (status === 'failed' || status === 'error') return 'failed'
  if (status === 'waiting') return 'waiting'
  return 'idle'
}

/** Highest-priority status among hidden rows in a collapsed folder or machine. */
function folderStatusDot(rows: readonly { readonly status: string }[]): 'failed' | 'working' | null {
  let hasWorking = false
  for (const row of rows) {
    const tone = statusDotTone(row.status)
    if (tone === 'failed') return 'failed'
    if (tone === 'working') hasWorking = true
  }
  return hasWorking ? 'working' : null
}

export function T3Sidebar({
  snap,
  page,
  sections,
  allRows,
  filter,
  setFilter,
  inbox,
  undo,
  onUndo,
  onDismissUndo,
  onSetViewMode,
  onToggleProjectScope,
  onClearProjectScope,
  onTogglePin,
  onSettle,
  onSnooze,
  onToggleShelf,
  onRemoveWorkspace,
  onRemoveRemoteHost,
  canGoBack,
  canGoForward,
  onGoBack,
  onGoForward,
  onOpen,
  sidebarToggle,
  runtimeStatus,
}: {
  readonly snap: Snapshot
  readonly page: DesktopPageName | null
  readonly sections: readonly SidebarSection<Snapshot['live'][number]>[]
  readonly allRows: readonly T3SidebarRow[]
  readonly filter: string
  readonly setFilter: (value: string) => void
  readonly inbox: ThreadInboxState
  readonly undo: InboxUndoEntry | null
  readonly onUndo: () => void
  readonly onDismissUndo: () => void
  readonly onSetViewMode: (mode: SidebarViewMode) => void
  readonly onToggleProjectScope: (cwd: string) => void
  readonly onClearProjectScope: () => void
  readonly onTogglePin: (id: string, title: string) => void
  readonly onSettle: (id: string, title: string, settled: boolean) => void
  readonly onSnooze: (id: string, title: string, wakeAt: number | null) => void
  readonly onToggleShelf: (shelf: 'snoozed' | 'settled') => void
  readonly onRemoveWorkspace?: (cwd: string, label: string) => void
  readonly onRemoveRemoteHost?: (alias: string) => void
  readonly canGoBack: boolean
  readonly canGoForward: boolean
  readonly onGoBack: () => void
  readonly onGoForward: () => void
  readonly onOpen: (panel: DesktopPanel) => void
  readonly sidebarToggle: ReactNode
  readonly runtimeStatus: ReactNode
}): ReactElement {
  const online = snap.connection === 'online'
  const needle = filter.trim().toLowerCase()
  const [scopeOpen, setScopeOpen] = useState(false)
  const [snoozeMenuId, setSnoozeMenuId] = useState<string | null>(null)
  type FolderMenuState =
    | { readonly kind: 'local'; readonly key: string; readonly cwd: string; readonly label: string; readonly x: number; readonly y: number }
    | { readonly kind: 'remote'; readonly key: string; readonly cwd: string; readonly label: string; readonly alias: string; readonly contextId?: number; readonly x: number; readonly y: number }
  const [folderMenu, setFolderMenu] = useState<FolderMenuState | null>(null)
  const [settledLimit, setSettledLimit] = useState(8)
  const [collapsed, setCollapsed] = useState<ReadonlySet<string>>(loadCollapsedSections)
  const [showAll, setShowAll] = useState<ReadonlySet<string>>(() => new Set())
  const scopeRef = useRef<HTMLDivElement>(null)
  const snoozeRef = useRef<HTMLDivElement>(null)
  const folderMenuRef = useRef<HTMLDivElement>(null)

  const activeProvider: AgentBrandProvider = isExternalAcpBinding(snap.acpBinding)
    ? snap.acpBinding.provider
    : 'xerxes'

  useEffect(() => {
    if (!scopeOpen) return
    const onDown = (event: MouseEvent): void => {
      if (event.target instanceof Node && !scopeRef.current?.contains(event.target)) setScopeOpen(false)
    }
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') { event.preventDefault(); setScopeOpen(false) }
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [scopeOpen])

  useEffect(() => {
    if (!snoozeMenuId) return
    const onDown = (event: MouseEvent): void => {
      if (event.target instanceof Node && !snoozeRef.current?.contains(event.target)) setSnoozeMenuId(null)
    }
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') { event.preventDefault(); setSnoozeMenuId(null) }
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [snoozeMenuId])

  useLayoutEffect(() => {
    const element = folderMenuRef.current
    if (!element || !folderMenu) return
    const place = (): void => {
      const rect = element.getBoundingClientRect()
      element.style.left = `${Math.max(8, Math.min(folderMenu.x, window.innerWidth - rect.width - 8))}px`
      element.style.top = `${Math.max(8, Math.min(folderMenu.y, window.innerHeight - rect.height - 8))}px`
    }
    place()
    element.querySelector<HTMLElement>('button')?.focus()
    window.addEventListener('resize', place)
    return () => window.removeEventListener('resize', place)
  }, [folderMenu])

  useEffect(() => {
    if (!folderMenu) return
    const onKey = (event: KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        setFolderMenu(null)
      }
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [folderMenu])

  const removeLocalWorkspace = (cwd: string, label: string): void => {
    setFolderMenu(null)
    if (onRemoveWorkspace) onRemoveWorkspace(cwd, label)
    else void store.removeWorkspace(cwd)
  }

  const removeRemoteWorkspace = (alias: string): void => {
    setFolderMenu(null)
    if (onRemoveRemoteHost) onRemoveRemoteHost(alias)
    else void store.removeRemoteHost(alias)
  }

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

  const hiddenWorkspaceSet = useMemo(
    () => new Set(inbox.hiddenWorkspaces.map(normalizeWorkspacePath).filter(Boolean)),
    [inbox.hiddenWorkspaces],
  )
  const isWorkspaceVisible = (cwd: string, isRemote: boolean): boolean => {
    if (isRemote || !cwd) return true
    return !hiddenWorkspaceSet.has(normalizeWorkspacePath(cwd))
  }

  const visibleSections = useMemo(
    () => sections.filter(section => isWorkspaceVisible(section.cwd, section.kind === 'context' && section.context.remote)),
    [sections, hiddenWorkspaceSet],
  )
  const visibleRows = useMemo(
    () => allRows.filter(row => row.current || isWorkspaceVisible(row.cwd, Boolean(row.host && row.host !== THIS_MAC))),
    [allRows, hiddenWorkspaceSet],
  )

  const projects = useMemo(() => {
    const map = new Map<string, { cwd: string; name: string; host: string; count: number }>()
    for (const section of visibleSections) {
      const existing = map.get(section.cwd)
      if (existing) existing.count += section.rows.length
      else map.set(section.cwd, { cwd: section.cwd, name: section.folder || workspaceName(section.cwd) || 'workspace', host: section.host, count: section.rows.length })
    }
    if (snap.cwd && isWorkspaceVisible(snap.cwd, false) && !map.has(snap.cwd)) {
      map.set(snap.cwd, { cwd: snap.cwd, name: workspaceName(snap.cwd) || 'workspace', host: THIS_MAC, count: 0 })
    }
    for (const dir of snap.workspaceDirectories ?? []) {
      if (dir && isWorkspaceVisible(dir, false) && !map.has(dir)) {
        map.set(dir, { cwd: dir, name: workspaceName(dir) || dir, host: THIS_MAC, count: 0 })
      }
    }
    for (const row of visibleRows) {
      const cwd = row.cwd || ''
      if (!cwd || map.has(cwd)) continue
      map.set(cwd, { cwd, name: workspaceName(cwd) || cwd, host: row.host || THIS_MAC, count: 1 })
    }
    return [...map.values()]
  }, [visibleSections, visibleRows, snap.cwd, snap.workspaceDirectories, hiddenWorkspaceSet])

  const activeScopeSet = useMemo(() => new Set(inbox.projectScope), [inbox.projectScope])
  const scopedRows = useMemo(
    () => (activeScopeSet.size === 0 ? visibleRows : visibleRows.filter(row => activeScopeSet.has(row.cwd))),
    [visibleRows, activeScopeSet],
  )

  const now = Date.now()
  const partitioned = useMemo(() => partitionInboxThreads(scopedRows, inbox, now), [scopedRows, inbox, now])
  const presets = useMemo(() => snoozePresets(now), [snoozeMenuId])
  const pinnedIdSet = useMemo(() => new Set(inbox.pinnedIds), [inbox.pinnedIds])
  const settledIdSet = useMemo(() => new Set(inbox.settledIds), [inbox.settledIds])

  const scopedSections = useMemo(
    () => visibleSections.filter(section => activeScopeSet.size === 0 || activeScopeSet.has(section.cwd)),
    [visibleSections, activeScopeSet],
  )
  const machineGroups = useMemo(() => groupSectionsByMachine(scopedSections), [scopedSections])

  const openRow = (row: T3SidebarRow): void => {
    if (row.contextId !== undefined) {
      void store.activateContext(row.contextId, row.id)
      return
    }
    void store.openSession(row.id)
  }

  const openMenu = (row: T3SidebarRow, x: number, y: number): void => {
    if (row.contextId !== undefined) return
    store.openSessionMenu({ id: row.id, key: row.key, title: row.title }, x, y)
  }

  const renderThreadCard = (
    row: T3SidebarRow,
    variant: 'inbox' | 'folder' | 'snoozed' | 'settled',
  ): ReactElement => {
    const title = row.snippet ?? row.title
    const projectLabel = workspaceName(row.cwd) || 'workspace'
    const tone = statusDotTone(row.status)
    const pinned = pinnedIdSet.has(row.id)
    const snoozed = isSnoozedActive(inbox.snoozed, row.id, now)
    const wakeAt = inbox.snoozed[row.id]
    const snoozeOpen = snoozeMenuId === row.id
    const rowBranch = row.branch || (row.cwd && row.cwd === snap.cwd ? snap.branch : '')
    const rowProvider: AgentBrandProvider = row.provider ?? (row.current ? activeProvider : 'xerxes')
    const displayAge = row.current && !row.age && tone !== 'working' ? 'now' : row.age

    if (variant === 'settled') {
      return (
        <div key={row.id} className={`t3-thread-wrap t3-thread-wrap--settled${row.current ? ' is-current' : ''}`}>
          <button
            type="button"
            className={`sess t3-settled-row${row.current ? ' is-current' : ''}`}
            onClick={() => openRow(row)}
            onContextMenu={event => { event.preventDefault(); openMenu(row, event.clientX, event.clientY) }}
            title={`${projectLabel} · ${title}`}
          >
            <ProjectMonogram name={projectLabel} seed={row.cwd} size={14} />
            <span className="sess__t t3-settled-row__title">{title}</span>
            {row.age && <span className="sess__age t3-thread__age">{row.age}</span>}
          </button>
          <div className="t3-thread__hover-actions">
            <button
              type="button"
              className="t3-thread__act"
              title="Restore thread"
              aria-label={`Restore ${title}`}
              onClick={event => { event.stopPropagation(); onSettle(row.id, title, false) }}
            >
              <T3Icon name="undo" size={13} />
            </button>
          </div>
        </div>
      )
    }

    const isNested = variant === 'folder'
    const showNestedSubline = isNested && row.current && Boolean(rowBranch || rowProvider)

    return (
      <div
        key={row.id}
        className={`t3-thread-wrap${isNested ? ' t3-thread-wrap--nested' : ''}${row.current ? ' is-current' : ''}${snoozeOpen ? ' has-menu' : ''}`}
        data-thread-id={row.id}
      >
        <button
          type="button"
          className={`sess t3-thread${isNested ? ' t3-thread--nested' : ''}${showNestedSubline ? ' t3-thread--nested-rich' : ''}${row.current ? ' is-current' : ''}`}
          aria-haspopup={row.contextId === undefined ? 'menu' : undefined}
          title={row.opensWindow ? 'Open this session while the current task continues' : `${projectLabel} · ${title}`}
          onClick={() => openRow(row)}
          onKeyDown={event => {
            if (event.key === 'ContextMenu' || (event.key === 'F10' && event.shiftKey)) {
              event.preventDefault()
              const rect = event.currentTarget.getBoundingClientRect()
              openMenu(row, rect.left, rect.bottom)
            }
          }}
          onContextMenu={event => {
            event.preventDefault()
            openMenu(row, event.clientX, event.clientY)
          }}
        >
          {isNested ? (
            <div className="t3-thread__nested-body">
              <div className="t3-thread__nested-top">
                {tone === 'working' && <span className="sess__dot t3-thread__nested-dot" data-status="working" />}
                {tone === 'failed' && <span className="sess__dot t3-thread__nested-dot" data-status="failed" />}
                <span className="sess__t t3-thread__title">{title}</span>
                <span className="t3-thread__right">
                  {tone === 'working' ? (
                    <span className="t3-thread__status" data-status="working">
                      <span>acting</span>
                      {row.age && <span className="sess__age">{row.age}</span>}
                    </span>
                  ) : tone === 'failed' ? (
                    <span className="t3-thread__status" data-status="failed">
                      <span>failed</span>
                    </span>
                  ) : displayAge ? (
                    <span className="sess__age t3-thread__age">{displayAge}</span>
                  ) : null}
                </span>
              </div>
              {showNestedSubline && (
                <div className="t3-thread__sub">
                  <span className="t3-thread__branch">{rowBranch || ''}</span>
                  <span className="t3-thread__provider" aria-hidden="true">
                    <RuntimeBrandIcon provider={rowProvider} size={13} />
                  </span>
                </div>
              )}
            </div>
          ) : (
            <div className="t3-thread__card">
              <div className="t3-thread__meta">
                <ProjectMonogram name={projectLabel} seed={row.cwd} size={14} />
                <span className="t3-thread__project">{projectLabel}</span>
                {row.host && row.host !== THIS_MAC && <span className="t3-thread__host">· {row.host}</span>}
                <span className="t3-thread__spacer" />
                <span className="t3-thread__right">
                  {tone === 'working' ? (
                    <span className="t3-thread__status" data-status="working">
                      <span className="sess__dot" data-status="working" />
                      <span>acting</span>
                      {row.age && <span className="sess__age">{row.age}</span>}
                    </span>
                  ) : tone === 'failed' ? (
                    <span className="t3-thread__status" data-status="failed">
                      <span className="sess__dot" data-status="failed" />
                      <span>failed</span>
                    </span>
                  ) : variant === 'snoozed' && wakeAt ? (
                    <span className="t3-thread__snooze-badge">
                      <T3Icon name="alarmClock" size={11} />
                      <span>{formatSnoozeRemaining(wakeAt, now)}</span>
                    </span>
                  ) : displayAge ? (
                    <span className="sess__age t3-thread__age">{displayAge}</span>
                  ) : null}
                </span>
              </div>
              <div className="t3-thread__title-row">
                <span className="sess__t t3-thread__title">{title}</span>
              </div>
              {(rowBranch || row.current) && (
                <div className="t3-thread__sub">
                  <span className="t3-thread__branch">{rowBranch || ''}</span>
                  <span className="t3-thread__provider" aria-hidden="true">
                    <RuntimeBrandIcon provider={rowProvider} size={13} />
                  </span>
                </div>
              )}
            </div>
          )}
        </button>

        {row.contextId === undefined && (
          <div className="t3-thread__hover-actions" aria-label="Thread quick actions">
            {variant === 'snoozed' ? (
              <button
                type="button"
                className="t3-thread__act"
                title="Wake thread now"
                aria-label={`Wake ${title}`}
                onClick={event => { event.stopPropagation(); onSnooze(row.id, title, null) }}
              >
                <T3Icon name="undo" size={13} />
              </button>
            ) : (
              <>
                <button
                  type="button"
                  className={`t3-thread__act${pinned ? ' is-active' : ''}`}
                  title={pinned ? 'Unpin thread' : 'Pin thread'}
                  aria-label={pinned ? `Unpin ${title}` : `Pin ${title}`}
                  onClick={event => { event.stopPropagation(); onTogglePin(row.id, title) }}
                >
                  <T3Icon name={pinned ? 'pinOff' : 'pin'} size={13} />
                </button>
                <button
                  type="button"
                  className={`t3-thread__act${snoozed || snoozeOpen ? ' is-active' : ''}`}
                  title="Snooze thread"
                  aria-label={`Snooze ${title}`}
                  aria-haspopup="menu"
                  aria-expanded={snoozeOpen}
                  onClick={event => {
                    event.stopPropagation()
                    setSnoozeMenuId(current => (current === row.id ? null : row.id))
                  }}
                >
                  <T3Icon name="alarmClock" size={13} />
                </button>
                <button
                  type="button"
                  className="t3-thread__act t3-thread__act--settle"
                  title="Settle thread"
                  aria-label={`Settle ${title}`}
                  onClick={event => { event.stopPropagation(); onSettle(row.id, title, true) }}
                >
                  <T3Icon name="check" size={13} />
                </button>
              </>
            )}
          </div>
        )}

        {snoozeOpen && (
          <div ref={snoozeRef} className="t3-snooze-menu" role="menu" aria-label={`Snooze ${title}`}>
            <div className="t3-snooze-menu__head">Snooze until</div>
            {presets.map(preset => (
              <button
                key={preset.id}
                type="button"
                role="menuitem"
                className="t3-snooze-menu__item"
                onClick={event => {
                  event.stopPropagation()
                  setSnoozeMenuId(null)
                  onSnooze(row.id, title, preset.wakeAt)
                }}
              >
                <span>{preset.label}</span>
                <span className="t3-snooze-menu__meta">{preset.detail}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    )
  }

  const shownSettled = partitioned.settled.slice(0, settledLimit)
  const hiddenSettled = Math.max(0, partitioned.settled.length - shownSettled.length)

  return (
    <aside id="session-sidebar" className="side t3-side" data-app-sidebar="true" data-sidebar-view={inbox.viewMode} aria-label="Sessions">
      {/* 52px T3 Sidebar Top Chrome: Toggle + Xerxes Code + adjacent Back/Forward */}
      <div className="side__head t3-side__head">
        {sidebarToggle}
        <button
          type="button"
          className="t3-wordmark"
          title="Return to active thread"
          onClick={() => onOpen(null)}
        >
          <span className="t3-wordmark__brand">Xerxes</span>
          <span className="t3-wordmark__sub">Code</span>
        </button>
        <div className="t3-side__history" role="group" aria-label="Navigation history">
          <button
            type="button"
            className="t3-icon-btn"
            disabled={!canGoBack}
            title="Back"
            aria-label="Go back"
            onClick={onGoBack}
          >
            <T3Icon name="arrowLeft" size={14} />
          </button>
          <button
            type="button"
            className="t3-icon-btn"
            disabled={!canGoForward}
            title="Forward"
            aria-label="Go forward"
            onClick={onGoForward}
          >
            <T3Icon name="arrowRight" size={14} />
          </button>
        </div>
      </div>

      {/* Borderless Search + Filter/View + Add Project + New Thread */}
      <div className="t3-side__toolbar">
        <label className="t3-side__search">
          <T3Icon name="search" size={14} className="t3-side__search-icon" />
          <input
            className="t3-side__search-input"
            value={filter}
            onChange={event => setFilter(event.target.value)}
            onKeyDown={event => {
              if (event.key === 'Escape' && filter) {
                event.preventDefault()
                setFilter('')
              } else if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
                event.preventDefault()
                store.openSessionSearch()
              }
            }}
            placeholder="Search"
            disabled={!online}
            aria-label="Filter threads"
            spellCheck={false}
          />
          {filter && (
            <button
              type="button"
              className="t3-side__search-clear"
              aria-label="Clear filter"
              onClick={() => setFilter('')}
            >
              <T3Icon name="close" size={12} />
            </button>
          )}
        </label>

        <div className="t3-side__scope-anchor" ref={scopeRef}>
          <button
            type="button"
            className={`t3-icon-btn t3-side__toolbtn${scopeOpen || activeScopeSet.size > 0 || inbox.viewMode === 'inbox' ? ' is-active' : ''}`}
            title={`Sidebar view (${inbox.viewMode === 'folders' ? 'Projects' : 'Inbox'})${activeScopeSet.size > 0 ? ` · ${activeScopeSet.size} filtered` : ''}`}
            aria-label="Sidebar view and project filter"
            aria-haspopup="dialog"
            aria-expanded={scopeOpen}
            onClick={() => setScopeOpen(value => !value)}
          >
            <T3Icon name="filterLines" size={14} />
            {activeScopeSet.size > 0 && <span className="t3-side__scope-dot" aria-hidden="true" />}
          </button>

          {scopeOpen && (
            <div className="t3-scope-pop" role="dialog" aria-label="Sidebar view and project filter">
              <div className="t3-scope-pop__sec">
                <div className="t3-scope-pop__label">View</div>
                <div className="t3-scope-pop__seg" role="radiogroup" aria-label="Sidebar layout">
                  <button
                    type="button"
                    role="radio"
                    aria-checked={inbox.viewMode === 'folders'}
                    className={inbox.viewMode === 'folders' ? 'is-on' : ''}
                    onClick={() => onSetViewMode('folders')}
                  >
                    <T3Icon name="folders" size={13} />
                    <span>Projects</span>
                  </button>
                  <button
                    type="button"
                    role="radio"
                    aria-checked={inbox.viewMode === 'inbox'}
                    className={inbox.viewMode === 'inbox' ? 'is-on' : ''}
                    onClick={() => onSetViewMode('inbox')}
                  >
                    <T3Icon name="inbox" size={13} />
                    <span>Inbox</span>
                  </button>
                </div>
              </div>

              {projects.length > 0 && (
                <div className="t3-scope-pop__sec">
                  <div className="t3-scope-pop__head">
                    <span className="t3-scope-pop__label">Projects</span>
                    {activeScopeSet.size > 0 && (
                      <button type="button" className="t3-scope-pop__clear" onClick={onClearProjectScope}>
                        Reset
                      </button>
                    )}
                  </div>
                  <div className="t3-scope-pop__list">
                    <button
                      type="button"
                      className={`t3-scope-pop__row${activeScopeSet.size === 0 ? ' is-selected' : ''}`}
                      onClick={onClearProjectScope}
                    >
                      <span>All projects</span>
                      <span className="t3-scope-pop__count">{allRows.length}</span>
                    </button>
                    {projects.map(project => {
                      const picked = activeScopeSet.has(project.cwd)
                      return (
                        <button
                          key={`${project.host}:${project.cwd}`}
                          type="button"
                          className={`t3-scope-pop__row${picked ? ' is-selected' : ''}`}
                          onClick={() => onToggleProjectScope(project.cwd)}
                        >
                          <ProjectMonogram name={project.name} seed={project.cwd} size={14} />
                          <span className="t3-scope-pop__name">
                            {project.name}
                            {project.host !== THIS_MAC ? ` · ${project.host}` : ''}
                          </span>
                          <span className="t3-scope-pop__count">{project.count}</span>
                        </button>
                      )
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        <button
          type="button"
          className="t3-icon-btn t3-side__toolbtn"
          title="Add project folder…"
          aria-label="Add project"
          onClick={() => store.chooseWorkspace()}
        >
          <T3Icon name="folderPlus" size={14} />
        </button>

        <button
          type="button"
          className="t3-icon-btn t3-side__toolbtn t3-side__new"
          disabled={!online}
          title={snap.cwd ? `New thread in ${workspaceName(snap.cwd)} (⌘N)` : 'New thread (⌘N)'}
          aria-label="New thread"
          onClick={() => {
            if (!snap.turnActive) onOpen(null)
            store.newChat()
          }}
        >
          <T3Icon name="squarePen" size={14} />
        </button>
      </div>

      {/* Main Scrollable Thread Area */}
      <nav className="side__list t3-side__list" aria-label="Threads">
        {/* Pinned threads always float at top in both Projects and Inbox views */}
        {partitioned.pinned.length > 0 && (
          <section className="t3-side__section" aria-label="Pinned threads">
            <div className="t3-side__sec-head">
              <T3Icon name="pin" size={11} />
              <span>Pinned</span>
              <span className="t3-side__sec-count">{partitioned.pinned.length}</span>
            </div>
            <div className="t3-side__cards">
              {partitioned.pinned.map(row => renderThreadCard(row, 'inbox'))}
            </div>
          </section>
        )}

        {inbox.viewMode === 'inbox' ? (
          <>
            {partitioned.active.length > 0 && (
              <section className="t3-side__section" aria-label="Active threads">
                {partitioned.pinned.length > 0 && (
                  <div className="t3-side__sec-head">
                    <span>Active</span>
                    <span className="t3-side__sec-count">{partitioned.active.length}</span>
                  </div>
                )}
                <div className="t3-side__cards">
                  {partitioned.active.map(row => renderThreadCard(row, 'inbox'))}
                </div>
              </section>
            )}

            {partitioned.pinned.length === 0 && partitioned.active.length === 0 && (
              <div className="side__empty t3-side__empty">
                {needle ? (
                  <>No thread matches “{filter.trim()}”. <button type="button" className="linkish" onClick={() => setFilter('')}>Clear filter</button></>
                ) : activeScopeSet.size > 0 ? (
                  <>No active threads in selected projects. <button type="button" className="linkish" onClick={onClearProjectScope}>Show all</button></>
                ) : partitioned.settled.length > 0 || partitioned.snoozed.length > 0 ? (
                  'Inbox zero — all threads are snoozed or settled.'
                ) : snap.noWorkspace ? (
                  snap.storageScope?.startsWith('ssh:') ? 'Remote threads appear after connecting' : 'Choose a project folder to begin'
                ) : online ? (
                  'No threads yet — start a new thread above.'
                ) : (
                  'Connecting to the workspace runtime…'
                )}
              </div>
            )}
          </>
        ) : (
          /* Default: T3 Code Machine → Project Folder → Nested Threads */
          <div className="t3-side__machines" data-machine-grouped="true">
            {machineGroups.map(machine => {
              const machineKey = `machine:${machine.host}`
              const machineFolded = !needle && collapsed.has(machineKey)

              // Materialize visible active rows for each project under this machine
              const renderedProjects = machine.sections
                .map(section => {
                  const folderRows: T3SidebarRow[] = section.kind === 'context'
                    ? section.rows
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
                        }))
                    : section.rows
                        .filter(row => {
                          const running = row.status === 'working' || row.status === 'running'
                          if (pinnedIdSet.has(row.id)) return false
                          if (!running && settledIdSet.has(row.id)) return false
                          if (!running && isSnoozedActive(inbox.snoozed, row.id, now)) return false
                          return true
                        })
                        .map(row => ({
                          id: row.id,
                          key: row.key,
                          title: row.title,
                          ...(snap.snippets[row.id] !== undefined ? { snippet: snap.snippets[row.id] } : {}),
                          status: row.status,
                          age: row.age,
                          current: row.current,
                          turns: row.turns,
                          cwd: row.cwd || section.cwd,
                          ...(row.cwd === snap.cwd && snap.branch ? { branch: snap.branch } : {}),
                          provider: row.current ? activeProvider : 'xerxes',
                          host: section.host,
                          opensWindow: snap.turnActive && row.id !== snap.currentId,
                        }))
                  return { section, folderRows }
                })
                .filter(entry => !needle || entry.folderRows.length > 0)

              if (needle && renderedProjects.length === 0) return null

              const allMachineRows = renderedProjects.flatMap(entry => entry.folderRows)
              const machineDot = machineFolded ? folderStatusDot(allMachineRows) : null

              return (
                <section
                  key={machineKey}
                  className={`t3-machine${machine.isLocal ? ' t3-machine--local' : ' t3-machine--remote'}`}
                  data-machine-group={machine.host}
                  data-collapsed={machineFolded || undefined}
                >
                  <div className="t3-machine__head">
                    <button
                      type="button"
                      className="t3-machine__cap"
                      aria-expanded={!machineFolded}
                      aria-haspopup={!machine.isLocal ? 'menu' : undefined}
                      title={`${machine.host} · ${renderedProjects.length} ${renderedProjects.length === 1 ? 'project' : 'projects'}${!machine.isLocal ? ' (right-click for options)' : ''}`}
                      onClick={() => toggleCollapsed(machineKey)}
                      onContextMenu={event => {
                        if (machine.isLocal) return
                        event.preventDefault()
                        const first = renderedProjects[0]?.section
                        const alias = (first?.kind === 'context' ? first.context.alias : undefined) || machine.host
                        setFolderMenu({
                          kind: 'remote',
                          key: machineKey,
                          cwd: first?.cwd ?? '',
                          label: machine.host,
                          alias,
                          ...(first?.kind === 'context' ? { contextId: first.context.id } : {}),
                          x: event.clientX,
                          y: event.clientY,
                        })
                      }}
                    >
                      <T3Icon name={machine.isLocal ? 'monitor' : 'server'} size={12} className="t3-machine__ico" />
                      <span className="t3-machine__name">{machine.host}</span>
                      {machineDot && <span className="t3-folder__dot" data-tone={machineDot} aria-hidden="true" />}
                      {machineFolded && allMachineRows.length > 0 && (
                        <span className="t3-machine__count">{allMachineRows.length}</span>
                      )}
                      <T3Icon
                        name="chevronDown"
                        size={11}
                        className={`t3-folder__chev${machineFolded ? ' is-collapsed' : ''}`}
                      />
                    </button>
                    {machine.isLocal && (
                      <button
                        type="button"
                        className="t3-machine__action"
                        title="Add project folder on This Mac…"
                        aria-label="Add project on This Mac"
                        onClick={() => store.chooseWorkspace()}
                      >
                        <T3Icon name="plus" size={12} />
                      </button>
                    )}
                  </div>

                  {!machineFolded && (
                    <div className="t3-machine__projects">
                      {renderedProjects.map(({ section, folderRows }) => {
                        const folded = !needle && collapsed.has(section.key)
                        const all = Boolean(needle) || showAll.has(section.key)
                        const { shown, hidden } = limitRows(folderRows, SIDEBAR_SECTION_ROWS, all, snap.currentId)
                        const dot = folded ? folderStatusDot(folderRows) : null

                        const openFolderContextMenu = (x: number, y: number): void => {
                          if (section.kind === 'context' && section.context.remote) {
                            setFolderMenu({
                              kind: 'remote',
                              key: section.key,
                              cwd: section.cwd,
                              label: section.folder,
                              alias: section.context.alias || section.host,
                              contextId: section.context.id,
                              x,
                              y,
                            })
                          } else {
                            setFolderMenu({
                              kind: 'local',
                              key: section.key,
                              cwd: section.cwd,
                              label: section.folder,
                              x,
                              y,
                            })
                          }
                        }

                        return (
                          <div
                            key={section.key}
                            className={`t3-folder${section.cwd === snap.cwd ? ' is-home' : ''}${folderMenu?.key === section.key ? ' has-menu' : ''}`}
                            data-project-folder={section.cwd}
                            data-collapsed={folded || undefined}
                          >
                            <div className="t3-folder__head">
                              <button
                                type="button"
                                className="t3-folder__cap"
                                aria-expanded={!folded}
                                aria-haspopup="menu"
                                title={`${section.cwd} — right-click for workspace actions`}
                                onClick={() => toggleCollapsed(section.key)}
                                onContextMenu={event => {
                                  event.preventDefault()
                                  openFolderContextMenu(event.clientX, event.clientY)
                                }}
                                onKeyDown={event => {
                                  if (event.key === 'ContextMenu' || (event.key === 'F10' && event.shiftKey)) {
                                    event.preventDefault()
                                    const rect = event.currentTarget.getBoundingClientRect()
                                    openFolderContextMenu(rect.left, rect.bottom)
                                  }
                                }}
                              >
                                <ProjectMonogram name={section.folder} seed={section.cwd} size={16} />
                                <span className="t3-folder__name">{section.folder}</span>
                                {dot && <span className="t3-folder__dot" data-tone={dot} aria-hidden="true" />}
                                {folderRows.length > 0 && (
                                  <span className="t3-folder__count">{folderRows.length}</span>
                                )}
                                <T3Icon
                                  name="chevronDown"
                                  size={12}
                                  className={`t3-folder__chev${folded ? ' is-collapsed' : ''}`}
                                />
                              </button>
                              {section.kind === 'context' ? (
                                <button
                                  type="button"
                                  className="t3-folder__new"
                                  title={`Open ${section.label}`}
                                  aria-label={`Open ${section.label}`}
                                  onClick={() => void store.activateContext(section.context.id)}
                                >
                                  <T3Icon name="arrowRight" size={12} />
                                </button>
                              ) : (
                                <button
                                  type="button"
                                  className="t3-folder__new"
                                  disabled={!online}
                                  title={`New thread in ${section.folder}`}
                                  aria-label={`New thread in ${section.folder}`}
                                  onClick={() => {
                                    if (!snap.turnActive) onOpen(null)
                                    store.newChatIn(section.cwd)
                                  }}
                                >
                                  <T3Icon name="squarePen" size={12} />
                                </button>
                              )}
                            </div>

                            {!folded && shown.length > 0 && (
                              <div className="t3-folder__rows">
                                {shown.map(row => renderThreadCard(row, 'folder'))}
                                {!needle && (hidden > 0 || (all && folderRows.length > SIDEBAR_SECTION_ROWS)) && (
                                  <button type="button" className="t3-folder__more" onClick={() => toggleShowAll(section.key)}>
                                    {hidden > 0 ? `Show ${hidden} more` : 'Show less'}
                                  </button>
                                )}
                              </div>
                            )}
                          </div>
                        )
                      })}
                    </div>
                  )}
                </section>
              )
            })}

            {machineGroups.length === 0 && (
              <div className="side__empty t3-side__empty">
                {snap.noWorkspace
                  ? (snap.storageScope?.startsWith('ssh:') ? 'Remote threads appear after connecting' : 'Choose a project folder to begin')
                  : online
                    ? 'No projects yet — click + above to add a folder.'
                    : 'Connecting to the workspace runtime…'}
              </div>
            )}
          </div>
        )}

        {/* Bottom Collapsible Shelves (Snoozed + T3-style `Settled  13 >`) */}
        {(partitioned.snoozed.length > 0 || partitioned.settled.length > 0) && (
          <div className="t3-side__shelves">
            {partitioned.snoozed.length > 0 && (
              <section className="t3-shelf" aria-label="Snoozed threads">
                <button
                  type="button"
                  className="t3-shelf__toggle"
                  aria-expanded={inbox.snoozedOpen}
                  onClick={() => onToggleShelf('snoozed')}
                >
                  <span>Snoozed</span>
                  <span className="t3-shelf__right">
                    <span className="t3-shelf__count">{partitioned.snoozed.length}</span>
                    <T3Icon name={inbox.snoozedOpen ? 'chevronDown' : 'chevronRight'} size={12} />
                  </span>
                </button>
                {inbox.snoozedOpen && (
                  <div className="t3-shelf__body">
                    {partitioned.snoozed.map(row => renderThreadCard(row, 'snoozed'))}
                  </div>
                )}
              </section>
            )}

            {partitioned.settled.length > 0 && (
              <section className="t3-shelf" aria-label="Settled threads">
                <button
                  type="button"
                  className="t3-shelf__toggle"
                  aria-expanded={inbox.settledOpen}
                  onClick={() => onToggleShelf('settled')}
                >
                  <span>Settled</span>
                  <span className="t3-shelf__right">
                    <span className="t3-shelf__count">{partitioned.settled.length}</span>
                    <T3Icon name={inbox.settledOpen ? 'chevronDown' : 'chevronRight'} size={12} />
                  </span>
                </button>
                {inbox.settledOpen && (
                  <div className="t3-shelf__body t3-shelf__body--settled">
                    {shownSettled.map(row => renderThreadCard(row, 'settled'))}
                    {hiddenSettled > 0 && (
                      <button
                        type="button"
                        className="t3-folder__more"
                        onClick={() => setSettledLimit(limit => limit + 12)}
                      >
                        Show {hiddenSettled} more
                      </button>
                    )}
                  </div>
                )}
              </section>
            )}
          </div>
        )}
      </nav>

      {folderMenu && (
        <>
          <div
            className="backdrop backdrop--clear"
            onClick={() => setFolderMenu(null)}
            onContextMenu={event => { event.preventDefault(); setFolderMenu(null) }}
          />
          <div
            ref={folderMenuRef}
            className="menu t3-folder-menu"
            role="menu"
            aria-label={`Workspace actions for ${folderMenu.label}`}
            style={{ left: folderMenu.x, top: folderMenu.y }}
            onKeyDown={event => {
              if (event.key === 'ArrowDown' || event.key === 'ArrowUp' || event.key === 'Home' || event.key === 'End') {
                event.preventDefault()
                const buttons = Array.from(folderMenuRef.current?.querySelectorAll<HTMLButtonElement>('button:not(:disabled)') ?? [])
                const current = buttons.indexOf(document.activeElement as HTMLButtonElement)
                const next = event.key === 'Home'
                  ? 0
                  : event.key === 'End'
                    ? buttons.length - 1
                    : (current + (event.key === 'ArrowDown' ? 1 : buttons.length - 1)) % buttons.length
                buttons[next]?.focus()
              }
            }}
          >
            {folderMenu.kind === 'local' ? (
              <>
                <button
                  type="button"
                  role="menuitem"
                  className="menu__item"
                  disabled={!online}
                  onClick={() => {
                    const target = folderMenu.cwd
                    setFolderMenu(null)
                    if (!snap.turnActive) onOpen(null)
                    store.newChatIn(target)
                  }}
                >
                  <T3Icon name="squarePen" size={13} />
                  <span>New thread</span>
                </button>
                {typeof window !== 'undefined' && typeof window.xerxes?.openPath === 'function' && (
                  <button
                    type="button"
                    role="menuitem"
                    className="menu__item"
                    onClick={() => {
                      const target = folderMenu.cwd
                      setFolderMenu(null)
                      void window.xerxes.openPath?.(target)
                    }}
                  >
                    <T3Icon name="folderOpen" size={13} />
                    <span>Reveal in Finder</span>
                  </button>
                )}
                <button
                  type="button"
                  role="menuitem"
                  className="menu__item"
                  onClick={() => {
                    const target = folderMenu.cwd
                    setFolderMenu(null)
                    void copyToClipboard(target)
                  }}
                >
                  <T3Icon name="copy" size={13} />
                  <span>Copy path</span>
                </button>
                <div className="menu__sep" />
                <button
                  type="button"
                  role="menuitem"
                  className="menu__item menu__item--danger"
                  title="Remove from sidebar without deleting any files on disk"
                  onClick={() => removeLocalWorkspace(folderMenu.cwd, folderMenu.label)}
                >
                  <T3Icon name="trash" size={13} />
                  <span>Remove workspace</span>
                </button>
              </>
            ) : (
              <>
                {folderMenu.contextId !== undefined && (
                  <button
                    type="button"
                    role="menuitem"
                    className="menu__item"
                    onClick={() => {
                      const id = folderMenu.contextId!
                      setFolderMenu(null)
                      void store.activateContext(id)
                    }}
                  >
                    <T3Icon name="arrowRight" size={13} />
                    <span>Open workspace</span>
                  </button>
                )}
                {folderMenu.cwd && (
                  <button
                    type="button"
                    role="menuitem"
                    className="menu__item"
                    onClick={() => {
                      const target = folderMenu.cwd
                      setFolderMenu(null)
                      void copyToClipboard(target)
                    }}
                  >
                    <T3Icon name="copy" size={13} />
                    <span>Copy path</span>
                  </button>
                )}
                <div className="menu__sep" />
                <button
                  type="button"
                  role="menuitem"
                  className="menu__item menu__item--danger"
                  title="Remove saved SSH host from sidebar without modifying the remote machine"
                  onClick={() => removeRemoteWorkspace(folderMenu.alias)}
                >
                  <T3Icon name="trash" size={13} />
                  <span>Remove SSH host</span>
                </button>
              </>
            )}
          </div>
        </>
      )}

      {/* Undo Notice Banner for settle / snooze / pin / workspace removal */}
      {undo && (
        <div className="t3-undo-notice" role="status" aria-live="polite">
          <span className="t3-undo-notice__text">{undo.verb} “{undo.title}”</span>
          <button type="button" className="t3-undo-notice__btn" onClick={onUndo}>
            <T3Icon name="undo" size={12} />
            <span>Undo</span>
          </button>
          <button type="button" className="t3-undo-notice__close" aria-label="Dismiss undo notice" onClick={onDismissUndo}>
            <T3Icon name="close" size={12} />
          </button>
        </div>
      )}

      {/* Minimal T3 Icon Footer: Settings · Skills · Schedules · Usage · Supervisor · Agents on left; Runtime Status circle on far right */}
      <footer className="studio-side-bottom t3-side__foot">
        <div className="t3-side__foot-icons" role="group" aria-label="Workspace destinations">
          <button
            type="button"
            className="t3-icon-btn vibes-foot__settings"
            title="Settings (⌘,)"
            aria-label="Settings"
            onClick={() => store.openSettings()}
          >
            <T3Icon name="settingsGear" size={14} />
          </button>
          <button
            type="button"
            className={`t3-icon-btn${page === 'extensions' ? ' is-selected' : ''}`}
            aria-pressed={page === 'extensions'}
            disabled={!online}
            title="Skills & tools"
            aria-label="Skills & tools"
            onClick={() => onOpen(page === 'extensions' ? null : 'extensions')}
          >
            <T3Icon name="workflow" size={14} />
            <LearnedBadge snap={snap} />
          </button>
          <button
            type="button"
            className={`t3-icon-btn${page === 'schedules' ? ' is-selected' : ''}`}
            aria-pressed={page === 'schedules'}
            disabled={!online}
            title="Scheduled jobs"
            aria-label="Scheduled jobs"
            onClick={() => onOpen(page === 'schedules' ? null : 'schedules')}
          >
            <T3Icon name="alarmClock" size={14} />
          </button>
          <button
            type="button"
            className="t3-icon-btn"
            disabled={!online}
            title="Usage & provider plans"
            aria-label="Usage"
            onClick={() => onOpen('usage')}
          >
            <T3Icon name="barChart" size={14} />
          </button>
          <button
            type="button"
            className={`t3-icon-btn${page === 'xerxes' ? ' is-selected' : ''}`}
            aria-pressed={page === 'xerxes'}
            title="XERXES Supervisor"
            aria-label="XERXES Supervisor"
            onClick={() => onOpen(page === 'xerxes' ? null : 'xerxes')}
          >
            <T3Icon name="sparkles" size={14} />
          </button>
          <button
            type="button"
            className={`t3-icon-btn${page === 'agents' ? ' is-selected' : ''}`}
            aria-pressed={page === 'agents'}
            title="Agents"
            aria-label="Agents"
            onClick={() => onOpen(page === 'agents' ? null : 'agents')}
          >
            <T3Icon name="bot" size={14} />
          </button>
        </div>
        <div className="t3-side__foot-status">{runtimeStatus}</div>
      </footer>
    </aside>
  )
}
