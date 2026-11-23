// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The tabbed workspace surfaces under the session header: Changes (diff
 * review folded from edit-family tool calls), Plan (the checklist the agent
 * proposed, live while plan mode is on), and Log (the raw daemon event
 * stream). Activity — the execution feed — lives in App.tsx because it
 * shares the transcript fold with the composer and the cards.
 */

import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react'

import { useDesktopNavigation } from './DesktopPanels.js'
import { Markdown } from './markdown.js'
import { TodoList } from './Todos.js'
import { isPlanReview, store, type Snapshot } from './store.js'
import type { DiffFile, LogEntry, PlanState } from './types.js'
import { Icon } from './Icon.js'

// ── Shared return bar ───────────────────────────────────────────────────

function WorkspaceReturnBar({ title, detail }: { readonly title: string; readonly detail?: string }): ReactElement {
  return (
    <div className="workspace-return">
      <button
        type="button"
        className="workspace-return__btn"
        aria-label="Back to conversation"
        title="Back to conversation (Esc)"
        onClick={() => store.setTab('activity')}
      >
        <span className="workspace-return__arrow" aria-hidden="true">←</span>
        <span>Back to conversation</span>
      </button>
      <span className="workspace-return__title">{title}</span>
      {detail && <span className="workspace-return__meta">{detail}</span>}
      <kbd className="workspace-return__hint" aria-hidden="true">Esc</kbd>
    </div>
  )
}

// ── Changes ─────────────────────────────────────────────────────────────

export function ChangesTab({ snap }: { snap: Snapshot }): ReactElement {
  const navigate = useDesktopNavigation()
  const allChanges = <button className="btn" onClick={() => navigate('review')}>Review working tree, including new files</button>
  const files = snap.changes
  const totals = useMemo(() => ({
    adds: files.reduce((sum, file) => sum + file.adds, 0),
    dels: files.reduce((sum, file) => sum + file.dels, 0),
  }), [files])
  const kept = snap.changesKept

  if (files.length === 0) {
    return (
      <div className="workspace-shell">
        <WorkspaceReturnBar title="Session edits" />
        <div className="tabempty">
          <div className="tabempty__mark">⎇</div>
          <h1>No recorded edits yet</h1>
          {allChanges}
          <p>File edits the agents make land here as reviewable diffs — per-file +/− and the exact hunks, folded from the edit calls as they stream.</p>
        </div>
      </div>
    )
  }
  return (
    <div className="workspace-shell">
      <WorkspaceReturnBar title="Session edits" detail={`${files.length} file${files.length === 1 ? '' : 's'}`} />
      <div className="changes">
        {allChanges}
        <div className="changes__bar">
          <span className="changes__stat">
            {files.length} file{files.length === 1 ? '' : 's'} · <span className="add">+{totals.adds}</span> <span className="del">−{totals.dels}</span>
          </span>
          <span style={{ flex: '1 1 auto' }} />
          {kept
            ? <span className="changes__stat">kept — tracking continues</span>
            : <button className="btn" onClick={() => store.ackChanges()}>Keep all</button>}
          <button
            className="btn btn--ghost"
            title="Reverse this session's recorded edits across every file"
            onClick={() => { void store.undoChanges(null) }}
          >
            Undo all
          </button>
        </div>
        {files.map(file => <DiffFileCard key={file.path} file={file} focusAt={snap.changesFocus?.path === file.path ? snap.changesFocus.at : undefined} />)}
        <p className="changes__note">
          Built from the file edits this task made. Your working tree is the source of truth — undo here asks the runtime to put a file back.
        </p>
      </div>
    </div>
  )
}

/**
 * One file's recorded edits. `focusAt` is set when the rail's Touched card
 * asked for this file: the card opens, scrolls into view and is marked for a
 * moment, so the click visibly lands on the file that was chosen.
 */
function DiffFileCard({ file, focusAt }: { file: DiffFile; focusAt?: number | undefined }): ReactElement {
  const [open, setOpen] = useState(true)
  const [focused, setFocused] = useState(false)
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (focusAt === undefined) return
    setOpen(true)
    setFocused(true)
    ref.current?.scrollIntoView({ block: 'start', behavior: 'smooth' })
    const timer = setTimeout(() => setFocused(false), 1_600)
    return () => clearTimeout(timer)
  }, [focusAt])
  return (
    <div className="dfile" ref={ref} data-focused={focused || undefined}>
      <div className="dfile__head">
        <span className="dot dot--done" />
        <button className="dfile__path" onClick={() => setOpen(value => !value)} title="Toggle hunks">
          {file.path}
          {file.isNew ? <span className="chipbtn dfile__new">new</span> : null}
        </button>
        <span className="dfile__stat">
          <span className="add">+{file.adds}</span> <span className="del">−{file.dels}</span>
        </span>
        <button
          className="chipbtn"
          title="Reverse this session's recorded edits to this file"
          onClick={() => { void store.undoChanges(file.path) }}
        >
          undo
        </button>
      </div>
      {open && file.hunks.map((line, index) => (
        <div key={index} className={`dline dline--${line.kind}`}>
          <span className="no">{line.kind === 'hunk' ? '…' : line.kind === 'del' ? '-' : line.kind === 'add' ? '+' : ' '}</span>
          <span className="tx">{line.text || ' '}</span>
        </div>
      ))}
    </div>
  )
}

// ── Plan ────────────────────────────────────────────────────────────────

export function PlanTab({ snap }: { snap: Snapshot }): ReactElement {
  const plan: PlanState | null = snap.plan
  // Only a PLAN review points back at Activity — a generic clarification
  // question is not a plan event.
  const review = snap.question && isPlanReview(snap.question) ? snap.question : null
  const items = plan?.items ?? []
  const done = items.filter(item => item.done).length
  const todos = snap.todos ?? []
  const doneTodos = todos.filter(item => item.status === 'completed').length
  const todoDetail = todos.length > 0 ? `${doneTodos}/${todos.length} done` : items.length > 0 ? `${done}/${items.length} done` : undefined

  if (!plan) {
    if (snap.todos != null) {
      return (
        <div className="workspace-shell">
          <WorkspaceReturnBar title="Plan & to-dos" {...(todoDetail ? { detail: todoDetail } : {})} />
          <div className="plan plan--tab"><TodoList items={snap.todos} /></div>
        </div>
      )
    }
    return (
      <div className="workspace-shell">
        <WorkspaceReturnBar title="Plan & to-dos" />
        <div className="tabempty">
          <div className="tabempty__mark"><Icon name="pause" size={22} /></div>
          <h1>{snap.planMode ? 'Planning in progress' : 'No plan yet'}</h1>
          <p>
            {snap.planMode
              ? 'Plan mode is on — the agent explores and proposes a checklist here before touching anything. Steer it from the composer.'
              : 'Switch the composer’s “Work directly” button to plan first (or ⌘K → plan mode), and the checklist the agent proposes is captured here for review.'}
          </p>
        </div>
      </div>
    )
  }
  return (
    <div className="workspace-shell">
      <WorkspaceReturnBar title="Plan & to-dos" {...(todoDetail ? { detail: todoDetail } : {})} />
      <div className="plan plan--tab">
        {snap.todos != null && <TodoList items={snap.todos} />}
        <div className="plan__head">
          <span className="plan__title">Working plan</span>
          <span className="plan__meta">
            {snap.planMode ? <><Icon name="pause" size={11} /> plan mode</> : 'approved / executing'} · turn {plan.turn}
            {items.length ? ` · ${done}/${items.length} done` : ''}
          </span>
        </div>
        {items.length > 0 && (
          <div className="pl-list">
            {items.map((item, index) => (
              <div key={index} className={`pl-item${item.done ? ' done' : ''}`}>
                <span className="pl-box" />
                <span className="pl-txt">{item.text}</span>
              </div>
            ))}
          </div>
        )}
        <div className="plan__doc">
          <Markdown text={plan.markdown} />
        </div>
        {review && (
          <div className="plan__actions">
            <span className="appr-policy">a plan review is pending in Activity — answer it there</span>
          </div>
        )}
      </div>
    </div>
  )
}

// ── Log ─────────────────────────────────────────────────────────────────

export function LogTab({ snap }: { snap: Snapshot }): ReactElement {
  const ref = useRef<HTMLDivElement>(null)
  // Stick to the newest line only while the human is already at the bottom —
  // the 1s turn tick re-renders this list, and an unconditional yank would
  // make scrolling up physically impossible mid-turn.
  const pinned = useRef(true)
  const onScroll = (): void => {
    const el = ref.current
    if (el) pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  }
  useEffect(() => {
    const el = ref.current
    if (el && pinned.current) el.scrollTop = el.scrollHeight
  })
  const entries: readonly LogEntry[] = snap.log
  if (entries.length === 0) {
    return (
      <div className="workspace-shell workspace-shell--log">
        <WorkspaceReturnBar title="Event log" />
        <div className="tabempty">
          <div className="tabempty__mark">≡</div>
          <h1>Event log</h1>
          <p>A diagnostic trace of what the runtime reported, newest last, one line per event. It fills while a task runs and keeps the most recent 400.</p>
        </div>
      </div>
    )
  }
  return (
    <div className="workspace-shell workspace-shell--log">
      <WorkspaceReturnBar title="Event log" detail={`${entries.length}`} />
      <div className="loglist" ref={ref} onScroll={onScroll}>
        {entries.map(entry => (
          <div key={entry.id} className="logrow">
            <span className="logrow__turn">t{entry.turn}</span>
            <span className="logrow__type">{entry.type}</span>
            <span className="logrow__sum">{entry.summary}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
