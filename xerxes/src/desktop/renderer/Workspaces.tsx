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

import { Markdown } from './markdown.js'
import { isPlanReview, store, type Snapshot } from './store.js'
import type { DiffFile, LogEntry, PlanState } from './types.js'

// ── Changes ─────────────────────────────────────────────────────────────

export function ChangesTab({ snap }: { snap: Snapshot }): ReactElement {
  const files = snap.changes
  const totals = useMemo(() => ({
    adds: files.reduce((sum, file) => sum + file.adds, 0),
    dels: files.reduce((sum, file) => sum + file.dels, 0),
  }), [files])
  const kept = snap.changesKept

  if (files.length === 0) {
    return (
      <div className="tabempty">
        <div className="tabempty__mark">⎇</div>
        <h1>No changes yet</h1>
        <p>File edits the agents make land here as reviewable diffs — per-file +/− and the exact hunks, folded from the edit calls as they stream.</p>
      </div>
    )
  }
  return (
    <div className="changes">
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
      {files.map(file => <DiffFileCard key={file.path} file={file} />)}
      <p className="changes__note">
        Folded live from <code>FileEditTool</code> / <code>WriteFile</code> calls — the daemon owns the worktree; nothing here rewrites it.
      </p>
    </div>
  )
}

function DiffFileCard({ file }: { file: DiffFile }): ReactElement {
  const [open, setOpen] = useState(true)
  return (
    <div className="dfile">
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

  if (!plan) {
    return (
      <div className="tabempty">
        <div className="tabempty__mark">⏸</div>
        <h1>{snap.planMode ? 'Planning in progress' : 'No plan yet'}</h1>
        <p>
          {snap.planMode
            ? 'Plan mode is on — the agent explores and proposes a checklist here before touching anything. Steer it from the composer.'
            : 'Toggle the ⏸ plan chip in the composer (or ⌘K → plan mode) and the proposed checklist will be captured here for review.'}
        </p>
      </div>
    )
  }
  return (
    <div className="plan plan--tab">
      <div className="plan__head">
        <span className="plan__title">Working plan</span>
        <span className="plan__meta">
          {snap.planMode ? '⏸ plan mode' : 'approved / executing'} · turn {plan.turn}
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
      <div className="tabempty">
        <div className="tabempty__mark">≡</div>
        <h1>Event stream</h1>
        <p>Every daemon event — wire order, one line each — as it arrived. Streaming a turn fills this ring (last 400 events).</p>
      </div>
    )
  }
  return (
    <div className="loglist" ref={ref} onScroll={onScroll}>
      {entries.map(entry => (
        <div key={entry.id} className="logrow">
          <span className="logrow__turn">t{entry.turn}</span>
          <span className="logrow__type">{entry.type}</span>
          <span className="logrow__sum">{entry.summary}</span>
        </div>
      ))}
    </div>
  )
}
