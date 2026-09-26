// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The activity rail's status head — the one part that is always populated.
 *
 * The rail used to open on ten collapsed sections, so it looked identical
 * whether the agent was idle, running three subagents, or had failed, and
 * it spent a quarter of the window showing five words and five triangles.
 * This answers the questions a person actually has while an agent works:
 * what is it doing, how long has it been, how far through, how much
 * context is left, and what is this costing.
 *
 * Everything here is read from the snapshot the store already keeps. There
 * is deliberately no percentage bar for the turn itself — the wire carries
 * no completion signal, and the todo count is the only honest progress the
 * daemon actually reports.
 */

import type { ReactElement } from 'react'

import { store, type Snapshot } from './store.js'
import { Icon } from './Icon.js'
import { toolPhrase } from './activityPhrase.js'
import { SessionDiagnostics } from './SessionDiagnostics.js'

/** 43 → "43s", 255 → "4m 15s". Matches the composer's clock. */
export function elapsedOf(seconds: number): string {
  if (seconds < 60) return `${seconds}s`
  const minutes = Math.floor(seconds / 60)
  return `${minutes}m ${seconds % 60}s`
}

const compact = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 })

export interface RailState {
  readonly tone: 'needs' | 'working' | 'failed' | 'plan' | 'idle'
  readonly label: string
}

export function railStateOf(snap: Snapshot): RailState {
  if (snap.approval || snap.question) return { tone: 'needs', label: 'Needs you' }
  if (snap.turnActive) return { tone: 'working', label: 'Working' }
  if (snap.submissionPending) return { tone: 'working', label: 'Starting' }
  if (snap.failed) return { tone: 'failed', label: 'Failed' }
  if (snap.planMode) return { tone: 'plan', label: 'Planning' }
  return { tone: 'idle', label: 'Idle' }
}

/**
 * The tool call the agent is inside right now, as one line. Reads the same
 * fold the transcript renders, so it can never disagree with the feed.
 */
export function currentActionOf(snap: Snapshot): { verb: string; detail: string } | null {
  if (!snap.turnActive) return null
  for (let index = snap.blocks.length - 1; index >= 0; index -= 1) {
    const block = snap.blocks[index]
    if (!block || block.kind !== 'tools') continue
    const running = block.items.find(item => item.state === 'working')
    // Same phrase as the feed header; the unabridged target rides the tooltip.
    if (running) return { verb: toolPhrase(running), detail: running.path || running.arg || '' }
  }
  return null
}

export function RailStatus({ snap }: { snap: Snapshot }): ReactElement {
  const state = railStateOf(snap)
  const action = currentActionOf(snap)
  const todos = snap.todos ?? []
  const doneTodos = todos.filter(todo => todo.status === 'completed').length
  // Only once the runtime has actually reported usage. Defaulting a null
  // to 0 drew a full-width empty gauge reading "0 / 400K", which claims
  // the context is untouched rather than admitting it is unknown.
  const context = snap.contextMax && snap.contextMax > 0 && snap.contextTokens != null
    ? Math.min(1, snap.contextTokens / snap.contextMax)
    : null
  const stoppable = snap.turnActive || snap.submissionPending
  return (
    <section className="railcard railcard--status railstatus" data-tone={state.tone} aria-label="Task status">
      <div className="railstatus__head">
        {/* A spinner only while work is actually in flight. Every other
            state is a resting one, and a dot that never moves says that
            more honestly than a spinner frozen mid-turn would. */}
        {state.tone === 'working'
          ? <span className="railstatus__spinner" aria-hidden="true"><Icon name="spinner" size={13} /></span>
          : <span className="railstatus__dot" aria-hidden="true" />}
        <strong>{state.label}</strong>
        {stoppable && <span className="railstatus__clock">{elapsedOf(snap.turnSeconds)}</span>}
        {snap.costUsd != null && snap.costUsd > 0 && !stoppable && (
          <span className="railstatus__cost">${snap.costUsd.toFixed(snap.costUsd < 0.01 ? 4 : 2)}</span>
        )}
        {stoppable && <button className="railstatus__stop" onClick={() => store.cancel()}>Stop</button>}
      </div>

      {action && (
        <p className="railstatus__action" title={action.detail || action.verb}>
          <Icon name="tools" size={13} />
          <span>{action.verb}</span>
        </p>
      )}

      {(todos.length > 0 || context !== null) && (
        <dl className="railstatus__facts">
          {todos.length > 0 && (
            <div>
              <dt>Steps</dt>
              <dd>{doneTodos} of {todos.length}</dd>
            </div>
          )}
          {context !== null && (
            <div>
              {/* Labelled on its own row. Sitting unlabelled under "Steps"
                  the bar read as step progress, which it never was. */}
              <dt>Context</dt>
              <dd>
                <span className="railstatus__track" role="img" aria-label={`${Math.round(context * 100)} percent of the context window used`}>
                  <span className="railstatus__fill" data-full={context > 0.85 || undefined} style={{ width: `${Math.round(context * 100)}%` }} />
                </span>
                {compact.format(snap.contextTokens ?? 0)} / {compact.format(snap.contextMax ?? 0)}
              </dd>
            </div>
          )}
        </dl>
      )}
      {/* This task's numbers live with the task; the Usage tab is about
          accounts (plans, keys, limits), not one session. */}
      <SessionDiagnostics snap={snap} />
    </section>
  )
}
