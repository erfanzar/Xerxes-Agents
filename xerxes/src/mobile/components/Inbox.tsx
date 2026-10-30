// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

import { plural, relativeTime } from '../../client/format.js'
import { inboxCounts, inboxTasks, presenceFor, workspaceAvailable, type InboxStatusFilter } from '../../client/hubStore.js'
import type { TaskSummary } from '../../client/hubTypes.js'
import { useHubState, useNow, useServices } from '../context.js'
import { taskRoute } from '../routes.js'
import { groupTasksByMachine, viewForTask, workspaceLabel } from '../viewModel.js'
import { AppLink } from './AppLink.js'
import { StatusBadge } from './StatusBadge.js'

const FILTERS: readonly (readonly [InboxStatusFilter, string])[] = [
  ['all', 'All'],
  ['needs_you', 'Needs you'],
  ['working', 'Working'],
  ['failed', 'Failed'],
  ['idle', 'Idle'],
]

function TaskCard({ task }: { task: TaskSummary }): ReactElement {
  const state = useHubState()
  const now = useNow()
  const available = workspaceAvailable(state.workspaces[task.workspaceId])
  const view = viewForTask(state, task)
  const chips = presenceFor(state, task.workspaceId, task.id)
  const needsYou = available && (task.pendingCount > 0 || task.status === 'needs_you')
  const body = (
    <>
      <span className="card-top">
        <span className="card-workspace">{workspaceLabel(state, task.workspaceId)}</span>
        {needsYou ? <span className="badge badge-attention">Needs you</span> : null}
      </span>
      <h3 className="card-title">{task.title}</h3>
      <StatusBadge view={view} />
      {task.goalSummary === '' ? null : <p className="card-goal">Goal: {task.goalSummary}</p>}
      <span className="card-bottom">
        <time dateTime={task.updatedAt}>Updated {relativeTime(task.updatedAt, now)}</time>
        {task.agents.active > 0 ? <span>{plural(task.agents.active, 'agent')} active</span> : null}
      </span>
      {chips.length === 0 ? null : (
        <ul className="chips" aria-label="People on this task">
          {chips.map(chip => (
            <li key={chip.userId} className="chip">{chip.name}{chip.state === 'typing' ? ' · typing' : ''}{chip.devices > 1 ? ` · ${chip.devices} devices` : ''}</li>
          ))}
        </ul>
      )}
    </>
  )
  return (
    <li className={`card${needsYou ? ' card-attention' : ''}${available ? '' : ' card-unavailable'}`}>
      {available
        ? <AppLink route={taskRoute(task.workspaceId, task.id)} className="card-link">{body}</AppLink>
        : <div className="card-link" aria-disabled="true">{body}<p className="card-note">Unavailable. This workspace is offline, so its tasks cannot be opened.</p></div>}
    </li>
  )
}

export interface InboxFilterChoice {
  readonly workspaceId?: string | undefined
  readonly status: InboxStatusFilter
}

/**
 * The Inbox filter outlives the Inbox screen: opening a task and coming back
 * keeps it. Keyed by the app's services object, so each app instance (and
 * each test) has its own.
 */
const FILTER_MEMORY = new WeakMap<object, InboxFilterChoice>()

export function rememberedInboxFilter(owner: object): InboxFilterChoice {
  return FILTER_MEMORY.get(owner) ?? { status: 'all' }
}

export function rememberInboxFilter(owner: object, choice: InboxFilterChoice): void {
  FILTER_MEMORY.set(owner, choice)
}

export function InboxTab(): ReactElement {
  const state = useHubState()
  const services = useServices()
  const { controller } = services
  // The scroll memory is created once per app; it anchors the filter memory too.
  const owner = services.scroll
  const [choice, setChoice] = useState<InboxFilterChoice>(() => rememberedInboxFilter(owner))
  const choose = (next: InboxFilterChoice): void => {
    rememberInboxFilter(owner, next)
    setChoice(next)
  }
  const workspaceId = choice.workspaceId
  const status = choice.status
  const setWorkspaceId = (id: string | undefined): void => choose({ ...choice, workspaceId: id })
  const setStatus = (value: InboxStatusFilter): void => choose({ ...choice, status: value })
  const tasks = inboxTasks(state, { ...(workspaceId === undefined ? {} : { workspaceId }), status })
  const counts = inboxCounts(state)
  const offline = state.workspaceOrder.map(id => state.workspaces[id]).filter(w => w !== undefined && !workspaceAvailable(w))

  return (
    <section aria-labelledby="inbox-title">
      <h1 id="inbox-title" className="page-title">Inbox</h1>
      <p className="summary">
        {counts.needsYou > 0 ? <strong>{plural(counts.needsYou, 'task')} {counts.needsYou === 1 ? 'needs' : 'need'} you</strong> : 'Nothing needs you'}
        {counts.working > 0 ? ` · ${counts.working} working` : ''}
        {counts.failed > 0 ? ` · ${counts.failed} failed` : ''}
      </p>

      {offline.length === 0 ? null : (
        <ul className="notes" aria-label="Unavailable workspaces">
          {offline.map(w => (w === undefined ? null : <li key={w.id}>{w.label} is unavailable ({w.health === 'needs_operator' ? 'needs attention on the hub' : 'offline'}).</li>))}
        </ul>
      )}

      <div className="filters" role="group" aria-label="Filter tasks">
        {state.workspaceOrder.length > 1 ? (
          <label className="field">
            <span>Workspace</span>
            <select value={workspaceId ?? ''} onChange={event => setWorkspaceId(event.target.value === '' ? undefined : event.target.value)}>
              <option value="">All workspaces</option>
              {state.workspaceOrder.map(id => <option key={id} value={id}>{workspaceLabel(state, id)}</option>)}
            </select>
          </label>
        ) : null}
        <div className="segments">
          {FILTERS.map(([value, label]) => (
            <button key={value} type="button" className="btn btn-segment" aria-pressed={status === value} onClick={() => setStatus(value)}>{label}</button>
          ))}
        </div>
      </div>

      {state.inboxError !== undefined ? (
        <div role="alert" className="notice is-error">
          <p>Could not load tasks: {state.inboxError}</p>
          <button type="button" className="btn" onClick={() => { void controller.refreshInbox() }}>Try again</button>
        </div>
      ) : null}

      {!state.inboxLoaded && state.inboxError === undefined ? <p aria-busy="true">Loading tasks…</p> : null}
      {state.inboxLoaded && tasks.length === 0 && state.workspaceOrder.length === 0 ? (
        <div className="empty-state">
          <h2>No workspaces connected</h2>
          <p>The Hub is running, but no machine is linked to it yet, so there are no tasks to show. In the Xerxes desktop app, open Settings → Hub &amp; phone and choose “Link this Mac”. Its projects then appear here on their own.</p>
        </div>
      ) : null}
      {state.inboxLoaded && tasks.length === 0 && state.workspaceOrder.length > 0 ? (
        status === 'all' && workspaceId === undefined
          ? <div className="empty-state"><h2>No tasks yet</h2><p>No open chats on {state.workspaceOrder.length === 1 ? workspaceLabel(state, state.workspaceOrder[0]!) : `${state.workspaceOrder.length} workspaces`}. Start one from the desktop or the Xerxes tab and it appears here.</p></div>
          : <p className="empty">No tasks match these filters.</p>
      ) : null}
      {groupTasksByMachine(state, tasks).map(group => (
        <section key={group.key} aria-label={group.label === undefined ? undefined : `Machine ${group.label}`}>
          {group.label === undefined ? null : <h2 className="section-title">{group.label}</h2>}
          <ul className="cards">{group.tasks.map(task => <TaskCard key={`${task.workspaceId}/${task.id}`} task={task} />)}</ul>
        </section>
      ))}
    </section>
  )
}
