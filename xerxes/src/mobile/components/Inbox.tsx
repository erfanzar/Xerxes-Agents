// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

import { plural, relativeTime } from '../../client/format.js'
import { canMutate, inboxCounts, inboxTasks, presenceFor, workspaceAvailable, type InboxStatusFilter } from '../../client/hubStore.js'
import type { TaskSummary } from '../../client/hubTypes.js'
import { useHubState, useNow, useServices } from '../context.js'
import { taskRoute } from '../routes.js'
import { groupTasksByMachine, viewForTask, workspaceLabel } from '../viewModel.js'
import { AppLink } from './AppLink.js'
import { Sheet } from './Sheet.js'
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
        <span className="card-time-group">
          <time dateTime={task.updatedAt}>{relativeTime(task.updatedAt, now)}</time>
          {available ? <span className="card-chevron" aria-hidden="true">›</span> : null}
        </span>
      </span>
      <h3 className="card-title">{task.title}</h3>
      {task.goalSummary === '' ? null : <p className="card-goal">{task.goalSummary}</p>}
      <span className="card-bottom">
        <StatusBadge view={view} />
        {task.agents.active > 0 ? <span className="card-agents">{plural(task.agents.active, 'agent')} active</span> : null}
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

function NewTaskSheet({
  workspaceIds,
  initialWorkspaceId,
  onClose,
}: {
  readonly workspaceIds: readonly string[]
  readonly initialWorkspaceId: string
  readonly onClose: () => void
}): ReactElement {
  const state = useHubState()
  const { controller, router } = useServices()
  const [selectedWs, setSelectedWs] = useState(initialWorkspaceId)
  const [prompt, setPrompt] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | undefined>(undefined)

  const submit = async (): Promise<void> => {
    if (busy || !selectedWs || !controller.createTask) return
    setBusy(true)
    setError(undefined)
    try {
      const result = await controller.createTask(selectedWs, prompt)
      if (!result.ok) {
        setError(result.message)
        return
      }
      onClose()
      router.navigate(taskRoute(result.workspaceId, result.sessionId))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Sheet title="New task" onClose={onClose}>
      <form className="new-task-form" onSubmit={event => { event.preventDefault(); void submit() }}>
        <label className="field">
          <span>Workspace</span>
          <select value={selectedWs} disabled={busy} onChange={event => setSelectedWs(event.target.value)}>
            {workspaceIds.map(id => <option key={id} value={id}>{workspaceLabel(state, id)}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Opening prompt (optional)</span>
          <textarea
            rows={3}
            value={prompt}
            disabled={busy}
            placeholder="Describe what you want Xerxes to work on…"
            onChange={event => setPrompt(event.target.value)}
          />
        </label>
        {error !== undefined ? <p role="alert" className="field-error">{error}</p> : null}
        <div className="approval-actions">
          <button type="submit" className="btn btn-primary" disabled={busy || !selectedWs}>
            {busy ? 'Starting…' : 'Start task'}
          </button>
          <button type="button" className="btn" disabled={busy} onClick={onClose}>Cancel</button>
        </div>
      </form>
    </Sheet>
  )
}

export function InboxTab(): ReactElement {
  const state = useHubState()
  const services = useServices()
  const { controller } = services
  const owner = services.scroll
  const [choice, setChoice] = useState<InboxFilterChoice>(() => rememberedInboxFilter(owner))
  const [query, setQuery] = useState('')
  const [newTaskOpen, setNewTaskOpen] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const choose = (next: InboxFilterChoice): void => {
    rememberInboxFilter(owner, next)
    setChoice(next)
  }
  const workspaceId = choice.workspaceId
  const status = choice.status
  const setWorkspaceId = (id: string | undefined): void => choose({ ...choice, workspaceId: id })
  const setStatus = (value: InboxStatusFilter): void => choose({ ...choice, status: value })
  const allFiltered = inboxTasks(state, { ...(workspaceId === undefined ? {} : { workspaceId }), status })
  const needle = query.trim().toLowerCase()
  const tasks = needle === ''
    ? allFiltered
    : allFiltered.filter(task => (
        task.title.toLowerCase().includes(needle)
        || task.goalSummary.toLowerCase().includes(needle)
        || workspaceLabel(state, task.workspaceId).toLowerCase().includes(needle)
      ))
  const counts = inboxCounts(state)
  const offline = state.workspaceOrder.map(id => state.workspaces[id]).filter(w => w !== undefined && !workspaceAvailable(w))
  const writableWorkspaces = state.workspaceOrder.filter(id => canMutate(state, id))
  const canStartTask = writableWorkspaces.length > 0 && controller.createTask !== undefined
  const defaultNewWs = (workspaceId !== undefined && writableWorkspaces.includes(workspaceId) ? workspaceId : writableWorkspaces[0]) ?? ''
  const live = state.connection.state === 'live'
  const showSearch = Object.keys(state.tasks).length > 0 || query !== ''

  const refresh = async (): Promise<void> => {
    if (refreshing || !live) return
    setRefreshing(true)
    try { await controller.refreshInbox() }
    finally { setRefreshing(false) }
  }

  const filterCount = (value: InboxStatusFilter): number => {
    if (value === 'needs_you') return counts.needsYou
    if (value === 'working') return counts.working
    if (value === 'failed') return counts.failed
    return 0
  }

  return (
    <section aria-labelledby="inbox-title">
      <div className="page-head">
        <h1 id="inbox-title" className="page-title">Inbox</h1>
        <div className="page-actions">
          <button type="button" className="btn" disabled={!live || refreshing} onClick={() => { void refresh() }}>
            <svg className="btn-icon" viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M13.5 8a5.5 5.5 0 1 1-1.4-3.7M13.5 2.5v2.8h-2.8" />
            </svg>
            <span>{refreshing ? 'Refreshing…' : 'Refresh'}</span>
          </button>
          {canStartTask ? (
            <button type="button" className="btn btn-primary" aria-haspopup="dialog" onClick={() => setNewTaskOpen(true)}>
              <svg className="btn-icon" viewBox="0 0 16 16" width="13" height="13" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
                <path d="M8 3.2v9.6M3.2 8h9.6" />
              </svg>
              <span>New task</span>
            </button>
          ) : null}
        </div>
      </div>
      <p className="summary inbox-summary">
        {counts.needsYou > 0 ? <strong>{plural(counts.needsYou, 'task')} {counts.needsYou === 1 ? 'needs' : 'need'} you</strong> : 'All caught up'}
        {counts.working > 0 ? ` · ${counts.working} working` : ''}
        {counts.failed > 0 ? ` · ${counts.failed} failed` : ''}
      </p>

      {offline.length === 0 ? null : (
        <ul className="notes inbox-alerts" aria-label="Unavailable workspaces">
          {offline.map(w => (w === undefined ? null : <li key={w.id}>{w.label} is unavailable ({w.health === 'needs_operator' ? 'needs attention on the hub' : 'offline'}).</li>))}
        </ul>
      )}

      {showSearch || state.workspaceOrder.length > 1 ? (
        <div className="inbox-controls">
          {showSearch ? (
            <div className="inbox-search">
              <svg className="inbox-search-icon" viewBox="0 0 16 16" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.65" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="7.2" cy="7.2" r="4.7" />
                <path d="M13.5 13.5l-2.9-2.9" />
              </svg>
              <input
                type="search"
                aria-label="Search tasks"
                placeholder="Search…"
                value={query}
                onChange={event => setQuery(event.target.value)}
              />
              {query !== '' ? (
                <button type="button" className="btn btn-quiet inbox-search-clear" onClick={() => setQuery('')}>Clear</button>
              ) : null}
            </div>
          ) : null}
          {state.workspaceOrder.length > 1 ? (
            <label className="inbox-ws-picker">
              <span className="visually-hidden">Workspace</span>
              <select value={workspaceId ?? ''} onChange={event => setWorkspaceId(event.target.value === '' ? undefined : event.target.value)}>
                <option value="">All workspaces</option>
                {state.workspaceOrder.map(id => <option key={id} value={id}>{workspaceLabel(state, id)}</option>)}
              </select>
            </label>
          ) : null}
        </div>
      ) : null}

      <div className="filters" role="group" aria-label="Filter tasks">
        <div className="segments">
          {FILTERS.map(([value, label]) => {
            const count = filterCount(value)
            return (
              <button key={value} type="button" className="btn btn-segment" aria-pressed={status === value} onClick={() => setStatus(value)}>
                <span>{label}</span>
                {count > 0 ? <span className="segment-count">{count}</span> : null}
              </button>
            )
          })}
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
        status === 'all' && workspaceId === undefined && needle === ''
          ? <div className="empty-state"><h2>No tasks yet</h2><p>No open chats on {state.workspaceOrder.length === 1 ? workspaceLabel(state, state.workspaceOrder[0]!) : `${state.workspaceOrder.length} workspaces`}. Start one right here or from the desktop and it appears here.</p></div>
          : <p className="empty">No tasks match these filters.</p>
      ) : null}
      {groupTasksByMachine(state, tasks).map(group => (
        <section key={group.key} aria-label={group.label === undefined ? undefined : `Machine ${group.label}`}>
          {group.label === undefined ? null : <h2 className="section-title">{group.label}</h2>}
          <ul className="cards">{group.tasks.map(task => <TaskCard key={`${task.workspaceId}/${task.id}`} task={task} />)}</ul>
        </section>
      ))}
      {newTaskOpen && canStartTask ? (
        <NewTaskSheet workspaceIds={writableWorkspaces} initialWorkspaceId={defaultNewWs} onClose={() => setNewTaskOpen(false)} />
      ) : null}
    </section>
  )
}
