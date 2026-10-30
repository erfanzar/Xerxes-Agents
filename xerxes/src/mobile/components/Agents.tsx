// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import { plural, relativeTime } from '../../client/format.js'
import type { AgentRow, AgentStep, TodoItem } from '../../client/taskTypes.js'
import { Markdown } from './Markdown.js'
import { Sheet } from './Sheet.js'
import { ToolRow } from './ToolRow.js'
import { agentStatus, argsSummary } from './toolView.js'

type LoadPhase = 'idle' | 'loading' | 'ready' | 'error'

/** What the agent list shows (`selected.agents`). */
export interface AgentListView {
  readonly phase: LoadPhase
  readonly rows: readonly AgentRow[]
  readonly error?: string | undefined
}

/** What one agent's step view shows (`selected.agentDetail`). */
export interface AgentDetailView {
  readonly phase: LoadPhase
  readonly agentId?: string | undefined
  readonly agent?: AgentRow | undefined
  readonly steps: readonly AgentStep[]
  readonly truncated: boolean
  readonly error?: string | undefined
}

/** The task's current todo list, as the agent last wrote it. */
export function TodoCard({ todos }: { todos: readonly TodoItem[] | undefined }): ReactElement | null {
  if (todos === undefined || todos.length === 0) return null
  const done = todos.filter(todo => todo.status === 'completed').length
  return (
    <details className="card todo-card" open>
      <summary className="todo-head">
        <span>Todo list</span>
        <span className="block-meta">{done} of {todos.length} done</span>
      </summary>
      <ul className="todo-list" aria-label="Todo list">
        {todos.map((todo, index) => (
          <li key={index} className={`todo todo-${todo.status}`}>
            <span className="todo-box" role="img" aria-label={todo.status === 'completed' ? 'done' : todo.status === 'in_progress' ? 'in progress' : 'not started'}>
              {todo.status === 'completed' ? '✓' : todo.status === 'in_progress' ? '•' : ''}
            </span>
            <span className="todo-text">{todo.content}</span>
          </li>
        ))}
      </ul>
    </details>
  )
}

function AgentStatusChip({ status }: { status: string }): ReactElement {
  const view = agentStatus(status)
  return <span className={`agent-status agent-${view.tone}`}>{view.active ? <span className="tool-spin" aria-hidden="true" /> : null}{view.label}</span>
}

function agentLine(agent: AgentRow, now: number): string {
  const parts: string[] = []
  if (agent.toolCalls !== undefined) parts.push(plural(agent.toolCalls, 'tool call'))
  if (agent.model !== undefined && agent.model !== '') parts.push(agent.model)
  const when = agent.updatedAt ?? agent.startedAt
  if (when !== undefined) parts.push(`updated ${relativeTime(when, now)}`)
  return parts.join(' · ')
}

function agentTitle(agent: AgentRow): string {
  return agent.title !== '' ? agent.title : agent.name !== '' ? agent.name : agent.id
}

/** One step of a subagent, rendered with the same rows as the main transcript. */
export function AgentStepView({ step }: { step: AgentStep }): ReactElement {
  switch (step.kind) {
    case 'tool':
      return <ToolRow tool={{ name: step.name ?? 'tool', summary: argsSummary(step.args), state: step.state ?? 'done', args: step.args, output: step.output }} />
    case 'thinking':
      return (
        <details className="fold">
          <summary>{step.state === 'working' ? 'Thinking…' : 'Thought it through'}</summary>
          <p className="block-text block-dim">{step.text ?? ''}</p>
        </details>
      )
    case 'text':
      return <Markdown text={step.text ?? ''} className="block-md" />
    case 'notice':
      return <p className={`block-notice${step.state === 'error' ? ' is-error' : ''}`}>{step.text ?? ''}</p>
  }
}

export interface AgentsPanelProps {
  readonly agents: AgentListView
  readonly detail: AgentDetailView
  readonly now: number
  readonly onOpen: (agentId: string) => void
  readonly onBack: () => void
  readonly onRetryList: () => void
  readonly onRetryDetail: (agentId: string) => void
}

function AgentList({ agents, now, onOpen, onRetryList }: Pick<AgentsPanelProps, 'agents' | 'now' | 'onOpen' | 'onRetryList'>): ReactElement {
  return (
    <>
      {agents.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not load the agents: {agents.error ?? 'unknown error'}</p>
          <button type="button" className="btn" onClick={onRetryList}>Try again</button>
        </div>
      ) : null}
      {(agents.phase === 'loading' || agents.phase === 'idle') && agents.rows.length === 0 ? <p role="status" aria-busy="true">Loading agents…</p> : null}
      {agents.phase === 'ready' && agents.rows.length === 0 ? <p className="empty">This task has not started any agents.</p> : null}
      {agents.rows.length === 0 ? null : (
        <ul className="agent-list" aria-label="Agents">
          {agents.rows.map(agent => (
            <li key={agent.id}>
              <button type="button" className="agent-row" onClick={() => onOpen(agent.id)}>
                <span className="agent-row-top">
                  <span className="agent-name">{agentTitle(agent)}</span>
                  <AgentStatusChip status={agent.status} />
                </span>
                {agent.name !== '' && agent.title !== '' && agent.name !== agent.title ? <span className="block-meta">{agent.name}</span> : null}
                {agentLine(agent, now) === '' ? null : <span className="block-meta">{agentLine(agent, now)}</span>}
                {agent.error === undefined ? null : <span className="field-error">{agent.error}</span>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </>
  )
}

function AgentDetail({ detail, now, onBack, onRetryDetail }: Pick<AgentsPanelProps, 'detail' | 'now' | 'onBack' | 'onRetryDetail'>): ReactElement {
  const agent = detail.agent
  return (
    <div className="agent-detail">
      <button type="button" className="btn btn-quiet" onClick={onBack}>‹ All agents</button>
      {agent === undefined ? null : (
        <header className="agent-detail-head">
          <h3 className="card-title">{agentTitle(agent)}</h3>
          <AgentStatusChip status={agent.status} />
          {agentLine(agent, now) === '' ? null : <p className="block-meta">{agentLine(agent, now)}</p>}
          {agent.summary === undefined || agent.summary === '' ? null : <p className="block-text">{agent.summary}</p>}
          {agent.error === undefined ? null : <p role="alert" className="field-error">{agent.error}</p>}
        </header>
      )}
      {detail.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not load this agent: {detail.error ?? 'unknown error'}</p>
          {detail.agentId === undefined ? null : <button type="button" className="btn" onClick={() => onRetryDetail(detail.agentId!)}>Try again</button>}
        </div>
      ) : null}
      {detail.phase === 'loading' && detail.steps.length === 0 ? <p role="status" aria-busy="true">Loading steps…</p> : null}
      {detail.truncated ? <p className="block-meta">Showing the latest steps only.</p> : null}
      {detail.phase === 'ready' && detail.steps.length === 0 ? <p className="empty">No steps recorded yet.</p> : null}
      {detail.steps.length === 0 ? null : (
        <ol className="blocks agent-steps" aria-label="Agent steps">
          {detail.steps.map((step, index) => (
            <li key={index} className={`block block-${step.kind === 'text' ? 'assistant' : step.kind === 'tool' ? 'tools' : step.kind}`}>
              <AgentStepView step={step} />
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}

/** The agent list, or one agent's own steps once a row is tapped. */
export function AgentsPanel(props: AgentsPanelProps): ReactElement {
  return props.detail.agentId === undefined ? <AgentList {...props} /> : <AgentDetail {...props} />
}

export function AgentsSheet(props: AgentsPanelProps & { readonly onClose: () => void }): ReactElement {
  const { onClose, ...panel } = props
  const title = panel.detail.agentId === undefined ? 'Agents' : panel.detail.agent === undefined ? 'Agent' : agentTitle(panel.detail.agent)
  return (
    <Sheet title={title} onClose={onClose} fullScreen>
      <AgentsPanel {...panel} />
    </Sheet>
  )
}

/** "3 agents · 1 running", from the loaded rows or else the inbox counts. */
export function agentsButtonLabel(rows: readonly AgentRow[], counts: { readonly total: number; readonly active: number } | undefined): string | undefined {
  const total = rows.length > 0 ? rows.length : counts?.total ?? 0
  if (total === 0) return undefined
  const active = rows.length > 0 ? rows.filter(row => agentStatus(row.status).active).length : counts?.active ?? 0
  return `${plural(total, 'agent')}${active > 0 ? ` · ${active} running` : ''}`
}
