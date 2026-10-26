// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import { plural } from '../../client/format.js'
import { canMutate, presenceFor } from '../../client/hubStore.js'
import { taskKey } from '../../client/hubTypes.js'
import { composerMode, isRunning, resolutionNotices } from '../../client/taskView.js'
import { useHubState, useServices } from '../context.js'
import { statusOfSelected, viewForSelected, workspaceLabel } from '../viewModel.js'
import { ApprovalSheet } from './ApprovalSheet.js'
import { TaskTranscript } from './TaskTranscript.js'
import { Composer } from './Composer.js'
import { StatusBadge } from './StatusBadge.js'
import { EffortSheet, GoalSheet, ModelSheet } from './TaskSheets.js'

type OpenSheet =
  | { readonly kind: 'model' | 'effort' | 'goal' }
  | { readonly kind: 'approval'; readonly id: string }

export function TaskTab({ workspaceId, sessionId }: { workspaceId?: string | undefined; sessionId?: string | undefined }): ReactElement {
  const state = useHubState()
  const { controller } = useServices()
  const [sheet, setSheet] = useState<OpenSheet | undefined>(undefined)

  useEffect(() => {
    if (workspaceId !== undefined && sessionId !== undefined) controller.openSession(workspaceId, sessionId)
  }, [controller, workspaceId, sessionId])

  const selected = state.selected
  const routed = workspaceId !== undefined && sessionId !== undefined
  if (selected === undefined || (routed && (selected.workspaceId !== workspaceId || selected.sessionId !== sessionId))) {
    return (
      <section aria-labelledby="task-title">
        <h1 id="task-title" className="page-title">Task</h1>
        {routed ? <p aria-busy="true">Opening task…</p> : <p className="empty">Choose a task from the Inbox to open it here.</p>}
      </section>
    )
  }

  const summary = state.tasks[taskKey(selected.workspaceId, selected.sessionId)]
  const workspace = state.workspaces[selected.workspaceId]
  const title = selected.tail.title ?? summary?.title ?? 'Task'
  const status = statusOfSelected(state, selected)
  const view = viewForSelected(state, selected)
  const mutable = canMutate(state, selected.workspaceId)
  const chips = presenceFor(state, selected.workspaceId, selected.sessionId)
  const notices = resolutionNotices(selected)
  const draftKey = `task:${selected.workspaceId}:${selected.sessionId}`
  const sendProblem = selected.send.phase === 'failed' || selected.send.phase === 'unknown' ? selected.send.message : undefined
  const approvalItem = sheet?.kind === 'approval' ? selected.tail.pending.find(item => item.id === sheet.id) : undefined
  const agents = summary?.agents

  return (
    <section aria-labelledby="task-title">
      <header className="task-head">
        <p className="card-workspace">{workspaceLabel(state, selected.workspaceId)}</p>
        <h1 id="task-title" className="page-title">{title}</h1>
        <div role="status" aria-live="polite"><StatusBadge view={view} size={32} /></div>
        {agents !== undefined && agents.total > 0 ? <p className="block-meta">{plural(agents.total, 'agent')}{agents.active > 0 ? `, ${agents.active} active` : ''}</p> : null}
        {chips.length === 0 ? null : (
          <ul className="chips" aria-label="People on this task">
            {chips.map(chip => <li key={chip.userId} className="chip">{chip.name}{chip.state === 'typing' ? ' · typing' : ''}</li>)}
          </ul>
        )}
        <div className="config-row">
          <button type="button" className="btn" onClick={() => setSheet({ kind: 'model' })}>Model: {selected.model.effective ?? 'not reported'}</button>
          <button type="button" className="btn" onClick={() => setSheet({ kind: 'effort' })}>Effort: {selected.effort.effective ?? 'not reported'}</button>
          <button type="button" className="btn" onClick={() => setSheet({ kind: 'goal' })}>Goal{selected.goal.effective ? '' : ': none'}</button>
        </div>
      </header>

      {selected.error !== undefined ? (
        <div role="alert" className="notice is-error">
          <p>{selected.phase === 'error' ? 'Could not open this task' : 'Could not refresh this task'}: {selected.error}</p>
          <button type="button" className="btn" onClick={() => { if (workspaceId !== undefined && sessionId !== undefined) controller.openSession(workspaceId, sessionId) }}>Try again</button>
        </div>
      ) : null}
      {selected.phase === 'loading' ? <p aria-busy="true">Loading task…</p> : null}
      {workspace?.readOnly === true ? (
        <p role="status" className="notice">View only. This machine runs a Xerxes version that cannot verify commands from the Hub. Update Xerxes there to send, steer or approve from your phone.</p>
      ) : null}

      {selected.tail.pending.length === 0 ? null : (
        <ul className="pending" aria-label="Waiting for you">
          {selected.tail.pending.map(item => {
            const decision = selected.decisions[item.id]
            return (
              <li key={item.id} className="card card-attention">
                <span className="badge badge-attention">Needs you</span>
                <h2 className="card-title">{item.title}</h2>
                <p>{item.kind === 'question' ? 'Xerxes is waiting for your answer.' : `${item.toolName} is waiting for your decision.`}</p>
                {decision?.phase === 'failed' ? <p role="alert" className="field-error">Not applied: {decision.message ?? 'The decision did not go through.'}</p> : null}
                <button type="button" className="btn btn-primary" onClick={() => setSheet({ kind: 'approval', id: item.id })}>Review and decide</button>
              </li>
            )
          })}
        </ul>
      )}

      {notices.length === 0 ? null : <ul className="notes" role="status">{notices.map(notice => <li key={notice.id}>{notice.text}</li>)}</ul>}
      {selected.tail.queuedSteers > 0 ? <p className="block-meta">{plural(selected.tail.queuedSteers, 'steer')} queued for the next step.</p> : null}

      <TaskTranscript key={`${selected.workspaceId}:${selected.sessionId}`} selected={selected} />

      <Composer draftKey={draftKey} mode={composerMode(status)} running={isRunning(status)} canSend={mutable} sendMessage={sendProblem} />

      {sheet?.kind === 'model' ? (
        <ModelSheet workspace={workspace} field={selected.model} canChange={mutable} onChoose={(profile, model) => { void controller.setModel(profile, model) }} onClose={() => setSheet(undefined)} />
      ) : null}
      {sheet?.kind === 'effort' ? <EffortSheet workspace={workspace} field={selected.effort} onClose={() => setSheet(undefined)} /> : null}
      {sheet?.kind === 'goal' ? (
        <GoalSheet field={selected.goal} canChange={mutable} onSave={objective => { void controller.setGoal(objective) }} onClose={() => setSheet(undefined)} />
      ) : null}
      {approvalItem !== undefined ? (
        <ApprovalSheet
          item={approvalItem}
          decision={selected.decisions[approvalItem.id]}
          canRespond={mutable}
          disabledReason={mutable ? undefined : 'Decisions need a live connection to this workspace.'}
          onDecide={decision => { void controller.decide(approvalItem.id, decision) }}
          onAnswer={answers => { void controller.answerQuestion(approvalItem.id, answers) }}
          onClose={() => setSheet(undefined)}
        />
      ) : null}
    </section>
  )
}
