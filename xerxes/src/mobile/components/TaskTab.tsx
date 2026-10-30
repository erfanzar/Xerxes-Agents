// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import { plural } from '../../client/format.js'
import { canMutate, presenceFor } from '../../client/hubStore.js'
import { taskKey } from '../../client/hubTypes.js'
import { composerMode, isRunning, resolutionNotices } from '../../client/taskView.js'
import { useHubState, useNow, useServices } from '../context.js'
import { statusOfSelected, viewForSelected, workspaceLabel } from '../viewModel.js'
import { AgentsSheet, TodoCard, agentsButtonLabel } from './Agents.js'
import { ApprovalSheet } from './ApprovalSheet.js'
import { InlineInteractions } from './Interactions.js'
import { TaskTranscript } from './TaskTranscript.js'
import { Composer } from './Composer.js'
import { StatusBadge } from './StatusBadge.js'
import { EffortSheet, GoalSheet, ModelSheet } from './TaskSheets.js'
import { agentStatus } from './toolView.js'

type OpenSheet =
  | { readonly kind: 'model' | 'effort' | 'goal' | 'agents' }
  | { readonly kind: 'approval'; readonly id: string }

/** How often an open, still-running agent's steps are re-read while its sheet is showing. */
export const AGENT_DETAIL_REFRESH_MS = 3_000

export function TaskTab({ workspaceId, sessionId }: { workspaceId?: string | undefined; sessionId?: string | undefined }): ReactElement {
  const state = useHubState()
  const now = useNow()
  const { controller, store } = useServices()
  const [sheet, setSheet] = useState<OpenSheet | undefined>(undefined)

  // Viewing while this screen is showing; idle once the person leaves it, so others stop seeing them here.
  useEffect(() => {
    if (workspaceId === undefined || sessionId === undefined) return
    // openSession announces 'viewing' itself once the task is open.
    controller.openSession(workspaceId, sessionId)
    return () => { controller.setPresence('idle') }
  }, [controller, workspaceId, sessionId])

  // Each task keeps its own sheets: a sheet opened on one task never shows over another.
  useEffect(() => { setSheet(undefined) }, [workspaceId, sessionId])

  const selected = state.selected
  const live = state.connection.state === 'live'
  const sendPhase = selected?.send.phase

  // A send that failed before the link dropped is stale once the link is back.
  useEffect(() => {
    const open = store.getSnapshot().selected
    if (live && sendPhase === 'failed' && open !== undefined) {
      store.dispatch({ type: 'send', state: { phase: 'idle' }, workspaceId: open.workspaceId, sessionId: open.sessionId })
    }
    // Only the transition to live matters; a failure while live stays visible.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, store])

  const detailId = selected?.agentDetail.agentId
  const detailActive = selected?.agentDetail.agent !== undefined && agentStatus(selected.agentDetail.agent.status).active
  const agentsOpen = sheet?.kind === 'agents'
  useEffect(() => {
    if (!agentsOpen || detailId === undefined || !detailActive || !live) return
    const timer = setInterval(() => { void controller.inspectAgent(detailId) }, AGENT_DETAIL_REFRESH_MS)
    return () => clearInterval(timer)
  }, [controller, agentsOpen, detailId, detailActive, live])

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
  const agentsLabel = agentsButtonLabel(selected.agents.rows, summary?.agents)
  const disabledReason = mutable
    ? undefined
    : workspace?.readOnly === true
      ? 'This machine runs a Xerxes version that cannot verify decisions from the Hub.'
      : 'Decisions need a live connection to this workspace.'

  const openAgents = (): void => {
    setSheet({ kind: 'agents' })
    void controller.loadAgents()
  }
  const closeSheet = (): void => {
    if (sheet?.kind === 'agents') controller.closeAgent()
    setSheet(undefined)
  }

  return (
    <section aria-labelledby="task-title">
      <header className="task-head">
        <p className="card-workspace">{workspaceLabel(state, selected.workspaceId)}</p>
        <h1 id="task-title" className="page-title">{title}</h1>
        <div role="status" aria-live="polite"><StatusBadge view={view} size={32} /></div>
        {chips.length === 0 ? null : (
          <ul className="chips" aria-label="People on this task">
            {chips.map(chip => <li key={chip.userId} className="chip">{chip.name}{chip.state === 'typing' ? ' · typing' : ''}</li>)}
          </ul>
        )}
        <div className="config-row">
          <button type="button" className="btn" aria-haspopup="dialog" onClick={openAgents}>{agentsLabel ?? 'Agents'}</button>
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

      <TaskTranscript key={`${selected.workspaceId}:${selected.sessionId}`} selected={selected}>
        {notices.length === 0 ? null : <ul className="notes" role="status">{notices.map(notice => <li key={notice.id}>{notice.text}</li>)}</ul>}
        {selected.tail.queuedSteers > 0 ? <p className="block-meta">{plural(selected.tail.queuedSteers, 'steer')} queued for the next step.</p> : null}
        <TodoCard todos={selected.tail.todos} />
        <InlineInteractions
          items={selected.tail.pending}
          decisions={selected.decisions}
          canRespond={mutable}
          disabledReason={disabledReason}
          onDecide={(id, decision) => { void controller.decide(id, decision) }}
          onAnswer={(id, answers) => { void controller.answerQuestion(id, answers) }}
          onDetails={id => setSheet({ kind: 'approval', id })}
        />
      </TaskTranscript>

      <Composer
        key={draftKey}
        draftKey={draftKey}
        mode={composerMode(status)}
        running={isRunning(status)}
        canSend={mutable}
        sendMessage={sendProblem}
        onEdited={() => {
          if (store.getSnapshot().selected?.send.phase === 'idle') return
          store.dispatch({ type: 'send', state: { phase: 'idle' }, workspaceId: selected.workspaceId, sessionId: selected.sessionId })
        }}
      />

      {sheet?.kind === 'model' ? (
        <ModelSheet workspace={workspace} field={selected.model} canChange={mutable} onChoose={(profile, model) => { void controller.setModel(profile, model) }} onClose={closeSheet} />
      ) : null}
      {sheet?.kind === 'effort' ? <EffortSheet workspace={workspace} field={selected.effort} onClose={closeSheet} /> : null}
      {sheet?.kind === 'goal' ? (
        <GoalSheet field={selected.goal} canChange={mutable} onSave={objective => { void controller.setGoal(objective) }} onClose={closeSheet} />
      ) : null}
      {sheet?.kind === 'agents' ? (
        <AgentsSheet
          agents={selected.agents}
          detail={selected.agentDetail}
          now={now}
          onOpen={id => { void controller.inspectAgent(id) }}
          onBack={() => controller.closeAgent()}
          onRetryList={() => { void controller.loadAgents() }}
          onRetryDetail={id => { void controller.inspectAgent(id) }}
          onClose={closeSheet}
        />
      ) : null}
      {approvalItem !== undefined ? (
        <ApprovalSheet
          item={approvalItem}
          decision={selected.decisions[approvalItem.id]}
          canRespond={mutable}
          disabledReason={disabledReason}
          onDecide={decision => { void controller.decide(approvalItem.id, decision) }}
          onAnswer={answers => { void controller.answerQuestion(approvalItem.id, answers) }}
          onClose={closeSheet}
        />
      ) : null}
    </section>
  )
}
