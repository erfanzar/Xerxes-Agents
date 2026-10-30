// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

import type { DecisionKind, DecisionState } from '../../client/hubStore.js'
import type { PendingInteraction } from '../../client/sessionFold.js'
import { DecisionStatus, HIDDEN_DETAILS_NOTE, QuestionForm, decisionLocked } from './Interactions.js'
import { Sheet } from './Sheet.js'

export type ApprovalStep = 'choose' | 'confirm-session'
export type ApprovalIntent = 'once' | 'deny' | 'session' | 'confirm-session' | 'back'

/**
 * The approval flow as a pure step function. Allow once and Deny decide at
 * once; the session-wide choice only moves to a confirmation step and decides
 * nothing until it is confirmed there.
 */
export function approvalStep(step: ApprovalStep, intent: ApprovalIntent): { readonly step: ApprovalStep; readonly decide?: DecisionKind } {
  if (intent === 'back') return { step: 'choose' }
  if (step === 'choose') {
    if (intent === 'once') return { step: 'choose', decide: 'once' }
    if (intent === 'deny') return { step: 'choose', decide: 'deny' }
    if (intent === 'session') return { step: 'confirm-session' }
    return { step }
  }
  if (intent === 'confirm-session') return { step: 'choose', decide: 'session' }
  return { step }
}

export interface ApprovalPanelProps {
  readonly item: PendingInteraction
  readonly decision?: DecisionState | undefined
  readonly canRespond: boolean
  readonly disabledReason?: string | undefined
  readonly onDecide: (decision: DecisionKind) => void
  readonly onAnswer: (answers: Readonly<Record<string, string>>) => void
  readonly initialStep?: ApprovalStep
}

/** The decision surface: real command, inputs, workspace and scope, with the safe choices first. */
export function ApprovalPanel({ item, decision, canRespond, disabledReason, onDecide, onAnswer, initialStep = 'choose' }: ApprovalPanelProps): ReactElement {
  const [step, setStep] = useState<ApprovalStep>(initialStep)
  const locked = decisionLocked(canRespond, decision)
  const act = (intent: ApprovalIntent): void => {
    const next = approvalStep(step, intent)
    setStep(next.step)
    if (next.decide !== undefined) onDecide(next.decide)
  }

  return (
    <div className="approval">
      <p className="approval-tool">
        <strong>{item.toolName}</strong>
        {item.requestedBy === undefined ? '' : ` · requested by ${item.requestedBy}`}
      </p>
      {item.workspace === undefined ? null : <p>Workspace: <code>{item.workspace}</code></p>}
      {item.command === undefined ? null : <pre className="block-code approval-command" aria-label="Command">{item.command}</pre>}
      {item.inputs.length === 0 ? null : (
        <dl className="approval-inputs">
          {item.inputs.map(([key, value]) => (
            <div key={key}><dt>{key}</dt><dd><code>{value}</code></dd></div>
          ))}
        </dl>
      )}
      {item.description === '' ? null : <p>{item.description}</p>}
      {item.detailsHidden ? <p className="notice">{HIDDEN_DETAILS_NOTE}</p> : null}

      <DecisionStatus decision={decision} />
      {!canRespond && disabledReason !== undefined ? <p className="notice">{disabledReason}</p> : null}

      {item.kind === 'question' ? (
        <QuestionForm item={item} locked={locked} onAnswer={onAnswer} />
      ) : step === 'choose' ? (
        <div className="approval-actions">
          {/* Nothing is approved blind: without the command and inputs only Deny is offered. */}
          <p className="approval-scope">{item.detailsHidden ? 'Deny refuses this request.' : 'Allow once approves only this request. Deny refuses it.'}</p>
          {item.detailsHidden ? null : <button type="button" className="btn btn-primary" disabled={locked} onClick={() => act('once')}>Allow once</button>}
          <button type="button" className="btn btn-danger" disabled={locked} onClick={() => act('deny')}>Deny</button>
          {item.detailsHidden ? null : <button type="button" className="btn btn-quiet" disabled={locked} onClick={() => act('session')}>Allow for this session…</button>}
        </div>
      ) : (
        <div className="approval-actions" role="group" aria-label="Confirm session-wide approval">
          <p className="notice">
            This allows {item.toolName} for the rest of this session without asking again. Anyone watching this task can still stop it. Only choose this if you trust everything the task may do with it.
          </p>
          <button type="button" className="btn btn-danger" disabled={locked} onClick={() => act('confirm-session')}>Yes, allow for this session</button>
          <button type="button" className="btn btn-primary" onClick={() => act('back')}>Go back</button>
        </div>
      )}
    </div>
  )
}

export function ApprovalSheet(props: ApprovalPanelProps & { onClose: () => void }): ReactElement {
  const { onClose, ...panel } = props
  return (
    <Sheet title={panel.item.title} onClose={onClose} tall>
      <ApprovalPanel {...panel} />
    </Sheet>
  )
}
