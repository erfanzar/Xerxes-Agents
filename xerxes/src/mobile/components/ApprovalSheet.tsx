// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

import type { DecisionKind, DecisionState } from '../../client/hubStore.js'
import type { PendingInteraction } from '../../client/sessionFold.js'
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

function QuestionForm({ item, locked, onAnswer }: { item: PendingInteraction; locked: boolean; onAnswer: (answers: Readonly<Record<string, string>>) => void }): ReactElement {
  const [answers, setAnswers] = useState<Readonly<Record<string, string>>>({})
  const complete = item.questions.length > 0 && item.questions.every(question => (answers[question.id] ?? '').trim() !== '')
  return (
    <form className="approval-questions" onSubmit={event => { event.preventDefault(); if (complete && !locked) onAnswer(answers) }}>
      {item.questions.map(question => (
        <fieldset key={question.id} className="question" disabled={locked}>
          <legend>{question.question}</legend>
          {question.options.map(option => (
            <button key={option} type="button" className="btn btn-segment" aria-pressed={answers[question.id] === option} onClick={() => setAnswers({ ...answers, [question.id]: option })}>{option}</button>
          ))}
          {question.allowFreeForm || question.options.length === 0 ? (
            <label className="field">
              <span>{question.options.length > 0 ? 'Something else' : 'Your answer'}</span>
              <input type="text" value={question.options.includes(answers[question.id] ?? '') ? '' : answers[question.id] ?? ''} onChange={event => setAnswers({ ...answers, [question.id]: event.target.value })} />
            </label>
          ) : null}
        </fieldset>
      ))}
      <button type="submit" className="btn btn-primary" disabled={!complete || locked}>Send answer</button>
    </form>
  )
}

/** The decision surface: real command, inputs, workspace and scope, with the safe choices first. */
export function ApprovalPanel({ item, decision, canRespond, disabledReason, onDecide, onAnswer, initialStep = 'choose' }: ApprovalPanelProps): ReactElement {
  const [step, setStep] = useState<ApprovalStep>(initialStep)
  const phase = decision?.phase
  const locked = !canRespond || phase === 'sending' || phase === 'unknown' || phase === 'resolved'
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
      {item.detailsHidden ? <p className="notice">The exact command and inputs are hidden for your role. Open this on the desktop to see them.</p> : null}

      {phase === 'sending' ? <p role="status">Sending your decision…</p> : null}
      {phase === 'failed' ? <p role="alert" className="field-error">Not applied: {decision?.message ?? 'The decision did not go through.'} You can try again.</p> : null}
      {phase === 'unknown' ? <p role="alert" className="field-error">{decision?.message ?? 'Could not confirm whether this was applied.'} Check the task before deciding again.</p> : null}
      {phase === 'resolved' ? <p role="status">{decision?.already === true ? (decision.by ? `Already resolved by ${decision.by}.` : decision.message ?? 'Already resolved by someone else.') : 'Decision sent.'}</p> : null}
      {!canRespond && disabledReason !== undefined ? <p className="notice">{disabledReason}</p> : null}

      {item.kind === 'question' ? (
        <QuestionForm item={item} locked={locked} onAnswer={onAnswer} />
      ) : step === 'choose' ? (
        <div className="approval-actions">
          <p className="approval-scope">Allow once approves only this request. Deny refuses it.</p>
          <button type="button" className="btn btn-primary" disabled={locked} onClick={() => act('once')}>Allow once</button>
          <button type="button" className="btn btn-danger" disabled={locked} onClick={() => act('deny')}>Deny</button>
          <button type="button" className="btn btn-quiet" disabled={locked} onClick={() => act('session')}>Allow for this session…</button>
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
