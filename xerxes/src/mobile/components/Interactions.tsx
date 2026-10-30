// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

import type { DecisionKind, DecisionState } from '../../client/hubStore.js'
import type { PendingInteraction, QuestionItem } from '../../client/sessionFold.js'
import { optionLabel } from './toolView.js'

/** Shown instead of the Allow buttons when the request arrived without its command and inputs. */
export const HIDDEN_DETAILS_NOTE =
  'The command and inputs were not sent to this device (your role cannot see them, or they were not recorded), so it cannot be approved blind from here. Deny it, or open the task on the desktop to approve.'

/** Decisions are locked offline, while one is in flight or unconfirmed, and once resolved. */
export function decisionLocked(canRespond: boolean, decision: DecisionState | undefined): boolean {
  const phase = decision?.phase
  return !canRespond || phase === 'sending' || phase === 'unknown' || phase === 'resolved'
}

export function DecisionStatus({ decision }: { decision: DecisionState | undefined }): ReactElement | null {
  switch (decision?.phase) {
    case 'sending': return <p role="status">Sending your decision…</p>
    case 'failed': return <p role="alert" className="field-error">Not applied: {decision.message ?? 'The decision did not go through.'} You can try again.</p>
    case 'unknown': return <p role="alert" className="field-error">{decision.message ?? 'Could not confirm whether this was applied.'} Check the task before deciding again.</p>
    case 'resolved':
      return <p role="status">{decision.already === true ? (decision.by ? `Already resolved by ${decision.by}.` : decision.message ?? 'Already resolved by someone else.') : 'Decision sent.'}</p>
    default: return null
  }
}

function QuestionField({ question, value, locked, onChange }: { question: QuestionItem; value: string; locked: boolean; onChange: (value: string) => void }): ReactElement {
  const typed = question.options.includes(value) ? '' : value
  const freeForm = question.allowFreeForm || question.options.length === 0
  return (
    <fieldset className="question" disabled={locked}>
      <legend>{question.question}</legend>
      {question.options.length === 0 ? null : (
        <div className="ask-options" role="group" aria-label="Options">
          {question.options.map(option => {
            const { label, recommended } = optionLabel(option)
            return (
              <button key={option} type="button" className="btn btn-segment ask-option" aria-pressed={value === option} onClick={() => onChange(option)}>
                {label}
                {recommended ? <span className="option-tag">Recommended</span> : null}
              </button>
            )
          })}
        </div>
      )}
      {freeForm ? (
        <label className="field">
          <span>{question.options.length > 0 ? 'Or type your own answer' : 'Your answer'}</span>
          <input type="text" value={typed} onChange={event => onChange(event.target.value)} />
        </label>
      ) : null}
    </fieldset>
  )
}

/** Every question of one request, answered together: option chips, free text where allowed, one Send. */
export function QuestionForm({ item, locked, onAnswer }: { item: PendingInteraction; locked: boolean; onAnswer: (answers: Readonly<Record<string, string>>) => void }): ReactElement {
  const [answers, setAnswers] = useState<Readonly<Record<string, string>>>({})
  const complete = item.questions.length > 0 && item.questions.every(question => (answers[question.id] ?? '').trim() !== '')
  const missing = item.questions.length - item.questions.filter(question => (answers[question.id] ?? '').trim() !== '').length
  if (item.questions.length === 0) {
    return <p className="notice">The question was not sent to this device. Open the task on the desktop to answer it.</p>
  }
  return (
    <form className="approval-questions" onSubmit={event => { event.preventDefault(); if (complete && !locked) onAnswer(answers) }}>
      {item.questions.map(question => (
        <QuestionField
          key={question.id}
          question={question}
          value={answers[question.id] ?? ''}
          locked={locked}
          onChange={value => setAnswers(current => ({ ...current, [question.id]: value }))}
        />
      ))}
      {item.questions.length > 1 && missing > 0 ? <p className="block-meta">{missing} of {item.questions.length} still need an answer.</p> : null}
      <button type="submit" className="btn btn-primary" disabled={!complete || locked}>{item.questions.length > 1 ? 'Send answers' : 'Send answer'}</button>
    </form>
  )
}

export interface InlineInteractionsProps {
  readonly items: readonly PendingInteraction[]
  readonly decisions: Readonly<Record<string, DecisionState>>
  readonly canRespond: boolean
  readonly disabledReason?: string | undefined
  readonly onDecide: (interactionId: string, decision: DecisionKind) => void
  readonly onAnswer: (interactionId: string, answers: Readonly<Record<string, string>>) => void
  readonly onDetails: (interactionId: string) => void
}

function PermissionCard({ item, decision, canRespond, disabledReason, onDecide, onDetails }: {
  item: PendingInteraction
  decision: DecisionState | undefined
  canRespond: boolean
  disabledReason: string | undefined
  onDecide: (decision: DecisionKind) => void
  onDetails: () => void
}): ReactElement {
  const locked = decisionLocked(canRespond, decision)
  return (
    <>
      <p className="approval-tool"><strong>{item.toolName}</strong>{item.requestedBy === undefined ? '' : ` · requested by ${item.requestedBy}`}</p>
      {item.command === undefined ? null : <pre className="block-code approval-command ask-scroll" aria-label="Command">{item.command}</pre>}
      {item.inputs.length === 0 ? null : (
        <dl className="approval-inputs ask-scroll">
          {item.inputs.map(([key, value]) => <div key={key}><dt>{key}</dt><dd><code>{value}</code></dd></div>)}
        </dl>
      )}
      {item.description === '' ? null : <p>{item.description}</p>}
      {item.detailsHidden ? <p className="notice">{HIDDEN_DETAILS_NOTE}</p> : null}
      <DecisionStatus decision={decision} />
      {!canRespond && disabledReason !== undefined ? <p className="notice">{disabledReason}</p> : null}
      <div className="approval-actions ask-actions" role="group" aria-label="Decide">
        {item.detailsHidden ? null : <button type="button" className="btn btn-primary" disabled={locked} onClick={() => onDecide('once')}>Allow</button>}
        {item.detailsHidden ? null : <button type="button" className="btn" disabled={locked} onClick={() => onDecide('session')}>Allow for session</button>}
        <button type="button" className="btn btn-danger" disabled={locked} onClick={() => onDecide('deny')}>Deny</button>
        <button type="button" className="btn btn-quiet" onClick={onDetails}>Details</button>
      </div>
    </>
  )
}

/** Questions and permission requests at the bottom of the chat, answered in place with one tap. */
export function InlineInteractions({ items, decisions, canRespond, disabledReason, onDecide, onAnswer, onDetails }: InlineInteractionsProps): ReactElement | null {
  if (items.length === 0) return null
  return (
    <ul className="pending inline-asks" aria-label="Waiting for you">
      {items.map(item => {
        const decision = decisions[item.id]
        return (
          <li key={item.id} className="card card-attention ask" data-kind={item.kind}>
            <span className="badge badge-attention">{item.kind === 'question' ? 'Question' : 'Permission'}</span>
            <h3 className="card-title">{item.kind === 'question' ? (item.requestedBy === undefined ? item.title : `${item.title} · ${item.requestedBy}`) : item.title}</h3>
            {item.kind === 'question' ? (
              <>
                <DecisionStatus decision={decision} />
                {!canRespond && disabledReason !== undefined ? <p className="notice">{disabledReason}</p> : null}
                <QuestionForm item={item} locked={decisionLocked(canRespond, decision)} onAnswer={answers => onAnswer(item.id, answers)} />
              </>
            ) : (
              <PermissionCard
                item={item}
                decision={decision}
                canRespond={canRespond}
                disabledReason={disabledReason}
                onDecide={kind => onDecide(item.id, kind)}
                onDetails={() => onDetails(item.id)}
              />
            )}
          </li>
        )
      })}
    </ul>
  )
}
