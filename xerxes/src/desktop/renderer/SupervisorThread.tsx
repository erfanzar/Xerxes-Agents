// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Fragment, type ReactElement } from 'react'

import { relativeTime } from '../../client/format.js'
import type { SupervisorConfirmation, SupervisorMessage, SupervisorStep } from '../../client/hubParsers.js'
import { withRecords } from '../../client/supervisorRecords.js'
import { endsTurn, stepPhrase, stepsFinished } from '../../client/supervisorSteps.js'
import { AgentOrb } from './AgentOrb.js'
import { elapsedOf } from './duration.js'
import { Icon } from './Icon.js'
import { Markdown } from './markdown.js'

/**
 * The XERXES conversation, shared by Chat and Activity. People and XERXES get
 * full messages; the Hub's notices ("Autopilot: woke for…", "Needs attention…",
 * "XERXES stopped…") are quiet one-line events. Consecutive identical notices
 * merge into one line with a count, a long run of notices folds its older lines
 * away, and a failure is the only notice drawn as an error.
 */

export type NoticeTone = 'info' | 'autopilot' | 'attention' | 'error'

export function noticeTone(text: string): NoticeTone {
  if (/^XERXES (stopped|failed)\b|\bcould not\b|^error\b/i.test(text)) return 'error'
  if (/^Needs attention\b|\bneeds you\b|^Watch:/i.test(text)) return 'attention'
  if (/^Autopilot\b|^XERXES did\b/i.test(text)) return 'autopilot'
  return 'info'
}

export interface EventLine {
  readonly id: string
  readonly text: string
  /** The latest occurrence. */
  readonly at: string
  readonly count: number
  readonly tone: NoticeTone
}

export type ThreadItem =
  | { readonly kind: 'message'; readonly message: SupervisorMessage }
  | { readonly kind: 'events'; readonly id: string; readonly lines: readonly EventLine[] }

/** Messages in order; each run of notices becomes one events item, identical neighbours merged. */
export function threadItems(messages: readonly SupervisorMessage[]): readonly ThreadItem[] {
  const items: ThreadItem[] = []
  for (const message of messages) {
    if (message.role !== 'system') {
      items.push({ kind: 'message', message })
      continue
    }
    const last = items.at(-1)
    const line: EventLine = { id: message.id, text: message.text, at: message.at, count: 1, tone: noticeTone(message.text) }
    if (last?.kind !== 'events') {
      items.push({ kind: 'events', id: message.id, lines: [line] })
      continue
    }
    const previous = last.lines.at(-1)
    const lines = previous !== undefined && previous.text === message.text
      ? [...last.lines.slice(0, -1), { ...previous, at: message.at, count: previous.count + 1 }]
      : [...last.lines, line]
    items[items.length - 1] = { kind: 'events', id: last.id, lines }
  }
  return items
}

/** Lines shown from a run of notices; older ones fold behind "N earlier events". */
export const VISIBLE_EVENTS = 3

const TONE_ICON = { info: 'info', autopilot: 'spark', attention: 'warning', error: 'error' } as const

function EventRow({ line, now }: { readonly line: EventLine; readonly now: number }): ReactElement {
  return (
    <li className="xevent" data-tone={line.tone}>
      <Icon name={TONE_ICON[line.tone]} size={13} />
      <span className="xevent__text">{line.text}</span>
      {line.count > 1 ? <span className="xevent__count" aria-label={`${line.count} times`}>×{line.count}</span> : null}
      <time className="xevent__time" dateTime={line.at}>{relativeTime(line.at, now)}</time>
    </li>
  )
}

function Events({ lines, now }: { readonly lines: readonly EventLine[]; readonly now: number }): ReactElement {
  // Folding one line saves nothing: only fold when at least two would hide.
  const fold = lines.length > VISIBLE_EVENTS + 1
  const hidden = fold ? lines.slice(0, lines.length - VISIBLE_EVENTS) : []
  const shown = fold ? lines.slice(-VISIBLE_EVENTS) : lines
  return (
    <>
      {hidden.length === 0 ? null : (
        <details className="xevents__more">
          <summary>{hidden.length} earlier event{hidden.length === 1 ? '' : 's'}</summary>
          <ul className="xevents__list">{hidden.map(line => <EventRow key={line.id} line={line} now={now} />)}</ul>
        </details>
      )}
      <ul className="xevents__list">{shown.map(line => <EventRow key={line.id} line={line} now={now} />)}</ul>
    </>
  )
}

function MessageRow({ message, now, pending = false }: { readonly message: SupervisorMessage; readonly now: number; readonly pending?: boolean }): ReactElement {
  const time = <time dateTime={message.at}>{pending ? 'just now' : relativeTime(message.at, now)}</time>
  const voice = message.source === 'voice' ? ' · voice' : ''
  if (message.role === 'xerxes') {
    return (
      <li className="supervisor__message supervisor__message--xerxes">
        <div className="xmsg__head">
          <span className="xmsg__avatar" aria-hidden="true">X</span>
          <span className="xmsg__who">XERXES</span>
          <span>{voice}</span>
          {time}
        </div>
        <div className="msg msg--agent"><Markdown text={message.text} className="msg__text" /></div>
      </li>
    )
  }
  return (
    <li className={`supervisor__message supervisor__message--user${pending ? ' supervisor__message--pending' : ''}`}>
      <p className="xmsg__meta">{pending ? 'You' : message.author ?? 'You'}{voice} · {time}</p>
      <div className="msg msg--user"><div className="msg__text">{message.text}</div></div>
    </li>
  )
}

// ── live steps ────────────────────────────────────────────────────────

const STEP_ICON = { running: 'spinner', done: 'check', error: 'error' } as const

export function StepList({ steps }: { readonly steps: readonly SupervisorStep[] }): ReactElement {
  return (
    <ol className="xsteps__list" aria-label="Steps">
      {steps.map(step => step.kind === 'say'
        ? <li key={step.id} className="xstep xstep--say">{step.text}</li>
        : (
          <li key={step.id} className="xstep" data-state={step.state}>
            <Icon name={STEP_ICON[step.state]} size={13} />
            <span>{step.text}</span>
          </li>
        ))}
    </ol>
  )
}

/** What XERXES is doing right now: a status line with a clock, then every step so far. */
export function LiveSteps({ steps, seconds, idle = 'Thinking…' }: {
  readonly steps: readonly SupervisorStep[]
  readonly seconds?: number
  /** The status line before the first step arrives. */
  readonly idle?: string
}): ReactElement {
  return (
    <div className="xsteps is-live" role="status" aria-live="polite">
      <div className="streamstatus supervisor__thinking">
        <AgentOrb size={20} state="working" />
        <span className="streamstatus__phrase">{stepPhrase(steps) ?? idle}</span>
        {seconds === undefined ? null : <span className="streamstatus__clock">{elapsedOf(seconds)}</span>}
      </div>
      {steps.length === 0 ? null : <StepList steps={steps} />}
    </div>
  )
}

function StepsSummary({ steps }: { readonly steps: readonly SupervisorStep[] }): ReactElement {
  const tools = steps.filter(step => step.kind === 'tool')
  const failed = tools.filter(step => step.state === 'error').length
  const label = tools.length === 0 ? 'Worked through it' : `Ran ${tools.length} step${tools.length === 1 ? '' : 's'}${failed > 0 ? ` · ${failed} failed` : ''}`
  return (
    <details className="xsteps">
      <summary>{label}</summary>
      <StepList steps={steps} />
    </details>
  )
}

/** Index of the item that finished the latest turn's steps, or -1 while they still run. */
function finishIndex(items: readonly ThreadItem[], steps: readonly SupervisorStep[], messages: readonly SupervisorMessage[]): number {
  if (steps.length === 0 || !stepsFinished(steps, messages)) return -1
  const at = Date.parse(steps.at(-1)!.at)
  return items.findIndex(item => item.kind === 'message'
    ? endsTurn(item.message) && Date.parse(item.message.at) >= at
    : item.lines.some(line => endsTurn({ role: 'system', text: line.text }) && Date.parse(line.at) >= at))
}

export interface ThreadMessagesProps {
  readonly messages: readonly SupervisorMessage[]
  /** What you just sent, shown until the Hub returns it. */
  readonly pending?: { readonly text: string; readonly at: string }
  readonly now: number
  readonly label: string
  /** The latest turn's steps: folded above the reply that finished them. */
  readonly steps?: readonly SupervisorStep[]
  /** Settled action cards: shown as event lines at the time they were made. */
  readonly records?: readonly SupervisorConfirmation[]
}

export function ThreadMessages({ messages, pending, now, label, steps, records }: ThreadMessagesProps): ReactElement | null {
  const all = withRecords(messages, records ?? [])
  const items = threadItems(all)
  if (items.length === 0 && pending === undefined) return null
  const finished = steps === undefined ? -1 : finishIndex(items, steps, all)
  return (
    <ol className="xthread" aria-label={label}>
      {items.map((item, index) => (
        <Fragment key={item.kind === 'message' ? item.message.id : item.id}>
          {index === finished && steps !== undefined ? <li className="xsteps-item"><StepsSummary steps={steps} /></li> : null}
          {item.kind === 'message'
            ? <MessageRow message={item.message} now={now} />
            : <li className="xevents"><Events lines={item.lines} now={now} /></li>}
        </Fragment>
      ))}
      {pending === undefined ? null : (
        <MessageRow key="pending" message={{ id: 'pending', role: 'user', text: pending.text, at: pending.at, source: 'text' }} now={now} pending />
      )}
    </ol>
  )
}
