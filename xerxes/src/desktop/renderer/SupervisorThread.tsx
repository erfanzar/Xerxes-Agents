// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import { relativeTime } from '../../client/format.js'
import type { SupervisorMessage } from '../../client/hubParsers.js'
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
  if (/^XERXES stopped\b|\bcould not\b|^error\b/i.test(text)) return 'error'
  if (/^Needs attention\b|\bneeds you\b|^Watch:/i.test(text)) return 'attention'
  if (/^Autopilot\b/i.test(text)) return 'autopilot'
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

export interface ThreadMessagesProps {
  readonly messages: readonly SupervisorMessage[]
  /** What you just sent, shown until the Hub returns it. */
  readonly pending?: { readonly text: string; readonly at: string }
  readonly now: number
  readonly label: string
}

export function ThreadMessages({ messages, pending, now, label }: ThreadMessagesProps): ReactElement | null {
  const items = threadItems(messages)
  if (items.length === 0 && pending === undefined) return null
  return (
    <ol className="xthread" aria-label={label}>
      {items.map(item => item.kind === 'message'
        ? <MessageRow key={item.message.id} message={item.message} now={now} />
        : <li key={item.id} className="xevents"><Events lines={item.lines} now={now} /></li>)}
      {pending === undefined ? null : (
        <MessageRow key="pending" message={{ id: 'pending', role: 'user', text: pending.text, at: pending.at, source: 'text' }} now={now} pending />
      )}
    </ol>
  )
}
