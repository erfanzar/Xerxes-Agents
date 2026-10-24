// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import { AUTOPILOT_THREAD_NOTE } from '../../client/autopilot.js'
import { relativeTime } from '../../client/format.js'
import type { AutopilotThreadState } from '../../client/hubStore.js'
import { ActionConfirmationCard } from './ActionConfirmationCard.js'
import { Markdown } from './markdown.js'

/**
 * The owner's private Autopilot thread: the Hub's wake-up messages, XERXES's
 * replies to them, the cards waiting for the owner's Confirm and the cards for
 * what XERXES did on its own. It is read-only here; its events never mix with
 * the shared thread.
 */

export interface AutopilotThreadViewProps {
  readonly thread: AutopilotThreadState
  readonly live: boolean
  readonly now: number
  readonly onRespond: (confirmationId: string, decision: 'confirm' | 'cancel') => void
}

export function AutopilotThreadView({ thread, live, now, onRespond }: AutopilotThreadViewProps): ReactElement {
  return (
    <section className="autopilot-thread" aria-label="Autopilot thread">
      <p className="studio-muted">{AUTOPILOT_THREAD_NOTE}</p>
      {thread.phase === 'loading' || thread.phase === 'idle' ? <p role="status" aria-busy="true" className="studio-muted">Opening the Autopilot thread…</p> : null}
      {thread.phase === 'error' ? <p role="alert" className="action-card__error">Could not open the Autopilot thread: {thread.error ?? 'unknown error'}</p> : null}
      {thread.phase === 'ready' && thread.messages.length === 0 && thread.confirmations.length === 0
        ? <p role="status" className="studio-muted">Nothing yet. Autopilot&apos;s wake-ups and the actions waiting for you will show up here.</p>
        : null}
      {thread.messages.length === 0 ? null : (
        <ol className="supervisor__messages" aria-label="Autopilot conversation">
          {thread.messages.map(message => (
            <li key={message.id} className={`supervisor__message supervisor__message--${message.role}`}>
              <p className="supervisor__who">
                {message.role === 'xerxes' ? 'XERXES' : message.role === 'system' ? 'Autopilot' : message.author ?? 'You'}
                <span> · <time dateTime={message.at}>{relativeTime(message.at, now)}</time></span>
              </p>
              {message.role === 'xerxes'
                ? <div className="msg msg--agent"><Markdown text={message.text} className="msg__text" /></div>
                : <p className="supervisor__notice">{message.text}</p>}
            </li>
          ))}
        </ol>
      )}
      {thread.confirmations.length === 0 ? null : (
        <ul className="supervisor__cards" aria-label="Autopilot actions">
          {thread.confirmations.map(item => (
            <li key={item.id}><ActionConfirmationCard item={item} canAct={live} onRespond={decision => onRespond(item.id, decision)} /></li>
          ))}
        </ul>
      )}
      {thread.working === true ? <p role="status" aria-live="polite" className="studio-muted">Autopilot is working…</p> : null}
    </section>
  )
}
