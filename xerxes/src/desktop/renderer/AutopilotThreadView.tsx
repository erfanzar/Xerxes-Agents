// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import { AUTOPILOT_THREAD_NOTE } from '../../client/autopilot.js'
import type { AutopilotThreadState } from '../../client/hubStore.js'
import { liveSteps } from '../../client/supervisorSteps.js'
import { ActionConfirmationCard } from './ActionConfirmationCard.js'
import { Icon } from './Icon.js'
import { LiveSteps, ThreadMessages } from './SupervisorThread.js'

/**
 * Activity: the owner's private Autopilot thread. The Hub's wake-ups, XERXES's
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
  const empty = thread.phase === 'ready' && thread.messages.length === 0 && thread.confirmations.length === 0
  const running = liveSteps(thread.steps, thread.messages)
  return (
    <section className="autopilot-thread" aria-label="Autopilot activity">
      <p className="xthread-note"><Icon name="eye" size={13} />{AUTOPILOT_THREAD_NOTE}</p>
      {thread.phase === 'loading' || thread.phase === 'idle' ? <p role="status" aria-busy="true" className="xempty xempty--wait">Opening Activity…</p> : null}
      {thread.phase === 'error' ? <p role="alert" className="xevent" data-tone="error">Could not open Activity: {thread.error ?? 'unknown error'}</p> : null}
      {empty ? (
        <div className="xempty" role="status">
          <strong>Nothing yet</strong>
          <p>Autopilot&apos;s wake-ups and the actions waiting for you will show up here.</p>
        </div>
      ) : null}
      <ThreadMessages messages={thread.messages} now={now} label="Autopilot conversation" {...(thread.steps ? { steps: thread.steps } : {})} />
      {thread.confirmations.length === 0 ? null : (
        <ul className="supervisor__cards" aria-label="Autopilot actions">
          {thread.confirmations.map(item => (
            <li key={item.id}><ActionConfirmationCard item={item} canAct={live} onRespond={decision => onRespond(item.id, decision)} /></li>
          ))}
        </ul>
      )}
      {thread.working === true || running.length > 0 ? <LiveSteps steps={running} idle="Autopilot is working…" /> : null}
    </section>
  )
}
