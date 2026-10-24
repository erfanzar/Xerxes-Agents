// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import type { ReactElement } from 'react'

import { autoCardNote } from '../../client/autopilot.js'
import type { ConfirmationState, SupervisorConfirmation } from '../../client/hubParsers.js'

/**
 * One action XERXES proposes. Nothing runs until a person presses Confirm, and
 * Confirm is not an approval of anything the task later asks for. The card says
 * who asked, what it touches and when it expires, and keeps the true outcome
 * visible: a failed or unknown result is never shown as done.
 */

const SETTLED: Readonly<Record<ConfirmationState, string | undefined>> = {
  pending: undefined,
  confirming: undefined,
  confirmed: 'Confirmed and sent.',
  cancelled: 'Cancelled. Nothing ran.',
  expired: 'Expired. Ask XERXES again if you still want this.',
  failed: undefined,
}

export interface ActionConfirmationCardProps {
  readonly item: SupervisorConfirmation
  /** False while the Hub link is down: the buttons stay visible but disabled. */
  readonly canAct: boolean
  readonly onRespond: (decision: 'confirm' | 'cancel') => void
}

export function ActionConfirmationCard({ item, canAct, onRespond }: ActionConfirmationCardProps): ReactElement {
  const settled = SETTLED[item.state]
  const titleId = `action-card-${item.id}`
  if (item.auto === true) {
    // Autopilot ran this without a Confirm tap: it is a record, never a question.
    return (
      <article className="action-card action-card--auto" aria-labelledby={titleId} data-state={item.state} data-auto="true">
        <h3 id={titleId} className="action-card__title">XERXES did: {item.tool}</h3>
        <p className="action-card__summary">{item.summary}</p>
        {item.targets.length === 0 ? null : <ul className="action-card__targets" aria-label="Affects">{item.targets.map(target => <li key={target}>{target}</li>)}</ul>}
        {item.state === 'failed'
          ? <p role="alert" className="action-card__error">{item.message ?? 'Autopilot tried this and it did not go through.'}</p>
          : <p role="status" className="action-card__result">{autoCardNote(item.confirmedBy)}</p>}
      </article>
    )
  }
  return (
    <article className="action-card" aria-labelledby={titleId} data-state={item.state}>
      <h3 id={titleId} className="action-card__title">XERXES proposes: {item.tool}</h3>
      <p className="action-card__summary">{item.summary}</p>
      {item.targets.length === 0 ? null : <ul className="action-card__targets" aria-label="Affects">{item.targets.map(target => <li key={target}>{target}</li>)}</ul>}
      <p className="action-card__meta">
        {item.requestedBy === undefined ? null : <>Requested by {item.requestedBy}. </>}
        {item.state === 'pending' ? 'Nothing runs until you confirm. ' : null}
        {item.expiresAt !== undefined && item.state === 'pending' ? <>Expires <time dateTime={item.expiresAt}>{new Date(item.expiresAt).toLocaleTimeString()}</time>.</> : null}
      </p>
      {item.state === 'pending' ? (
        <div className="action-card__actions" role="group" aria-label="Confirm or cancel this action">
          <button type="button" className="btn btn-primary" disabled={!canAct} onClick={() => onRespond('confirm')}>Confirm</button>
          <button type="button" className="btn" disabled={!canAct} onClick={() => onRespond('cancel')}>Cancel</button>
        </div>
      ) : null}
      {item.state === 'confirming' ? <p role="status">Sending…</p> : null}
      {settled === undefined ? null : <p role="status" className="action-card__result">{settled}</p>}
      {item.state === 'failed' ? <p role="alert" className="action-card__error">{item.message ?? 'That did not go through.'}</p> : null}
    </article>
  )
}
