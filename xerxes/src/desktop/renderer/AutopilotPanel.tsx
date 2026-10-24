// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

import {
  AUTOPILOT_EXPIRY_LAPSED_NOTE, AUTOPILOT_EXPLANATION, AUTOPILOT_INSTRUCTION_PLACEHOLDER, AUTOPILOT_READ_ONLY_NOTE, autopilotExpiresAt,
  autopilotExpiryOptions, defaultAutopilotExpiry, effectiveExpiryChoice, isExpiryLapsed, type AutopilotExpiryChoice,
} from '../../client/autopilot.js'
import { AUTOPILOT_INSTRUCTION_LIMIT, type AutopilotInput } from '../../client/hubController.js'
import type { AutopilotState } from '../../client/hubStore.js'
import type { AutopilotRow } from '../../client/hubTypes.js'

/**
 * Autopilot, per workspace. The Hub owner turns it on; XERXES then answers a
 * task's questions and runs its own actions without a Confirm tap. Permission
 * requests (commands, edits) always stay with a person. Everyone else sees
 * the same rows read-only. `active` is the Hub's verdict and is shown as is.
 */

export interface AutopilotPanelProps {
  readonly autopilot: AutopilotState
  /** False while the Hub link is down: saving is disabled. */
  readonly live: boolean
  readonly onSave: (input: AutopilotInput) => Promise<boolean>
  readonly onRetry: () => void
  readonly onClose: () => void
}

function stateText(row: AutopilotRow): string {
  if (row.active) return 'On'
  return row.enabled ? 'Off (expired or the owner changed)' : 'Off'
}

function ReadOnlyRow({ row }: { readonly row: AutopilotRow }): ReactElement {
  return (
    <li className="autopilot__row" aria-label={`Autopilot for ${row.label}`}>
      <div className="hubphone__row">
        <strong>{row.label}</strong>
        <span className={`autopilot__state${row.active ? ' is-on' : ''}`}>{stateText(row)}</span>
      </div>
      {row.enabledBy === null ? null : <p className="studio-muted">Turned on by {row.enabledBy.name}.</p>}
      {row.expiresAt === null ? null : <p className="studio-muted">Until <time dateTime={row.expiresAt}>{new Date(row.expiresAt).toLocaleString()}</time>.</p>}
      {row.instruction === '' ? null : <p className="autopilot__instruction">{row.instruction}</p>}
    </li>
  )
}

function EditableRow({ row, live, saving, onSave }: {
  readonly row: AutopilotRow
  readonly live: boolean
  readonly saving: boolean
  readonly onSave: (input: AutopilotInput) => Promise<boolean>
}): ReactElement {
  const [enabled, setEnabled] = useState(row.enabled)
  const [instruction, setInstruction] = useState(row.instruction)
  const [expiry, setExpiry] = useState<AutopilotExpiryChoice>(() => defaultAutopilotExpiry(row, Date.now()))
  const id = `autopilot-${row.workspaceId}`
  const tooLong = instruction.length > AUTOPILOT_INSTRUCTION_LIMIT
  const nowMs = Date.now()
  // "Keep" is not on offer once the expiry has passed; the select shows what is really chosen.
  const choice = effectiveExpiryChoice(expiry, row, nowMs)
  const lapsed = isExpiryLapsed(row.expiresAt, nowMs)
  const save = async (): Promise<void> => {
    if (tooLong) return
    const expiresAt = autopilotExpiresAt(choice, Date.now())
    const ok = await onSave({
      workspaceId: row.workspaceId,
      enabled,
      instruction,
      ...(expiresAt === undefined ? {} : { expiresAt }),
    })
    // The saved row now carries the expiry, so the select goes back to "keep" (or "no expiry").
    if (ok) setExpiry(defaultAutopilotExpiry({ expiresAt: expiresAt === undefined ? row.expiresAt : expiresAt }, Date.now()))
  }
  return (
    <li className="autopilot__row" aria-label={`Autopilot for ${row.label}`}>
      <form onSubmit={event => { event.preventDefault(); void save() }}>
        <div className="hubphone__row">
          <label htmlFor={`${id}-on`}>
            <input id={`${id}-on`} type="checkbox" checked={enabled} onChange={event => setEnabled(event.target.checked)} /> <strong>{row.label}</strong>
          </label>
          <span className={`autopilot__state${row.active ? ' is-on' : ''}`}>{stateText(row)}</span>
        </div>
        <div className="field">
          <label htmlFor={`${id}-instruction`}>Standing instruction</label>
          <textarea
            id={`${id}-instruction`}
            rows={2}
            maxLength={AUTOPILOT_INSTRUCTION_LIMIT}
            value={instruction}
            placeholder={AUTOPILOT_INSTRUCTION_PLACEHOLDER}
            onChange={event => setInstruction(event.target.value)}
          />
          <span className="studio-muted">{instruction.length}/{AUTOPILOT_INSTRUCTION_LIMIT}</span>
        </div>
        <div className="field">
          <label htmlFor={`${id}-expiry`}>Stop automatically</label>
          <select id={`${id}-expiry`} value={choice} onChange={event => setExpiry(event.target.value as AutopilotExpiryChoice)}>
            {autopilotExpiryOptions(row, nowMs).map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </div>
        {row.expiresAt === null ? null : lapsed
          ? <p role="status" className="studio-muted">{AUTOPILOT_EXPIRY_LAPSED_NOTE}</p>
          : <p className="studio-muted">Currently until <time dateTime={row.expiresAt}>{new Date(row.expiresAt).toLocaleString()}</time>.</p>}
        {row.enabledBy === null ? null : <p className="studio-muted">Turned on by {row.enabledBy.name}.</p>}
        <div className="hubphone__row">
          <button type="submit" className="btn btn--solid" disabled={!live || saving || tooLong}>{saving ? 'Saving…' : 'Save'}</button>
        </div>
      </form>
    </li>
  )
}

export function AutopilotPanel({ autopilot, live, onSave, onRetry, onClose }: AutopilotPanelProps): ReactElement {
  return (
    <section className="autopilot" aria-label="Autopilot">
      <div className="hubphone__row">
        <strong>Autopilot</strong>
        <button type="button" className="btn" onClick={onClose}>Close</button>
      </div>
      <p className="studio-muted">{AUTOPILOT_EXPLANATION}</p>
      {autopilot.phase === 'loading' && autopilot.rows.length === 0 ? <p role="status" aria-busy="true" className="studio-muted">Reading Autopilot settings…</p> : null}
      {autopilot.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not read Autopilot: {autopilot.error ?? 'unknown error'}</p>
          <button type="button" className="btn" onClick={onRetry}>Try again</button>
        </div>
      ) : null}
      {autopilot.phase === 'ready' || autopilot.rows.length > 0 ? (
        <>
          {!autopilot.canEdit ? <p role="status" className="studio-muted">{AUTOPILOT_READ_ONLY_NOTE}</p> : null}
          {autopilot.rows.length === 0 ? <p role="status" className="notice">No workspace is linked to this Hub yet.</p> : (
            <ul className="autopilot__rows" aria-label="Workspaces">
              {autopilot.rows.map(row => autopilot.canEdit
                ? <EditableRow key={row.workspaceId} row={row} live={live} saving={autopilot.savingId === row.workspaceId} onSave={onSave} />
                : <ReadOnlyRow key={row.workspaceId} row={row} />)}
            </ul>
          )}
        </>
      ) : null}
      {autopilot.saveError === undefined ? null : <p role="alert" className="action-card__error">{autopilot.saveError}</p>}
    </section>
  )
}
