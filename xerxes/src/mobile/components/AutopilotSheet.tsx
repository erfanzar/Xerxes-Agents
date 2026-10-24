// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import {
  AUTOPILOT_EXPIRY_LAPSED_NOTE, AUTOPILOT_EXPLANATION, AUTOPILOT_INSTRUCTION_PLACEHOLDER, AUTOPILOT_READ_ONLY_NOTE, autopilotExpiresAt,
  autopilotExpiryOptions, defaultAutopilotExpiry, effectiveExpiryChoice, isExpiryLapsed, type AutopilotExpiryChoice,
} from '../../client/autopilot.js'
import { AUTOPILOT_INSTRUCTION_LIMIT } from '../../client/hubController.js'
import type { AutopilotRow } from '../../client/hubTypes.js'
import { useHubState, useServices } from '../context.js'
import { Sheet } from './Sheet.js'

/**
 * Autopilot from the phone: per workspace, a switch, a standing instruction
 * and an expiry. Only the Hub owner can change it; everyone else sees the same
 * rows read-only. `active` is the Hub's verdict and is shown as is.
 */

function stateText(row: AutopilotRow): string {
  if (row.active) return 'On'
  return row.enabled ? 'Off (expired or the owner changed)' : 'Off'
}

function ReadOnlyRow({ row }: { row: AutopilotRow }): ReactElement {
  return (
    <li className="card" aria-label={`Autopilot for ${row.label}`}>
      <h3 className="card-title">{row.label} · {stateText(row)}</h3>
      {row.enabledBy === null ? null : <p className="block-meta">Turned on by {row.enabledBy.name}.</p>}
      {row.expiresAt === null ? null : <p className="block-meta">Until <time dateTime={row.expiresAt}>{new Date(row.expiresAt).toLocaleString()}</time>.</p>}
      {row.instruction === '' ? null : <p className="block-text">{row.instruction}</p>}
    </li>
  )
}

function EditableRow({ row }: { row: AutopilotRow }): ReactElement {
  const state = useHubState()
  const { controller } = useServices()
  const [enabled, setEnabled] = useState(row.enabled)
  const [instruction, setInstruction] = useState(row.instruction)
  const [expiry, setExpiry] = useState<AutopilotExpiryChoice>(() => defaultAutopilotExpiry(row, Date.now()))
  const live = state.connection.state === 'live'
  const saving = state.autopilot.savingId === row.workspaceId
  const tooLong = instruction.length > AUTOPILOT_INSTRUCTION_LIMIT
  const nowMs = Date.now()
  // "Keep" is not on offer once the expiry has passed; the select shows what is really chosen.
  const choice = effectiveExpiryChoice(expiry, row, nowMs)
  const lapsed = isExpiryLapsed(row.expiresAt, nowMs)
  const save = async (): Promise<void> => {
    if (tooLong || !live || saving) return
    const expiresAt = autopilotExpiresAt(choice, Date.now())
    const ok = await controller.setAutopilot({
      workspaceId: row.workspaceId,
      enabled,
      instruction,
      ...(expiresAt === undefined ? {} : { expiresAt }),
    })
    if (ok) setExpiry(defaultAutopilotExpiry({ expiresAt: expiresAt === undefined ? row.expiresAt : expiresAt }, Date.now()))
  }
  return (
    <li className="card" aria-label={`Autopilot for ${row.label}`}>
      <form className="model-form" onSubmit={event => { event.preventDefault(); void save() }}>
        <label className="field">
          <span>{row.label} · {stateText(row)}</span>
          <input type="checkbox" checked={enabled} onChange={event => setEnabled(event.target.checked)} /> Autopilot on
        </label>
        <label className="field">
          <span>Standing instruction</span>
          <textarea
            rows={3}
            maxLength={AUTOPILOT_INSTRUCTION_LIMIT}
            value={instruction}
            placeholder={AUTOPILOT_INSTRUCTION_PLACEHOLDER}
            onChange={event => setInstruction(event.target.value)}
          />
          <span className="block-meta">{instruction.length}/{AUTOPILOT_INSTRUCTION_LIMIT}</span>
        </label>
        <label className="field">
          <span>Stop automatically</span>
          <select value={choice} onChange={event => setExpiry(event.target.value as AutopilotExpiryChoice)}>
            {autopilotExpiryOptions(row, nowMs).map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        {row.expiresAt === null ? null : lapsed
          ? <p role="status" className="block-meta">{AUTOPILOT_EXPIRY_LAPSED_NOTE}</p>
          : <p className="block-meta">Currently until <time dateTime={row.expiresAt}>{new Date(row.expiresAt).toLocaleString()}</time>.</p>}
        <div className="composer-row">
          <button type="submit" className="btn btn-primary" disabled={!live || saving || tooLong}>{saving ? 'Saving…' : 'Save'}</button>
        </div>
      </form>
    </li>
  )
}

/** The Autopilot rows, for owners editable and for everyone else read-only. */
export function AutopilotBody(): ReactElement {
  const { autopilot } = useHubState()
  const { controller } = useServices()
  return (
    <>
      <p className="summary">{AUTOPILOT_EXPLANATION}</p>
      {autopilot.phase === 'loading' && autopilot.rows.length === 0 ? <p role="status" aria-busy="true">Reading Autopilot settings…</p> : null}
      {autopilot.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not read Autopilot: {autopilot.error ?? 'unknown error'}</p>
          <button type="button" className="btn" onClick={() => { void controller.loadAutopilot() }}>Try again</button>
        </div>
      ) : null}
      {autopilot.phase === 'ready' || autopilot.rows.length > 0 ? (
        <>
          {autopilot.canEdit ? null : <p role="status" className="block-meta">{AUTOPILOT_READ_ONLY_NOTE}</p>}
          {autopilot.rows.length === 0 ? <p role="status" className="notice">No workspace is linked to this Hub yet.</p> : (
            <ul className="cards" aria-label="Workspaces">
              {autopilot.rows.map(row => autopilot.canEdit
                ? <EditableRow key={row.workspaceId} row={row} />
                : <ReadOnlyRow key={row.workspaceId} row={row} />)}
            </ul>
          )}
        </>
      ) : null}
      {autopilot.saveError === undefined ? null : <p role="alert" className="field-error">{autopilot.saveError}</p>}
    </>
  )
}

export function AutopilotSheet({ onClose }: { onClose: () => void }): ReactElement {
  const { controller } = useServices()
  useEffect(() => { void controller.loadAutopilot() }, [controller])
  return (
    <Sheet title="Autopilot" onClose={onClose} tall>
      <AutopilotBody />
    </Sheet>
  )
}
