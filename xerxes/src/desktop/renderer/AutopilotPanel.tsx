// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

import {
  AUTOPILOT_ALL_NOTE, AUTOPILOT_EXPIRY_LAPSED_NOTE, AUTOPILOT_EXPLANATION, AUTOPILOT_INSTRUCTION_PLACEHOLDER, AUTOPILOT_MODE_OPTIONS, AUTOPILOT_READ_ONLY_NOTE,
  autopilotExpiresAt, autopilotExpiryOptions, autopilotSettingsEnabled, autopilotStateText, autopilotToggleInput, defaultAutopilotExpiry, effectiveExpiryChoice,
  isExpiryLapsed, type AutopilotExpiryChoice, type AutopilotMode,
} from '../../client/autopilot.js'
import { AUTOPILOT_INSTRUCTION_LIMIT, type AutopilotInput } from '../../client/hubController.js'
import type { AutopilotState } from '../../client/hubStore.js'
import type { AutopilotRow } from '../../client/hubTypes.js'

/**
 * Autopilot: one line per workspace with a switch that saves at once, and an
 * "All workspaces" switch on top for 24/7 everywhere. A row's instruction, mode
 * and expiry sit behind its Settings disclosure. Everyone but the Hub owner sees
 * the same lines read-only. `active` is the Hub's verdict and is shown as is.
 */

export interface AutopilotPanelProps {
  readonly autopilot: AutopilotState
  /** False while the Hub link is down: switches and Save are disabled. */
  readonly live: boolean
  readonly onSave: (input: AutopilotInput) => Promise<boolean>
  readonly onRetry: () => void
  readonly onClose: () => void
}

function Switch({ row, label, disabled, onSave }: {
  readonly row: AutopilotRow
  readonly label: string
  readonly disabled: boolean
  readonly onSave: (input: AutopilotInput) => Promise<boolean>
}): ReactElement {
  return (
    <button
      type="button"
      className={`switch${row.active ? ' is-on' : ''}`}
      role="switch"
      aria-checked={row.active}
      aria-label={`${row.active ? 'Turn off' : 'Turn on'} Autopilot for ${label}`}
      disabled={disabled}
      onClick={() => { void onSave(autopilotToggleInput(row, Date.now())) }}
    />
  )
}

function Settings({ row, live, saving, onSave }: {
  readonly row: AutopilotRow
  readonly live: boolean
  readonly saving: boolean
  readonly onSave: (input: AutopilotInput) => Promise<boolean>
}): ReactElement {
  const [instruction, setInstruction] = useState(row.instruction)
  const [mode, setMode] = useState<AutopilotMode>(row.mode)
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
      enabled: autopilotSettingsEnabled(row),
      instruction,
      mode,
      ...(expiresAt === undefined ? {} : { expiresAt }),
    })
    if (ok) setExpiry(defaultAutopilotExpiry({ expiresAt: expiresAt === undefined ? row.expiresAt : expiresAt }, Date.now()))
  }
  return (
    <details className="autopilot__settings">
      <summary>Settings</summary>
      <form onSubmit={event => { event.preventDefault(); void save() }}>
        <div className="autopilot__grid">
          <label htmlFor={`${id}-mode`}>Mode</label>
          <select id={`${id}-mode`} value={mode} onChange={event => setMode(event.target.value === 'observe' ? 'observe' : 'auto')}>
            {AUTOPILOT_MODE_OPTIONS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
          <label htmlFor={`${id}-expiry`}>Stop</label>
          <select id={`${id}-expiry`} value={choice} onChange={event => setExpiry(event.target.value as AutopilotExpiryChoice)}>
            {autopilotExpiryOptions(row, nowMs).map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </div>
        <label htmlFor={`${id}-instruction`} className="studio-muted">Standing instruction</label>
        <textarea
          id={`${id}-instruction`}
          rows={2}
          maxLength={AUTOPILOT_INSTRUCTION_LIMIT}
          value={instruction}
          placeholder={AUTOPILOT_INSTRUCTION_PLACEHOLDER}
          onChange={event => setInstruction(event.target.value)}
        />
        {lapsed ? <p role="status" className="studio-muted">{AUTOPILOT_EXPIRY_LAPSED_NOTE}</p> : null}
        <div className="autopilot__actions">
          <span className="studio-muted">{instruction.length}/{AUTOPILOT_INSTRUCTION_LIMIT}</span>
          <button type="submit" className="btn" disabled={!live || saving || tooLong}>{saving ? 'Saving…' : 'Save'}</button>
        </div>
      </form>
    </details>
  )
}

function Line({ row, label, note, canEdit, live, saving, onSave }: {
  readonly row: AutopilotRow
  readonly label: string
  readonly note?: string
  readonly canEdit: boolean
  readonly live: boolean
  readonly saving: boolean
  readonly onSave: (input: AutopilotInput) => Promise<boolean>
}): ReactElement {
  const meta = [
    autopilotStateText(row),
    row.active && row.expiresAt !== null && row.viaAll !== true ? `until ${new Date(row.expiresAt).toLocaleString()}` : undefined,
    row.enabledBy === null || row.viaAll === true ? undefined : `Turned on by ${row.enabledBy.name}.`,
  ].filter((part): part is string => part !== undefined)
  return (
    <li className={`autopilot__line${row.active ? ' is-on' : ''}`} aria-label={`Autopilot for ${label}`}>
      <div className="autopilot__head">
        <div className="autopilot__text">
          <strong>{label}</strong>
          <span className={`autopilot__state${row.active ? ' is-on' : ''}`}>{meta.join(' · ')}</span>
          {note === undefined ? null : <span className="studio-muted">{note}</span>}
          {row.instruction === '' ? null : <span className="autopilot__instruction">{row.instruction}</span>}
        </div>
        {canEdit ? <Switch row={row} label={label} disabled={!live || saving} onSave={onSave} /> : null}
      </div>
      {canEdit ? <Settings row={row} live={live} saving={saving} onSave={onSave} /> : null}
    </li>
  )
}

export function AutopilotPanel({ autopilot, live, onSave, onRetry, onClose }: AutopilotPanelProps): ReactElement {
  const all = autopilot.all
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
          <ul className="autopilot__rows" aria-label="Workspaces">
            {all === undefined ? null : (
              <Line key="*" row={all} label="All workspaces" note={AUTOPILOT_ALL_NOTE} canEdit={autopilot.canEdit} live={live}
                saving={autopilot.savingId === all.workspaceId} onSave={onSave} />
            )}
            {autopilot.rows.map(row => (
              <Line key={row.workspaceId} row={row} label={row.label} canEdit={autopilot.canEdit} live={live}
                saving={autopilot.savingId === row.workspaceId} onSave={onSave} />
            ))}
          </ul>
          {autopilot.rows.length === 0 ? <p role="status" className="notice">No workspace is linked to this Hub yet.</p> : null}
        </>
      ) : null}
      {autopilot.saveError === undefined ? null : <p role="alert" className="action-card__error">{autopilot.saveError}</p>}
    </section>
  )
}
