// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import {
  AUTOPILOT_ALL_NOTE, AUTOPILOT_EXPIRY_LAPSED_NOTE, AUTOPILOT_EXPLANATION, AUTOPILOT_INSTRUCTION_PLACEHOLDER, AUTOPILOT_MODE_OPTIONS, AUTOPILOT_READ_ONLY_NOTE,
  autopilotExpiresAt, autopilotExpiryOptions, autopilotSettingsEnabled, autopilotStateText, autopilotToggleInput, defaultAutopilotExpiry, effectiveExpiryChoice,
  isExpiryLapsed, type AutopilotExpiryChoice, type AutopilotMode,
} from '../../client/autopilot.js'
import { AUTOPILOT_INSTRUCTION_LIMIT } from '../../client/hubController.js'
import type { AutopilotRow } from '../../client/hubTypes.js'
import { useHubState, useServices } from '../context.js'
import { Sheet } from './Sheet.js'

/**
 * Autopilot from the phone: an "All workspaces" switch for 24/7 everywhere, then
 * one line per workspace with a switch that saves at once. A row's instruction,
 * mode and expiry sit behind its Settings disclosure. Only the Hub owner can
 * change it; everyone else sees the same lines read-only.
 */

function Settings({ row }: { row: AutopilotRow }): ReactElement {
  const state = useHubState()
  const { controller } = useServices()
  const [instruction, setInstruction] = useState(row.instruction)
  const [mode, setMode] = useState<AutopilotMode>(row.mode)
  const [expiry, setExpiry] = useState<AutopilotExpiryChoice>(() => defaultAutopilotExpiry(row, Date.now()))
  const live = state.connection.state === 'live'
  const saving = state.autopilot.savingId === row.workspaceId
  const tooLong = instruction.length > AUTOPILOT_INSTRUCTION_LIMIT
  const nowMs = Date.now()
  const choice = effectiveExpiryChoice(expiry, row, nowMs)
  const lapsed = isExpiryLapsed(row.expiresAt, nowMs)
  const save = async (): Promise<void> => {
    if (tooLong || !live || saving) return
    const expiresAt = autopilotExpiresAt(choice, Date.now())
    const ok = await controller.setAutopilot({
      workspaceId: row.workspaceId,
      enabled: autopilotSettingsEnabled(row),
      instruction,
      mode,
      ...(expiresAt === undefined ? {} : { expiresAt }),
    })
    if (ok) setExpiry(defaultAutopilotExpiry({ expiresAt: expiresAt === undefined ? row.expiresAt : expiresAt }, Date.now()))
  }
  return (
    <details className="ap-settings">
      <summary>Settings</summary>
      <form className="model-form" onSubmit={event => { event.preventDefault(); void save() }}>
        <label className="field">
          <span>Mode</span>
          <select value={mode} onChange={event => setMode(event.target.value === 'observe' ? 'observe' : 'auto')}>
            {AUTOPILOT_MODE_OPTIONS.map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Stop</span>
          <select value={choice} onChange={event => setExpiry(event.target.value as AutopilotExpiryChoice)}>
            {autopilotExpiryOptions(row, nowMs).map(option => <option key={option.value} value={option.value}>{option.label}</option>)}
          </select>
        </label>
        <label className="field">
          <span>Standing instruction</span>
          <textarea
            rows={2}
            maxLength={AUTOPILOT_INSTRUCTION_LIMIT}
            value={instruction}
            placeholder={AUTOPILOT_INSTRUCTION_PLACEHOLDER}
            onChange={event => setInstruction(event.target.value)}
          />
        </label>
        {lapsed ? <p role="status" className="block-meta">{AUTOPILOT_EXPIRY_LAPSED_NOTE}</p> : null}
        <div className="composer-row">
          <span className="block-meta">{instruction.length}/{AUTOPILOT_INSTRUCTION_LIMIT}</span>
          <button type="submit" className="btn" disabled={!live || saving || tooLong}>{saving ? 'Saving…' : 'Save'}</button>
        </div>
      </form>
    </details>
  )
}

function Line({ row, label, note }: { row: AutopilotRow; label: string; note?: string }): ReactElement {
  const state = useHubState()
  const { controller } = useServices()
  const canEdit = state.autopilot.canEdit
  const live = state.connection.state === 'live'
  const saving = state.autopilot.savingId === row.workspaceId
  const meta = [
    autopilotStateText(row),
    row.active && row.expiresAt !== null && row.viaAll !== true ? `until ${new Date(row.expiresAt).toLocaleString()}` : undefined,
    row.enabledBy === null || row.viaAll === true ? undefined : `Turned on by ${row.enabledBy.name}.`,
  ].filter((part): part is string => part !== undefined)
  return (
    <li className={`ap-line${row.active ? ' is-on' : ''}`} aria-label={`Autopilot for ${label}`}>
      <div className="ap-head">
        <div className="ap-text">
          <strong>{label}</strong>
          <span className="block-meta">{meta.join(' · ')}</span>
          {note === undefined ? null : <span className="block-meta">{note}</span>}
          {row.instruction === '' ? null : <span className="ap-instruction">{row.instruction}</span>}
        </div>
        {canEdit ? (
          <button
            type="button"
            className={`ap-switch${row.active ? ' is-on' : ''}`}
            role="switch"
            aria-checked={row.active}
            aria-label={`${row.active ? 'Turn off' : 'Turn on'} Autopilot for ${label}`}
            disabled={!live || saving}
            onClick={() => { void controller.setAutopilot(autopilotToggleInput(row, Date.now())) }}
          />
        ) : null}
      </div>
      {canEdit ? <Settings row={row} /> : null}
    </li>
  )
}

/** The Autopilot lines, for owners switchable and for everyone else read-only. */
export function AutopilotBody(): ReactElement {
  const { autopilot } = useHubState()
  const { controller } = useServices()
  const all = autopilot.all
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
          <ul className="ap-rows" aria-label="Workspaces">
            {all === undefined ? null : <Line key="*" row={all} label="All workspaces" note={AUTOPILOT_ALL_NOTE} />}
            {autopilot.rows.map(row => <Line key={row.workspaceId} row={row} label={row.label} />)}
          </ul>
          {autopilot.rows.length === 0 ? <p role="status" className="notice">No workspace is linked to this Hub yet.</p> : null}
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
