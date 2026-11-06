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
import { ChipChoice } from './XChoice.js'

/**
 * The Autopilot tab of the XERXES controls dialog. On top, one card for "All
 * workspaces" (the 24/7 switch, and what it may do when it wakes). Below, one
 * line per workspace: a status dot, its state, a switch that saves at once and
 * an Edit button that opens that workspace's settings in place. Everyone but
 * the Hub owner sees the same lines read-only. `active` is the Hub's verdict.
 */

export interface AutopilotPanelProps {
  readonly autopilot: AutopilotState
  /** False while the Hub link is down: switches and Save are disabled. */
  readonly live: boolean
  readonly onSave: (input: AutopilotInput) => Promise<boolean>
  readonly onRetry: () => void
}

function Toggle({ row, label, disabled, onSave }: {
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

/** Watch only | Act, as a segmented control. */
export function ModeSeg({ value, disabled, label, onChange }: {
  readonly value: AutopilotMode
  readonly disabled: boolean
  readonly label: string
  readonly onChange: (mode: AutopilotMode) => void
}): ReactElement {
  return (
    <div className="seg xap__seg" role="radiogroup" aria-label={label}>
      {AUTOPILOT_MODE_OPTIONS.map(option => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          title={option.label}
          className={value === option.value ? 'is-on' : undefined}
          disabled={disabled}
          onClick={() => { if (value !== option.value) onChange(option.value) }}
        >{option.short}</button>
      ))}
    </div>
  )
}

function rowMeta(row: AutopilotRow): string {
  return [
    autopilotStateText(row),
    row.active && row.expiresAt !== null && row.viaAll !== true ? `until ${new Date(row.expiresAt).toLocaleString()}` : undefined,
    row.enabledBy === null || row.viaAll === true ? undefined : `Turned on by ${row.enabledBy.name}.`,
  ].filter((part): part is string => part !== undefined).join(' · ')
}

/** One workspace's settings, opened in place by Edit. */
export function AutopilotEditor({ row, live, saving, onSave, onDone }: {
  readonly row: AutopilotRow
  readonly live: boolean
  readonly saving: boolean
  readonly onSave: (input: AutopilotInput) => Promise<boolean>
  readonly onDone: () => void
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
    if (ok) onDone()
  }
  return (
    <form className="xap__editor" aria-label={`Autopilot settings for ${row.label}`} onSubmit={event => { event.preventDefault(); void save() }}>
      <div className="xap__grid">
        <span className="xap__label">When it wakes</span>
        <ModeSeg value={mode} disabled={!live || saving} label={`Mode for ${row.label}`} onChange={setMode} />
        <span className="xap__label">Stop</span>
        <ChipChoice label={`When Autopilot stops for ${row.label}`} value={choice} options={autopilotExpiryOptions(row, nowMs)} disabled={!live || saving} onChange={setExpiry} />
        <label className="xap__label" htmlFor={`${id}-instruction`}>Instruction</label>
        <textarea
          id={`${id}-instruction`}
          rows={3}
          maxLength={AUTOPILOT_INSTRUCTION_LIMIT}
          value={instruction}
          placeholder={AUTOPILOT_INSTRUCTION_PLACEHOLDER}
          onChange={event => setInstruction(event.target.value)}
        />
      </div>
      {lapsed ? <p role="status" className="xap__meta">{AUTOPILOT_EXPIRY_LAPSED_NOTE}</p> : null}
      <div className="xap__actions">
        <span className="xap__meta">{instruction.length}/{AUTOPILOT_INSTRUCTION_LIMIT}</span>
        <span className="xap__flex" />
        <button type="button" className="btn" onClick={onDone}>Cancel</button>
        <button type="submit" className="btn btn--solid" disabled={!live || saving || tooLong}>{saving ? 'Saving…' : 'Save'}</button>
      </div>
    </form>
  )
}

function AllCard({ row, canEdit, live, saving, onSave }: {
  readonly row: AutopilotRow
  readonly canEdit: boolean
  readonly live: boolean
  readonly saving: boolean
  readonly onSave: (input: AutopilotInput) => Promise<boolean>
}): ReactElement {
  return (
    <div className={`xap__all${row.active ? ' is-on' : ''}`} role="group" aria-label="Autopilot for All workspaces">
      <div className="xap__line">
        <span className={`xap__dot${row.active ? ' is-on' : ''}`} aria-hidden="true" />
        <div className="xap__text">
          <strong>All workspaces</strong>
          <span className="xap__meta">{AUTOPILOT_ALL_NOTE}</span>
        </div>
        {canEdit ? <Toggle row={row} label="All workspaces" disabled={!live || saving} onSave={onSave} /> : <span className="xap__state">{autopilotStateText(row)}</span>}
      </div>
      {canEdit ? (
        <div className="xap__allmode">
          <span className="xap__label">When it wakes</span>
          <ModeSeg
            value={row.mode}
            disabled={!live || saving}
            label="Mode for All workspaces"
            onChange={mode => { void onSave({ workspaceId: row.workspaceId, enabled: row.enabled, mode }) }}
          />
        </div>
      ) : null}
    </div>
  )
}

function Line({ row, canEdit, live, saving, editing, onEdit, onSave }: {
  readonly row: AutopilotRow
  readonly canEdit: boolean
  readonly live: boolean
  readonly saving: boolean
  readonly editing: boolean
  readonly onEdit: (open: boolean) => void
  readonly onSave: (input: AutopilotInput) => Promise<boolean>
}): ReactElement {
  return (
    <li className={`xap__row${row.active ? ' is-on' : ''}${editing ? ' is-editing' : ''}`} aria-label={`Autopilot for ${row.label}`}>
      <div className="xap__line">
        <span className={`xap__dot${row.active ? ' is-on' : ''}`} aria-hidden="true" />
        <div className="xap__text">
          <strong>{row.label}</strong>
          <span className="xap__meta">{rowMeta(row)}</span>
          {row.instruction === '' || editing ? null : <span className="xap__instruction">{row.instruction}</span>}
        </div>
        {canEdit ? (
          <>
            <button type="button" className="btn xap__edit" aria-expanded={editing} onClick={() => onEdit(!editing)}>Edit</button>
            <Toggle row={row} label={row.label} disabled={!live || saving} onSave={onSave} />
          </>
        ) : null}
      </div>
      {canEdit && editing ? <AutopilotEditor row={row} live={live} saving={saving} onSave={onSave} onDone={() => onEdit(false)} /> : null}
    </li>
  )
}

export function AutopilotPanel({ autopilot, live, onSave, onRetry }: AutopilotPanelProps): ReactElement {
  const [editing, setEditing] = useState<string | undefined>(undefined)
  const all = autopilot.all
  return (
    <section className="xap" aria-label="Autopilot">
      <p className="modal__sub">{AUTOPILOT_EXPLANATION}</p>
      {autopilot.phase === 'loading' && autopilot.rows.length === 0 ? <p role="status" aria-busy="true" className="xap__meta">Reading Autopilot settings…</p> : null}
      {autopilot.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not read Autopilot: {autopilot.error ?? 'unknown error'}</p>
          <button type="button" className="btn" onClick={onRetry}>Try again</button>
        </div>
      ) : null}
      {autopilot.phase === 'ready' || autopilot.rows.length > 0 ? (
        <>
          {!autopilot.canEdit ? <p role="status" className="xap__meta">{AUTOPILOT_READ_ONLY_NOTE}</p> : null}
          {all === undefined ? null : (
            <AllCard row={all} canEdit={autopilot.canEdit} live={live} saving={autopilot.savingId === all.workspaceId} onSave={onSave} />
          )}
          <div className="cap xap__cap">Workspaces</div>
          {autopilot.rows.length === 0 ? <p role="status" className="notice">No workspace is linked to this Hub yet.</p> : (
            <ul className="xap__list" aria-label="Workspaces">
              {autopilot.rows.map(row => (
                <Line
                  key={row.workspaceId}
                  row={row}
                  canEdit={autopilot.canEdit}
                  live={live}
                  saving={autopilot.savingId === row.workspaceId}
                  editing={editing === row.workspaceId}
                  onEdit={open => setEditing(open ? row.workspaceId : undefined)}
                  onSave={onSave}
                />
              ))}
            </ul>
          )}
        </>
      ) : null}
      {autopilot.saveError === undefined ? null : <p role="alert" className="action-card__error">{autopilot.saveError}</p>}
    </section>
  )
}
