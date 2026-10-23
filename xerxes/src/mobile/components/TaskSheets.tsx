// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'

import type { ConfigField } from '../../client/hubStore.js'
import type { WorkspaceView } from '../../client/hubTypes.js'
import { configNote } from '../../client/taskView.js'
import { Sheet } from './Sheet.js'

function Status({ field, label }: { field: ConfigField; label: string }): ReactElement {
  const note = configNote(field)
  return (
    <>
      <p>{label}: <strong>{field.effective ?? 'not reported'}</strong></p>
      {field.requested !== undefined && field.requested !== field.effective ? <p>Requested: <strong>{field.requested}</strong></p> : null}
      {note === undefined ? null : <p role={field.phase === 'rejected' || field.phase === 'unknown' ? 'alert' : 'status'} className={field.phase === 'rejected' || field.phase === 'unknown' ? 'field-error' : 'notice'}>{note}</p>}
    </>
  )
}

export function ModelSheet({ workspace, field, canChange, onChoose, onClose }: {
  workspace: WorkspaceView | undefined
  field: ConfigField
  canChange: boolean
  onChoose: (profile: string | undefined, model: string) => void
  onClose: () => void
}): ReactElement {
  const [manual, setManual] = useState('')
  const choices = workspace?.models ?? []
  const busy = field.phase === 'requested'
  return (
    <Sheet title="Model" onClose={onClose} tall>
      <Status field={field} label="Current model" />
      {!canChange ? <p className="notice">Changing the model needs a live connection to this workspace.</p> : null}
      {choices.length > 0 ? (
        <ul className="choices">
          {choices.map(choice => (
            <li key={`${choice.profile}/${choice.model}`}>
              <button type="button" className="btn btn-segment" aria-pressed={field.effective === choice.model} disabled={!canChange || busy} onClick={() => onChoose(choice.profile, choice.model)}>{choice.label}</button>
            </li>
          ))}
        </ul>
      ) : <p className="notice">This workspace did not publish a list of models. You can type a model name it already knows.</p>}
      <form onSubmit={event => { event.preventDefault(); if (manual.trim() !== '') onChoose(undefined, manual.trim()) }}>
        <label className="field">
          <span>Model name</span>
          <input type="text" value={manual} onChange={event => setManual(event.target.value)} disabled={!canChange || busy} autoCapitalize="off" autoCorrect="off" />
        </label>
        <button type="submit" className="btn btn-primary" disabled={!canChange || busy || manual.trim() === ''}>Request this model</button>
      </form>
      <p className="notice">The change applies to this task only. It is shown as the current model once the task reports it.</p>
    </Sheet>
  )
}

export function EffortSheet({ workspace, field, onClose }: { workspace: WorkspaceView | undefined; field: ConfigField; onClose: () => void }): ReactElement {
  const efforts = workspace?.efforts ?? []
  return (
    <Sheet title="Reasoning effort" onClose={onClose}>
      <Status field={field} label="Current effort" />
      {efforts.length > 0 ? <p>This workspace supports: {efforts.join(', ')}.</p> : null}
      <p className="notice">The Hub has no command for changing reasoning effort from a phone yet, so it is shown here but cannot be changed. Change it on the desktop.</p>
    </Sheet>
  )
}

export function GoalSheet({ field, canChange, onSave, onClose }: { field: ConfigField & { readonly revision?: number }; canChange: boolean; onSave: (objective: string) => void; onClose: () => void }): ReactElement {
  const [draft, setDraft] = useState(field.effective ?? '')
  const busy = field.phase === 'requested'
  return (
    <Sheet title="Goal" onClose={onClose} tall>
      <Status field={field} label="Current goal" />
      {!canChange ? <p className="notice">Changing the goal needs a live connection to this workspace.</p> : null}
      <form onSubmit={event => { event.preventDefault(); if (draft.trim() !== '') onSave(draft.trim()) }}>
        <label className="field">
          <span>Goal</span>
          <textarea rows={5} value={draft} onChange={event => setDraft(event.target.value)} disabled={!canChange || busy} />
        </label>
        <button type="submit" className="btn btn-primary" disabled={!canChange || busy || draft.trim() === ''}>Save goal</button>
      </form>
    </Sheet>
  )
}
