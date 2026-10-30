// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import type { SupervisorChoice, SupervisorConfigStatus } from '../../client/hubTypes.js'
import { useHubState, useServices } from '../context.js'
import { Sheet } from './Sheet.js'

/**
 * XERXES's model, chosen from the phone: machine, then provider profile, then
 * model, then thinking effort. Every list comes from the Hub
 * (`supervisor.config.options`); nothing is typed in and no credential is shown.
 * Only the Hub owner can save; everyone else sees the current choice.
 */

export const READ_ONLY_NOTE = "Only the Hub owner can change XERXES's model."
export const NO_MACHINE_NOTE = "No linked machine can run XERXES's model yet. In the Xerxes desktop app, open Settings → Hub & phone and link a machine."

const STATUS_NOTE: Readonly<Partial<Record<SupervisorConfigStatus, string>>> = {
  machine_offline: "The machine that runs XERXES's model is offline. Pick another machine or wait for it to come back.",
  invalid: "The saved model choice no longer works. Choose again.",
}

function ReadOnlyChoice({ config, status, message }: { config: SupervisorChoice | undefined; status: string; message: string | undefined }): ReactElement {
  if (config === undefined) {
    return <p>XERXES has no model yet. The Hub owner chooses one from this screen.</p>
  }
  const note = message ?? STATUS_NOTE[status as SupervisorConfigStatus]
  return (
    <>
      <dl className="model-summary">
        <div><dt>Machine</dt><dd>{config.machineLabel}</dd></div>
        <div><dt>Provider profile</dt><dd>{config.profile}</dd></div>
        <div><dt>Model</dt><dd>{config.model}</dd></div>
        <div><dt>Thinking effort</dt><dd>{config.effort ?? 'Default'}</dd></div>
      </dl>
      {note === undefined ? null : <p role="status" className="block-meta">{note}</p>}
      <p className="block-meta">{READ_ONLY_NOTE}</p>
    </>
  )
}

function ModelEditor({ onSaved }: { onSaved: (() => void) | undefined }): ReactElement {
  const state = useHubState()
  const { controller } = useServices()
  const { machines, supervisorConfig: sc } = state
  const live = state.connection.state === 'live'
  const candidates = machines.list.filter(machine => machine.modelsSupported)
  const current = sc.config
  const loaded = sc.options.options

  const [machineId, setMachineId] = useState(() => current?.machineId ?? loaded?.machineId ?? '')
  const [profile, setProfile] = useState(() => current?.profile ?? loaded?.profile ?? '')
  const [model, setModel] = useState(() => current?.model ?? loaded?.model ?? '')
  const [effort, setEffort] = useState(() => current?.effort ?? '')

  useEffect(() => {
    if (live && machines.phase === 'idle') void controller.machinesList()
  }, [controller, live, machines.phase])

  // Preselect a machine once the list is known: the first one that is online.
  useEffect(() => {
    if (machineId !== '') return
    const pick = candidates.find(machine => machine.health === 'online') ?? candidates[0]
    if (pick !== undefined) setMachineId(pick.id)
  }, [machineId, candidates])

  // Ask the Hub what this machine offers when we do not have it yet.
  const offered = loaded !== undefined && loaded.machineId === machineId ? loaded : undefined
  useEffect(() => {
    if (!live || machineId === '' || offered !== undefined || sc.options.phase === 'loading' || sc.options.phase === 'error') return
    void controller.supervisorConfigOptions({ machineId, ...(profile === '' ? {} : { profile }), ...(model === '' ? {} : { model }) })
    // eslint-disable-next-line react-hooks/exhaustive-deps -- profile/model only seed the first question
  }, [controller, live, machineId, offered, sc.options.phase])

  const profiles = offered?.profiles ?? []
  const profileValue = profiles.some(item => item.name === profile) ? profile : offered?.profile ?? ''
  const modelIds = offered?.models.map(item => item.id) ?? []
  const modelValue = modelIds.includes(model) ? model : offered?.model ?? ''
  const efforts = offered?.efforts ?? []
  const effortValue = efforts.some(item => item.effort === effort) ? effort : ''
  const effortInfo = efforts.find(item => item.effort === effortValue)
  const loading = sc.options.phase === 'loading'

  const pick = (next: { machine?: string; profile?: string; model?: string }): void => {
    const nextMachine = next.machine ?? machineId
    const nextProfile = next.profile ?? profileValue
    setEffort('')
    if (next.machine !== undefined) { setMachineId(next.machine); setProfile(''); setModel('') }
    else if (next.profile !== undefined) { setProfile(next.profile); setModel('') }
    else if (next.model !== undefined) setModel(next.model)
    if (nextMachine === '') return
    void controller.supervisorConfigOptions({
      machineId: nextMachine,
      ...(next.machine !== undefined || nextProfile === '' ? {} : { profile: nextProfile }),
      ...(next.model === undefined ? {} : { model: next.model }),
    })
  }

  const canSave = live && !sc.saving && !loading && machineId !== '' && profileValue !== '' && modelValue !== ''
  const save = async (): Promise<void> => {
    if (!canSave) return
    const ok = await controller.supervisorConfigSet({ machineId, profile: profileValue, model: modelValue, effort: effortValue === '' ? null : effortValue })
    if (ok) onSaved?.()
  }

  if (machines.phase === 'error' && candidates.length === 0) {
    return (
      <div role="alert" className="notice is-error">
        <p>Could not load the machines: {machines.error ?? 'unknown error'}</p>
        <button type="button" className="btn" disabled={!live} onClick={() => { void controller.machinesList() }}>Try again</button>
      </div>
    )
  }

  if (machines.phase === 'ready' && candidates.length === 0 && current === undefined) {
    return <p>{NO_MACHINE_NOTE}</p>
  }

  const note = sc.message ?? STATUS_NOTE[sc.status as SupervisorConfigStatus]
  return (
    <form className="model-form" aria-label="XERXES model" aria-busy={loading} onSubmit={event => { event.preventDefault(); void save() }}>
      {note === undefined || sc.status === 'unconfigured' ? null : <p role="status" className="block-meta">{note}</p>}
      <label className="field">
        <span>Machine</span>
        <select value={machineId} onChange={event => pick({ machine: event.target.value })}>
          {machineId === '' ? <option value="">Choose a machine</option> : null}
          {machineId !== '' && !candidates.some(machine => machine.id === machineId) ? <option value={machineId}>{current?.machineLabel ?? machineId}</option> : null}
          {candidates.map(machine => <option key={machine.id} value={machine.id}>{machine.label}{machine.health === 'online' ? '' : ` (${machine.health})`}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Provider profile</span>
        <select value={profileValue} disabled={offered === undefined} onChange={event => pick({ profile: event.target.value })}>
          {profileValue === '' ? <option value="">{loading ? 'Loading…' : 'Choose a provider profile'}</option> : null}
          {profiles.map(item => <option key={item.name} value={item.name}>{item.label}{item.provider !== '' && item.provider !== item.label ? ` · ${item.provider}` : ''}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Model</span>
        <select value={modelValue} disabled={offered === undefined || profileValue === ''} onChange={event => pick({ model: event.target.value })}>
          {modelValue === '' ? <option value="">{loading ? 'Loading…' : 'Choose a model'}</option> : null}
          {offered?.models.map(item => <option key={item.id} value={item.id}>{item.label}</option>)}
        </select>
      </label>
      <label className="field">
        <span>Thinking effort</span>
        <select value={effortValue} disabled={offered === undefined || modelValue === ''} onChange={event => setEffort(event.target.value)}>
          <option value="">{offered?.defaultEffort ? `Default (${offered.defaultEffort})` : 'Default'}</option>
          {efforts.map(item => <option key={item.effort} value={item.effort}>{item.effort}</option>)}
        </select>
        {effortInfo?.description === undefined ? null : <span className="block-meta">{effortInfo.description}</span>}
      </label>
      {sc.options.phase === 'error' ? (
        <div role="alert" className="notice is-error">
          <p>Could not load the choices: {sc.options.error ?? 'unknown error'}</p>
          <button type="button" className="btn" onClick={() => { if (machineId !== '') void controller.supervisorConfigOptions({ machineId }) }}>Try again</button>
        </div>
      ) : null}
      {sc.saveError === undefined ? null : <p role="alert" className="field-error">{sc.saveError}</p>}
      {live ? null : <p className="block-meta">Saving needs a live connection.</p>}
      <div className="composer-row">
        <button type="submit" className="btn btn-primary" disabled={!canSave}>{sc.saving ? 'Saving…' : 'Save'}</button>
      </div>
    </form>
  )
}

/** The chooser in a modal sheet, opened from the Xerxes tab's Model button. */
export function ModelSheet({ onClose }: { onClose: () => void }): ReactElement {
  return (
    <Sheet title="XERXES model" onClose={onClose}>
      <ModelChooser onSaved={onClose} />
    </Sheet>
  )
}

/** The chooser for owners, the current choice for everyone else. */
export function ModelChooser({ onSaved }: { onSaved?: () => void }): ReactElement {
  const { controller } = useServices()
  const { supervisorConfig: sc } = useHubState()
  if (sc.phase === 'error' && sc.status === 'unknown') {
    return (
      <div role="alert" className="notice is-error">
        <p>Could not read XERXES's model: {sc.error ?? 'unknown error'}</p>
        <button type="button" className="btn" onClick={() => { void controller.supervisorConfigGet() }}>Try again</button>
      </div>
    )
  }
  if (sc.status === 'unknown' && sc.phase !== 'ready') return <p aria-busy="true">Loading…</p>
  if (!sc.canEdit) return <ReadOnlyChoice config={sc.config} status={sc.status} message={sc.message} />
  return <ModelEditor onSaved={onSaved} />
}
