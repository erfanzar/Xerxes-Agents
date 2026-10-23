// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import type { SupervisorConfigInput, SupervisorOptionsInput } from '../../client/hubController.js'
import type { MachinesState, SupervisorConfigState } from '../../client/hubStore.js'

/**
 * Which model XERXES the supervisor runs on: machine, then provider profile,
 * then model, then thinking effort. Each list is filled by the Hub from the
 * chosen machine's own provider profiles; no key or base URL is ever shown.
 * Only the Hub owner can save; everyone else sees the current choice.
 */

export interface ModelChooserProps {
  readonly machines: MachinesState
  readonly config: SupervisorConfigState
  readonly onOptions: (input: SupervisorOptionsInput) => void
  readonly onSave: (input: SupervisorConfigInput) => Promise<boolean>
  /** Called once when the chooser appears, to refresh machines and the saved choice. */
  readonly onOpen?: () => void
  /** Absent while XERXES has no model yet: there is nothing to go back to. */
  readonly onClose?: () => void
}

export function ModelChooser({ machines, config, onOptions, onSave, onOpen, onClose }: ModelChooserProps): ReactElement {
  const only = machines.list.length === 1 ? machines.list[0]!.id : ''
  const [machineId, setMachineId] = useState(config.config?.machineId ?? only)
  const [profile, setProfile] = useState(config.config?.profile ?? '')
  const [model, setModel] = useState(config.config?.model ?? '')
  const [effort, setEffort] = useState(config.config?.effort ?? '')
  const editable = config.canEdit

  useEffect(() => { onOpen?.() }, [])
  // A single machine needs no choosing; it may arrive after the chooser opens.
  useEffect(() => { if (machineId === '' && only !== '') setMachineId(only) }, [machineId, only])
  useEffect(() => {
    if (machineId === '') return
    onOptions({ machineId, ...(profile === '' ? {} : { profile }), ...(model === '' ? {} : { model }) })
  }, [machineId])

  const options = config.options.options
  const forMachine = options !== undefined && options.machineId === machineId ? options : undefined
  const forProfile = forMachine !== undefined && forMachine.profile !== undefined && forMachine.profile === profile ? forMachine : undefined
  const forModel = forProfile !== undefined && forProfile.model !== undefined && forProfile.model === model ? forProfile : undefined
  useEffect(() => {
    if (forMachine?.profile !== undefined && profile === '') setProfile(forMachine.profile)
  }, [forMachine, profile])
  useEffect(() => {
    if (forProfile?.model !== undefined && model === '') setModel(forProfile.model)
  }, [forProfile, model])

  const loading = config.options.phase === 'loading'
  const ready = machineId !== '' && profile !== '' && model !== ''

  const chooseMachine = (id: string): void => {
    setMachineId(id)
    setProfile('')
    setModel('')
    setEffort('')
  }
  const chooseProfile = (name: string): void => {
    setProfile(name)
    setModel('')
    setEffort('')
    if (name !== '') onOptions({ machineId, profile: name })
  }
  const chooseModel = (id: string): void => {
    setModel(id)
    setEffort('')
    if (id !== '') onOptions({ machineId, profile, model: id })
  }
  const save = async (): Promise<void> => {
    if (!ready || !editable) return
    if (await onSave({ machineId, profile, model, effort: effort === '' ? null : effort })) onClose?.()
  }

  return (
    <form className="modelchooser" aria-label="XERXES model" aria-busy={loading} onSubmit={event => { event.preventDefault(); void save() }}>
      <div className="hubphone__row">
        <strong>XERXES model</strong>
        {onClose ? <button type="button" className="btn" onClick={onClose}>Close</button> : null}
      </div>
      {config.status === 'unconfigured' ? <p className="studio-muted">XERXES has no model yet. Choose a machine, a provider profile and a model.</p> : null}
      {!editable ? <p role="status" className="studio-muted">Only the Hub owner can change XERXES&apos;s model. This is what it uses now.</p> : null}
      {machines.error ? <p role="alert" className="action-card__error">{machines.error}</p> : null}
      {machines.phase === 'ready' && machines.list.length === 0 ? <p role="status" className="notice">No machine is linked to this Hub yet. Use “Link this Mac” in Settings → Hub &amp; phone.</p> : null}

      <div className="field">
        <label htmlFor="xerxes-model-machine">Machine</label>
        <select id="xerxes-model-machine" value={machineId} disabled={!editable} onChange={event => chooseMachine(event.target.value)}>
          <option value="">Choose a machine…</option>
          {machines.list.map(machine => (
            <option key={machine.id} value={machine.id} disabled={!machine.modelsSupported && machine.id !== machineId}>
              {`${machine.label}${machine.health === 'online' ? '' : ` (${machine.health})`}${machine.modelsSupported ? '' : ' — cannot run models'}`}
            </option>
          ))}
        </select>
      </div>
      <div className="field">
        <label htmlFor="xerxes-model-profile">Provider profile</label>
        <select id="xerxes-model-profile" value={profile} disabled={!editable || machineId === ''} onChange={event => chooseProfile(event.target.value)}>
          <option value="">Choose a profile…</option>
          {(forMachine?.profiles ?? []).map(item => <option key={item.name} value={item.name}>{`${item.label} · ${item.provider}`}</option>)}
        </select>
      </div>
      <div className="field">
        <label htmlFor="xerxes-model-model">Model</label>
        <select id="xerxes-model-model" value={model} disabled={!editable || profile === ''} onChange={event => chooseModel(event.target.value)}>
          <option value="">Choose a model…</option>
          {(forProfile?.models ?? []).map(item => <option key={item.id} value={item.id}>{item.label === '' ? item.id : item.label}</option>)}
        </select>
      </div>
      <div className="field">
        <label htmlFor="xerxes-model-effort">Thinking effort</label>
        <select id="xerxes-model-effort" value={effort} disabled={!editable || model === ''} onChange={event => setEffort(event.target.value)}>
          <option value="">{forModel?.defaultEffort ? `Provider default (${forModel.defaultEffort})` : 'Provider default'}</option>
          {(forModel?.efforts ?? []).map(item => <option key={item.effort} value={item.effort}>{item.description ? `${item.effort} — ${item.description}` : item.effort}</option>)}
        </select>
      </div>

      {loading ? <p role="status" className="studio-muted">Asking the machine what it offers…</p> : null}
      {config.options.phase === 'error' ? <p role="alert" className="action-card__error">{config.options.error ?? 'Could not load the choices.'}</p> : null}
      {config.saveError ? <p role="alert" className="action-card__error">{config.saveError}</p> : null}
      <div className="hubphone__row">
        <button type="submit" className="btn btn--solid" disabled={!editable || !ready || config.saving}>{config.saving ? 'Saving…' : 'Save'}</button>
      </div>
    </form>
  )
}
