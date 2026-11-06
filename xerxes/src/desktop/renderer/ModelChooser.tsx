// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useState, type ReactElement } from 'react'

import type { SupervisorConfigInput, SupervisorOptionsInput } from '../../client/hubController.js'
import type { MachinesState, SupervisorConfigState } from '../../client/hubStore.js'
import { ChipChoice, ListChoice } from './XChoice.js'

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
  /** Inside the XERXES controls dialog: the dialog has the title and Close. */
  readonly embedded?: boolean
}

export function ModelChooser({ machines, config, onOptions, onSave, onOpen, onClose, embedded }: ModelChooserProps): ReactElement {
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
      {embedded === true ? null : (
        <div className="hubphone__row">
          <strong>XERXES model</strong>
          {onClose ? <button type="button" className="btn" onClick={onClose}>Close</button> : null}
        </div>
      )}
      {config.status === 'unconfigured' ? <p className="studio-muted">XERXES has no model yet. Choose a machine, a provider profile and a model.</p> : null}
      {!editable ? <p role="status" className="studio-muted">Only the Hub owner can change XERXES&apos;s model. This is what it uses now.</p> : null}
      {machines.error ? <p role="alert" className="action-card__error">{machines.error}</p> : null}
      {machines.phase === 'ready' && machines.list.length === 0 ? <p role="status" className="notice">No machine is linked to this Hub yet. Use “Link this Mac” in Settings → Hub &amp; phone.</p> : null}

      <div className="xfield">
        <span className="xfield__label">Machine</span>
        <ListChoice
          label="Machine"
          value={machineId}
          options={machines.list.map(machine => ({
            value: machine.id,
            label: machine.label,
            hint: [machine.health === 'online' ? '' : machine.health, machine.modelsSupported ? '' : 'cannot run models'].filter(Boolean).join(' · '),
            disabled: !machine.modelsSupported && machine.id !== machineId,
          }))}
          disabled={!editable}
          empty="No machine is linked yet."
          onChange={chooseMachine}
        />
      </div>
      <div className="xfield">
        <span className="xfield__label">Provider profile</span>
        <ListChoice
          label="Provider profile"
          value={profile}
          options={(forMachine?.profiles ?? []).map(item => ({ value: item.name, label: item.label, hint: item.provider }))}
          disabled={!editable || machineId === ''}
          empty={machineId === '' ? 'Choose a machine first.' : loading ? 'Asking the machine…' : 'This machine has no provider profile.'}
          onChange={chooseProfile}
        />
      </div>
      <div className="xfield">
        <span className="xfield__label">Model</span>
        <ListChoice
          label="Model"
          value={model}
          options={(forProfile?.models ?? []).map(item => ({ value: item.id, label: item.label === '' ? item.id : item.label, hint: item.label === '' || item.label === item.id ? '' : item.id }))}
          disabled={!editable || profile === ''}
          empty={profile === '' ? 'Choose a profile first.' : loading ? 'Asking the machine…' : 'This profile lists no models.'}
          onChange={chooseModel}
        />
      </div>
      <div className="xfield">
        <span className="xfield__label">Thinking effort</span>
        {model === '' ? <p className="xlist__empty">Choose a model first.</p> : (
          <ChipChoice
            label="Thinking effort"
            value={effort}
            options={[
              { value: '', label: forModel?.defaultEffort ? `Default (${forModel.defaultEffort})` : 'Default' },
              ...(forModel?.efforts ?? []).map(item => ({ value: item.effort, label: item.effort, ...(item.description ? { hint: item.description } : {}) })),
            ]}
            disabled={!editable}
            onChange={setEffort}
          />
        )}
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
