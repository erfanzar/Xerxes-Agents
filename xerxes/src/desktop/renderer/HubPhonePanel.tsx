// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from 'react'

import { displayOrigin, type CockpitMachine, type CockpitPairing, type CockpitStatus } from '../hubShared.js'
import { CopyButton } from './CopyButton.js'
import { HUB_CHANGED_EVENT, cockpitCall, windowCockpitApi } from './hubClient.js'

/**
 * Settings → "Hub & phone". One field (the Hub's SSH address) and three
 * buttons: link this Mac, pair a phone, confirm it. Every action runs in the
 * desktop's main process; this page only shows what it reports. Nothing here
 * ever holds a key, a cookie or a token.
 */

export type LinkState =
  | { readonly phase: 'idle' }
  | { readonly phase: 'running'; readonly lines: readonly string[] }
  | { readonly phase: 'error'; readonly message: string }

export interface MachinesView {
  readonly phase: 'idle' | 'loading' | 'ready' | 'error'
  readonly machines: readonly CockpitMachine[]
  readonly error?: string
}

export type PairState =
  | { readonly phase: 'idle' }
  | { readonly phase: 'creating' }
  | { readonly phase: 'ready'; readonly pairing: CockpitPairing; readonly confirming: boolean; readonly error?: string }
  | { readonly phase: 'confirmed'; readonly deviceId: string }
  | { readonly phase: 'error'; readonly message: string }

export const EMPTY_STATUS: CockpitStatus = {
  target: null, origin: null, wsUrl: null, fingerprint: null, hubVersion: null, linkedMachineId: null, hasSession: false,
}

const HEALTH_TEXT: Readonly<Record<string, string>> = {
  online: 'Online',
  connecting: 'Connecting',
  stale: 'Not responding',
  offline: 'Offline',
  needs_operator: 'Needs attention',
}

export function healthText(health: string): string {
  return HEALTH_TEXT[health] ?? 'Unknown'
}

export interface HubPhoneViewProps {
  /** False outside the desktop app (no cockpit bridge). */
  readonly available: boolean
  /** `undefined` while the saved address is being read. */
  readonly status: CockpitStatus | undefined
  readonly address: string
  readonly onAddress: (value: string) => void
  readonly saving: boolean
  readonly saveError?: string
  readonly onSave: () => void
  readonly onForget: () => void
  readonly link: LinkState
  readonly onLink: () => void
  readonly machines: MachinesView
  readonly onRefreshMachines: () => void
  readonly pair: PairState
  readonly onPair: () => void
  readonly onConfirm: (pairingId: string, code: string) => void
}

function MachineList({ view, onRefresh }: { readonly view: MachinesView; readonly onRefresh: () => void }): ReactElement {
  return (
    <div className="hubphone__machines" aria-label="Machines on this Hub">
      <div className="hubphone__row">
        <strong>Machines and projects</strong>
        <button type="button" className="btn" disabled={view.phase === 'loading'} onClick={onRefresh}>{view.phase === 'loading' ? 'Refreshing…' : 'Refresh'}</button>
      </div>
      {view.error ? <p role="alert" className="action-card__error">{view.error}</p> : null}
      {view.phase === 'ready' && view.machines.length === 0 ? <p className="studio-muted">The Hub has no machines yet.</p> : null}
      <ul className="hubphone__list">
        {view.machines.map(machine => (
          <li key={machine.id}>
            <p className="hubphone__machine">{machine.label}{machine.discover ? ' · all projects' : ''}</p>
            {machine.projects.length === 0 ? <p className="studio-muted">No projects found yet. They appear as soon as a task runs there.</p> : (
              <ul className="hubphone__projects">
                {machine.projects.map(project => (
                  <li key={project.workspaceId}>
                    <span>{project.label}</span>
                    <span className={`hubphone__health hubphone__health--${project.health}`}>{healthText(project.health)}</span>
                  </li>
                ))}
              </ul>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}

function PairingBlock({ pairing, confirming, error, onConfirm }: {
  readonly pairing: CockpitPairing
  readonly confirming: boolean
  readonly error?: string
  readonly onConfirm: (pairingId: string, code: string) => void
}): ReactElement {
  return (
    <div className="hubphone__pair" aria-label="Pair a phone">
      {pairing.qrDataUrl !== null
        ? <img className="hubphone__qr" src={pairing.qrDataUrl} alt="QR code for pairing the phone" width={192} height={192} />
        : <p className="studio-muted">No QR code was made. Open the link below on the phone instead.</p>}
      <div className="field">
        <label htmlFor="hub-pair-link">Pairing link</label>
        <input id="hub-pair-link" type="text" readOnly value={pairing.url} spellCheck={false} onFocus={event => event.currentTarget.select()} />
        <CopyButton text={pairing.url} label="Copy link" />
      </div>
      <p>Comparison code <strong className="hubphone__code">{pairing.comparisonCode}</strong></p>
      <p className="studio-muted">Open the link on the phone (or scan the QR code). If the phone shows the same code, confirm it here. The link works once and expires.</p>
      {pairing.confirmRequired
        ? <button type="button" className="btn btn--solid" disabled={confirming} onClick={() => onConfirm(pairing.pairingId, pairing.comparisonCode)}>{confirming ? 'Confirming…' : `The phone shows ${pairing.comparisonCode}`}</button>
        : <p role="status">No confirmation is needed. The phone is ready once it opens the link.</p>}
      {error ? <p role="alert" className="action-card__error">{error}</p> : null}
    </div>
  )
}

export function HubPhoneView(props: HubPhoneViewProps): ReactElement {
  const { available, status, link, pair, machines } = props
  return (
    <section className="hubphone" aria-labelledby="hubphone-title">
      <h3 id="hubphone-title">Hub &amp; phone</h3>
      <p className="studio-muted">Your Hub is the always-on server that lets your phone and the XERXES tab reach every machine. Enter its SSH address once and this app does the rest.</p>
      {!available ? (
        <p role="status" className="notice">Hub &amp; phone is only available in the Xerxes desktop app.</p>
      ) : status === undefined ? (
        <p aria-busy="true">Reading the saved Hub…</p>
      ) : (
        <>
          <form className="hubphone__form" onSubmit={event => { event.preventDefault(); props.onSave() }}>
            <div className="field">
              <label htmlFor="hub-ssh-address">Hub SSH address</label>
              <input
                id="hub-ssh-address" type="text" value={props.address} placeholder="xerxes@209.38.102.184"
                spellCheck={false} autoComplete="off" autoCapitalize="off" disabled={props.saving}
                onChange={event => props.onAddress(event.target.value)}
              />
            </div>
            <div className="hubphone__row">
              <button type="submit" className="btn btn--solid" disabled={props.saving || props.address.trim() === ''}>
                {props.saving ? 'Checking the Hub…' : status.target === null ? 'Save and connect' : 'Save'}
              </button>
              {status.target !== null ? <button type="button" className="btn" disabled={props.saving} onClick={props.onForget}>Forget this Hub</button> : null}
            </div>
            {props.saveError ? <p role="alert" className="action-card__error">{props.saveError}</p> : null}
          </form>

          {status.target === null || status.origin === null ? (
            <p className="studio-muted">No Hub is saved on this Mac yet.</p>
          ) : (
            <>
              <dl className="hubphone__facts">
                <dt>Web address</dt><dd>{displayOrigin(status.origin)}</dd>
                {status.hubVersion !== null ? <><dt>Version</dt><dd>{status.hubVersion}</dd></> : null}
                {status.fingerprint !== null ? <><dt>Key</dt><dd className="hubphone__mono">{status.fingerprint}</dd></> : null}
              </dl>

              <div className="hubphone__block">
                <div className="hubphone__row">
                  <div>
                    <strong>This Mac</strong>
                    <p className="studio-muted">{status.linkedMachineId !== null ? `Linked to the Hub as ${status.linkedMachineId}. Every project on this Mac is connected automatically.` : 'Not linked yet. Linking connects every project on this Mac to the Hub.'}</p>
                  </div>
                  <button type="button" className="btn btn--solid" disabled={link.phase === 'running'} onClick={props.onLink}>
                    {link.phase === 'running' ? 'Linking…' : status.linkedMachineId !== null ? 'Link this Mac again' : 'Link this Mac'}
                  </button>
                </div>
                {link.phase === 'running' ? (
                  <ul className="hubphone__progress" role="status" aria-label="Link progress">
                    {link.lines.length === 0 ? <li>Starting…</li> : link.lines.map((line, index) => <li key={`${index}:${line}`}>{line}</li>)}
                  </ul>
                ) : null}
                {link.phase === 'error' ? <p role="alert" className="action-card__error">{link.message}</p> : null}
                {status.linkedMachineId !== null || machines.machines.length > 0 || machines.error ? <MachineList view={machines} onRefresh={props.onRefreshMachines} /> : null}
              </div>

              <div className="hubphone__block">
                <div className="hubphone__row">
                  <div>
                    <strong>Phone</strong>
                    <p className="studio-muted">Pair a phone to follow tasks and talk to XERXES from anywhere.</p>
                  </div>
                  <button type="button" className="btn btn--solid" disabled={pair.phase === 'creating'} onClick={props.onPair}>
                    {pair.phase === 'creating' ? 'Creating link…' : pair.phase === 'idle' ? 'Pair a phone' : 'New pairing link'}
                  </button>
                </div>
                {pair.phase === 'ready' ? <PairingBlock pairing={pair.pairing} confirming={pair.confirming} {...(pair.error === undefined ? {} : { error: pair.error })} onConfirm={props.onConfirm} /> : null}
                {pair.phase === 'confirmed' ? <p role="status" className="notice">Phone paired. It can open the Hub now.</p> : null}
                {pair.phase === 'error' ? <p role="alert" className="action-card__error">{pair.message}</p> : null}
              </div>
            </>
          )}
        </>
      )}
    </section>
  )
}

/** The panel wired to the desktop's main process. */
export function HubPhoneCard(): ReactElement {
  const api = useMemo(() => windowCockpitApi(), [])
  const [status, setStatus] = useState<CockpitStatus | undefined>(undefined)
  const [address, setAddress] = useState('')
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | undefined>(undefined)
  const [link, setLink] = useState<LinkState>({ phase: 'idle' })
  const [machines, setMachines] = useState<MachinesView>({ phase: 'idle', machines: [] })
  const [pair, setPair] = useState<PairState>({ phase: 'idle' })
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => { alive.current = false }
  }, [])

  const loadMachines = useCallback(async (): Promise<void> => {
    if (api === undefined) return
    setMachines(previous => ({ phase: 'loading', machines: previous.machines }))
    const result = await cockpitCall(api, 'machines')
    if (!alive.current) return
    setMachines(result.ok ? { phase: 'ready', machines: result.machines } : { phase: 'error', machines: [], error: result.message })
  }, [api])

  useEffect(() => {
    if (api === undefined) return undefined
    let cancelled = false
    void cockpitCall(api, 'status').then(result => {
      if (cancelled) return
      if (result.ok) {
        setStatus(result.status)
        setAddress(result.status.target ?? '')
        if (result.status.linkedMachineId !== null) void loadMachines()
      } else {
        setStatus(EMPTY_STATUS)
        setSaveError(result.message)
      }
    })
    return () => { cancelled = true }
  }, [api, loadMachines])

  useEffect(() => api?.onProgress(progress => {
    setLink(previous => (previous.phase === 'running' ? { phase: 'running', lines: [...previous.lines, progress.line].slice(-6) } : previous))
  }), [api])

  const save = async (): Promise<void> => {
    if (api === undefined) return
    setSaving(true)
    setSaveError(undefined)
    const result = await cockpitCall(api, 'save', { target: address })
    if (!alive.current) return
    setSaving(false)
    if (!result.ok) {
      setSaveError(result.message)
      return
    }
    setStatus(result.status)
    setAddress(result.status.target ?? address)
    setLink({ phase: 'idle' })
    setPair({ phase: 'idle' })
    if (result.status.linkedMachineId !== null) void loadMachines()
    else setMachines({ phase: 'idle', machines: [] })
    window.dispatchEvent(new Event(HUB_CHANGED_EVENT))
  }

  const forget = async (): Promise<void> => {
    if (api === undefined) return
    const result = await cockpitCall(api, 'forget')
    if (!alive.current) return
    if (!result.ok) {
      setSaveError(result.message)
      return
    }
    setStatus(result.status)
    setAddress('')
    setSaveError(undefined)
    setLink({ phase: 'idle' })
    setPair({ phase: 'idle' })
    setMachines({ phase: 'idle', machines: [] })
    window.dispatchEvent(new Event(HUB_CHANGED_EVENT))
  }

  const linkMac = async (): Promise<void> => {
    if (api === undefined || status?.target === null || status === undefined) return
    setLink({ phase: 'running', lines: [] })
    const result = await cockpitCall(api, 'link', { target: status.target })
    if (!alive.current) return
    if (!result.ok) {
      setLink({ phase: 'error', message: result.message })
      return
    }
    setStatus(result.status)
    setLink({ phase: 'idle' })
    setMachines(result.machinesError === null ? { phase: 'ready', machines: result.machines } : { phase: 'error', machines: [], error: result.machinesError })
  }

  const pairPhone = async (): Promise<void> => {
    if (api === undefined) return
    setPair({ phase: 'creating' })
    const result = await cockpitCall(api, 'pair', { label: 'Phone' })
    if (!alive.current) return
    setPair(result.ok ? { phase: 'ready', pairing: result.pairing, confirming: false } : { phase: 'error', message: result.message })
  }

  const confirm = async (pairingId: string, code: string): Promise<void> => {
    if (api === undefined) return
    setPair(previous => (previous.phase === 'ready' ? { phase: 'ready', pairing: previous.pairing, confirming: true } : previous))
    const result = await cockpitCall(api, 'confirm', { pairingId, code })
    if (!alive.current) return
    setPair(previous => {
      if (result.ok) return { phase: 'confirmed', deviceId: result.deviceId }
      return previous.phase === 'ready' ? { phase: 'ready', pairing: previous.pairing, confirming: false, error: result.message } : previous
    })
  }

  return (
    <HubPhoneView
      available={api !== undefined}
      status={api === undefined ? EMPTY_STATUS : status}
      address={address}
      onAddress={setAddress}
      saving={saving}
      {...(saveError === undefined ? {} : { saveError })}
      onSave={() => { void save() }}
      onForget={() => { void forget() }}
      link={link}
      onLink={() => { void linkMac() }}
      machines={machines}
      onRefreshMachines={() => { void loadMachines() }}
      pair={pair}
      onPair={() => { void pairPhone() }}
      onConfirm={(id, code) => { void confirm(id, code) }}
    />
  )
}
