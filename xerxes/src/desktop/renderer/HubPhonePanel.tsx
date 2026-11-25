// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useCallback, useEffect, useMemo, useRef, useState, type ReactElement } from 'react'

import { relativeTime } from '../../client/format.js'
import { displayOrigin, type CockpitDevice, type CockpitMachine, type CockpitPairing, type CockpitStatus } from '../hubShared.js'
import { CopyButton } from './CopyButton.js'
import { HUB_CHANGED_EVENT, cockpitCall, windowCockpitApi } from './hubClient.js'
import { localMachineLabel } from './runModeView.js'

/**
 * Settings → "Hub & phone". Redesigned cockpit for connecting this Mac to a
 * Hub, inspecting its fleet of machines & projects, pairing phones with a
 * custom label, and managing/revoking paired devices.
 *
 * Every action runs in the desktop's main process; this view only renders
 * validated, secret-free metadata (and renders QR codes strictly as inert
 * base64 `<img>` elements — never inline `<svg>`).
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

export interface DevicesView {
  readonly phase: 'idle' | 'loading' | 'ready' | 'error'
  readonly devices: readonly CockpitDevice[]
  readonly error?: string
  readonly revokingId?: string
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

const DEVICE_STATUS_TEXT: Readonly<Record<string, string>> = {
  active: 'Active',
  pending: 'Pending',
  revoked: 'Revoked',
  expired: 'Expired',
  offline: 'Offline',
}

/** Human-readable device status label for paired devices. */
export function deviceStatusText(status: string): string {
  const mapped = DEVICE_STATUS_TEXT[status]
  if (mapped) return mapped
  const cleaned = status.replace(/[_-]+/g, ' ').trim()
  return cleaned ? cleaned[0]!.toUpperCase() + cleaned.slice(1) : 'Unknown'
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
  readonly deviceLabel?: string
  readonly onDeviceLabel?: (value: string) => void
  readonly devices?: DevicesView
  readonly onRefreshDevices?: () => void
  readonly onRevokeDevice?: (deviceId: string) => void
  readonly onOpenWeb?: (origin: string) => void
}

function summarizeFleet(machines: readonly CockpitMachine[]): { totalMachines: number; totalProjects: number; onlineProjects: number } {
  let totalProjects = 0
  let onlineProjects = 0
  for (const machine of machines) {
    totalProjects += machine.projects.length
    for (const project of machine.projects) {
      if (project.health === 'online') onlineProjects += 1
    }
  }
  return { totalMachines: machines.length, totalProjects, onlineProjects }
}

function formatExpiry(iso: string): string | null {
  const ms = Date.parse(iso)
  if (!Number.isFinite(ms)) return null
  return new Date(ms).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function MachineList({ view, onRefresh }: { readonly view: MachinesView; readonly onRefresh: () => void }): ReactElement {
  const summary = summarizeFleet(view.machines)
  return (
    <div className="hubphone__machines" aria-label="Machines on this Hub">
      <div className="hubphone__row">
        <div className="hubphone__subhead">
          <strong>Machines and projects</strong>
          {view.phase === 'ready' && summary.totalMachines > 0 ? (
            <span className="hubphone__pill">
              {summary.totalMachines} {summary.totalMachines === 1 ? 'machine' : 'machines'} · {summary.onlineProjects}/{summary.totalProjects} online
            </span>
          ) : null}
        </div>
        <button type="button" className="btn" disabled={view.phase === 'loading'} onClick={onRefresh}>
          {view.phase === 'loading' ? 'Refreshing…' : 'Refresh'}
        </button>
      </div>
      {view.error ? <p role="alert" className="action-card__error">{view.error}</p> : null}
      {view.phase === 'ready' && view.machines.length === 0 ? <p className="studio-muted">The Hub has no machines yet.</p> : null}
      <ul className="hubphone__list">
        {view.machines.map(machine => {
          const onlineCount = machine.projects.filter(p => p.health === 'online').length
          return (
            <li key={machine.id} className="hubphone__machine-card">
              <div className="hubphone__machine-head">
                <p className="hubphone__machine">{machine.label}{machine.discover ? ' · all projects' : ''}</p>
                <span className="hubphone__pill">
                  {machine.projects.length === 0 ? 'Idle' : `${onlineCount}/${machine.projects.length} online`}
                </span>
              </div>
              {machine.projects.length === 0 ? <p className="studio-muted">No projects found yet. They appear as soon as a task runs there.</p> : (
                <ul className="hubphone__projects">
                  {machine.projects.map(project => (
                    <li key={project.workspaceId} className="hubphone__project-row" title={project.projectDir || undefined}>
                      <span className="hubphone__project-name">
                        <span className={`hubphone__dot hubphone__dot--${project.health}`} aria-hidden="true" />
                        <span>{project.label}</span>
                      </span>
                      <span className={`hubphone__health hubphone__health--${project.health}`}>{healthText(project.health)}</span>
                    </li>
                  ))}
                </ul>
              )}
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function DevicesList({
  view,
  onRefresh,
  onRevoke,
}: {
  readonly view: DevicesView
  readonly onRefresh?: () => void
  readonly onRevoke?: (deviceId: string) => void
}): ReactElement {
  const [confirmingId, setConfirmingId] = useState<string | null>(null)
  const active = view.devices.filter(device => device.status !== 'revoked')
  const now = Date.now()
  return (
    <div className="hubphone__devices" aria-label="Paired devices">
      <div className="hubphone__row">
        <div className="hubphone__subhead">
          <strong>Paired devices</strong>
          {view.phase === 'ready' ? <span className="hubphone__pill">{active.length} active</span> : null}
        </div>
        {onRefresh ? (
          <button type="button" className="btn" disabled={view.phase === 'loading'} onClick={onRefresh}>
            {view.phase === 'loading' ? 'Refreshing…' : 'Refresh devices'}
          </button>
        ) : null}
      </div>
      {view.error ? <p role="alert" className="action-card__error">{view.error}</p> : null}
      {view.phase === 'ready' && active.length === 0 ? <p className="studio-muted">No paired phones or desktop sessions yet.</p> : null}
      {active.length > 0 ? (
        <ul className="hubphone__device-list">
          {active.map(device => {
            const busy = view.revokingId === device.id
            const asking = confirmingId === device.id
            return (
              <li key={device.id} className="hubphone__device-row">
                <div className="hubphone__device-info">
                  <span className="hubphone__device-title">
                    <span className={`hubphone__dot ${device.status === 'active' ? 'hubphone__dot--online' : 'hubphone__dot--connecting'}`} aria-hidden="true" />
                    <strong>{device.label}</strong>
                    <span className="hubphone__pill">{deviceStatusText(device.status)}</span>
                  </span>
                  <span className="studio-muted hubphone__device-meta">
                    {device.displayName ? `${device.displayName} · ` : ''}
                    {device.lastSeenAt ? `Last seen ${relativeTime(device.lastSeenAt, now)}` : 'Never connected yet'}
                  </span>
                </div>
                {onRevoke ? (
                  <div className="hubphone__device-actions">
                    {asking ? (
                      <>
                        <button
                          type="button"
                          className="btn btn--danger"
                          disabled={busy}
                          onClick={() => { setConfirmingId(null); onRevoke(device.id) }}
                        >
                          {busy ? 'Revoking…' : 'Confirm revoke'}
                        </button>
                        <button type="button" className="btn" disabled={busy} onClick={() => setConfirmingId(null)}>Cancel</button>
                      </>
                    ) : (
                      <button type="button" className="btn" disabled={busy} onClick={() => setConfirmingId(device.id)}>
                        {busy ? 'Revoking…' : 'Revoke'}
                      </button>
                    )}
                  </div>
                ) : null}
              </li>
            )
          })}
        </ul>
      ) : null}
    </div>
  )
}

function PairingBlock({ pairing, confirming, error, onConfirm }: {
  readonly pairing: CockpitPairing
  readonly confirming: boolean
  readonly error?: string
  readonly onConfirm: (pairingId: string, code: string) => void
}): ReactElement {
  const expiresLabel = formatExpiry(pairing.expiresAt)
  return (
    <div className="hubphone__pair" aria-label="Pair a phone">
      <div className="hubphone__pair-stage">
        <div className="hubphone__pair-qrwrap">
          {pairing.qrDataUrl !== null
            ? <img className="hubphone__qr" src={pairing.qrDataUrl} alt="QR code for pairing the phone" width={192} height={192} />
            : <p className="studio-muted">No QR code was made. Open the link below on the phone instead.</p>}
        </div>
        <div className="hubphone__pair-details">
          <div className="hubphone__code-card">
            <span className="studio-muted">Comparison code</span>
            <strong className="hubphone__code">{pairing.comparisonCode}</strong>
            {expiresLabel ? <span className="hubphone__pill">Expires {expiresLabel}</span> : null}
          </div>
          <p className="studio-muted">Open the link on the phone (or scan the QR code). If the phone shows the same code, confirm it here. The link works once and expires.</p>
          {pairing.confirmRequired
            ? <button type="button" className="btn btn--solid" disabled={confirming} onClick={() => onConfirm(pairing.pairingId, pairing.comparisonCode)}>{confirming ? 'Confirming…' : `The phone shows ${pairing.comparisonCode}`}</button>
            : <p role="status">No confirmation is needed. The phone is ready once it opens the link.</p>}
        </div>
      </div>
      <div className="field hubphone__link-field">
        <label htmlFor="hub-pair-link">Pairing link</label>
        <div className="hubphone__copy-row">
          <input id="hub-pair-link" type="text" readOnly value={pairing.url} spellCheck={false} onFocus={event => event.currentTarget.select()} />
          <CopyButton text={pairing.url} label="Copy link" />
        </div>
      </div>
      {error ? <p role="alert" className="action-card__error">{error}</p> : null}
    </div>
  )
}

export function HubPhoneView(props: HubPhoneViewProps): ReactElement {
  const { available, status, link, pair, machines, devices } = props
  const [confirmForget, setConfirmForget] = useState(false)
  const connected = status !== undefined && status.target !== null && status.origin !== null
  const machine = localMachineLabel()
  const machineTitle = localMachineLabel(true)
  useEffect(() => {
    if (status?.target === null) setConfirmForget(false)
  }, [status?.target])
  return (
    <section className="hubphone" aria-labelledby="hubphone-title">
      <header className="hubphone__header">
        <div className="hubphone__title-row">
          <h2 id="hubphone-title" className="modal__title">Hub &amp; phone</h2>
          {status !== undefined ? (
            <span className={`hubphone__status-badge ${connected ? 'hubphone__status-badge--live' : ''}`}>
              <span className={`hubphone__dot ${connected ? 'hubphone__dot--online' : 'hubphone__dot--offline'}`} aria-hidden="true" />
              {connected ? 'Connected' : 'Not connected'}
            </span>
          ) : null}
        </div>
        <p className="modal__sub">Your Hub is the always-on server that lets your phone and the XERXES tab reach every machine. Enter its SSH address once and this app does the rest.</p>
      </header>

      {!available ? (
        <p role="status" className="notice">Hub &amp; phone is only available in the Xerxes desktop app.</p>
      ) : status === undefined ? (
        <p role="status" aria-busy="true">Reading the saved Hub…</p>
      ) : (
        <>
          <div className="hubphone__card">
            <form className="hubphone__form" onSubmit={event => { event.preventDefault(); props.onSave() }}>
              <div className="field">
                <label htmlFor="hub-ssh-address">Hub SSH address</label>
                <input
                  id="hub-ssh-address" type="text" value={props.address} placeholder="xerxes@203.0.113.10"
                  spellCheck={false} autoComplete="off" autoCapitalize="off" disabled={props.saving}
                  aria-invalid={props.saveError ? true : undefined}
                  {...(props.saveError ? { 'aria-describedby': 'hub-ssh-error' } : {})}
                  onChange={event => props.onAddress(event.target.value)}
                />
              </div>
              <div className="hubphone__row">
                <div className="hubphone__actions">
                  <button type="submit" className={connected ? 'btn' : 'btn btn--solid'} disabled={props.saving || props.address.trim() === ''}>
                    {props.saving ? 'Checking the Hub…' : status.target === null ? 'Save and connect' : 'Save'}
                  </button>
                  {status.target !== null ? (
                    confirmForget ? (
                      <>
                        <button
                          type="button"
                          className="btn btn--danger"
                          disabled={props.saving}
                          onClick={() => { setConfirmForget(false); props.onForget() }}
                        >
                          Confirm forget
                        </button>
                        <button type="button" className="btn" disabled={props.saving} onClick={() => setConfirmForget(false)}>
                          Cancel
                        </button>
                      </>
                    ) : (
                      <button type="button" className="btn" disabled={props.saving} onClick={() => setConfirmForget(true)}>
                        Forget this Hub
                      </button>
                    )
                  ) : null}
                </div>
                {connected && props.onOpenWeb ? (
                  <button type="button" className="btn" onClick={() => props.onOpenWeb?.(status.origin!)}>Open Hub web app</button>
                ) : null}
              </div>
              {props.saveError ? <p id="hub-ssh-error" role="alert" className="action-card__error">{props.saveError}</p> : null}
            </form>

            {!connected ? (
              <p className="studio-muted hubphone__empty-note">No Hub is saved on {machine} yet.</p>
            ) : (
              <dl className="hubphone__facts">
                <dt>Web address</dt><dd>{displayOrigin(status.origin!)}</dd>
                {status.hubVersion !== null ? <><dt>Version</dt><dd>{status.hubVersion}</dd></> : null}
                {status.fingerprint !== null ? <><dt>Key</dt><dd className="hubphone__mono">{status.fingerprint}</dd></> : null}
              </dl>
            )}
          </div>

          {connected ? (
            <>
              <div className="hubphone__card hubphone__block">
                <div className="hubphone__row">
                  <div>
                    <div className="hubphone__subhead">
                      <strong>{machineTitle}</strong>
                      <span className="hubphone__pill">{status.linkedMachineId !== null ? 'Linked' : 'Unlinked'}</span>
                    </div>
                    <p className="studio-muted">{status.linkedMachineId !== null ? `Linked to the Hub as ${status.linkedMachineId}. Every project on ${machine} is connected automatically.` : `Not linked yet. Linking connects every project on ${machine} to the Hub.`}</p>
                  </div>
                  <button type="button" className={status.linkedMachineId === null ? 'btn btn--solid' : 'btn'} disabled={link.phase === 'running'} onClick={props.onLink}>
                    {link.phase === 'running' ? 'Linking…' : status.linkedMachineId !== null ? `Link ${machine} again` : `Link ${machine}`}
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

              <div className="hubphone__card hubphone__block">
                <div className="hubphone__row">
                  <div>
                    <strong>Phone</strong>
                    <p className="studio-muted">Pair a phone to follow tasks, start new ones, and talk to XERXES from anywhere.</p>
                  </div>
                  <form className="hubphone__pair-controls" onSubmit={event => { event.preventDefault(); if (pair.phase !== 'creating') props.onPair() }}>
                    {props.onDeviceLabel ? (
                      <input
                        id="hub-phone-label"
                        className="hubphone__label-input"
                        type="text"
                        aria-label="Phone label"
                        value={props.deviceLabel ?? 'Phone'}
                        placeholder="Phone"
                        maxLength={40}
                        disabled={pair.phase === 'creating'}
                        onChange={event => props.onDeviceLabel?.(event.target.value)}
                      />
                    ) : null}
                    <button type="submit" className={status.linkedMachineId !== null && pair.phase === 'idle' ? 'btn btn--solid' : 'btn'} disabled={pair.phase === 'creating'}>
                      {pair.phase === 'creating' ? 'Creating link…' : pair.phase === 'idle' ? 'Pair a phone' : 'New pairing link'}
                    </button>
                  </form>
                </div>
                {pair.phase === 'ready' ? <PairingBlock pairing={pair.pairing} confirming={pair.confirming} {...(pair.error === undefined ? {} : { error: pair.error })} onConfirm={props.onConfirm} /> : null}
                {pair.phase === 'confirmed' ? <p role="status" className="notice">Phone paired. It can open the Hub now.</p> : null}
                {pair.phase === 'error' ? <p role="alert" className="action-card__error">{pair.message}</p> : null}
                {devices && (devices.phase !== 'idle' || devices.devices.length > 0 || devices.error) ? (
                  <DevicesList view={devices} {...(props.onRefreshDevices ? { onRefresh: props.onRefreshDevices } : {})} {...(props.onRevokeDevice ? { onRevoke: props.onRevokeDevice } : {})} />
                ) : null}
              </div>
            </>
          ) : null}
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
  const [deviceLabel, setDeviceLabel] = useState('Phone')
  const [saving, setSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | undefined>(undefined)
  const [link, setLink] = useState<LinkState>({ phase: 'idle' })
  const [machines, setMachines] = useState<MachinesView>({ phase: 'idle', machines: [] })
  const [devices, setDevices] = useState<DevicesView>({ phase: 'idle', devices: [] })
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

  const loadDevices = useCallback(async (): Promise<void> => {
    if (api === undefined) return
    setDevices(previous => ({ phase: 'loading', devices: previous.devices }))
    const result = await cockpitCall(api, 'devices')
    if (!alive.current) return
    setDevices(result.ok ? { phase: 'ready', devices: result.devices } : { phase: 'error', devices: [], error: result.message })
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
        if (result.status.target !== null && result.status.origin !== null) void loadDevices()
      } else {
        setStatus(EMPTY_STATUS)
        setSaveError(result.message)
      }
    })
    return () => { cancelled = true }
  }, [api, loadMachines, loadDevices])

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
    if (result.status.target !== null && result.status.origin !== null) void loadDevices()
    else setDevices({ phase: 'idle', devices: [] })
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
    setDevices({ phase: 'idle', devices: [] })
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
    const cleanLabel = deviceLabel.trim() || 'Phone'
    const result = await cockpitCall(api, 'pair', { label: cleanLabel })
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
    if (result.ok) void loadDevices()
  }

  const revokeDevice = async (deviceId: string): Promise<void> => {
    if (api === undefined) return
    setDevices(previous => ({ ...previous, revokingId: deviceId }))
    const result = await cockpitCall(api, 'revokeDevice', { deviceId })
    if (!alive.current) return
    if (!result.ok) {
      setDevices(previous => ({ phase: 'ready', devices: previous.devices, error: result.message }))
      return
    }
    void loadDevices()
  }

  const openWeb = useCallback((origin: string): void => {
    void window.xerxes?.openExternal?.(origin).catch(() => undefined)
  }, [])

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
      deviceLabel={deviceLabel}
      onDeviceLabel={setDeviceLabel}
      devices={devices}
      onRefreshDevices={() => { void loadDevices() }}
      onRevokeDevice={id => { void revokeDevice(id) }}
      onOpenWeb={openWeb}
    />
  )
}
