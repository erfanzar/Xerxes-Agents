// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * macOS privacy permissions: Settings → macOS permissions (one row per
 * permission, Open Settings / Request / Re-check) and the non-blocking notice
 * shown at start when computer use is missing Accessibility or Screen
 * Recording. The main process reads the statuses and owns every System
 * Settings link; this page only names a permission id.
 */

import { useCallback, useEffect, useRef, useState, type ReactElement } from 'react'

import {
  asMacPermissionsState, missingNoticePermissions,
  type MacPermissionAction, type MacPermissionId, type MacPermissionItem, type MacPermissionStatus, type MacPermissionsState,
} from '../macPermissionsShared.js'
import { isVscodeHost } from './host.js'
import { store } from './store.js'

/** The slice of `window.xerxes` this panel needs. */
export interface MacPermissionsApi {
  call(action: MacPermissionAction, params?: Record<string, unknown>): Promise<unknown>
  onState(handler: (state: MacPermissionsState) => void): () => void
}

export function windowMacPermissionsApi(): MacPermissionsApi | undefined {
  // The VS Code webview shares this bridge object but its host has no such channels.
  if (typeof window === 'undefined' || isVscodeHost()) return undefined
  const bridge = window.xerxes
  if (!bridge?.macPermissions || !bridge.onMacPermissions) return undefined
  const call = bridge.macPermissions.bind(bridge)
  const on = bridge.onMacPermissions.bind(bridge)
  return { call: (action, params) => call(action, params ?? {}), onState: handler => on(handler) }
}

type CallOutcome = { readonly ok: true; readonly state: MacPermissionsState; readonly openedSettings: boolean } | { readonly ok: false; readonly message: string }

/** One call with the answer validated; a rejected promise becomes a message. */
export async function macPermissionsCall(api: MacPermissionsApi, action: MacPermissionAction, params?: Record<string, unknown>): Promise<CallOutcome> {
  try {
    const raw = await api.call(action, params)
    if (raw && typeof raw === 'object' && (raw as { ok?: unknown }).ok === true) {
      const state = asMacPermissionsState((raw as { state?: unknown }).state)
      if (state) return { ok: true, state, openedSettings: (raw as { opened_settings?: unknown }).opened_settings === true }
    }
    const message = raw && typeof raw === 'object' && typeof (raw as { message?: unknown }).message === 'string' ? (raw as { message: string }).message : 'The permissions check returned an unexpected answer.'
    return { ok: false, message }
  } catch (error) {
    return { ok: false, message: error instanceof Error ? error.message : String(error) }
  }
}

const STATUS_TEXT: Record<MacPermissionStatus, string> = {
  granted: 'On',
  denied: 'Off',
  not_determined: 'Not asked yet',
  unknown: 'Cannot be checked',
  not_applicable: 'Not applicable',
}

function statusDot(status: MacPermissionStatus): string {
  if (status === 'granted') return 'dot dot--done'
  if (status === 'denied' || status === 'not_determined') return 'dot dot--fail'
  return 'dot'
}

/** Whether the screen was ever not granted in this state; macOS needs a relaunch to pick a new grant up. */
function screenMayNeedRelaunch(state: MacPermissionsState): boolean {
  return state.items.some(item => item.id === 'screen' && item.status !== 'granted' && item.status !== 'not_applicable')
}

function hostNote(app: string): string {
  return `macOS gives these permissions to the app that runs Xerxes. For computer use started from this app that is ${app}. If the runtime was started by the VS Code extension, VS Code needs them instead; from a terminal, that terminal app.`
}

export interface MacPermissionsViewProps {
  readonly available: boolean
  readonly state: MacPermissionsState | undefined
  readonly busy: MacPermissionId | 'recheck' | null
  readonly error?: string | undefined
  readonly onOpen: (id: MacPermissionId) => void
  readonly onRequest: (id: MacPermissionId) => void
  readonly onRecheck: () => void
  readonly onRemind: (remind: boolean) => void
  readonly onRelaunch: () => void
}

function PermissionRow({ item, busy, onOpen, onRequest }: { item: MacPermissionItem; busy: boolean; onOpen: () => void; onRequest: () => void }): ReactElement {
  return (
    <div className="row macperm-row" data-permission={item.id}>
      <span className={statusDot(item.status)} aria-hidden="true" />
      <div className="row__main">
        <div className="row__t">{item.label} <span className={`macperm-status macperm-status--${item.status}`}>{STATUS_TEXT[item.status]}</span></div>
        <div className="row__s">{item.required_for}</div>
        {item.detail && <div className="row__s macperm-detail">{item.detail}</div>}
      </div>
      <div className="macperm-actions">
        {item.can_request && <button className="btn" disabled={busy} onClick={onRequest}>Request</button>}
        <button className="btn" disabled={busy} onClick={onOpen}>Open Settings</button>
      </div>
    </div>
  )
}

/** Settings → macOS permissions, as a pure view (tests render it directly). */
export function MacPermissionsView(props: MacPermissionsViewProps): ReactElement {
  const { available, state, busy, error } = props
  return (
    <section aria-labelledby="macperm-title">
      <h2 className="modal__title" id="macperm-title">macOS permissions</h2>
      {!available ? <p role="status" className="notice">macOS permissions are only checked in the Xerxes desktop app.</p>
        : !state ? <p role="status">Checking permissions…</p>
        : state.platform !== 'darwin' ? <p role="status" className="notice">These permissions only exist on macOS; nothing to turn on here.</p>
        : (
          <>
            <p className="modal__sub">{hostNote(state.app_label)} Statuses are read without asking macOS to prompt; Request shows macOS's own dialog only when you click it.</p>
            {error && <p role="alert" className="studio-error">{error}</p>}
            <div className="rowlist">
              {state.items.map(item => (
                <PermissionRow key={item.id} item={item} busy={busy !== null} onOpen={() => props.onOpen(item.id)} onRequest={() => props.onRequest(item.id)} />
              ))}
            </div>
            {screenMayNeedRelaunch(state) && <p className="modal__sub">macOS applies a new Screen Recording grant only after {state.app_label} relaunches.</p>}
            <div className="macperm-footer">
              <button className="btn btn--ghost" disabled={busy !== null} onClick={props.onRecheck}>{busy === 'recheck' ? 'Checking…' : 'Re-check'}</button>
              <button className="btn btn--ghost" onClick={props.onRelaunch}>Relaunch {state.app_label}</button>
              <label className="macperm-remind">
                <input type="checkbox" checked={!state.dismissed} onChange={event => props.onRemind(event.currentTarget.checked)} />
                Remind me at start when computer use is missing a permission
              </label>
            </div>
            <p className="modal__sub">Last checked {new Date(state.checked_at).toLocaleTimeString()}. Coming back to this window from System Settings checks again.</p>
          </>
        )}
    </section>
  )
}

export interface MacPermissionsNoticeProps {
  readonly state: MacPermissionsState | undefined
  /** "Not now" for this window only. */
  readonly hidden: boolean
  readonly onOpen: (id: MacPermissionId) => void
  readonly onLater: () => void
  readonly onDismiss: () => void
  readonly onRelaunch: () => void
  readonly onDetails: () => void
}

/** Whether the startup notice shows for this state. */
export function macPermissionsNoticeVisible(state: MacPermissionsState | undefined, hidden: boolean): boolean {
  return Boolean(state && !hidden && state.platform === 'darwin' && state.notice && !state.dismissed && missingNoticePermissions(state.items).length > 0)
}

/** The non-blocking notice: missing Accessibility / Screen Recording, with a way to the right pane. */
export function MacPermissionsNoticeView(props: MacPermissionsNoticeProps): ReactElement | null {
  const { state } = props
  if (!state || !macPermissionsNoticeVisible(state, props.hidden)) return null
  const missing = missingNoticePermissions(state.items)
  const names = missing.map(item => item.label).join(' and ')
  const first = missing[0]!
  const screen = missing.some(item => item.id === 'screen')
  return (
    <div className="macperm-notice" role="status" aria-live="polite">
      <div className="macperm-notice__text">
        <strong>Computer use needs Accessibility and Screen Recording for {state.app_label}.</strong>
        <span> {names} {missing.length > 1 ? 'are' : 'is'} off. Turn {missing.length > 1 ? 'them' : 'it'} on in System Settings › Privacy &amp; Security.{screen ? ` macOS often needs ${state.app_label} relaunched after Screen Recording is turned on.` : ''}</span>
      </div>
      <div className="macperm-notice__actions">
        <button className="btn btn--solid" onClick={() => props.onOpen(first.id)}>Open settings</button>
        {missing.length > 1 && missing.slice(1).map(item => <button key={item.id} className="btn" onClick={() => props.onOpen(item.id)}>Open {item.label}</button>)}
        {screen && <button className="btn" onClick={props.onRelaunch}>Relaunch now</button>}
        <button className="btn btn--ghost" onClick={props.onDetails}>Details</button>
        <button className="btn btn--ghost" onClick={props.onLater}>Not now</button>
        <button className="btn btn--ghost" onClick={props.onDismiss}>Don't remind me</button>
      </div>
    </div>
  )
}

/** Shared state for the wired panel and notice: the latest state, kept fresh by main's pushes. */
function useMacPermissions(api: MacPermissionsApi | undefined): {
  state: MacPermissionsState | undefined
  busy: MacPermissionId | 'recheck' | null
  error: string | undefined
  run: (action: MacPermissionAction, params?: Record<string, unknown>, busy?: MacPermissionId | 'recheck') => Promise<void>
} {
  const [state, setState] = useState<MacPermissionsState | undefined>(undefined)
  const [busy, setBusy] = useState<MacPermissionId | 'recheck' | null>(null)
  const [error, setError] = useState<string | undefined>(undefined)
  const alive = useRef(true)
  useEffect(() => {
    alive.current = true
    return () => { alive.current = false }
  }, [])
  const run = useCallback(async (action: MacPermissionAction, params?: Record<string, unknown>, mark?: MacPermissionId | 'recheck'): Promise<void> => {
    if (api === undefined) return
    if (mark !== undefined) setBusy(mark)
    setError(undefined)
    const result = await macPermissionsCall(api, action, params)
    if (!alive.current) return
    if (mark !== undefined) setBusy(null)
    if (result.ok) setState(result.state)
    else setError(result.message)
  }, [api])
  useEffect(() => {
    if (api === undefined) return undefined
    const stop = api.onState(next => { if (alive.current) setState(next) })
    void run('status')
    return stop
  }, [api, run])
  return { state, busy, error, run }
}

/** Settings → macOS permissions, wired to the main process. */
export function MacPermissionsCard(): ReactElement {
  const [api] = useState(() => windowMacPermissionsApi())
  const { state, busy, error, run } = useMacPermissions(api)
  return (
    <MacPermissionsView
      available={api !== undefined}
      state={state}
      busy={busy}
      error={error}
      onOpen={id => { void run('open', { id }, id) }}
      onRequest={id => { void run('request', { id }, id) }}
      onRecheck={() => { void run('status', undefined, 'recheck') }}
      onRemind={remind => { void run(remind ? 'remind' : 'dismiss') }}
      onRelaunch={() => { void run('relaunch') }}
    />
  )
}

/** The startup notice, wired to the main process. Renders nothing outside the desktop app or off macOS. */
export function MacPermissionsNotice(): ReactElement | null {
  const [api] = useState(() => windowMacPermissionsApi())
  const { state, run } = useMacPermissions(api)
  const [hidden, setHidden] = useState(false)
  if (api === undefined) return null
  return (
    <MacPermissionsNoticeView
      state={state}
      hidden={hidden}
      onOpen={id => { void run('open', { id }) }}
      onLater={() => setHidden(true)}
      onDismiss={() => { void run('dismiss') }}
      onRelaunch={() => { void run('relaunch') }}
      onDetails={() => store.openSettings('macos')}
    />
  )
}
