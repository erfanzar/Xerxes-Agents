// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The Hub notice, like the macOS permissions one: shown when no Hub is set up
 * or the saved Hub does not answer its /healthz. The main process does the
 * checking (main/hubHealth.ts); this page only shows the validated state and
 * names an action.
 */

import { useCallback, useEffect, useRef, useState, type ReactElement } from 'react'

import { asHubHealthState, hubNoticeKind, type HubHealthAction, type HubHealthState, type HubNoticeKind } from '../hubHealthShared.js'
import { isVscodeHost } from './host.js'
import { store } from './store.js'

/** The slice of `window.xerxes` the notice needs. */
export interface HubHealthApi {
  call(action: HubHealthAction, params?: Record<string, unknown>): Promise<unknown>
  onState(handler: (state: HubHealthState) => void): () => void
}

export function windowHubHealthApi(): HubHealthApi | undefined {
  // The VS Code webview shares this bridge object but its host has no such channels.
  if (typeof window === 'undefined' || isVscodeHost()) return undefined
  const bridge = window.xerxes
  if (!bridge?.hubHealth || !bridge.onHubHealth) return undefined
  const call = bridge.hubHealth.bind(bridge)
  const on = bridge.onHubHealth.bind(bridge)
  return { call: (action, params) => call(action, params ?? {}), onState: handler => on(handler) }
}

/** The state from one call's answer, or undefined when it failed or is off-contract. */
export function hubHealthStateOf(raw: unknown): HubHealthState | undefined {
  if (!raw || typeof raw !== 'object' || (raw as { ok?: unknown }).ok !== true) return undefined
  return asHubHealthState((raw as { state?: unknown }).state)
}

export interface HubHealthNoticeProps {
  readonly state: HubHealthState | undefined
  /** "Not now" for this window only. */
  readonly hidden: boolean
  readonly checking: boolean
  readonly onSettings: () => void
  readonly onRecheck: () => void
  readonly onLater: () => void
  readonly onDismiss: (kind: HubNoticeKind) => void
}

/**
 * The non-blocking notice, for a Hub someone set up that stopped answering.
 * Nothing when the Hub is fine, unknown, or the person opted out, and nothing
 * when no Hub was ever set up: most people never need one, and setting one up
 * lives in Settings › Hub & phone.
 */
export function HubHealthNoticeView(props: HubHealthNoticeProps): ReactElement | null {
  const { state } = props
  if (!state || props.hidden) return null
  const kind = hubNoticeKind(state)
  if (kind === null || kind === 'not_configured') return null
  const title = 'The Hub is not reachable.'
  const body = `${state.origin ?? state.target ?? 'The saved Hub'} did not answer${state.detail ? ` (${state.detail})` : ''}. The phone app, XERXES and Hub relays are offline until it is back. Your workspaces keep running.`
  return (
    <div className="macperm-notice hub-notice" role="status" aria-live="polite" data-kind={kind}>
      <div className="macperm-notice__text">
        <strong>{title}</strong>
        <span>{body}</span>
        {state.target && <span className="hub-notice__target">Hub host: <code>{state.target}</code>{state.checked_at ? ` · checked ${new Date(state.checked_at).toLocaleTimeString()}` : ''}</span>}
      </div>
      <div className="macperm-notice__actions">
        <button className="btn btn--solid" onClick={props.onSettings}>Hub settings</button>
        <button className="btn" disabled={props.checking} onClick={props.onRecheck}>{props.checking ? 'Checking…' : 'Re-check'}</button>
        <button className="btn btn--ghost" onClick={props.onLater}>Not now</button>
        <button className="btn btn--ghost" onClick={() => props.onDismiss(kind)}>Don't remind me</button>
      </div>
    </div>
  )
}

/** The notice, wired to the main process. Renders nothing outside the desktop app. */
export function HubHealthNotice(): ReactElement | null {
  const [api] = useState(() => windowHubHealthApi())
  const [state, setState] = useState<HubHealthState | undefined>(undefined)
  const [hidden, setHidden] = useState(false)
  const [checking, setChecking] = useState(false)
  const alive = useRef(true)
  useEffect(() => { alive.current = true; return () => { alive.current = false } }, [])
  const run = useCallback(async (action: HubHealthAction, params?: Record<string, unknown>): Promise<void> => {
    if (api === undefined) return
    try {
      const next = hubHealthStateOf(await api.call(action, params))
      if (next && alive.current) setState(next)
    } catch { /* the notice is advisory; a failed call leaves the last state */ }
  }, [api])
  useEffect(() => {
    if (api === undefined) return undefined
    const stop = api.onState(next => {
      if (!alive.current) return
      setState(next)
      // A fresh problem after recovery deserves to be seen again.
      if (next.kind === 'online') setHidden(false)
    })
    void run('status')
    return stop
  }, [api, run])
  if (api === undefined) return null
  return (
    <HubHealthNoticeView
      state={state}
      hidden={hidden}
      checking={checking}
      onSettings={() => store.openSettings('hub')}
      onRecheck={() => { setChecking(true); void run('recheck').finally(() => { if (alive.current) setChecking(false) }) }}
      onLater={() => setHidden(true)}
      onDismiss={kind => { void run('dismiss', { kind }) }}
    />
  )
}
