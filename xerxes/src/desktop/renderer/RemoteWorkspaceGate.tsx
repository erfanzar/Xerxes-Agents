// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState } from 'react'
import type { XerxesBridge } from './types.js'
import { desktopError } from './desktopRpc.js'

export interface RemoteOpeningState {
  machine?: { alias: string; target: string; workspacePath: string }
  connecting: boolean
  connected: boolean
  error: string
  resume_session_id?: string
}

export function remoteOpeningState(value: unknown): RemoteOpeningState {
  if (!value || typeof value !== 'object') throw new Error('Could not read SSH connection status.')
  const row = value as Record<string, unknown>
  if (row.ok === false) throw new Error(typeof row.error === 'string' ? row.error : 'SSH operation failed.')
  const m = row.machine as Record<string, unknown> | undefined
  return {
    ...(m && typeof m.alias === 'string' && typeof m.target === 'string' && typeof m.workspacePath === 'string'
      ? { machine: { alias: m.alias, target: m.target, workspacePath: m.workspacePath } } : {}),
    connecting: row.connecting === true, connected: row.connected === true,
    error: typeof row.error === 'string' ? row.error : '',
    ...(typeof row.resume_session_id === 'string' ? { resume_session_id: row.resume_session_id } : {}),
  }
}

export function RemoteWorkspaceGate({ remote = typeof window === 'undefined' ? undefined : window.xerxes.remote }: { remote?: XerxesBridge['remote'] }) {
  const [state, setState] = useState<RemoteOpeningState>({ connecting: true, connected: false, error: '' })
  const [busy, setBusy] = useState(false)
  const alive = useRef(false), operation = useRef(false)
  useEffect(() => {
    alive.current = true
    let reading = false
    const read = async () => {
      if (reading || operation.current) return
      reading = true
      try {
        if (!remote) throw new Error('SSH connections require the current desktop app.')
        const next = remoteOpeningState(await remote('status', {}))
        if (alive.current && !operation.current) setState(next)
      } catch (error) {
        if (alive.current && !operation.current) setState(s => ({ ...s, connecting: false, error: desktopError(error) }))
      } finally { reading = false }
    }
    void read()
    const timer = setInterval(() => void read(), 1000)
    return () => { alive.current = false; clearInterval(timer) }
  }, [remote])
  const cancel = async () => {
    if (!remote) return
    try { remoteOpeningState(await remote('cancel', {})) }
    catch (error) { if (alive.current) setState(s => ({ ...s, error: desktopError(error) })) }
  }
  const act = async () => {
    if (!remote || operation.current) return
    operation.current = true
    setBusy(true)
    setState(s => ({ ...s, error: '', connecting: true }))
    try {
      remoteOpeningState(await remote('connect', { machine: state.machine, resume_session_id: state.resume_session_id }))
    } catch (error) {
      if (alive.current) setState(s => ({ ...s, connecting: false, error: desktopError(error) }))
    } finally { operation.current = false; if (alive.current) setBusy(false) }
  }
  return <main className="chat"><section className="wsgate" aria-label="SSH workspace connection">
    <h1>{state.connecting ? 'Connecting to SSH workspace' : state.connected ? 'Opening your session' : 'Could not open SSH workspace'}</h1>
    {state.machine && <p>{state.machine.alias} · {state.machine.target}<br />{state.machine.workspacePath}</p>}
    {state.error && <p className="connection-error" role="alert">{state.error}</p>}
    <p role="status">{state.connecting ? 'Preparing the remote runtime. You can use your other workspaces while it connects.' : state.connected ? 'Restoring the saved conversation.' : 'Your saved workspace and session are retained. Retry after resolving the connection error.'}</p>
    {state.connecting ? <button className="btn" onClick={() => void cancel()}>Cancel connection</button>
      : !state.connected && <button className="btn btn--solid" disabled={busy || !state.machine} onClick={() => void act()}>Retry SSH connection</button>}
  </section></main>
}
