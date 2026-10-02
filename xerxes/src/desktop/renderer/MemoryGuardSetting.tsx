// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Settings → General → Memory guard: how much memory one agent command (with
 * everything it starts) may use before the runtime stops it. Read and saved
 * through memory_guard.get/save; the default is half the computer's memory.
 */

import { useEffect, useState, type ReactElement } from 'react'

import { desktopCall, desktopError, type RpcRecord } from './desktopRpc.js'
import type { Snapshot } from './store.js'

interface GuardState {
  readonly supported: boolean
  /** Saved limit in MB; null means the default. 0 is off. */
  readonly limitMb: number | null
  readonly effectiveMb: number
  readonly defaultMb: number
  readonly totalMb: number
  readonly environmentOverride: boolean
}

export function guardStateOf(result: RpcRecord): GuardState {
  const num = (value: unknown, fallback = 0) => typeof value === 'number' && Number.isFinite(value) ? value : fallback
  return {
    supported: result.supported === true,
    limitMb: typeof result.limit_mb === 'number' ? result.limit_mb : null,
    effectiveMb: num(result.effective_mb),
    defaultMb: num(result.default_mb),
    totalMb: num(result.total_mb),
    environmentOverride: result.environment_override === true,
  }
}

const gb = (mb: number) => Math.round((mb / 1024) * 10) / 10

export function MemoryGuardSetting({ snap }: { snap: Snapshot }): ReactElement | null {
  const [state, setState] = useState<GuardState | null>(null)
  const [draft, setDraft] = useState('')
  const [message, setMessage] = useState<{ kind: 'error' | 'saved'; text: string } | null>(null)
  const online = snap.connection === 'online'

  useEffect(() => {
    if (!online) return
    let live = true
    void desktopCall(window.xerxes, snap.sessionKey, 'memory_guard.get').then(result => {
      if (!live) return
      const next = guardStateOf(result)
      setState(next)
      setDraft(String(gb(next.effectiveMb || next.defaultMb)))
    }).catch(() => { /* An older runtime has no guard; the row stays hidden. */ })
    return () => { live = false }
  }, [snap.sessionKey, online])

  if (!state) return null
  const on = state.effectiveMb > 0
  const save = async (limitMb: number | null) => {
    setMessage(null)
    try {
      const next = guardStateOf(await desktopCall(window.xerxes, snap.sessionKey, 'memory_guard.save', { limit_mb: limitMb }))
      setState(next)
      setDraft(String(gb(next.effectiveMb || next.defaultMb)))
      setMessage({ kind: 'saved', text: next.effectiveMb > 0 ? `Saved. Commands are stopped past ${gb(next.effectiveMb)} GB.` : 'Saved. The memory guard is off.' })
    } catch (error) { setMessage({ kind: 'error', text: desktopError(error) }) }
  }
  const applyDraft = () => {
    const value = Number(draft)
    if (!Number.isFinite(value) || value <= 0) { setMessage({ kind: 'error', text: 'Enter a limit in GB above 0, or turn the guard off.' }); return }
    void save(Math.round(value * 1024))
  }

  return (
    <>
      <div className="row">
        <div className="row__main">
          <div className="row__t">Memory guard</div>
          <div className="row__s">
            {!state.supported ? 'Not available on this computer.'
              : on ? `Stops an agent command, with everything it started, once it uses more than ${gb(state.effectiveMb)} GB${state.limitMb === null ? ` (half of this computer's ${gb(state.totalMb)} GB)` : ''}. The agent is told why.`
                : 'Off: an agent command can use as much memory as it wants.'}
            {state.environmentOverride && ' Set by XERXES_COMMAND_MEMORY_LIMIT_MB, which overrides this setting.'}
          </div>
        </div>
        <button
          className={`switch${on ? ' is-on' : ''}`}
          role="switch"
          aria-checked={on}
          aria-label="Memory guard"
          disabled={!state.supported || !online || state.environmentOverride}
          onClick={() => void save(on ? 0 : null)}
        />
      </div>
      {on && state.supported && !state.environmentOverride && (
        <div className="row memguard">
          <label className="memguard__field">
            <span>Limit per command</span>
            <input type="number" min={0.5} step={0.5} value={draft} onChange={event => { setDraft(event.target.value); setMessage(null) }}
              onKeyDown={event => { if (event.key === 'Enter') applyDraft() }} aria-label="Memory limit in GB" />
            <span>GB</span>
          </label>
          <button className="btn" disabled={Number(draft) * 1024 === state.effectiveMb} onClick={applyDraft}>Save</button>
          {state.limitMb !== null && <button className="btn btn--ghost" onClick={() => void save(null)}>Use default ({gb(state.defaultMb)} GB)</button>}
        </div>
      )}
      {message && <p className={message.kind === 'error' ? 'studio-error' : 'row__s'} role={message.kind === 'error' ? 'alert' : 'status'}>{message.text}</p>}
    </>
  )
}
