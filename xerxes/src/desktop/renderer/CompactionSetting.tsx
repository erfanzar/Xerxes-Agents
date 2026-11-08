// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Settings → General → Auto-compact: when a task's history is summarized to
 * make room. A share of the model's context window (60%) or a token count
 * (400k); read and saved through compaction.settings.get/save.
 */

import { useEffect, useState, type ReactElement } from 'react'

import { desktopCall, desktopError, type RpcRecord } from './desktopRpc.js'
import type { Snapshot } from './store.js'

export interface CompactionState {
  /** A share of the window (0 < n <= 1), a token count (n > 1), or 0 for off. */
  readonly threshold: number
  readonly label: string
  readonly defaultLabel: string
  /** Whether a person chose a value; otherwise the default applies. */
  readonly saved: boolean
}

export function compactionStateOf(result: RpcRecord): CompactionState {
  const threshold = typeof result.threshold === 'number' && Number.isFinite(result.threshold) && result.threshold >= 0 ? result.threshold : 0
  return {
    threshold,
    label: typeof result.label === 'string' ? result.label : '',
    defaultLabel: typeof result.default_label === 'string' ? result.default_label : '',
    saved: result.saved === true,
  }
}

export function compactionSummary(state: CompactionState): string {
  if (state.threshold <= 0) return 'Off: history is summarized only when you run /compact or a provider refuses a request for its size.'
  if (state.threshold <= 1) return `A task's history is summarized once it fills ${state.label} of the model's context window. The full history is archived first.`
  return `A task's history is summarized once it reaches ${state.label}; a model with a smaller window compacts at ${state.defaultLabel} of it instead. The full history is archived first.`
}

export function CompactionSetting({ snap }: { snap: Snapshot }): ReactElement | null {
  const [state, setState] = useState<CompactionState | null>(null)
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState<{ kind: 'error' | 'saved'; text: string } | null>(null)
  const online = snap.connection === 'online'

  useEffect(() => {
    if (!online) return
    let live = true
    void desktopCall(window.xerxes, snap.sessionKey, 'compaction.settings.get').then(result => {
      if (!live) return
      const next = compactionStateOf(result)
      setState(next)
      setDraft(next.label)
    }).catch(() => { /* An older runtime has no setting; the row stays hidden. */ })
    return () => { live = false }
  }, [snap.sessionKey, online])

  if (!state) return null
  const save = async (threshold: string | null) => {
    setMessage(null)
    setBusy(true)
    try {
      const next = compactionStateOf(await desktopCall(window.xerxes, snap.sessionKey, 'compaction.settings.save', { threshold }))
      setState(next)
      setDraft(next.label)
      setMessage({ kind: 'saved', text: next.threshold > 0 ? `Saved. Tasks compact at ${next.label}.` : 'Saved. Automatic compaction is off.' })
    } catch (error) {
      setMessage({ kind: 'error', text: desktopError(error) })
    } finally {
      setBusy(false)
    }
  }
  const apply = () => {
    const text = draft.trim()
    if (!text) { setMessage({ kind: 'error', text: 'Enter a percentage such as 60%, a token count such as 400k, or off.' }); return }
    void save(text)
  }

  return (
    <>
      <div className="row">
        <div className="row__main">
          <div className="row__t">Auto-compact</div>
          <div className="row__s">{compactionSummary(state)}</div>
        </div>
      </div>
      <div className="row memguard">
        <label className="memguard__field">
          <span>Compact at</span>
          <input className="compact-threshold__input" value={draft} spellCheck={false} placeholder="60%, 0.7 or 400k"
            aria-label="Auto-compact threshold" disabled={!online || busy}
            onChange={event => { setDraft(event.target.value); setMessage(null) }}
            onKeyDown={event => { if (event.key === 'Enter') apply() }} />
        </label>
        <button className="btn" disabled={!online || busy || draft.trim() === state.label} onClick={apply}>Save</button>
        {state.saved && <button className="btn btn--ghost" disabled={!online || busy} onClick={() => void save(null)}>Use default ({state.defaultLabel})</button>}
      </div>
      {message && <p className={message.kind === 'error' ? 'studio-error' : 'row__s'} role={message.kind === 'error' ? 'alert' : 'status'}>{message.text}</p>}
    </>
  )
}
