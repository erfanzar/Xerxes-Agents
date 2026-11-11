// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Every file the current task wrote, by any tool or script: the runtime
 * records them at the end of each turn (`outputs.list`). A row opens the file
 * in the Files panel's viewer; Open and Show in Finder appear when the host
 * offers them (`window.xerxes.fileAction`). Results come first, then the
 * scripts and scratch files the task wrote along the way.
 */

import { useCallback, useEffect, useRef, useState, type ReactElement } from 'react'

import { desktopCall } from './desktopRpc.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import { Icon } from './Icon.js'
import type { Snapshot } from './store.js'

export interface TaskOutput {
  readonly path: string
  readonly turn: number
  readonly size: number
  readonly modifiedAt: number
}

export interface TaskOutputs {
  readonly cwd: string
  readonly outputs: readonly TaskOutput[]
}

const SCRIPT = /\.(py|sh|bash|zsh|js|mjs|cjs|ts|rb|pl|r|ps1|bat|ipynb)$/i

/** `outputs.list` as the panel reads it; malformed rows are dropped, never guessed. */
export function taskOutputsView(value: unknown): TaskOutputs | null {
  if (!value || typeof value !== 'object') return null
  const item = value as Record<string, unknown>
  if (typeof item.cwd !== 'string' || !Array.isArray(item.outputs)) return null
  const outputs = item.outputs.flatMap(raw => {
    const row = raw && typeof raw === 'object' ? raw as Record<string, unknown> : {}
    const { path, turn, size, modified_at: modifiedAt } = row
    return typeof path === 'string' && path && typeof turn === 'number' && typeof size === 'number' && typeof modifiedAt === 'number'
      ? [{ path, turn, size, modifiedAt }]
      : []
  })
  return { cwd: item.cwd, outputs }
}

/** Results first, then scripts and scratch files; newest first within each. */
export function groupOutputs(outputs: readonly TaskOutput[]): { readonly results: TaskOutput[]; readonly scripts: TaskOutput[] } {
  const byTime = [...outputs].sort((a, b) => b.modifiedAt - a.modifiedAt || a.path.localeCompare(b.path))
  const scratch = (path: string) => SCRIPT.test(path) || /(^|\/)(_|\.)/.test(path) || /(^|\/)__pycache__\//.test(path)
  return { results: byTime.filter(output => !scratch(output.path)), scripts: byTime.filter(output => scratch(output.path)) }
}

export function fileSize(bytes: number): string {
  if (bytes < 1_000) return `${bytes} B`
  if (bytes < 1_000_000) return `${(bytes / 1_000).toFixed(bytes < 10_000 ? 1 : 0)} KB`
  return `${(bytes / 1_000_000).toFixed(1)} MB`
}

function ago(at: number, now: number): string {
  const seconds = Math.max(0, Math.round((now - at) / 1000))
  if (seconds < 60) return 'just now'
  const minutes = Math.round(seconds / 60)
  if (minutes < 60) return `${minutes} min ago`
  const hours = Math.round(minutes / 60)
  return hours < 24 ? `${hours} h ago` : new Date(at).toLocaleDateString()
}

/** This task's outputs, refreshed when it changes and after each turn. Null on a runtime without the list. */
export function useTaskOutputs(snap: Snapshot): { readonly outputs: TaskOutputs | null; readonly error: string; readonly refresh: () => void } {
  const [outputs, setOutputs] = useState<TaskOutputs | null>(null)
  const [error, setError] = useState('')
  const task = `${snap.sessionKey}\u0000${snap.currentId}`
  const latest = useRef(task)
  latest.current = task
  const online = snap.connection === 'online'
  const refresh = useCallback(() => {
    if (!online) return
    const asked = task
    desktopCall(window.xerxes, snap.sessionKey, 'outputs.list', {})
      .then(result => { if (latest.current === asked) { setOutputs(taskOutputsView(result)); setError('') } })
      .catch((failure: unknown) => {
        if (latest.current !== asked) return
        const message = failure instanceof Error ? failure.message : String(failure)
        // No task yet: nothing written. An older runtime: say so once.
        if (/no active session/i.test(message)) { setOutputs({ cwd: snap.cwd, outputs: [] }); setError('') }
        else setError(/unknown method/i.test(message) ? 'The Outputs list needs the latest runtime.' : message)
      })
  }, [online, snap.sessionKey, snap.cwd, task])
  useEffect(() => { setOutputs(null) }, [task])
  useEffect(() => { if (!snap.turnActive) refresh() }, [refresh, snap.turnActive])
  return { outputs, error, refresh }
}

function OutputRow({ output, cwd, now }: { output: TaskOutput; cwd: string; now: number }): ReactElement {
  const navigate = useDesktopNavigation()
  const [failure, setFailure] = useState('')
  const slash = output.path.lastIndexOf('/')
  const name = output.path.slice(slash + 1)
  const folder = slash > 0 ? output.path.slice(0, slash) : ''
  const host = window.xerxes as { fileAction?: (action: 'open' | 'reveal', path: string, cwd: string) => Promise<{ ok: boolean; error?: string }> }
  const act = (action: 'open' | 'reveal') => {
    setFailure('')
    void host.fileAction?.(action, output.path, cwd)
      .then(result => { if (!result.ok) setFailure(result.error ?? 'could not open the file') })
      .catch((error: unknown) => setFailure(error instanceof Error ? error.message : String(error)))
  }
  return <li className="output-row">
    <button className="output-row__main" title={`Preview ${output.path}`} onClick={() => navigate('files', output.path)}>
      <Icon name="file" size={15} />
      <span className="output-row__text">
        <span className="output-row__name">{name}</span>
        <span className="output-row__meta">{[folder, fileSize(output.size), `turn ${output.turn}`, ago(output.modifiedAt, now)].filter(Boolean).join(' · ')}</span>
      </span>
    </button>
    {host.fileAction && <span className="output-row__actions">
      <button className="btn btn--ghost" onClick={() => act('open')} title="Open in its default app">Open</button>
      <button className="btn btn--ghost" onClick={() => act('reveal')} title="Show in Finder">Show</button>
    </span>}
    {failure && <p className="output-row__error" role="alert">{failure}</p>}
  </li>
}

/** The list itself, for the rail panel and the Artifacts page. */
export function OutputsList({ snap }: { snap: Snapshot }): ReactElement {
  const { outputs, error } = useTaskOutputs(snap)
  const now = Date.now()
  if (error) return <p className="usage__error" role="alert">{error}</p>
  if (!outputs) return <p className="usage__empty" role="status">{snap.connection === 'online' ? 'Looking for the files this task wrote…' : 'Outputs appear when the runtime is connected.'}</p>
  if (!outputs.outputs.length) return <p className="usage__empty">Nothing written yet. Files this task creates, by any tool or script, appear here.</p>
  const { results, scripts } = groupOutputs(outputs.outputs)
  return <div className="outputs">
    {results.length > 0 && <section className="usage__section" aria-labelledby="outputs-results">
      <header className="usage__head"><h3 id="outputs-results">Results</h3><span className="usage__sub">{results.length} file{results.length === 1 ? '' : 's'}</span></header>
      <ul className="output-list">{results.map(output => <OutputRow key={output.path} output={output} cwd={outputs.cwd} now={now} />)}</ul>
    </section>}
    {scripts.length > 0 && <section className="usage__section" aria-labelledby="outputs-scripts">
      <header className="usage__head"><h3 id="outputs-scripts">Scripts and scratch files</h3><span className="usage__sub">{scripts.length}</span></header>
      <ul className="output-list">{scripts.map(output => <OutputRow key={output.path} output={output} cwd={outputs.cwd} now={now} />)}</ul>
    </section>}
  </div>
}

export function OutputsPanel({ snap }: { snap: Snapshot }): ReactElement {
  return <div className="usage"><OutputsList snap={snap} /></div>
}
