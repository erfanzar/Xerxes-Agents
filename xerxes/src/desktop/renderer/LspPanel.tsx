// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useRef, useState, type ReactElement } from 'react'
import { desktopCall, desktopError, record, records, text } from './desktopRpc.js'
import type { Snapshot } from './store.js'

type Server = { name: string; enabled: boolean; languageId: string; extensions: string[]; timeoutMs: number }
type View = { revision: string; servers: Server[]; warnings: string[] }
export function readLspView(value: unknown): View {
  const data = record(value)
  if (typeof data.revision !== 'string' || !Array.isArray(data.warnings) || !data.warnings.every(v => typeof v === 'string')) throw new Error('Invalid language server settings')
  const servers = records(data.servers).map(row => {
    if (!text(row.name) || typeof row.enabled !== 'boolean' || typeof row.languageId !== 'string' || !Array.isArray(row.extensions) || !row.extensions.every(v => typeof v === 'string') || typeof row.timeoutMs !== 'number') throw new Error('Invalid language server settings')
    return { name: row.name as string, enabled: row.enabled, languageId: row.languageId, extensions: row.extensions as string[], timeoutMs: row.timeoutMs }
  })
  return { revision: data.revision, servers, warnings: data.warnings as string[] }
}

export function lspChanges(form: FormData): Record<string, unknown> {
  const changes: Record<string, unknown> = {
    enabled: form.get('enabled') === 'on', languageId: String(form.get('languageId') ?? '').trim(),
    extensions: String(form.get('extensions') ?? '').split(',').map(s => s.trim()).filter(Boolean),
    timeoutMs: Number(form.get('timeoutMs')),
  }
  const command = String(form.get('command') ?? '').trim()
  if (command) changes.command = command
  for (const key of ['args', 'env']) {
    const raw = String(form.get(key) ?? '').trim()
    if (!raw) continue
    const parsed: unknown = JSON.parse(raw)
    if (parsed !== null && (key === 'args' ? !Array.isArray(parsed) || !parsed.every(v => typeof v === 'string') : !parsed || typeof parsed !== 'object' || Array.isArray(parsed) || !Object.values(parsed).every(v => typeof v === 'string'))) throw new Error(key === 'args' ? 'Arguments must be a JSON array of strings.' : 'Environment must be a JSON object with string values.')
    changes[key] = parsed
  }
  return changes
}

export function LspCard({ snap }: { snap: Snapshot }): ReactElement {
  const [view, setView] = useState<View | null>(null)
  const [editing, setEditing] = useState<Server | 'new' | null>(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [busy, setBusy] = useState(false)
  const pending = useRef(false)
  const epoch = useRef(0)
  const available = snap.connection === 'online' && !!snap.sessionKey
  const run = async (method: string, params: Record<string, unknown> = {}): Promise<void> => {
    if (pending.current || !available) return
    const generation = epoch.current
    pending.current = true; setBusy(true); setError(''); setNotice('')
    try {
      const result = await desktopCall(window.xerxes, snap.sessionKey, method, params)
      if (generation !== epoch.current) return
      if (method === 'lsp.release') setNotice(text(result.message))
      else { setView(readLspView(result)); setEditing(null) }
    } catch (failure) { if (generation === epoch.current) setError(desktopError(failure)) }
    finally { if (generation === epoch.current) { pending.current = false; setBusy(false) } }
  }
  useEffect(() => {
    epoch.current++; pending.current = false; setView(null); setEditing(null); setBusy(false)
    void run('lsp.settings.get')
    return () => { epoch.current++ }
  }, [snap.sessionKey, snap.connection])
  const server = editing && editing !== 'new' ? editing : null
  return <>
    <h2 className="modal__title">Language servers</h2>
    <p className="modal__sub">Configure code navigation and diagnostics. Servers start when a task needs them.</p>
    {!available && <p role="status">Open a connected session to manage language servers.</p>}
    {error && <p className="studio-error" role="alert">{error}</p>}
    {notice && <p role="status">{notice}</p>}
    {busy && <p role="status">Updating language servers…</p>}
    {view?.warnings.map((warning, i) => <p role="status" key={i}>{warning}</p>)}
    <div className="rowlist">{view?.servers.map(row => <div className="row" key={row.name}>
      <div className="row__main"><div className="row__t">{row.name}</div><div className="row__s">{row.languageId} · {row.extensions.join(', ')} · {row.enabled ? 'Enabled' : 'Disabled'}</div></div>
      <button className="btn btn--ghost" disabled={busy || !available} onClick={() => setEditing(row)}>Edit</button>
      <button className="btn btn--ghost" disabled={busy || !available} onClick={() => void run('lsp.release', { name: row.name })}>Release host</button>
    </div>)}</div>
    {view && !view.servers.length && <p>No language servers configured.</p>}
    <div className="lsp-actions"><button className="btn btn--ghost" disabled={busy || !available} onClick={() => void run('lsp.settings.get')}>Reload</button><button className="btn" disabled={busy || !available || !view} onClick={() => setEditing('new')}>Add server</button></div>
    {editing && <form className="lsp-editor" key={server?.name ?? 'new'} onSubmit={event => {
      event.preventDefault()
      try { const data = new FormData(event.currentTarget); void run('lsp.settings.save', { revision: view?.revision, name: server?.name ?? String(data.get('name')).trim(), action: server ? 'update' : 'create', changes: lspChanges(data) }) }
      catch (failure) { setError(desktopError(failure)) }
    }}>
      <h3>{server ? `Edit ${server.name}` : 'Add language server'}</h3>
      <fieldset disabled={busy || !available}>
        {!server && <div className="field"><label htmlFor="lsp-name">Name</label><input id="lsp-name" name="name" required maxLength={128}/></div>}
        <label><input name="enabled" type="checkbox" defaultChecked={server?.enabled ?? true}/> Enabled</label>
        <div className="field"><label htmlFor="lsp-language">Language ID</label><input id="lsp-language" name="languageId" required defaultValue={server?.languageId}/></div>
        <div className="field"><label htmlFor="lsp-ext">File extensions, separated by commas</label><input id="lsp-ext" name="extensions" required placeholder=".ts, .tsx" defaultValue={server?.extensions.join(', ')}/></div>
        <div className="field"><label htmlFor="lsp-timeout">Timeout (milliseconds)</label><input id="lsp-timeout" name="timeoutMs" type="number" min={1} required defaultValue={server?.timeoutMs ?? 30000}/></div>
        <p className="modal__sub">{server ? 'Saved command, arguments and environment are private. Leave a field blank to keep its value.' : 'Enter the executable and any arguments it needs.'} Use null to clear arguments or environment.</p>
        <div className="field"><label htmlFor="lsp-command">Executable</label><input id="lsp-command" name="command" required={!server} autoComplete="off"/></div>
        <div className="field"><label htmlFor="lsp-args">Arguments (JSON array)</label><textarea id="lsp-args" name="args" rows={2} placeholder={'["--stdio"]'}/></div>
        <div className="field"><label htmlFor="lsp-env">Environment (JSON object)</label><textarea id="lsp-env" name="env" rows={2} autoComplete="off" spellCheck={false}/></div>
        <div className="lsp-actions"><button className="btn" type="submit">Save server</button><button className="btn btn--ghost" type="button" onClick={() => setEditing(null)}>Cancel</button>{server && <button className="btn btn--ghost" type="button" onClick={() => void run('lsp.settings.save', { revision: view?.revision, name: server.name, action: 'remove' })}>Remove server</button>}</div>
      </fieldset>
    </form>}
  </>
}
