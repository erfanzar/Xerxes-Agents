// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Settings → Terminals: every shell the daemon currently tracks for this
 * session — foreground execs, background commands, PTY seats — with the
 * control surface `terminal.control` actually allows per row (write,
 * interrupt, kill). Inspect reads a retained output tail; it never drains
 * the model's own buffer.
 */

import { useEffect, useState, type ReactElement } from 'react'

import { store, type Snapshot } from './store.js'
import type { TerminalRow } from './types.js'
import { desktopError } from './desktopRpc.js'

function clockOf(epoch: number | undefined): string {
  if (epoch === undefined) return ''
  const date = new Date(epoch)
  if (Number.isNaN(date.getTime())) return ''
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

export function TerminalsCard({ snap }: { snap: Snapshot }): ReactElement {
  // Terminals live and die with turns; the open view re-reads, plus a manual refresh.
  useEffect(() => { store.loadTerminals() }, [])
  const terminals = snap.terminals
  const online = snap.connection === 'online'

  return (
    <>
      <h2 className="modal__title">Terminals</h2>
      <p className="modal__sub">
        Commands started in this session. Inspect retained output without interrupting the process.
      </p>
      {snap.terminalsError && <p className="studio-error" role="alert">{snap.terminalsError}</p>}
      {!online && (
        <div className="row">
          <span className="dot dot--idle" />
          <div className="row__main">
            <div className="row__t">Daemon offline</div>
            <div className="row__s">reconnect to list and control terminals</div>
          </div>
        </div>
      )}
      <div className="rowlist">
        {terminals.length === 0 && online && !snap.terminalsError && (
          <div className="row">
            <span className="dot dot--idle" />
            <div className="row__main">
              <div className="row__t">{snap.terminalsLoading ? 'Asking the daemon…' : 'No terminals yet'}</div>
              <div className="row__s">commands the agents run this session appear here while they exist</div>
            </div>
          </div>
        )}
        {terminals.map(row => (
          <TerminalCard key={row.id} row={row} online={online} />
        ))}
      </div>
      <div style={{ display: 'flex', gap: 8, paddingTop: 16 }}>
        <button className="btn btn--ghost" disabled={!online || snap.terminalsLoading} onClick={() => store.loadTerminals()}>
          ↻ Refresh
        </button>
      </div>
    </>
  )
}

export function TerminalCard({ row, online }: { row: TerminalRow; online: boolean }): ReactElement {
  const [open, setOpen] = useState(false)
  const [output, setOutput] = useState('')
  const [truncated, setTruncated] = useState(false)
  const [loading, setLoading] = useState(false)
  const [draft, setDraft] = useState('')
  const [error, setError] = useState('')
  const [pending, setPending] = useState<'write' | 'interrupt' | 'kill' | null>(null)
  const running = row.running
  const state = running ? 'running' : row.exitCode === null ? 'exited' : `exit ${row.exitCode}`

  const inspect = (): void => {
    setLoading(true)
    setError('')
    store.inspectTerminal(row.id)
      .then(detail => {
        if (detail) {
          setOutput(detail.output)
          setTruncated(detail.outputTruncated)
          setOpen(true)
        } else setError('This terminal no longer has a retained result. Refresh the terminal list.')
      })
      .catch(failure => setError(desktopError(failure)))
      .finally(() => setLoading(false))
  }

  const control = async (action: 'write' | 'interrupt' | 'kill'): Promise<void> => {
    if (pending || !online || !running) return
    setPending(action)
    setError('')
    try {
      await store.controlTerminal(row.id, action, action === 'write' ? `${draft}\n` : undefined)
      if (action === 'write') setDraft('')
    } catch (failure) {
      setError(desktopError(failure))
    } finally {
      setPending(null)
    }
  }

  const send = (): void => {
    // The trailing newline is the Enter key — `terminal.control` deliberately
    // does not trim `chars`.
    if (!draft.trim()) return
    void control('write')
  }

  return (
    <section className="terminal-row" aria-busy={pending !== null}>
      <div className="terminal-row__head">
        <span className={`dot ${running ? 'dot--live' : 'dot--idle'}`} />
        <div className="terminal-row__identity">
          <code className="terminal-row__command">{row.command || row.label || 'Terminal'}</code>
          <div className="terminal-row__meta">
            {state}
            {clockOf(row.startedAt) ? ` · started ${clockOf(row.startedAt)}` : ''}
            {!running && clockOf(row.endedAt) ? ` · ended ${clockOf(row.endedAt)}` : ''}
          </div>
        </div>
      <div className="terminal-row__actions">
        <button className="chipbtn" disabled={!online || loading || pending !== null} onClick={inspect}>
          {loading ? 'Reading…' : open ? 'Refresh output' : 'Inspect'}
        </button>
        {row.canInterrupt && running && (
          <button
            className="chipbtn"
            disabled={!online || pending !== null}
            title="Send Ctrl+C to the live process"
            onClick={() => { void control('interrupt') }}
          >
            Interrupt
          </button>
        )}
        {row.canKill && running && (
          <button
            className="chipbtn chipbtn--danger"
            disabled={!online || pending !== null}
            title="Terminate the process (SIGTERM)"
            onClick={() => { if (window.confirm(`Kill terminal ${row.label || row.id}?`)) void control('kill') }}
          >
            Kill
          </button>
        )}
      </div>
      </div>
      {error && <p className="studio-error" role="alert">{error}</p>}
      {open && (
        <div className="terminal-row__output">
          <pre tabIndex={0} role="region" aria-label="Terminal output">{output || (row.running ? 'No output yet.' : 'No output recorded.')}</pre>
          {truncated && <div className="row__s">older output was dropped from the mirror</div>}
          {row.canWrite && running && (
            <div className="findwrap">
              <input
                className="side__search findwrap__input"
                value={draft}
                aria-label="Terminal input"
                disabled={!online || pending !== null}
                spellCheck={false}
                placeholder="send a line to this shell…"
                onChange={event => setDraft(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter') { event.preventDefault(); send() }
                  if (event.key === 'Escape') setDraft('')
                }}
              />
              <button className="btn btn--solid" disabled={!draft.trim() || !online || pending !== null} onClick={send}>{pending === 'write' ? 'Sending…' : 'Send ⏎'}</button>
            </div>
          )}
          <div className="preset-actions">
            <button className="btn btn--ghost" onClick={() => setOpen(false)}>Close</button>
          </div>
          <details className="terminal-row__diagnostics"><summary>Process details</summary><p>{row.kind}{row.pid ? ` · PID ${row.pid}` : ''}</p><p>{row.id}</p></details>
        </div>
      )}
    </section>
  )
}
