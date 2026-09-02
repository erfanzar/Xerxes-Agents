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
        Shells the agents started while working in this session — watching never consumes: the retained tail below is a mirror, so peeking never steals the model's output.
      </p>
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
        {terminals.length === 0 && online && (
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

function TerminalCard({ row, online }: { row: TerminalRow; online: boolean }): ReactElement {
  const [open, setOpen] = useState(false)
  const [output, setOutput] = useState('')
  const [truncated, setTruncated] = useState(false)
  const [loading, setLoading] = useState(false)
  const [draft, setDraft] = useState('')
  const running = row.running
  const state = running ? 'running' : row.exitCode === null ? 'exited' : `exit ${row.exitCode}`

  const inspect = (): void => {
    setLoading(true)
    store.inspectTerminal(row.id)
      .then(detail => {
        if (detail) {
          setOutput(detail.output)
          setTruncated(detail.outputTruncated)
          setOpen(true)
        }
      })
      .catch(() => {})
      .finally(() => setLoading(false))
  }

  const send = (): void => {
    // The trailing newline is the Enter key — `terminal.control` deliberately
    // does not trim `chars`.
    if (!draft.trim()) return
    store.controlTerminal(row.id, 'write', `${draft}\n`)
    setDraft('')
  }

  return (
    <div className="pcard">
      <div className="pcard__main">
        <span className={`dot ${running ? 'dot--live' : 'dot--idle'}`} />
        <span className="pcard__text">
          <span className="pcard__name">{row.label || row.command.slice(0, 48) || row.id}</span>
          <span className="pcard__meta">
            <code>{row.kind}</code> · {state}
            {row.pid ? ` · pid ${row.pid}` : ''}
            {clockOf(row.startedAt) ? ` · started ${clockOf(row.startedAt)}` : ''}
            {running && clockOf(row.endedAt) ? ` · ended ${clockOf(row.endedAt)}` : ''}
            {` · ${row.outputChars} chars seen`}
          </span>
          <span className="pcard__meta" title={row.command}>
            <code>{row.command}</code>
          </span>
        </span>
      </div>
      <div className="pcard__actions">
        <button className="chipbtn" disabled={!online || loading} onClick={inspect}>
          {loading ? 'Reading…' : open ? '↻ Output' : 'Inspect'}
        </button>
        {row.canInterrupt && running && (
          <button
            className="chipbtn"
            disabled={!online}
            title="Send Ctrl+C to the live process"
            onClick={() => store.controlTerminal(row.id, 'interrupt')}
          >
            Interrupt
          </button>
        )}
        {row.canKill && running && (
          <button
            className="chipbtn chipbtn--danger"
            disabled={!online}
            title="Terminate the process (SIGTERM)"
            onClick={() => { if (window.confirm(`Kill terminal ${row.label || row.id}?`)) store.controlTerminal(row.id, 'kill') }}
          >
            Kill
          </button>
        )}
      </div>
      {open && (
        <div className="provform">
          <div className="row__t">Output tail · {row.id}</div>
          <pre className="preset-composition">{output || (row.running ? '(no output yet)' : '(no output recorded)')}</pre>
          {truncated && <div className="row__s">older output was dropped from the mirror</div>}
          {row.canWrite && running && (
            <div className="findwrap">
              <input
                className="side__search findwrap__input"
                value={draft}
                spellCheck={false}
                placeholder="send a line to this shell…"
                onChange={event => setDraft(event.target.value)}
                onKeyDown={event => {
                  if (event.key === 'Enter') { event.preventDefault(); send() }
                  if (event.key === 'Escape') setDraft('')
                }}
              />
              <button className="btn btn--solid" disabled={!draft.trim() || !online} onClick={send}>Send ⏎</button>
            </div>
          )}
          <div className="preset-actions">
            <button className="btn btn--ghost" onClick={() => setOpen(false)}>Close</button>
          </div>
        </div>
      )}
    </div>
  )
}
