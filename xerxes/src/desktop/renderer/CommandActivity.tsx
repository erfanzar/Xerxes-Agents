// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useState, type ReactElement } from 'react'
import { desktopCall, desktopError, record, text, type RpcRecord } from './desktopRpc.js'
import { OutputViewer, readableOutput } from './OutputViewer.js'
import { Icon } from './Icon.js'
import { commandGist } from './commandGist.js'

/** A bounded summary first; read only this command's retained output on expansion. */
export function CommandActivity({ row, sessionKey, online }: {row: RpcRecord; sessionKey: string; online: boolean}): ReactElement {
  const [open, setOpen] = useState(false)
  const [detail, setDetail] = useState<RpcRecord | null>(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [retry, setRetry] = useState(0)
  const running = row.state === 'running'
  const started = typeof row.startedAt === 'number' ? row.startedAt : null
  const ended = typeof row.endedAt === 'number' ? row.endedAt : Date.now()
  const seconds = started === null ? null : Math.max(0, Math.floor((ended - started) / 1000))
  const elapsed = seconds === null ? '' : seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m ${seconds % 60}s`
  useEffect(() => {
    if (!open || !online) return
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      try {
        const result = await desktopCall(window.xerxes, sessionKey, 'terminal.inspect', {terminal_id: row.id, max_output_chars: 24000})
        if (current) {setDetail(record(result.terminal));setError('')}
      } catch (failure) { if (current) setError(desktopError(failure)) }
      if (current && running) timer = setTimeout(() => void load(), 2000)
    }
    void load()
    return () => {current=false;if(timer)clearTimeout(timer)}
  }, [open, online, row.id, sessionKey, running, retry])
  const interrupt = async () => {
    if (busy || !online) return
    setBusy(true)
    try {await desktopCall(window.xerxes, sessionKey, 'terminal.control', {terminal_id:row.id, action:'interrupt'});setRetry(value=>value+1)}
    catch (failure) {setError(desktopError(failure))}
    finally {setBusy(false)}
  }
  return <details className="command-activity" data-state={text(row.state)} open={open} onToggle={event=>setOpen(event.currentTarget.open)}>
    {/* One line: the command, its state as a dot, and how long it took.
        The old summary led with the words "Shell command" and pushed the
        command itself to a third line, so the least useful thing on the
        row was the most prominent. */}
    <summary title={`${text(row.title)}\n${text(row.detail)}`}>
      <Icon name="chevron" size={12}/>
      <span className="command-activity__dot" aria-hidden="true" />
      <code className="command-activity__preview">{commandGist(text(row.title))}</code>
      {!online
        ? <span className="command-activity__state">disconnected</span>
        : running ? <span className="command-activity__state">running</span> : null}
      {elapsed && <span className="command-activity__elapsed">{elapsed}</span>}
    </summary>
    {open && <div className="command-activity__body">
<details className="command-activity__source"><summary>Full command</summary><pre className="command-activity__command" aria-label="Full command" tabIndex={0}>{text(detail?.command) || text(row.title)}</pre></details>
      <div className="command-activity__toolbar"><strong>{running ? 'Updates every 2 seconds' : 'Recorded output'}</strong><button disabled={!online} onClick={()=>setRetry(value=>value+1)}>Refresh</button>
        {running && detail?.canInterrupt === true && <button disabled={!online || busy} onClick={()=>void interrupt()}>{busy ? 'Requesting…' : 'Interrupt'}</button>}
      </div>
      {!online && <p role="status">Disconnected. Retained output is shown; reconnect to refresh or control this command.</p>}
      {error && <p role="alert" className="studio-error">{error}</p>}
<OutputViewer text={detail ? readableOutput(text(detail.output)) : ''} empty={detail ? 'No output recorded yet.' : error ? 'Output is unavailable.' : 'Loading output…'} />
      {detail?.outputTruncated === true && <p className="studio-muted">Showing the retained output tail.</p>}
    </div>}
  </details>
}
