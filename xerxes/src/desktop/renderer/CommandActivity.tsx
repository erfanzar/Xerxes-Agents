// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { useEffect, useState, type ReactElement } from 'react'
import { desktopCall, desktopError, record, text, type RpcRecord } from './desktopRpc.js'
import { pageIsInactive } from './pageVisibility.js'
import { OutputViewer, readableOutput } from './OutputViewer.js'
import { Icon } from './Icon.js'
import { commandGist } from './commandGist.js'
import { elapsedOf } from './duration.js'

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
  const elapsed = seconds === null ? '' : elapsedOf(seconds)
  const fullCommand = text(detail?.command) || text(row.title)
  // The daemon's label is the agent's description, or the first words of the command when none was given.
  const label = text(row.label).trim()
  const heading = label && !text(row.title).startsWith(label.replace(/…$/, '')) ? label : commandGist(text(row.title))
  const state = text(row.state)
  const endLabel = state === 'completed' ? 'Completed' : state === 'failed' ? 'Failed' : state ? state[0]!.toUpperCase() + state.slice(1) : 'Finished'
  const outcome = !online ? 'Disconnected. Showing the retained output.'
    : running ? 'Running. Updates every 2 seconds.'
    : `${endLabel}${typeof row.exitCode === 'number' ? ` with exit code ${row.exitCode}` : ''}.`
  useEffect(() => {
    if (!open || !online) return
    let current = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const load = async () => {
      if (pageIsInactive()) { if (current && running) timer = setTimeout(() => void load(), 5000); return }
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
    {/* Claude Code's task card: what the command is for (the agent's description, else the
        command's gist), and its duration. The raw command waits inside. */}
    <summary title={`${heading}\n${text(row.title)}`}>
      <Icon name="chevron" size={12}/>
      {/* A span, not <code>: the inline-code chip style painted a grey pill behind every row. */}
      <span className="command-activity__preview">{heading}</span>
      {!online
        ? <span className="command-activity__state">Disconnected</span>
        : running ? <span className="command-activity__state">running</span> : null}
      {elapsed && <span className="command-activity__elapsed">{elapsed}</span>}
    </summary>
    {/* Open: `$ command`, its output, a hairline, then one sentence saying how it ended. */}
    {open && <div className="command-activity__body">
      <pre className="command-activity__command" aria-label="Full command" tabIndex={0}><span aria-hidden="true">$ </span>{fullCommand}</pre>
      {(detail ? readableOutput(text(detail.output)).trim() : '') !== '' && <OutputViewer compact text={readableOutput(text(detail!.output))} />}
      {error && <p role="alert" className="studio-error">{error}</p>}
      <div className="command-activity__foot">
        {/* One short outcome line; the command is already shown above, never repeated here. */}
        <span>{outcome}{detail?.outputTruncated === true ? ' Showing the last part of its output.' : ''}</span>
        {running && <button type="button" disabled={!online} onClick={()=>setRetry(value=>value+1)}>Refresh</button>}
        {running && detail?.canInterrupt === true && <button type="button" disabled={!online || busy} onClick={()=>void interrupt()}>{busy ? 'Interrupting…' : 'Interrupt'}</button>}
      </div>
    </div>}
  </details>
}
