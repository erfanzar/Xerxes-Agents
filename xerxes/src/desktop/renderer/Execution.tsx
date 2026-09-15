// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'
import type { ToolItem } from './types.js'
import { Icon } from './Icon.js'
import { StructuredResult, structuredOutput } from './StructuredResult.js'

function record(text: string): Record<string, unknown> | null {
  try {
    const value: unknown = JSON.parse(text)
    return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
  } catch { return null }
}
function shellArgument(value: string): string {
  return /^[a-zA-Z0-9_./:=,+-]+$/.test(value) ? value : `'${value.replaceAll("'", "'\\''")}'`
}
export function executionView(item: ToolItem) {
  const input = record(item.input)
  const output = record(item.output)
  const commandParts = Array.isArray(output?.command) ? output.command : input && typeof input.cmd === 'string' && Array.isArray(input.args) ? [input.cmd, ...input.args] : null
  const command = commandParts?.every((part): part is string => typeof part === 'string')
    ? commandParts.map(shellArgument).join(' ')
    : typeof input?.cmd === 'string' ? input.cmd : typeof input?.command === 'string' ? input.command : null
  return {
    command,
    stdout: typeof output?.stdout === 'string' ? output.stdout : null,
    stderr: typeof output?.stderr === 'string' ? output.stderr : null,
    exitCode: typeof output?.exitCode === 'number' ? output.exitCode : null,
    cwd: typeof output?.cwd === 'string' ? output.cwd : null,
  }
}
function CopyButton({ text, label }: { text: string; label: string }): ReactElement {
  const [status, setStatus] = useState('')
  return <button onClick={() => { void navigator.clipboard.writeText(text).then(() => setStatus('Copied'), () => setStatus('Copy failed')) }}>{status || label}</button>
}
export function ToolCallRow({ item, label }: { item: ToolItem; label: string }): ReactElement {
  const view = executionView(item)
  const failed = item.state === 'failed' || (view.exitCode !== null && view.exitCode !== 0)
  const target = view.command || item.path || item.arg
  return <details className="toolrow execution-row" data-state={failed ? 'failed' : item.state}>
    <summary>
      <span className="execution-row__disclosure"><Icon name="chevron" size={12} /></span>
      <span className="execution-row__label"><Icon name={view.command ? 'terminal' : 'tools'} size={14} />{label}</span>
      <span className="execution-row__state">{item.state === 'working' ? 'Running' : failed ? 'Failed' : 'Done'}{item.dur && item.dur !== '0.0s' ? ` · ${item.dur}` : ''}</span>
      {target && target.trim() !== '{}' && <code className="execution-row__target">{target}</code>}
      {item.diff && <span className="execution-row__diff"><span className="add">+{item.diff.adds}</span> <span className="del">−{item.diff.dels}</span></span>}
      {failed && (item.error || view.stderr) && <span className="execution-row__error">{item.error || view.stderr}</span>}
    </summary>
    <ExecutionDetails item={item} />
  </details>
}
export function ExecutionDetails({ item }: { item: ToolItem }): ReactElement {
  const [expanded, setExpanded] = useState(false)
  const [wrap, setWrap] = useState(true)
  const view = executionView(item)
  const failed = item.state === 'failed' || (view.exitCode !== null && view.exitCode !== 0)
  const readableOutput = view.stdout ?? item.output
  const structured = view.stdout === null ? structuredOutput(item.output) : null
  return <div className="execution">
    <div className="execution__status" data-failed={failed || undefined}>
      <span>{item.state === 'working' ? 'Running' : failed ? 'Failed' : 'Completed'}{view.exitCode !== null ? ` · Exit ${view.exitCode}` : ''}</span>
      {view.cwd && <span title={view.cwd}>{view.cwd}</span>}
    </div>
    {view.command && <pre className="execution__command">{view.command}</pre>}
    {readableOutput && <><div className="execution__viewer-actions"><span>{structured ? 'Result' : 'Output'}</span><button onClick={()=>setExpanded(value=>!value)} aria-expanded={expanded}>{expanded ? 'Compact view' : 'Expand output'}</button>{!structured && <button onClick={()=>setWrap(value=>!value)} aria-pressed={wrap}>Wrap lines</button>}</div><div className="execution__viewer" data-expanded={expanded || undefined} tabIndex={0} role="region" aria-label="Tool output">{structured ? <StructuredResult value={structured}/> : <pre className="execution__output" data-wrap={wrap} aria-label="Command output">{readableOutput}</pre>}</div></>}
    {view.stderr && <div className="execution__error"><strong>Standard error</strong><pre>{view.stderr}</pre></div>}
    {item.error && <div className="execution__error"><strong>Error</strong><pre>{item.error}</pre></div>}
    {!readableOutput && !item.error && !view.stderr && <p className="execution__empty">{item.state === 'working' ? 'Waiting for output…' : 'No output'}</p>}
    <div className="execution__actions">
      {view.command && <CopyButton text={view.command} label="Copy command" />}
      {readableOutput && <CopyButton text={readableOutput} label="Copy output" />}
    </div>
    <details className="execution__raw"><summary>Raw details</summary><code>{item.name} · {item.id}</code><strong>Input</strong><pre>{item.input || '(no arguments)'}</pre><strong>Result</strong><pre>{item.output || '(no result)'}</pre></details>
  </div>
}
