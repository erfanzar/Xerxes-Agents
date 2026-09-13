// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useState, type ReactElement } from 'react'
import type { ToolItem } from './types.js'

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
export function ExecutionDetails({ item }: { item: ToolItem }): ReactElement {
  const view = executionView(item)
  const failed = item.state === 'failed' || (view.exitCode !== null && view.exitCode !== 0)
  const readableOutput = view.stdout ?? item.output
  return <div className="execution">
    <div className="execution__status" data-failed={failed || undefined}>
      <span>{item.state === 'working' ? 'Running' : failed ? 'Failed' : 'Completed'}{view.exitCode !== null ? ` · Exit ${view.exitCode}` : ''}</span>
      {view.cwd && <span title={view.cwd}>{view.cwd}</span>}
    </div>
    {view.command && <pre className="execution__command">{view.command}</pre>}
    {readableOutput && <pre className="execution__output" aria-label="Command output">{readableOutput}</pre>}
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
