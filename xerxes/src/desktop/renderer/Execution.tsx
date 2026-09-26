// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { memo, useState, type ReactElement } from 'react'
import type { ToolItem } from './types.js'
import { OutputViewer, readableOutput } from './OutputViewer.js'
import { Icon } from './Icon.js'
import { StructuredResult, structuredOutput } from './StructuredResult.js'
import { CopyButton } from './CopyButton.js'

function record(text: string): Record<string, unknown> | null {
  try {
    const value: unknown = JSON.parse(text)
    return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : null
  } catch { return null }
}
const failureByItem = new WeakMap<ToolItem, boolean>()
/** Share failure status with collapsed group headers without reparsing unchanged results. */
export function toolHasFailed(item: ToolItem): boolean {
  if (item.state === 'failed') return true
  const cached = failureByItem.get(item)
  if (cached !== undefined) return cached
  const exitCode = record(item.output)?.exitCode
  const failed = typeof exitCode === 'number' && exitCode !== 0
  failureByItem.set(item, failed)
  return failed
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
export const ToolCallRow = memo(function ToolCallRow({ item, label }: { item: ToolItem; label: string }): ReactElement {
  const [inspected, setInspected] = useState(false)
  const view = executionView(item)
  const failed = toolHasFailed(item)
  const target = view.command || item.path || item.arg
  return <details className="toolrow execution-row" data-state={failed ? 'failed' : item.state} data-command={view.command ? '' : undefined} onToggle={event => { if (event.currentTarget.open) setInspected(true) }}>
    <summary>
      <span className="execution-row__disclosure"><Icon name="chevron" size={12} /></span>
      <span className="execution-row__label"><Icon name={view.command ? 'terminal' : 'tools'} size={14} />{label}</span>
      <span className="execution-row__state">{item.state === 'working' ? 'Running' : failed ? 'Failed' : 'Done'}{item.dur && item.dur !== '0.0s' ? ` · ${item.dur}` : ''}</span>
      {target && target.trim() !== '{}' && <code className="execution-row__target">{target}</code>}
      {item.diff && <span className="execution-row__diff"><span className="add">+{item.diff.adds}</span> <span className="del">−{item.diff.dels}</span></span>}
      {failed && (item.error || view.stderr) && <span className="execution-row__error">{item.error || view.stderr}</span>}
    </summary>
    {inspected && <ExecutionDetails item={item} />}
  </details>
})
export function ExecutionDetails({ item }: { item: ToolItem }): ReactElement {
  const [expanded, setExpanded] = useState(false)
  const view = executionView(item)
  const failed = toolHasFailed(item)
  const output = view.stdout ?? readableOutput(item.output)
  const structured = view.stdout === null && output === item.output ? structuredOutput(item.output) : null
  return <div className="execution">
    <div className="execution__status" data-failed={failed || undefined}>
      <span>{item.state === 'working' ? 'Running' : failed ? 'Failed' : 'Completed'}{view.exitCode !== null ? ` · Exit ${view.exitCode}` : ''}</span>
      {view.cwd && <span title={view.cwd}>{view.cwd}</span>}
    </div>
    {view.command && <pre className="execution__command">{view.command}</pre>}
    {output && (structured ? <><div className="execution__viewer-actions"><span>Result</span><button onClick={()=>setExpanded(value=>!value)} aria-expanded={expanded}>{expanded ? 'Compact view' : 'Expand result'}</button></div><div className="execution__viewer" data-expanded={expanded || undefined} tabIndex={0} role="region" aria-label="Tool output"><StructuredResult value={structured}/></div></> : <OutputViewer text={output} />)}
    {view.stderr && <div className="execution__error"><strong>Standard error</strong><pre>{view.stderr}</pre></div>}
    {item.error && item.error !== output && item.error !== view.stderr && <div className="execution__error"><strong>Error</strong><pre>{item.error}</pre></div>}
    {!output && !item.error && !view.stderr && <p className="execution__empty">{item.state === 'working' ? 'Waiting for output…' : 'No output'}</p>}
    <div className="execution__actions">
      {view.command && <CopyButton text={view.command} label="Copy command" />}
      {/* glob, list_dir, memory, browser, skill and MCP results all land in
          the structured branch, which had no copy control — the output was
          reachable only by drag-selecting it. */}
      {output && <CopyButton text={output} label="Copy output" />}
      {item.input && <CopyButton text={item.input} label="Copy arguments" />}
    </div>
    <details className="execution__raw"><summary>Raw details</summary><code>{item.name} · {item.id}</code><strong>Input</strong><pre>{item.input || '(no arguments)'}</pre><strong>Result</strong><pre>{item.output || '(no result)'}</pre></details>
  </div>
}
