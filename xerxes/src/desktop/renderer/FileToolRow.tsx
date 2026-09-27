// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Read, write and edit calls drawn as what they did to a file (Claude
 * Code's), rather than as a generic tool row with raw arguments: an edit is
 * its diff, a write is the new text, a read is the lines it returned. The
 * path opens the file in the Files panel. The raw call stays one click away.
 */

import { memo, useState, type ReactElement } from 'react'
import type { ToolItem } from './types.js'
import { Icon } from './Icon.js'
import { CopyButton } from './CopyButton.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import { parseArgs } from './blocks.js'

export type FileTool =
  | { readonly kind: 'read'; readonly path: string; readonly offset: number }
  | { readonly kind: 'edit'; readonly path: string; readonly before: string; readonly after: string; readonly wholeFile: boolean }
  | { readonly kind: 'write'; readonly path: string; readonly content: string; readonly append: boolean }

/** The file tool a call is, from its name and arguments; null for any other tool. */
export function fileToolOf(item: Pick<ToolItem, 'name' | 'input'>): FileTool | null {
  const name = item.name.split(/[.:]/).pop()?.toLowerCase() ?? ''
  if (!['readfile', 'fileedittool', 'writefile', 'appendfile'].includes(name)) return null
  const args = parseArgs(item.input)
  const path = typeof args.file_path === 'string' ? args.file_path : typeof args.path === 'string' ? args.path : ''
  if (!path) return null
  const text = (value: unknown): string => typeof value === 'string' ? value : ''
  if (name === 'readfile') return { kind: 'read', path, offset: typeof args.offset === 'number' && args.offset > 0 ? Math.floor(args.offset) : 0 }
  if (name === 'fileedittool') return { kind: 'edit', path, before: text(args.old_string), after: text(args.new_string), wholeFile: args.edit_mode === 'whole_file' }
  return { kind: 'write', path, content: text(args.content), append: name === 'appendfile' }
}

export type DiffLine = { readonly kind: 'ctx' | 'add' | 'del'; readonly text: string }

const splitLines = (text: string): string[] => text ? text.replace(/\n$/, '').split('\n') : []

/**
 * A line diff of an edit's old and new text: common lines as context,
 * the rest removed then added, in order (longest common subsequence).
 * Past a size where that gets costly, it is all-removed then all-added.
 */
export function lineDiff(before: string, after: string, limit = 400): DiffLine[] {
  const a = splitLines(before), b = splitLines(after)
  if (a.length > limit || b.length > limit) return [...a.map(text => ({ kind: 'del' as const, text })), ...b.map(text => ({ kind: 'add' as const, text }))]
  const table = Array.from({ length: a.length + 1 }, () => new Array<number>(b.length + 1).fill(0))
  for (let i = a.length - 1; i >= 0; i -= 1) for (let j = b.length - 1; j >= 0; j -= 1) table[i]![j] = a[i] === b[j] ? table[i + 1]![j + 1]! + 1 : Math.max(table[i + 1]![j]!, table[i]![j + 1]!)
  const out: DiffLine[] = []
  let i = 0, j = 0
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) { out.push({ kind: 'ctx', text: a[i]! }); i += 1; j += 1 }
    else if (j < b.length && (i >= a.length || table[i]![j + 1]! >= table[i + 1]![j]!)) { out.push({ kind: 'add', text: b[j]! }); j += 1 }
    else { out.push({ kind: 'del', text: a[i]! }); i += 1 }
  }
  // Removals first within each changed run reads like a patch.
  const ordered: DiffLine[] = []
  for (let k = 0; k < out.length;) {
    if (out[k]!.kind === 'ctx') { ordered.push(out[k]!); k += 1; continue }
    const run: DiffLine[] = []
    while (k < out.length && out[k]!.kind !== 'ctx') run.push(out[k++]!)
    ordered.push(...run.filter(line => line.kind === 'del'), ...run.filter(line => line.kind === 'add'))
  }
  return ordered
}

/** A read's returned text without the tool's own trailing continuation notice. */
export function readBody(output: string): string {
  return output.replace(/\n\n\[ReadFile\] Showing lines[\s\S]*$/, '')
}

const PREVIEW_LINES = 14
const VERBS = { read: 'Read', edit: 'Edit', write: 'Write' } as const

function DiffBlock({ lines, numberFrom }: { lines: readonly DiffLine[]; numberFrom?: number }): ReactElement {
  const [all, setAll] = useState(false)
  const shown = all ? lines : lines.slice(0, PREVIEW_LINES)
  return <div className="filetool__code">
    <pre tabIndex={0} aria-label="Changes">{shown.map((line, index) => <span key={index} className={`filetool__line filetool__line--${line.kind}`}>
      <span className="filetool__gutter" aria-hidden="true">{numberFrom !== undefined ? numberFrom + index : line.kind === 'add' ? '+' : line.kind === 'del' ? '−' : ''}</span>
      <code>{line.text || ' '}</code>
    </span>)}</pre>
    {lines.length > PREVIEW_LINES && <button className="filetool__more" onClick={() => setAll(value => !value)}>{all ? 'Show less' : `Show ${lines.length - PREVIEW_LINES} more line${lines.length - PREVIEW_LINES === 1 ? '' : 's'}`}</button>}
  </div>
}

export const FileToolRow = memo(function FileToolRow({ item, tool, failed }: { item: ToolItem; tool: FileTool; failed: boolean }): ReactElement {
  const open = useDesktopNavigation()
  const [expanded, setExpanded] = useState(false)
  const lines = tool.kind === 'edit' ? lineDiff(tool.before, tool.after)
    : tool.kind === 'write' ? splitLines(tool.content).map(text => ({ kind: 'add' as const, text }))
    : splitLines(readBody(item.output)).map(text => ({ kind: 'ctx' as const, text }))
  const adds = lines.filter(line => line.kind === 'add').length
  const dels = lines.filter(line => line.kind === 'del').length
  const verb = tool.kind === 'write' && tool.append ? 'Append' : tool.kind === 'edit' && tool.wholeFile ? 'Rewrite' : VERBS[tool.kind]
  const error = failed ? (item.error || item.output).split('\n')[0]!.replace(/^Tool execution failed: /, '').slice(0, 240) : ''
  // An edit or write shows what it changes; a read folds its text away.
  const body = !failed && (tool.kind !== 'read' || expanded)
  return <div className="filetool" data-kind={tool.kind} data-state={failed ? 'failed' : item.state}>
    <div className="filetool__head">
      <span className="filetool__verb">{item.state === 'working' ? <Icon name="spinner" size={13} /> : null}{verb}</span>
      <button className="filetool__path" title={`Open ${tool.path} in Files`} onClick={() => open('files', tool.path)}>{tool.path}</button>
      {tool.kind === 'read' && !failed && item.state !== 'working' && lines.length > 0 && <button className="filetool__toggle" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>
        {`lines ${tool.offset + 1}–${tool.offset + lines.length}`}<Icon name="chevron" size={11} />
      </button>}
      {tool.kind !== 'read' && !failed && item.state !== 'working' && <span className="filetool__stats"><span className="add">+{adds}</span>{tool.kind === 'edit' && <span className="del">−{dels}</span>}</span>}
      <span className="filetool__state">{failed ? 'Failed' : item.state === 'working' ? 'Running' : item.dur && item.dur !== '0.0s' ? item.dur : ''}</span>
    </div>
    {error && <p className="filetool__error" title={item.error || item.output}>{error}</p>}
    {body && lines.length > 0 && <DiffBlock lines={lines} {...(tool.kind === 'read' ? { numberFrom: tool.offset + 1 } : {})} />}
    {body && tool.kind === 'read' && <div className="filetool__actions"><CopyButton icon text={readBody(item.output)} label="Copy text" /></div>}
  </div>
})
