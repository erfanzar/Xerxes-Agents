// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Read, write, edit, patch and outline calls rendered as semantic file cards
 * with clickable basename + muted parent directory, crisp `+N −M` diff pills,
 * collapsible unified diff views with line-number gutters, outline symbol
 * lists, copy actions, and clean error callouts.
 */

import { createContext, memo, useContext, useEffect, useMemo, useState, type ReactElement } from 'react'
import type { ToolItem } from './types.js'
import { Icon, type IconName } from './Icon.js'
import { CopyButton } from './CopyButton.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import { ToolImages } from './ImageView.js'
import { parseArgs, trimmedChangedLines } from './blocks.js'

export type DiffLine = {
  readonly kind: 'ctx' | 'add' | 'del'
  readonly text: string
  readonly oldLine?: number
  readonly newLine?: number
}

export interface PatchFileDiff {
  readonly path: string
  readonly moveTo?: string
  readonly action: 'add' | 'update' | 'delete'
  readonly lines: readonly DiffLine[]
  readonly adds: number
  readonly dels: number
}

export interface OutlineSymbolEntry {
  readonly lineRange: string
  readonly kind: string
  readonly signature: string
  readonly exported: boolean
  readonly depth: number
}

export interface ParsedOutline {
  readonly header?: string
  readonly importsLine?: string
  readonly symbols: readonly OutlineSymbolEntry[]
}

export type FileTool =
  | { readonly kind: 'read'; readonly path: string; readonly offset: number }
  | { readonly kind: 'edit'; readonly path: string; readonly before: string; readonly after: string; readonly wholeFile: boolean }
  | { readonly kind: 'write'; readonly path: string; readonly content: string; readonly append: boolean }
  | { readonly kind: 'patch'; readonly path: string; readonly files: readonly PatchFileDiff[]; readonly dryRun: boolean }
  | { readonly kind: 'outline'; readonly path: string; readonly exportedOnly: boolean }

/** Optional group-level diff expansion broadcast from ToolTimelineGroup. */
export interface ToolTimelineDiffCommand {
  readonly mode: 'default' | 'expand' | 'collapse'
  readonly rev: number
}

export const ToolTimelineDiffContext = createContext<ToolTimelineDiffCommand>({ mode: 'default', rev: 0 })

const splitLines = (text: string): string[] => text ? text.replace(/\r\n/g, '\n').replace(/\n$/, '').split('\n') : []

function unwrapCodeFence(text: string): string {
  const trimmed = text.trim()
  const match = /^```[a-zA-Z0-9_-]*\n([\s\S]*?)\n```$/.exec(trimmed)
  return match?.[1] ?? text
}

function cleanPatchPath(raw: string): string {
  const withoutTimestamp = raw.split('\t')[0]?.trim() ?? ''
  const unquoted = withoutTimestamp.replace(/^["']|["']$/g, '').trim()
  return unquoted.replace(/^\.\//, '')
}

function parseHunkStarts(header: string): { oldLine?: number; newLine?: number } {
  const match = /^@@\s+-(\d+)(?:,\d+)?\s+\+(\d+)(?:,\d+)?\s+@@/.exec(header.trim())
  if (!match) return {}
  return { oldLine: Number(match[1]), newLine: Number(match[2]) }
}

/**
 * Parse V4A (`*** Begin Patch` / `*** Update File:`) or standard unified diff
 * (`--- a/...` / `+++ b/...`) text into per-file structured diff hunks.
 */
export function parsePatchDiffs(rawPatch: string, fallbackPath = ''): PatchFileDiff[] {
  const normalized = unwrapCodeFence(rawPatch.replace(/\r\n/g, '\n'))
  if (!normalized.trim()) return []
  const lines = normalized.split('\n')
  if (normalized.endsWith('\n') && lines.length > 0 && lines[lines.length - 1] === '') lines.pop()

  interface MutablePatchFile {
    path: string
    moveTo: string | undefined
    action: 'add' | 'update' | 'delete'
    lines: DiffLine[]
    adds: number
    dels: number
    oldCursor: number | undefined
    newCursor: number | undefined
  }

  const files: PatchFileDiff[] = []
  const state: { current: MutablePatchFile | null } = { current: null }

  const flush = (): void => {
    const cur = state.current
    if (!cur) return
    if (cur.path || cur.lines.length > 0) {
      files.push({
        path: cur.path || fallbackPath || 'patch',
        ...(cur.moveTo ? { moveTo: cur.moveTo } : {}),
        action: cur.action,
        lines: cur.lines,
        adds: cur.adds,
        dels: cur.dels,
      })
    }
    state.current = null
  }

  const startFile = (path: string, action: 'add' | 'update' | 'delete'): MutablePatchFile => {
    flush()
    const next: MutablePatchFile = {
      path: cleanPatchPath(path),
      moveTo: undefined,
      action,
      lines: [],
      adds: 0,
      dels: 0,
      oldCursor: undefined,
      newCursor: undefined,
    }
    state.current = next
    return next
  }

  const pushDiffLine = (kind: DiffLine['kind'], text: string): void => {
    const target = state.current ?? startFile(fallbackPath || 'patch', 'update')
    const oldLine = kind !== 'add' ? target.oldCursor : undefined
    const newLine = kind !== 'del' ? target.newCursor : undefined
    if (kind !== 'add' && target.oldCursor !== undefined) target.oldCursor += 1
    if (kind !== 'del' && target.newCursor !== undefined) target.newCursor += 1
    target.lines.push({
      kind,
      text,
      ...(oldLine !== undefined ? { oldLine } : {}),
      ...(newLine !== undefined ? { newLine } : {}),
    })
    if (kind === 'add') target.adds += 1
    if (kind === 'del') target.dels += 1
  }

  let pendingOldPath = ''
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i]!
    const trimmed = line.trim()

    if (trimmed === '*** Begin Patch' || trimmed === '*** End Patch' || trimmed === '*** End of File') {
      continue
    }
    if (line.startsWith('*** Update File:')) {
      startFile(line.slice('*** Update File:'.length), 'update')
      continue
    }
    if (line.startsWith('*** Add File:')) {
      const added = startFile(line.slice('*** Add File:'.length), 'add')
      added.newCursor = 1
      continue
    }
    if (line.startsWith('*** Delete File:')) {
      const removed = startFile(line.slice('*** Delete File:'.length), 'delete')
      removed.oldCursor = 1
      continue
    }
    if (line.startsWith('*** Move to:') && state.current) {
      state.current.moveTo = cleanPatchPath(line.slice('*** Move to:'.length))
      continue
    }
    if (line.startsWith('diff --git ')) {
      flush()
      pendingOldPath = ''
      continue
    }
    if (/^(?:index |new file mode |deleted file mode |old mode |new mode |similarity index |rename from |rename to )/.test(line)) {
      if (line.startsWith('rename to ') && state.current) {
        state.current.moveTo = cleanPatchPath(line.slice('rename to '.length))
      }
      continue
    }
    if (line.startsWith('--- ')) {
      const raw = cleanPatchPath(line.slice(4))
      pendingOldPath = raw === '/dev/null' ? '/dev/null' : raw.startsWith('a/') ? raw.slice(2) : raw
      continue
    }
    if (line.startsWith('+++ ')) {
      const raw = cleanPatchPath(line.slice(4))
      const newPath = raw === '/dev/null' ? '/dev/null' : raw.startsWith('b/') ? raw.slice(2) : raw
      if (pendingOldPath === '/dev/null' && newPath !== '/dev/null') {
        startFile(newPath, 'add')
      } else if (newPath === '/dev/null' && pendingOldPath && pendingOldPath !== '/dev/null') {
        startFile(pendingOldPath, 'delete')
      } else {
        startFile(newPath !== '/dev/null' ? newPath : pendingOldPath, 'update')
      }
      pendingOldPath = ''
      continue
    }
    if (line.startsWith('@@')) {
      const target = state.current ?? startFile(fallbackPath || 'patch', 'update')
      const starts = parseHunkStarts(line)
      target.oldCursor = starts.oldLine
      target.newCursor = starts.newLine
      continue
    }
    if (line.startsWith('\\ No newline at end of file')) continue

    if (line.startsWith('+')) {
      pushDiffLine('add', line.slice(1))
    } else if (line.startsWith('-')) {
      pushDiffLine('del', line.slice(1))
    } else if (line.startsWith(' ')) {
      pushDiffLine('ctx', line.slice(1))
    } else if (state.current && state.current.action === 'add' && line.length > 0) {
      pushDiffLine('add', line)
    }
  }

  flush()
  return files
}

/** Parse FileOutlineTool textual (or JSON) output into structured symbol entries. */
export function parseOutlineSymbols(output: string): ParsedOutline {
  const trimmed = output.trim()
  if (!trimmed) return { symbols: [] }

  // Support structured JSON payloads if returned by an outline tool variant.
  if (trimmed.startsWith('{') || trimmed.startsWith('[')) {
    try {
      const parsed: unknown = JSON.parse(trimmed)
      const list = Array.isArray(parsed)
        ? parsed
        : parsed && typeof parsed === 'object' && Array.isArray((parsed as { symbols?: unknown }).symbols)
          ? (parsed as { symbols: unknown[] }).symbols
          : null
      if (list) {
        const symbols: OutlineSymbolEntry[] = []
        const walk = (entries: readonly unknown[], depth: number): void => {
          for (const entry of entries) {
            if (!entry || typeof entry !== 'object') continue
            const rec = entry as Record<string, unknown>
            const kind = typeof rec.kind === 'string' ? rec.kind : 'symbol'
            const signature = typeof rec.signature === 'string'
              ? rec.signature
              : typeof rec.name === 'string'
                ? rec.name
                : ''
            if (!signature) continue
            const start = typeof rec.startLine === 'number' ? rec.startLine : typeof rec.line === 'number' ? rec.line : undefined
            const end = typeof rec.endLine === 'number' ? rec.endLine : start
            const lineRange = start !== undefined ? (end !== undefined && end !== start ? `${start}–${end}` : String(start)) : ''
            symbols.push({ lineRange, kind, signature, exported: rec.exported === true, depth })
            if (Array.isArray(rec.children)) walk(rec.children, depth + 1)
          }
        }
        walk(list, 0)
        return { symbols }
      }
    } catch {
      // Fall through to text parser.
    }
  }

  let header: string | undefined
  let importsLine: string | undefined
  const symbols: OutlineSymbolEntry[] = []
  const symbolRe = /^(\s*)\[(\d+(?:-\d+)?)\]\s+(export\s+)?([a-zA-Z_-]+)\s+(.+)$/

  for (const rawLine of trimmed.replace(/\r\n/g, '\n').split('\n')) {
    if (!header && rawLine.startsWith('# ')) {
      header = rawLine.slice(2).trim()
      continue
    }
    if (!importsLine && rawLine.startsWith('imports [')) {
      importsLine = rawLine.trim()
      continue
    }
    const match = symbolRe.exec(rawLine)
    if (match) {
      const indent = match[1]?.length ?? 0
      const lineRange = match[2]!.replace('-', '–')
      const exported = Boolean(match[3])
      const kind = match[4]!
      const signature = match[5]!.trim()
      symbols.push({ lineRange, kind, signature, exported, depth: Math.floor(indent / 2) })
    }
  }

  return {
    ...(header ? { header } : {}),
    ...(importsLine ? { importsLine } : {}),
    symbols,
  }
}

function normalizeFileToolName(rawName: string | undefined): string {
  const safe = typeof rawName === 'string' ? rawName : ''
  return (safe.split(/[.:/]/).pop() ?? safe).trim().toLowerCase().replace(/[\s_-]+/g, '')
}

export interface StructuredToolDiff {
  readonly adds?: number
  readonly dels?: number
  readonly path?: string
  readonly old_text?: string
  readonly new_text?: string
  readonly oldText?: string
  readonly newText?: string
}

function firstStringValue(...values: readonly unknown[]): string | undefined {
  for (const value of values) {
    if (typeof value === 'string') return value
  }
  return undefined
}

/** The file tool a call is, from its name, arguments, and optional structured diffs; null for any other tool. */
export function fileToolOf(
  item: Pick<ToolItem, 'name' | 'input'> & {
    readonly path?: string
    readonly diff?: StructuredToolDiff
    readonly diffs?: readonly StructuredToolDiff[]
  },
): FileTool | null {
  const name = normalizeFileToolName(item.name)
  const args = parseArgs(item.input ?? '')
  const diffObj = item.diff
  const path =
    (typeof args.file_path === 'string' && args.file_path.trim())
    || (typeof args.path === 'string' && args.path.trim())
    || (typeof args.Path === 'string' && args.Path.trim())
    || (typeof diffObj?.path === 'string' && diffObj.path.trim())
    || (typeof item.diffs?.[0]?.path === 'string' && item.diffs[0].path.trim())
    || (typeof item.path === 'string' && item.path.trim())
    || ''

  if (item.diffs && item.diffs.length > 1) {
    const files: PatchFileDiff[] = []
    for (const entry of item.diffs) {
      const entryPath = (typeof entry.path === 'string' && entry.path.trim()) || path
      const rawBefore = firstStringValue(entry.old_text, entry.oldText)
      const rawAfter = firstStringValue(entry.new_text, entry.newText)
      if (!entryPath && rawBefore === undefined && rawAfter === undefined) continue
      let before = rawBefore ?? ''
      let after = rawAfter ?? ''
      if (before && after) {
        const span = trimmedChangedLines(before, after, 3)
        before = span.before.join('\n')
        after = span.after.join('\n')
      }
      const lines = lineDiff(before, after)
      const adds = lines.filter(line => line.kind === 'add').length
      const dels = lines.filter(line => line.kind === 'del').length
      const action: PatchFileDiff['action'] = !rawBefore && rawAfter !== undefined
        ? 'add'
        : rawBefore !== undefined && rawAfter === ''
          ? 'delete'
          : 'update'
      files.push({
        path: entryPath || 'file',
        action,
        lines,
        adds,
        dels,
      })
    }
    if (files.length > 1) {
      return {
        kind: 'patch',
        path: files[0]!.path,
        files,
        dryRun: false,
      }
    }
  }

  if (name === 'applypatchtool' || name === 'applypatch') {
    const patchText = typeof args.patch === 'string' ? args.patch : typeof args.diff === 'string' ? args.diff : ''
    if (!patchText.trim()) return null
    const files = parsePatchDiffs(patchText, path)
    if (files.length === 0) return null
    return {
      kind: 'patch',
      path: files[0]!.path,
      files,
      dryRun: args.dry_run === true,
    }
  }

  if (name === 'fileoutlinetool' || name === 'fileoutline') {
    if (!path) return null
    return {
      kind: 'outline',
      path,
      exportedOnly: args.exported_only === true,
    }
  }

  const diffBefore = firstStringValue(diffObj?.old_text, diffObj?.oldText)
  const diffAfter = firstStringValue(diffObj?.new_text, diffObj?.newText)
  const hasStructuredDiffText = diffBefore !== undefined || diffAfter !== undefined
  const isKnownFileTool = ['readfile', 'fileedittool', 'editfile', 'writefile', 'appendfile'].includes(name)
  if (!isKnownFileTool && !hasStructuredDiffText) return null
  if (!path) return null

  if (name === 'readfile') {
    const offset =
      typeof args.offset === 'number' && args.offset > 0
        ? Math.floor(args.offset)
        : typeof args.StartLine === 'number' && args.StartLine > 1
          ? Math.floor(args.StartLine) - 1
          : 0
    return { kind: 'read', path, offset }
  }
  if (name === 'fileedittool' || name === 'editfile' || (!isKnownFileTool && hasStructuredDiffText)) {
    const argBefore = firstStringValue(args.old_string, args.Target, args.old_text, args.oldText)
    const argAfter = firstStringValue(args.new_string, args.Replacement, args.new_text, args.newText)
    const hasArgEdit = argBefore !== undefined || argAfter !== undefined
    let before = hasArgEdit ? (argBefore ?? '') : (diffBefore ?? '')
    let after = hasArgEdit ? (argAfter ?? '') : (diffAfter ?? '')
    if (!hasArgEdit && diffBefore && diffAfter) {
      const span = trimmedChangedLines(diffBefore, diffAfter, 3)
      before = span.before.join('\n')
      after = span.after.join('\n')
    }
    return { kind: 'edit', path, before, after, wholeFile: args.edit_mode === 'whole_file' }
  }
  const content = firstStringValue(args.content, args.Content, diffAfter) ?? ''
  return { kind: 'write', path, content, append: name === 'appendfile' }
}

/**
 * Compute accurate `+adds −dels` totals for any diff-bearing file tool
 * (`FileEditTool`, `edit_file`, `WriteFile`, `write_file`, `AppendFile`,
 * `ApplyPatchTool`). Returns null for non-mutating or non-file tools.
 */
export function fileToolStats(
  item: Pick<ToolItem, 'name' | 'input'> & {
    readonly path?: string
    readonly diff?: StructuredToolDiff
    readonly diffs?: readonly StructuredToolDiff[]
  },
): { readonly adds: number; readonly dels: number } | null {
  const tool = fileToolOf(item)
  if (!tool) {
    return item.diff && typeof item.diff.adds === 'number' && typeof item.diff.dels === 'number'
      ? { adds: item.diff.adds, dels: item.diff.dels }
      : null
  }
  if (tool.kind === 'edit') {
    const lines = lineDiff(tool.before, tool.after)
    return {
      adds: lines.filter(line => line.kind === 'add').length,
      dels: lines.filter(line => line.kind === 'del').length,
    }
  }
  if (tool.kind === 'write') {
    return { adds: splitLines(tool.content).length, dels: 0 }
  }
  if (tool.kind === 'patch') {
    return {
      adds: tool.files.reduce((sum, file) => sum + file.adds, 0),
      dels: tool.files.reduce((sum, file) => sum + file.dels, 0),
    }
  }
  return null
}

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

function splitDisplayPath(rawPath: string): { readonly dir: string; readonly base: string } {
  const normalized = rawPath.replace(/\\/g, '/')
  const lastSlash = normalized.lastIndexOf('/')
  if (lastSlash < 0) return { dir: '', base: rawPath }
  return {
    dir: normalized.slice(0, lastSlash + 1),
    base: normalized.slice(lastSlash + 1) || rawPath,
  }
}

function formatDiffForCopy(lines: readonly DiffLine[]): string {
  return lines.map(line => `${line.kind === 'add' ? '+' : line.kind === 'del' ? '-' : ' '}${line.text}`).join('\n')
}

const PREVIEW_LINES = 14
const VERBS = { read: 'Read', edit: 'Edit', write: 'Write', patch: 'Patch', outline: 'Outline' } as const

function iconForFileTool(kind: FileTool['kind']): IconName {
  switch (kind) {
    case 'read': return 'file'
    case 'edit': return 'changes'
    case 'write': return 'note'
    case 'patch': return 'changes'
    case 'outline': return 'listBullets'
  }
}

function symbolKindTag(kind: string): string {
  switch (kind.toLowerCase()) {
    case 'function': return 'fn'
    case 'method': return 'fn'
    case 'interface': return 'iface'
    case 'class': return 'class'
    case 'type': return 'type'
    case 'enum': return 'enum'
    case 'struct': return 'struct'
    case 'trait': return 'trait'
    case 'impl': return 'impl'
    case 'constant': return 'const'
    case 'variable': return 'var'
    case 'property': return 'prop'
    case 'module': return 'mod'
    default: return kind.slice(0, 6)
  }
}

function DiffBlock({ lines, numberFrom, forceAll = false }: {
  readonly lines: readonly DiffLine[]
  readonly numberFrom?: number
  readonly forceAll?: boolean
}): ReactElement {
  const [all, setAll] = useState(forceAll)
  useEffect(() => {
    if (forceAll) setAll(true)
  }, [forceAll])
  const shown = all ? lines : lines.slice(0, PREVIEW_LINES)
  return <div className="filetool__code">
    <pre tabIndex={0} aria-label="Changes">{shown.map((line, index) => {
      const lineNum = numberFrom !== undefined
        ? numberFrom + index
        : line.kind === 'add'
          ? line.newLine
          : line.kind === 'del'
            ? line.oldLine
            : (line.newLine ?? line.oldLine)
      const sign = numberFrom !== undefined ? '' : line.kind === 'add' ? '+' : line.kind === 'del' ? '−' : ' '
      return <span key={index} className={`filetool__line filetool__line--${line.kind}`}>
        <span className="filetool__gutter" aria-hidden="true">
          {lineNum !== undefined && <span className="filetool__lnum">{lineNum}</span>}
          {sign && <span className="filetool__sign">{sign}</span>}
        </span>
        <code>{line.text || ' '}</code>
      </span>
    })}</pre>
    {lines.length > PREVIEW_LINES && (
      <button type="button" className="filetool__more" onClick={() => setAll(value => !value)}>
        {all ? 'Show less' : `Show ${lines.length - PREVIEW_LINES} more line${lines.length - PREVIEW_LINES === 1 ? '' : 's'}`}
      </button>
    )}
  </div>
}

function FilePathButton({ path, onOpen }: { readonly path: string; readonly onOpen: (path: string) => void }): ReactElement {
  const { dir, base } = splitDisplayPath(path)
  return <button
    type="button"
    className="filetool__path"
    data-path={path}
    title={`Open ${path} in Files`}
    onClick={() => onOpen(path)}
  >
    {dir && <span className="filetool__dir">{dir}</span>}
    <span className="filetool__base">{base}</span>
  </button>
}

function OutlineList({ parsed }: { readonly parsed: ParsedOutline }): ReactElement {
  return <div className="filetool__outline" role="region" aria-label="File symbols">
    {parsed.importsLine && <div className="filetool__outline-imports">{parsed.importsLine}</div>}
    {parsed.symbols.length === 0 ? (
      <div className="filetool__outline-empty">No matching symbols</div>
    ) : (
      <ul className="filetool__symbols">
        {parsed.symbols.map((sym, index) => (
          <li key={`${sym.lineRange}-${sym.signature}-${index}`} className="filetool__symbol" style={{ paddingLeft: `${8 + sym.depth * 14}px` }}>
            <span className="filetool__symbol-kind" data-kind={sym.kind.toLowerCase()}>{symbolKindTag(sym.kind)}</span>
            {sym.exported && <span className="filetool__symbol-exp">export</span>}
            <code className="filetool__symbol-sig">{sym.signature}</code>
            {sym.lineRange && <span className="filetool__symbol-range">L{sym.lineRange}</span>}
          </li>
        ))}
      </ul>
    )}
  </div>
}

export const FileToolRow = memo(function FileToolRow({ item, tool, failed, defaultExpanded }: {
  item: ToolItem
  tool: FileTool
  failed: boolean
  defaultExpanded?: boolean
}): ReactElement {
  const open = useDesktopNavigation()
  const diffCmd = useContext(ToolTimelineDiffContext)
  const isDiffKind = tool.kind === 'edit' || tool.kind === 'write' || tool.kind === 'patch'
  const [expanded, setExpanded] = useState(defaultExpanded ?? isDiffKind)
  const [forceAllLines, setForceAllLines] = useState(false)

  useEffect(() => {
    if (diffCmd.rev === 0 || !isDiffKind) return
    if (diffCmd.mode === 'expand') {
      setExpanded(true)
      setForceAllLines(true)
    } else if (diffCmd.mode === 'collapse') {
      setExpanded(false)
      setForceAllLines(false)
    }
  }, [diffCmd.mode, diffCmd.rev, isDiffKind])

  const editOrWriteOrReadLines = useMemo<readonly DiffLine[]>(() => {
    if (tool.kind === 'edit') return lineDiff(tool.before, tool.after)
    if (tool.kind === 'write') return splitLines(tool.content).map((text, index) => ({ kind: 'add' as const, text, newLine: index + 1 }))
    if (tool.kind === 'read') return splitLines(readBody(item.output)).map(text => ({ kind: 'ctx' as const, text }))
    return []
  }, [tool, item.output])

  const parsedOutline = useMemo<ParsedOutline | null>(
    () => tool.kind === 'outline' ? parseOutlineSymbols(item.output) : null,
    [tool.kind, item.output],
  )

  const { adds, dels } = useMemo(() => {
    if (tool.kind === 'patch') {
      return {
        adds: tool.files.reduce((sum, file) => sum + file.adds, 0),
        dels: tool.files.reduce((sum, file) => sum + file.dels, 0),
      }
    }
    return {
      adds: editOrWriteOrReadLines.filter(line => line.kind === 'add').length,
      dels: editOrWriteOrReadLines.filter(line => line.kind === 'del').length,
    }
  }, [tool, editOrWriteOrReadLines])

  const verb = tool.kind === 'write' && tool.append
    ? 'Append'
    : tool.kind === 'edit' && tool.wholeFile
      ? 'Rewrite'
      : tool.kind === 'patch' && tool.dryRun
        ? 'Check patch'
        : VERBS[tool.kind]

  const error = failed ? (item.error || item.output).split('\n')[0]!.replace(/^Tool execution failed: /, '').slice(0, 240) : ''
  const hasDiffContent = tool.kind === 'patch'
    ? tool.files.some(file => file.lines.length > 0)
    : isDiffKind && editOrWriteOrReadLines.length > 0

  const showBody = !failed && expanded
  const copyPayload = useMemo(() => {
    if (tool.kind === 'read') return readBody(item.output)
    if (tool.kind === 'write') return tool.content
    if (tool.kind === 'edit') return formatDiffForCopy(editOrWriteOrReadLines)
    if (tool.kind === 'patch') {
      return tool.files
        .map(file => `--- ${file.path}\n${formatDiffForCopy(file.lines)}`)
        .join('\n\n')
    }
    return item.output
  }, [tool, item.output, editOrWriteOrReadLines])

  const extraPatchFiles = tool.kind === 'patch' && tool.files.length > 1 ? tool.files.length - 1 : 0
  const symbolCount = parsedOutline?.symbols.length ?? 0

  return <div className="filetool" data-kind={tool.kind} data-state={failed ? 'failed' : item.state} data-expanded={showBody || undefined}>
    <div className="filetool__head">
      <span className="filetool__icon" data-kind={tool.kind} aria-hidden="true">
        <Icon name={item.state === 'working' ? 'spinner' : iconForFileTool(tool.kind)} size={13} />
      </span>
      <span className="filetool__verb">{verb}</span>
      <FilePathButton path={tool.path} onOpen={target => open('files', target)} />
      {extraPatchFiles > 0 && (
        <span className="filetool__badge">+{extraPatchFiles} file{extraPatchFiles === 1 ? '' : 's'}</span>
      )}
      {tool.kind === 'read' && !failed && item.state !== 'working' && editOrWriteOrReadLines.length > 0 && (
        <button type="button" className="filetool__toggle" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>
          {`lines ${tool.offset + 1}–${tool.offset + editOrWriteOrReadLines.length}`}<Icon name="chevron" size={11} />
        </button>
      )}
      {tool.kind === 'outline' && !failed && item.state !== 'working' && (
        <button type="button" className="filetool__toggle" aria-expanded={expanded} onClick={() => setExpanded(value => !value)}>
          {symbolCount > 0 ? `${symbolCount} symbol${symbolCount === 1 ? '' : 's'}` : 'Outline'}<Icon name="chevron" size={11} />
        </button>
      )}
      {isDiffKind && !failed && item.state !== 'working' && (
        <span className="filetool__stats">
          <span className="add">+{adds}</span>
          {(tool.kind === 'edit' || (tool.kind === 'patch' && dels > 0)) && <span className="del">−{dels}</span>}
        </span>
      )}
      {hasDiffContent && !failed && (
        <button
          type="button"
          className="filetool__toggle filetool__toggle--diff"
          aria-expanded={expanded}
          aria-label={expanded ? 'Hide diff' : 'Show diff'}
          onClick={() => setExpanded(value => !value)}
        >
          <Icon name="chevron" size={11} />
        </button>
      )}
      <span className="filetool__state">
        {failed ? 'Failed' : item.state === 'working' ? 'Running' : item.dur && item.dur !== '0.0s' ? item.dur : ''}
      </span>
    </div>
    {error && <p className="filetool__error" title={item.error || item.output}>{error}</p>}
    {showBody && tool.kind === 'patch' && tool.files.length > 0 && (
      <div className="filetool__patch-files">
        {tool.files.map((file, idx) => (
          <div key={`${file.path}-${idx}`} className="filetool__patch-file">
            {tool.files.length > 1 && (
              <div className="filetool__patch-head">
                <span className="filetool__patch-op" data-op={file.action}>{file.action}</span>
                <FilePathButton path={file.path} onOpen={target => open('files', target)} />
                {file.moveTo && <span className="filetool__patch-move">→ {file.moveTo}</span>}
                <span className="filetool__stats">
                  <span className="add">+{file.adds}</span>
                  {file.dels > 0 && <span className="del">−{file.dels}</span>}
                </span>
              </div>
            )}
            {file.lines.length > 0 && <DiffBlock lines={file.lines} forceAll={forceAllLines} />}
          </div>
        ))}
        {copyPayload && <div className="filetool__actions"><CopyButton icon text={copyPayload} label="Copy patch" /></div>}
      </div>
    )}
    {showBody && tool.kind === 'outline' && parsedOutline && (
      <>
        <OutlineList parsed={parsedOutline} />
        {copyPayload && <div className="filetool__actions"><CopyButton icon text={copyPayload} label="Copy outline" /></div>}
      </>
    )}
    {showBody && (tool.kind === 'edit' || tool.kind === 'write' || tool.kind === 'read') && editOrWriteOrReadLines.length > 0 && (
      <>
        <DiffBlock
          lines={editOrWriteOrReadLines}
          {...(tool.kind === 'read' ? { numberFrom: tool.offset + 1 } : {})}
          forceAll={forceAllLines}
        />
        {copyPayload && (
          <div className="filetool__actions">
            <CopyButton icon text={copyPayload} label={tool.kind === 'read' ? 'Copy text' : tool.kind === 'write' ? 'Copy content' : 'Copy diff'} />
          </div>
        )}
      </>
    )}
    {showBody && <ToolImages item={item} />}
  </div>
})
