// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Pure helpers behind the editor's completion popup (positions, triggers, edits). */

export interface CompletionItemView {
  readonly label: string
  readonly kind?: number
  readonly detail?: string
  readonly documentation?: string
  readonly insertText: string
  /** Replaced span on the cursor's line, as character offsets within that line. */
  readonly replace?: { readonly start: number; readonly end: number }
}
export type Completer = (text: string, line: number, character: number) => Promise<readonly CompletionItemView[]>

const record = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)

/** Validate the daemon's reply; anything malformed is dropped, never shown. */
export function completionItemsOf(value: unknown): CompletionItemView[] {
  if (!Array.isArray(value)) return []
  const items: CompletionItemView[] = []
  for (const item of value.slice(0, 200)) {
    if (!record(item) || typeof item.label !== 'string' || !item.label || typeof item.insertText !== 'string') continue
    const replace = record(item.replace) && Number.isSafeInteger(item.replace.start) && Number.isSafeInteger(item.replace.end)
      && Number(item.replace.start) >= 0 && Number(item.replace.end) >= Number(item.replace.start)
      ? { start: Number(item.replace.start), end: Number(item.replace.end) } : undefined
    items.push({
      label: item.label, insertText: item.insertText,
      ...(typeof item.kind === 'number' ? { kind: item.kind } : {}),
      ...(typeof item.detail === 'string' ? { detail: item.detail } : {}),
      ...(typeof item.documentation === 'string' ? { documentation: item.documentation } : {}),
      ...(replace ? { replace } : {}),
    })
  }
  return items
}

export function cursorPosition(text: string, offset: number): { line: number; character: number; lineStart: number } {
  const before = text.slice(0, offset)
  const lineStart = before.lastIndexOf('\n') + 1
  let line = 0
  for (let index = before.indexOf('\n'); index !== -1; index = before.indexOf('\n', index + 1)) line++
  return { line, character: offset - lineStart, lineStart }
}

/** The identifier being typed right before the cursor. */
export function completionPrefix(text: string, offset: number): string {
  return /[A-Za-z_][A-Za-z0-9_]*$/.exec(text.slice(0, offset))?.[0] ?? ''
}

/** Open the popup after a name character, a dot, or a space in an import line. */
export function shouldComplete(text: string, offset: number): boolean {
  const { lineStart } = cursorPosition(text, offset)
  const line = text.slice(lineStart, offset)
  if (/^\s*#/.test(line) || /(^|[^\\])(['"])(?:(?!\2).)*$/.test(line)) return false
  const last = line.at(-1)
  if (last === undefined) return false
  // A name (not a number like `1.`) or a closing bracket right before the dot.
  if (last === '.') return /(?<![0-9A-Za-z_])[A-Za-z_][A-Za-z0-9_]*\.$|[)\]]\.$/.test(line)
  if (last === ' ') return /^\s*(from\s+[\w.]+\s+import|import|from)\s+$/.test(line)
  // Any name character, digits included once a name has started (`A1`, `x2`).
  return /[A-Za-z0-9_]/.test(last) && completionPrefix(text, offset).length >= 1
}

/** Leading whitespace of the caret's line, plus one indent unit after a line ending in `:` (Python blocks). */
export function newlineIndent(text: string, offset: number, unit: string): string {
  const { lineStart } = cursorPosition(text, offset)
  const before = text.slice(lineStart, offset)
  const indent = /^[ \t]*/.exec(before)?.[0] ?? ''
  return /:\s*(#.*)?$/.test(before) ? indent + unit : indent
}

/** Items matching the typed prefix: exact-case starts first, then case-insensitive, then contains. */
export function filterCompletions(items: readonly CompletionItemView[], prefix: string): CompletionItemView[] {
  if (!prefix) return [...items]
  const lower = prefix.toLowerCase()
  const rank = (label: string): number => label.startsWith(prefix) ? 0 : label.toLowerCase().startsWith(lower) ? 1 : label.toLowerCase().includes(lower) ? 2 : 3
  return items.map((item, index) => ({ item, index, rank: rank(item.label) }))
    .filter(entry => entry.rank < 3)
    .sort((a, b) => a.rank - b.rank || a.index - b.index)
    .map(entry => entry.item)
}

/** The absolute span to replace and the text to put there. */
export function completionEdit(text: string, offset: number, item: CompletionItemView): { start: number; end: number; insert: string } {
  const { lineStart } = cursorPosition(text, offset)
  const lineEnd = text.indexOf('\n', offset) === -1 ? text.length : text.indexOf('\n', offset)
  if (item.replace && lineStart + item.replace.end <= lineEnd && item.replace.start <= offset - lineStart) {
    return { start: lineStart + item.replace.start, end: Math.max(offset, lineStart + item.replace.end), insert: item.insertText }
  }
  return { start: offset - completionPrefix(text, offset).length, end: offset, insert: item.insertText }
}

/** Short tag per LSP CompletionItemKind. */
export function completionKindTag(kind: number | undefined): string {
  switch (kind) {
    case 2: return 'm'
    case 3: return 'ƒ'
    case 4: return 'c'
    case 5: case 10: return 'p'
    case 6: return 'v'
    case 7: case 22: return 'C'
    case 9: return 'M'
    case 13: case 20: return 'e'
    case 14: return 'k'
    case 21: return 'K'
    default: return '·'
  }
}
