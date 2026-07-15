// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type PythonTokenKind = 'comment' | 'name' | 'string' | 'symbol'
export type PythonScopeKind = 'class' | 'function' | 'generic' | 'module'

export interface PythonToken {
  end: number
  kind: PythonTokenKind
  plainString: boolean
  start: number
  text: string
}

export interface PythonStatement {
  end: number
  indent: number
  start: number
  tokens: readonly PythonToken[]
}

export interface PythonScope {
  bodyIndent: number
  id: number
  kind: PythonScopeKind
  statements: PythonStatement[]
}

export interface PythonStatementContext {
  firstInScope: boolean
  scope: PythonScope
  statement: PythonStatement
}

export interface PythonSourceAnalysis {
  comments: readonly PythonToken[]
  contexts: readonly PythonStatementContext[]
  scopes: readonly PythonScope[]
  valid: boolean
}

export interface SourceEdit {
  end: number
  replacement: string
  start: number
}

interface PendingScope {
  headerIndent: number
  kind: PythonScopeKind
}

interface ScannedSource {
  comments: PythonToken[]
  statements: PythonStatement[]
  valid: boolean
}

const STRING_PREFIX = /^[bfru]+$/iu

/** Analyze Python source without invoking an external Python interpreter. */
export function analyzePythonSource(source: string): PythonSourceAnalysis {
  const scanned = scanPythonSource(source)
  const moduleScope: PythonScope = { bodyIndent: 0, id: 0, kind: 'module', statements: [] }
  const scopes: PythonScope[] = [moduleScope]
  const activeScopes: PythonScope[] = [moduleScope]
  const contexts: PythonStatementContext[] = []
  let pending: PendingScope | undefined

  for (const statement of scanned.statements) {
    if (pending !== undefined) {
      if (statement.indent > pending.headerIndent) {
        const scope: PythonScope = {
          bodyIndent: statement.indent,
          id: scopes.length,
          kind: pending.kind,
          statements: [],
        }
        scopes.push(scope)
        activeScopes.push(scope)
      }
      pending = undefined
    }

    while (activeScopes.length > 1 && statement.indent < activeScopes.at(-1)!.bodyIndent) {
      activeScopes.pop()
    }

    const scope = activeScopes.at(-1)!
    if (statement.indent !== scope.bodyIndent) continue

    const firstInScope = scope.statements.length === 0
    scope.statements.push(statement)
    contexts.push({ firstInScope, scope, statement })

    const kind = suiteKind(statement)
    if (kind !== undefined) pending = { headerIndent: statement.indent, kind }
  }

  return { comments: scanned.comments, contexts, scopes, valid: scanned.valid }
}

/** Return plain-string expression tokens when a statement is only a string literal expression. */
export function barePlainStringTokens(statement: PythonStatement): readonly PythonToken[] | undefined {
  let depth = 0
  let state: 'afterStrings' | 'beforeStrings' | 'done' | 'strings' = 'beforeStrings'
  const strings: PythonToken[] = []

  for (const token of statement.tokens) {
    if (token.kind === 'string') {
      if (!token.plainString || state === 'afterStrings' || state === 'done') return undefined
      strings.push(token)
      state = 'strings'
      continue
    }
    if (token.kind !== 'symbol') return undefined
    if (token.text === '(') {
      if (state !== 'beforeStrings') return undefined
      depth += 1
      continue
    }
    if (token.text !== ')') return undefined
    if (state !== 'strings' && state !== 'afterStrings') return undefined
    if (depth === 0) return undefined
    depth -= 1
    state = depth === 0 ? 'done' : 'afterStrings'
  }

  if (strings.length === 0 || depth !== 0) return undefined
  if (state !== 'strings' && state !== 'done') return undefined
  return strings
}

/** Apply non-overlapping source edits while preserving all untouched text. */
export function applySourceEdits(source: string, edits: readonly SourceEdit[]): string {
  if (edits.length === 0) return source
  const ordered = [...edits].sort((left, right) => left.start - right.start || left.end - right.end)
  let cursor = 0
  let result = ''

  for (const edit of ordered) {
    if (edit.start < cursor || edit.end < edit.start || edit.end > source.length) {
      throw new RangeError('Source edits must be ordered, non-overlapping, and within the source text.')
    }
    result += source.slice(cursor, edit.start)
    result += edit.replacement
    cursor = edit.end
  }

  return result + source.slice(cursor)
}

/** Expand a range to whole physical lines and any blank lines that immediately follow it. */
export function lineRemovalRange(source: string, start: number, end: number): SourceEdit {
  const lineStart = source.lastIndexOf('\n', Math.max(0, start - 1)) + 1
  let cursor = endOfLine(source, end)

  while (cursor < source.length) {
    const next = endOfLine(source, cursor)
    const line = source.slice(cursor, next).replace(/[\r\n]+$/u, '')
    if (line.trim() !== '') break
    cursor = next
  }

  return { end: cursor, replacement: '', start: lineStart }
}

/** Normalize blank lines after removing Python comments and docstrings. */
export function normalizeStrippedSource(source: string, sourceEndedWithNewline: boolean): string {
  const cleaned: string[] = []

  for (const line of splitLinesKeepingEndings(source)) {
    const match = line.match(/(\r\n|\n|\r)$/u)
    const ending = match?.[0] ?? ''
    const body = ending === '' ? line : line.slice(0, -ending.length)
    const normalized = `${body.replace(/[\t ]+$/u, '')}${ending}`
    const blank = normalized.replace(/[\r\n]+$/u, '').trim() === ''
    if (blank && cleaned.length > 0 && cleaned.at(-1)!.replace(/[\r\n]+$/u, '').trim() === '') continue
    cleaned.push(normalized)
  }

  while (cleaned.length > 0 && cleaned.at(-1)!.replace(/[\r\n]+$/u, '').trim() === '') {
    cleaned.pop()
  }

  let result = cleaned.join('')
  if (sourceEndedWithNewline && !result.endsWith('\n')) result += '\n'
  return result
}

function scanPythonSource(source: string): ScannedSource {
  const comments: PythonToken[] = []
  const statements: PythonStatement[] = []
  let current: PythonToken[] = []
  let bracketDepth = 0
  let cursor = 0
  let valid = true

  const flush = (): void => {
    if (current.length === 0) return
    const first = current[0]
    const last = current.at(-1)
    if (first === undefined || last === undefined) return
    statements.push({
      end: last.end,
      indent: indentationAt(source, first.start),
      start: first.start,
      tokens: current,
    })
    current = []
  }

  while (cursor < source.length) {
    const character = source[cursor]!
    if (character === ' ' || character === '\t' || character === '\f') {
      cursor += 1
      continue
    }
    if (character === '\r' || character === '\n') {
      const newlineStart = cursor
      if (character === '\r' && source[cursor + 1] === '\n') cursor += 2
      else cursor += 1
      if (bracketDepth === 0 && !hasExplicitLineContinuation(source, newlineStart)) flush()
      continue
    }
    if (character === '#') {
      const start = cursor
      while (cursor < source.length && source[cursor] !== '\r' && source[cursor] !== '\n') cursor += 1
      comments.push({ end: cursor, kind: 'comment', plainString: false, start, text: source.slice(start, cursor) })
      continue
    }

    const string = stringAt(source, cursor)
    if (string !== undefined) {
      const scanned = scanString(source, string.quoteStart)
      if (!scanned.terminated) valid = false
      current.push({
        end: scanned.end,
        kind: 'string',
        plainString: !/[bf]/iu.test(string.prefix),
        start: cursor,
        text: source.slice(cursor, scanned.end),
      })
      cursor = scanned.end
      continue
    }

    if (isNameStart(character)) {
      const start = cursor
      cursor += 1
      while (cursor < source.length && isNamePart(source[cursor]!)) cursor += 1
      current.push({ end: cursor, kind: 'name', plainString: false, start, text: source.slice(start, cursor) })
      continue
    }

    if (character === '(' || character === '[' || character === '{') bracketDepth += 1
    if (character === ')' || character === ']' || character === '}') {
      bracketDepth -= 1
      if (bracketDepth < 0) {
        valid = false
        bracketDepth = 0
      }
    }
    current.push({ end: cursor + 1, kind: 'symbol', plainString: false, start: cursor, text: character })
    cursor += 1
  }

  flush()
  if (bracketDepth !== 0) valid = false
  return { comments, statements, valid }
}

function suiteKind(statement: PythonStatement): PythonScopeKind | undefined {
  const last = statement.tokens.at(-1)
  if (last?.kind !== 'symbol' || last.text !== ':') return undefined
  const first = statement.tokens[0]
  if (first?.kind !== 'name') return 'generic'
  if (first.text === 'class') return 'class'
  if (first.text === 'def') return 'function'
  const second = statement.tokens[1]
  if (first.text === 'async' && second?.kind === 'name' && second.text === 'def') return 'function'
  return 'generic'
}

function stringAt(source: string, start: number): { prefix: string; quoteStart: number } | undefined {
  const character = source[start]
  if (character === '"' || character === "'") return { prefix: '', quoteStart: start }
  if (character === undefined || !STRING_PREFIX.test(character)) return undefined
  if (start > 0 && isNamePart(source[start - 1]!)) return undefined

  let quoteStart = start
  while (quoteStart < source.length && quoteStart - start < 3 && STRING_PREFIX.test(source[quoteStart]!)) {
    quoteStart += 1
  }
  const quote = source[quoteStart]
  if (quote !== '"' && quote !== "'") return undefined
  return { prefix: source.slice(start, quoteStart), quoteStart }
}

function scanString(source: string, quoteStart: number): { end: number; terminated: boolean } {
  const quote = source[quoteStart]!
  const triple = source.slice(quoteStart, quoteStart + 3) === quote.repeat(3)
  const delimiterLength = triple ? 3 : 1
  let cursor = quoteStart + delimiterLength

  while (cursor < source.length) {
    const character = source[cursor]!
    if (character === '\\') {
      cursor += 2
      continue
    }
    if (!triple && (character === '\r' || character === '\n')) return { end: cursor, terminated: false }
    if (triple && source.slice(cursor, cursor + 3) === quote.repeat(3)) {
      return { end: cursor + 3, terminated: true }
    }
    if (!triple && character === quote) return { end: cursor + 1, terminated: true }
    cursor += 1
  }

  return { end: source.length, terminated: false }
}

function indentationAt(source: string, offset: number): number {
  const lineStart = source.lastIndexOf('\n', Math.max(0, offset - 1)) + 1
  let column = 0
  for (let cursor = lineStart; cursor < offset; cursor += 1) {
    const character = source[cursor]
    if (character === ' ') column += 1
    else if (character === '\t') column += 8 - (column % 8)
    else if (character !== '\f') break
  }
  return column
}

function endOfLine(source: string, offset: number): number {
  const newline = source.indexOf('\n', offset)
  return newline === -1 ? source.length : newline + 1
}

function hasExplicitLineContinuation(source: string, newlineStart: number): boolean {
  let cursor = newlineStart - 1
  if (source[cursor] === '\r') cursor -= 1
  let slashCount = 0
  while (cursor >= 0 && source[cursor] === '\\') {
    slashCount += 1
    cursor -= 1
  }
  return slashCount % 2 === 1
}

function isNamePart(character: string): boolean {
  return /[A-Za-z0-9_]/u.test(character)
}

function isNameStart(character: string): boolean {
  return /[A-Za-z_]/u.test(character)
}

function splitLinesKeepingEndings(source: string): string[] {
  const lines: string[] = []
  let start = 0
  for (let index = 0; index < source.length; index += 1) {
    const character = source[index]
    if (character !== '\n' && character !== '\r') continue
    if (character === '\r' && source[index + 1] === '\n') index += 1
    lines.push(source.slice(start, index + 1))
    start = index + 1
  }
  if (start < source.length) lines.push(source.slice(start))
  return lines
}
