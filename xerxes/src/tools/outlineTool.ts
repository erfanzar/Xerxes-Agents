// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readdir, stat } from 'node:fs/promises'
import { dirname, extname, join } from 'node:path'

import { ValidationError } from '../core/errors.js'
import type { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { GitignoreMatcher } from './gitignore.js'
import { optionalBoolean, optionalInteger, requireRange, requiredString } from './inputs.js'
import { BINARY_SNIFF_BYTES, looksBinary } from './officeText.js'
import { WorkspacePathError, type WorkspacePathResolver } from './pathSafety.js'

export type OutlineLanguage = 'typescript' | 'javascript' | 'python' | 'go' | 'rust' | 'unknown'

export type OutlineSymbolKind =
  | 'export'
  | 'class'
  | 'interface'
  | 'type'
  | 'enum'
  | 'struct'
  | 'trait'
  | 'impl'
  | 'module'
  | 'function'
  | 'method'
  | 'constant'
  | 'variable'
  | 'property'

export interface OutlineSymbol {
  readonly name: string
  readonly kind: OutlineSymbolKind
  readonly signature: string
  /** 1-based inclusive start line. */
  readonly startLine: number
  /** 1-based inclusive end line. */
  readonly endLine: number
  readonly exported: boolean
  readonly children?: readonly OutlineSymbol[]
}

export interface OutlineImportSummary {
  readonly count: number
  readonly startLine: number
  readonly endLine: number
  readonly modules: readonly string[]
}

export interface FileOutline {
  readonly filePath: string
  readonly language: OutlineLanguage
  readonly lineCount: number
  readonly imports?: OutlineImportSummary
  readonly symbols: readonly OutlineSymbol[]
}

export interface ExtractOutlineOptions {
  readonly includeImports?: boolean
  readonly exportedOnly?: boolean
  readonly maxDepth?: number
}

const DEFAULT_MAX_DEPTH = 2
const DEFAULT_MAX_FILES = 10
const MAX_BATCH_FILES = 25
const MAX_OUTLINE_FILE_BYTES = 16 * 1024 * 1024
const MAX_SIGNATURE_CHARS = 120

const READ_ONLY_OUTLINE_CAPABILITIES = Object.freeze({
  concurrencySafe: true,
  defer: false,
  destructive: false,
  openWorld: false,
  readOnly: true,
} as const)

const SUPPORTED_EXTENSIONS = new Set([
  '.ts',
  '.tsx',
  '.mts',
  '.cts',
  '.js',
  '.jsx',
  '.mjs',
  '.cjs',
  '.py',
  '.pyi',
  '.go',
  '.rs',
])

export const FILE_OUTLINE_TOOL_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'FileOutlineTool',
    description:
      'Extract a compact structural symbol outline with 1-based [startLine-endLine] spans from a workspace '
      + 'file, directory, or glob batch (up to max_files, default 10) without spawning external processes. '
      + 'Supports TypeScript/TSX/JS/JSX, Python, Go, and Rust (classes, structs, traits, impls, interfaces, '
      + 'methods, functions, type aliases, enums, constants, and imports/exports). Use this first on large '
      + 'files to locate exact symbol line spans before calling ReadFile(offset, limit).',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        file_path: {
          type: 'string',
          minLength: 1,
          description: 'Workspace-relative file path, directory path, or glob pattern (for example src/foo.ts or src/**/*.ts).',
        },
        include_imports: {
          type: 'boolean',
          default: true,
          description: 'Include summarized import/use spans at the top of each file.',
        },
        exported_only: {
          type: 'boolean',
          default: false,
          description: 'Show only exported/public declarations.',
        },
        max_depth: {
          type: 'integer',
          minimum: 1,
          maximum: 8,
          default: DEFAULT_MAX_DEPTH,
          description: 'Maximum symbol nesting depth (1 = top-level only, 2 = classes/impls and their members).',
        },
        max_files: {
          type: 'integer',
          minimum: 1,
          maximum: MAX_BATCH_FILES,
          default: DEFAULT_MAX_FILES,
          description: 'Maximum files outlined when file_path is a directory or glob.',
        },
      },
      required: ['file_path'],
    },
  },
}

export function registerFileOutlineTool(registry: ToolRegistry, paths: WorkspacePathResolver): void {
  registry.register(
    FILE_OUTLINE_TOOL_DEFINITION,
    inputs => outlineFile(inputs, paths),
    'default',
    READ_ONLY_OUTLINE_CAPABILITIES,
  )
}

export async function outlineFile(inputs: JsonObject, paths: WorkspacePathResolver): Promise<string> {
  const filePath = requiredString(inputs, 'file_path')
  const includeImports = optionalBoolean(inputs, 'include_imports', true)
  const exportedOnly = optionalBoolean(inputs, 'exported_only', false)
  const maxDepth = requireRange(optionalInteger(inputs, 'max_depth', DEFAULT_MAX_DEPTH), 'max_depth', 1, 8)
  const maxFiles = requireRange(optionalInteger(inputs, 'max_files', DEFAULT_MAX_FILES), 'max_files', 1, MAX_BATCH_FILES)

  const resolvedTargets = await collectOutlineTargets(filePath, maxFiles, paths)
  if (resolvedTargets.files.length === 0) {
    return `[FileOutlineTool] No supported source files matched "${filePath}".`
  }

  const sections: string[] = []
  for (const target of resolvedTargets.files) {
    const info = await stat(target.absolutePath)
    if (info.size > MAX_OUTLINE_FILE_BYTES) {
      sections.push(`# ${target.displayPath} (skipped: ${info.size} bytes exceeds ${MAX_OUTLINE_FILE_BYTES}-byte limit)`)
      continue
    }
    const sniff = new Uint8Array(await Bun.file(target.absolutePath).slice(0, BINARY_SNIFF_BYTES).arrayBuffer())
    if (looksBinary(sniff)) {
      sections.push(`# ${target.displayPath} (skipped: binary file)`)
      continue
    }
    const content = await Bun.file(target.absolutePath).text()
    const outline = extractFileOutline(target.displayPath, content, {
      includeImports,
      exportedOnly,
      maxDepth,
    })
    sections.push(formatFileOutline(outline, { includeImports }))
  }

  if (resolvedTargets.truncated) {
    sections.push(`... (showing first ${maxFiles} files; narrow file_path or raise max_files up to ${MAX_BATCH_FILES})`)
  }
  return sections.join('\n\n')
}

export function detectOutlineLanguage(filePath: string): OutlineLanguage {
  const ext = extname(filePath.toLowerCase())
  if (ext === '.ts' || ext === '.tsx' || ext === '.mts' || ext === '.cts') return 'typescript'
  if (ext === '.js' || ext === '.jsx' || ext === '.mjs' || ext === '.cjs') return 'javascript'
  if (ext === '.py' || ext === '.pyi') return 'python'
  if (ext === '.go') return 'go'
  if (ext === '.rs') return 'rust'
  return 'unknown'
}

/**
 * Pure, zero-subprocess structural symbol extractor across TS/JS, Python, Go, and Rust.
 */
export function extractFileOutline(
  filePath: string,
  content: string,
  options: ExtractOutlineOptions = {},
): FileOutline {
  const includeImports = options.includeImports ?? true
  const exportedOnly = options.exportedOnly ?? false
  const maxDepth = options.maxDepth ?? DEFAULT_MAX_DEPTH
  const normalized = content.replaceAll('\r\n', '\n')
  const rawLines = normalized === '' ? [] : (normalized.endsWith('\n') ? normalized.slice(0, -1).split('\n') : normalized.split('\n'))
  const language = detectOutlineLanguage(filePath)

  let extracted: { readonly imports?: OutlineImportSummary; readonly symbols: readonly OutlineSymbol[] }
  if (language === 'typescript' || language === 'javascript') {
    extracted = extractTsJsOutline(rawLines, normalized)
  } else if (language === 'python') {
    extracted = extractPythonOutline(rawLines, normalized)
  } else if (language === 'go') {
    extracted = extractGoOutline(rawLines, normalized)
  } else if (language === 'rust') {
    extracted = extractRustOutline(rawLines, normalized)
  } else {
    extracted = { symbols: [] }
  }

  const prunedSymbols = filterAndTrimSymbols(extracted.symbols, exportedOnly, maxDepth, 1)
  return {
    filePath,
    language,
    lineCount: rawLines.length,
    ...(includeImports && extracted.imports !== undefined ? { imports: extracted.imports } : {}),
    symbols: prunedSymbols,
  }
}

export function formatFileOutline(
  outline: FileOutline,
  options: { readonly includeImports?: boolean } = {},
): string {
  const includeImports = options.includeImports ?? true
  const out: string[] = [`# ${outline.filePath} (${outline.language}, ${outline.lineCount} line${outline.lineCount === 1 ? '' : 's'})`]

  if (includeImports && outline.imports && outline.imports.count > 0) {
    const previewMods = outline.imports.modules.slice(0, 8)
    const restCount = outline.imports.modules.length - previewMods.length
    const suffix = restCount > 0 ? `, +${restCount} more` : ''
    const modList = previewMods.length > 0 ? `: ${previewMods.join(', ')}${suffix}` : ''
    out.push(`imports [${outline.imports.startLine}-${outline.imports.endLine}] (${outline.imports.count})${modList}`)
  }

  if (outline.symbols.length === 0) {
    out.push('(no matching symbols)')
    return out.join('\n')
  }

  const renderTree = (items: readonly OutlineSymbol[], indent: string): void => {
    for (const sym of items) {
      const exportTag = sym.exported ? 'export ' : ''
      out.push(`${indent}[${sym.startLine}-${sym.endLine}] ${exportTag}${sym.kind} ${sym.signature}`)
      if (sym.children && sym.children.length > 0) {
        renderTree(sym.children, `${indent}  `)
      }
    }
  }

  renderTree(outline.symbols, '')
  return out.join('\n')
}

function filterAndTrimSymbols(
  symbols: readonly OutlineSymbol[],
  exportedOnly: boolean,
  maxDepth: number,
  currentDepth: number,
): readonly OutlineSymbol[] {
  const result: OutlineSymbol[] = []
  for (const sym of symbols) {
    const filteredChildren = sym.children && currentDepth < maxDepth
      ? filterAndTrimSymbols(sym.children, exportedOnly && sym.kind !== 'class' && sym.kind !== 'interface' && sym.kind !== 'trait', maxDepth, currentDepth + 1)
      : undefined

    if (exportedOnly && !sym.exported) {
      // In Rust, an `impl Type` block is itself not marked `pub`, yet may expose `pub fn` methods.
      if (sym.kind === 'impl' && filteredChildren && filteredChildren.length > 0) {
        result.push({
          name: sym.name,
          kind: sym.kind,
          signature: sym.signature,
          startLine: sym.startLine,
          endLine: sym.endLine,
          exported: sym.exported,
          children: filteredChildren,
        })
      }
      continue
    }

    const base: OutlineSymbol = {
      name: sym.name,
      kind: sym.kind,
      signature: sym.signature,
      startLine: sym.startLine,
      endLine: sym.endLine,
      exported: sym.exported,
      ...(filteredChildren && filteredChildren.length > 0 ? { children: filteredChildren } : {}),
    }
    result.push(base)
  }
  return result
}

// ============================================================================
// Lexical Brace & Delimiter Scanner (TS/JS, Go, Rust)
// ============================================================================

interface BraceToken {
  readonly openOffset: number
  readonly closeOffset: number
  readonly openLine: number
  readonly closeLine: number
  readonly depth: number // 0 = outer-most `{ ... }`
}

interface BraceScanIndex {
  /** Brace depth at the start of each 0-based line. */
  readonly lineStartBraceDepth: readonly number[]
  /** Paren depth at the start of each 0-based line. */
  readonly lineStartParenDepth: readonly number[]
  /** True when the start of the 0-based line is inside a block comment or multiline string. */
  readonly lineStartInCommentOrString: readonly boolean[]
  /** Code-only view of each 0-based line (comments/strings replaced with spaces for safe regex matching). */
  readonly scrubbedLines: readonly string[]
  /** Matched `{ ... }` pairs in source order. */
  readonly braces: readonly BraceToken[]
  /** Matched `( ... )` pairs in source order. */
  readonly parens: readonly BraceToken[]
}

function scanBraceLanguage(text: string, lineCount: number, mode: 'js' | 'go' | 'rust'): BraceScanIndex {
  const lineStartBraceDepth = new Array<number>(lineCount).fill(0)
  const lineStartParenDepth = new Array<number>(lineCount).fill(0)
  const lineStartInCommentOrString = new Array<boolean>(lineCount).fill(false)
  const scrubbedChars = text.split('')

  const braceStack: Array<{ readonly offset: number; readonly line: number; readonly depth: number }> = []
  const parenStack: Array<{ readonly offset: number; readonly line: number; readonly depth: number }> = []
  const templateBraceTargets: number[] = []

  const braces: BraceToken[] = []
  const parens: BraceToken[] = []

  let line = 1
  let i = 0
  const len = text.length

  const markLineTransition = (inLiteral: boolean): void => {
    line += 1
    if (line - 1 < lineCount) {
      lineStartBraceDepth[line - 1] = braceStack.length
      lineStartParenDepth[line - 1] = parenStack.length
      lineStartInCommentOrString[line - 1] = inLiteral
    }
  }

  const scrub = (pos: number): void => {
    if (scrubbedChars[pos] !== '\n') {
      scrubbedChars[pos] = ' '
    }
  }

  while (i < len) {
    const ch = text[i]!
    const next = text[i + 1]

    if (ch === '\n') {
      markLineTransition(false)
      i += 1
      continue
    }

    // Line comment `//`
    if (ch === '/' && next === '/') {
      while (i < len && text[i] !== '\n') {
        scrub(i)
        i += 1
      }
      continue
    }

    // Block comment `/* ... */` (nested in Rust)
    if (ch === '/' && next === '*') {
      scrub(i)
      scrub(i + 1)
      i += 2
      let commentDepth = 1
      while (i < len && commentDepth > 0) {
        if (text[i] === '\n') {
          markLineTransition(true)
          i += 1
          continue
        }
        if (mode === 'rust' && text[i] === '/' && text[i + 1] === '*') {
          commentDepth += 1
          scrub(i)
          scrub(i + 1)
          i += 2
          continue
        }
        if (text[i] === '*' && text[i + 1] === '/') {
          commentDepth -= 1
          scrub(i)
          scrub(i + 1)
          i += 2
          continue
        }
        scrub(i)
        i += 1
      }
      continue
    }

    // Rust raw strings: r"...", r#"..."#, br#"..."#
    if (mode === 'rust' && (ch === 'r' || (ch === 'b' && next === 'r'))) {
      const rStart = ch === 'b' ? i + 1 : i
      if (rStart === 0 || !/[A-Za-z0-9_]/.test(text[rStart - 1]!)) {
        let hashes = 0
        let cursor = rStart + 1
        while (cursor < len && text[cursor] === '#') {
          hashes += 1
          cursor += 1
        }
        if (cursor < len && text[cursor] === '"' && (hashes > 0 || text[rStart] === 'r')) {
          for (let k = i; k <= cursor; k += 1) scrub(k)
          i = cursor + 1
          while (i < len) {
            if (text[i] === '\n') {
              markLineTransition(true)
              i += 1
              continue
            }
            if (text[i] === '"') {
              let closeHashes = 0
              while (closeHashes < hashes && text[i + 1 + closeHashes] === '#') {
                closeHashes += 1
              }
              if (closeHashes === hashes) {
                for (let k = 0; k <= hashes; k += 1) scrub(i + k)
                i += hashes + 1
                break
              }
            }
            scrub(i)
            i += 1
          }
          continue
        }
      }
    }

    // Double-quoted string `"..."`
    if (ch === '"') {
      i += 1
      while (i < len) {
        if (text[i] === '\n') {
          markLineTransition(true)
          i += 1
          continue
        }
        if (text[i] === '\\') {
          scrub(i)
          if (i + 1 < len && text[i + 1] !== '\n') {
            scrub(i + 1)
            i += 2
            continue
          }
          i += 1
          continue
        }
        if (text[i] === '"') {
          i += 1
          break
        }
        scrub(i)
        i += 1
      }
      continue
    }

    // Single-quoted literal `'...'` (careful: in Rust `'a` is a lifetime, NOT a string!)
    if (ch === '\'') {
      if (mode === 'rust') {
        const charMatch = text.slice(i).match(/^'(?:\\(?:x[\da-fA-F]{2}|u\{[\da-fA-F_]+\}|.)|[^'\\\n])'/)
        if (charMatch) {
          for (let k = 1; k < charMatch[0].length - 1; k += 1) scrub(i + k)
          i += charMatch[0].length
          continue
        }
        // Rust lifetime token (`'a`, `'static`, `'_`): leave as normal token.
        i += 1
        continue
      }
      i += 1
      while (i < len && text[i] !== '\n') {
        if (text[i] === '\\') {
          scrub(i)
          if (i + 1 < len && text[i + 1] !== '\n') {
            scrub(i + 1)
            i += 2
            continue
          }
          i += 1
          continue
        }
        if (text[i] === '\'') {
          i += 1
          break
        }
        scrub(i)
        i += 1
      }
      continue
    }

    // Backtick literal: Go raw string or JS/TS template literal
    if (ch === '`' && (mode === 'go' || mode === 'js')) {
      scrub(i)
      i += 1
      if (mode === 'go') {
        while (i < len) {
          if (text[i] === '\n') {
            markLineTransition(true)
            i += 1
            continue
          }
          if (text[i] === '`') {
            scrub(i)
            i += 1
            break
          }
          scrub(i)
          i += 1
        }
        continue
      }
      // JS/TS template literal with `${...}` interpolation tracking
      while (i < len) {
        if (text[i] === '\n') {
          markLineTransition(true)
          i += 1
          continue
        }
        if (text[i] === '\\') {
          scrub(i)
          if (i + 1 < len && text[i + 1] !== '\n') {
            scrub(i + 1)
            i += 2
            continue
          }
          i += 1
          continue
        }
        if (text[i] === '$' && text[i + 1] === '{') {
          scrub(i)
          scrub(i + 1)
          templateBraceTargets.push(braceStack.length)
          braceStack.push({ offset: i + 1, line, depth: braceStack.length })
          i += 2
          break
        }
        if (text[i] === '`') {
          scrub(i)
          i += 1
          break
        }
        scrub(i)
        i += 1
      }
      continue
    }

    // JS/TS same-line regex literal `/pattern/flags` (avoids braces/quotes inside regexes)
    if (mode === 'js' && ch === '/' && next !== '/' && next !== '*') {
      if (isLikelyRegexStart(scrubbedChars, i)) {
        const regexEnd = scanJsRegexSameLine(text, i)
        if (regexEnd > i) {
          for (let k = i; k < regexEnd; k += 1) scrub(k)
          i = regexEnd
          continue
        }
      }
    }

    if (ch === '{') {
      braceStack.push({ offset: i, line, depth: braceStack.length })
      i += 1
      continue
    }

    if (ch === '}') {
      const opened = braceStack.pop()
      if (opened !== undefined) {
        braces.push({
          openOffset: opened.offset,
          closeOffset: i,
          openLine: opened.line,
          closeLine: line,
          depth: opened.depth,
        })
        if (
          mode === 'js'
          && templateBraceTargets.length > 0
          && templateBraceTargets[templateBraceTargets.length - 1] === braceStack.length
        ) {
          templateBraceTargets.pop()
          scrub(i)
          i += 1
          // Resume scanning rest of JS template literal after `${...}` closes
          while (i < len) {
            if (text[i] === '\n') {
              markLineTransition(true)
              i += 1
              continue
            }
            if (text[i] === '\\') {
              scrub(i)
              if (i + 1 < len && text[i + 1] !== '\n') {
                scrub(i + 1)
                i += 2
                continue
              }
              i += 1
              continue
            }
            if (text[i] === '$' && text[i + 1] === '{') {
              scrub(i)
              scrub(i + 1)
              templateBraceTargets.push(braceStack.length)
              braceStack.push({ offset: i + 1, line, depth: braceStack.length })
              i += 2
              break
            }
            if (text[i] === '`') {
              scrub(i)
              i += 1
              break
            }
            scrub(i)
            i += 1
          }
          continue
        }
      }
      i += 1
      continue
    }

    if (ch === '(') {
      parenStack.push({ offset: i, line, depth: parenStack.length })
      i += 1
      continue
    }

    if (ch === ')') {
      const opened = parenStack.pop()
      if (opened !== undefined) {
        parens.push({
          openOffset: opened.offset,
          closeOffset: i,
          openLine: opened.line,
          closeLine: line,
          depth: opened.depth,
        })
      }
      i += 1
      continue
    }

    i += 1
  }

  braces.sort((a, b) => a.openOffset - b.openOffset)
  parens.sort((a, b) => a.openOffset - b.openOffset)

  const scrubbedText = scrubbedChars.join('')
  const scrubbedLines = lineCount === 0
    ? []
    : (scrubbedText.endsWith('\n') ? scrubbedText.slice(0, -1).split('\n') : scrubbedText.split('\n'))

  return {
    lineStartBraceDepth,
    lineStartParenDepth,
    lineStartInCommentOrString,
    scrubbedLines,
    braces,
    parens,
  }
}

function isLikelyRegexStart(scrubbed: readonly string[], slashPos: number): boolean {
  let cursor = slashPos - 1
  while (cursor >= 0 && (scrubbed[cursor] === ' ' || scrubbed[cursor] === '\t' || scrubbed[cursor] === '\n')) {
    cursor -= 1
  }
  if (cursor < 0) return true
  const prev = scrubbed[cursor]!
  if ('=([,;:!&|?+-*~^%<>/{'.includes(prev)) return true
  // Check preceding keyword like `return`, `typeof`, `case`
  let wordStart = cursor
  while (wordStart > 0 && /[A-Za-z0-9_$]/.test(scrubbed[wordStart - 1]!)) {
    wordStart -= 1
  }
  const word = scrubbed.slice(wordStart, cursor + 1).join('')
  return word === 'return' || word === 'case' || word === 'throw' || word === 'typeof' || word === 'void' || word === 'in' || word === 'of'
}

function scanJsRegexSameLine(text: string, startSlash: number): number {
  let i = startSlash + 1
  let inCharClass = false
  if (i >= text.length || text[i] === '\n' || text[i] === ' ') return -1
  while (i < text.length && text[i] !== '\n') {
    const ch = text[i]!
    if (ch === '\\') {
      if (i + 1 >= text.length || text[i + 1] === '\n') return -1
      i += 2
      continue
    }
    if (ch === '[') {
      inCharClass = true
      i += 1
      continue
    }
    if (ch === ']' && inCharClass) {
      inCharClass = false
      i += 1
      continue
    }
    if (ch === '/' && !inCharClass) {
      i += 1
      while (i < text.length && /[dgimsuvy]/.test(text[i]!)) {
        i += 1
      }
      return i
    }
    i += 1
  }
  return -1
}

function findBlockEndLine(
  scan: BraceScanIndex,
  fromLine1Based: number,
  expectedBraceDepth: number,
  maxSearchLines = 40,
): { readonly bodyBrace?: BraceToken; readonly endLine: number } {
  const maxLine = Math.min(scan.scrubbedLines.length, fromLine1Based + maxSearchLines)
  // Find first `{` at `expectedBraceDepth` opened on `[fromLine1Based .. maxLine]` before a top-level `;` or blank statement boundary.
  let semicolonLine: number | undefined
  for (let ln = fromLine1Based; ln <= maxLine; ln += 1) {
    const brace = scan.braces.find(b => b.openLine === ln && b.depth === expectedBraceDepth)
    if (brace !== undefined) {
      return { bodyBrace: brace, endLine: brace.closeLine }
    }
    const scrubbed = scan.scrubbedLines[ln - 1] ?? ''
    if (scrubbed.includes(';') && (scan.lineStartParenDepth[ln - 1] ?? 0) === 0) {
      semicolonLine = ln
      break
    }
    // Check if an unclosed paren started on `fromLine1Based` spans further lines
    const paren = scan.parens.find(p => p.openLine === ln && p.depth === 0)
    if (paren !== undefined && paren.closeLine > ln) {
      const afterParenBrace = scan.braces.find(
        b => b.openLine >= paren.closeLine && b.openLine <= paren.closeLine + 3 && b.depth === expectedBraceDepth,
      )
      if (afterParenBrace !== undefined) {
        return { bodyBrace: afterParenBrace, endLine: afterParenBrace.closeLine }
      }
    }
    if (ln > fromLine1Based && scrubbed.trim() === '' && (scan.lineStartParenDepth[ln - 1] ?? 0) === 0) {
      return { endLine: ln - 1 }
    }
  }
  if (semicolonLine !== undefined) {
    return { endLine: semicolonLine }
  }
  return { endLine: fromLine1Based }
}

// ============================================================================
// TypeScript / JavaScript Extractor
// ============================================================================

function extractTsJsOutline(
  rawLines: readonly string[],
  fullText: string,
): { readonly imports?: OutlineImportSummary; readonly symbols: readonly OutlineSymbol[] } {
  if (rawLines.length === 0) return { symbols: [] }
  const scan = scanBraceLanguage(fullText, rawLines.length, 'js')
  const symbols: OutlineSymbol[] = []
  const importModules: string[] = []
  let importCount = 0
  let importStart = Number.POSITIVE_INFINITY
  let importEnd = 0

  let lineIndex = 0
  while (lineIndex < rawLines.length) {
    if (scan.lineStartInCommentOrString[lineIndex] || (scan.lineStartBraceDepth[lineIndex] ?? 0) !== 0 || (scan.lineStartParenDepth[lineIndex] ?? 0) !== 0) {
      lineIndex += 1
      continue
    }

    const scrubbed = scan.scrubbedLines[lineIndex]!.trim()
    if (!scrubbed) {
      lineIndex += 1
      continue
    }

    // Track leading decorators (`@Component(...)`)
    let decoratorStartLine: number | undefined
    if (scrubbed.startsWith('@')) {
      decoratorStartLine = lineIndex + 1
      while (lineIndex < rawLines.length && (scan.scrubbedLines[lineIndex] ?? '').trim().startsWith('@')) {
        const paren = scan.parens.find(p => p.openLine === lineIndex + 1 && p.depth === 0)
        lineIndex = paren ? paren.closeLine : lineIndex + 1
      }
      if (lineIndex >= rawLines.length) break
    }

    const currentScrubbed = (scan.scrubbedLines[lineIndex] ?? '').trim()
    const currentRaw = (rawLines[lineIndex] ?? '').trim()
    const declLine1 = lineIndex + 1
    const startLine = decoratorStartLine ?? declLine1

    // Top-level imports (`import ...` excluding `import foo = require(...)`? Capture both!)
    if (/^import\b/.test(currentScrubbed)) {
      const endLine = findStatementEndLine(rawLines, scan, declLine1, 0)
      importCount += 1
      importStart = Math.min(importStart, declLine1)
      importEnd = Math.max(importEnd, endLine)
      const slice = rawLines.slice(declLine1 - 1, endLine).join(' ')
      const specMatch = slice.match(/\bfrom\s*['"]([^'"]+)['"]/) ?? slice.match(/^import\s*['"]([^'"]+)['"]/)
      if (specMatch?.[1] && !importModules.includes(specMatch[1])) {
        importModules.push(specMatch[1])
      }
      lineIndex = endLine
      continue
    }

    // Re-export statements (`export * from '...'` or `export { ... } from '...'`)
    if (/^export\s+(?:type\s+)?(?:\*|\{)/.test(currentScrubbed)) {
      const endLine = findStatementEndLine(rawLines, scan, declLine1, 0)
      const sig = compactSignature(rawLines.slice(declLine1 - 1, endLine).join(' ').replace(/^export\s+/, ''))
      symbols.push({
        name: 're-export',
        kind: 'export',
        signature: sig,
        startLine: declLine1,
        endLine,
        exported: true,
      })
      lineIndex = endLine
      continue
    }

    const exported = /^export\b/.test(currentScrubbed)

    // Class declaration
    const classMatch = currentScrubbed.match(/^(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?class\s+([A-Za-z_$][A-Za-z0-9_$]*)/)
    if (classMatch?.[1]) {
      const name = classMatch[1]
      const block = findBlockEndLine(scan, declLine1, 0)
      const sig = extractHeaderBeforeBrace(rawLines, declLine1, block.bodyBrace?.openLine ?? block.endLine)
        .replace(/^export\s+/, '')
        .replace(/^(?:default\s+)?(?:declare\s+)?(?:abstract\s+)?class\s+/, '')
      const children = block.bodyBrace
        ? extractTsClassMembers(rawLines, scan, block.bodyBrace.openLine, block.bodyBrace.closeLine, 1)
        : []
      symbols.push({
        name,
        kind: 'class',
        signature: compactSignature(sig),
        startLine,
        endLine: block.endLine,
        exported,
        ...(children.length > 0 ? { children } : {}),
      })
      lineIndex = block.endLine
      continue
    }

    // Interface declaration
    const ifaceMatch = currentScrubbed.match(/^(?:export\s+)?(?:default\s+)?(?:declare\s+)?interface\s+([A-Za-z_$][A-Za-z0-9_$]*)/)
    if (ifaceMatch?.[1]) {
      const name = ifaceMatch[1]
      const block = findBlockEndLine(scan, declLine1, 0)
      const sig = extractHeaderBeforeBrace(rawLines, declLine1, block.bodyBrace?.openLine ?? block.endLine)
        .replace(/^export\s+/, '')
        .replace(/^(?:default\s+)?(?:declare\s+)?interface\s+/, '')
      const children = block.bodyBrace
        ? extractTsInterfaceMembers(rawLines, scan, block.bodyBrace.openLine, block.bodyBrace.closeLine, 1)
        : []
      symbols.push({
        name,
        kind: 'interface',
        signature: compactSignature(sig),
        startLine,
        endLine: block.endLine,
        exported,
        ...(children.length > 0 ? { children } : {}),
      })
      lineIndex = block.endLine
      continue
    }

    // Enum declaration
    const enumMatch = currentScrubbed.match(/^(?:export\s+)?(?:declare\s+)?(?:const\s+)?enum\s+([A-Za-z_$][A-Za-z0-9_$]*)/)
    if (enumMatch?.[1]) {
      const name = enumMatch[1]
      const block = findBlockEndLine(scan, declLine1, 0)
      const sig = extractHeaderBeforeBrace(rawLines, declLine1, block.bodyBrace?.openLine ?? block.endLine)
        .replace(/^export\s+/, '')
        .replace(/^const\s+enum\s+/, '')
        .replace(/^enum\s+/, '')
      symbols.push({
        name,
        kind: 'enum',
        signature: compactSignature(sig || name),
        startLine,
        endLine: block.endLine,
        exported,
      })
      lineIndex = block.endLine
      continue
    }

    // Type alias (`type Foo = ...`)
    const typeMatch = currentScrubbed.match(/^(?:export\s+)?(?:declare\s+)?type\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*(?:<|=)/)
    if (typeMatch?.[1]) {
      const name = typeMatch[1]
      const endLine = findTsTypeAliasEndLine(rawLines, scan, declLine1)
      const firstLineClean = currentRaw.replace(/^export\s+/, '').replace(/^declare\s+/, '').replace(/^type\s+/, '')
      symbols.push({
        name,
        kind: 'type',
        signature: compactSignature(firstLineClean),
        startLine,
        endLine,
        exported,
      })
      lineIndex = endLine
      continue
    }

    // Function declaration (`function foo(...)`, `async function* foo(...)`)
    const fnMatch = currentScrubbed.match(/^(?:export\s+)?(?:default\s+)?(?:declare\s+)?(?:async\s+)?function\s*\*?\s*([A-Za-z_$][A-Za-z0-9_$]*)/)
    if (fnMatch?.[1]) {
      const name = fnMatch[1]
      const block = findBlockEndLine(scan, declLine1, 0)
      const sig = extractHeaderBeforeBrace(rawLines, declLine1, block.bodyBrace?.openLine ?? block.endLine)
        .replace(/^export\s+/, '')
        .replace(/^default\s+/, '')
        .replace(/^declare\s+/, '')
        .replace(/^(async\s+)?function(\s*\*)?\s+/, '$1')
      symbols.push({
        name,
        kind: 'function',
        signature: compactSignature(sig),
        startLine,
        endLine: block.endLine,
        exported,
      })
      lineIndex = block.endLine
      continue
    }

    // Top-level const / let / var (including arrow functions & tool definitions)
    const varMatch = currentScrubbed.match(/^(?:export\s+)?(?:declare\s+)?(const|let|var)\s+([A-Za-z_$][A-Za-z0-9_$]*)/)
    if (varMatch?.[1] && varMatch[2]) {
      const declKeyword = varMatch[1]
      const name = varMatch[2]
      const endLine = findStatementEndLine(rawLines, scan, declLine1, 0)
      const spanScrubbed = scan.scrubbedLines.slice(declLine1 - 1, Math.min(endLine, declLine1 + 4)).join(' ')
      const isFnExpr = /=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=>/.test(spanScrubbed)
        || /=\s*(?:async\s+)?function\b/.test(spanScrubbed)
      const kind: OutlineSymbolKind = isFnExpr ? 'function' : declKeyword === 'const' ? 'constant' : 'variable'
      const rawHeader = rawLines[declLine1 - 1]!.trim().replace(/^export\s+/, '').replace(/^(?:const|let|var)\s+/, '')
      symbols.push({
        name,
        kind,
        signature: compactSignature(rawHeader),
        startLine,
        endLine,
        exported,
      })
      lineIndex = endLine
      continue
    }

    // Namespace / module block
    const nsMatch = currentScrubbed.match(/^(?:export\s+)?(?:declare\s+)?(?:namespace|module)\s+([A-Za-z_$][A-Za-z0-9_$]*)/)
    if (nsMatch?.[1]) {
      const name = nsMatch[1]
      const block = findBlockEndLine(scan, declLine1, 0)
      symbols.push({
        name,
        kind: 'module',
        signature: name,
        startLine,
        endLine: block.endLine,
        exported,
      })
      lineIndex = block.endLine
      continue
    }

    lineIndex += 1
  }

  return {
    ...(importCount > 0 ? {
      imports: {
        count: importCount,
        startLine: importStart,
        endLine: importEnd,
        modules: importModules,
      },
    } : {}),
    symbols,
  }
}

function extractTsClassMembers(
  rawLines: readonly string[],
  scan: BraceScanIndex,
  openLine1: number,
  closeLine1: number,
  classBraceDepth: number,
): OutlineSymbol[] {
  const members: OutlineSymbol[] = []
  let lineIdx = openLine1 // start at line after `{` unless single-line; openLine1 is 1-based so index openLine1 is line openLine1 + 1
  while (lineIdx < closeLine1 - 1) {
    if (
      scan.lineStartInCommentOrString[lineIdx]
      || (scan.lineStartBraceDepth[lineIdx] ?? 0) !== classBraceDepth
      || (scan.lineStartParenDepth[lineIdx] ?? 0) !== 0
    ) {
      lineIdx += 1
      continue
    }

    const scrubbed = (scan.scrubbedLines[lineIdx] ?? '').trim()
    if (!scrubbed || scrubbed === '}' || scrubbed === ';') {
      lineIdx += 1
      continue
    }

    let decoratorStart: number | undefined
    if (scrubbed.startsWith('@')) {
      decoratorStart = lineIdx + 1
      while (lineIdx < closeLine1 - 1 && (scan.scrubbedLines[lineIdx] ?? '').trim().startsWith('@')) {
        lineIdx += 1
      }
      if (lineIdx >= closeLine1 - 1) break
    }

    const memberScrubbed = (scan.scrubbedLines[lineIdx] ?? '').trim()
    const memberLine1 = lineIdx + 1
    const startLine = decoratorStart ?? memberLine1
    const isPrivate = /^(?:private\b|#)/.test(memberScrubbed) || /\bprivate\s+/.test(memberScrubbed)

    // Method / constructor / getter / setter
    const methodMatch = memberScrubbed.match(
      /^(?:(?:public|protected|private|readonly|static|abstract|override|async)\s+)*(?:(get|set)\s+)?(\*?\s*#?[A-Za-z_$][A-Za-z0-9_$]*)\s*(?:<[^>]*>)?\s*\(/,
    )
    if (methodMatch?.[2] && !['if', 'for', 'while', 'switch', 'catch'].includes(methodMatch[2].trim())) {
      const accessor = methodMatch[1] ? `${methodMatch[1]} ` : ''
      const rawName = methodMatch[2].replace(/^\*\s*/, '').trim()
      const block = findBlockEndLine(scan, memberLine1, classBraceDepth)
      const endLine = Math.min(closeLine1 - 1, Math.max(memberLine1, block.endLine))
      const sig = extractHeaderBeforeBrace(rawLines, memberLine1, block.bodyBrace?.openLine ?? endLine)
      members.push({
        name: `${accessor}${rawName}`.trim(),
        kind: 'method',
        signature: compactSignature(sig),
        startLine,
        endLine,
        exported: !isPrivate && !rawName.startsWith('#'),
      })
      lineIdx = endLine
      continue
    }

    // Arrow-function class field (`readonly run = async (...) => { ... }`)
    const arrowFieldMatch = memberScrubbed.match(
      /^(?:(?:public|protected|private|readonly|static|override)\s+)*(#?[A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*(?::[^=]+)?=>/,
    )
    if (arrowFieldMatch?.[1]) {
      const rawName = arrowFieldMatch[1]
      const endLine = Math.min(closeLine1 - 1, findStatementEndLine(rawLines, scan, memberLine1, classBraceDepth))
      members.push({
        name: rawName,
        kind: 'method',
        signature: compactSignature((rawLines[memberLine1 - 1] ?? '').trim()),
        startLine,
        endLine,
        exported: !isPrivate && !rawName.startsWith('#'),
      })
      lineIdx = endLine
      continue
    }

    lineIdx += 1
  }
  return members
}

function extractTsInterfaceMembers(
  rawLines: readonly string[],
  scan: BraceScanIndex,
  openLine1: number,
  closeLine1: number,
  ifaceBraceDepth: number,
): OutlineSymbol[] {
  const members: OutlineSymbol[] = []
  for (let lineIdx = openLine1; lineIdx < closeLine1 - 1; lineIdx += 1) {
    if (
      scan.lineStartInCommentOrString[lineIdx]
      || (scan.lineStartBraceDepth[lineIdx] ?? 0) !== ifaceBraceDepth
    ) {
      continue
    }
    const scrubbed = (scan.scrubbedLines[lineIdx] ?? '').trim()
    if (!scrubbed || scrubbed === '}') continue
    const methodSig = scrubbed.match(/^(?:readonly\s+)?([A-Za-z_$][A-Za-z0-9_$]*)\??\s*(?:<[^>]*>)?\s*\(/)
    if (methodSig?.[1]) {
      const line1 = lineIdx + 1
      const endLine = Math.min(closeLine1 - 1, findStatementEndLine(rawLines, scan, line1, ifaceBraceDepth))
      members.push({
        name: methodSig[1],
        kind: 'method',
        signature: compactSignature((rawLines[lineIdx] ?? '').trim().replace(/[;,]$/, '')),
        startLine: line1,
        endLine,
        exported: true,
      })
      lineIdx = endLine - 1
    }
  }
  return members
}

// ============================================================================
// Python Extractor
// ============================================================================

interface PythonLineInfo {
  readonly indent: number
  readonly scrubbed: string
  readonly inTripleStringStart: boolean
  readonly parenDepthStart: number
}

function scanPythonLines(rawLines: readonly string[]): readonly PythonLineInfo[] {
  const infos: PythonLineInfo[] = []
  let inTriple: '"""' | '\'\'\'' | undefined
  let parenDepth = 0

  for (let idx = 0; idx < rawLines.length; idx += 1) {
    const raw = rawLines[idx]!
    const inTripleStart = inTriple !== undefined
    const parenStart = parenDepth
    const chars = raw.split('')
    let i = 0

    while (i < raw.length) {
      if (inTriple !== undefined) {
        if (raw[i] === '\\') {
          chars[i] = ' '
          if (i + 1 < raw.length) chars[i + 1] = ' '
          i += 2
          continue
        }
        if (raw.startsWith(inTriple, i)) {
          chars[i] = ' '
          chars[i + 1] = ' '
          chars[i + 2] = ' '
          inTriple = undefined
          i += 3
          continue
        }
        chars[i] = ' '
        i += 1
        continue
      }

      const ch = raw[i]!
      if (ch === '#') {
        while (i < raw.length) {
          chars[i] = ' '
          i += 1
        }
        break
      }
      if (raw.startsWith('"""', i) || raw.startsWith('\'\'\'', i)) {
        const delim = raw.slice(i, i + 3) as '"""' | '\'\'\''
        chars[i] = ' '
        chars[i + 1] = ' '
        chars[i + 2] = ' '
        i += 3
        let closedSameLine = false
        while (i < raw.length) {
          if (raw[i] === '\\') {
            chars[i] = ' '
            if (i + 1 < raw.length) chars[i + 1] = ' '
            i += 2
            continue
          }
          if (raw.startsWith(delim, i)) {
            chars[i] = ' '
            chars[i + 1] = ' '
            chars[i + 2] = ' '
            i += 3
            closedSameLine = true
            break
          }
          chars[i] = ' '
          i += 1
        }
        if (!closedSameLine) {
          inTriple = delim
        }
        continue
      }
      if (ch === '"' || ch === '\'') {
        const quote = ch
        chars[i] = ' '
        i += 1
        while (i < raw.length) {
          if (raw[i] === '\\') {
            chars[i] = ' '
            if (i + 1 < raw.length) chars[i + 1] = ' '
            i += 2
            continue
          }
          if (raw[i] === quote) {
            chars[i] = ' '
            i += 1
            break
          }
          chars[i] = ' '
          i += 1
        }
        continue
      }
      if (ch === '(' || ch === '[' || ch === '{') parenDepth += 1
      else if ((ch === ')' || ch === ']' || ch === '}') && parenDepth > 0) parenDepth -= 1
      i += 1
    }

    const scrubbed = chars.join('')
    const indent = countPythonIndent(raw)
    infos.push({
      indent,
      scrubbed,
      inTripleStringStart: inTripleStart,
      parenDepthStart: parenStart,
    })
  }

  return infos
}

function countPythonIndent(line: string): number {
  let col = 0
  for (let i = 0; i < line.length; i += 1) {
    if (line[i] === ' ') col += 1
    else if (line[i] === '\t') col += 4
    else break
  }
  return col
}

function extractPythonOutline(
  rawLines: readonly string[],
  fullText: string,
): { readonly imports?: OutlineImportSummary; readonly symbols: readonly OutlineSymbol[] } {
  if (rawLines.length === 0) return { symbols: [] }
  const infos = scanPythonLines(rawLines)
  const explicitDunderAll = parsePythonDunderAll(fullText)

  const importModules: string[] = []
  let importCount = 0
  let importStart = Number.POSITIVE_INFINITY
  let importEnd = 0

  for (let idx = 0; idx < rawLines.length; idx += 1) {
    const info = infos[idx]!
    if (info.inTripleStringStart || info.parenDepthStart > 0 || info.indent !== 0) continue
    const trimmed = info.scrubbed.trim()
    const fromMatch = trimmed.match(/^from\s+([A-Za-z0-9_.]+)\s+import\b/)
    const impMatch = trimmed.match(/^import\s+([A-Za-z0-9_.,\s]+)/)
    if (fromMatch?.[1] || impMatch?.[1]) {
      const start1 = idx + 1
      let endIdx = idx
      while (endIdx + 1 < rawLines.length && infos[endIdx + 1]!.parenDepthStart > 0) {
        endIdx += 1
      }
      importCount += 1
      importStart = Math.min(importStart, start1)
      importEnd = Math.max(importEnd, endIdx + 1)
      if (fromMatch?.[1]) {
        if (!importModules.includes(fromMatch[1])) importModules.push(fromMatch[1])
      } else if (impMatch?.[1]) {
        for (const part of impMatch[1].split(',')) {
          const modName = part.trim().split(/\s+as\s+/)[0]?.trim()
          if (modName && !importModules.includes(modName)) importModules.push(modName)
        }
      }
      idx = endIdx
    }
  }

  const isPublic = (name: string): boolean =>
    explicitDunderAll !== undefined ? explicitDunderAll.has(name) : !name.startsWith('_')

  const parseSuite = (startIdx: number, endIdxExclusive: number, expectedIndent: number | undefined, insideClass: boolean): OutlineSymbol[] => {
    const items: OutlineSymbol[] = []
    let idx = startIdx
    let suiteIndent = expectedIndent

    while (idx < endIdxExclusive) {
      const info = infos[idx]!
      if (info.inTripleStringStart || info.parenDepthStart > 0) {
        idx += 1
        continue
      }
      const trimmed = info.scrubbed.trim()
      if (!trimmed) {
        idx += 1
        continue
      }
      if (suiteIndent === undefined) {
        suiteIndent = info.indent
      }
      if (info.indent !== suiteIndent) {
        idx += 1
        continue
      }

      // Capture decorator block `@...`
      let decoratorStart1: number | undefined
      if (trimmed.startsWith('@')) {
        decoratorStart1 = idx + 1
        while (idx < endIdxExclusive) {
          const decInfo = infos[idx]!
          const decTrimmed = decInfo.scrubbed.trim()
          if (!decInfo.inTripleStringStart && decInfo.parenDepthStart === 0 && decTrimmed !== '' && !decTrimmed.startsWith('@')) {
            break
          }
          idx += 1
        }
        if (idx >= endIdxExclusive) break
      }

      const declInfo = infos[idx]!
      const declTrimmed = declInfo.scrubbed.trim()
      if (declInfo.indent !== suiteIndent) {
        continue
      }

      const classMatch = declTrimmed.match(/^class\s+([A-Za-z_][A-Za-z0-9_]*)/)
      if (classMatch?.[1]) {
        const name = classMatch[1]
        const headerEndIdx = findPythonHeaderEndIdx(infos, idx, endIdxExclusive)
        const suiteEndIdx = findPythonSuiteEndIdx(rawLines, infos, headerEndIdx + 1, endIdxExclusive, suiteIndent)
        const endLine = Math.max(headerEndIdx + 1, suiteEndIdx + 1)
        const headerRaw = joinPythonHeader(rawLines, idx, headerEndIdx).replace(/^class\s+/, '')
        const children = suiteEndIdx > headerEndIdx
          ? parseSuite(headerEndIdx + 1, suiteEndIdx + 1, undefined, true)
          : []
        items.push({
          name,
          kind: 'class',
          signature: compactSignature(headerRaw),
          startLine: decoratorStart1 ?? (idx + 1),
          endLine,
          exported: isPublic(name),
          ...(children.length > 0 ? { children } : {}),
        })
        idx = Math.max(idx + 1, suiteEndIdx + 1)
        continue
      }

      const defMatch = declTrimmed.match(/^(async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)/)
      if (defMatch?.[2]) {
        const asyncPrefix = defMatch[1] ? 'async ' : ''
        const name = defMatch[2]
        const headerEndIdx = findPythonHeaderEndIdx(infos, idx, endIdxExclusive)
        const suiteEndIdx = findPythonSuiteEndIdx(rawLines, infos, headerEndIdx + 1, endIdxExclusive, suiteIndent)
        const endLine = Math.max(headerEndIdx + 1, suiteEndIdx + 1)
        const headerRaw = joinPythonHeader(rawLines, idx, headerEndIdx).replace(/^(?:async\s+)?def\s+/, asyncPrefix)
        const children = suiteEndIdx > headerEndIdx
          ? parseSuite(headerEndIdx + 1, suiteEndIdx + 1, undefined, false)
          : []
        items.push({
          name,
          kind: insideClass ? 'method' : 'function',
          signature: compactSignature(headerRaw),
          startLine: decoratorStart1 ?? (idx + 1),
          endLine,
          exported: insideClass ? (!name.startsWith('_') || (name.startsWith('__') && name.endsWith('__'))) : isPublic(name),
          ...(children.length > 0 ? { children } : {}),
        })
        idx = Math.max(idx + 1, suiteEndIdx + 1)
        continue
      }

      // Python 3.12 `type Foo = ...` or top-level constants `MAX_ITEMS = ...`
      if (suiteIndent === 0 && !insideClass) {
        const pyTypeMatch = declTrimmed.match(/^type\s+([A-Za-z_][A-Za-z0-9_]*)\s*[=[[]/)
        if (pyTypeMatch?.[1]) {
          let stmtEnd = idx
          while (stmtEnd + 1 < endIdxExclusive && infos[stmtEnd + 1]!.parenDepthStart > 0) stmtEnd += 1
          items.push({
            name: pyTypeMatch[1],
            kind: 'type',
            signature: compactSignature(rawLines[idx]!.trim().replace(/^type\s+/, '')),
            startLine: idx + 1,
            endLine: stmtEnd + 1,
            exported: isPublic(pyTypeMatch[1]),
          })
          idx = stmtEnd + 1
          continue
        }

        const constMatch = declTrimmed.match(/^([A-Z][A-Z0-9_]*)\s*(?::[^=]+)?=/)
        if (constMatch?.[1]) {
          let stmtEnd = idx
          while (stmtEnd + 1 < endIdxExclusive && infos[stmtEnd + 1]!.parenDepthStart > 0) stmtEnd += 1
          items.push({
            name: constMatch[1],
            kind: 'constant',
            signature: compactSignature(rawLines[idx]!.trim()),
            startLine: idx + 1,
            endLine: stmtEnd + 1,
            exported: isPublic(constMatch[1]),
          })
          idx = stmtEnd + 1
          continue
        }
      }

      idx += 1
    }

    return items
  }

  const symbols = parseSuite(0, rawLines.length, 0, false)
  return {
    ...(importCount > 0 ? {
      imports: {
        count: importCount,
        startLine: importStart,
        endLine: importEnd,
        modules: importModules,
      },
    } : {}),
    symbols,
  }
}

function findPythonHeaderEndIdx(infos: readonly PythonLineInfo[], startIdx: number, limitExclusive: number): number {
  let idx = startIdx
  while (idx + 1 < limitExclusive && infos[idx + 1]!.parenDepthStart > 0) {
    idx += 1
  }
  return idx
}

function findPythonSuiteEndIdx(
  rawLines: readonly string[],
  infos: readonly PythonLineInfo[],
  bodyStartIdx: number,
  limitExclusive: number,
  declIndent: number,
): number {
  let lastBodyLineIdx = bodyStartIdx - 1
  for (let idx = bodyStartIdx; idx < limitExclusive; idx += 1) {
    const info = infos[idx]!
    if (info.inTripleStringStart || info.parenDepthStart > 0) {
      lastBodyLineIdx = idx
      continue
    }
    const rawTrimmed = rawLines[idx]!.trim()
    if (rawTrimmed === '') continue
    if (rawTrimmed.startsWith('#')) {
      if (info.indent > declIndent) {
        continue
      }
      // Top-level comment at `<= declIndent`: check if the suite actually continues after it.
      continue
    }
    if (info.indent <= declIndent) {
      break
    }
    lastBodyLineIdx = idx
  }
  return lastBodyLineIdx
}

function joinPythonHeader(rawLines: readonly string[], startIdx: number, endIdx: number): string {
  const joined = rawLines
    .slice(startIdx, endIdx + 1)
    .map(line => line.replace(/#.*$/, '').trim())
    .join(' ')
  return joined.replace(/:\s*$/, '')
}

function parsePythonDunderAll(fullText: string): ReadonlySet<string> | undefined {
  const match = fullText.match(/^[ \t]*__all__\s*=\s*[[(]([\s\S]*?)[\])]/m)
  if (!match?.[1]) return undefined
  const names = new Set<string>()
  for (const item of match[1].matchAll(/['"]([A-Za-z_][A-Za-z0-9_]*)['"]/g)) {
    if (item[1]) names.add(item[1])
  }
  return names.size > 0 ? names : undefined
}

// ============================================================================
// Go Extractor
// ============================================================================

function extractGoOutline(
  rawLines: readonly string[],
  fullText: string,
): { readonly imports?: OutlineImportSummary; readonly symbols: readonly OutlineSymbol[] } {
  if (rawLines.length === 0) return { symbols: [] }
  const scan = scanBraceLanguage(fullText, rawLines.length, 'go')
  const topSymbols: OutlineSymbol[] = []
  const structMethods = new Map<string, OutlineSymbol[]>()
  const structIndices = new Map<string, number>()

  const importModules: string[] = []
  let importCount = 0
  let importStart = Number.POSITIVE_INFINITY
  let importEnd = 0

  let lineIndex = 0
  while (lineIndex < rawLines.length) {
    if (
      scan.lineStartInCommentOrString[lineIndex]
      || (scan.lineStartBraceDepth[lineIndex] ?? 0) !== 0
      || (scan.lineStartParenDepth[lineIndex] ?? 0) !== 0
    ) {
      lineIndex += 1
      continue
    }

    const scrubbed = (scan.scrubbedLines[lineIndex] ?? '').trim()
    const rawLine = (rawLines[lineIndex] ?? '').trim()
    const line1 = lineIndex + 1

    if (!scrubbed || scrubbed.startsWith('package ')) {
      lineIndex += 1
      continue
    }

    // `import "..."` or `import ( ... )`
    if (/^import\b/.test(scrubbed)) {
      const paren = scan.parens.find(p => p.openLine === line1 && p.depth === 0)
      const endLine = paren ? paren.closeLine : line1
      importStart = Math.min(importStart, line1)
      importEnd = Math.max(importEnd, endLine)
      for (let ln = line1; ln <= endLine; ln += 1) {
        const strMatch = rawLines[ln - 1]!.match(/"([^"]+)"/)
        if (strMatch?.[1]) {
          importCount += 1
          if (!importModules.includes(strMatch[1])) importModules.push(strMatch[1])
        }
      }
      lineIndex = endLine
      continue
    }

    // `type Name struct { ... }` / `type Name interface { ... }` / `type Name ...`
    if (/^type\b/.test(scrubbed)) {
      const parenGroup = scan.parens.find(p => p.openLine === line1 && p.depth === 0)
      if (parenGroup && /^type\s*\(/.test(scrubbed)) {
        // Grouped `type ( ... )`
        for (let innerIdx = line1; innerIdx < parenGroup.closeLine - 1; innerIdx += 1) {
          const innerScrub = (scan.scrubbedLines[innerIdx] ?? '').trim()
          const innerMatch = innerScrub.match(/^([A-Za-z_][A-Za-z0-9_]*)\s+(struct|interface)?/)
          if (innerMatch?.[1]) {
            const innerLine1 = innerIdx + 1
            const kind: OutlineSymbolKind = innerMatch[2] === 'struct' ? 'struct' : innerMatch[2] === 'interface' ? 'interface' : 'type'
            const brace = scan.braces.find(b => b.openLine === innerLine1 && b.depth === 0)
            const endInner = brace ? brace.closeLine : innerLine1
            const sym: OutlineSymbol = {
              name: innerMatch[1],
              kind,
              signature: compactSignature((rawLines[innerIdx] ?? '').trim().replace(/\{.*$/, '')),
              startLine: innerLine1,
              endLine: endInner,
              exported: /^[A-Z]/.test(innerMatch[1]),
            }
            if (kind === 'struct') structIndices.set(innerMatch[1], topSymbols.length)
            topSymbols.push(sym)
            innerIdx = endInner - 1
          }
        }
        lineIndex = parenGroup.closeLine
        continue
      }

      const typeMatch = scrubbed.match(/^type\s+([A-Za-z_][A-Za-z0-9_]*)(?:\s*\[[^\]]*\])?\s*(=?\s*struct\b|=?\s*interface\b)?/)
      if (typeMatch?.[1]) {
        const name = typeMatch[1]
        const rhs = (typeMatch[2] ?? '').replace(/^=\s*/, '')
        const kind: OutlineSymbolKind = rhs === 'struct' ? 'struct' : rhs === 'interface' ? 'interface' : 'type'
        const block = findBlockEndLine(scan, line1, 0, 5)
        const sig = extractHeaderBeforeBrace(rawLines, line1, block.bodyBrace?.openLine ?? block.endLine)
          .replace(/^type\s+/, '')
          .replace(/\s*=?\s*(?:struct|interface)$/, '')
        const ifaceChildren = kind === 'interface' && block.bodyBrace
          ? extractGoInterfaceMethods(rawLines, scan, block.bodyBrace.openLine, block.bodyBrace.closeLine)
          : []
        if (kind === 'struct') {
          structIndices.set(name, topSymbols.length)
        }
        topSymbols.push({
          name,
          kind,
          signature: compactSignature(sig),
          startLine: line1,
          endLine: block.endLine,
          exported: /^[A-Z]/.test(name),
          ...(ifaceChildren.length > 0 ? { children: ifaceChildren } : {}),
        })
        lineIndex = block.endLine
        continue
      }
    }

    // Receiver method: `func (r *Server) Run(...) ... { ... }`
    const recvMatch = rawLine.match(/^func\s*\(\s*(?:[A-Za-z_][A-Za-z0-9_]*\s+)?\*?\s*([A-Za-z_][A-Za-z0-9_]*)(?:\[[^\]]*\])?\s*\)\s*([A-Za-z_][A-Za-z0-9_]*)/)
    if (recvMatch?.[1] && recvMatch[2] && /^func\s*\(/.test(scrubbed)) {
      const receiverType = recvMatch[1]
      const methodName = recvMatch[2]
      const block = findBlockEndLine(scan, line1, 0)
      const sig = extractHeaderBeforeBrace(rawLines, line1, block.bodyBrace?.openLine ?? block.endLine).replace(/^func\s*/, '')
      const methodSym: OutlineSymbol = {
        name: `${receiverType}.${methodName}`,
        kind: 'method',
        signature: compactSignature(sig),
        startLine: line1,
        endLine: block.endLine,
        exported: /^[A-Z]/.test(methodName),
      }
      if (structIndices.has(receiverType)) {
        const list = structMethods.get(receiverType) ?? []
        list.push(methodSym)
        structMethods.set(receiverType, list)
      } else {
        topSymbols.push(methodSym)
      }
      lineIndex = block.endLine
      continue
    }

    // Plain function: `func Foo(...) ... { ... }`
    const fnMatch = scrubbed.match(/^func\s+([A-Za-z_][A-Za-z0-9_]*)/)
    if (fnMatch?.[1]) {
      const name = fnMatch[1]
      const block = findBlockEndLine(scan, line1, 0)
      const sig = extractHeaderBeforeBrace(rawLines, line1, block.bodyBrace?.openLine ?? block.endLine).replace(/^func\s+/, '')
      topSymbols.push({
        name,
        kind: 'function',
        signature: compactSignature(sig),
        startLine: line1,
        endLine: block.endLine,
        exported: /^[A-Z]/.test(name),
      })
      lineIndex = block.endLine
      continue
    }

    // `const` / `var`
    const constVarMatch = scrubbed.match(/^(const|var)\b/)
    if (constVarMatch?.[1]) {
      const kw = constVarMatch[1]
      const kind: OutlineSymbolKind = kw === 'const' ? 'constant' : 'variable'
      const paren = scan.parens.find(p => p.openLine === line1 && p.depth === 0)
      if (paren && /^(?:const|var)\s*\(/.test(scrubbed)) {
        for (let inner = line1; inner < paren.closeLine - 1; inner += 1) {
          const itemScrub = (scan.scrubbedLines[inner] ?? '').trim()
          const nameMatch = itemScrub.match(/^([A-Za-z_][A-Za-z0-9_]*)\b/)
          if (nameMatch?.[1] && nameMatch[1] !== 'iota') {
            topSymbols.push({
              name: nameMatch[1],
              kind,
              signature: compactSignature((rawLines[inner] ?? '').trim()),
              startLine: inner + 1,
              endLine: inner + 1,
              exported: /^[A-Z]/.test(nameMatch[1]),
            })
          }
        }
        lineIndex = paren.closeLine
        continue
      }
      const singleMatch = scrubbed.match(/^(?:const|var)\s+([A-Za-z_][A-Za-z0-9_]*)/)
      if (singleMatch?.[1]) {
        const brace = scan.braces.find(b => b.openLine === line1 && b.depth === 0)
        const endLine = brace ? brace.closeLine : line1
        topSymbols.push({
          name: singleMatch[1],
          kind,
          signature: compactSignature(rawLine.replace(/^(?:const|var)\s+/, '')),
          startLine: line1,
          endLine,
          exported: /^[A-Z]/.test(singleMatch[1]),
        })
        lineIndex = endLine
        continue
      }
    }

    lineIndex += 1
  }

  // Attach receiver methods onto structs defined in the same file (while preserving top-level order).
  for (const [structName, methods] of structMethods) {
    const idx = structIndices.get(structName)
    if (idx !== undefined && topSymbols[idx] !== undefined) {
      const existing = topSymbols[idx]!
      topSymbols[idx] = {
        ...existing,
        children: [...(existing.children ?? []), ...methods],
      }
    } else {
      topSymbols.push(...methods)
    }
  }

  return {
    ...(importCount > 0 ? {
      imports: {
        count: importCount,
        startLine: importStart,
        endLine: importEnd,
        modules: importModules,
      },
    } : {}),
    symbols: topSymbols,
  }
}

function extractGoInterfaceMethods(
  rawLines: readonly string[],
  scan: BraceScanIndex,
  openLine1: number,
  closeLine1: number,
): OutlineSymbol[] {
  const methods: OutlineSymbol[] = []
  for (let idx = openLine1; idx < closeLine1 - 1; idx += 1) {
    const scrubbed = (scan.scrubbedLines[idx] ?? '').trim()
    if (!scrubbed || scrubbed === '}') continue
    const m = scrubbed.match(/^([A-Za-z_][A-Za-z0-9_]*)\s*\(/)
    if (m?.[1]) {
      methods.push({
        name: m[1],
        kind: 'method',
        signature: compactSignature((rawLines[idx] ?? '').trim()),
        startLine: idx + 1,
        endLine: idx + 1,
        exported: /^[A-Z]/.test(m[1]),
      })
    }
  }
  return methods
}

// ============================================================================
// Rust Extractor
// ============================================================================

function extractRustOutline(
  rawLines: readonly string[],
  fullText: string,
): { readonly imports?: OutlineImportSummary; readonly symbols: readonly OutlineSymbol[] } {
  if (rawLines.length === 0) return { symbols: [] }
  const scan = scanBraceLanguage(fullText, rawLines.length, 'rust')
  const symbols: OutlineSymbol[] = []

  const importModules: string[] = []
  let importCount = 0
  let importStart = Number.POSITIVE_INFINITY
  let importEnd = 0

  let lineIndex = 0
  while (lineIndex < rawLines.length) {
    if (
      scan.lineStartInCommentOrString[lineIndex]
      || (scan.lineStartBraceDepth[lineIndex] ?? 0) !== 0
      || (scan.lineStartParenDepth[lineIndex] ?? 0) !== 0
    ) {
      lineIndex += 1
      continue
    }

    const scrubbed = (scan.scrubbedLines[lineIndex] ?? '').trim()
    if (!scrubbed) {
      lineIndex += 1
      continue
    }

    // Track Rust attributes `#[...]` (excluding inner `#![...]` crate attributes)
    let attrStartLine1: number | undefined
    let macroExported = false
    if (scrubbed.startsWith('#[')) {
      attrStartLine1 = lineIndex + 1
      while (lineIndex < rawLines.length) {
        const attrScrub = (scan.scrubbedLines[lineIndex] ?? '').trim()
        if (!attrScrub.startsWith('#[')) break
        if (attrScrub.includes('macro_export')) macroExported = true
        lineIndex += 1
      }
      if (lineIndex >= rawLines.length) break
    }

    const currentScrubbed = (scan.scrubbedLines[lineIndex] ?? '').trim()
    const declLine1 = lineIndex + 1
    const startLine = attrStartLine1 ?? declLine1

    // `use ...;` or `pub use ...;`
    const useMatch = currentScrubbed.match(/^(pub(?:\([^)]*\))?\s+)?use\s+(.+)$/)
    if (useMatch) {
      const endLine = findStatementEndLine(rawLines, scan, declLine1, 0)
      const isPubUse = Boolean(useMatch[1])
      const spec = rawLines.slice(declLine1 - 1, endLine).join(' ').replace(/^(?:pub(?:\([^)]*\))?\s+)?use\s+/, '').replace(/;\s*$/, '').trim()
      if (isPubUse) {
        symbols.push({
          name: 'pub use',
          kind: 'export',
          signature: compactSignature(spec),
          startLine: declLine1,
          endLine,
          exported: true,
        })
      } else {
        importCount += 1
        importStart = Math.min(importStart, declLine1)
        importEnd = Math.max(importEnd, endLine)
        const rootPath = spec.split('::{')[0]?.trim() ?? spec
        if (rootPath && !importModules.includes(rootPath)) importModules.push(rootPath)
      }
      lineIndex = endLine
      continue
    }

    const exported = /^pub\b/.test(currentScrubbed) || macroExported
    const strippedVis = currentScrubbed.replace(/^pub(?:\s*\([^)]*\))?\s+/, '')

    // `struct`
    const structMatch = strippedVis.match(/^struct\s+([A-Za-z_][A-Za-z0-9_]*)/)
    if (structMatch?.[1]) {
      const block = findBlockEndLine(scan, declLine1, 0, 12)
      const sig = extractHeaderBeforeBrace(rawLines, declLine1, block.bodyBrace?.openLine ?? block.endLine)
        .replace(/^pub(?:\s*\([^)]*\))?\s+/, '')
        .replace(/^struct\s+/, '')
      symbols.push({
        name: structMatch[1],
        kind: 'struct',
        signature: compactSignature(sig),
        startLine,
        endLine: block.endLine,
        exported,
      })
      lineIndex = block.endLine
      continue
    }

    // `enum`
    const enumMatch = strippedVis.match(/^enum\s+([A-Za-z_][A-Za-z0-9_]*)/)
    if (enumMatch?.[1]) {
      const block = findBlockEndLine(scan, declLine1, 0, 12)
      const sig = extractHeaderBeforeBrace(rawLines, declLine1, block.bodyBrace?.openLine ?? block.endLine)
        .replace(/^pub(?:\s*\([^)]*\))?\s+/, '')
        .replace(/^enum\s+/, '')
      symbols.push({
        name: enumMatch[1],
        kind: 'enum',
        signature: compactSignature(sig),
        startLine,
        endLine: block.endLine,
        exported,
      })
      lineIndex = block.endLine
      continue
    }

    // `trait`
    const traitMatch = strippedVis.match(/^(?:unsafe\s+)?trait\s+([A-Za-z_][A-Za-z0-9_]*)/)
    if (traitMatch?.[1]) {
      const block = findBlockEndLine(scan, declLine1, 0, 16)
      const sig = extractHeaderBeforeBrace(rawLines, declLine1, block.bodyBrace?.openLine ?? block.endLine)
        .replace(/^pub(?:\s*\([^)]*\))?\s+/, '')
        .replace(/^(?:unsafe\s+)?trait\s+/, '')
      const children = block.bodyBrace
        ? extractRustImplOrTraitMembers(rawLines, scan, block.bodyBrace.openLine, block.bodyBrace.closeLine, 1, true)
        : []
      symbols.push({
        name: traitMatch[1],
        kind: 'trait',
        signature: compactSignature(sig),
        startLine,
        endLine: block.endLine,
        exported,
        ...(children.length > 0 ? { children } : {}),
      })
      lineIndex = block.endLine
      continue
    }

    // `impl`
    if (/^(?:unsafe\s+)?impl\b/.test(strippedVis)) {
      const block = findBlockEndLine(scan, declLine1, 0, 16)
      const header = extractHeaderBeforeBrace(rawLines, declLine1, block.bodyBrace?.openLine ?? block.endLine)
        .replace(/^(?:unsafe\s+)?impl\s*/, '')
      const isTraitImpl = /\bfor\b/.test(header)
      const children = block.bodyBrace
        ? extractRustImplOrTraitMembers(rawLines, scan, block.bodyBrace.openLine, block.bodyBrace.closeLine, 1, isTraitImpl)
        : []
      symbols.push({
        name: compactSignature(header),
        kind: 'impl',
        signature: compactSignature(header),
        startLine,
        endLine: block.endLine,
        exported: isTraitImpl || children.some(c => c.exported),
        ...(children.length > 0 ? { children } : {}),
      })
      lineIndex = block.endLine
      continue
    }

    // `fn`
    const fnMatch = strippedVis.match(/^(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern(?:\s+"[^"]*")?\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)/)
    if (fnMatch?.[1]) {
      const block = findBlockEndLine(scan, declLine1, 0, 20)
      const sig = extractHeaderBeforeBrace(rawLines, declLine1, block.bodyBrace?.openLine ?? block.endLine)
        .replace(/^pub(?:\s*\([^)]*\))?\s+/, '')
        .replace(/\bfn\s+/, '')
      symbols.push({
        name: fnMatch[1],
        kind: 'function',
        signature: compactSignature(sig),
        startLine,
        endLine: block.endLine,
        exported,
      })
      lineIndex = block.endLine
      continue
    }

    // `type` alias
    const typeMatch = strippedVis.match(/^type\s+([A-Za-z_][A-Za-z0-9_]*)/)
    if (typeMatch?.[1]) {
      const endLine = findStatementEndLine(rawLines, scan, declLine1, 0)
      const sig = rawLines.slice(declLine1 - 1, endLine).join(' ')
        .replace(/^pub(?:\s*\([^)]*\))?\s+/, '')
        .replace(/^type\s+/, '')
        .replace(/;\s*$/, '')
      symbols.push({
        name: typeMatch[1],
        kind: 'type',
        signature: compactSignature(sig),
        startLine,
        endLine,
        exported,
      })
      lineIndex = endLine
      continue
    }

    // `const` / `static`
    const constMatch = strippedVis.match(/^(const|static)\s+(?:mut\s+)?([A-Za-z_][A-Za-z0-9_]*)/)
    if (constMatch?.[2]) {
      const endLine = findStatementEndLine(rawLines, scan, declLine1, 0)
      const sig = rawLines[declLine1 - 1]!.trim()
        .replace(/^pub(?:\s*\([^)]*\))?\s+/, '')
        .replace(/^(?:const|static)\s+/, '')
        .replace(/;\s*$/, '')
      symbols.push({
        name: constMatch[2],
        kind: 'constant',
        signature: compactSignature(sig),
        startLine,
        endLine,
        exported,
      })
      lineIndex = endLine
      continue
    }

    // `mod`
    const modMatch = strippedVis.match(/^mod\s+([A-Za-z_][A-Za-z0-9_]*)/)
    if (modMatch?.[1]) {
      const block = findBlockEndLine(scan, declLine1, 0, 5)
      symbols.push({
        name: modMatch[1],
        kind: 'module',
        signature: modMatch[1],
        startLine,
        endLine: block.endLine,
        exported,
      })
      lineIndex = block.endLine
      continue
    }

    // `macro_rules! name`
    const macroMatch = strippedVis.match(/^macro_rules!\s*([A-Za-z_][A-Za-z0-9_]*)/)
    if (macroMatch?.[1]) {
      const block = findBlockEndLine(scan, declLine1, 0, 5)
      symbols.push({
        name: `${macroMatch[1]}!`,
        kind: 'function',
        signature: `macro_rules! ${macroMatch[1]}`,
        startLine,
        endLine: block.endLine,
        exported,
      })
      lineIndex = block.endLine
      continue
    }

    lineIndex += 1
  }

  return {
    ...(importCount > 0 ? {
      imports: {
        count: importCount,
        startLine: importStart,
        endLine: importEnd,
        modules: importModules,
      },
    } : {}),
    symbols,
  }
}

function extractRustImplOrTraitMembers(
  rawLines: readonly string[],
  scan: BraceScanIndex,
  openLine1: number,
  closeLine1: number,
  braceDepth: number,
  defaultExported: boolean,
): OutlineSymbol[] {
  const members: OutlineSymbol[] = []
  let lineIdx = openLine1
  while (lineIdx < closeLine1 - 1) {
    if (
      scan.lineStartInCommentOrString[lineIdx]
      || (scan.lineStartBraceDepth[lineIdx] ?? 0) !== braceDepth
      || (scan.lineStartParenDepth[lineIdx] ?? 0) !== 0
    ) {
      lineIdx += 1
      continue
    }
    const scrubbed = (scan.scrubbedLines[lineIdx] ?? '').trim()
    if (!scrubbed || scrubbed === '}') {
      lineIdx += 1
      continue
    }

    let attrStart: number | undefined
    if (scrubbed.startsWith('#[')) {
      attrStart = lineIdx + 1
      while (lineIdx < closeLine1 - 1 && (scan.scrubbedLines[lineIdx] ?? '').trim().startsWith('#[')) {
        lineIdx += 1
      }
      if (lineIdx >= closeLine1 - 1) break
    }

    const memberScrub = (scan.scrubbedLines[lineIdx] ?? '').trim()
    const line1 = lineIdx + 1
    const startLine = attrStart ?? line1
    const isPub = defaultExported || /^pub\b/.test(memberScrub)
    const stripped = memberScrub.replace(/^pub(?:\s*\([^)]*\))?\s+/, '')

    const fnMatch = stripped.match(/^(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?(?:extern(?:\s+"[^"]*")?\s+)?fn\s+([A-Za-z_][A-Za-z0-9_]*)/)
    if (fnMatch?.[1]) {
      const block = findBlockEndLine(scan, line1, braceDepth, 16)
      const endLine = Math.min(closeLine1 - 1, Math.max(line1, block.endLine))
      const sig = extractHeaderBeforeBrace(rawLines, line1, block.bodyBrace?.openLine ?? endLine)
        .replace(/^pub(?:\s*\([^)]*\))?\s+/, '')
        .replace(/\bfn\s+/, '')
      members.push({
        name: fnMatch[1],
        kind: 'method',
        signature: compactSignature(sig),
        startLine,
        endLine,
        exported: isPub,
      })
      lineIdx = endLine
      continue
    }

    lineIdx += 1
  }
  return members
}

// ============================================================================
// Shared Statement / Signature Helpers & Batch File Discovery
// ============================================================================

function findStatementEndLine(
  rawLines: readonly string[],
  scan: BraceScanIndex,
  fromLine1: number,
  baseBraceDepth: number,
): number {
  let endLine = fromLine1
  const limit = Math.min(rawLines.length, fromLine1 + 500)
  for (let ln = fromLine1; ln <= limit; ln += 1) {
    const lineScrub = (scan.scrubbedLines[ln - 1] ?? '').trim()
    if (ln > fromLine1 && lineScrub === '') {
      return ln - 1
    }
    endLine = ln
    const brace = scan.braces.find(b => b.openLine === ln && b.depth === baseBraceDepth)
    if (brace !== undefined && brace.closeLine > endLine) {
      endLine = brace.closeLine
      ln = endLine
    }
    const paren = scan.parens.find(p => p.openLine === ln && p.depth === 0)
    if (paren !== undefined && paren.closeLine > endLine) {
      endLine = paren.closeLine
      ln = endLine
    }
    const scrubbedAtEnd = (scan.scrubbedLines[endLine - 1] ?? '').trim()
    if (scrubbedAtEnd.endsWith(';') || !/[=,|+&(]$|=>$/.test(scrubbedAtEnd)) {
      return endLine
    }
  }
  return endLine
}

function findTsTypeAliasEndLine(rawLines: readonly string[], scan: BraceScanIndex, fromLine1: number): number {
  let endLine = findStatementEndLine(rawLines, scan, fromLine1, 0)
  while (endLine < rawLines.length) {
    const currentScrub = (scan.scrubbedLines[endLine - 1] ?? '').trim()
    if (currentScrub.endsWith(';')) break
    const nextScrub = (scan.scrubbedLines[endLine] ?? '').trim()
    if (nextScrub.startsWith('|') || nextScrub.startsWith('&') || nextScrub.startsWith('?')) {
      const brace = scan.braces.find(b => b.openLine === endLine + 1 && b.depth === 0)
      endLine = brace ? brace.closeLine : endLine + 1
      continue
    }
    break
  }
  return endLine
}

function extractHeaderBeforeBrace(rawLines: readonly string[], startLine1: number, braceLine1: number): string {
  const slice = rawLines.slice(startLine1 - 1, Math.max(startLine1, braceLine1))
  const joined = slice
    .map(line => line.replace(/\/\/.*$/, '').trim())
    .join(' ')
  const braceIdx = joined.indexOf('{')
  const header = braceIdx >= 0 ? joined.slice(0, braceIdx) : joined
  return header.replace(/;\s*$/, '').trim()
}

function compactSignature(raw: string): string {
  const collapsed = raw.replace(/\s+/g, ' ').trim()
  return collapsed.length <= MAX_SIGNATURE_CHARS ? collapsed : `${collapsed.slice(0, MAX_SIGNATURE_CHARS - 1)}…`
}

interface OutlineTargetFile {
  readonly absolutePath: string
  readonly displayPath: string
}

async function collectOutlineTargets(
  filePath: string,
  maxFiles: number,
  paths: WorkspacePathResolver,
): Promise<{ readonly files: readonly OutlineTargetFile[]; readonly truncated: boolean }> {
  if (/[*?[{]/.test(filePath)) {
    validateGlobPattern(filePath)
    const root = await paths.resolve('.')
    const matcher = await GitignoreMatcher.create(root, paths)
    const glob = new Bun.Glob(filePath)
    const matches: OutlineTargetFile[] = []
    for await (const rel of glob.scan({ cwd: root, dot: false, followSymlinks: false, onlyFiles: true })) {
      const posixRel = rel.replaceAll('\\', '/')
      if (posixRel.split('/').includes('.git') || posixRel.split('/').includes('node_modules')) continue
      if (!SUPPORTED_EXTENSIONS.has(extname(posixRel.toLowerCase()))) continue
      let resolved: string
      try {
        resolved = await paths.resolve(join(root, rel))
      } catch (error) {
        if (error instanceof WorkspacePathError) continue
        throw error
      }
      if (matcher && await matcher.isIgnored(resolved, false)) continue
      const displayPath = (await paths.relative(resolved)).replaceAll('\\', '/')
      matches.push({ absolutePath: resolved, displayPath })
    }
    matches.sort((a, b) => a.displayPath.localeCompare(b.displayPath))
    return {
      files: matches.slice(0, maxFiles),
      truncated: matches.length > maxFiles,
    }
  }

  const resolved = await paths.resolveReadable(filePath)
  let info
  try {
    info = await stat(resolved)
  } catch (error) {
    if (isNotFound(error)) {
      throw new ValidationError('file_path', 'must refer to an existing file, directory, or glob pattern', filePath)
    }
    throw error
  }

  if (info.isFile()) {
    let displayPath = filePath.replaceAll('\\', '/')
    try {
      displayPath = (await paths.relative(resolved)).replaceAll('\\', '/')
    } catch {
      // Read-only roots outside workspace keep caller-visible path.
    }
    return { files: [{ absolutePath: resolved, displayPath }], truncated: false }
  }

  if (!info.isDirectory()) {
    throw new ValidationError('file_path', 'must refer to a regular file or directory', filePath)
  }

  const matcher = await GitignoreMatcher.create(resolved, paths)
  const collected: OutlineTargetFile[] = []
  let truncated = false

  const walk = async (dir: string): Promise<void> => {
    const entries = await readdir(dir, { withFileTypes: true })
    entries.sort((a, b) => a.name.localeCompare(b.name))
    for (const entry of entries) {
      if (entry.name.startsWith('.') || entry.name === 'node_modules' || entry.isSymbolicLink()) continue
      const candidate = join(dir, entry.name)
      const isDir = entry.isDirectory()
      if (matcher && await matcher.isIgnored(candidate, isDir)) continue
      let checked: string
      try {
        checked = await paths.resolve(candidate)
      } catch (error) {
        if (error instanceof WorkspacePathError) continue
        throw error
      }
      if (isDir) {
        await walk(checked)
        if (truncated) return
      } else if (entry.isFile() && SUPPORTED_EXTENSIONS.has(extname(entry.name.toLowerCase()))) {
        if (collected.length >= maxFiles) {
          truncated = true
          return
        }
        const displayPath = (await paths.relative(checked)).replaceAll('\\', '/')
        collected.push({ absolutePath: checked, displayPath })
      }
    }
  }

  await walk(resolved)
  return { files: collected, truncated }
}

function validateGlobPattern(pattern: string): void {
  const hasTraversal = pattern.split(/[\\/]/).includes('..')
  if (pattern.includes('\0') || pattern.startsWith('/') || pattern.startsWith('\\') || hasTraversal) {
    throw new ValidationError('file_path', 'glob pattern must be workspace-relative and must not contain parent traversal', pattern)
  }
}

function isNotFound(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}
