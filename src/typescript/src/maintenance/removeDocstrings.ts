// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { errorMessage, parseFileArguments, readUtf8Text, writeTextFile, type WriteOptions } from './io.js'
import {
  analyzePythonSource,
  applySourceEdits,
  barePlainStringTokens,
  normalizeStrippedSource,
  type PythonStatement,
  type SourceEdit,
} from './pythonSource.js'

export interface DocstringRemovalResult {
  changed: boolean
  commentsRemoved: number
  docstringsRemoved: number
  emptySuitesPreserved: number
  text: string
  valid: boolean
}

export interface DocstringRemovalFileResult extends DocstringRemovalResult {
  path: string
}

/** Remove Python comments and docstrings without needing a Python runtime. */
export function removeDocstringsAndComments(source: string): DocstringRemovalResult {
  const analysis = analyzePythonSource(source)
  if (!analysis.valid) {
    return {
      changed: false,
      commentsRemoved: 0,
      docstringsRemoved: 0,
      emptySuitesPreserved: 0,
      text: source,
      valid: false,
    }
  }

  const removable = new Set<PythonStatement>()
  for (const context of analysis.contexts) {
    if (barePlainStringTokens(context.statement) === undefined) continue
    const removeAsAttribute = context.scope.kind === 'class' || context.scope.kind === 'module'
    const removeAsDocstring = context.firstInScope
      && (context.scope.kind === 'class' || context.scope.kind === 'function' || context.scope.kind === 'module')
    if (removeAsAttribute || removeAsDocstring) removable.add(context.statement)
  }

  const preserveWithPass = new Set<PythonStatement>()
  for (const scope of analysis.scopes) {
    if (scope.kind !== 'class' && scope.kind !== 'function') continue
    if (scope.statements.length === 0 || !scope.statements.every(statement => removable.has(statement))) continue
    const first = scope.statements[0]
    if (first !== undefined) preserveWithPass.add(first)
  }

  const docstringEdits: SourceEdit[] = []
  for (const context of analysis.contexts) {
    if (!removable.has(context.statement)) continue
    docstringEdits.push({
      end: context.statement.end,
      replacement: preserveWithPass.has(context.statement) ? 'pass' : '',
      start: context.statement.start,
    })
  }

  const commentEdits = analysis.comments
    .filter(comment => !docstringEdits.some(edit => comment.start >= edit.start && comment.end <= edit.end))
    .map(comment => ({ end: comment.end, replacement: '', start: comment.start }))
  const edits = [...docstringEdits, ...commentEdits]
  if (edits.length === 0) {
    return {
      changed: false,
      commentsRemoved: 0,
      docstringsRemoved: 0,
      emptySuitesPreserved: 0,
      text: source,
      valid: true,
    }
  }

  const stripped = normalizeStrippedSource(applySourceEdits(source, edits), source.endsWith('\n'))
  const valid = analyzePythonSource(stripped).valid
  if (!valid) {
    return {
      changed: false,
      commentsRemoved: 0,
      docstringsRemoved: 0,
      emptySuitesPreserved: 0,
      text: source,
      valid: false,
    }
  }
  return {
    changed: stripped !== source,
    commentsRemoved: analysis.comments.length,
    docstringsRemoved: removable.size,
    emptySuitesPreserved: preserveWithPass.size,
    text: stripped,
    valid: true,
  }
}

/** Transform explicitly named Python files with dry-run and atomic-write controls. */
export async function removeDocstringsFromFiles(
  paths: readonly string[],
  options: WriteOptions = {},
): Promise<DocstringRemovalFileResult[]> {
  const results: DocstringRemovalFileResult[] = []
  for (const path of paths) {
    const source = await readUtf8Text(path)
    const result = removeDocstringsAndComments(source)
    if (result.changed) await writeTextFile(path, result.text, options)
    results.push({ ...result, path })
  }
  return results
}

export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  try {
    const parsed = parseFileArguments(
      args,
      'Usage: bun removeDocstrings.ts [--dry-run] [--no-atomic] <file1.py> [file2.py ...]',
    )
    if (parsed === undefined) return 0
    if (parsed.paths.length === 0) throw new Error('At least one Python source path is required.')

    const results = await removeDocstringsFromFiles(parsed.paths, parsed)
    for (const result of results) {
      if (!result.valid) console.log(`SKIP (lexically invalid Python): ${result.path}`)
      else console.log(`${result.changed ? (parsed.dryRun ? 'WOULD PROCESS' : 'PROCESSED') : 'NO CHANGES'}: ${result.path}`)
    }
    return 0
  } catch (error) {
    console.error(`remove-docstrings: ${errorMessage(error)}`)
    return 1
  }
}

if (import.meta.main) {
  process.exitCode = await main()
}
