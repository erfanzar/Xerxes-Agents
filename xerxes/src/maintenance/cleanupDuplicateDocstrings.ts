// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { errorMessage, parseFileArguments, readUtf8Text, writeTextFile, type WriteOptions } from './io.js'
import { analyzePythonSource, applySourceEdits, barePlainStringTokens, lineRemovalRange } from './pythonSource.js'

export interface DuplicateDocstringResult {
  changed: boolean
  removed: number
  text: string
  valid: boolean
}

export interface DuplicateDocstringFileResult extends DuplicateDocstringResult {
  path: string
}

/** Remove non-docstring standalone literals from module and class bodies. */
export function cleanupDuplicateDocstrings(source: string): DuplicateDocstringResult {
  const analysis = analyzePythonSource(source)
  if (!analysis.valid) return { changed: false, removed: 0, text: source, valid: false }

  const removals = analysis.contexts.flatMap(context => {
    if (context.scope.kind !== 'class' && context.scope.kind !== 'module') return []
    if (context.firstInScope || barePlainStringTokens(context.statement) === undefined) return []
    return [lineRemovalRange(source, context.statement.start, context.statement.end)]
  })
  const text = applySourceEdits(source, removals)
  const valid = analyzePythonSource(text).valid
  if (!valid) return { changed: false, removed: 0, text: source, valid: false }
  return { changed: text !== source, removed: removals.length, text, valid: true }
}

/** Transform explicitly named Python files with dry-run and atomic-write controls. */
export async function cleanupDuplicateDocstringFiles(
  paths: readonly string[],
  options: WriteOptions = {},
): Promise<DuplicateDocstringFileResult[]> {
  const results: DuplicateDocstringFileResult[] = []
  for (const path of paths) {
    const source = await readUtf8Text(path)
    const result = cleanupDuplicateDocstrings(source)
    if (result.changed) await writeTextFile(path, result.text, options)
    results.push({ ...result, path })
  }
  return results
}

export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  try {
    const parsed = parseFileArguments(
      args,
      'Usage: bun cleanupDuplicateDocstrings.ts [--dry-run] [--no-atomic] <file1.py> [file2.py ...]',
    )
    if (parsed === undefined) return 0
    if (parsed.paths.length === 0) throw new Error('At least one Python source path is required.')

    const results = await cleanupDuplicateDocstringFiles(parsed.paths, parsed)
    let total = 0
    for (const result of results) {
      if (!result.valid) {
        console.log(`  SKIP (lexically invalid Python): ${result.path}`)
        continue
      }
      if (!result.changed) continue
      total += 1
      console.log(`  ${parsed.dryRun ? 'WOULD CLEAN' : 'CLEANED'}: ${result.path}`)
    }
    console.log(`\nTotal files ${parsed.dryRun ? 'that would be cleaned' : 'cleaned'}: ${total}`)
    return 0
  } catch (error) {
    console.error(`cleanup-duplicate-docstrings: ${errorMessage(error)}`)
    return 1
  }
}

if (import.meta.main) {
  process.exitCode = await main()
}
