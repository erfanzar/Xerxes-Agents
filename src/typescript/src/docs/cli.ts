// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { relative } from 'node:path'

import { buildBunDocsSite } from './siteBuilder.js'

/** Run the native Bun documentation builder. */
export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  try {
    const options = parseArguments(args)
    if (options === undefined) return 0
    const result = await buildBunDocsSite(options)
    const changed = result.changes.filter(change => change.action !== 'unchanged')
    const verb = result.dryRun ? 'Would update' : 'Updated'
    const summary = [
      `${verb} ${result.documents.length} static documentation pages`,
      `and ${result.typeScriptApiModules} TypeScript API modules.`,
    ].join(' ')
    console.log(summary)
    for (const change of changed) {
      console.log(`  ${change.action.toUpperCase()}: ${relative(result.outputDirectory, change.path) || change.path}`)
    }
    for (const diagnostic of result.diagnostics) {
      const line = diagnostic.line === undefined ? '' : `:${diagnostic.line}`
      console.warn(`  WARNING ${diagnostic.source}${line}: ${diagnostic.message}`)
    }
    return 0
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    console.error(`bun-docs: ${message}`)
    return 1
  }
}

if (import.meta.main) process.exitCode = await main()

function parseArguments(args: readonly string[]): Parameters<typeof buildBunDocsSite>[0] | undefined {
  type MutableBuildOptions = {
    -readonly [Key in keyof NonNullable<Parameters<typeof buildBunDocsSite>[0]>]: NonNullable<
      Parameters<typeof buildBunDocsSite>[0]
    >[Key]
  }
  const options: MutableBuildOptions = {}
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === '--help' || argument === '-h') {
      console.log([
        'Usage: bun docs/cli.ts [--config <path>] [--source <directory>] [--output <directory>]',
        '                         [--api-source <directory>] [--api-package <name>] [--title <title>] [--dry-run]',
        '',
        'The native builder reads Markdown sources and emits static HTML with a generated TypeScript API reference.',
      ].join('\n'))
      return undefined
    }
    if (argument === '--dry-run') {
      options.dryRun = true
      continue
    }
    const value = args[index + 1]
    if (value === undefined || value.startsWith('--')) throw new Error(`Missing value for ${argument}`)
    index += 1
    if (argument === '--config') options.configPath = value
    else if (argument === '--source') options.sourceDirectory = value
    else if (argument === '--output') options.outputDirectory = value
    else if (argument === '--api-source') options.apiSourceDirectory = value
    else if (argument === '--api-package') options.apiPackageName = value
    else if (argument === '--title') options.title = value
    else throw new Error(`Unknown argument: ${argument}`)
  }
  return options
}
