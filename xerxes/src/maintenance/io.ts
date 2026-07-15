// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { chmod, readFile, rename, rm, stat, writeFile } from 'node:fs/promises'
import { basename, dirname, join, resolve } from 'node:path'
import { randomUUID } from 'node:crypto'

export interface WriteOptions {
  atomic?: boolean
  dryRun?: boolean
}

export interface ParsedFileArguments {
  atomic: boolean
  dryRun: boolean
  paths: string[]
}

export function resolvePaths(paths: readonly string[]): string[] {
  return paths.map(path => resolve(path))
}

export async function readUtf8Text(path: string): Promise<string> {
  return new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(await readFile(path))
}

export async function writeTextFile(path: string, content: string, options: WriteOptions = {}): Promise<void> {
  if (options.dryRun === true) return
  if (options.atomic === false) {
    await writeFile(path, content, 'utf8')
    return
  }

  const source = await stat(path)
  const mode = source.mode & 0o777
  const temporary = join(dirname(path), `.${basename(path)}.${process.pid}.${randomUUID()}.tmp`)
  try {
    await writeFile(temporary, content, { encoding: 'utf8', mode })
    await chmod(temporary, mode)
    await rename(temporary, path)
  } catch (error) {
    await rm(temporary, { force: true })
    throw error
  }
}

export function parseFileArguments(args: readonly string[], usage: string): ParsedFileArguments | undefined {
  let atomic = true
  let dryRun = false
  const paths: string[] = []
  let positional = false

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === undefined) continue
    if (argument === '--') {
      positional = true
      continue
    }
    if (!positional && (argument === '--help' || argument === '-h')) {
      console.log(usage)
      return undefined
    }
    if (!positional && argument === '--dry-run') {
      dryRun = true
      continue
    }
    if (!positional && argument === '--no-atomic') {
      atomic = false
      continue
    }
    if (!positional && argument.startsWith('-')) {
      throw new Error(`Unknown option: ${argument}`)
    }
    paths.push(argument)
  }

  return { atomic, dryRun, paths: resolvePaths(paths) }
}

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
