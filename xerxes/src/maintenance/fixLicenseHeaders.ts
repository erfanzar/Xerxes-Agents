// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readdir } from 'node:fs/promises'
import { extname, join, relative, resolve } from 'node:path'

import { errorMessage, readUtf8Text, writeTextFile, type WriteOptions } from './io.js'

export const CANONICAL_COPYRIGHT_LINE = '# Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).'
export const LICENSE_URL_LINE = '#     https://www.apache.org/licenses/LICENSE-2.0'
export const LICENSE_TRAILER_MARKER = 'Unless required by applicable law'
export const LICENSE_TRAILER = [
  '#',
  '# Unless required by applicable law or agreed to in writing, software',
  '# distributed under the License is distributed on an "AS IS" BASIS,',
  '# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.',
  '# See the License for the specific language governing permissions and',
  '# limitations under the License.',
].join('\n')

const COPYRIGHT = /^# Copyright 2026 (?:The )?Xerxes-Agents Author(?: @erfanzar \(Erfan Zare Chavoshi\))?\.?$/gmu
const EXTENSIONS = new Set(['.py', '.sh', '.yml', '.yaml'])
const FILE_NAMES = new Set(['Dockerfile'])
const SKIPPED_DIRECTORIES = new Set([
  '.git',
  '.pytest_cache',
  '.ruff_cache',
  '.venv',
  '__pycache__',
  'build',
  'dist',
  'node_modules',
])

export interface LicenseFix {
  fixes: readonly ('copyright' | 'trailer')[]
  path: string
}

export interface LicenseHeaderRun extends LicenseFix {
  changed: boolean
}

/** Normalize Xerxes Apache headers in source files below one explicit root directory. */
export async function fixLicenseHeaders(root: string, options: WriteOptions = {}): Promise<LicenseHeaderRun[]> {
  const normalizedRoot = resolve(root)
  const changes: LicenseHeaderRun[] = []

  for (const path of await targetFiles(normalizedRoot)) {
    let source: string
    try {
      source = await readUtf8Text(path)
    } catch {
      continue
    }
    const fixed = fixLicenseText(source)
    if (fixed.fixes.length === 0 || fixed.text === source) continue
    await writeTextFile(path, fixed.text, options)
    changes.push({ changed: options.dryRun !== true, fixes: fixed.fixes, path })
  }

  return changes.sort((left, right) => left.path.localeCompare(right.path))
}

/** Return the normalized text and the exact repairs that were necessary. */
export function fixLicenseText(source: string): { fixes: readonly ('copyright' | 'trailer')[]; text: string } {
  const fixes: ('copyright' | 'trailer')[] = []
  let text = source.replace(COPYRIGHT, CANONICAL_COPYRIGHT_LINE)
  if (text !== source) fixes.push('copyright')

  if (text.includes(LICENSE_URL_LINE) && !text.includes(LICENSE_TRAILER_MARKER)) {
    const urlOffset = text.indexOf(LICENSE_URL_LINE)
    const newline = text.indexOf('\n', urlOffset)
    const insertion = newline === -1 ? text.length : newline + 1
    const separator = newline === -1 ? '\n' : ''
    text = `${text.slice(0, insertion)}${separator}${LICENSE_TRAILER}\n${text.slice(insertion)}`
    fixes.push('trailer')
  }

  return { fixes, text }
}

export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  try {
    const parsed = parseArguments(args)
    if (parsed === undefined) return 0
    const changes = await fixLicenseHeaders(parsed.root, { atomic: parsed.atomic, dryRun: parsed.dryRun })
    console.log(`${parsed.dryRun ? 'Files that would change' : 'Files changed'}: ${changes.length}`)
    for (const change of changes) {
      const path = relative(parsed.root, change.path) || change.path
      console.log(`  ${path}  [${change.fixes.join(', ')}]`)
    }
    return 0
  } catch (error) {
    console.error(`fix-license-headers: ${errorMessage(error)}`)
    return 1
  }
}

if (import.meta.main) {
  process.exitCode = await main()
}

async function targetFiles(root: string): Promise<string[]> {
  const files: string[] = []
  const directories = [root]

  while (directories.length > 0) {
    const directory = directories.pop()
    if (directory === undefined) continue
    const entries = await readdir(directory, { withFileTypes: true })
    for (const entry of entries.sort((left, right) => left.name.localeCompare(right.name))) {
      const path = join(directory, entry.name)
      if (entry.isDirectory()) {
        if (!SKIPPED_DIRECTORIES.has(entry.name)) directories.push(path)
        continue
      }
      if (entry.isFile() && shouldVisit(entry.name)) files.push(path)
    }
  }

  return files
}

function parseArguments(args: readonly string[]): { atomic: boolean; dryRun: boolean; root: string } | undefined {
  let atomic = true
  let dryRun = false
  let root = resolve(process.cwd())

  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === undefined) continue
    if (argument === '--help' || argument === '-h') {
      console.log('Usage: bun fixLicenseHeaders.ts [--root <directory>] [--dry-run] [--no-atomic]')
      return undefined
    }
    if (argument === '--dry-run') {
      dryRun = true
      continue
    }
    if (argument === '--no-atomic') {
      atomic = false
      continue
    }
    if (argument === '--root') {
      const candidate = args[index + 1]
      if (candidate === undefined) throw new Error('--root requires a directory path.')
      root = resolve(candidate)
      index += 1
      continue
    }
    throw new Error(`Unknown option: ${argument}`)
  }

  return { atomic, dryRun, root }
}

function shouldVisit(fileName: string): boolean {
  return FILE_NAMES.has(fileName) || EXTENSIONS.has(extname(fileName))
}
