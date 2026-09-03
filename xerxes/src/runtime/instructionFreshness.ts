// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Instruction-file freshness (borrowed from DSH's reconciliation idea,
 * delivered through Xerxes' volatile-layer machinery instead of chat
 * messages).
 *
 * Bootstrap injects XERXES.md / AGENTS.md / CLAUDE.md content into the stable
 * system band once per session; without freshness, an edit during a long
 * session silently leaves the model working from stale instructions until a
 * reload. This module re-reads the discovered files at turn assembly, compares
 * SHA-256 digests against the values recorded in session metadata, and renders
 * exactly one volatile `instruction_updates` layer when something changed —
 * carrying the new content inline so the update is authoritative, not a
 * pointer the model must chase.
 *
 * Change detection runs at every turn rather than only on Xerxes' own tool
 * touches (DSH's rule): external editor changes matter just as much, and a
 * handful of bounded reads per turn costs microseconds.
 */

import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { scanContextContent } from '../security/promptScanner.js'
import { INSTRUCTION_FILE_CANDIDATES, MAX_BOOTSTRAP_INSTRUCTION_FILE_BYTES } from './bootstrap.js'

/** Session-metadata key for the recorded path → SHA-256 map. */
const DIGESTS_KEY = 'instruction_file_digests'

/** Per-update content ceiling inside the volatile layer. */
const MAX_UPDATE_CONTENT_BYTES = MAX_BOOTSTRAP_INSTRUCTION_FILE_BYTES
/** Aggregate ceiling for one freshness layer. */
const MAX_UPDATE_LAYER_BYTES = 32 * 1024
/** Walk-up depth, matching bootstrap's project instruction search. */
const WALK_UP_DEPTH = 10

export interface InstructionFileSnapshot {
  readonly digest: string
  readonly name: string
  readonly path: string
}

function sha256(text: string): string {
  return createHash('sha256').update(text, 'utf8').digest('hex')
}

/** Read the recorded digest map; unknown shapes read as empty. */
export function readInstructionDigests(
  metadata: Readonly<Record<string, unknown>>,
): Record<string, string> {
  const raw = metadata[DIGESTS_KEY]
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return {}
  const out: Record<string, string> = {}
  for (const [path, digest] of Object.entries(raw)) {
    if (typeof digest === 'string' && digest) out[path] = digest
  }
  return out
}

/**
 * Discover instruction files exactly the way bootstrap does: each candidate
 * name resolves to the nearest file walking up from `cwd`, so a repo-root
 * AGENTS.md still governs a session started in a subdirectory, and a closer
 * file shadows a farther one of the same name.
 */
export async function discoverInstructionFiles(cwd: string): Promise<InstructionFileSnapshot[]> {
  const found: InstructionFileSnapshot[] = []
  for (const name of INSTRUCTION_FILE_CANDIDATES) {
    let current = cwd
    for (let depth = 0; depth < WALK_UP_DEPTH; depth += 1) {
      const candidate = join(current, name)
      const text = await readFile(candidate, 'utf8').catch(() => undefined)
      if (text !== undefined && text.trim()) {
        found.push({ digest: sha256(text), name, path: candidate })
        break
      }
      const parent = dirname(current)
      if (parent === current) break
      current = parent
    }
  }
  return found
}

/**
 * Diff the live instruction files against the recorded digests and, when they
 * drifted, render the volatile layer announcing the change with fresh content.
 * The recorded map is updated in `metadata` either way so the next turn diffs
 * against the newest baseline.
 *
 * The first turn after bootstrap records the baseline silently — the stable
 * band already carries the same content, so announcing it would be noise.
 */
export async function instructionFileUpdateLayer(
  cwd: string,
  metadata: Record<string, unknown>,
): Promise<string> {
  const current = await discoverInstructionFiles(cwd)
  // Baseline existence is the KEY's presence with a well-formed map, not a
  // non-empty one: a turn after every file was deleted legitimately records
  // an empty baseline, and later additions must still be announced; malformed
  // stored values read as never-recorded.
  const rawBaseline = metadata[DIGESTS_KEY]
  const baselineExists =
    rawBaseline !== undefined &&
    typeof rawBaseline === 'object' &&
    rawBaseline !== null &&
    !Array.isArray(rawBaseline)
  const prior = readInstructionDigests(metadata)
  const next: Record<string, string> = {}
  for (const file of current) next[file.path] = file.digest
  metadata[DIGESTS_KEY] = next

  const added: InstructionFileSnapshot[] = []
  const changed: InstructionFileSnapshot[] = []
  const removed: string[] = []
  for (const file of current) {
    const old = prior[file.path]
    if (old === undefined) added.push(file)
    else if (old !== file.digest) changed.push(file)
  }
  for (const path of Object.keys(prior)) {
    if (next[path] === undefined) removed.push(path)
  }

  // First contact: bootstrap already injected this exact content into the
  // stable band, so only establish the baseline.
  if (!baselineExists || (!added.length && !changed.length && !removed.length)) {
    return ''
  }

  const sections: string[] = [
    '# Updated project instructions',
    'Instruction files changed on disk since this session loaded them. The stable project-context section above predates this change; the content below is current and takes precedence.',
  ]
  let budget = MAX_UPDATE_LAYER_BYTES

  const appendFile = async (file: InstructionFileSnapshot, verb: 'Added' | 'Updated') => {
    if (budget <= 0) return
    const raw = await readFile(file.path, 'utf8').catch(() => undefined)
    if (raw === undefined) return
    const scanned = scanContextContent(raw.slice(0, MAX_UPDATE_CONTENT_BYTES), file.path)
    const header = `\n## ${verb}: ${file.path}\n`
    const take = scanned.slice(0, budget)
    sections.push(header + take)
    budget -= Buffer.byteLength(header + take, 'utf8')
  }

  for (const file of changed) await appendFile(file, 'Updated')
  for (const file of added) await appendFile(file, 'Added')
  for (const path of removed) {
    sections.push(`\n## Removed: ${path}\nThe instructions from this file no longer apply.`)
  }
  return sections.join('\n')
}
