// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { afterEach, describe, expect, test } from 'bun:test'

import {
  discoverInstructionFiles,
  instructionFileUpdateLayer,
  readInstructionDigests,
} from '../src/runtime/instructionFreshness.js'

const scratchDirs: string[] = []

afterEach(() => {
  for (const dir of scratchDirs.splice(0)) {
    rmSync(dir, { force: true, recursive: true })
  }
})

function scratch(): string {
  const dir = mkdtempSync(join(tmpdir(), 'xerxes-freshness-'))
  scratchDirs.push(dir)
  return dir
}

describe('discoverInstructionFiles', () => {
  test('finds nearest file per name walking up, including .local.md overlays', async () => {
    const root = scratch()
    const sub = join(root, 'pkg', 'deep')
    mkdirSync(sub, { recursive: true })
    writeFileSync(join(root, 'AGENTS.md'), 'root agents')
    writeFileSync(join(root, 'pkg', 'AGENTS.md'), 'pkg agents')
    writeFileSync(join(root, 'AGENTS.local.md'), 'personal overlay')

    const found = await discoverInstructionFiles(sub)
    const byName = new Map(found.map(file => [file.name, file]))
    // Nearest AGENTS.md wins (pkg), and the overlay is a separate candidate.
    expect(byName.get('AGENTS.md')?.path).toBe(join(root, 'pkg', 'AGENTS.md'))
    expect(byName.get('AGENTS.local.md')?.path).toBe(join(root, 'AGENTS.local.md'))
  })
})

describe('instructionFileUpdateLayer', () => {
  test('first contact records the baseline silently', async () => {
    const root = scratch()
    writeFileSync(join(root, 'AGENTS.md'), 'v1')
    const metadata: Record<string, unknown> = {}

    const layer = await instructionFileUpdateLayer(root, metadata)
    expect(layer).toBe('')
    expect(Object.keys(readInstructionDigests(metadata))).toEqual([join(root, 'AGENTS.md')])
  })

  test('an edited file announces itself with fresh scanned content', async () => {
    const root = scratch()
    const path = join(root, 'AGENTS.md')
    writeFileSync(path, 'v1 instructions')
    const metadata: Record<string, unknown> = {}
    await instructionFileUpdateLayer(root, metadata)

    writeFileSync(path, 'v2 instructions: always run bun run verify')
    const layer = await instructionFileUpdateLayer(root, metadata)

    expect(layer).toContain('Updated project instructions')
    expect(layer).toContain(`Updated: ${path}`)
    expect(layer).toContain('v2 instructions: always run bun run verify')
    expect(layer).not.toContain('v1 instructions')
    // A third turn with no further edits stays silent.
    expect(await instructionFileUpdateLayer(root, metadata)).toBe('')
  })

  test('a deleted file announces removal; a new file announces addition', async () => {
    const root = scratch()
    const path = join(root, 'AGENTS.md')
    writeFileSync(path, 'v1')
    const metadata: Record<string, unknown> = {}
    await instructionFileUpdateLayer(root, metadata)

    rmSync(path)
    const removed = await instructionFileUpdateLayer(root, metadata)
    expect(removed).toContain(`Removed: ${path}`)
    expect(removed).toContain('no longer apply')

    writeFileSync(join(root, 'CLAUDE.md'), 'claude rules')
    const added = await instructionFileUpdateLayer(root, metadata)
    expect(added).toContain(`Added: ${join(root, 'CLAUDE.md')}`)
    expect(added).toContain('claude rules')
  })

  test('malformed stored digests read as an empty baseline', async () => {
    const root = scratch()
    writeFileSync(join(root, 'AGENTS.md'), 'v1')
    const metadata: Record<string, unknown> = { instruction_file_digests: 'garbage' }
    expect(await instructionFileUpdateLayer(root, metadata)).toBe('')
    expect(Object.keys(readInstructionDigests(metadata))).toHaveLength(1)
  })
})
