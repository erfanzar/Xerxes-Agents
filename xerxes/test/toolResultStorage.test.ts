// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import {
  ToolResultStorage,
  serializeToolResultPayload,
  sha1Short,
} from '../src/context/toolResultStorage.js'

test('tool-result storage retains small values inline and spills large text to a compatible reference', async () => {
  await inTemporaryDirectory(async directory => {
    const store = new ToolResultStorage(directory, { inlineLimit: 10 })
    expect(store.maybeStore('read_file', 'hello')).toBe('hello')
    expect(store.maybeStore('read_file', '😀')).toBe('😀')

    const reference = store.maybeStore('read_file', 'x'.repeat(100))
    expect(ToolResultStorage.isRef(reference)).toBe(true)
    expect(typeof reference).toBe('string')
    const referenceText = reference as string
    expect(ToolResultStorage.parseRef(referenceText)).toBe('read_file_' + sha1Short('x'.repeat(100)))
    expect(referenceText).toContain(':100:')
    expect(store.fetch(referenceText)).toBe('x'.repeat(100))
  })
})

test('tool-result storage reads JSON payloads back after LRU eviction and deduplicates identical results', async () => {
  await inTemporaryDirectory(async directory => {
    const store = new ToolResultStorage(directory, { inlineLimit: 10, lruSize: 2 })
    const data = { key: 'value'.repeat(100) }
    const first = store.maybeStore('structured', data) as string
    const repeated = store.maybeStore('structured', data) as string
    store.maybeStore('other_a', 'a'.repeat(100))
    store.maybeStore('other_b', 'b'.repeat(100))

    expect(first).toBe(repeated)
    expect(store.fetch(first)).toEqual(data)
    expect(store.listRefs()).toEqual([
      referenceId(store.maybeStore('other_a', 'a'.repeat(100)) as string),
      referenceId(store.maybeStore('other_b', 'b'.repeat(100)) as string),
      referenceId(first),
    ].sort())
  })
})

test('tool-result storage lists, prunes, and safely ignores missing references', async () => {
  await inTemporaryDirectory(async directory => {
    const store = new ToolResultStorage(directory, { inlineLimit: 1 })
    for (let index = 0; index < 5; index += 1) {
      store.maybeStore('tool_' + index, 'payload_' + index)
    }

    expect(store.listRefs()).toHaveLength(5)
    expect(store.prune(2)).toBe(3)
    expect(store.listRefs()).toHaveLength(2)
    expect(store.fetch('does_not_exist')).toBeUndefined()
    expect(ToolResultStorage.isRef('plain text')).toBe(false)
    expect(ToolResultStorage.parseRef('plain text')).toBeUndefined()
  })
})

test('tool-result payload serialization and input validation are explicit', () => {
  expect(serializeToolResultPayload({ key: 'value' })).toBe('{"key":"value"}')
  expect(() => new ToolResultStorage('')).toThrow('baseDirectory must be non-empty')
  expect(() => new ToolResultStorage('/tmp', { lruSize: 0 })).toThrow('lruSize must be a positive integer')
})

async function inTemporaryDirectory(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-tool-result-storage-'))
  try {
    await run(directory)
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
}

function referenceId(value: string): string {
  const parsed = ToolResultStorage.parseRef(value)
  if (parsed === undefined) throw new Error('expected a tool-result reference')
  return parsed
}
