// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'

import { expect, test } from 'bun:test'

import { AnthropicMessagesClient } from '../src/llms/anthropic.js'

import {
  type CachedSystemPrompt,
  type SystemPromptSegment,
  EPHEMERAL_CACHE_CONTROL,
  SUPPORTS_CACHING,
  cacheableSystemPrompt,
  extractCacheTokens,
  joinSystemSegments,
  wrapSystemSegmentsWithCache,
  wrapSystemWithCache,
  wrapToolsWithCache,
} from '../src/streaming/promptCaching.js'

test('prompt caching exposes Anthropic as the supported provider and marks non-empty system prompts', () => {
  expect(SUPPORTS_CACHING).toEqual(['anthropic'])
  expect(wrapSystemWithCache('')).toBe('')
  expect(wrapSystemWithCache('Stable instructions.')).toEqual([{
    type: 'text',
    text: 'Stable instructions.',
    cache_control: { type: 'ephemeral' },
  }])
})

test('tool cache wrapping copies schemas and leaves exactly one tail breakpoint', () => {
  const tools = [{
    name: 'ReadFile',
    cache_control: { type: 'ephemeral' },
    input_schema: { type: 'object' },
  }, {
    name: 'WriteFile',
    cache_control: { type: 'persistent' },
    input_schema: { type: 'object' },
  }]

  const wrapped = wrapToolsWithCache(tools)
  expect(wrapped).not.toBe(tools)
  expect(wrapped).toEqual([{
    name: 'ReadFile',
    input_schema: { type: 'object' },
  }, {
    name: 'WriteFile',
    input_schema: { type: 'object' },
    cache_control: EPHEMERAL_CACHE_CONTROL,
  }])
  expect(tools).toEqual([{
    name: 'ReadFile',
    cache_control: { type: 'ephemeral' },
    input_schema: { type: 'object' },
  }, {
    name: 'WriteFile',
    cache_control: { type: 'persistent' },
    input_schema: { type: 'object' },
  }])

  const empty: readonly Record<string, unknown>[] = []
  expect(wrapToolsWithCache(empty)).toBe(empty)
})

test('the cached system prefix survives a memory mutation between turns', () => {
  const turnOne = wrapSystemSegmentsWithCache(promptSegments('Recalled: user prefers bun test.'))
  const turnTwo = wrapSystemSegmentsWithCache(promptSegments(
    'Recalled: user prefers bun test.\nRecalled: the daemon owns turnRunner.',
  ))

  // The whole point of the change: the agent writing to its memory before the
  // turn ends must not invalidate the prefix on the next request.
  expect(prefixHash(turnTwo)).toBe(prefixHash(turnOne))
  expect(blockAt(turnOne, 0).text).toBe('Bootstrap contract.\n\nAgent instructions.\n\nWorkspace is trusted.')
  expect(blockAt(turnOne, 0).cache_control).toEqual(EPHEMERAL_CACHE_CONTROL)

  // The volatile tail is the block that moves, and it carries no breakpoint.
  expect(blockAt(turnOne, 1)).toEqual({ type: 'text', text: '\n\nRecalled: user prefers bun test.' })
  expect(blockAt(turnTwo, 1).cache_control).toBeUndefined()
  expect(blockAt(turnTwo, 1).text).not.toBe(blockAt(turnOne, 1).text)

  // Reordering stable sources ahead of volatile ones is the only content
  // change; the bytes the model reads still match the joined single string.
  expect(blockTexts(turnOne).join('')).toBe(joinSystemSegments(promptSegments('Recalled: user prefers bun test.')))
})

test('segment wrapping degenerates safely when a partition is empty', () => {
  expect(wrapSystemSegmentsWithCache([])).toBe('')
  expect(wrapSystemSegmentsWithCache([{ name: 'memory', text: '', volatile: true }])).toBe('')

  expect(wrapSystemSegmentsWithCache([
    { name: 'bootstrap', text: 'Bootstrap contract.' },
    { name: 'memory', text: '', volatile: true },
  ])).toEqual([{ type: 'text', text: 'Bootstrap contract.', cache_control: EPHEMERAL_CACHE_CONTROL }])

  // Nothing outlives the turn, so a breakpoint would only buy cache writes.
  expect(wrapSystemSegmentsWithCache([
    { name: 'memory', text: 'Recalled: nothing stable.', volatile: true },
  ])).toEqual([{ type: 'text', text: 'Recalled: nothing stable.' }])
})

test('joined segments resolve back to two blocks while unregistered prompts keep one', () => {
  const joined = joinSystemSegments(promptSegments('Recalled: registry lookup.'))
  expect(joined).toBe(
    'Bootstrap contract.\n\nAgent instructions.\n\nWorkspace is trusted.\n\nRecalled: registry lookup.',
  )

  const resolved = cacheableSystemPrompt(joined)
  expect(blockTexts(resolved)).toEqual([
    'Bootstrap contract.\n\nAgent instructions.\n\nWorkspace is trusted.',
    '\n\nRecalled: registry lookup.',
  ])
  expect(blockAt(resolved, 0).cache_control).toEqual(EPHEMERAL_CACHE_CONTROL)

  // A caller that has not migrated still gets the previous whole-string block.
  expect(cacheableSystemPrompt('Unregistered instructions.')).toEqual([{
    type: 'text',
    text: 'Unregistered instructions.',
    cache_control: EPHEMERAL_CACHE_CONTROL,
  }])
  expect(cacheableSystemPrompt('')).toBe('')
})

test('the segment registry is bounded so long-lived daemons cannot leak prompts', () => {
  const evicted = joinSystemSegments([{ name: 'bootstrap', text: 'Evictable prefix.' }])
  for (let index = 0; index < 8; index += 1) {
    joinSystemSegments([
      { name: 'bootstrap', text: `Filler prefix ${index}.` },
      { name: 'memory', text: 'Filler memory.', volatile: true },
    ])
  }

  // Dropping the oldest entry loses only the split, never the prompt itself.
  expect(cacheableSystemPrompt(evicted)).toEqual([{
    type: 'text',
    text: 'Evictable prefix.',
    cache_control: EPHEMERAL_CACHE_CONTROL,
  }])
})

test('cache token extraction accepts record and SDK-shaped usage while defaulting malformed values', () => {
  expect(extractCacheTokens({
    cache_read_input_tokens: 13,
    cache_creation_input_tokens: 7,
  })).toEqual([13, 7])

  class Usage {
    readonly cache_creation_input_tokens = '5'
    readonly cache_read_input_tokens = 11.8
  }

  expect(extractCacheTokens(new Usage())).toEqual([11, 5])
  expect(extractCacheTokens({
    cache_read_input_tokens: Number.POSITIVE_INFINITY,
    cache_creation_input_tokens: 'not-a-number',
  })).toEqual([0, 0])
  expect(extractCacheTokens(undefined)).toEqual([0, 0])
})

/** Daemon-shaped assembly: memory is declared before the static addendum. */
function promptSegments(memory: string): readonly SystemPromptSegment[] {
  return [
    { name: 'bootstrap', text: 'Bootstrap contract.' },
    { name: 'agent', text: 'Agent instructions.' },
    { name: 'memory', text: memory, volatile: true },
    { name: 'workspace-addendum', text: 'Workspace is trusted.' },
  ]
}

function prefixHash(prompt: CachedSystemPrompt): string {
  return createHash('sha256').update(blockAt(prompt, 0).text).digest('hex')
}

function blockTexts(prompt: CachedSystemPrompt): readonly string[] {
  expect(typeof prompt).not.toBe('string')
  return (prompt as readonly { readonly text: string }[]).map(block => block.text)
}

function blockAt(prompt: CachedSystemPrompt, index: number): {
  readonly cache_control?: { readonly type: 'ephemeral' }
  readonly text: string
  readonly type: 'text'
} {
  expect(typeof prompt).not.toBe('string')
  const blocks = prompt as readonly {
    readonly cache_control?: { readonly type: 'ephemeral' }
    readonly text: string
    readonly type: 'text'
  }[]
  const block = blocks[index]
  if (!block) {
    throw new Error(`expected a system block at index ${index}, saw ${blocks.length}`)
  }
  return block
}

test('the Anthropic client moves the cache breakpoint off the volatile memory section', async () => {
  const segments = [
    { name: 'bootstrap', text: 'STABLE bootstrap prompt' },
    { name: 'agent', text: 'STABLE agent prompt' },
    { name: 'memory', text: 'VOLATILE memory rewritten every turn', volatile: true },
  ]
  const messages = [
    { role: 'system' as const, content: segments.map(segment => segment.text).join('\n\n') },
    { role: 'user' as const, content: 'hi' },
  ]
  const send = async (systemSegments?: typeof segments) => {
    let body: Record<string, unknown> = {}
    const client = new AnthropicMessagesClient({
      apiKey: 'test-key',
      fetchImplementation: (async (_url: unknown, init: { body: string }) => {
        body = JSON.parse(init.body) as Record<string, unknown>
        return new Response(JSON.stringify({ content: [{ type: 'text', text: 'ok' }] }), { status: 200 })
      }) as never,
    })
    await client.complete({ model: 'claude-opus-4', messages, ...(systemSegments ? { systemSegments } : {}) } as never)
    return body.system as { text: string; cache_control?: unknown }[]
  }

  // Without segments the whole prompt is one cached block, so rewriting memory
  // — which the shipped prompt instructs the agent to do — invalidates all of it.
  const joined = await send()
  expect(joined).toHaveLength(1)
  expect(joined[0]?.text).toContain('VOLATILE')
  expect(joined[0]?.cache_control).toBeDefined()

  const split = await send(segments)
  expect(split).toHaveLength(2)
  expect(split[0]?.text).toBe('STABLE bootstrap prompt\n\nSTABLE agent prompt')
  expect(split[0]?.cache_control).toBeDefined()
  expect(split[1]?.text).toContain('VOLATILE')
  expect(split[1]?.cache_control).toBeUndefined()
})

test('segments that do not reproduce the converted system text fall back to whole-string caching', async () => {
  // A stray system message in the transcript must never be dropped to win a
  // cache hit, so the segmented path is used only when it is provably lossless.
  let body: Record<string, unknown> = {}
  const client = new AnthropicMessagesClient({
    apiKey: 'test-key',
    fetchImplementation: (async (_url: unknown, init: { body: string }) => {
      body = JSON.parse(init.body) as Record<string, unknown>
      return new Response(JSON.stringify({ content: [{ type: 'text', text: 'ok' }] }), { status: 200 })
    }) as never,
  })
  await client.complete({
    model: 'claude-opus-4',
    messages: [
      { role: 'system', content: 'declared segment' },
      { role: 'system', content: 'extra system message nobody declared' },
      { role: 'user', content: 'hi' },
    ],
    systemSegments: [{ name: 'bootstrap', text: 'declared segment' }],
  } as never)
  const blocks = body.system as { text: string }[]
  expect(blocks).toHaveLength(1)
  expect(blocks[0]?.text).toContain('extra system message nobody declared')
})
