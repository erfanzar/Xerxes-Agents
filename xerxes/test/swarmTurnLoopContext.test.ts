// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  COMPACTION_ARCHIVE_NOTE,
  COMPACTION_REFERENCE_PREFIX,
  ContextCompressor,
  ProviderTokenCounter,
  SmartTokenCounter,
  estimateContextTokens,
  pruneToolMessages,
  pruneToolResult,
} from '../src/context/index.js'
import { omittedScreenshotMarker, screenshotParts } from '../src/context/screenshotSuperseder.js'
import type { LlmDelta } from '../src/llms/client.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

function buildScreenshotToolPayload(
  width: number,
  height: number,
  base64Body: string,
  mediaType = 'image/png',
): string {
  return JSON.stringify({
    _multimodal: true,
    text_summary: `Screen capture: ${width}x${height}\nFocused window: Terminal`,
    content: [
      { type: 'text', text: `Screen capture: ${width}x${height}` },
      { type: 'image_url', image_url: { url: `data:${mediaType};base64,${base64Body}` } },
    ],
  })
}

test('ProviderTokenCounter prices JSON-serialized screenshot tool results as images instead of base64 text', () => {
  const largeBase64 = 'iVBORw0KGgoA'.repeat(30_000) // 360 KB of base64 (~90K chars/4 tokens if miscounted)
  const screenshotContent = buildScreenshotToolPayload(1920, 1080, largeBase64)
  const messages = [
    { role: 'user', content: 'inspect my desktop' },
    {
      role: 'assistant',
      content: '',
      tool_calls: [{ id: 'shot-1', type: 'function', function: { name: 'computer_use', arguments: { action: 'screenshot' } } }],
    },
    { role: 'tool', tool_call_id: 'shot-1', name: 'computer_use', content: screenshotContent },
  ]

  const tokens = ProviderTokenCounter.countTokensForProvider(messages, 'anthropic')
  expect(tokens).toBeGreaterThan(1_600)
  expect(tokens).toBeLessThan(2_200)
  expect(ProviderTokenCounter.messagesToText(messages)).not.toContain(largeBase64.slice(0, 48))

  // Media types containing digits (e.g. image/jp2) and line-wrapped base64 payloads also parse and summarize cleanly.
  const wrappedJp2 = buildScreenshotToolPayload(1280, 720, `${'QUJD'.repeat(50)}\r\n${'REVG'.repeat(50)}`, 'image/jp2')
  expect(screenshotParts(wrappedJp2)?.images[0]?.mediaType).toBe('image/jp2')
  expect(omittedScreenshotMarker(wrappedJp2)).toBe('[screenshot omitted: 300 bytes, 1280x720]')
})

test('in-turn auto-compaction check does not falsely fail when the latest screenshot sits in the protected tail', async () => {
  const counter = new SmartTokenCounter({ model: 'anthropic/claude-sonnet-4-5' })
  const screenshotPayload = buildScreenshotToolPayload(1568, 1018, 'A'.repeat(320_000))
  const computerUse: ToolDefinition = {
    type: 'function',
    function: { name: 'computer_use', description: 'Capture.', parameters: { type: 'object', properties: {} } },
  }

  let round = 0
  let compactions = 0
  const state = createAgentState()
  const events: StreamEvent[] = []
  for await (const event of runTurn({
    model: 'anthropic/claude-sonnet-4-5',
    permissionMode: 'accept-all',
    state,
    tools: [computerUse],
    userMessage: 'capture screen and describe it',
  }, {
    // 50,000-token threshold: a 320 KB base64 string miscounted as text (~80K tokens) used to trip this!
    contextCompactionDue: messages => estimateContextTokens(
      messages as unknown as readonly Record<string, unknown>[],
      { model: 'anthropic/claude-sonnet-4-5', tokenCounter: counter },
    ) >= 50_000,
    reduceContext: async messages => {
      compactions += 1
      return { messages, tokensFreed: 0 }
    },
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        round += 1
        if (round === 1) {
          yield {
            toolCalls: [{ id: 'shot-live', type: 'function', function: { name: 'computer_use', arguments: {} } }],
          }
          return
        }
        yield { content: 'The terminal window is focused.' }
      },
    },
    toolExecutor: { async execute() { return screenshotPayload } },
  })) {
    events.push(event)
  }

  expect(compactions).toBe(0)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'completed' })
})

test('pruneToolResult and pruneToolMessages collapse older screenshots to omitted markers instead of broken base64 slices', () => {
  const screenshotPayload = buildScreenshotToolPayload(1440, 900, 'B'.repeat(12_000))
  const single = pruneToolResult(screenshotPayload, { maxChars: 4_000 })
  expect(single).toEqual({
    content: '[screenshot omitted: 9000 bytes, 1440x900]',
    pruned: true,
  })

  const messages = [
    { role: 'user', content: 'open browser' },
    { role: 'tool', tool_call_id: 'old-shot', content: screenshotPayload },
    { role: 'user', content: 'next step' },
    { role: 'assistant', content: 'working' },
  ]
  const pruned = pruneToolMessages(messages, { protectLast: 2 })
  expect(pruned.prunedCount).toBe(1)
  expect(pruned.messages[1]?.content).toBe('[screenshot omitted: 9000 bytes, 1440x900]')
})

test('ContextCompressor deduplicates reference headers across iterative passes and skips empty summary-only middles', () => {
  let summarizerCalls = 0
  const compressor = new ContextCompressor({
    contextWindow: 120,
    forceSummarize: true,
    protectFirst: 1,
    protectLast: 1,
    summarizer: slice => {
      summarizerCalls += 1
      return `pass-${summarizerCalls}(${slice.length})`
    },
    summaryMinTokens: 1,
    threshold: 0.2,
  })

  const first = compressor.compress([
    { role: 'system', content: 'system prompt' },
    { role: 'user', content: 'turn 1 question' },
    { role: 'assistant', content: 'turn 1 answer' },
    { role: 'user', content: 'turn 2 question' },
  ])
  expect(first.compressed).toBe(true)
  expect(first.metadata.strategy).toBe('first-pass')
  expect(summarizerCalls).toBe(1)

  // Simulate daemon archive-note decoration on the stored summary before pass 2.
  const withArchiveNote = first.messages.map((message, index) => index === 1
    ? {
        ...message,
        content: `${COMPACTION_REFERENCE_PREFIX}\n${COMPACTION_ARCHIVE_NOTE}\n\npass-1(2)`,
      }
    : message)

  // Re-running compress immediately with NO new middle messages (only the prior summary sits between head and tail)
  // must be a clean no-op rather than invoking the summarizer on `[]`.
  const noop = compressor.compress(withArchiveNote)
  expect(noop.compressed).toBe(false)
  expect(noop.metadata.strategy).toBe('no-middle')
  expect(summarizerCalls).toBe(1)

  // Adding new turns produces an iterative summary with exactly ONE reference header.
  const second = compressor.compress([
    ...withArchiveNote,
    { role: 'assistant', content: 'turn 2 answer' },
    { role: 'user', content: 'turn 3 question' },
  ])
  expect(second.compressed).toBe(true)
  expect(second.metadata.strategy).toBe('iterative')
  expect(summarizerCalls).toBe(2)

  const summaryContent = String(second.messages[1]?.content ?? '')
  const prefixMatches = summaryContent.split(COMPACTION_REFERENCE_PREFIX).length - 1
  expect(prefixMatches).toBe(1)
  expect(summaryContent).not.toContain(COMPACTION_ARCHIVE_NOTE)
  expect(summaryContent).toBe(`${COMPACTION_REFERENCE_PREFIX}\n\npass-1(2)\n\n---\n\npass-2(2)`)
})
