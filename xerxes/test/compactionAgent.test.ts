// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CompactionAgent,
  CompactionResponseShapeError,
  completionText,
  type CompactionCompletionRequest,
} from '../src/agents/compactionAgent.js'
import { DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS } from '../src/context/index.js'

function history(): Array<Record<string, unknown>> {
  return [
    { role: 'system', content: 'Remain factual.' },
    { role: 'user', content: 'rename the daemon socket path '.repeat(40) },
    { role: 'assistant', content: 'renaming it now '.repeat(40) },
    { role: 'user', content: 'latest request' },
  ]
}

test('the live compaction call carries the sectioned template and a budget that fits it', async () => {
  const requests: CompactionCompletionRequest[] = []
  const agent = new CompactionAgent({
    model: 'gpt-test',
    completion: request => {
      requests.push(request)
      return 'durable summary of the resolved request'
    },
  })

  await agent.summarizeMessages(history())

  expect(requests).toHaveLength(1)
  const request = requests[0]
  expect(request?.maxTokens).toBe(DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS)
  // 2_048 tokens truncated the enumerated sections mid-summary and stored the fragment.
  expect(request?.maxTokens).toBeGreaterThan(2_048)
  expect(request?.prompt).toContain('## User requests')
  expect(request?.prompt).toContain('## Next step')
  expect(request?.prompt).toContain('preserved live tail')
  expect(request?.prompt).toContain('CONTEXT TO SUMMARIZE:')
  expect(request?.prompt).toContain('rename the daemon socket path')

  expect(agent.summaryMaxTokens).toBe(DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS)
  expect(new CompactionAgent({ completion: () => '', summaryMaxTokens: 32_000 }).summaryMaxTokens).toBe(32_000)
  expect(() => new CompactionAgent({ completion: () => '', summaryMaxTokens: 0 })).toThrow(RangeError)
})

test('the analysis scratchpad never reaches the stored summary', async () => {
  const agent = new CompactionAgent({
    model: 'gpt-test',
    completion: () => ({
      choices: [{
        message: {
          content: '<analysis>\nuser turns: two\n</analysis>\n\n## User requests\n- rename the socket path',
        },
      }],
    }),
  })

  const compacted = await agent.summarizeMessages(history())
  const stored = compacted.map(message => String(message.content)).join('\n')

  expect(stored).toContain('## User requests')
  expect(stored).not.toContain('<analysis>')
  expect(stored).not.toContain('user turns: two')
})

test('an unreadable response shape is typed data, not an indistinguishable provider failure', async () => {
  expect(completionText('plain text')).toEqual({ ok: true, text: 'plain text' })
  expect(completionText({ content: 'from content' })).toEqual({ ok: true, text: 'from content' })
  expect(completionText({ choices: [{ message: { content: 'from choices' } }] })).toEqual({
    ok: true,
    text: 'from choices',
  })

  const failure = completionText({ choices: [{ message: { content: { unexpected: true } } }] })
  expect(failure.ok).toBe(false)
  // The detail names keys only: response values can carry session content into daemon logs.
  expect(failure.ok === false && failure.detail).toBe('object with keys choices')

  const agent = new CompactionAgent({
    model: 'gpt-test',
    completion: () => ({ content: 12 }),
  })
  const shapeResult = await agent.summarizeContextResult('x'.repeat(400))
  expect(shapeResult.ok).toBe(false)
  expect(shapeResult.ok === false && shapeResult.detail).toBe('object with keys content')

  let thrown: unknown
  try {
    await agent.summarizeContext('x'.repeat(400))
  } catch (error) {
    thrown = error
  }
  expect(thrown).toBeInstanceOf(CompactionResponseShapeError)
  expect(String(thrown)).toContain('unusable response shape')

  const transport = new CompactionAgent({
    model: 'gpt-test',
    completion: () => {
      throw new Error('connection reset')
    },
  })
  let providerError: unknown
  try {
    await transport.summarizeContext('x'.repeat(400))
  } catch (error) {
    providerError = error
  }
  expect(providerError).toBeInstanceOf(Error)
  expect(providerError).not.toBeInstanceOf(CompactionResponseShapeError)
})
