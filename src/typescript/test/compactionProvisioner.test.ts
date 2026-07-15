// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  AdvancedCompactionStrategy,
  COMPACTION_SUMMARY_PREFIX,
  CompactionStrategy,
  CompactionProvisioner,
  PriorityBasedStrategy,
  SlidingWindowStrategy,
  SmartCompactionStrategy,
  SummarizationStrategy,
  TruncateStrategy,
  getCompactionStrategy,
  renderMessagesForSummary,
  type CompactionModelRequest,
} from '../src/context/index.js'

function messages(): Array<Record<string, unknown>> {
  return [
    { role: 'system', content: 'Keep the response factual.' },
    { role: 'user', content: 'old request '.repeat(90) },
    { role: 'assistant', content: 'old answer '.repeat(90) },
    { role: 'user', content: 'latest request' },
  ]
}

test('provisioner compacts through an injected model port and preserves the live tail', () => {
  const requests: CompactionModelRequest[] = []
  const provisioner = new CompactionProvisioner({
    model: 'gpt-4o',
    maxContextTokens: 240,
    thresholdTokens: 1,
    targetTokens: 80,
    summaryMaxTokens: 700,
    modelPort: request => {
      requests.push(request)
      return 'AGENT MEMORY: old request was resolved.'
    },
  })

  const result = provisioner.compact(messages(), { force: true, previousSummary: 'prior durable state' })

  expect(result.compacted).toBe(true)
  expect(result.reason).toBe('compacted')
  expect(result.messages[0]).toEqual(messages()[0])
  expect(result.messages.at(-1)).toEqual(messages().at(-1))
  expect(result.messages.some(message => String(message.content).startsWith(COMPACTION_SUMMARY_PREFIX))).toBe(true)
  expect(result.messages.some(message => String(message.content).includes('AGENT MEMORY'))).toBe(true)
  expect(result.tokensAfter).toBeLessThan(result.tokensBefore)
  expect(requests).toHaveLength(1)
  expect(requests[0]).toMatchObject({
    model: 'gpt-4o',
    maxTokens: 700,
    temperature: 0.2,
    previousSummary: 'prior durable state',
  })
  expect(requests[0]?.prompt).toContain('Conversation history to compact:')
  expect(requests[0]?.prompt).toContain('old request')
})

test('provisioner does not drop history without an agent and surfaces agent failures deterministically', () => {
  const history = messages()
  const incoming = [{ role: 'user', content: 'incoming turn '.repeat(90) }]
  const noAgent = new CompactionProvisioner({
    model: 'gpt-4o',
    maxContextTokens: 240,
    thresholdTokens: 1,
    targetTokens: 80,
  })
  const skipped = noAgent.compactBeforeAppend(history, incoming)

  expect(skipped.compacted).toBe(false)
  expect(skipped.messages).toEqual(history)
  expect(skipped.reason).toBe('no_summary_agent')
  expect(noAgent.shouldCompact(history, { force: true })).toBe(true)

  const failing = new CompactionProvisioner({
    model: 'gpt-4o',
    maxContextTokens: 240,
    thresholdTokens: 1,
    targetTokens: 80,
    summaryAgent: () => {
      throw new Error('model unavailable')
    },
  }).compact(history, { force: true })

  expect(failing.compacted).toBe(false)
  expect(failing.messages).toEqual(history)
  expect(failing.reason).toBe('summary_agent_failed')
  expect(failing.error).toBe('model unavailable')
})

test('provisioner retains the shared compressor prune-only result without invoking the model', () => {
  let modelCalls = 0
  const result = new CompactionProvisioner({
    model: 'gpt-4o',
    maxContextTokens: 2_000,
    thresholdTokens: 1_500,
    targetTokens: 1_000,
    modelPort: () => {
      modelCalls += 1
      return 'unused summary'
    },
  }).compact([
    { role: 'user', content: 'keep this request' },
    { role: 'tool', content: 'x'.repeat(10_000) },
    { role: 'user', content: 'latest request' },
  ])

  expect(result.compacted).toBe(true)
  expect(result.reason).toBe('pruned')
  expect(result.summarizedCount).toBe(0)
  expect(modelCalls).toBe(0)
  expect(result.tokensAfter).toBeLessThan(result.tokensBefore)
})

test('summary rendering and strategy selection are stable and model-backed smart compaction is tagged', () => {
  const rendered = renderMessagesForSummary([
    { role: 'assistant', content: { z: 1, a: 2 }, tool_calls: [{ id: 'call-1', name: 'ReadFile' }] },
  ])
  expect(rendered).toContain('{"a":2,"z":1}')
  expect(rendered).toContain('tool_calls=[{"id":"call-1","name":"ReadFile"}]')

  const options = {
    model: 'gpt-4o',
    targetTokens: 80,
    modelPort: () => 'strategy summary',
  }
  expect(getCompactionStrategy(CompactionStrategy.SUMMARIZE, options)).toBeInstanceOf(SummarizationStrategy)
  expect(getCompactionStrategy('sliding_window', options)).toBeInstanceOf(SlidingWindowStrategy)
  expect(getCompactionStrategy('priority_based', options)).toBeInstanceOf(PriorityBasedStrategy)
  expect(getCompactionStrategy('truncate', options)).toBeInstanceOf(TruncateStrategy)
  expect(getCompactionStrategy('advanced', options)).toBeInstanceOf(AdvancedCompactionStrategy)
  expect(getCompactionStrategy('unknown', options)).toBeInstanceOf(SummarizationStrategy)

  const smart = getCompactionStrategy('smart', options)
  expect(smart).toBeInstanceOf(SmartCompactionStrategy)
  const result = smart.compact(messages())
  expect(result.stats.strategy).toBe('smart')
  expect(result.stats.summaryCreated).toBe(true)
  expect(result.stats.substrategy).toBe('summarization')
})
