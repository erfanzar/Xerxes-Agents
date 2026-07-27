// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  AdvancedCompactionStrategy,
  COMPACTION_LENGTH_INSTRUCTIONS,
  COMPACTION_SUMMARY_PREFIX,
  CompactionStrategy,
  CompactionProvisioner,
  PriorityBasedStrategy,
  SlidingWindowStrategy,
  SmartCompactionStrategy,
  SummarizationStrategy,
  TruncateStrategy,
  buildCompactionPromptFromText,
  getCompactionStrategy,
  renderMessagesForSummary,
  stripCompactionAnalysis,
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
  expect(requests[0]?.prompt).toContain('CONTEXT TO SUMMARIZE:')
  expect(requests[0]?.prompt).toContain('old request')
  expect(requests[0]?.prompt).toContain('EXISTING SUMMARY TO REFRESH')
  expect(requests[0]?.prompt).toContain('prior durable state')
})

test('the shared compaction template covers every high-loss section and its own placement', () => {
  const prompt = buildCompactionPromptFromText({
    context: 'Message 1 [USER]\nrename the daemon socket path',
    preserveTopics: ['socket path', 'daemon restart'],
    targetLength: 'detailed',
  })

  expect(prompt).toContain('<analysis>')
  expect(prompt).toContain('</analysis>')
  expect(prompt).toContain('## User requests')
  expect(prompt).toContain('## Current task')
  expect(prompt).toContain('## Files touched')
  expect(prompt).toContain('## Decisions')
  expect(prompt).toContain('## Errors and fixes')
  expect(prompt).toContain('## Open questions')
  expect(prompt).toContain('## Next step')
  expect(prompt).toContain('Every non-tool user message in the slice, in order, one bullet each')
  expect(prompt).toContain('quoted VERBATIM')
  expect(prompt).toContain('absolute path')
  // The summary lands between a preserved head and a preserved live tail, so restating the tail
  // spends the compacted window on messages the agent can already read.
  expect(prompt).toContain('preserved live tail')
  expect(prompt).toContain('do not re-describe the recent turns that follow you')
  expect(prompt).toContain(COMPACTION_LENGTH_INSTRUCTIONS.detailed)
  expect(prompt).toContain('- Ensure these topics are covered: socket path, daemon restart')
  expect(prompt).not.toContain('EXISTING SUMMARY TO REFRESH')
  expect(prompt.endsWith('Begin with <analysis>.')).toBe(true)
})

test('the analysis scratchpad is stripped, including from a response truncated inside the block', () => {
  expect(stripCompactionAnalysis('<analysis>counting turns</analysis>\n\n## User requests\n- one')).toBe(
    '## User requests\n- one',
  )
  expect(stripCompactionAnalysis('## Next step\nrun the tests\n<analysis>still thinking about')).toBe(
    '## Next step\nrun the tests',
  )
  expect(stripCompactionAnalysis('<analysis>a</analysis>x<analysis>b</analysis>y')).toBe('xy')
  expect(stripCompactionAnalysis('## Decisions\nnone')).toBe('## Decisions\nnone')
})

test('tool traffic renders as one call line carrying its own outcome', () => {
  const rendered = renderMessagesForSummary([
    { role: 'user', content: 'read the config' },
    {
      role: 'assistant',
      content: 'Reading it now.',
      tool_calls: [
        { id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: '{"path": "/repo/a.ts"}' } },
        { id: 'call-2', type: 'function', function: { name: 'Bash', arguments: '{"command": "bun test"}' } },
      ],
    },
    { role: 'tool', tool_call_id: 'call-1', content: 'export const port = 4_000\n' },
    { role: 'tool', tool_call_id: 'call-2', content: '1 fail', is_error: true },
  ])

  expect(rendered).toContain('called ReadFile({"path": "/repo/a.ts"}) -> export const port = 4_000')
  expect(rendered).toContain('called Bash({"command": "bun test"}) -> error: 1 fail')
  expect(rendered).not.toContain('tool_calls=')
  // Both results are folded into their call lines, so only the user and assistant turns remain.
  expect(rendered).toContain('Message 2 [ASSISTANT]')
  expect(rendered).not.toContain('Message 3')
})

test('rendering keeps unpaired tool results and truncates oversized call payloads', () => {
  const rendered = renderMessagesForSummary([
    { role: 'tool', tool_call_id: 'orphan-1', content: 'result from before this window' },
    { role: 'assistant', tool_calls: [{ id: 'call-9', name: 'WriteFile', input: { body: 'x'.repeat(400) } }] },
  ])

  expect(rendered).toContain('Message 1 [TOOL]')
  expect(rendered).toContain('result from before this window')
  expect(rendered).toContain('tool_call_id=orphan-1')
  expect(rendered).toContain('called WriteFile(')
  expect(rendered).toContain('-> (no result in this slice)')
  expect(rendered).toContain('chars)')
  expect(rendered.length).toBeLessThan(500)
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
  expect(rendered).toContain('called ReadFile() -> (no result in this slice)')

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
