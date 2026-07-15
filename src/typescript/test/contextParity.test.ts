// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import {
  AdvancedCompactionStrategy,
  COMPACTION_REFERENCE_PREFIX,
  CompactionStrategy,
  ContextCompressor,
  ProviderTokenCounter,
  RepoMapper,
  SlidingWindowStrategy,
  SmartCompactionStrategy,
  SmartTokenCounter,
  SummarizationStrategy,
  TruncateStrategy,
  defaultPriorityScorer,
  getCompactionStrategy,
  naiveSummarizer,
  pruneToolMessages,
  pruneToolResult,
} from '../src/context/index.js'

test('provider-aware token counting preserves the Python model-routing and capacity contracts', () => {
  expect(ProviderTokenCounter.detectProvider('gpt-4o')).toBe('openai')
  expect(ProviderTokenCounter.detectProvider('o1-preview')).toBe('openai')
  expect(ProviderTokenCounter.detectProvider('claude-3-sonnet')).toBe('anthropic')
  expect(ProviderTokenCounter.detectProvider('gemini-pro')).toBe('google')
  expect(ProviderTokenCounter.detectProvider('palm-2')).toBe('google')
  expect(ProviderTokenCounter.detectProvider('llama-3.3')).toBe('meta')
  expect(ProviderTokenCounter.detectProvider('mixtral-8x7b')).toBe('mistral')
  expect(ProviderTokenCounter.detectProvider('unknown-model')).toBeUndefined()
  expect(ProviderTokenCounter.detectProvider('')).toBeUndefined()

  const messages = [{ content: 'missing role' }, { role: 'user', content: 'hello' }]
  expect(ProviderTokenCounter.messagesToText(messages)).toBe(': missing role\nuser: hello')
  expect(ProviderTokenCounter.countTokensForProvider(messages, 'openai', 'gpt-4')).toBeGreaterThan(0)
  expect(ProviderTokenCounter.countTokensForProvider('hello', undefined, 'gpt-4')).toBeGreaterThan(0)

  const counter = new SmartTokenCounter({ model: 'gpt-4' })
  expect(counter.provider).toBe('openai')
  expect(new SmartTokenCounter({ provider: 'anthropic', model: 'gpt-4' }).provider).toBe('anthropic')
  expect(new SmartTokenCounter().provider).toBeUndefined()
  expect(counter.countRemainingCapacity('hello', 100)).toBeGreaterThan(0)
  expect(counter.countRemainingCapacity('hello '.repeat(1_000), 1)).toBe(0)
  expect(counter.estimateCompressionRatio('one two three four', 'one')).toBeGreaterThan(0)
  expect(counter.estimateCompressionRatio('', 'one')).toBe(0)
  expect(counter.estimateCompressionRatio('same', 'same')).toBe(0)
})

test('tool-result pre-pruning preserves exact, structured, and protected values while replacing binary blobs', () => {
  expect(pruneToolResult('x'.repeat(100), { maxChars: 100 })).toEqual({ content: 'x'.repeat(100), pruned: false })
  expect(pruneToolResult({ value: true })).toEqual({ content: { value: true }, pruned: false })

  const binary = Array.from({ length: 2_000 }, (_, index) => String.fromCharCode(index % 32)).join('')
  const binaryResult = pruneToolResult(binary, { maxChars: 100 })
  expect(binaryResult).toMatchObject({ pruned: true })
  expect(binaryResult.content).toContain('binary content')

  const long = 'y'.repeat(50_000)
  const input = [
    { role: 'user', content: long },
    { role: 'tool', content: long, tool_call_id: 'first' },
    { role: 'tool', content: long, tool_call_id: 'protected' },
    { role: 'assistant', content: long },
  ]
  const result = pruneToolMessages(input, { protectLast: 2 })
  expect(result.prunedCount).toBe(1)
  expect(result.messages[0]?.content).toBe(long)
  expect(result.messages[1]?.content).not.toBe(long)
  expect(result.messages[2]?.content).toBe(long)
  expect(result.messages[3]?.content).toBe(long)
  expect(input[1]?.content).toBe(long)
})

test('native compressor retains the Python summary framing, validation, and iterative history handoff', () => {
  expect(naiveSummarizer([] as Array<Record<string, unknown>>, 100)).toBe('')
  expect(naiveSummarizer([
    { role: 'user', content: { task: 'inspect', enabled: true } },
    { role: 'assistant', content: 'x'.repeat(250) },
  ], 100)).toEqual([
    '- user: {"task":"inspect","enabled":true}',
    `- assistant: ${'x'.repeat(200)}…`,
  ].join('\n'))
  expect(() => new ContextCompressor({ threshold: 0 })).toThrow('threshold')
  expect(() => new ContextCompressor({ protectFirst: -1 })).toThrow('protectFirst')

  const compressor = new ContextCompressor({
    contextWindow: 100,
    protectFirst: 1,
    protectLast: 1,
    summarizer: messages => `handoff for ${messages.length} message(s)`,
    summaryMinTokens: 1,
    threshold: 0.1,
  })
  const first = compressor.compress([
    { role: 'system', content: 'system constraints' },
    { role: 'user', content: 'old question' },
    { role: 'assistant', content: 'old answer' },
    { role: 'user', content: 'latest question' },
  ])
  expect(first.metadata.strategy).toBe('first-pass')
  expect(String(first.messages[1]?.content)).toContain(COMPACTION_REFERENCE_PREFIX)
  expect(first.messages.at(-1)).toEqual({ role: 'user', content: 'latest question' })

  const second = compressor.compress([...first.messages, { role: 'assistant', content: 'latest answer' }])
  expect(second.metadata.strategy).toBe('iterative')
  expect(second.messages.some(message => String(message.content).includes('handoff for'))).toBe(true)
})

test('compaction strategy aliases are non-destructive without a model and model-backed summaries keep a system prefix', () => {
  const messages = [
    { role: 'system', content: 'keep facts' },
    { role: 'user', content: 'old request '.repeat(120) },
    { role: 'assistant', content: 'old answer '.repeat(120) },
    { role: 'user', content: 'latest request' },
  ]
  const options = { model: 'gpt-4o', targetTokens: 60 }
  const aliases = [
    CompactionStrategy.SUMMARIZE,
    CompactionStrategy.SLIDING_WINDOW,
    CompactionStrategy.PRIORITY_BASED,
    CompactionStrategy.TRUNCATE,
    CompactionStrategy.ADVANCED,
  ]
  for (const alias of aliases) {
    const result = getCompactionStrategy(alias, options).compact(messages)
    expect(result.messages).toEqual(messages)
    expect(result.stats.reason).toBe('no_summary_agent')
  }

  const smart = new SmartCompactionStrategy({
    ...options,
    modelPort: () => 'model-backed durable summary',
  }).compact(messages)
  expect(smart.stats.strategy).toBe('smart')
  expect(smart.stats.substrategy).toBe('summarization')
  expect(smart.messages[0]).toEqual(messages[0])
  expect(smart.messages.some(message => String(message.content).includes('model-backed durable summary'))).toBe(true)

  expect(new SlidingWindowStrategy(options)).toBeInstanceOf(SlidingWindowStrategy)
  expect(new SummarizationStrategy(options)).toBeInstanceOf(SummarizationStrategy)
  expect(new TruncateStrategy(options)).toBeInstanceOf(TruncateStrategy)
  expect(new AdvancedCompactionStrategy(options)).toBeInstanceOf(AdvancedCompactionStrategy)
  expect(defaultPriorityScorer({ role: 'system', content: 'x' }, 0)).toBeGreaterThan(0.5)
  expect(defaultPriorityScorer({ role: 'assistant', content: 'x', tool_calls: [] }, 5)).toBeGreaterThan(0.5)
  expect(defaultPriorityScorer({ role: 'user', content: 'x'.repeat(600) }, 0)).toBeGreaterThanOrEqual(0.6)
})

test('repo mapping retains public Python declarations, skips private/broken definitions, and honors gitignore', async () => {
  await inTemporaryDirectory(async root => {
    await writeFile(join(root, 'app.py'), [
      'MAX_RETRIES = 3',
      'def public_function(value):',
      '    return value',
      'async def async_fetch(url: str) -> str:',
      '    return url',
      'class DataProcessor:',
      '    def process(self, value):',
      '        return value',
      '    def _hidden(self):',
      '        return value',
      'def _private():',
      '    return None',
    ].join('\n'))
    await writeFile(join(root, 'broken.py'), 'def (\n')
    await writeFile(join(root, 'good.py'), 'def works():\n    return True\n')
    await mkdir(join(root, 'ignored'))
    await writeFile(join(root, 'ignored', 'secret.py'), 'SECRET = "not visible"\n')
    await writeFile(join(root, '.gitignore'), 'ignored/\n')

    const result = await new RepoMapper({ tokenBudget: 2_000 }).build(root)
    expect(result.text).toContain('MAX_RETRIES')
    expect(result.text).toContain('public_function')
    expect(result.text).toContain('async_fetch')
    expect(result.text).toContain('DataProcessor.process')
    expect(result.text).toContain('works')
    expect(result.text).not.toContain('_hidden')
    expect(result.text).not.toContain('_private')
    expect(result.text).not.toContain('SECRET')
  })
})

async function inTemporaryDirectory(run: (directory: string) => Promise<void>): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-context-parity-'))
  try {
    await run(directory)
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
}
