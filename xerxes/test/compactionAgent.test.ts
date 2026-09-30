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
import { SmartTokenCounter } from '../src/context/tokenCounter.js'

test('oversized recovery bounds every request and preserves both ends of the source', async () => {
  const requests: CompactionCompletionRequest[] = []
  const counter = new SmartTokenCounter()
  const agent = new CompactionAgent({ maxContextTokens: 4096, summaryMaxTokens: 256, completion: request => {
    requests.push(request)
    expect(counter.countTokens(request.prompt)).toBeLessThan(4096)
    return 'Summary of this chronological segment with the unfinished work.'
  } })
  const original = 'FIRST-MARKER\n' + 'Long execution output and important work. '.repeat(4000) + '\nLAST-MARKER'
  expect(await agent.summarizeContext(original)).toContain('unfinished work')
  expect(requests.length).toBeGreaterThan(2)
  expect(requests.some(request => request.prompt.includes('FIRST-MARKER'))).toBe(true)
  expect(requests.some(request => request.prompt.includes('LAST-MARKER'))).toBe(true)
})

test('a failed chunk leaves the original transcript untouched', async () => {
  let calls = 0
  const messages = history()
  messages.splice(2, 0, { role: 'assistant', content: 'important history '.repeat(20_000) })
  const before = structuredClone(messages)
  const agent = new CompactionAgent({ maxContextTokens: 4096, completion: () => {
    if (++calls === 2) throw new Error('provider unavailable')
    return 'summary'
  } })
  await expect(agent.summarizeMessages(messages)).rejects.toThrow('provider unavailable')
  expect(messages).toEqual(before)
})

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

test('large model windows do not expand individual compaction requests', async () => {
  const counter = new SmartTokenCounter()
  const requests: CompactionCompletionRequest[] = []
  const agent = new CompactionAgent({ maxContextTokens: 264_000, completion: request => {
    requests.push(request)
    expect(counter.countTokens(request.prompt)).toBeLessThanOrEqual(32_000)
    return 'Preserved work summary.'
  } })
  await agent.summarizeContext('FIRST-MARKER\n' + 'Long command output from the existing session. '.repeat(18_000) + '\nLAST-MARKER')
  expect(requests.length).toBeGreaterThan(2)
  expect(requests.some(request => request.prompt.includes('FIRST-MARKER'))).toBe(true)
  expect(requests.some(request => request.prompt.includes('LAST-MARKER'))).toBe(true)
})

test('a late chunk timeout retries only that segment and retains earlier summaries', async () => {
  const requests: string[] = []
  let failed = false
  const agent = new CompactionAgent({ maxRequestTokens: 16000, completion: request => {
    requests.push(request.prompt)
    if (requests.length === 2 && !failed) { failed = true; throw new Error('The operation timed out.') }
    return 'Preserved completed work and remaining tasks.'
  } })
  const text = 'FIRST-SEGMENT-MARKER\n' + 'Important history and command evidence. '.repeat(10000)
  await expect(agent.summarizeContext(text)).resolves.toContain('Preserved')
  expect(failed).toBe(true)
  expect(requests.filter(prompt => prompt.includes('FIRST-SEGMENT-MARKER'))).toHaveLength(1)
  expect(requests[2]!.length).toBeLessThan(requests[1]!.length)
})

test('independent summary segments run at most three at a time and retain source order', async () => {
  let active = 0; let peak = 0; let sequence = 0;
  const prompts: string[] = [];
  const agent = new CompactionAgent({ maxRequestTokens: 4096, completion: async request => {
    const id = ++sequence; prompts.push(request.prompt);
    active++; peak = Math.max(peak, active);
    await Bun.sleep(id % 3 === 0 ? 1 : 10);
    active--;
    return `Preserved segment ${id} with its evidence.`;
  } });
  await agent.summarizeContext('Chronological repository evidence and important user requests. '.repeat(3000));
  expect(peak).toBe(3);
  const combined = prompts.at(-1)!;
  expect(combined.indexOf('Preserved segment 1')).toBeLessThan(combined.indexOf('Preserved segment 2'));
  expect(combined.indexOf('Preserved segment 2')).toBeLessThan(combined.indexOf('Preserved segment 3'));
});

test('a failed parallel segment drains its siblings before returning failure', async () => {
  let active = 0;
  let calls = 0;
  const failure = new Error('provider refused this summary');
  const agent = new CompactionAgent({ maxRequestTokens: 4096, completion: async () => {
    const id = ++calls;
    active++;
    try {
      if (id === 1) throw failure;
      await Bun.sleep(20);
      return 'Completed summary.';
    } finally { active--; }
  } });
  await expect(agent.summarizeContext('Important user constraints and saved work. '.repeat(3000))).rejects.toBe(failure);
  expect(active).toBe(0);
  expect(calls).toBe(3);
});

test('a carried summary past twice the budget is folded into one, instead of growing every pass', async () => {
  const counter = new SmartTokenCounter()
  const requests: CompactionCompletionRequest[] = []
  const agent = new CompactionAgent({
    model: 'gpt-test',
    maxContextTokens: 200_000,
    summaryMaxTokens: 2_048,
    completion: request => {
      requests.push(request)
      return request.prompt.includes('Earlier summary of this session')
        ? 'CONSOLIDATED: goal, open work and recovery references.'
        : 'NEW: what happened this pass.'
    },
  })
  // Nineteen passes of appending left a ~25K-token summary (live session, 2026-09-30).
  const carried = '[CONTEXT COMPACTION — REFERENCE ONLY]\n\n' + 'OLD-DETAIL established fact. '.repeat(9_000)
  const messages: Array<Record<string, unknown>> = [
    { role: 'user', content: carried },
    ...Array.from({ length: 40 }, (_, index) => ({ role: index % 2 ? 'assistant' : 'user', content: `step ${index} `.repeat(400) })),
    { role: 'user', content: 'latest request' },
  ]
  const compacted = await agent.summarizeMessages(messages)
  const summaries = compacted.filter(message => typeof message.content === 'string' && message.content.startsWith('[CONTEXT COMPACTION'))
  expect(summaries).toHaveLength(1)
  const summary = String(summaries[0]!.content)
  expect(summary).toContain('CONSOLIDATED')
  expect(summary).not.toContain('OLD-DETAIL')
  // One reference prefix, not one nested per pass.
  expect(summary.split('[CONTEXT COMPACTION').length).toBe(2)
  expect(counter.countTokens(summary)).toBeLessThan(2_048)
  // The consolidation saw the earlier summary without its prefix.
  const merge = requests.find(request => request.prompt.includes('Earlier summary of this session'))
  expect(merge?.prompt).toContain('OLD-DETAIL')
})

test('a small carried summary is kept and extended, not re-summarized', async () => {
  const requests: CompactionCompletionRequest[] = []
  const agent = new CompactionAgent({ model: 'gpt-test', summaryMaxTokens: 2_048, completion: request => { requests.push(request); return 'NEW: this pass.' } })
  const messages: Array<Record<string, unknown>> = [
    { role: 'user', content: '[CONTEXT COMPACTION — REFERENCE ONLY]\n\nEARLIER: a short summary.' },
    ...Array.from({ length: 20 }, (_, index) => ({ role: index % 2 ? 'assistant' : 'user', content: `step ${index} `.repeat(300) })),
    { role: 'user', content: 'latest request' },
  ]
  const compacted = await agent.summarizeMessages(messages)
  const summary = String(compacted.find(message => String(message.content).startsWith('[CONTEXT COMPACTION'))?.content)
  expect(summary).toContain('EARLIER: a short summary.')
  expect(summary).toContain('NEW: this pass.')
  expect(requests.some(request => request.prompt.includes('Earlier summary of this session'))).toBe(false)
})
