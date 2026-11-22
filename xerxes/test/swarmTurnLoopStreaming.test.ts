// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import type { ToolLoopBlockAuditInput } from '../src/audit/emitter.js'
import type { ToolExecutionContext } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import type { RunSafetyPort } from '../src/runtime/runSafety.js'
import type { BoundCommandSandbox } from '../src/security/commandSandbox.js'
import { createAgentState, type PermissionRequest, type StreamEvent } from '../src/streaming/events.js'
import {
  DEFAULT_MAX_IDENTICAL_FAILED_TOOL_ROUNDS,
  IDENTICAL_FAILED_TOOL_LOOP_PATTERN,
  IDENTICAL_FAILED_TOOL_NUDGE_THRESHOLD,
  roughPromptTokens,
  runTurn,
} from '../src/streaming/loop.js'
import { messagesToAnthropic, messagesToOpenAi } from '../src/streaming/messages.js'
import type { PermissionMode } from '../src/streaming/permissions.js'
import { ThinkingParser, splitThinkingTags } from '../src/streaming/thinkingParser.js'
import {
  parseRepairableJsonContainer,
  repairDeclaredArguments,
  repairExecCommandInputs,
  repairToolCall,
} from '../src/streaming/toolCallRepair.js'
import type { ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

async function collect(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const result: StreamEvent[] = []
  for await (const event of events) result.push(event)
  return result
}

const READ_FILE: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ReadFile',
    description: 'Read a file.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      required: ['path'],
      properties: {
        path: { type: 'string' },
        offset: { type: 'integer' },
      },
    },
  },
}

const EXEC_COMMAND: ToolDefinition = {
  type: 'function',
  function: {
    name: 'exec_command',
    description: 'Execute a command.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      required: ['cmd'],
      properties: {
        cmd: { type: 'string' },
        args: { type: 'array', items: { type: 'string' } },
        workdir: { type: 'string' },
        unsandboxed: { type: 'boolean' },
      },
    },
  },
}

const PLUGIN_MANAGE: ToolDefinition = {
  type: 'function',
  function: {
    name: 'PluginManageTool',
    description: 'Manage plugins.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      required: ['action'],
      properties: {
        action: { type: 'string' },
        path: { type: 'string' },
        skip_tests: { type: 'boolean' },
      },
    },
  },
}

const COMPUTER_USE: ToolDefinition = {
  type: 'function',
  function: {
    name: 'computer_use',
    description: 'Desktop interaction.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      required: ['action'],
      properties: {
        action: { type: 'string' },
        x: { type: 'integer' },
        y: { type: 'integer' },
      },
    },
  },
}

test('live permissionModeProvider transitions update runSafety disposition and commandSandbox mid-turn', async () => {
  let liveMode: PermissionMode = 'plan'
  const fakeSandbox = { kind: 'seatbelt' } as unknown as BoundCommandSandbox
  const observedContexts: Array<BoundCommandSandbox | undefined> = []
  const prompts: string[] = []
  const runSafety: RunSafetyPort = {
    toolDisposition: call => call.function.name === 'exec_command' ? 'allow' : undefined,
    commandSandbox: () => fakeSandbox,
    approved: () => undefined,
  }

  let round = 0
  const state = createAgentState()
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'plan',
    permissionModeProvider: () => liveMode,
    state,
    tools: [EXEC_COMMAND],
    userMessage: 'leave plan mode then run build',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        round += 1
        if (round === 1) {
          // Simulate ExitPlanModeTool or operator switching session mode to auto mid-turn.
          liveMode = 'auto'
          yield {
            toolCalls: [{
              id: 'call-sandboxed',
              type: 'function',
              function: { name: 'exec_command', arguments: { cmd: 'bun', args: ['build'] } },
            }],
          }
          return
        }
        if (round === 2) {
          // Switching from auto to accept-all mid-turn drops commandSandbox.
          liveMode = 'accept-all'
          yield {
            toolCalls: [{
              id: 'call-unsandboxed-yolo',
              type: 'function',
              function: { name: 'exec_command', arguments: { cmd: 'bun', args: ['build'] } },
            }],
          }
          return
        }
        yield { content: 'Done.' }
      },
    },
    permissionBroker: {
      async request(req) {
        prompts.push(req.description)
        return 'approve'
      },
    },
    runSafety,
    toolExecutor: {
      async execute(_call: ToolCall, context: ToolExecutionContext): Promise<string> {
        observedContexts.push(context.commandSandbox)
        return '{"exitCode":0}'
      },
    },
  }))

  // Round 1 ran under live `auto`: runSafety allowed it without prompting AND attached commandSandbox.
  expect(prompts).toEqual([])
  expect(observedContexts).toEqual([fakeSandbox, undefined])
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'completed' })
})

test('mixed unconfigured and configured tool calls keep complete tool_result pairing on early stop', async () => {
  const state = createAgentState()
  let executed = 0
  const events = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [READ_FILE],
    userMessage: 'mixed call stop',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        yield {
          toolCalls: [
            { id: 'call-hidden', type: 'function', function: { name: 'unconfigured_tool', arguments: {} } },
            { id: 'call-exposed', type: 'function', function: { name: 'ReadFile', arguments: { path: 'a.ts' } } },
          ],
        }
      },
    },
    onUnconfiguredToolCalls: () => 'stop',
    toolExecutor: {
      async execute(): Promise<string> {
        executed += 1
        return 'unexpected'
      },
    },
  }))

  expect(executed).toBe(0)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'unconfigured_tools' })
  const toolEnds = events.filter((event): event is Extract<StreamEvent, { type: 'tool_end' }> => event.type === 'tool_end')
  expect(toolEnds.map(event => event.result.toolCallId)).toEqual(['call-hidden', 'call-exposed'])
  expect(toolEnds[1]?.result.result).toBe('Cancelled before execution.')

  const toolMessages = state.messages.filter(message => message.role === 'tool')
  expect(toolMessages.map(message => message.tool_call_id)).toEqual(['call-hidden', 'call-exposed'])
  // Both Anthropic and OpenAI wire converters accept the resulting state cleanly.
  expect(messagesToAnthropic(state.messages)).toHaveLength(3)
  expect(messagesToOpenAi(state.messages)).toHaveLength(4)
})

test('a rejecting permissionBroker or mid-round consumer break never leaves orphaned assistant tool_calls', async () => {
  // 1. Permission broker rejects with AbortError when signal aborts during approval prompt.
  const abortController = new AbortController()
  const abortedState = createAgentState()
  const abortedEvents = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'manual',
    state: abortedState,
    tools: [READ_FILE],
    userMessage: 'approve then cancel',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        yield {
          toolCalls: [
            { id: 'call-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'one.ts' } } },
            { id: 'call-2', type: 'function', function: { name: 'ReadFile', arguments: { path: 'two.ts' } } },
          ],
        }
      },
    },
    permissionBroker: {
      async request(_req, sig): Promise<'approve' | 'reject'> {
        abortController.abort(new DOMException('User cancelled approval', 'AbortError'))
        throw sig?.reason ?? new DOMException('Aborted', 'AbortError')
      },
    },
  }, abortController.signal))

  expect(abortedEvents.at(-1)).toMatchObject({ type: 'turn_done', reason: 'aborted' })
  expect(abortedState.messages.filter(message => message.role === 'tool').map(message => message.tool_call_id)).toEqual([
    'call-1',
    'call-2',
  ])

  // 2. Consumer detaches (breaks out of for-await) right on tool_start.
  const detachedState = createAgentState()
  const stream = runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state: detachedState,
    tools: [READ_FILE],
    userMessage: 'detach on tool_start',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        yield {
          toolCalls: [
            { id: 'detach-1', type: 'function', function: { name: 'ReadFile', arguments: { path: 'a.ts' } } },
            { id: 'detach-2', type: 'function', function: { name: 'ReadFile', arguments: { path: 'b.ts' } } },
          ],
        }
      },
    },
    toolExecutor: { async execute() { return 'ok' } },
  })
  for await (const event of stream) {
    if (event.type === 'tool_start') break
  }
  expect(detachedState.messages.filter(message => message.role === 'tool').map(message => message.tool_call_id)).toEqual([
    'detach-1',
    'detach-2',
  ])
})

test('streamed multi-chunk and interleaved reasoning attaches complete per-call blocks to tool_start and tool_end', async () => {
  let round = 0
  const events = await collect(runTurn({
    model: 'claude-sonnet-4-5',
    permissionMode: 'accept-all',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'inspect two files',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        round += 1
        if (round === 1) {
          yield { thinking: 'First, ' }
          yield { thinking: 'inspect ' }
          yield { thinking: 'alpha.ts.' }
          yield {
            toolCalls: [{ id: 'call-alpha', type: 'function', function: { name: 'ReadFile', arguments: { path: 'alpha.ts' } } }],
          }
          yield { thinking: 'Next, ' }
          yield { thinking: 'inspect beta.ts.' }
          yield {
            toolCalls: [{ id: 'call-beta', type: 'function', function: { name: 'ReadFile', arguments: { path: 'beta.ts' } } }],
          }
          return
        }
        yield { content: 'Both inspected.' }
      },
    },
    toolExecutor: { async execute() { return 'contents' } },
  }))

  const starts = events.filter((event): event is Extract<StreamEvent, { type: 'tool_start' }> => event.type === 'tool_start')
  const ends = events.filter((event): event is Extract<StreamEvent, { type: 'tool_end' }> => event.type === 'tool_end')
  expect(starts.map(event => [event.call.id, event.reasoning])).toEqual([
    ['call-alpha', 'First, inspect alpha.ts.'],
    ['call-beta', 'Next, inspect beta.ts.'],
  ])
  expect(ends.map(event => [event.result.toolCallId, event.reasoning])).toEqual([
    ['call-alpha', 'First, inspect alpha.ts.'],
    ['call-beta', 'Next, inspect beta.ts.'],
  ])
})

test('drainSteer("final") runs only when the turn actually ends, never before empty-reply or objective continuations', async () => {
  const boundaries: Array<'start' | 'final' | undefined> = []
  const pendingSteers: string[] = []
  let round = 0
  const state = createAgentState()
  const events = await collect(runTurn({
    interactionMode: 'objective',
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [EXEC_COMMAND],
    userMessage: 'verify tests pass',
  }, {
    drainSteer: boundary => {
      boundaries.push(boundary)
      if (boundary === 'final') return pendingSteers.splice(0)
      return pendingSteers.splice(0)
    },
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        round += 1
        if (round === 1) {
          // Steer arrives while the model emits an unverified narrative stop in objective mode.
          pendingSteers.push('focus on unit tests only')
          yield { content: 'I will check things soon.' }
          return
        }
        if (round === 2) {
          yield {
            toolCalls: [{
              id: 'call-test',
              type: 'function',
              function: { name: 'exec_command', arguments: { cmd: 'bun', args: ['test'] } },
            }],
          }
          return
        }
        // Steer arrives during the true final verified answer.
        pendingSteers.push('save a note for next turn')
        yield { content: 'All tests pass.' }
      },
    },
    toolExecutor: {
      async execute(): Promise<string> {
        return JSON.stringify({ exitCode: 0, stdout: '1 pass', stderr: '', timedOut: false })
      },
    },
  }))

  // Round 1 continued via objective gate, so 'final' was NOT drained after round 1;
  // instead round 2 drained the steer live as `[steer from user]` AFTER `[Objective gate]`.
  expect(boundaries).toEqual([undefined, undefined, undefined, 'final'])
  const userContents = state.messages
    .filter(message => message.role === 'user')
    .map(message => String(message.content))
  const gateIndex = userContents.findIndex(text => text.startsWith('[Objective gate]'))
  const liveSteerIndex = userContents.findIndex(text => text === '[steer from user]\nfocus on unit tests only')
  const savedSteerIndex = userContents.findIndex(text => text === '[steer from user saved for next turn]\nsave a note for next turn')
  expect(gateIndex).toBeGreaterThan(0)
  expect(liveSteerIndex).toBeGreaterThan(gateIndex)
  expect(savedSteerIndex).toBeGreaterThan(liveSteerIndex)
  expect(events.some(event => event.type === 'text' && event.text.includes('Steer saved for next turn: focus on unit tests only'))).toBe(false)
  expect(events.some(event => event.type === 'text' && event.text.includes('Steer saved for next turn: save a note for next turn'))).toBe(true)
})

test('request-only systemPrompt sizes first-chunk prefill budget and metadata-only deltas preserve remaining prefill wait', async () => {
  expect(roughPromptTokens([{ role: 'user', content: 'hi' }], undefined, 's'.repeat(12_000))).toBeGreaterThan(3_000)

  const sleep = (ms: number, sig?: AbortSignal) => new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    sig?.addEventListener('abort', () => { clearTimeout(timer); reject(sig.reason) }, { once: true })
  })

  let calls = 0
  const events = await collect(runTurn({
    model: 'local-model',
    state: createAgentState(),
    systemPrompt: 's'.repeat(12_000),
    systemPromptRequestOnly: true,
    userMessage: 'hi',
  }, {
    llm: {
      async *stream(_req: CompletionRequest, sig?: AbortSignal): AsyncGenerator<LlmDelta> {
        calls += 1
        // Emit an immediate metadata-only frame (like message_start) before prefill finishes.
        yield { usage: { inputTokens: 3_000, outputTokens: 0 } }
        await sleep(120, sig)
        yield { content: 'answered after prefill', usage: { inputTokens: 3_000, outputTokens: 4 } }
      },
    },
    retryDelays: [0],
    streamInactivityTimeoutMs: 35,
  }))

  expect(calls).toBe(1)
  expect(events.some(event => event.type === 'provider_retry')).toBe(false)
  expect(events.find(event => event.type === 'provider_wait' && event.active)).toMatchObject({
    type: 'provider_wait',
    active: true,
    promptTokens: expect.any(Number),
  })
})

test('decode throughput accounts for streamed reasoning accurately without progressive toolCall byte inflation', async () => {
  // 1. Streamed reasoning + content: 200 reasoning tokens + 100 visible tokens over 3,000ms = 100 tok/s.
  const streamedTimes = [1_000, 1_200, 3_200, 4_200]
  const streamedEvents = await collect(runTurn({
    model: 'deepseek/deepseek-reasoner',
    state: createAgentState(),
    userMessage: 'reason and answer',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        yield { thinking: 'r'.repeat(800) }
        yield { content: 'c'.repeat(400), usage: { inputTokens: 20, outputTokens: 300, reasoningTokens: 200 } }
      },
    },
    now: () => streamedTimes.shift() ?? 4_200,
  }))
  const streamedUsage = streamedEvents.find(event => event.type === 'usage_update')
  expect(streamedUsage).toMatchObject({ type: 'usage_update', tokensPerSecond: 100 })

  // 2. Hidden reasoning with a tiny 5-word summary at t=1.2s, visible answer at t=5.2s..6.2s:
  //    100 visible tokens over 1,000ms = 100 tok/s (not dragged down by the 4s hidden reasoning gap).
  const summaryTimes = [1_000, 1_200, 5_200, 6_200]
  const summaryEvents = await collect(runTurn({
    model: 'openai/gpt-5',
    state: createAgentState(),
    userMessage: 'summarize reasoning then answer',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        yield { thinking: 'Short summary.' }
        yield { content: 'v'.repeat(400), usage: { inputTokens: 20, outputTokens: 2_100, reasoningTokens: 2_000 } }
      },
    },
    now: () => summaryTimes.shift() ?? 6_200,
  }))
  const summaryUsage = summaryEvents.find(event => event.type === 'usage_update')
  expect(summaryUsage).toMatchObject({ type: 'usage_update', tokensPerSecond: 100 })

  // 3. Progressive toolCalls snapshots with hidden unstreamed tokens do not quadratically inflate byte bounds.
  const progressiveTimes = [1_000, 1_100, 2_100]
  const progressiveEvents = await collect(runTurn({
    model: 'claude-code/opus',
    permissionMode: 'accept-all',
    state: createAgentState(),
    tools: [READ_FILE],
    userMessage: 'progressive tool deltas',
  }, {
    llm: {
      async *stream(req): AsyncGenerator<LlmDelta> {
        if (req.messages.some(message => message.role === 'tool')) {
          yield { content: 'done' }
          return
        }
        yield { content: 'ok' }
        for (let step = 1; step <= 40; step += 1) {
          yield {
            toolCalls: [{
              id: 'call-prog',
              type: 'function',
              function: { name: 'ReadFile', arguments: { path: 'a'.repeat(step) } },
            }],
          }
        }
        yield { usage: { inputTokens: 10, outputTokens: 2_000 } }
      },
    },
    now: () => progressiveTimes.shift() ?? 2_500,
    toolExecutor: { async execute() { return 'body' } },
  }))
  const firstUsage = progressiveEvents.find(event => event.type === 'usage_update')
  expect(firstUsage).not.toHaveProperty('tokensPerSecond')
})

test('ThinkingParser keeps inline-code closing tags inside reasoning and prose across arbitrary chunk splits', () => {
  const reasoningSample = '<think>Verify how `</think>` and `</thinking>` behave in chunks.</think>Final visible reply.'
  for (const chunkSize of [reasoningSample.length, 9, 3, 1]) {
    const parser = new ThinkingParser()
    const parts = []
    for (let index = 0; index < reasoningSample.length; index += chunkSize) {
      parts.push(...parser.process(reasoningSample.slice(index, index + chunkSize)))
    }
    parts.push(...parser.process(''))
    expect(parts.filter(part => part.type === 'thinking').map(part => part.text).join('')).toBe(
      'Verify how `</think>` and `</thinking>` behave in chunks.',
    )
    expect(parts.filter(part => part.type === 'text').map(part => part.text).join('')).toBe('Final visible reply.')
  }

  const proseMention = 'DeepSeek opens reasoning with <think> tags and closes them with `</think>`.'
  for (const chunkSize of [proseMention.length, 6, 1]) {
    const parser = new ThinkingParser()
    const parts = []
    for (let index = 0; index < proseMention.length; index += chunkSize) {
      parts.push(...parser.process(proseMention.slice(index, index + chunkSize)))
    }
    parts.push(...parser.process(''))
    expect(parts.filter(part => part.type === 'thinking')).toEqual([])
    expect(parts.map(part => part.text).join('')).toBe(proseMention)
  }
  expect(splitThinkingTags(proseMention)).toEqual({ visible: proseMention, thinking: '' })
})

test('schema-aware repairToolCall normalizes aliases, scalar literals, and fenced/trailing-comma JSON before permission gates', async () => {
  expect(parseRepairableJsonContainer('```json\n["-la", "src",]\n```', 'array')).toEqual(['-la', 'src'])
  expect(parseRepairableJsonContainer('{"a": 1, "b": "keep ,] literal", }', 'object')).toEqual({
    a: 1,
    b: 'keep ,] literal',
  })
  expect(repairExecCommandInputs({ cmd: 'wc -l', args: '```json\n["a.txt",]\n```' })).toEqual({
    cmd: 'wc',
    args: ['-l', 'a.txt'],
  })
  expect(repairDeclaredArguments({ file_path: 'README.md', offset: '10' }, READ_FILE.function.parameters)).toEqual({
    path: 'README.md',
    offset: 10,
  })
  // Near-miss scalar strings remain untouched so validation still rejects them loudly.
  expect(repairDeclaredArguments({ action: 'install', skip_tests: 'yes' }, PLUGIN_MANAGE.function.parameters)).toEqual({
    action: 'install',
    skip_tests: 'yes',
  })
  // Without a schema, non-exec_command calls stay identical references.
  const untouched: ToolCall = { id: 'c', type: 'function', function: { name: 'ReadFile', arguments: { file_path: 'a.ts' } } }
  expect(repairToolCall(untouched)).toBe(untouched)

  // End-to-end: permission prompts see coerced `skip_tests: true` and numeric `(x, y)` coordinates,
  // and `exec_command({ command: "wc -l" })` repairs into read-only argv that runs in `auto` without prompting.
  const requests: PermissionRequest[] = []
  let round = 0
  await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'auto',
    state: createAgentState(),
    tools: [PLUGIN_MANAGE, COMPUTER_USE, EXEC_COMMAND],
    userMessage: 'check pre-gate repairs',
  }, {
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        round += 1
        if (round === 1) {
          yield {
            toolCalls: [
              {
                id: 'call-plugin',
                type: 'function',
                function: { name: 'PluginManageTool', arguments: { action: 'install', path: '/tmp/p', skip_tests: 'true' } },
              },
              {
                id: 'call-click',
                type: 'function',
                function: { name: 'computer_use', arguments: { action: 'click', x: '120', y: '340' } },
              },
              {
                id: 'call-wc',
                type: 'function',
                function: { name: 'exec_command', arguments: { command: 'wc -l', args: '["README.md",]' } },
              },
            ],
          }
          return
        }
        yield { content: 'All checked.' }
      },
    },
    permissionBroker: {
      async request(req) {
        requests.push(req)
        return 'reject'
      },
    },
    toolExecutor: { async execute() { return '{"exitCode":0}' } },
  }))

  // `wc -l README.md` was auto-allowed after alias + JSON + argv repair; only plugin & click prompted.
  expect(requests).toHaveLength(2)
  expect(requests[0]?.description).toContain('WITHOUT TESTS (skip_tests requested).')
  expect(requests[1]?.description).toBe('Desktop: click at (120, 340)')
})

test('identical failing tool-call loop guard nudges on repeat and stops runaway loops while preserving recovery', async () => {
  // 1. Model repeats identical failing call twice, receives harness nudge, adjusts path on round 3, and succeeds!
  let recoveryRound = 0
  const recoveryRequests: CompletionRequest[] = []
  const recoveryState = createAgentState()
  const recoveryEvents = await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state: recoveryState,
    tools: [READ_FILE],
    userMessage: 'read the config',
  }, {
    llm: {
      async *stream(req): AsyncGenerator<LlmDelta> {
        recoveryRequests.push(req)
        recoveryRound += 1
        if (recoveryRound <= IDENTICAL_FAILED_TOOL_NUDGE_THRESHOLD) {
          yield {
            toolCalls: [{
              id: `fail-${recoveryRound}`,
              type: 'function',
              function: { name: 'ReadFile', arguments: { path: 'missing.json' } },
            }],
          }
          return
        }
        if (recoveryRound === IDENTICAL_FAILED_TOOL_NUDGE_THRESHOLD + 1) {
          yield {
            toolCalls: [{
              id: 'ok-3',
              type: 'function',
              function: { name: 'ReadFile', arguments: { path: 'present.json' } },
            }],
          }
          return
        }
        yield { content: 'Recovered with present.json.' }
      },
    },
    toolExecutor: {
      async execute(call): Promise<string> {
        if (call.function.arguments.path === 'missing.json') {
          throw new Error('ENOENT: missing.json does not exist')
        }
        return '{"ok":true}'
      },
    },
  }))

  expect(recoveryRound).toBe(4)
  expect(JSON.stringify(recoveryRequests[2]?.messages)).toContain('You repeated the exact same failing call to ReadFile')
  expect(recoveryEvents.at(-1)).toMatchObject({ type: 'turn_done', reason: 'completed' })

  // 2. Stuck model ignores the nudge and repeats the exact failing call (with reordered keys) forever -> stops at ceiling.
  let stuckRounds = 0
  const audits: ToolLoopBlockAuditInput[] = []
  const stuckState = createAgentState()
  const stuckEvents = await collect(runTurn({
    agentId: 'worker-1',
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    sessionId: 'sess-loop',
    state: stuckState,
    tools: [READ_FILE],
    userMessage: 'stuck reading missing file',
  }, {
    auditToolLoopBlock: input => { audits.push(input) },
    llm: {
      async *stream(): AsyncGenerator<LlmDelta> {
        stuckRounds += 1
        const argumentsObject = stuckRounds % 2 === 0
          ? { offset: 0, path: 'missing.json' }
          : { path: 'missing.json', offset: 0 }
        yield {
          toolCalls: [{
            id: `stuck-${stuckRounds}`,
            type: 'function',
            function: { name: 'ReadFile', arguments: argumentsObject },
          }],
        }
      },
    },
    toolExecutor: {
      async execute(): Promise<string> {
        throw new Error('ENOENT: missing.json')
      },
    },
  }))

  expect(stuckRounds).toBe(DEFAULT_MAX_IDENTICAL_FAILED_TOOL_ROUNDS)
  expect(stuckEvents.at(-1)).toMatchObject({ type: 'turn_done', reason: 'tool_budget_exhausted' })
  expect(stuckEvents.filter(event => event.type === 'text').map(event => event.text).join('')).toContain(
    `repeated the identical failing ReadFile call ${DEFAULT_MAX_IDENTICAL_FAILED_TOOL_ROUNDS} consecutive rounds`,
  )
  expect(audits).toEqual([{
    agentId: 'worker-1',
    count: DEFAULT_MAX_IDENTICAL_FAILED_TOOL_ROUNDS,
    pattern: IDENTICAL_FAILED_TOOL_LOOP_PATTERN,
    sessionId: 'sess-loop',
    toolName: 'ReadFile',
  }])
  expect(stuckState.messages.filter(message => message.role === 'tool')).toHaveLength(DEFAULT_MAX_IDENTICAL_FAILED_TOOL_ROUNDS)

  // 3. Invalid maxIdenticalFailedToolRounds fails fast with RangeError.
  await expect(collect(runTurn({
    maxIdenticalFailedToolRounds: 1.5,
    model: 'gpt-4o',
    state: createAgentState(),
    userMessage: 'bad config',
  }, {
    llm: { async *stream() { yield { content: 'unreachable' } } },
  }))).rejects.toThrow('maxIdenticalFailedToolRounds must be a safe integer')
})

test('aborting during provider_wait microtask still enters the constructed stream generator once before settling aborted', async () => {
  const controller = new AbortController()
  let enteredStreamBody = false
  const events: StreamEvent[] = []
  for await (const event of runTurn({
    model: 'gpt-4o',
    state: createAgentState(),
    userMessage: 'prompt before immediate EOF abort',
  }, {
    llm: {
      async *stream(_req: CompletionRequest, sig?: AbortSignal): AsyncGenerator<LlmDelta> {
        enteredStreamBody = true
        sig?.throwIfAborted()
        yield { content: 'unreachable' }
      },
    },
    retryDelays: [],
  }, controller.signal)) {
    events.push(event)
    if (event.type === 'provider_wait' && event.active) {
      controller.abort(new Error('ACP prompt cancelled'))
    }
  }

  expect(enteredStreamBody).toBe(true)
  expect(events.at(-1)).toMatchObject({ type: 'turn_done', reason: 'aborted' })
})

