// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { BedrockRuntimeServiceException } from '@aws-sdk/client-bedrock-runtime'
import { describe, expect, test } from 'bun:test'

import {
  BedrockConverseClient,
  bedrockFinishReason,
  buildBedrockConverseInput,
  buildBedrockThinkingFields,
  formatBedrockError,
  normalizeBedrockToolCallId,
  resolveBedrockConfig,
  resolveBedrockModel,
  type BedrockConverseInput,
  type BedrockConverseTransport,
  type BedrockEnv,
  type BedrockResolvedConfig,
} from '../src/llms/bedrock.js'
import { createLlmClient, type CompletionRequest, type LlmDelta } from '../src/llms/client.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

const EMPTY_ENV: BedrockEnv = {}

const SEARCH_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'search',
    description: 'Search the web.',
    parameters: { type: 'object', properties: { query: { type: 'string' } }, required: ['query'] },
  },
}

function claudeRequest(overrides: Partial<CompletionRequest> = {}): CompletionRequest {
  return {
    model: 'amazon-bedrock/anthropic.claude-opus-4-1-20250805-v1:0',
    messages: [{ role: 'user', content: 'hello' }],
    ...overrides,
  }
}

function novaRequest(overrides: Partial<CompletionRequest> = {}): CompletionRequest {
  return {
    model: 'amazon-bedrock/amazon.nova-lite-v1:0',
    messages: [{ role: 'user', content: 'hello' }],
    ...overrides,
  }
}

const resolvedClaude = resolveBedrockModel(claudeRequest(), EMPTY_ENV)
const resolvedNova = resolveBedrockModel(novaRequest(), EMPTY_ENV)
const resolvedOpus46 = resolveBedrockModel(
  { ...claudeRequest(), model: 'amazon-bedrock/anthropic.claude-opus-4-6-v1' },
  EMPTY_ENV,
)

describe('resolveBedrockConfig', () => {
  const base = (env: BedrockEnv, modelId = 'anthropic.claude-opus-4-1-20250805-v1:0', bearerToken?: string) => ({
    baseUrl: 'https://bedrock-runtime.us-east-1.amazonaws.com' as string | undefined,
    bearerToken,
    env,
    modelId,
  })

  test('pins the standard endpoint and defaults to us-east-1 with no ambient config', () => {
    const config = resolveBedrockConfig(base(EMPTY_ENV))
    expect(config.region).toBe('us-east-1')
    expect(config.endpoint).toBe('https://bedrock-runtime.us-east-1.amazonaws.com')
    expect(config.credentials).toBeUndefined()
    expect(config.token).toBeUndefined()
  })

  test('AWS_REGION routes the SDK and stops explicit endpoint pinning', () => {
    const config = resolveBedrockConfig(base({ AWS_REGION: 'eu-central-1' }))
    expect(config.region).toBe('eu-central-1')
    expect(config.endpoint).toBeUndefined()
  })

  test('an ARN model id takes its region from the ARN', () => {
    const config = resolveBedrockConfig({
      baseUrl: undefined,
      bearerToken: undefined,
      env: { AWS_REGION: 'eu-central-1' },
      modelId: 'arn:aws:bedrock:us-west-2:123456789012:inference-profile/us.anthropic.claude-opus-4-6-v1',
    })
    expect(config.region).toBe('us-west-2')
  })

  test('AWS_PROFILE suppresses env credentials so the SDK chain owns resolution', () => {
    const config = resolveBedrockConfig(base({
      AWS_PROFILE: 'prod',
      AWS_ACCESS_KEY_ID: 'AKIA-ENV',
      AWS_SECRET_ACCESS_KEY: 'secret',
    }))
    expect(config.profile).toBe('prod')
    expect(config.credentials).toBeUndefined()
  })

  test('env access keys become static credentials without a profile', () => {
    const config = resolveBedrockConfig(base({
      AWS_ACCESS_KEY_ID: 'AKIA-ENV',
      AWS_SECRET_ACCESS_KEY: 'secret',
      AWS_SESSION_TOKEN: 'token-1',
    }))
    expect(config.credentials).toEqual({
      accessKeyId: 'AKIA-ENV',
      secretAccessKey: 'secret',
      sessionToken: 'token-1',
    })
  })

  test('a bearer token switches to httpBearerAuth and outranks env keys', () => {
    const config = resolveBedrockConfig(
      base({ AWS_ACCESS_KEY_ID: 'AKIA-ENV', AWS_SECRET_ACCESS_KEY: 'secret' }, undefined, 'bedrock-api-key'),
    )
    expect(config.token).toEqual({ token: 'bedrock-api-key' })
    expect(config.authSchemePreference).toEqual(['httpBearerAuth'])
    expect(config.credentials).toBeUndefined()
  })

  test('AWS_BEDROCK_SKIP_AUTH pins dummy credentials and drops the bearer scheme', () => {
    const config = resolveBedrockConfig(base({ AWS_BEDROCK_SKIP_AUTH: '1' }, undefined, 'bedrock-api-key'))
    expect(config.credentials).toEqual({ accessKeyId: 'dummy-access-key', secretAccessKey: 'dummy-secret-key' })
    expect(config.token).toBeUndefined()
  })

  test('a custom (non-standard) endpoint is always used explicitly', () => {
    const config = resolveBedrockConfig({
      baseUrl: 'https://bedrock.vpc.internal:8443',
      bearerToken: undefined,
      env: { AWS_REGION: 'us-east-2' },
      modelId: 'anthropic.claude-opus-4-1-20250805-v1:0',
    })
    expect(config.endpoint).toBe('https://bedrock.vpc.internal:8443')
    expect(config.region).toBe('us-east-2')
  })
})

describe('buildBedrockConverseInput', () => {
  test('system blocks carry the cache point only for caching-capable models', () => {
    const messages = [
      { role: 'system' as const, content: 'be terse' },
      { role: 'user' as const, content: 'hello' },
    ]
    const claude = buildBedrockConverseInput(
      claudeRequest({ messages }),
      { env: EMPTY_ENV, model: resolvedClaude },
    ).system as Record<string, unknown>[]
    expect(claude).toHaveLength(2)
    expect(claude[0]).toEqual({ text: 'be terse' })
    expect(claude[1]?.cachePoint).toBeDefined()

    const nova = buildBedrockConverseInput(
      novaRequest({ messages }),
      { env: EMPTY_ENV, model: resolvedNova },
    ).system as Record<string, unknown>[]
    expect(nova).toHaveLength(1)
  })

  test('PI_CACHE_RETENTION=long sets a one-hour TTL on cache points', () => {
    const input = buildBedrockConverseInput(claudeRequest({
      messages: [
        { role: 'system', content: 'be terse' },
        { role: 'user', content: 'hello' },
      ],
    }), { env: { PI_CACHE_RETENTION: 'long' }, model: resolvedClaude })
    const system = input.system as Record<string, unknown>[]
    expect((system[1]?.cachePoint as Record<string, unknown>).ttl).toBe('1h')
  })

  test('tool config maps choices and omits entirely on none or no tools', () => {
    const auto = buildBedrockConverseInput(
      claudeRequest({ tools: [SEARCH_TOOL], toolChoice: 'auto' }),
      { env: EMPTY_ENV, model: resolvedClaude },
    )
    const autoConfig = auto.toolConfig as Record<string, unknown>
    expect(autoConfig.toolChoice).toEqual({ auto: {} })
    expect(autoConfig.tools).toEqual([{
      toolSpec: {
        name: 'search',
        description: 'Search the web.',
        inputSchema: { json: SEARCH_TOOL.function.parameters },
      },
    }])

    const anyChoice = buildBedrockConverseInput(
      claudeRequest({ tools: [SEARCH_TOOL], toolChoice: 'any' }),
      { env: EMPTY_ENV, model: resolvedClaude },
    )
    expect((anyChoice.toolConfig as Record<string, unknown>).toolChoice).toEqual({ any: {} })

    const none = buildBedrockConverseInput(
      claudeRequest({ tools: [SEARCH_TOOL], toolChoice: 'none' }),
      { env: EMPTY_ENV, model: resolvedClaude },
    )
    expect(none.toolConfig).toBeUndefined()
    expect(buildBedrockConverseInput(claudeRequest(), { env: EMPTY_ENV, model: resolvedClaude }).toolConfig)
      .toBeUndefined()
  })

  test('tool call ids are normalized for Bedrock and results group into one user message', () => {
    const input = buildBedrockConverseInput(claudeRequest({
      messages: [
        { role: 'user', content: 'search' },
        {
          role: 'assistant',
          content: '',
          tool_calls: [{
            id: 'call|abc$1',
            type: 'function',
            function: { name: 'search', arguments: { query: 'x' } },
          }],
        },
        { role: 'tool', tool_call_id: 'call|abc$1', content: 'result one' },
        { role: 'tool', tool_call_id: 'call|abc$1', content: 'result two', is_error: true },
      ],
    }), { env: EMPTY_ENV, model: resolvedClaude })

    const messages = input.messages as Record<string, unknown>[]
    const assistant = messages[1]
    expect(assistant?.role).toBe('assistant')
    expect(assistant?.content).toEqual([
      { toolUse: { toolUseId: 'call_abc_1', name: 'search', input: { query: 'x' } } },
    ])
    const grouped = messages[2]
    expect(grouped?.role).toBe('user')
    const blocks = grouped?.content as Record<string, unknown>[]
    // Two tool results plus the Claude cache point that trails them.
    expect(blocks).toHaveLength(3)
    expect(blocks[0]).toEqual({
      toolResult: { toolUseId: 'call_abc_1', content: [{ text: 'result one' }], status: 'success' },
    })
    expect(blocks[1]).toEqual({
      toolResult: { toolUseId: 'call_abc_1', content: [{ text: 'result two' }], status: 'error' },
    })
    expect(blocks[2]?.cachePoint).toBeDefined()
  })

  test('blank user text falls back to the Bedrock-required placeholder', () => {
    const input = buildBedrockConverseInput(
      claudeRequest({ messages: [{ role: 'user', content: '   ' }] }),
      { env: EMPTY_ENV, model: resolvedClaude },
    )
    const messages = input.messages as Record<string, unknown>[]
    expect(messages[0]?.content).toEqual([{ text: '<empty>' }])
  })

  test('budget-based Claude thinking expands the output ceiling and sets the beta', () => {
    const input = buildBedrockConverseInput(claudeRequest({
      maxTokens: 4096,
      thinking: { effort: 'medium', budgetTokens: 8192 },
    }), { env: EMPTY_ENV, model: resolvedClaude })
    // pi-ai math: ceiling = min(base + budget, modelMax 32000) = 12288.
    expect(input.inferenceConfig).toEqual({ maxTokens: 12_288 })
    expect(input.additionalModelRequestFields).toEqual({
      thinking: { type: 'enabled', budget_tokens: 8192, display: 'summarized' },
      anthropic_beta: ['interleaved-thinking-2025-05-14'],
    })
  })

  test('adaptive-thinking Claude models use output_config effort from the level map', () => {
    const input = buildBedrockConverseInput({
      ...claudeRequest(),
      model: 'amazon-bedrock/anthropic.claude-opus-4-6-v1',
      thinking: { effort: 'max' },
    }, { env: EMPTY_ENV, model: resolvedOpus46 })
    expect(input.additionalModelRequestFields).toEqual({
      thinking: { type: 'adaptive', display: 'summarized' },
      output_config: { effort: 'max' },
    })
    expect(input.inferenceConfig).toEqual({ maxTokens: 128_000 })
  })

  test('GovCloud targets omit the thinking display field', () => {
    const fields = buildBedrockThinkingFields({
      adaptiveThinking: false,
      capabilities: resolvedClaude.capabilities,
      govCloud: true,
      thinking: { effort: 'medium', budgetTokens: 2048 },
    })
    expect(fields).toEqual({
      thinking: { type: 'enabled', budget_tokens: 2048 },
      anthropic_beta: ['interleaved-thinking-2025-05-14'],
    })
  })

  test('non-Claude models get no thinking fields and no default output cap', () => {
    const input = buildBedrockConverseInput(novaRequest({
      thinking: { effort: 'high' },
    }), { env: EMPTY_ENV, model: resolvedNova })
    expect(input.additionalModelRequestFields).toBeUndefined()
    expect(input.inferenceConfig).toEqual({})
  })
})

describe('normalizeBedrockToolCallId', () => {
  test('strips unsupported characters and caps length at 64', () => {
    expect(normalizeBedrockToolCallId('call|abc$1')).toBe('call_abc_1')
    expect(normalizeBedrockToolCallId(`x${'y'.repeat(200)}`)).toHaveLength(64)
  })
})

describe('bedrockFinishReason', () => {
  test('maps Converse stop reasons onto the neutral vocabulary', () => {
    expect(bedrockFinishReason('end_turn')).toBe('stop')
    expect(bedrockFinishReason('stop_sequence')).toBe('stop')
    expect(bedrockFinishReason('max_tokens')).toBe('length')
    expect(bedrockFinishReason('model_context_window_exceeded')).toBe('length')
    expect(bedrockFinishReason('tool_use')).toBe('tool_calls')
    expect(() => bedrockFinishReason('guardrail_intervened')).toThrow(/guardrail_intervened/)
  })
})

describe('formatBedrockError', () => {
  test('prefixes SDK exceptions and hints at the data-retention docs', () => {
    const error = sdkException(
      'ValidationException',
      "data retention mode 'default' is not available",
      400,
    )
    const message = formatBedrockError(error)
    expect(message).toContain('Validation error:')
    expect(message).toContain('docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html')
    expect(formatBedrockError(new Error('plain'))).toBe('plain')
  })
})

/** Build a real SDK service exception so instanceof-gated formatting applies. */
function sdkException(name: string, message: string, status: number): BedrockRuntimeServiceException {
  const error = new BedrockRuntimeServiceException({
    message,
    name,
    $fault: 'client',
    $metadata: { httpStatusCode: status },
  })
  return error
}

/** Scripted transport: replays the given events without any network access. */
function scriptedTransport(
  events: readonly unknown[],
  options: { readonly sendError?: unknown } = {},
): { readonly transport: BedrockConverseTransport; readonly inputs: BedrockConverseInput[] } {
  const inputs: BedrockConverseInput[] = []
  return {
    inputs,
    transport: {
      async send(input) {
        inputs.push(input)
        if (options.sendError !== undefined) throw options.sendError
        return {
          stream: (async function* replay() {
            for (const event of events) yield event
          })(),
        }
      },
    },
  }
}

function collect(stream: AsyncIterable<LlmDelta>): Promise<LlmDelta[]> {
  return Array.fromAsync(stream)
}

describe('BedrockConverseClient streaming', () => {
  test('translates text, thinking, tool calls, usage, and the stop reason', async () => {
    const scripted = scriptedTransport([
      { messageStart: { role: 'assistant' } },
      { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'he' } } },
      { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'llo' } } },
      { contentBlockDelta: { contentBlockIndex: 1, delta: { reasoningContent: { reasoningText: { text: 'think', signature: 'sig-1' } } } } },
      { contentBlockStart: { contentBlockIndex: 2, start: { toolUse: { toolUseId: 'tu_1', name: 'search' } } } },
      { contentBlockDelta: { contentBlockIndex: 2, delta: { toolUse: { input: '{"query":' } } } },
      { contentBlockDelta: { contentBlockIndex: 2, delta: { toolUse: { input: '"bedrock"}' } } } },
      { contentBlockStop: { contentBlockIndex: 2 } },
      { messageStop: { stopReason: 'tool_use' } },
      { metadata: { usage: { inputTokens: 21, outputTokens: 7, cacheReadInputTokens: 5, cacheWriteInputTokens: 9, totalTokens: 42 } } },
    ])
    const client = new BedrockConverseClient({ env: EMPTY_ENV, createClient: () => scripted.transport })
    const deltas = await collect(client.stream(claudeRequest({
      tools: [SEARCH_TOOL],
      messages: [
        { role: 'system', content: 'be terse' },
        { role: 'user', content: 'hello' },
      ],
    })))

    expect(deltas.some(delta => delta.content === 'he')).toBe(true)
    expect(deltas.some(delta => delta.content === 'llo')).toBe(true)
    const thinking = deltas.find(delta => delta.thinking === 'think')
    expect(thinking?.thinking).toBe('think')
    expect(deltas.find(delta => delta.thinkingSignature === 'sig-1')?.thinkingSignature).toBe('sig-1')
    const toolDelta = deltas.find(delta => delta.toolCalls !== undefined)
    expect(toolDelta?.toolCalls).toEqual([{
      id: 'tu_1',
      type: 'function',
      function: { name: 'search', arguments: { query: 'bedrock' } },
    }])
    expect(deltas.find(delta => delta.finishReason !== undefined)?.finishReason).toBe('tool_calls')
    const usage = deltas.find(delta => delta.usage !== undefined)?.usage
    expect(usage).toEqual({ inputTokens: 21, outputTokens: 7, cacheReadTokens: 5, cacheCreationTokens: 9 })

    // The command input carried the system blocks and the tool spec.
    expect(Array.isArray(scripted.inputs[0]?.system)).toBe(true)
    expect(scripted.inputs[0]?.toolConfig).toBeDefined()
  })

  test('redacted reasoning arrives as a placeholder plus base64 signature', async () => {
    const payload = new Uint8Array([1, 2, 3])
    const scripted = scriptedTransport([
      { contentBlockDelta: { contentBlockIndex: 0, delta: { reasoningContent: { redactedContent: payload } } } },
      { messageStop: { stopReason: 'end_turn' } },
    ])
    const client = new BedrockConverseClient({ env: EMPTY_ENV, createClient: () => scripted.transport })
    const deltas = await collect(client.stream(novaRequest()))
    const redacted = deltas.find(delta => delta.thinking === '[Reasoning redacted]')
    expect(redacted?.thinkingSignature).toBe(Buffer.from(payload).toString('base64'))
  })

  test('a send failure becomes a prefixed ProviderError carrying the HTTP status', async () => {
    const sendError = sdkException('ThrottlingException', 'rate exceeded', 429)
    const scripted = scriptedTransport([], { sendError })
    const client = new BedrockConverseClient({ env: EMPTY_ENV, createClient: () => scripted.transport })
    await expect(client.stream(claudeRequest()).next()).rejects.toMatchObject({
      clientType: 'amazon-bedrock',
      message: expect.stringContaining('Throttling error: rate exceeded'),
      details: { status: 429 },
    })
  })

  test('a mid-stream throttling exception item rejects the stream', async () => {
    const throttling = sdkException('ThrottlingException', 'slow down', 429)
    const scripted = scriptedTransport([
      { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'partial' } } },
      { throttlingException: throttling },
    ])
    const client = new BedrockConverseClient({ env: EMPTY_ENV, createClient: () => scripted.transport })
    await expect(collect(client.stream(claudeRequest()))).rejects.toMatchObject({
      message: expect.stringContaining('Throttling error: slow down'),
    })
  })

  test('a stream without a stop reason is an error, not a silent success', async () => {
    const scripted = scriptedTransport([
      { contentBlockDelta: { contentBlockIndex: 0, delta: { text: 'orphan' } } },
    ])
    const client = new BedrockConverseClient({ env: EMPTY_ENV, createClient: () => scripted.transport })
    await expect(collect(client.stream(claudeRequest()))).rejects.toThrow(/without a stop reason/)
  })

  test('the client resolves bearer-token auth from AWS_BEARER_TOKEN_BEDROCK', async () => {
    let seenConfig: BedrockResolvedConfig | undefined
    const transport: BedrockConverseTransport = {
      async send() {
        return { stream: (async function* one() { yield { messageStop: { stopReason: 'end_turn' } } })() }
      },
    }
    const client = new BedrockConverseClient({
      env: { AWS_BEARER_TOKEN_BEDROCK: 'tok-1' },
      createClient: config => {
        seenConfig = config
        return transport
      },
    })
    await collect(client.stream(claudeRequest()))
    expect(seenConfig?.token).toEqual({ token: 'tok-1' })
    expect(seenConfig?.authSchemePreference).toEqual(['httpBearerAuth'])
  })
})

describe('provider wiring', () => {
  test('bedrock routes through the dedicated client with explicit prefixes and aliases', () => {
    const client = createLlmClient('bedrock/anthropic.claude-opus-4-1-20250805-v1:0')
    expect(client).toBeInstanceOf(BedrockConverseClient)
    const aliased = createLlmClient('anthropic.claude-opus-4-1-20250805-v1:0', { provider: 'amazon_bedrock' })
    expect(aliased).toBeInstanceOf(BedrockConverseClient)
  })
})
