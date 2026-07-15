// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AgentSwitchTrigger as RuntimeAgentSwitchTrigger } from '../src/agents/orchestrator.js'
import { CompactionStrategy as RuntimeCompactionStrategy } from '../src/context/compactionStrategies.js'
import { ExecutionStatus as RuntimeExecutionStatus } from '../src/runtime/executionRegistry.js'
import {
  AgentSwitchTrigger,
  CompactionStrategy,
  ExecutionStatus,
  FUNCTION_CALLING_CAPABILITY,
  FunctionCallStrategy,
  createRequestFunctionCall,
  createStreamChunk,
  geminiContent,
  requestFunctionCallFromToolCall,
  streamChunkIsThinking,
  toolCallFromRequestFunctionCall,
  withRequestFunctionCall,
  type StreamingResponseType,
} from '../src/types/functionExecution.js'

test('function-execution types reuse the existing status, switching, capability, and compaction vocabularies', () => {
  expect(ExecutionStatus).toBe(RuntimeExecutionStatus)
  expect(AgentSwitchTrigger).toBe(RuntimeAgentSwitchTrigger)
  expect(CompactionStrategy).toBe(RuntimeCompactionStrategy)
  expect(FUNCTION_CALLING_CAPABILITY).toEqual({
    name: 'function_calling',
    description: 'Can use tools and function calls',
  })
  expect(FunctionCallStrategy).toEqual({
    SEQUENTIAL: 'sequential',
    PARALLEL: 'parallel',
    CONDITIONAL: 'conditional',
    PIPELINE: 'pipeline',
  })
})

test('request call factories synchronize ids, preserve canonical JSON arguments, and return replacement values for transitions', () => {
  const initial = createRequestFunctionCall({
    name: 'ReadFile',
    arguments: { path: 'README.md' },
    id: 'ignored-id',
    callId: 'authoritative-id',
    dependencies: ['prepare'],
    timeout: 10,
  })
  expect(initial).toEqual({
    name: 'ReadFile',
    arguments: { path: 'README.md' },
    id: 'authoritative-id',
    callId: 'authoritative-id',
    dependencies: ['prepare'],
    retryCount: 0,
    maxRetries: 3,
    status: 'pending',
    timeout: 10,
  })

  const completed = withRequestFunctionCall(initial, {
    status: ExecutionStatus.SUCCESS,
    result: { bytes: 42 },
    retryCount: 1,
  })
  expect(completed).toEqual({ ...initial, status: 'success', result: { bytes: 42 }, retryCount: 1 })
  expect(completed).not.toBe(initial)

  const fromToolCall = requestFunctionCallFromToolCall({
    id: 'call-tool',
    type: 'function',
    function: { name: 'ListFiles', arguments: { path: 'src' } },
  })
  expect(toolCallFromRequestFunctionCall(fromToolCall)).toEqual({
    id: 'call-tool',
    type: 'function',
    function: { name: 'ListFiles', arguments: { path: 'src' } },
  })
})

test('stream chunks normalize defaults and expose provider-neutral Gemini and reasoning helpers', () => {
  const chunk = createStreamChunk({
    agentId: 'agent-a',
    bufferedContent: '<think>inspect',
    content: 'fallback',
    chunk: { _result: { text: 'Gemini text' } },
    toolCalls: [{ id: 'call-1', type: 'function', functionName: 'ReadFile', arguments: '{"path":"README.md"}', isComplete: true }],
  })

  expect(chunk).toMatchObject({ type: 'stream_chunk', agentId: 'agent-a', reinvoked: false })
  expect(geminiContent(chunk)).toBe('Gemini text')
  expect(streamChunkIsThinking(chunk)).toBe(true)
  expect(streamChunkIsThinking(createStreamChunk({ bufferedContent: '<reason>x</reason>' }))).toBe(false)
  expect(geminiContent(createStreamChunk({ chunk: { _result: {} }, content: 'fallback' }))).toBe('fallback')
})

test('legacy event union remains discriminated without being confused with the current agent-loop events', () => {
  const events: StreamingResponseType[] = [
    { type: 'function_detection', message: 'Processing function calls...', agentId: 'agent-a' },
    { type: 'function_execution_start', functionName: 'ReadFile', functionId: 'call-1', progress: 'Reading', agentId: 'agent-a' },
    { type: 'reinvoke_signal', message: 'continue', agentId: 'agent-a' },
  ]

  expect(events.map(event => event.type)).toEqual([
    'function_detection',
    'function_execution_start',
    'reinvoke_signal',
  ])
})
