// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { existsSync, mkdtempSync, readFileSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  COLORS,
  ColorFormatter,
  XerxesLogger as ConsoleLogger,
  configureConsoleLogger,
  createStreamCallback,
  getConsoleLogger,
  logAgentStart,
  logDelegation,
  logError,
  logRetry,
  logStep,
  logSuccess,
  logTaskComplete,
  logTaskStart,
  logThinking,
  logWarning,
  setVerbosity,
  type LogOutput,
} from '../src/logging/index.js'
import {
  type StructuredTraceAttributes,
  type StructuredTraceSpan,
  type StructuredTracer,
  XerxesLogger as StructuredLogger,
  configureLogging,
  getLogger as getStructuredLogger,
} from '../src/logging/structured.js'
import { StructuredLogging, XerxesLogger as RootConsoleLogger } from '../src/index.js'

class CapturedOutput implements LogOutput {
  readonly chunks: string[] = []

  constructor(readonly isTTY = false) {}

  write(chunk: string): void {
    this.chunks.push(chunk)
  }

  get text(): string {
    return this.chunks.join('')
  }
}

const fixedClock = (): Date => new Date('2026-07-13T12:34:56.000Z')

test('console logger formats levels and multiline context through an injected non-TTY stream', () => {
  const output = new CapturedOutput()
  const bound = { session_id: 'session-1' }
  const logger = new ConsoleLogger({ name: 'worker', level: 'INFO', stream: output, clock: fixedClock }).bind(bound)
  bound.session_id = 'mutated'

  expect(logger.debug('hidden')).toBeUndefined()
  const record = logger.info('first\nsecond', { tool: 'read_file' })
  expect(record).toMatchObject({ level: 'INFO', name: 'worker', context: { session_id: 'session-1', tool: 'read_file' } })
  expect(output.text).toBe(
    '(12:34:56 worker) first {"session_id":"session-1","tool":"read_file"}\n'
      + '(12:34:56 worker) second {"session_id":"session-1","tool":"read_file"}\n',
  )

  const formatter = new ColorFormatter({ color: true })
  expect(formatter.format({ timestamp: fixedClock().toISOString(), name: 'worker', level: 'INFO', message: 'ready' }))
    .toContain(COLORS.BLUE_PURPLE)
  logStep('plan', 'map the repository', 'GREEN', logger)
  expect(output.text).toContain('[plan] map the repository')
})

test('stream callback renderer keeps text and event banners on an injected stream', () => {
  const output = new CapturedOutput()
  const callback = createStreamCallback({ stream: output, color: false })
  callback({ type: 'text', text: 'Answering' })
  callback({
    type: 'tool_start',
    call: { id: 'call-1', type: 'function', function: { name: 'read_file', arguments: { path: 'README.md' } } },
  })
  callback({
    type: 'tool_end',
    result: { toolCallId: 'call-1', name: 'read_file', permitted: true, durationMs: 250, result: 'ok' },
  })

  expect(output.text).toBe('Answering\n🛠️ Calling read_file\n✅ read_file completed in 0.25s\n')
})

test('structured logger emits JSON records with immutable bound context and named domain events', () => {
  const output = new CapturedOutput()
  const context = { session_id: 'session-1' }
  const logger = new StructuredLogger({ name: 'structured', level: 'DEBUG', stream: output, clock: fixedClock }).bind(context)
  context.session_id = 'mutated'

  const record = logger.logFunctionCall({
    agentId: 'coder',
    functionName: 'search',
    arguments: { query: 'logging' },
    result: 'x'.repeat(205),
    duration: 0.5,
  })
  expect(record).toMatchObject({ event: 'Function call completed', event_name: 'function_call', session_id: 'session-1' })
  expect(Object.isFrozen(record)).toBeTrue()
  const functionCall = JSON.parse(output.chunks[0] ?? '') as Record<string, unknown>
  expect(functionCall).toMatchObject({
    timestamp: '2026-07-13T12:34:56.000Z',
    level: 'info',
    logger: 'structured',
    event: 'Function call completed',
    event_name: 'function_call',
    agent_id: 'coder',
    status: 'success',
  })
  expect(String(functionCall.result)).toHaveLength(200)

  logger.logLlmRequest({
    provider: 'openai', model: 'gpt-test', promptTokens: 3, completionTokens: 5, duration: 0.2,
    error: new Error('rate limited'),
  })
  const failed = JSON.parse(output.chunks[1] ?? '') as Record<string, unknown>
  expect(failed).toMatchObject({ level: 'error', event_name: 'llm_request', status: 'error', error: 'rate limited' })

  logger.setLevel('ERROR')
  expect(logger.logAgentSwitch({ fromAgent: 'coder', toAgent: 'reviewer' })).toBeUndefined()
  expect(new TextDecoder().decode(logger.getMetrics())).toContain('xerxes_function_calls_total')
})

test('structured human output and root exports remain explicitly available', () => {
  const output = new CapturedOutput()
  const logger = new StructuredLogger({ name: 'human', enableJson: false, stream: output, clock: fixedClock })
  logger.info('started', { ready: true })
  expect(output.text).toBe('2026-07-13T12:34:56.000Z - human - INFO - started {"ready":true}\n')

  const configured = configureLogging({ name: 'configured', stream: output })
  expect(getStructuredLogger()).toBe(configured)
  expect(RootConsoleLogger).toBe(ConsoleLogger)
  expect(StructuredLogging.XerxesLogger).toBe(StructuredLogger)
})

test('console shortcuts preserve non-TTY messages, level control, and singleton configuration', () => {
  const output = new CapturedOutput()
  const logger = new ConsoleLogger({ color: false, level: 'DEBUG', stream: output, clock: fixedClock })
  logThinking('planner', logger)
  logSuccess('saved', logger)
  logError('failed', logger)
  logWarning('careful', logger)
  logRetry(2, 3, 'temporary', logger)
  logDelegation('planner', 'coder', logger)
  logAgentStart(undefined, logger)
  logTaskStart('inspect', 'coder', logger)
  logTaskComplete('inspect', 1.25, logger)

  expect(output.text).toContain('(🧠 planner) is thinking...')
  expect(output.text).toContain('🚀 saved')
  expect(output.text).toContain('❌ failed')
  expect(output.text).toContain('⚠️ careful')
  expect(output.text).toContain('⏳ Retry 2/3: temporary')
  expect(output.text).toContain('📌 Delegation: planner → coder')
  expect(output.text).toContain('Agent is started.')
  expect(output.text).toContain('Task Started: inspect (Agent: coder)')
  expect(output.text).toContain('Task Completed: inspect (1.25s)')

  const configured = configureConsoleLogger({ color: false, level: 'INFO', stream: output, clock: fixedClock })
  setVerbosity('DEBUG')
  expect(getConsoleLogger()).toBe(configured)
  expect(configured.level).toBe('DEBUG')
})

test('structured logger writes rotating files, exports Prometheus metrics, and scopes host tracing', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'xerxes-logging-'))
  try {
    const output = new CapturedOutput()
    const tracer = new RecordingTracer()
    const logFile = join(directory, 'nested', 'xerxes.log')
    const logger = new StructuredLogger({
      name: 'telemetry',
      level: 'DEBUG',
      stream: output,
      clock: fixedClock,
      logFile,
      maxLogFileBytes: 80,
      maxLogFileBackups: 1,
      enableJson: false,
      enableTracing: true,
      tracer,
    })
    logger.logFunctionCall({
      agentId: 'coder',
      functionName: 'read_file',
      arguments: { path: 'README.md' },
      duration: 0.5,
    })
    logger.logFunctionCall({
      agentId: 'coder',
      functionName: 'read_file',
      arguments: {},
      error: new TypeError('bad input'),
      duration: 1,
    })
    logger.logAgentSwitch({ fromAgent: 'planner', toAgent: 'coder' })
    logger.logLlmRequest({ provider: 'openai', model: 'gpt-test', promptTokens: 3, completionTokens: 5, duration: 0.2 })
    logger.logMemoryOperation({ operation: 'add', memoryType: 'short', agentId: 'coder', entryCount: 2 })
    logger.logMemoryOperation({ operation: 'remove', memoryType: 'short', agentId: 'coder' })
    logger.info('force a second rotated record')

    const metrics = new TextDecoder().decode(logger.getMetrics())
    expect(metrics).toContain('xerxes_function_calls_total{agent_id="coder",function_name="read_file",status="success"} 1')
    expect(metrics).toContain('xerxes_function_calls_total{agent_id="coder",function_name="read_file",status="error"} 1')
    expect(metrics).toContain('xerxes_function_duration_seconds_count{agent_id="coder",function_name="read_file"} 2')
    expect(metrics).toContain('xerxes_switches_total{from_agent="planner",to_agent="coder"} 1')
    expect(metrics).toContain('xerxes_llm_tokens_total{model="gpt-test",provider="openai",type="prompt"} 3')
    expect(metrics).toContain('xerxes_memory_entries{agent_id="coder",memory_type="short"} 1')
    expect(metrics).toContain('xerxes_errors_total{component="function_executor",error_type="TypeError"} 1')
    expect(existsSync(logFile)).toBeTrue()
    expect(existsSync(`${logFile}.1`)).toBeTrue()
    expect(readFileSync(logFile, 'utf8').length + readFileSync(`${logFile}.1`, 'utf8').length).toBeGreaterThan(0)

    const success = logger.withSpan('one', { request_id: 'r1' }, span => {
      span?.setAttribute?.('tool', 'read_file')
      return 'done'
    })
    expect(success).toBe('done')
    expect(tracer.spans[0]).toMatchObject({
      name: 'one',
      attributes: { 'service.name': 'telemetry', request_id: 'r1', tool: 'read_file' },
      ended: true,
    })

    await expect(logger.withSpan('failing', {}, async () => {
      throw new Error('trace failure')
    })).rejects.toThrow('trace failure')
    expect(tracer.spans[1]).toMatchObject({ ended: true, exceptions: ['trace failure'], status: { code: 'error', message: 'trace failure' } })

    const noTrace = new StructuredLogger({ enableTracing: false, tracer })
    expect(noTrace.startSpan('disabled')).toBeUndefined()
  } finally {
    rmSync(directory, { recursive: true, force: true })
  }
})

class RecordingSpan implements StructuredTraceSpan {
  readonly attributes: Record<string, StructuredTraceAttributes[string]>
  readonly exceptions: string[] = []
  ended = false
  status: { code: 'error' | 'ok'; message?: string } | undefined

  constructor(readonly name: string, attributes: StructuredTraceAttributes | undefined) {
    this.attributes = { ...(attributes ?? {}) }
  }

  end(): void {
    this.ended = true
  }

  recordException(error: unknown): void {
    this.exceptions.push(error instanceof Error ? error.message : String(error))
  }

  setAttribute(name: string, value: StructuredTraceAttributes[string]): void {
    this.attributes[name] = value
  }

  setStatus(status: Readonly<{ code: 'error' | 'ok'; message?: string }>): void {
    this.status = { ...status }
  }
}

class RecordingTracer implements StructuredTracer {
  readonly spans: RecordingSpan[] = []

  startSpan(name: string, options?: Readonly<{ attributes?: StructuredTraceAttributes }>): StructuredTraceSpan {
    const span = new RecordingSpan(name, options?.attributes)
    this.spans.push(span)
    return span
  }
}
