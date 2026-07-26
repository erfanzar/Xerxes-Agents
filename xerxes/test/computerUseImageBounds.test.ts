// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  isScreenshotToolResult,
  omittedScreenshotMarker,
  supersedeScreenshotToolResults,
} from '../src/context/screenshotSuperseder.js'
import { ClientError, ConfigurationError, ValidationError } from '../src/core/errors.js'
import type { ToolExecutor } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { createAgentState, type StreamEvent } from '../src/streaming/events.js'
import { runTurn } from '../src/streaming/loop.js'
import {
  normalizeCaptureResult,
  type CaptureResult,
} from '../src/tools/computerUse/backend.js'
import {
  MacOSComputerUsePort,
  resolveComputerUseCaptureBounds,
  type MacOSCommandRunner,
  type MacOSComputerUsePortOptions,
} from '../src/tools/computerUse/macosPort.js'
import { formatCaptureResult } from '../src/tools/computerUse/tool.js'
import type { ChatMessage } from '../src/types/messages.js'
import type { ToolCall, ToolDefinition } from '../src/types/toolCalls.js'

const PNG_HEADER_B64 = 'iVBORw0KGgo='
const JPEG_HEADER_B64 = '/9j/4AAQSkZJRg=='

interface FakeRunner {
  readonly calls: string[][]
  readonly runner: MacOSCommandRunner
}

/** Runner whose sips convert pass fails, exercising the honest fallback. */
function fallbackRunner(): FakeRunner {
  const calls: string[][] = []
  const runner: MacOSCommandRunner = async argv => {
    calls.push([...argv])
    const executable = argv[0] ?? ''
    const script = argv[4] ?? ''
    if (executable.endsWith('osascript') && script.includes('NSScreen')) {
      return { code: 0, stderr: '', stdout: '1800,1169,2\n' }
    }
    if (executable.endsWith('sips') && argv.includes('-g')) {
      return { code: 0, stderr: '', stdout: '/tmp/xerxes-cua-test.png\n  pixelWidth: 3600\n  pixelHeight: 2338\n' }
    }
    if (executable.endsWith('sips') && argv.includes('--out')) {
      return { code: 1, stderr: 'no writable codec', stdout: '' }
    }
    return { code: 0, stderr: '', stdout: '' }
  }
  return { calls, runner }
}

function successRunner(): FakeRunner {
  const calls: string[][] = []
  const runner: MacOSCommandRunner = async argv => {
    calls.push([...argv])
    const executable = argv[0] ?? ''
    const script = argv[4] ?? ''
    if (executable.endsWith('osascript') && script.includes('NSScreen')) {
      return { code: 0, stderr: '', stdout: '1800,1169,2\n' }
    }
    if (executable.endsWith('sips') && argv.includes('-g')) {
      return { code: 0, stderr: '', stdout: '/tmp/xerxes-cua-test.png\n  pixelWidth: 3600\n  pixelHeight: 2338\n' }
    }
    return { code: 0, stderr: '', stdout: '' }
  }
  return { calls, runner }
}

function makePort(fake: FakeRunner, extra: MacOSComputerUsePortOptions = {}): MacOSComputerUsePort {
  return new MacOSComputerUsePort({
    fileExists: () => true,
    platform: 'darwin',
    readFile: async () => new Uint8Array([0x89, 0x50, 0x4e, 0x47]),
    removeFile: async () => undefined,
    runner: fake.runner,
    tmpDir: '/tmp',
    uniqueId: () => 'test',
    ...extra,
  })
}

test('capture bounds honor maxCaptureEdge and jpegQuality options in the sips pipeline', async () => {
  const fake = successRunner()
  const capture = await makePort(fake, { jpegQuality: 50, maxCaptureEdge: 800 }).capture({ mode: 'vision' })

  expect(capture.width).toBe(800)
  expect(capture.height).toBe(520)
  expect(capture.mediaType).toBe('image/jpeg')
  const convert = fake.calls.at(-1)
  expect(convert?.[0]).toBe('/usr/bin/sips')
  expect(convert).toContain('formatOptions')
  expect(convert?.[convert.indexOf('formatOptions') + 1]).toBe('50')
  expect(convert).toContain('520')
  expect(convert).toContain('800')
})

test('port rejects out-of-range capture bounds instead of silently clamping', () => {
  expect(() => makePort(successRunner(), { maxCaptureEdge: 10 })).toThrow(ValidationError)
  expect(() => makePort(successRunner(), { jpegQuality: 0 })).toThrow(ValidationError)
  expect(() => makePort(successRunner(), { jpegQuality: 101 })).toThrow(ValidationError)
})

test('capture falls back honestly: original PNG kept with a size warning when re-encode fails', async () => {
  const fake = fallbackRunner()
  const capture = await makePort(fake).capture({ mode: 'vision' })

  expect(capture.mediaType).toBe('image/png')
  expect(capture.width).toBe(3600)
  expect(capture.height).toBe(2338)
  expect(capture.warning).toContain('could not be downscaled')
  expect(capture.warning).toContain('3600x2338')

  // Clicks on the fallback full-size image map through the raw-pixel scale.
  await makePort(fake).capture({ mode: 'vision' })
})

test('resolveComputerUseCaptureBounds wires env and settings with env precedence', () => {
  expect(resolveComputerUseCaptureBounds({}, {})).toEqual({ jpegQuality: 70, maxCaptureEdge: 1568 })
  expect(resolveComputerUseCaptureBounds({ computer_use_max_edge: 1024 }, {})).toEqual({ jpegQuality: 70, maxCaptureEdge: 1024 })
  expect(resolveComputerUseCaptureBounds({}, { XERXES_COMPUTER_USE_MAX_EDGE: '800' })).toEqual({ jpegQuality: 70, maxCaptureEdge: 800 })
  expect(resolveComputerUseCaptureBounds({}, { XERXES_COMPUTER_USE_JPEG_QUALITY: '55' })).toEqual({ jpegQuality: 55, maxCaptureEdge: 1568 })
  // Environment wins over settings, matching XERXES_COMPUTER_USE precedence.
  expect(resolveComputerUseCaptureBounds({ computer_use_max_edge: 1024 }, { XERXES_COMPUTER_USE_MAX_EDGE: '800' }).maxCaptureEdge).toBe(800)
})

test('resolveComputerUseCaptureBounds fails typed on invalid values', () => {
  expect(() => resolveComputerUseCaptureBounds({}, { XERXES_COMPUTER_USE_MAX_EDGE: 'abc' })).toThrow(ConfigurationError)
  expect(() => resolveComputerUseCaptureBounds({}, { XERXES_COMPUTER_USE_MAX_EDGE: '100' })).toThrow(ConfigurationError)
  expect(() => resolveComputerUseCaptureBounds({}, { XERXES_COMPUTER_USE_JPEG_QUALITY: '0' })).toThrow(ConfigurationError)
  expect(() => resolveComputerUseCaptureBounds({ computer_use_jpeg_quality: 'high' }, {})).toThrow(ConfigurationError)
})

test('normalizeCaptureResult rejects oversized base64 with a typed error, never truncation', () => {
  const oversized: CaptureResult = {
    elements: [],
    height: 10,
    mode: 'vision',
    pngB64: 'A'.repeat(33 * 1024 * 1024),
    width: 10,
  }
  expect(() => normalizeCaptureResult(oversized)).toThrow(ClientError)
  expect(() => normalizeCaptureResult(oversized)).toThrow('invalid or oversized PNG base64 data')
})

test('normalizeCaptureResult sniffs real image bytes and rejects impostors', () => {
  const base = { elements: [], height: 10, mode: 'vision' as const, width: 10 }

  const png = normalizeCaptureResult({ ...base, pngB64: PNG_HEADER_B64 })
  expect(png.mediaType).toBe('image/png')
  const jpeg = normalizeCaptureResult({ ...base, pngB64: JPEG_HEADER_B64 })
  expect(jpeg.mediaType).toBe('image/jpeg')
  const declared = normalizeCaptureResult({ ...base, mediaType: 'image/jpeg', pngB64: JPEG_HEADER_B64 })
  expect(declared.mediaType).toBe('image/jpeg')

  // Non-image bytes are a typed boundary failure.
  expect(() => normalizeCaptureResult({ ...base, pngB64: Buffer.from('plain text, not an image').toString('base64') }))
    .toThrow(ClientError)
  // Declared type contradicting the sniffed bytes is rejected.
  expect(() => normalizeCaptureResult({ ...base, mediaType: 'image/png', pngB64: JPEG_HEADER_B64 }))
    .toThrow(ClientError)
  // Unsupported declared types are rejected.
  expect(() => normalizeCaptureResult({ ...base, mediaType: 'image/tiff', pngB64: PNG_HEADER_B64 }))
    .toThrow(ClientError)
})

test('formatCaptureResult emits the declared media type and surfaces capture warnings', () => {
  const result = formatCaptureResult({
    elements: [],
    height: 1018,
    mediaType: 'image/jpeg',
    mode: 'vision',
    pngB64: JPEG_HEADER_B64,
    pngBytesLength: 12,
    warning: 'screenshot could not be downscaled; full-size kept inline',
    width: 1568,
  })
  expect(JSON.stringify(result)).toContain('data:image/jpeg;base64,')
  if (!('_multimodal' in result)) throw new Error('expected a multimodal capture result')
  expect(result.text_summary).toContain('Warning: screenshot could not be downscaled')
})

function screenshotPayload(width: number, height: number, base64: string): string {
  return JSON.stringify({
    _multimodal: true,
    content: [
      { type: 'text', text: `Screen capture: ${width}x${height}` },
      { type: 'image_url', image_url: { url: `data:image/jpeg;base64,${base64}` } },
    ],
    text_summary: `Screen capture: ${width}x${height}`,
  })
}

test('supersedeScreenshotToolResults keeps only the latest capture inline', () => {
  const first = screenshotPayload(1568, 1018, 'A'.repeat(400))
  const second = screenshotPayload(800, 520, 'B'.repeat(200))
  const messages: ChatMessage[] = [
    { role: 'user', content: 'look at my screen' },
    { role: 'tool', content: first, name: 'computer_use', tool_call_id: 'call-1' },
    { role: 'tool', content: 'plain text result', name: 'ReadFile', tool_call_id: 'call-2' },
    { role: 'tool', content: second, name: 'computer_use', tool_call_id: 'call-3' },
  ]

  expect(isScreenshotToolResult(first)).toBe(true)
  expect(isScreenshotToolResult('plain text result')).toBe(false)
  expect(supersedeScreenshotToolResults(messages)).toBe(1)

  const [ , superseded, untouched, latest ] = messages
  expect(superseded?.role).toBe('tool')
  if (superseded?.role !== 'tool') throw new Error('unreachable')
  // 400 base64 chars decode to 300 bytes; dimensions come from the summary.
  expect(superseded.content).toBe('[screenshot omitted: 300 bytes, 1568x1018]')
  // Tool-call pairing and message identity are preserved.
  expect(superseded.tool_call_id).toBe('call-1')
  expect(superseded.name).toBe('computer_use')
  expect(untouched?.content).toBe('plain text result')
  expect(latest?.content).toBe(second)

  // A second sweep is a no-op: markers are not re-processed.
  expect(supersedeScreenshotToolResults(messages)).toBe(0)
  expect(latest?.content).toBe(second)
})

test('omittedScreenshotMarker degrades gracefully on unparseable payloads', () => {
  expect(omittedScreenshotMarker('{"_multimodal":true,')).toBe('[screenshot omitted: unknown bytes, unknown dimensions]')
  expect(omittedScreenshotMarker('[screenshot omitted: 1 bytes, 1x1]')).toBe('[screenshot omitted: 1 bytes, 1x1]')
})

const COMPUTER_USE: ToolDefinition = {
  type: 'function',
  function: {
    name: 'computer_use',
    description: 'Desktop control.',
    parameters: { type: 'object', properties: { action: { type: 'string' } } },
  },
}

function captureCall(id: string): ToolCall {
  return { id, type: 'function', function: { name: 'computer_use', arguments: { action: 'capture' } } }
}

async function collect(events: AsyncIterable<StreamEvent>): Promise<StreamEvent[]> {
  const result: StreamEvent[] = []
  for await (const event of events) result.push(event)
  return result
}

class TwoCapturesThenTextClient implements LlmClient {
  readonly requests: CompletionRequest[] = []

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request)
    if (this.requests.length <= 2) {
      yield { toolCalls: [captureCall(`call-${this.requests.length}`)], usage: { inputTokens: 5, outputTokens: 5 } }
      return
    }
    yield { content: 'done', usage: { inputTokens: 5, outputTokens: 5 } }
  }
}

test('streaming loop collapses superseded screenshots so only the latest capture stays inline', async () => {
  const state = createAgentState()
  const executor: ToolExecutor = {
    async execute(): Promise<string> {
      return screenshotPayload(1568, 1018, 'C'.repeat(800))
    },
  }

  await collect(runTurn({
    model: 'gpt-4o',
    permissionMode: 'accept-all',
    state,
    tools: [COMPUTER_USE],
    userMessage: 'watch the screen twice',
  }, {
    llm: new TwoCapturesThenTextClient(),
    toolExecutor: executor,
  }))

  const toolMessages = state.messages.filter(message => message.role === 'tool')
  expect(toolMessages).toHaveLength(2)
  const [superseded, latest] = toolMessages
  expect(superseded?.content).toBe('[screenshot omitted: 600 bytes, 1568x1018]')
  expect(typeof latest?.content === 'string' && latest.content.startsWith('{"_multimodal":true')).toBe(true)
  // Provider round 3 saw the first screenshot already collapsed.
  const finalRequest = state.messages
  expect(finalRequest.filter(message => typeof message.content === 'string'
    && (message.content as string).startsWith('{"_multimodal":true'))).toHaveLength(1)
})
