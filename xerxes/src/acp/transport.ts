// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { errorMessage } from '../executors/toolRegistry.js'
import { AcpServer } from './server.js'
import {
  ACP_JSON_RPC_ERRORS,
  acpJsonRpcFailure,
  acpJsonRpcSuccess,
  isAcpJsonRpcId,
  isAcpRecord,
  type AcpJsonRpcId,
  type AcpWireEvent,
} from './types.js'

export type AcpStdioWriter = (line: string) => void | Promise<void>

/** Matches the daemon socket frame cap: one NDJSON frame may not exceed 16 MiB. */
export const MAX_ACP_FRAME_BYTES = 16 * 1024 * 1024

/** Incrementally accounts for bytes in the current NDJSON frame. */
export class AcpFrameByteCounter {
  private currentBytes = 0
  private scannedBytes = 0

  constructor(private readonly maxBytes = MAX_ACP_FRAME_BYTES) {}

  append(chunk: Uint8Array): void {
    for (const byte of chunk) {
      this.scannedBytes += 1
      this.currentBytes = byte === 0x0a ? 0 : this.currentBytes + 1
    }
  }

  get exceedsLimit(): boolean {
    return this.currentBytes > this.maxBytes
  }

  /** Exposed so fragmented-input tests can assert accounting remains linear. */
  get bytesScanned(): number {
    return this.scannedBytes
  }
}

interface ParsedRequest {
  readonly canReply: boolean
  readonly id: AcpJsonRpcId
  readonly method: string
  readonly params: Record<string, unknown>
}

type Sender = (frame: unknown) => Promise<void>

/**
 * Newline-delimited JSON-RPC 2.0 transport for an ACP server.
 *
 * A `prompt` is intentionally dispatched on an independent async worker so
 * `cancel`, `respond_permission`, and `respond_question` frames remain
 * processable while the prompt emits incremental `session/update` events.
 */
export class StdioJsonRpcServer {
  private readonly promptShutdown = new AbortController()
  private readonly workers = new Set<Promise<void>>()
  private running = true
  private writeQueue: Promise<void> = Promise.resolve()

  constructor(private readonly server: AcpServer) {}

  async serve(input: ReadableStream<Uint8Array>, write: AcpStdioWriter): Promise<void> {
    const reader = input.getReader()
    const decoder = new TextDecoder()
    const send = this.serializedSender(write)
    const frameBytes = new AcpFrameByteCounter()
    let buffer = ''
    try {
      while (this.running) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }
        frameBytes.append(value)
        buffer += decoder.decode(value, { stream: true })
        buffer = await this.handleCompleteLines(buffer, send)
        if (this.running && frameBytes.exceedsLimit) {
          await send(acpJsonRpcFailure(
            null,
            ACP_JSON_RPC_ERRORS.invalidRequest,
            `frame exceeds maximum size of ${MAX_ACP_FRAME_BYTES} bytes`,
          ))
          this.running = false
          await reader.cancel().catch(() => undefined)
          break
        }
      }
      buffer += decoder.decode()
      if (this.running && buffer.trim()) {
        await this.handleLine(buffer.trim(), send)
      }
    } finally {
      reader.releaseLock()
      this.stopPrompts()
      await Promise.allSettled([...this.workers])
      await this.writeQueue
    }
  }

  private serializedSender(write: AcpStdioWriter): Sender {
    return frame => {
      const line = `${JSON.stringify(frame)}\n`
      const sent = this.writeQueue.then(() => write(line)).then(() => undefined)
      this.writeQueue = sent.catch(() => undefined)
      return sent
    }
  }

  private async handleCompleteLines(buffer: string, send: Sender): Promise<string> {
    let newline = buffer.indexOf('\n')
    while (newline >= 0) {
      const rawLine = buffer.slice(0, newline)
      buffer = buffer.slice(newline + 1)
      if (utf8ByteLength(rawLine) > MAX_ACP_FRAME_BYTES) {
        await send(acpJsonRpcFailure(
          null,
          ACP_JSON_RPC_ERRORS.invalidRequest,
          `frame exceeds maximum size of ${MAX_ACP_FRAME_BYTES} bytes`,
        ))
        this.running = false
        return buffer
      }
      const line = rawLine.trim()
      if (line) {
        await this.handleLine(line, send)
      }
      if (!this.running) {
        return buffer
      }
      newline = buffer.indexOf('\n')
    }
    return buffer
  }

  private async handleLine(line: string, send: Sender): Promise<void> {
    let value: unknown
    try {
      value = JSON.parse(line) as unknown
    } catch (error) {
      await send(acpJsonRpcFailure(null, ACP_JSON_RPC_ERRORS.parseError, `parse error: ${errorMessage(error)}`))
      return
    }
    if (!isAcpRecord(value) || value.jsonrpc !== '2.0' || typeof value.method !== 'string') {
      const id = isAcpRecord(value) && isAcpJsonRpcId(value.id) ? value.id : null
      await send(acpJsonRpcFailure(id, ACP_JSON_RPC_ERRORS.invalidRequest, 'invalid request'))
      return
    }

    const hasId = Object.hasOwn(value, 'id')
    if (hasId && !isAcpJsonRpcId(value.id)) {
      await send(acpJsonRpcFailure(null, ACP_JSON_RPC_ERRORS.invalidRequest, 'JSON-RPC request id must be a string, number, or null'))
      return
    }
    const id = hasId ? value.id as AcpJsonRpcId : null
    // JSON-RPC notifications are identified by an omitted id. Although null
    // ids are discouraged by the spec, they are still requests and receive a
    // null-id response for compatibility with clients that use them.
    const canReply = hasId
    if (value.params !== undefined && !isAcpRecord(value.params)) {
      if (canReply) {
        await send(acpJsonRpcFailure(id, ACP_JSON_RPC_ERRORS.invalidParams, 'params must be an object'))
      }
      return
    }
    const request: ParsedRequest = {
      id,
      canReply,
      method: value.method,
      params: value.params ?? {},
    }
    try {
      await this.dispatch(request, send)
    } catch (error) {
      if (request.canReply) {
        await send(acpJsonRpcFailure(request.id, ACP_JSON_RPC_ERRORS.internalError, `${errorMessage(error)}`))
      }
    }
  }

  private async dispatch(request: ParsedRequest, send: Sender): Promise<void> {
    const method = request.method.replaceAll('/', '_').toLowerCase()
    switch (method) {
      case 'initialize':
        await this.result(request, this.server.initialize(recordParameter(request.params.client_info)), send)
        return
      case 'list_tools':
      case 'tools_list':
        await this.result(request, this.server.listTools(), send)
        return
      case 'list_models':
      case 'models_list':
        await this.result(request, this.server.listModels(), send)
        return
      case 'open_session':
      case 'session_new':
      case 'session_open':
        {
          const model = modelParameter(request.params.model)
        await this.result(request, this.server.openSession(stringParameter(request.params.cwd), {
          ...(model !== undefined ? { model } : {}),
          ...(typeof request.params.title === 'string' ? { title: request.params.title } : {}),
        }), send)
        return
        }
      case 'list_sessions':
      case 'session_list':
        await this.result(request, this.server.listSessions(), send)
        return
      case 'set_model':
      case 'session_set_model':
        await this.result(request, this.server.setModel(stringParameter(request.params.session_id), modelParameter(request.params.model) ?? null), send)
        return
      case 'cancel':
      case 'session_cancel':
        await this.result(request, this.server.cancel(stringParameter(request.params.session_id)), send)
        return
      case 'close_session':
      case 'session_close':
        await this.result(request, this.server.closeSession(stringParameter(request.params.session_id)), send)
        return
      case 'respond_permission':
      case 'permission_respond':
        if (typeof request.params.allow !== 'boolean') {
          if (request.canReply) {
            await send(acpJsonRpcFailure(
              request.id,
              ACP_JSON_RPC_ERRORS.invalidParams,
              'respond_permission requires a boolean allow parameter',
            ))
          }
          return
        }
        await this.result(request, this.server.respondPermission(
          stringParameter(request.params.permission_id),
          request.params.allow,
        ), send)
        return
      case 'pending_permissions':
      case 'permission_pending':
        await this.result(request, this.server.pendingPermissions(), send)
        return
      case 'respond_question':
      case 'question_respond':
      case 'respond_input':
      case 'input_respond':
        await this.result(request, this.server.respondQuestion(
          stringParameter(request.params.input_id),
          stringParameter(request.params.answer),
        ), send)
        return
      case 'pending_questions':
      case 'question_pending':
      case 'pending_inputs':
        await this.result(request, this.server.pendingQuestions(), send)
        return
      case 'prompt':
      case 'session_prompt':
        await this.startPrompt(request, send)
        return
      case 'shutdown':
      case 'exit':
        this.running = false
        await this.result(request, { ok: true }, send)
        this.stopPrompts()
        return
      default:
        if (request.canReply) {
          await send(acpJsonRpcFailure(request.id, ACP_JSON_RPC_ERRORS.methodNotFound, `unknown method: ${request.method}`))
        }
    }
  }

  private async result(request: ParsedRequest, result: unknown, send: Sender): Promise<void> {
    if (!request.canReply) {
      return
    }
    await send(acpJsonRpcSuccess(request.id, result))
  }

  private async startPrompt(request: ParsedRequest, send: Sender): Promise<void> {
    const sessionId = stringParameter(request.params.session_id)
    const session = this.server.sessions.get(sessionId)
    if (!session) {
      if (request.canReply) {
        await send(acpJsonRpcFailure(request.id, ACP_JSON_RPC_ERRORS.invalidRequest, `unknown session: ${sessionId}`))
      }
      return
    }
    const text = typeof request.params.text === 'string'
      ? request.params.text
      : stringParameter(request.params.prompt)
    const worker = this.runPromptWorker(request, sessionId, text, send)
    this.workers.add(worker)
    void worker.then(
      () => this.workers.delete(worker),
      () => this.workers.delete(worker),
    )
  }

  private async runPromptWorker(request: ParsedRequest, sessionId: string, text: string, send: Sender): Promise<void> {
    let acceptsUpdates = true
    const prompt = this.server.prompt(sessionId, text, async event => {
      if (!acceptsUpdates) {
        return
      }
      await send(sessionUpdate(sessionId, request.canReply ? request.id : null, event))
    })
    const outcome = await Promise.race([
      prompt.then(
        result => ({ kind: 'result' as const, result }),
        error => ({ kind: 'error' as const, error }),
      ),
      aborted(this.promptShutdown.signal).then(async () => {
        // Let an already-settling handler publish its contractual response;
        // only detach workers that remain pending after shutdown cancellation.
        await new Promise(resolve => setTimeout(resolve, 0))
        return { kind: 'shutdown' as const }
      }),
    ])
    acceptsUpdates = false
    if (outcome.kind === 'shutdown' || !request.canReply) {
      return
    }
    if (outcome.kind === 'result') {
      await send(acpJsonRpcSuccess(request.id, outcome.result))
      return
    }
    await send(acpJsonRpcFailure(request.id, ACP_JSON_RPC_ERRORS.internalError, errorMessage(outcome.error)))
  }

  private stopPrompts(): void {
    this.server.shutdown()
    if (!this.promptShutdown.signal.aborted) {
      this.promptShutdown.abort()
    }
  }
}

/** Serve an ACP JSON-RPC endpoint from injected Bun/WHATWG stdio streams. */
export async function serveACPStdio(
  server: AcpServer,
  input: ReadableStream<Uint8Array>,
  write: AcpStdioWriter,
): Promise<void> {
  await new StdioJsonRpcServer(server).serve(input, write)
}

function aborted(signal: AbortSignal): Promise<void> {
  if (signal.aborted) {
    return Promise.resolve()
  }
  return new Promise(resolve => signal.addEventListener('abort', () => resolve(), { once: true }))
}

function utf8ByteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength
}

function recordParameter(value: unknown): Record<string, unknown> | undefined {
  return isAcpRecord(value) ? value : undefined
}

function stringParameter(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function modelParameter(value: unknown): string | null | undefined {
  if (value === null || typeof value === 'string') {
    return value
  }
  return undefined
}

function sessionUpdate(sessionId: string, requestId: AcpJsonRpcId, event: AcpWireEvent): Record<string, unknown> {
  return {
    jsonrpc: '2.0',
    method: 'session/update',
    params: {
      session_id: sessionId,
      request_id: requestId,
      event,
    },
  }
}
