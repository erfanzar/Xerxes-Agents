// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Buffer } from 'node:buffer'

import { ClientError } from '../../core/errors.js'
import { MCPClient } from '../../mcp/client.js'
import type { MCPToolCallResult } from '../../mcp/types.js'
import type { JsonObject } from '../../types/toolCalls.js'
import {
  ComputerUseUnavailableError,
  type ActionResult,
  type CaptureRequest,
  type CaptureResult,
  type ClickRequest,
  type ComputerUseBackend,
  type DragRequest,
  type KeyRequest,
  type ScrollRequest,
  type SetValueRequest,
  type TextRequest,
  type UIElement,
} from './backend.js'

const CUA_DRIVER_INSTALL_HINT = [
  'No cua-driver backend is configured.',
  'Install and run cua-driver behind an explicitly configured MCP transport, then inject CuaBackend into ComputerUseSession.',
  'The Bun runtime never downloads or auto-installs desktop automation software.',
].join(' ')

const WINDOW_LINE = /^-\s+(.+?)\s+\(pid\s+(\d+)\)\s+.*\[window_id:\s+(\d+)\]/gm
const ELEMENT_LINE = /^\s*(?:-\s+)?\[(\d+)\]\s+(\w+)(?:\s+"([^"]*)"|(?:\s+\(\d+\))?\s+id=([^\s\[\]]*))?/gm

/**
 * Narrow MCP boundary required by the CUA adapter.
 *
 * It is deliberately injected. This module does not inspect PATH, launch a
 * shell command, or install cua-driver. A host can use MCPClient below after
 * it has deliberately supplied a server configuration.
 */
export interface CuaDriverMcpPort {
  callTool(name: string, arguments_: JsonObject, signal?: AbortSignal): MCPToolCallResult | Promise<MCPToolCallResult>
  isAvailable(signal?: AbortSignal): boolean | Promise<boolean>
  start?(signal?: AbortSignal): void | Promise<void>
  stop?(signal?: AbortSignal): void | Promise<void>
}

export interface MCPClientCuaDriverPortOptions {
  /**
   * Host-defined provision check. It must not perform a privileged desktop
   * action. The default only means that the caller deliberately configured an
   * MCP client; actual connection failures still make start fail explicitly.
   */
  readonly isProvisioned?: (signal?: AbortSignal) => boolean | Promise<boolean>
}

/**
 * Adapter for the repository's MCP client after a host has chosen its process
 * command and security policy. It is not registered or constructed implicitly.
 */
export class MCPClientCuaDriverPort implements CuaDriverMcpPort {
  constructor(
    private readonly client: Pick<MCPClient, 'callTool' | 'connect' | 'disconnect'>,
    private readonly options: MCPClientCuaDriverPortOptions = {},
  ) {}

  async isAvailable(signal?: AbortSignal): Promise<boolean> {
    throwIfAborted(signal)
    return this.options.isProvisioned === undefined ? true : Boolean(await this.options.isProvisioned(signal))
  }

  async start(signal?: AbortSignal): Promise<void> {
    throwIfAborted(signal)
    await this.client.connect()
  }

  async stop(signal?: AbortSignal): Promise<void> {
    throwIfAborted(signal)
    await this.client.disconnect()
  }

  async callTool(name: string, arguments_: JsonObject, signal?: AbortSignal): Promise<MCPToolCallResult> {
    throwIfAborted(signal)
    return this.client.callTool(name, arguments_)
  }
}

/**
 * CUA-driver implementation of the generic computer-use backend.
 *
 * It faithfully maps the driver MCP surface into normalized desktop actions,
 * but owns no process launch policy. Pair it with an injected CuaDriverMcpPort
 * and a ComputerUseSession to make desktop access available to a tool call.
 */
export class CuaBackend implements ComputerUseBackend {
  private started = false

  constructor(private readonly driver: CuaDriverMcpPort) {}

  async start(signal?: AbortSignal): Promise<void> {
    if (this.started) return
    if (!await this.isAvailable(signal)) {
      throw new ComputerUseUnavailableError(cuaDriverInstallHint())
    }
    await this.driver.start?.(signal)
    this.started = true
  }

  async stop(signal?: AbortSignal): Promise<void> {
    if (!this.started) return
    try {
      await this.driver.stop?.(signal)
    } finally {
      this.started = false
    }
  }

  async isAvailable(signal?: AbortSignal): Promise<boolean> {
    throwIfAborted(signal)
    return Boolean(await this.driver.isAvailable(signal))
  }

  async capture(request: CaptureRequest, signal?: AbortSignal): Promise<CaptureResult> {
    const arguments_: JsonObject = { mode: request.mode }
    if (request.app !== undefined) arguments_.app = request.app
    const result = await this.call('capture', arguments_, signal)
    const pngB64 = firstPng(result)
    const dimensions = pngB64 === undefined ? { width: 0, height: 0 } : pngDimensions(pngB64)
    return {
      mode: request.mode,
      width: dimensions.width,
      height: dimensions.height,
      ...(pngB64 === undefined ? {} : { pngB64, pngBytesLength: decodedLength(pngB64) }),
      elements: parseCuaElements(textFromResult(result)),
    }
  }

  async click(request: ClickRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickWithAction('click', request, signal)
  }

  async doubleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickWithAction('double_click', { ...request, button: 'left', clickCount: 2 }, signal)
  }

  async rightClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickWithAction('right_click', { ...request, button: 'right', clickCount: 1 }, signal)
  }

  async middleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickWithAction('middle_click', { ...request, button: 'middle', clickCount: 1 }, signal)
  }

  async drag(request: DragRequest, signal?: AbortSignal): Promise<ActionResult> {
    const arguments_: JsonObject = captureArguments(request.captureAfter)
    assignOptionalNumber(arguments_, 'start_element', request.startElement)
    assignOptionalNumber(arguments_, 'start_x', request.startX)
    assignOptionalNumber(arguments_, 'start_y', request.startY)
    assignOptionalNumber(arguments_, 'end_element', request.endElement)
    assignOptionalNumber(arguments_, 'end_x', request.endX)
    assignOptionalNumber(arguments_, 'end_y', request.endY)
    return this.actionFromResult('drag', await this.call('drag', arguments_, signal))
  }

  async scroll(request: ScrollRequest, signal?: AbortSignal): Promise<ActionResult> {
    const arguments_: JsonObject = { ...captureArguments(request.captureAfter), dx: request.dx, dy: request.dy }
    assignOptionalNumber(arguments_, 'element', request.element)
    assignOptionalNumber(arguments_, 'x', request.x)
    assignOptionalNumber(arguments_, 'y', request.y)
    return this.actionFromResult('scroll', await this.call('scroll', arguments_, signal))
  }

  async type(request: TextRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.actionFromResult(
      'type',
      await this.call('type', { ...captureArguments(request.captureAfter), text: request.text }, signal),
    )
  }

  async key(request: KeyRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.actionFromResult(
      'key',
      await this.call('key', { ...captureArguments(request.captureAfter), key: request.key }, signal),
    )
  }

  async setValue(request: SetValueRequest, signal?: AbortSignal): Promise<ActionResult> {
    const arguments_: JsonObject = { ...captureArguments(request.captureAfter), value: request.value }
    assignOptionalNumber(arguments_, 'element', request.element)
    return this.actionFromResult('set_value', await this.call('set_value', arguments_, signal))
  }

  async wait(ms: number, signal?: AbortSignal): Promise<ActionResult> {
    return this.actionFromResult('wait', await this.call('wait', { ms }, signal))
  }

  async listApps(signal?: AbortSignal): Promise<ActionResult> {
    const result = await this.call('list_apps', {}, signal)
    const message = textFromResult(result)
    return {
      ok: !result.isError,
      action: 'list_apps',
      message,
      meta: { windows: parseCuaWindows(message) },
    }
  }

  async focusApp(app: string, signal?: AbortSignal): Promise<ActionResult> {
    return this.actionFromResult('focus_app', await this.call('focus_app', { app }, signal))
  }

  private async clickWithAction(action: string, request: ClickRequest, signal?: AbortSignal): Promise<ActionResult> {
    const arguments_: JsonObject = {
      ...captureArguments(request.captureAfter),
      button: request.button,
      click_count: request.clickCount,
    }
    assignOptionalNumber(arguments_, 'element', request.element)
    assignOptionalNumber(arguments_, 'x', request.x)
    assignOptionalNumber(arguments_, 'y', request.y)
    return this.actionFromResult(action, await this.call('click', arguments_, signal))
  }

  private async call(name: string, arguments_: JsonObject, signal: AbortSignal | undefined): Promise<MCPToolCallResult> {
    throwIfAborted(signal)
    if (!this.started) {
      throw new ComputerUseUnavailableError('CuaBackend has not been started; use it through ComputerUseSession')
    }
    return this.driver.callTool(name, arguments_, signal)
  }

  private actionFromResult(action: string, result: MCPToolCallResult): ActionResult {
    return {
      ok: !result.isError,
      action,
      message: textFromResult(result),
    }
  }
}

/** Human-readable setup guidance only. It never attempts to install software. */
export function cuaDriverInstallHint(): string {
  return CUA_DRIVER_INSTALL_HINT
}

/** Parse the CUA driver's list_windows markdown into JSON-safe window records. */
export function parseCuaWindows(text: string): JsonObject[] {
  const windows: JsonObject[] = []
  WINDOW_LINE.lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = WINDOW_LINE.exec(text)) !== null) {
    const app = match[1]?.trim()
    const pid = parsePositiveInteger(match[2])
    const windowId = parsePositiveInteger(match[3])
    if (app && pid !== undefined && windowId !== undefined) {
      windows.push({ app, pid, window_id: windowId })
    }
  }
  return windows
}

/** Parse the CUA driver's accessibility-tree markdown into generic UI elements. */
export function parseCuaElements(text: string): UIElement[] {
  const elements: UIElement[] = []
  ELEMENT_LINE.lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = ELEMENT_LINE.exec(text)) !== null) {
    const index = parsePositiveInteger(match[1])
    const role = match[2]?.trim()
    if (index === undefined || !role) continue
    const label = (match[3] ?? match[4] ?? '').trim()
    elements.push(label ? { index, role, label } : { index, role })
  }
  return elements
}

/** Decode PNG IHDR dimensions when a cua-driver response includes a valid header. */
export function pngDimensions(pngB64: string): { readonly height: number; readonly width: number } {
  try {
    const bytes = Buffer.from(pngB64, 'base64')
    if (bytes.length < 24) return { width: 0, height: 0 }
    return {
      width: bytes.readUInt32BE(16),
      height: bytes.readUInt32BE(20),
    }
  } catch {
    return { width: 0, height: 0 }
  }
}

function captureArguments(captureAfter: boolean): JsonObject {
  return captureAfter ? { capture_after: true } : {}
}

function assignOptionalNumber(target: JsonObject, key: string, value: number | undefined): void {
  if (value !== undefined) target[key] = value
}

function textFromResult(result: MCPToolCallResult): string {
  return result.content
    .filter((content): content is Extract<MCPToolCallResult['content'][number], { readonly type: 'text' }> => content.type === 'text')
    .map(content => content.text)
    .join('\n')
}

function firstPng(result: MCPToolCallResult): string | undefined {
  const image = result.content.find(
    (content): content is Extract<MCPToolCallResult['content'][number], { readonly type: 'image' }> =>
      content.type === 'image' && content.mimeType.toLowerCase().includes('png'),
  )
  return image?.data
}

function decodedLength(base64: string): number {
  return Buffer.byteLength(base64, 'base64')
}

function parsePositiveInteger(value: string | undefined): number | undefined {
  if (value === undefined || !/^\d+$/.test(value)) return undefined
  const number = Number(value)
  return Number.isSafeInteger(number) && number > 0 ? number : undefined
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new ClientError('computer_use', 'computer-use action was cancelled before the backend call')
  }
}
