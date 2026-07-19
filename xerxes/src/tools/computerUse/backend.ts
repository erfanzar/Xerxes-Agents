// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ClientError, ConfigurationError } from '../../core/errors.js'
import type { JsonObject, JsonValue } from '../../types/toolCalls.js'

const MAX_CAPTURE_DIMENSION = 100_000
const MAX_CAPTURE_ELEMENTS = 10_000
const MAX_PNG_BASE64_CHARACTERS = 32 * 1024 * 1024

export type ComputerCaptureMode = 'ax' | 'som' | 'vision'
export type ComputerMouseButton = 'left' | 'middle' | 'right'

/** One interactable accessibility element exposed by a privileged desktop backend. */
export interface UIElement {
  readonly app?: string
  readonly attributes?: JsonObject
  readonly bounds?: readonly [number, number, number, number]
  readonly index: number
  readonly label?: string
  readonly pid?: number
  readonly role: string
  readonly windowId?: number
}

/** A screenshot and/or accessibility tree captured by a privileged backend. */
export interface CaptureResult {
  readonly app?: string
  readonly elements?: readonly UIElement[]
  readonly height: number
  readonly mode: ComputerCaptureMode
  readonly pngB64?: string
  readonly pngBytesLength?: number
  readonly width: number
  readonly windowTitle?: string
}

/** Outcome from a desktop interaction. A post-action capture is optional. */
export interface ActionResult {
  readonly action: string
  readonly capture?: CaptureResult
  readonly message?: string
  readonly meta?: JsonObject
  readonly ok: boolean
}

export interface CaptureRequest {
  readonly app?: string
  readonly mode: ComputerCaptureMode
}

export interface ClickRequest {
  readonly button: ComputerMouseButton
  readonly captureAfter: boolean
  readonly clickCount: number
  readonly element?: number
  readonly x?: number
  readonly y?: number
}

export interface DragRequest {
  readonly captureAfter: boolean
  readonly endElement?: number
  readonly endX?: number
  readonly endY?: number
  readonly startElement?: number
  readonly startX?: number
  readonly startY?: number
}

export interface ScrollRequest {
  readonly captureAfter: boolean
  readonly dx: number
  readonly dy: number
  readonly element?: number
  readonly x?: number
  readonly y?: number
}

export interface TextRequest {
  readonly captureAfter: boolean
  readonly text: string
}

export interface KeyRequest {
  readonly captureAfter: boolean
  readonly key: string
}

export interface SetValueRequest {
  readonly captureAfter: boolean
  readonly element?: number
  readonly value: string
}

export interface MouseMoveRequest {
  readonly captureAfter: boolean
  readonly x: number
  readonly y: number
}

/**
 * Privileged desktop boundary.
 *
 * The Bun runtime deliberately includes no desktop automation implementation.
 * Hosts inject a port backed by an audited native service, a remote CUA
 * driver, or another explicitly approved integration. A missing or unavailable
 * port is always an error; no browser or cursor simulation is substituted.
 */
export interface ComputerUsePort {
  capture(request: CaptureRequest, signal?: AbortSignal): CaptureResult | Promise<CaptureResult>
  click(request: ClickRequest, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  /** Report the current cursor position in logical points (meta: x, y). */
  cursorPosition?(signal?: AbortSignal): ActionResult | Promise<ActionResult>
  doubleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  drag(request: DragRequest, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  focusApp(app: string, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  isAvailable(signal?: AbortSignal): boolean | Promise<boolean>
  key(request: KeyRequest, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  listApps(signal?: AbortSignal): ActionResult | Promise<ActionResult>
  middleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  /** Move the cursor without pressing a button. */
  mouseMove?(request: MouseMoveRequest, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  rightClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  scroll(request: ScrollRequest, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  setValue(request: SetValueRequest, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  start?(signal?: AbortSignal): void | Promise<void>
  stop?(signal?: AbortSignal): void | Promise<void>
  /** Triple-click, e.g. to select a paragraph. */
  tripleClick?(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  type(request: TextRequest, signal?: AbortSignal): ActionResult | Promise<ActionResult>
  wait(ms: number, signal?: AbortSignal): ActionResult | Promise<ActionResult>
}

/** Alias retained for integrations that describe this boundary as a backend. */
export interface ComputerUseBackend extends ComputerUsePort {}

export class ComputerUseUnavailableError extends ClientError {
  constructor(message: string, cause: unknown = undefined) {
    super('computer_use', message, cause)
  }
}

export interface ComputerUseSessionOptions {
  readonly port?: ComputerUsePort
}

/**
 * Serialized lifecycle and validation boundary around a configured desktop port.
 *
 * A desktop target is shared state. Serializing calls prevents concurrent model
 * turns from interleaving pointer and keyboard actions. The session also
 * validates every backend result before it is passed back into model context.
 */
export class ComputerUseSession {
  private operationTail: Promise<void> = Promise.resolve()
  private port: ComputerUsePort | undefined
  private started = false

  constructor(options: ComputerUseSessionOptions = {}) {
    this.port = options.port
  }

  /**
   * Configure a port before the session starts.
   *
   * Replacing a started backend could leave a privileged child service alive,
   * so callers must create a fresh session or stop this one first.
   */
  setPort(port: ComputerUsePort | undefined): void {
    if (this.started) {
      throw new ConfigurationError('computerUse.port', 'cannot replace a started ComputerUsePort; stop it first')
    }
    this.port = port
  }

  async isAvailable(signal?: AbortSignal): Promise<boolean> {
    const port = this.port
    if (port === undefined) return false
    return Boolean(await port.isAvailable(signal))
  }

  async start(signal?: AbortSignal): Promise<void> {
    await this.exclusive(async () => {
      await this.requireStartedPort(signal)
    })
  }

  async stop(signal?: AbortSignal): Promise<void> {
    await this.exclusive(async () => {
      const port = this.port
      if (!this.started || port === undefined) return
      try {
        await port.stop?.(signal)
      } finally {
        this.started = false
      }
    })
  }

  async capture(request: CaptureRequest, signal?: AbortSignal): Promise<CaptureResult> {
    return this.exclusive(async () => normalizeCaptureResult(await (await this.requireStartedPort(signal)).capture(request, signal)))
  }

  async click(request: ClickRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('click', signal, port => port.click(request, signal))
  }

  async doubleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('double_click', signal, port => port.doubleClick(request, signal))
  }

  async rightClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('right_click', signal, port => port.rightClick(request, signal))
  }

  async middleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('middle_click', signal, port => port.middleClick(request, signal))
  }

  async tripleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('triple_click', signal, port => {
      if (port.tripleClick === undefined) {
        throw new ComputerUseUnavailableError('configured ComputerUsePort does not support triple_click')
      }
      return port.tripleClick(request, signal)
    })
  }

  async mouseMove(request: MouseMoveRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('mouse_move', signal, port => {
      if (port.mouseMove === undefined) {
        throw new ComputerUseUnavailableError('configured ComputerUsePort does not support mouse_move')
      }
      return port.mouseMove(request, signal)
    })
  }

  async cursorPosition(signal?: AbortSignal): Promise<ActionResult> {
    return this.action('cursor_position', signal, port => {
      if (port.cursorPosition === undefined) {
        throw new ComputerUseUnavailableError('configured ComputerUsePort does not support cursor_position')
      }
      return port.cursorPosition(signal)
    })
  }

  async drag(request: DragRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('drag', signal, port => port.drag(request, signal))
  }

  async scroll(request: ScrollRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('scroll', signal, port => port.scroll(request, signal))
  }

  async type(request: TextRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('type', signal, port => port.type(request, signal))
  }

  async key(request: KeyRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('key', signal, port => port.key(request, signal))
  }

  async setValue(request: SetValueRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('set_value', signal, port => port.setValue(request, signal))
  }

  async wait(ms: number, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('wait', signal, port => port.wait(ms, signal))
  }

  async listApps(signal?: AbortSignal): Promise<ActionResult> {
    return this.action('list_apps', signal, port => port.listApps(signal))
  }

  async focusApp(app: string, signal?: AbortSignal): Promise<ActionResult> {
    return this.action('focus_app', signal, port => port.focusApp(app, signal))
  }

  private async action(
    expectedAction: string,
    signal: AbortSignal | undefined,
    invoke: (port: ComputerUsePort) => ActionResult | Promise<ActionResult>,
  ): Promise<ActionResult> {
    return this.exclusive(async () => normalizeActionResult(await invoke(await this.requireStartedPort(signal)), expectedAction))
  }

  private async requireStartedPort(signal: AbortSignal | undefined): Promise<ComputerUsePort> {
    const port = this.port
    if (port === undefined) {
      throw new ComputerUseUnavailableError(
        'no ComputerUsePort is configured; inject a privileged desktop backend before registering computer_use',
      )
    }

    let available: boolean
    try {
      available = Boolean(await port.isAvailable(signal))
    } catch (error) {
      throw new ComputerUseUnavailableError('configured ComputerUsePort could not determine availability', error)
    }
    if (!available) {
      throw new ComputerUseUnavailableError('configured ComputerUsePort is unavailable on this host')
    }
    if (!this.started) {
      try {
        await port.start?.(signal)
      } catch (error) {
        throw new ComputerUseUnavailableError('configured ComputerUsePort could not start', error)
      }
      this.started = true
    }
    return port
  }

  private async exclusive<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.operationTail
    let release: (() => void) | undefined
    const current = new Promise<void>(resolve => {
      release = resolve
    })
    this.operationTail = current
    await previous
    try {
      return await operation()
    } finally {
      release?.()
      if (this.operationTail === current) {
        this.operationTail = Promise.resolve()
      }
    }
  }
}

/** Validate and normalize untrusted backend capture data at the tool boundary. */
export function normalizeCaptureResult(source: CaptureResult): CaptureResult {
  if (source === null || typeof source !== 'object') {
    throw new ClientError('computer_use', 'ComputerUsePort returned an invalid capture result')
  }
  const mode = captureMode(source.mode)
  const pngB64 = source.pngB64 === undefined ? undefined : pngBase64(source.pngB64)
  const pngBytesLength = source.pngBytesLength === undefined
    ? undefined
    : nonNegativeInteger(source.pngBytesLength, 'capture png byte length')
  if (pngBytesLength !== undefined && pngBytesLength > MAX_PNG_BASE64_CHARACTERS) {
    throw new ClientError('computer_use', 'ComputerUsePort returned an oversized screenshot byte length')
  }
  const elements = source.elements === undefined
    ? []
    : normalizeElements(source.elements)
  const app = source.app === undefined ? undefined : portString(source.app, 'capture app')
  const windowTitle = source.windowTitle === undefined ? undefined : portString(source.windowTitle, 'capture window title')
  return Object.freeze({
    mode,
    width: captureDimension(source.width, 'capture width'),
    height: captureDimension(source.height, 'capture height'),
    ...(pngB64 === undefined ? {} : { pngB64 }),
    ...(pngBytesLength === undefined ? {} : { pngBytesLength }),
    elements: Object.freeze(elements),
    ...(app === undefined ? {} : { app }),
    ...(windowTitle === undefined ? {} : { windowTitle }),
  })
}

/** Validate and normalize untrusted backend action data at the tool boundary. */
export function normalizeActionResult(source: ActionResult, expectedAction?: string): ActionResult {
  if (source === null || typeof source !== 'object' || typeof source.ok !== 'boolean') {
    throw new ClientError('computer_use', 'ComputerUsePort returned an invalid action result')
  }
  const action = portNonBlankString(source.action, 'action name')
  if (expectedAction !== undefined && action !== expectedAction) {
    throw new ClientError(
      'computer_use',
      'ComputerUsePort reported action ' + action + ' while ' + expectedAction + ' was requested',
    )
  }
  const message = source.message === undefined ? '' : portString(source.message, 'action message')
  const meta = source.meta === undefined ? {} : jsonObject(source.meta, 'action metadata')
  const capture = source.capture === undefined ? undefined : normalizeCaptureResult(source.capture)
  return Object.freeze({
    ok: source.ok,
    action,
    message,
    meta,
    ...(capture === undefined ? {} : { capture }),
  })
}

function normalizeElements(source: readonly UIElement[]): UIElement[] {
  if (!Array.isArray(source)) {
    throw new ClientError('computer_use', 'ComputerUsePort capture elements must be an array')
  }
  if (source.length > MAX_CAPTURE_ELEMENTS) {
    throw new ClientError('computer_use', 'ComputerUsePort returned too many capture elements')
  }
  const indexes = new Set<number>()
  return source.map((element, index) => {
    if (element === null || typeof element !== 'object') {
      throw new ClientError('computer_use', 'ComputerUsePort returned an invalid capture element at index ' + index)
    }
    const normalizedIndex = positiveInteger(element.index, 'element index')
    if (indexes.has(normalizedIndex)) {
      throw new ClientError('computer_use', 'ComputerUsePort returned duplicate element index ' + normalizedIndex)
    }
    indexes.add(normalizedIndex)
    const bounds = element.bounds === undefined ? undefined : normalizeBounds(element.bounds)
    const label = element.label === undefined ? undefined : portString(element.label, 'element label')
    const app = element.app === undefined ? undefined : portString(element.app, 'element app')
    const pid = element.pid === undefined ? undefined : nonNegativeInteger(element.pid, 'element pid')
    const windowId = element.windowId === undefined ? undefined : nonNegativeInteger(element.windowId, 'element window id')
    const attributes = element.attributes === undefined ? undefined : jsonObject(element.attributes, 'element attributes')
    return Object.freeze({
      index: normalizedIndex,
      role: portNonBlankString(element.role, 'element role'),
      ...(label === undefined ? {} : { label }),
      ...(bounds === undefined ? {} : { bounds }),
      ...(app === undefined ? {} : { app }),
      ...(pid === undefined ? {} : { pid }),
      ...(windowId === undefined ? {} : { windowId }),
      ...(attributes === undefined ? {} : { attributes }),
    })
  })
}

function normalizeBounds(source: readonly number[]): readonly [number, number, number, number] {
  if (!Array.isArray(source) || source.length !== 4) {
    throw new ClientError('computer_use', 'ComputerUsePort element bounds must contain x, y, width, and height')
  }
  const values = source.map((value, index) => integer(value, 'element bounds[' + index + ']'))
  const x = values[0]
  const y = values[1]
  const width = values[2]
  const height = values[3]
  if (x === undefined || y === undefined || width === undefined || height === undefined || width < 0 || height < 0) {
    throw new ClientError('computer_use', 'ComputerUsePort returned invalid element bounds')
  }
  return Object.freeze([x, y, width, height])
}

function captureMode(value: unknown): ComputerCaptureMode {
  if (value === 'som' || value === 'vision' || value === 'ax') return value
  throw new ClientError('computer_use', 'ComputerUsePort returned an invalid capture mode')
}

function captureDimension(value: unknown, label: string): number {
  const dimension = nonNegativeInteger(value, label)
  if (dimension > MAX_CAPTURE_DIMENSION) {
    throw new ClientError('computer_use', 'ComputerUsePort returned an oversized ' + label)
  }
  return dimension
}

function positiveInteger(value: unknown, label: string): number {
  const integerValue = integer(value, label)
  if (integerValue < 1) {
    throw new ClientError('computer_use', 'ComputerUsePort returned a non-positive ' + label)
  }
  return integerValue
}

function nonNegativeInteger(value: unknown, label: string): number {
  const integerValue = integer(value, label)
  if (integerValue < 0) {
    throw new ClientError('computer_use', 'ComputerUsePort returned a negative ' + label)
  }
  return integerValue
}

function integer(value: unknown, label: string): number {
  if (typeof value !== 'number' || !Number.isSafeInteger(value)) {
    throw new ClientError('computer_use', 'ComputerUsePort returned an invalid ' + label)
  }
  return value
}

function portString(value: unknown, label: string): string {
  if (typeof value !== 'string') {
    throw new ClientError('computer_use', 'ComputerUsePort returned an invalid ' + label)
  }
  return value
}

function portNonBlankString(value: unknown, label: string): string {
  const normalized = portString(value, label).trim()
  if (!normalized) {
    throw new ClientError('computer_use', 'ComputerUsePort returned an empty ' + label)
  }
  return normalized
}

function pngBase64(value: unknown): string {
  const base64 = portNonBlankString(value, 'PNG base64 data')
  if (base64.length > MAX_PNG_BASE64_CHARACTERS || base64.length % 4 === 1 || !/^[A-Za-z0-9+/]*={0,2}$/.test(base64)) {
    throw new ClientError('computer_use', 'ComputerUsePort returned invalid or oversized PNG base64 data')
  }
  return base64
}

function jsonObject(value: unknown, label: string): JsonObject {
  if (!isJsonObject(value)) {
    throw new ClientError('computer_use', 'ComputerUsePort returned invalid ' + label)
  }
  return Object.freeze({ ...value })
}

function isJsonObject(value: unknown): value is JsonObject {
  return typeof value === 'object' && value !== null && !Array.isArray(value) && Object.values(value).every(isJsonValue)
}

function isJsonValue(value: unknown): value is JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return true
  if (typeof value === 'number') return Number.isFinite(value)
  if (Array.isArray(value)) return value.every(isJsonValue)
  return typeof value === 'object' && value !== null && Object.values(value).every(isJsonValue)
}
