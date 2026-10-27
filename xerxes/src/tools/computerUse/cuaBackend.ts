// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Buffer } from 'node:buffer'

import { ClientError } from '../../core/errors.js'
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
  type MouseMoveRequest,
  type ScrollRequest,
  type SetValueRequest,
  type TextRequest,
  type UIElement,
} from './backend.js'

export { MCPClientCuaDriverPort } from './cuaDriverPort.js'
export type { MCPClientCuaDriverPortOptions } from './cuaDriverPort.js'

export interface CuaDriverMcpPort {
  callTool(name: string, arguments_: JsonObject, signal?: AbortSignal): MCPToolCallResult | Promise<MCPToolCallResult>
  isAvailable(signal?: AbortSignal): boolean | Promise<boolean>
  start?(signal?: AbortSignal): void | Promise<void>
  stop?(signal?: AbortSignal): void | Promise<void>
}

export interface CuaBackendOptions {
  readonly maxDimension?: number
  readonly maxElements?: number
  /** A runtime host owns the shared transport; standalone backends own theirs. */
  readonly ownsDriver?: boolean
}

interface TargetWindow { readonly pid: number; readonly window_id: number }
interface ElementHandle {
  readonly driverIndex: number
  readonly token?: string
  readonly bounds?: readonly [number, number, number, number]
}
interface Snapshot {
  readonly target?: TargetWindow
  readonly request: CaptureRequest
  readonly snapshotId?: string
  readonly screenshotScale: number
  readonly width: number
  readonly height: number
  readonly elements: ReadonlyMap<number, ElementHandle>
}

const PERMISSION_HINT = 'CuaDriver needs Accessibility and Screen Recording permissions under its own identity (com.trycua.driver). Run `cua-driver permissions grant` (or the configured binary with `permissions grant`), then retry. Xerxes never grants permissions or changes driver configuration.'

/** CuaDriver 0.28: exact window targets, opaque AX handles, and screenshot-pixel input. */
export class CuaBackend implements ComputerUseBackend {
  private started = false
  private snapshot: Snapshot | undefined
  private elementsValid = false

  constructor(private readonly driver: CuaDriverMcpPort, private readonly options: CuaBackendOptions = {}) {}

  async isAvailable(signal?: AbortSignal): Promise<boolean> {
    throwIfAborted(signal)
    if (!await this.driver.isAvailable(signal)) throw new ComputerUseUnavailableError(cuaDriverInstallHint())
    return true
  }

  async start(signal?: AbortSignal): Promise<void> {
    if (this.started) return
    await this.isAvailable(signal)
    try {
      await this.driver.start?.(signal)
      // Read only: never prompt for a grant or change the driver's configuration.
      const permissions = await this.driver.callTool('check_permissions', { prompt: false, probe_direct_capture: false }, signal)
      const state = structured(permissions)
      if (permissions.isError || state.accessibility !== true || state.screen_recording !== true || state.screen_recording_capturable === false) {
        throw new ComputerUseUnavailableError(PERMISSION_HINT + '\n' + textFromResult(permissions))
      }
      this.started = true
    } catch (error) {
      if (this.options.ownsDriver !== false) await this.driver.stop?.()
      if (error instanceof ComputerUseUnavailableError) throw error
      throw new ComputerUseUnavailableError('CuaDriver is unavailable: ' + errorMessage(error) + '. ' + PERMISSION_HINT, error)
    }
  }

  async stop(): Promise<void> {
    this.snapshot = undefined
    this.elementsValid = false
    this.started = false
    if (this.options.ownsDriver !== false) await this.driver.stop?.()
  }

  async capture(request: CaptureRequest, signal?: AbortSignal): Promise<CaptureResult> {
    // A failed/new capture must not leave the previous capture's handles usable.
    this.snapshot = undefined
    this.elementsValid = false
    const target = await this.resolveWindow(request.app, signal)
    if (!target && request.app !== undefined) {
      throw new ClientError('computer_use', 'CuaDriver: no usable window for ' + request.app + '; focus_app explicitly to open it')
    }
    return this.captureTarget(request, target, signal)
  }

  click(request: ClickRequest, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickLike('click', request, request.button, request.clickCount, signal)
  }

  doubleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickLike('double_click', request, 'left', 2, signal)
  }

  rightClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickLike('right_click', request, 'right', 1, signal)
  }

  middleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickLike('middle_click', request, 'middle', 1, signal)
  }

  tripleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>, signal?: AbortSignal): Promise<ActionResult> {
    return this.clickLike('triple_click', request, 'left', 3, signal)
  }

  async drag(request: DragRequest, signal?: AbortSignal): Promise<ActionResult> {
    const from = this.point(request.startX, request.startY, request.startElement)
    const to = this.point(request.endX, request.endY, request.endElement)
    return this.act('drag', 'drag', { ...this.targetArguments(), from_x: from.x, from_y: from.y, to_x: to.x, to_y: to.y }, request.captureAfter, signal)
  }

  async scroll(request: ScrollRequest, signal?: AbortSignal): Promise<ActionResult> {
    const target = request.element !== undefined ? this.elementArguments(request.element) : this.targetArguments()
    if (request.x !== undefined || request.y !== undefined) Object.assign(target, this.point(request.x, request.y))
    // The public tool uses pixel deltas; 0.28 uses wheel notches (1..50).
    // Preserve both axes, stopping at the first refused/failed driver action.
    const axes = [{ delta: request.dy, positive: 'down', negative: 'up' }, { delta: request.dx, positive: 'right', negative: 'left' }]
    let result: MCPToolCallResult = { content: [{ type: 'text', text: 'No scroll requested (zero deltas).' }] }
    for (const axis of axes) {
      if (!axis.delta) continue
      this.elementsValid = false
      result = await this.call('scroll', { ...target, direction: axis.delta > 0 ? axis.positive : axis.negative, amount: Math.min(50, Math.max(1, Math.round(Math.abs(axis.delta) / 40))), by: 'line' }, signal)
      if (result.isError) break
    }
    return this.finish('scroll', result, request.captureAfter, signal)
  }

  async type(request: TextRequest, signal?: AbortSignal): Promise<ActionResult> {
    const target = request.element === undefined ? await this.keyboardTarget(signal) : this.elementArguments(request.element)
    return this.act('type', 'type_text', { ...target, text: request.text }, request.captureAfter, signal)
  }

  async key(request: KeyRequest, signal?: AbortSignal): Promise<ActionResult> {
    const keys = request.key.toLowerCase().split('+').map(part => part.trim())
    const aliases: Readonly<Record<string, string>> = { command: 'cmd', control: 'ctrl', option: 'alt', enter: 'return', esc: 'escape', arrowup: 'up', arrowdown: 'down', arrowleft: 'left', arrowright: 'right' }
    const normalized = keys.map(key => aliases[key] ?? key)
    if (normalized.some(key => !key) || normalized.slice(0, -1).some(key => !['cmd', 'ctrl', 'alt', 'shift', 'fn'].includes(key))) {
      return { action: 'key', ok: false, message: 'CuaDriver: invalid key chord; use command+a, shift+tab, or Enter' }
    }
    const target = await this.keyboardTarget(signal)
    return normalized.length > 1
      ? this.act('key', 'hotkey', { ...target, keys: normalized }, request.captureAfter, signal)
      : this.act('key', 'press_key', { ...target, key: normalized[0] ?? '' }, request.captureAfter, signal)
  }

  setValue(request: SetValueRequest, signal?: AbortSignal): Promise<ActionResult> {
    if (request.element === undefined) throw recapture('set_value requires an element index')
    return this.act('set_value', 'set_value', { ...this.elementArguments(request.element), value: request.value }, request.captureAfter, signal)
  }

  async mouseMove(request: MouseMoveRequest, signal?: AbortSignal): Promise<ActionResult> {
    const snapshot = this.requireSnapshot()
    const target: JsonObject = snapshot.target === undefined
      ? { kind: 'desktop', display_id: 'primary' }
      : { kind: 'window', ...snapshot.target }
    return this.act('mouse_move', 'move_cursor', { target, scope: snapshot.target ? 'window' : 'desktop', ...this.point(request.x, request.y) }, request.captureAfter, signal)
  }

  async cursorPosition(signal?: AbortSignal): Promise<ActionResult> {
    const result = await this.call('get_cursor_position', {}, signal)
    return { ...this.actionResult('cursor_position', result), meta: structured(result) }
  }

  async wait(ms: number, signal?: AbortSignal): Promise<ActionResult> {
    throwIfAborted(signal)
    await new Promise<void>((resolve, reject) => {
      const abort = (): void => { clearTimeout(timer); reject(new ClientError('computer_use', 'computer-use wait was cancelled')) }
      const timer = setTimeout(() => { signal?.removeEventListener('abort', abort); resolve() }, ms)
      signal?.addEventListener('abort', abort, { once: true })
    })
    return { action: 'wait', ok: true, message: `CuaDriver: waited ${ms}ms` }
  }

  async listApps(signal?: AbortSignal): Promise<ActionResult> {
    const result = await this.call('list_apps', {}, signal)
    return { ...this.actionResult('list_apps', result), meta: structured(result) }
  }

  async focusApp(app: string, signal?: AbortSignal): Promise<ActionResult> {
    this.snapshot = undefined
    this.elementsValid = false
    let apps = await this.apps(signal)
    let match = findApp(apps, app)
    if (!match || !positive(match.pid) || match.running === false) {
      if (!app.trim() || /^\d+$/.test(app.trim())) return { action: 'focus_app', ok: false, message: 'CuaDriver: no running app for that pid; launching requires an app name' }
      // Only explicit focus_app may launch; capture never opens an app.
      const launched = await this.call('launch_app', { name: app }, signal)
      if (launched.isError) return this.actionResult('focus_app', launched)
      apps = await this.apps(signal)
      match = findApp(apps, app)
    }
    if (!match || !positive(match.pid)) return { action: 'focus_app', ok: false, message: 'CuaDriver: app is not running after launch: ' + app }
    return this.actionResult('focus_app', await this.call('bring_to_front', { pid: match.pid }, signal))
  }

  private async captureTarget(request: CaptureRequest, target: TargetWindow | undefined, signal?: AbortSignal): Promise<CaptureResult> {
    this.snapshot = undefined
    this.elementsValid = false
    const result = await this.call(target ? 'get_window_state' : 'get_desktop_state', target ? {
      ...target,
      include_screenshot: request.mode !== 'ax',
      include_accessibility_tree: request.mode !== 'vision',
      max_dimension: this.options.maxDimension ?? 1568,
      max_elements: this.options.maxElements ?? 300,
    } : {}, signal)
    requireSuccess(result)
    const state = structured(result)
    const image = state.screenshot_frame_valid === false ? undefined : result.content.find(content => content.type === 'image')
    const dimensions = image?.mimeType === 'image/png' ? pngDimensions(image.data) : { width: 0, height: 0 }
    const width = image ? dimension(state.screenshot_width) || dimensions.width : 0
    const height = image ? dimension(state.screenshot_height) || dimensions.height : 0
    const handles = new Map<number, ElementHandle>()
    const elements: UIElement[] = []
    const windowBounds = record(state.window_bounds)
    const scaleX = width / (finite(windowBounds.width) || 1)
    const scaleY = height / (finite(windowBounds.height) || 1)
    for (const row of records(state.elements).slice(0, this.options.maxElements ?? 300)) {
      if (typeof row.element_index !== 'number' || !Number.isSafeInteger(row.element_index) || row.element_index < 0 || typeof row.role !== 'string' || !row.role.trim()) continue
      // CuaDriver starts at zero; the public Xerxes schema starts at one.
      const index = elements.length + 1
      const frame = record(row.frame)
      const bounds: readonly [number, number, number, number] | undefined = width && height && ['x', 'y', 'w', 'h'].every(key => typeof frame[key] === 'number')
        ? [Math.round((finite(frame.x) - finite(windowBounds.x)) * scaleX), Math.round((finite(frame.y) - finite(windowBounds.y)) * scaleY), Math.max(0, Math.round(finite(frame.w) * scaleX)), Math.max(0, Math.round(finite(frame.h) * scaleY))]
        : undefined
      const token = typeof row.element_token === 'string' && row.element_token ? row.element_token : undefined
      handles.set(index, { driverIndex: row.element_index, ...(token ? { token } : {}), ...(bounds ? { bounds } : {}) })
      const value = typeof row.value === 'string' ? row.value : typeof row.value === 'number' || typeof row.value === 'boolean' ? String(row.value) : ''
      const label = [typeof row.label === 'string' ? row.label : '', value ? `value: ${value.slice(0, 160)}` : ''].filter(Boolean).join(' — ').slice(0, 500)
      elements.push({ index, role: row.role, ...(label ? { label } : {}), ...(bounds ? { bounds } : {}), ...(target ? { pid: target.pid, windowId: target.window_id } : {}) })
    }
    this.snapshot = {
      request, ...(target ? { target } : {}),
      ...(typeof state.snapshot_id === 'string' ? { snapshotId: state.snapshot_id } : {}),
      screenshotScale: finite(state.screenshot_scale) || 1, width, height, elements: handles,
    }
    this.elementsValid = true
    const warnings = [
      !target ? 'Desktop scope: full-display native pixels; CuaDriver does not support max_dimension for desktop captures.' : '',
      typeof state.degraded_reason === 'string' ? state.degraded_reason : '',
      request.mode !== 'ax' && !image ? 'CuaDriver returned no screenshot/pixel frame; use AX elements or capture again.' + screenshotErrorHint(state.screenshot_error) : '',
    ].filter(Boolean)
    return {
      mode: request.mode, width, height, elements, backend: 'cua',
      ...(typeof state.app_name === 'string' ? { app: state.app_name } : {}),
      ...(typeof state.window_title === 'string' ? { windowTitle: state.window_title } : {}),
      ...(image ? { pngB64: image.data, mediaType: image.mimeType, pngBytesLength: Buffer.byteLength(image.data, 'base64') } : {}),
      ...(warnings.length ? { warning: warnings.join(' ') } : {}),
    }
  }

  private async apps(signal?: AbortSignal): Promise<JsonObject[]> {
    const result = await this.call('list_apps', {}, signal)
    requireSuccess(result)
    return records(structured(result).apps)
  }

  private async resolveWindow(app: string | undefined, signal?: AbortSignal): Promise<TargetWindow | undefined> {
    const apps = await this.apps(signal)
    const match = app === undefined ? apps.find(row => row.active === true && positive(row.pid)) : findApp(apps, app)
    if (app !== undefined && (!match || !positive(match.pid) || match.running === false)) return undefined
    const result = await this.call('list_windows', match && positive(match.pid) ? { pid: match.pid } : { on_screen_only: true }, signal)
    requireSuccess(result)
    const windows = records(structured(result).windows)
      .filter(row => positive(row.pid) && positive(row.window_id) && (!match || row.pid === match.pid))
      .filter(row => { const b = record(row.bounds); return finite(b.width) > 0 && finite(b.height) > 50 })
      .sort((a, b) => Number(b.on_current_space === true) - Number(a.on_current_space === true)
        || Number(b.is_on_screen === true) - Number(a.is_on_screen === true)
        || Number(typeof b.title === 'string' && b.title.trim().length > 0) - Number(typeof a.title === 'string' && a.title.trim().length > 0)
        || finite(a.z_index) - finite(b.z_index))
    const window = windows[0]
    return window && positive(window.pid) && positive(window.window_id) ? { pid: window.pid, window_id: window.window_id } : undefined
  }

  private requireSnapshot(): Snapshot {
    if (!this.snapshot) throw recapture('no target snapshot')
    return this.snapshot
  }

  private targetArguments(): JsonObject {
    const snapshot = this.requireSnapshot()
    return snapshot.target ? { ...snapshot.target, scope: 'window' } : { scope: 'desktop' }
  }

  private elementArguments(index: number): JsonObject {
    const snapshot = this.requireSnapshot()
    const handle = snapshot.elements.get(index)
    if (!snapshot.target || !this.elementsValid || !handle) throw recapture('missing, unknown, or stale element index ' + index)
    if (handle.token) return { ...snapshot.target, element_token: handle.token }
    if (snapshot.snapshotId) return { ...snapshot.target, element_index: handle.driverIndex, snapshot_id: snapshot.snapshotId }
    throw recapture('driver returned no snapshot-bound handle for element ' + index)
  }

  private point(x: number | undefined, y: number | undefined, element?: number): { x: number; y: number } {
    const snapshot = this.requireSnapshot()
    if (element !== undefined) {
      this.elementArguments(element)
      const bounds = snapshot.elements.get(element)?.bounds
      if (!bounds) throw recapture('element has no screenshot bounds')
      x = bounds[0] + bounds[2] / 2
      y = bounds[1] + bounds[3] / 2
    }
    if (!snapshot.width || !snapshot.height || x === undefined || y === undefined || !Number.isFinite(x) || !Number.isFinite(y) || x < 0 || y < 0 || x >= snapshot.width || y >= snapshot.height) {
      throw recapture('coordinates are outside the last screenshot or no pixel frame is available')
    }
    // IMPORTANT: identity transform. 0.28 expects pixels in the *returned*
    // screenshot and reverses max_dimension AND Retina backing scale itself.
    // screenshot_scale is a backing scale (e.g. 2), NOT a factor to apply here.
    return { x, y }
  }

  private async keyboardTarget(signal?: AbortSignal): Promise<JsonObject> {
    if (this.snapshot) return this.targetArguments()
    const app = (await this.apps(signal)).find(row => row.active === true && positive(row.pid))
    return app && positive(app.pid) ? { pid: app.pid } : { scope: 'desktop' }
  }

  private clickLike(action: string, request: Omit<ClickRequest, 'button' | 'clickCount'>, button: string, count: number, signal?: AbortSignal): Promise<ActionResult> {
    // Generic click.count applies ONLY to pixels. The dedicated double_click
    // implements element double-clicks; triple-click has to use the AX center.
    const target = request.element === undefined || count === 3
      ? { ...this.targetArguments(), ...this.point(request.x, request.y, request.element) }
      : this.elementArguments(request.element)
    if (action === 'double_click' && request.element !== undefined) {
      return this.act(action, 'double_click', target, request.captureAfter, signal)
    }
    return this.act(action, 'click', { ...target, button, count }, request.captureAfter, signal)
  }

  private async act(action: string, tool: string, arguments_: JsonObject, captureAfter: boolean, signal?: AbortSignal): Promise<ActionResult> {
    // Invalidate BEFORE sending: a cancelled/failed call may have taken effect.
    this.elementsValid = false
    return this.finish(action, await this.call(tool, arguments_, signal), captureAfter, signal)
  }

  private async finish(action: string, result: MCPToolCallResult, captureAfter: boolean, signal?: AbortSignal): Promise<ActionResult> {
    const outcome = this.actionResult(action, result)
    if (!outcome.ok || !captureAfter) return outcome
    const snapshot = this.snapshot
    try {
      const capture = snapshot ? await this.captureTarget(snapshot.request, snapshot.target, signal) : await this.capture({ mode: 'som' }, signal)
      return { ...outcome, capture }
    } catch (error) {
      return { ...outcome, message: (outcome.message ?? '') + '; capture_after failed: ' + errorMessage(error) }
    }
  }

  private actionResult(action: string, result: MCPToolCallResult): ActionResult {
    const text = textFromResult(result) || JSON.stringify(structured(result))
    if (result.isError && /stale|snapshot|token/i.test(text)) this.elementsValid = false
    return { action, ok: result.isError !== true, message: 'CuaDriver: ' + text + (result.isError && /stale|snapshot|token/i.test(text) ? '; capture again before using an element' : '') }
  }

  private async call(name: string, arguments_: JsonObject, signal?: AbortSignal): Promise<MCPToolCallResult> {
    throwIfAborted(signal)
    if (!this.started) throw new ComputerUseUnavailableError('CuaBackend has not been started; use ComputerUseSession')
    return this.driver.callTool(name, arguments_, signal)
  }
}

export function cuaDriverInstallHint(): string {
  return 'CuaDriver is unavailable. Install CuaDriver 0.28 or set computer_use_cua_driver / XERXES_CUA_DRIVER to its absolute binary path. ' + PERMISSION_HINT
}

/** Legacy markdown readers remain useful for exported driver logs, not action handles. */
export function parseCuaWindows(text: string): JsonObject[] {
  return [...text.matchAll(/^-\s+(.+?)\s+\(pid\s+(\d+)\)\s+.*\[window_id:\s+(\d+)\]/gm)]
    .map(match => ({ app: match[1]?.trim() ?? '', pid: Number(match[2]), window_id: Number(match[3]) }))
}

export function parseCuaElements(text: string): UIElement[] {
  return [...text.matchAll(/^\s*(?:-\s+)?\[(\d+)\]\s+(\w+)(?:\s+"([^"]*)"|(?:\s+\(\d+\))?\s+id=([^\s\[\]]*))?/gm)]
    .filter(match => Number(match[1]) > 0)
    .map(match => ({ index: Number(match[1]), role: match[2] ?? '', ...((match[3] ?? match[4]) ? { label: (match[3] ?? match[4] ?? '').trim() } : {}) }))
}

export function pngDimensions(base64: string): { width: number; height: number } {
  const bytes = Buffer.from(base64.slice(0, 32), 'base64')
  return bytes.length >= 24 && bytes.subarray(0, 8).equals(Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]))
    ? { width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) }
    : { width: 0, height: 0 }
}

function record(value: unknown): JsonObject {
  return value !== null && typeof value === 'object' && !Array.isArray(value) ? value as JsonObject : {}
}
function records(value: unknown): JsonObject[] { return Array.isArray(value) ? value.filter(row => row !== null && typeof row === 'object' && !Array.isArray(row)) : [] }
function finite(value: unknown): number { return typeof value === 'number' && Number.isFinite(value) ? value : 0 }
function positive(value: unknown): value is number { return typeof value === 'number' && Number.isSafeInteger(value) && value > 0 }
function dimension(value: unknown): number { return positive(value) ? value : 0 }
function findApp(apps: readonly JsonObject[], app: string): JsonObject | undefined {
  const name = app.trim().toLowerCase()
  return apps.find(row => [row.name, row.bundle_id, String(row.pid)].some(value => typeof value === 'string' && value.toLowerCase() === name))
}
function textFromResult(result: MCPToolCallResult): string { return result.content.filter(content => content.type === 'text').map(content => content.text).join('\n') }
function structured(result: MCPToolCallResult): JsonObject {
  if (result.structuredContent) return result.structuredContent
  // Some transports put the same JSON object in a text block.
  for (const content of result.content) {
    if (content.type !== 'text') continue
    try { const parsed: unknown = JSON.parse(content.text); if (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) return record(parsed) } catch { /* Ordinary markdown/text is not structured content. */ }
  }
  return {}
}
function requireSuccess(result: MCPToolCallResult): void {
  if (result.isError) {
    const message = textFromResult(result) || JSON.stringify(structured(result))
    if (/permission|accessibility.*(denied|missing)|screen.recording.*(denied|missing)/i.test(message)) throw new ComputerUseUnavailableError(PERMISSION_HINT + '\n' + message)
    throw new ClientError('computer_use', 'CuaDriver: ' + message)
  }
}
/** CuaDriver's own reason for a missing frame (e.g. an off-screen or minimized window), bounded. */
function screenshotErrorHint(value: unknown): string {
  const error = record(value)
  const code = typeof error.code === 'string' ? error.code.slice(0, 80) : ''
  const reason = typeof error.reason === 'string' ? error.reason.slice(0, 300) : ''
  if (!code && !reason) return ''
  return ` Driver reason${code ? ` (${code})` : ''}: ${reason || 'unspecified'}. If the window is off-screen or minimized, focus_app to bring it forward, then capture again.`
}
function recapture(reason: string): ClientError { return new ClientError('computer_use', 'CuaDriver: ' + reason + '; capture again before addressing elements or pixels') }
function throwIfAborted(signal?: AbortSignal): void { if (signal?.aborted) throw new ClientError('computer_use', 'computer-use action was cancelled before the driver call') }
function errorMessage(error: unknown): string { return error instanceof Error ? error.message : String(error) }
