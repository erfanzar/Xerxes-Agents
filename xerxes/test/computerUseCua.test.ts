// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { Buffer } from 'node:buffer'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { MCPToolCallOptions } from '../src/mcp/client.js'
import type { MCPToolCallResult } from '../src/mcp/types.js'
import { ComputerUseSession, ComputerUseUnavailableError, type ComputerUsePort } from '../src/tools/computerUse/backend.js'
import { CuaBackend, type CuaDriverMcpPort } from '../src/tools/computerUse/cuaBackend.js'
import { MCPClientCuaDriverPort } from '../src/tools/computerUse/cuaDriverPort.js'
import { DEFAULT_CUA_DRIVER, MacOSComputerUseHost } from '../src/tools/computerUse/host.js'
import { registerComputerUseTool } from '../src/tools/computerUse/tool.js'
import type { JsonObject } from '../src/types/toolCalls.js'

const png = Buffer.alloc(24)
Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]).copy(png)
png.writeUInt32BE(1567, 16)
png.writeUInt32BE(984, 20)
const PNG = png.toString('base64')
const ok = (data: JsonObject = {}): MCPToolCallResult => ({ content: [], structuredContent: data })
const failed = (message: string): MCPToolCallResult => ({ isError: true, content: [{ type: 'text', text: message }] })

class FakeDriver implements CuaDriverMcpPort {
  calls: Array<{ name: string; args: JsonObject }> = []
  starts = 0
  stops = 0
  available = true
  permissions = ok({ accessibility: true, screen_recording: true })
  apps = [{ name: 'Xerxes Agents', bundle_id: 'dev.xerxes.agents', pid: 42, running: true, active: true }]
  windows: JsonObject[] = [
    { pid: 42, window_id: 99, is_on_screen: false, z_index: 1, bounds: { x: 0, y: 0, width: 1800, height: 39 } },
    { pid: 42, window_id: 100, is_on_screen: true, z_index: 19, bounds: { x: 0, y: 39, width: 1800, height: 1130 } },
  ]
  state: JsonObject = {
    app_name: 'Xerxes Agents', window_title: 'Workspace', snapshot_id: 's00000001',
    screenshot_width: 1567, screenshot_height: 984, screenshot_scale: 2,
    window_bounds: { x: 0, y: 39, width: 1800, height: 1130 },
    elements: [
      { element_index: 0, element_token: 's00000001:0', role: 'AXWindow', label: 'Workspace', frame: { x: 0, y: 39, w: 1800, h: 1130 } },
      { element_index: 7, element_token: 's00000001:7', role: 'AXTextField', label: 'Search', value: 'hello', frame: { x: 100, y: 139, w: 100, h: 40 } },
    ],
  }
  actionResult: MCPToolCallResult = { content: [{ type: 'text', text: 'driver dispatched action; verify from a fresh state' }] }
  captureError: MCPToolCallResult | undefined
  jpeg = false
  snapshots = 0
  isAvailable(): boolean { return this.available }
  start(): void { this.starts++ }
  stop(): void { this.stops++ }
  callTool(name: string, args: JsonObject): MCPToolCallResult {
    this.calls.push({ name, args })
    if (name === 'check_permissions') return this.permissions
    if (name === 'list_apps') return ok({ apps: this.apps })
    if (name === 'list_windows') return ok({ windows: this.windows })
    if (name === 'get_window_state' || name === 'get_desktop_state') {
      this.snapshots++
      if (this.captureError) return this.captureError
      return {
        structuredContent: this.state,
        content: args.include_screenshot === false ? [] : [{ type: 'image', mimeType: this.jpeg ? 'image/jpeg' : 'image/png', data: this.jpeg ? '/9j/2Q==' : PNG }],
      }
    }
    if (name === 'get_cursor_position') return { content: [{ type: 'text', text: 'cursor at 14, 25 points' }], structuredContent: { x: 14, y: 25 } }
    return this.actionResult
  }
  last(): { name: string; args: JsonObject } | undefined { return this.calls.at(-1) }
}

async function fixture(): Promise<{ driver: FakeDriver; backend: CuaBackend; session: ComputerUseSession }> {
  const driver = new FakeDriver()
  const backend = new CuaBackend(driver)
  const session = new ComputerUseSession({ port: backend })
  await session.start()
  return { driver, backend, session }
}

const click = { button: 'left' as const, clickCount: 1, captureAfter: false }

test('0.28 capture returns an image and our 1-based elements with value, from the exact window', async () => {
  const { driver, session } = await fixture()
  const registry = new ToolRegistry()
  registerComputerUseTool(registry, { session })
  const value: unknown = JSON.parse(await registry.execute({ id: 'capture', type: 'function', function: { name: 'computer_use', arguments: { action: 'capture', app: 'dev.xerxes.agents' } } }, { metadata: {} }))
  expect(value).toMatchObject({ _multimodal: true, text_summary: expect.stringContaining('[2] AXTextField "Search — value: hello"') })
  expect(JSON.stringify(value)).toContain('data:image/png;base64,' + PNG)
  expect(JSON.stringify(value)).toContain('Backend: cua')
  expect(driver.last()).toEqual({ name: 'get_window_state', args: { pid: 42, window_id: 100, include_screenshot: true, include_accessibility_tree: true, max_dimension: 1568, max_elements: 300 } })
  expect(driver.calls[0]).toEqual({ name: 'check_permissions', args: { prompt: false, probe_direct_capture: false } })
  await session.stop()
  expect(driver.stops).toBe(1)
})

test('element click sends the right opaque token, and missing/unknown/stale indices never send input', async () => {
  const { driver, session } = await fixture()
  await expect(session.click({ ...click, element: 1 })).rejects.toThrow('capture again')
  expect(driver.calls.some(call => call.name === 'click')).toBe(false)
  await session.capture({ mode: 'som' })
  await expect(session.click({ ...click, element: 55 })).rejects.toThrow('capture again')
  const action = await session.click({ ...click, element: 2 })
  expect(action.ok).toBe(true)
  expect(driver.last()).toEqual({ name: 'click', args: { pid: 42, window_id: 100, element_token: 's00000001:7', button: 'left', count: 1 } })
  const count = driver.calls.length
  await expect(session.click({ ...click, element: 2 })).rejects.toThrow('stale')
  expect(driver.calls.length).toBe(count)
})

test('returned-image pixels are passed unchanged despite 2x backing scale and 1568 max_dimension', async () => {
  const { driver, session } = await fixture()
  const capture = await session.capture({ mode: 'som' })
  expect([capture.width, capture.height]).toEqual([1567, 984])
  await session.click({ ...click, x: 783, y: 492 })
  expect(driver.last()?.args).toEqual({ pid: 42, window_id: 100, scope: 'window', x: 783, y: 492, button: 'left', count: 1 })
  await session.drag({ startX: 100, startY: 200, endX: 800, endY: 600, captureAfter: false })
  expect(driver.last()).toEqual({ name: 'drag', args: { pid: 42, window_id: 100, scope: 'window', from_x: 100, from_y: 200, to_x: 800, to_y: 600 } })
  await expect(session.click({ ...click, x: 1567, y: 0 })).rejects.toThrow('outside')
  await session.scroll({ dx: -80, dy: 120, x: 200, y: 300, captureAfter: false })
  expect(driver.calls.slice(-2)).toEqual([
    { name: 'scroll', args: { pid: 42, window_id: 100, scope: 'window', x: 200, y: 300, direction: 'down', amount: 3, by: 'line' } },
    { name: 'scroll', args: { pid: 42, window_id: 100, scope: 'window', x: 200, y: 300, direction: 'left', amount: 2, by: 'line' } },
  ])
})

test('element drags convert screen-point AX frames into returned-image pixels, not Retina pixels', async () => {
  const { driver, session } = await fixture()
  const capture = await session.capture({ mode: 'som' })
  const bounds = capture.elements?.[1]?.bounds
  expect(bounds).toEqual([87, 87, 87, 35])
  await session.drag({ startElement: 2, endX: 800, endY: 600, captureAfter: false })
  expect(driver.last()?.args.from_x).toBe(130.5)
  expect(driver.last()?.args.from_y).toBe(104.5)
})

test.each(['doubleClick', 'rightClick', 'middleClick', 'tripleClick'] as const)('%s uses a 0.28 operation that actually supports its button/count', async method => {
  const { driver, session } = await fixture()
  await session.capture({ mode: 'som' })
  await session[method]({ element: 2, captureAfter: false })
  if (method === 'doubleClick') {
    expect(driver.last()).toEqual({ name: 'double_click', args: { pid: 42, window_id: 100, element_token: 's00000001:7' } })
  } else if (method === 'tripleClick') {
    expect(driver.last()).toEqual({ name: 'click', args: { pid: 42, window_id: 100, scope: 'window', x: 130.5, y: 104.5, button: 'left', count: 3 } })
  } else {
    expect(driver.last()).toEqual({ name: 'click', args: { pid: 42, window_id: 100, element_token: 's00000001:7', button: method === 'rightClick' ? 'right' : 'middle', count: 1 } })
  }
})

test('AX-only captures keep token addressing; tokenless elements require snapshot_id and original index', async () => {
  const { driver, session } = await fixture()
  driver.state.elements = [{ element_index: 7, role: 'AXButton', label: 'Save' }]
  const capture = await session.capture({ mode: 'ax' })
  expect(capture.pngB64).toBeUndefined()
  await session.click({ ...click, element: 1 })
  expect(driver.last()?.args).toEqual({ pid: 42, window_id: 100, element_index: 7, snapshot_id: 's00000001', button: 'left', count: 1 })
  delete driver.state.snapshot_id
  await session.capture({ mode: 'ax' })
  await expect(session.click({ ...click, element: 1 })).rejects.toThrow('no snapshot-bound handle')
})

test('type by element, set_value, chords, single keys and frontmost typing use 0.28 names', async () => {
  const { driver, session } = await fixture()
  await session.type({ text: 'frontmost', captureAfter: false })
  expect(driver.last()).toEqual({ name: 'type_text', args: { pid: 42, text: 'frontmost' } })
  await session.capture({ mode: 'som' })
  await session.type({ text: 'into field', element: 2, captureAfter: false })
  expect(driver.last()).toEqual({ name: 'type_text', args: { pid: 42, window_id: 100, element_token: 's00000001:7', text: 'into field' } })
  await session.capture({ mode: 'som' })
  await session.setValue({ element: 2, value: '', captureAfter: false })
  expect(driver.last()).toEqual({ name: 'set_value', args: { pid: 42, window_id: 100, element_token: 's00000001:7', value: '' } })
  await session.key({ key: 'command+shift+a', captureAfter: false })
  expect(driver.last()).toEqual({ name: 'hotkey', args: { pid: 42, window_id: 100, scope: 'window', keys: ['cmd', 'shift', 'a'] } })
  await session.key({ key: 'Enter', captureAfter: false })
  expect(driver.last()).toEqual({ name: 'press_key', args: { pid: 42, window_id: 100, scope: 'window', key: 'return' } })
  const count = driver.calls.length
  expect((await session.key({ key: 'command+shfit+a', captureAfter: false })).ok).toBe(false)
  expect(driver.calls.length).toBe(count)
})

test('driver isError and stale-token errors stay ok:false; capture_after is fresh and only after success', async () => {
  const { driver, session } = await fixture()
  await session.capture({ mode: 'som' })
  driver.actionResult = failed('stale element token')
  const refused = await session.click({ ...click, element: 2, captureAfter: true })
  expect(refused.ok).toBe(false)
  expect(refused.message).toContain('stale element token')
  expect(refused.message).toContain('capture again')
  expect(driver.snapshots).toBe(1)
  driver.actionResult = ok({ effect: 'unverifiable' })
  const action = await session.click({ ...click, x: 10, y: 20, captureAfter: true })
  expect(action.ok).toBe(true)
  expect(action.capture?.elements).toHaveLength(2)
  expect(driver.snapshots).toBe(2)
  expect(driver.last()?.name).toBe('get_window_state')
  expect(driver.last()?.args.pid).toBe(42)
  expect(driver.last()?.args.window_id).toBe(100)
})

test('JPEG media type is preserved and failed captures clear the previous snapshot', async () => {
  const { driver, session } = await fixture()
  driver.jpeg = true
  expect((await session.capture({ mode: 'som' })).mediaType).toBe('image/jpeg')
  driver.captureError = failed('window_id_not_found')
  await expect(session.capture({ mode: 'som' })).rejects.toThrow('window_id_not_found')
  await expect(session.click({ ...click, element: 2 })).rejects.toThrow('capture again')
})

test('no-app/no-window capture falls back to desktop and remembers native-pixel scope', async () => {
  const { driver, session } = await fixture()
  driver.windows = []
  const capture = await session.capture({ mode: 'som' })
  expect(driver.last()?.name).toBe('get_desktop_state')
  expect(capture.warning).toContain('native pixels')
  await session.click({ ...click, x: 10, y: 20 })
  expect(driver.last()?.args).toEqual({ scope: 'desktop', x: 10, y: 20, button: 'left', count: 1 })
  await expect(session.capture({ mode: 'som', app: 'Xerxes Agents' })).rejects.toThrow('no usable window')
  expect(driver.calls.some(call => call.name === 'launch_app')).toBe(false)
})

test('focus_app explicitly brings a running app forward and clears handles; capture never launches', async () => {
  const { driver, session } = await fixture()
  await session.capture({ mode: 'som' })
  await session.focusApp('Xerxes Agents')
  expect(driver.last()).toEqual({ name: 'bring_to_front', args: { pid: 42 } })
  await expect(session.click({ ...click, element: 2 })).rejects.toThrow('capture again')
  await expect(session.capture({ mode: 'som', app: 'Not running' })).rejects.toThrow('no usable window')
  expect(driver.calls.some(call => call.name === 'launch_app')).toBe(false)
  driver.actionResult = failed('launch refused')
  const result = await session.focusApp('Not running')
  expect(driver.last()).toEqual({ name: 'launch_app', args: { name: 'Not running' } })
  expect(result.ok).toBe(false)
})

test('cursor reads preserve logical-point metadata; mouse_move uses an exact screenshot target; wait is abortable', async () => {
  const { driver, session } = await fixture()
  expect((await session.cursorPosition()).meta).toEqual({ x: 14, y: 25 })
  await session.capture({ mode: 'som' })
  await session.mouseMove({ x: 400, y: 200, captureAfter: false })
  expect(driver.last()).toEqual({ name: 'move_cursor', args: { target: { kind: 'window', pid: 42, window_id: 100 }, scope: 'window', x: 400, y: 200 } })
  const controller = new AbortController()
  const waiting = session.wait(10000, controller.signal)
  await Bun.sleep(1)
  controller.abort()
  await expect(waiting).rejects.toThrow('cancelled')
  expect(driver.calls.some(call => call.name === 'wait')).toBe(false)
})

test('missing/unknown permissions fail with an actionable unavailable error, never silently falling back', async () => {
  for (const permissions of [ok({ accessibility: false, screen_recording: true }), ok({ accessibility: true, screen_recording: false }), ok({}), failed('permission denied')]) {
    const driver = new FakeDriver()
    driver.permissions = permissions
    const session = new ComputerUseSession({ port: new CuaBackend(driver) })
    await expect(session.capture({ mode: 'som' })).rejects.toBeInstanceOf(ComputerUseUnavailableError)
    await expect(session.capture({ mode: 'som' })).rejects.toThrow('cua-driver permissions grant')
    expect(driver.calls.every(call => call.name === 'check_permissions')).toBe(true)
    expect(driver.stops).toBe(2)
  }
})

function fakeHost(binaryExists = true): { host: MacOSComputerUseHost; driver: FakeDriver; macos: ComputerUsePort; paths: string[] } {
  const driver = new FakeDriver()
  const macos = new CuaBackend(driver)
  const paths: string[] = []
  const host = new MacOSComputerUseHost({ platform: 'darwin', hostAvailable: () => true, fileExists: () => binaryExists, createDriver: path => { paths.push(path); return driver }, createMacOSPort: () => macos })
  return { host, driver, macos, paths }
}

test.each(['auto', 'cua', 'macos'] as const)('backend selection %s stays lazy and disposal closes owned driver', async backend => {
  const { host, driver, paths } = fakeHost()
  const options = host.toolOptions({ computer_use_backend: backend }, {})
  expect(options).toBeDefined()
  expect(driver.starts).toBe(0)
  await options?.session?.capture({ mode: 'som' })
  expect(driver.starts).toBe(1)
  expect(paths).toEqual(backend === 'macos' ? [] : [DEFAULT_CUA_DRIVER])
  await host.dispose()
  expect(driver.stops).toBe(1)
})

test('auto chooses macos when no binary exists; cua forces missing driver availability errors', async () => {
  const fallback = fakeHost(false)
  await fallback.host.toolOptions({}, {})?.session?.capture({ mode: 'som' })
  expect(fallback.paths).toEqual([])
  await fallback.host.dispose()
  const forced = fakeHost(false)
  forced.driver.available = false
  const options = forced.host.toolOptions({ computer_use_backend: 'cua' }, {})
  await expect(options?.session?.capture({ mode: 'som' })).rejects.toThrow('CuaDriver is unavailable')
  expect(forced.paths).toEqual([DEFAULT_CUA_DRIVER])
  await forced.host.dispose()
})

test('opt-out wins; configured path uses env precedence; one runtime transport survives rebuilds and callers have separate snapshots', async () => {
  const { host, driver, paths } = fakeHost()
  expect(host.toolOptions({ computer_use_enabled: false, computer_use_max_edge: 'invalid' }, {})).toBeUndefined()
  expect(host.toolOptions({ computer_use_enabled: true }, { XERXES_COMPUTER_USE: 'off' })).toBeUndefined()
  expect(paths).toEqual([])
  const settings = { computer_use_backend: 'auto', computer_use_cua_driver: '/settings/driver' }
  const env = { XERXES_CUA_DRIVER: '/env/driver' }
  const a = host.toolOptions(settings, env)
  const b = host.toolOptions(settings, env)
  expect(a?.session).toBe(b?.session)
  expect(paths).toEqual(['/env/driver'])
  const first = await a?.resolveSession?.({ sessionId: 'one', metadata: {} })
  const second = await a?.resolveSession?.({ sessionId: 'two', metadata: {} })
  await first?.capture({ mode: 'som' })
  await expect(second?.click({ ...click, element: 1 })).rejects.toThrow('capture again')
  expect(driver.calls.some(call => call.name === 'click')).toBe(false)
  await host.dispose()
  expect(driver.stops).toBe(1)
})

test('window resolution avoids unnamed helper windows even when the app is on another Space', async () => {
  const { driver, session } = await fixture()
  driver.windows = [
    { pid: 42, window_id: 99, is_on_screen: false, z_index: 1, title: '', bounds: { width: 500, height: 500 } },
    { pid: 42, window_id: 100, is_on_screen: false, z_index: 19, title: 'Workspace', bounds: { width: 1800, height: 1130 } },
  ]
  await session.capture({ mode: 'som', app: 'Xerxes Agents' })
  expect(driver.last()?.args.window_id).toBe(100)
})

test('unprovable driver pixel frames never permit pixel input, but remain honest AX captures', async () => {
  const { driver, session } = await fixture()
  driver.state.screenshot_frame_valid = false
  driver.state.degraded_reason = 'px_frame_mismatch'
  const capture = await session.capture({ mode: 'som' })
  expect(capture.pngB64).toBeUndefined()
  expect(capture.warning).toContain('px_frame_mismatch')
  driver.state.screenshot_error = { code: 'px_capture_unavailable', reason: 'window 100 capture failed: off-screen' }
  const offscreen = await session.capture({ mode: 'som' })
  expect(offscreen.warning).toContain('px_capture_unavailable')
  expect(offscreen.warning).toContain('off-screen')
  expect(offscreen.warning).toContain('focus_app')
  await expect(session.click({ ...click, x: 10, y: 20 })).rejects.toThrow('no pixel frame')
  expect(driver.calls.some(call => call.name === 'click')).toBe(false)
  await session.click({ ...click, element: 2 })
  expect(driver.last()?.args.element_token).toBe('s00000001:7')
})

test('invalid CuaDriver path fails explicitly, except opt-out still wins', () => {
  const { host } = fakeHost()
  expect(() => host.toolOptions({ computer_use_cua_driver: 'relative/path' }, {})).toThrow('absolute')
  expect(host.toolOptions({ computer_use_enabled: false, computer_use_cua_driver: 'relative/path' }, {})).toBeUndefined()
})

class FakeClient {
  connects = 0
  disconnects = 0
  signals: AbortSignal[] = []
  hangConnect = false
  hangCall = false
  connect(): Promise<void> { this.connects++; return this.hangConnect ? new Promise(() => {}) : Promise.resolve() }
  disconnect(): Promise<void> { this.disconnects++; return Promise.resolve() }
  callTool(_name: string, _args: JsonObject, options: MCPToolCallOptions = {}): Promise<MCPToolCallResult> {
    if (options.signal) this.signals.push(options.signal)
    return this.hangCall ? new Promise(() => {}) : Promise.resolve(ok({ elements: [{ element_index: 0, element_token: 's00000001:0' }] }))
  }
}

test('stdio MCP adapter connects lazily once, forwards structured content, and closes once on dispose', async () => {
  const client = new FakeClient()
  const port = new MCPClientCuaDriverPort(client)
  expect(client.connects).toBe(0)
  await Promise.all([port.start(), port.start()])
  expect(client.connects).toBe(1)
  expect((await port.callTool('get_window_state', {})).structuredContent?.elements).toHaveLength(1)
  expect(client.signals).toHaveLength(1)
  await port.stop()
  await port.stop()
  expect(client.disconnects).toBe(1)
  await expect(port.callTool('click', {})).rejects.toThrow('closed')
})

test.each(['connect', 'call'] as const)('stdio adapter bounds %s timeout and closes without retrying', async phase => {
  const client = new FakeClient()
  client.hangConnect = phase === 'connect'
  client.hangCall = phase === 'call'
  const port = new MCPClientCuaDriverPort(client, { timeoutMs: 10 })
  await expect(port.callTool('click', {})).rejects.toThrow('timed out after 10ms')
  expect(client.connects).toBe(1)
  expect(client.disconnects).toBe(1)
  expect(client.signals.every(signal => signal.aborted)).toBe(true)
  await expect(port.callTool('click', {})).rejects.toThrow('closed')
})

test('a second caller can abort while waiting on another caller\'s shared handshake', async () => {
  const client = new FakeClient()
  client.hangConnect = true
  const port = new MCPClientCuaDriverPort(client)
  const controller = new AbortController()
  const outcomes = Promise.allSettled([port.start(), port.start(controller.signal)])
  await Bun.sleep(1)
  controller.abort()
  const results = await outcomes
  expect(results[1]).toMatchObject({ status: 'rejected', reason: expect.objectContaining({ message: expect.stringContaining('cancelled') }) })
  expect(results[0].status).toBe('rejected')
  expect(client.connects).toBe(1)
  expect(client.disconnects).toBe(1)
})

test('stdio adapter disposal rejects an in-flight call promptly, even if the client never settles it', async () => {
  const client = new FakeClient()
  client.hangCall = true
  const port = new MCPClientCuaDriverPort(client)
  const pending = port.callTool('click', {})
  await Bun.sleep(1)
  await port.stop()
  await expect(pending).rejects.toThrow('closed')
  expect(client.signals[0]?.aborted).toBe(true)
  expect(client.disconnects).toBe(1)
})

class EndingClient extends FakeClient {
  calls: { name: string; args: JsonObject }[] = []
  ended = true
  reviveFails = false
  override callTool(name: string, args: JsonObject): Promise<MCPToolCallResult> {
    this.calls.push({ name, args })
    if (name === 'start_session') {
      if (this.reviveFails) return Promise.resolve({ isError: true, content: [{ type: 'text', text: 'cannot revive' }] })
      this.ended = false
      return Promise.resolve({ content: [{ type: 'text', text: `Session '${String(args.session)}' is active.` }] })
    }
    if (this.ended) {
      return Promise.resolve({ isError: true, content: [{ type: 'text', text: `session 'mcp-1-2' has ended; tool call '${name}' was rejected. Call start_session with this id to revive it before issuing further actions, or use a new session id.` }] })
    }
    return Promise.resolve(ok({ done: name }))
  }
}

test('stdio adapter revives an ended CuaDriver session and repeats the rejected call once', async () => {
  const client = new EndingClient()
  const port = new MCPClientCuaDriverPort(client)
  const result = await port.callTool('list_apps', {})
  expect(result.isError).not.toBe(true)
  expect(client.calls.map(call => call.name)).toEqual(['list_apps', 'start_session', 'list_apps'])
  expect(client.calls[1]?.args).toEqual({ session: 'mcp-1-2' })
  expect(client.disconnects).toBe(0)
})

test('a failed revival returns the original rejection without repeating the call', async () => {
  const client = new EndingClient()
  client.reviveFails = true
  const port = new MCPClientCuaDriverPort(client)
  const result = await port.callTool('click', {})
  expect(result.isError).toBe(true)
  expect(client.calls.map(call => call.name)).toEqual(['click', 'start_session'])
})

test('other driver errors are never retried', async () => {
  const client = new EndingClient()
  client.ended = false
  const port = new MCPClientCuaDriverPort(client)
  client.callTool = (name: string, args: JsonObject) => {
    client.calls.push({ name, args })
    return Promise.resolve({ isError: true, content: [{ type: 'text', text: 'element token is stale' }] })
  }
  expect((await port.callTool('click', {})).isError).toBe(true)
  expect(client.calls).toHaveLength(1)
})

test('stdio adapter propagates cancellation to MCP, closes, and never retries input', async () => {
  const client = new FakeClient()
  client.hangCall = true
  const port = new MCPClientCuaDriverPort(client)
  const controller = new AbortController()
  const pending = port.callTool('click', {}, controller.signal)
  await Bun.sleep(1)
  controller.abort()
  await expect(pending).rejects.toThrow('cancelled')
  expect(client.signals[0]?.aborted).toBe(true)
  expect(client.disconnects).toBe(1)
})
