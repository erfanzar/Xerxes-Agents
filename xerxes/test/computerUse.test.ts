// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { Buffer } from 'node:buffer'

import { expect, test } from 'bun:test'

import { ClientError } from '../src/core/errors.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { MCPToolCallResult } from '../src/mcp/types.js'
import {
  ComputerUseSession,
  type ActionResult,
  type CaptureRequest,
  type CaptureResult,
  type ClickRequest,
  type ComputerUsePort,
  type DragRequest,
  type KeyRequest,
  type ScrollRequest,
  type SetValueRequest,
  type TextRequest,
} from '../src/tools/computerUse/backend.js'
import {
  CuaBackend,
  type CuaDriverMcpPort,
  parseCuaElements,
  parseCuaWindows,
} from '../src/tools/computerUse/cuaBackend.js'
import { registerComputerUseTool } from '../src/tools/computerUse/tool.js'
import type { JsonObject, ToolCall } from '../src/types/toolCalls.js'

test('computer_use delegates every privileged action only through the injected port', async () => {
  const port = new FakeComputerPort()
  const registry = new ToolRegistry()
  registerComputerUseTool(registry, { session: new ComputerUseSession({ port }) })

  const capture = await execute(registry, {
    action: 'capture',
    mode: 'som',
    max_elements: 1,
  })
  expect(capture._multimodal).toBe(true)
  expect(capture.text_summary).toContain('Screen capture: 1280x720')
  expect(capture.text_summary).toContain('Elements (1 shown+):')
  expect(JSON.stringify(capture)).toContain('data:image/png;base64,UE5H')

  const action = await execute(registry, {
    action: 'click',
    element: 2,
    capture_after: true,
  })
  expect(action._multimodal).toBe(true)
  expect(action.text_summary).toContain('Screen capture: 1280x720')
  expect(JSON.stringify(action)).toContain('Action: click')

  expect(await execute(registry, {
    action: 'drag',
    start_x: 10,
    start_y: 20,
    end_element: 4,
  })).toEqual({
    ok: true,
    action: 'drag',
    message: 'dragged',
    meta: {},
  })
  expect(await execute(registry, { action: 'scroll', dx: -4, dy: 80 })).toEqual({
    ok: true,
    action: 'scroll',
    message: 'scrolled',
    meta: {},
  })
  expect(await execute(registry, { action: 'type', text: 'hello' })).toEqual({
    ok: true,
    action: 'type',
    message: 'typed hello',
    meta: {},
  })
  expect(await execute(registry, { action: 'key', key: 'Enter' })).toEqual({
    ok: true,
    action: 'key',
    message: 'key Enter',
    meta: {},
  })
  expect(await execute(registry, { action: 'set_value', element: 2, value: '' })).toEqual({
    ok: true,
    action: 'set_value',
    message: 'set',
    meta: {},
  })
  expect(await execute(registry, { action: 'wait', ms: 5 })).toEqual({
    ok: true,
    action: 'wait',
    message: 'waited 5',
    meta: {},
  })
  expect(await execute(registry, { action: 'list_apps' })).toEqual({
    ok: true,
    action: 'list_apps',
    message: 'one app',
    meta: {},
  })
  expect(await execute(registry, { action: 'focus_app', app: 'Safari' })).toEqual({
    ok: true,
    action: 'focus_app',
    message: 'focused Safari',
    meta: {},
  })

  expect(port.events).toEqual([
    'available',
    'start',
    'capture:som',
    'available',
    'click:2',
    'available',
    'drag:10,20->4',
    'available',
    'scroll:-4,80',
    'available',
    'type:hello',
    'available',
    'key:Enter',
    'available',
    'set:2:',
    'available',
    'wait:5',
    'available',
    'list_apps',
    'available',
    'focus:Safari',
  ])
})

test('computer_use rejects malformed action targets and never treats a browser as a desktop fallback', async () => {
  const registry = new ToolRegistry()
  registerComputerUseTool(registry, { session: new ComputerUseSession({ port: new FakeComputerPort() }) })

  await expect(registry.execute(call({ action: 'click', x: 2 }), { metadata: {} }))
    .rejects.toThrow('must provide both x and y coordinates')
  await expect(registry.execute(call({ action: 'drag', start_element: 1, end_x: 2, end_y: 3, app: 'Safari' }), { metadata: {} }))
    .rejects.toThrow('is not valid for action=drag')
  await expect(registry.execute(call({ action: 'capture', mode: 'bad' }), { metadata: {} }))
    .rejects.toThrow('must be som, vision, or ax')
  await expect(registry.execute(call({ action: 'scroll', dx: 100_001 }), { metadata: {} }))
    .rejects.toThrow('must be an integer between -100000 and 100000')

  const unconfigured = new ToolRegistry()
  registerComputerUseTool(unconfigured, { session: new ComputerUseSession() })
  await expect(unconfigured.execute(call({ action: 'capture' }), { metadata: {} }))
    .rejects.toThrow('no ComputerUsePort is configured')

  const unavailable = new ToolRegistry()
  registerComputerUseTool(unavailable, { session: new ComputerUseSession({ port: new FakeComputerPort(false) }) })
  await expect(unavailable.execute(call({ action: 'capture' }), { metadata: {} }))
    .rejects.toThrow('configured ComputerUsePort is unavailable')
})

test('computer-use session rejects malformed backend results before model-visible serialization', async () => {
  const port = new FakeComputerPort()
  port.captureResult = {
    mode: 'som',
    width: 1,
    height: 1,
    elements: [
      { index: 1, role: 'AXButton' },
      { index: 1, role: 'AXButton' },
    ],
  }
  const session = new ComputerUseSession({ port })

  await expect(session.capture({ mode: 'som' })).rejects.toBeInstanceOf(ClientError)
  await expect(session.capture({ mode: 'som' })).rejects.toThrow('duplicate element index')
})

test('CuaBackend maps an injected MCP transport without launching or installing anything itself', async () => {
  const driver = new FakeCuaDriver()
  const backend = new CuaBackend(driver)
  const session = new ComputerUseSession({ port: backend })
  const registry = new ToolRegistry()
  registerComputerUseTool(registry, { session })

  const capture = await execute(registry, { action: 'capture', mode: 'som' })
  expect(capture._multimodal).toBe(true)
  expect(capture.text_summary).toContain('Screen capture: 640x480')
  expect(capture.text_summary).toContain('[7] AXButton "Save"')

  expect(await execute(registry, { action: 'right_click', x: 12, y: 8 })).toEqual({
    ok: true,
    action: 'right_click',
    message: 'clicked',
    meta: {},
  })
  expect(driver.events).toEqual([
    'available',
    'available',
    'start',
    'capture:{"mode":"som"}',
    'available',
    'click:{"button":"right","click_count":1,"x":12,"y":8}',
  ])

  expect(parseCuaWindows('- Safari (pid 123) "Docs" [window_id: 456]')).toEqual([
    { app: 'Safari', pid: 123, window_id: 456 },
  ])
  expect(parseCuaElements('- [7] AXButton "Save"\n[8] AXTextField (2) id=Search')).toEqual([
    { index: 7, role: 'AXButton', label: 'Save' },
    { index: 8, role: 'AXTextField', label: 'Search' },
  ])
})

async function execute(registry: ToolRegistry, arguments_: JsonObject): Promise<JsonObject> {
  return JSON.parse(await registry.execute(call(arguments_), { metadata: {} })) as JsonObject
}

function call(arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: 'function',
    function: { name: 'computer_use', arguments: arguments_ },
  }
}

class FakeComputerPort implements ComputerUsePort {
  readonly events: string[] = []
  captureResult: CaptureResult = capture()

  constructor(private readonly available = true) {}

  isAvailable(): boolean {
    this.events.push('available')
    return this.available
  }

  start(): void {
    this.events.push('start')
  }

  capture(request: CaptureRequest): CaptureResult {
    this.events.push('capture:' + request.mode)
    return this.captureResult
  }

  click(request: ClickRequest): ActionResult {
    this.events.push('click:' + (request.element ?? request.x ?? 'unknown'))
    return result('click', 'clicked', request.captureAfter)
  }

  doubleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>): ActionResult {
    this.events.push('double:' + (request.element ?? request.x ?? 'unknown'))
    return result('double_click', 'double clicked', request.captureAfter)
  }

  rightClick(request: Omit<ClickRequest, 'button' | 'clickCount'>): ActionResult {
    this.events.push('right:' + (request.element ?? request.x ?? 'unknown'))
    return result('right_click', 'right clicked', request.captureAfter)
  }

  middleClick(request: Omit<ClickRequest, 'button' | 'clickCount'>): ActionResult {
    this.events.push('middle:' + (request.element ?? request.x ?? 'unknown'))
    return result('middle_click', 'middle clicked', request.captureAfter)
  }

  drag(request: DragRequest): ActionResult {
    this.events.push('drag:' + (request.startX ?? request.startElement) + ',' + (request.startY ?? '') + '->' + (request.endElement ?? request.endX))
    return result('drag', 'dragged')
  }

  scroll(request: ScrollRequest): ActionResult {
    this.events.push('scroll:' + request.dx + ',' + request.dy)
    return result('scroll', 'scrolled')
  }

  type(request: TextRequest): ActionResult {
    this.events.push('type:' + request.text)
    return result('type', 'typed ' + request.text)
  }

  key(request: KeyRequest): ActionResult {
    this.events.push('key:' + request.key)
    return result('key', 'key ' + request.key)
  }

  setValue(request: SetValueRequest): ActionResult {
    this.events.push('set:' + (request.element ?? '') + ':' + request.value)
    return result('set_value', 'set')
  }

  wait(ms: number): ActionResult {
    this.events.push('wait:' + ms)
    return result('wait', 'waited ' + ms)
  }

  listApps(): ActionResult {
    this.events.push('list_apps')
    return result('list_apps', 'one app')
  }

  focusApp(app: string): ActionResult {
    this.events.push('focus:' + app)
    return result('focus_app', 'focused ' + app)
  }
}

class FakeCuaDriver implements CuaDriverMcpPort {
  readonly events: string[] = []

  isAvailable(): boolean {
    this.events.push('available')
    return true
  }

  start(): void {
    this.events.push('start')
  }

  callTool(name: string, arguments_: JsonObject): MCPToolCallResult {
    this.events.push(name + ':' + JSON.stringify(arguments_))
    if (name === 'capture') {
      return {
        content: [
          { type: 'text', text: '- [7] AXButton "Save"' },
          { type: 'image', mimeType: 'image/png', data: png640x480() },
        ],
      }
    }
    return { content: [{ type: 'text', text: 'clicked' }] }
  }
}

function capture(): CaptureResult {
  return {
    mode: 'som',
    width: 1280,
    height: 720,
    pngB64: 'UE5H',
    elements: [
      { index: 1, role: 'AXTextField', label: 'Search' },
      { index: 2, role: 'AXButton', label: 'Submit' },
    ],
  }
}

function result(action: string, message: string, captureAfter = false): ActionResult {
  return {
    ok: true,
    action,
    message,
    ...(captureAfter ? { capture: capture() } : {}),
  }
}

function png640x480(): string {
  const bytes = Buffer.alloc(24)
  bytes.writeUInt32BE(640, 16)
  bytes.writeUInt32BE(480, 20)
  return bytes.toString('base64')
}
