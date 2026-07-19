// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../../core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../../types/toolCalls.js'
import { optionalBoolean, optionalInteger } from '../inputs.js'
import {
  ComputerUseSession,
  ComputerUseUnavailableError,
  type ActionResult,
  type CaptureResult,
} from './backend.js'
import { COMPUTER_USE_DEFINITION } from './schema.js'

const DEFAULT_CAPTURE_MODE = 'som'
const DEFAULT_MAX_ELEMENTS = 100
const DEFAULT_WAIT_MS = 1_000
const MAX_COORDINATE = 100_000
const MAX_ELEMENTS = 1_000
const MAX_SCROLL_DELTA = 100_000
const MAX_WAIT_MS = 300_000

const ACTIONS = [
  'capture',
  'click',
  'double_click',
  'triple_click',
  'right_click',
  'middle_click',
  'mouse_move',
  'cursor_position',
  'drag',
  'scroll',
  'type',
  'key',
  'set_value',
  'wait',
  'list_apps',
  'focus_app',
] as const

type ComputerUseAction = (typeof ACTIONS)[number]

const ACTION_FIELDS: Readonly<Record<ComputerUseAction, readonly string[]>> = {
  capture: ['action', 'mode', 'app', 'max_elements'],
  click: ['action', 'element', 'x', 'y', 'capture_after'],
  double_click: ['action', 'element', 'x', 'y', 'capture_after'],
  triple_click: ['action', 'element', 'x', 'y', 'capture_after'],
  right_click: ['action', 'element', 'x', 'y', 'capture_after'],
  middle_click: ['action', 'element', 'x', 'y', 'capture_after'],
  mouse_move: ['action', 'x', 'y', 'capture_after'],
  cursor_position: ['action'],
  drag: [
    'action',
    'start_element',
    'start_x',
    'start_y',
    'end_element',
    'end_x',
    'end_y',
    'capture_after',
  ],
  scroll: ['action', 'element', 'x', 'y', 'dx', 'dy', 'capture_after'],
  type: ['action', 'text', 'capture_after'],
  key: ['action', 'key', 'capture_after'],
  set_value: ['action', 'element', 'value', 'capture_after'],
  wait: ['action', 'ms'],
  list_apps: ['action'],
  focus_app: ['action', 'app'],
}

export interface ComputerUseToolsOptions {
  /** Static desktop session for a single-session host. */
  readonly session?: ComputerUseSession
  /** Resolve the desktop session dynamically for per-agent or per-session ownership. */
  readonly resolveSession?: (
    context: ToolExecutionContext,
  ) => ComputerUseSession | undefined | Promise<ComputerUseSession | undefined>
}

export interface MultimodalComputerUseResult {
  readonly _multimodal: true
  readonly content: readonly [JsonObject, JsonObject]
  readonly text_summary: string
}

export interface TextComputerUseResult {
  readonly result: string
}

export type ComputerUseResult = JsonObject | MultimodalComputerUseResult | TextComputerUseResult

/** Register computer_use only against a deliberately injected session or resolver. */
export function registerComputerUseTool(
  registry: ToolRegistry,
  options: ComputerUseToolsOptions = {},
  agentId = 'default',
): void {
  registry.register(COMPUTER_USE_DEFINITION, (inputs, context, signal) => computerUse(inputs, context, options, signal), agentId)
}

/** Execute one strictly validated desktop action through the configured session. */
export async function computerUse(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: ComputerUseToolsOptions,
  signal?: AbortSignal,
): Promise<ComputerUseResult> {
  const action = actionFrom(inputs)
  assertActionFields(inputs, action)
  const session = await resolveSession(context, options)

  switch (action) {
    case 'capture': {
      const maxElements = boundedInteger(inputs, 'max_elements', DEFAULT_MAX_ELEMENTS, 1, MAX_ELEMENTS)
      const app = optionalNonBlankString(inputs, 'app')
      const capture = await session.capture({
        mode: captureMode(inputs),
        ...(app === undefined ? {} : { app }),
      }, signal)
      return formatCaptureResult(capture, maxElements)
    }
    case 'click':
      return formatActionResult(await session.click(clickRequest(inputs), signal), captureAfter(inputs))
    case 'double_click':
      return formatActionResult(await session.doubleClick(secondaryClickRequest(inputs), signal), captureAfter(inputs))
    case 'triple_click':
      return formatActionResult(await session.tripleClick(secondaryClickRequest(inputs), signal), captureAfter(inputs))
    case 'right_click':
      return formatActionResult(await session.rightClick(secondaryClickRequest(inputs), signal), captureAfter(inputs))
    case 'middle_click':
      return formatActionResult(await session.middleClick(secondaryClickRequest(inputs), signal), captureAfter(inputs))
    case 'mouse_move':
      return formatActionResult(await session.mouseMove(mouseMoveRequest(inputs), signal), captureAfter(inputs))
    case 'cursor_position':
      return formatActionResult(await session.cursorPosition(signal))
    case 'drag':
      return formatActionResult(await session.drag(dragRequest(inputs), signal), captureAfter(inputs))
    case 'scroll':
      return formatActionResult(await session.scroll(scrollRequest(inputs), signal), captureAfter(inputs))
    case 'type':
      return formatActionResult(
        await session.type({ text: requiredStringValue(inputs, 'text'), captureAfter: captureAfter(inputs) }, signal),
        captureAfter(inputs),
      )
    case 'key':
      return formatActionResult(
        await session.key({ key: requiredNonBlankString(inputs, 'key'), captureAfter: captureAfter(inputs) }, signal),
        captureAfter(inputs),
      )
    case 'set_value':
      return formatActionResult(
        await session.setValue({
          value: requiredStringValue(inputs, 'value'),
          captureAfter: captureAfter(inputs),
          ...optionalElement(inputs, 'element'),
        }, signal),
        captureAfter(inputs),
      )
    case 'wait':
      return formatActionResult(await session.wait(boundedInteger(inputs, 'ms', DEFAULT_WAIT_MS, 0, MAX_WAIT_MS), signal))
    case 'list_apps':
      return formatActionResult(await session.listApps(signal))
    case 'focus_app':
      return formatActionResult(await session.focusApp(requiredNonBlankString(inputs, 'app'), signal))
  }
}

/** Serialize a capture as Python-compatible multimodal tool content when an image is available. */
export function formatCaptureResult(capture: CaptureResult, maxElements = DEFAULT_MAX_ELEMENTS): ComputerUseResult {
  const boundedMaxElements = validateRange(maxElements, 'max_elements', 1, MAX_ELEMENTS)
  const allElements = capture.elements ?? []
  const elements = allElements.slice(0, boundedMaxElements)
  const truncated = allElements.length > elements.length
  const lines = [
    'Screen capture: ' + capture.width + 'x' + capture.height,
    'Mode: ' + capture.mode,
  ]
  if (capture.app) lines.push('App: ' + capture.app)
  if (capture.windowTitle) lines.push('Window: ' + capture.windowTitle)
  if (elements.length) {
    lines.push('')
    lines.push('Elements (' + elements.length + ' shown' + (truncated ? '+' : '') + '):')
    for (const element of elements) {
      lines.push('  [' + element.index + '] ' + element.role + (element.label ? ' "' + element.label + '"' : ''))
    }
    if (truncated) {
      lines.push('  ... (' + (allElements.length - elements.length) + ' more elements)')
    }
  } else {
    lines.push('')
    lines.push('No interactable elements detected.')
  }
  const textSummary = lines.join('\n')
  if (capture.pngB64 === undefined) return { result: textSummary }
  return {
    _multimodal: true,
    content: [
      { type: 'text', text: textSummary },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,' + capture.pngB64 } },
    ],
    text_summary: textSummary,
  }
}

/** Serialize an action result, preserving a post-action capture when present. */
export function formatActionResult(result: ActionResult, captureAfter = false): ComputerUseResult {
  if (captureAfter && result.capture !== undefined) {
    const capture = formatCaptureResult(result.capture)
    if (isMultimodalCapture(capture)) {
      return {
        ...capture,
        content: [
          {
            type: 'text',
            text: 'Action: ' + result.action + '\n' + (result.message ?? '') + '\n\n' + capture.text_summary,
          },
          capture.content[1],
        ],
      }
    }
  }
  return {
    ok: result.ok,
    action: result.action,
    message: result.message ?? '',
    meta: result.meta ?? {},
  }
}

async function resolveSession(context: ToolExecutionContext, options: ComputerUseToolsOptions): Promise<ComputerUseSession> {
  const resolved = options.resolveSession === undefined ? undefined : await options.resolveSession(context)
  const session = resolved ?? options.session
  if (session === undefined) {
    throw new ComputerUseUnavailableError(
      'computer_use is unavailable because no ComputerUseSession is configured for this session',
    )
  }
  return session
}

function actionFrom(inputs: JsonObject): ComputerUseAction {
  const value = inputs.action
  if (typeof value !== 'string' || !ACTIONS.includes(value as ComputerUseAction)) {
    throw new ValidationError('action', 'must be one of: ' + ACTIONS.join(', '), value)
  }
  return value as ComputerUseAction
}

function assertActionFields(inputs: JsonObject, action: ComputerUseAction): void {
  const allowed = new Set(ACTION_FIELDS[action])
  for (const field of Object.keys(inputs)) {
    if (!allowed.has(field)) {
      throw new ValidationError('computer_use.' + field, 'is not valid for action=' + action, inputs[field])
    }
  }
}

function captureMode(inputs: JsonObject): 'ax' | 'som' | 'vision' {
  const value = inputs.mode ?? DEFAULT_CAPTURE_MODE
  if (value === 'som' || value === 'vision' || value === 'ax') return value
  throw new ValidationError('mode', 'must be som, vision, or ax', value)
}

function captureAfter(inputs: JsonObject): boolean {
  return optionalBoolean(inputs, 'capture_after', false)
}

function clickRequest(inputs: JsonObject): {
  readonly button: 'left'
  readonly captureAfter: boolean
  readonly clickCount: 1
  readonly element?: number
  readonly x?: number
  readonly y?: number
} {
  return {
    button: 'left',
    clickCount: 1,
    captureAfter: captureAfter(inputs),
    ...pointRequest(inputs),
  }
}

function secondaryClickRequest(inputs: JsonObject): {
  readonly captureAfter: boolean
  readonly element?: number
  readonly x?: number
  readonly y?: number
} {
  return {
    captureAfter: captureAfter(inputs),
    ...pointRequest(inputs),
  }
}

function mouseMoveRequest(inputs: JsonObject): { readonly captureAfter: boolean; readonly x: number; readonly y: number } {
  const target = pointRequest(inputs)
  if (target.x === undefined || target.y === undefined) {
    throw new ValidationError('target', 'mouse_move requires x/y coordinates')
  }
  return { captureAfter: captureAfter(inputs), x: target.x, y: target.y }
}

function pointRequest(inputs: JsonObject): { readonly element?: number; readonly x?: number; readonly y?: number } {
  return targetRequest(inputs, '', true)
}

function dragRequest(inputs: JsonObject): {
  readonly captureAfter: boolean
  readonly endElement?: number
  readonly endX?: number
  readonly endY?: number
  readonly startElement?: number
  readonly startX?: number
  readonly startY?: number
} {
  const start = targetRequest(inputs, 'start_', true)
  const end = targetRequest(inputs, 'end_', true)
  return {
    captureAfter: captureAfter(inputs),
    ...(start.element === undefined ? {} : { startElement: start.element }),
    ...(start.x === undefined ? {} : { startX: start.x }),
    ...(start.y === undefined ? {} : { startY: start.y }),
    ...(end.element === undefined ? {} : { endElement: end.element }),
    ...(end.x === undefined ? {} : { endX: end.x }),
    ...(end.y === undefined ? {} : { endY: end.y }),
  }
}

function scrollRequest(inputs: JsonObject): {
  readonly captureAfter: boolean
  readonly dx: number
  readonly dy: number
  readonly element?: number
  readonly x?: number
  readonly y?: number
} {
  return {
    captureAfter: captureAfter(inputs),
    dx: boundedInteger(inputs, 'dx', 0, -MAX_SCROLL_DELTA, MAX_SCROLL_DELTA),
    dy: boundedInteger(inputs, 'dy', 0, -MAX_SCROLL_DELTA, MAX_SCROLL_DELTA),
    ...targetRequest(inputs, '', false),
  }
}

function targetRequest(
  inputs: JsonObject,
  prefix: string,
  required: boolean,
): { readonly element?: number; readonly x?: number; readonly y?: number } {
  const elementName = prefix + 'element'
  const xName = prefix + 'x'
  const yName = prefix + 'y'
  const element = optionalElement(inputs, elementName).element
  const x = optionalCoordinate(inputs, xName)
  const y = optionalCoordinate(inputs, yName)
  if (element !== undefined && (x !== undefined || y !== undefined)) {
    throw new ValidationError(prefix || 'target', 'must specify an element or x/y coordinates, not both')
  }
  if ((x === undefined) !== (y === undefined)) {
    throw new ValidationError(prefix || 'target', 'must provide both x and y coordinates')
  }
  if (required && element === undefined && x === undefined) {
    throw new ValidationError(prefix || 'target', 'requires an element or x/y coordinates')
  }
  return {
    ...(element === undefined ? {} : { element }),
    ...(x === undefined ? {} : { x }),
    ...(y === undefined ? {} : { y }),
  }
}

function optionalElement(inputs: JsonObject, name: string): { readonly element?: number } {
  if (inputs[name] === undefined) return {}
  return { element: boundedInteger(inputs, name, 0, 1, Number.MAX_SAFE_INTEGER) }
}

function optionalCoordinate(inputs: JsonObject, name: string): number | undefined {
  if (inputs[name] === undefined) return undefined
  return boundedInteger(inputs, name, 0, 0, MAX_COORDINATE)
}

function boundedInteger(inputs: JsonObject, name: string, defaultValue: number, minimum: number, maximum: number): number {
  return validateRange(optionalInteger(inputs, name, defaultValue), name, minimum, maximum)
}

function validateRange(value: number, field: string, minimum: number, maximum: number): number {
  if (!Number.isInteger(value) || value < minimum || value > maximum) {
    throw new ValidationError(field, 'must be an integer between ' + minimum + ' and ' + maximum, value)
  }
  return value
}

function requiredStringValue(inputs: JsonObject, name: string): string {
  const value = inputs[name]
  if (typeof value !== 'string') {
    throw new ValidationError(name, 'must be a string', value)
  }
  return value
}

function requiredNonBlankString(inputs: JsonObject, name: string): string {
  const value = requiredStringValue(inputs, name).trim()
  if (!value) {
    throw new ValidationError(name, 'must not be blank', inputs[name])
  }
  return value
}

function optionalNonBlankString(inputs: JsonObject, name: string): string | undefined {
  if (inputs[name] === undefined) return undefined
  return requiredNonBlankString(inputs, name)
}

function isMultimodalCapture(value: ComputerUseResult): value is MultimodalComputerUseResult {
  return typeof value === 'object' && value !== null && '_multimodal' in value && value._multimodal === true
}

export { COMPUTER_USE_DEFINITION }
