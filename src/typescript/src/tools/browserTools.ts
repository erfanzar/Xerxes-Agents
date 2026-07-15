// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ClientError, ValidationError } from '../core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../executors/toolRegistry.js'
import { checkUrl, type UrlSafetyOptions } from '../security/urlSafety.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import { optionalBoolean, optionalInteger, requireRange, requiredString } from './inputs.js'

const DEFAULT_SCROLL_DELTA = 400
const MAX_CONSOLE_MESSAGES = 200
const MAX_IMAGES = 200
const MAX_SNAPSHOT_ELEMENTS = 200
const MAX_SNAPSHOT_TEXT = 4_000
const MAX_VISION_TEXT = 1_000
const MAX_SCROLL_DELTA = 100_000

/** A page element addressable by the browser tool surface. */
export interface BrowserElement {
  readonly href?: string
  readonly name: string
  readonly ref: string
  readonly role: string
  readonly tag: string
}

/** Metadata for one image visible on the current browser page. */
export interface BrowserImage {
  readonly alt: string
  readonly src: string
}

/**
 * Complete current-page state supplied by a concrete browser adapter.
 *
 * The core runtime intentionally does not parse HTML or launch a browser.
 * An adapter may use Playwright, a remote browser provider, or an audited
 * host integration, but must provide this normalized state at the boundary.
 */
export interface BrowserDocument {
  readonly console: readonly string[]
  readonly elements: readonly BrowserElement[]
  readonly images: readonly BrowserImage[]
  readonly scrollY: number
  readonly text: string
  readonly title: string
  readonly url: string
}

/** A browser action may include refreshed page state after a navigation or DOM update. */
export interface BrowserActionResult {
  readonly document?: BrowserDocument
  readonly note?: string
  readonly ok: boolean
  readonly reason?: string
}

/** Scroll actions report the actual scroll position only when the adapter can observe it. */
export interface BrowserScrollResult extends BrowserActionResult {
  readonly scrollY?: number
}

/** Browser screenshot/vision payload. Image data remains base64 for JSON-safe tool output. */
export interface BrowserVisionCapture {
  readonly format: string
  readonly imageB64: string
  readonly summary: string
  readonly text?: string
  readonly url: string
}

export interface BrowserImagesResult {
  readonly images: readonly BrowserImage[]
  readonly url: string
}

export interface BrowserConsoleResult {
  readonly console: readonly string[]
  readonly url: string
}

/**
 * Host-provided browser boundary.
 *
 * No browser package is imported by this module. A port implementation is
 * responsible for browser lifecycle and for validating every resolved redirect
 * destination, including DNS-rebinding defenses, before it is loaded. The
 * session below performs public-URL checks before navigation and validates all
 * URLs returned by the adapter as a second boundary.
 */
export interface BrowserPort {
  back?(): Promise<BrowserActionResult>
  click?(ref: string): Promise<BrowserActionResult>
  consoleLog?(): Promise<BrowserConsoleResult>
  getImages?(): Promise<BrowserImagesResult>
  navigate?(url: string): Promise<BrowserDocument>
  press?(key: string): Promise<BrowserActionResult>
  scroll?(dy: number): Promise<BrowserScrollResult>
  snapshot?(): Promise<BrowserDocument>
  typeText?(ref: string, text: string, options: { readonly submit: boolean }): Promise<BrowserActionResult>
  vision?(): Promise<BrowserVisionCapture>
  close?(): Promise<void>
}

export interface BrowserSessionOptions {
  readonly port?: BrowserPort
  readonly urlSafety?: UrlSafetyOptions
}

/**
 * Session-scoped, serialized browser tool façade around an injected browser port.
 *
 * It deliberately has no HTTP or Playwright fallback: a missing adapter is an
 * explicit configuration error, not a misleading simulated browser result.
 */
export class BrowserSession {
  private document: BrowserDocument | undefined
  private operationTail: Promise<void> = Promise.resolve()
  private port: BrowserPort | undefined
  private readonly urlSafety: UrlSafetyOptions

  constructor(options: BrowserSessionOptions = {}) {
    this.port = options.port
    this.urlSafety = options.urlSafety ?? {}
  }

  setPort(port: BrowserPort | undefined): void {
    this.port = port
    this.document = undefined
  }

  async navigate(url: string): Promise<BrowserDocument> {
    return this.exclusive(async () => {
      const requestedUrl = nonBlank(url, 'url')
      this.assertSafeUrl(requestedUrl)
      const port = this.requirePort()
      const operation = port.navigate
      if (operation === undefined) this.unsupported('navigate')
      return this.recordDocument(await operation.call(port, requestedUrl), true)
    })
  }

  async back(): Promise<BrowserActionResult> {
    return this.exclusive(async () => {
      const port = this.requirePort()
      const operation = port.back
      if (operation === undefined) this.unsupported('back')
      return this.recordAction(await operation.call(port))
    })
  }

  async snapshot(): Promise<BrowserDocument> {
    return this.exclusive(async () => {
      const port = this.requirePort()
      const operation = port.snapshot
      if (operation === undefined) this.unsupported('snapshot')
      return this.recordDocument(await operation.call(port), false)
    })
  }

  async vision(): Promise<BrowserVisionCapture> {
    return this.exclusive(async () => {
      const port = this.requirePort()
      const operation = port.vision
      if (operation === undefined) this.unsupported('vision')
      return normalizeVision(await operation.call(port), this.urlSafety)
    })
  }

  async getImages(): Promise<BrowserImagesResult> {
    return this.exclusive(async () => {
      const port = this.requirePort()
      const operation = port.getImages
      if (operation === undefined) this.unsupported('get_images')
      return normalizeImages(await operation.call(port), this.urlSafety)
    })
  }

  async consoleLog(): Promise<BrowserConsoleResult> {
    return this.exclusive(async () => {
      const port = this.requirePort()
      const operation = port.consoleLog
      if (operation === undefined) this.unsupported('console')
      return normalizeConsole(await operation.call(port), this.urlSafety)
    })
  }

  async click(ref: string): Promise<BrowserActionResult> {
    return this.exclusive(async () => {
      const port = this.requirePort()
      const operation = port.click
      if (operation === undefined) this.unsupported('click')
      return this.recordAction(await operation.call(port, nonBlank(ref, 'ref')))
    })
  }

  async typeText(ref: string, text: string, submit = false): Promise<BrowserActionResult> {
    return this.exclusive(async () => {
      const port = this.requirePort()
      const operation = port.typeText
      if (operation === undefined) this.unsupported('type')
      return this.recordAction(await operation.call(port, nonBlank(ref, 'ref'), nonBlank(text, 'text'), { submit }))
    })
  }

  async press(key: string): Promise<BrowserActionResult> {
    return this.exclusive(async () => {
      const port = this.requirePort()
      const operation = port.press
      if (operation === undefined) this.unsupported('press')
      return this.recordAction(await operation.call(port, nonBlank(key, 'key')))
    })
  }

  async scroll(dy: number): Promise<BrowserScrollResult> {
    return this.exclusive(async () => {
      if (!Number.isInteger(dy) || dy < -MAX_SCROLL_DELTA || dy > MAX_SCROLL_DELTA) {
        throw new ValidationError('dy', 'must be an integer between -100000 and 100000', dy)
      }
      const port = this.requirePort()
      const operation = port.scroll
      if (operation === undefined) this.unsupported('scroll')
      const action = await operation.call(port, dy)
      const recorded = this.recordAction(action)
      const reportedScroll = action.scrollY ?? recorded.document?.scrollY
      return Object.freeze({
        ...recorded,
        ...(reportedScroll === undefined ? {} : { scrollY: normalizeScrollY(reportedScroll) }),
      })
    })
  }

  async close(): Promise<void> {
    await this.exclusive(async () => {
      try {
        await this.port?.close?.()
      } finally {
        this.document = undefined
      }
    })
  }

  private recordDocument(source: BrowserDocument, requireUrl = false): BrowserDocument {
    const document = normalizeDocument(source, this.urlSafety, requireUrl)
    this.document = document
    return document
  }

  private recordAction(source: BrowserActionResult): BrowserActionResult {
    if (typeof source?.ok !== 'boolean') {
      throw new ClientError('browser', 'browser port returned an invalid action result')
    }
    const document = source.document === undefined ? undefined : this.recordDocument(source.document)
    const reason = source.reason === undefined ? undefined : requiredPortString(source.reason, 'action reason')
    const note = source.note === undefined ? undefined : requiredPortString(source.note, 'action note')
    return Object.freeze({
      ok: source.ok,
      ...(reason === undefined ? {} : { reason }),
      ...(note === undefined ? {} : { note }),
      ...(document === undefined ? {} : { document }),
    })
  }

  private requirePort(): BrowserPort {
    if (this.port === undefined) {
      throw new ClientError('browser', 'no browser port is configured; inject a BrowserPort backed by a real browser service')
    }
    return this.port
  }

  private unsupported(operation: string): never {
    throw new ClientError('browser', 'configured browser port does not support ' + operation)
  }

  private assertSafeUrl(url: string): void {
    const decision = checkUrl(url, this.urlSafety)
    if (!decision.allowed) {
      throw new ValidationError('url', decision.reason, url)
    }
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
    }
  }
}

export interface BrowserToolsOptions {
  /** Static session for a single-session host. */
  readonly session?: BrowserSession
  /** Resolve the browser session dynamically for per-agent or per-session browser ownership. */
  readonly resolveSession?: (context: ToolExecutionContext) => BrowserSession | undefined | Promise<BrowserSession | undefined>
}

export const BROWSER_NAVIGATE_DEFINITION: ToolDefinition = definition(
  'browser_navigate',
  'Navigate the configured browser to a public HTTP(S) URL.',
  {
    url: { type: 'string', description: 'Public HTTP(S) URL to open.' },
  },
  ['url'],
)

export const BROWSER_BACK_DEFINITION: ToolDefinition = definition(
  'browser_back',
  'Navigate back one entry in the configured browser history.',
)

export const BROWSER_SNAPSHOT_DEFINITION: ToolDefinition = definition(
  'browser_snapshot',
  'Return the current browser page text and interactive element references.',
)

export const BROWSER_VISION_DEFINITION: ToolDefinition = definition(
  'browser_vision',
  'Capture a browser screenshot through the configured browser port.',
)

export const BROWSER_GET_IMAGES_DEFINITION: ToolDefinition = definition(
  'browser_get_images',
  'List image metadata from the current browser page.',
)

export const BROWSER_CONSOLE_DEFINITION: ToolDefinition = definition(
  'browser_console',
  'Return console messages captured by the configured browser port.',
)

export const BROWSER_CLICK_DEFINITION: ToolDefinition = definition(
  'browser_click',
  'Click an interactive element by its reference from browser_snapshot.',
  {
    ref: { type: 'string', description: 'Element reference from browser_snapshot.' },
  },
  ['ref'],
)

export const BROWSER_TYPE_DEFINITION: ToolDefinition = definition(
  'browser_type',
  'Type non-empty text into an interactive element by reference.',
  {
    ref: { type: 'string', description: 'Input or textarea reference from browser_snapshot.' },
    text: { type: 'string', description: 'Non-empty text to type.' },
    submit: { type: 'boolean', default: false, description: 'Press Enter after filling the field.' },
  },
  ['ref', 'text'],
)

export const BROWSER_PRESS_DEFINITION: ToolDefinition = definition(
  'browser_press',
  'Press one keyboard key in the configured browser.',
  {
    key: { type: 'string', description: 'Keyboard key, such as Enter, Escape, or ArrowDown.' },
  },
  ['key'],
)

export const BROWSER_SCROLL_DEFINITION: ToolDefinition = definition(
  'browser_scroll',
  'Scroll the configured browser viewport vertically.',
  {
    dy: {
      type: 'integer',
      default: DEFAULT_SCROLL_DELTA,
      minimum: -MAX_SCROLL_DELTA,
      maximum: MAX_SCROLL_DELTA,
      description: 'Vertical pixel delta. Positive scrolls down.',
    },
  },
)

export const BROWSER_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  BROWSER_NAVIGATE_DEFINITION,
  BROWSER_BACK_DEFINITION,
  BROWSER_SNAPSHOT_DEFINITION,
  BROWSER_VISION_DEFINITION,
  BROWSER_GET_IMAGES_DEFINITION,
  BROWSER_CONSOLE_DEFINITION,
  BROWSER_CLICK_DEFINITION,
  BROWSER_TYPE_DEFINITION,
  BROWSER_PRESS_DEFINITION,
  BROWSER_SCROLL_DEFINITION,
]

/** Register the browser tool family against an explicitly supplied session resolver. */
export function registerBrowserTools(
  registry: ToolRegistry,
  options: BrowserToolsOptions,
  agentId = 'default',
): void {
  registry.register(BROWSER_NAVIGATE_DEFINITION, async (inputs, context) => browserNavigate(inputs, context, options), agentId)
  registry.register(BROWSER_BACK_DEFINITION, async (inputs, context) => browserBack(inputs, context, options), agentId)
  registry.register(BROWSER_SNAPSHOT_DEFINITION, async (inputs, context) => browserSnapshot(inputs, context, options), agentId)
  registry.register(BROWSER_VISION_DEFINITION, async (inputs, context) => browserVision(inputs, context, options), agentId)
  registry.register(BROWSER_GET_IMAGES_DEFINITION, async (inputs, context) => browserGetImages(inputs, context, options), agentId)
  registry.register(BROWSER_CONSOLE_DEFINITION, async (inputs, context) => browserConsole(inputs, context, options), agentId)
  registry.register(BROWSER_CLICK_DEFINITION, async (inputs, context) => browserClick(inputs, context, options), agentId)
  registry.register(BROWSER_TYPE_DEFINITION, async (inputs, context) => browserType(inputs, context, options), agentId)
  registry.register(BROWSER_PRESS_DEFINITION, async (inputs, context) => browserPress(inputs, context, options), agentId)
  registry.register(BROWSER_SCROLL_DEFINITION, async (inputs, context) => browserScroll(inputs, context, options), agentId)
}

export async function browserNavigate(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: BrowserToolsOptions,
): Promise<JsonObject> {
  const document = await resolveSession(context, options).then(session => session.navigate(nonBlank(requiredString(inputs, 'url'), 'url')))
  return navigationWire(document)
}

export async function browserBack(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: BrowserToolsOptions,
): Promise<JsonObject> {
  const action = await resolveSession(context, options).then(session => session.back())
  if (action.ok && action.document !== undefined) {
    return { ...navigationWire(action.document), ...actionWire(action) }
  }
  return actionWire(action)
}

export async function browserSnapshot(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: BrowserToolsOptions,
): Promise<JsonObject> {
  return snapshotWire(await resolveSession(context, options).then(session => session.snapshot()))
}

export async function browserVision(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: BrowserToolsOptions,
): Promise<JsonObject> {
  const vision = await resolveSession(context, options).then(session => session.vision())
  return {
    url: vision.url,
    image_b64: vision.imageB64,
    format: vision.format,
    summary: vision.summary,
    ...(vision.text === undefined ? {} : { text: vision.text.slice(0, MAX_VISION_TEXT) }),
  }
}

export async function browserGetImages(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: BrowserToolsOptions,
): Promise<JsonObject> {
  const result = await resolveSession(context, options).then(session => session.getImages())
  return {
    url: result.url,
    images: result.images.slice(0, MAX_IMAGES).map(image => ({ src: image.src, alt: image.alt })),
  }
}

export async function browserConsole(
  _inputs: JsonObject,
  context: ToolExecutionContext,
  options: BrowserToolsOptions,
): Promise<JsonObject> {
  const result = await resolveSession(context, options).then(session => session.consoleLog())
  return { url: result.url, console: result.console.slice(-MAX_CONSOLE_MESSAGES) }
}

export async function browserClick(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: BrowserToolsOptions,
): Promise<JsonObject> {
  const ref = nonBlank(requiredString(inputs, 'ref'), 'ref')
  return actionWire(await resolveSession(context, options).then(session => session.click(ref)), { ref })
}

export async function browserType(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: BrowserToolsOptions,
): Promise<JsonObject> {
  const ref = nonBlank(requiredString(inputs, 'ref'), 'ref')
  const text = nonBlank(requiredString(inputs, 'text'), 'text')
  const submit = optionalBoolean(inputs, 'submit', false)
  const action = await resolveSession(context, options).then(session => session.typeText(ref, text, submit))
  return actionWire(action, action.ok ? { ref, submitted: submit } : {})
}

export async function browserPress(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: BrowserToolsOptions,
): Promise<JsonObject> {
  const key = nonBlank(requiredString(inputs, 'key'), 'key')
  const action = await resolveSession(context, options).then(session => session.press(key))
  return actionWire(action, action.ok ? { key } : {})
}

export async function browserScroll(
  inputs: JsonObject,
  context: ToolExecutionContext,
  options: BrowserToolsOptions,
): Promise<JsonObject> {
  const dy = requireRange(optionalInteger(inputs, 'dy', DEFAULT_SCROLL_DELTA), 'dy', -MAX_SCROLL_DELTA, MAX_SCROLL_DELTA)
  const action = await resolveSession(context, options).then(session => session.scroll(dy))
  const wire = actionWire(action)
  if (!action.ok) return wire
  return {
    ...wire,
    scroll_y: action.scrollY ?? null,
  }
}

async function resolveSession(context: ToolExecutionContext, options: BrowserToolsOptions): Promise<BrowserSession> {
  const resolved = options.resolveSession === undefined ? undefined : await options.resolveSession(context)
  const session = resolved ?? options.session
  if (session === undefined) {
    throw new ClientError('browser', 'browser tools are unavailable because no BrowserSession is configured for this session')
  }
  return session
}

function definition(name: string, description: string, properties: JsonObject = {}, required: string[] = []): ToolDefinition {
  return {
    type: 'function',
    function: {
      name,
      description,
      parameters: {
        type: 'object',
        additionalProperties: false,
        properties,
        ...(required.length === 0 ? {} : { required }),
      },
    },
  }
}

function navigationWire(document: BrowserDocument): JsonObject {
  return {
    url: document.url,
    title: document.title,
    elements: document.elements.length,
  }
}

function snapshotWire(document: BrowserDocument): JsonObject {
  return {
    url: document.url,
    title: document.title,
    text: document.text.slice(0, MAX_SNAPSHOT_TEXT),
    elements: document.elements.slice(0, MAX_SNAPSHOT_ELEMENTS).map(element => ({
      ref: element.ref,
      tag: element.tag,
      role: element.role,
      name: element.name,
      href: element.href ?? '',
    })),
    scroll_y: document.scrollY,
  }
}

function actionWire(
  action: BrowserActionResult,
  successFields: Record<string, string | boolean> = {},
): JsonObject {
  if (!action.ok) {
    return {
      ok: false,
      reason: action.reason ?? 'browser adapter reported failure without a reason',
      ...(action.note === undefined ? {} : { note: action.note }),
    }
  }
  return {
    ok: true,
    ...successFields,
    ...(action.note === undefined ? {} : { note: action.note }),
  }
}

function normalizeDocument(source: BrowserDocument, urlSafety: UrlSafetyOptions, requireUrl: boolean): BrowserDocument {
  if (source === null || typeof source !== 'object') {
    throw new ClientError('browser', 'browser port returned an invalid page document')
  }
  const url = normalizePortUrl(source.url, 'document URL', urlSafety, requireUrl)
  const title = requiredPortString(source.title, 'document title')
  const text = requiredPortString(source.text, 'document text')
  const scrollY = normalizeScrollY(source.scrollY)
  if (!Array.isArray(source.elements) || !Array.isArray(source.images) || !Array.isArray(source.console)) {
    throw new ClientError('browser', 'browser port document must include elements, images, and console arrays')
  }
  return Object.freeze({
    url,
    title,
    text,
    scrollY,
    elements: Object.freeze(source.elements.map(normalizeElement)),
    images: Object.freeze(source.images.map(normalizeImage)),
    console: Object.freeze(source.console.map((message, index) => requiredPortString(message, 'console message ' + index))),
  })
}

function normalizeVision(source: BrowserVisionCapture, urlSafety: UrlSafetyOptions): BrowserVisionCapture {
  if (source === null || typeof source !== 'object') {
    throw new ClientError('browser', 'browser port returned an invalid vision result')
  }
  const url = normalizePortUrl(source.url, 'vision URL', urlSafety, false)
  const text = source.text === undefined ? undefined : requiredPortString(source.text, 'vision text')
  return Object.freeze({
    url,
    imageB64: requiredPortString(source.imageB64, 'vision image data'),
    format: nonBlankPortString(source.format, 'vision format'),
    summary: requiredPortString(source.summary, 'vision summary'),
    ...(text === undefined ? {} : { text }),
  })
}

function normalizeImages(source: BrowserImagesResult, urlSafety: UrlSafetyOptions): BrowserImagesResult {
  if (source === null || typeof source !== 'object' || !Array.isArray(source.images)) {
    throw new ClientError('browser', 'browser port returned an invalid image result')
  }
  const url = normalizePortUrl(source.url, 'image result URL', urlSafety, false)
  return Object.freeze({ url, images: Object.freeze(source.images.map(normalizeImage)) })
}

function normalizeConsole(source: BrowserConsoleResult, urlSafety: UrlSafetyOptions): BrowserConsoleResult {
  if (source === null || typeof source !== 'object' || !Array.isArray(source.console)) {
    throw new ClientError('browser', 'browser port returned an invalid console result')
  }
  const url = normalizePortUrl(source.url, 'console result URL', urlSafety, false)
  return Object.freeze({
    url,
    console: Object.freeze(source.console.map((message, index) => requiredPortString(message, 'console message ' + index))),
  })
}

function normalizeElement(source: BrowserElement): BrowserElement {
  if (source === null || typeof source !== 'object') {
    throw new ClientError('browser', 'browser port returned an invalid page element')
  }
  const href = source.href === undefined ? undefined : requiredPortString(source.href, 'element href')
  return Object.freeze({
    ref: nonBlankPortString(source.ref, 'element ref'),
    tag: requiredPortString(source.tag, 'element tag'),
    role: requiredPortString(source.role, 'element role'),
    name: requiredPortString(source.name, 'element name'),
    ...(href === undefined ? {} : { href }),
  })
}

function normalizeImage(source: BrowserImage): BrowserImage {
  if (source === null || typeof source !== 'object') {
    throw new ClientError('browser', 'browser port returned an invalid image')
  }
  return Object.freeze({
    src: requiredPortString(source.src, 'image src'),
    alt: requiredPortString(source.alt, 'image alt'),
  })
}

function normalizeScrollY(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new ClientError('browser', 'browser port returned an invalid scroll position')
  }
  return value
}

function assertSafePortUrl(url: string, urlSafety: UrlSafetyOptions): void {
  const decision = checkUrl(url, urlSafety)
  if (!decision.allowed) {
    throw new ClientError('browser', 'browser port returned a disallowed URL: ' + decision.reason)
  }
}

function normalizePortUrl(value: unknown, label: string, urlSafety: UrlSafetyOptions, requireUrl: boolean): string {
  const url = requiredPortString(value, label).trim()
  if (!url) {
    if (requireUrl) {
      throw new ClientError('browser', 'browser port returned an empty ' + label)
    }
    return ''
  }
  assertSafePortUrl(url, urlSafety)
  return url
}

function requiredPortString(value: unknown, label: string): string {
  if (typeof value !== 'string') {
    throw new ClientError('browser', 'browser port returned an invalid ' + label)
  }
  return value
}

function nonBlankPortString(value: unknown, label: string): string {
  const string = requiredPortString(value, label).trim()
  if (!string) {
    throw new ClientError('browser', 'browser port returned an empty ' + label)
  }
  return string
}

function nonBlank(value: string, field: string): string {
  const trimmed = value.trim()
  if (!trimmed) {
    throw new ValidationError(field, 'must not be blank', value)
  }
  return trimmed
}
