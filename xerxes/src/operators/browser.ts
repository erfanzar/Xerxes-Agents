// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ClientError, ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import { checkUrl, type UrlSafetyOptions } from '../security/urlSafety.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import type { CdpBrowserAdapterOptions } from './cdpBrowser.js'

export interface BrowserLink {
  readonly id?: number
  readonly url: string
}

export interface BrowserPageState {
  readonly linkMap: ReadonlyMap<number, string>
  readonly refId: string
  readonly title: string
  readonly url: string
}

export interface BrowserPageObservation {
  readonly contentPreview?: string
  readonly links?: readonly BrowserLink[]
  readonly refId?: string
  readonly title?: string
  readonly url: string
}

export interface BrowserOpenRequest {
  readonly refId?: string
  readonly url?: string
  readonly waitMs: number
}

export interface BrowserClickRequest {
  readonly refId: string
  readonly selector?: string
  readonly text?: string
  readonly waitMs: number
}

export interface BrowserFindResult {
  readonly matchCount: number
  readonly matches: readonly string[]
  readonly pattern: string
  readonly refId: string
}

export interface BrowserScreenshotResult {
  readonly fullPage: boolean
  readonly path: string
  readonly refId: string
}

/** JSON-safe connection information exposed to a daemon or terminal UI. */
export interface BrowserConnectionStatus {
  readonly connected: boolean
  /** A credential-redacted endpoint when the concrete adapter can expose one safely. */
  readonly endpoint?: string
  /** Concrete adapter family, such as `cdp` or `injected`. */
  readonly kind: string
}

/**
 * Browser-driver port. A Playwright, Browserbase, or remote browser adapter can
 * implement this without leaking its package into the Bun runtime core.
 *
 * Implementations must pin/re-check the resolved host before navigation to
 * defend against DNS rebinding; `BrowserManager` only performs syntax and
 * obvious private-network screening.
 */
export interface BrowserAdapter {
  click(request: BrowserClickRequest): Promise<BrowserPageObservation>
  find(refId: string, pattern: string): Promise<BrowserFindResult>
  open(request: BrowserOpenRequest): Promise<BrowserPageObservation>
  screenshot(refId: string, options: { readonly fullPage: boolean; readonly path?: string }): Promise<BrowserScreenshotResult>
  close?(): Promise<void>
  /** Optional JSON-safe status; adapters must not return credentials here. */
  connectionStatus?(): BrowserConnectionStatus
}

export interface BrowserManagerOptions {
  readonly adapter?: BrowserAdapter
  readonly urlSafety?: UrlSafetyOptions
}

/** Lazy browser session manager backed by an optional typed driver adapter. */
export class BrowserManager {
  private adapter: BrowserAdapter | undefined
  private readonly pages = new Map<string, BrowserPageState>()
  private readonly urlSafety: UrlSafetyOptions

  constructor(options: BrowserManagerOptions = {}) {
    this.adapter = options.adapter
    this.urlSafety = options.urlSafety ?? {}
  }

  setAdapter(adapter: BrowserAdapter | undefined): void {
    this.adapter = adapter
    this.pages.clear()
  }

  /** Attach a real Chromium DevTools Protocol endpoint without a Python or Playwright subprocess. */
  async connectCdp(endpoint: string, options: CdpBrowserAdapterOptions = {}): Promise<BrowserConnectionStatus> {
    const { CdpBrowserAdapter } = await import('./cdpBrowser.js')
    const adapter = await CdpBrowserAdapter.connect(endpoint, options)
    try {
      await this.disconnect()
    } catch (error) {
      await adapter.close()
      throw error
    }
    this.adapter = adapter
    return this.connectionStatus()
  }

  /** Detach from the configured browser without trying to terminate the external Chromium process. */
  async disconnect(): Promise<void> {
    const adapter = this.adapter
    this.adapter = undefined
    this.pages.clear()
    await adapter?.close?.()
  }

  /** Report only safe connection metadata; a configured injected adapter has no endpoint by default. */
  connectionStatus(): BrowserConnectionStatus {
    const adapter = this.adapter
    if (adapter === undefined) return Object.freeze({ connected: false, kind: 'none' })
    return adapter.connectionStatus?.() ?? Object.freeze({ connected: true, kind: 'injected' })
  }

  async open(options: { readonly refId?: string; readonly url?: string; readonly waitMs?: number }): Promise<BrowserPageObservation & {
    readonly links: readonly BrowserLink[]
    readonly refId: string
    readonly title: string
  }> {
    const waitMs = normalizeWait(options.waitMs)
    const refId = options.refId?.trim()
    const url = options.url?.trim()
    if (!refId && !url) throw new ValidationError('browser.open', 'requires url or ref_id')
    if (url) this.assertSafeUrl(url)
    if (refId && !this.pages.has(refId)) throw new ValidationError('ref_id', 'browser page not found', refId)
    const observation = await this.requireAdapter().open({
      ...(refId === undefined ? {} : { refId }),
      ...(url === undefined ? {} : { url }),
      waitMs,
    })
    const assignedRefId = observation.refId?.trim() || refId || this.nextRefId()
    return this.record(assignedRefId, observation)
  }

  async click(
    refId: string,
    options: { readonly linkId?: number; readonly selector?: string; readonly text?: string; readonly waitMs?: number },
  ): Promise<BrowserPageObservation & { readonly links: readonly BrowserLink[]; readonly refId: string; readonly title: string }> {
    const state = this.requirePage(refId)
    const targets = Number(options.linkId !== undefined) + Number(Boolean(options.selector?.trim())) + Number(Boolean(options.text?.trim()))
    if (targets !== 1) throw new ValidationError('browser.click', 'requires exactly one of link_id, selector, or text')
    if (options.linkId !== undefined) {
      const url = state.linkMap.get(options.linkId)
      if (url === undefined) throw new ValidationError('link_id', 'was not found for browser page', options.linkId)
      return this.open({
        refId,
        url,
        ...(options.waitMs === undefined ? {} : { waitMs: options.waitMs }),
      })
    }
    const observation = await this.requireAdapter().click({
      refId,
      ...(options.selector?.trim() ? { selector: options.selector.trim() } : {}),
      ...(options.text?.trim() ? { text: options.text.trim() } : {}),
      waitMs: normalizeWait(options.waitMs),
    })
    return this.record(refId, observation)
  }

  async find(refId: string, pattern: string): Promise<BrowserFindResult> {
    this.requirePage(refId)
    if (!pattern) throw new ValidationError('pattern', 'must not be empty', pattern)
    const result = await this.requireAdapter().find(refId, pattern)
    return Object.freeze({
      refId,
      pattern,
      matchCount: result.matchCount,
      matches: Object.freeze([...result.matches].slice(0, 20)),
    })
  }

  async screenshot(
    refId: string,
    options: { readonly fullPage?: boolean; readonly path?: string } = {},
  ): Promise<BrowserScreenshotResult> {
    this.requirePage(refId)
    return this.requireAdapter().screenshot(refId, {
      fullPage: options.fullPage ?? true,
      ...(options.path?.trim() ? { path: options.path.trim() } : {}),
    })
  }

  listPages(): Array<{ readonly refId: string; readonly title: string; readonly url: string }> {
    return [...this.pages.values()]
      .sort((left, right) => left.refId.localeCompare(right.refId))
      .map(state => Object.freeze({ refId: state.refId, url: state.url, title: state.title }))
  }

  async close(): Promise<void> {
    await this.disconnect()
  }

  private record(
    refId: string,
    observation: BrowserPageObservation,
  ): BrowserPageObservation & { readonly links: readonly BrowserLink[]; readonly refId: string; readonly title: string } {
    this.assertSafeUrl(observation.url)
    const links = normalizeLinks(observation.links)
    const linkMap = new Map(links.map(link => [link.id ?? 0, link.url]))
    const state: BrowserPageState = Object.freeze({
      refId,
      url: observation.url,
      title: observation.title ?? '',
      linkMap,
    })
    this.pages.set(refId, state)
    return Object.freeze({
      refId,
      url: state.url,
      title: state.title,
      ...(observation.contentPreview === undefined ? {} : { contentPreview: observation.contentPreview.slice(0, 2_000) }),
      links: Object.freeze(links),
    })
  }

  private assertSafeUrl(url: string): void {
    const decision = checkUrl(url, this.urlSafety)
    if (!decision.allowed) throw new ValidationError('url', decision.reason, url)
  }

  private requireAdapter(): BrowserAdapter {
    if (this.adapter === undefined) {
      throw new ClientError('browser', 'no browser adapter is configured; install or inject a Playwright-compatible adapter')
    }
    return this.adapter
  }

  private requirePage(refId: string): BrowserPageState {
    const page = this.pages.get(refId)
    if (page === undefined) throw new ValidationError('ref_id', 'browser page not found', refId)
    return page
  }

  private nextRefId(): string {
    return `page_${crypto.randomUUID().replaceAll('-', '').slice(0, 10)}`
  }
}

/**
 * Register the real `web.*` browser tools around one shared manager.
 *
 * Embedding runtimes call this only when they own a manager that can be
 * connected to an explicit browser adapter. Before `/browser connect`, calls
 * fail with the manager's actionable configuration error rather than returning
 * fabricated page data.
 */
export function registerBrowserManagerTools(
  registry: ToolRegistry,
  browserManager: BrowserManager,
  agentId = 'default',
): readonly ToolDefinition[] {
  const tools: readonly { readonly definition: ToolDefinition; readonly handler: (inputs: JsonObject) => Promise<unknown> }[] = [
    {
      definition: browserToolDefinition('web.open', 'Open or revisit a page through the connected Chromium CDP browser.', {
        url: stringSchema('Public http(s) URL to navigate to.'),
        ref_id: stringSchema('Existing browser page reference.'),
        wait_ms: integerSchema('Milliseconds to wait after navigation.'),
      }),
      handler: async inputs => {
        const url = optionalString(inputs, 'url')
        const refId = optionalString(inputs, 'ref_id')
        const waitMs = optionalInteger(inputs, 'wait_ms')
        return browserOpenWire(await browserManager.open({
          ...(url === undefined ? {} : { url }),
          ...(refId === undefined ? {} : { refId }),
          ...(waitMs === undefined ? {} : { waitMs }),
        }))
      },
    },
    {
      definition: browserToolDefinition('web.click', 'Click a discovered link, CSS selector, or visible-text target.', {
        ref_id: stringSchema('Browser page reference.'),
        link_id: integerSchema('Numeric link id returned by web.open.'),
        selector: stringSchema('CSS selector target.'),
        text: stringSchema('Visible text target.'),
        wait_ms: integerSchema('Milliseconds to wait after interaction.'),
      }, ['ref_id']),
      handler: async inputs => {
        const linkId = optionalInteger(inputs, 'link_id')
        const selector = optionalString(inputs, 'selector')
        const text = optionalString(inputs, 'text')
        const waitMs = optionalInteger(inputs, 'wait_ms')
        return browserOpenWire(await browserManager.click(requiredString(inputs, 'ref_id'), {
          ...(linkId === undefined ? {} : { linkId }),
          ...(selector === undefined ? {} : { selector }),
          ...(text === undefined ? {} : { text }),
          ...(waitMs === undefined ? {} : { waitMs }),
        }))
      },
    },
    {
      definition: browserToolDefinition('web.find', 'Find case-insensitive regular-expression matches in a page.', {
        ref_id: stringSchema('Browser page reference.'),
        pattern: stringSchema('Regular expression to find.'),
      }, ['ref_id', 'pattern']),
      handler: async inputs => browserFindWire(await browserManager.find(
        requiredString(inputs, 'ref_id'),
        requiredString(inputs, 'pattern'),
      )),
    },
    {
      definition: browserToolDefinition('web.screenshot', 'Capture a PNG under the configured browser screenshot directory.', {
        ref_id: stringSchema('Browser page reference.'),
        path: stringSchema('Optional path beneath the configured screenshot directory.'),
        full_page: booleanSchema('Capture the entire page rather than the viewport.'),
      }, ['ref_id']),
      handler: async inputs => {
        const path = optionalString(inputs, 'path')
        const fullPage = optionalBoolean(inputs, 'full_page')
        return browserScreenshotWire(await browserManager.screenshot(requiredString(inputs, 'ref_id'), {
          ...(path === undefined ? {} : { path }),
          ...(fullPage === undefined ? {} : { fullPage }),
        }))
      },
    },
  ]
  for (const tool of tools) registry.replace(tool.definition, tool.handler, agentId)
  return tools.map(tool => tool.definition)
}

function normalizeWait(value: number | undefined): number {
  const wait = value ?? 500
  if (!Number.isInteger(wait) || wait < 0) throw new ValidationError('wait_ms', 'must be a non-negative integer', wait)
  return wait
}

function normalizeLinks(links: readonly BrowserLink[] | undefined): BrowserLink[] {
  const normalized: BrowserLink[] = []
  const usedIds = new Set<number>()
  for (const source of links ?? []) {
    const url = source.url.trim()
    if (!url) continue
    let id = source.id
    if (id === undefined || !Number.isInteger(id) || id < 0 || usedIds.has(id)) {
      id = nextLinkId(usedIds)
    }
    usedIds.add(id)
    normalized.push(Object.freeze({ id, url }))
  }
  return normalized
}

function nextLinkId(usedIds: ReadonlySet<number>): number {
  let id = 0
  while (usedIds.has(id)) id += 1
  return id
}

function browserToolDefinition(
  name: string,
  description: string,
  properties: Record<string, unknown>,
  required: readonly string[] = [],
): ToolDefinition {
  return Object.freeze({
    type: 'function',
    function: Object.freeze({
      name,
      description,
      parameters: Object.freeze({
        type: 'object',
        additionalProperties: false,
        properties,
        ...(required.length ? { required } : {}),
      }),
    }),
  })
}

function stringSchema(description: string): Record<string, unknown> {
  return { type: 'string', description }
}

function integerSchema(description: string): Record<string, unknown> {
  return { type: 'integer', description }
}

function booleanSchema(description: string): Record<string, unknown> {
  return { type: 'boolean', description }
}

function requiredString(inputs: JsonObject, field: string): string {
  const value = inputs[field]
  if (typeof value !== 'string' || !value.trim()) throw new ValidationError(field, 'must be a non-empty string', value)
  return value.trim()
}

function optionalString(inputs: JsonObject, field: string): string | undefined {
  const value = inputs[field]
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'string') throw new ValidationError(field, 'must be a string', value)
  const normalized = value.trim()
  return normalized || undefined
}

function optionalInteger(inputs: JsonObject, field: string): number | undefined {
  const value = inputs[field]
  if (value === undefined || value === null) return undefined
  if (!Number.isInteger(value)) throw new ValidationError(field, 'must be an integer', value)
  return Number(value)
}

function optionalBoolean(inputs: JsonObject, field: string): boolean | undefined {
  const value = inputs[field]
  if (value === undefined || value === null) return undefined
  if (typeof value !== 'boolean') throw new ValidationError(field, 'must be a boolean', value)
  return value
}

function browserOpenWire(value: BrowserPageObservation & {
  readonly links: readonly BrowserLink[]
  readonly refId: string
  readonly title: string
}): Record<string, unknown> {
  return Object.freeze({
    ref_id: value.refId,
    url: value.url,
    title: value.title,
    content_preview: value.contentPreview ?? '',
    links: value.links.map(link => ({ id: link.id, url: link.url })),
  })
}

function browserFindWire(value: BrowserFindResult): Record<string, unknown> {
  return Object.freeze({
    ref_id: value.refId,
    pattern: value.pattern,
    match_count: value.matchCount,
    matches: [...value.matches],
  })
}

function browserScreenshotWire(value: BrowserScreenshotResult): Record<string, unknown> {
  return Object.freeze({ ref_id: value.refId, path: value.path, full_page: value.fullPage })
}
