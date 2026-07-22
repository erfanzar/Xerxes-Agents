// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat, mkdir } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path'

import { ClientError, ValidationError } from '../core/errors.js'
import type {
  BrowserAdapter,
  BrowserClickRequest,
  BrowserConnectionStatus,
  BrowserFindResult,
  BrowserOpenRequest,
  BrowserPageObservation,
  BrowserScreenshotResult,
} from './browser.js'

const CDP_COMMAND_TIMEOUT = 15_000
const CDP_CONNECT_TIMEOUT = 10_000
const DEFAULT_SCREENSHOT_DIRECTORY = join(tmpdir(), 'xerxes-browser')
/** Cap on collected page links so a hostile DOM cannot flood the transcript. */
const MAX_INSPECTION_LINKS = 200
/** Cap on accepted base64 screenshot payloads (~37.5MB of PNG) before decoding. */
const MAX_SCREENSHOT_BASE64_CHARS = 50 * 1024 * 1024
/** Bound the model-supplied find pattern before it runs on the page main thread. */
const MAX_FIND_PATTERN_CHARS = 256
/** Cap the in-page subject and iteration budget for the bounded find exec loop. */
const MAX_FIND_SUBJECT_CHARS = 1_000_000
const MAX_FIND_ITERATIONS = 10_000

interface CdpPage {
  readonly refId: string
  readonly sessionId: string
  readonly targetId: string
}

interface CdpResponseError {
  readonly code?: unknown
  readonly data?: unknown
  readonly message?: unknown
}

interface CdpResponseFrame {
  readonly error?: CdpResponseError
  readonly id?: unknown
  readonly result?: unknown
}

/** Minimal transport boundary for a Chromium DevTools Protocol connection. */
export interface CdpConnection {
  close(): void | Promise<void>
  command(
    method: string,
    params?: Readonly<Record<string, unknown>>,
    sessionId?: string,
  ): Promise<unknown>
}

/** Injectable connection factory used by the adapter and deterministic tests. */
export interface CdpConnectionFactory {
  connect(endpoint: string): Promise<CdpConnection>
}

/** Explicit HTTP boundary used only to discover a browser-level CDP WebSocket URL. */
export type CdpFetch = (url: string, init?: RequestInit) => Promise<Response>

export interface CdpBrowserAdapterOptions {
  /**
   * Explicit additional hostnames the discovered webSocketDebuggerUrl may point
   * at. Without an entry here the resolved WebSocket must stay on the host and
   * port of the supplied HTTP endpoint, which blocks SSRF relay through a
   * compromised debugging endpoint.
   */
  readonly allowedWebSocketHosts?: readonly string[]
  readonly connectionFactory?: CdpConnectionFactory
  readonly fetchImplementation?: CdpFetch
  /** Output directory for screenshots. Explicit paths must remain beneath it. */
  readonly screenshotDirectory?: string
  /** Injectable clock for deterministic tests. */
  readonly sleep?: (milliseconds: number) => Promise<void>
}

/**
 * Bun-native Chromium DevTools Protocol adapter for the operator `web.*` surface.
 *
 * Xerxes never launches, installs, or owns Chromium here. Callers explicitly
 * connect it to a browser already running with a remote-debugging endpoint.
 */
export class CdpBrowserAdapter implements BrowserAdapter {
  readonly #connection: CdpConnection
  readonly #displayEndpoint: string
  readonly #pages = new Map<string, CdpPage>()
  readonly #screenshotDirectory: string
  readonly #sleep: (milliseconds: number) => Promise<void>
  #closed = false

  private constructor(
    connection: CdpConnection,
    displayEndpoint: string,
    options: CdpBrowserAdapterOptions,
  ) {
    this.#connection = connection
    this.#displayEndpoint = displayEndpoint
    this.#screenshotDirectory = resolve(options.screenshotDirectory ?? DEFAULT_SCREENSHOT_DIRECTORY)
    this.#sleep = options.sleep ?? (milliseconds => Bun.sleep(milliseconds))
  }

  /** Connect and verify an already-running Chromium-family browser. */
  static async connect(endpoint: string, options: CdpBrowserAdapterOptions = {}): Promise<CdpBrowserAdapter> {
    const requested = cdpEndpointUrl(endpoint)
    const webSocketEndpoint = await resolveCdpWebSocketUrl(
      requested,
      options.fetchImplementation ?? defaultFetch,
      { allowedHosts: options.allowedWebSocketHosts ?? [] },
    )
    const connection = await (options.connectionFactory ?? new BunCdpConnectionFactory()).connect(webSocketEndpoint)
    try {
      await connection.command('Browser.getVersion')
    } catch (error) {
      await connection.close()
      throw new ClientError('cdp', 'endpoint did not accept Browser.getVersion', error)
    }
    return new CdpBrowserAdapter(connection, redactedCdpEndpoint(requested.toString()), options)
  }

  connectionStatus(): BrowserConnectionStatus {
    return Object.freeze({
      connected: !this.#closed,
      endpoint: this.#displayEndpoint,
      kind: 'cdp',
    })
  }

  async open(request: BrowserOpenRequest): Promise<BrowserPageObservation> {
    this.assertOpen()
    const refId = request.refId?.trim()
    const url = request.url?.trim()
    const page = refId ? this.requirePage(refId) : await this.createPage()
    if (url) await this.navigate(page, url)
    if (request.waitMs > 0) await this.#sleep(request.waitMs)
    return this.inspect(page)
  }

  async click(request: BrowserClickRequest): Promise<BrowserPageObservation> {
    this.assertOpen()
    const page = this.requirePage(request.refId)
    const selector = request.selector?.trim()
    const text = request.text?.trim()
    const expression = selector
      ? clickSelectorExpression(selector)
      : text
        ? clickTextExpression(text)
        : undefined
    if (!expression) throw new ValidationError('browser.click', 'requires selector or text')
    const result = recordValue(await this.evaluate(page, expression))
    if (result.ok !== true) {
      throw new ValidationError('browser.click', stringValue(result.reason) || 'target was not found')
    }
    if (request.waitMs > 0) await this.#sleep(request.waitMs)
    return this.inspect(page)
  }

  async find(refId: string, pattern: string): Promise<BrowserFindResult> {
    this.assertOpen()
    const page = this.requirePage(refId)
    // Validate before the pattern compiles and runs on the page main thread,
    // where catastrophic backtracking would block the whole tab.
    assertSafeFindPattern(pattern)
    const result = recordValue(await this.evaluate(page, findExpression(pattern)))
    if (result.ok !== true) {
      throw new ValidationError('pattern', stringValue(result.reason) || 'is not a valid regular expression', pattern)
    }
    const matches = stringArray(result.matches).slice(0, 20)
    return Object.freeze({
      refId,
      pattern,
      matchCount: nonNegativeInteger(result.matchCount, 'matchCount'),
      matches: Object.freeze(matches),
    })
  }

  async screenshot(
    refId: string,
    options: { readonly fullPage: boolean; readonly path?: string },
  ): Promise<BrowserScreenshotResult> {
    this.assertOpen()
    const page = this.requirePage(refId)
    const response = recordValue(await this.command('Page.captureScreenshot', {
      format: 'png',
      captureBeyondViewport: options.fullPage,
    }, page.sessionId))
    const data = stringValue(response.data)
    if (!data) throw new ClientError('cdp', 'Page.captureScreenshot returned no PNG data')
    if (data.length > MAX_SCREENSHOT_BASE64_CHARS) {
      throw new ClientError('cdp', `screenshot exceeded the ${MAX_SCREENSHOT_BASE64_CHARS}-character base64 capture limit`)
    }
    const path = this.screenshotPath(refId, options.path)
    await this.prepareScreenshotDirectory(path)
    try {
      await Bun.write(path, Buffer.from(data, 'base64'))
    } catch (error) {
      throw new ClientError('cdp', `could not write screenshot to ${path}`, error)
    }
    return Object.freeze({ refId, path, fullPage: options.fullPage })
  }

  /** Detach and close every target Xerxes opened; the externally owned browser keeps running. */
  async close(): Promise<void> {
    if (this.#closed) return
    let firstError: unknown
    for (const page of this.#pages.values()) {
      try {
        await this.command('Target.detachFromTarget', { sessionId: page.sessionId })
      } catch (error) {
        firstError ??= error
      }
      try {
        // Pages Xerxes created would otherwise stay open in the user's browser
        // after disconnect; close only the targets this adapter created.
        await this.command('Target.closeTarget', { targetId: page.targetId })
      } catch (error) {
        firstError ??= error
      }
    }
    this.#pages.clear()
    this.#closed = true
    try {
      await this.#connection.close()
    } catch (error) {
      firstError ??= error
    }
    if (firstError !== undefined) {
      throw new ClientError('cdp', 'could not fully detach browser session', firstError)
    }
  }

  private async createPage(): Promise<CdpPage> {
    const created = recordValue(await this.command('Target.createTarget', { url: 'about:blank' }))
    const targetId = requiredText(created.targetId, 'Target.createTarget targetId')
    let attached: Record<string, unknown>
    try {
      attached = recordValue(await this.command('Target.attachToTarget', {
        targetId,
        flatten: true,
      }))
    } catch (error) {
      // Do not leak the freshly created target when the attach fails.
      await this.command('Target.closeTarget', { targetId }).catch(() => undefined)
      throw error
    }
    const sessionId = requiredText(attached.sessionId, 'Target.attachToTarget sessionId')
    const page = Object.freeze({
      refId: `page_${crypto.randomUUID().replaceAll('-', '').slice(0, 10)}`,
      sessionId,
      targetId,
    })
    this.#pages.set(page.refId, page)
    return page
  }

  private async navigate(page: CdpPage, url: string): Promise<void> {
    const response = recordValue(await this.command('Page.navigate', { url }, page.sessionId))
    const errorText = stringValue(response.errorText)
    if (errorText) throw new ClientError('cdp', `navigation failed: ${errorText}`)
  }

  private async inspect(page: CdpPage): Promise<BrowserPageObservation> {
    const value = recordValue(await this.evaluate(page, inspectionExpression()))
    const url = requiredText(value.url, 'browser page url')
    const links = arrayValue(value.links)
      .map(item => stringValue(recordValue(item).url))
      .filter((link): link is string => Boolean(link))
      .map((link, id) => Object.freeze({ id, url: link }))
    return Object.freeze({
      refId: page.refId,
      url,
      title: stringValue(value.title) || '',
      contentPreview: stringValue(value.contentPreview) || '',
      links: Object.freeze(links),
    })
  }

  private async evaluate(page: CdpPage, expression: string): Promise<unknown> {
    const response = recordValue(await this.command('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    }, page.sessionId))
    if (response.exceptionDetails !== undefined) {
      throw new ClientError('cdp', `page evaluation failed: ${exceptionMessage(response.exceptionDetails)}`)
    }
    const result = recordValue(response.result)
    if (!Object.hasOwn(result, 'value')) {
      throw new ClientError('cdp', 'page evaluation returned no serializable value')
    }
    return result.value
  }

  private async command(
    method: string,
    params?: Readonly<Record<string, unknown>>,
    sessionId?: string,
  ): Promise<unknown> {
    if (this.#closed) throw new ClientError('cdp', 'browser connection is closed')
    return this.#connection.command(method, params, sessionId)
  }

  private requirePage(refId: string): CdpPage {
    const page = this.#pages.get(refId)
    if (!page) throw new ValidationError('ref_id', 'browser page not found', refId)
    return page
  }

  private screenshotPath(refId: string, requested: string | undefined): string {
    const directory = this.#screenshotDirectory
    const candidate = requested?.trim()
      ? resolve(requested)
      : join(directory, `${refId.replaceAll(/[^a-zA-Z0-9_-]/g, '_')}.png`)
    const relativePath = relative(directory, candidate)
    if (
      !relativePath ||
      relativePath === '..' ||
      relativePath.startsWith(`..${sep}`) ||
      isAbsolute(relativePath)
    ) {
      throw new ValidationError('path', `must be within the configured screenshot directory: ${directory}`, requested)
    }
    return candidate
  }

  private assertOpen(): void {
    if (this.#closed) throw new ClientError('cdp', 'browser connection is closed')
  }

  /** Create the screenshot directory with private permissions and reject symlinked paths. */
  private async prepareScreenshotDirectory(path: string): Promise<void> {
    await mkdir(this.#screenshotDirectory, { recursive: true, mode: 0o700 })
    await assertRealDirectory(this.#screenshotDirectory)
    const parent = dirname(path)
    if (parent !== this.#screenshotDirectory) {
      await mkdir(parent, { recursive: true, mode: 0o700 })
      await assertRealDirectory(parent)
    }
  }
}

/** Reject a pre-existing attacker-controlled symlink or non-directory at a screenshot path. */
async function assertRealDirectory(path: string): Promise<void> {
  const stats = await lstat(path).catch(() => undefined)
  if (!stats?.isDirectory() || stats.isSymbolicLink()) {
    throw new ClientError('cdp', `screenshot directory is not a real directory: ${path}`)
  }
}

/** Resolve an HTTP remote-debugging endpoint or a direct browser WebSocket URL. */
export async function resolveCdpWebSocketUrl(
  endpoint: URL,
  fetchImplementation: CdpFetch = defaultFetch,
  options: { readonly allowedHosts?: readonly string[] } = {},
): Promise<string> {
  if (endpoint.protocol === 'ws:' || endpoint.protocol === 'wss:') return endpoint.toString()
  const versionUrl = new URL('/json/version', endpoint)
  let response: Response
  try {
    // Never follow redirects: a redirect would let the endpoint silently swap the
    // discovery origin and defeat the host pinning enforced below.
    response = await fetchImplementation(versionUrl.toString(), { redirect: 'error' })
  } catch (error) {
    throw new ClientError('cdp', `could not reach ${redactedCdpEndpoint(versionUrl.toString())}`, error)
  }
  if (!response.ok) {
    throw new ClientError('cdp', `remote debugging endpoint returned HTTP ${response.status}`)
  }
  const payload = recordValue(await response.json().catch(() => undefined))
  const webSocketDebuggerUrl = stringValue(payload.webSocketDebuggerUrl)
  if (!webSocketDebuggerUrl) {
    throw new ClientError('cdp', 'remote debugging endpoint did not return webSocketDebuggerUrl')
  }
  const resolved = cdpEndpointUrl(webSocketDebuggerUrl)
  if (resolved.protocol !== 'ws:' && resolved.protocol !== 'wss:') {
    throw new ClientError('cdp', 'webSocketDebuggerUrl must use ws:// or wss://')
  }
  assertPinnedWebSocketEndpoint(endpoint, resolved, options.allowedHosts ?? [])
  return resolved.toString()
}

/**
 * Pin the discovered WebSocket URL to the supplied endpoint so a compromised
 * debugging endpoint cannot relay Xerxes into an internal target (SSRF). An
 * explicitly allow-listed hostname bypasses the pin by deliberate configuration.
 */
function assertPinnedWebSocketEndpoint(endpoint: URL, resolved: URL, allowedHosts: readonly string[]): void {
  const resolvedHost = resolved.hostname.toLowerCase()
  if (allowedHosts.some(host => host.trim().toLowerCase() === resolvedHost)) return
  if (resolvedHost !== endpoint.hostname.toLowerCase() || effectivePort(resolved) !== effectivePort(endpoint)) {
    throw new ClientError(
      'cdp',
      `webSocketDebuggerUrl host ${redactedCdpEndpoint(resolved.toString())} does not match the supplied endpoint`
        + ' host and port; add it to allowedWebSocketHosts to trust it explicitly',
    )
  }
}

/** Effective port for pinning comparisons, filling in protocol defaults. */
function effectivePort(endpoint: URL): string {
  if (endpoint.port) return endpoint.port
  return endpoint.protocol === 'https:' || endpoint.protocol === 'wss:' ? '443' : '80'
}

/** Validate a user-supplied browser endpoint without imposing public-web URL policy on an explicit local control port. */
export function cdpEndpointUrl(value: string): URL {
  const raw = value.trim()
  if (!raw) throw new ValidationError('browser.cdp_url', 'must not be empty', value)
  let endpoint: URL
  try {
    endpoint = new URL(raw)
  } catch (error) {
    throw new ValidationError('browser.cdp_url', 'must be an absolute http(s) or ws(s) URL', value, { cause: error })
  }
  if (!['http:', 'https:', 'ws:', 'wss:'].includes(endpoint.protocol)) {
    throw new ValidationError('browser.cdp_url', 'must use http(s) or ws(s)', value)
  }
  if (!endpoint.hostname) throw new ValidationError('browser.cdp_url', 'must include a host', value)
  return endpoint
}

/** Remove query credentials and basic-auth data before displaying an endpoint to a user. */
export function redactedCdpEndpoint(value: string): string {
  const endpoint = cdpEndpointUrl(value)
  endpoint.username = ''
  endpoint.password = ''
  endpoint.search = ''
  endpoint.hash = ''
  if (endpoint.protocol === 'http:' || endpoint.protocol === 'https:') {
    endpoint.pathname = ''
  }
  return endpoint.toString().replace(/\/$/, '')
}

/** Bun's native WebSocket client behind the injectable CDP transport boundary. */
export class BunCdpConnectionFactory implements CdpConnectionFactory {
  async connect(endpoint: string): Promise<CdpConnection> {
    const url = cdpEndpointUrl(endpoint)
    if (url.protocol !== 'ws:' && url.protocol !== 'wss:') {
      throw new ValidationError('browser.cdp_url', 'CDP transport must use ws:// or wss://', endpoint)
    }
    return BunCdpConnection.connect(url.toString())
  }
}

class BunCdpConnection implements CdpConnection {
  readonly #pending = new Map<number, PendingCommand>()
  #closed = false
  #nextId = 1

  private constructor(private readonly socket: WebSocket) {
    socket.addEventListener('message', event => {
      void messageText(event.data).then(text => this.receive(text), error => this.fail(error))
    })
    socket.addEventListener('error', event => this.fail(new ClientError('cdp', 'WebSocket error', event)))
    socket.addEventListener('close', event => {
      if (!this.#closed) this.fail(new ClientError('cdp', `WebSocket closed (${event.code})`))
    })
  }

  static async connect(endpoint: string): Promise<BunCdpConnection> {
    return new Promise((resolveConnection, rejectConnection) => {
      let socket: WebSocket
      try {
        socket = new WebSocket(endpoint)
      } catch (error) {
        rejectConnection(new ClientError('cdp', 'could not create WebSocket', error))
        return
      }
      let settled = false
      const timer = setTimeout(() => {
        if (settled) return
        settled = true
        socket.close()
        rejectConnection(new ClientError('cdp', `connection timed out after ${CDP_CONNECT_TIMEOUT}ms`))
      }, CDP_CONNECT_TIMEOUT)
      socket.addEventListener('open', () => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        resolveConnection(new BunCdpConnection(socket))
      }, { once: true })
      socket.addEventListener('error', event => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        rejectConnection(new ClientError('cdp', 'WebSocket could not connect', event))
      }, { once: true })
      socket.addEventListener('close', event => {
        if (settled) return
        settled = true
        clearTimeout(timer)
        rejectConnection(new ClientError('cdp', `WebSocket closed before opening (${event.code})`))
      }, { once: true })
    })
  }

  command(
    method: string,
    params?: Readonly<Record<string, unknown>>,
    sessionId?: string,
  ): Promise<unknown> {
    if (this.#closed || this.socket.readyState !== WebSocket.OPEN) {
      return Promise.reject(new ClientError('cdp', 'WebSocket is not open'))
    }
    const id = this.#nextId++
    const frame = {
      id,
      method,
      ...(params === undefined ? {} : { params }),
      ...(sessionId === undefined ? {} : { sessionId }),
    }
    return new Promise((resolveCommand, rejectCommand) => {
      const timer = setTimeout(() => {
        this.#pending.delete(id)
        rejectCommand(new ClientError('cdp', `${method} timed out after ${CDP_COMMAND_TIMEOUT}ms`))
      }, CDP_COMMAND_TIMEOUT)
      this.#pending.set(id, { resolve: resolveCommand, reject: rejectCommand, timer })
      try {
        this.socket.send(JSON.stringify(frame))
      } catch (error) {
        clearTimeout(timer)
        this.#pending.delete(id)
        rejectCommand(new ClientError('cdp', `could not send ${method}`, error))
      }
    })
  }

  close(): void {
    if (this.#closed) return
    this.#closed = true
    this.socket.close()
    this.rejectPending(new ClientError('cdp', 'browser connection closed'))
  }

  private receive(text: string): void {
    let frame: CdpResponseFrame
    try {
      frame = JSON.parse(text) as CdpResponseFrame
    } catch {
      return
    }
    if (typeof frame.id !== 'number') return
    const pending = this.#pending.get(frame.id)
    if (!pending) return
    this.#pending.delete(frame.id)
    clearTimeout(pending.timer)
    if (frame.error) {
      pending.reject(new ClientError('cdp', cdpErrorMessage(frame.error), frame.error))
      return
    }
    pending.resolve(frame.result ?? {})
  }

  private fail(error: unknown): void {
    if (this.#closed) return
    this.#closed = true
    this.rejectPending(error instanceof Error ? error : new ClientError('cdp', String(error)))
  }

  private rejectPending(error: Error): void {
    for (const pending of this.#pending.values()) {
      clearTimeout(pending.timer)
      pending.reject(error)
    }
    this.#pending.clear()
  }
}

interface PendingCommand {
  readonly reject: (reason?: unknown) => void
  readonly resolve: (value: unknown) => void
  readonly timer: ReturnType<typeof setTimeout>
}

function inspectionExpression(): string {
  return `(() => ({
    url: String(globalThis.location?.href ?? ''),
    title: String(document.title ?? ''),
    contentPreview: String(document.body?.innerText ?? '').slice(0, 2000),
    links: Array.from(document.querySelectorAll('a[href]')).slice(0, ${MAX_INSPECTION_LINKS}).map((anchor) => ({ url: String((anchor).href ?? '') })).filter((link) => Boolean(link.url))
  }))()`
}

function clickSelectorExpression(selector: string): string {
  return `(() => {
    const selector = ${JSON.stringify(selector)};
    const target = document.querySelector(selector);
    if (!(target instanceof HTMLElement)) return { ok: false, reason: 'no element matches selector' };
    target.click();
    return { ok: true };
  })()`
}

function clickTextExpression(text: string): string {
  return `(() => {
    const text = ${JSON.stringify(text)};
    const candidates = Array.from(document.querySelectorAll('a,button,input[type="button"],input[type="submit"],[role="button"],[onclick]'));
    const target = candidates.find((element) => String((element instanceof HTMLInputElement ? element.value : element.textContent) ?? '').includes(text));
    if (!(target instanceof HTMLElement)) return { ok: false, reason: 'no clickable element contains the requested text' };
    target.click();
    return { ok: true };
  })()`
}

function findExpression(pattern: string): string {
  return `(() => {
    const pattern = ${JSON.stringify(pattern)};
    let expression;
    try { expression = new RegExp(pattern, 'gi'); }
    catch (error) { return { ok: false, reason: String(error instanceof Error ? error.message : error) }; }
    const text = String(document.body?.innerText ?? '').slice(0, ${MAX_FIND_SUBJECT_CHARS});
    const matches = [];
    let matchCount = 0;
    let iterations = 0;
    let match;
    // Bounded exec loop: cap both the subject size and the iteration count so a
    // dense-match pattern cannot pin the tab main thread indefinitely.
    while (iterations < ${MAX_FIND_ITERATIONS} && (match = expression.exec(text)) !== null) {
      matchCount += 1;
      if (matches.length < 20) matches.push(String(match[0] ?? ''));
      iterations += 1;
      if (match[0] === '') expression.lastIndex += 1;
    }
    return { ok: true, matchCount, matches, truncated: iterations >= ${MAX_FIND_ITERATIONS} };
  })()`
}

/**
 * Best-effort rejection of catastrophic-backtracking regex shapes before a
 * model-supplied pattern compiles and runs synchronously on the page main
 * thread. Mirrors the documented heuristic used by the native file-edit tool:
 * overlong patterns, unbounded quantifiers on groups that already contain a
 * quantifier, and adjacent unbounded wildcard quantifiers are rejected.
 */
function assertSafeFindPattern(pattern: string): void {
  if (pattern.length > MAX_FIND_PATTERN_CHARS) {
    throw new ValidationError('pattern', `must be at most ${MAX_FIND_PATTERN_CHARS} characters`, pattern)
  }
  const groupStack: boolean[] = []
  let previousAtomDotUnbounded = false
  let index = 0
  const noteQuantifier = (): void => {
    if (groupStack.length > 0) groupStack[groupStack.length - 1] = true
  }
  while (index < pattern.length) {
    const char = pattern[index] ?? ''
    if (char === '\\') {
      index += 2
      previousAtomDotUnbounded = false
      const quantifier = readFindQuantifier(pattern, index)
      if (quantifier !== undefined) {
        noteQuantifier()
        index = quantifier.next
      }
      continue
    }
    if (char === '[') {
      index = skipFindCharacterClass(pattern, index)
      previousAtomDotUnbounded = false
      const quantifier = readFindQuantifier(pattern, index)
      if (quantifier !== undefined) {
        noteQuantifier()
        index = quantifier.next
      }
      continue
    }
    if (char === '(') {
      groupStack.push(false)
      previousAtomDotUnbounded = false
      index += 1
      continue
    }
    if (char === ')') {
      const hadInnerQuantifier = groupStack.pop() ?? false
      index += 1
      const quantifier = readFindQuantifier(pattern, index)
      if (quantifier !== undefined) {
        if (hadInnerQuantifier && quantifier.unbounded) {
          throw new ValidationError(
            'pattern',
            'must not apply an unbounded quantifier to a group that already contains a quantifier'
              + ' (catastrophic-backtracking risk)',
            pattern,
          )
        }
        noteQuantifier()
        index = quantifier.next
      }
      previousAtomDotUnbounded = false
      continue
    }
    const isDot = char === '.'
    index += 1
    const quantifier = readFindQuantifier(pattern, index)
    if (quantifier !== undefined) {
      noteQuantifier()
      if (isDot && quantifier.unbounded) {
        if (previousAtomDotUnbounded) {
          throw new ValidationError(
            'pattern',
            'must not place adjacent unbounded wildcard quantifiers (catastrophic-backtracking risk)',
            pattern,
          )
        }
        previousAtomDotUnbounded = true
      } else {
        previousAtomDotUnbounded = false
      }
      index = quantifier.next
    } else {
      previousAtomDotUnbounded = false
    }
  }
}

/** Read a `*`, `+`, `?`, or `{m,n}` quantifier at index, skipping a lazy `?` suffix. */
function readFindQuantifier(
  pattern: string,
  index: number,
): { readonly next: number; readonly unbounded: boolean } | undefined {
  const char = pattern[index]
  if (char === '*' || char === '+') {
    return { next: index + 1 + (pattern[index + 1] === '?' ? 1 : 0), unbounded: true }
  }
  if (char === '?') {
    return { next: index + 1 + (pattern[index + 1] === '?' ? 1 : 0), unbounded: false }
  }
  if (char === '{') {
    const match = /^\{(\d+)(?:,(\d*))?\}/.exec(pattern.slice(index))
    if (match === null) return undefined
    // {n} and {n,m} are bounded; {n,} is unbounded.
    const unbounded = match[0].includes(',') && match[2] === ''
    return { next: index + match[0].length + (pattern[index + match[0].length] === '?' ? 1 : 0), unbounded }
  }
  return undefined
}

/** Return the index just past the character class that opens at start. */
function skipFindCharacterClass(pattern: string, start: number): number {
  let index = start + 1
  if (pattern[index] === '^') index += 1
  if (pattern[index] === ']') index += 1
  while (index < pattern.length && pattern[index] !== ']') {
    index += pattern[index] === '\\' ? 2 : 1
  }
  return Math.min(index + 1, pattern.length)
}

function cdpErrorMessage(error: CdpResponseError): string {
  const message = stringValue(error.message)
  const code = typeof error.code === 'number' ? ` (${error.code})` : ''
  return message ? `CDP command failed${code}: ${message}` : `CDP command failed${code}`
}

function exceptionMessage(value: unknown): string {
  const details = recordValue(value)
  return stringValue(details.text) || stringValue(recordValue(details.exception).description) || 'unknown exception'
}

function requiredText(value: unknown, field: string): string {
  const text = stringValue(value)
  if (!text) throw new ClientError('cdp', `${field} was missing from the CDP response`)
  return text
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined
}

function stringArray(value: unknown): string[] {
  return arrayValue(value).flatMap(item => typeof item === 'string' ? [item] : [])
}

function arrayValue(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function nonNegativeInteger(value: unknown, field: string): number {
  if (!Number.isInteger(value) || Number(value) < 0) {
    throw new ClientError('cdp', `${field} was not a non-negative integer`)
  }
  return Number(value)
}

function recordValue(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

async function messageText(value: unknown): Promise<string> {
  if (typeof value === 'string') return value
  if (value instanceof ArrayBuffer) return new TextDecoder().decode(value)
  if (ArrayBuffer.isView(value)) return new TextDecoder().decode(value)
  if (value instanceof Blob) return value.text()
  throw new ClientError('cdp', 'WebSocket delivered an unsupported frame')
}

const defaultFetch: CdpFetch = (url, init) => fetch(url, init)
