// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ClientError, ConfigurationError } from '../core/errors.js'
import type { JsonValue } from '../types/toolCalls.js'

const BROWSERBASE_SESSIONS_URL = 'https://api.browserbase.com/v1/sessions'
const BROWSER_USE_SESSIONS_URL = 'https://api.browser-use.com/v1/sessions'
const SENSITIVE_METADATA_KEY = /(api[_-]?key|authorization|bearer|cookie|credential|password|secret|token)/i

/** Browser connection roles supported by the native Bun runtime. */
export const SUPPORTED_BROWSER_PROVIDERS = Object.freeze([
  'local',
  'camofox',
  'browserbase',
  'browser_use',
  'firecrawl',
] as const)

export type BrowserProviderName = (typeof SUPPORTED_BROWSER_PROVIDERS)[number]
export type BrowserProviderSessionKind = 'local' | 'remote' | 'request'
export type BrowserProviderSessionMetadata = Readonly<Record<string, JsonValue>>

/** Public, JSON-safe representation of a provider session. */
export interface BrowserProviderSessionRecord {
  readonly cdp_url_present: boolean
  readonly kind: BrowserProviderSessionKind
  readonly metadata: BrowserProviderSessionMetadata
  readonly provider: BrowserProviderName
  readonly session_id: string
}

export interface BrowserProviderSessionOptions {
  /** Kept in an ECMAScript-private field and never included in serialized records. */
  readonly cdpUrl?: string
  readonly kind: BrowserProviderSessionKind
  readonly metadata?: Readonly<Record<string, JsonValue>>
  readonly provider: BrowserProviderName
  readonly sessionId: string
}

/**
 * Opaque provider-session descriptor.
 *
 * A CDP connection URL can contain a bearer credential. It therefore remains
 * private and is available only to a trusted host that is composing a browser
 * adapter; JSON serialization emits only whether such a URL exists.
 */
export class BrowserProviderSession {
  readonly kind: BrowserProviderSessionKind
  readonly metadata: BrowserProviderSessionMetadata
  readonly provider: BrowserProviderName
  readonly sessionId: string
  #cdpUrl: string | undefined

  constructor(options: BrowserProviderSessionOptions) {
    if (!isBrowserProviderName(options.provider)) {
      throw new TypeError('browser provider session must use a supported provider')
    }
    if (!options.sessionId.trim()) {
      throw new TypeError('browser provider session id must not be empty')
    }
    this.provider = options.provider
    this.sessionId = options.sessionId
    this.kind = options.kind
    this.metadata = copyMetadata(options.metadata ?? {})
    this.#cdpUrl = privateConnectionUrl(options.cdpUrl)
    Object.freeze(this)
  }

  /** Return a connection URL only for the trusted host that builds an adapter. */
  cdpUrlForHost(): string | undefined {
    return this.#cdpUrl
  }

  /** Return a fresh immutable record that is safe to return from a tool call. */
  toRecord(): BrowserProviderSessionRecord {
    return Object.freeze({
      provider: this.provider,
      session_id: this.sessionId,
      kind: this.kind,
      metadata: copyMetadata(this.metadata),
      cdp_url_present: this.#cdpUrl !== undefined,
    })
  }

  toJSON(): BrowserProviderSessionRecord {
    return this.toRecord()
  }
}

export interface BrowserProviderOpenOptions {
  readonly headless?: boolean
}

/** One explicit provider implementation registered with {@link BrowserProviderRegistry}. */
export interface BrowserProvider {
  readonly name: BrowserProviderName
  close(session: BrowserProviderSession): Promise<void>
  open(options?: BrowserProviderOpenOptions): Promise<BrowserProviderSession>
}

/** Explicit HTTP boundary for providers that create managed remote sessions. */
export type BrowserProviderFetcher = (url: string, init: RequestInit) => Promise<Response>

/** Injectable session-ID source for deterministic hosts and tests. */
export type BrowserProviderSessionIdFactory = () => string

/** A local browser host owns the actual automation implementation and lifecycle. */
export interface LocalBrowserHost {
  closeBrowser(session: LocalBrowserHostSession): Promise<void>
  openBrowser(options: LocalBrowserHostOpenOptions): Promise<LocalBrowserHostSession>
}

export interface LocalBrowserHostOpenOptions {
  readonly headless: boolean
}

/** Opaque host-owned resource; it is never copied into a provider session record. */
export interface LocalBrowserHostSession {
  readonly cdpUrl?: string
  readonly id: string
}

export interface LocalProviderOptions {
  readonly host?: LocalBrowserHost
  readonly idFactory?: BrowserProviderSessionIdFactory
}

/** Local browser role backed exclusively by an injected host-owned adapter factory. */
export class LocalProvider implements BrowserProvider {
  readonly name = 'local' as const
  readonly #host: LocalBrowserHost | undefined
  readonly #idFactory: BrowserProviderSessionIdFactory
  readonly #sessions = new Map<string, LocalBrowserHostSession>()

  constructor(options: LocalProviderOptions = {}) {
    this.#host = options.host
    this.#idFactory = options.idFactory ?? defaultSessionId
  }

  async open(options: BrowserProviderOpenOptions = {}): Promise<BrowserProviderSession> {
    const host = this.#host
    if (host === undefined) {
      throw new ConfigurationError('local.host', 'must be injected before opening a local browser session')
    }
    const sessionId = nextSessionId(this.#idFactory, this.name)
    if (this.#sessions.has(sessionId)) {
      throw new ClientError(this.name, 'session identifier is already active')
    }
    const hostSession = await externalCall(this.name, 'browser host could not open a session', () => host.openBrowser({
      headless: headless(options),
    }))
    assertHostSession(hostSession, this.name)
    this.#sessions.set(sessionId, hostSession)
    return new BrowserProviderSession({
      provider: this.name,
      sessionId,
      kind: 'local',
      metadata: { headless: headless(options) },
      ...(hostSession.cdpUrl === undefined ? {} : { cdpUrl: hostSession.cdpUrl }),
    })
  }

  async close(session: BrowserProviderSession): Promise<void> {
    if (session.provider !== this.name) return
    const hostSession = this.#sessions.get(session.sessionId)
    if (hostSession === undefined) return
    this.#sessions.delete(session.sessionId)
    const host = this.#host
    if (host === undefined) return
    await externalCall(this.name, 'browser host could not close the session', () => host.closeBrowser(hostSession))
  }
}

/** Injected process launcher for Camofox; the runtime never invokes Node or Bun process APIs itself. */
export interface CamofoxSpawner {
  spawn(request: CamofoxSpawnRequest): Promise<CamofoxProcess>
}

export interface CamofoxSpawnRequest {
  readonly command: readonly string[]
  readonly headless: boolean
}

/** Opaque Camofox process handle kept private by the provider. */
export interface CamofoxProcess {
  close(): Promise<void>
  readonly cdpUrl?: string
  readonly id: string
}

export interface CamofoxProviderOptions {
  /** Host-selected launch command. No implicit Node executable or script is used. */
  readonly command?: readonly string[]
  readonly idFactory?: BrowserProviderSessionIdFactory
  readonly spawner?: CamofoxSpawner
}

/** Camofox role that delegates all process lifecycle work through an injected spawner. */
export class CamofoxProvider implements BrowserProvider {
  readonly name = 'camofox' as const
  readonly #command: readonly string[] | undefined
  readonly #idFactory: BrowserProviderSessionIdFactory
  readonly #processes = new Map<string, CamofoxProcess>()
  readonly #spawner: CamofoxSpawner | undefined

  constructor(options: CamofoxProviderOptions = {}) {
    this.#command = options.command === undefined ? undefined : normalizedCommand(options.command)
    this.#spawner = options.spawner
    this.#idFactory = options.idFactory ?? defaultSessionId
  }

  async open(options: BrowserProviderOpenOptions = {}): Promise<BrowserProviderSession> {
    const spawner = this.#spawner
    if (spawner === undefined) {
      throw new ConfigurationError('camofox.spawner', 'must be injected before opening a Camofox session')
    }
    const command = this.#command
    if (command === undefined) {
      throw new ConfigurationError('camofox.command', 'must be explicitly configured before opening a Camofox session')
    }
    const sessionId = nextSessionId(this.#idFactory, this.name)
    if (this.#processes.has(sessionId)) {
      throw new ClientError(this.name, 'session identifier is already active')
    }
    const process = await externalCall(this.name, 'Camofox host could not start a session', () => spawner.spawn({
      command,
      headless: headless(options),
    }))
    assertCamofoxProcess(process)
    this.#processes.set(sessionId, process)
    return new BrowserProviderSession({
      provider: this.name,
      sessionId,
      kind: 'local',
      metadata: { headless: headless(options) },
      ...(process.cdpUrl === undefined ? {} : { cdpUrl: process.cdpUrl }),
    })
  }

  async close(session: BrowserProviderSession): Promise<void> {
    if (session.provider !== this.name) return
    const process = this.#processes.get(session.sessionId)
    if (process === undefined) return
    this.#processes.delete(session.sessionId)
    await externalCall(this.name, 'Camofox host could not close the session', () => process.close())
  }
}

export interface BrowserbaseProviderConfig {
  readonly apiKey: string
  readonly endpoint?: string
  readonly projectId: string
}

export interface BrowserbaseProviderOptions {
  /** Explicit credential/configuration object; environment variables are never read. */
  readonly config?: BrowserbaseProviderConfig
  readonly fetcher?: BrowserProviderFetcher
  readonly idFactory?: BrowserProviderSessionIdFactory
}

/** Browserbase role that creates one remote CDP session through an injected HTTP port. */
export class BrowserbaseProvider implements BrowserProvider {
  readonly name = 'browserbase' as const
  readonly #config: BrowserbaseProviderConfig | undefined
  readonly #fetcher: BrowserProviderFetcher | undefined
  readonly #idFactory: BrowserProviderSessionIdFactory

  constructor(options: BrowserbaseProviderOptions = {}) {
    this.#config = options.config
    this.#fetcher = options.fetcher
    this.#idFactory = options.idFactory ?? defaultSessionId
  }

  async open(_options: BrowserProviderOpenOptions = {}): Promise<BrowserProviderSession> {
    const config = browserbaseConfig(this.#config)
    const fetcher = requiredFetcher(this.#fetcher, this.name)
    const url = endpoint(config.endpoint, BROWSERBASE_SESSIONS_URL, 'browserbase.endpoint')
    const payload = await postSession(this.name, fetcher, url, {
      'content-type': 'application/json',
      'x-bb-api-key': config.apiKey,
    }, { projectId: config.projectId })
    const cdpUrl = responseConnectionUrl(payload, 'connectUrl', this.name)
    return new BrowserProviderSession({
      provider: this.name,
      sessionId: nextSessionId(this.#idFactory, this.name),
      kind: 'remote',
      cdpUrl,
    })
  }

  async close(_session: BrowserProviderSession): Promise<void> {}
}

export interface BrowserUseProviderConfig {
  readonly apiKey: string
  readonly endpoint?: string
}

export interface BrowserUseProviderOptions {
  /** Explicit credential/configuration object; environment variables are never read. */
  readonly config?: BrowserUseProviderConfig
  readonly fetcher?: BrowserProviderFetcher
  readonly idFactory?: BrowserProviderSessionIdFactory
}

/** Browser Use role that requests one remote CDP session through an injected HTTP port. */
export class BrowserUseProvider implements BrowserProvider {
  readonly name = 'browser_use' as const
  readonly #config: BrowserUseProviderConfig | undefined
  readonly #fetcher: BrowserProviderFetcher | undefined
  readonly #idFactory: BrowserProviderSessionIdFactory

  constructor(options: BrowserUseProviderOptions = {}) {
    this.#config = options.config
    this.#fetcher = options.fetcher
    this.#idFactory = options.idFactory ?? defaultSessionId
  }

  async open(options: BrowserProviderOpenOptions = {}): Promise<BrowserProviderSession> {
    const config = browserUseConfig(this.#config)
    const fetcher = requiredFetcher(this.#fetcher, this.name)
    const url = endpoint(config.endpoint, BROWSER_USE_SESSIONS_URL, 'browser_use.endpoint')
    const payload = await postSession(this.name, fetcher, url, {
      authorization: `Bearer ${config.apiKey}`,
      'content-type': 'application/json',
    }, { headless: headless(options) })
    const cdpUrl = responseConnectionUrl(payload, 'cdp_url', this.name)
    return new BrowserProviderSession({
      provider: this.name,
      sessionId: nextSessionId(this.#idFactory, this.name),
      kind: 'remote',
      cdpUrl,
      metadata: { headless: headless(options) },
    })
  }

  async close(_session: BrowserProviderSession): Promise<void> {}
}

export interface FirecrawlProviderConfig {
  readonly apiKey: string
}

export interface FirecrawlProviderOptions {
  /** Explicit Firecrawl credentials; this provider does not read process environment state. */
  readonly config?: FirecrawlProviderConfig
  readonly idFactory?: BrowserProviderSessionIdFactory
}

/**
 * Firecrawl's role is request-oriented rather than a persistent browser session.
 * It records only an opaque, configured capability marker; a host must inject a
 * separate Firecrawl client for actual scrape operations.
 */
export class FirecrawlProvider implements BrowserProvider {
  readonly name = 'firecrawl' as const
  readonly #config: FirecrawlProviderConfig | undefined
  readonly #idFactory: BrowserProviderSessionIdFactory

  constructor(options: FirecrawlProviderOptions = {}) {
    this.#config = options.config
    this.#idFactory = options.idFactory ?? defaultSessionId
  }

  async open(options: BrowserProviderOpenOptions = {}): Promise<BrowserProviderSession> {
    firecrawlConfig(this.#config)
    return new BrowserProviderSession({
      provider: this.name,
      sessionId: nextSessionId(this.#idFactory, this.name),
      kind: 'request',
      metadata: { headless: headless(options) },
    })
  }

  async close(_session: BrowserProviderSession): Promise<void> {}
}

/** Typed, process-local registry of the five browser-provider roles. */
export class BrowserProviderRegistry {
  readonly #providers = new Map<BrowserProviderName, BrowserProvider>()

  constructor(providers: readonly BrowserProvider[] = []) {
    for (const provider of providers) this.register(provider)
  }

  register(provider: BrowserProvider): void {
    this.#providers.set(provider.name, provider)
  }

  get(name: BrowserProviderName): BrowserProvider | undefined {
    return this.#providers.get(name)
  }

  names(): readonly BrowserProviderName[] {
    return Object.freeze(SUPPORTED_BROWSER_PROVIDERS.filter(name => this.#providers.has(name)))
  }

  async open(name: BrowserProviderName, options: BrowserProviderOpenOptions = {}): Promise<BrowserProviderSession> {
    const provider = this.#providers.get(name)
    if (provider === undefined) {
      throw new ConfigurationError(`browserProvider.${name}`, 'is not registered')
    }
    const session = await provider.open(options)
    if (session.provider !== name) {
      throw new ClientError(name, 'provider returned a session for a different provider')
    }
    return session
  }

  async close(session: BrowserProviderSession): Promise<void> {
    const provider = this.#providers.get(session.provider)
    if (provider === undefined) {
      throw new ConfigurationError(`browserProvider.${session.provider}`, 'is not registered')
    }
    await provider.close(session)
  }
}

export interface BuiltinBrowserProviderOptions {
  readonly browserbase?: BrowserbaseProviderOptions
  readonly browserUse?: BrowserUseProviderOptions
  readonly camofox?: CamofoxProviderOptions
  readonly firecrawl?: FirecrawlProviderOptions
  readonly local?: LocalProviderOptions
}

/**
 * Build the complete provider registry without selecting an implementation.
 *
 * Every external capability remains unavailable until its host port/config is
 * explicitly injected into the corresponding provider options.
 */
export function createBrowserProviderRegistry(options: BuiltinBrowserProviderOptions = {}): BrowserProviderRegistry {
  return new BrowserProviderRegistry([
    new LocalProvider(options.local),
    new CamofoxProvider(options.camofox),
    new BrowserbaseProvider(options.browserbase),
    new BrowserUseProvider(options.browserUse),
    new FirecrawlProvider(options.firecrawl),
  ])
}

function isBrowserProviderName(value: string): value is BrowserProviderName {
  return (SUPPORTED_BROWSER_PROVIDERS as readonly string[]).includes(value)
}

function headless(options: BrowserProviderOpenOptions): boolean {
  return options.headless ?? true
}

function nextSessionId(factory: BrowserProviderSessionIdFactory, provider: BrowserProviderName): string {
  const sessionId = factory()
  if (typeof sessionId !== 'string' || !sessionId.trim()) {
    throw new ConfigurationError(`browserProvider.${provider}.idFactory`, 'must return a non-empty session identifier')
  }
  return sessionId
}

function defaultSessionId(): string {
  return crypto.randomUUID().replaceAll('-', '')
}

function assertHostSession(value: LocalBrowserHostSession, provider: BrowserProviderName): void {
  if (typeof value !== 'object' || value === null || typeof value.id !== 'string' || !value.id.trim()) {
    throw new ClientError(provider, 'browser host returned an invalid session')
  }
  if (value.cdpUrl !== undefined && (typeof value.cdpUrl !== 'string' || !value.cdpUrl.trim())) {
    throw new ClientError(provider, 'browser host returned an invalid connection URL')
  }
}

function assertCamofoxProcess(value: CamofoxProcess): void {
  if (
    typeof value !== 'object'
    || value === null
    || typeof value.id !== 'string'
    || !value.id.trim()
    || typeof value.close !== 'function'
  ) {
    throw new ClientError('camofox', 'Camofox host returned an invalid process handle')
  }
  if (value.cdpUrl !== undefined && (typeof value.cdpUrl !== 'string' || !value.cdpUrl.trim())) {
    throw new ClientError('camofox', 'Camofox host returned an invalid connection URL')
  }
}

function normalizedCommand(command: readonly string[]): readonly string[] {
  if (!command.length || command.some(part => typeof part !== 'string' || !part.trim())) {
    throw new ConfigurationError('camofox.command', 'must contain non-empty command parts')
  }
  return Object.freeze([...command])
}

function browserbaseConfig(config: BrowserbaseProviderConfig | undefined): BrowserbaseProviderConfig {
  if (config === undefined) {
    throw new ConfigurationError('browserbase.config', 'must be injected before opening a Browserbase session')
  }
  return {
    apiKey: requiredConfigString(config.apiKey, 'browserbase.apiKey'),
    projectId: requiredConfigString(config.projectId, 'browserbase.projectId'),
    ...(config.endpoint === undefined ? {} : { endpoint: config.endpoint }),
  }
}

function browserUseConfig(config: BrowserUseProviderConfig | undefined): BrowserUseProviderConfig {
  if (config === undefined) {
    throw new ConfigurationError('browser_use.config', 'must be injected before opening a Browser Use session')
  }
  return {
    apiKey: requiredConfigString(config.apiKey, 'browser_use.apiKey'),
    ...(config.endpoint === undefined ? {} : { endpoint: config.endpoint }),
  }
}

function firecrawlConfig(config: FirecrawlProviderConfig | undefined): FirecrawlProviderConfig {
  if (config === undefined) {
    throw new ConfigurationError('firecrawl.config', 'must be injected before opening a Firecrawl session')
  }
  return { apiKey: requiredConfigString(config.apiKey, 'firecrawl.apiKey') }
}

function requiredConfigString(value: string, key: string): string {
  if (typeof value !== 'string' || !value.trim()) {
    throw new ConfigurationError(key, 'must be supplied through the injected provider configuration')
  }
  return value
}

function requiredFetcher(
  fetcher: BrowserProviderFetcher | undefined,
  provider: BrowserProviderName,
): BrowserProviderFetcher {
  if (fetcher === undefined) {
    throw new ConfigurationError(`${provider}.fetcher`, 'must be injected before opening a remote browser session')
  }
  return fetcher
}

function endpoint(value: string | undefined, fallback: string, key: string): string {
  if (value === undefined) return fallback
  if (!value.trim()) {
    throw new ConfigurationError(key, 'must be an absolute HTTPS URL')
  }
  try {
    const parsed = new URL(value)
    if (parsed.protocol !== 'https:' || parsed.username || parsed.password) {
      throw new Error('invalid endpoint')
    }
    return parsed.toString()
  } catch {
    throw new ConfigurationError(key, 'must be an absolute HTTPS URL')
  }
}

async function postSession(
  provider: BrowserProviderName,
  fetcher: BrowserProviderFetcher,
  url: string,
  headers: Readonly<Record<string, string>>,
  body: Record<string, JsonValue>,
): Promise<Record<string, unknown>> {
  let response: Response
  try {
    response = await fetcher(url, {
      method: 'POST',
      headers,
      body: JSON.stringify(body),
    })
  } catch {
    throw new ClientError(provider, 'session request failed')
  }
  if (!response.ok) {
    throw new ClientError(provider, `session request failed with status ${response.status}`)
  }
  let payload: unknown
  try {
    payload = await response.json()
  } catch {
    throw new ClientError(provider, 'session response was not valid JSON')
  }
  if (!isPlainRecord(payload)) {
    throw new ClientError(provider, 'session response was not a JSON object')
  }
  return payload
}

function responseConnectionUrl(payload: Record<string, unknown>, key: string, provider: BrowserProviderName): string {
  const value = payload[key]
  if (typeof value !== 'string' || !value.trim()) {
    throw new ClientError(provider, 'session response did not include a connection URL')
  }
  return value
}

async function externalCall<T>(provider: BrowserProviderName, message: string, operation: () => Promise<T>): Promise<T> {
  try {
    return await operation()
  } catch {
    throw new ClientError(provider, message)
  }
}

function privateConnectionUrl(value: string | undefined): string | undefined {
  if (value === undefined) return undefined
  if (typeof value !== 'string' || !value.trim()) {
    throw new TypeError('browser provider connection URL must be a non-empty string')
  }
  return value
}

function copyMetadata(metadata: Readonly<Record<string, unknown>>): BrowserProviderSessionMetadata {
  const copied: Record<string, JsonValue> = {}
  for (const [key, value] of Object.entries(metadata)) {
    if (SENSITIVE_METADATA_KEY.test(key)) continue
    copied[key] = copyJsonValue(value)
  }
  return Object.freeze(copied)
}

function copyJsonValue(value: unknown): JsonValue {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return value
  if (typeof value === 'number') {
    if (Number.isFinite(value)) return value
    throw new TypeError('browser provider session metadata must contain JSON-serializable values')
  }
  if (Array.isArray(value)) {
    return Object.freeze(value.map(copyJsonValue)) as unknown as JsonValue
  }
  if (isPlainRecord(value)) {
    const copied: Record<string, JsonValue> = {}
    for (const [key, nested] of Object.entries(value)) {
      if (SENSITIVE_METADATA_KEY.test(key)) continue
      copied[key] = copyJsonValue(nested)
    }
    return Object.freeze(copied) as JsonValue
  }
  throw new TypeError('browser provider session metadata must contain JSON-serializable values')
}

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(value)
  return prototype === Object.prototype || prototype === null
}
