// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * GitHub Copilot OAuth session.
 *
 * A Copilot subscription (Individual, Business, Enterprise) can drive the
 * Copilot proxy API with an OAuth access token instead of a metered PAT.
 * The client id below is the public first-party device-flow client the GitHub
 * Copilot tooling ships; the flow is the RFC 8628 device grant against
 * github.com plus a token exchange against `copilot_internal/v2/token`.
 *
 * The exchanged token is NOT a JWT — it is a semicolon-delimited claim string
 * (`tid=…;exp=…;sku=…;proxy-ep=…`) whose `proxy-ep` names the API base the
 * token is valid for. That derivation lives in {@link copilotApiBase}; it is
 * read per request rather than cached, because the base follows the SKU on
 * the token, not the account.
 *
 * Persistence is an `auth.json`-style credential file at
 * `<xerxesHome>/auth/copilot.json`: `access` is the exchanged Copilot token,
 * `refresh` is the long-lived GitHub OAuth token it was minted from, and
 * `expires` is epoch SECONDS (Copilot reports short-lived tokens; everything
 * in this module compares in seconds).
 */

import { mkdir, readFile, rm, writeFile } from 'node:fs/promises'
import { dirname, join } from 'node:path'

import { ConfigurationError, ProviderError } from '../core/errors.js'
import { xerxesHome as defaultXerxesHome } from '../daemon/paths.js'

/** Provider key carried in errors surfaced from this module. */
export const COPILOT_PROVIDER = 'github-copilot'

/** Public device-flow client id of the GitHub Copilot tooling. */
export const COPILOT_CLIENT_ID = 'Iv1.b507a08c87ecfe98'

/** OAuth scope requested for the device flow. */
export const COPILOT_SCOPE = 'read:user'

export const COPILOT_DEVICE_CODE_URL = 'https://github.com/login/device/code'
export const COPILOT_TOKEN_URL = 'https://github.com/login/oauth/access_token'
export const COPILOT_TOKEN_EXCHANGE_URL = 'https://api.github.com/copilot_internal/v2/token'

/** RFC 8628 device grant type string used when polling for the token. */
export const COPILOT_DEVICE_GRANT_TYPE = 'urn:ietf:params:oauth:grant-type:device_code'

/** API version the Copilot `/models` catalog is read under. */
export const COPILOT_MODELS_API_VERSION = '2026-06-01'

/** Default API base for individually-licensed Copilot tokens. */
export const COPILOT_API_BASE_DEFAULT = 'https://api.individual.githubcopilot.com'

/**
 * Refresh this far ahead of expiry. Copilot tokens live ~30 minutes and the
 * exchange is cheap, but a token that expires in flight surfaces as a 401
 * mid-stream, which the agent loop cannot retry cleanly once tokens have
 * been emitted.
 */
export const COPILOT_REFRESH_SKEW_SECONDS = 300

/**
 * Deadline for one refresh/exchange HTTP call.
 *
 * Bun's `fetch` has no default timeout, so an unbounded call here never
 * settles — and a hung exchange wedges every later caller waiting on the
 * same credential. Generous, because the exchange is short and the cost of
 * a false timeout is a spurious re-authentication prompt.
 */
export const COPILOT_REQUEST_TIMEOUT_MS = 15_000

/** Identifies the client surface to GitHub and the Copilot proxy (pi-ai's pinned versions). */
export const COPILOT_USER_AGENT = 'GitHubCopilotChat/0.35.0'
export const COPILOT_EDITOR_VERSION = 'vscode/1.107.0'
export const COPILOT_EDITOR_PLUGIN_VERSION = 'copilot-chat/0.35.0'
export const COPILOT_INTEGRATION_ID = 'vscode-chat'

/** A resolved, ready-to-send Copilot credential. */
export interface CopilotCredential {
  /** Exchanged Copilot proxy token, sent as the bearer credential. */
  access: string
  /** Long-lived GitHub OAuth token the Copilot token was minted from. */
  refresh: string
  /** Access-token expiry in epoch seconds. */
  expires: number
  /** Non-default API base reported by the exchange, when present. */
  enterpriseUrl?: string
  /** Model ids the parent wiring has already resolved for this credential. */
  availableModelIds?: string[]
}

export interface CopilotSessionOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly fetchImplementation?: typeof fetch
  /** Overrides `<xerxesHome>` for the credential file location. */
  readonly xerxesHome?: string
  /** Seconds since the epoch; injected so expiry logic is testable. */
  readonly now?: () => number
  /** Injectable delay so device-flow pacing is observable without real sleeps. */
  readonly sleep?: (ms: number) => Promise<void>
}

/** Bound one request, honouring a caller's abort as well as the deadline. */
function requestDeadline(timeoutMs: number, signal?: AbortSignal): {
  readonly dispose: () => void
  readonly signal: AbortSignal
} {
  const controller = new AbortController()
  const timer = setTimeout(
    () => controller.abort(new Error(`Copilot request timed out after ${timeoutMs}ms`)),
    timeoutMs,
  )
  const dispose = (): void => clearTimeout(timer)
  if (!signal) return { dispose, signal: controller.signal }
  if (signal.aborted) controller.abort(signal.reason)
  else signal.addEventListener('abort', () => controller.abort(signal.reason), { once: true })
  return { dispose, signal: controller.signal }
}

const defaultSleep = (ms: number): Promise<void> => new Promise(resolve => setTimeout(resolve, ms))

/**
 * Derive the Copilot API base from the token's own claims.
 *
 * Precedence: the `proxy-ep` claim (what the proxy actually fronts for this
 * SKU) first; then an enterprise/domain claim, spelled `https://copilot-api.<domain>`;
 * finally the individual default. The claims are separated by `;`, so the
 * regexes are anchored to that delimiter rather than trusting arbitrary
 * substring positions.
 */
export function copilotApiBase(token: string): string {
  // pi-ai: the JWT's proxy-ep names the proxy host; the API base swaps the
  // `proxy.` subdomain for `api.` (proxy.individual.githubcopilot.com →
  // api.individual.githubcopilot.com).
  const proxy = /proxy-ep=(?:https?:\/\/)?([^;,\s"']+)/i.exec(token)?.[1]
  if (proxy) {
    const host = proxy.replace(/\/+$/, '')
    return `https://${host.replace(/^proxy\./i, 'api.')}`
  }
  const enterprise = /(?:^|;)\s*(?:enterprise|ent|domain)=([a-z0-9.-]+\.[a-z]{2,})/i.exec(token)?.[1]
  if (enterprise) {
    return `https://copilot-api.${enterprise.replace(/^https?:\/\//i, '').replace(/\/+$/, '')}`
  }
  return COPILOT_API_BASE_DEFAULT
}

/** Headers that authorize one Copilot proxy request. */
export function copilotAuthHeaders(credential: CopilotCredential): Record<string, string> {
  return {
    Authorization: `Bearer ${credential.access}`,
    Accept: 'application/json',
    'User-Agent': COPILOT_USER_AGENT,
    'Editor-Version': COPILOT_EDITOR_VERSION,
    'Editor-Plugin-Version': COPILOT_EDITOR_PLUGIN_VERSION,
    'Copilot-Integration-Id': COPILOT_INTEGRATION_ID,
  }
}

/**
 * Per-request Copilot headers the proxy reads beyond authorization.
 *
 * `X-Initiator` tells the backend who drove the turn — a user message starts
 * the conversation (`user`), everything the agent does on its own is
 * `agent`. `Openai-Intent` is fixed to the conversation-edits surface this
 * integration uses. Vision requests must opt in explicitly.
 */
export function copilotRequestHeaders(options: {
  readonly hasImages?: boolean
  readonly lastMessageRole?: string
} = {}): Record<string, string> {
  const headers: Record<string, string> = {
    'X-Initiator': options.lastMessageRole?.trim().toLowerCase() === 'user' ? 'user' : 'agent',
    'Openai-Intent': 'conversation-edits',
  }
  if (options.hasImages) headers['Copilot-Vision-Request'] = 'true'
  return headers
}

/**
 * Read the model ids the credential may actually use from `{apiBase}/models`.
 *
 * Filtering keeps a model only when it can call tools, its license policy
 * allows use (`policy.state` present and not `enabled` drops it; an absent
 * policy falls back to allowed), and it is not hidden from model pickers.
 * A 429 is retried, honouring `retry-after` seconds capped at 30s.
 */
export async function fetchCopilotModels(
  credential: CopilotCredential,
  options: {
    readonly apiBase?: string
    readonly fetchImplementation?: typeof fetch
    readonly signal?: AbortSignal
    readonly sleep?: (ms: number) => Promise<void>
  } = {},
): Promise<string[]> {
  const request = options.fetchImplementation ?? fetch
  const sleep = options.sleep ?? defaultSleep
  const url = `${(options.apiBase ?? copilotApiBase(credential.access)).replace(/\/+$/, '')}/models`
  for (let attempt = 0; ; attempt += 1) {
    const deadline = requestDeadline(COPILOT_REQUEST_TIMEOUT_MS, options.signal)
    let response: Response
    try {
      response = await request(url, {
        headers: {
          ...copilotAuthHeaders(credential),
          'X-GitHub-Api-Version': COPILOT_MODELS_API_VERSION,
        },
        signal: deadline.signal,
      })
    } finally {
      deadline.dispose()
    }
    if (response.status === 429 && attempt < 2) {
      const retryAfter = Number(response.headers.get('retry-after'))
      const seconds = Number.isFinite(retryAfter) && retryAfter >= 0 ? Math.min(retryAfter, 30) : 1
      await sleep(seconds * 1_000)
      continue
    }
    if (!response.ok) {
      throw new ProviderError(
        COPILOT_PROVIDER,
        `Copilot model list request failed (${response.status}): ${(await response.text()).slice(0, 512)}`,
      )
    }
    const payload = asRecord(await response.json())
    const entries = Array.isArray(payload?.data) ? payload.data : []
    const ids: string[] = []
    for (const entry of entries) {
      const record = asRecord(entry)
      if (!record || !isSelectableCopilotModel(record)) continue
      const id = stringField(record, 'id')
      if (id) ids.push(id)
    }
    return ids
  }
}

function isSelectableCopilotModel(record: Record<string, unknown>): boolean {
  const capabilities = asRecord(record.capabilities)
  const supports = asRecord(capabilities?.supports)
  if (supports?.tool_calls === false || capabilities?.tool_calls === false) return false
  const state = stringField(asRecord(record.policy), 'state')
  if (state !== undefined && state !== 'enabled') return false
  if (record.model_picker_enabled === false) return false
  return true
}

interface DeviceFlowStart {
  readonly deviceCode: string
  readonly expiresInSeconds: number
  readonly intervalSeconds: number
  readonly userCode: string
  readonly verificationUri: string
}

/**
 * Resolves a live Copilot credential, exchanging or re-minting as needed.
 *
 * Resolution order: a stored `<xerxesHome>/auth/copilot.json` session first,
 * then an ambient GitHub token (`COPILOT_GITHUB_TOKEN`, `GH_TOKEN`,
 * `GITHUB_TOKEN`) adopted as the refresh credential — so Xerxes works on a
 * machine that already has a GitHub token exported without a device-flow
 * round trip. Concurrent callers on one session share one in-flight
 * resolution so a near-expiry turn does not race N exchanges.
 */
export class CopilotSession {
  private readonly environment: Readonly<Record<string, string | undefined>>
  private readonly fetchImplementation: typeof fetch | undefined
  private readonly home: string
  private readonly now: () => number
  private pending: Promise<CopilotCredential> | undefined
  private readonly sleep: (ms: number) => Promise<void>

  constructor(options: CopilotSessionOptions = {}) {
    this.environment = options.environment ?? process.env
    this.fetchImplementation = options.fetchImplementation
    this.home = options.xerxesHome ?? defaultXerxesHome(this.environment)
    this.now = options.now ?? (() => Date.now() / 1000)
    this.sleep = options.sleep ?? defaultSleep
  }

  /**
   * Return a usable credential, re-exchanging when it is at or near expiry.
   *
   * The in-flight mirror coalesces concurrent callers within this instance;
   * the exchange rotates nothing (the GitHub refresh token is long-lived),
   * so unlike the Codex chain no cross-instance single-flight is required.
   */
  async credential(signal?: AbortSignal): Promise<CopilotCredential> {
    if (this.pending) return this.pending
    const flight = this.resolve(signal)
    const tracked = flight.finally(() => {
      if (this.pending === tracked) this.pending = undefined
    })
    this.pending = tracked
    return tracked
  }

  /**
   * Re-mint the Copilot access token from the credential's GitHub token.
   *
   * The GitHub token itself does not rotate, so a failed exchange leaves the
   * stored credential intact and the caller free to retry.
   */
  async refresh(credential: CopilotCredential, signal?: AbortSignal): Promise<CopilotCredential> {
    if (!credential.refresh) {
      throw new ConfigurationError(
        'copilot_auth',
        "Copilot credential carries no refresh token. Run 'xerxes auth login copilot'.",
      )
    }
    const fresh = await this.exchange(credential.refresh, signal)
    const enterpriseUrl = fresh.enterpriseUrl ?? credential.enterpriseUrl
    const next: CopilotCredential = {
      access: fresh.access,
      refresh: credential.refresh,
      expires: fresh.expires,
      ...(enterpriseUrl === undefined ? {} : { enterpriseUrl }),
      ...(credential.availableModelIds === undefined ? {} : { availableModelIds: credential.availableModelIds }),
    }
    await this.persist(next)
    return next
  }

  /**
   * Run the RFC 8628 device flow to mint a fresh session.
   *
   * `onUserCode` fires once with the code to type and the URL to type it at,
   * then this polls the token endpoint at the flow's interval, stretching it
   * by 5s per `slow_down` as the grant requires, until GitHub issues the
   * token or the device code expires.
   */
  async login(
    onUserCode: (userCode: string, verificationUri: string) => void,
    signal?: AbortSignal,
  ): Promise<CopilotCredential> {
    const device = await this.startDeviceFlow(signal)
    onUserCode(device.userCode, device.verificationUri)
    const githubToken = await this.pollForDeviceToken(device, signal)
    const fresh = await this.exchange(githubToken, signal)
    const credential: CopilotCredential = {
      access: fresh.access,
      refresh: githubToken,
      expires: fresh.expires,
      ...(fresh.enterpriseUrl === undefined ? {} : { enterpriseUrl: fresh.enterpriseUrl }),
    }
    await this.persist(credential)
    return credential
  }

  private async resolve(signal?: AbortSignal): Promise<CopilotCredential> {
    const stored = await this.loadStored()
    const refresh = stored?.refresh ?? this.environmentToken()
    if (!refresh) {
      throw new ConfigurationError(
        'copilot_auth',
        "No Copilot session found. Run 'xerxes auth login copilot', or set COPILOT_GITHUB_TOKEN.",
      )
    }
    if (stored && stored.access && !this.isExpired(stored.expires)) {
      return { ...stored, refresh }
    }
    return this.refresh({
      access: stored?.access ?? '',
      refresh,
      expires: stored?.expires ?? 0,
      ...(stored?.enterpriseUrl === undefined ? {} : { enterpriseUrl: stored.enterpriseUrl }),
    }, signal)
  }

  private isExpired(expires: number): boolean {
    return expires - COPILOT_REFRESH_SKEW_SECONDS <= this.now()
  }

  /** First ambient GitHub token from the environment, when no session is stored. */
  private environmentToken(): string | undefined {
    for (const key of ['COPILOT_GITHUB_TOKEN', 'GH_TOKEN', 'GITHUB_TOKEN']) {
      const value = this.environment[key]?.trim()
      if (value) return value
    }
    return undefined
  }

  private async startDeviceFlow(signal?: AbortSignal): Promise<DeviceFlowStart> {
    const response = await this.postForm(COPILOT_DEVICE_CODE_URL, {
      client_id: COPILOT_CLIENT_ID,
      scope: COPILOT_SCOPE,
    }, signal)
    const payload = asRecord(await response.json())
    const deviceCode = stringField(payload, 'device_code')
    const userCode = stringField(payload, 'user_code')
    const verificationUri = stringField(payload, 'verification_uri')
    if (!deviceCode || !userCode || !verificationUri) {
      throw new ProviderError(COPILOT_PROVIDER, 'device flow start returned an incomplete payload')
    }
    return {
      deviceCode,
      userCode,
      verificationUri,
      // RFC 8628 defaults: 5s pacing, 900s lifetime, when GitHub omits them.
      intervalSeconds: numberField(payload, 'interval') ?? 5,
      expiresInSeconds: numberField(payload, 'expires_in') ?? 900,
    }
  }

  private async pollForDeviceToken(device: DeviceFlowStart, signal?: AbortSignal): Promise<string> {
    const deadline = this.now() + device.expiresInSeconds
    let interval = device.intervalSeconds
    for (;;) {
      if (this.now() > deadline) {
        throw new ProviderError(COPILOT_PROVIDER, 'device code expired before authorization completed')
      }
      await this.sleep(Math.max(0, interval) * 1_000)
      const response = await this.postForm(COPILOT_TOKEN_URL, {
        client_id: COPILOT_CLIENT_ID,
        device_code: device.deviceCode,
        grant_type: COPILOT_DEVICE_GRANT_TYPE,
      }, signal)
      const payload = asRecord(await response.json())
      const token = stringField(payload, 'access_token')
      if (token) return token
      const error = stringField(payload, 'error')
      if (error === 'authorization_pending') continue
      if (error === 'slow_down') {
        // RFC 8628: the poll interval MUST grow by 5 seconds on slow_down.
        interval += 5
        continue
      }
      throw new ProviderError(COPILOT_PROVIDER, `device flow failed: ${error ?? 'unknown error'}`)
    }
  }

  private async postForm(
    url: string,
    fields: Record<string, string>,
    signal?: AbortSignal,
  ): Promise<Response> {
    const request = this.fetchImplementation ?? fetch
    const deadline = requestDeadline(COPILOT_REQUEST_TIMEOUT_MS, signal)
    try {
      return await request(url, {
        method: 'POST',
        headers: {
          Accept: 'application/json',
          'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: new URLSearchParams(fields).toString(),
        signal: deadline.signal,
      })
    } finally {
      deadline.dispose()
    }
  }

  /** Exchange the long-lived GitHub token for a short-lived Copilot token. */
  private async exchange(
    githubToken: string,
    signal?: AbortSignal,
  ): Promise<{ access: string; enterpriseUrl?: string; expires: number }> {
    const request = this.fetchImplementation ?? fetch
    const deadline = requestDeadline(COPILOT_REQUEST_TIMEOUT_MS, signal)
    let response: Response
    try {
      response = await request(COPILOT_TOKEN_EXCHANGE_URL, {
        headers: {
          Authorization: `Bearer ${githubToken}`,
          Accept: 'application/json',
          'Editor-Version': COPILOT_EDITOR_VERSION,
          'Editor-Plugin-Version': COPILOT_EDITOR_PLUGIN_VERSION,
          'Copilot-Integration-Id': COPILOT_INTEGRATION_ID,
        },
        signal: deadline.signal,
      })
    } finally {
      deadline.dispose()
    }
    if (!response.ok) {
      throw new ProviderError(
        COPILOT_PROVIDER,
        `Copilot token exchange failed (${response.status}): ${(await response.text()).slice(0, 512)}`,
      )
    }
    const payload = asRecord(await response.json())
    const access = stringField(payload, 'token')
    if (!access) {
      throw new ProviderError(COPILOT_PROVIDER, 'Copilot token exchange returned no token')
    }
    const expiresAt = numberField(payload, 'expires_at')
    const refreshIn = numberField(payload, 'refresh_in')
    if (expiresAt === undefined && refreshIn === undefined) {
      throw new ProviderError(COPILOT_PROVIDER, 'Copilot token exchange reported no expiry')
    }
    const endpoints = asRecord(payload?.endpoints)
    const api = stringField(endpoints, 'api')
    const enterpriseUrl = api !== undefined && !isIndividualApiBase(api) ? api : undefined
    return {
      access,
      expires: expiresAt ?? Math.floor(this.now()) + refreshIn!,
      ...(enterpriseUrl === undefined ? {} : { enterpriseUrl }),
    }
  }

  private credentialPath(): string {
    return join(this.home, 'auth', 'copilot.json')
  }

  /** Where the credential persists, for status/logout surfaces. */
  storedPath(): string {
    return this.credentialPath()
  }

  /** The persisted credential without re-exchanging, for `auth status`. */
  async stored(): Promise<CopilotCredential | undefined> {
    return this.loadStored()
  }

  /** Remove the stored credential. Returns whether one existed. */
  async logout(): Promise<boolean> {
    try {
      await rm(this.credentialPath())
      return true
    } catch {
      return false
    }
  }

  private async loadStored(): Promise<CopilotCredential | undefined> {
    let raw: string
    try {
      raw = await readFile(this.credentialPath(), 'utf8')
    } catch {
      return undefined
    }
    let parsed: unknown
    try {
      parsed = JSON.parse(raw)
    } catch {
      return undefined
    }
    const record = asRecord(parsed)
    const access = stringField(record, 'access')
    const refresh = stringField(record, 'refresh')
    const expires = numberField(record, 'expires')
    if (!access || !refresh || expires === undefined) return undefined
    const enterpriseUrl = stringField(record, 'enterpriseUrl')
    const availableModelIds = Array.isArray(record?.availableModelIds)
      ? record?.availableModelIds.filter((id): id is string => typeof id === 'string' && id.length > 0)
      : undefined
    return {
      access,
      refresh,
      expires,
      ...(enterpriseUrl === undefined ? {} : { enterpriseUrl }),
      ...(availableModelIds === undefined || availableModelIds.length === 0
        ? {}
        : { availableModelIds }),
    }
  }

  private async persist(credential: CopilotCredential): Promise<void> {
    const path = this.credentialPath()
    await mkdir(dirname(path), { recursive: true })
    await writeFile(path, `${JSON.stringify(credential, null, 2)}\n`, 'utf8')
  }
}

function isIndividualApiBase(api: string): boolean {
  try {
    return new URL(api).host === new URL(COPILOT_API_BASE_DEFAULT).host
  } catch {
    return false
  }
}

function asRecord(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined
}

function stringField(record: Record<string, unknown> | undefined, key: string): string | undefined {
  const value = record?.[key]
  return typeof value === 'string' && value ? value : undefined
}

function numberField(record: Record<string, unknown> | undefined, key: string): number | undefined {
  const value = record?.[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}
