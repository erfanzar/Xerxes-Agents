// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ClientError, ConfigurationError, ValidationError } from '../core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../executors/toolRegistry.js'
import { isJsonObject, type JsonObject, type JsonValue, type ToolDefinition } from '../types/toolCalls.js'
import { optionalInteger, optionalString, requiredString } from './inputs.js'

const DEFAULT_ENTITY_LIMIT = 200
const DEFAULT_TIMEOUT_MS = 15_000
const MAX_ENTITY_LIMIT = 1_000
const MAX_TIMEOUT_MS = 300_000
const DOMAIN_PATTERN = /^[a-z0-9_]+$/
const ENTITY_ID_PATTERN = /^[a-z0-9_]+(?:\.[a-z0-9_]+)+$/

/** Injectable native-fetch boundary for a deliberately configured Home Assistant instance. */
export type HomeAssistantFetch = (
  input: RequestInfo | URL,
  init?: RequestInit,
) => Promise<Response>

/**
 * Explicit Home Assistant REST configuration.
 *
 * Credentials are never read from process environment variables. Hosts must
 * provide a token directly, or deliberately opt into an unauthenticated local
 * gateway with allowUnauthenticated.
 */
export interface HomeAssistantClientOptions {
  readonly allowUnauthenticated?: boolean
  readonly baseUrl: string
  readonly fetchImplementation?: HomeAssistantFetch
  readonly timeoutMs?: number
  readonly token?: string
}

/** Safe typed error that never includes response bodies or credential values. */
export class HomeAssistantRequestError extends ClientError {
  readonly status: number | undefined

  constructor(message: string, status: number | undefined = undefined, cause: unknown = undefined) {
    super('home_assistant', message, cause, status === undefined ? {} : { status })
    this.status = status
  }
}

/**
 * Small REST client for a single configured Home Assistant installation.
 *
 * This intentionally has no singleton and no environment fallback. A host can
 * instantiate one client per tenant/session and inject a mock or hardened fetch
 * implementation at its network boundary.
 */
export class HomeAssistantClient {
  private readonly allowUnauthenticated: boolean
  private readonly baseUrl: URL
  private readonly fetchImplementation: HomeAssistantFetch
  private readonly timeoutMs: number
  private readonly token: string

  constructor(options: HomeAssistantClientOptions) {
    this.allowUnauthenticated = options.allowUnauthenticated ?? false
    this.baseUrl = normalizeBaseUrl(options.baseUrl)
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.timeoutMs = boundedTimeout(options.timeoutMs ?? DEFAULT_TIMEOUT_MS)
    this.token = normalizedToken(options.token)

    if (!this.token && !this.allowUnauthenticated) {
      throw new ConfigurationError(
        'homeAssistant.token',
        'must be supplied explicitly; set allowUnauthenticated only for an intentionally unauthenticated gateway',
      )
    }
  }

  async listStates(signal?: AbortSignal): Promise<readonly JsonObject[]> {
    return requiredObjectArray(await this.requestJson('GET', 'api/states', undefined, signal), 'states')
  }

  async getState(entityId: string, signal?: AbortSignal): Promise<JsonObject | undefined> {
    const normalizedEntityId = homeAssistantEntityId(entityId)
    try {
      const response = await this.requestJson(
        'GET',
        'api/states/' + encodeURIComponent(normalizedEntityId),
        undefined,
        signal,
      )
      if (response === undefined) {
        throw new HomeAssistantRequestError('Home Assistant returned an empty state response')
      }
      return requiredObject(response, 'state')
    } catch (error) {
      if (error instanceof HomeAssistantRequestError && error.status === 404) {
        return undefined
      }
      throw error
    }
  }

  async listServices(signal?: AbortSignal): Promise<readonly JsonObject[]> {
    return requiredObjectArray(await this.requestJson('GET', 'api/services', undefined, signal), 'services')
  }

  async callService(
    domain: string,
    service: string,
    data: JsonObject = {},
    signal?: AbortSignal,
  ): Promise<readonly JsonObject[]> {
    const normalizedDomain = homeAssistantDomain(domain, 'domain')
    const normalizedService = homeAssistantDomain(service, 'service')
    const response = await this.requestJson(
      'POST',
      'api/services/' + encodeURIComponent(normalizedDomain) + '/' + encodeURIComponent(normalizedService),
      data,
      signal,
    )
    if (response === undefined) {
      return []
    }
    return requiredObjectArray(response, 'service response')
  }

  private async requestJson(
    method: 'GET' | 'POST',
    path: string,
    body: JsonObject | undefined,
    signal: AbortSignal | undefined,
  ): Promise<JsonValue | undefined> {
    const headers: Record<string, string> = { Accept: 'application/json' }
    if (body !== undefined) {
      headers['Content-Type'] = 'application/json'
    }
    if (this.token) {
      headers.Authorization = 'Bearer ' + this.token
    }

    const init: RequestInit = {
      headers,
      method,
      signal: requestSignal(signal, this.timeoutMs),
    }
    if (body !== undefined) {
      init.body = JSON.stringify(body)
    }

    let response: Response
    try {
      response = await this.fetchImplementation(this.endpoint(path), init)
    } catch (error) {
      throw new HomeAssistantRequestError('could not reach the configured Home Assistant instance', undefined, error)
    }
    if (!response.ok) {
      throw new HomeAssistantRequestError('Home Assistant request failed with HTTP ' + response.status, response.status)
    }

    let text: string
    try {
      text = await response.text()
    } catch (error) {
      throw new HomeAssistantRequestError('could not read the Home Assistant response', response.status, error)
    }
    if (!text.trim()) {
      return undefined
    }

    try {
      const parsed: unknown = JSON.parse(text)
      if (!isJsonValue(parsed)) {
        throw new Error('response was not JSON-serializable')
      }
      return parsed
    } catch (error) {
      throw new HomeAssistantRequestError('Home Assistant returned invalid JSON', response.status, error)
    }
  }

  private endpoint(path: string): URL {
    const normalized = path.replace(/^\/+/, '')
    if (!normalized || normalized.includes('?') || normalized.includes('#')) {
      throw new ValidationError('homeAssistant.path', 'must be a non-empty relative API path', path)
    }
    return new URL(normalized, this.baseUrl)
  }
}

export const HA_LIST_ENTITIES_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ha_list_entities',
    description: 'List Home Assistant entity states, optionally filtered by domain and area.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        area: { type: 'string', description: 'Optional Home Assistant area ID or area name.' },
        domain: { type: 'string', description: 'Optional entity domain, such as light or switch.' },
        limit: { type: 'integer', minimum: 1, maximum: MAX_ENTITY_LIMIT, default: DEFAULT_ENTITY_LIMIT },
      },
    },
  },
}

export const HA_LIST_SERVICES_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ha_list_services',
    description: 'List Home Assistant service domains, optionally limited to one domain.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        domain: { type: 'string', description: 'Optional service domain, such as light.' },
      },
    },
  },
}

export const HA_GET_STATE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ha_get_state',
    description: 'Get the current state, attributes, and timestamps for one Home Assistant entity.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        entity_id: { type: 'string', description: 'Entity ID, such as light.living_room.' },
      },
      required: ['entity_id'],
    },
  },
}

export const HA_CALL_SERVICE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'ha_call_service',
    description: 'Call an explicitly configured Home Assistant service.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        data: { type: 'object', description: 'JSON service data, for example an entity_id.' },
        domain: { type: 'string', description: 'Service domain, such as light.' },
        service: { type: 'string', description: 'Service name, such as turn_on.' },
      },
      required: ['domain', 'service'],
    },
  },
}

export const HOME_ASSISTANT_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  HA_LIST_ENTITIES_DEFINITION,
  HA_LIST_SERVICES_DEFINITION,
  HA_GET_STATE_DEFINITION,
  HA_CALL_SERVICE_DEFINITION,
]

/**
 * Host-bound configuration for Home Assistant tools.
 *
 * A resolver supports multi-session hosts without a mutable process-global
 * client. The static client is useful for single-installation deployments.
 */
export interface HomeAssistantToolsOptions {
  readonly client?: HomeAssistantClient
  readonly resolveClient?: (context: ToolExecutionContext) => HomeAssistantClient | undefined
}

/** Register Home Assistant tools only after the host supplies an explicit client or resolver. */
export function registerHomeAssistantTools(registry: ToolRegistry, options: HomeAssistantToolsOptions): void {
  if (!options.client && !options.resolveClient) {
    throw new ConfigurationError('homeAssistant.client', 'must be supplied when registering Home Assistant tools')
  }

  registry.register(HA_LIST_ENTITIES_DEFINITION, (inputs, context, signal) => (
    listHomeAssistantEntities(inputs, clientFor(options, context), signal)
  ))
  registry.register(HA_LIST_SERVICES_DEFINITION, (inputs, context, signal) => (
    listHomeAssistantServices(inputs, clientFor(options, context), signal)
  ))
  registry.register(HA_GET_STATE_DEFINITION, (inputs, context, signal) => (
    getHomeAssistantState(inputs, clientFor(options, context), signal)
  ))
  registry.register(HA_CALL_SERVICE_DEFINITION, (inputs, context, signal) => (
    callHomeAssistantService(inputs, clientFor(options, context), signal)
  ))
}

/** List and summarize entities in the Python tool's stable JSON shape. */
export async function listHomeAssistantEntities(
  inputs: JsonObject,
  client: HomeAssistantClient,
  signal?: AbortSignal,
): Promise<JsonObject> {
  const domain = optionalDomain(inputs, 'domain')
  const area = optionalArea(inputs, 'area')
  const limit = entityLimit(inputs)

  try {
    const states = await client.listStates(signal)
    const entities = states
      .filter(state => stateMatches(state, domain, area))
      .slice(0, limit)
      .map(stateSummary)
    return { count: entities.length, entities }
  } catch (error) {
    return { count: 0, entities: [], error: toolErrorCode(error) }
  }
}

/** List services in the Python tool's stable JSON shape. */
export async function listHomeAssistantServices(
  inputs: JsonObject,
  client: HomeAssistantClient,
  signal?: AbortSignal,
): Promise<JsonObject> {
  const domain = optionalDomain(inputs, 'domain')
  try {
    const catalog = await client.listServices(signal)
    const domains = domain === undefined
      ? catalog
      : catalog.filter(serviceDomain => serviceDomain.domain === domain)
    return { domains: [...domains] }
  } catch (error) {
    return { domains: [], error: toolErrorCode(error) }
  }
}

/** Return one entity state, with a safe structured result for missing or failed reads. */
export async function getHomeAssistantState(
  inputs: JsonObject,
  client: HomeAssistantClient,
  signal?: AbortSignal,
): Promise<JsonObject> {
  const entityId = homeAssistantEntityId(requiredString(inputs, 'entity_id'))
  try {
    const state = await client.getState(entityId, signal)
    if (!state || typeof state.entity_id !== 'string') {
      return { entity_id: entityId, error: 'not_found' }
    }
    return stateDetail(state)
  } catch (error) {
    return { entity_id: entityId, error: toolErrorCode(error) }
  }
}

/** Call a Home Assistant service and return a body-safe status result. */
export async function callHomeAssistantService(
  inputs: JsonObject,
  client: HomeAssistantClient,
  signal?: AbortSignal,
): Promise<JsonObject> {
  const domain = homeAssistantDomain(requiredString(inputs, 'domain'), 'domain')
  const service = homeAssistantDomain(requiredString(inputs, 'service'), 'service')
  const data = serviceData(inputs)

  try {
    const changed = await client.callService(domain, service, data, signal)
    return { changed: [...changed], domain, ok: true, service }
  } catch (error) {
    return { domain, error: toolErrorCode(error), ok: false, service }
  }
}

/** Filter raw state records by optional Home Assistant domain and area metadata. */
export function filterHomeAssistantEntities(
  states: readonly JsonObject[],
  domain: string | undefined = undefined,
  area: string | undefined = undefined,
): JsonObject[] {
  return states.filter(state => stateMatches(state, domain, area))
}

function clientFor(options: HomeAssistantToolsOptions, context: ToolExecutionContext): HomeAssistantClient {
  const resolved = options.resolveClient?.(context) ?? options.client
  if (!resolved) {
    throw new ConfigurationError('homeAssistant.client', 'is not configured for this tool context')
  }
  return resolved
}

function stateMatches(state: JsonObject, domain: string | undefined, area: string | undefined): boolean {
  const entityId = typeof state.entity_id === 'string' ? state.entity_id : ''
  if (domain !== undefined && !entityId.startsWith(domain + '.')) {
    return false
  }
  if (area === undefined) {
    return true
  }
  const attributes = isJsonObject(state.attributes) ? state.attributes : {}
  return attributes.area_id === area || attributes.area === area
}

function stateSummary(state: JsonObject): JsonObject {
  return {
    attributes: isJsonObject(state.attributes) ? state.attributes : {},
    entity_id: typeof state.entity_id === 'string' ? state.entity_id : null,
    state: typeof state.state === 'string' ? state.state : null,
  }
}

function stateDetail(state: JsonObject): JsonObject {
  return {
    attributes: isJsonObject(state.attributes) ? state.attributes : {},
    entity_id: typeof state.entity_id === 'string' ? state.entity_id : null,
    last_changed: typeof state.last_changed === 'string' ? state.last_changed : '',
    last_updated: typeof state.last_updated === 'string' ? state.last_updated : '',
    state: typeof state.state === 'string' ? state.state : null,
  }
}

function serviceData(inputs: JsonObject): JsonObject {
  const data = inputs.data
  if (data === undefined) {
    return {}
  }
  if (!isJsonObject(data)) {
    throw new ValidationError('data', 'must be a JSON object', data)
  }
  return data
}

function optionalDomain(inputs: JsonObject, field: string): string | undefined {
  const value = optionalString(inputs, field)
  return value === undefined ? undefined : homeAssistantDomain(value, field)
}

function optionalArea(inputs: JsonObject, field: string): string | undefined {
  const value = optionalString(inputs, field)
  if (value === undefined) {
    return undefined
  }
  const normalized = value.trim()
  if (!normalized) {
    throw new ValidationError(field, 'must be a non-empty string when supplied', value)
  }
  return normalized
}

function entityLimit(inputs: JsonObject): number {
  const limit = optionalInteger(inputs, 'limit', DEFAULT_ENTITY_LIMIT)
  if (limit < 1 || limit > MAX_ENTITY_LIMIT) {
    throw new ValidationError('limit', 'must be between 1 and ' + MAX_ENTITY_LIMIT, limit)
  }
  return limit
}

function homeAssistantEntityId(value: string): string {
  const normalized = value.trim()
  if (!ENTITY_ID_PATTERN.test(normalized)) {
    throw new ValidationError('entity_id', 'must be a valid Home Assistant entity ID', value)
  }
  return normalized
}

function homeAssistantDomain(value: string, field: string): string {
  const normalized = value.trim()
  if (!DOMAIN_PATTERN.test(normalized)) {
    throw new ValidationError(field, 'must contain only lowercase letters, digits, and underscores', value)
  }
  return normalized
}

function normalizeBaseUrl(value: string): URL {
  let parsed: URL
  try {
    parsed = new URL(value)
  } catch (error) {
    throw new ConfigurationError('homeAssistant.baseUrl', 'must be an absolute HTTP(S) URL', { cause: error })
  }
  if (parsed.protocol !== 'http:' && parsed.protocol !== 'https:') {
    throw new ConfigurationError('homeAssistant.baseUrl', 'must use HTTP or HTTPS')
  }
  if (parsed.username || parsed.password) {
    throw new ConfigurationError('homeAssistant.baseUrl', 'must not embed credentials')
  }
  if (parsed.search || parsed.hash) {
    throw new ConfigurationError('homeAssistant.baseUrl', 'must not include a query string or fragment')
  }
  if (!parsed.pathname.endsWith('/')) {
    parsed.pathname += '/'
  }
  return parsed
}

function normalizedToken(value: string | undefined): string {
  const token = value?.trim() ?? ''
  if (token.includes('\n') || token.includes('\r')) {
    throw new ConfigurationError('homeAssistant.token', 'must not contain line breaks')
  }
  return token
}

function boundedTimeout(value: number): number {
  if (!Number.isInteger(value) || value < 1 || value > MAX_TIMEOUT_MS) {
    throw new ConfigurationError('homeAssistant.timeoutMs', 'must be an integer between 1 and ' + MAX_TIMEOUT_MS)
  }
  return value
}

function requestSignal(signal: AbortSignal | undefined, timeoutMs: number): AbortSignal {
  const timeout = AbortSignal.timeout(timeoutMs)
  return signal === undefined ? timeout : AbortSignal.any([signal, timeout])
}

function requiredObjectArray(value: JsonValue | undefined, field: string): JsonObject[] {
  if (!Array.isArray(value)) {
    throw new HomeAssistantRequestError('Home Assistant returned a malformed ' + field + ' response')
  }
  const objects: JsonObject[] = []
  for (const entry of value) {
    if (!isJsonObject(entry)) {
      throw new HomeAssistantRequestError('Home Assistant returned a malformed ' + field + ' response')
    }
    objects.push(entry)
  }
  return objects
}

function requiredObject(value: JsonValue, field: string): JsonObject {
  if (!isJsonObject(value)) {
    throw new HomeAssistantRequestError('Home Assistant returned a malformed ' + field + ' response')
  }
  return value
}

function isJsonValue(value: unknown): value is JsonValue {
  if (
    value === null
    || typeof value === 'boolean'
    || typeof value === 'number'
    || typeof value === 'string'
  ) {
    return true
  }
  if (Array.isArray(value)) {
    return value.every(isJsonValue)
  }
  return isJsonObject(value) && Object.values(value).every(isJsonValue)
}

function toolErrorCode(error: unknown): string {
  if (error instanceof ConfigurationError) {
    return 'not_configured'
  }
  if (error instanceof HomeAssistantRequestError) {
    if (error.status === 401 || error.status === 403) {
      return 'unauthorized'
    }
    if (error.status === 404) {
      return 'not_found'
    }
  }
  return 'request_failed'
}
