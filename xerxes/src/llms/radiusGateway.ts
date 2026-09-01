// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Radius gateway config client (pi-ai parity, dist/providers/radius-config.ts).
 *
 * The gateway serves its model catalog and pi-messages endpoint base URL from
 * `GET <gateway>/v1/config`. Everything else about the gateway (auth,
 * transport) rides the pi-messages client in ./piMessages.ts.
 */

import type { FetchImplementation } from './client.js'

/** Default hosted Radius gateway (pi-ai parity). */
export const DEFAULT_RADIUS_GATEWAY = 'https://radius.pi.dev'

/** Per-million-token pricing with optional request-wide input tiers (Pi `ModelCost`). */
export interface RadiusGatewayModelCost {
  cacheRead: number
  cacheWrite: number
  input: number
  output: number
  tiers?: { inputTokensAbove: number; input: number; output: number; cacheRead: number; cacheWrite: number }[]
}

/** One gateway model entry as served by `/v1/config` (Pi `RadiusGatewayModel`). */
export interface RadiusGatewayModel {
  contextWindow: number
  cost: RadiusGatewayModelCost
  id: string
  input: ('text' | 'image')[]
  maxTokens: number
  name: string
  reasoning: boolean
  thinkingLevelMap?: Record<string, string | null>
}

/** Gateway catalog: the pi-messages root plus its models (Pi `RadiusGatewayConfig`). */
export interface RadiusGatewayConfig {
  baseUrl: string
  models: RadiusGatewayModel[]
}

/** A gateway model resolved for the pi-messages transport (Pi `Model<"pi-messages">` fields). */
export interface RadiusGatewayModelSpec extends RadiusGatewayModel {
  api: 'pi-messages'
  baseUrl: string
  provider: string
}

function isRadiusGatewayModel(value: unknown): value is RadiusGatewayModel {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) return false
  const model = value as Partial<RadiusGatewayModel>
  return (
    typeof model.id === 'string' &&
    typeof model.name === 'string' &&
    typeof model.reasoning === 'boolean' &&
    Array.isArray(model.input) &&
    typeof model.cost === 'object' &&
    model.cost !== null &&
    !Array.isArray(model.cost) &&
    typeof model.contextWindow === 'number' &&
    typeof model.maxTokens === 'number'
  )
}

/** Validate an untrusted config record; `undefined` when it is not a config. */
export function sanitizeRadiusGatewayConfig(config: unknown): RadiusGatewayConfig | undefined {
  if (typeof config !== 'object' || config === null || Array.isArray(config)) return undefined
  const { baseUrl, models } = config as Partial<RadiusGatewayConfig>
  if (typeof baseUrl !== 'string' || !Array.isArray(models)) return undefined
  return {
    baseUrl,
    models: models.filter(isRadiusGatewayModel).map(model => ({ ...model })),
  }
}

/** Accept either a bare host or a full URL and produce a scheme-less-trailing origin. */
export function normalizeRadiusGatewayUrl(value: string): string {
  const withScheme = /^https?:\/\//iu.test(value) ? value : `https://${value}`
  return withScheme.replace(/\/+$/u, '')
}

/**
 * Validate a stored credential's gateway config (Pi `getRadiusCredentialConfig`
 * minus the OAuth credential wrapper, which is the caller's concern).
 */
export function getRadiusCredentialConfig(config: unknown): RadiusGatewayConfig | undefined {
  return sanitizeRadiusGatewayConfig(config)
}

/** Resolve the gateway's models for one provider id (Pi `getRadiusModelsFromConfig`). */
export function getRadiusModelsFromConfig(providerId: string, config: RadiusGatewayConfig): RadiusGatewayModelSpec[] {
  return config.models.map(model => ({
    ...model,
    api: 'pi-messages' as const,
    provider: providerId,
    baseUrl: config.baseUrl,
  }))
}

function truncateHttpBody(body: string): string {
  const trimmed = body.trim()
  return trimmed.length > 512 ? `${trimmed.slice(0, 512)}…` : trimmed
}

export interface LoadRadiusGatewayConfigOptions {
  readonly fetchImplementation?: FetchImplementation
}

/**
 * Load the gateway catalog from `GET <gateway>/v1/config` (pi-ai parity,
 * including its exact error strings and validation).
 *
 * Like Pi, the gateway argument is used verbatim in `new URL` — pass a value
 * through {@link normalizeRadiusGatewayUrl} first; a scheme-less host throws
 * from the URL constructor rather than being guessed at.
 */
export async function loadRadiusGatewayConfig(
  gateway: string,
  apiKeyOrToken?: string,
  signal?: AbortSignal,
  options: LoadRadiusGatewayConfigOptions = {},
): Promise<RadiusGatewayConfig> {
  const request = options.fetchImplementation ?? fetch
  const headers: Record<string, string> = { accept: 'application/json' }
  if (apiKeyOrToken) headers.authorization = `Bearer ${apiKeyOrToken}`
  const response = await request(new URL('/v1/config', gateway), { headers, ...(signal ? { signal } : {}) })
  if (!response.ok) {
    throw new Error(
      `Could not load Radius config from ${gateway}: ${response.status}: ${truncateHttpBody(await response.text())}`,
    )
  }
  const config = sanitizeRadiusGatewayConfig(await response.json())
  if (!config) throw new Error(`Invalid Radius config from ${gateway}`)
  return config
}
