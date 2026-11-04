// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Parser for the routing settings (`runtime.routing`, `runtime.routing_routes`,
 * env `XERXES_ROUTING`). Routing is OFF unless a mode is set; a bad value is a
 * `ValidationError` naming the key, never a quietly looser setting.
 */

import { ValidationError } from '../core/errors.js'
import type { DataTrust, RouteCandidate, RoutingMode } from './types.js'

/**
 * A route as written in settings. `trust` is optional here: when it is absent
 * the router resolves it from the profile (and fails closed to `cloud` when it
 * cannot).
 */
export interface ConfiguredRoute extends Omit<RouteCandidate, 'trust'> {
  readonly trust?: DataTrust
}

export interface RoutingConfig {
  readonly mode: RoutingMode
  readonly routes: readonly ConfiguredRoute[]
}

const MODES: readonly RoutingMode[] = ['off', 'rules', 'learned']
const TRUSTS: readonly DataTrust[] = ['local', 'cloud']
const ROUTE_KEYS: ReadonlySet<string> = new Set([
  'id',
  'profile',
  'model',
  'trust',
  'tier',
  'input_usd_per_mtok',
  'output_usd_per_mtok',
  'context_limit',
])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function oneOf<T extends string>(field: string, value: unknown, allowed: readonly T[]): T {
  const text = typeof value === 'string' ? value.trim().toLowerCase() : undefined
  const match = allowed.find((candidate) => candidate === text)
  if (match === undefined) throw new ValidationError(field, `must be one of ${allowed.join(', ')}`, value)
  return match
}

function nonEmptyString(field: string, value: unknown): string {
  if (typeof value !== 'string' || value.trim() === '') throw new ValidationError(field, 'must be a non-empty string', value)
  return value.trim()
}

function price(field: string, value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) {
    throw new ValidationError(field, 'must be a non-negative number (USD per million tokens)', value)
  }
  return value
}

function parseRoute(raw: unknown, index: number): ConfiguredRoute {
  const base = `routing_routes[${index}]`
  if (!isRecord(raw)) throw new ValidationError(base, 'must be an object with id, profile and model', raw)
  for (const key of Object.keys(raw)) {
    if (!ROUTE_KEYS.has(key)) throw new ValidationError(`${base}.${key}`, 'unknown setting', raw[key])
  }
  const id = nonEmptyString(`${base}.id`, raw.id)
  const profile = nonEmptyString(`${base}.profile`, raw.profile)
  const model = nonEmptyString(`${base}.model`, raw.model)
  const route: {
    id: string
    profile: string
    model: string
    trust?: DataTrust
    tier?: 1 | 2 | 3
    inputUsdPerMTok?: number
    outputUsdPerMTok?: number
    contextLimit?: number
  } = { id, profile, model }
  if (raw.trust !== undefined) route.trust = oneOf(`${base}.trust`, raw.trust, TRUSTS)
  if (raw.tier !== undefined) {
    const tier = raw.tier
    if (tier !== 1 && tier !== 2 && tier !== 3) throw new ValidationError(`${base}.tier`, 'must be 1, 2 or 3', tier)
    route.tier = tier
  }
  if (raw.input_usd_per_mtok !== undefined) route.inputUsdPerMTok = price(`${base}.input_usd_per_mtok`, raw.input_usd_per_mtok)
  if (raw.output_usd_per_mtok !== undefined) route.outputUsdPerMTok = price(`${base}.output_usd_per_mtok`, raw.output_usd_per_mtok)
  if (raw.context_limit !== undefined) {
    const limit = raw.context_limit
    if (typeof limit !== 'number' || !Number.isInteger(limit) || limit <= 0) {
      throw new ValidationError(`${base}.context_limit`, 'must be a positive integer', limit)
    }
    route.contextLimit = limit
  }
  return Object.freeze(route)
}

/**
 * Parse `runtime.routing` and `runtime.routing_routes`. `XERXES_ROUTING` wins
 * over the file. With nothing set the mode is `off` and there are no routes.
 */
export function parseRoutingConfig(
  runtime: Record<string, unknown>,
  env: Record<string, string | undefined>,
): RoutingConfig {
  const envMode = env.XERXES_ROUTING?.trim()
  const mode: RoutingMode = envMode
    ? oneOf('XERXES_ROUTING', envMode, MODES)
    : runtime.routing === undefined ? 'off' : oneOf('routing', runtime.routing, MODES)

  const rawRoutes = runtime.routing_routes
  if (rawRoutes !== undefined && !Array.isArray(rawRoutes)) {
    throw new ValidationError('routing_routes', 'must be an array of routes', rawRoutes)
  }
  const routes: ConfiguredRoute[] = []
  const seen = new Set<string>()
  const list: readonly unknown[] = rawRoutes ?? []
  list.forEach((raw, index) => {
    const route = parseRoute(raw, index)
    if (seen.has(route.id)) throw new ValidationError(`routing_routes[${index}].id`, `duplicate route id "${route.id}"`, route.id)
    seen.add(route.id)
    routes.push(route)
  })
  return Object.freeze({ mode, routes: Object.freeze(routes) })
}
