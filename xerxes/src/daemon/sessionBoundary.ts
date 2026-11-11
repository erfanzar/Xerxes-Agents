// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolve } from 'node:path'

import type { DataTrust, OutboundGate, OutboundRequestSummary } from '../security/boundaryTypes.js'
import {
  boundaryPermissionPort,
  DataBoundary,
  parseDataBoundaryConfig,
  type BoundaryPermissionPort,
} from '../security/dataBoundary.js'
import { DisclosureLog } from '../security/disclosureLog.js'
import type { OutboundGateContext } from './outboundGate.js'

/** What the daemon needs to know about a session to place it on the boundary. */
export interface BoundarySession {
  readonly id?: string
  readonly cwd: string
  readonly model?: string
  readonly metadata: Record<string, unknown>
}

export interface SessionBoundaryOptions<S extends BoundarySession> {
  /** The daemon's `runtime` settings (merged file + live settings). */
  readonly runtime: () => Record<string, unknown>
  readonly env?: Record<string, string | undefined>
  /** Where the session's model sends data. Errors resolving it count as cloud. */
  readonly trustOf: (session: S) => DataTrust
  /** Shared with the subagent host so a child's taint is seen by its parent's checks. */
  readonly registry?: BoundaryRegistry
}

/**
 * One {@link DataBoundary} per workspace root and settings. The main
 * conversation's permission gate and its subagents' gates take theirs from
 * the same registry, so taint recorded by a local child is what the filter
 * on that child's reply reads. New settings start a new boundary.
 */
export class BoundaryRegistry {
  private readonly entries = new Map<string, { readonly key: string; readonly boundary: DataBoundary }>()

  /** The boundary for `root`, or undefined when the mode is off. A malformed setting throws. */
  boundaryFor(root: string, runtime: Record<string, unknown>, env: Record<string, string | undefined> = process.env): DataBoundary | undefined {
    const key = resolve(root)
    const config = parseDataBoundaryConfig(runtime, env, { root: key })
    if (config.mode === 'off') return undefined
    const settings = JSON.stringify(config)
    let entry = this.entries.get(key)
    if (entry === undefined || entry.key !== settings) {
      entry = { key: settings, boundary: new DataBoundary(config, key) }
      this.entries.set(key, entry)
    }
    return entry.boundary
  }
}

/**
 * Build the per-session boundary port for the turn runner.
 *
 * One {@link DataBoundary} is kept per workspace root so taint survives
 * across turns. Mode `off` returns undefined, which leaves the permission
 * gate exactly as it is today. A malformed setting throws (fail closed): the
 * turn fails with the validation error instead of running unprotected.
 */
export function createSessionBoundaryFactory<S extends BoundarySession>(
  options: SessionBoundaryOptions<S>,
): (session: S) => BoundaryPermissionPort | undefined {
  const registry = options.registry ?? new BoundaryRegistry()
  return session => {
    const boundary = registry.boundaryFor(session.cwd, options.runtime(), options.env ?? process.env)
    if (boundary === undefined) return undefined
    const trust = safeTrust(() => options.trustOf(session))
    return boundaryPermissionPort(boundary, () => trust)
  }
}

function safeTrust(resolveTrust: () => DataTrust): DataTrust {
  try {
    return resolveTrust()
  } catch {
    return 'cloud'
  }
}

export interface OutboundGateFactoryOptions<S extends BoundarySession> {
  readonly runtime: () => Record<string, unknown>
  readonly env?: Record<string, string | undefined>
  /** Where the session's model sends data, and the provider name shown to the user. */
  readonly routeOf: (session: S, model: string) => { readonly trust: DataTrust; readonly provider: string }
  /** The approval card. Absent: every request that needs approval is refused (fail closed). */
  readonly approve?: (summary: OutboundRequestSummary, signal?: AbortSignal) => Promise<boolean>
  readonly log?: DisclosureLog
}

/**
 * Build the per-session outbound gate for the turn runner. One gate per
 * workspace root (and settings) so `approve_outbound: first` is remembered
 * across turns. Mode `off` returns undefined: model calls are not wrapped.
 */
export function createOutboundGateFactory<S extends BoundarySession>(
  options: OutboundGateFactoryOptions<S>,
): (session: S, model: string) => OutboundGateContext | undefined {
  const gates = new Map<string, { readonly key: string; readonly gate: OutboundGate }>()
  let log = options.log
  return (session, model) => {
    const root = resolve(session.cwd)
    const config = parseDataBoundaryConfig(options.runtime(), options.env ?? process.env, { root })
    if (config.mode === 'off') return undefined
    const key = JSON.stringify(config)
    let entry = gates.get(root)
    if (entry === undefined || entry.key !== key) {
      log ??= new DisclosureLog()
      entry = {
        key,
        gate: new DataBoundary(config, root).outboundGate({ approve: options.approve ?? (async () => false), log }),
      }
      gates.set(root, entry)
    }
    let route: { readonly trust: DataTrust; readonly provider: string }
    try {
      route = options.routeOf(session, model)
    } catch {
      route = { trust: 'cloud', provider: model }
    }
    return {
      gate: entry.gate,
      trust: route.trust,
      provider: route.provider,
      purpose: 'turn',
      ...(session.id === undefined ? {} : { sessionId: session.id }),
    }
  }
}
