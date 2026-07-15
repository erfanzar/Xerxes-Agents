// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  MemoryProviderRegistry,
  memoryProviderRegistry,
  type MemoryProvider,
} from '../provider.js'
import { ByteRoverProvider } from './byterover.js'
import {
  type ExternalMemoryUpstream,
  type MemoryPluginEnvironment,
} from './base.js'
import { HindsightProvider } from './hindsight.js'
import { HolographicProvider } from './holographic.js'
import { HonchoProvider } from './honcho.js'
import { type MemoryPluginHttpTransport } from './http.js'
import { Mem0Provider } from './mem0.js'
import { OpenVikingProvider } from './openviking.js'
import { RetainDBProvider } from './retaindb.js'
import { SupermemoryProvider } from './supermemory.js'

/** Names of the external-memory providers formerly registered by the Python plugin package. */
export const BUILTIN_MEMORY_PROVIDER_NAMES = Object.freeze([
  'honcho',
  'mem0',
  'hindsight',
  'holographic',
  'retaindb',
  'openviking',
  'supermemory',
  'byterover',
] as const)

export type BuiltinMemoryProviderName = typeof BUILTIN_MEMORY_PROVIDER_NAMES[number]

/**
 * Explicit host ports needed to construct every built-in memory provider.
 *
 * The native runtime deliberately does not discover SDKs, read credentials,
 * or make ambient network requests while importing the plugin package. The
 * embedding application supplies every effectful boundary here, then calls
 * {@link registerBuiltinMemoryProviders} during its own startup sequence.
 */
export interface BuiltinMemoryProviderDependencies {
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly honchoUpstream: ExternalMemoryUpstream
  readonly holographicUpstream: ExternalMemoryUpstream
  readonly httpTransport: MemoryPluginHttpTransport
  readonly retainDbUpstream: ExternalMemoryUpstream
}

/** Build the complete built-in provider set without mutating a registry. */
export function createBuiltinMemoryProviders(
  dependencies: BuiltinMemoryProviderDependencies,
): readonly MemoryProvider[] {
  const clock = dependencies.clock === undefined ? {} : { clock: dependencies.clock }
  return Object.freeze([
    new HonchoProvider({
      environment: dependencies.environment,
      upstream: dependencies.honchoUpstream,
      ...clock,
    }),
    new Mem0Provider({
      environment: dependencies.environment,
      transport: dependencies.httpTransport,
      ...clock,
    }),
    new HindsightProvider({
      environment: dependencies.environment,
      transport: dependencies.httpTransport,
      ...clock,
    }),
    new HolographicProvider({
      environment: dependencies.environment,
      upstream: dependencies.holographicUpstream,
      ...clock,
    }),
    new RetainDBProvider({
      environment: dependencies.environment,
      upstream: dependencies.retainDbUpstream,
      ...clock,
    }),
    new OpenVikingProvider({
      environment: dependencies.environment,
      transport: dependencies.httpTransport,
      ...clock,
    }),
    new SupermemoryProvider({
      environment: dependencies.environment,
      transport: dependencies.httpTransport,
      ...clock,
    }),
    new ByteRoverProvider({
      environment: dependencies.environment,
      transport: dependencies.httpTransport,
      ...clock,
    }),
  ])
}

/**
 * Register all built-ins in one registry without choosing an active provider.
 *
 * Re-registering intentionally replaces same-named providers, matching the
 * legacy registry's replacement behavior while preserving its active slot.
 */
export function registerBuiltinMemoryProviders(
  dependencies: BuiltinMemoryProviderDependencies,
  registry: MemoryProviderRegistry = memoryProviderRegistry(),
): readonly MemoryProvider[] {
  const providers = createBuiltinMemoryProviders(dependencies)
  for (const provider of providers) registry.register(provider)
  return providers
}
