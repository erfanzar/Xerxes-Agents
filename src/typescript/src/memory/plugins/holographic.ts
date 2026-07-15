// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  ExternalMemoryProviderBase,
  type ExternalMemoryUpstream,
  type MemoryPluginEnvironment,
} from './base.js'

/** Host-provided local fact-store configuration for Holographic memory. */
export interface HolographicProviderOptions {
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly upstream: ExternalMemoryUpstream
}

/**
 * Holographic fact-store adapter.
 *
 * The host owns the persistent store and can retain Holographic-specific
 * trust or temporal-decay metadata without coupling this package to a file,
 * SQLite implementation, or implicit environment lookup.
 */
export class HolographicProvider extends ExternalMemoryProviderBase {
  constructor(options: HolographicProviderOptions) {
    super({
      name: 'holographic',
      namespaceLabel: 'holo',
      environment: options.environment,
      upstream: options.upstream,
      ...(options.clock === undefined ? {} : { clock: options.clock }),
    })
  }
}
