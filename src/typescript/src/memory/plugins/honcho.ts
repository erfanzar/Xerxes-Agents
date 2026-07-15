// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  ExternalMemoryProviderBase,
  type ExternalMemoryUpstream,
  type MemoryPluginEnvironment,
} from './base.js'

/** Explicit Honcho configuration. The embedding host supplies the actual SDK or HTTP adapter. */
export interface HonchoProviderOptions {
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly upstream: ExternalMemoryUpstream
}

/** Honcho notes provider without import-time SDK probing or an implicit client. */
export class HonchoProvider extends ExternalMemoryProviderBase {
  constructor(options: HonchoProviderOptions) {
    super({
      name: 'honcho',
      namespaceLabel: 'honcho',
      requiredEnvironment: ['HONCHO_API_KEY'],
      environment: options.environment,
      upstream: options.upstream,
      ...(options.clock === undefined ? {} : { clock: options.clock }),
    })
  }
}
