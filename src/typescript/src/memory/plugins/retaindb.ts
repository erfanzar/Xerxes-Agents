// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  ExternalMemoryProviderBase,
  type ExternalMemoryUpstream,
  type MemoryPluginEnvironment,
} from './base.js'

/** Explicit RetainDB configuration. The host owns the write-behind queue and transport. */
export interface RetainDBProviderOptions {
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly upstream: ExternalMemoryUpstream
}

/** RetainDB adapter with a host-owned durable queue or remote client. */
export class RetainDBProvider extends ExternalMemoryProviderBase {
  constructor(options: RetainDBProviderOptions) {
    super({
      name: 'retaindb',
      namespaceLabel: 'retain',
      requiredEnvironment: ['RETAINDB_API_KEY'],
      environment: options.environment,
      upstream: options.upstream,
      ...(options.clock === undefined ? {} : { clock: options.clock }),
    })
  }
}
