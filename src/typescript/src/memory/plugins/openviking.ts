// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  ExternalMemoryProviderBase,
  memoryPluginLimit,
  requiredMemoryPluginArgument,
  requiredMemoryPluginEnvironment,
  type MemoryPluginEnvironment,
} from './base.js'
import { HttpMemoryUpstream, memoryPluginPathSegment, memoryPluginUrl, type MemoryPluginHttpTransport } from './http.js'

/** Explicit OpenViking endpoint configuration and host-owned HTTP transport. */
export interface OpenVikingProviderOptions {
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly transport: MemoryPluginHttpTransport
}

/** OpenViking context-DB adapter. Its base URL comes from the explicit endpoint port. */
export class OpenVikingProvider extends ExternalMemoryProviderBase {
  constructor(options: OpenVikingProviderOptions) {
    super({
      name: 'openviking',
      namespaceLabel: 'viking',
      requiredEnvironment: ['OPENVIKING_ENDPOINT', 'OPENVIKING_API_KEY'],
      environment: options.environment,
      upstream: new HttpMemoryUpstream({
        providerName: 'openviking',
        transport: options.transport,
        requestFor: (action, arguments_) => {
          const endpoint = requiredMemoryPluginEnvironment(options.environment, 'OPENVIKING_ENDPOINT')
          const apiKey = requiredMemoryPluginEnvironment(options.environment, 'OPENVIKING_API_KEY')
          const headers = { Authorization: `Bearer ${apiKey}` }
          if (action === 'add') {
            return {
              method: 'POST',
              url: memoryPluginUrl(endpoint, 'v1/contexts'),
              headers,
              body: { content: requiredMemoryPluginArgument(arguments_, 'content') },
            }
          }
          if (action === 'search') {
            return {
              method: 'POST',
              url: memoryPluginUrl(endpoint, 'v1/contexts/search'),
              headers,
              body: { query: requiredMemoryPluginArgument(arguments_, 'query') },
            }
          }
          if (action === 'list') {
            return {
              method: 'GET',
              url: memoryPluginUrl(endpoint, 'v1/contexts', { limit: memoryPluginLimit(arguments_, 20) }),
              headers,
            }
          }
          const entryId = requiredMemoryPluginArgument(arguments_, 'entry_id')
          return {
            method: 'DELETE',
            url: memoryPluginUrl(endpoint, `v1/contexts/${memoryPluginPathSegment(entryId)}`),
            headers,
          }
        },
      }),
      ...(options.clock === undefined ? {} : { clock: options.clock }),
    })
  }
}
