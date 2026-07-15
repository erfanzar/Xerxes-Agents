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

/** Default Supermemory cloud API endpoint. */
export const DEFAULT_SUPERMEMORY_BASE_URL = 'https://api.supermemory.ai/v1'

/** Explicit Supermemory configuration and host-owned HTTP transport. */
export interface SupermemoryProviderOptions {
  readonly baseUrl?: string
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly transport: MemoryPluginHttpTransport
}

/** Supermemory HTTP adapter with no implicit client or network fallback. */
export class SupermemoryProvider extends ExternalMemoryProviderBase {
  readonly baseUrl: string

  constructor(options: SupermemoryProviderOptions) {
    const baseUrl = options.baseUrl ?? DEFAULT_SUPERMEMORY_BASE_URL
    super({
      name: 'supermemory',
      namespaceLabel: 'super',
      requiredEnvironment: ['SUPERMEMORY_API_KEY'],
      environment: options.environment,
      upstream: new HttpMemoryUpstream({
        providerName: 'supermemory',
        transport: options.transport,
        requestFor: (action, arguments_) => {
          const apiKey = requiredMemoryPluginEnvironment(options.environment, 'SUPERMEMORY_API_KEY')
          const headers = { Authorization: `Bearer ${apiKey}` }
          if (action === 'add') {
            return {
              method: 'POST',
              url: memoryPluginUrl(baseUrl, 'memories'),
              headers,
              body: { content: requiredMemoryPluginArgument(arguments_, 'content') },
            }
          }
          if (action === 'search') {
            return {
              method: 'POST',
              url: memoryPluginUrl(baseUrl, 'memories/search'),
              headers,
              body: { q: requiredMemoryPluginArgument(arguments_, 'query') },
            }
          }
          if (action === 'list') {
            return {
              method: 'GET',
              url: memoryPluginUrl(baseUrl, 'memories', { limit: memoryPluginLimit(arguments_, 20) }),
              headers,
            }
          }
          const entryId = requiredMemoryPluginArgument(arguments_, 'entry_id')
          return {
            method: 'DELETE',
            url: memoryPluginUrl(baseUrl, `memories/${memoryPluginPathSegment(entryId)}`),
            headers,
          }
        },
      }),
      ...(options.clock === undefined ? {} : { clock: options.clock }),
    })
    this.baseUrl = baseUrl
  }
}
