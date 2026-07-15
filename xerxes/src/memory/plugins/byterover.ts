// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  ExternalMemoryProviderBase,
  memoryPluginLimit,
  optionalMemoryPluginArgument,
  requiredMemoryPluginArgument,
  requiredMemoryPluginEnvironment,
  type MemoryPluginEnvironment,
} from './base.js'
import { HttpMemoryUpstream, memoryPluginPathSegment, memoryPluginUrl, type MemoryPluginHttpTransport } from './http.js'

/** Default ByteRover cloud API endpoint. */
export const DEFAULT_BYTEROVER_BASE_URL = 'https://api.byterover.dev/v1'

/** Explicit ByteRover configuration and host-owned HTTP transport. */
export interface ByteRoverProviderOptions {
  readonly baseUrl?: string
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly transport: MemoryPluginHttpTransport
}

/** ByteRover hierarchical-context adapter. */
export class ByteRoverProvider extends ExternalMemoryProviderBase {
  readonly baseUrl: string

  constructor(options: ByteRoverProviderOptions) {
    const baseUrl = options.baseUrl ?? DEFAULT_BYTEROVER_BASE_URL
    super({
      name: 'byterover',
      namespaceLabel: 'brv',
      requiredEnvironment: ['BRV_API_KEY'],
      environment: options.environment,
      upstream: new HttpMemoryUpstream({
        providerName: 'byterover',
        transport: options.transport,
        requestFor: (action, arguments_) => {
          const apiKey = requiredMemoryPluginEnvironment(options.environment, 'BRV_API_KEY')
          const headers = { Authorization: `Bearer ${apiKey}` }
          if (action === 'add') {
            const parent = optionalMemoryPluginArgument(arguments_, 'parent')
            return {
              method: 'POST',
              url: memoryPluginUrl(baseUrl, 'nodes'),
              headers,
              body: {
                content: requiredMemoryPluginArgument(arguments_, 'content'),
                ...(parent === undefined ? {} : { parent }),
              },
            }
          }
          if (action === 'search') {
            return {
              method: 'POST',
              url: memoryPluginUrl(baseUrl, 'nodes/search'),
              headers,
              body: { query: requiredMemoryPluginArgument(arguments_, 'query') },
            }
          }
          if (action === 'list') {
            return {
              method: 'GET',
              url: memoryPluginUrl(baseUrl, 'nodes', { limit: memoryPluginLimit(arguments_, 20) }),
              headers,
            }
          }
          const entryId = requiredMemoryPluginArgument(arguments_, 'entry_id')
          return {
            method: 'DELETE',
            url: memoryPluginUrl(baseUrl, `nodes/${memoryPluginPathSegment(entryId)}`),
            headers,
          }
        },
      }),
      ...(options.clock === undefined ? {} : { clock: options.clock }),
    })
    this.baseUrl = baseUrl
  }
}
