// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  ExternalMemoryProviderBase,
  memoryPluginEnvironmentValue,
  memoryPluginLimit,
  requiredMemoryPluginArgument,
  requiredMemoryPluginEnvironment,
  type MemoryPluginEnvironment,
} from './base.js'
import { HttpMemoryUpstream, memoryPluginPathSegment, memoryPluginUrl, type MemoryPluginHttpTransport } from './http.js'

/** Default Mem0 cloud API endpoint. An embedding application may override it explicitly. */
export const DEFAULT_MEM0_BASE_URL = 'https://api.mem0.ai/v1'

/** Explicit Mem0 configuration and host-owned HTTP transport. */
export interface Mem0ProviderOptions {
  readonly baseUrl?: string
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly transport: MemoryPluginHttpTransport
}

/** Mem0 HTTP adapter with no global fetch, SDK import, or process-environment lookup. */
export class Mem0Provider extends ExternalMemoryProviderBase {
  readonly baseUrl: string

  constructor(options: Mem0ProviderOptions) {
    const baseUrl = options.baseUrl ?? DEFAULT_MEM0_BASE_URL
    super({
      name: 'mem0',
      namespaceLabel: 'mem0',
      requiredEnvironment: ['MEM0_API_KEY'],
      environment: options.environment,
      upstream: new HttpMemoryUpstream({
        providerName: 'mem0',
        transport: options.transport,
        requestFor: (action, arguments_) => {
          const apiKey = requiredMemoryPluginEnvironment(options.environment, 'MEM0_API_KEY')
          const userId = memoryPluginEnvironmentValue(options.environment, 'MEM0_USER_ID', 'xerxes')
          const headers = { Authorization: `Bearer ${apiKey}` }
          if (action === 'add') {
            return {
              method: 'POST',
              url: memoryPluginUrl(baseUrl, 'memories'),
              headers,
              body: {
                messages: [{ role: 'user', content: requiredMemoryPluginArgument(arguments_, 'content') }],
                user_id: userId,
              },
            }
          }
          if (action === 'search') {
            return {
              method: 'POST',
              url: memoryPluginUrl(baseUrl, 'memories/search'),
              headers,
              body: { query: requiredMemoryPluginArgument(arguments_, 'query'), user_id: userId },
            }
          }
          if (action === 'list') {
            return {
              method: 'GET',
              url: memoryPluginUrl(baseUrl, 'memories', { user_id: userId, limit: memoryPluginLimit(arguments_, 20) }),
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
