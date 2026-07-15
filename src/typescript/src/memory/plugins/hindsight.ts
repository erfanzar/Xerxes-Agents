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

/** Default Hindsight knowledge-graph API endpoint. */
export const DEFAULT_HINDSIGHT_BASE_URL = 'https://api.hindsight.ai/v1'

/** Explicit Hindsight configuration and host-owned HTTP transport. */
export interface HindsightProviderOptions {
  readonly baseUrl?: string
  readonly clock?: () => number
  readonly environment: MemoryPluginEnvironment
  readonly transport: MemoryPluginHttpTransport
}

/** Hindsight bank adapter with a configurable search budget. */
export class HindsightProvider extends ExternalMemoryProviderBase {
  readonly baseUrl: string

  constructor(options: HindsightProviderOptions) {
    const baseUrl = options.baseUrl ?? DEFAULT_HINDSIGHT_BASE_URL
    super({
      name: 'hindsight',
      namespaceLabel: 'hindsight',
      requiredEnvironment: ['HINDSIGHT_API_KEY', 'HINDSIGHT_BANK_ID'],
      environment: options.environment,
      upstream: new HttpMemoryUpstream({
        providerName: 'hindsight',
        transport: options.transport,
        requestFor: (action, arguments_) => {
          const apiKey = requiredMemoryPluginEnvironment(options.environment, 'HINDSIGHT_API_KEY')
          const bankId = requiredMemoryPluginEnvironment(options.environment, 'HINDSIGHT_BANK_ID')
          const bankPath = `banks/${memoryPluginPathSegment(bankId)}`
          const headers = { 'X-Api-Key': apiKey }
          if (action === 'add') {
            return {
              method: 'POST',
              url: memoryPluginUrl(baseUrl, `${bankPath}/entries`),
              headers,
              body: { content: requiredMemoryPluginArgument(arguments_, 'content') },
            }
          }
          if (action === 'search') {
            return {
              method: 'POST',
              url: memoryPluginUrl(baseUrl, `${bankPath}/search`),
              headers,
              body: {
                query: requiredMemoryPluginArgument(arguments_, 'query'),
                budget: memoryPluginEnvironmentValue(options.environment, 'HINDSIGHT_BUDGET', 'mid'),
              },
            }
          }
          if (action === 'list') {
            return {
              method: 'GET',
              url: memoryPluginUrl(baseUrl, `${bankPath}/entries`, { limit: memoryPluginLimit(arguments_, 20) }),
              headers,
            }
          }
          return {
            method: 'DELETE',
            url: memoryPluginUrl(
              baseUrl,
              `${bankPath}/entries/${memoryPluginPathSegment(requiredMemoryPluginArgument(arguments_, 'entry_id'))}`,
            ),
            headers,
          }
        },
      }),
      ...(options.clock === undefined ? {} : { clock: options.clock }),
    })
    this.baseUrl = baseUrl
  }
}
