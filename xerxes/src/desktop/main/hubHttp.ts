// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The one HTTP call shape the Hub sign-in needs: a JSON POST that sends an
 * `Origin` header and reports the raw `Set-Cookie` lines. Built on
 * `node:http(s)` rather than `fetch`, because the Hub's refresh route insists
 * on an `Origin` and `fetch` implementations may drop that header.
 */

import http from 'node:http'
import https from 'node:https'

export interface HubHttpRequest {
  readonly url: string
  readonly headers: Readonly<Record<string, string>>
  readonly body: string
  readonly timeoutMs: number
}

export interface HubHttpResponse {
  readonly status: number
  readonly setCookies: readonly string[]
  readonly body: string
}

export type HubHttp = (request: HubHttpRequest) => Promise<HubHttpResponse>

const MAX_RESPONSE_BYTES = 256 * 1024

export const nodeHubHttp: HubHttp = request =>
  new Promise<HubHttpResponse>((resolve, reject) => {
    let url: URL
    try {
      url = new URL(request.url)
    } catch {
      reject(new Error('The Hub address is not a valid URL.'))
      return
    }
    const secure = url.protocol === 'https:'
    if (!secure && url.protocol !== 'http:') {
      reject(new Error('The Hub address must be http(s).'))
      return
    }
    const body = Buffer.from(request.body, 'utf8')
    const req = (secure ? https : http).request(
      url,
      {
        method: 'POST',
        agent: false,
        timeout: request.timeoutMs,
        headers: { accept: 'application/json', ...request.headers, 'content-length': String(body.length) },
      },
      response => {
        const chunks: Buffer[] = []
        let size = 0
        response.on('data', (chunk: Buffer) => {
          size += chunk.length
          if (size > MAX_RESPONSE_BYTES) {
            response.destroy(new Error('The Hub answered with too much data.'))
            return
          }
          chunks.push(chunk)
        })
        response.on('error', reject)
        response.on('end', () => {
          resolve({ status: response.statusCode ?? 0, setCookies: response.headers['set-cookie'] ?? [], body: Buffer.concat(chunks).toString('utf8') })
        })
      },
    )
    req.on('timeout', () => req.destroy(new Error('The Hub did not answer in time.')))
    req.on('error', reject)
    req.end(body)
  })
