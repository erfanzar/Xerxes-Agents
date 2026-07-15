// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ClientError, ConfigurationError } from '../src/core/errors.js'
import {
  BrowserProviderSession,
  BrowserbaseProvider,
  BrowserUseProvider,
  CamofoxProvider,
  FirecrawlProvider,
  LocalProvider,
  createBrowserProviderRegistry,
  type BrowserProviderFetcher,
  type CamofoxSpawner,
  type LocalBrowserHost,
} from '../src/operators/index.js'

test('complete provider registry preserves all five roles but requires explicit host configuration', async () => {
  const providers = createBrowserProviderRegistry()

  expect(providers.names()).toEqual(['local', 'camofox', 'browserbase', 'browser_use', 'firecrawl'])
  await expect(providers.open('local')).rejects.toBeInstanceOf(ConfigurationError)
  await expect(providers.open('camofox')).rejects.toBeInstanceOf(ConfigurationError)
  await expect(providers.open('browserbase')).rejects.toBeInstanceOf(ConfigurationError)
  await expect(providers.open('browser_use')).rejects.toBeInstanceOf(ConfigurationError)
  await expect(providers.open('firecrawl')).rejects.toBeInstanceOf(ConfigurationError)
})

test('local provider delegates lifecycle to an injected host and serializes only a safe session record', async () => {
  const events: string[] = []
  const host: LocalBrowserHost = {
    async openBrowser(options) {
      events.push(`open:${options.headless}`)
      return { id: 'host-local-1', cdpUrl: 'ws://127.0.0.1/devtools?token=local-secret' }
    },
    async closeBrowser(session) {
      events.push(`close:${session.id}`)
    },
  }
  const provider = new LocalProvider({ host, idFactory: () => 'local-session' })
  const session = await provider.open({ headless: false })

  expect(session.toRecord()).toEqual({
    provider: 'local',
    session_id: 'local-session',
    kind: 'local',
    metadata: { headless: false },
    cdp_url_present: true,
  })
  expect(JSON.stringify(session)).not.toContain('local-secret')
  expect(session.cdpUrlForHost()).toContain('local-secret')

  await provider.close(session)
  expect(events).toEqual(['open:false', 'close:host-local-1'])
})

test('Camofox uses the configured host spawner without a Bun or Node process bridge', async () => {
  const requests: Array<{ readonly command: readonly string[]; readonly headless: boolean }> = []
  const events: string[] = []
  const spawner: CamofoxSpawner = {
    async spawn(request) {
      requests.push(request)
      return {
        id: 'camofox-host-1',
        cdpUrl: 'ws://127.0.0.1/camofox?token=camofox-secret',
        async close() {
          events.push('close')
        },
      }
    },
  }
  const provider = new CamofoxProvider({
    command: ['camofox-host', '--serve'],
    spawner,
    idFactory: () => 'camofox-session',
  })
  const session = await provider.open()

  expect(requests).toEqual([{ command: ['camofox-host', '--serve'], headless: true }])
  expect(JSON.stringify(session)).not.toContain('camofox-secret')
  expect(session.toRecord()).toMatchObject({
    provider: 'camofox',
    session_id: 'camofox-session',
    cdp_url_present: true,
  })

  await provider.close(session)
  expect(events).toEqual(['close'])
})

test('Browserbase and Browser Use make redacted injected HTTP requests and keep CDP credentials private', async () => {
  const browserbaseRequests: RecordedRequest[] = []
  const browserUseRequests: RecordedRequest[] = []
  const browserbaseFetcher: BrowserProviderFetcher = async (url, init) => {
    browserbaseRequests.push(recordedRequest(url, init))
    return Response.json({
      id: 'browserbase-host-id',
      connectUrl: 'wss://connect.browserbase.test?token=browserbase-secret',
    })
  }
  const browserUseFetcher: BrowserProviderFetcher = async (url, init) => {
    browserUseRequests.push(recordedRequest(url, init))
    return Response.json({ id: 'browser-use-host-id', cdp_url: 'wss://browser-use.test?token=browser-use-secret' })
  }
  const browserbase = new BrowserbaseProvider({
    config: { apiKey: 'browserbase-api-key', projectId: 'project-1' },
    fetcher: browserbaseFetcher,
    idFactory: () => 'browserbase-session',
  })
  const browserUse = new BrowserUseProvider({
    config: { apiKey: 'browser-use-api-key' },
    fetcher: browserUseFetcher,
    idFactory: () => 'browser-use-session',
  })

  const browserbaseSession = await browserbase.open()
  const browserUseSession = await browserUse.open({ headless: false })

  expect(browserbaseRequests).toEqual([{
    url: 'https://api.browserbase.com/v1/sessions',
    headers: expect.any(Headers),
    body: JSON.stringify({ projectId: 'project-1' }),
  }])
  expect(browserbaseRequests[0]?.headers.get('x-bb-api-key')).toBe('browserbase-api-key')
  expect(browserUseRequests).toEqual([{
    url: 'https://api.browser-use.com/v1/sessions',
    headers: expect.any(Headers),
    body: JSON.stringify({ headless: false }),
  }])
  expect(browserUseRequests[0]?.headers.get('authorization')).toBe('Bearer browser-use-api-key')
  expect(JSON.stringify(browserbaseSession)).not.toContain('browserbase-secret')
  expect(JSON.stringify(browserbaseSession)).not.toContain('browserbase-api-key')
  expect(JSON.stringify(browserUseSession)).not.toContain('browser-use-secret')
  expect(JSON.stringify(browserUseSession)).not.toContain('browser-use-api-key')
  expect(browserbaseSession.cdpUrlForHost()).toContain('browserbase-secret')
  expect(browserUseSession.cdpUrlForHost()).toContain('browser-use-secret')
})

test('Firecrawl is a configured request capability, not fabricated browser automation', async () => {
  const provider = new FirecrawlProvider({
    config: { apiKey: 'firecrawl-api-key' },
    idFactory: () => 'firecrawl-session',
  })
  const session = await provider.open()

  expect(session.toRecord()).toEqual({
    provider: 'firecrawl',
    session_id: 'firecrawl-session',
    kind: 'request',
    metadata: { headless: true },
    cdp_url_present: false,
  })
  expect(JSON.stringify(session)).not.toContain('firecrawl-api-key')
})

test('provider failures redact external response and metadata credentials', async () => {
  const provider = new BrowserbaseProvider({
    config: { apiKey: 'browserbase-api-key', projectId: 'project-1' },
    fetcher: async () => {
      throw new Error('request carried browserbase-api-key')
    },
  })
  await expect(provider.open()).rejects.toEqual(expect.objectContaining({
    name: ClientError.name,
    message: 'Client browserbase: session request failed',
  }))

  const session = new BrowserProviderSession({
    provider: 'local',
    sessionId: 'redacted-metadata',
    kind: 'local',
    metadata: { api_key: 'do-not-leak', token_count: 3, visible: true },
  })
  expect(session.toRecord().metadata).toEqual({ visible: true })
})

interface RecordedRequest {
  readonly body: string | null
  readonly headers: Headers
  readonly url: string
}

function recordedRequest(url: string, init: RequestInit): RecordedRequest {
  return {
    url,
    headers: new Headers(init.headers),
    body: typeof init.body === 'string' ? init.body : null,
  }
}
