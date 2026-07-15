// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  GoogleWorkspaceBridge,
  GoogleWorkspaceClient,
  GoogleWorkspaceOAuthClient,
  googleWorkspaceOAuthConfigFromClientSecret,
  googleWorkspaceRequestUrl,
  googleWorkspaceSetupGuidance,
  type GooglePendingAuthorization,
  type GoogleWorkspaceAuthorizationStorage,
  type GoogleWorkspaceEndpoints,
  type GoogleWorkspaceToken,
} from '../src/skills/googleWorkspace/index.js'
import type { SkillFetch } from '../src/skills/http.js'

const ENDPOINTS: GoogleWorkspaceEndpoints = {
  calendar: 'https://calendar.google.test/calendar/v3/',
  docs: 'https://docs.google.test/v1/',
  drive: 'https://drive.google.test/drive/v3/',
  gmail: 'https://gmail.google.test/gmail/v1/',
  people: 'https://people.google.test/v1/',
  sheets: 'https://sheets.google.test/v4/',
}

class MemoryGoogleStorage implements GoogleWorkspaceAuthorizationStorage {
  pending: GooglePendingAuthorization | undefined
  token: GoogleWorkspaceToken | undefined

  async loadPendingAuthorization(): Promise<GooglePendingAuthorization | undefined> {
    return this.pending
  }

  async loadToken(): Promise<GoogleWorkspaceToken | undefined> {
    return this.token
  }

  async removePendingAuthorization(): Promise<void> {
    this.pending = undefined
  }

  async removeToken(): Promise<void> {
    this.token = undefined
  }

  async savePendingAuthorization(pending: GooglePendingAuthorization): Promise<void> {
    this.pending = pending
  }

  async saveToken(token: GoogleWorkspaceToken): Promise<void> {
    this.token = token
  }
}

function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { headers: { 'Content-Type': 'application/json' }, status })
}

function workspaceClient(fetchImplementation: SkillFetch): GoogleWorkspaceClient {
  return new GoogleWorkspaceClient({
    endpoints: ENDPOINTS,
    fetchImplementation,
    tokenProvider: { accessToken: async () => 'fixture-bearer-token' },
  })
}

test('Google Workspace OAuth uses caller-owned PKCE storage and browser ports', async () => {
  const storage = new MemoryGoogleStorage()
  const opened: string[] = []
  const forms: URLSearchParams[] = []
  let sequence = 1
  const oauth = new GoogleWorkspaceOAuthClient({
    authorizationEndpoint: 'https://accounts.google.test/authorize',
    clientId: 'fixture-client-id',
    clientSecret: 'fixture-client-secret',
    redirectUri: 'http://127.0.0.1:4567/callback',
    scopes: ['scope.one', 'scope.two'],
    tokenEndpoint: 'https://accounts.google.test/token',
  }, {
    browser: { open: async url => { opened.push(url) } },
    fetchImplementation: async (_input, init) => {
      forms.push(new URLSearchParams(String(init?.body)))
      return json({ access_token: 'fixture-issued-token', expires_in: 3_600, refresh_token: 'fixture-refresh-token', scope: 'scope.one' })
    },
    now: () => 1_000,
    randomBytes: size => new Uint8Array(Array.from({ length: size }, () => sequence++)),
    storage,
  })

  const authorization = await oauth.beginAuthorization({ openBrowser: true })
  const authorizeUrl = new URL(authorization.url)
  expect(opened).toEqual([authorization.url])
  expect(authorizeUrl.searchParams.get('client_id')).toBe('fixture-client-id')
  expect(authorizeUrl.searchParams.get('code_challenge_method')).toBe('S256')
  expect(authorizeUrl.searchParams.get('scope')).toBe('scope.one scope.two')
  expect(storage.pending?.state).toBe(authorization.state)

  await oauth.completeAuthorization(`http://127.0.0.1:4567/callback?code=fixture-code&state=${encodeURIComponent(authorization.state)}`)
  expect(forms).toHaveLength(1)
  expect(forms[0]?.get('grant_type')).toBe('authorization_code')
  expect(forms[0]?.get('code_verifier')).toBe(authorization.codeVerifier)
  expect(storage.pending).toBeUndefined()
  expect((await oauth.authorizationStatus()).state).toBe('partial')
  expect((await oauth.authorizationStatus()).missingScopes).toEqual(['scope.two'])
})

test('Google OAuth refreshes expired credentials and revokes only through injected storage/fetch', async () => {
  const storage = new MemoryGoogleStorage()
  storage.token = {
    accessToken: 'fixture-expired-token',
    expiresAt: 10,
    refreshToken: 'fixture-refresh-token',
    scopes: ['scope.one'],
    tokenType: 'Bearer',
  }
  const requests: Array<{ readonly form: URLSearchParams; readonly url: URL }> = []
  const oauth = new GoogleWorkspaceOAuthClient({
    clientId: 'fixture-client-id',
    redirectUri: 'http://127.0.0.1:4567/callback',
    scopes: ['scope.one'],
    tokenEndpoint: 'https://accounts.google.test/token',
    revocationEndpoint: 'https://accounts.google.test/revoke',
  }, {
    fetchImplementation: async (input, init) => {
      const url = new URL(String(input))
      requests.push({ form: new URLSearchParams(String(init?.body)), url })
      if (url.pathname === '/token') return json({ access_token: 'fixture-refreshed-token', expires_in: 300 })
      return new Response(null, { status: 200 })
    },
    now: () => 100,
    storage,
  })

  expect(await oauth.accessToken()).toBeTruthy()
  expect(requests[0]?.form.get('grant_type')).toBe('refresh_token')
  expect(await oauth.revoke()).toBeTrue()
  expect(requests[1]?.url.pathname).toBe('/revoke')
  expect(storage.token).toBeUndefined()
})

test('Google REST client builds bounded URLs and maps Gmail metadata, bodies, and sends natively', async () => {
  const calls: Array<{ readonly body: string; readonly headers: HeadersInit | undefined; readonly method: string; readonly url: URL }> = []
  const client = workspaceClient(async (input, init) => {
    const url = new URL(String(input))
    calls.push({ body: String(init?.body ?? ''), headers: init?.headers, method: init?.method ?? 'GET', url })
    if (url.pathname.endsWith('/users/me/messages') && url.searchParams.has('q')) return json({ messages: [{ id: 'message-1' }] })
    if (url.pathname.endsWith('/users/me/messages/message-1') && url.searchParams.get('format') === 'metadata') {
      return json({
        id: 'message-1',
        labelIds: ['INBOX'],
        payload: { headers: [
          { name: 'From', value: 'sender@example.test' },
          { name: 'To', value: 'recipient@example.test' },
          { name: 'Subject', value: 'Status' },
          { name: 'Date', value: 'Mon, 1 Jan 2026 00:00:00 +0000' },
        ] },
        snippet: 'summary',
        threadId: 'thread-1',
      })
    }
    if (url.pathname.endsWith('/users/me/messages/message-1')) {
      return json({
        id: 'message-1',
        labelIds: ['INBOX'],
        payload: {
          headers: [{ name: 'Subject', value: 'Status' }],
          parts: [{ mimeType: 'multipart/alternative', parts: [{ body: { data: Buffer.from('native body').toString('base64url') }, mimeType: 'text/plain' }] }],
        },
        snippet: 'summary',
        threadId: 'thread-1',
      })
    }
    if (url.pathname.endsWith('/users/me/messages/send')) return json({ id: 'sent-1', threadId: 'thread-1' })
    throw new Error(`unexpected request ${url}`)
  })

  const summaries = await client.gmailSearch('is:unread', { maxResults: 3 })
  expect(summaries).toEqual([expect.objectContaining({ from: 'sender@example.test', subject: 'Status' })])
  const message = await client.gmailGet('message-1')
  expect(message.body).toBe('native body')
  const sent = await client.gmailSend({ body: 'Hello from Bun', subject: 'Native send', to: 'recipient@example.test' })
  expect(sent.status).toBe('sent')
  const sendCall = calls.at(-1)
  const raw = JSON.parse(sendCall?.body ?? '{}').raw as string
  expect(Buffer.from(raw, 'base64url').toString('utf8')).toContain('Subject: Native send')
  expect((sendCall?.headers as Record<string, string>).Authorization).toBe('Bearer fixture-bearer-token')
  await expect(client.gmailSend({ body: 'x', subject: 'bad\r\nBcc: injected', to: 'recipient@example.test' })).rejects.toThrow('newline')

  const url = googleWorkspaceRequestUrl(ENDPOINTS.gmail, 'users/me/messages', { metadataHeaders: ['From', 'Subject'], q: 'is:unread' })
  expect(url.searchParams.getAll('metadataHeaders')).toEqual(['From', 'Subject'])
  expect(() => googleWorkspaceRequestUrl(ENDPOINTS.gmail, 'https://unexpected.example.test/')).toThrow('relative path')
})

test('Google Workspace REST operations cover Calendar, Drive, People, Sheets, and Docs', async () => {
  const calls: URL[] = []
  const client = workspaceClient(async (input, init) => {
    const url = new URL(String(input))
    calls.push(url)
    if (url.hostname === 'calendar.google.test') {
      if (init?.method === 'POST') return json({ htmlLink: 'https://calendar.google.test/event', id: 'event-1', summary: 'Planning' })
      if (init?.method === 'DELETE') return new Response(null, { status: 204 })
      return json({ items: [{ end: { dateTime: '2026-01-02T11:00:00Z' }, id: 'event-1', start: { dateTime: '2026-01-02T10:00:00Z' }, summary: 'Planning' }] })
    }
    if (url.hostname === 'drive.google.test') return json({ files: [{ id: 'file-1', mimeType: 'text/plain', modifiedTime: '2026-01-01T00:00:00Z', name: 'Notes', webViewLink: 'https://drive.google.test/file-1' }] })
    if (url.hostname === 'people.google.test') return json({ connections: [{ emailAddresses: [{ value: 'person@example.test' }], names: [{ displayName: 'Person' }], phoneNumbers: [{ value: '+1-555-0100' }] }] })
    if (url.hostname === 'sheets.google.test') {
      if (url.pathname.endsWith(':append')) return json({ updates: { updatedCells: 2 } })
      if (init?.method === 'PUT') return json({ updatedCells: 2, updatedRange: 'Sheet1!A1:B1' })
      return json({ values: [['one', 2]] })
    }
    if (url.hostname === 'docs.google.test') {
      return json({ body: { content: [{ paragraph: { elements: [{ textRun: { content: 'First ' } }, { textRun: { content: 'second' } }] } }] }, documentId: 'doc-1', title: 'Doc' })
    }
    throw new Error(`unexpected request ${url}`)
  })

  const events = await client.calendarList({ end: '2026-01-03T00:00:00', start: '2026-01-01T00:00:00' })
  expect(events[0]?.summary).toBe('Planning')
  expect(await client.calendarCreate({ end: '2026-01-02T11:00:00', start: '2026-01-02T10:00:00', summary: 'Planning' })).toEqual(expect.objectContaining({ status: 'created' }))
  expect(await client.calendarDelete('event-1')).toEqual({ eventId: 'event-1', status: 'deleted' })
  expect((await client.driveSearch("O'Hara"))[0]?.name).toBe('Notes')
  expect(calls.find(url => url.hostname === 'drive.google.test')?.searchParams.get('q')).toBe("fullText contains 'O\\'Hara'")
  expect((await client.contactsList())[0]).toEqual({ emails: ['person@example.test'], name: 'Person', phones: ['+1-555-0100'] })
  expect(await client.sheetsGet('sheet-1', 'Sheet1!A1:B1')).toEqual([['one', 2]])
  expect(await client.sheetsUpdate('sheet-1', 'Sheet1!A1:B1', [['one', 2]])).toEqual({ updatedCells: 2, updatedRange: 'Sheet1!A1:B1' })
  expect(await client.sheetsAppend('sheet-1', 'Sheet1!A1', [['three', 4]])).toEqual({ updatedCells: 2 })
  expect(await client.docsGet('doc-1')).toEqual({ body: 'First second', documentId: 'doc-1', title: 'Doc' })
})

test('structured bridge dispatches native commands and setup guidance never needs ambient configuration', async () => {
  const client = workspaceClient(async (input) => {
    const url = new URL(String(input))
    if (url.pathname.endsWith('/users/me/labels')) return json({ labels: [{ id: 'INBOX', name: 'Inbox', type: 'system' }] })
    throw new Error(`unexpected request ${url}`)
  })
  const bridge = new GoogleWorkspaceBridge({ client })
  const labels = await bridge.dispatch({ action: 'labels', service: 'gmail' })
  expect(labels).toEqual([{ id: 'INBOX', name: 'Inbox', type: 'system' }])
  expect((await bridge.dispatch({ action: 'guidance', service: 'setup' }) as ReturnType<typeof googleWorkspaceSetupGuidance>).security).toContain('does not discover')
  await expect(bridge.dispatch({ action: 'status', service: 'setup' })).rejects.toThrow('inject an explicit')

  const config = googleWorkspaceOAuthConfigFromClientSecret({
    installed: {
      auth_uri: 'https://accounts.google.test/authorize',
      client_id: 'fixture-client-id',
      client_secret: 'fixture-client-secret',
      token_uri: 'https://accounts.google.test/token',
    },
  }, { redirectUri: 'http://127.0.0.1:4567/callback' })
  expect(config.clientId).toBe('fixture-client-id')
  expect(googleWorkspaceSetupGuidance().steps).toHaveLength(5)
})
