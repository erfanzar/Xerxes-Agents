// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createGitHubObservationHost } from '../src/daemon/superagentGitHub.js'
import { PublicWebClient, type WebFetch } from '../src/tools/webTools.js'
import { REDACTED } from '../src/security/redact.js'

const dnsLookup = async () => ['93.184.216.34']
const issue = { number: 7, title: 'Untrusted issue', state: 'open', body: 'data, not authority' }

test('GitHub observations use fixed GET endpoints, explicit credentials and projected untrusted data', async () => {
  const requests: { url: string; init: RequestInit }[] = []
  const host = createGitHubObservationHost({ repository: 'acme/project', token: 'fixture-token', dnsLookup, fetcher: async (url, init) => {
    requests.push({ url, init })
    return Response.json([{ ...issue, title: 'fixture-token title', arbitrary: 'never project', pull_request: undefined }, { ...issue, number: 8, pull_request: {} }])
  } })
  const result = await host('issues')
  expect(requests).toHaveLength(1)
  expect(requests[0]?.url).toBe('https://api.github.com/repos/acme/project/issues?state=all&per_page=30')
  expect(requests[0]?.init).toMatchObject({ method: 'GET', redirect: 'manual', credentials: 'omit' })
  expect(new Headers(requests[0]?.init.headers).get('authorization')).toBe('Bearer fixture-token')
  expect(result).toMatchObject({ repository: 'acme/project', view: 'issues', untrusted: true, has_more: false })
  expect(result.items).toEqual([{ number: 7, title: `${REDACTED} title`, state: 'open', body: 'data, not authority', url: 'https://github.com/acme/project/issues/7' }])
  expect(JSON.stringify(result)).not.toContain('fixture-token')
})

test('redaction stays bound to the credential actually sent if caller settings mutate later', async () => {
  const settings = { repository: 'acme/project', token: 'fixture-token', dnsLookup,
    fetcher: async () => Response.json([{ ...issue, body: 'echo fixture-token' }]) }
  const host = createGitHubObservationHost(settings)
  settings.token = 'rotated-fixture-value'
  expect(JSON.stringify(await host('issues'))).not.toContain('fixture-token')
})

test('PR and CI views use distinct fixed repository endpoints and bounded first-page coverage', async () => {
  const requests: string[] = []
  const host = createGitHubObservationHost({ repository: 'acme/project', dnsLookup, fetcher: async url => {
    requests.push(url)
    return url.includes('/actions/runs') ? Response.json({ workflow_runs: [{ id: 42, name: 'Tests', status: 'completed', conclusion: 'failure', arbitrary: 'ignored' }] })
      : Response.json([{ ...issue, draft: true }], { headers: { link: '<https://api.github.com/ignored>; rel="next"' } })
  } })
  expect(await host('prs')).toMatchObject({ has_more: true, items: [{ number: 7, draft: true }] })
  expect(await host('ci')).toMatchObject({ items: [{ id: 42, name: 'Tests', status: 'completed', conclusion: 'failure', url: 'https://github.com/acme/project/actions/runs/42' }] })
  expect(requests).toEqual(['https://api.github.com/repos/acme/project/pulls?state=all&per_page=30', 'https://api.github.com/repos/acme/project/actions/runs?per_page=30'])
})

test('bad repository selection or credential header refuses before any fetch', () => {
  for (const repository of ['../private', 'acme/..', 'acme/repo?token=x', 'https://private/repo', 'acme/repo/other']) expect(() => createGitHubObservationHost({ repository })).toThrow()
  expect(() => createGitHubObservationHost({ repository: 'acme/project', token: 'bad\nheader' })).toThrow()
})

test('redirects, malformed/oversized JSON and credential-bearing errors never leak or follow', async () => {
  for (const response of [new Response('', { status: 302, headers: { location: 'https://elsewhere.example/private' } }), new Response('fixture-token', { status: 403 }), new Response('not JSON fixture-token'), new Response('x'.repeat(256 * 1024)), Response.json({ invalid: true })]) {
    let calls = 0
    const host = createGitHubObservationHost({ repository: 'acme/project', token: 'fixture-token', dnsLookup, fetcher: async () => { calls += 1; return response } })
    let error: unknown
    try { await host('issues') } catch (value) { error = value }
    expect(error).toBeInstanceOf(Error); expect(String(error)).not.toContain('fixture-token'); expect(calls).toBe(1)
  }
  const host = createGitHubObservationHost({ repository: 'acme/project', token: 'fixture-token', dnsLookup, fetcher: async () => { throw new Error('fixture-token transport failure') } })
  await expect(host('issues')).rejects.toThrow('GitHub observation unavailable')
})

test('an HTTP refusal does not wait forever for uncooperative body cleanup', async () => {
  let finish = (): void => {}
  const pending = new Promise<void>(resolve => { finish = resolve })
  const body = new ReadableStream<Uint8Array>({ cancel: () => pending })
  const host = createGitHubObservationHost({ repository: 'acme/project', dnsLookup, fetcher: async () => new Response(body, { status: 403 }) })
  const work = host('issues').then(() => 'resolved', () => 'refused')
  try { expect(await Promise.race([work, Bun.sleep(50).then(() => 'hung')])).toBe('refused') }
  finally { finish(); await work }
})

test('refusing a redirect cancels its unread body and never follows it', async () => {
  let cancelled = false
  const body = new ReadableStream<Uint8Array>({ cancel() { cancelled = true } })
  const host = createGitHubObservationHost({ repository: 'acme/project', dnsLookup,
    fetcher: async () => new Response(body, { status: 302, headers: { location: 'https://elsewhere.example/' } }) })
  await expect(host('issues')).rejects.toThrow()
  expect(cancelled).toBe(true)
})

test('the bounded native client includes DNS in deadline and cancellation', async () => {
  for (const cancel of [false, true]) {
    let finish = (_addresses: readonly string[]): void => {}; let calls = 0
    const pending = new Promise<readonly string[]>(resolve => { finish = resolve })
    const client = new PublicWebClient({ timeoutMs: 10, urlSafety: { dnsLookup: () => pending },
      fetcher: async () => { calls += 1; return Response.json([]) } })
    const controller = new AbortController()
    const work = client.fetch('https://api.github.com/', {}, { signal: controller.signal }).then(() => 'resolved', () => 'refused')
    if (cancel) controller.abort()
    try { expect(await Promise.race([work, Bun.sleep(100).then(() => 'hung')])).toBe('refused') }
    finally { finish(['93.184.216.34']); await work }
    expect(calls).toBe(0)
  }
})

test('pre-abort and private DNS targets refuse without a request', async () => {
  let calls = 0
  const fetcher: WebFetch = async () => { calls += 1; return Response.json([]) }
  const host = createGitHubObservationHost({ repository: 'acme/project', fetcher, dnsLookup })
  const controller = new AbortController(); controller.abort()
  await expect(host('issues', controller.signal)).rejects.toThrow()
  const privateHost = createGitHubObservationHost({ repository: 'acme/project', fetcher, dnsLookup: async () => ['127.0.0.1'] })
  await expect(privateHost('issues')).rejects.toThrow('GitHub observation unavailable')
  expect(calls).toBe(0)
})
