// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, readFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { expect, test } from 'bun:test'

import { checkPackage, isBlocked, parseResponse, type OSVFetch } from '../src/mcp/osv.js'

test('OSV checks post the requested package tuple and classify high-severity advisories', async () => {
  let body: unknown
  const fetchImplementation: OSVFetch = async (_input, init) => {
    body = JSON.parse(String(init?.body)) as unknown
    return Response.json({
      vulns: [
        { id: 'GHSA-example', summary: 'Example', aliases: ['CVE-2026-1'], database_specific: { severity: 'HIGH' } },
        { id: 'MAL-2026-1', summary: 'Malware' },
      ],
    })
  }

  const vulnerabilities = await checkPackage('npm', 'example-package', '1.2.3', { fetchImplementation })
  expect(body).toEqual({ package: { ecosystem: 'npm', name: 'example-package' }, version: '1.2.3' })
  expect(vulnerabilities).toEqual([
    { id: 'GHSA-example', summary: 'Example', severity: 'HIGH', aliases: ['CVE-2026-1'] },
    { id: 'MAL-2026-1', summary: 'Malware', severity: '', aliases: [] },
  ])
  expect(isBlocked(vulnerabilities)).toBeTrue()
  expect(isBlocked([{ id: 'GHSA-low', summary: '', severity: 'LOW', aliases: [] }])).toBeFalse()
})

test('OSV cache is fresh only for its configured TTL and is written as JSON', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-osv-'))
  try {
    let requests = 0
    const fetchImplementation: OSVFetch = async () => {
      requests += 1
      return Response.json({ vulns: [{ id: 'GHSA-cached', summary: 'Cached', database_specific: { severity: 'MEDIUM' } }] })
    }
    let now = 1_000
    const options = { cacheDirectory: directory, cacheTtlMs: 10, fetchImplementation, now: () => now }

    await checkPackage('npm', 'cached', undefined, options)
    now = 1_009
    const fromCache = await checkPackage('npm', 'cached', undefined, options)
    now = 1_010
    await checkPackage('npm', 'cached', undefined, options)

    expect(fromCache[0]?.id).toBe('GHSA-cached')
    expect(requests).toBe(2)
    expect(JSON.parse(await readFile(join(directory, 'osv_cache.json'), 'utf8'))).toMatchObject({
      'npm::cached::': { fetchedAt: 1_010 },
    })
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('OSV network or malformed responses fail open without exposing invalid records', async () => {
  const failed: OSVFetch = async () => { throw new Error('network unavailable') }
  const malformed: OSVFetch = async () => Response.json({ vulns: [{ id: 42, aliases: ['CVE', 3] }] })

  await expect(checkPackage('npm', 'offline', undefined, { fetchImplementation: failed })).resolves.toEqual([])
  await expect(checkPackage('npm', 'malformed', undefined, { fetchImplementation: malformed })).resolves.toEqual([
    { id: '', summary: '', severity: '', aliases: ['CVE'] },
  ])
  expect(parseResponse({ vulns: 'not an array' })).toEqual([])
  await expect(checkPackage('npm', ' ', undefined)).rejects.toThrow('name must not be empty')
})
