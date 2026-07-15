// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import {
  CredentialFileRegistry,
  CredentialPathError,
  selectCredentialEnvironment,
  syncPull,
  syncPush,
} from '../src/security/index.js'
import type { FileSyncCopyRequest, FileSyncPorts, FileSyncStatRequest } from '../src/security/index.js'

test('credential files stay inside caller-supplied roots and environment selection is explicit', () => {
  const registry = new CredentialFileRegistry({
    allowedRoots: ['/host/credentials', '/home/agent/.config'],
    baseDirectory: '/host/credentials',
    homeDirectory: '/home/agent',
  })

  expect(registry.register('aws/credentials')).toBe('/host/credentials/aws/credentials')
  expect(registry.register('~/.config/service/token')).toBe('/home/agent/.config/service/token')
  expect(registry.allowedPaths()).toEqual([
    '/home/agent/.config/service/token',
    '/host/credentials/aws/credentials',
  ])
  expect(registry.isAllowed('/host/credentials/aws/credentials')).toBeTrue()
  expect(registry.isAllowed('../outside')).toBeFalse()
  expect(registry.unregister('aws/credentials')).toBeTrue()
  expect(registry.isAllowed('aws/credentials')).toBeFalse()
  expect(() => registry.register('/etc/shadow')).toThrow(CredentialPathError)
  expect(() => new CredentialFileRegistry({
    allowedRoots: ['/host/credentials'],
    baseDirectory: '/host/credentials',
  }).register('~/secret')).toThrow(CredentialPathError)

  const inheritedEnvironment = Object.create({ INHERITED_TOKEN: 'must-not-leak' }) as Record<string, string>
  inheritedEnvironment.EXPLICIT_TOKEN = 'visible'
  const selected = selectCredentialEnvironment(
    ['EXPLICIT_TOKEN', 'INHERITED_TOKEN', 'MISSING_TOKEN'],
    inheritedEnvironment,
  )
  expect(selected).toEqual({ EXPLICIT_TOKEN: 'visible' })
  expect(Object.isFrozen(selected)).toBeTrue()

  registry.clear()
  expect(registry.allowedPaths()).toEqual([])
})

test('push sync reports byte caps, missing, failed, and escaped files without aborting the batch', async () => {
  const copyRequests: FileSyncCopyRequest[] = []
  const statRequests: FileSyncStatRequest[] = []
  const sizes = new Map<string, number>([
    ['/host/workspace/small.txt', 4],
    ['/host/workspace/big.bin', 32],
    ['/host/workspace/broken.txt', 5],
  ])
  const ports: FileSyncPorts = {
    copy: async request => {
      copyRequests.push(request)
      if (request.source.endsWith('/broken.txt')) {
        throw new Error('transfer unavailable')
      }
    },
    stat: async request => {
      statRequests.push(request)
      const size = sizes.get(request.path)
      return size === undefined ? undefined : { size }
    },
  }

  const results = await syncPush([
    { localPath: 'small.txt', remotePath: 'input/small.txt', metadata: { request: 'small' } },
    { localPath: 'big.bin', remotePath: 'input/big.bin' },
    { localPath: 'missing.txt', remotePath: 'input/missing.txt' },
    { localPath: 'broken.txt', remotePath: 'input/broken.txt' },
    { localPath: '../secret.txt', remotePath: 'input/secret.txt' },
    { localPath: 'small.txt', remotePath: '/outside/secret.txt' },
  ], ports, {
    localRoot: '/host/workspace',
    maxBytes: 10,
    remoteRoot: '/sandbox/workspace',
  })

  expect(results).toEqual([
    {
      bytes: 4,
      direction: 'push',
      localPath: '/host/workspace/small.txt',
      metadata: { request: 'small' },
      remotePath: '/sandbox/workspace/input/small.txt',
      status: 'copied',
    },
    {
      bytes: 32,
      direction: 'push',
      localPath: '/host/workspace/big.bin',
      metadata: {},
      reason: 'max_bytes_exceeded',
      remotePath: '/sandbox/workspace/input/big.bin',
      status: 'skipped',
    },
    {
      bytes: 0,
      direction: 'push',
      localPath: '/host/workspace/missing.txt',
      metadata: {},
      reason: 'missing',
      remotePath: '/sandbox/workspace/input/missing.txt',
      status: 'skipped',
    },
    {
      direction: 'push',
      error: expect.stringContaining('copy failed: transfer unavailable'),
      localPath: '/host/workspace/broken.txt',
      metadata: {},
      remotePath: '/sandbox/workspace/input/broken.txt',
      status: 'failed',
    },
    {
      direction: 'push',
      error: expect.stringContaining('path validation failed'),
      localPath: '../secret.txt',
      metadata: {},
      remotePath: 'input/secret.txt',
      status: 'failed',
    },
    {
      direction: 'push',
      error: expect.stringContaining('path validation failed'),
      localPath: 'small.txt',
      metadata: {},
      remotePath: '/outside/secret.txt',
      status: 'failed',
    },
  ])
  expect(statRequests.map(request => request.path)).toEqual([
    '/host/workspace/small.txt',
    '/host/workspace/big.bin',
    '/host/workspace/missing.txt',
    '/host/workspace/broken.txt',
  ])
  expect(copyRequests).toEqual([
    {
      destination: '/sandbox/workspace/input/small.txt',
      destinationLocation: 'remote',
      direction: 'push',
      source: '/host/workspace/small.txt',
      sourceLocation: 'local',
    },
    {
      destination: '/sandbox/workspace/input/broken.txt',
      destinationLocation: 'remote',
      direction: 'push',
      source: '/host/workspace/broken.txt',
      sourceLocation: 'local',
    },
  ])
})

test('pull sync uses remote source paths and continues after a transfer failure', async () => {
  const copyRequests: FileSyncCopyRequest[] = []
  const ports: FileSyncPorts = {
    copy: async request => {
      copyRequests.push(request)
      if (request.source.endsWith('/bad.json')) {
        throw new Error('remote read denied')
      }
    },
    stat: async request => {
      const sizes: Readonly<Record<string, number>> = {
        '/sandbox/workspace/exports/bad.json': 3,
        '/sandbox/workspace/exports/good.json': 7,
        '/sandbox/workspace/exports/large.json': 20,
      }
      const size = sizes[request.path]
      return size === undefined ? undefined : { size }
    },
  }

  const results = await syncPull([
    { localPath: 'downloads/good.json', remotePath: 'exports/good.json' },
    { localPath: 'downloads/bad.json', remotePath: 'exports/bad.json' },
    { localPath: 'downloads/large.json', remotePath: 'exports/large.json' },
  ], ports, {
    localRoot: '/host/output',
    maxBytes: 10,
    remoteRoot: '/sandbox/workspace',
  })

  expect(results.map(result => result.status)).toEqual(['copied', 'failed', 'skipped'])
  expect(results[1]).toMatchObject({ error: expect.stringContaining('copy failed: remote read denied') })
  expect(results[2]).toMatchObject({ bytes: 20, reason: 'max_bytes_exceeded' })
  expect(copyRequests).toEqual([
    {
      destination: '/host/output/downloads/good.json',
      destinationLocation: 'local',
      direction: 'pull',
      source: '/sandbox/workspace/exports/good.json',
      sourceLocation: 'remote',
    },
    {
      destination: '/host/output/downloads/bad.json',
      destinationLocation: 'local',
      direction: 'pull',
      source: '/sandbox/workspace/exports/bad.json',
      sourceLocation: 'remote',
    },
  ])
})
