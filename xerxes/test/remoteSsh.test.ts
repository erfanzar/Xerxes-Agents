// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { parseRemoteTarget, RemoteHostRegistry } from '../src/remote/ssh.js'

describe('parseRemoteTarget', () => {
  test('parses user@host', () => {
    expect(parseRemoteTarget('alice@example.com')).toEqual({
      alias: 'alice@example.com',
      host: 'example.com',
      user: 'alice',
      workspacePath: '~',
    })
  })

  test('parses user@host:port', () => {
    expect(parseRemoteTarget('alice@example.com:2222')).toEqual({
      alias: 'alice@example.com:2222',
      host: 'example.com',
      port: 2222,
      user: 'alice',
      workspacePath: '~',
    })
  })

  test('parses user@host/path', () => {
    expect(parseRemoteTarget('alice@example.com/code/project')).toEqual({
      alias: 'alice@example.com',
      host: 'example.com',
      user: 'alice',
      workspacePath: 'code/project',
    })
  })

  test('parses user@host:port/path', () => {
    expect(parseRemoteTarget('alice@example.com:2222/code/project')).toEqual({
      alias: 'alice@example.com:2222',
      host: 'example.com',
      port: 2222,
      user: 'alice',
      workspacePath: 'code/project',
    })
  })

  test('rejects invalid format', () => {
    expect(() => parseRemoteTarget('not-a-target')).toThrow('invalid format')
  })
})

describe('RemoteHostRegistry', () => {
  test('adds, lists, and removes hosts', async () => {
    const registry = new RemoteHostRegistry('/tmp/xerxes-remote-test.json')
    await registry.load()

    registry.add({
      alias: 'dev',
      host: 'dev.example.com',
      user: 'alice',
      workspacePath: '/code',
    })

    expect(registry.get('dev')).toEqual({
      alias: 'dev',
      host: 'dev.example.com',
      user: 'alice',
      workspacePath: '/code',
    })
    expect(registry.list()).toHaveLength(1)

    expect(registry.remove('dev')).toBe(true)
    expect(registry.list()).toHaveLength(0)
  })
})
