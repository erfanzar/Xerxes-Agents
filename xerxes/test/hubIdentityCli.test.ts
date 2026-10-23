// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'

import { parseDuration, runHubIdentityCli } from '../src/hub/identity/cli.js'
import { cleanupAll, tempPaths } from './hubIdentityHarness.js'

afterEach(cleanupAll)

async function cli(paths: ReturnType<typeof tempPaths>['paths'], args: string[]): Promise<{ code: number; out: string[]; err: string[] }> {
  const out: string[] = []
  const err: string[] = []
  const code = await runHubIdentityCli(args, {
    paths,
    environment: { XERXES_HUB_PUBLIC_ORIGINS: 'https://hub.test' },
    out: line => out.push(line),
    err: line => err.push(line),
  })
  return { code, out, err }
}

describe('hub identity CLI', () => {
  test('owner init, pair, devices, audit verify', async () => {
    const { paths } = tempPaths()
    expect((await cli(paths, ['owner', 'init', '--name', 'Erfan'])).code).toBe(0)
    const again = await cli(paths, ['owner', 'init', '--name', 'Other'])
    expect(again.code).toBe(1)
    expect(again.err.join('\n')).toContain('owner already exists')

    const pair = await cli(paths, ['pair', '--label', 'Pixel', '--no-confirm'])
    expect(pair.code).toBe(0)
    const text = pair.out.join('\n')
    expect(text).toContain('https://hub.test/pair#')
    expect(text).toMatch(/comparison code: \d{6}/)
    expect(text).toContain('no confirmation required')

    const confirmable = await cli(paths, ['pair'])
    expect(confirmable.out.join('\n')).toMatch(/pair confirm pr_\S+ \d{6}/)

    expect((await cli(paths, ['devices', 'list'])).code).toBe(0)
    const verify = await cli(paths, ['audit', 'verify'])
    expect(verify.code).toBe(0)
    expect(verify.out.join('\n')).toContain('audit chain ok')
  })

  test('invite create validates flags', async () => {
    const { paths } = tempPaths()
    await cli(paths, ['owner', 'init', '--name', 'Erfan'])
    const created = await cli(paths, ['invite', 'create', '--workspace', 'w1', '--session', 's1', '--role', 'viewer', '--expires', '2h'])
    expect(created.code).toBe(0)
    expect(created.out.join('\n')).toContain('/invite#')
    expect((await cli(paths, ['invite', 'create', '--workspace', 'w1', '--role', 'owner'])).code).toBe(1)
    expect((await cli(paths, ['invite', 'create', '--workspace', 'w1'])).code).toBe(2)
    expect((await cli(paths, ['pair', '--role', 'god'])).code).toBe(2)
  })

  test('usage errors', async () => {
    const { paths } = tempPaths()
    expect((await cli(paths, [])).code).toBe(2)
    expect((await cli(paths, ['bogus'])).code).toBe(2)
    expect((await cli(paths, ['owner', 'init'])).code).toBe(2)
    expect((await cli(paths, ['pair'])).code).toBe(1) // no owner yet
  })

  test('parseDuration', () => {
    expect(parseDuration('30m')).toBe(1_800_000)
    expect(parseDuration('24h')).toBe(86_400_000)
    expect(() => parseDuration('soon')).toThrow()
  })
})
