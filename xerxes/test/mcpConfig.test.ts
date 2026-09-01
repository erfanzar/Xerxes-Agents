// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { loadMcpConfig } from '../src/mcp/config.js'

describe('loadMcpConfig', () => {
  const cleanup: string[] = []
  afterEach(async () => {
    for (const dir of cleanup.splice(0)) await rm(dir, { recursive: true, force: true })
  })

  const tempConfig = async (content: string | null): Promise<string> => {
    const dir = await mkdtemp(join(tmpdir(), 'xerxes-mcp-config-'))
    cleanup.push(dir)
    const path = join(dir, 'mcp.json')
    if (content !== null) await writeFile(path, content)
    return path
  }

  test('a missing file means no servers and no warnings', async () => {
    const path = await tempConfig(null)
    expect(loadMcpConfig(path)).toEqual({ servers: [], warnings: [] })
  })

  test('the list shape loads valid servers and skips disabled ones', async () => {
    const path = await tempConfig(JSON.stringify({
      servers: [
        { name: 'filesystem', command: 'npx', args: ['-y', 'mcp-fs'] },
        { name: 'github', url: 'https://mcp.example.com', transport: 'streamable_http' },
        { name: 'off', command: 'noop', enabled: false },
      ],
    }))
    const config = loadMcpConfig(path)
    expect(config.servers.map(server => server.name)).toEqual(['filesystem', 'github'])
    expect(config.warnings).toEqual([])
  })

  test('the map shape is accepted with names folded in', async () => {
    const path = await tempConfig(JSON.stringify({
      filesystem: { command: 'npx' },
    }))
    const config = loadMcpConfig(path)
    expect(config.servers).toHaveLength(1)
    expect(config.servers[0]?.name).toBe('filesystem')
  })

  test('invalid entries become warnings; valid siblings still load', async () => {
    const path = await tempConfig(JSON.stringify({
      servers: [
        { command: 'npx' },
        { name: 'broken' },
        { name: 'good', command: 'noop' },
        'not an object',
      ],
    }))
    const config = loadMcpConfig(path)
    expect(config.servers.map(server => server.name)).toEqual(['good'])
    expect(config.warnings).toHaveLength(3)
    expect(config.warnings.some(warning => warning.includes("'broken'"))).toBe(true)
  })

  test('malformed JSON is one actionable warning, not a crash', async () => {
    const path = await tempConfig('{ nope')
    const config = loadMcpConfig(path)
    expect(config.servers).toEqual([])
    expect(config.warnings[0]).toContain('not valid JSON')
  })
})
