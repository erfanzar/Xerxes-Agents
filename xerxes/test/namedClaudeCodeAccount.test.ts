// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { ProviderAccountRegistry } from '../src/auth/providerAccountRegistry.js'
import { ClaudeCodeClient, type ClaudeCodeLauncher } from '../src/llms/claudeCode.js'
import { createLlmClient } from '../src/llms/client.js'

test('named Claude Code keeps toolchain paths but uses only its own token and identity', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-named-cli-'))
  try {
    const root = join(directory, 'accounts')
    const registry = new ProviderAccountRegistry(root, join(directory, 'profiles.json'), { PATH: '/fixture/toolchain/bin', TMPDIR: '/fixture/tmp', CLAUDE_CODE_CLI: '/fixture/claude', CLAUDE_CODE_OAUTH_TOKEN: 'fixture-ambient', ANTHROPIC_API_KEY: 'fixture-other', CLAUDE_CODE_USE_BEDROCK: '1', CLAUDE_CODE_USE_VERTEX: '1' })
    const fingerprints: Array<string | undefined> = []
    for (const label of ['First', 'Second']) {
      const account = await registry.create('claude-code', label)
      await mkdir(join(root, account.id, 'auth'), { recursive: true })
      await writeFile(join(root, account.id, 'auth', 'anthropic-oauth.json'), JSON.stringify({ access: `fixture-${label}`, refresh: 'fixture-refresh', expires: 4102444800 }))
      const options = registry.sessionOptions(account.id)
      expect(options.claudeEnvironment?.PATH).toBe('/fixture/toolchain/bin')
      expect(options.claudeEnvironment?.TMPDIR).toBe('/fixture/tmp')
      expect(options.claudeEnvironment?.CLAUDE_CODE_OAUTH_TOKEN).toBeUndefined()
      expect(options.claudeEnvironment?.ANTHROPIC_API_KEY).toBeUndefined()
      const launches: Record<string, string>[] = []
      const launch: ClaudeCodeLauncher = (_argv, spawn) => {
        launches.push(spawn.env)
        return { lines: (async function* () { yield JSON.stringify({ type: 'result', subtype: 'success', is_error: false, result: 'done', usage: { input_tokens: 1, output_tokens: 1 } }) })(), exited: Promise.resolve(0), stderr: Promise.resolve(''), kill: () => {} }
      }
      const client = new ClaudeCodeClient({ environment: options.claudeEnvironment, isolatedAuth: true, subscriptionSession: options.anthropicOAuthSession, executable: '/fixture/claude', workingDirectory: directory, launch })
      fingerprints.push(await client.authFingerprint())
      for await (const _delta of client.stream({ model: 'sonnet', messages: [{ role: 'user', content: 'fixture' }], toolChoice: 'none' })) { /* offline injected launcher */ }
      expect(launches).toHaveLength(1)
      expect(launches[0]?.CLAUDE_CODE_OAUTH_TOKEN).toBe(`fixture-${label}`)
      expect(launches[0]?.CLAUDE_CONFIG_DIR).toBe(join(root, account.id))
      expect(launches[0]?.CLAUDE_CODE_USE_BEDROCK).toBeUndefined()
      expect(launches[0]?.CLAUDE_CODE_USE_VERTEX).toBeUndefined()
      const routed = createLlmClient('sonnet', { provider: 'claude-code' }, options)
      expect(await routed.authFingerprint?.()).toBe(fingerprints.at(-1))
    }
    expect(fingerprints.every(value => typeof value === 'string' && value.length > 0)).toBe(true)
    expect(fingerprints[0]).not.toBe(fingerprints[1])
    const empty = await registry.create('claude-code', 'Empty')
    const options = registry.sessionOptions(empty.id)
    const client = new ClaudeCodeClient({ environment: options.claudeEnvironment, isolatedAuth: true, subscriptionSession: options.anthropicOAuthSession, executable: '/fixture/claude', workingDirectory: directory, launch: () => { throw new Error('Must not launch') } })
    const run = async () => { for await (const _delta of client.stream({ model: 'sonnet', messages: [{ role: 'user', content: 'fixture' }], toolChoice: 'none' })) { /* no output */ } }
    await expect(run()).rejects.toThrow('No Claude Code sign-in found')
  } finally { await rm(directory, { recursive: true, force: true }) }
})
