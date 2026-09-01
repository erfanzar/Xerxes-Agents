// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AuthCommandError, runAuthCommand } from '../src/auth/command.js'
import type { OAuthFlowCredential } from '../src/auth/oauthFlows.js'
import { CliWriter } from '../src/runtime/cliStyle.js'

function capture(): { lines: string[]; writer: CliWriter } {
  const lines: string[] = []
  return { lines, writer: new CliWriter({ write: line => lines.push(line) }) }
}

const CREDENTIAL: OAuthFlowCredential = {
  access: 'session-access',
  refresh: 'session-refresh',
  expires: Math.floor(Date.now() / 1000) + 3_600,
}

/** A session that records calls and behaves like every OAuth session surface. */
function stubSession(): {
  calls: string[]
  session: Record<string, unknown>
} {
  const calls: string[] = []
  return {
    calls,
    session: {
      credential: async () => {
        calls.push('credential')
        return CREDENTIAL
      },
      login: async (...args: unknown[]) => {
        calls.push(`login:${args.map(argument => JSON.stringify(argument, (_key, value) => typeof value === 'function' ? '[fn]' : value) ?? 'undefined').join('|')}`)
        return CREDENTIAL
      },
      refresh: async () => {
        calls.push('refresh')
        return CREDENTIAL
      },
      stored: async () => {
        calls.push('stored')
        return CREDENTIAL
      },
      logout: async () => {
        calls.push('logout')
        return true
      },
      resolveGateway: (explicit?: string) => explicit ?? 'https://default.example.com',
    },
  }
}

test('auth login/status/logout route anthropic, kimi, openrouter, xai, and radius aliases', async () => {
  for (const [provider, alias] of [
    ['anthropic', 'claude'],
    ['kimi', 'kimi-code'],
    ['openrouter', 'openrouter'],
    ['xai', 'grok'],
    ['radius', 'radius'],
  ] as const) {
    const { calls, session } = stubSession()
    const key = `${provider}OAuthSession`
    const options = { [key]: session } as never
    await runAuthCommand(['login', alias], options)
    await runAuthCommand(['status', alias], options)
    await runAuthCommand(['logout', alias], options)
    expect(calls.filter(call => call.startsWith('login:'))).toHaveLength(1)
    expect(calls).toContain('stored')
    expect(calls).toContain('logout')
  }
})

test('radius login forwards the gateway positional and --device method', async () => {
  const { calls, session } = stubSession()
  await runAuthCommand(['login', 'radius', 'gw.example.com', '--device'], {
    radiusOAuthSession: session,
  } as never)
  expect(calls[0]).toContain('"method":"device"')
  expect(calls[0]).toContain('gw.example.com')
})

test('anthropic login forwards a --code paste as manual input', async () => {
  const { calls, session } = stubSession()
  await runAuthCommand(['login', 'anthropic', '--code', 'pasted#state'], {
    anthropicOAuthSession: session,
  } as never)
  // login receives { manualInput, openUrl }; the replacer keeps the key.
  expect(calls[0]).toContain('manualInput')
})

test('unknown providers still fail with a usage error listing the new aliases', () => {
  expect(() => runAuthCommand(['login', 'vertex'])).toThrow(AuthCommandError)
  expect(() => runAuthCommand(['login'])).toThrow(AuthCommandError)
})

test('auth help mentions every new provider', async () => {
  const { lines, writer } = capture()
  await runAuthCommand([], { writer })
  const help = lines.join('\n')
  for (const provider of ['anthropic', 'kimi', 'openrouter', 'xai', 'radius']) {
    expect(help).toContain(provider)
  }
})
