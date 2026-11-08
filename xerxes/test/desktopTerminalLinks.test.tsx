// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { openTerminalLink } from '../src/desktop/renderer/terminalLinks.js'

// Synthetic OAuth parameters only: no live state, codes, or credentials.
const oauth = 'https://auth.openai.com/oauth/authorize?response_type=code&client_id=test-client&redirect_uri=http%3A%2F%2F127.0.0.1%3A1455%2Fauth%2Fcallback&code_challenge=test-challenge&code_challenge_method=S256&state=test-state&scope=openid+profile+offline_access&originator=codex-tui'

test('OK hands the actual OAuth URL to the host unchanged, never a blank popup', async () => {
  const opened: string[] = []
  const prompts: string[] = []
  const error = await openTerminalLink(oauth, message => { prompts.push(message); return true }, {
    openExternal: async url => { opened.push(url); return true },
  })
  expect(error).toBeNull()
  expect(prompts).toHaveLength(1)
  expect(prompts[0]).toContain(oauth)
  expect(opened).toEqual([oauth])
})

test('Cancel never calls the browser bridge', async () => {
  let opens = 0
  expect(await openTerminalLink(oauth, () => false, { openExternal: async () => { opens += 1; return true } })).toBeNull()
  expect(opens).toBe(0)
})

test('unsafe, malformed, and oversized terminal links never prompt or reach the host', async () => {
  let prompts = 0
  let opens = 0
  for (const uri of ['javascript:alert(1)', 'data:text/html,hi', 'file:///tmp/private.txt', 'mailto:user@example.com', 'ftp://example.com/', 'vscode://file/tmp/file', 'about:blank', '//example.com/', '', 'https://example.com/\nnext', 'https://example.com/' + 'x'.repeat(2048)]) {
    expect(await openTerminalLink(uri, () => { prompts += 1; return true }, { openExternal: async () => { opens += 1; return true } })).toContain('Only valid HTTP and HTTPS')
  }
  expect(prompts).toBe(0)
  expect(opens).toBe(0)
})

test('HTTP and HTTPS links use the existing bridge normalization', async () => {
  const opened: string[] = []
  const bridge = { openExternal: async (url: string) => { opened.push(url); return true } }
  expect(await openTerminalLink('http://localhost:8080', () => true, bridge)).toBeNull()
  expect(await openTerminalLink('HTTPS://EXAMPLE.COM', () => true, bridge)).toBeNull()
  expect(opened).toEqual(['http://localhost:8080/', 'https://example.com/'])
})

test('a false host result is a visible, actionable failure instead of silent success', async () => {
  const error = await openTerminalLink(oauth, () => true, { openExternal: async () => false })
  expect(error).toContain('Could not open')
  expect(error).toContain('Copy the link')
})

test('host rejection is handled without leaking OAuth parameters into the error', async () => {
  const error = await openTerminalLink(oauth, () => true, { openExternal: async () => { throw new Error(`OS refused ${oauth}`) } })
  expect(error).toContain('Could not open')
  expect(error).not.toContain('test-state')
  expect(error).not.toContain('code_challenge')
})

test('a missing link opener reports failure rather than pretending the browser opened', async () => {
  expect(await openTerminalLink(oauth, () => true, {})).toContain('Could not open')
})
