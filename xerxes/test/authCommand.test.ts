// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { AuthCommandError, runAuthCommand } from '../src/auth/command.js'
import type { CopilotCredential, CopilotSession } from '../src/auth/copilotAuth.js'
import { CliWriter } from '../src/runtime/cliStyle.js'

function capture(): { lines: string[]; writer: CliWriter } {
  const lines: string[] = []
  return { lines, writer: new CliWriter({ write: line => lines.push(line) }) }
}

function copilotSession(options: {
  credential?: CopilotCredential | undefined
  loggedOut?: boolean
}): CopilotSession {
  return {
    stored: async () => options.credential,
    logout: async () => options.loggedOut ?? options.credential !== undefined,
  } as unknown as CopilotSession
}

const CREDENTIAL: CopilotCredential = {
  access: 'copilot-access',
  refresh: 'gho_test',
  expires: Math.floor(Date.now() / 1000) + 3_600,
}

test('auth status copilot reports the stored session and its expiry', async () => {
  const { lines, writer } = capture()
  const code = await runAuthCommand(['status', 'copilot'], {
    copilotSession: copilotSession({ credential: CREDENTIAL }),
    writer,
  })
  expect(code).toBe(0)
  expect(lines.join('\n')).toContain('Signed in (github-copilot)')
})

test('auth status copilot without a session fails with the login hint', async () => {
  const { lines, writer } = capture()
  const saved = { ...process.env }
  delete process.env.COPILOT_GITHUB_TOKEN
  delete process.env.GH_TOKEN
  delete process.env.GITHUB_TOKEN
  try {
    const code = await runAuthCommand(['status', 'copilot'], {
      copilotSession: copilotSession({}),
      writer,
    })
    expect(code).toBe(1)
    expect(lines.join('\n')).toContain('xerxes auth login copilot')
  } finally {
    process.env = saved
  }
})

test('auth logout copilot removes the stored session', async () => {
  const { lines, writer } = capture()
  const code = await runAuthCommand(['logout', 'gh-copilot'], {
    copilotSession: copilotSession({ credential: CREDENTIAL }),
    writer,
  })
  expect(code).toBe(0)
  expect(lines.join('\n')).toContain('Removed the stored github-copilot session')
})

test('auth rejects unknown providers as usage errors', async () => {
  const { writer } = capture()
  await expect(runAuthCommand(['login', 'gemini'], { writer }))
    .rejects.toBeInstanceOf(AuthCommandError)
  await expect(runAuthCommand(['login'], { writer })).rejects.toThrow(/requires a provider/)
})
