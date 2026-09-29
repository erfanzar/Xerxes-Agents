// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { CLAUDE_CODE_PROFILE_NAME, ProfileStore, profileLabel } from '../src/bridge/profiles.js'
import { profileOwningModel } from '../src/daemon/sessionProvider.js'

async function withStore(document: unknown, body: (store: ProfileStore, path: string) => Promise<void> | void): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-profiles-'))
  const path = join(root, 'profiles.json')
  try {
    await writeFile(path, JSON.stringify(document))
    await body(new ProfileStore(path), path)
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

const legacyClaudeCode = { name: 'cc', provider: 'claude-code', base_url: 'claude-code://local', api_key: '', model: 'claude-code/opus', model_capabilities: {}, sampling: { reasoning_effort: 'high' } }

test('the built-in Claude Code profile is called claude-code', () => {
  expect(CLAUDE_CODE_PROFILE_NAME).toBe('claude-code')
  expect(profileLabel('claude-code')).toBe('Claude Code')
})

test('a store saved under cc is read as claude-code, selection and settings included', async () => {
  await withStore({ active: 'cc', profiles: { cc: legacyClaudeCode } }, async (store, path) => {
    expect(store.active()?.name).toBe('claude-code')
    expect(store.active()?.model).toBe('claude-code/opus')
    expect(store.active()?.sampling).toEqual({ reasoning_effort: 'high' })
    expect(store.list().map(profile => profile.name)).not.toContain('cc')
    // Old sessions and `/provider cc` still resolve.
    expect(store.get('cc')?.name).toBe('claude-code')
    expect(store.setActive('cc')).toBe(true)
    // The first write persists the rename.
    const saved = JSON.parse(await readFile(path, 'utf8'))
    expect(saved.active).toBe('claude-code')
    expect(Object.keys(saved.profiles)).toEqual(['claude-code'])
  })
})

test('a user profile that is merely named cc is not taken for Claude Code', async () => {
  const mine = { name: 'cc', provider: 'openai', base_url: 'https://example.test/v1', api_key: 'k', model: 'm', model_capabilities: {}, sampling: {} }
  await withStore({ active: 'cc', profiles: { cc: mine } }, store => {
    expect(store.get('cc')?.provider).toBe('openai')
    expect(store.active()?.name).toBe('cc')
  })
})

test('a cc/ model prefix still names the Claude Code profile', async () => {
  await withStore({ active: 'cc', profiles: { cc: legacyClaudeCode } }, store => {
    expect(profileOwningModel(store.list(), 'cc/sonnet')?.name).toBe('claude-code')
  })
})
