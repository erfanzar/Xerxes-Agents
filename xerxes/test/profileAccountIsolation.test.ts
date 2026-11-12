// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { profileSessionOptions } from '../src/auth/profileSessionOptions.js'
import { runtimeConnection } from '../src/daemon/runtimeConnection.js'
import { supervisorConfigFromEnvironment } from '../src/hub/cli.js'
import { inheritedSelectionValidator } from '../src/runtime/profileInventory.js'
import type { ProviderProfile } from '../src/bridge/profiles.js'

const profile: ProviderProfile = { name: 'bound-gemini', provider: 'gemini', model: 'gemini-2.5-flash', api_key: '', base_url: 'https://generativelanguage.googleapis.com/v1beta', sampling: {}, isolated_auth: true }

test('runtime connection retains dedicated API-key authentication isolation', () => {
  const connection = runtimeConnection({ channels: {}, control: {}, maxConcurrentTurns: 1, projectDirectory: '/workspace', runtime: {}, workspace: {} }, profile)
  expect(connection).toMatchObject({ isolatedAuth: true })
  expect(profileSessionOptions({ isolated_auth: true })).toEqual({ isolatedAuth: true })
})

test('Hub supervisor does not borrow ambient credentials for a dedicated account', () => {
  const previous = process.env.GEMINI_API_KEY
  process.env.GEMINI_API_KEY = 'fixture-ambient-not-the-account'
  try {
    const result = supervisorConfigFromEnvironment({ XERXES_HUB_SUPERVISOR_PROFILE: profile.name, XERXES_HUB_SUPERVISOR_MODEL: profile.model }, { get: () => profile })
    expect(result.kind).toBe('unconfigured')
    expect(JSON.stringify(result)).not.toContain('fixture-ambient')
  } finally {
    if (previous === undefined) delete process.env.GEMINI_API_KEY
    else process.env.GEMINI_API_KEY = previous
  }
})

test('inherited model validation preserves authentication isolation before any discovery', async () => {
  const previous = process.env.GEMINI_API_KEY
  process.env.GEMINI_API_KEY = 'fixture-ambient-not-the-account'
  try {
    const validate = inheritedSelectionValidator({ provider: profile.provider, model: profile.model, baseUrl: profile.base_url, permissionMode: 'accept-all', isolatedAuth: true })
    await expect(validate('gemini-other-fixture')).rejects.toThrow('dedicated API key')
  } finally {
    if (previous === undefined) delete process.env.GEMINI_API_KEY
    else process.env.GEMINI_API_KEY = previous
  }
})
