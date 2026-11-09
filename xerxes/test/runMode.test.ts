// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { applyRunMode, readRunMode, runtimeSettingsFile, type RunModeProfile } from '../src/daemon/runMode.js'

const bonsai: RunModeProfile = { name: 'bonsai-local', model: 'ternary-bonsai-2-27b', base_url: 'http://127.0.0.1:11581/v1', data_trust: 'local' }
const claude: RunModeProfile = { name: 'claude-code', model: 'claude-code/opus', base_url: 'claude-code://local' }
const openrouter: RunModeProfile = { name: 'openrouter', model: 'deepseek/deepseek-v4.1-flash', base_url: 'https://openrouter.ai/api/v1' }
const profiles = new Map([bonsai, claude, openrouter].map(profile => [profile.name, profile]))
const profile = (name: string) => profiles.get(name)

function read(saved: Record<string, unknown>, worker: RunModeProfile | undefined, loaded: Record<string, unknown> = saved, env: Record<string, string> = {}) {
  return readRunMode({ saved, loaded, env, worker, profile })
}

test('the mode follows the worker and fully local mode; nothing is chosen until a worker is', () => {
  expect(read({}, undefined).mode).toBeNull()
  expect(read({ local_only: true }, bonsai)).toMatchObject({ mode: 'private', localOnly: true, worker: { profile: 'bonsai-local', local: true } })
  expect(read({ advisor: { profile: 'claude-code' } }, bonsai)).toMatchObject({
    mode: 'hybrid',
    advisor: { profile: 'claude-code', model: 'claude-code/opus', found: true },
    budgetUsd: 1,
  })
  expect(read({}, openrouter)).toMatchObject({ mode: 'cloud', worker: { local: false } })
})

test('a saved change the daemon has not started with is reported as needing a restart', () => {
  expect(read({ local_only: true }, bonsai, {}).restartPending).toBe(true)
  expect(read({ advisor: { profile: 'openrouter' } }, bonsai, { advisor: { profile: 'claude-code' } }).restartPending).toBe(true)
  expect(read({ escalation: { max_usd: 3 } }, bonsai, {}).restartPending).toBe(false)
  expect(read({}, bonsai, {}, { XERXES_LOCAL_ONLY: '1' })).toMatchObject({ mode: 'private', lockedBy: ['XERXES_LOCAL_ONLY'] })
})

test('a mode change writes only its own settings and keeps the rest', () => {
  const saved = { model_card: 'x', advisor: { profile: 'claude-code', model: 'claude-code/opus', max_calls: 3 }, escalation: { mode: 'on' } }
  expect(applyRunMode(saved, { mode: 'private' })).toEqual({ ...saved, local_only: true })
  const hybrid = applyRunMode({ ...saved, local_only: true }, { mode: 'hybrid', advisorProfile: 'openrouter', budgetUsd: 2.5 })
  // A model picked for the previous advisor is dropped with it; its other limits stay.
  expect(hybrid).toEqual({ model_card: 'x', advisor: { profile: 'openrouter', max_calls: 3 }, escalation: { mode: 'on', max_usd: 2.5 } })
  expect(applyRunMode({ advisor: { profile: 'claude-code' } }, { mode: 'cloud', advisorProfile: null })).toEqual({})
})

test('a budget the runtime would refuse on start is refused when saved', () => {
  expect(() => applyRunMode({}, { mode: 'hybrid', budgetUsd: 0 })).toThrow('escalation.max_usd')
  expect(() => applyRunMode({}, { mode: 'hybrid', budgetUsd: Number.NaN })).toThrow('escalation.max_usd')
})

test('the settings file keeps other sections, starts empty when missing and is owner-only', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'xerxes-run-mode-'))
  try {
    const path = join(dir, 'daemon', 'config.json')
    const missing = runtimeSettingsFile(path)
    expect(missing.read()).toEqual({})
    missing.write({ local_only: true })
    expect(JSON.parse(await readFile(path, 'utf8'))).toEqual({ runtime: { local_only: true } })
    expect((await stat(path)).mode & 0o777).toBe(0o600)

    await writeFile(path, JSON.stringify({ control: { websocket_port: 1 }, runtime: { model: 'm', local_only: true } }))
    const file = runtimeSettingsFile(path)
    file.write(applyRunMode(file.read(), { mode: 'hybrid' }))
    expect(JSON.parse(await readFile(path, 'utf8'))).toEqual({ control: { websocket_port: 1 }, runtime: { model: 'm' } })

    await writeFile(path, '[1]')
    expect(() => runtimeSettingsFile(path).read()).toThrow('must be a JSON object')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})
