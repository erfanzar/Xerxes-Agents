// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { applyRunMode, HYBRID_BOUNDARY, readRunMode, runModeTiers, runtimeSettingsFile, type RunModeProfile } from '../src/daemon/runMode.js'

const bonsai: RunModeProfile = { name: 'bonsai-local', model: 'ternary-bonsai-2-27b', base_url: 'http://127.0.0.1:11581/v1', data_trust: 'local' }
const claude: RunModeProfile = { name: 'claude-code', model: 'claude-code/opus', base_url: 'claude-code://local' }
const openrouter: RunModeProfile = { name: 'openrouter', model: 'deepseek/deepseek-v4.1-flash', base_url: 'https://openrouter.ai/api/v1' }
const hybridRuntime = { data_boundary: 'hybrid', boundary: { ...HYBRID_BOUNDARY } }

function read(saved: Record<string, unknown>, worker: RunModeProfile | undefined, loaded: Record<string, unknown> = saved, env: Record<string, string> = {}) {
  return readRunMode({ saved, loaded, env, worker })
}

test('the mode follows the main profile, fully local mode and the Hybrid boundary', () => {
  expect(read({}, undefined).mode).toBeNull()
  expect(read({ local_only: true }, bonsai)).toMatchObject({ mode: 'private', localOnly: true, worker: { profile: 'bonsai-local', local: true } })
  expect(read(hybridRuntime, claude)).toMatchObject({ mode: 'hybrid', worker: { profile: 'claude-code', local: false } })
  expect(read({}, openrouter)).toMatchObject({ mode: 'cloud', worker: { local: false } })
  // The old Hybrid (Bonsai leading, a cloud advisor) is no longer a mode: the person picks again.
  expect(read({ advisor: { profile: 'claude-code' } }, bonsai).mode).toBeNull()
})

test('a saved change the daemon has not started with is reported as needing a restart', () => {
  expect(read({ local_only: true }, bonsai, {}).restartPending).toBe(true)
  expect(read(hybridRuntime, claude, {}).restartPending).toBe(true)
  expect(read(hybridRuntime, claude, hybridRuntime).restartPending).toBe(false)
  expect(read({ escalation: { max_usd: 3 } }, bonsai, {}).restartPending).toBe(false)
  expect(read({}, bonsai, {}, { XERXES_LOCAL_ONLY: '1' })).toMatchObject({ mode: 'private', lockedBy: ['XERXES_LOCAL_ONLY'] })
  expect(read({}, claude, {}, { XERXES_DATA_BOUNDARY: 'hybrid' })).toMatchObject({ mode: 'hybrid', lockedBy: ['XERXES_DATA_BOUNDARY'] })
})

test('a mode change writes only its own settings and keeps the rest', () => {
  const saved = { model_card: 'x', advisor: { profile: 'claude-code', model: 'claude-code/opus', max_calls: 3 }, escalation: { mode: 'on' }, boundary: { output_dir: 'results' } }
  // The advisor is no longer part of a mode: its profile goes, its other limits stay.
  expect(applyRunMode(saved, 'private')).toEqual({ model_card: 'x', advisor: { max_calls: 3 }, escalation: { mode: 'on' }, boundary: { output_dir: 'results' }, local_only: true })
  const hybrid = applyRunMode({ ...saved, local_only: true }, 'hybrid')
  expect(hybrid).toEqual({
    model_card: 'x', advisor: { max_calls: 3 }, escalation: { mode: 'on' },
    data_boundary: 'hybrid', boundary: { output_dir: 'results', private_paths: ['**'], cloud_results: 'summary', approve_outbound: 'off' },
  })
  // Leaving Hybrid takes its boundary off; a boundary someone set in another mode is left alone.
  expect(applyRunMode(hybrid, 'cloud')).toEqual({ model_card: 'x', advisor: { max_calls: 3 }, escalation: { mode: 'on' } })
  expect(applyRunMode({ data_boundary: 'private' }, 'cloud')).toEqual({ data_boundary: 'private' })
  expect(applyRunMode({ advisor: { profile: 'claude-code' } }, 'cloud')).toEqual({})
})

test('each mode places subagents: Bonsai answering directly for light and balanced work, smart on the lead in Hybrid', () => {
  const fast = { model: 'ternary-bonsai-2-27b', provider_profile: 'bonsai-local', reasoning_effort: 'off' }
  expect(runModeTiers('hybrid', bonsai, claude)).toEqual({ default: 'balanced', light: fast, balanced: fast, smart: { model: 'claude-code/opus', provider_profile: 'claude-code' } })
  expect(runModeTiers('private', bonsai, bonsai)).toEqual({ default: 'balanced', light: fast, balanced: fast, smart: { model: 'ternary-bonsai-2-27b', provider_profile: 'bonsai-local' } })
  expect(runModeTiers('cloud', undefined, claude)).toEqual({})
  expect(() => runModeTiers('hybrid', undefined, claude)).toThrow('Bonsai')
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
    file.write(applyRunMode(file.read(), 'cloud'))
    expect(JSON.parse(await readFile(path, 'utf8'))).toEqual({ control: { websocket_port: 1 }, runtime: { model: 'm' } })

    await writeFile(path, '[1]')
    expect(() => runtimeSettingsFile(path).read()).toThrow('must be a JSON object')
  } finally {
    await rm(dir, { recursive: true, force: true })
  }
})
