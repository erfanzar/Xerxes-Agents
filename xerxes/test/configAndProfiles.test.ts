// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { ProfileStore } from '../src/bridge/profiles.js'
import { daemonConfigPath, loadSystemDaemonConfig, resolveEnvironmentReferences } from '../src/daemon/config.js'
import { runtimeConnection } from '../src/daemon/runtimeConnection.js'

test('daemon config merges legacy/nested records and applies environment priority', async () => {
  const normalizedHome = await mkdtemp(join(tmpdir(), 'xerxes-config-'))
  try {
    await mkdir(join(normalizedHome, 'daemon'), { recursive: true })
    await writeFile(daemonConfigPath(normalizedHome), JSON.stringify({
      runtime: { model: 'saved-model', api_key_env: 'SAVED_KEY' },
      control: { websocket_port: 3333 },
      ws_host: '0.0.0.0',
      model: 'legacy-model',
    }), 'utf8')
    const config = loadSystemDaemonConfig({
      home: normalizedHome,
      projectDirectory: '/project',
      environment: { XERXES_MODEL: 'environment-model', SAVED_KEY: 'secret-from-env' },
    })
    expect(config.runtime.model).toBe('environment-model')
    expect(config.control.websocket_host).toBe('0.0.0.0')
    expect(config.control.websocket_port).toBe(3333)
    expect(resolveEnvironmentReferences(config.runtime, { SAVED_KEY: 'secret-from-env' })).toMatchObject({ api_key: 'secret-from-env' })
  } finally {
    await rm(normalizedHome, { recursive: true, force: true })
  }
})

test('profile store preserves active selection and filters sampling keys', async () => {
  const normalizedHome = await mkdtemp(join(tmpdir(), 'xerxes-profiles-'))
  try {
    const store = new ProfileStore(join(normalizedHome, 'profiles.json'))
    store.save({ name: 'local', baseUrl: 'http://localhost:11434/v1/', apiKey: '', model: 'llama3.3' })
    expect(store.active()).toMatchObject({ name: 'local', provider: 'ollama', base_url: 'http://localhost:11434/v1' })
    expect(store.updateSampling('local', { temperature: 0.2, not_supported: true })).toMatchObject({ sampling: { temperature: 0.2 } })
    expect(store.setActive('missing')).toBe(false)
    expect(store.list().find(profile => profile.name === 'local')).toMatchObject({ active: true })
    expect(store.get('local')).toMatchObject({ name: 'local', model: 'llama3.3' })
    expect(store.get('missing')).toBeUndefined()
    expect(store.get('__proto__')).toBeUndefined()
    expect(store.get('constructor')).toBeUndefined()
    expect(store.updateSampling('__proto__', { temperature: 0.2 })).toBeUndefined()
    expect(store.updateSampling('constructor', { temperature: 0.2 })).toBeUndefined()
    expect(store.get('__proto__')).toBeUndefined()
    expect(store.get('constructor')).toBeUndefined()
    expect((await stat(store.filePath)).mode & 0o777).toBe(0o600)
  } finally {
    await rm(normalizedHome, { recursive: true, force: true })
  }
})

test('runtime connection prefers explicit daemon settings over a saved provider profile', () => {
  const connection = runtimeConnection({
    runtime: {
      model: 'gpt-4o',
      api_key: 'daemon-key',
      max_tokens: 1234,
      temperature: 0.2,
      permission_mode: 'plan',
      responses_api: 'true',
    },
    control: {}, workspace: {}, channels: {}, projectDirectory: '/project', maxConcurrentTurns: 8,
  }, {
    name: 'saved', base_url: 'https://api.openai.com/v1', api_key: 'profile-key', model: 'gpt-4.1', provider: 'openai', sampling: {},
  })
  expect(connection).toEqual({
    model: 'gpt-4o', apiKey: 'daemon-key', baseUrl: 'https://api.openai.com/v1', provider: 'openai',
    maxTokens: 1234, temperature: 0.2, permissionMode: 'plan', responsesApi: true,
  })
})

test('runtime connection defaults to YOLO while preserving explicit stricter modes', () => {
  const base = {
    runtime: { model: 'gpt-4o' },
    control: {},
    workspace: {},
    channels: {},
    projectDirectory: '/project',
    maxConcurrentTurns: 8,
  }

  expect(runtimeConnection(base, undefined)?.permissionMode).toBe('accept-all')
  expect(runtimeConnection({ ...base, runtime: { model: 'gpt-4o', permission_mode: 'manual' } }, undefined)?.permissionMode)
    .toBe('manual')
})
