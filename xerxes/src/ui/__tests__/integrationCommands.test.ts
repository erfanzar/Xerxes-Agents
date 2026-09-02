// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { resetOverlayState } from '../app/overlayStore.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

const flush = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

function makeSlashContext(response: unknown) {
  const request = vi.fn().mockResolvedValue(response)
  const sys: string[] = []
  const page: string[] = []
  const context = {
    composer: {
      enqueue: vi.fn(),
      hasSelection: false,
      paste: vi.fn(),
      queueRef: { current: [] },
      selection: {
        captureScrolledRows: vi.fn(),
        clearSelection: vi.fn(),
        copySelection: vi.fn(),
        copySelectionNoClear: vi.fn(),
        getState: vi.fn(),
        shiftAnchor: vi.fn(),
        shiftSelection: vi.fn(),
        version: vi.fn()
      },
      setInput: vi.fn()
    },
    gateway: { gw: { request }, rpc: request },
    local: {
      catalog: null,
      getHistoryItems: vi.fn(() => []),
      getLastUserMsg: vi.fn(() => ''),
      maybeWarn: vi.fn(),
      setCatalog: vi.fn()
    },
    session: {
      closeSession: vi.fn(),
      die: vi.fn(),
      dieWithCode: vi.fn(),
      guardBusySessionSwitch: vi.fn(),
      newLiveSession: vi.fn(),
      newSession: vi.fn(),
      resetVisibleHistory: vi.fn(),
      resumeById: vi.fn(),
      setSessionStartedAt: vi.fn()
    },
    slashFlightRef: { current: 0 },
    transcript: {
      dispatch: vi.fn(),
      page: (text: string) => page.push(text),
      panel: vi.fn(),
      send: vi.fn(),
      setHistoryItems: vi.fn(),
      sys: (text: string) => sys.push(text),
      trimLastExchange: vi.fn(items => items)
    },
    voice: { setVoiceEnabled: vi.fn(), setVoiceRecordKey: vi.fn(), setVoiceTts: vi.fn() }
  } as never

  return { context, page, request, sys }
}

describe('integration slash commands', () => {
  beforeEach(() => {
    patchUiState({ sid: 's1' })
  })

  afterEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it('/channels lists gateways with enabled state', async () => {
    const { context, page, request } = makeSlashContext({
      channels: [
        { adapter_name: 'telegram', enabled: true, name: 'telegram' },
        { adapter_name: 'discord', enabled: false, last_error: 'missing token', name: 'discord' }
      ],
      channels_available: true,
      channels_configured: true,
      ok: true
    })

    createSlashHandler(context)('/channels')
    await flush()

    expect(request).toHaveBeenCalledWith('channel.list', {})
    expect(page[0]).toContain('telegram')
    expect(page[0]).toContain('● telegram')
    expect(page[0]).toContain('○ discord')
    expect(page[0]).toContain('missing token')
  })

  it('/channels enable forwards the channel name and renders the refreshed list', async () => {
    const { context, page, request } = makeSlashContext({
      channels: [{ enabled: true, name: 'slack' }],
      ok: true
    })

    createSlashHandler(context)('/channels enable slack')
    await flush()

    expect(request).toHaveBeenCalledWith('channel.enable', { name: 'slack' })
    expect(page[0]).toContain('slack')
  })

  it('/channels enable surfaces a daemon-side failure instead of inventing success', async () => {
    const { context, request, sys } = makeSlashContext({ error: 'channel manager is not configured', ok: false })

    createSlashHandler(context)('/channels enable slack')
    await flush()

    expect(request).toHaveBeenCalledWith('channel.enable', { name: 'slack' })
    expect(sys[0]).toContain('channel manager is not configured')
  })

  it('/providers lists profiles with the active marker', async () => {
    const { context, page, request } = makeSlashContext({
      ok: true,
      profiles: [
        { active: true, base_url: 'https://api.openai.com/v1', model: 'gpt-4o', name: 'main', provider: 'openai' },
        { active: false, base_url: 'http://localhost:11434/v1', model: 'llama3', name: 'local', provider: 'ollama' }
      ]
    })

    createSlashHandler(context)('/providers')
    await flush()

    expect(request).toHaveBeenCalledWith('provider_list', {})
    expect(page[0]).toContain('● active')
    expect(page[0]).toContain('main')
    expect(page[0]).toContain('local')
  })

  it('/providers use selects a profile then refreshes the list', async () => {
    const { context, request } = makeSlashContext({ ok: true, profiles: [] })

    createSlashHandler(context)('/providers use local')
    await flush()

    expect(request).toHaveBeenNthCalledWith(1, 'provider_select', { name: 'local' })
    expect(request).toHaveBeenNthCalledWith(2, 'provider_list', {})
  })

  it('/providers add forwards typed flags to provider_save', async () => {
    const { context, request } = makeSlashContext({ ok: true, profiles: [] })

    createSlashHandler(context)('/providers add work --type openai --model gpt-4o --key sk-test --base-url https://example.test/v1')
    await flush()

    expect(request).toHaveBeenNthCalledWith(1, 'provider_save', {
      api_key: 'sk-test',
      base_url: 'https://example.test/v1',
      model: 'gpt-4o',
      name: 'work',
      provider: 'openai'
    })
  })

  it('/providers add without a model prints usage instead of calling the daemon', async () => {
    const { context, request, sys } = makeSlashContext({ ok: true })

    createSlashHandler(context)('/providers add work --type openai')
    await flush()

    expect(request).not.toHaveBeenCalledWith('provider_save', expect.anything())
    expect(sys[0]).toContain('usage: /providers add')
  })

  it('/worktree create posts the action and session id', async () => {
    const { context, request, sys } = makeSlashContext({ ok: true, path: '/tmp/wt-feat' })

    createSlashHandler(context)('/worktree create feat-branch')
    await flush()

    expect(request).toHaveBeenCalledWith('workspace.worktree', {
      action: 'create',
      name: 'feat-branch',
      session_id: 's1'
    })
    expect(sys[0]).toContain('/tmp/wt-feat')
  })

  it('/creator-trace renders forge trace rows', async () => {
    const { context, page, request } = makeSlashContext({
      ok: true,
      trace: [{ action: 'define', at: 123, detail: 'ok', name: 'lint-fix', status: 'created', version: '1' }]
    })

    createSlashHandler(context)('/creator-trace')
    await flush()

    expect(request).toHaveBeenCalledWith('creator_trace', { session_id: 's1' })
    expect(page[0]).toContain('lint-fix')
    expect(page[0]).toContain('created')
  })
})
