// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'

const flush = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

function makeContext(request: ReturnType<typeof vi.fn>) {
  const sys: string[] = []
  const page: string[] = []
  const send: string[] = []

  return {
    context: {
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
      gateway: {
        gw: { request },
        rpc: request
      },
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
        page: (text: string) => page.push(text),
        panel: vi.fn(),
        send: (text: string) => send.push(text),
        setHistoryItems: vi.fn(),
        sys: (text: string) => sys.push(text),
        trimLastExchange: vi.fn(items => items)
      },
      voice: {
        setVoiceEnabled: vi.fn(),
        setVoiceRecordKey: vi.fn(),
        setVoiceTts: vi.fn()
      }
    } as never,
    page,
    send,
    sys
  }
}

describe('createSlashHandler', () => {
  afterEach(() => resetUiState())

  it('keeps empty slash responses out of the transcript', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({})
    const { context, page, sys } = makeContext(request)

    createSlashHandler(context)('/remote-command')
    await flush()

    expect(request).toHaveBeenCalledWith('slash.exec', { command: 'remote-command', session_id: 's1' })
    expect(sys).toEqual([])
    expect(page).toEqual([])
  })

  it('does not render legacy empty command-dispatch output as no output', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockRejectedValueOnce(new Error('missing method')).mockResolvedValueOnce({
      output: '',
      type: 'exec'
    })
    const { context, sys } = makeContext(request)

    createSlashHandler(context)('/remote-command')
    await flush()

    expect(request).toHaveBeenLastCalledWith('command.dispatch', {
      arg: '',
      name: 'remote-command',
      session_id: 's1'
    })
    expect(sys).toEqual([])
  })

  it('renders modern slash output returned through the fallback path', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockRejectedValueOnce(new Error('missing method')).mockResolvedValueOnce({
      output: 'done'
    })
    const { context, sys } = makeContext(request)

    createSlashHandler(context)('/remote-command')
    await flush()

    expect(sys).toEqual(['done'])
  })
})
