// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { patchUiState, resetUiState } from '../app/uiStore.js'
import type { Msg, SlashCatalog } from '../types.js'

const flush = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

function makeContext(request: ReturnType<typeof vi.fn>, catalog: null | SlashCatalog = null) {
  const dieWithCode = vi.fn()
  const sys: string[] = []
  const page: string[] = []
  const send: string[] = []
  let historyItems: Msg[] = []
  const setHistoryItems = (update: Msg[] | ((items: Msg[]) => Msg[])) => {
    historyItems = typeof update === 'function' ? update(historyItems) : update
  }

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
        catalog,
        getHistoryItems: vi.fn(() => []),
        getLastUserMsg: vi.fn(() => ''),
        maybeWarn: vi.fn(),
        setCatalog: vi.fn()
      },
      session: {
        closeSession: vi.fn(),
        die: vi.fn(),
        dieWithCode,
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
        setHistoryItems: vi.fn(setHistoryItems),
        sys: (text: string) => sys.push(text),
        trimLastExchange: vi.fn(items => items)
      },
      voice: {
        setVoiceEnabled: vi.fn(),
        setVoiceRecordKey: vi.fn(),
        setVoiceTts: vi.fn()
      }
    } as never,
    dieWithCode,
    getHistoryItems: () => historyItems,
    page,
    send,
    setHistoryItems,
    sys
  }
}

describe('createSlashHandler', () => {
  afterEach(() => resetUiState())

  it.each([
    ['/stop', 'stop'],
    ['/reload', 'reload'],
    ['/reload-mcp', 'reload-mcp'],
    ['/rollback list', 'snapshots'],
    ['/reload-skills', 'reload'],
    ['/skills', 'skills'],
    ['/skills list', 'skills'],
    ['/plugins', 'plugins'],
    ['/tools list', 'tools'],
    ['/image a native sunset', 'image a native sunset'],
    ['/voice status', 'voice status']
  ])('routes %s through the native daemon slash handler', async (input, command) => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({})
    const { context, sys } = makeContext(request)

    createSlashHandler(context)(input)
    await flush()

    expect(request).toHaveBeenCalledWith('slash.exec', { command, session_id: 's1' })
    const methods = request.mock.calls.map(([method]) => method)

    for (const retired of [
      'process.stop',
      'reload.env',
      'reload.mcp',
      'rollback.list',
      'skills.reload',
      'skills.manage',
      'plugins.manage',
      'tools.configure',
      'voice.toggle'
    ]) {
      expect(methods).not.toContain(retired)
    }

    if (input === '/voice status') {
      expect(sys).toEqual([
        'voice capture is not implemented in this native Bun TUI; recording shortcuts are disabled.'
      ])
    } else {
      expect(sys).toEqual([])
    }
  })

  it('routes every yolo toggle through the daemon instead of the retired config shim', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({})
    const { context, sys } = makeContext(request)
    const handleSlash = createSlashHandler(context)

    handleSlash('/yolo')
    await flush()
    handleSlash('/yolo')
    await flush()

    expect(request.mock.calls).toEqual([
      ['slash.exec', { command: 'yolo', session_id: 's1' }],
      ['slash.exec', { command: 'yolo', session_id: 's1' }]
    ])
    expect(request).not.toHaveBeenCalledWith('config.set', expect.anything())
    expect(sys).toEqual([])
  })

  it.each(['/agents pause', '/agents resume', '/replay list', '/replay load /tmp/tree.json', '/tools disable shell'])(
    'does not call a retired RPC for unavailable native control %s',
    async input => {
      patchUiState({ sid: 's1' })
      const request = vi.fn().mockResolvedValue({})
      const { context, sys } = makeContext(request)

      createSlashHandler(context)(input)
      await flush()

      expect(request).not.toHaveBeenCalled()
      expect(sys).toHaveLength(1)
      expect(sys[0]).toMatch(/^unavailable in the native Bun daemon:/)
    }
  )

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

  it.each([
    {
      catalog: {
        canon: {},
        categories: [{ name: 'project skills', pairs: [['/deepscan', 'deep scan']] }],
        pairs: [],
        skillCount: 1,
        sub: {}
      } satisfies SlashCatalog,
      input: '/deepscan audit auth'
    },
    { catalog: null, input: '/skill deepscan audit auth' }
  ])('renders a model-turning skill command as authored user input: $input', async ({ catalog, input }) => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({})
    const fixture = makeContext(request, catalog)

    fixture.setHistoryItems([
      { kind: 'intro', role: 'system', text: '' },
      { kind: 'slash', role: 'system', text: input }
    ])

    createSlashHandler(fixture.context)(input)
    await flush()

    expect(fixture.getHistoryItems()).toEqual([
      { kind: 'intro', role: 'system', text: '' },
      { role: 'user', text: input }
    ])
    expect(request).toHaveBeenCalledWith('slash.exec', {
      command: input.slice(1),
      session_id: 's1'
    })
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

  it('shows native Bun update status without exiting the TUI or invoking a retired wrapper code', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({
      applied: false,
      command: 'bun run xerxes update',
      next_steps: [
        'bun run xerxes update --dry-run --spec file:./release-preview',
        'bun run xerxes update --apply --spec file:./release-preview'
      ],
      summary: 'current (HEAD abc123)'
    })
    const { context, dieWithCode, page, sys } = makeContext(request)

    createSlashHandler(context)('/update')
    await flush()

    expect(request).toHaveBeenCalledWith('runtime.update_status', {})
    expect(sys).toEqual(['checking Bun update status…'])
    expect(page).toEqual([
      [
        'Bun update status',
        'current (HEAD abc123)',
        '',
        'No update was run from the TUI.',
        'Status command: bun run xerxes update',
        'bun run xerxes update --dry-run --spec file:./release-preview',
        'bun run xerxes update --apply --spec file:./release-preview'
      ].join('\n')
    ])
    expect(dieWithCode).not.toHaveBeenCalled()
  })
})
