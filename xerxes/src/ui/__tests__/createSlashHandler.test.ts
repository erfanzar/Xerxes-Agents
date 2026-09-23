// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest'

import { createSlashHandler } from '../app/createSlashHandler.js'
import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
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
  const dispatch: string[] = []
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
        dispatch: (text: string) => dispatch.push(text),
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
    dispatch,
    getHistoryItems: () => historyItems,
    page,
    send,
    setHistoryItems,
    sys
  }
}

describe('createSlashHandler', () => {
  it.each(['/custom-agents', '/agents edit'])('opens project specialist editing with %s without resetting chat', async input => {
    patchUiState({ sid: 's1' })
    const request = vi.fn()
    const { context, send } = makeContext(request)
    createSlashHandler(context)(input)
    await flush()
    expect(getOverlayState().customAgentEditor).toBe(true)
    expect(send).toEqual([])
    expect(context.session.resetVisibleHistory).not.toHaveBeenCalled()
  })
  it('opens the capabilities guide without a provider call or resetting the conversation', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn()
    const { context, page, send } = makeContext(request)
    createSlashHandler(context)('/features')
    await flush()
    expect(getOverlayState().capabilities).toBe(true)
    expect(page).toEqual([])
    expect(send).toEqual([])
    expect(request).not.toHaveBeenCalled()
    expect(context.session.resetVisibleHistory).not.toHaveBeenCalled()
  })
  it('opens the machine picker without erasing history', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn(async () => ({ output: 'error: Remote workspace switching is not implemented in the TUI.' }))
    const { context, sys, send } = makeContext(request)
    createSlashHandler(context)('/machine')
    await flush()
    expect(request).not.toHaveBeenCalled()
    expect(getOverlayState().machinePicker).toBe(true)
    expect(send).toEqual([])
    expect(context.session.resetVisibleHistory).not.toHaveBeenCalled()
    expect(context.transcript.setHistoryItems).not.toHaveBeenCalled()
  })
  it('routes repository setup and existing workflow trust through the daemon without erasing history', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn(async () => ({ ok: true, queued: true }))
    const { context, send } = makeContext(request)
    createSlashHandler(context)('/init focus on kernel tests')
    await flush()
    expect(request).toHaveBeenCalledWith('slash.exec', { command: 'init focus on kernel tests', session_id: 's1' })
    createSlashHandler(context)('/skills trust repo-test')
    await flush()
    expect(request).toHaveBeenCalledWith('slash.exec', { command: 'skills trust repo-test', session_id: 's1' })
    expect(send).toEqual([])
    expect(context.session.resetVisibleHistory).not.toHaveBeenCalled()
    expect(context.transcript.setHistoryItems).not.toHaveBeenCalled()
  })
  it('routes unlimited goals without replacing history or sending a model message', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn(async () => ({ ok: true, text: 'Goal limits removed' }))
    const { context, page, send } = makeContext(request)
    createSlashHandler(context)('/goal unlimited')
    await flush()
    expect(request).toHaveBeenCalledWith('session.goal', { input: 'unlimited', session_id: 's1' })
    expect(page).toEqual(['Goal limits removed'])
    expect(send).toEqual([])
    expect(context.session.resetVisibleHistory).not.toHaveBeenCalled()
  })
  it('routes milestone commands without replacing history or sending a model message', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn(async () => ({ ok: true, text: 'Current milestone: Verify recovery' }))
    const { context, page, send } = makeContext(request)
    createSlashHandler(context)('/goal milestone Verify recovery')
    await flush()
    expect(request).toHaveBeenCalledWith('session.goal', { input: 'milestone Verify recovery', session_id: 's1' })
    expect(page).toEqual(['Current milestone: Verify recovery'])
    expect(send).toEqual([])
    expect(context.session.resetVisibleHistory).not.toHaveBeenCalled()
    expect(context.transcript.setHistoryItems).not.toHaveBeenCalled()
  })
  it('branches through the native command and loads the branch while preserving the source', async () => {
    const request = vi.fn(async () => ({ ok: true, session: { id: 'branch-id' } }))
    const { context } = makeContext(request)
    createSlashHandler(context)('/branch --through-turn 2 Review alternative')
    await flush()
    expect(request).toHaveBeenCalledWith('slash', { command: '/branch --through-turn 2 Review alternative' })
    expect(context.session.resumeById).toHaveBeenCalledWith('branch-id', { keepCurrent: true })
    expect(context.session.closeSession).not.toHaveBeenCalled()
  })
  it('leaves the current session alone when branching fails', async () => {
    const { context, sys } = makeContext(vi.fn(async () => ({ ok: false, error: 'turn is running' })))
    createSlashHandler(context)('/branch')
    await flush(); await flush()
    expect(context.session.resumeById).not.toHaveBeenCalled()
    expect(context.session.closeSession).not.toHaveBeenCalled()
    expect(sys.join(' ')).toContain('turn is running')
  })
  it('opens conversation loops without sending a model message', () => {
    const { context, send, sys } = makeContext(vi.fn());
    createSlashHandler(context)('/loop');
    expect(getOverlayState().loops).toBe(true);
    expect(getOverlayState().schedules).toBe(false);
    expect(send).toEqual([]); expect(sys).toEqual([]);
  });
  it('opens context inspection without a model turn', () => {
    const { context, send, sys } = makeContext(vi.fn());
    createSlashHandler(context)('/context');
    expect(getOverlayState().contextInspector).toBe(true);
    expect(send).toEqual([]); expect(sys).toEqual([]);
  });
  it('opens LSP settings without dispatching a model message', () => {
    const { context, send, sys } = makeContext(vi.fn());
    createSlashHandler(context)('/config lsp');
    expect(getOverlayState().lspSettings).toBe(true);
    expect(send).toEqual([]);
    expect(sys).toEqual([]);
  });
  it('opens MCP settings without dispatching a model message', () => {
    const request = vi.fn();
    const { context, send, sys } = makeContext(request);
    createSlashHandler(context)('/config mcp');
    expect(getOverlayState().mcpSettings).toBe(true);
    expect(send).toEqual([]);
    expect(sys).toEqual([]);
  });
  afterEach(() => {
    resetOverlayState()
    resetUiState()
  })

  it.each([
    ['/stop', 'stop'],
    ['/reload', 'reload'],
    ['/reload-mcp', 'reload-mcp'],
    ['/mcp status', 'mcp status'],
    ['/mcp reconnect fixture', 'mcp reconnect fixture'],
    ['/rollback list', 'snapshots'],
    [`/rollback apply target ${'a'.repeat(64)}`, `rollback apply target ${'a'.repeat(64)}`],
    ['/rollback diff snapshot-id', 'rollback diff snapshot-id'],
    ['/reload-skills', 'reload'],
    ['/skills', 'skills'],
    ['/skills list', 'skills'],
    ['/skills inspect review', 'skills inspect review'],
    ['/skills diagnostics', 'skills diagnostics'],
    ['/plugins', 'plugins'],
    ['/plugins inspect fixture', 'plugins inspect fixture'],
    ['/plugins install /tmp/module.ts', 'plugins install /tmp/module.ts'],
    ['/plugins enable fixture', 'plugins enable fixture'],
    ['/plugins disable fixture', 'plugins disable fixture'],
    ['/skills search review', 'skills search review'],
    ['/skills install /tmp/skill', 'skills install /tmp/skill'],
    ['/tools list', 'tools'],
    ['/image a native sunset', 'image a native sunset']
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

    expect(sys).toEqual([])
  })

  it('answers /voice locally instead of forwarding a control the daemon cannot apply', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({})
    const { context, send, sys } = makeContext(request)

    createSlashHandler(context)('/voice status')
    await flush()

    // The daemon only re-emits `/voice` as a `ui_command` event, which no
    // client handles. Forwarding it added a second line claiming the control
    // had been delivered, directly contradicting the honest one below.
    expect(request).not.toHaveBeenCalled()
    expect(send).toEqual([])
    expect(sys).toEqual([
      'voice capture is not implemented in this native Bun TUI; recording shortcuts are disabled.'
    ])
  })

  it('still validates /voice arguments before answering', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({})
    const { context, sys } = makeContext(request)

    createSlashHandler(context)('/voice nonsense')
    await flush()

    expect(request).not.toHaveBeenCalled()
    expect(sys).toEqual(['usage: /voice [on|off|tts|status]'])
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

  it('/background detaches the current chat and optionally dispatches an instruction through normal busy routing', () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn()
    const fixture = makeContext(request)

    createSlashHandler(fixture.context)('/background keep working on the tests')

    expect(fixture.dispatch).toEqual(['keep working on the tests'])
    expect(getOverlayState().sessions).toBe(true)
    expect(request).not.toHaveBeenCalled()
  })

  it('keeps /btw daemon-owned instead of aliasing it to /background', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({})
    const fixture = makeContext(request)

    createSlashHandler(fixture.context)('/btw what changed?')
    await flush()

    expect(fixture.dispatch).toEqual([])
    expect(getOverlayState().sessions).toBe(false)
    expect(request).toHaveBeenCalledWith('slash.exec', { command: 'btw what changed?', session_id: 's1' })
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

  it.each([
    ['/remove-memory', 'daemon.wipe_memory', 'Wipe ALL Xerxes memory?'],
    ['/remove-history', 'daemon.wipe_history', 'Wipe ALL chat history?']
  ])('%s opens a danger confirm before calling %s', async (input, method, title) => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({ ok: true, removed: { bytes: 64, files: 2 } })
    const { context, sys } = makeContext(request)

    createSlashHandler(context)(input)
    await flush()

    const confirm = getOverlayState().confirm
    expect(confirm?.danger).toBe(true)
    expect(confirm?.title).toBe(title)
    expect(confirm?.detail).toMatch(/cannot be undone/i)
    expect(request).not.toHaveBeenCalled()
    expect(sys).toEqual([])

    confirm?.onConfirm()
    await flush()

    expect(request).toHaveBeenCalledWith(method, {})
    expect(sys).toEqual([
      `${method === 'daemon.wipe_memory' ? 'memory wiped' : 'history wiped'}: 2 file(s), 64 B removed`
    ])
  })

  it('/remove-history without confirmation does not invoke the wipe RPC', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({ ok: true, removed: { bytes: 0, files: 0 } })
    const { context } = makeContext(request)

    createSlashHandler(context)('/remove-history')
    await flush()

    expect(request).not.toHaveBeenCalled()
  })

  it('surfaces a daemon-side refusal instead of a success message', async () => {
    patchUiState({ sid: 's1' })
    const request = vi.fn().mockResolvedValue({ error: 'a turn is mid-write', ok: false })
    const { context, sys } = makeContext(request)

    createSlashHandler(context)('/remove-history')
    await flush()
    getOverlayState().confirm?.onConfirm()
    await flush()

    expect(request).toHaveBeenCalledWith('daemon.wipe_history', {})
    expect(sys).toEqual(['could not history wiped: a turn is mid-write'])
  })
})
