// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { getOverlayState, resetOverlayState } from '../app/overlayStore.js'
import { coreCommands } from '../app/slash/commands/core.js'
import type { SlashRunCtx } from '../app/slash/types.js'
import { COPY_USAGE, copyTextToClipboard } from '../lib/copyText.js'
import type { Msg } from '../types.js'

vi.mock('../lib/copyText.js', async importOriginal => {
  const actual = await importOriginal<typeof import('../lib/copyText.js')>()

  return { ...actual, copyTextToClipboard: vi.fn() }
})

const mockedCopy = vi.mocked(copyTextToClipboard)

const copyCommand = coreCommands.find(command => command.name === 'copy')!

const history: Msg[] = [
  { kind: 'intro', role: 'system', text: '' },
  { role: 'user', text: 'how do I copy' },
  { role: 'assistant', text: 'use /copy' },
  { role: 'user', text: 'thanks' },
  { role: 'assistant', text: 'anytime' }
]

const flush = async () => {
  await Promise.resolve()
  await Promise.resolve()
}

function makeCtx(items: Msg[] = history) {
  const sys: string[] = []
  const ctx = {
    composer: { hasSelection: false },
    local: { getHistoryItems: () => items },
    stale: () => false,
    transcript: { sys: (text: string) => sys.push(text) }
  } as unknown as SlashRunCtx

  return { ctx, sys }
}

describe('/copy slash command', () => {
  beforeEach(() => {
    mockedCopy.mockReset()
    mockedCopy.mockImplementation(async text => ({ backend: 'native' as const, characters: text.length }))
  })

  afterEach(() => {
    resetOverlayState()
  })

  it('copies the last assistant message by default with a character count', async () => {
    const { ctx, sys } = makeCtx()

    await copyCommand.run('1', ctx, '/copy 1')
    await flush()

    expect(mockedCopy).toHaveBeenCalledWith('use /copy')
    expect(sys).toEqual(['copied 9 characters'])
  })

  it('copies the last user message via /copy user', async () => {
    const { ctx, sys } = makeCtx()

    await copyCommand.run('user', ctx, '/copy user')
    await flush()

    expect(mockedCopy).toHaveBeenCalledWith('thanks')
    expect(sys).toEqual(['copied 6 characters'])
  })

  it('copies the nth user message via /copy user n', async () => {
    const { ctx } = makeCtx()

    await copyCommand.run('user 1', ctx, '/copy user 1')
    await flush()

    expect(mockedCopy).toHaveBeenCalledWith('how do I copy')
  })

  it('copies the newest message of any role via /copy last', async () => {
    const { ctx } = makeCtx()

    await copyCommand.run('last', ctx, '/copy last')
    await flush()

    expect(mockedCopy).toHaveBeenCalledWith('anytime')
  })

  it('copies the full role-labeled transcript via /copy all', async () => {
    const { ctx } = makeCtx()

    await copyCommand.run('all', ctx, '/copy all')
    await flush()

    const text = mockedCopy.mock.calls[0]![0]!

    expect(text).toContain('[You #1]\nhow do I copy')
    expect(text).toContain('[Xerxes #1]\nuse /copy')
    expect(text).toContain('[You #2]\nthanks')
    expect(text).toContain('[Xerxes #2]\nanytime')
  })

  it.each(['bogus', 'user x', '0', 'all 2'])('prints usage for /copy %s', async arg => {
    const { ctx, sys } = makeCtx()

    await copyCommand.run(arg, ctx, `/copy ${arg}`)
    await flush()

    expect(sys).toEqual([COPY_USAGE])
    expect(mockedCopy).not.toHaveBeenCalled()
  })

  it('bare /copy opens the message picker with all roles, newest snapshot', async () => {
    const { ctx, sys } = makeCtx()

    await copyCommand.run('', ctx, '/copy')
    await flush()

    const picker = getOverlayState().copyPicker

    expect(picker).not.toBeNull()
    expect(picker!.items.map(item => `${item.role}:${item.ordinal}`)).toEqual([
      'user:1',
      'assistant:1',
      'user:2',
      'assistant:2'
    ])
    expect(sys).toEqual([])
    expect(mockedCopy).not.toHaveBeenCalled()
  })

  it('bare /copy with an empty transcript says there is nothing to copy', async () => {
    const { ctx, sys } = makeCtx([])

    await copyCommand.run('', ctx, '/copy')
    await flush()

    expect(sys).toEqual(['nothing to copy — start a conversation first'])
    expect(getOverlayState().copyPicker).toBeNull()
  })

  it('reports the OSC52 fallback honestly when native tools fail', async () => {
    mockedCopy.mockImplementation(async text => ({ backend: 'osc52' as const, characters: text.length }))
    const { ctx, sys } = makeCtx()

    await copyCommand.run('1', ctx, '/copy 1')
    await flush()

    expect(sys[0]).toContain('OSC52')
    expect(sys[0]).toContain('9 characters')
  })

  it('reports an honest error when every backend fails', async () => {
    mockedCopy.mockImplementation(async text => ({ backend: null, characters: text.length }))
    const { ctx, sys } = makeCtx()

    await copyCommand.run('1', ctx, '/copy 1')
    await flush()

    expect(sys[0]).toContain('copy failed')
  })
})
