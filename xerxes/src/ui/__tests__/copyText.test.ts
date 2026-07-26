// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it, vi } from 'vitest'

import type { Msg } from '../types.js'
import {
  COPY_USAGE,
  copyableMessages,
  copyLatestAssistantMessage,
  copyPreview,
  copyTextToClipboard,
  forceOsc52Clipboard,
  formatCopyOutcome,
  formatTranscriptForCopy,
  resolveCopyArg
} from '../lib/copyText.js'

const transcript: Msg[] = [
  { kind: 'intro', role: 'system', text: '' },
  { role: 'user', text: 'first question' },
  { role: 'assistant', text: 'first answer' },
  { role: 'tool', text: 'tool output' },
  { role: 'user', text: 'second question' },
  { role: 'assistant', text: 'second answer' },
  { role: 'assistant', text: '   ' }
]

const items = copyableMessages(transcript)

describe('copyableMessages', () => {
  it('keeps non-empty user + assistant text with per-role ordinals', () => {
    expect(items).toEqual([
      { ordinal: 1, role: 'user', text: 'first question' },
      { ordinal: 1, role: 'assistant', text: 'first answer' },
      { ordinal: 2, role: 'user', text: 'second question' },
      { ordinal: 2, role: 'assistant', text: 'second answer' }
    ])
  })
})

describe('resolveCopyArg', () => {
  it('bare arg opens the picker with a snapshot of all copyable messages', () => {
    const resolution = resolveCopyArg('', items)
    expect(resolution.kind).toBe('picker')
    if (resolution.kind === 'picker') {
      expect(resolution.items).toHaveLength(4)
    }
  })

  it('numeric form keeps nth-assistant semantics, clamped to the last one', () => {
    expect(resolveCopyArg('1', items)).toEqual({ kind: 'text', text: 'first answer' })
    expect(resolveCopyArg('2', items)).toEqual({ kind: 'text', text: 'second answer' })
    expect(resolveCopyArg('99', items)).toEqual({ kind: 'text', text: 'second answer' })
  })

  it('/copy user defaults to the last user message', () => {
    expect(resolveCopyArg('user', items)).toEqual({ kind: 'text', text: 'second question' })
  })

  it('/copy user n picks the nth user message, clamped', () => {
    expect(resolveCopyArg('user 1', items)).toEqual({ kind: 'text', text: 'first question' })
    expect(resolveCopyArg('user 99', items)).toEqual({ kind: 'text', text: 'second question' })
  })

  it('/copy last picks the newest message of any role', () => {
    expect(resolveCopyArg('last', items)).toEqual({ kind: 'text', text: 'second answer' })
  })

  it('/copy all renders the full role-labeled transcript', () => {
    const resolution = resolveCopyArg('all', items)
    expect(resolution.kind).toBe('text')
    if (resolution.kind === 'text') {
      expect(resolution.text).toBe(formatTranscriptForCopy(items))
      expect(resolution.text).toContain('[You #1]\nfirst question')
      expect(resolution.text).toContain('[Xerxes #2]\nsecond answer')
    }
  })

  it.each(['bogus', '1x', 'user x', 'user 0', '0', 'all 2', 'last 1', 'user 1 extra', 'two words'])(
    'rejects %j with a usage error',
    arg => {
      expect(resolveCopyArg(arg, items)).toEqual({ kind: 'usage' })
    }
  )

  it('reports empty states honestly', () => {
    expect(resolveCopyArg('', [])).toEqual({ kind: 'empty', message: 'nothing to copy — start a conversation first' })
    expect(resolveCopyArg('1', [])).toEqual({ kind: 'empty', message: 'nothing to copy — start a conversation first' })
    const onlyUser = items.filter(message => message.role === 'user')
    expect(resolveCopyArg('1', onlyUser)).toEqual({ kind: 'empty', message: 'no assistant messages to copy yet' })
    const onlyAssistant = items.filter(message => message.role === 'assistant')
    expect(resolveCopyArg('user', onlyAssistant)).toEqual({ kind: 'empty', message: 'no user messages to copy yet' })
  })

  it('exposes a usage string covering every form', () => {
    expect(COPY_USAGE).toContain('/copy [n]')
    expect(COPY_USAGE).toContain('/copy user [n]')
    expect(COPY_USAGE).toContain('/copy last')
    expect(COPY_USAGE).toContain('/copy all')
  })
})

describe('copyPreview', () => {
  it('flattens whitespace and clamps to the width', () => {
    expect(copyPreview('line one\n  line two\tthree', 80)).toBe('line one line two three')
    expect(copyPreview('abcdefghij', 5)).toBe('abcd…')
  })
})

describe('copyTextToClipboard fallback ordering', () => {
  it('uses the native backend first and reports native', async () => {
    const native = vi.fn(async () => true)
    const osc52 = vi.fn(() => true)

    const outcome = await copyTextToClipboard('hello', { env: {}, native, osc52 })

    expect(outcome).toEqual({ backend: 'native', characters: 5 })
    expect(native).toHaveBeenCalledWith('hello')
    expect(osc52).not.toHaveBeenCalled()
  })

  it('falls back to OSC52 when native fails and reports osc52', async () => {
    const native = vi.fn(async () => false)
    const osc52 = vi.fn(() => true)

    const outcome = await copyTextToClipboard('hello', { env: {}, native, osc52 })

    expect(outcome).toEqual({ backend: 'osc52', characters: 5 })
    expect(osc52).toHaveBeenCalledWith('hello')
  })

  it('falls back to OSC52 when native throws', async () => {
    const outcome = await copyTextToClipboard('hello', {
      env: {},
      native: async () => {
        throw new Error('spawn failed')
      },
      osc52: () => true
    })

    expect(outcome.backend).toBe('osc52')
  })

  it('reports failure honestly when both backends fail', async () => {
    const outcome = await copyTextToClipboard('hello', {
      env: {},
      native: async () => false,
      osc52: () => false
    })

    expect(outcome).toEqual({ backend: null, characters: 5 })
    expect(formatCopyOutcome(outcome)).toContain('copy failed')
  })

  it('XERXES_TUI_FORCE_OSC52=1 skips native entirely', async () => {
    const native = vi.fn(async () => true)
    const osc52 = vi.fn(() => true)

    const outcome = await copyTextToClipboard('hello', {
      env: { XERXES_TUI_FORCE_OSC52: '1' },
      native,
      osc52
    })

    expect(outcome.backend).toBe('osc52')
    expect(native).not.toHaveBeenCalled()
  })

  it('treats 0/false/off as not forced', () => {
    expect(forceOsc52Clipboard({ XERXES_TUI_FORCE_OSC52: '1' })).toBe(true)
    expect(forceOsc52Clipboard({ XERXES_TUI_FORCE_OSC52: '0' })).toBe(false)
    expect(forceOsc52Clipboard({ XERXES_TUI_FORCE_OSC52: 'false' })).toBe(false)
    expect(forceOsc52Clipboard({ XERXES_TUI_FORCE_OSC52: 'off' })).toBe(false)
    expect(forceOsc52Clipboard({})).toBe(false)
  })

  it('never copies empty text', async () => {
    const native = vi.fn(async () => true)
    const outcome = await copyTextToClipboard('', { env: {}, native })
    expect(outcome).toEqual({ backend: null, characters: 0 })
    expect(native).not.toHaveBeenCalled()
  })
})

describe('formatCopyOutcome', () => {
  it('names the backend that succeeded', () => {
    expect(formatCopyOutcome({ backend: 'native', characters: 12 })).toBe('copied 12 characters')
    expect(formatCopyOutcome({ backend: 'osc52', characters: 7 })).toContain('OSC52')
    expect(formatCopyOutcome({ backend: null, characters: 3 })).toContain('copy failed')
  })
})

describe('copyLatestAssistantMessage', () => {
  it('copies the newest assistant text and reports the character count', async () => {
    const copy = vi.fn(async () => ({ backend: 'native', characters: 13 }) as const)
    const message = await copyLatestAssistantMessage(transcript, copy)

    expect(copy).toHaveBeenCalledWith('second answer')
    expect(message).toBe('copied 13 characters')
  })

  it('says when there is nothing to copy', async () => {
    const copy = vi.fn()
    const message = await copyLatestAssistantMessage([{ role: 'user', text: 'hi' }], copy)

    expect(message).toBe('nothing to copy — no assistant message yet')
    expect(copy).not.toHaveBeenCalled()
  })

  it('surfaces clipboard errors instead of throwing', async () => {
    const message = await copyLatestAssistantMessage(transcript, async () => {
      throw new Error('boom')
    })

    expect(message).toBe('copy failed: Error: boom')
  })
})
