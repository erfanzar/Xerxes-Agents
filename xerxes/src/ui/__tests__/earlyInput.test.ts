// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { testRender } from '@opentui/react/test-utils'
import { act, createElement } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useComposerState } from '../app/useComposerState.js'
import type { GatewayClient } from '../gatewayClient.js'
import {
  EMPTY_EARLY_INPUT,
  type EarlyInputStdin,
  dropLastGrapheme,
  finalizeEarlyInputText,
  reduceEarlyInput,
  setEarlyInputText,
  startEarlyInputCapture,
  takeEarlyInputText
} from '../lib/earlyInput.js'

const feed = (...chunks: (string | Uint8Array)[]) => chunks.reduce(reduceEarlyInput, EMPTY_EARLY_INPUT)

describe('reduceEarlyInput', () => {
  it('accumulates plain typing across chunks', () => {
    expect(feed('fix ', 'the ', 'build').text).toBe('fix the build')
  })

  it('decodes byte chunks, including multi-byte characters', () => {
    expect(feed(new TextEncoder().encode('héllo ✅')).text).toBe('héllo ✅')
  })

  it('translates CR to LF and folds CRLF into a single newline', () => {
    expect(feed('one\rtwo\r\nthree').text).toBe('one\ntwo\nthree')
  })

  it('keeps tabs but drops other control bytes', () => {
    expect(feed('a\tb\x00\x01\x1f\x7fz').text).toBe('a\tz')
  })

  it('strips CSI sequences such as arrow keys and mouse reports', () => {
    expect(feed('a\x1b[Db\x1b[<0;10;5Mc').text).toBe('abc')
  })

  it('strips SS3 sequences from application cursor mode', () => {
    expect(feed('a\x1bOAb').text).toBe('ab')
  })

  it('strips OSC sequences terminated by BEL or ST', () => {
    expect(feed('a\x1b]0;title\x07b\x1b]52;c;ZGF0YQ==\x1b\\c').text).toBe('abc')
  })

  it('drops Alt-modified two-character escapes', () => {
    expect(feed('a\x1bbz').text).toBe('az')
  })

  it('carries an escape sequence split across chunk boundaries', () => {
    const split = reduceEarlyInput(reduceEarlyInput(EMPTY_EARLY_INPUT, 'go\x1b['), 'Dnow')

    // Without the carry the tail leaks into the composer as a literal "[D".
    expect(split.text).toBe('gonow')
    expect(split.pending).toBe('')
  })

  it('reports a pending tail while a sequence is unfinished', () => {
    expect(reduceEarlyInput(EMPTY_EARLY_INPUT, 'hi\x1b]0;ti').pending).toBe('\x1b]0;ti')
  })

  it('abandons a runaway unterminated sequence instead of swallowing all input', () => {
    expect(reduceEarlyInput(EMPTY_EARLY_INPUT, `\x1b]0;${'x'.repeat(400)}`).pending).toBe('')
  })

  it('applies backspace and DEL by grapheme', () => {
    expect(feed('abc\x7f').text).toBe('ab')
    expect(feed('ab\x08\x08\x08').text).toBe('')
    expect(feed('hi 👍\x7f').text).toBe('hi ')
  })

  it('flags interrupt on Ctrl-C without keeping the byte', () => {
    const state = feed('oops\x03')

    expect(state.interrupt).toBe(true)
    expect(state.text).toBe('oops')
  })

  it('bounds the captured text so a startup paste cannot seed megabytes', () => {
    expect(feed('y'.repeat(20_000)).text.length).toBe(4096)
  })

  it('is pure — the previous state is never mutated', () => {
    const previous = reduceEarlyInput(EMPTY_EARLY_INPUT, 'seed')

    reduceEarlyInput(previous, ' more')

    expect(previous.text).toBe('seed')
    expect(EMPTY_EARLY_INPUT.text).toBe('')
  })
})

describe('dropLastGrapheme', () => {
  it('removes a whole surrogate pair rather than half of one', () => {
    expect(dropLastGrapheme('a😀')).toBe('a')
  })

  it('is a no-op on empty text', () => {
    expect(dropLastGrapheme('')).toBe('')
  })
})

describe('finalizeEarlyInputText', () => {
  it('trims the trailing newline left by an Enter press', () => {
    expect(finalizeEarlyInputText(feed('ship it\r'))).toBe('ship it')
  })

  it('preserves interior newlines and leading text', () => {
    expect(finalizeEarlyInputText(feed('one\rtwo'))).toBe('one\ntwo')
  })

  it('discards everything once Ctrl-C was pressed', () => {
    expect(finalizeEarlyInputText(feed('draft\x03'))).toBe('')
  })
})

class FakeStdin implements EarlyInputStdin {
  isTTY = true
  rawMode: boolean | null = null
  private listeners: ((chunk: string | Uint8Array) => void)[] = []

  emit(chunk: string | Uint8Array) {
    for (const listener of [...this.listeners]) {
      listener(chunk)
    }
  }

  get listenerCount() {
    return this.listeners.length
  }

  off(_event: 'data', listener: (chunk: string | Uint8Array) => void) {
    this.listeners = this.listeners.filter(entry => entry !== listener)
  }

  on(_event: 'data', listener: (chunk: string | Uint8Array) => void) {
    this.listeners.push(listener)
  }

  setRawMode(mode: boolean) {
    this.rawMode = mode
  }
}

describe('startEarlyInputCapture', () => {
  it('enables raw mode and collects typing until stop', () => {
    const stdin = new FakeStdin()
    const capture = startEarlyInputCapture({ stdin })

    expect(stdin.rawMode).toBe(true)
    stdin.emit('hello\x1b[D')
    stdin.emit(' there\r')

    expect(capture.stop()).toBe('hello there')
    expect(stdin.listenerCount).toBe(0)
  })

  it('leaves raw mode enabled at stop so the renderer handover does not flicker', () => {
    const stdin = new FakeStdin()

    startEarlyInputCapture({ stdin }).stop()

    expect(stdin.rawMode).toBe(true)
  })

  it('ignores bytes that arrive after stop', () => {
    const stdin = new FakeStdin()
    const capture = startEarlyInputCapture({ stdin })

    stdin.emit('kept')
    capture.stop()
    stdin.emit('lost')

    expect(capture.stop()).toBe('kept')
  })

  it('reports Ctrl-C exactly once', () => {
    const stdin = new FakeStdin()
    const onInterrupt = vi.fn()

    startEarlyInputCapture({ onInterrupt, stdin })
    stdin.emit('a\x03')
    stdin.emit('b\x03')

    expect(onInterrupt).toHaveBeenCalledTimes(1)
  })

  it('does no capture and touches no mode when stdin is not a tty', () => {
    const stdin = new FakeStdin()

    stdin.isTTY = false

    expect(startEarlyInputCapture({ stdin }).stop()).toBe('')
    expect(stdin.rawMode).toBeNull()
    expect(stdin.listenerCount).toBe(0)
  })
})

describe('early input handoff', () => {
  beforeEach(() => {
    takeEarlyInputText()
  })

  it('hands the captured text to the first reader', () => {
    setEarlyInputText('seeded prompt')

    expect(takeEarlyInputText()).toBe('seeded prompt')
  })

  it('clears after one read so a remount cannot resurrect old keystrokes', () => {
    setEarlyInputText('seeded prompt')
    takeEarlyInputText()

    expect(takeEarlyInputText()).toBe('')
  })
})

describe('composer seeding', () => {
  /** Mount the composer hook and report the value it starts with. */
  const initialComposerInput = async (): Promise<null | string> => {
    let observed: null | string = null
    const submits: string[] = []

    const Probe = () => {
      const { state } = useComposerState({
        catalog: null,
        gw: {} as GatewayClient,
        onClipboardPaste: () => {},
        submitRef: { current: value => submits.push(value) }
      })

      observed = state.input

      return null
    }

    const setup = await testRender(createElement(Probe), { height: 6, width: 40 })

    try {
      await setup.flush()

      // Seeding must never send: the text was typed blind during startup.
      expect(submits).toEqual([])

      return observed
    } finally {
      act(() => setup.renderer.destroy())
    }
  }

  beforeEach(() => {
    takeEarlyInputText()
  })

  it('starts the composer with the startup keystrokes instead of submitting them', async () => {
    setEarlyInputText('review the diff')

    expect(await initialComposerInput()).toBe('review the diff')
  })

  it('starts empty when nothing was typed during startup', async () => {
    expect(await initialComposerInput()).toBe('')
  })
})
