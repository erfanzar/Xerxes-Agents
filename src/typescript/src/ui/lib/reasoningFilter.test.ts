// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { ReasoningFilter } from './reasoningFilter.js'

const collect = (chunks: string[], filter = new ReasoningFilter()) => {
  let thinking = ''
  let visible = ''

  for (const chunk of chunks) {
    const output = filter.feed(chunk)

    thinking += output.thinking
    visible += output.visible
  }

  const tail = filter.flush()

  return { filter, thinking: thinking + tail.thinking, visible: visible + tail.visible }
}

describe('ReasoningFilter', () => {
  it('passes ordinary streamed text through on flush', () => {
    const result = collect(['hello world'])

    expect(result.visible).toBe('hello world')
    expect(result.thinking).toBe('')
  })

  it('suppresses reasoning blocks and records their content', () => {
    const result = collect(['before <reasoning>private plan</reasoning> after'])

    expect(result.visible).toBe('before  after')
    expect(result.thinking).toBe('private plan')
    expect(result.filter.thinkingLog).toBe('private plan')
  })

  it('suppresses an opening tag split across chunks', () => {
    const result = collect(['a <reaso', 'ning>hidden</reasoning> c'])

    expect(result.visible).toBe('a  c')
    expect(result.thinking).toBe('hidden')
  })

  it('matches tags case-insensitively by default', () => {
    const result = collect(['<THINK>secret</THINK>shown'])

    expect(result.visible).toBe('shown')
    expect(result.thinking).toBe('secret')
  })

  it('suppresses every supported scratchpad spelling', () => {
    for (const tag of ['think', 'thinking', 'reasoning', 'reasoning_scratchpad', 'scratchpad']) {
      const result = collect([`before <${tag}>hidden</${tag}> after`])

      expect(result.visible).toBe('before  after')
      expect(result.thinking).toBe('hidden')
    }
  })

  it('treats an unclosed suppressed block as thinking on flush', () => {
    const result = collect(['before <scratchpad>private'])

    expect(result.visible).toBe('before ')
    expect(result.thinking).toBe('private')
  })

  it('can opt out of case-insensitive matching', () => {
    const result = collect(['<THINK>not suppressed</THINK>'], new ReasoningFilter({ caseInsensitive: false }))

    expect(result.visible).toBe('<THINK>not suppressed</THINK>')
    expect(result.thinking).toBe('')
  })
})
