// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import {
  appendTranscriptMessage,
  hasTranscriptContent,
  removeTranscriptMessage,
  visibleTranscriptDetails,
  visibleTranscriptMessages
} from './messages.js'

describe('appendTranscriptMessage', () => {
  it('merges adjacent tool-only shelves into one transcript row', () => {
    const out = appendTranscriptMessage([{ kind: 'trail', role: 'system', text: '', tools: ['Terminal("one") ✓'] }], {
      kind: 'trail',
      role: 'system',
      text: '',
      tools: ['Terminal("two") ✓']
    })

    expect(out).toEqual([
      { kind: 'trail', role: 'system', text: '', tools: ['Terminal("one") ✓', 'Terminal("two") ✓'] }
    ])
  })

  it('merges tool shelves into the nearest thinking shelf', () => {
    const out = appendTranscriptMessage(
      [{ kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['Terminal("one") ✓'] }],
      { kind: 'trail', role: 'system', text: '', tools: ['Terminal("two") ✓'] }
    )

    expect(out).toEqual([
      { kind: 'trail', role: 'system', text: '', thinking: 'plan', tools: ['Terminal("one") ✓', 'Terminal("two") ✓'] }
    ])
  })
})

describe('removeTranscriptMessage', () => {
  it('removes only the rejected optimistic bubble by identity', () => {
    const older = { role: 'user' as const, text: 'same prompt' }
    const optimistic = { role: 'user' as const, text: 'same prompt' }

    expect(removeTranscriptMessage([older, optimistic], optimistic)).toEqual([older])
  })
})

describe('visibleTranscriptMessages', () => {
  it('keeps the startup intro until a real transcript message exists', () => {
    const intro = { kind: 'intro' as const, role: 'system' as const, text: '' }
    const provider = { kind: 'slash' as const, role: 'system' as const, text: '/provider' }
    const user = { role: 'user' as const, text: 'hello' }

    expect(visibleTranscriptMessages([intro])).toEqual([intro])
    expect(hasTranscriptContent([intro, provider])).toBe(false)
    expect(visibleTranscriptMessages([intro, provider])).toEqual([intro])
    expect(hasTranscriptContent([intro, provider, user])).toBe(true)
    expect(visibleTranscriptMessages([intro, provider, user])).toEqual([provider, user])
  })

  it('reveals command echoes when a provider flow emits a result', () => {
    const intro = { kind: 'intro' as const, role: 'system' as const, text: '' }
    const provider = { kind: 'slash' as const, role: 'system' as const, text: '/provider' }
    const result = { role: 'system' as const, text: 'Switched to kimi-code.' }

    expect(visibleTranscriptMessages([intro, provider, provider])).toEqual([intro])
    expect(hasTranscriptContent([intro, provider, result])).toBe(true)
    expect(visibleTranscriptMessages([intro, provider, result])).toEqual([provider, result])
    expect(visibleTranscriptMessages([intro])).toEqual([intro])
  })

  it('drops non-rendering bookkeeping trails before long-session virtualization', () => {
    const user = { role: 'user' as const, text: 'inspect the project' }
    const invisible = Array.from({ length: 1_000 }, () => ({
      kind: 'trail' as const,
      role: 'system' as const,
      text: '',
      toolTokens: 12
    }))
    const thinking = { kind: 'trail' as const, role: 'system' as const, text: '', thinking: 'checking files' }
    const tools = { kind: 'trail' as const, role: 'system' as const, text: '', tools: ['Read File src/a.ts ✓'] }
    const agents = {
      kind: 'trail' as const,
      role: 'system' as const,
      subagents: [
        {
          depth: 0,
          goal: 'panel-owned work',
          id: 'agent-1',
          index: 0,
          notes: [],
          parentId: null,
          status: 'running' as const,
          taskCount: 1,
          thinking: [],
          toolCount: 0,
          tools: []
        }
      ],
      text: ''
    }

    expect(visibleTranscriptMessages([user, ...invisible, agents, thinking, tools])).toEqual([user, thinking, tools])
    expect(hasTranscriptContent(invisible)).toBe(false)
  })

  it('filters hidden detail-only rows before they enter the virtual height model', () => {
    const user = { role: 'user' as const, text: 'inspect' }
    const thinking = { kind: 'trail' as const, role: 'system' as const, text: '', thinking: 'private plan' }
    const tools = { kind: 'trail' as const, role: 'system' as const, text: '', tools: ['Read File src/a.ts ✓'] }
    const mixed = {
      kind: 'trail' as const,
      role: 'system' as const,
      text: '',
      thinking: 'hidden thought',
      tools: ['Read File src/b.ts ✓']
    }

    expect(
      visibleTranscriptDetails([user, thinking, tools, mixed], {
        subagents: false,
        thinking: false,
        tools: true
      })
    ).toEqual([user, tools, mixed])
    expect(
      visibleTranscriptDetails([user, thinking, tools, mixed], {
        subagents: false,
        thinking: false,
        tools: false
      })
    ).toEqual([user])
  })
})
