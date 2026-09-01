// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'

import { sessionToMarkdown } from '../src/desktop/renderer/exportMarkdown.js'

const stamp = new Date('2026-08-29T12:00:00Z')

describe('sessionToMarkdown', () => {
  test('renders the metadata header with title, model and cwd', () => {
    const md = sessionToMarkdown({
      id: 'aa19f402',
      title: 'Ship cancel-safe loop',
      model: 'kimi-for-coding',
      cwd: '/repo',
      messages: [],
    }, stamp)
    expect(md).toContain('# Ship cancel-safe loop')
    expect(md).toContain('Exported from Xerxes · 2026-08-29 12:00')
    expect(md).toContain('session `aa19f402`')
    expect(md).toContain('model `kimi-for-coding`')
    expect(md).toContain('cwd `/repo`')
  })

  test('walks user, thinking and agent turns in stream order', () => {
    const md = sessionToMarkdown({
      title: 'T',
      messages: [
        { role: 'user', content: 'fix the flake' },
        { role: 'assistant', content: 'on it' },
        { role: 'assistant', content: 'done — retry wraps the assertion' },
      ],
      thinking_content: ['the flake is a race', 'green now'],
    })
    const you = md.indexOf('## You\n\nfix the flake')
    const thinking = md.indexOf('> the flake is a race')
    const agent = md.indexOf('## Agent\n\non it')
    expect(you).toBeGreaterThanOrEqual(0)
    expect(thinking).toBeGreaterThan(you)
    expect(agent).toBeGreaterThan(thinking)
    expect(md).toContain('> green now')
  })

  test('tool messages render their stored execution, not a fabricated row', () => {
    const md = sessionToMarkdown({
      title: 'T',
      messages: [
        { role: 'user', content: 'run it' },
        { role: 'tool', content: '3 files changed' },
        { role: 'assistant', content: 'done' },
      ],
      tool_executions: [{ name: 'exec_command', inputs: { cmd: ['git', 'status'] }, toolCallId: 't1', durationMs: 420 }],
    })
    expect(md).toContain('**`exec_command`** `git status` — _( 0.4s )_')
    expect(md).toContain('```\n3 files changed\n```')
    expect(md).not.toContain('**`tool`**')
  })

  test('a tool message without a stored execution is skipped entirely', () => {
    const md = sessionToMarkdown({
      title: 'T',
      messages: [{ role: 'user', content: 'x' }, { role: 'tool', content: 'orphan result' }],
    })
    expect(md).not.toContain('orphan result')
    expect(md).toContain('## You')
  })

  test('args fall back to compact JSON and fences widen when needed', () => {
    const md = sessionToMarkdown({
      title: 'T',
      messages: [
        { role: 'tool', content: 'output with ``` nested fence' },
        { role: 'tool', content: 'plain' },
      ],
      tool_executions: [
        { name: 'Write', inputs: { content: 'x' } },
        { name: 'Read', inputs: { file_path: '/repo/a.ts' } },
      ],
    })
    expect(md).toContain('**`Write`** `{"content":"x"}`')
    expect(md).toContain('**`Read`** `/repo/a.ts`')
    expect(md).toContain('````\noutput with ``` nested fence\n````')
  })

  test('untitled sessions fall back to the id and parts fold like the replay', () => {
    const md = sessionToMarkdown({
      id: 'aa19f402',
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'think', think: 'planning' },
            { type: 'tool_use', name: 'exec_command', input: { cmd: ['bun', 'test'] } },
            { type: 'text', text: 'all green' },
          ],
        },
      ],
    })
    expect(md).toContain('# Session aa19f402')
    expect(md).toContain('> planning')
    expect(md).toContain('**`exec_command`** `bun test`')
    expect(md).toContain('## Agent\n\nall green')
  })
})
