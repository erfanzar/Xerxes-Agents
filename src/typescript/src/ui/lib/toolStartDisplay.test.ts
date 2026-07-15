// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { summarizeToolStartDisplay } from './toolStartDisplay.js'

describe('summarizeToolStartDisplay', () => {
  it('summarizes SpawnAgents without exposing prompt arguments', () => {
    const args = JSON.stringify({
      agents: [
        { name: 'runtime', prompt: 'read every runtime file and produce a long report' },
        { name: 'tools', prompt: 'read every tool file and produce a long report' }
      ],
      wait: true
    })

    const display = summarizeToolStartDisplay('SpawnAgents', '', args)

    expect(display).toEqual({ context: '2 agents: runtime, tools · wait=true' })
    expect('verboseArgs' in display).toBe(false)
    expect(display.context).not.toContain('long report')
  })

  it('reduces ordinary file args to a one-line path without retaining raw JSON', () => {
    expect(summarizeToolStartDisplay('ReadFile', 'x.py', '{"file_path":"x.py"}')).toEqual({
      context: 'x.py'
    })
  })

  it('formats command tools like a compact Grok row', () => {
    expect(
      summarizeToolStartDisplay(
        'ExecCommand',
        '',
        JSON.stringify({ args: ['-la'], cmd: 'ls', cwd: '.', timeout_ms: 10_000 })
      )
    ).toEqual({ context: 'ls -la' })
  })

  it('keeps search intent and path but drops unrelated argument fields', () => {
    expect(
      summarizeToolStartDisplay(
        'GrepTool',
        '',
        JSON.stringify({ include: '*.ts', path: 'src/typescript/src/ui', pattern: 'streamSegments' })
      )
    ).toEqual({ context: 'streamSegments in src/typescript/src/ui' })
  })

  it('summarizes WriteFile without exposing full content', () => {
    const args = JSON.stringify({
      file_path: '/workspace/Xerxes-Agents/AGENT_NOTES.md',
      content: 'hello\n'.repeat(600),
      overwrite: true
    })

    const display = summarizeToolStartDisplay('WriteFile', '', args)

    expect(display.context).toBe(
      'write /workspace/Xerxes-Agents/AGENT_NOTES.md · 3.6k chars · overwrite=true'
    )
    expect('verboseArgs' in display).toBe(false)
    expect(display.context).not.toContain('hello')
  })

  it('summarizes lowercase write_file calls', () => {
    const args = JSON.stringify({ file_path: 'docs/report.md', content: 'updated report' })

    expect(summarizeToolStartDisplay('write_file', '', args)).toEqual({
      context: 'write docs/report.md · 14 chars'
    })
  })

  it('summarizes file moves without exposing raw args', () => {
    const args = JSON.stringify({
      source: '/repo/src/old-name.ts',
      destination: '/repo/src/new-name.ts',
      overwrite: false
    })

    expect(summarizeToolStartDisplay('move_file', '', args)).toEqual({
      context: '/repo/src/old-name.ts -> /repo/src/new-name.ts'
    })
  })

  it('summarizes FileSystemTools move operations', () => {
    const args = JSON.stringify({ operation: 'move', path: 'tmp/a.txt', destination: 'tmp/b.txt' })

    expect(summarizeToolStartDisplay('FileSystemTools', '', args)).toEqual({
      context: 'tmp/a.txt -> tmp/b.txt'
    })
  })
})
