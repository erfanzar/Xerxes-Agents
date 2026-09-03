// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { spawnRosterFromLine, summarizeToolStartDisplay } from './toolStartDisplay.js'

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
    expect(spawnRosterFromLine(`Spawn Agents("${display.context}") ✓`)).toEqual({
      extra: 0,
      names: ['runtime', 'tools']
    })
  })

  it('uses agent titles and a safe reattach context when raw SpawnAgents JSON was truncated', () => {
    expect(
      summarizeToolStartDisplay(
        'Spawn Agents',
        '2 agents: Analyze structure, Audit security',
        '{"agents":[{"title":"Analyze structure","prompt":"truncated…'
      )
    ).toEqual({ context: '2 agents: Analyze structure, Audit security' })

    expect(
      summarizeToolStartDisplay(
        'SpawnAgents',
        '',
        JSON.stringify({ agents: [{ title: 'Analyze structure' }, { title: 'Audit security' }] })
      )
    ).toEqual({ context: '2 agents: Analyze structure, Audit security' })
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
        JSON.stringify({ include: '*.ts', path: 'xerxes/src/ui', pattern: 'streamSegments' })
      )
    ).toEqual({ context: 'streamSegments in xerxes/src/ui' })
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

  // The exact row that shipped in the transcript for months:
  //   → Task Output Tool {"task_id":"r9-timeout"}
  it('names an identifier-addressed call instead of echoing its JSON', () => {
    expect(summarizeToolStartDisplay('Task Output Tool', '{"task_id":"r9-timeout"}')).toEqual({
      context: 'r9-timeout'
    })
    expect(summarizeToolStartDisplay('subagent_status', '', JSON.stringify({ agent_id: 'reviewer-3' }))).toEqual({
      context: 'reviewer-3'
    })
  })

  it('renders unrecognized scalar payloads as key=value, never as a blob', () => {
    const args = JSON.stringify({ limit: 20, offset: 5, ascending: true })

    expect(summarizeToolStartDisplay('list_things', '', args)).toEqual({
      context: 'limit=20 · offset=5 · ascending=true'
    })
  })

  it('never carries a serialized object into the transcript', () => {
    // An object of nothing but nested values has no readable summary; an
    // empty row beats a wall of JSON.
    const args = JSON.stringify({ filter: { nested: true } })

    expect(summarizeToolStartDisplay('weird_tool', args, args).context).toBe('')
  })

  it('still prefers a path or query over an identifier', () => {
    const args = JSON.stringify({ id: 'abc123', file_path: 'src/one.ts' })

    expect(summarizeToolStartDisplay('read_file', '', args)).toEqual({ context: 'src/one.ts' })
  })
})
