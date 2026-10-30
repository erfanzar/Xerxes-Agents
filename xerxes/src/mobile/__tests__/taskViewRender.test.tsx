// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { renderToStaticMarkup } from 'react-dom/server'
import { describe, expect, it } from 'vitest'

import type { PendingInteraction } from '../../client/sessionFold.js'
import type { AgentRow } from '../../client/taskTypes.js'
import { AgentsPanel, TodoCard, agentsButtonLabel } from '../components/Agents.js'
import { remainingAfterSend, withTranscript } from '../components/Composer.js'
import { diffCounts, diffLines, diffRows } from '../components/diffLines.js'
import { InlineInteractions } from '../components/Interactions.js'
import { keyboardInset, trackKeyboardInset } from '../components/keyboardInset.js'
import { Markdown } from '../components/Markdown.js'
import { JumpToLatest } from '../components/TaskTranscript.js'
import { ToolRow } from '../components/ToolRow.js'
import { agentStatus, formatDuration, optionLabel, outputPreview, toolTarget } from '../components/toolView.js'
import { ConfirmationCard, supervisorAuthor } from '../components/XerxesTab.js'
import { reconnectOf } from '../MobileApp.js'
import { rememberInboxFilter, rememberedInboxFilter } from '../components/Inbox.js'
import { ModelChooser } from '../components/ModelChooser.js'
import { ServicesProvider } from '../context.js'
import { liveState, stubController, withSupervisorConfig } from './helpers.js'
import { makeServices, render } from './renderHelpers.js'
import { createTranscriptScroll } from '../transcriptScroll.js'

const plain = (html: string): string => html.replaceAll('<!-- -->', '').replaceAll('&#x27;', "'").replaceAll('&quot;', '"')
const noop = (): void => {}

describe('tool rows', () => {
  it('shows the verb, the exact command, state, duration and the output inside', () => {
    const html = plain(renderToStaticMarkup(<ToolRow tool={{ name: 'exec_command', summary: 'bun test', state: 'done', args: 'bun test --timeout 15000 xerxes/test/a.test.ts', output: '3 pass\n0 fail', durationMs: 4200 }} />))
    expect(html).toContain('<span class="tool-verb">Ran</span>')
    expect(html).toContain('<code class="tool-target">bun test --timeout 15000 xerxes/test/a.test.ts</code>')
    expect(html).toContain('aria-label="Command">bun test --timeout 15000 xerxes/test/a.test.ts</pre>')
    expect(html).toContain('aria-label="Output">3 pass\n0 fail</pre>')
    expect(html).toContain('4.2s')
    expect(html).toContain('aria-label="done"')
  })

  it('marks running and failed calls, and shows JSON args for non-shell tools', () => {
    const running = plain(renderToStaticMarkup(<ToolRow tool={{ name: 'GrepTool', summary: 'parseFrame', state: 'working', args: '{\n  "pattern": "parseFrame"\n}' }} />))
    expect(running).toContain('>Searching<')
    expect(running).toContain('class="tool-spin"')
    expect(running).toContain('aria-label="Arguments"')
    expect(running).toContain('"pattern": "parseFrame"')
    const failed = plain(renderToStaticMarkup(<ToolRow tool={{ name: 'exec_command', summary: 'false', state: 'error', output: 'exit 1' }} />))
    expect(failed).toContain('data-state="error"')
    expect(failed).toContain('>failed<')
  })

  it('collapses output beyond 20 lines behind "Show all"', () => {
    const long = Array.from({ length: 45 }, (_, i) => `line ${i + 1}`).join('\n')
    const html = plain(renderToStaticMarkup(<ToolRow tool={{ name: 'exec_command', summary: 'x', state: 'done', output: long }} />))
    expect(html).toContain('line 20</pre>')
    expect(html).not.toContain('line 21')
    expect(html).toContain('Show all 45 lines')
    expect(outputPreview(long, true).clipped).toBe(false)
  })

  it('renders an edit as a unified diff with + and - markers instead of its raw args', () => {
    const html = plain(renderToStaticMarkup(<ToolRow tool={{ name: 'FileEditTool', summary: 'a.ts', state: 'done', args: '{"old_string":"x"}', diff: { path: 'src/a.ts', before: 'one\ntwo\nthree', after: 'one\nTWO\nthree' } }} />))
    expect(html).toContain('<code class="tool-target">src/a.ts</code>')
    expect(html).toContain('>Edited<')
    expect(html).toContain('<span class="diff-line diff-del"><span class="diff-mark">-</span>two</span>')
    expect(html).toContain('<span class="diff-line diff-add"><span class="diff-mark">+</span>TWO</span>')
    expect(html).toContain('+1 −1')
    expect(html).not.toContain('old_string')
  })

  it('formats durations and picks the target', () => {
    expect(formatDuration(350)).toBe('350ms')
    expect(formatDuration(38_000)).toBe('38s')
    expect(formatDuration(125_000)).toBe('2m 05s')
    expect(formatDuration(undefined)).toBe('')
    expect(toolTarget({ name: 'ReadFile', summary: 'src/x.ts', state: 'done' })).toBe('src/x.ts')
  })
})

describe('line diff', () => {
  it('keeps context around changes and folds long unchanged runs into a gap', () => {
    const before = Array.from({ length: 30 }, (_, i) => `l${i}`).join('\n')
    const after = before.replace('l15', 'changed')
    const ops = diffLines(before, after)
    expect(diffCounts(ops)).toEqual({ added: 1, removed: 1 })
    const rows = diffRows(ops)
    expect(rows[0]).toEqual({ kind: 'gap', count: 12 })
    expect(rows.at(-1)).toEqual({ kind: 'gap', count: 11 })
    expect(rows.filter(row => row.kind === 'same')).toHaveLength(6)
  })

  it('treats a whole-file write (empty before) as all additions', () => {
    expect(diffCounts(diffLines('', 'a\nb'))).toEqual({ added: 2, removed: 0 })
  })
})

describe('todo list card', () => {
  it('shows each item with its checkbox state and the done count', () => {
    const html = plain(renderToStaticMarkup(<TodoCard todos={[
      { content: 'Read the code', status: 'completed' },
      { content: 'Fix the bug', status: 'in_progress' },
      { content: 'Run tests', status: 'pending' },
    ]} />))
    expect(html).toContain('1 of 3 done')
    expect(html).toContain('class="todo todo-completed"')
    expect(html).toContain('aria-label="in progress"')
    expect(html).toContain('aria-label="not started"')
    expect(renderToStaticMarkup(<TodoCard todos={[]} />)).toBe('')
  })
})

describe('agents panel', () => {
  const rows: AgentRow[] = [
    { id: 'a1', name: 'reviewer', title: 'Review the diff', status: 'running', toolCalls: 4, model: 'opus', updatedAt: '2026-10-21T11:58:00Z' },
    { id: 'a2', name: 'tester', title: 'Run tests', status: 'failed', error: 'tests failed' },
  ]
  const base = { now: Date.parse('2026-10-21T12:00:00Z'), onOpen: noop, onBack: noop, onRetryList: noop, onRetryDetail: noop }
  const idleDetail = { phase: 'idle' as const, steps: [], truncated: false }

  it('lists agents with title, status, tool count and update time', () => {
    const html = plain(renderToStaticMarkup(<AgentsPanel {...base} agents={{ phase: 'ready', rows }} detail={idleDetail} />))
    expect(html).toContain('Review the diff')
    expect(html).toContain('>Running<')
    expect(html).toContain('4 tool calls · opus · updated 2m ago')
    expect(html).toContain('>Failed<')
    expect(html).toContain('tests failed')
    expect(agentsButtonLabel(rows, undefined)).toBe('2 agents · 1 running')
    expect(agentsButtonLabel([], { total: 3, active: 0 })).toBe('3 agents')
    expect(agentsButtonLabel([], undefined)).toBeUndefined()
  })

  it('shows loading, empty and error-with-retry states', () => {
    expect(plain(renderToStaticMarkup(<AgentsPanel {...base} agents={{ phase: 'loading', rows: [] }} detail={idleDetail} />))).toContain('Loading agents…')
    expect(plain(renderToStaticMarkup(<AgentsPanel {...base} agents={{ phase: 'ready', rows: [] }} detail={idleDetail} />))).toContain('has not started any agents')
    const failed = plain(renderToStaticMarkup(<AgentsPanel {...base} agents={{ phase: 'error', rows: [], error: 'not_found' }} detail={idleDetail} />))
    expect(failed).toContain('Could not load the agents: not_found')
    expect(failed).toContain('>Try again<')
  })

  it("renders one agent's steps with the same tool rows as the transcript", () => {
    const html = plain(renderToStaticMarkup(<AgentsPanel {...base} agents={{ phase: 'ready', rows }} detail={{
      phase: 'ready', agentId: 'a1', agent: rows[0]!, truncated: true,
      steps: [
        { kind: 'thinking', text: 'look at files' },
        { kind: 'tool', name: 'exec_command', args: 'git diff --stat', output: '2 files changed', state: 'done' },
        { kind: 'text', text: 'Looks **good**' },
        { kind: 'notice', text: 'cancelled', state: 'error' },
      ],
    }} />))
    expect(html).toContain('‹ All agents')
    expect(html).toContain('<code class="tool-target">git diff --stat</code>')
    expect(html).toContain('2 files changed')
    expect(html).toContain('<strong>good</strong>')
    expect(html).toContain('Showing the latest steps only.')
    expect(html).toContain('block-notice is-error')
    const error = plain(renderToStaticMarkup(<AgentsPanel {...base} agents={{ phase: 'ready', rows }} detail={{ phase: 'error', agentId: 'a2', steps: [], truncated: false, error: 'offline' }} />))
    expect(error).toContain('Could not load this agent: offline')
    expect(error).toContain('>Try again<')
  })

  it('maps daemon status strings to a tone', () => {
    expect(agentStatus('queued')).toMatchObject({ label: 'Queued', active: true })
    expect(agentStatus('completed')).toMatchObject({ tone: 'done', active: false })
    expect(agentStatus('interrupted')).toMatchObject({ tone: 'problem' })
  })
})

describe('inline questions and permissions', () => {
  const permission: PendingInteraction = {
    id: 'p1', kind: 'permission', toolName: 'exec_command', title: 'Run this command?', description: 'Deletes the build',
    command: 'rm -rf build', inputs: [['cwd', '/proj']], detailsHidden: false, questions: [],
  }
  const question: PendingInteraction = {
    id: 'q1', kind: 'question', toolName: 'question', title: 'Xerxes has a question', description: '', inputs: [], detailsHidden: false,
    questions: [
      { id: 'a', question: 'Which approach?', options: ['Patch the script, then the tests (Recommended)', 'Rewrite it'], allowFreeForm: true },
      { id: 'b', question: 'Ship today?', options: ['Yes', 'No'], allowFreeForm: false },
    ],
  }
  const props = { decisions: {}, onDecide: noop, onAnswer: noop, onDetails: noop }

  it('puts the command and Allow / Allow for session / Deny on the card itself', () => {
    const html = plain(renderToStaticMarkup(<InlineInteractions {...props} items={[permission]} canRespond />))
    expect(html).toContain('aria-label="Command">rm -rf build</pre>')
    expect(html).toContain('Deletes the build')
    const allow = html.indexOf('>Allow<')
    expect(allow).toBeGreaterThan(-1)
    expect(html.indexOf('>Allow for session<')).toBeGreaterThan(allow)
    expect(html.indexOf('>Deny<')).toBeGreaterThan(allow)
    expect(html).not.toMatch(/<button[^>]*disabled[^>]*>Allow</)
  })

  it('offers only Deny when the details are missing (no blind Allow)', () => {
    const html = plain(renderToStaticMarkup(<InlineInteractions {...props} items={[{ ...permission, command: undefined as unknown as string, inputs: [], detailsHidden: true }]} canRespond />))
    expect(html).not.toContain('>Allow<')
    expect(html).not.toContain('>Allow for session<')
    expect(html).toContain('>Deny<')
    expect(html).toContain('cannot be approved blind')
  })

  it('shows every question with tappable options, the recommended one tagged, free text and one Send', () => {
    const html = plain(renderToStaticMarkup(<InlineInteractions {...props} items={[question]} canRespond />))
    expect(html).toContain('Which approach?')
    expect(html).toContain('Ship today?')
    expect(html).toContain('Patch the script, then the tests<span class="option-tag">Recommended</span>')
    expect(html).not.toContain('(Recommended)')
    expect(html.match(/<input type="text"/g)).toHaveLength(1)
    expect(html).toContain('2 of 2 still need an answer.')
    expect(html).toMatch(/<button type="submit" class="btn btn-primary" disabled="">Send answers/)
    expect(optionLabel('Plain')).toEqual({ label: 'Plain', recommended: false })
  })

  it('disables answering with a reason when the person cannot respond, and says when a decision failed', () => {
    const html = plain(renderToStaticMarkup(<InlineInteractions {...props} items={[permission]} canRespond={false} disabledReason="Decisions need a live connection to this workspace." decisions={{ p1: { phase: 'failed', decision: 'once', message: 'busy' } }} />))
    expect(html).toContain('Decisions need a live connection to this workspace.')
    expect(html).toMatch(/<button[^>]*disabled[^>]*>Allow</)
    expect(html).toContain('Not applied: busy')
  })

  it('says a question without its text cannot be answered here instead of showing nothing', () => {
    const html = plain(renderToStaticMarkup(<InlineInteractions {...props} items={[{ ...question, questions: [] }]} canRespond />))
    expect(html).toContain('was not sent to this device')
  })
})

describe('scroll follow and jump pill', () => {
  it('renders the pill only while scrolled away, naming what is waiting', () => {
    expect(renderToStaticMarkup(<JumpToLatest visible={false} onJump={noop} />)).toBe('')
    expect(plain(renderToStaticMarkup(<JumpToLatest visible waiting={2} onJump={noop} />))).toContain('2 waiting for you · Jump to latest')
  })

  it('reports pinned state and jumps back to the bottom, then follows again', () => {
    let y = 0
    let height = 2000
    const scroll = createTranscriptScroll({ y: () => y, height: () => height, viewport: () => 500, top: () => -y, blocks: () => [], scrollTo: v => { y = v } })
    scroll.changed()
    expect(scroll.pinned()).toBe(true)
    y = 600
    scroll.moved()
    expect(scroll.pinned()).toBe(false)
    height = 2600
    scroll.changed()
    expect(y).toBe(600)
    scroll.toBottom()
    expect(y).toBe(2100)
    expect(scroll.pinned()).toBe(true)
    height = 3000
    scroll.changed()
    expect(y).toBe(2500)
  })
})

describe('keyboard inset', () => {
  it('is the part of the layout viewport the keyboard covers', () => {
    expect(keyboardInset(800, { height: 500, offsetTop: 0 })).toBe(300)
    expect(keyboardInset(800, { height: 800, offsetTop: 0 })).toBe(0)
    expect(keyboardInset(800, null)).toBe(0)
  })

  it('writes --keyboard-inset on resize and removes it on detach', () => {
    const listeners = new Map<string, () => void>()
    const viewport = { height: 800, offsetTop: 0, addEventListener: (type: string, fn: () => void) => { listeners.set(type, fn) }, removeEventListener: (type: string) => { listeners.delete(type) } }
    const props = new Map<string, string>()
    const root = { style: { setProperty: (k: string, v: string) => { props.set(k, v) }, removeProperty: (k: string) => { props.delete(k); return '' } } }
    const detach = trackKeyboardInset({ innerHeight: 800, visualViewport: viewport as never }, root as never)
    expect(props.get('--keyboard-inset')).toBe('0px')
    viewport.height = 460
    listeners.get('resize')!()
    expect(props.get('--keyboard-inset')).toBe('340px')
    detach()
    expect(listeners.size).toBe(0)
    expect(props.has('--keyboard-inset')).toBe(false)
  })
})

describe('composer send and voice helpers', () => {
  it('keeps text typed while a send was in flight (UI-02)', () => {
    expect(remainingAfterSend('fix it', 'fix it')).toBe('')
    expect(remainingAfterSend('fix it and more', 'fix it')).toBe('and more')
    expect(remainingAfterSend('rewritten', 'fix it')).toBe('rewritten')
  })

  it('appends a transcript to the latest text (UI-10)', () => {
    expect(withTranscript('', 'hello')).toBe('hello')
    expect(withTranscript('typed', 'spoken')).toBe('typed spoken')
  })
})

describe('markdown fixes', () => {
  const html = (text: string): string => renderToStaticMarkup(<Markdown text={text} />).replaceAll('<!-- -->', '')

  it('leaves globs and paths alone but still renders emphasis (UI-11)', () => {
    expect(html('match src/**/*.ts and a*b*c files')).not.toContain('<em>')
    expect(html('match src/**/*.ts and a*b*c files')).toContain('src/**/*.ts')
    expect(html('this is *important* and **bold**')).toContain('<em>important</em>')
    expect(html('this is *important* and **bold**')).toContain('<strong>bold</strong>')
  })

  it('opens a fence whose info string has spaces, and closes only with a matching marker (UI-12)', () => {
    const out = html('```ts title="a b"\nconst x = 1\n```\n\nafter')
    expect(out).toContain('<pre class="md-code" data-lang="ts"><code>const x = 1</code></pre>')
    expect(out).toContain('<p>after</p>')
    expect(html('````\n```\ninner\n````\nafter')).toContain('<code>```\ninner</code>')
  })
})

describe('XERXES tab rows and cards', () => {
  it('labels Hub notices as Autopilot / Watch / Notice, never "You" (XT-4)', () => {
    expect(supervisorAuthor({ role: 'system', text: 'Autopilot did: steer' })).toBe('Autopilot')
    expect(supervisorAuthor({ role: 'system', text: 'Watch: woke for loop' })).toBe('Watch')
    expect(supervisorAuthor({ role: 'system', text: 'Approved a permission request' })).toBe('Notice')
    expect(supervisorAuthor({ role: 'user', text: 'hi' })).toBe('You')
    expect(supervisorAuthor({ role: 'user', text: 'hi', author: 'Ann' })).toBe('Ann')
  })

  it('says an auto card is still running instead of "Done automatically" (XT-9)', () => {
    const running = plain(renderToStaticMarkup(<ConfirmationCard item={{ id: 'a', tool: 'task.stop', summary: 'Stop it', targets: [], state: 'confirming', auto: true, confirmedBy: 'Ann' }} canAct onRespond={noop} />))
    expect(running).toContain('XERXES is doing: task.stop')
    expect(running).toContain('Working…')
    expect(running).not.toContain('Done automatically')
  })

  it('keeps a Try again button on a failed card (XT-7 UI half)', () => {
    const failed = plain(renderToStaticMarkup(<ConfirmationCard item={{ id: 'c', tool: 'task.stop', summary: 'Stop it', targets: [], state: 'failed', message: 'Hub busy' }} canAct onRespond={noop} />))
    expect(failed).toContain('Hub busy')
    expect(failed).toContain('>Try again<')
  })
})

describe('inbox filter memory and model chooser errors', () => {
  it('keeps the Inbox filter per app across screen changes (UI-14)', () => {
    const owner = {}
    expect(rememberedInboxFilter(owner)).toEqual({ status: 'all' })
    rememberInboxFilter(owner, { status: 'failed', workspaceId: 'w1' })
    expect(rememberedInboxFilter(owner)).toEqual({ status: 'failed', workspaceId: 'w1' })
    expect(rememberedInboxFilter({})).toEqual({ status: 'all' })
  })

  it('shows the machine-list error with Try again instead of an empty picker (XT-11)', () => {
    const base = withSupervisorConfig(liveState(), { phase: 'ready', status: 'unconfigured', canEdit: true })
    const state = { ...base, machines: { phase: 'error' as const, list: [], error: 'Hub unavailable' } }
    const html = plain(renderToStaticMarkup(<ServicesProvider services={makeServices(state, '/xerxes')}><ModelChooser /></ServicesProvider>))
    expect(html).toContain('Could not load the machines: Hub unavailable')
    expect(html).toContain('>Try again<')
    expect(html).not.toContain('<select')
  })

  it('offers Retry in the banner only while the link is down (C1 UI half)', () => {
    const down = { ...liveState(), connection: { state: 'reconnecting' as const, attempt: 2 } }
    const controller = stubController({ resync: async () => {} })
    expect(plain(render(down, '/', { controller }))).toContain('>Retry<')
    expect(plain(render(liveState(), '/', { controller }))).not.toContain('>Retry<')
  })
})

describe('connection retry', () => {
  it('uses reconnectNow when present, else resync, else offers nothing', () => {
    const calls: string[] = []
    reconnectOf({ reconnectNow: () => calls.push('now') })?.()
    reconnectOf({ resync: async () => { calls.push('resync') } })?.()
    expect(calls).toEqual(['now', 'resync'])
    expect(reconnectOf({})).toBeUndefined()
  })
})
