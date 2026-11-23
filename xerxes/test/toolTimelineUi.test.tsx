// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { ToolTimelineGroup, summarizeToolTimeline } from '../src/desktop/renderer/ToolTimeline.js'
import { SwarmToolCard, isSwarmToolName, swarmSpawnSummaryOf } from '../src/desktop/renderer/SwarmToolCard.js'
import {
  FileToolRow,
  fileToolOf,
  fileToolStats,
  parseOutlineSymbols,
  parsePatchDiffs,
} from '../src/desktop/renderer/FileToolRow.js'
import { ToolCallRow, ToolCallsSection, classifyToolItem } from '../src/desktop/renderer/Execution.js'
import {
  SideQuestionModelPicker,
  SideQuestionPanel,
  buildSideQuestionModelChoices,
} from '../src/desktop/renderer/SideQuestion.js'
import { Store, type Snapshot } from '../src/desktop/renderer/store.js'
import {
  XerxesBrandEmblem,
  XerxesStatusMark,
  XerxesThinkingMark,
} from '../src/desktop/renderer/XerxesSprites.js'
import { SideQuestionCard } from '../src/vscode/ui/SideQuestionCard.js'
import type { Block, ToolItem } from '../src/desktop/renderer/types.js'

const makeTool = (
  name: string,
  input: Record<string, unknown> | string = {},
  overrides: Partial<ToolItem> = {},
): ToolItem => ({
  id: overrides.id ?? `call-${name.toLowerCase()}`,
  verb: overrides.verb ?? name,
  arg: overrides.arg ?? '',
  dur: overrides.dur ?? '0.4s',
  state: overrides.state ?? 'done',
  name,
  input: typeof input === 'string' ? input : JSON.stringify(input),
  output: overrides.output ?? '',
  ...(overrides.error ? { error: overrides.error } : {}),
  ...(overrides.path ? { path: overrides.path } : {}),
  ...(overrides.diff ? { diff: overrides.diff } : {}),
})

describe('ToolTimelineGroup', () => {
  test('renders borderless inline thinking mark and 1–3 item live rolling preview strip while running and collapsed', () => {
    const blocks: Block[] = [
      { kind: 'thinking', id: 1, text: '**Updating SwarmBoard**', streaming: false },
      {
        kind: 'tools',
        id: 2,
        running: true,
        items: [
          makeTool('ReadFile', { file_path: 'src/SwarmBoard.tsx' }, { id: 't1', dur: '0.2s', state: 'done', output: 'export const x = 1\n' }),
          makeTool('FileEditTool', { file_path: 'src/SwarmBoard.tsx', old_string: 'a\n', new_string: 'b\nc\n' }, { id: 't2', state: 'working', path: 'src/SwarmBoard.tsx' }),
          makeTool('exec_command', { cmd: 'bun', args: ['test', 'xerxes/test/swarm.test.ts'] }, { id: 't3', state: 'working' }),
        ],
      },
    ]

    const html = renderToStaticMarkup(createElement(ToolTimelineGroup, { blocks, active: true, turnActive: true }))
    expect(html).toContain('class="tool-timeline-wrap tool-timeline is-running"')
    expect(html).toContain('<details class="activity-group is-running">')
    expect(html).not.toContain('<details class="activity-group is-running" open')
    expect(html).toContain('Working')
    expect(html).toContain('tool-timeline__mark')
    expect(html).toContain('tool-timeline__live-strip')
    expect(html).toContain('SwarmBoard.tsx')
    expect(html).toContain('$ bun test xerxes/test/swarm.test.ts')
    // Collapsed disclosure stays plain & inline — step count lives in title tooltip and expanded toolbar, not on the collapsed bar.
    expect(html).toContain('4 steps')
    expect(html).not.toContain('tool-timeline__status-icon--done')
    expect(html).not.toContain('tool-timeline__metrics')
    expect(html).not.toContain('tool-timeline__pill--failed')
  })

  test('renders plain borderless settled headline (zero checkmark, zero pill, rich tooltip) and full stats in expanded toolbar', () => {
    const blocks: Block[] = [
      {
        kind: 'tools',
        id: 10,
        running: false,
        items: [
          makeTool('FileEditTool', { file_path: 'src/a.ts', old_string: 'one\ntwo\n', new_string: 'one\nTWO\nthree\nfour\n' }, { id: 'e1', dur: '0.5s' }),
          makeTool('WriteFile', { file_path: 'src/b.ts', content: 'line1\nline2\nline3\n' }, { id: 'e2', dur: '0.3s' }),
          makeTool('exec_command', { command: 'bun run check' }, { id: 'c1', dur: '1.2s', output: JSON.stringify({ stdout: 'ok', exitCode: 0 }) }),
          makeTool('exec_command', { command: 'bun test' }, { id: 'c2', dur: '2.0s', state: 'failed', output: JSON.stringify({ stderr: 'boom', exitCode: 1 }) }),
          makeTool('ReadFile', { file_path: 'src/c.ts' }, { id: 'r1', dur: '0.2s', output: 'abc' }),
        ],
      },
    ]

    const stats = summarizeToolTimeline(blocks, false)
    // e1: +3 −1, e2: +3 −0 => +6 −1
    expect(stats.adds).toBe(6)
    expect(stats.dels).toBe(1)
    expect(stats.stepCount).toBe(5)
    expect(stats.failures).toBe(1)
    expect(stats.durationLabel).toBe('4.2s')
    expect(stats.headline).toContain('Edited 2 files')
    expect(stats.headline).toContain('Ran 2 commands')
    expect(stats.headline).toContain('Read 1 file')

    // Collapsed view: quiet Claude Code inline line, zero checkmark/warning icon, zero pill box, rich title tooltip
    const collapsedHtml = renderToStaticMarkup(createElement(ToolTimelineGroup, { blocks, active: false, turnActive: false }))
    expect(collapsedHtml).toContain('Edited 2 files · Ran 2 commands · Read 1 file')
    expect(collapsedHtml).toContain('1 failed')
    expect(collapsedHtml).toContain('tool-timeline__failed')
    expect(collapsedHtml).toContain('+6 −1 · 5 steps · 4.2s · 1 failed')
    expect(collapsedHtml).not.toContain('tool-timeline__status-icon--done')
    expect(collapsedHtml).not.toContain('tool-timeline__status-icon--failed')
    expect(collapsedHtml).not.toContain('tool-timeline__pill--failed')
    expect(collapsedHtml).not.toContain('tool-timeline__metrics')
    expect(collapsedHtml).not.toContain('tool-timeline__live-strip')

    // Clean settled group with zero failures also renders zero checkmark / status icons
    const cleanBlocks: Block[] = [
      {
        kind: 'tools',
        id: 11,
        running: false,
        items: [
          makeTool('FileEditTool', { file_path: 'src/a.ts', old_string: 'one\n', new_string: 'one\ntwo\n' }, { id: 'ok1', dur: '0.4s' }),
        ],
      },
    ]
    const cleanHtml = renderToStaticMarkup(createElement(ToolTimelineGroup, { blocks: cleanBlocks, active: false, turnActive: false }))
    expect(cleanHtml).toContain('Edited 1 file')
    expect(cleanHtml).not.toContain('tool-timeline__status-icon')
    expect(cleanHtml).not.toContain('M173.66,98.34') // checkCircle SVG path must never appear on settled group

    // Expanded view surfaces exact stats readout (+6 −1 · 5 steps · 4.2s) in the quiet toolbar
    const expandedHtml = renderToStaticMarkup(createElement(ToolTimelineGroup, { blocks, active: false, turnActive: false, defaultOpen: true }))
    expect(expandedHtml).toContain('tool-timeline__toolbar')
    expect(expandedHtml).toContain('>+6<')
    expect(expandedHtml).toContain('>−1<')
    expect(expandedHtml).toContain('5 steps')
    expect(expandedHtml).toContain('4.2s')
    expect(expandedHtml).toContain('Expand all diffs')
    expect(expandedHtml).toContain('>Collapse<')
  })

  test('renders standalone error-notice-only groups directly inline without Runtime error box, triangle, or 1 failed pill', () => {
    const shortError: Block[] = [
      { kind: 'notice', id: 50, error: true, text: 'Rate limit exceeded — try again in 30s.' },
    ]
    const shortHtml = renderToStaticMarkup(createElement(ToolTimelineGroup, { blocks: shortError, active: false, turnActive: false }))
    expect(shortHtml).toContain('tool-timeline__inline-error')
    expect(shortHtml).toContain('Rate limit exceeded — try again in 30s.')
    expect(shortHtml).not.toContain('Runtime error')
    expect(shortHtml).not.toContain('1 failed')
    expect(shortHtml).not.toContain('1 step')
    expect(shortHtml).not.toContain('<details')
    expect(shortHtml).not.toContain('tool-timeline__status-icon')
    expect(shortHtml).not.toContain('tool-timeline__pill--failed')

    const longError: Block[] = [
      { kind: 'notice', id: 51, error: true, text: 'Provider request failed (500)\nline 2\nline 3\nline 4\nstack trace detail' },
    ]
    const longHtml = renderToStaticMarkup(createElement(ToolTimelineGroup, { blocks: longError, active: false, turnActive: false }))
    expect(longHtml).toContain('tool-timeline__inline-error-fold')
    expect(longHtml).toContain('Provider request failed (500)')
    expect(longHtml).not.toContain('Runtime error')
    expect(longHtml).not.toContain('1 failed')
    expect(longHtml).not.toContain('tool-timeline__status-icon')
  })

  test('keeps saved ArtifactCards and Swarm spawn summaries visible outside a collapsed settled group', () => {
    const blocks: Block[] = [
      {
        kind: 'tools',
        id: 20,
        running: false,
        items: [
          makeTool(
            'swarm',
            { action: 'add', name: 'verifier', model: 'claude-sonnet-4-6', workspace: 'share', brief: 'Verify the tool timeline UI.' },
            {
              id: 'sw1',
              output: JSON.stringify({
                status: 'added',
                added: [{ name: 'verifier', role: 'worker', model: 'claude-sonnet-4-6', model_display_name: 'Claude Sonnet 4.6', workspace: 'share' }],
                brief_seqs: [12],
              }),
            },
          ),
          makeTool(
            'ArtifactTool',
            { action: 'create', kind: 'markdown' },
            { id: 'art1', output: 'Created artifact "UX Verification Report" (id rep123, version 1)' },
          ),
          makeTool('ReadFile', { file_path: 'secret-collapsed.ts' }, { id: 'r1', output: 'hidden' }),
        ],
      },
    ]

    const html = renderToStaticMarkup(createElement(ToolTimelineGroup, { blocks, active: false, turnActive: false }))
    expect(html).not.toContain('secret-collapsed.ts')
    expect(html).toContain('class="tool-timeline__outside"')
    expect(html).toContain('swarm-spawn-summary')
    expect(html).toContain('Spawned')
    expect(html).toContain('verifier')
    expect(html).toContain('share · worktree')
    expect(html).toContain('Open Immortals →')
    expect(html).toContain('class="artifact-card"')
    expect(html).toContain('UX Verification Report')
  })

  test('renders whisper-thin vertical timeline rail (zero dots on settled steps), reasoning, checkpoints, and bulk toolbar when expanded with > 4 tools', () => {
    const blocks: Block[] = [
      { kind: 'thinking', id: 1, text: '**Planning changes**\nInspecting renderer.', streaming: false },
      {
        kind: 'tools',
        id: 2,
        running: false,
        items: [
          makeTool('ReadFile', { file_path: 'src/a.ts' }, { id: 't1', output: 'const a = 1' }),
          makeTool('GrepTool', { pattern: 'ToolCallRow', path: 'xerxes/src' }, { id: 't2', output: 'a.ts:1\nb.ts:2\n' }),
          makeTool('FileEditTool', { file_path: 'src/a.ts', old_string: '1', new_string: '2' }, { id: 't3' }),
          makeTool('exec_command', { command: 'bun test' }, { id: 't4', output: JSON.stringify({ stdout: 'pass', exitCode: 0 }) }),
          makeTool('swarm', { action: 'status' }, { id: 't5', output: JSON.stringify({ head: 5, members: [{ name: 'coordinator', state: 'waiting' }] }) }),
        ],
      },
      { kind: 'checkpoint', id: 3, turn: 2, adds: 14, dels: 3 },
    ]

    const html = renderToStaticMarkup(createElement(ToolTimelineGroup, { blocks, active: false, turnActive: false, defaultOpen: true }))
    expect(html).toContain('tool-timeline__toolbar')
    expect(html).toContain('Expand all diffs')
    expect(html).toContain('>Collapse<')
    expect(html).toContain('tool-timeline__rail')
    // Normal settled steps carry zero cartoon dots/diamonds
    expect(html).not.toContain('tool-timeline__node')
    expect(html).toContain('data-kind="thinking"')
    expect(html).toContain('data-kind="tool"')
    expect(html).toContain('data-kind="checkpoint"')
    expect(html).toContain('turn 2 end')
    expect(html).toContain('Checked roster · 1 member')
    const immortalsStats = summarizeToolTimeline(
      [
        {
          kind: 'tools',
          id: 99,
          running: false,
          items: [
            makeTool('swarm', { action: 'status' }, { id: 'im1' }),
            makeTool('post_message', { text: '@all Ready' }, { id: 'im2' }),
          ],
        },
      ],
      false,
    )
    expect(immortalsStats.headline).toBe('Immortals · 2 actions')
  })
})

describe('SwarmToolCard', () => {
  test('isSwarmToolName matches all native Swarm tools case-insensitively and strips prefixes', () => {
    for (const name of ['swarm', 'Swarm', 'mcp.xerxes:swarm', 'post_message', 'tools.post_message', 'read_board', 'swarm_wait', 'swarm_overview', 'SwarmWaitTool']) {
      expect(isSwarmToolName(name)).toBe(true)
    }
    for (const other of ['ReadFile', 'exec_command', 'GrepTool', 'ArtifactTool', 'send_message']) {
      expect(isSwarmToolName(other)).toBe(false)
    }
  })

  test('renders swarm add (single & sharded group) with Simurgh sprite, model pill, workspace badge, brief, and Open Swarm button', () => {
    const single = makeTool(
      'swarm',
      { action: 'add', name: 'tool-ux', model: 'claude-opus-4-6', workspace: 'share', brief: 'Overhaul @desktop-wire tool groups cleanly.' },
      {
        output: JSON.stringify({
          status: 'added',
          added: [{ name: 'tool-ux', role: 'worker', model: 'claude-opus-4-6', model_display_name: 'Claude Opus 4.6', workspace: 'share' }],
          brief_seqs: [136],
        }),
      },
    )
    const html = renderToStaticMarkup(createElement(SwarmToolCard, { item: single }))
    expect(html).toContain('Spawned')
    expect(html).toContain('tool-ux')
    expect(html).toContain('Claude Opus 4.6')
    expect(html).toContain('share · worktree')
    expect(html).toContain('#136')
    expect(html).toContain('swarm-tool__mention')
    expect(html).toContain('@desktop-wire')
    expect(html).toContain('Open Immortals →')
    expect(html).toContain('<svg')
    expect(swarmSpawnSummaryOf(single)?.displayTarget).toBe('tool-ux')

    const group = makeTool(
      'swarm',
      { action: 'add', name: 'worker', n: 3, workspace: 'branch', brief: 'Run parallel shard tests' },
      { output: JSON.stringify({ status: 'added', added: [{ name: 'worker-1', workspace: 'branch' }, { name: 'worker-2', workspace: 'branch' }, { name: 'worker-3', workspace: 'branch' }] }) },
    )
    const groupHtml = renderToStaticMarkup(createElement(SwarmToolCard, { item: group }))
    expect(groupHtml).toContain('worker-1..3')
    expect(groupHtml).toContain('branch · clone')
  })

  test('renders swarm model, kill, title, status, models, and channel actions cleanly', () => {
    const modelHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool(
        'swarm',
        { action: 'model', name: 'tool-ux', model: 'claude-sonnet-4-6', reason: 'Quota exhausted on opus' },
        { output: JSON.stringify({ status: 'switched', name: 'tool-ux', previous_model: 'claude-opus-4-6', model: 'claude-sonnet-4-6', model_display_name: 'Claude Sonnet 4.6', resumed: true, reason: 'Quota exhausted on opus' }) },
      ),
    }))
    expect(modelHtml).toContain('Switched')
    expect(modelHtml).toContain('@tool-ux')
    expect(modelHtml).toContain('Claude Sonnet 4.6')
    expect(modelHtml).toContain('Quota exhausted on opus')

    const killHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool('swarm', { action: 'kill', name: 'old-worker' }, { output: JSON.stringify({ status: 'killed', killed: ['old-worker'] }) }),
    }))
    expect(killHtml).toContain('Retired')
    expect(killHtml).toContain('@old-worker')
    expect(killHtml).toContain('retired')

    const titleHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool('swarm', { action: 'title', title: 'Tool UX Wave' }, { output: JSON.stringify({ status: 'titled', title: 'Tool UX Wave' }) }),
    }))
    expect(titleHtml).toContain('Named Immortals')
    expect(titleHtml).toContain('Tool UX Wave')

    const statusHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool('swarm', { action: 'status' }, { output: JSON.stringify({ head: 24, members: [{ name: 'coordinator', state: 'waiting' }, { name: 'tool-ux', state: 'working' }] }) }),
    }))
    expect(statusHtml).toContain('Checked roster · 2 members')
    expect(statusHtml).toContain('1 working')

    const modelsHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool('swarm', { action: 'models', all: true }, { output: JSON.stringify({ models: [{ id: 'a' }, { id: 'b' }, { id: 'c' }] }) }),
    }))
    expect(modelsHtml).toContain('Listed 3 models')

    const channelHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool('swarm', { action: 'channel', op: 'create', channel: 'ui-review', topic: 'Visual QA' }, { output: JSON.stringify({ status: 'created', channel: { name: 'ui-review', topic: 'Visual QA' } }) }),
    }))
    expect(channelHtml).toContain('Channel')
    expect(channelHtml).toContain('#ui-review')
    expect(channelHtml).toContain('Visual QA')
  })

  test('renders post_message, read_board, swarm_wait, and swarm_overview with semantic pills and expandable JSON', () => {
    const postHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool(
        'post_message',
        { text: '@desktop-wire @graph-verifier [done] Tool timeline shipped cleanly (#136)' },
        { output: JSON.stringify({ status: 'posted', seq: 145, channel: 'main', woken: ['desktop-wire'] }) },
      ),
      defaultExpanded: true,
    }))
    expect(postHtml).toContain('#145')
    expect(postHtml).toContain('swarm-tool__mention')
    expect(postHtml).toContain('@desktop-wire')
    expect(postHtml).toContain('swarm-tool__done-tag')
    expect(postHtml).toContain('Open Immortals →')
    expect(postHtml).toContain('Raw JSON')

    const readMsgsHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool('read_board', { since: 130 }, { output: JSON.stringify({ head: 138, messages: [{ seq: 136, agent: 'coordinator', text: 'Go' }, { seq: 137, agent: 'acp-bridge', text: 'Working' }] }) }),
    }))
    expect(readMsgsHtml).toContain('Read board')
    expect(readMsgsHtml).toContain('#130 → #138')
    expect(readMsgsHtml).toContain('2 messages')

    const readRosterHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool('read_board', { status: true }, { output: JSON.stringify({ head: 138, active: true, title: 'Swarm', members: [{ name: 'coordinator' }, { name: 'tool-ux' }, { name: 'brand-sprites' }] }) }),
    }))
    expect(readRosterHtml).toContain('Roster status · 3 members')

    const waitSleepingHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool('swarm_wait', { since: 145 }, { state: 'working' }),
    }))
    expect(waitSleepingHtml).toContain('Waiting for board activity')
    expect(waitSleepingHtml).toContain('#145')

    const waitWokenHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool('swarm_wait', { since: 145 }, { output: JSON.stringify({ status: 'woken', wake_seq: 149, head: 149, message: '@tool-ux please verify' }) }),
    }))
    expect(waitWokenHtml).toContain('Woken by')
    expect(waitWokenHtml).toContain('#149')

    const overviewHtml = renderToStaticMarkup(createElement(SwarmToolCard, {
      item: makeTool('swarm_overview', { summary: 'All 4 UI waves converging', markdown: '## Wave\n- Tool UX done' }, { output: JSON.stringify({ status: 'updated', head: 150 }) }),
    }))
    expect(overviewHtml).toContain('Updated live overview')
    expect(overviewHtml).toContain('All 4 UI waves converging')
  })

  test('ToolCallRow routes Swarm tools directly to SwarmToolCard instead of generic row', () => {
    const html = renderToStaticMarkup(createElement(ToolCallRow, {
      item: makeTool('post_message', { text: '@all Ready for verification' }, { output: JSON.stringify({ status: 'posted', seq: 9 }) }),
      label: 'Post message',
    }))
    expect(html).toContain('class="swarm-tool"')
    expect(html).not.toContain('execution-row')
  })
})

describe('FileToolRow (Read, Edit, Write, ApplyPatchTool, FileOutlineTool)', () => {
  test('parses and renders ApplyPatchTool V4A multi-file and unified diff hunks with +N −M badges', () => {
    const v4aPatch = [
      '*** Begin Patch',
      '*** Update File: xerxes/src/desktop/renderer/Execution.tsx',
      '@@ export function ToolCallRow',
      ' const x = 1',
      '-const old = true',
      '+const next = true',
      '+const extra = 2',
      '*** Add File: xerxes/src/desktop/renderer/NewCard.tsx',
      '+export const NewCard = () => null',
      '*** End Patch',
    ].join('\n')

    const parsed = parsePatchDiffs(v4aPatch)
    expect(parsed).toHaveLength(2)
    expect(parsed[0]).toMatchObject({ path: 'xerxes/src/desktop/renderer/Execution.tsx', action: 'update', adds: 2, dels: 1 })
    expect(parsed[1]).toMatchObject({ path: 'xerxes/src/desktop/renderer/NewCard.tsx', action: 'add', adds: 1, dels: 0 })

    const patchItem = makeTool('ApplyPatchTool', { patch: v4aPatch })
    const tool = fileToolOf(patchItem)
    expect(tool?.kind).toBe('patch')
    expect(fileToolStats(patchItem)).toEqual({ adds: 3, dels: 1 })

    const html = renderToStaticMarkup(createElement(FileToolRow, { item: patchItem, tool: tool!, failed: false }))
    expect(html).toContain('>Patch<')
    expect(html).toContain('Execution.tsx')
    expect(html).toContain('+1 file')
    expect(html).toContain('>+3<')
    expect(html).toContain('>−1<')
    expect(html).toContain('filetool__line--add')
    expect(html).toContain('filetool__line--del')

    const unifiedPatch = [
      '--- a/src/foo.ts',
      '+++ b/src/foo.ts',
      '@@ -10,3 +10,3 @@ export function foo()',
      ' const a = 1',
      '-return a',
      '+return a + 1',
    ].join('\n')
    const uniItem = makeTool('apply_patch', { patch: unifiedPatch })
    const uniTool = fileToolOf(uniItem)
    expect(uniTool?.kind).toBe('patch')
    expect(fileToolStats(uniItem)).toEqual({ adds: 1, dels: 1 })
  })

  test('parses and renders FileOutlineTool with clickable path, symbol count pill, and expandable symbol list', () => {
    const outlineOutput = [
      '# xerxes/src/desktop/renderer/Execution.tsx (typescript, 180 lines)',
      'imports [4-14] (11): react, ./types.js, ./Icon.js',
      '[23-31] export function toolHasFailed(item: ToolItem): boolean',
      '[35-49] export function executionView(item: ToolItem)',
      '[71-85] export const ToolCallRow',
      '  [76-80] method renderBody()',
    ].join('\n')

    const parsed = parseOutlineSymbols(outlineOutput)
    expect(parsed.symbols).toHaveLength(4)
    expect(parsed.symbols[0]).toMatchObject({ kind: 'function', exported: true, lineRange: '23–31', depth: 0 })
    expect(parsed.symbols[3]).toMatchObject({ kind: 'method', exported: false, depth: 1 })

    const item = makeTool('FileOutlineTool', { file_path: 'xerxes/src/desktop/renderer/Execution.tsx' }, { output: outlineOutput })
    const tool = fileToolOf(item)
    expect(tool).toEqual({ kind: 'outline', path: 'xerxes/src/desktop/renderer/Execution.tsx', exportedOnly: false })

    const html = renderToStaticMarkup(createElement(FileToolRow, { item, tool: tool!, failed: false, defaultExpanded: true }))
    expect(html).toContain('>Outline<')
    expect(html).toContain('xerxes/src/desktop/renderer/')
    expect(html).toContain('Execution.tsx')
    expect(html).toContain('4 symbols')
    expect(html).toContain('filetool__symbols')
    expect(html).toContain('toolHasFailed')
    expect(html).toContain('L23–31')
  })

  test('renders ReadFile, FileEditTool, and WriteFile with basename/dir split and diff chrome', () => {
    const readItem = makeTool('ReadFile', { file_path: 'xerxes/src/cli.ts', offset: 10 }, { output: 'line 1\nline 2\n' })
    const readHtml = renderToStaticMarkup(createElement(FileToolRow, { item: readItem, tool: fileToolOf(readItem)!, failed: false }))
    expect(readHtml).toContain('>Read<')
    expect(readHtml).toContain('filetool__dir')
    expect(readHtml).toContain('cli.ts')
    expect(readHtml).toContain('lines 11–12')

    const writeItem = makeTool('WriteFile', { file_path: 'docs/guide.md', content: '# Guide\nBody\n' })
    const writeHtml = renderToStaticMarkup(createElement(FileToolRow, { item: writeItem, tool: fileToolOf(writeItem)!, failed: false }))
    expect(writeHtml).toContain('>Write<')
    expect(writeHtml).toContain('guide.md')
    expect(writeHtml).toContain('>+2<')
  })
})

describe('Semantic Tool Formatting (Shell, Search, Web, Skill, MCP, Memory, Schedule, Monitor)', () => {
  test('formats Shell, Search (Grep/Glob/ListDir/LSP), Web, Skill, MCP, Memory, Schedule, and Monitor without raw JSON', () => {
    const shell = makeTool('bash', { command: 'bun test xerxes/test/toolTimelineUi.test.tsx' }, { output: JSON.stringify({ stdout: '12 pass', exitCode: 0 }) })
    const shellHtml = renderToStaticMarkup(createElement(ToolCallRow, { item: shell, label: 'Bash' }))
    expect(shellHtml).toContain('data-category="shell"')
    expect(shellHtml).toContain('$ ')
    expect(shellHtml).toContain('bun test xerxes/test/toolTimelineUi.test.tsx')
    expect(shellHtml).toContain('exit 0')

    const grep = makeTool('GrepTool', { pattern: 'ToolTimelineGroup', path: 'xerxes/src' }, { arg: '{"pattern":"ToolTimelineGroup"}', output: 'a.ts:1\nb.ts:2\nc.ts:3\n' })
    const grepHtml = renderToStaticMarkup(createElement(ToolCallRow, { item: grep, label: 'Grep' }))
    expect(grepHtml).toContain('data-category="search"')
    expect(grepHtml).toContain('>Search<')
    expect(grepHtml).toContain('ToolTimelineGroup')
    expect(grepHtml).toContain('xerxes/src')
    expect(grepHtml).toContain('3 matches')
    expect(grepHtml).not.toContain('{"pattern"')

    const glob = makeTool('GlobTool', { pattern: '**/*.tsx', path: 'xerxes/src/desktop' }, { output: JSON.stringify({ files: ['a.tsx', 'b.tsx'] }) })
    const globHtml = renderToStaticMarkup(createElement(ToolCallRow, { item: glob, label: 'Glob' }))
    expect(globHtml).toContain('>Find files<')
    expect(globHtml).toContain('**/*.tsx')
    expect(globHtml).toContain('2 files')

    const webSearch = makeTool('web_search', { query: 'bun sqlite fts5', domain: 'bun.sh' }, { output: JSON.stringify({ results: [{ title: '1' }, { title: '2' }] }) })
    const webHtml = renderToStaticMarkup(createElement(ToolCallRow, { item: webSearch, label: 'Web search' }))
    expect(webHtml).toContain('data-category="web"')
    expect(webHtml).toContain('>Web search<')
    expect(webHtml).toContain('bun sqlite fts5')
    expect(webHtml).toContain('bun.sh')

    const webFetch = makeTool('web_fetch', { url: 'https://docs.anthropic.com/en/docs/claude-code' })
    expect(classifyToolItem(webFetch)).toMatchObject({ category: 'web', badge: 'Web fetch', target: 'docs.anthropic.com/en/docs/claude-code' })

    const skill = makeTool('skill_manage', { skill: 'swarm-coordinator', action: 'load' })
    const skillHtml = renderToStaticMarkup(createElement(ToolCallRow, { item: skill, label: 'Skill manage' }))
    expect(skillHtml).toContain('data-category="skill"')
    expect(skillHtml).toContain('>Skill<')
    expect(skillHtml).toContain('swarm-coordinator')

    const mcp = makeTool('mcp.github:list_pull_requests', { repo: 'erfanzar/xerxes-agents' })
    const mcpHtml = renderToStaticMarkup(createElement(ToolCallRow, { item: mcp, label: 'List pull requests' }))
    expect(mcpHtml).toContain('data-category="mcp"')
    expect(mcpHtml).toContain('>MCP<')
    expect(mcpHtml).toContain('github')
    expect(mcpHtml).toContain('List pull requests · erfanzar/xerxes-agents')

    const memory = makeTool('recall_memory', { query: 'desktop css tokens' })
    expect(classifyToolItem(memory)).toMatchObject({ category: 'memory', badge: 'Memory', target: 'desktop css tokens' })

    const schedule = makeTool('manage_schedule', { cron: '*/5 * * * *', name: 'build-check' })
    expect(classifyToolItem(schedule)).toMatchObject({ category: 'schedule', badge: 'Schedule', target: '*/5 * * * * · build-check' })

    const monitor = makeTool('monitor_file', { file_path: 'xerxes/src/cli.ts' })
    expect(classifyToolItem(monitor)).toMatchObject({ category: 'monitor', badge: 'Monitor', target: 'xerxes/src/cli.ts' })

    // Unknown tool with raw JSON in item.arg still never dumps raw JSON into summary
    const unknown = makeTool('custom_widget', { target: 'dashboard', count: 3 }, { arg: '{"target":"dashboard","count":3}' })
    const unknownInfo = classifyToolItem(unknown)
    expect(unknownInfo.target).toBe('dashboard')
    expect(unknownInfo.target).not.toContain('{"target"')
  })

  test('ToolCallsSection shows aggregate diff stats and live active-step preview while collapsed', () => {
    const items: ToolItem[] = [
      makeTool('FileEditTool', { file_path: 'src/a.ts', old_string: 'x\n', new_string: 'y\nz\n' }, { id: 'f1' }),
      makeTool('exec_command', { command: 'bun run test' }, { id: 'c1', state: 'working' }),
    ]
    const html = renderToStaticMarkup(createElement(ToolCallsSection, { items }))
    expect(html).toContain('<details class="toolcalls">')
    expect(html).toContain('>2<')
    expect(html).toContain('>+2<')
    expect(html).toContain('>−1<')
    expect(html).toContain('1 running')
    expect(html).toContain('tool-timeline__live-strip')
    expect(html).toContain('$ bun run test')
  })

  test('layout.css provides dynamic theme tokens with zero hardcoded purple/cyan accents, borderless Claude Code disclosures, and prefers-reduced-motion', () => {
    const css = readFileSync(join(import.meta.dir, '../src/desktop/renderer/layout.css'), 'utf8')
    expect(css).toContain('--tx-accent:var(--x-accent);')
    expect(css).toContain('--tx-violet:var(--tx-accent);')
    expect(css).toContain('--tx-cyan:var(--x-activity);')
    expect(css).toContain('--tx-emerald:var(--x-done);')
    expect(css).toContain('--tx-rose:var(--x-failed);')
    expect(css).toContain('--tx-amber:var(--x-needs);')
    expect(css).not.toMatch(/#a855f7|#8b5cf6|#7e22ce|rgba\(168,\s*85,\s*247|#38bdf8|#0284c7/i)
    // Borderless Claude Code ToolTimelineGroup container & inline disclosure header
    expect(css).toContain('.tool-timeline-wrap>.activity-group{margin:0;padding:0;border:0;border-radius:0;background:transparent;box-shadow:none;overflow:visible}')
    expect(css).toContain('summary.tool-timeline__header{display:inline-flex;align-items:center;justify-content:flex-start;gap:6px;min-height:24px;max-width:100%;padding:3px 0;margin:0;border:0;border-radius:0;background:transparent;')
    expect(css).not.toContain('.tool-timeline__status-icon--done')
    expect(css).not.toContain('.tool-timeline__pill--failed')
    expect(css).toContain('.tool-timeline__working{display:inline;font-weight:500;color:inherit}')
    expect(css).toContain('@keyframes live-text-shine')
    expect(css).toContain('.tool-timeline__meta{display:inline-flex;align-items:center;gap:5px;padding:0;border:0;background:transparent;')
    expect(css).toContain('.tool-timeline-wrap>.activity-group>.tool-timeline__body{margin:4px 0 8px 4px;padding:2px 0 2px 14px;border:0;border-left:1px solid var(--x-hairline);border-radius:0;background:transparent;')
    expect(css).toContain('.tool-timeline__live-strip{display:flex;flex-direction:column;gap:2px;margin:2px 0 4px 4px;padding:2px 0 2px 14px;border:0;border-left:1px solid var(--x-hairline);border-radius:0;background:transparent}')
    expect(css).toContain('.swarm-spawn-summary{display:flex;align-items:center;flex-wrap:wrap;gap:6px 8px;padding:8px 12px;border:1px solid var(--x-hairline);border-radius:10px;background:var(--x-card);')
    expect(css).toContain('.swarm-spawn-summary button.swarm-tool__open{margin-left:auto;flex:none}')
    expect(css).toContain('.execution-row__main{display:flex;align-items:center;flex-wrap:nowrap;')
    expect(css).toContain('.execution-row[open][data-command] .execution-row__target')
    expect(css).toContain('.tool-timeline__rail')
    expect(css).toContain('.swarm-tool{display:flex;flex-direction:column;gap:4px;padding:2px 0;border:0;border-radius:0;background:transparent;')
    expect(css).toContain('.swarm-tool__mention')
    expect(css).toContain(":root[data-theme='light']:not(#layout):not(#tool-ux)")
    expect(css).toContain('@media (prefers-reduced-motion:reduce)')
  })

  test('XerxesThinkingMark, XerxesStatusMark, and Hero XerxesBrandEmblem adapt 100% dynamically to CSS theme tokens', () => {
    const HEX_RE = /#[0-9a-fA-F]{6}\b/g
    const thinkingActive = renderToStaticMarkup(createElement(XerxesThinkingMark, { size: 15, active: true }))
    expect(thinkingActive).toContain('var(--x-accent')
    expect(thinkingActive).toContain('var(--x-working')
    expect(thinkingActive.match(HEX_RE) ?? []).toEqual([])

    const compactWorking = renderToStaticMarkup(createElement(XerxesStatusMark, { state: 'working', size: 20, live: true }))
    expect(compactWorking).toContain('var(--x-accent')
    expect(compactWorking).toContain('var(--x-working')
    expect(compactWorking.match(HEX_RE) ?? []).toEqual([])

    const compactDone = renderToStaticMarkup(createElement(XerxesStatusMark, { state: 'done', size: 18, live: false }))
    expect(compactDone).toContain('var(--x-done')
    expect(compactDone.match(HEX_RE) ?? []).toEqual([])

    const compactFailed = renderToStaticMarkup(createElement(XerxesStatusMark, { state: 'failed', size: 18, live: true }))
    expect(compactFailed).toContain('var(--x-failed')
    expect(compactFailed.match(HEX_RE) ?? []).toEqual([])

    const compactWaiting = renderToStaticMarkup(createElement(XerxesStatusMark, { state: 'approval', size: 18, live: true }))
    expect(compactWaiting).toContain('var(--x-needs')
    expect(compactWaiting.match(HEX_RE) ?? []).toEqual([])

    // Hero Xerxes mark adapts to theme tokens across dark & light palettes
    const heroEmblem = renderToStaticMarkup(createElement(XerxesBrandEmblem, { size: 48, animate: true }))
    expect(heroEmblem).toContain('var(--x-title')
    expect(heroEmblem).toContain('var(--x-hairline')
    expect(heroEmblem.match(HEX_RE) ?? []).toEqual([])

    const regalMark = renderToStaticMarkup(createElement(XerxesStatusMark, { state: 'working', size: 48, live: true }))
    expect(regalMark).toContain('var(--x-working')
    expect(regalMark).toContain('var(--x-hairline')
    expect(regalMark.match(HEX_RE) ?? []).toEqual([])
  })
})

describe('Side Question Vibes Model Popover (Desktop & VS Code)', () => {
  const sampleModels = [
    { id: 'claude-opus-4-6', provider: 'Anthropic', label: 'Claude Opus 4.6' },
    { id: 'claude-haiku-4-5', provider: 'Anthropic', label: 'Claude Haiku 4.5' },
    { id: 'gemini-2.5-flash', provider: 'Google', label: 'Gemini 2.5 Flash' },
    { id: 'claude-opus-4-6', provider: 'Anthropic', label: 'Duplicate' },
  ]

  function makeSideQuestionSnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
    const base = new Store().getSnapshot()
    return {
      ...base,
      connection: 'online',
      cwd: '/workspace/xerxes',
      model: 'gpt-6.2-sol',
      models: sampleModels,
      sideQuestionModel: '',
      ...overrides,
    }
  }

  test('dedupes discovered models and preserves custom model selections', () => {
    const choices = buildSideQuestionModelChoices(sampleModels, 'openai/gpt-5-mini')
    expect(choices.map(c => c.id)).toEqual([
      'openai/gpt-5-mini',
      'claude-opus-4-6',
      'claude-haiku-4-5',
      'gemini-2.5-flash',
    ])
    expect(choices[0]).toMatchObject({ id: 'openai/gpt-5-mini', label: 'gpt-5-mini', provider: 'openai' })
  })

  test('renders sleek Vibes popover button (zero raw <select>) with Session default, models, search (>6), loading state, and custom ID input', () => {
    const closedHtml = renderToStaticMarkup(createElement(SideQuestionPanel, {
      snap: makeSideQuestionSnapshot(),
    }))
    expect(closedHtml).not.toContain('<select')
    expect(closedHtml).toContain('side-question__model-btn')
    expect(closedHtml).toContain('aria-label="Side question model"')
    expect(closedHtml).toContain('Session default (gpt-6.2-sol)')
    expect(closedHtml).not.toContain('btw-model-pop')

    const openHtml = renderToStaticMarkup(createElement(SideQuestionPanel, {
      snap: makeSideQuestionSnapshot({ sideQuestionModel: 'claude-haiku-4-5' }),
      defaultModelPickerOpen: true,
    }))
    expect(openHtml).not.toContain('<select')
    expect(openHtml).toContain('Claude Haiku 4.5')
    expect(openHtml).toContain('class="btw-model-pop"')
    expect(openHtml).toContain('Side question model')
    expect(openHtml).toContain('Session default')
    expect(openHtml).toContain('gpt-6.2-sol')
    expect(openHtml).toContain('Gemini 2.5 Flash')
    expect(openHtml).toContain('Anthropic')
    expect(openHtml).toContain('Google')
    expect(openHtml).toContain('placeholder="Custom model ID…"')
    // <= 6 models omits the search input
    expect(openHtml).not.toContain('btw-model-pop__filter')

    // > 6 models shows the inline search filter
    const manyModels = Array.from({ length: 8 }, (_, idx) => ({
      id: `model-${idx + 1}`,
      provider: 'Provider',
      label: `Model ${idx + 1}`,
    }))
    const searchHtml = renderToStaticMarkup(createElement(SideQuestionModelPicker, {
      models: manyModels,
      sessionModel: 'gpt-6.2-sol',
      defaultOpen: true,
    }))
    expect(searchHtml).toContain('btw-model-pop__filter')
    expect(searchHtml).toContain('placeholder="Filter models…"')

    // Loading state with empty models displays subtle Loading models… status
    const loadingHtml = renderToStaticMarkup(createElement(SideQuestionModelPicker, {
      models: [],
      modelsLoading: true,
      sessionModel: 'gpt-6.2-sol',
      defaultOpen: true,
    }))
    expect(loadingHtml).toContain('Loading models…')
    expect(loadingHtml).toContain('Session default')

    // VS Code SideQuestionCard uses the same custom popover with zero raw <select>
    const vscodeHtml = renderToStaticMarkup(createElement(SideQuestionCard, {
      snap: makeSideQuestionSnapshot(),
      entries: [],
      defaultModelPickerOpen: true,
      onEntriesChange: () => {},
      onClear: () => {},
      onClose: () => {},
    }))
    expect(vscodeHtml).not.toContain('<select')
    expect(vscodeHtml).toContain('xv-btw__model-btn')
    expect(vscodeHtml).toContain('class="btw-model-pop"')
    expect(vscodeHtml).toContain('Session default')
    expect(vscodeHtml).toContain('Claude Haiku 4.5')
    expect(vscodeHtml).toContain('Custom model ID…')
  })

  test('SideQuestion auto-loads models via store.loadModels() and atelier.css/vscode.css style hairline focus & portal layer', () => {
    const sideQuestionSrc = readFileSync(join(import.meta.dir, '../src/desktop/renderer/SideQuestion.tsx'), 'utf8')
    const vscodeCardSrc = readFileSync(join(import.meta.dir, '../src/vscode/ui/SideQuestionCard.tsx'), 'utf8')
    const atelierCss = readFileSync(join(import.meta.dir, '../src/desktop/renderer/atelier.css'), 'utf8')
    const vscodeCss = readFileSync(join(import.meta.dir, '../src/vscode/ui/vscode.css'), 'utf8')

    expect(sideQuestionSrc).toContain('store.loadModels(')
    expect(vscodeCardSrc).toContain('store.loadModels(')
    expect(atelierCss).toContain('.picker-layer>.btw-model-pop{pointer-events:auto;z-index:1}')
    expect(atelierCss).toContain('.side-question__model-btn:focus-visible{outline:none;box-shadow:0 0 0 1px var(--x-accent)}')
    expect(vscodeCss).toContain('.xv-btw__model-btn:focus-visible{outline:none;box-shadow:0 0 0 1px var(--x-accent)}')
    expect(vscodeCss).toContain(':root[data-host=\'vscode\'] .btw-model-pop{')
  })
})
