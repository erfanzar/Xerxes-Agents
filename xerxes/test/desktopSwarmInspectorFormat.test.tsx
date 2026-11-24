// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { SwarmBoardView } from '../src/desktop/renderer/SwarmBoard.js'
import {
  formatInspectorToolArgs,
  formatInspectorToolResult,
  stripAgyToolTimestampBoilerplate,
  stripInspectorSystemEnvelope,
} from '../src/desktop/renderer/swarmInspectorFormat.js'
import type { SwarmMemberInspectResult, SwarmStatusSnapshot } from '../src/desktop/renderer/swarmState.js'

describe('Immortals worker inspector formatting', () => {
  test('strips closed and unclosed <SYSTEM_MESSAGE> wrappers and classifies board wakes', () => {
    const unclosed = stripInspectorSystemEnvelope(
      '<SYSTEM_MESSAGE>\n[board] the latest message is #14 (you have read up to #13). Read anything you have missed.',
    )
    expect(unclosed.isSystemWake).toBe(true)
    expect(unclosed.text).toBe('[board] the latest message is #14 (you have read up to #13). Read anything you have missed.')
    expect(unclosed.text).not.toContain('<SYSTEM_MESSAGE>')

    const closed = stripInspectorSystemEnvelope(
      '<SYSTEM_MESSAGE>\n[Message] timestamp=2026-11-23T21:35:25Z content=[swarm] check #14\n</SYSTEM_MESSAGE>',
    )
    expect(closed.isSystemWake).toBe(true)
    expect(closed.text).toBe('[swarm] check #14')

    const normal = stripInspectorSystemEnvelope('Implement the Phase-2 loop design.')
    expect(normal.isSystemWake).toBe(false)
    expect(normal.text).toBe('Implement the Phase-2 loop design.')
  })

  test('strips duplicated Created At / Completed At headers and formats read_board + swarm_wait cleanly', () => {
    const rawReadBoard = [
      'Created At: 2026-11-23T21:35:25Z',
      'Completed At: 2026-11-23T21:35:25Z',
      'Created At: 2026-11-23T21:35:25Z',
      'Completed At: 2026-11-23T21:35:25Z',
      JSON.stringify({
        messages: [{ seq: 14, agent: 'verifier', text: '@arch-designer Phase-2 design audit PASSED.' }],
        head: 14,
      }),
    ].join('\n')

    expect(stripAgyToolTimestampBoilerplate(rawReadBoard).startsWith('{')).toBe(true)
    const formattedRead = formatInspectorToolResult('read_board', rawReadBoard)
    expect(formattedRead.collapsible).toBe(true)
    expect(formattedRead.summary).toBe('Read 1 board message (#14 from @verifier) · head #14')
    expect(formattedRead.body).toContain('#14 @verifier: @arch-designer Phase-2 design audit PASSED.')
    expect(formattedRead.body).not.toContain('Created At:')

    const rawWait = [
      'Created At: 2026-11-23T21:35:40Z',
      'Completed At: 2026-11-23T21:35:40Z',
      'Created At: 2026-11-23T21:35:40Z',
      'Completed At: 2026-11-23T21:35:40Z',
      'Waiting for a message that calls on you by name, or on @all posted after #14.',
    ].join('\n')
    const formattedWait = formatInspectorToolResult('swarm_wait', rawWait)
    expect(formattedWait.collapsible).toBe(false)
    expect(formattedWait.summary).toBe('Waiting for a message that calls on you by name, or on @all posted after #14.')

    expect(formatInspectorToolArgs('{"mentions_only":true,"since":"14"}')).toBe(
      '{\n  "mentions_only": true,\n  "since": "14"\n}',
    )
  })

  test('SwarmBoardView worker inspector renders Markdown bold headings, clean board wake notices, and cleaned tool outputs', () => {
    const status: SwarmStatusSnapshot = {
      active: true,
      session_id: 's1',
      title: 'Qwen 3.8-27B Loop Transformer Analysis',
      board: 'local',
      open_board: '',
      head: 14,
      tail_from_seq: 1,
      overview: null,
      channels: [{ name: 'main', topic: '', created_by: 'coordinator', archived: false, subscriber_count: 2 }],
      members: [
        {
          name: 'arch-designer',
          role: 'worker',
          model: 'gemini-4-neon-xhigh',
          model_display_name: 'gemini-4-neon-xhigh',
          workspace: 'inherit',
          state: 'waiting',
          channels: ['main'],
          last_read_seq: 14,
          last_post_seq: 13,
        },
      ],
    }

    const inspect: SwarmMemberInspectResult = {
      member: status.members[0]!,
      total_messages: 4,
      messages: [
        {
          index: 0,
          role: 'user',
          text: '<SYSTEM_MESSAGE>\n[board] the latest message is #14 (you have read up to #13).',
        },
        {
          index: 1,
          role: 'tool',
          tool_name: 'read_board',
          text: 'Created At: 2026-11-23T21:35:25Z\nCompleted At: 2026-11-23T21:35:25Z\nCreated At: 2026-11-23T21:35:25Z\nCompleted At: 2026-11-23T21:35:25Z\n{"messages":[{"seq":14,"agent":"verifier","text":"Audit PASSED"}],"head":14}',
        },
        {
          index: 2,
          role: 'assistant',
          text: '**Celebrating Design Audit Success**\n\nPhase-2 audit passed.',
          tool_calls: [{ id: 'tc-1', name: 'swarm_wait', args_preview: '{"mentions_only":true,"since":"14"}' }],
        },
        {
          index: 3,
          role: 'tool',
          tool_name: 'swarm_wait',
          text: 'Created At: 2026-11-23T21:35:40Z\nCompleted At: 2026-11-23T21:35:40Z\nWaiting for a message after #14.',
        },
      ],
    }

    const html = renderToStaticMarkup(
      createElement(SwarmBoardView, {
        status,
        channels: status.channels,
        members: status.members,
        counts: { total: 1, active: 0, waiting: 1, stopped: 0, killed: 0 },
        messages: [],
        visibleMessages: [],
        models: [],
        modelsLoading: false,
        online: true,
        loading: false,
        busyAction: '',
        error: '',
        activeTab: 'members',
        selectedChannel: 'main',
        highlightedSeq: null,
        feedbackOpen: false,
        feedbackDraft: '',
        postDraft: '',
        switchingMember: '',
        modelDrafts: {},
        inspectedMember: 'arch-designer',
        inspectResult: inspect,
        inspectLoading: false,
        onSelectTab: () => {},
        onRefresh: () => {},
        onStopAll: () => {},
        onToggleFeedback: () => {},
        onChangeFeedbackDraft: () => {},
        onSubmitFeedback: () => {},
        onCloseFeedback: () => {},
        onSelectChannel: () => {},
        onInsertMention: () => {},
        onInsertSeqRef: () => {},
        onJumpToSeq: () => {},
        onKillMember: () => {},
        onInspectMember: () => {},
        onCloseInspect: () => {},
        onToggleModelSwitch: () => {},
        onChangeModelDraft: () => {},
        onSubmitModelSwitch: () => {},
        onCancelModelSwitch: () => {},
        onChangePostDraft: () => {},
        onSubmitPost: () => {},
      }),
    )

    expect(html).not.toContain('&lt;SYSTEM_MESSAGE&gt;')
    expect(html).not.toContain('Created At:')
    expect(html).not.toContain('**Celebrating Design Audit Success**')
    expect(html).toContain('<strong>Celebrating Design Audit Success</strong>')
    expect(html).toContain('Read 1 board message (#14 from @verifier) · head #14')
    expect(html).toContain('Waiting for a message after #14.')
    expect(html).toContain('swarm-inspector__turn--wake')
  })
})
