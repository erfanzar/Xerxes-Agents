// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { afterEach, describe, expect, it, vi } from 'vitest'

import { getTurnState } from '../app/turnStore.js'
import { turnController } from '../app/turnController.js'
import { getUiState, patchUiState } from '../app/uiStore.js'
import {
  clearSpawnHistory,
  getSpawnHistory,
  reconcileSpawnHistorySubagent
} from '../app/spawnHistoryStore.js'
import type { SubagentEventPayload } from '../gatewayTypes.js'
import { toolTrailLabel } from '../lib/text.js'

const subagent = (id: string, index: number, goal: string): SubagentEventPayload => ({
  agent_name: index === 0 ? 'runtime-audit' : 'test-review',
  agent_type: index === 0 ? 'researcher' : 'reviewer',
  depth: 1,
  goal,
  parent_id: 'root-agent',
  status: 'running',
  subagent_id: id,
  task_count: 2,
  task_index: index,
  tool_count: index
})

const seedParallelSubagents = () => {
  turnController.startMessage()

  // Deliver out of order to prove the stable daemon ids and task indexes,
  // rather than arrival order, define the two durable rows.
  turnController.upsertSubagent(subagent('review-child', 1, 'run verification'), current => ({
    notes: [...current.notes, 'checking tests'],
    status: 'running'
  }))
  turnController.upsertSubagent(subagent('research-child', 0, 'inspect runtime'), current => ({
    notes: [...current.notes, 'reading sources'],
    status: 'running'
  }))
  turnController.upsertSubagent(subagent('research-child', 0, 'inspect runtime'), current => ({
    summary: 'runtime mapped',
    status: 'completed'
  }))
}

describe('turnController', () => {
  afterEach(() => {
    turnController.fullReset()
    clearSpawnHistory()
    vi.useRealTimers()
  })

  it('keeps TodoWriteTool state pinned instead of archiving it into the transcript', () => {
    const todos = [{ content: 'verify the fix', id: '1', status: 'completed' as const }]

    turnController.fullReset()
    turnController.recordTodos(todos)

    const result = turnController.recordMessageComplete({ text: 'Done.' })

    expect(result.finalMessages.some(msg => msg.kind === 'trail' && Boolean(msg.todos?.length))).toBe(false)
    expect(result.finalMessages).toEqual([{ role: 'assistant', text: 'Done.' }])
    expect(getTurnState().todos).toEqual(todos)
  })

  it('clears pinned todos when the next assistant turn starts', () => {
    const todos = [{ content: 'verify the fix', id: '1', status: 'completed' as const }]

    turnController.recordTodos(todos)
    expect(getTurnState().todos).toEqual(todos)

    turnController.startMessage()

    expect(getTurnState().todos).toEqual([])
  })

  it('never flashes a reasoning tag split across live deltas', () => {
    vi.useFakeTimers()
    patchUiState({ streaming: true })
    turnController.startMessage()
    turnController.recordMessageDelta({ text: 'Visible <reaso' })

    vi.runOnlyPendingTimers()

    expect(getTurnState().streaming).not.toContain('<reaso')

    turnController.recordMessageDelta({ text: 'ning>hidden</reasoning> after enough text to leave the filter tail.' })

    vi.runOnlyPendingTimers()

    expect(getTurnState().streaming).toContain('Visible')
    expect(getTurnState().streaming).toContain('after')
    expect(getTurnState().streaming).not.toContain('hidden')
    expect(getTurnState().streaming).not.toContain('<reasoning>')
  })

  it('keeps every completed tool visible while later stream events arrive', () => {
    turnController.startMessage()
    turnController.recordReasoningDelta('I should inspect the workspace.', true)
    turnController.recordToolStart('tool-1', 'read_file', 'src/one.ts')
    turnController.recordToolComplete('tool-1', 'read_file', undefined, 'loaded one', 0.1)

    expect(getTurnState().streamPendingTools).toHaveLength(1)
    expect(getTurnState().streamPendingTools[0]).toContain('Read File("src/one.ts")')

    // Starting the next call flushes prior narration/details into a settled
    // live segment. The completed call must remain on the OpenTUI live shelf.
    turnController.recordMessageDelta({ text: 'Now I will inspect the second file.' })
    turnController.recordToolStart('tool-2', 'read_file', 'src/two.ts')

    expect(getTurnState().streamPendingTools).toHaveLength(1)
    expect(getTurnState().streamPendingTools[0]).toContain('Read File("src/one.ts")')
    expect(getTurnState().streamSegments.flatMap(message => message.tools ?? [])[0]).toContain(
      'Read File("src/one.ts")'
    )
    expect(getTurnState().streamSegments.some(message => message.text === 'Now I will inspect the second file.')).toBe(
      true
    )

    turnController.recordToolComplete('tool-2', 'read_file', undefined, 'loaded two', 0.2)

    expect(getTurnState().streamPendingTools).toHaveLength(2)
    expect(getTurnState().streamPendingTools[0]).toContain('Read File("src/one.ts")')
    expect(getTurnState().streamPendingTools[1]).toContain('Read File("src/two.ts")')

    const { finalMessages } = turnController.recordMessageComplete({ text: 'Done.' })
    const persistedTools = finalMessages.flatMap(message => message.tools ?? [])

    expect(persistedTools).toHaveLength(2)
    expect(persistedTools[0]).toContain('Read File("src/one.ts")')
    expect(persistedTools[1]).toContain('Read File("src/two.ts")')
  })

  it('treats repeated normal and inline tool completions as idempotent', () => {
    const diff = ['--- a/src/a.ts', '+++ b/src/a.ts', '@@ -1 +1 @@', '-old', '+new'].join('\n')
    turnController.startMessage()
    turnController.recordToolStart('read-1', 'ReadFile', 'src/a.ts')
    turnController.recordToolComplete('read-1', 'ReadFile', undefined, 'loaded', 0.1)
    turnController.recordToolComplete('read-1', 'ReadFile', undefined, 'loaded', 0.1)
    turnController.recordToolStart('edit-1', 'FileEditTool', 'src/a.ts')
    turnController.recordInlineDiffToolComplete(diff, 'edit-1', 'FileEditTool', undefined, 0.2)
    turnController.recordInlineDiffToolComplete(diff, 'edit-1', 'FileEditTool', undefined, 0.2)

    expect(getTurnState().streamPendingTools).toHaveLength(2)
    const { finalMessages } = turnController.recordMessageComplete({ text: 'Done.' })
    expect(finalMessages.flatMap(message => message.tools ?? [])).toHaveLength(2)
  })

  it('keeps a deduplicated inline-diff tool row when the final answer repeats the diff', () => {
    const diff = ['--- a/src/a.ts', '+++ b/src/a.ts', '@@ -1 +1 @@', '-old', '+new'].join('\n')
    turnController.startMessage()
    turnController.recordToolStart('edit-1', 'FileEditTool', 'src/a.ts')
    turnController.recordInlineDiffToolComplete(diff, 'edit-1', 'FileEditTool', undefined, 0.2)

    const { finalMessages } = turnController.recordMessageComplete({
      text: `Applied the change.\n\n\`\`\`diff\n${diff}\n\`\`\``
    })

    expect(finalMessages.some(message => message.kind === 'diff')).toBe(false)
    const toolRow = finalMessages.find(message => message.kind === 'trail' && message.tools?.length)
    expect(toolRow?.tools).toHaveLength(1)
    expect(finalMessages.indexOf(toolRow!)).toBeLessThan(finalMessages.findIndex(message => message.role === 'assistant'))
  })

  it('does not persist a phantom trail for tool-token metadata alone', () => {
    turnController.startMessage()
    turnController.toolTokenAcc = 42

    const { finalMessages } = turnController.recordMessageComplete({ text: 'Done.' })

    expect(finalMessages).toEqual([{ role: 'assistant', text: 'Done.' }])
  })

  it('does not duplicate a clarification row already persisted by its answer', () => {
    turnController.startMessage()
    turnController.recordToolStart('clarify-1', 'clarify', 'Which provider?')
    turnController.persistedToolIds.add('clarify-1')
    turnController.recordToolComplete('clarify-1', 'clarify', undefined, 'answered', 0.1)
    expect(getTurnState().streamPendingTools).toEqual([])

    turnController.recordToolStart('clarify-legacy', 'clarify', 'Which model?')
    turnController.persistedToolLabels.add(toolTrailLabel('clarify'))
    turnController.recordToolComplete('clarify-legacy', 'clarify', undefined, 'answered', 0.1)
    expect(getTurnState().streamPendingTools).toEqual([])
  })

  it('preserves every reasoning phase with the tool call that followed it', () => {
    turnController.startMessage()
    turnController.recordReasoningDelta('First I will inspect one file.', true)
    turnController.recordToolStart('tool-1', 'read_file', 'src/one.ts')
    turnController.recordToolComplete('tool-1', 'read_file', undefined, 'raw first result', 0.1)
    turnController.recordReasoningDelta('Now I need the second file.', true)
    turnController.recordToolStart('tool-2', 'read_file', 'src/two.ts')
    turnController.recordToolComplete('tool-2', 'read_file', undefined, 'raw second result', 0.2)

    const live = getTurnState().streamSegments

    expect(live).toHaveLength(2)
    expect(live[0]).toMatchObject({ thinking: 'First I will inspect one file.' })
    expect(live[0]?.tools?.[0]).toContain('Read File("src/one.ts")')
    expect(live[1]).toMatchObject({ thinking: 'Now I need the second file.' })
    expect(live[1]?.tools?.[0]).toContain('Read File("src/two.ts")')

    const { finalMessages } = turnController.recordMessageComplete({ text: 'Done.' })
    const thinking = finalMessages.filter(message => message.thinking).map(message => message.thinking)

    expect(thinking).toEqual(['First I will inspect one file.', 'Now I need the second file.'])
  })

  it('stores successful tools as one-line rows without raw Args or Result payloads', () => {
    turnController.startMessage()
    turnController.recordToolStart('tool-1', 'exec_command', 'ls -la')
    turnController.recordToolComplete(
      'tool-1',
      'exec_command',
      undefined,
      'completed successfully',
      0.1,
      undefined,
      '{"stdout":"thousands of bytes that belong in model context, not the TUI"}'
    )

    const line = getTurnState().streamPendingTools[0] ?? ''

    expect(line).toContain('Exec Command("ls -la")')
    expect(line).not.toContain('Args:')
    expect(line).not.toContain('Result:')
    expect(line).not.toContain('stdout')
    expect(line.split('\n')).toHaveLength(1)
  })

  it('returns completed tool history when a turn ends in an error', () => {
    turnController.startMessage()
    turnController.recordToolStart('tool-1', 'read_file', 'src/one.ts')
    turnController.recordToolComplete('tool-1', 'read_file', undefined, 'loaded one', 0.1)

    const preserved = turnController.recordError()

    expect(preserved.flatMap(message => message.tools ?? [])).toHaveLength(1)
    expect(preserved.flatMap(message => message.tools ?? [])[0]).toContain('Read File("src/one.ts")')
    expect(getTurnState().streamPendingTools).toEqual([])
  })

  it('keeps multiple subagents as stable rows and persists them when the message completes', () => {
    seedParallelSubagents()

    expect(getTurnState().subagents.map(agent => agent.id)).toEqual(['research-child', 'review-child'])
    expect(getTurnState().subagents[0]).toMatchObject({
      name: 'runtime-audit',
      notes: ['reading sources'],
      status: 'completed',
      summary: 'runtime mapped'
    })

    const { finalMessages } = turnController.recordMessageComplete({ text: 'Parallel work finished.' })
    const trail = finalMessages.find(message => message.subagents?.length)

    expect(trail).toMatchObject({ kind: 'trail', role: 'system', text: '' })
    expect(trail?.subagents?.map(agent => agent.name)).toEqual(['runtime-audit', 'test-review'])
    expect(trail?.subagents?.map(agent => agent.id)).toEqual(['research-child', 'review-child'])
    expect(trail?.subagents?.map(agent => agent.status)).toEqual(['completed', 'interrupted'])
    expect(getSpawnHistory()[0]?.subagents.map(agent => agent.status)).toEqual(['completed', 'interrupted'])
    expect(finalMessages.at(-1)).toEqual({ role: 'assistant', text: 'Parallel work finished.' })
    expect(getTurnState().subagents).toEqual([])

    reconcileSpawnHistorySubagent(
      {
        goal: 'run verification',
        status: 'completed',
        subagent_id: 'review-child',
        summary: 'verification finished after the parent boundary',
        task_index: 1
      },
      () => ({ status: 'completed', summary: 'verification finished after the parent boundary' })
    )
    expect(getSpawnHistory()[0]?.subagents[1]).toMatchObject({
      id: 'review-child',
      status: 'completed',
      summary: 'verification finished after the parent boundary'
    })
  })

  it('archives queued subagents as interrupted when the parent message completes', () => {
    turnController.startMessage()
    turnController.upsertSubagent(subagent('queued-child', 0, 'wait for a worker slot'), () => ({ status: 'queued' }))

    const { finalMessages } = turnController.recordMessageComplete({ text: 'Parent finished.' })
    const archived = finalMessages.find(message => message.subagents?.length)?.subagents?.[0]

    expect(archived).toMatchObject({ id: 'queued-child', status: 'interrupted' })
    expect(getSpawnHistory()[0]?.subagents[0]).toMatchObject({ id: 'queued-child', status: 'interrupted' })
  })

  it('persists every active subagent on a daemon-confirmed interruption before clearing live state', () => {
    seedParallelSubagents()
    const sys = vi.fn()
    const request = vi.fn().mockResolvedValue({ ok: true })

    turnController.interruptTurn({ gw: { request }, sid: 'session-interrupt', sys })

    expect(request).toHaveBeenCalledWith('session.interrupt', { session_id: 'session-interrupt' })
    expect(getUiState().status).toBe('interrupting…')

    // The [interrupted] archive is deferred until the daemon confirms the
    // cancellation on its settle edge (a cancelled turn_end).
    const { finalMessages, wasInterrupted } = turnController.recordMessageComplete({ interrupted: true })

    expect(wasInterrupted).toBe(true)
    expect(finalMessages).toHaveLength(1)
    expect(finalMessages[0]).toMatchObject({ role: 'assistant', text: '*[interrupted]*' })
    expect(finalMessages[0]?.subagents?.map(agent => agent.id)).toEqual(['research-child', 'review-child'])
    expect(finalMessages[0]?.subagents?.map(agent => agent.status)).toEqual(['completed', 'interrupted'])
    expect(sys).not.toHaveBeenCalled()
    expect(getTurnState().subagents).toEqual([])
  })

  it('prefers the turn real ending when a natural completion races the interrupt request', () => {
    turnController.startMessage()
    turnController.recordMessageDelta({ text: 'The complete answer.' })
    const sys = vi.fn()
    const request = vi.fn().mockResolvedValue({ ok: true })

    turnController.interruptTurn({ gw: { request }, sid: 'session-race', sys })

    // turn_end was already in flight (not cancelled): the real final
    // messages win over the pending interrupt request.
    const { finalMessages, finalText, wasInterrupted } = turnController.recordMessageComplete({})

    expect(wasInterrupted).toBe(false)
    expect(finalText).toBe('The complete answer.')
    expect(finalMessages).toEqual([{ role: 'assistant', text: 'The complete answer.' }])
    expect(finalMessages.some(message => message.text.includes('[interrupted]'))).toBe(false)
    expect(getUiState().busy).toBe(false)
  })

  it('treats the legacy [interrupted] marker as a daemon confirmation', () => {
    turnController.startMessage()
    turnController.recordMessageDelta({ text: 'Partial draft' })
    const request = vi.fn().mockResolvedValue({ ok: true })

    turnController.interruptTurn({ gw: { request }, sid: 'session-legacy', sys: vi.fn() })
    const { finalMessages, wasInterrupted } = turnController.recordMessageComplete({ text: '[interrupted]' })

    expect(wasInterrupted).toBe(true)
    expect(finalMessages).toEqual([{ role: 'assistant', text: 'Partial draft\n\n*[interrupted]*' }])
  })

  it('reverts a failed interrupt request instead of showing a false interrupted', async () => {
    turnController.startMessage()
    const sys = vi.fn()
    const request = vi.fn().mockRejectedValue(new Error('gateway socket closed'))

    turnController.interruptTurn({ gw: { request }, sid: 'session-flaky', sys })
    expect(getUiState().status).toBe('interrupting…')

    await new Promise(resolve => setTimeout(resolve, 0))

    // The daemon never confirmed: live recording re-arms, the failure is
    // reported once, and no false "interrupted" lands anywhere.
    expect(sys).toHaveBeenCalledWith('error: interrupt failed: gateway socket closed')
    expect(getUiState().status).toBe('running…')
    expect(getUiState().busy).toBe(true)

    // The turn is still alive — its natural completion renders normally.
    turnController.recordMessageDelta({ text: 'Still finished.' })
    const { finalMessages, wasInterrupted } = turnController.recordMessageComplete({})

    expect(wasInterrupted).toBe(false)
    expect(finalMessages).toEqual([{ role: 'assistant', text: 'Still finished.' }])
  })

  it('returns every live subagent as transcript state when the turn errors', () => {
    seedParallelSubagents()

    const preserved = turnController.recordError()
    const trail = preserved.find(message => message.subagents?.length)

    expect(trail).toMatchObject({ kind: 'trail', role: 'system', text: '' })
    expect(trail?.subagents?.map(agent => agent.id)).toEqual(['research-child', 'review-child'])
    expect(trail?.subagents?.map(agent => agent.status)).toEqual(['completed', 'interrupted'])
    expect(getTurnState().subagents).toEqual([])
  })

  it('anchors an accepted steer as one ordinary user message between live assistant segments', () => {
    turnController.startMessage()
    turnController.recordMessageDelta({ text: 'First answer segment.' })
    turnController.recordUserSteer('Please also verify the tests.')

    expect(getTurnState().streamSegments).toEqual([
      { role: 'assistant', text: 'First answer segment.' },
      { role: 'user', text: 'Please also verify the tests.' }
    ])

    turnController.recordMessageDelta({ text: ' Second answer segment.' })
    const { finalMessages } = turnController.recordMessageComplete({
      text: 'First answer segment. Second answer segment.'
    })

    expect(finalMessages).toEqual([
      { role: 'assistant', text: 'First answer segment.' },
      { role: 'user', text: 'Please also verify the tests.' },
      { role: 'assistant', text: 'Second answer segment.' }
    ])
    expect(finalMessages.filter(message => message.role === 'user')).toHaveLength(1)
  })
})
