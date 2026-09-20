// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import { ExecutionDetails, executionView, ToolCallRow } from '../src/desktop/renderer/Execution.js'
import { activityFleetRows, AgentRoster } from '../src/desktop/renderer/AgentRoster.js'
import { CommandActivity } from '../src/desktop/renderer/CommandActivity.js'
import type { SessionRow } from '../src/desktop/renderer/types.js'
import type { ToolItem } from '../src/desktop/renderer/types.js'
import { BlockBuilder } from '../src/desktop/renderer/blocks.js'
import { activityGroupKey, groupActivity, keyedActivityGroups } from '../src/desktop/renderer/activityGroups.js'

const item: ToolItem = { id: 'call-1', name: 'exec_command', verb: 'exec_command', arg: 'sed', dur: '0.0s', state: 'done', input: JSON.stringify({ cmd: 'sed', args: ['-n', '10,20p', 'path with spaces/file.ts'] }), output: JSON.stringify({ stdout: 'function run() {\n  return 1\n}\n', stderr: '', exitCode: 0, cwd: '/repo' }) }
test('collapsed calls do not construct large output viewers', () => {
  const output = 'Large result line\n'.repeat(10_000)
  const html = renderToStaticMarkup(createElement(ToolCallRow, { label: 'Exec command', item: { ...item, output } }))
  expect(html).toContain('Exec command')
  expect(html).not.toContain('Large result line')
  expect(html).not.toContain('execution__viewer')
  expect(html.length).toBeLessThan(3000)
})
test('collapsed tool rows show complete command arguments and surface nonzero exit failures', () => {
  const html = renderToStaticMarkup(createElement(ToolCallRow, { label: 'Exec command', item: { ...item, output: JSON.stringify({ exitCode: 2, stderr: 'Permission denied' }) } }))
  const summary = html.slice(0, html.indexOf('</summary>'))
  expect(summary).toContain('10,20p')
  expect(summary).toContain('path with spaces/file.ts')
  expect(summary).toContain('Failed')
  expect(summary).toContain('Permission denied')
})
test('agent roster puts active work first and collapses failed and completed history', () => {
  const base: SessionRow = { id: 'a', key: 'a', title: 'Completed review', status: 'completed', age: '', current: false, kind: 'subagent', turns: 0, messages: 0, cwd: '', untitled: false }
  const html = renderToStaticMarkup(createElement(AgentRoster, { rows: [base, { ...base, id: 'b', title: 'Active review', status: 'running' }, { ...base, id: 'c', title: 'Failed review', status: 'failed', agentDetails: { summary: 'Checked cancellation', error: 'Permission denied', model: 'test-model', toolCount: 4, filesRead: ['src/deep/path.ts'], filesWritten: [] } }] }))
  expect(html.indexOf('Active review')).toBeLessThan(html.indexOf('Failed review'))
  expect(html.indexOf('Past agents')).toBeLessThan(html.indexOf('Failed review'))
  expect(html).toContain('2 · 1 failed')
  expect(html.indexOf('Active review')).toBeLessThan(html.indexOf('Completed review'))
  expect(html).toContain('Permission denied')
  expect(html).toContain('src/deep/path.ts')
  expect(html).toContain('4 tools')
  expect(html).not.toContain('0 turns')
})
test('execution decodes output line breaks and safely displays argument boundaries', () => {
  const view = executionView(item)
  expect(view.command).toBe("sed -n 10,20p 'path with spaces/file.ts'")
  expect(view.stdout).toBe('function run() {\n  return 1\n}\n')
  const html = renderToStaticMarkup(createElement(ExecutionDetails, { item }))
  expect(html).toContain('aria-label="Command output">function run() {\n  return 1\n}')
  expect(html).toContain('<summary>Raw details</summary>')
  expect(html).toContain('Copy command')
})
test('nonzero results expose failure and stderr even without a transport error', () => {
  const html = renderToStaticMarkup(createElement(ExecutionDetails, { item: { ...item, output: JSON.stringify({ stdout: '', stderr: 'Permission denied\n', exitCode: 2 }) } }))
  expect(html).toContain('Failed · Exit 2')
  expect(html).toContain('Permission denied\n')
})
test('plain output and malformed payloads remain visible without invented success data', () => {
  const html = renderToStaticMarkup(createElement(ExecutionDetails, { item: { ...item, output: 'partial { output', error: 'Cancelled by user', state: 'failed' } }))
  expect(html).toContain('partial { output')
  expect(html).toContain('Cancelled by user')
  expect(html).not.toContain('Exit 0')
})

test('activity grouping preserves prose and keeps approval operations outside disclosures', async () => {
  const { groupActivity } = await import('../src/desktop/renderer/activityGroups.js')
  const blocks = [{ kind: 'thinking' as const, id: 1, text: 'Inspect', streaming: false }, { kind: 'tools' as const, id: 2, items: [item], running: false }, { kind: 'agent' as const, id: 3, text: 'Result', streaming: false }]
  expect(groupActivity(blocks).map(group => group.length)).toEqual([2, 1])
  expect(groupActivity(blocks, item.id).map(group => group.length)).toEqual([2, 1])
  expect(groupActivity(blocks).flat()).toEqual(blocks)
})

test('activity identity survives tool results, subsequent calls and turn finalization', () => {
  for (const error of ['', 'Permission denied', 'Cancelled by user']) {
    const builder = new BlockBuilder()
    builder.push('think_part', { think: 'Inspect the files' })
    builder.push('tool_call', { id: 'first-call', name: 'read_file', arguments: { path: 'README.md' } })
    const key = activityGroupKey(groupActivity(builder.snapshot(true))[0]!)
    builder.push('tool_result', { tool_call_id: 'first-call', return_value: 'Contents', error })
    builder.push('tool_call', { id: 'second-call', name: 'exec_command', arguments: { cmd: 'bun test' } })
    expect(activityGroupKey(groupActivity(builder.snapshot(true))[0]!)).toBe(key)
    builder.push('tool_result', { tool_call_id: 'second-call', return_value: 'Tests passed' })
    builder.finalize()
    expect(activityGroupKey(groupActivity(builder.snapshot(false))[0]!)).toBe(key)
  }
})

test('reused call IDs in different turns get separate disclosure identities', () => {
  const groups = keyedActivityGroups([
    { kind: 'user', id: 1, text: 'First request' },
    { kind: 'tools', id: 2, items: [item], running: false },
    { kind: 'user', id: 3, text: 'Second request' },
    { kind: 'tools', id: 4, items: [item], running: false },
  ])
  expect(groups[1]!.key).not.toBe(groups[3]!.key)
})

test('completed agent history is collapsed without hiding failures or active agents', () => {
  const base: SessionRow = { id: 'done', key: 'done', title: 'Finished research', status: 'completed', age: '', current: false, kind: 'subagent', turns: 0, messages: 0, cwd: '', untitled: false, agentDetails: { toolCount: 0 } }
  const html = renderToStaticMarkup(createElement(AgentRoster, { rows: [base, { ...base, id: 'live', title: 'Research in progress', status: 'running' }] }))
  expect(html).toContain('<details class="agent-roster__history">')
  expect(html.indexOf('Research in progress')).toBeLessThan(html.indexOf('agent-roster__history'))
  expect(html.indexOf('Finished research')).toBeGreaterThan(html.indexOf('agent-roster__history'))
  expect(html).not.toContain('0 tools')
})

test('reasoning owns the work group before tools, notices and agents arrive', () => {
  const builder = new BlockBuilder()
  builder.push('think_part', { think: 'Inspect' })
  const initial = keyedActivityGroups(builder.snapshot(true))[0]!.key
  builder.push('tool_call', { id: 'slow-call', name: 'exec_command', arguments: { cmd: 'bun test' } })
  builder.push('notification', { message: 'Background agent completed' })
  builder.pushAgents([{ key: 'child', title: 'Reviewer', status: 'working' }])
  const during = builder.snapshot(true)
  expect(keyedActivityGroups(during)).toHaveLength(1)
  expect(keyedActivityGroups(during)[0]!.key).toBe(initial)
  expect(during.find(block => block.kind === 'tools')?.kind === 'tools' && (during.find(block => block.kind === 'tools') as Extract<typeof during[number], { kind: 'tools' }>).items[0]?.state).toBe('working')
  builder.push('tool_result', { tool_call_id: 'slow-call', return_value: 'kept result' })
  expect(builder.snapshot(true).filter(block => block.kind === 'tools')).toHaveLength(1)
  builder.finalize()
  expect(keyedActivityGroups(builder.snapshot(false))[0]!.key).toBe(initial)
  expect(builder.snapshot(false).some(block => block.kind === 'tools' && block.items[0]?.output === 'kept result')).toBe(true)
})

test('unconfirmed spawn requests stay visible without fake agent controls', () => {
  const rows=activityFleetRows([], [{kind:'agents',id:1,members:[{key:'call:0',title:'Review code',status:'working'}]}])
  expect(rows).toHaveLength(1)
  expect(rows[0]?.agentDetails?.provisional).toBe(true)
  const html=renderToStaticMarkup(createElement(AgentRoster,{rows}))
  expect(html).toContain('Awaiting runtime status')
  expect(html).not.toContain('Stop agent')
  expect(activityFleetRows([{...rows[0]!,id:'real-id',status:'working',agentDetails:undefined}], [{kind:'agents',id:1,members:[{key:'call:0',runtimeId:'real-id',title:'Different title',status:'working'}]}])).toHaveLength(1)
})

test('long commands use a collapsed compact summary without constructing output', () => {
  const html=renderToStaticMarkup(createElement(CommandActivity,{row:{id:'command-1',kind:'shell',title:'env '+ 'LONG_VARIABLE=value '.repeat(80)+'bun test',detail:'/repo',state:'running'},sessionKey:'session',online:true}))
  expect(html).toContain('command-activity__preview')
  expect(html).toContain('Shell command')
  expect(html).toContain('Running')
  expect(html).not.toContain('aria-label="Command output"')
  expect(html).not.toContain('<strong>env ')
})
