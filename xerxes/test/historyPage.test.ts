// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { test, expect } from 'bun:test'
import { historyLimit, sessionHistoryPage } from '../src/daemon/historyPage.js'
import type { DaemonSession } from '../src/daemon/runtime.js'
const source = () => ({id:'session-a', messages:Array.from({length:250},(_,i)=>({role:i%2?'assistant':'user',content:`Message ${i}`})),toolExecutions:[],thinkingContent:[]} satisfies Pick<DaemonSession,'id'|'messages'|'toolExecutions'|'thinkingContent'>)
test('history pages are newest first to fetch, chronological to render, and append stable',()=>{
 const s=source(), tail=sessionHistoryPage(s,100)
 expect(tail.actions).toHaveLength(100)
 expect(tail.actions[0]?.messages[0]?.content).toBe('Message 150')
 s.messages.push({role:'assistant',content:'New live message'})
 const middle=sessionHistoryPage(s,100,tail.before),first=sessionHistoryPage(s,100,middle.before)
 expect(middle.actions[0]?.messages[0]?.content).toBe('Message 50')
 expect(first.actions).toHaveLength(50)
 expect(first.has_more).toBe(false)
 expect(new Set([...first.actions,...middle.actions,...tail.actions].map(a=>a.id)).size).toBe(250)
})
test('tool calls and results stay together even at a page boundary',()=>{
 const s:Pick<DaemonSession,'id'|'messages'|'toolExecutions'|'thinkingContent'>=source()
 s.messages.push({role:'assistant',content:'',tool_calls:[{id:'call',function:{name:'ReadFileTool',arguments:{path:'long.ts'}}}]},{role:'tool',tool_call_id:'call',content:'line 1\nline 2'})
 s.toolExecutions.push({toolCallId:'call',name:'ReadFileTool',result:'line 1\nline 2',durationMs:20})
 const page=sessionHistoryPage(s,1)
 expect(page.actions).toHaveLength(1)
 expect(page.actions[0]?.messages).toHaveLength(2)
 expect(page.actions[0]?.executions).toHaveLength(1)
 expect(page.actions[0]?.messages[1]?.content).toBe('line 1\nline 2')
})
test('changed history and cross-session cursors fail without mutating the source',()=>{
 const s=source(),cursor=sessionHistoryPage(s,100).before
 expect(()=>sessionHistoryPage({...s,id:'other'},100,cursor)).toThrow('History changed')
 s.messages.splice(0,20)
 expect(()=>sessionHistoryPage(s,100,cursor)).toThrow('History changed')
 expect(s.messages).toHaveLength(230)
 expect(()=>sessionHistoryPage(s,100,'bad')).toThrow('invalid history cursor')
 for(const limit of [-1,101,'100',1.5])expect(()=>historyLimit(limit)).toThrow()
 expect(historyLimit(0)).toBe(0)
 expect(sessionHistoryPage({...s,messages:[]},100)).toMatchObject({actions:[],has_more:false,before:null})
})

test('outcomes remain ordered and appear once when paging text, no-output, and tool-only turns', () => {
  const s: Pick<DaemonSession, 'id'|'messages'|'toolExecutions'|'thinkingContent'> = {id: 'outcomes', toolExecutions: [], thinkingContent: [], messages: [
    {role: 'user', content: 'Failed before output', turn_outcome: {version: 1, reason: 'provider_failed'}},
    {role: 'assistant', content: 'Tool narration', tool_calls: [{id: 'c', function: {name: 'read_file', arguments: '{}'}}]},
    {role: 'tool', tool_call_id: 'c', content: 'result', turn_outcome: {version: 1, reason: 'aborted'}},
    {role: 'assistant', content: 'Answer', turn_outcome: {version: 1, reason: 'completed'}},
  ]}
  const rows: DaemonSession['messages'] = []
  let before: string | null = null
  do {
    const page = sessionHistoryPage(s, 1, before)
    rows.unshift(...page.actions.flatMap(action => action.messages))
    before = page.before
  } while (before)
  expect(rows.filter(m => m.turn_outcome).map(m => (m.turn_outcome as {reason: string}).reason)).toEqual(['provider_failed', 'aborted', 'completed'])
  expect(rows.filter(m => m.content).map(m => m.content)).toEqual(['Failed before output', 'Tool narration', 'result', 'Answer'])
  expect(s.messages[0]?.turn_outcome).toEqual({version: 1, reason: 'provider_failed'})
})

test('overlapping archived executions render once while repeated user prompts remain distinct', () => {
  const call = { role: 'assistant', content: '', tool_calls: [{ id: 'same-call', function: { name: 'ReadFile', arguments: '{}' } }] }
  const result = { role: 'tool', tool_call_id: 'same-call', content: 'output' }
  const session = { id: 'archives', messages: [{ role: 'user', content: 'continue' }, call, result, { role: 'user', content: 'continue' }, call, result], toolExecutions: [], thinkingContent: [] }
  const actions = sessionHistoryPage(session, 100).actions
  expect(actions.filter(action => action.messages[0]?.tool_calls)).toHaveLength(1)
  expect(actions.filter(action => action.messages[0]?.role === 'user')).toHaveLength(2)
})
