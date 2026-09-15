// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { foldAgentEvent } from '../src/desktop/renderer/agentEvents.js'
import type { SessionRow } from '../src/desktop/renderer/types.js'
const event = (type: string, payload: Record<string, unknown> = {}, id='child') => ({agent_id:id,title:`Agent ${id}`,model:'test-model',goal:'Check cancellation',event:{type,payload}})
test('child calls pair by id across concurrent results and preserve denied output',()=>{
 let rows:readonly SessionRow[]=foldAgentEvent([],event('turn_begin'),100)
 rows=foldAgentEvent(rows,event('tool_call',{id:'one',name:'read_file',arguments:'src/a.ts'}),110)
 rows=foldAgentEvent(rows,event('tool_call',{id:'two',name:'exec_command',arguments:'bun test'}),120)
 rows=foldAgentEvent(rows,event('tool_result',{tool_call_id:'two',name:'exec_command',permitted:false,return_value:'Permission denied',duration_ms:250}),370)
 rows=foldAgentEvent(rows,event('tool_result',{tool_call_id:'one',name:'read_file',return_value:'export const a = 1',duration_ms:500}),610)
 expect(rows[0]?.agentDetails?.toolCalls?.map(call=>[call.id,call.state])).toEqual([['one','done'],['two','failed']])
 expect(rows[0]?.agentDetails?.toolCalls?.[1]?.error).toBe('Permission denied')
 expect(rows[0]?.agentDetails?.toolCalls?.[0]?.output).toBe('export const a = 1')
})
test('nested events retain child identity and terminal state survives late progress',()=>{
 let rows:readonly SessionRow[]=foldAgentEvent([],event('turn_begin'))
 rows=foldAgentEvent(rows,{...event('subagent_event'),event:{type:'subagent_event',payload:event('turn_begin',{},'grandchild')}})
 expect(rows.map(row=>row.id)).toEqual(['child','grandchild'])
 rows=foldAgentEvent(rows,event('turn_end',{status:'cancelled',summary:'Stopped by user'}))
 rows=foldAgentEvent(rows,event('think_part',{think:'Late frame'}))
 expect(rows[0]?.status).toBe('cancelled')
 expect(rows[0]?.agentDetails?.summary).toBe('Stopped by user')
 expect(rows[1]?.status).toBe('running')
})
test('completion without a result does not paint pending tools as successful',()=>{
 let rows:readonly SessionRow[]=foldAgentEvent([],event('tool_call',{id:'pending',name:'exec_command'}))
 rows=foldAgentEvent(rows,event('turn_end',{status:'interrupted'}))
 expect(rows[0]?.agentDetails?.toolCalls?.[0]?.state).toBe('failed')
 expect(rows[0]?.agentDetails?.toolCalls?.[0]?.error).toContain('before a tool result')
})
test('malformed frames are ignored and progress is bounded',()=>{
 expect(foldAgentEvent([],{event:null})).toEqual([])
 let rows:readonly SessionRow[]=[]
 for(let i=0;i<120;i++)rows=foldAgentEvent(rows,event('text_part',{text:`line ${i}`}))
 expect(rows[0]?.agentDetails?.notes).toHaveLength(40)
})
