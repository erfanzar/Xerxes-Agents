// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { createNativeSubagentHost } from '../src/daemon/subagentHost.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { registerClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'
import type { CompletionRequest } from '../src/llms/client.js'

async function fixture() {
  const dir = await mkdtemp(join(tmpdir(),'xr-followup-'))
  const requests: CompletionRequest[] = []
  let fail = false
  const registry = new ToolRegistry()
  const options = {agentDefinitions:BUILTIN_AGENTS,cwd:dir,eventBus:new DaemonSubagentEventBus(),model:'fixture-model',permissionMode:'accept-all' as const,tools:[],toolExecutor:registry,
    transcriptStore:new DaemonTranscriptStore({currentProjectDirectory:dir,directory:join(dir,'sessions')}),
    llm:{async *stream(request:CompletionRequest){requests.push(structuredClone(request));if(fail){fail=false;throw new Error('synthetic provider failure')}yield {content:'Reply: '+request.messages.filter(m=>m.role==='user').at(-1)?.content}}}}
  const host = createNativeSubagentHost(options)
  registerClaudeAgentTools(registry,{manager:host.managerPort})
  const send = (target:string,message:string,sessionId='parent') => registry.execute({id:crypto.randomUUID(),type:'function',function:{name:'SendMessageTool',arguments:{target,message}}},{sessionId,metadata:{}})
  const spawn = () => host.managerPort.spawn({promptProfile:'coder',sourceAgentId:'parent',nickname:'worker',message:'First task'})
  return {dir,host,options,requests,send,spawn,setFail(){fail=true},async close(){await host.manager.shutdown();await rm(dir,{recursive:true,force:true})}}
}

test('SendMessageTool continues a completed child and serializes concurrent id/name follow-ups without losing history',async()=>{
  const f=await fixture()
  try {
    const first=await f.spawn();await f.host.managerPort.wait([first.id],5000)
    const replies=await Promise.all([f.send(first.id,'Second task'),f.send('worker','Third task')])
    expect(replies.every(r=>JSON.parse(r).id===first.id)).toBe(true)
    await f.host.managerPort.wait([first.id],5000)
    expect(f.requests).toHaveLength(3)
    expect(f.requests.at(-1)?.messages.filter(m=>m.role==='user').map(m=>m.content)).toEqual(['First task','Second task','Third task'])
    expect(f.host.managerPort.listHandles()).toHaveLength(1)
    expect(f.host.managerPort.listHandles()[0]).toMatchObject({id:first.id,historySessionId:first.historySessionId,status:'completed',model:'fixture-model'})
    await expect(f.send(first.id,'foreign','other-parent')).rejects.toThrow()
    expect(f.requests).toHaveLength(3)
  }finally{await f.close()}
})

test('failed delivery validation releases the input queue and a failed child accepts new work',async()=>{
  const f=await fixture()
  try {
    f.setFail();const first=await f.spawn();await f.host.managerPort.wait([first.id],5000)
    await expect(f.send(first.id,' ')).rejects.toThrow()
    await f.send(first.id,'Recover now');await f.host.managerPort.wait([first.id],5000)
    expect(f.requests).toHaveLength(2)
    expect(f.host.managerPort.listHandles()[0]).toMatchObject({id:first.id,status:'completed',lastInput:'Recover now'})
    f.host.managerPort.close(first.id)
    await expect(f.send(first.id,'Do not revive closed work')).rejects.toThrow('closed')
    f.host.invalidateAll()
    await expect(f.send(first.id,'Do not bypass policy')).rejects.toThrow('invalidated')
    expect(f.requests).toHaveLength(2)
  }finally{await f.close()}
})

test('a follow-up after daemon recovery preserves the stable child and persisted conversation',async()=>{
  const f=await fixture()
  let next:ReturnType<typeof createNativeSubagentHost>|undefined
  try {
    const first=await f.spawn();await f.host.managerPort.wait([first.id],5000)
    const snapshots=f.host.managerPort.listHandles();await f.host.manager.shutdown()
    next=createNativeSubagentHost(f.options)
    expect(next.turnCoordinator.restore?.('parent',snapshots)).toBe(1)
    const replies=await Promise.all([next.managerPort.sendInput(first.id,{message:'After restart'}),next.managerPort.sendInput('worker',{message:'Another follow-up'})])
    expect(replies.map(r=>r.id)).toEqual([first.id,first.id])
    await next.managerPort.wait([first.id],5000)
    expect(f.requests.at(-1)?.messages.filter(m=>m.role==='user').map(m=>m.content)).toEqual(['First task','After restart','Another follow-up'])
    expect(next.managerPort.listHandles()[0]?.historySessionId).toBe(first.historySessionId)
  }finally{await next?.manager.shutdown();await f.close()}
})

test('an interrupted open child accepts follow-up work after its cancelled turn settles',async()=>{
  const f=await fixture()
  let host:ReturnType<typeof createNativeSubagentHost>|undefined
  const started=Promise.withResolvers<void>()
  let calls=0
  try {
    host=createNativeSubagentHost({...f.options,llm:{async *stream(_request,signal){
      if(++calls===1){started.resolve();await new Promise<void>((_resolve,reject)=>{if(signal?.aborted)reject(signal.reason);else signal?.addEventListener('abort',()=>reject(signal.reason),{once:true})})}
      yield {content:'Continued after interruption'}
    }}})
    const child=await host.managerPort.spawn({promptProfile:'coder',sourceAgentId:'parent',message:'Wait'})
    await started.promise
    expect(host.interruptSource('parent')).toBe(1)
    const continued=await host.managerPort.sendInput(child.id,{message:'Continue now'})
    expect(continued.id).toBe(child.id)
    await host.managerPort.wait([child.id],5000)
    expect(calls).toBe(2)
    expect(host.managerPort.listHandles()[0]).toMatchObject({status:'completed',lastOutput:'Continued after interruption',historySessionId:child.historySessionId})
  }finally{await host?.manager.shutdown();await f.close()}
})
