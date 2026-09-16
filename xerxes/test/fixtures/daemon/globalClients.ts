// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, mkdir, rm } from 'node:fs/promises'
import { DaemonRpc } from '../../../src/desktop/main/daemon.js'
import { GatewayClient, daemonPaths, legacyProjectDaemonPaths, resolveProjectDir } from '../../../src/ui/gatewayClient.js'
import { requestDaemonControl } from '../../../src/daemon/controlClient.js'
type Status = Record<string, unknown>
type Sessions = { sessions: Array<{ status: string; turn_count: number }> }
const requests: unknown[] = []
const provider = Bun.serve({ hostname:'127.0.0.1', port:0, async fetch(req) {
 const body = await req.json() as { stream?: boolean }; requests.push(body)
 const text = JSON.stringify(body).includes('ALPHA_WORKSPACE_ONLY') ? 'Alpha verified' : 'Beta verified'
 if (!body.stream) return Response.json({id:'fixture',object:'chat.completion',model:'gpt-4o',choices:[{index:0,message:{role:'assistant',content:text},finish_reason:'stop'}],usage:{prompt_tokens:10,completion_tokens:3,total_tokens:13}})
 const chunks=[{id:'fixture',object:'chat.completion.chunk',choices:[{index:0,delta:{role:'assistant',content:text},finish_reason:null}]},{id:'fixture',object:'chat.completion.chunk',choices:[{index:0,delta:{},finish_reason:'stop'}],usage:{prompt_tokens:10,completion_tokens:3,total_tokens:13}}]
 return new Response(chunks.map(x=>'data: '+JSON.stringify(x)+'\n\n').join('')+'data: [DONE]\n\n',{headers:{'content-type':'text/event-stream'}})
}})
process.env.XERXES_MODEL='gpt-4o'
process.env.XERXES_BASE_URL='http://127.0.0.1:'+provider.port+'/v1'
process.env.XERXES_API_KEY='local-fixture-only'
const home = await mkdtemp('/tmp/xglobal-')
process.env.XERXES_HOME=home
process.env.XERXES_BUN_DAEMON=new URL('../../../src/cli.ts', import.meta.url).pathname
process.env.XERXES_BUN=process.execPath
delete process.env.XERXES_DAEMON_SOCKET
await mkdir(home+'/a'); await mkdir(home+'/b')
await Bun.write(home+'/a/XERXES.md','ALPHA_WORKSPACE_ONLY')
await Bun.write(home+'/b/XERXES.md','BETA_WORKSPACE_ONLY')
const legacyAddress=legacyProjectDaemonPaths(resolveProjectDir(home+'/a'))
const legacy=Bun.spawn([process.execPath,process.env.XERXES_BUN_DAEMON,'daemon','--project-dir',home+'/a','--socket',legacyAddress.socketPath,'--pid-file',legacyAddress.pidPath],{stdout:'ignore',stderr:'ignore'})
let legacyReady=false
const legacyDeadline=Date.now()+10000
while(Date.now()<legacyDeadline) {
 try { await requestDaemonControl(legacyAddress.socketPath,'runtime.status',{}, {timeoutMs:1000}); legacyReady=true; break }
 catch { await new Promise(r=>setTimeout(r,25)) }
}
if(!legacyReady){ legacy.kill(); throw new Error('Legacy fixture did not start') }
const desktop=new DaemonRpc({projectDir:home+'/a',env:process.env,startupTimeoutMs:30000})
const tui=new GatewayClient({projectDir:home+'/b',bunDaemonPath:process.env.XERXES_BUN_DAEMON,bunBinary:process.execPath})
const socket=daemonPaths(home+'/a').socketPath
try {
 const [opened] = await Promise.all([desktop.call('session.open',{session_key:'desktop-a',project_dir:home+'/a'}),tui.start()])
 const openedB=await tui.request('session.open',{session_key:'tui-b',project_dir:home+'/b'})
 const d=await desktop.call<Status>('runtime.status',{})
 const t=await tui.request<Status>('runtime.status',{})
 if (opened.ok !== true || (openedB as Status).ok !== true) throw new Error('Could not open both workspaces')
 console.log('Shared PID:',d.pid)
 if(d.pid===legacy.pid)throw new Error('Idle legacy daemon was not migrated')
 if(await legacy.exited!==0)throw new Error('Legacy daemon did not shut down cleanly')
 console.log('PASS: idle legacy workspace migrated automatically')
 if (JSON.stringify(d)!==JSON.stringify(t)) throw new Error('Different daemon status across clients')
 const list=await tui.request<Sessions>('session.active_list',{})
 if(list.sessions.length!==2)throw new Error('Expected two shared live sessions')
 await Promise.all([desktop.call('turn.submit',{session_key:'desktop-a',text:'Reply hello, no tools needed'}),tui.request('turn.submit',{session_key:'tui-b',text:'Reply hello, no tools needed'})])
 const deadline=Date.now()+20000
 let finished=false
 while(Date.now()<deadline) {
   const state=await desktop.call<Sessions>('session.active_list',{})
   if(state.sessions.every((x)=>x.status==='idle' && x.turn_count>=1)) { finished=true; break }
   await new Promise(r=>setTimeout(r,100))
 }
 if(!finished)throw new Error('Concurrent turns did not finish')
 const alpha=requests.filter(r=>JSON.stringify(r).includes('ALPHA_WORKSPACE_ONLY'))
 const beta=requests.filter(r=>JSON.stringify(r).includes('BETA_WORKSPACE_ONLY'))
 if(!alpha.length||!beta.length||alpha.some(r=>JSON.stringify(r).includes('BETA_WORKSPACE_ONLY'))||beta.some(r=>JSON.stringify(r).includes('ALPHA_WORKSPACE_ONLY'))) throw new Error('Workspace prompt isolation failed')
 console.log('PASS: simultaneous production turns completed through local provider; project instructions stayed isolated')
 const secondWindow = new DaemonRpc({projectDir:home+'/b',env:process.env})
 try {
   const { Store } = await import('../../../src/desktop/renderer/store.js')
   const a = opened as {session:{id:string}}
   Object.assign(globalThis, { window: { xerxes: { onEvent: () => () => {} } } })
   const view = new Store()
   view.start({
     call: (method, params) => secondWindow.call(method, params),
     getWorkspace: async () => home+'/b',
     getResumeSession: async () => a.session.id,
   })
   const until=Date.now()+10000
   while(view.getSnapshot().connection === 'connecting' && Date.now()<until) await new Promise(r=>setTimeout(r,25))
   const snapshot=view.getSnapshot()
   if(snapshot.connection!=='online' || !snapshot.cwd.endsWith('/b') || snapshot.currentId===a.session.id) throw new Error('Desktop stale workspace recovery failed: '+snapshot.error)
   const status=await secondWindow.call<Status>('runtime.status',{})
   if(status.pid!==d.pid) throw new Error('Workspace switch started another daemon')
   const original=await desktop.call<{session:{id:string}}>('session.open',{session_key:'desktop-a'})
   if(original.session.id!==a.session.id) throw new Error('Workspace recovery replaced original session')
   console.log('PASS: real desktop store recovered a cross-project saved session on the SAME daemon; original session unchanged')
 } finally { secondWindow.dispose(); delete (globalThis as {window?:unknown}).window }
 const beforeDetach=await desktop.call<Sessions>('session.active_list',{})
 tui.kill('verification detach')
 const after=await desktop.call<Sessions>('session.active_list',{})
 if(after.sessions.length!==beforeDetach.sessions.length)throw new Error('TUI exit lost sessions')
 console.log('PASS: desktop and TUI share two workspace sessions; TUI exit leaves daemon alive')
} finally {
 tui.close(); desktop.dispose()
 if(legacy.exitCode===null)legacy.kill()
 await requestDaemonControl(socket,'shutdown',{}).catch(error=>console.error(String(error)))
 // Only the isolated verification daemon is stopped.
 await rm(home, {recursive:true,force:true})
 provider.stop(true)
}
