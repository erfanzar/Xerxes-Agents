// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Isolated daemon acceptance data; no external provider or user workspace access.
import {mkdir, realpath} from 'node:fs/promises'
import {join} from 'node:path'
import {DaemonServer} from '../../../src/daemon/server.js'
import {InMemoryDaemonRuntime} from '../../../src/daemon/runtime.js'
import {JobStore} from '../../../src/cron/jobs.js'
import {TerminalRegistry} from '../../../src/runtime/terminalRegistry.js'
const requestedRoot=Bun.argv[2]
if(!requestedRoot?.startsWith('/tmp/xerxes-agent-inspection-'))throw new Error('Use an owned /tmp/xerxes-agent-inspection-* directory')
await mkdir(requestedRoot,{recursive:true})
const root=await realpath(requestedRoot)
const terminals=new TerminalRegistry()
const runtime=new InMemoryDaemonRuntime({async *run(){yield {type:'text_part',payload:{text:'The inspection fixture has a retained child record and a real Bun command. No external provider is used.'}}}},{model:'fixture',...(Bun.env.XERXES_FIXTURE_BUILD_ID ? {buildId:Bun.env.XERXES_FIXTURE_BUILD_ID} : {}),currentProjectDirectory:root,sessionDirectory:join(root,'sessions')})
const session=await runtime.openSession('inspection-session')
await runtime.submitTurn(session.sessionKey,'Inspect the saved agent and command output',()=>{})
session.metadata.title='Agent inspection acceptance'
session.metadata.xerxes_subagent_snapshots_v1=[{id:'inspection-child',title:'Review saved session continuity',status:'completed',agent_id:'reviewer',model:'fixture',provider_profile:'fixture-profile',reasoning_effort:'high',tool_count:2,files_read:['session.ts'],files_written:[],summary:'The fixture retains a completed review for inspection.',last_input:'Inspect session and cancellation boundaries.',last_output:'Retained agent evidence\nCancellation was checked.\nReconnect was checked.\nThis is deterministic acceptance data.'}]
await runtime.flushSessions()
const command="for(let i=0;i<80;i++)console.log('Verified boundary '+i+': output remains readable');"
const handle=terminals.open({id:'inspection-shell',ownerSessionId:session.id,kind:'background',cwd:root,command:'bun -e '+JSON.stringify(command)})
const child=Bun.spawn([process.execPath,'-e',command],{cwd:root,stdout:'pipe',stderr:'pipe'})
handle.append(await new Response(child.stdout).text());handle.close(await child.exited)
const server=new DaemonServer({runtime,terminalRegistry:terminals,projectDirectory:root,socketPath:join(root,'rpc.sock'),cronLeasePath:join(root,'cron.lease'),cronStoreFactory:()=>new JobStore(join(root,'cron/jobs.json'))})
await server.start()
await Bun.write(join(root,'ready.json'),JSON.stringify({pid:process.pid,sessionId:session.id,sessionKey:session.sessionKey,socket:join(root,'rpc.sock')}))
console.log(JSON.stringify({ready:true,pid:process.pid,sessionId:session.id}))
process.once('SIGTERM',()=>{void server.stop().then(()=>process.exit(0))})
