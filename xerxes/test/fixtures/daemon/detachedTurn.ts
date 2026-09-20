// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Opt-in real-process acceptance fixture. No provider, credentials, or user tasks.
import {mkdir} from 'node:fs/promises'
import {join} from 'node:path'
import {DaemonServer} from '../../../src/daemon/server.js'
import {InMemoryDaemonRuntime} from '../../../src/daemon/runtime.js'
import {JobStore} from '../../../src/cron/jobs.js'

const root=Bun.argv[2]
if(!root || !root.startsWith('/tmp/xerxes-detach-'))throw new Error('Choose an owned /tmp/xerxes-detach-* verification directory')
await mkdir(root,{recursive:true})
const runtime=new InMemoryDaemonRuntime({async *run(session,_text,signal){
  await Bun.write(join(root,'started.json'),JSON.stringify({sessionId:session.id,sessionKey:session.sessionKey,pid:process.pid,startedAt:Date.now()}))
  yield {type:'text_part',payload:{text:'Work started. This daemon will finish after the client closes.\n'}}
  while(!signal.aborted && !await Bun.file(join(root,'finish')).exists())await Bun.sleep(50)
  if(signal.aborted){await Bun.write(join(root,'cancelled'),'cancelled');return}
  yield {type:'text_part',payload:{text:'FINISHED_WITHOUT_CLIENT\n'}}
  await Bun.write(join(root,'completed.json'),JSON.stringify({finishedAt:Date.now(),pid:process.pid,cancelled:signal.aborted}))
}},{model:'fixture',currentProjectDirectory:root,sessionDirectory:join(root,'sessions')})
const server=new DaemonServer({runtime,projectDirectory:root,socketPath:join(root,'rpc.sock'),cronStoreFactory:()=>new JobStore(join(root,'cron/jobs.json'))})
await server.start()
console.log(JSON.stringify({ready:true,pid:process.pid,socket:join(root,'rpc.sock')}))
process.once('SIGTERM',()=>{void server.stop().then(()=>process.exit(0))})
