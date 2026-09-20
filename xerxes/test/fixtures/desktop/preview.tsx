// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Isolated desktop stress fixture. Never imported by the production entry. */
import { createRoot } from 'react-dom/client'
import { useSyncExternalStore } from 'react'
import { restoreAppearance } from '../../../src/desktop/renderer/appearance.js'
import { Shell } from '../../../src/desktop/renderer/App.js'
import { BlockBuilder, blocksFromStoredMessages } from '../../../src/desktop/renderer/blocks.js'
import { foldAgentEvent } from '../../../src/desktop/renderer/agentEvents.js'
import { store, type Snapshot } from '../../../src/desktop/renderer/store.js'
import type { TerminalDetail, XerxesBridge } from '../../../src/desktop/renderer/types.js'

const patch = (value: Partial<Snapshot>) => { const port = store as unknown as { patch(value: Partial<Snapshot>): void; builder: { reset(blocks: Snapshot['blocks']): void }; turnCount: number }; if(value.blocks) port.builder.reset(value.blocks); if(value.turnCount !== undefined) port.turnCount=value.turnCount; port.patch(value) }
let fixtureFolderAttempts=0
let fixtureWorkMonitor=false
let fixtureCommandError=false
let fixtureAgentError=false
const cwd='/fixture/xerxes-workspace'
const session=(i:number) => ({ id:`session-${i}`,key:`session-${i}`,title:['Review transport cancellation behavior','Improve workspace file navigation','Investigate provider retry handling'][i%3]+` ${i+1}`,status:'idle',age:`${i+1}m`,current:i===0,kind:'main',turns:i%5,messages:2,cwd,untitled:false })
const long='Preserve every runtime integration and saved session. Review cancellation, reconnect behavior, command discovery and deeply nested file paths. Keep navigation available while work is running. '
const skills=Array.from({length:167},(_,i)=>({name:`workspace-review-${String(i+1).padStart(3,'0')}`,description:`Review workspace behavior ${i+1}, including cancellation, keyboard access and saved sessions.`,kind:'skill',source:'fixture',enabled:true}))
let schedules=Array.from({length:12},(_,i)=>({id:`schedule-${i}`,prompt:`Review workspace ${i+1}: ${long}`,schedule:'0 8 * * 1-5',timezone:'UTC',paused:i%3===0,execution_state:'idle',next_run_at:'2026-09-14T08:00:00Z',revision:1}))
const scheduledRuns=Array.from({length:8},(_,i)=>({id:`run-${i}`,ownerSessionId:'session-0',sourceId:'schedule-0',workspace:cwd,kind:i%2===0?'schedule':'terminal',revision:1,unread:true,title:'Review workspace '+(i+1),state:i===0?'running':i%3===0?'failed':'succeeded',startedAt:Date.UTC(2026,8,12-i,8),error:i%3===0?'Provider configuration is missing a model.':null,output:Array.from({length:120},(_,line)=>`  Step ${line+1}: checked workspace state`).join('\n'),outputTruncated:false,exitCode:i%3===0?1:0}))
const base: Partial<Snapshot>={connection:'online',cwd,model:'fixture/model-long-context',currentId:'session-0',sessionKey:'session-0',currentTitle:'Review the desktop workspace',sessions:Array.from({length:40},(_,i)=>session(i)),models:Array.from({length:60},(_,i)=>({id:`fixture/model-${String(i).padStart(2,'0')}-long-context`,provider:'Fixture'})),reasoningEffort:'high',reasoningLevels:['low','medium','high','xhigh'].map(effort=>({effort,label:effort})),permissionMode:'ask',goal:`Objective: ${long.repeat(5)}\nStatus: active\nRounds: 4`,fleet:Array.from({length:8},(_,i)=>({...session(i),id:`agent-${i}`,title:['Inspect transport cancellation and reconnect behavior','Review session persistence across workspace changes','Audit file navigation and deeply nested project paths','Verify model selection and effort menu keyboard access','Check the composer with a long multiline draft','Inspect streaming and recoverable provider errors','Review settings and onboarding affordances','Check the large diff and code review gutters'][i]!,kind:'subagent',turns:0,status:i<3?'working':'done',age:''})),commands:[{name:'help',description:'Show available commands'},{name:'goal',description:'Manage the task objective'},{name:'skill',description:'Run a workspace skill'}],turnCount:4,contextTokens:62100,contextMax:262144,llmSteps:24,toolSteps:48,llmDurationMs:465000,toolDurationMs:170000,ttftMs:5900,tokensPerSecond:99.6,cacheHitRate:0.5,inputTokens:123000,daemonWarning:null,noWorkspace:false}
const blocks: Snapshot['blocks']=[{kind:'user',id:1,text:long+'\n\n'+long+'\n\nConfirm the result with populated states.'},{kind:'agent',id:2,text:'The layout keeps navigation independent of the conversation.\n\nI am checking the following behaviors:\n\n- Persistent sidebar widths\n- Full-width multiline composer\n- File selection and code review\n\n```ts\nconst retry = await transport.reconnect({ signal });\nif (retry.ready) resumeSession(session.id);\n```\n\nAuthentication and certificate errors remain visible.',streaming:false}]
blocks.push(
  {kind:'thinking',id:3,text:'Inspecting deeply nested transport code before changing reconnect behavior.',streaming:false},
  {kind:'tools',id:4,running:false,items:[{id:'fixture-exec',verb:'exec_command',name:'exec_command',arg:'sed -n 10,30p src/runtime/transport/connections/recovery/session-reconnect-controller.ts',state:'done',dur:'0.0s',input:JSON.stringify({cmd:'sed',args:['-n','10,30p','src/runtime/transport/connections/recovery/session-reconnect-controller.ts']}),output:JSON.stringify({command:['sed','-n','10,30p','src/runtime/transport/connections/recovery/session-reconnect-controller.ts'],stdout:'export async function reconnect(signal: AbortSignal) {\n  const connection = await connect({ signal });\n  return connection;\n}\n',stderr:'',exitCode:0,cwd})}]},
  {kind:'thinking',id:5,text:'Checking the error path with a deterministic failed operation.',streaming:false},
  {kind:'tools',id:6,running:false,items:[{id:'fixture-failure',verb:'exec_command',name:'exec_command',arg:'bun test transport',state:'failed',dur:'1.2s',input:JSON.stringify({cmd:'bun',args:['test','transport']}),output:JSON.stringify({stdout:'1 test failed\n',stderr:'Expected cancellation to preserve the saved session.\nReceived: session missing\n',exitCode:1,cwd})}]}
)
const lines=[{kind:'file',text:'src/runtime/transport/connections/recovery/session-reconnect-controller.ts'},{kind:'hunk',text:'@@ -1,200 +1,200 @@'},...Array.from({length:200},(_,i)=>({kind:i%9===0?'add':i%11===0?'del':'context',text:(i%9===0?'+':i%11===0?'-':' ')+`  const connection${i} = await reconnectSession(session.id, { signal, preserveDraft: true });`,oldLine:i%9===0?undefined:i+1,newLine:i%11===0?undefined:i+1})),{kind:'file',text:'src/desktop/layout.ts'}, {kind:'hunk',text:'@@ -1 +1 @@'}, {kind:'add',text:'+ export const width = 760',newLine:1}]
let fixtureExportDenied=false
let specialistRows=Array.from({length:8},(_,i)=>({id:'reviewer-'+i,description:long,error:''}))
const specialistBodies:Record<string,{content:string;revision:string}>=Object.fromEntries(specialistRows.map(row=>[row.id,{content:'---\nname: '+row.id+'\ndescription: Review workspace changes.\n---\nCheck cancellation and persistence.\n',revision:'revision-1'}]))
let fixturePresets=[{id:'reviewer',name:'Workspace reviewer',description:'Review runtime changes and report verified defects.',trust:'system',is_default:true,manageable:false},{id:'unavailable',name:'Unavailable preset',description:'Exercises a file that becomes unreadable after discovery.',trust:'project',is_default:false,manageable:true}]
const presetBodies:Record<string,string>={reviewer:'# Workspace reviewer\n\nReview runtime changes and report verified defects.'}
let fixtureChannels=[{name:'fixture-gateway',adapter_name:'Fixture',enabled:false},{name:'fixture-denied',adapter_name:'Fixture',enabled:false}]
let fixtureTerminal: TerminalDetail = { id:'fixture-shell',kind:'pty',label:'Review shell',command:'bun test src/runtime/transport/connections/recovery.test.ts',cwd,running:true,startedAt:Date.now(),exitCode:null,outputChars:50,canWrite:true,canInterrupt:true,canKill:true,output:'Ready for input.\n  Whitespace is retained.\n',outputTruncated:false }
const fixtureDeliveries=[{id:'delivery-1',platform:'fixture',recipient:'test-room',state:'uncertain',attempts:1,content:'Saved run output for review.'}]
let monitorRows:Record<string,unknown>[]=[]
let goalRevision=1
let goalEvidence:unknown=undefined
let contextRevision=1
const contextControls=new Map<string,{pinned:boolean;excluded:boolean}>()
let lspRevision=1
let lspServers=[{name:'typescript',enabled:true,languageId:'typescript',extensions:['.ts','.tsx'],timeoutMs:30000,configuredFields:['command','args']}]
const bridge: XerxesBridge={onEvent:()=>()=>{},async call<T>(method:string,params:Record<string,unknown>={}) {
  let result:unknown={ok:false,error:`Fixture does not implement ${method}`}
  if(method==='complete') {const query=String(params.text||'');result={completions:query.startsWith('./') ? [{label:'src/',value:'./src/',meta:'dir'},{label:'session-reconnect-controller.ts',value:'./src/runtime/transport/connections/recovery/session-reconnect-controller.ts',meta:'file'},{label:'unavailable.txt',value:'./unavailable.txt',meta:'file'}] : skills.filter(s=>s.name.includes(query.replace('/skill ','').replace('/',''))).map(s=>({label:s.name,value:`/skill ${s.name}`,meta:s.description}))}}
  else if(method==='monitor.list')result={ok:true,monitors:monitorRows}
  else if(method==='monitor.sources')result={ok:true,webhooks:[{name:'build-events'}]}
  else if(method==='monitor.create'){
    const kind=String(params.source_kind||'terminal');const source=kind==='file'?{kind,path:params.file_path,workspace:cwd}:kind==='websocket'?{kind,url:params.websocket_url}:kind==='webhook'?{kind,name:params.webhook_name}:{kind,terminalId:params.terminal_id}
    const row={id:'watch-'+monitorRows.length,terminalId:String(params.terminal_id||''),match:String(params.match||''),state:'watching',trigger:params.trigger,source,expiresAt:Date.now()+3600000,stopAction:'stop-watch',events:[{text:'Source attached; waiting for a match.'}],...(params.react?{reactionHealth:{state:'waiting',attempts:0,maxReactions:params.max_reactions,policy:{revision:'1',maxReactions:params.max_reactions,maxDurationMs:Number(params.reaction_timeout_seconds)*1000,maxTotalTokens:params.max_total_tokens??null}}}:{})};monitorRows.push(row);result={ok:true,monitor:row}
  }
  else if(method==='monitor.inspect'||method==='monitor.stop'||method==='monitor.update'){
    const row=monitorRows.find(row=>row.id===params.monitor_id)
    if(!row)result={ok:false,error:'Monitor unavailable'}
    else if(method==='monitor.update'&&params.max_reactions===9)result={ok:false,error:'Fixture policy revision changed; refresh and retry.'}
    else{if(method==='monitor.stop'){row.state='stopped';row.stopAction=null}if(method==='monitor.update')row.reactionHealth={state:'waiting',attempts:0,maxReactions:params.max_reactions,policy:{revision:'2',maxReactions:params.max_reactions,maxDurationMs:Number(params.reaction_timeout_seconds)*1000,maxTotalTokens:params.max_total_tokens}};result={ok:true,monitor:row}}
  }
  else if(method==='goal.inspect'||method==='goal.decision') {
    if(method==='goal.decision'&&(params.session_key!==undefined||params.revision!==goalRevision||params.session_id!=='session-0'||params.goal_id!=='fixture-goal')) result={ok:false,error:'The goal changed. Refresh and review it again.'}
    else if(method==='goal.decision'&&String(params.summary).includes('reject')) result={ok:false,error:'Fixture decision rejected.'}
    else {
      if(method==='goal.decision'){goalRevision++;goalEvidence={kind:'user-decision',decisionId:'fixture-decision',summary:String(params.summary),recordedAt:Date.now()}}
      result={ok:true,session_id:'session-0',goal:{id:'fixture-goal',revision:goalRevision,objective:long,phase:'active',roundsStarted:4,maxGoalRounds:20,currentMilestone:'Verify the work before accepting it.',criteria:[{id:'tests',description:'Preserve session isolation and verify cancellation.',...(goalEvidence?{evidence:goalEvidence}:{})}]},token_usage:{inputTokens:1000,outputTokens:200,measuredCalls:2,settledCalls:2,pendingCalls:0,complete:true},continuation:null}
    }
  }
  else if(method==='context.inspect') {
    const section=String(params.section||'instructions'), offset=Number(params.offset||0)
    result={ok:true,generation:'context-'+contextRevision,controls_revision:contextRevision,note:'Fixture context; no provider request.',section,offset,next_offset:offset===0?20:null,sections:['instructions','memory','conversation','tools','compaction'].map(id=>({id,count:21,available:true,estimated_tokens:2300,provenance:'Saved session'})),entries:Array.from({length:offset===0?20:1},(_,i)=>({index:offset+i,title:'Source '+(offset+i+1),text:'Preserve source line breaks.\nThis is a retained context excerpt.',truncated:false,estimated_tokens:100,...(section==='memory'?{control:{scope:'project',path:'notes-'+(offset+i)+'.md',...(contextControls.get('notes-'+(offset+i)+'.md')||{pinned:false,excluded:false})}}:{})}))}
  }
  else if(method==='context.control') {
    if(params.path==='notes-1.md') result={ok:false,error:'Context changed; refresh before changing controls.'}
    else {const key=String(params.path),previous=contextControls.get(key)||{pinned:false,excluded:false};contextControls.set(key,{pinned:params.action==='pin'?true:params.action==='unpin'||params.action==='exclude'?false:previous.pinned,excluded:params.action==='exclude'?true:params.action==='include'||params.action==='pin'?false:previous.excluded});contextRevision++;result={ok:true,revision:contextRevision,applies_next_turn:true}}
  }
  else if(method==='lsp.settings.get') result={ok:true,revision:String(lspRevision),servers:lspServers,warnings:[]}
  else if(method==='lsp.release') result={ok:true,message:'Host released; the next request starts it lazily.'}
  else if(method==='lsp.settings.save') {
    const changes=params.changes as Record<string,unknown>|undefined
    if(params.revision!==String(lspRevision)) result={ok:false,error:'Settings changed; reload before saving.'}
    else if(changes?.command==='reject') result={ok:false,error:'Fixture rejected executable; correct it and retry.'}
    else {
      if(params.action==='remove') lspServers=lspServers.filter(row=>row.name!==params.name)
      else {
        const next={name:String(params.name),enabled:changes?.enabled===true,languageId:String(changes?.languageId),extensions:changes?.extensions as string[],timeoutMs:Number(changes?.timeoutMs),configuredFields:['command']}
        lspServers=params.action==='create'?[...lspServers,next]:lspServers.map(row=>row.name===params.name?next:row)
      }
      lspRevision++;result={ok:true,revision:String(lspRevision),servers:lspServers,warnings:[]}
    }
  }
  else if(method==='reasoning_levels')result={levels:['low','medium','high','xhigh'].map(effort=>({effort,description:effort+' reasoning'})),current:store.getSnapshot().reasoningEffort}
  else if(method==='provider_list')result={ok:true,profiles:[{name:'Workspace review provider with a long descriptive name',provider:'openai',model:'fixture/model-long-context',active:true},{name:'Secondary development provider',provider:'openai',model:'fixture/review-model',active:false}]}
  else if(method==='provider_types')result={ok:true,types:[{name:'openai',base_url:'https://api.openai.com/v1',api_key_env:'OPENAI_API_KEY'}]}
  else if(method==='capabilities.list') result={skills,tools:[],plugins:[]}
  else if(method==='capabilities.inspect') result={name:params.name,description:long,instructions:'# Workspace review\n\n'+long}
  else if(method==='workspace.filePreview'){await new Promise(resolve=>setTimeout(resolve,String(params.path).includes('unavailable')?100:700));if(String(params.path).includes('unavailable'))result={ok:false,error:'Fixture file cannot be read. Choose another file.'};else result={ok:true,path:params.path,content:Array.from({length:240},(_,i)=>`export const recoveryStep${i + 1} = async (session: Session) => resumeSession(session.id);`).join('\n'),truncated:false}}
  else if(method==='workspace.list')result={ok:true,inventory:{records:[{id:'managed-1',path:'/fixture/isolated/task',branch:'task/review',taskId:'session-0'}]}}
  else if(method==='workspace.inspect')result={ok:true,review:{id:'managed-1',path:'/fixture/isolated/task',branch:'task/review',taskId:'session-0',reviewId:'review-1',base:'base-sha',head:'head-sha',status:'ready',diff:'diff --git a/src/check.ts b/src/check.ts\n--- a/src/check.ts\n+++ b/src/check.ts\n@@ -1 +1 @@\n-export const value = false\n+export const value = true\n'}}
  else if(method==='workspace.checkApply')result={ok:true,check:{reviewId:'review-1',canApply:true,destination:cwd,destinationState:'a'.repeat(64)}}
  else if(method==='workspace.apply')result={ok:false,error:'Destination changed since the check; review and check again.'}
  else if(method==='workspace.integrations')result={ok:true,inventory:{records:[{id:'integration-1',backupPath:'/fixture/backups/one',destination:cwd,status:'needs-recovery',paths:['src/check.ts']}]}}
  else if(method==='workspace.integration.inspect')result={ok:true,inspection:{id:'integration-1',files:[{path:'src/check.ts',action:'conflict',reason:'Concurrent edit will be preserved.'}]}}
  else if(method==='workspace.recover')result={ok:true,recovery:{id:'integration-1',status:'needs-recovery',conflicts:['src/check.ts']}}
  else if(method==='workspace.diff')result={kind:'diff',diff:{lines,untracked:[]}}
  else if(method==='background.activity')result={rows:fixtureWorkMonitor?[{id:'fixture-shell',kind:'shell',title:'env CHECK_MODE=full '+Array.from({length:40},(_,i)=>`--completed-xml .cache/results/verification-pass-${i}.xml`).join(' ')+' bun test',state:'running',detail:cwd,startedAt:Date.now()-320000},...Array.from({length:36},(_,i)=>({id:'old-'+i,kind:'shell',title:'bun test previous-'+i,state:i%2?'failed':'completed',detail:cwd}))]:new URLSearchParams(location.search).get('scenario')==='Background history'?[{id:'live-watch',kind:'watcher',title:'Watch source changes',state:'watching',detail:'README.md'},...Array.from({length:120},(_,i)=>({id:'old-'+i,kind:'shell',title:'Completed validation '+i,state:i%10===0?'failed':'completed',detail:cwd}))]:[]}
  else if(method==='background.status')result={shells:0,watchers:0}
  else if(method==='schedule.deliveries')result={ok:true,deliveries:fixtureDeliveries}
  else if(method==='schedule.delivery.inspect')result={ok:true,delivery:fixtureDeliveries.find(row=>row.id===params.delivery_id)}
  else if(method==='schedule.delivery.resolve'){const row=fixtureDeliveries.find(row=>row.id===params.delivery_id);if(!row||row.attempts!==params.attempts)result={ok:false,error:'Delivery changed; refresh first.'};else{row.state=params.decision==='sent'?'sent':'pending';result={ok:true}}}
  else if(method==='schedule.delivery.send')result={ok:false,error:'Fixture delivery unavailable. No message was sent.'}
  else if(method==='schedule.list')result={ok:true,jobs:schedules}
  else if(method==='schedule.preview')result=String(params.schedule).split(' ').length===5?{ok:true,next_run_at:'2026-09-14T08:00:00Z'}:{ok:false,error:'Invalid cron expression'}
  else if(method==='schedule.create'){schedules=[...schedules,{id:`schedule-${schedules.length}`,prompt:String(params.prompt),schedule:String(params.schedule),timezone:String(params.timezone),paused:false,execution_state:'idle',next_run_at:'2026-09-14T08:00:00Z',revision:1}];result={ok:true}}
  else if(method==='schedule.update'){schedules=schedules.map(job=>job.id===params.schedule_id?{...job,prompt:String(params.prompt),schedule:String(params.schedule),timezone:String(params.timezone),revision:job.revision+1}:job);result={ok:true}}
  else if(method==='schedule.pause'||method==='schedule.resume'){schedules=schedules.map(job=>job.id===params.schedule_id?{...job,paused:method==='schedule.pause'}:job);result={ok:true}}
  else if(method==='schedule.remove'){schedules=schedules.filter(job=>job.id!==params.schedule_id);result={ok:true}}
  else if(method==='run.list')result={ok:true,runs:scheduledRuns.filter(run=>(!params.kind||run.kind===params.kind)&&(!params.state||run.state===params.state)&&(!params.unread_only||run.unread)&&(typeof params.before_started_at!=='number'||run.startedAt<params.before_started_at)).slice(0,4).map(({output,...run})=>run),has_more:params.before_started_at===undefined}
  else if(method==='run.inspect'){const run=scheduledRuns.find(run=>run.id===params.run_id);result={ok:true,run:run?{...run,cancel_label:run.state==='running'?'Cancel run':null}:null}}
  else if(method==='run.acknowledge'||method==='run.cancel') {const run=scheduledRuns.find(run=>run.id===params.run_id);if(!run||run.revision!==params.revision)result={ok:false,error:'Run changed; refresh before changing it.'};else if(method==='run.cancel')result={ok:false,error:'Fixture cancellation rejected; refresh before cancelling.'};else{run.unread=false;run.revision++;result={ok:true,run}}}
  else if(method==='agentPreset.projectList')result={ok:true,agents:specialistRows}
  else if(method==='agentPreset.projectRead')result={ok:true,...specialistBodies[String(params.id)]}
  else if(method==='agentPreset.projectGenerate'){
    await new Promise(resolve=>setTimeout(resolve,1000))
    result=String(params.description).includes('reject')?{ok:false,error:'Fixture provider rejected specialist generation.'}:{ok:true,id:'generated-reviewer',content:'---\nname: generated-reviewer\ndescription: Review workspace changes.\n---\nCheck cancellation and persistence.\n'}
  }
  else if(method==='agentPreset.projectWrite'){
    await new Promise(resolve=>setTimeout(resolve,1000))
    const content=String(params.content),id=String(params.id||content.match(/name:\s*([a-z0-9-]+)/)?.[1]||'')
    if(!id||content.includes('invalid: ['))result={ok:false,error:'Invalid specialist: provide valid YAML name and description.'}
    else if(params.revision!==null&&specialistBodies[id]?.revision!==params.revision)result={ok:false,error:'Specialist changed since it was opened. Reload before saving.'}
    else {specialistBodies[id]={content,revision:'revision-'+Date.now()};if(!specialistRows.some(row=>row.id===id))specialistRows.push({id,description:'Generated reviewer',error:''});result={ok:true,id}}
  }
  else if(method==='agentPreset.list')result={ok:true,presets:fixturePresets}
  else if(method==='agentPreset.read')result=params.agent_preset==='unavailable'?{ok:false,error:'Preset file is unavailable. Refresh the list and check file permissions.'}:{ok:true,content:presetBodies[String(params.agent_preset)]||''}
  else if(method==='agentPreset.copy'){
    await new Promise(resolve=>setTimeout(resolve,1000))
    const id=String(params.agent_preset)
    if(id==='rejected-preset')result={ok:false,error:'Fixture preset location is read-only. Choose a writable destination.'}
    else if(fixturePresets.some(row=>row.id===id))result={ok:false,error:'A preset with this identifier already exists.'}
    else {fixturePresets.push({id,name:String(params.name||id),description:'Isolated editable preset',trust:'project',is_default:false,manageable:true});presetBodies[id]=presetBodies[String(params.from)]||'';result={ok:true}}
  }
  else if(method==='agentPreset.write'){
    await new Promise(resolve=>setTimeout(resolve,1000))
    if(String(params.content).includes('invalid: ['))result={ok:false,error:'Invalid preset YAML: unclosed sequence.'}
    else {presetBodies[String(params.agent_preset)]=String(params.content);result={ok:true}}
  }
  else if(method==='agentPreset.remove'){
    await new Promise(resolve=>setTimeout(resolve,600))
    if(params.agent_preset==='unavailable')result={ok:false,error:'Fixture preset cannot be removed while its files are read-only.'}
    else {fixturePresets=fixturePresets.filter(row=>row.id!==params.agent_preset);result={ok:true}}
  }
  else if(method==='agentPreset.openDocument')result={ok:true,path:'/fixture/presets/'+String(params.agent_preset)}
  else if(method==='agentPreset.select')result=params.agent_preset==='unavailable'?{ok:false,error:'This preset is unavailable for the current session.'}:{ok:true,agent_preset:params.agent_preset}
  else if(method==='agentPreset.setDefault'){fixturePresets=fixturePresets.map(row=>({...row,is_default:row.id===params.agent_preset}));result={ok:true}}
  else if(method==='session.list')result={sessions:base.sessions}
  else if(method==='session.title'){
    if(params.title==='Rejected title')result={ok:false,error:'This session is read-only. Choose another session or retry after reconnecting.'}
    else {const title=String(params.title);base.sessions=base.sessions.map(row=>row.key===params.session_key?{...row,title}:row);patch({sessions:base.sessions,...(params.session_key===base.sessionKey?{currentTitle:title}:{})});result={ok:true}}
  }
  else if(method==='session.status'&&fixtureExportDenied)result={ok:false,error:'Transcript export denied by the fixture runtime.'}
  else if(method==='session.status')result={session:{id:params.session_key,title:base.sessions.find(row=>row.key===params.session_key)?.title||'Fixture export',cwd,model:base.model,mcp_status:{"fixture-files":{connected:true,tools:12,resources:2,prompts:1},"fixture-review":{connected:false,lastError:"Fixture endpoint unavailable"}},messages:2,transcript:[{role:'user',content:long},{role:'assistant',content:'Verified export fixture.\n\nWhitespace stays readable.'}]}}
  else if(method==='session.active_list')result={sessions:[]}
  else if(method==='subagent.interrupt')result={ok:false,error:'This fixture runtime does not support stopping the selected agent.'}
  else if(method==='subagent.retry')result=params.message==='deny'?{ok:false,error:'Retry denied by runtime policy'}:{ok:true,agent:{id:params.task,status:'running'}}
  else if(method==='initialize'){
    const id=String(params.resume_session_id||'session-0')
    const transcript=new URLSearchParams(location.search).has('resume-stress')
      ? Array.from({length:160},(_,i)=>({role:i%2?'assistant':'user',content:`${id} message ${i+1}\n\n${long.repeat(3)}`}))
      : undefined
    await new Promise(resolve=>setTimeout(resolve,120))
    result={session_id:id,cwd,model:base.model,session:{id,key:id,title:`Review workspace ${id}`,cwd,transcript}}
  }
  else if(method==='slash' && params.command==='/reload-mcp')result={ok:false,error:'Fixture MCP configuration is unreadable'}
  else if(method==='slash'){const command=String(params.command||'');if(command.startsWith('/model '))patch({model:command.slice(7)});if(command.startsWith('/thinking '))patch({reasoningEffort:command.slice(10)});result={ok:true,text:command.startsWith('/goal')?base.goal:''}}
  else if(method==='context_breakdown')result={system_prompt_tokens:2100,tools_tokens:10000,messages_tokens:50000,total_tokens:62100,context_limit:262144}
  else if(method==='turn.cancel'){patch({turnActive:false,steerQueue:[]});result={ok:true}}
  else if(method==='turn.submit') {setTimeout(()=>patch({blocks:[{kind:'user',id:10,text:String(params.message||params.text||'Fixture message')},blocks[1]!],turnActive:false,turnCount:5}),0);result={ok:true}}
  else if(method==='channel.list')result={ok:true,channels_available:true,channels_configured:true,channels:fixtureChannels}
  else if(method==='channel.enable'||method==='channel.disable'){
    await new Promise(resolve=>setTimeout(resolve,1600))
    if(params.name==='fixture-denied')result={ok:false,error:'Fixture gateway credentials unavailable'}
    else {fixtureChannels=fixtureChannels.map(row=>row.name===params.name?{...row,enabled:method==='channel.enable'}:row);result={ok:true,channels:fixtureChannels}}
  }
  else if(method==='runtime.status')result={ok:true}
  else if(method==='terminal.list')result={ok:true,terminals:[fixtureTerminal]}
  else if(method==='subagent.inspect'){
    await new Promise(resolve=>setTimeout(resolve,params.task==='live-0'?180:30))
    result=fixtureAgentError?{ok:false,error:'Agent detail read failed. Retry.'}:{ok:true,agent:{id:params.task,agent_id:'reviewer',model:'fixture/model',provider_profile:'work-profile',reasoning_effort:'high',prompt:'Review cancellation, session persistence and reconnect handling without changing unrelated work.',output:'Review started.\nInspected src/runtime/session.ts\nCancellation retains the saved transcript.\n'+Array.from({length:80},(_,i)=>'Evidence '+(i+1)+': verified boundary').join('\n')}}
  }
  else if(method==='terminal.inspect')result=fixtureCommandError?{ok:false,error:'Command output is temporarily unavailable. Retry.'}:{ok:true,terminal:fixtureTerminal}
  else if(method==='terminal.control'){
    await new Promise(resolve=>setTimeout(resolve,1200))
    if(String(params.chars).includes('reject'))result={ok:false,error:'Fixture terminal input denied. The draft has not been sent.'}
    else {
      fixtureTerminal=params.action==='write'?{...fixtureTerminal,output:fixtureTerminal.output+String(params.chars)}:{...fixtureTerminal,running:false,exitCode:params.action==='interrupt'?130:143,endedAt:Date.now()}
      result={ok:true,terminal:fixtureTerminal}
    }
  }
  return result as T
}, openPath:async path=>!path.endsWith('/unavailable'), getWorkspace:async()=>cwd,chooseWorkspace:async()=>{await new Promise(resolve=>setTimeout(resolve,800));if(++fixtureFolderAttempts===1)throw new Error("Fixture folder access denied. Choose an accessible folder.");patch({cwd:'/fixture/second-workspace'});return true},useWorkspace:async dir=>{patch({cwd:dir});return true}}
Object.defineProperty(window,'xerxes',{value:bridge})
;(store as unknown as {bridge:XerxesBridge}).bridge=bridge
localStorage.setItem('xerxes.desktop.setup.v1','done')
restoreAppearance()
base.fleet=base.fleet?.map((row,i)=>({...row,status:i===3?"failed":row.status,agentDetails:{model:"fixture/reviewer",toolCount:12+i,summary:i<3?"Reviewing session persistence and cancellation boundaries.":"Checked the workspace behavior and recorded the findings.",error:i===3?"Permission denied while reading the protected workspace file.":"",inputTokens:4200+i*100,outputTokens:800,filesRead:["src/runtime/transport/connections/recovery/session-reconnect-controller.ts"],filesWritten:i<3?[]:["src/desktop/layout.ts"]}}));
patch({...base,blocks})
function Preview(){return <Shell snap={useSyncExternalStore(store.subscribe,store.getSnapshot)} />}
createRoot(document.getElementById('root')!).render(<Preview/> )
const host=window as unknown as {fixture?:{onScenario(handler:(name:string)=>void):void}}
let activityBuilder = new BlockBuilder()
const scenario = (name:string) => {
  if(name==='Agent failure'){fixtureAgentError=true;return}
  if(name==='Agent recover'){fixtureAgentError=false;return}
  if(name==='Command failure'){fixtureCommandError=true;return}
  if(name==='Command recover'){fixtureCommandError=false;return}
  if(name==='Command disconnect'){patch({connection:'offline'});return}
  if(name==='Command reconnect'){patch({connection:'online'});return}
  fixtureWorkMonitor=name==='Work monitor'
  if(fixtureWorkMonitor)fixtureTerminal={...fixtureTerminal,output:'Ready for input.\nRunning the workspace verification suite.\n'+Array.from({length:120},(_,i)=>i%12===0?'FAIL tests/runtime/recovery/session-reconnect.test.ts — expected retained session, received missing history':'PASS tests/runtime/persistence/checkpoint-'+i+'.test.ts · saved state retained').join('\n')}
  if (name.startsWith('Activity ')) {
    if (name === 'Activity start') {
      activityBuilder = new BlockBuilder()
      activityBuilder.push('think_part', {think: 'Checking the workspace before calling tools.'})
    }
    if (name === 'Activity tool') activityBuilder.push('tool_call', {id: 'slow-call', name: 'exec_command', arguments: {cmd: 'bun test'}})
    if (name === 'Activity notice') {
      activityBuilder.push('notification', {message: 'Background reviewer finished'})
      activityBuilder.pushAgents([{key: 'reviewer', title: 'Review workspace', status: 'working'}])
    }
    if (name === 'Activity result') activityBuilder.push('tool_result', {tool_call_id: 'slow-call', return_value: 'Verification output retained', duration_ms: 1200})
    if (name === 'Activity done') {
      activityBuilder.pushAgents([{key: 'reviewer', title: 'Review workspace', status: 'done'}])
      activityBuilder.finalize()
    }
    patch({...base, goal: '', fleet: [], turnActive: name !== 'Activity done', blocks: activityBuilder.snapshot(name !== 'Activity done')})
    return
  }
  fixtureExportDenied=name==='Export error'
  patch({...base,blocks,turnActive:false,failed:null,networkRetrying:false,settingsOpen:false,pickerOpen:false,modelMenuOpen:false,reasoningPickerOpen:false})
  if(name==='Work monitor')patch({goal:'',turnActive:true,fleet:[...Array.from({length:58},(_,i)=>({...session(i),id:'past-'+i,kind:'subagent',status:i%3?'completed':'failed',title:'Previous review '+i,agentDetails:{summary:'Previous attempt finished.',error:i%3?'':'Previous attempt failed',model:'fixture/model',filesRead:[],filesWritten:[]}})),...Array.from({length:3},(_,i)=>({...session(i),id:'live-'+i,kind:'subagent',status:'running',title:'Running review '+(i+1),agentDetails:{baseAgent:i===2?'test-engineer':'reviewer',model:'fixture/model',providerProfile:'work-profile',reasoningEffort:'high',goal:'Review cancellation and reconnect boundaries for task '+(i+1),summary:'Inspecting session recovery and recorded output.',error:'',filesRead:['src/runtime/session.ts'],filesWritten:[],notes:['Checking the live cancellation path.','The saved transcript survives reconnect.'],toolCalls:[{id:'inspect-'+i,name:'ReadFile',verb:'Read file',arg:'src/runtime/session.ts',input:'{}',state:'done',dur:'0.1s',output:'export const preserveSession = true;'}]}}))],blocks:[{kind:'user',id:900,text:'Review the current changes while the test command runs.'},{kind:'agents',id:901,members:Array.from({length:3},(_,i)=>({key:'call-'+i,runtimeId:'live-'+i,title:'Running review '+(i+1),status:'working'}))}]})
  if(name==='Artifacts')patch({changes:[{path:'src/runtime/transport/connections/recovery/session-reconnect-controller.ts',adds:24,dels:8,isNew:false,hunks:[],turn:1},{path:'src/desktop/layout.ts',adds:1,dels:0,isNew:true,hunks:[],turn:1}]})
  if(name==='Runtime update')patch({daemonWarning:'The workspace runtime predates this app build. Existing work is still running.',turnActive:true})
  if(name==='Runtime offline')patch({connection:'offline'})
  if(name==='Agent timeline'){
    let fleet:Snapshot['fleet']=[]
    const events=[['turn_begin',{}],['think_part',{think:'Checking cancellation before changing files.'}],['tool_call',{id:'read-1',name:'read_file',arguments:'src/runtime/transport/connections/recovery.ts'}],['tool_result',{tool_call_id:'read-1',name:'read_file',return_value:'export function recover() {\n  return connect();\n}',duration_ms:120}],['tool_call',{id:'test-1',name:'exec_command',arguments:'bun test recovery'}],['tool_result',{tool_call_id:'test-1',name:'exec_command',return_value:'Permission denied',permitted:false,duration_ms:900}],['turn_end',{status:'failed',summary:'The test command was denied by policy.'}]] as const
    for(const [type,payload] of events)fleet=foldAgentEvent(fleet,{agent_id:'timeline-agent',title:'Review transport recovery',model:'fixture/reviewer',goal:'Verify cancellation and reconnect behavior.',tool_count:2,event:{type,payload}})
    patch({fleet})
  }
  if(name==='Queued context')patch({queue:[{id:90001,text:'When do you think it will be finished?\nPlease include the remaining checks.'}]})
  if(name==='Welcome')patch({blocks:[],turnCount:0,goal:'',fleet:[],todos:null,plan:null,llmSteps:0,toolSteps:0,llmDurationMs:0,toolDurationMs:0,inputTokens:0,ttftMs:null,tokensPerSecond:null,cacheHitRate:null})
  if(name==='Continuation')patch({blocks:[]})
  if(name==='Research')patch({goal:'Status: active\nObjective: Compare three workspace options and prepare a decision brief.\nRounds: 2',todos:[{id:'a',content:'Read the supplied research notes',status:'completed'},{id:'b',content:'Compare the options and identify missing evidence',status:'in_progress'},{id:'c',content:'Write the decision brief',status:'pending'}],blocks:[{kind:'user',id:101,text:'Compare the three options in my notes. Explain the tradeoffs and tell me what still needs verification.'},{kind:'tools',id:102,running:false,items:[{id:'research-read',verb:'read_file',name:'read_file',arg:'research/workspace-options.md',path:'research/workspace-options.md',dur:'0.2s',state:'done',input:'{}',output:'Synthetic verification notes\nOption A: flexible membership; limited meeting space.\nOption B: dedicated rooms; annual commitment.\nOption C: remote-first; requires separate event space.'}]},{kind:'agent',id:103,text:'## Comparison from your notes\n\nThis is synthetic content for interface verification.\n\n| Option | Advantage | Open question |\n| --- | --- | --- |\n| A | Flexible membership | Can meeting rooms be reserved reliably? |\n| B | Dedicated space | What are the cancellation terms? |\n| C | Remote flexibility | What does occasional event space cost? |\n\nThe notes support a comparison, but not a final recommendation. I’m separating confirmed details from the questions that need evidence.',streaming:false}]})
  if(name==='Goal output' || name==='Long output')patch({blocks:[{kind:'tools',id:901,running:false,items:[{id:'result-preview',verb:'Get goal',name:'get_goal',arg:'{}',input:'{}',state:'done',dur:'',output:name==='Long output' ? Array.from({length:200},(_,i)=>`Line ${i+1}: ${'Verify runtime behavior. '.repeat(8)}`).join('\n') : JSON.stringify({goal:{id:'goal-example',revision:53,objective:'Finish the roadmap while preserving every runtime integration and saved session.',phase:'paused',currentMilestone:'Qualify runtime contracts and cancellation recovery; collect the full verification results.',criteria:[{id:'tls-default',description:'Keep certificate verification enabled for every runtime connection.'},{id:'session-replay',description:'Restore saved messages, todo state and tool output after reconnecting.'},{id:'desktop-layout',description:'Keep navigation and output readable at narrow and wide window sizes.'}],roundsStarted:18,maxGoalRounds:40}})}]}]})
  if(name==='Todos' || name==='Queued context')patch({todos:[{id:'1',content:'Review existing runtime contracts',status:'completed'},{id:'2',content:'Verify reconnect and cancellation behavior',status:'in_progress'},{id:'3',content:'Inspect the rebuilt desktop app at narrow and wide sizes',status:'pending'}]})
  if(name==='Wide transcript')patch({blocks:blocksFromStoredMessages([
    {role:'user',content:'Review the long output without moving the conversation sideways.'},
    {role:'assistant',content:'| Path | Result |\n| --- | --- |\n| '+ 'deep_directory_'.repeat(120) +' | Complete |\n\n```text\n'+ 'long_output_'.repeat(160)+'\n```\n\nThe conversation remains readable after this output.'},
  ])})
  if(name==='Restored labels')patch({blocks:blocksFromStoredMessages([
    {role:'user',text:'Monitor reaction · 1–8',content:'Expanded monitor evidence that belongs in the provider context.'},
    {role:'assistant',content:'The new evidence confirms that reconnecting preserves the current task.\n\nNo additional action is needed.'},
    {role:'user',text:'/skill workspace-review',content:[{type:'text',text:'Expanded internal skill instructions.'}]},
    {role:'assistant',content:'I’ll review the workspace and report specific findings.'},
  ])})
  if(name==='Approval keyboard')patch({approval:{id:'approval-fixture',action:'read_file',description:'Read a selected local file'}})
  if(name==='Question keyboard')patch({question:{requestId:'question-fixture',items:[{id:'scope',question:'Choose the scope',options:['Selected files','Whole workspace'],allowFreeform:false},{id:'notes',question:'Any constraints?',options:[],allowFreeform:true,placeholder:'Your constraints'}]}})
  if(name==='Plan keyboard')patch({question:{requestId:'plan-keyboard',items:[{id:'answer',question:'Review the plan',options:['Approve plan — start acting','Keep planning'],allowFreeform:true}]},plan:{items:[],turn:1,markdown:'# Proposed plan\nReview the selected files before editing.'}})
  if(name==='Streaming')patch({turnActive:true,turnSeconds:42,blocks:[blocks[0]!,{...blocks[1]!,streaming:true} as Snapshot['blocks'][number]]})
  if(name==='Execution')patch({turnActive:true,blocks:[...blocks,{kind:'agents',id:8,members:(base.fleet ?? []).map(row=>({key:row.id,title:row.title,status:row.status}))},{kind:'thinking',id:9,text:'Checking the cancellation regression.',streaming:true},{kind:'tools',id:10,running:true,items:[{id:'running-test',verb:'exec_command',name:'exec_command',arg:'bun',dur:'12s',state:'working',input:JSON.stringify({cmd:'bun',args:['test','src/runtime/transport/connections/recovery/session-reconnect-controller.test.ts']}),output:''}]}]})
  if(name==='Reconnecting')patch({turnActive:true,networkRetrying:true,turnSeconds:58})
  if(name==='Error')patch({failed:{error:'Authentication failed: configure a valid provider credential in Settings.',turn:5,lastUser:long}})
  if(name==='Compaction error') {
    const error='Automatic context compaction failed: Client openai-codex: Responses API stream request failed (400): {"detail":"Bad Request"}. Original conversation retained.'
    patch({blocks:[{kind:'user',id:1,text:'Continue the goal.'},{kind:'notice',id:2,error:true,text:error},{kind:'agent',id:3,text:`[Error: ${error}]`,streaming:false}],failed:{error,turn:109,lastUser:'Continue the goal.'}})
  }
  if(name==='Onboarding')window.dispatchEvent(new Event('xerxes:setup'))
}
host.fixture?.onScenario(scenario)
window.addEventListener('keydown', event => { if(event.altKey && (event.metaKey || event.ctrlKey)) { const names=['Welcome','Populated','Streaming','Reconnecting','Error','Onboarding','Restored labels','Export error','Artifacts','Runtime update']; const name=names[Number(event.code.replace('Digit',''))-1]; if(name){event.preventDefault();scenario(name)} } })

const requestedScenario = new URLSearchParams(location.search).get('scenario'); if(requestedScenario) scenario(requestedScenario)
