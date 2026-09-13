// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Isolated desktop stress fixture. Never imported by the production entry. */
import { createRoot } from 'react-dom/client'
import { useSyncExternalStore } from 'react'
import { restoreAppearance } from '../../../src/desktop/renderer/appearance.js'
import { Shell } from '../../../src/desktop/renderer/App.js'
import { blocksFromStoredMessages } from '../../../src/desktop/renderer/blocks.js'
import { store, type Snapshot } from '../../../src/desktop/renderer/store.js'
import type { TerminalDetail, XerxesBridge } from '../../../src/desktop/renderer/types.js'

const patch = (value: Partial<Snapshot>) => { const port = store as unknown as { patch(value: Partial<Snapshot>): void; builder: { reset(blocks: Snapshot['blocks']): void }; turnCount: number }; if(value.blocks) port.builder.reset(value.blocks); if(value.turnCount !== undefined) port.turnCount=value.turnCount; port.patch(value) }
let fixtureFolderAttempts=0
const cwd='/fixture/xerxes-workspace'
const session=(i:number) => ({ id:`session-${i}`,key:`session-${i}`,title:['Review transport cancellation behavior','Improve workspace file navigation','Investigate provider retry handling'][i%3]+` ${i+1}`,status:'idle',age:`${i+1}m`,current:i===0,kind:'main',turns:i%5,messages:2,cwd,untitled:false })
const long='Preserve every runtime integration and saved session. Review cancellation, reconnect behavior, command discovery and deeply nested file paths. Keep navigation available while work is running. '
const skills=Array.from({length:167},(_,i)=>({name:`workspace-review-${String(i+1).padStart(3,'0')}`,description:`Review workspace behavior ${i+1}, including cancellation, keyboard access and saved sessions.`,kind:'skill',source:'fixture',enabled:true}))
let schedules=Array.from({length:12},(_,i)=>({id:`schedule-${i}`,prompt:`Review workspace ${i+1}: ${long}`,schedule:'0 8 * * 1-5',timezone:'UTC',paused:i%3===0,execution_state:'idle',next_run_at:'2026-09-14T08:00:00Z',revision:1}))
const scheduledRuns=Array.from({length:8},(_,i)=>({id:`run-${i}`,title:'Review workspace',state:i%3===0?'failed':'succeeded',startedAt:Date.UTC(2026,8,12-i,8),error:i%3===0?'Provider configuration is missing a model.':null,output:Array.from({length:120},(_,line)=>`  Step ${line+1}: checked workspace state`).join('\n'),outputTruncated:false,exitCode:i%3===0?1:0}))
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
const bridge: XerxesBridge={onEvent:()=>()=>{},async call<T>(method:string,params:Record<string,unknown>={}) {
  let result:unknown={ok:false,error:`Fixture does not implement ${method}`}
  if(method==='complete') {const query=String(params.text||'');result={completions:query.startsWith('./') ? [{label:'src/',value:'./src/',meta:'dir'},{label:'session-reconnect-controller.ts',value:'./src/runtime/transport/connections/recovery/session-reconnect-controller.ts',meta:'file'},{label:'unavailable.txt',value:'./unavailable.txt',meta:'file'}] : skills.filter(s=>s.name.includes(query.replace('/skill ','').replace('/',''))).map(s=>({label:s.name,value:`/skill ${s.name}`,meta:s.description}))}}
  else if(method==='reasoning_levels')result={levels:['low','medium','high','xhigh'].map(effort=>({effort,description:effort+' reasoning'})),current:store.getSnapshot().reasoningEffort}
  else if(method==='provider_list')result={ok:true,profiles:[{name:'Workspace review provider with a long descriptive name',provider:'openai',model:'fixture/model-long-context',active:true},{name:'Secondary development provider',provider:'openai',model:'fixture/review-model',active:false}]}
  else if(method==='provider_types')result={ok:true,types:[{name:'openai',base_url:'https://api.openai.com/v1',api_key_env:'OPENAI_API_KEY'}]}
  else if(method==='capabilities.list') result={skills,tools:[],plugins:[]}
  else if(method==='capabilities.inspect') result={name:params.name,description:long,instructions:'# Workspace review\n\n'+long}
  else if(method==='workspace.filePreview'){await new Promise(resolve=>setTimeout(resolve,String(params.path).includes('unavailable')?100:700));if(String(params.path).includes('unavailable'))result={ok:false,error:'Fixture file cannot be read. Choose another file.'};else result={ok:true,path:params.path,content:Array.from({length:240},(_,i)=>`export const recoveryStep${i + 1} = async (session: Session) => resumeSession(session.id);`).join('\n'),truncated:false}}
  else if(method==='workspace.diff')result={kind:'diff',diff:{lines,untracked:[]}}
  else if(method==='background.activity')result={rows:[]}
  else if(method==='background.status')result={shells:0,watchers:0}
  else if(method==='schedule.list')result={ok:true,jobs:schedules}
  else if(method==='schedule.preview')result=String(params.schedule).split(' ').length===5?{ok:true,next_run_at:'2026-09-14T08:00:00Z'}:{ok:false,error:'Invalid cron expression'}
  else if(method==='schedule.create'){schedules=[...schedules,{id:`schedule-${schedules.length}`,prompt:String(params.prompt),schedule:String(params.schedule),timezone:String(params.timezone),paused:false,execution_state:'idle',next_run_at:'2026-09-14T08:00:00Z',revision:1}];result={ok:true}}
  else if(method==='schedule.update'){schedules=schedules.map(job=>job.id===params.schedule_id?{...job,prompt:String(params.prompt),schedule:String(params.schedule),timezone:String(params.timezone),revision:job.revision+1}:job);result={ok:true}}
  else if(method==='schedule.pause'||method==='schedule.resume'){schedules=schedules.map(job=>job.id===params.schedule_id?{...job,paused:method==='schedule.pause'}:job);result={ok:true}}
  else if(method==='schedule.remove'){schedules=schedules.filter(job=>job.id!==params.schedule_id);result={ok:true}}
  else if(method==='run.list')result={ok:true,runs:scheduledRuns.filter(run=>typeof params.before_started_at!=='number'||run.startedAt<params.before_started_at).slice(0,4).map(({output,...run})=>run),has_more:params.before_started_at===undefined}
  else if(method==='run.inspect')result={ok:true,run:scheduledRuns.find(run=>run.id===params.run_id)}
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
  else if(method==='initialize'){const id=String(params.resume_session_id||'session-0');result={session_id:id,cwd,model:base.model,session:{id,key:id,title:`Review workspace ${id}`,cwd}}}
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
  else if(method==='terminal.inspect')result={ok:true,terminal:fixtureTerminal}
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
patch({...base,blocks})
function Preview(){return <Shell snap={useSyncExternalStore(store.subscribe,store.getSnapshot)} />}
createRoot(document.getElementById('root')!).render(<Preview/> )
const host=window as unknown as {fixture?:{onScenario(handler:(name:string)=>void):void}}
const scenario = (name:string) => {
  fixtureExportDenied=name==='Export error'
  patch({...base,blocks,turnActive:false,failed:null,networkRetrying:false,settingsOpen:false,pickerOpen:false,modelMenuOpen:false,reasoningPickerOpen:false})
  if(name==='Artifacts')patch({changes:[{path:'src/runtime/transport/connections/recovery/session-reconnect-controller.ts',adds:24,dels:8,isNew:false,hunks:[],turn:1},{path:'src/desktop/layout.ts',adds:1,dels:0,isNew:true,hunks:[],turn:1}]})
  if(name==='Runtime update')patch({daemonWarning:'The workspace runtime predates this app build. Existing work is still running.',turnActive:true})
  if(name==='Runtime offline')patch({connection:'offline'})
  if(name==='Welcome')patch({blocks:[],turnCount:0})
  if(name==='Restored labels')patch({blocks:blocksFromStoredMessages([
    {role:'user',text:'Monitor reaction · 1–8',content:'Expanded monitor evidence that belongs in the provider context.'},
    {role:'assistant',content:'The new evidence confirms that reconnecting preserves the current task.\n\nNo additional action is needed.'},
    {role:'user',text:'/skill workspace-review',content:[{type:'text',text:'Expanded internal skill instructions.'}]},
    {role:'assistant',content:'I’ll review the workspace and report specific findings.'},
  ])})
  if(name==='Streaming')patch({turnActive:true,turnSeconds:42,blocks:[blocks[0]!,{...blocks[1]!,streaming:true} as Snapshot['blocks'][number]]})
  if(name==='Reconnecting')patch({turnActive:true,networkRetrying:true,turnSeconds:58})
  if(name==='Error')patch({failed:{error:'Authentication failed: configure a valid provider credential in Settings.',turn:5,lastUser:long}})
  if(name==='Onboarding')window.dispatchEvent(new Event('xerxes:setup'))
}
host.fixture?.onScenario(scenario)
window.addEventListener('keydown', event => { if(event.altKey && (event.metaKey || event.ctrlKey)) { const names=['Welcome','Populated','Streaming','Reconnecting','Error','Onboarding','Restored labels','Export error','Artifacts','Runtime update']; const name=names[Number(event.code.replace('Digit',''))-1]; if(name){event.preventDefault();scenario(name)} } })

const requestedScenario = new URLSearchParams(location.search).get('scenario'); if(requestedScenario) scenario(requestedScenario)
