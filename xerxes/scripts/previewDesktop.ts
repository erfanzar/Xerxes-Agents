// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Isolated native renderer verification. No daemon, credentials, or real sessions. */
import {resolve,join} from 'node:path'
import {tmpdir} from 'node:os'
import {mkdtemp} from 'node:fs/promises'
import {themeStylesheet} from '../src/desktop/tokens.js'
import {mkdir,copyFile,writeFile} from 'node:fs/promises'
const root=resolve(import.meta.dir, '../..'),out=await mkdtemp(join(tmpdir(),'xerxes-desktop-qa-'))
await mkdir(out,{recursive:true})
const result=await Bun.build({entrypoints:[root+'/xerxes/test/fixtures/desktop/preview.tsx'],outdir:out,target:'browser',format:'esm'})
if(!result.success)throw new Error(result.logs.join('\n'))
for(const file of ['app.css','atelier.css'])await copyFile(root+'/xerxes/src/desktop/renderer/'+file,out+'/'+file)
await writeFile(out+'/theme.css',themeStylesheet())
await writeFile(out+'/index.html',`<!doctype html><html data-theme="dark"><head><meta charset="utf-8"><meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:;"><title>Xerxes Layout Verification</title><link rel="stylesheet" href="theme.css"><link rel="stylesheet" href="app.css"><link rel="stylesheet" href="atelier.css"></head><body><div id="root"></div><script type="module" src="preview.js"></script></body></html>`)
await writeFile(out+'/preload.cjs',`const {contextBridge,ipcRenderer}=require('electron');contextBridge.exposeInMainWorld('fixture',{onScenario:fn=>ipcRenderer.on('fixture',(_,name)=>fn(name))});`)
await writeFile(out+'/main.cjs',`const{app,BrowserWindow,Menu}=require('electron');app.setName('Xerxes Layout Verification');app.setPath('userData',__dirname+'/profile');app.whenReady().then(()=>{const win=new BrowserWindow({width:1440,height:900,minWidth:620,minHeight:480,title:'Xerxes Layout Verification',titleBarStyle:'hiddenInset',backgroundColor:'#202124',webPreferences:{preload:__dirname+'/preload.cjs',sandbox:true,contextIsolation:true,nodeIntegration:false}});Menu.setApplicationMenu(Menu.buildFromTemplate([{label:'Fixture',submenu:[...['Welcome','Populated','Streaming','Reconnecting','Error','Onboarding'].map(label=>({label,click:()=>win.webContents.send('fixture',label)})),{type:'separator'},...[[ 'Narrow',720,700],['Normal',1200,820],['Wide',1600,1000]].map(([label,w,h])=>({label,click:()=>win.setSize(w,h)})),{role:'reload'},{role:'quit'}]},{role:'editMenu'},{role:'viewMenu'}]));win.loadFile(__dirname+'/index.html');});`)
const verifyAgents=Bun.argv.includes("--verify-agent-inspection")
const verifyMonitor=Bun.argv.includes('--verify-work-monitor')
const child=Bun.spawn([root+'/node_modules/electron/dist/Electron.app/Contents/MacOS/Electron',verifyAgents?root+"/xerxes/test/fixtures/desktop/agentInspectionVerification.cjs":verifyMonitor?root+'/xerxes/test/fixtures/desktop/workMonitorVerification.cjs':out+'/main.cjs'],{env:{...Bun.env,XERXES_QA_ROOT:out},stdout:Bun.file(out+'/stdout.log'),stderr:Bun.file(out+'/stderr.log')})
console.log('Fixture PID',child.pid,'Artifacts:',out)
process.exitCode=await child.exited
