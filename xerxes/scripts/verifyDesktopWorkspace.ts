// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
// Opt-in macOS native acceptance. Run bun run --cwd xerxes build:desktop first.
import {mkdtemp, mkdir, writeFile, readdir, readFile} from 'node:fs/promises'
import {tmpdir} from 'node:os'
import {join} from 'node:path'
// Short owned home keeps the macOS Unix socket below its path-length limit.
const out=await mkdtemp('/tmp/xerxes-ui-native-')
const workspace=join(out,'workspace'),home=join(out,'home')
await mkdir(workspace);await mkdir(home)
for(const folder of ['f1','f2','f3','f4','f5','f6','f7/p1'])await mkdir(join(workspace,folder),{recursive:true})
for(const file of ['f4/p1','f4/p2','f7/p1/c1','f7/p1/c2'])await writeFile(join(workspace,file),'Nested file '+file+'\n')
for(let i=0;i<65;i++)await writeFile(join(workspace,'f2',`file-${String(i).padStart(2,'0')}.txt`),'Paged file\n')
await writeFile(join(workspace,'tracked.txt'),'original\n')
async function git(args:string[]){const p=Bun.spawn(['git',...args],{cwd:workspace,stdout:'ignore',stderr:'pipe'});if(await p.exited)throw new Error(await new Response(p.stderr).text())}
await git(['init','-q']);await git(['add','tracked.txt']);await git(['-c','user.name=Verification','-c','user.email=qa@localhost','commit','-qm','Initial fixture'])
await writeFile(join(workspace,'tracked.txt'),Array.from({length:5000},(_,i)=>'Tracked changed row '+i).join('\n'))
await writeFile(join(workspace,'new file ü.ts'),'export const NEW_UNTRACKED_VISIBLE = true\n')
await writeFile(join(home,'desktop.json'),JSON.stringify({workspace}))

console.log(out)

const repo=new URL('../../', import.meta.url).pathname
const child=Bun.spawn([repo+'node_modules/electron/dist/Electron.app/Contents/MacOS/Electron',repo+'xerxes/test/fixtures/desktop/workspaceVerification.cjs'],{env:{...process.env,XERXES_HOME:home,XERXES_BUN:process.execPath,XERXES_BUN_DAEMON:repo+'xerxes/dist/cli.js',XERXES_QA_ROOT:out,XERXES_QA_ENTRY:repo+'xerxes/dist/desktop/main.js'},stdout:'inherit',stderr:'inherit'})
try { process.exitCode=await child.exited } finally {
  // Only this verification's daemon; user daemons and workspaces are untouched.
  for(const name of await readdir(join(home,'daemon')).catch(()=>[])) if(name.endsWith('.pid')) {
    const pid=Number(await readFile(join(home,'daemon',name),'utf8'));if(Number.isInteger(pid)&&pid>1)try{process.kill(pid,'SIGTERM')}catch{}
  }
}
console.log('Native verification evidence:',out)
