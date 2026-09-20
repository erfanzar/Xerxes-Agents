// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
const {app,BrowserWindow,webContents}=require('electron');
const fs=require('fs'),path=require('path'),assert=require('assert/strict');
const out=process.env.XERXES_QA_ROOT;
app.setPath('userData',path.join(out,'profile-'+Date.now()));
const delay=ms=>new Promise(r=>setTimeout(r,ms));
let wc,win;
const run=async code=>{try{return await wc.executeJavaScript(code,true)}catch(e){console.error("Renderer command failed:",code);throw e}};
const sel=q=>`document.querySelector(${JSON.stringify(q)})`;const all=q=>`document.querySelectorAll(${JSON.stringify(q)})`;
const until=async(code)=>{for(let i=0;i<100;i++){if(await run(code))return;await delay(100)}throw Error('Timed out: '+code)};
const click=async(selector)=>{await until(`Boolean(document.querySelector(${JSON.stringify(selector)}))`);await run(`document.querySelector(${JSON.stringify(selector)}).click()`);await delay(200)};
const capture=async(name)=>{fs.writeFileSync(path.join(out,name+'.png'),(await wc.capturePage()).toPNG());fs.writeFileSync(path.join(out,name+'.txt'),await run('document.body.innerText'))};
(async()=>{
await import(process.env.XERXES_QA_ENTRY);
await app.whenReady();
for(let i=0;i<100;i++){win=BrowserWindow.getAllWindows()[0];wc=webContents.getAllWebContents().find(c=>c.getURL().includes('/renderer/'));if(wc&&!wc.isLoading())break;await delay(150)}
await until('Boolean(document.querySelector(".app"))');
await run(`localStorage.setItem('xerxes.desktop.setup.v1','done');document.documentElement.dataset.userTheme='dark';document.documentElement.dataset.theme='dark'`);
wc.reload();await delay(1300);
await click('.desktop-rail header nav button:first-child');
await until('document.querySelectorAll("[role=treeitem]").length>=7');
const find=(name)=>`[role="treeitem"][title="Expand folder: ./${name}/"]`;
await click(find('f4'));await until(`${all('[aria-level="2"]')}.length===2`);
await click(find('f7'));await click(find('f7/p1'));
assert.equal(await run(`${all('[aria-level="3"]')}.length`),2);
assert.equal(await run(`${all('[aria-level="1"]')}.length>=7`),true);
await capture('recursive-tree');
await click('[title="Select file: ./f7/p1/c1"]');await until('document.body.innerText.includes("Nested file f7/p1/c1")');
await click('[title="Collapse folder: ./f7/"]');await click(find('f7'));
assert.equal(await run(`${sel('[title="Collapse folder: ./f7/p1/"]')}.getAttribute('aria-expanded')`),'true');
await click(find('f2'));await until(`${all('[title^="Select file: ./f2/"]')}.length===50`);
await run(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='Load more entries').click()`);
await until(`${all('[title^="Select file: ./f2/"]')}.length===65`);
await click('.desktop-rail header nav button:nth-child(2)');await until(`Boolean(${sel('[aria-label="Changed files"]')})`);
await until(`Array.from(document.querySelectorAll('button')).some(b=>b.textContent==='Load more new files')`);
await run(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='Load more new files').click()`);
await until(`Boolean(${sel('[aria-label="Changed files"] button[title="new file ü.ts"]')})`);
await run(`Array.from(document.querySelectorAll('[aria-label="Changed files"] button')).find(b=>b.textContent.includes('new file ü.ts')).click()`);
await until(`${sel('[aria-label="Diff contents"]')}?.textContent.includes('NEW_UNTRACKED_VISIBLE')`);
await capture('new-file-diff');
assert.equal(await run('getComputedStyle(document.querySelector(".top")).paddingLeft'),'88px');
win.setFullScreen(true);await until('document.querySelector(".app").classList.contains("app--no-traffic-lights")');
assert.equal(await run('getComputedStyle(document.querySelector(".top")).paddingLeft'),'12px');
await delay(700);await capture('fullscreen');
win.setFullScreen(false);await until('!document.querySelector(".app").classList.contains("app--no-traffic-lights")');
assert.equal(await run('getComputedStyle(document.querySelector(".top")).paddingLeft'),'88px');
await delay(700);await capture('windowed');
fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({passed:true,checks:['recursive siblings','nested children','collapse preserves descendants','65-entry paging','untracked diff after 5000 tracked rows','fullscreen inset 12px','normal inset restored 88px']},null,2));
console.log('PASS '+out);app.quit();
})().catch(async e=>{console.error(e);if(wc){console.error((await run('document.body.innerText')).slice(0,12000));await capture('failure')}fs.writeFileSync(path.join(out,'failure.txt'),String(e));app.exit(1)})
