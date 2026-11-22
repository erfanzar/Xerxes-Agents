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
const openView=async(name)=>{
  if(await run(`Boolean(document.querySelector(${JSON.stringify('.floatrail button[aria-label="'+name+'"]')}))`)){
    await click(`.floatrail button[aria-label="${name}"]`);return;
  }
  await click('.rail-add__trigger');
  await until(`Array.from(document.querySelectorAll('.rail-add__item')).some(b=>b.textContent.trim()===${JSON.stringify(name)})`);
  await run(`Array.from(document.querySelectorAll('.rail-add__item')).find(b=>b.textContent.trim()===${JSON.stringify(name)}).click()`);
  await delay(200);
};
const capture=async(name)=>{fs.writeFileSync(path.join(out,name+'.png'),(await wc.capturePage()).toPNG());fs.writeFileSync(path.join(out,name+'.txt'),await run('document.body.innerText'))};
(async()=>{
await import(process.env.XERXES_QA_ENTRY);
await app.whenReady();
for(let i=0;i<100;i++){win=BrowserWindow.getAllWindows()[0];wc=webContents.getAllWebContents().find(c=>c.getURL().includes('/renderer/'));if(wc&&!wc.isLoading())break;await delay(150)}
await until('Boolean(document.querySelector(".app"))');
await run(`localStorage.setItem('xerxes.desktop.setup.v1','done');document.documentElement.dataset.userTheme='dark';document.documentElement.dataset.theme='dark'`);
wc.reload();await delay(1300);
await until('Boolean(document.querySelector(".app"))');
await openView('Files');
await until('document.querySelectorAll("[role=treeitem]").length>=7');
const find=(name)=>`[role="treeitem"][title="Expand folder: ./${name}/"]`;
await click(find('f4'));await until(`${all('[aria-level="2"]')}.length===2`);
await click(find('f7'));await click(find('f7/p1'));
await until(`${all('[aria-level="3"]')}.length===2`);
assert.equal(await run(`${all('[aria-level="3"]')}.length`),2);
assert.equal(await run(`${all('[aria-level="1"]')}.length>=7`),true);
await capture('recursive-tree');
await click('[title="Collapse folder: ./f7/"]');await click(find('f7'));
assert.equal(await run(`${sel('[title="Collapse folder: ./f7/p1/"]')}.getAttribute('aria-expanded')`),'true');
await click(find('f2'));await until(`${all('[title^="Select file: ./f2/"]')}.length===50`);
await run(`Array.from(document.querySelectorAll('button')).find(b=>b.textContent==='Load more entries').click()`);
await until(`${all('[title^="Select file: ./f2/"]')}.length===65`);
await click('[title="Select file: ./f7/p1/c1"]');await until('document.body.innerText.includes("Nested file f7/p1/c1")');
await openView('Changes');await until(`Boolean(${sel('.scm')})`);
await until(`Boolean(${sel('.scm-file__open[title="new file ü.ts"]')})`);
await click('.scm-file__open[title="new file ü.ts"]');
await until(`${sel('[aria-label="Diff contents"]')}?.textContent.includes('NEW_UNTRACKED_VISIBLE')`);
await capture('new-file-diff');
const windowedInset=await run('getComputedStyle(document.querySelector(".side__head")).paddingLeft');
assert.ok(parseFloat(windowedInset)>=78,'expected traffic-light inset >=78px, got '+windowedInset);
win.setFullScreen(true);await until('document.querySelector(".app").classList.contains("app--no-traffic-lights")');
const fullscreenInset=await run('getComputedStyle(document.querySelector(".side__head")).paddingLeft');
assert.ok(parseFloat(fullscreenInset)<=12,'expected fullscreen inset <=12px, got '+fullscreenInset);
await delay(700);await capture('fullscreen');
win.setFullScreen(false);await until('!document.querySelector(".app").classList.contains("app--no-traffic-lights")');
assert.equal(await run('getComputedStyle(document.querySelector(".side__head")).paddingLeft'),windowedInset);
await delay(700);await capture('windowed');
fs.writeFileSync(path.join(out,'result.json'),JSON.stringify({passed:true,windowedInset,fullscreenInset,checks:['recursive siblings','nested children','collapse preserves descendants','65-entry paging','untracked diff after 5000 tracked rows','fullscreen inset collapsed','normal inset restored']},null,2));
console.log('PASS '+out);app.quit();
})().catch(async e=>{console.error(e);if(wc){console.error((await run('document.body.innerText')).slice(0,12000));await capture('failure')}fs.writeFileSync(path.join(out,'failure.txt'),String(e));app.exit(1)})
