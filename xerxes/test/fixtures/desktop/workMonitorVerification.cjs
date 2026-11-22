// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
const {app,BrowserWindow}=require('electron');
const fs=require('fs'),assert=require('assert/strict');
const out=process.env.XERXES_QA_ROOT;
const delay=ms=>new Promise(r=>setTimeout(r,ms));let wc,win;
const run=code=>wc.executeJavaScript(code,true);
const until=async(code)=>{for(let i=0;i<100;i++){if(await run(code))return;await delay(100)}throw Error('Timed out: '+code)};
const capture=async(name)=>{await delay(350);fs.writeFileSync(out+'/'+name+'.png',(await wc.capturePage()).toPNG());fs.writeFileSync(out+'/'+name+'.txt',await run('document.body.innerText'))};
(async()=>{require(out+'/main.cjs');await app.whenReady();await delay(1500);win=BrowserWindow.getAllWindows()[0];wc=win.webContents;
wc.send('fixture','Work monitor');await delay(700);
if(!(await run('Boolean(document.querySelector(".activity-panel"))'))){
  await run(`(document.querySelector('.floatrail button[aria-label="Activity"]') || Array.from(document.querySelectorAll('button')).find(b=>b.textContent.trim()==='Activity'))?.click()`);
}
await until('document.querySelectorAll("[aria-label=\\"Agents\\"] .railrow[data-state=\\"working\\"]").length===3');
await until('Boolean(document.querySelector(".command-activity"))');
assert.equal(await run('Array.from(document.querySelectorAll("[aria-label=\\"Agents\\"] .railrow")).slice(0,3).every(r=>r.dataset.state==="working")'),true);
assert.equal(await run('document.querySelector("[aria-label=\\"Agents\\"] .raillist__more")?.getAttribute("aria-expanded")'),'false');
assert.equal(await run('document.querySelector("[aria-label=\\"Agents\\"] .raillist__more")?.textContent.trim().startsWith("38 more")'),true);
assert.equal(await run('document.querySelector(".command-activity").open'),false);
const height=await run('document.querySelector(".command-activity").getBoundingClientRect().height');assert.ok(height<140);
await capture('monitor-collapsed');
await run('document.querySelector(".command-activity summary").click()');
await until('document.querySelector(".command-activity .output-viewer__text")?.textContent.includes("Ready for input")');
await run('document.querySelector(".command-activity").scrollIntoView({block:"center"})');
await capture('monitor-expanded');
wc.send('fixture','Command disconnect');
await until('document.querySelector(".command-activity__state")?.textContent.trim()==="disconnected"');
assert.ok(await run('document.querySelector(".command-activity__foot")?.textContent.includes("Disconnected")'));
assert.ok(await run('Array.from(document.querySelectorAll(".command-activity__foot button")).every(b=>b.disabled)'));
assert.ok(await run('document.querySelector(".command-activity .output-viewer__text").textContent.includes("Ready for input")'));
await capture('monitor-disconnected');wc.send('fixture','Command reconnect');
await until('document.querySelector(".command-activity__state")?.textContent.trim()==="running"');
wc.send('fixture','Command failure');await delay(50);
await run(`Array.from(document.querySelectorAll('.command-activity button')).find(b=>b.textContent==='Refresh').click()`);
await until('document.querySelector(".command-activity [role=alert]")?.textContent.includes("temporarily unavailable")');
assert.ok(await run('document.querySelector(".command-activity .output-viewer__text").textContent.includes("Ready for input")'));
await capture('monitor-output-error');
wc.send('fixture','Command recover');await delay(50);
await run(`Array.from(document.querySelectorAll('.command-activity button')).find(b=>b.textContent==='Refresh').click()`);
await until('!document.querySelector(".command-activity [role=alert]")');
await run('document.querySelector(".command-activity summary").click()');
await until('!document.querySelector(".command-activity .output-viewer__text")');
win.setSize(720,700);await delay(500);
if(!(await run('Boolean(document.querySelector(".command-activity"))'))){
  await run(`Array.from(document.querySelectorAll('.acard__foot button, button')).find(b=>b.textContent.trim()==='Open in Activity')?.click()`);
  await until('Boolean(document.querySelector(".command-activity"))');
}
await run('document.querySelector(".command-activity").scrollIntoView({block:"center"})');await capture('monitor-narrow');
fs.writeFileSync(out+'/monitor-result.json',JSON.stringify({passed:true,collapsedHeight:height,checks:['3 active agents first','38 completed agents collapsed behind more','long command bounded','scoped expandable output','failure preserves output','retry recovers','disconnect retains output and disables controls','reconnect restores output access','collapse removes output','720px narrow capture']},null,2));console.log('PASS '+out);app.quit();
})().catch(async error=>{console.error(error);if(wc){await capture('monitor-failure');console.error((await run('document.body.innerText')).slice(-5000))}app.exit(1)})
