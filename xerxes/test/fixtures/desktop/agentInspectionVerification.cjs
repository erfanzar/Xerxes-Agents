// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
const {app,BrowserWindow}=require('electron');
const fs=require('fs'),assert=require('assert/strict');
const out=process.env.XERXES_QA_ROOT;
const delay=ms=>new Promise(r=>setTimeout(r,ms));let wc,win;
const run=async code=>{try{return await wc.executeJavaScript(code,true)}catch(error){console.error(code);throw error}};
const until=async(code)=>{for(let i=0;i<100;i++){if(await run(code))return;await delay(100)}throw Error('Timed out: '+code)};
const capture=async(name)=>{await delay(250);fs.writeFileSync(out+'/'+name+'.png',(await wc.capturePage()).toPNG());fs.writeFileSync(out+'/'+name+'.txt',await run('document.body.innerText'))};
const openOutputFold=async()=>{
  await until(`Array.from(document.querySelectorAll('.agent-card details.agent-card__fold>summary')).some(s=>s.textContent.trim()==='Output')`);
  await run(`(() => { const s = Array.from(document.querySelectorAll('.agent-card details.agent-card__fold>summary')).find(el => el.textContent.trim() === 'Output'); if (s && !s.parentElement.open) s.click(); })()`);
};
(async()=>{require(out+'/main.cjs');await app.whenReady();await delay(1200);win=BrowserWindow.getAllWindows()[0];wc=win.webContents;
wc.send('fixture','Work monitor');await delay(400);
await until('document.querySelectorAll("[aria-label=\\"Agents\\"] .railrow[data-state=\\"working\\"]").length===3');
await until('document.querySelectorAll("button.acard__row").length===3');
await run('document.querySelector("button.acard__row").click()');
await until('document.querySelector(".agent-card")?.textContent.includes("reviewer")');
await openOutputFold();
await until('document.querySelector(".agent-card .output-viewer__text")?.textContent.includes("Evidence 80")');
assert.ok(await run('document.querySelector(".agent-card").textContent.includes("Base agent")'));
assert.ok(await run('document.querySelector(".agent-card").textContent.includes("work-profile")'));
assert.ok(await run('document.querySelector(".agent-card").textContent.includes("Assigned task")'));
assert.equal(await run('document.querySelectorAll(".agent-inspector .command-activity").length'),0);
await capture('agent-inspector-wide');
// Switching agents while a read is in flight cannot display the previous identity.
await run('(() => { const rows = document.querySelectorAll("[aria-label=\\"Agents\\"] .railrow[data-state=\\"working\\"]"); rows[1].click(); rows[2].click(); })()');
await until('document.querySelector(".agent-card .railcard__title")?.textContent==="Running review 3"');
await until('document.querySelector(".agent-card")?.textContent.includes("test-engineer")');
await openOutputFold();
await until('document.querySelector(".agent-card .output-viewer__text")?.textContent.includes("Evidence 80")');
wc.send('fixture','Agent failure');
await until('document.querySelector(".agent-inspector [role=alert]")?.textContent.includes("detail read failed")');
assert.ok(await run('document.querySelector(".agent-card .output-viewer__text").textContent.includes("Evidence 80")'));
wc.send('fixture','Agent recover');await delay(30);
await run(`Array.from(document.querySelectorAll('.agent-inspector button')).find(b=>b.textContent==='Retry agent details').click()`);
await until('!document.querySelector(".agent-inspector [role=alert]")');
wc.send('fixture','Command disconnect');
await until('document.querySelector(".agent-inspector [role=status]")?.textContent.includes("Disconnected")');
assert.equal(await run('document.querySelector(".agent-inspector fieldset").disabled'),true);
wc.send('fixture','Command reconnect');
await until('!document.querySelector(".agent-inspector [role=status]")');
win.setSize(720,700);await delay(300);await capture('agent-inspector-narrow');
await run('document.querySelector(".agent-crumb__parent").click()');
await until('!document.querySelector(".agent-inspector")');
if(!(await run('Boolean(document.querySelector(".command-activity"))'))){
  await run(`Array.from(document.querySelectorAll('.acard__foot button, button')).find(b=>b.textContent.trim()==='Open in Activity')?.click()`);
}
await until('Boolean(document.querySelector(".command-activity"))');
await run('document.querySelector(".command-activity summary").click()');
await until('document.querySelector(".command-activity .output-viewer__text")?.textContent.includes("Ready for input")');
await run('document.querySelector(".command-activity").scrollIntoView({block:"center"})');
assert.equal(await run('Boolean(document.querySelector(".command-activity__command"))'),true);
assert.equal(await run('getComputedStyle(document.querySelector(".command-activity .output-viewer__text")).whiteSpace'),'pre-wrap');
await capture('command-reader-narrow');
await run('document.querySelector(".command-activity button[aria-label=\\"Expand output\\"]").click()');
await until('document.querySelector(".output-dialog")?.open');
assert.ok(await run('document.querySelector(".output-dialog").getBoundingClientRect().right<=innerWidth'));
await run('document.querySelector(".output-dialog [aria-pressed]").click()');
assert.equal(await run('getComputedStyle(document.querySelector(".output-dialog pre")).whiteSpace'),'pre');
await capture('command-expanded-narrow');
wc.sendInputEvent({type:'keyDown',keyCode:'Escape'});wc.sendInputEvent({type:'keyUp',keyCode:'Escape'});
await until('!document.querySelector(".output-dialog")');
assert.equal(await run('document.activeElement?.getAttribute("aria-label")'),'Expand output');
win.setSize(1440,900);await delay(250);
await run('document.querySelector(".command-activity button[aria-label=\\"Expand output\\"]").click()');await until('document.querySelector(".output-dialog")?.open');await capture('command-expanded-wide');
await run('document.querySelector(".output-dialog button[aria-label=\\"Close expanded output\\"]")?.click() || document.querySelector(".output-dialog button[aria-label]")?.click()');
wc.send('fixture','Command failure');await delay(30);
await run(`Array.from(document.querySelectorAll('.command-activity button')).find(b=>b.textContent==='Refresh').click()`);
await until('document.querySelector(".command-activity [role=alert]")?.textContent.includes("temporarily unavailable")');
assert.ok(await run('document.querySelector(".command-activity .output-viewer__text").textContent.includes("Ready for input")'));
wc.send('fixture','Command recover');await delay(30);
await run(`Array.from(document.querySelectorAll('.command-activity button')).find(b=>b.textContent==='Refresh').click()`);
await until('!document.querySelector(".command-activity [role=alert]")');
await run('document.querySelector("[aria-label=\\"Agents\\"] .railrow").click()');
await until('Boolean(document.querySelector(".agent-inspector"))');
fs.writeFileSync(out+'/agent-inspection-result.json',JSON.stringify({passed:true,checks:['chat agent rows clickable','roster opens same inspector','base agent and provider metadata','assigned task and retained output','rapid agent switch','error retains evidence','retry recovers','disconnect disables controls','reconnect recovers','narrow inspector','command collapsed independently','wrapped output','expanded reading surface','Escape restores focus','command failure retains output']},null,2));console.log('PASS '+out);app.quit();
})().catch(async error=>{console.error(error);if(wc){await capture('inspection-failure');console.error((await run('document.body.innerText')).slice(-5000))}app.exit(1)})
