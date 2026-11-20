// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { strict as assert } from 'node:assert'
import { join } from 'node:path'
import { pathToFileURL } from 'node:url'
import { setTimeout as delay } from 'node:timers/promises'
import { app, BrowserWindow } from 'electron'

void verify().catch(error => { console.error(error); app.exit(1) })

async function verify() {
await import(pathToFileURL(join(process.env.XERXES_QA_ROOT, 'main.cjs')).href)
await app.whenReady()
const wc = BrowserWindow.getAllWindows()[0].webContents
const js = (code) => wc.executeJavaScript(code, true)
async function until(code, description, timeout = 2500) {
  const end = Date.now() + timeout
  while (Date.now() < end) {
    if (await js(code)) return
    await delay(50)
  }
  assert.fail(`${description}: ${await js("document.querySelector('.desktop-rail')?.innerText ?? '(no inspector)'")}`)
}
const revision = (version) => `document.querySelector('.rsec--detail h1')?.textContent === 'Artifact revision ${version}'`
try {
  await until("!!document.querySelector('.chat')", 'workspace did not render')
  wc.send('fixture', 'Artifact card')
  await until("!!document.querySelector('.artifact-card__open')", 'artifact card missing')
  await js("document.querySelector('.artifact-card__open').click()")
  await until(revision(1), 'Open did not go directly to the selected document')
  assert.equal(await js("document.querySelector('[aria-selected=true][role=tab]')?.textContent"), 'Plan: grouped usage')
  assert.equal(await js("document.querySelector('.chat')?.getBoundingClientRect().width > 0 && getComputedStyle(document.querySelector('.chat')).display !== 'none'"), true)
  console.log('PASS transcript card opens the selected document beside chat')

  wc.send('fixture', 'Refresh artifact')
  await until(revision(2), 'open body did not refresh on the tool result', 1800)
  console.log('PASS open artifact refreshes immediately on an update')

  await js("document.querySelector('[aria-label=\"Edit artifact\"]').click()")
  await until("document.querySelector('.rsec__editor')?.value.includes('revision 2')", 'editor not open')
  await js(`(() => {const el = document.querySelector('.rsec__editor'); Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set.call(el, 'Unsaved draft'); el.dispatchEvent(new Event('input', { bubbles: true }))})()`)
  wc.send('fixture', 'Refresh artifact')
  await until("document.querySelector('.rsec__meta')?.textContent.includes('v3')", 'edit view did not observe the newer revision')
  assert.equal(await js("document.querySelector('.rsec__editor').value"), 'Unsaved draft')
  await js("[...document.querySelectorAll('.rsec button')].find(button => button.textContent === 'Cancel').click()")
  await until(revision(3), 'cancel did not reveal latest revision')
  console.log('PASS live update preserves the edit draft')

  wc.send('fixture', 'Refresh artifact via poll')
  await until(revision(4), 'list poll did not refresh the document without an event', 8500)
  console.log('PASS list revision polling refreshes missed events')

  wc.send('fixture', 'Todo navigation')
  const todo = "[...document.querySelectorAll('.composer-task-summary button')].find(button => button.textContent.includes('Verify the return to chat'))"
  await until(`Boolean(${todo})`, 'to-do shortcut missing')
  await js(`${todo}.click()`)
  await until("!document.querySelector('.stream[aria-label=\"Conversation\"]')", 'to-do shortcut did not open the plan')
  const back = "[...document.querySelectorAll('.composer-task-summary button')].find(button => button.textContent.includes('Back to conversation'))"
  await until(`Boolean(${back})`, 'no return control above the composer')
  await js(`${back}.click()`)
  await until(`(() => { const feed = document.querySelector('.stream[aria-label="Conversation"]'); return !!feed && feed.textContent.includes('Latest fixture reply') && feed.scrollHeight > feed.clientHeight && feed.scrollHeight - feed.scrollTop - feed.clientHeight < 5 })()`, 'return control did not restore the bottom of the conversation')
  console.log('PASS to-do shortcut offers a return to the latest conversation above the composer')
  app.exit(0)
} catch (error) {
  console.error(error)
  app.exit(1)
}
}
