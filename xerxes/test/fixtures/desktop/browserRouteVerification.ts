// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

// Bundle this entry to a temporary directory with Electron external, then run
// it in Electron. The SOCKS peers send synthetic pages and cannot forward traffic.
import { strict as assert } from 'node:assert'
import { ChildProcess } from 'node:child_process'
import { mkdtemp, rm } from 'node:fs/promises'
import { createServer, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { app, BrowserWindow, session, webContents } from 'electron'

import { BrowserTab } from '../../../src/desktop/main/browserTab.js'
import { RemoteNetwork, waitForLoopback } from '../../../src/desktop/main/remoteNetwork.js'

const machine = { alias: 'fixture', target: 'fixture.invalid', workspacePath: '/fixture' }
const target = 'http://localhost:9999/'

async function peer(title: string, refused = false) {
  const sockets = new Set<Socket>()
  const server = createServer(socket => {
    sockets.add(socket)
    socket.once('close', () => sockets.delete(socket))
    socket.on('error', () => socket.destroy())
    let bytes = Buffer.alloc(0)
    let stage: 'hello' | 'connect' | 'http' = 'hello'
    socket.on('data', chunk => {
      bytes = Buffer.concat([bytes, chunk])
      if (stage === 'hello') {
        if (bytes.length < 2 || bytes.length < 2 + bytes[1]!) return
        if (bytes[0] !== 5) { socket.destroy(); return }
        bytes = bytes.subarray(2 + bytes[1]!)
        stage = 'connect'
        socket.write(Buffer.from([5, 0]))
      }
      if (stage === 'connect') {
        if (bytes.length < 5) return
        const size = bytes[3] === 3 ? 5 + bytes[4]! + 2 : bytes[3] === 1 ? 10 : 22
        if (bytes.length < size) return
        // Only accept SOCKS CONNECT to the expected remote localhost. This
        // also proves Chromium hasn't bypassed the proxy for loopback URLs.
        const allowed = bytes[0] === 5 && bytes[1] === 1 && bytes[3] === 3
          && bytes.subarray(5, size - 2).toString() === 'localhost' && bytes.readUInt16BE(size - 2) === 9999
        const reply = Buffer.from([5, allowed && !refused ? 0 : 5, 0, 1, 127, 0, 0, 1, 0, 0])
        if (!allowed || refused) { socket.end(reply); return }
        socket.write(reply)
        bytes = bytes.subarray(size)
        stage = 'http'
      }
      if (stage === 'http' && bytes.includes('\r\n\r\n')) {
        const body = `<title>${title}</title>`
        socket.end(`HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nCache-Control: no-store\r\nContent-Length: ${Buffer.byteLength(body)}\r\nConnection: close\r\n\r\n${body}`)
      }
    })
  })
  await new Promise<void>((resolve, reject) => { server.once('error', reject); server.listen(0, '127.0.0.1', resolve) })
  const address = server.address()
  if (!address || typeof address === 'string') throw Error('Missing fixture listener')
  return { port: address.port, close() { for (const socket of sockets) socket.destroy(); server.close() } }
}

async function until(tab: BrowserTab, description: string, ready: () => boolean) {
  const deadline = Date.now() + 4000
  while (!ready() && Date.now() < deadline) await delay(25)
  assert.ok(ready(), `${description}: ${JSON.stringify(tab.state())}`)
}

async function recover(window: BrowserWindow, action: 'reload' | 'open', firstLoadFails = false) {
  const first = await peer('first route', firstLoadFails)
  const next = await peer('replacement route')
  const ports = [first.port, next.port]
  const children: ChildProcess[] = []
  const network = new RemoteNetwork(machine, {
    freePort: async () => { const port = ports.shift(); assert.ok(port); return port },
    spawn: () => { const child = new ChildProcess(); children.push(child); return child },
    waitReady: (port, signal) => waitForLoopback(port, signal, 1000),
  })
  const tab = new BrowserTab(window, () => network, () => {})
  try {
    await tab.open(target)
    if (firstLoadFails) await until(tab, 'expected the initial SOCKS failure', () => !!tab.state().error)
    else await until(tab, 'initial page did not load through the first proxy', () => tab.state().title === 'first route')
    assert.equal(tab.state().url, target, 'failed first load must retain its retry address')
    first.close()
    children[0]?.emit('exit', 1)
    if (action === 'open') await tab.open(target)
    else await tab.navigate('reload')
    assert.equal(children.length, 2, `${action} must acquire the replacement SOCKS listener before loading`)
    await until(tab, `${action} did not recover through the new proxy`, () => tab.state().title === 'replacement route' && !tab.state().error)
  } finally {
    tab.close()
    network.close()
    first.close()
    next.close()
  }
}

async function resetDuringSetup(window: BrowserWindow) {
  const proxy = await peer('stale route')
  let finish!: () => void
  let entering!: () => void
  const pending = new Promise<void>(resolve => { finish = resolve })
  const entered = new Promise<void>(resolve => { entering = resolve })
  const network = new RemoteNetwork(machine, {
    freePort: async () => proxy.port,
    spawn: () => new ChildProcess(),
    waitReady: async () => { entering(); await pending },
  })
  const tab = new BrowserTab(window, () => network, () => {})
  try {
    const opening = tab.open(target)
    const rejected = assert.rejects(opening, /workspace changed/i)
    await entered
    tab.reset()
    finish()
    await rejected
    assert.equal(tab.state().url, '')
  } finally {
    tab.close()
    network.close()
    proxy.close()
  }
}

async function remoteWebRtcPolicy(window: BrowserWindow) {
  const proxy = await peer('remote WebRTC policy')
  const network = new RemoteNetwork(machine, {
    freePort: async () => proxy.port,
    spawn: () => new ChildProcess(),
    waitReady: (port, signal) => waitForLoopback(port, signal, 1000),
  })
  const tab = new BrowserTab(window, () => network, () => {})
  try {
    await tab.open(target)
    await until(tab, 'remote WebRTC fixture did not load', () => tab.state().title === 'remote WebRTC policy')
    const page = webContents.getAllWebContents().find(contents => contents.getTitle() === 'remote WebRTC policy')
    assert.ok(page, 'expected a native remote Browser view')
    assert.equal(page.getWebRTCIPHandlingPolicy(), 'disable_non_proxied_udp')
    assert.notEqual(window.webContents.getWebRTCIPHandlingPolicy(), 'disable_non_proxied_udp', 'leave the shell policy alone')
  } finally {
    tab.close()
    network.close()
    proxy.close()
  }
}

const profile = await mkdtemp(join(tmpdir(), 'xerxes-browser-route-'))
app.setPath('userData', profile)
// Let the fixture set its exit code after closing the last window.
app.on('window-all-closed', () => {})
await app.whenReady()
const window = new BrowserWindow({ show: false })
// Suppress Electron's own default-session networking. The tab has a dedicated
// in-memory partition whose only proxy is the synthetic peer above.
await session.defaultSession.setProxy({ proxyRules: 'socks5://127.0.0.1:1', proxyBypassRules: '<-loopback>' })
let failed = 0
for (const [name, verify] of [
  ['reload replaces a disconnected SOCKS listener', () => recover(window, 'reload')],
  ['address resubmission recovers the SOCKS route', () => recover(window, 'open')],
  ['failed first navigation stays retryable', () => recover(window, 'reload', true)],
  ['workspace reset discards a pending browser view', () => resetDuringSetup(window)],
  ['remote Browser disables non-proxied WebRTC UDP', () => remoteWebRtcPolicy(window)],
] as const) {
  try { await verify(); console.log(`PASS ${name}`) }
  catch (error) { failed++; console.error(`FAIL ${name}`, error) }
}
window.destroy()
await rm(profile, { recursive: true, force: true })
app.exit(failed ? 1 : 0)
