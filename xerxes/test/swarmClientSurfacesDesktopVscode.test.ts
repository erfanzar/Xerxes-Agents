// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from 'node:fs'
import http from 'node:http'
import { createServer, type Server, type Socket as NetSocket } from 'node:net'
import { tmpdir } from 'node:os'
import * as nodePath from 'node:path'
import { PassThrough } from 'node:stream'

import { afterEach, describe, expect, test } from 'bun:test'

import { DaemonRpc } from '../src/desktop/main/daemon.js'
import { nodeHubHttp } from '../src/desktop/main/hubHttp.js'
import { closePayload, connectHubSocket, wrapHubSocket } from '../src/desktop/main/hubSocket.js'
import { canonicalProjectDir, daemonAddress, legacyProjectDaemonPaths } from '../src/desktop/main/spawn.js'
import { isAllowedRendererDaemonMethod } from '../src/desktop/main/windowRoutes.js'
import type { PushFrame, ResultFrame } from '../src/vscode/messages.js'
import { PanelHost, isAllowedRevealPath, type PanelDaemon, type PanelPorts } from '../src/vscode/panelHost.js'
import { webviewTransport } from '../src/vscode/webviewTransport.js'

const tempDirs: string[] = []
function makeTempDir(prefix: string): string {
  const dir = mkdtempSync(nodePath.join(tmpdir(), prefix))
  tempDirs.push(dir)
  return dir
}

function listenUnix(server: Server, sockPath: string): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    server.once('error', reject)
    server.listen(sockPath, () => {
      server.removeListener('error', reject)
      resolve()
    })
  })
}

afterEach(() => {
  while (tempDirs.length > 0) {
    const dir = tempDirs.pop()!
    rmSync(dir, { recursive: true, force: true })
  }
})

/** Build an unmasked RFC 6455 server frame for wrapHubSocket tests. */
function serverWsFrame(opcode: number, payload: Uint8Array, fin = true): Buffer {
  const len = payload.length
  const head = len < 126 ? 2 : 4
  const out = Buffer.alloc(head + len)
  out[0] = (fin ? 0x80 : 0x00) | opcode
  if (len < 126) out[1] = len
  else {
    out[1] = 126
    out.writeUInt16BE(len, 2)
  }
  Buffer.from(payload).copy(out, head)
  return out
}

function makePorts(overrides: Partial<PanelPorts> = {}) {
  const memory = new Map<string, string>()
  const editor: string[] = []
  const value: PanelPorts = {
    workspace: '/work/repo',
    workspaceFolders: () => ['/work/repo'],
    remoteName: undefined,
    chooseFolder: async () => null,
    openFolder: async () => undefined,
    openPanel: () => undefined,
    revealPath: async () => true,
    backgroundData: async name => `data:${name}`,
    memory: {
      get: key => memory.get(key),
      set: (key, next) => {
        if (next === undefined) memory.delete(key)
        else memory.set(key, next)
      },
    },
    notify: () => undefined,
    visible: () => true,
    openFile: async (file, line) => {
      editor.push(`open:${file}:${line ?? ''}`)
      return true
    },
    openDiff: async file => {
      editor.push(`diff:${file}`)
      return true
    },
    pickFiles: async () => [],
    showSourceControl: async () => true,
    setStyle: async () => true,
    ...overrides,
  }
  return { value, memory, editor }
}

describe('VS Code PanelHost & webview boundary hardening', () => {
  test('PanelHost never spawns an orphaned daemon or posts frames after dispose()', async () => {
    let connectCalls = 0
    let disposedCount = 0
    let unblockFirst: (() => void) | undefined
    const daemon: PanelDaemon = {
      call: () =>
        new Promise<Record<string, unknown>>(resolve => {
          unblockFirst = () => resolve({ ok: true })
        }),
      onEvent: () => undefined,
      offEvent: () => undefined,
      dispose: () => {
        disposedCount += 1
      },
    }
    const posted: Array<ResultFrame | PushFrame> = []
    const host = new PanelHost(makePorts().value, frame => posted.push(frame), () => {
      connectCalls += 1
      return daemon
    })

    const inflight = host.receive({ kind: 'invoke', id: 1, channel: 'daemon:call', args: ['runtime.status', {}] })
    expect(connectCalls).toBe(1)

    host.dispose()
    expect(disposedCount).toBe(1)

    // Completing the in-flight call and receiving further frames after dispose must neither post nor reconnect.
    unblockFirst?.()
    await inflight
    await host.receive({ kind: 'invoke', id: 2, channel: 'daemon:call', args: ['runtime.status', {}] })
    expect(connectCalls).toBe(1)
    expect(posted).toEqual([])
  })

  test('PanelHost validates session IDs, editor file/line inputs, and blocks host-only daemon RPCs', async () => {
    const env = makePorts()
    env.memory.set('xerxes.resumeSession', '../bad-session-id')
    const posted: Array<ResultFrame | PushFrame> = []
    const daemonCalls: string[] = []
    const daemon: PanelDaemon = {
      call: async method => {
        daemonCalls.push(method)
        return { ok: true }
      },
      onEvent: () => undefined,
      offEvent: () => undefined,
      dispose: () => undefined,
    }
    const host = new PanelHost(env.value, frame => posted.push(frame), () => daemon)

    await host.receive({ kind: 'invoke', id: 1, channel: 'desktop:resume', args: [] })
    expect(posted.at(-1)).toEqual({ kind: 'result', id: 1, ok: true, value: null })

    for (const [id, channel, args] of [
      [2, 'host:open-file', [undefined]],
      [3, 'host:open-file', ['src/a.ts', 0]],
      [4, 'host:open-file', ['src/a.ts', -2]],
      [5, 'host:open-file', ['src/a.ts', Number.NaN]],
      [6, 'host:open-file', ['src/\u0000evil.ts', 1]],
      [7, 'host:open-diff', ['   ']],
      [8, 'daemon:call', ['provider.remote.reply', {}]],
      [9, 'daemon:call', ['connection.lease', {}]],
      [10, 'daemon:call', ['runtime.status', ['not-an-object']]],
    ] as const) {
      await host.receive({ kind: 'invoke', id, channel, args: [...args] })
      expect(posted.at(-1)).toMatchObject({ kind: 'result', id, ok: false })
    }

    expect(env.editor).toEqual([])
    expect(daemonCalls).toEqual([])
  })

  test('isAllowedRevealPath enforces containment across POSIX, dot-prefixed names, and Windows drives', () => {
    expect(isAllowedRevealPath('/work/repo/src/index.ts', ['/work/repo', '/Users/alice/.xerxes'], nodePath.posix)).toBe(true)
    expect(isAllowedRevealPath('/work/repo', ['/work/repo'], nodePath.posix)).toBe(true)
    // Valid dot-prefixed child inside workspace must not be mistaken for parent traversal.
    expect(isAllowedRevealPath('/work/repo/..cache/file.txt', ['/work/repo'], nodePath.posix)).toBe(true)
    // Traversal, sibling prefix, relative target, relative root, and NUL byte are all rejected.
    expect(isAllowedRevealPath('/work/repo-other/secret.txt', ['/work/repo'], nodePath.posix)).toBe(false)
    expect(isAllowedRevealPath('/work/repo/../secret.txt', ['/work/repo'], nodePath.posix)).toBe(false)
    expect(isAllowedRevealPath('relative/file.txt', ['/work/repo'], nodePath.posix)).toBe(false)
    expect(isAllowedRevealPath('/tmp/.xerxes/config.json', ['.xerxes'], nodePath.posix)).toBe(false)
    expect(isAllowedRevealPath('/work/repo/bad\u0000.txt', ['/work/repo'], nodePath.posix)).toBe(false)

    // Windows cross-drive escape (`relative('C:\\repo', 'D:\\secret') === 'D:\\secret'`) must be rejected.
    expect(isAllowedRevealPath('C:\\work\\repo\\src\\a.ts', ['C:\\work\\repo'], nodePath.win32)).toBe(true)
    expect(isAllowedRevealPath('D:\\secret\\keys.json', ['C:\\work\\repo', 'C:\\Users\\me\\.xerxes'], nodePath.win32)).toBe(false)
  })

  test('webviewTransport cleans up pending requests on synchronous postMessage errors and handles listener unsubscribe during dispatch', async () => {
    let listener: ((event: Event) => void) | undefined
    const postedIds: number[] = []
    let shouldThrow = true
    const transport = webviewTransport(
      {
        postMessage(message) {
          if (shouldThrow) throw new Error('DataCloneError: could not clone object')
          const frame = message as { id: number }
          postedIds.push(frame.id)
          listener?.({ data: { kind: 'result', id: frame.id, ok: true, value: 'ok' } } as unknown as Event)
        },
      },
      {
        addEventListener(_type: string, fn: EventListenerOrEventListenerObject) {
          listener = fn as (event: Event) => void
        },
      },
      () => undefined,
    )

    await expect(transport.invoke('desktop:workspace')).rejects.toThrow('DataCloneError')
    shouldThrow = false
    await expect(transport.invoke('desktop:workspace')).resolves.toBe('ok')
    expect(postedIds).toEqual([2])

    const seen: string[] = []
    let offFirst = (): void => undefined
    offFirst = transport.on('daemon:event', () => {
      seen.push('first')
      offFirst()
    })
    const offSecond = transport.on('daemon:event', () => {
      seen.push('second')
    })

    listener?.({ data: { kind: 'push', channel: 'daemon:event', value: {} } } as unknown as Event)
    expect(seen).toEqual(['first', 'second'])

    seen.length = 0
    listener?.({ data: { kind: 'push', channel: 'daemon:event', value: {} } } as unknown as Event)
    expect(seen).toEqual(['second'])
    offSecond()
  })
})

describe('Desktop main process transport & IPC hardening', () => {
  test('isAllowedRendererDaemonMethod blocks host-only private methods and malformed names', () => {
    expect(isAllowedRendererDaemonMethod('runtime.status')).toBe(true)
    expect(isAllowedRendererDaemonMethod('session.open')).toBe(true)
    expect(isAllowedRendererDaemonMethod('provider.remote.reply')).toBe(false)
    expect(isAllowedRendererDaemonMethod('connection.lease')).toBe(false)
    expect(isAllowedRendererDaemonMethod('')).toBe(false)
    expect(isAllowedRendererDaemonMethod('bad method')).toBe(false)
    expect(isAllowedRendererDaemonMethod(null)).toBe(false)
  })

  test('DaemonRpc detaches legacy project daemon socket cleanly after idle migration without false disconnect events', async () => {
    const root = realpathSync(makeTempDir('m'))
    const rawProjectDir = nodePath.join(root, 'p')
    const homeDir = nodePath.join(root, 'h')
    mkdirSync(rawProjectDir, { recursive: true })
    const canonicalDir = canonicalProjectDir(rawProjectDir)
    const env = { XERXES_HOME: homeDir }
    const legacySockPath = legacyProjectDaemonPaths(canonicalDir, env).socketPath
    const sharedSockPath = daemonAddress(canonicalDir, env).socketPath
    mkdirSync(nodePath.dirname(legacySockPath), { recursive: true })
    mkdirSync(nodePath.dirname(sharedSockPath), { recursive: true })

    let legacyAlive = true
    const writeFlush = (sock: NetSocket, payload: unknown): void => {
      setImmediate(() => {
        if (sock.destroyed) return
        sock.write(`${JSON.stringify(payload)}\n`)
        ;(sock as NetSocket & { flush?: () => void }).flush?.()
      })
    }

    const legacySockets: NetSocket[] = []
    const legacyServer = createServer(sock => {
      legacySockets.push(sock)
      sock.setEncoding('utf8')
      let buf = ''
      sock.on('data', chunk => {
        buf += chunk
        let nl = buf.indexOf('\n')
        while (nl !== -1) {
          const line = buf.slice(0, nl)
          buf = buf.slice(nl + 1)
          if (line.trim()) {
            const req = JSON.parse(line) as { id: number; method: string }
            if (req.method === 'runtime.status') {
              writeFlush(sock, { jsonrpc: '2.0', id: req.id, result: { ok: true, pid: 4242 } })
            } else if (req.method === 'runtime.restart_if_idle') {
              legacyAlive = false
              writeFlush(sock, { jsonrpc: '2.0', id: req.id, result: { ok: true } })
            }
          }
          nl = buf.indexOf('\n')
        }
      })
    })
    const sharedServer = createServer(sock => {
      sock.setEncoding('utf8')
      let buf = ''
      sock.on('data', chunk => {
        buf += chunk
        let nl = buf.indexOf('\n')
        while (nl !== -1) {
          const line = buf.slice(0, nl)
          buf = buf.slice(nl + 1)
          if (line.trim()) {
            const req = JSON.parse(line) as { id: number; method: string }
            writeFlush(sock, { jsonrpc: '2.0', id: req.id, result: { ok: true, migrated: req.method } })
          }
          nl = buf.indexOf('\n')
        }
      })
    })

    await listenUnix(legacyServer, legacySockPath)
    await listenUnix(sharedServer, sharedSockPath)

    const rpc = new DaemonRpc({
      projectDir: rawProjectDir,
      env,
      deadlineMs: 2_000,
      isProcessAlive: () => legacyAlive,
    })
    const transitions: boolean[] = []
    rpc.onConnection(online => transitions.push(online))

    try {
      const res = await rpc.call<{ ok: boolean; migrated: string }>('ping', {})
      expect(res).toEqual({ ok: true, migrated: 'ping' })
      await new Promise(resolve => setTimeout(resolve, 40))
      expect(legacySockets[0]?.destroyed).toBe(true)
      expect(transitions).toEqual([true])
      expect(rpc.online).toBe(true)
    } finally {
      rpc.dispose()
      legacyServer.close()
      sharedServer.close()
    }
  })

  test('DaemonRpc aborts in-flight socket connect on dispose() without attaching a zombie socket', async () => {
    const root = makeTempDir('xdd-')
    const sockPath = nodePath.join(root, 'slow.sock')
    const accepted: NetSocket[] = []
    const server = createServer(sock => {
      accepted.push(sock)
    })
    await new Promise<void>(resolve => server.listen(sockPath, resolve))

    const rpc = new DaemonRpc({ projectDir: root, socketPath: sockPath, deadlineMs: 1_000 })
    const transitions: boolean[] = []
    rpc.onConnection(online => transitions.push(online))

    const pending = rpc.call('runtime.status', {})
    rpc.dispose()

    await expect(pending).rejects.toThrow()
    await new Promise(resolve => setTimeout(resolve, 40))
    expect(rpc.online).toBe(false)
    expect(transitions).toEqual([])
    for (const sock of accepted) sock.destroy()
    server.close()
  })

  test('hubSocket closePayload preserves valid UTF-8 at the 123-byte RFC 6455 boundary', () => {
    const decoder = new TextDecoder('utf-8', { fatal: true })
    for (const pad of [120, 121, 122, 123]) {
      const reason = 'a'.repeat(pad) + '…trailing'
      const payload = closePayload(1000, reason)
      expect(payload.length).toBeLessThanOrEqual(125)
      expect(() => decoder.decode(payload.subarray(2))).not.toThrow()
    }
  })

  test('wrapHubSocket handles fragmented binary messages and rejects illegal 1-byte close frames', () => {
    const duplex = new PassThrough()
    const events: string[] = []
    let closeInfo: { code: number; reason: string } | undefined
    wrapHubSocket(
      duplex,
      {
        open: () => events.push('open'),
        message: text => events.push(`msg:${text}`),
        binary: () => events.push('binary'),
        error: err => events.push(`err:${err}`),
        close: (code, reason) => {
          closeInfo = { code, reason }
        },
      },
      { maxMessageBytes: 1024 },
    )

    // Fragmented binary (0x2 fin=false + 0x0 fin=true) must report binary once without failing with 1002.
    duplex.write(serverWsFrame(0x2, Buffer.from([1, 2, 3]), false))
    duplex.write(serverWsFrame(0x0, Buffer.from([4, 5]), true))
    // Followed by a normal text frame on the same connection.
    duplex.write(serverWsFrame(0x1, Buffer.from('alive', 'utf8'), true))
    expect(events).toEqual(['binary', 'msg:alive'])
    expect(closeInfo).toBeUndefined()

    // Illegal 1-byte close frame (RFC 6455 §5.5.1) must fail with 1002.
    duplex.write(serverWsFrame(0x8, Buffer.from([0x03]), true))
    expect(closeInfo?.code).toBe(1002)
  })

  test('connectHubSocket destroys late-upgraded sockets when closed before HTTP 101 arrives', async () => {
    let serverUpgradedSocket: NetSocket | undefined
    const server = http.createServer()
    server.on('upgrade', (req, socket) => {
      serverUpgradedSocket = socket as NetSocket
      const key = String(req.headers['sec-websocket-key'] ?? '')
      const accept = createHash('sha1')
        .update(`${key}258EAFA5-E914-47DA-95CA-C5AB0DC85B11`)
        .digest('base64')
      setTimeout(() => {
        if (!socket.destroyed) {
          socket.write(
            `HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Accept: ${accept}\r\n\r\n`,
          )
        }
      }, 25)
    })

    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    const address = server.address() as { port: number }
    const seen: string[] = []

    try {
      const hubSock = connectHubSocket(
        {
          url: `ws://127.0.0.1:${address.port}/ws`,
          origin: `http://127.0.0.1:${address.port}`,
          cookie: '__Host-xerxes_session=tok',
        },
        {
          open: () => seen.push('open'),
          message: () => seen.push('message'),
          binary: () => seen.push('binary'),
          error: msg => seen.push(`error:${msg}`),
          close: code => seen.push(`close:${code}`),
        },
      )

      hubSock.close(1000, 'cancelled')
      await new Promise(resolve => setTimeout(resolve, 60))
      expect(seen).toEqual(['close:1000'])
      expect(serverUpgradedSocket === undefined || serverUpgradedSocket.destroyed).toBe(true)
    } finally {
      server.close()
    }
  })

  test('nodeHubHttp rejects instead of hanging when the server drops a chunked response mid-body', async () => {
    const server = http.createServer((_req, res) => {
      res.writeHead(200, { 'content-type': 'application/json', 'transfer-encoding': 'chunked' })
      res.write('{"partial":')
      setTimeout(() => res.socket?.destroy(), 15)
    })
    await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
    const address = server.address() as { port: number }

    try {
      await expect(
        nodeHubHttp({
          url: `http://127.0.0.1:${address.port}/api/hub/v1/session/refresh`,
          headers: { origin: `http://127.0.0.1:${address.port}`, 'content-type': 'application/json' },
          body: '{}',
          timeoutMs: 2_000,
        }),
      ).rejects.toThrow()
    } finally {
      server.close()
    }
  })
})
