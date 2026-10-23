// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'

import { VOICE_PATH } from '../src/hub/voice/protocol.js'
import { createHubVoiceRoute } from '../src/hub/voice/route.js'
import { disposeAll, registerCleanup, startFixture, until } from './hubHarness.js'
import { Rig, control } from './hubVoiceHarness.js'

afterEach(disposeAll)

interface VoiceClient {
  readonly ws: WebSocket
  readonly messages: Record<string, unknown>[]
  closeCode: number | undefined
}

async function connectVoice(port: number, who: string | undefined, origin: string | null = `http://127.0.0.1:${port}`): Promise<VoiceClient | undefined> {
  const headers: Record<string, string> = {}
  if (origin !== null) headers.origin = origin
  if (who !== undefined) headers.cookie = `who=${who}`
  const BunWebSocket = WebSocket as unknown as new (url: string, options: { headers: Record<string, string> }) => WebSocket
  const ws = new BunWebSocket(`ws://127.0.0.1:${port}${VOICE_PATH}`, { headers })
  const client: VoiceClient = { ws, messages: [], closeCode: undefined }
  ws.addEventListener('message', event => {
    if (typeof event.data === 'string') client.messages.push(JSON.parse(event.data) as Record<string, unknown>)
  })
  ws.addEventListener('close', event => { client.closeCode = event.code })
  const opened = await new Promise<boolean>(resolve => {
    ws.addEventListener('open', () => resolve(true))
    ws.addEventListener('error', () => resolve(false))
    ws.addEventListener('close', () => resolve(false))
  })
  if (!opened) return undefined
  registerCleanup(() => { try { ws.close() } catch { /* already closed */ } })
  return client
}

describe('voice socket mounted on the Hub server', () => {
  test('an authenticated person reaches the voice session; the Hub socket keeps working beside it', async () => {
    const rig = new Rig()
    const probe = await startFixture()
    const route = createHubVoiceRoute({ identity: probe.identity, supervisor: rig.bridge, config: rig.config, clock: rig.clock })
    registerCleanup(() => route.close())
    const fixture = await startFixture({ server: { voice: route } })

    const voice = await connectVoice(fixture.port, 'alice')
    expect(voice).toBeDefined()
    voice!.ws.send(control({ type: 'capabilities' }))
    const reply = await until(() => voice!.messages.find(message => message.type === 'capabilities'))
    expect(reply).toMatchObject({ type: 'capabilities', capabilities: { enabled: true, modes: ['push_to_talk'] } })
    expect(route.sessionCount).toBe(1)

    const hub = await fixture.client('alice')
    expect(await hub.ok('hub.hello')).toMatchObject({})

    voice!.ws.close()
    await until(() => route.sessionCount === 0 || undefined)
  })

  test('anonymous callers and foreign origins are refused before any voice session exists', async () => {
    const rig = new Rig()
    const probe = await startFixture()
    const route = createHubVoiceRoute({ identity: probe.identity, supervisor: rig.bridge, config: rig.config, clock: rig.clock })
    registerCleanup(() => route.close())
    const fixture = await startFixture({ server: { voice: route } })

    expect(await connectVoice(fixture.port, undefined)).toBeUndefined()
    expect(await connectVoice(fixture.port, 'alice', 'https://evil.example')).toBeUndefined()
    const plain = await fetch(fixture.url(VOICE_PATH), { headers: { cookie: 'who=alice' } })
    expect(plain.status).toBe(426)
    expect(route.sessionCount).toBe(0)
  })

  test('without a voice route the path is not served, so the phone probe fails and the talk button stays hidden', async () => {
    const fixture = await startFixture()
    expect(await connectVoice(fixture.port, 'alice')).toBeUndefined()
    const response = await fetch(fixture.url(VOICE_PATH), { headers: { cookie: 'who=alice' } })
    expect(response.status).toBe(404)
  })

  test('stopping the server ends voice sessions with 1001 (read from the raw close frame)', async () => {
    const rig = new Rig()
    const probe = await startFixture()
    const route = createHubVoiceRoute({ identity: probe.identity, supervisor: rig.bridge, config: rig.config, clock: rig.clock })
    const fixture = await startFixture({ server: { voice: route } })
    // The Bun WebSocket client rewrites a server 1001 to 1000, so read the wire bytes directly.
    let received = Buffer.alloc(0)
    const socket = await Bun.connect({
      hostname: '127.0.0.1', port: fixture.port,
      socket: {
        data(_socket, chunk) { received = Buffer.concat([received, chunk]) },
        open(open) {
          open.write([
            `GET ${VOICE_PATH} HTTP/1.1`, `Host: 127.0.0.1:${fixture.port}`, 'Upgrade: websocket', 'Connection: Upgrade',
            `Origin: http://127.0.0.1:${fixture.port}`, 'Cookie: who=alice', 'Sec-WebSocket-Version: 13',
            'Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==', '', '',
          ].join('\r\n'))
        },
      },
    })
    registerCleanup(() => { socket.end() })
    await until(() => received.includes('\r\n\r\n') || undefined)
    expect(received.toString('latin1').startsWith('HTTP/1.1 101')).toBe(true)
    await until(() => route.sessionCount === 1 || undefined)
    await fixture.server.stop()
    const closeCode = (): number | undefined => {
      let offset = received.indexOf('\r\n\r\n') + 4
      while (offset + 2 <= received.length) {
        const opcode = received[offset]! & 0x0f
        const length = received[offset + 1]! & 0x7f
        if (length >= 126) return undefined
        if (offset + 2 + length > received.length) return undefined
        if (opcode === 0x8) return length >= 2 ? received.readUInt16BE(offset + 2) : 1005
        offset += 2 + length
      }
      return undefined
    }
    expect(await until(closeCode)).toBe(1001)
    expect(route.sessionCount).toBe(0)
  })
})
