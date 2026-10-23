// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { HubConnection, type HubConnectionOptions, type SocketHandlers, type SocketLike, type Timers } from '../../client/hubConnection.js'
import { HUB_PROTOCOL_VERSION, type HubEvent } from '../../hub/protocol.js'

export interface Frame {
  readonly v: number
  readonly kind: string
  readonly id: string
  readonly method: string
  readonly params: Record<string, unknown>
}

export class FakeTimers implements Timers {
  now = 0
  private seq = 0
  private readonly items = new Map<number, { at: number; cb: () => void }>()
  setTimeout(cb: () => void, ms: number): unknown {
    const id = ++this.seq
    this.items.set(id, { at: this.now + ms, cb })
    return id
  }
  clearTimeout(handle: unknown): void {
    this.items.delete(handle as number)
  }
  advance(ms: number): void {
    const target = this.now + ms
    for (;;) {
      let best: [number, { at: number; cb: () => void }] | undefined
      for (const entry of this.items) if (entry[1].at <= target && (best === undefined || entry[1].at < best[1].at)) best = entry
      if (best === undefined) break
      this.items.delete(best[0])
      this.now = Math.max(this.now, best[1].at)
      best[1].cb()
    }
    this.now = target
  }
}

type ErrorReply = { readonly __error: { readonly code: string; readonly message: string } }
export type Reply = Record<string, unknown> | ErrorReply | undefined

function isErrorReply(reply: Reply): reply is ErrorReply {
  return reply !== undefined && typeof reply === 'object' && '__error' in reply
}

export class FakeSocket implements SocketLike {
  readonly sent: Frame[] = []
  closedWith: number | undefined
  constructor(readonly handlers: SocketHandlers, private readonly hub: FakeHub) {}
  send(data: string): void {
    const frame = JSON.parse(data) as Frame
    this.sent.push(frame)
    this.hub.onRequest(this, frame)
  }
  close(code?: number): void {
    this.closedWith = code ?? 1000
  }
  deliver(frame: unknown): void {
    this.handlers.message(JSON.stringify(frame))
  }
  respond(id: string, result: Record<string, unknown>): void {
    this.deliver({ v: HUB_PROTOCOL_VERSION, kind: 'response', id, ok: true, result })
  }
  fail(id: string, code: string, message: string): void {
    this.deliver({ v: HUB_PROTOCOL_VERSION, kind: 'response', id, ok: false, error: { code, message } })
  }
  event(type: string, payload: Record<string, unknown>): void {
    this.deliver({ v: HUB_PROTOCOL_VERSION, kind: 'event', type, payload })
  }
  open(): void {
    this.handlers.open()
  }
  drop(code = 1006): void {
    this.handlers.close(code, '')
  }
  framesFor(method: string): Frame[] {
    return this.sent.filter(frame => frame.method === method)
  }
}

export class FakeHub {
  readonly sockets: FakeSocket[] = []
  readonly handlers = new Map<string, (params: Record<string, unknown>, socket: FakeSocket) => Reply>()
  private subscriptions = 0
  constructor() {
    this.handlers.set('hub.hello', () => ({ principal: { id: 'u1', display_name: 'Ann' } }))
    this.handlers.set('hub.events.subscribe', () => ({ subscription_id: `sub${++this.subscriptions}`, watermark: 'W', oldest_cursor: 'O', reset_required: false }))
    this.handlers.set('hub.events.unsubscribe', () => ({}))
    this.handlers.set('hub.events.ack', () => ({}))
  }
  readonly factory = (_url: string, handlers: SocketHandlers): SocketLike => {
    const socket = new FakeSocket(handlers, this)
    this.sockets.push(socket)
    return socket
  }
  onRequest(socket: FakeSocket, frame: Frame): void {
    const handler = this.handlers.get(frame.method)
    if (handler === undefined) return
    const reply = handler(frame.params, socket)
    if (reply === undefined) return
    queueMicrotask(() => {
      if (isErrorReply(reply)) socket.fail(frame.id, reply.__error.code, reply.__error.message)
      else socket.respond(frame.id, reply as Record<string, unknown>)
    })
  }
  socket(index: number): FakeSocket {
    const found = this.sockets[index]
    if (found === undefined) throw new Error(`no socket #${index}`)
    return found
  }
  all(method: string): Frame[] {
    return this.sockets.flatMap(socket => socket.framesFor(method))
  }
}

/** Let every queued microtask chain finish. */
export function settle(): Promise<void> {
  return new Promise(resolve => { setTimeout(resolve, 0) })
}

export interface Harness {
  readonly hub: FakeHub
  readonly timers: FakeTimers
  readonly conn: HubConnection
  readonly errors: Error[]
}

export function harness(options: Partial<HubConnectionOptions> = {}): Harness {
  const hub = new FakeHub()
  const timers = new FakeTimers()
  const errors: Error[] = []
  const conn = new HubConnection({
    url: 'wss://hub.example/api/hub/v1/ws',
    socketFactory: hub.factory,
    timers,
    random: () => 0.5,
    onProtocolError: error => errors.push(error),
    ...options,
  })
  return { hub, timers, conn, errors }
}

/** Connect and finish the hello exchange. */
export async function goLive(h: Harness): Promise<FakeSocket> {
  h.conn.connect()
  await settle()
  const socket = h.hub.socket(h.hub.sockets.length - 1)
  socket.open()
  await settle()
  return socket
}

let counter = 0
export function ev(sequence: number, extra: Partial<HubEvent> = {}): HubEvent {
  counter += 1
  return {
    event_id: `ev-${sequence}-${counter}`, sequence, cursor: `cur${sequence}`, workspace_id: 'w1', session_id: 's1',
    type: 'text_part', payload: { text: 'x' }, occurred_at: '2026-10-21T12:00:00.000Z', ...extra,
  }
}

export function batch(socket: FakeSocket, subscriptionId: string, events: readonly HubEvent[], through: string): void {
  socket.event('hub.event_batch', { subscription_id: subscriptionId, events, through_cursor: through })
}
