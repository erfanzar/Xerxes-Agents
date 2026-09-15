// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Route host capabilities by the sending renderer, never by the focused window. */
export class WindowRoutes<Event extends { sender: { id: number } }> {
  private readonly windows = new Map<number, Map<string, (event: Event, args: unknown[]) => unknown>>()

  bind<Args extends unknown[], Result>(id: number, channel: string, handler: (event: Event, ...args: Args) => Result): void {
    let routes = this.windows.get(id)
    if (!routes) { routes = new Map(); this.windows.set(id, routes) }
    routes.set(channel, (event, args) => handler(event, ...args as Args))
  }

  invoke(channel: string, event: Event, args: unknown[]): unknown {
    const handler = this.windows.get(event.sender.id)?.get(channel)
    if (!handler) throw new Error('This workspace window is closed or unavailable')
    return handler(event, args)
  }

  remove(id: number): void { this.windows.delete(id) }
}

export type WindowEvent = (type: string, payload: Record<string, unknown>) => void
export interface WindowConnection {
  onEvent(handler: WindowEvent): void
  offEvent(handler: WindowEvent): void
}

/** Owns subscriptions only; closing a window never shuts down its project daemon. */
export class WindowConnections<Connection extends WindowConnection> {
  private readonly entries = new Map<number, { connection: Connection; forward: WindowEvent }>()
  get(id: number): Connection | undefined { return this.entries.get(id)?.connection }
  attach(id: number, connection: Connection, send: WindowEvent): void {
    this.detach(id)
    const forward: WindowEvent = (type, payload) => {
      if (this.entries.get(id)?.connection === connection) send(type, payload)
    }
    this.entries.set(id, { connection, forward })
    connection.onEvent(forward)
  }
  detach(id: number): void {
    const previous = this.entries.get(id)
    this.entries.delete(id)
    if (previous) previous.connection.offEvent(previous.forward)
  }
}
