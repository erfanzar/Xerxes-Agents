// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { WindowConnections, WindowRoutes, type WindowEvent } from '../src/desktop/main/windowRoutes.js'

class Connection {
  readonly handlers = new Set<WindowEvent>()
  onEvent(handler: WindowEvent): void { this.handlers.add(handler) }
  offEvent(handler: WindowEvent): void { this.handlers.delete(handler) }
  emit(message: string): void { for (const handler of this.handlers) handler('message', { message }) }
}

test('workspace operations route by sender and closing one window leaves the other usable', async () => {
  const routes = new WindowRoutes<{ sender: { id: number } }>()
  let first = '/project-a', second = '/project-b'
  routes.bind(1, 'switch', (_event, path: string) => { first = path })
  routes.bind(2, 'switch', (_event, path: string) => { second = path })
  routes.invoke('switch', { sender: { id: 2 } }, ['/project-c'])
  expect(first).toBe('/project-a')
  expect(second).toBe('/project-c')
  routes.remove(2)
  expect(() => routes.invoke('switch', { sender: { id: 2 } }, ['/project-d'])).toThrow('closed')
  routes.invoke('switch', { sender: { id: 1 } }, ['/project-e'])
  expect(first).toBe('/project-e')
  expect(() => routes.invoke('switch', { sender: { id: 999 } }, ['/other'])).toThrow('unavailable')
})

test('concurrent daemon streams, replacement, and late events stay in their owning windows', () => {
  const bindings = new WindowConnections<Connection>()
  const a = new Connection(), b = new Connection(), c = new Connection()
  const first: string[] = [], second: string[] = []
  bindings.attach(1, a, (_type, payload) => first.push(String(payload.message)))
  bindings.attach(2, b, (_type, payload) => second.push(String(payload.message)))
  a.emit('a1'); b.emit('b1'); a.emit('a2')
  const queuedOld = [...a.handlers][0]!
  bindings.attach(1, c, (_type, payload) => first.push(String(payload.message)))
  a.emit('stale'); queuedOld('message', { message: 'queued stale' }); c.emit('c1'); b.emit('b2')
  expect(first).toEqual(['a1', 'a2', 'c1'])
  expect(second).toEqual(['b1', 'b2'])
  bindings.detach(1)
  c.emit('closed'); b.emit('b3')
  expect(c.handlers.size).toBe(0)
  expect(bindings.get(1)).toBeUndefined()
  expect(bindings.get(2)).toBe(b)
  expect(second).toEqual(['b1', 'b2', 'b3'])
})
