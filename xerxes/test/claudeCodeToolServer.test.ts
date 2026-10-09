// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { expect, test } from 'bun:test'
import { declareClaudeCodeTools } from '../src/llms/claudeCodeToolServer.js'

const rpc = async (url: string, body: unknown) => {
  const response = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })
  return { status: response.status, body: response.status === 202 ? undefined : await response.json() as Record<string, any> }
}

test('declares tools to Claude Code over MCP, never runs them, and stops serving when released', async () => {
  const declaration = declareClaudeCodeTools([
    { type: 'function', function: { name: 'read_file', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' } } } } },
    { type: 'function', function: { name: 'fs.write', description: 'Write', parameters: { type: 'object', properties: {} } } },
  ])
  expect(declaration.url).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/mcp\/[0-9a-f-]{36}$/)
  const init = await rpc(declaration.url, { jsonrpc: '2.0', id: 1, method: 'initialize', params: { protocolVersion: '2025-06-18' } })
  expect(init.body?.result).toMatchObject({ protocolVersion: '2025-06-18', capabilities: { tools: {} } })
  expect((await rpc(declaration.url, { jsonrpc: '2.0', method: 'notifications/initialized' })).status).toBe(202)
  const listed = await rpc(declaration.url, { jsonrpc: '2.0', id: 2, method: 'tools/list' })
  // A name outside MCP's grammar is declared under a legal one and mapped back.
  expect(listed.body?.result.tools.map((tool: { name: string }) => tool.name)).toEqual(['read_file', 'fs_write'])
  expect(declaration.toolName('mcp__xerxes__fs_write')).toBe('fs.write')
  expect(declaration.toolName('read_file')).toBe('read_file')
  const called = await rpc(declaration.url, { jsonrpc: '2.0', id: 3, method: 'tools/call', params: { name: 'read_file', arguments: {} } })
  expect(called.body?.result.isError).toBe(true)
  // Another request's path is not this one's.
  expect((await fetch(declaration.url.replace(/[0-9a-f-]{36}$/, crypto.randomUUID()), { method: 'POST', body: '{}' })).status).toBe(404)
  const other = declareClaudeCodeTools([])
  declaration.release()
  expect((await fetch(declaration.url, { method: 'POST', body: '{}' })).status).toBe(404)
  other.release()
})
