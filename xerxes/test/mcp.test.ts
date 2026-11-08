// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { MCPClient } from '../src/mcp/client.js'
import { MCPConnectionError, MCPProtocolError, type MCPJsonRpcResponse } from '../src/mcp/types.js'

const TEST_MCP_SERVER = String.raw`
let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
  buffer += chunk;
  let newline = buffer.indexOf('\n');
  while (newline >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (line) {
      handle(JSON.parse(line));
    }
    newline = buffer.indexOf('\n');
  }
});

function handle(request) {
  if (!Object.hasOwn(request, 'id')) {
    return;
  }
  const params = request.params || {};
  let result;
  switch (request.method) {
    case 'initialize':
      result = {
        protocolVersion: '2024-11-05',
        capabilities: { tools: {} },
        serverInfo: { name: 'fixture-mcp', version: '1.2.3' },
      };
      break;
    case 'tools/list':
      result = {
        tools: [{
          name: 'echo',
          description: 'Echo a value.',
          inputSchema: { type: 'object', properties: { value: { type: 'string' } } },
        }],
      };
      break;
    case 'resources/list':
      result = {
        resources: [{ uri: 'memo://welcome', name: 'Welcome', description: 'Greeting', mimeType: 'text/plain' }],
      };
      break;
    case 'prompts/list':
      result = { prompts: [{ name: 'greet', description: 'Greeting prompt', arguments: [{ name: 'name', required: true }] }] };
      break;
    case 'tools/call':
      result = { content: [{ type: 'text', text: 'echo:' + params.arguments.value }] };
      break;
    case 'resources/read':
      result = { contents: [{ uri: params.uri, mimeType: 'text/plain', text: 'hello resource' }] };
      break;
    case 'prompts/get':
      result = { messages: [{ role: 'user', content: { type: 'text', text: 'hello ' + params.arguments.name } }] };
      break;
    default:
      process.stdout.write(JSON.stringify({
        jsonrpc: '2.0', id: request.id, error: { code: -32601, message: 'method not found' },
      }) + '\n');
      return;
  }
  process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id: request.id, result }) + '\n');
}
`

test('MCPClient completes stdio initialization, discovery, tool calls, resources, and prompts', async () => {
  const client = new MCPClient({
    name: 'fixture',
    command: process.execPath,
    args: ['-e', TEST_MCP_SERVER],
    timeoutMs: 5_000,
  })
  try {
    await client.connect()
    expect(client.connected).toBeTrue()
    expect(client.serverInfo).toEqual({ name: 'fixture-mcp', version: '1.2.3' })
    expect(client.tools).toEqual([
      {
        name: 'echo',
        description: 'Echo a value.',
        inputSchema: { type: 'object', properties: { value: { type: 'string' } } },
        serverName: 'fixture',
      },
    ])
    expect(client.resources).toEqual([
      {
        uri: 'memo://welcome',
        name: 'Welcome',
        description: 'Greeting',
        mimeType: 'text/plain',
        serverName: 'fixture',
      },
    ])
    expect(client.prompts).toEqual([
      {
        name: 'greet',
        description: 'Greeting prompt',
        arguments: [{ name: 'name', required: true }],
        serverName: 'fixture',
      },
    ])

    await expect(client.callTool('echo', { value: 'hi' })).resolves.toEqual({
      content: [{ type: 'text', text: 'echo:hi' }],
    })
    await expect(client.readResource('memo://welcome')).resolves.toEqual({
      contents: [{ uri: 'memo://welcome', mimeType: 'text/plain', text: 'hello resource' }],
    })
    await expect(client.getPrompt('greet', { name: 'Ada' })).resolves.toEqual({
      messages: [{ role: 'user', content: { type: 'text', text: 'hello Ada' } }],
    })
  } finally {
    await client.disconnect()
  }
})

const TEST_MCP_CHATTY_SERVER = String.raw`
let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', chunk => {
  buffer += chunk;
  let newline = buffer.indexOf('\n');
  while (newline >= 0) {
    const line = buffer.slice(0, newline).trim();
    buffer = buffer.slice(newline + 1);
    if (line) {
      handle(JSON.parse(line));
    }
    newline = buffer.indexOf('\n');
  }
});

function write(frame) {
  process.stdout.write(JSON.stringify(frame) + '\n');
}

function handle(request) {
  if (!Object.hasOwn(request, 'id')) {
    return;
  }
  const params = request.params || {};
  switch (request.method) {
    case 'initialize':
      process.stdout.write('fixture boot diagnostic (not json)\n');
      write({ jsonrpc: '2.0', method: 'notifications/message', params: { level: 'info' } });
      write({ id: 'unrelated', result: {} });
      write({
        jsonrpc: '2.0',
        id: request.id,
        result: {
          protocolVersion: '2024-11-05',
          capabilities: { tools: {} },
          serverInfo: { name: 'chatty-mcp', version: '0.0.1' },
        },
      });
      return;
    case 'tools/list':
      write({ jsonrpc: '2.0', id: request.id, result: { tools: [
        { name: 'echo', inputSchema: { type: 'object' } },
        { name: 'hold', inputSchema: { type: 'object' } },
        { name: 'poison', inputSchema: { type: 'object' } },
      ] } });
      return;
    case 'tools/call':
      if (params.name === 'hold') {
        return;
      }
      if (params.name === 'poison') {
        write({ id: request.id });
        return;
      }
      process.stdout.write('another non-json diagnostic\n');
      write({ jsonrpc: '2.0', method: 'notifications/message', params: { level: 'info' } });
      write({ jsonrpc: '2.0', id: request.id, result: { content: [{ type: 'text', text: 'echo:ok' }] } });
      return;
    default:
      write({ jsonrpc: '2.0', id: request.id, error: { code: -32601, message: 'method not found' } });
  }
}
`

test('MCPClient survives garbage stdout lines, malformed frames, and throwing notification handlers', async () => {
  const debug: string[] = []
  const client = new MCPClient(
    { name: 'chatty', command: process.execPath, args: ['-e', TEST_MCP_CHATTY_SERVER], timeoutMs: 5_000 },
    { debug: message => { debug.push(message) } },
  )
  client.onNotification(() => {
    throw new Error('subscriber exploded')
  })
  try {
    await client.connect()
    expect(client.connected).toBeTrue()
    expect(client.serverInfo).toEqual({ name: 'chatty-mcp', version: '0.0.1' })

    await expect(client.callTool('echo')).resolves.toEqual({
      content: [{ type: 'text', text: 'echo:ok' }],
    })

    // A malformed frame rejects only the request it implicates.
    await expect(client.callTool('poison')).rejects.toBeInstanceOf(MCPProtocolError)
    await expect(client.callTool('echo')).resolves.toEqual({
      content: [{ type: 'text', text: 'echo:ok' }],
    })
    expect(client.connected).toBeTrue()
    expect(debug.some(message => message.includes('non-JSON'))).toBe(true)
    expect(debug.some(message => message.includes('subscriber exploded'))).toBe(true)
  } finally {
    await client.disconnect()
  }
})

test('MCPClient.callTool aborts through an AbortSignal and discards the pending request', async () => {
  const client = new MCPClient({
    name: 'chatty',
    command: process.execPath,
    args: ['-e', TEST_MCP_CHATTY_SERVER],
    timeoutMs: 30_000,
  })
  try {
    await client.connect()

    const preAborted = new AbortController()
    preAborted.abort()
    await expect(client.callTool('echo', {}, { signal: preAborted.signal }))
      .rejects.toBeInstanceOf(MCPConnectionError)

    const controller = new AbortController()
    const held = client.callTool('hold', {}, { signal: controller.signal })
    controller.abort()
    await expect(held).rejects.toBeInstanceOf(MCPConnectionError)

    // The connection stays healthy after a discarded in-flight request.
    await expect(client.callTool('echo')).resolves.toEqual({
      content: [{ type: 'text', text: 'echo:ok' }],
    })
  } finally {
    await client.disconnect()
  }
})

function resultOf(response: MCPJsonRpcResponse | undefined): Record<string, unknown> {
  if (!response || !('result' in response)) {
    throw new Error('Expected an MCP JSON-RPC success response')
  }
  return response.result
}
