// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Xerxes' tools, declared to Claude Code as native tools over MCP.
 *
 * `claude -p` has no stop sequences, so a text tool protocol ends a reply only
 * when the model writes the block close; when it does not, the reply runs to
 * the output limit (one subagent wrote 1,336 calls and 2,673 blank lines in a
 * single 23-minute reply). Declared as MCP tools, its calls are real tool_use
 * blocks and the API itself ends the reply (`stop_reason: tool_use`).
 *
 * The server only declares. Xerxes executes every call through its own loop,
 * permissions and policy; Claude Code runs with `--permission-mode dontAsk`,
 * so it never asks to run them, and `tools/call` here refuses all the same.
 * It listens on loopback, under one unguessable path per request.
 */
import type { ToolDefinition } from '../types/toolCalls.js'

/** MCP's tool name grammar; Claude Code shows each as `mcp__xerxes__<name>`. */
const MCP_TOOL_NAME = /^[A-Za-z0-9_-]{1,64}$/
/** The server name in Claude Code's MCP config, and so the tool name prefix. */
export const CLAUDE_CODE_TOOL_SERVER = 'xerxes'
export const CLAUDE_CODE_TOOL_PREFIX = `mcp__${CLAUDE_CODE_TOOL_SERVER}__`

interface McpTool {
  readonly name: string
  readonly description: string
  readonly inputSchema: Readonly<Record<string, unknown>>
}

export interface ClaudeCodeToolDeclaration {
  /** The endpoint for this request's MCP config. */
  readonly url: string
  /** The Xerxes tool a name Claude Code reports (with or without its prefix) refers to. */
  toolName(reported: string): string | undefined
  /** Stops serving this request's tools; the server closes with its last declaration. */
  release(): void
}

let server: ReturnType<typeof Bun.serve> | undefined
const routes = new Map<string, readonly McpTool[]>()

/** Serve `tools` to one Claude Code request until released. */
export function declareClaudeCodeTools(tools: readonly ToolDefinition[]): ClaudeCodeToolDeclaration {
  const names = new Map<string, string>()
  const declared: McpTool[] = []
  for (const tool of tools) {
    const name = mcpName(tool.function.name, names)
    names.set(name, tool.function.name)
    declared.push({
      name,
      description: tool.function.description ?? '',
      inputSchema: isRecord(tool.function.parameters) ? tool.function.parameters : { type: 'object', properties: {} },
    })
  }
  const token = crypto.randomUUID()
  routes.set(token, declared)
  server ??= Bun.serve({ hostname: '127.0.0.1', port: 0, fetch: answer })
  const url = `http://127.0.0.1:${server.port}/mcp/${token}`
  let released = false
  return {
    url,
    toolName: reported => names.get(reported.startsWith(CLAUDE_CODE_TOOL_PREFIX) ? reported.slice(CLAUDE_CODE_TOOL_PREFIX.length) : reported),
    release: () => {
      if (released) return
      released = true
      routes.delete(token)
      if (!routes.size && server) {
        void server.stop(true)
        server = undefined
      }
    },
  }
}

/** An MCP-legal name for `name`, unique among `taken`. */
function mcpName(name: string, taken: ReadonlyMap<string, string>): string {
  const base = MCP_TOOL_NAME.test(name) ? name : name.replace(/[^A-Za-z0-9_-]/g, '_').slice(0, 64) || 'tool'
  let candidate = base
  for (let index = 2; taken.has(candidate); index += 1) candidate = `${base.slice(0, 60)}_${index}`
  return candidate
}

async function answer(request: Request): Promise<Response> {
  const token = /^\/mcp\/([0-9a-f-]{36})$/.exec(new URL(request.url).pathname)?.[1]
  const tools = token ? routes.get(token) : undefined
  if (!tools) return new Response(null, { status: 404 })
  // Streamable HTTP without a server-initiated stream: POST only.
  if (request.method === 'DELETE') return new Response(null, { status: 200 })
  if (request.method !== 'POST') return new Response(null, { status: 405 })
  let body: unknown
  try { body = await request.json() } catch { return Response.json(rpcError(null, -32700, 'Parse error'), { status: 400 }) }
  const replies = (Array.isArray(body) ? body : [body]).map(message => reply(message, tools)).filter(item => item !== undefined)
  if (!replies.length) return new Response(null, { status: 202 })
  return Response.json(Array.isArray(body) ? replies : replies[0])
}

function reply(message: unknown, tools: readonly McpTool[]): unknown {
  if (!isRecord(message)) return rpcError(null, -32600, 'Invalid request')
  const id = message.id
  // Notifications (no id) get no answer.
  if (id === undefined) return undefined
  switch (message.method) {
    case 'initialize': {
      const params = isRecord(message.params) ? message.params : {}
      return rpcResult(id, {
        protocolVersion: typeof params.protocolVersion === 'string' ? params.protocolVersion : '2025-06-18',
        capabilities: { tools: {} },
        serverInfo: { name: CLAUDE_CODE_TOOL_SERVER, version: '1' },
      })
    }
    case 'tools/list':
      return rpcResult(id, { tools })
    case 'tools/call':
      // Claude Code runs in dontAsk mode and never calls; if it did, the
      // call must not look like it ran.
      return rpcResult(id, { isError: true, content: [{ type: 'text', text: 'Xerxes runs this tool itself; its result arrives in the next message.' }] })
    case 'ping':
      return rpcResult(id, {})
    default:
      return typeof message.method === 'string' && message.method.startsWith('server/') ? rpcResult(id, {}) : rpcError(id, -32601, 'Method not found')
  }
}

const rpcResult = (id: unknown, result: unknown) => ({ jsonrpc: '2.0', id, result })
const rpcError = (id: unknown, code: number, message: string) => ({ jsonrpc: '2.0', id, error: { code, message } })

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}
