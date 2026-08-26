// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { execFileSync } from 'node:child_process'
import { EventEmitter } from 'node:events'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import type { Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it, vi } from 'vitest'

import { GatewayClient, MAX_GATEWAY_FRAME_BYTES, resolveProjectDir } from '../gatewayClient.js'
import type { SessionActiveListResponse } from '../gatewayTypes.js'
import type { SessionInfo } from '../types.js'

interface SessionCreateResult {
  info: SessionInfo
  session_id: string
}

interface SessionResumeResult extends SessionCreateResult {
  message_count: number
  messages: Array<{ role: string; text?: string }>
  resumed: string
}

const initGitProject = () => {
  const dir = mkdtempSync(join(tmpdir(), 'xerxes-tui-project-'))
  writeFileSync(join(dir, 'package.json'), JSON.stringify({ name: 'xerxes-agent', version: '9.9.9' }, null, 2))
  writeFileSync(join(dir, 'README.md'), '# test\n')
  execFileSync('git', ['init'], { cwd: dir, stdio: 'ignore' })
  execFileSync('git', ['add', '.'], { cwd: dir, stdio: 'ignore' })
  execFileSync('git', ['-c', 'user.email=test@example.com', '-c', 'user.name=Xerxes Test', 'commit', '-m', 'init'], {
    cwd: dir,
    stdio: 'ignore'
  })

  return dir
}

/** Minimal Socket stand-in: an EventEmitter with the surface attachSocket/close touch. */
const fakeSocket = () => {
  const emitter = new EventEmitter()

  return Object.assign(emitter, {
    destroy: vi.fn(),
    end: vi.fn(),
    setEncoding: vi.fn(),
    write: vi.fn((_frame: string, cb?: (error?: Error | null) => void) => {
      cb?.(null)

      return true
    })
  }) as unknown as Socket
}

const attachFakeSocket = (client: GatewayClient, socket: Socket) => {
  ;(client as unknown as { attachSocket: (sock: Socket) => void }).attachSocket(socket)
}

describe('GatewayClient session lifecycle', () => {
  it('keeps the retired config mtime poll stable when the native daemon has no revision source', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:config-mtime' })

    await expect(client.request('config.get', { key: 'mtime' })).resolves.toEqual({ mtime: 0 })
    await expect(client.request('config.get', { key: 'mtime' })).resolves.toEqual({ mtime: 0 })
  })

  it('lifts daemon session ids onto every adapted subagent event', () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:subagent-session-routing' })
    const events: Array<{ session_id?: string; type: string }> = []
    const privateClient = client as unknown as { onLine: (line: string) => void }

    client.on('event', event => events.push(event as { session_id?: string; type: string }))
    privateClient.onLine(
      JSON.stringify({
        jsonrpc: '2.0',
        method: 'event',
        params: {
          payload: {
            agent_id: 'child-1',
            event: { payload: {}, type: 'TurnBegin' },
            goal: 'inspect runtime',
            session_id: 'session-a',
            task_index: 0
          },
          type: 'subagent_event'
        }
      })
    )

    expect(events).toHaveLength(1)
    expect(events[0]).toMatchObject({ session_id: 'session-a', type: 'subagent.start' })
  })

  it('rejects valid JSON that is not a protocol frame without throwing from the socket handler', () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:invalid-frame' })
    const protocolErrors: unknown[] = []
    const privateClient = client as unknown as { onLine: (line: string) => void }

    client.on('gateway.protocol_error', event => protocolErrors.push(event))

    expect(() => privateClient.onLine('null')).not.toThrow()
    expect(() => privateClient.onLine('[]')).not.toThrow()
    expect(() =>
      privateClient.onLine(
        JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { payload: [], type: 'text_part' } })
      )
    ).not.toThrow()
    expect(protocolErrors).toHaveLength(3)
  })

  it('normalizes tolerated PascalCase bridge aliases before emitting UI events', () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:event-alias' })
    const events: unknown[] = []
    const privateClient = client as unknown as { onLine: (line: string) => void }

    client.on('message.delta', event => events.push(event))
    privateClient.onLine(
      JSON.stringify({
        jsonrpc: '2.0',
        method: 'event',
        params: { payload: { text: 'hello' }, type: 'TextPart' }
      })
    )

    expect(events).toEqual([{ payload: { text: 'hello' }, type: 'message.delta' }])
  })

  it('routes active and saved session lists to distinct daemon RPCs', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:sessions' })
    const calls: string[] = []
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async (method, params) => {
      calls.push(method)
      if (method === 'session.active_list') {
        expect(params).toEqual({ current_session_id: 'live1' })
        return {
          ok: true,
          sessions: [{
            active_turn_id: 'turn1',
            id: 'live1',
            inflight: { assistant: 'working', streaming: true, user: 'audit session routing' },
            key: 'test:sessions',
            last_active: 1_751_018_100,
            messages: 3,
            title: 'live work'
          }]
        }
      }
      if (method === 'session.list') {
        expect(params).toEqual({ include_subagents: true, kind: 'all' })
        return {
          ok: true,
          sessions: [
            {
              key: 'old1',
              messages: 2,
              session_id: 'old1',
              title: 'saved work',
              updated_at: '2026-06-27T10:00:00+00:00'
            },
            {
              agent_id: 'researcher',
              kind: 'subagent',
              messages: 7,
              model: 'provider/research-model',
              parent_session_id: 'old1',
              root_session_id: 'old1',
              session_id: 'agent1',
              status: 'completed',
              subagent_id: 'subagent_policy',
              title: 'Policy review',
              updated_at: '2026-06-27T10:05:00+00:00'
            }
          ]
        }
      }
      throw new Error(`unexpected ${method}`)
    }

    const active = await client.request('session.active_list', { current_session_id: 'live1' })
    const saved = await client.request('session.list', { include_subagents: true, kind: 'all' })

    expect(calls).toEqual(['session.active_list', 'session.list'])
    expect(active).toMatchObject({
      sessions: [{
        activity: 'audit session routing',
        id: 'live1',
        last_active: 1_751_018_100,
        message_count: 3,
        status: 'working',
        title: 'live work'
      }]
    })
    expect(saved).toMatchObject({
      sessions: [
        { id: 'old1', message_count: 2, source: 'saved', title: 'saved work' },
        {
          agent_id: 'researcher',
          id: 'agent1',
          kind: 'subagent',
          message_count: 7,
          model: 'provider/research-model',
          parent_session_id: 'old1',
          root_session_id: 'old1',
          source: 'saved',
          status: 'completed',
          subagent_id: 'subagent_policy',
          title: 'Policy review'
        }
      ]
    })
  })

  it('attaches the daemon connection when activating a tab so following slash commands use that session', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:activate' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })
      if (method === 'session.active_list') {
        return {
          ok: true,
          sessions: [{ id: 'tab-b', key: 'daemon:key-b', messages: 0, status: 'idle', title: 'Tab B' }]
        }
      }
      if (method === 'session.status') {
        return {
          ok: true,
          session: { cwd: '/worktrees/tab-b', id: 'tab-b', key: 'daemon:key-b', status: 'idle' }
        }
      }
      if (method === 'session.open') {
        return {
          ok: true,
          session: { id: 'tab-b', key: 'daemon:key-b', messages: 0, status: 'idle', transcript: [] }
        }
      }
      if (method === 'slash') {
        return { ok: true, output: 'renamed active tab' }
      }
      throw new Error(`unexpected ${method}`)
    }

    await client.request('session.active_list', {})
    await client.request('session.activate', { session_id: 'tab-b' })
    await client.request('slash.exec', { command: '/title Active tab' })

    expect(calls.slice(1)).toEqual([
      { method: 'session.status', params: { session_key: 'daemon:key-b' } },
      {
        method: 'session.open',
        params: { project_dir: '/worktrees/tab-b', session_key: 'daemon:key-b' }
      },
      { method: 'slash', params: { command: '/title Active tab' } }
    ])
  })

  it('preserves starting and waiting statuses even before an active turn id exists', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:live-status' })
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async () => ({
      ok: true,
      sessions: [
        { id: 'starting', key: 'key-starting', messages: 0, status: 'starting' },
        { id: 'waiting', key: 'key-waiting', messages: 1, status: 'waiting' }
      ]
    })

    const result = await client.request<SessionActiveListResponse>('session.active_list', {})

    expect(result.sessions?.map(session => session.status)).toEqual(['starting', 'waiting'])
  })

  it('peeks at a live session without attaching the daemon connection', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:peek' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })
      if (method === 'session.active_list') {
        return { ok: true, sessions: [{ id: 'other', key: 'daemon:other', messages: 1, status: 'working' }] }
      }
      if (method === 'session.status') {
        return {
          ok: true,
          session: {
            id: 'other',
            inflight: { assistant: 'half done', streaming: true, user: 'keep going' },
            key: 'daemon:other',
            status: 'working',
            transcript: [{ role: 'user', content: 'inspect this' }]
          }
        }
      }
      throw new Error(`unexpected ${method}`)
    }

    await client.request('session.active_list', {})
    const result = await client.request('session.peek', { session_id: 'other' })

    expect(result).toMatchObject({
      inflight: { assistant: 'half done', streaming: true, user: 'keep going' },
      messages: [{ role: 'user', text: 'inspect this' }],
      session_id: 'other',
      status: 'working'
    })
    expect(calls).toEqual([
      { method: 'session.active_list', params: {} },
      { method: 'session.status', params: { session_key: 'daemon:other' } }
    ])
    expect(calls.some(call => call.method === 'session.open')).toBe(false)
  })

  it('never fabricates a title from prompt, key, or id', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:untitled' })
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async method => method === 'session.active_list'
      ? { ok: true, sessions: [{ id: 'opaque-live-id', key: 'opaque-live-key', messages: 0, title: '   ' }] }
      : {
          ok: true,
          sessions: [{ key: 'opaque-saved-key', messages: 1, session_id: 'opaque-saved-id', title: '' }]
        }

    // An unnamed session stays blank all the way to the renderer, which is
    // what lets the header show a bare mode label and the picker show its own
    // "not yet named" placeholder. Inventing one here hid both.
    await expect(client.request('session.active_list', {})).resolves.toMatchObject({
      sessions: [{ preview: '', title: '' }]
    })
    await expect(client.request('session.list', {})).resolves.toMatchObject({
      sessions: [{ preview: '', title: '' }]
    })
  })

  it('preserves once, session, permanent, and deny approval scopes for the addressed request', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:approval-scopes' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const privateClient = client as unknown as {
      approvalRespond: (params: Record<string, unknown>) => Promise<unknown>
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })

      return { ok: true }
    }

    for (const choice of ['once', 'session', 'always', 'deny']) {
      await privateClient.approvalRespond({ choice, request_id: 'permission-42' })
    }

    expect(calls).toEqual([
      { method: 'permission_response', params: { request_id: 'permission-42', response: 'approve' } },
      { method: 'permission_response', params: { request_id: 'permission-42', response: 'approve_for_session' } },
      { method: 'permission_response', params: { request_id: 'permission-42', response: 'always' } },
      { method: 'permission_response', params: { request_id: 'permission-42', response: 'reject' } }
    ])
  })

  it('uses the live runtime profile identity and discovers profiles without fabricating catalog entries', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:model-picker' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const privateClient = client as unknown as {
      configSet: (params: Record<string, unknown>) => Promise<Record<string, unknown>>
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })

      if (method === 'provider_list') {
        return {
          ok: true,
          profiles: [
            { active: false, model: 'gpt-4.1', name: 'openai-dev', provider: 'openai' },
            { active: true, model: 'kimi-for-coding', name: 'kimi-local', provider: 'kimi-code' }
          ]
        }
      }

      if (method === 'session.status') {
        return { ok: true, session: { model: 'k3', profile_name: 'openai-dev' } }
      }

      if (method === 'fetch_models') {
        return { ok: true, models: ['k3', 'kimi-k2.5', 'k3', ''], source: 'remote' }
      }

      return { ok: true }
    }

    const options = await client.request<Record<string, unknown>>('model.options', { session_id: 'live-session' })
    expect(options).toEqual({
      model: 'k3',
      provider: 'openai-dev',
      providers: [
        {
          configured_model: 'gpt-4.1',
          is_current: true,
          name: 'openai-dev',
          provider_type: 'openai',
          slug: 'openai-dev'
        },
        {
          configured_model: 'kimi-for-coding',
          is_current: false,
          name: 'kimi-local',
          provider_type: 'kimi-code',
          slug: 'kimi-local'
        }
      ]
    })
    expect(options).not.toHaveProperty('providers.0.models')
    expect(options).not.toHaveProperty('providers.0.total_models')
    expect(options).not.toHaveProperty('providers.0.authenticated')
    await expect(client.request('model.models', { profile_name: 'kimi-local' })).resolves.toEqual({
      models: ['k3', 'kimi-k2.5'],
      source: 'remote'
    })
    await expect(
      privateClient.configSet({ key: 'model', value: 'gpt-4.1 --provider openai-dev --tui-session' })
    ).resolves.toEqual({ value: 'gpt-4.1' })

    expect(calls.slice(0, 3)).toEqual([
      { method: 'provider_list', params: {} },
      { method: 'session.status', params: { session_key: 'live-session' } },
      { method: 'fetch_models', params: { profile_name: 'kimi-local' } }
    ])
    expect(calls.slice(-2)).toEqual([
      { method: 'provider_select', params: { name: 'openai-dev' } },
      { method: 'runtime.reload', params: { model: 'gpt-4.1' } }
    ])
  })

  it('does not mark a stored profile current when the live runtime matches none', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:model-picker-unmatched' })
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async method => {
      if (method === 'provider_list') {
        return {
          ok: true,
          profiles: [
            { active: false, model: 'gpt-4.1', name: 'openai-dev', provider: 'openai' },
            { active: true, model: 'kimi-for-coding', name: 'kimi-local', provider: 'kimi-code' }
          ]
        }
      }
      if (method === 'session.status') {
        return { ok: true, session: { model: 'runtime-override', profile_name: null } }
      }
      return { ok: true }
    }

    await expect(client.request('model.options', {})).resolves.toEqual({
      model: 'runtime-override',
      provider: '',
      providers: [
        {
          configured_model: 'gpt-4.1',
          is_current: false,
          name: 'openai-dev',
          provider_type: 'openai',
          slug: 'openai-dev'
        },
        {
          configured_model: 'kimi-for-coding',
          is_current: false,
          name: 'kimi-local',
          provider_type: 'kimi-code',
          slug: 'kimi-local'
        }
      ]
    })
  })

  it('returns normalized reasoning info after config.set reasoning', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:reasoning' })
    const privateClient = client as unknown as {
      configSet: (params: Record<string, unknown>) => Promise<Record<string, unknown>>
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async (method, params) => {
      expect(method).toBe('runtime.reload')
      expect(params).toEqual({ reasoning_effort: 'high' })
      return { ok: true, reasoning_effort: 'high' }
    }

    const result = await privateClient.configSet({ key: 'reasoning', value: 'high' })

    expect(result).toEqual({ info: { reasoning_effort: 'high' }, value: 'high' })
  })

  it('returns the daemon-canonical mode and rejects failed config.set mode requests', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:mode' })
    const privateClient = client as unknown as {
      configSet: (params: Record<string, unknown>) => Promise<Record<string, unknown>>
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async (method, params) => {
      expect(method).toBe('set_mode')
      expect(params).toEqual({ mode: 'planner', session_key: 'session-1' })
      return { ok: true, mode: 'plan', plan_mode: true }
    }
    await expect(privateClient.configSet({ key: 'mode', session_id: 'session-1', value: 'planner' }))
      .resolves.toEqual({ info: { mode: 'plan', plan_mode: true }, value: 'plan' })

    privateClient.rawRequest = async () => ({ ok: false, error: 'no active session' })
    await expect(privateClient.configSet({ key: 'mode', session_id: 'session-1', value: 'code' }))
      .rejects.toThrow('no active session')
  })

  it('uses real native session RPCs instead of fabricating title, compact, save, undo, or recent-session results', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:session-rpcs' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })

      switch (method) {
        case 'session.title':
          return { ok: true, title: 'Native session' }
        case 'session.delete':
          return { deleted: true, ok: true, session_id: 'saved-1' }
        case 'session.compress':
          return { compacted: true, ok: true, tokens_after: 80, tokens_before: 160 }
        case 'session.save':
          return { ok: true, session: { path: '/tmp/session.json' } }
        case 'session.undo':
          return { dropped: 2, ok: true }
        case 'session.most_recent':
          return { ok: true, session: { id: 'aabbccdd', title: 'Recent native session' } }
        default:
          throw new Error(`unexpected ${method}`)
      }
    }

    await expect(client.request('session.title', { session_id: 'live-1', title: 'Native session' })).resolves.toEqual({
      title: 'Native session'
    })
    await expect(client.request('session.delete', { session_id: 'saved-1' })).resolves.toEqual({ deleted: 'saved-1' })
    await expect(client.request('session.compress', { session_id: 'live-1' })).resolves.toEqual({
      after_tokens: 80,
      before_tokens: 160,
      summary: { headline: 'context compacted', noop: false, token_line: '160 → 80 tokens' }
    })
    await expect(client.request('session.save', { session_id: 'live-1' })).resolves.toEqual({
      file: '/tmp/session.json'
    })
    await expect(client.request('session.undo', { session_id: 'live-1' })).resolves.toEqual({ removed: 2 })
    await expect(client.request('session.most_recent')).resolves.toEqual({
      session_id: 'aabbccdd',
      source: 'saved',
      title: 'Recent native session'
    })

    expect(calls).toEqual([
      { method: 'session.title', params: { session_key: 'live-1', title: 'Native session' } },
      { method: 'session.delete', params: { session_id: 'saved-1' } },
      { method: 'session.compress', params: { session_key: 'live-1' } },
      { method: 'session.save', params: { session_key: 'live-1' } },
      { method: 'session.undo', params: { session_key: 'live-1' } },
      { method: 'session.most_recent', params: { project_dir: resolveProjectDir(process.cwd()) } }
    ])
  })

  it('rejects a native application-level session failure instead of returning a false success result', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:session-rpc-failure' })
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async () => ({ error: 'turn is running', ok: false })

    await expect(client.request('session.undo', { session_id: 'live-1' })).rejects.toThrow('turn is running')
  })

  it.each([
    ['prompt.submit', { session_id: 'live-1', text: 'hello' }, 'turn rejected'],
    ['prompt.background', { session_id: 'live-1', text: 'hello' }, 'background rejected'],
    ['session.interrupt', { session_id: 'live-1' }, 'no active turn']
  ])('rejects native %s soft failures so the TUI can leave its optimistic state', async (method, params, error) => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:turn-rpc-failure' })
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async () => ({ error, ok: false })

    await expect(client.request(method, params)).rejects.toThrow(error)
  })

  it('forwards authored and provider-facing prompt text separately', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:attachment-submit' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }
    privateClient.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })
      return { ok: true }
    }

    await client.request('prompt.submit', {
      display_text: 'review [Pasted 20 lines] @context.md',
      session_id: 'live-attachment',
      submission_id: 'submission-1',
      text: 'review expanded paste @context.md'
    })

    expect(calls).toEqual([
      {
        method: 'turn.submit',
        params: {
          display_text: 'review [Pasted 20 lines] @context.md',
          session_key: 'live-attachment',
          submission_id: 'submission-1',
          text: 'review expanded paste @context.md'
        }
      }
    ])
  })

  it('routes browser management to the real native CDP daemon endpoint', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:browser-rpc' })
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async (method, params = {}) => {
      expect(method).toBe('browser.manage')
      expect(params).toEqual({ action: 'connect', url: 'http://127.0.0.1:9222' })
      return {
        ok: true,
        pages: [{ ref_id: 'page-1', title: 'Xerxes', url: 'https://example.test/' }],
        status: { connected: true, endpoint: 'http://127.0.0.1:9222', kind: 'cdp' }
      }
    }

    await expect(
      client.request('browser.manage', {
        action: 'connect',
        session_id: 'ignored-by-native-browser-manager',
        url: 'http://127.0.0.1:9222'
      })
    ).resolves.toMatchObject({
      connected: true,
      kind: 'cdp',
      pages: [{ ref_id: 'page-1' }]
    })
  })

  it('preserves initialize event metadata when the raw response is stale', async () => {
    const projectDir = initGitProject()

    try {
      const head = execFileSync('git', ['-C', projectDir, 'rev-parse', '--short=12', 'HEAD'], {
        encoding: 'utf8'
      }).trim()
      const client = new GatewayClient({ projectDir, sessionKey: 'test:session' })
      const privateClient = client as unknown as {
        rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
      }

      privateClient.rawRequest = async method => {
        expect(method).toBe('initialize')
        client.emit('session.info', {
          payload: {
            cwd: projectDir,
            model: 'claude-code/opus',
            skillDescriptions: { deepscan: 'deep scan' },
            skills: { skills: ['deepscan', 'eternal-army'] },
            tools: { tools: ['ReadFile'] },
            usage: { calls: 0, context_max: 1_000_000, context_used: 0, input: 0, output: 0, total: 0 }
          },
          type: 'session.info'
        })
        client.emit('status.update', {
          payload: { usage: { calls: 0, context_max: 1_000_000, context_used: 123, input: 0, output: 0, total: 0 } },
          type: 'status.update'
        })

        return {
          model: 'claude-code/opus',
          ok: true,
          session: {
            cwd: projectDir,
            id: 's1',
            messages: 0,
            mode: 'code',
            model: 'claude-code/opus'
          },
          skills: 2,
          tools: 1
        }
      }

      const result = (await client.request('session.create', {})) as SessionCreateResult

      expect(result.session_id).toBe('s1')
      expect(result.info.cwd).toBe(projectDir)
      expect(result.info.version).toBe('9.9.9')
      expect(result.info.head_hash).toBe(head)
      expect(result.info.skills).toEqual({ skills: ['deepscan', 'eternal-army'] })
      expect(result.info.skillDescriptions).toEqual({ deepscan: 'deep scan' })
      expect(result.info.usage?.context_max).toBe(1_000_000)
      expect(result.info.usage?.context_used).toBe(123)
    } finally {
      rmSync(projectDir, { force: true, recursive: true })
    }
  })

  it('batches initialize replay events into the resumed transcript when the daemon returns only a count', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:resume' })
    const forwarded: string[] = []
    const privateClient = client as unknown as {
      onLine: (line: string) => void
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    client.on('event', event => {
      if ((event as { type?: string }).type === 'transcript.append') {
        forwarded.push((event as { type: string }).type)
      }
    })

    let capturedKey = ''
    privateClient.rawRequest = async (method, params) => {
      expect(method).toBe('initialize')
      // Resume must not alias the connection onto the raw session id: another
      // connection resuming the same session would derive the identical key.
      expect(params).toMatchObject({ resume_session_id: 'aabbccdd' })
      capturedKey = String(params?.session_key ?? '')
      expect(capturedKey).toMatch(/^tui:/)
      expect(capturedKey).not.toBe('aabbccdd')

      for (const payload of [
        { body: '✨ inspect the auth flow', category: 'history', type: 'replay_user' },
        { body: 'The flow starts in auth.ts.', category: 'history', type: 'replay_assistant' }
      ]) {
        privateClient.onLine(
          JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { payload, type: 'notification' } })
        )
      }

      // A replay-shaped row tagged for a *different* live session belongs to
      // that session's stream; the resume capture must not swallow it.
      privateClient.onLine(
        JSON.stringify({
          jsonrpc: '2.0',
          method: 'event',
          params: {
            payload: {
              body: 'other session row',
              category: 'history',
              session_id: 'ff001122',
              type: 'replay_assistant'
            },
            type: 'notification'
          }
        })
      )

      return {
        cwd: process.cwd(),
        model: 'kimi-for-coding',
        ok: true,
        session: {
          cwd: process.cwd(),
          id: 'aabbccdd',
          message_count: 2,
          messages: 2,
          mode: 'code',
          model: 'kimi-for-coding'
        }
      }
    }

    const result = (await client.request('session.resume', { session_id: 'aabbccdd' })) as SessionResumeResult

    expect(result).toMatchObject({ message_count: 2, resumed: 'aabbccdd', session_id: 'aabbccdd' })
    expect(result.messages).toEqual([
      { role: 'user', text: 'inspect the auth flow' },
      { role: 'assistant', text: 'The flow starts in auth.ts.' }
    ])
    // Only the foreign-session row is forwarded; the resumed session's own
    // replay is batched into the hydrate-once transcript.
    expect(forwarded).toEqual(['transcript.append'])

    // The minted key is remembered, so follow-up calls for the session route
    // through it instead of the raw session id.
    const routed: string[] = []
    privateClient.rawRequest = async (method, params) => {
      routed.push(`${method}:${String(params?.session_key ?? '')}`)
      return { ok: true }
    }
    await client.request('session.interrupt', { session_id: 'aabbccdd' })
    expect(routed).toEqual([`cancel:${capturedKey}`])
  })

  it('emits gateway.closed on an unexpected socket death but never from a deliberate kill', () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:closed-event' })
    const closedEvents: unknown[] = []
    const exitEvents: unknown[] = []

    client.on('gateway.closed', event => closedEvents.push(event))
    client.on('exit', event => exitEvents.push(event))

    // Unexpected death: the daemon end of a live socket disappears.
    const first = fakeSocket()
    attachFakeSocket(client, first)
    first.emit('close')

    expect(closedEvents).toHaveLength(1)

    // Deliberate shutdown: kill() marks the client closed first, so the same
    // socket teardown must not look like a crash to recovery subscribers.
    const second = fakeSocket()
    attachFakeSocket(client, second)
    client.kill('test')
    second.emit('close')

    expect(closedEvents).toHaveLength(1)
    expect(exitEvents).toHaveLength(1)
  })

  it('rejects pending requests immediately on close instead of hanging until the RPC timeout', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:close-rejects' })
    attachFakeSocket(client, fakeSocket())

    const pending = client.request('session.title', {})
    const pendingMap = client as unknown as { pending: Map<number, unknown> }

    expect(pendingMap.pending.size).toBe(1)

    client.close()

    await expect(pending).rejects.toThrow('gateway closed')
    expect(pendingMap.pending.size).toBe(0)
  })

  it('drops a partial frame buffered from a dead socket when attaching its replacement', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:buffer-reset' })
    const protocolErrors: unknown[] = []

    client.on('gateway.protocol_error', event => protocolErrors.push(event))

    const first = fakeSocket()
    attachFakeSocket(client, first)
    first.emit('data', '{"jsonrpc":"2.0","id":1,"resu')

    const second = fakeSocket()
    attachFakeSocket(client, second)

    const response = client.request('session.title', {})
    second.emit('data', '{"jsonrpc":"2.0","id":1,"result":{"title":"clean"}}\n')

    await expect(response).resolves.toEqual({ title: 'clean' })
    expect(protocolErrors).toHaveLength(0)
  })

  it('closes with a protocol error when an unterminated frame exceeds the daemon frame cap', () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:frame-cap' })
    const socket = fakeSocket()
    const protocolErrors: Array<{ payload?: { message?: string }; type?: string }> = []

    client.on('gateway.protocol_error', event =>
      protocolErrors.push(event as { payload?: { message?: string }; type?: string })
    )
    attachFakeSocket(client, socket)
    socket.emit('data', 'x'.repeat(MAX_GATEWAY_FRAME_BYTES + 1))

    expect(protocolErrors).toEqual([
      {
        payload: { message: `gateway frame exceeds maximum size of ${MAX_GATEWAY_FRAME_BYTES} bytes` },
        type: 'gateway.protocol_error'
      }
    ])
    expect(socket.destroy).toHaveBeenCalledOnce()
    expect((client as unknown as { buffer: string }).buffer).toBe('')
  })

  it('commits the session key only after a successful initialize', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:key-commit' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }
    let initializeCalls = 0

    privateClient.rawRequest = async (method, params = {}) => {
      calls.push({ method, params })

      if (method === 'initialize') {
        initializeCalls += 1

        if (initializeCalls === 1) {
          return { error: 'provider missing', ok: false }
        }

        return { ok: true, session: { id: `s${initializeCalls}` } }
      }

      return { ok: true }
    }

    // A failed initialize keeps the previous session key.
    await expect(client.request('session.create', {})).rejects.toThrow('provider missing')
    await client.request('session.title', {})
    expect(calls.at(-1)).toEqual({ method: 'session.title', params: { session_key: 'test:key-commit' } })

    // A successful initialize commits the freshly minted key.
    await client.request('session.create', {})
    const createdKey = String(calls.filter(call => call.method === 'initialize').at(-1)!.params.session_key)

    expect(createdKey).not.toBe('test:key-commit')

    await client.request('session.title', {})
    expect(calls.at(-1)).toEqual({ method: 'session.title', params: { session_key: createdKey } })
  })

  it('omits fabricated timestamps from active session rows', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:active-timestamps' })
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
    }

    privateClient.rawRequest = async method => {
      expect(method).toBe('session.active_list')

      return { ok: true, sessions: [{ id: 'live1', key: 'test:active-timestamps', messages: 2, title: 'live work' }] }
    }

    const result = await client.request<SessionActiveListResponse>('session.active_list', {})

    expect(result.sessions?.[0]).toMatchObject({ id: 'live1', message_count: 2, status: 'idle', title: 'live work' })
    expect(result.sessions?.[0]).not.toHaveProperty('last_active')
    expect(result.sessions?.[0]).not.toHaveProperty('started_at')
  })

  it('bounds the session key map like a simple LRU', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:key-lru' })
    const titleKeys: string[] = []
    const privateClient = client as unknown as {
      rawRequest: (method: string, params?: Record<string, unknown>) => Promise<Record<string, unknown>>
      rememberSessionKey: (id: string, key: string) => void
    }

    privateClient.rawRequest = async (method, params = {}) => {
      if (method === 'session.title') {
        titleKeys.push(String(params.session_key))
      }

      return { ok: true }
    }

    for (let index = 0; index < 205; index += 1) {
      privateClient.rememberSessionKey(`sid-${index}`, `key-${index}`)
    }

    // The map holds at most 200 entries: sid-0 … sid-4 are evicted, sid-5 on
    // is retained, and the newest entry always resolves to its daemon key.
    await client.request('session.title', { session_id: 'sid-4' })
    await client.request('session.title', { session_id: 'sid-5' })
    await client.request('session.title', { session_id: 'sid-204' })

    expect(titleKeys).toEqual(['sid-4', 'key-5', 'key-204'])
  })
})

describe('stale daemon rejection message', () => {
  // The remedy has to be executable, and it has to be the RIGHT remedy. The
  // old text ended every rejection with "restart it explicitly when idle" —
  // no command, and the wrong advice for a daemon that was already idle and
  // was only refused because its provenance could not be proven.
  const reject = async (opts: { busy: boolean; pid?: number }): Promise<string> => {
    const client = new GatewayClient({
      expectedDaemonBuildId: 'expected-build',
      projectDir: process.cwd(),
      sessionKey: 'test:stale-daemon'
    })
    const priv = client as unknown as {
      detachSocketSilently: () => Promise<void>
      ensureConnectedDaemonCurrent: (socket: string, pid: string) => Promise<boolean>
      probeDaemonIdentity: () => Promise<Record<string, unknown>>
      rawRequest: (method: string) => Promise<unknown>
    }

    priv.detachSocketSilently = async () => undefined
    priv.probeDaemonIdentity = async () => ({
      active_subagents: opts.busy ? 1 : 0,
      daemon_build_id: 'running-build',
      daemon_protocol: 1,
      pid: opts.pid,
      runtime: 'bun'
    })
    // Session activity comes from a separate RPC, and an unanswered one is
    // deliberately treated as busy. Stub it, or every case looks busy and the
    // branches stop being distinguishable.
    priv.rawRequest = async () => ({
      sessions: opts.busy ? [{ active_turn_id: 'turn-1', status: 'working' }] : []
    })

    try {
      await priv.ensureConnectedDaemonCurrent('/tmp/x.sock', '/tmp/x.pid')
    } catch (error) {
      return error instanceof Error ? error.message : String(error)
    }

    return ''
  }

  it('names the exact command when the daemon is idle but not ours', async () => {
    const message = await reject({ busy: false, pid: 4242 })

    expect(message).toContain('build mismatch (running running-build, expected expected-build)')
    expect(message).toContain('not started by this Xerxes install')
    expect(message).toContain('kill 4242')
    // It is already idle; waiting for idleness was never the remedy.
    expect(message).not.toContain('idle')
  })

  it('says to wait when something is genuinely still working', async () => {
    const message = await reject({ busy: true, pid: 4242 })

    expect(message).toContain('still working')
    expect(message).toContain('goes idle')
    expect(message).toContain('kill 4242')
  })

  it('points at the pid file when it has no pid to name', async () => {
    const message = await reject({ busy: false })

    expect(message).toContain('/tmp/x.pid')
    expect(message).not.toContain('kill undefined')
  })
})
