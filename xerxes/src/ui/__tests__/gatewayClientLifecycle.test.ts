// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { GatewayClient } from '../gatewayClient.js'
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
          sessions: [{ active_turn_id: 'turn1', id: 'live1', key: 'test:sessions', messages: 3, title: 'live work' }]
        }
      }
      if (method === 'session.list') {
        expect(params).toEqual({ limit: 200 })
        return {
          ok: true,
          sessions: [
            {
              key: 'old1',
              messages: 2,
              session_id: 'old1',
              title: 'saved work',
              updated_at: '2026-06-27T10:00:00+00:00'
            }
          ]
        }
      }
      throw new Error(`unexpected ${method}`)
    }

    const active = await client.request('session.active_list', { current_session_id: 'live1' })
    const saved = await client.request('session.list', { limit: 200 })

    expect(calls).toEqual(['session.active_list', 'session.list'])
    expect(active).toMatchObject({
      sessions: [{ id: 'live1', message_count: 3, status: 'working', title: 'live work' }]
    })
    expect(saved).toMatchObject({
      sessions: [{ id: 'old1', message_count: 2, source: 'saved', title: 'saved work' }]
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

  it('switches the selected provider profile before applying the picker model', async () => {
    const client = new GatewayClient({ projectDir: process.cwd(), sessionKey: 'test:model-picker' })
    const calls: Array<{ method: string; params: Record<string, unknown> }> = []
    const privateClient = client as unknown as {
      configSet: (params: Record<string, unknown>) => Promise<Record<string, unknown>>
      modelOptions: () => Promise<Record<string, unknown>>
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

      return { ok: true }
    }

    await expect(privateClient.modelOptions()).resolves.toMatchObject({ model: 'kimi-for-coding' })
    await expect(
      privateClient.configSet({ key: 'model', value: 'gpt-4.1 --provider openai-dev --tui-session' })
    ).resolves.toEqual({ value: 'gpt-4.1' })

    expect(calls.slice(-2)).toEqual([
      { method: 'provider_select', params: { name: 'openai-dev' } },
      { method: 'runtime.reload', params: { model: 'gpt-4.1' } }
    ])
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
      { method: 'session.most_recent', params: {} }
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
      text: 'review expanded paste @context.md'
    })

    expect(calls).toEqual([
      {
        method: 'turn.submit',
        params: {
          display_text: 'review [Pasted 20 lines] @context.md',
          session_key: 'live-attachment',
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

    privateClient.rawRequest = async (method, params) => {
      expect(method).toBe('initialize')
      expect(params).toMatchObject({ resume_session_id: 'aabbccdd', session_key: 'aabbccdd' })

      for (const payload of [
        { body: '✨ inspect the auth flow', category: 'history', type: 'replay_user' },
        { body: 'The flow starts in auth.ts.', category: 'history', type: 'replay_assistant' }
      ]) {
        privateClient.onLine(
          JSON.stringify({ jsonrpc: '2.0', method: 'event', params: { payload, type: 'notification' } })
        )
      }

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
    expect(forwarded).toEqual([])
  })
})
