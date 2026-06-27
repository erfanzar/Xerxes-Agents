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

const initGitProject = () => {
  const dir = mkdtempSync(join(tmpdir(), 'xerxes-tui-project-'))
  writeFileSync(join(dir, 'pyproject.toml'), ['[project]', 'name = "xerxes-agent"', 'version = "9.9.9"', ''].join('\n'))
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
})
