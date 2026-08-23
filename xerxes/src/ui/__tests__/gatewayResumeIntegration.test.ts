// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { mkdtemp, realpath, rm } from 'node:fs/promises'
import { connect, type Socket } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { InMemoryDaemonRuntime } from '../../daemon/runtime.js'
import { DaemonServer } from '../../daemon/server.js'
import { DaemonTranscriptStore } from '../../session/daemonTranscript.js'
import { toTranscriptMessages } from '../domain/messages.js'
import { GatewayClient } from '../gatewayClient.js'
import type { SessionResumeResponse } from '../gatewayTypes.js'

describe('GatewayClient resumed transcript integration', () => {
  it('hydrates persisted daemon history once through the real socket protocol', async () => {
    // Canonicalize the temp dir: on macOS /tmp is a symlink to /private/tmp,
    // and the daemon rejects a transcript whose stored project dir differs
    // from the (realpath-resolved) project dir the client connects with.
    const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-gateway-resume-')))
    const sessionDirectory = join(directory, 'sessions')
    const socketPath = join(directory, 'daemon.sock')
    const workspaceRoot = join(directory, 'agents')
    const sessionId = 'c1d2e3f4'
    const store = new DaemonTranscriptStore({
      currentProjectDirectory: directory,
      directory: sessionDirectory,
      workspaceRoot
    })

    await store.save({
      agentId: 'default',
      cwd: directory,
      extra: {},
      format: 'bun-v2',
      interactionMode: 'code',
      key: sessionId,
      messages: [
        { content: 'inspect the resume path', role: 'user' },
        { content: 'The persisted answer is visible.', role: 'assistant' }
      ],
      metadata: {},
      pendingResumeReplays: [],
      planMode: false,
      schemaVersion: undefined,
      sessionId,
      thinkingContent: [],
      toolExecutions: [],
      totalInputTokens: 12,
      totalOutputTokens: 8,
      turnCount: 1,
      updatedAt: '2026-07-16T00:00:00.000Z',
      workspace: join(workspaceRoot, 'default')
    })

    const server = new DaemonServer({
      runtime: new InMemoryDaemonRuntime(undefined, {
        currentProjectDirectory: directory,
        model: 'gpt-4o',
        sessionDirectory,
        workspaceRoot
      }),
      socketPath
    })
    const client = new GatewayClient({ projectDir: directory, sessionKey: 'test:resume-integration' })

    await server.start()

    try {
      const socket = await connectSocket(socketPath)
      const privateClient = client as unknown as { attachSocket: (socket: Socket) => void }
      privateClient.attachSocket(socket)
      const forwarded: string[] = []

      client.on('event', event => {
        if ((event as { type?: string }).type === 'transcript.append') {
          forwarded.push((event as { type: string }).type)
        }
      })

      const resumed = await client.request<SessionResumeResponse>('session.resume', { session_id: sessionId })

      expect(resumed).toMatchObject({ message_count: 2, resumed: sessionId, session_id: sessionId })
      expect(resumed.messages).toEqual([
        { role: 'user', text: 'inspect the resume path' },
        { role: 'assistant', text: 'The persisted answer is visible.' }
      ])
      expect(forwarded).toEqual([])
    } finally {
      client.close()
      await server.stop()
      await rm(directory, { force: true, recursive: true })
    }
  })

  it('hydrates persisted thinking traces and tool rows on resume', async () => {
    const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-gateway-resume-tools-')))
    const sessionDirectory = join(directory, 'sessions')
    const socketPath = join(directory, 'daemon.sock')
    const workspaceRoot = join(directory, 'agents')
    const sessionId = 'd5e6f7a8'
    const store = new DaemonTranscriptStore({
      currentProjectDirectory: directory,
      directory: sessionDirectory,
      workspaceRoot
    })

    await store.save({
      agentId: 'default',
      cwd: directory,
      extra: {},
      format: 'bun-v2',
      interactionMode: 'code',
      key: sessionId,
      messages: [
        { content: 'inspect auth', role: 'user' },
        {
          content: 'Let me read the file.',
          role: 'assistant',
          thinking: 'reasoning trace',
          tool_calls: [
            {
              function: { arguments: { path: 'src/auth.ts' }, name: 'ReadFile' },
              id: 'call_1',
              type: 'function'
            }
          ]
        },
        { content: 'file body', name: 'ReadFile', role: 'tool', tool_call_id: 'call_1' },
        { content: 'The flow starts in auth.ts.', role: 'assistant' }
      ],
      metadata: {},
      pendingResumeReplays: [],
      planMode: false,
      schemaVersion: undefined,
      sessionId,
      thinkingContent: ['reasoning trace'],
      toolExecutions: [
        {
          display_blocks: [],
          duration_ms: 200,
          name: 'ReadFile',
          permitted: true,
          result: 'file body',
          return_value: 'file body',
          tool_call_id: 'call_1'
        }
      ],
      totalInputTokens: 12,
      totalOutputTokens: 8,
      turnCount: 1,
      updatedAt: '2026-07-16T00:00:00.000Z',
      workspace: join(workspaceRoot, 'default')
    })

    const server = new DaemonServer({
      runtime: new InMemoryDaemonRuntime(undefined, {
        currentProjectDirectory: directory,
        model: 'gpt-4o',
        sessionDirectory,
        workspaceRoot
      }),
      socketPath
    })
    const client = new GatewayClient({ projectDir: directory, sessionKey: 'test:resume-tools' })

    await server.start()

    try {
      const socket = await connectSocket(socketPath)
      const privateClient = client as unknown as { attachSocket: (socket: Socket) => void }
      privateClient.attachSocket(socket)

      const resumed = await client.request<SessionResumeResponse>('session.resume', { session_id: sessionId })

      expect(resumed).toMatchObject({ resumed: sessionId, session_id: sessionId })
      expect(resumed.messages).toEqual([
        { role: 'user', text: 'inspect auth' },
        { role: 'assistant', text: 'Let me read the file.', thinking: 'reasoning trace' },
        {
          // Argument blobs are summarized like live rows: path value, not JSON.
          context: 'src/auth.ts',
          duration_s: 0.2,
          name: 'ReadFile',
          role: 'tool'
        },
        { role: 'assistant', text: 'The flow starts in auth.ts.' }
      ])

      const hydrated = toTranscriptMessages(resumed.messages)
      expect(hydrated).toHaveLength(3)
      expect(hydrated[0]).toEqual({ role: 'user', text: 'inspect auth' })
      expect(hydrated[1]).toEqual({
        role: 'assistant',
        text: 'Let me read the file.',
        thinking: 'reasoning trace'
      })
      expect(hydrated[2]).toMatchObject({ role: 'assistant', text: 'The flow starts in auth.ts.' })
      expect(hydrated[2]?.tools).toHaveLength(1)
      expect(hydrated[2]?.tools?.[0]).toContain('Read File("src/auth.ts")')
      expect(hydrated[2]?.tools?.[0]).toMatch(/✓$/)
    } finally {
      client.close()
      await server.stop()
      await rm(directory, { force: true, recursive: true })
    }
  })
})

const connectSocket = (path: string) =>
  new Promise<Socket>((resolve, reject) => {
    const socket = connect({ path })
    socket.once('connect', () => resolve(socket))
    socket.once('error', reject)
  })
