// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, realpathSync, rmSync } from 'node:fs'
import { homedir } from 'node:os'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import { afterEach, describe, expect, it } from 'vitest'

import { daemonPaths, resolveProjectDir } from '../gatewayClient.js'
import { isResponseFrame, normalizeEventType } from '../gatewayTypes.js'

describe('daemonPaths', () => {
  const prevHome = process.env.XERXES_HOME
  const prevSock = process.env.XERXES_DAEMON_SOCKET

  afterEach(() => {
    if (prevHome === undefined) delete process.env.XERXES_HOME
    else process.env.XERXES_HOME = prevHome
    if (prevSock === undefined) delete process.env.XERXES_DAEMON_SOCKET
    else process.env.XERXES_DAEMON_SOCKET = prevSock
  })

  it('derives the per-project digest socket under $XERXES_HOME/daemon/projects', () => {
    delete process.env.XERXES_DAEMON_SOCKET
    process.env.XERXES_HOME = '/tmp/xh'
    const dir = '/Users/me/proj'
    const digest = createHash('sha256').update(dir, 'utf8').digest('hex').slice(0, 16)
    const { socketPath, pidPath } = daemonPaths(dir)
    expect(socketPath).toBe(`/tmp/xh/daemon/projects/${digest}.sock`)
    expect(pidPath).toBe(`/tmp/xh/daemon/projects/${digest}.pid`)
  })

  it('defaults $XERXES_HOME to ~/.xerxes', () => {
    delete process.env.XERXES_HOME
    delete process.env.XERXES_DAEMON_SOCKET
    const { socketPath } = daemonPaths('/x')
    expect(socketPath.startsWith(join(homedir(), '.xerxes', 'daemon', 'projects'))).toBe(true)
    expect(socketPath.endsWith('.sock')).toBe(true)
  })

  it('honors XERXES_DAEMON_SOCKET override for the socket, keeps digest pid', () => {
    process.env.XERXES_HOME = '/tmp/xh'
    process.env.XERXES_DAEMON_SOCKET = '/run/custom.sock'
    const { socketPath, pidPath } = daemonPaths('/x')
    expect(socketPath).toBe('/run/custom.sock')
    expect(pidPath).toContain('/tmp/xh/daemon/projects/')
  })

  it('produces a 16-hex-char digest', () => {
    delete process.env.XERXES_DAEMON_SOCKET
    process.env.XERXES_HOME = '/tmp/xh'
    const { socketPath } = daemonPaths('/any/where')
    const name = socketPath.split('/').pop() ?? ''
    expect(name).toMatch(/^[0-9a-f]{16}\.sock$/)
  })

  it('canonicalizes subdirectories to the git repository root', () => {
    const root = mkdtempSync(join(tmpdir(), 'xerxes-gateway-'))
    try {
      execFileSync('git', ['init'], { cwd: root, stdio: 'ignore' })
      const child = join(root, 'src', 'ui-tui')
      mkdirSync(child, { recursive: true })
      expect(resolveProjectDir(child)).toBe(realpathSync(root))
    } finally {
      rmSync(root, { force: true, recursive: true })
    }
  })
})

describe('normalizeEventType', () => {
  it('folds PascalCase (bridge) names to snake_case (socket)', () => {
    expect(normalizeEventType('TextPart')).toBe('text_part')
    expect(normalizeEventType('QuestionRequest')).toBe('question_request')
    expect(normalizeEventType('MCPLoadingBegin')).toBe('mcp_loading_begin')
    expect(normalizeEventType('SubagentEvent')).toBe('subagent_event')
  })

  it('passes snake_case through unchanged', () => {
    expect(normalizeEventType('text_part')).toBe('text_part')
    expect(normalizeEventType('init_done')).toBe('init_done')
  })

  it('leaves unknown names untouched', () => {
    expect(normalizeEventType('something_new')).toBe('something_new')
  })
})

describe('isResponseFrame', () => {
  it('treats id-bearing frames as responses', () => {
    expect(isResponseFrame({ id: 1, result: {} })).toBe(true)
    expect(isResponseFrame({ id: 0, result: {} })).toBe(true)
  })

  it('treats id-less event frames as not-responses', () => {
    expect(isResponseFrame({ method: 'event', params: {} })).toBe(false)
    expect(isResponseFrame({})).toBe(false)
  })
})
