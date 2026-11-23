// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, it } from 'bun:test'
import { existsSync, rmSync } from 'node:fs'
import { mkdtempSafe, sanitizeDesktopInheritedEnv, sanitizeTmpdirEnv } from '../src/desktop/main/safeTmp.js'

describe('safeTmp', () => {
  it('strips missing TMPDIR/TEMP/TMP paths while keeping existing ones', () => {
    const env: NodeJS.ProcessEnv = {
      TMPDIR: '/nonexistent/xerxes-ephemeral-tmp-dir-12345',
      TEMP: '/tmp',
      TMP: '/nonexistent/xerxes-ephemeral-tmp-dir-67890',
    }
    sanitizeTmpdirEnv(env)
    expect(env.TMPDIR).toBeUndefined()
    expect(env.TMP).toBeUndefined()
    expect(env.TEMP).toBe('/tmp')
  })

  it('strips inherited SSH session and outer Antigravity sandbox vars while preserving SSH_AUTH_SOCK', () => {
    const env: NodeJS.ProcessEnv = {
      SSH_CONNECTION: '127.0.0.1 12345 127.0.0.1 22',
      SSH_CLIENT: '127.0.0.1 12345 22',
      SSH_TTY: '/dev/ttys001',
      SSH_AUTH_SOCK: '/private/tmp/com.apple.launchd/Listeners',
      AGY_ACP_FORCE_FILE_STORAGE: '1',
      ANTIGRAVITY_APP_DATA_DIR: '/tmp/outer-host/antigravity-acp',
      ANTIGRAVITY_AGENT: '1',
      ANTIGRAVITY_CSRF_TOKEN: 'secret-token',
      GEMINI_HOME: '/tmp/outer-host',
      HOME: '/Users/example',
    }
    sanitizeDesktopInheritedEnv(env)
    expect(env.SSH_CONNECTION).toBeUndefined()
    expect(env.SSH_CLIENT).toBeUndefined()
    expect(env.SSH_TTY).toBeUndefined()
    expect(env.AGY_ACP_FORCE_FILE_STORAGE).toBeUndefined()
    expect(env.ANTIGRAVITY_APP_DATA_DIR).toBeUndefined()
    expect(env.ANTIGRAVITY_AGENT).toBeUndefined()
    expect(env.ANTIGRAVITY_CSRF_TOKEN).toBeUndefined()
    expect(env.GEMINI_HOME).toBeUndefined()
    expect(env.SSH_AUTH_SOCK).toBe('/private/tmp/com.apple.launchd/Listeners')
    expect(env.HOME).toBe('/Users/example')
  })

  it('preserves user-configured GEMINI_HOME when no outer Antigravity host markers are present', () => {
    const env: NodeJS.ProcessEnv = {
      SSH_CONNECTION: '127.0.0.1 12345 127.0.0.1 22',
      GEMINI_HOME: '/Users/example/custom-gemini',
    }
    sanitizeDesktopInheritedEnv(env)
    expect(env.SSH_CONNECTION).toBeUndefined()
    expect(env.GEMINI_HOME).toBe('/Users/example/custom-gemini')
  })

  it('falls back to /tmp when process.env.TMPDIR points to a deleted directory', async () => {
    const prev = process.env.TMPDIR
    process.env.TMPDIR = '/nonexistent/xerxes-deleted-run-dir-99999'
    let created = ''
    try {
      created = await mkdtempSafe('xt-test-')
      expect(existsSync(created)).toBe(true)
      expect(created.startsWith('/tmp/xt-test-') || created.startsWith('/private/tmp/xt-test-')).toBe(true)
    } finally {
      if (created) rmSync(created, { recursive: true, force: true })
      if (prev === undefined) delete process.env.TMPDIR
      else process.env.TMPDIR = prev
    }
  })
})
