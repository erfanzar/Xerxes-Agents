// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { EventEmitter } from 'node:events'
import { mkdir, mkdtemp, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { Readable, Writable } from 'node:stream'

import { expect, test } from 'bun:test'

import { loadAgyRuntimeEnv } from '../src/auth/agyRuntimeEnv.js'
import { ProviderAccountRegistry } from '../src/auth/providerAccountRegistry.js'
import {
  AgyCredentialTransfer,
  AgyLoginManager,
  adcCredentialsPath,
  agyOauthTokenPath,
  decodeAgyOauthToken,
  inspectLocalAgyCredentials,
  remoteAgyReceipt,
  remoteAgyReceiverSource,
  spawnAgySecretWriter,
  type AgyRemoteSyncPayload,
} from '../src/desktop/main/agyCredentials.js'
import { createBridge } from '../src/desktop/bridgeApi.js'
import {
  CODEX_PACKAGE,
  ensureAgyCli,
  ensureCodexCli,
  ensureDaemonCompanionClis,
  runInstallCommand,
} from '../src/runtime/companionInstall.js'

const target = { alias: 'gpu', target: 'user@example.com', workspacePath: '/work/repo' }

test('inspectLocalAgyCredentials reports available modes without ever exposing secrets', async () => {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-agy-inspect-'))
  try {
    const adcSecret = 'super-secret-adc-refresh-token'
    const geminiSecret = 'AIzaSy-secret-gemini-key'
    await mkdir(join(home, '.local', 'bin'), { recursive: true })
    await writeFile(join(home, '.local', 'bin', 'agy'), '#!/bin/sh\n', { mode: 0o755 })
    await mkdir(join(home, '.config', 'gcloud'), { recursive: true })
    await writeFile(adcCredentialsPath(home), JSON.stringify({
      type: 'authorized_user',
      quota_project_id: 'demo-proj',
      refresh_token: adcSecret,
      client_secret: 'client-secret-value',
    }), { mode: 0o600 })

    const registry = new ProviderAccountRegistry(join(home, 'accounts'), join(home, 'profiles.json'))
    await registry.createGemini('Work', geminiSecret, 'gemini-2.5-flash')

    const status = await inspectLocalAgyCredentials({
      home,
      env: { PATH: '/nonexistent' },
      registry,
      hasKeyringEntry: async () => true,
    }, { destination: target.target })

    expect(status.agyInstalled).toBe(true)
    expect(status.canSync).toBe(true)
    expect(status.recommendedMode).toBe('keyring')
    expect(status.sources.map(s => [s.mode, s.available, s.headlessSafe])).toEqual([
      ['keyring', true, true],
      ['adc', true, true],
      ['gemini_api_key', true, true],
    ])
    const serialized = JSON.stringify(status)
    expect(serialized).not.toContain(adcSecret)
    expect(serialized).not.toContain(geminiSecret)
    expect(serialized).not.toContain('client-secret-value')
  } finally {
    await rm(home, { recursive: true, force: true })
  }
})

test('decodeAgyOauthToken decodes go-keyring-base64, hex, and raw JSON, and receiver writes mode 0600 headless token file', async () => {
  const tokenObj = { token: { access_token: 'secret-access', refresh_token: 'secret-refresh' }, auth_method: 'gcp', id_token_email: 'user@example.com' }
  const json = JSON.stringify(tokenObj)
  const wrapped = 'go-keyring-base64:' + Buffer.from(json, 'utf8').toString('base64')
  const hexWrapped = Buffer.from(wrapped, 'utf8').toString('hex')
  expect(decodeAgyOauthToken(wrapped)).toEqual(tokenObj)
  expect(decodeAgyOauthToken(hexWrapped)).toEqual(tokenObj)
  expect(decodeAgyOauthToken(json)).toEqual(tokenObj)
  expect(decodeAgyOauthToken('not-valid')).toBeUndefined()

  const remoteHome = await mkdtemp(join(tmpdir(), 'xerxes-agy-oauth-remote-'))
  try {
    await mkdir(join(remoteHome, '.gemini', 'antigravity-cli'), { recursive: true })
    await writeFile(join(remoteHome, '.gemini', 'antigravity-cli', 'settings.json'), JSON.stringify({ modelProvider: 'gemini', model: 'Gemini 4 Neon (High)' }))
    const payload: AgyRemoteSyncPayload = {
      kind: 'agy',
      mode: 'keyring',
      secret: wrapped,
      oauthToken: tokenObj,
    }
    const proc = Bun.spawn([process.execPath, '-e', remoteAgyReceiverSource()], {
      stdin: new TextEncoder().encode(JSON.stringify(payload)),
      stdout: 'pipe',
      stderr: 'pipe',
      env: { ...process.env, HOME: remoteHome, DBUS_SESSION_BUS_ADDRESS: '', XERXES_SKIP_OS_KEYRING: '1' },
    })
    const [stdout, code] = await Promise.all([new Response(proc.stdout).text(), proc.exited])
    expect(code).toBe(0)
    expect(remoteAgyReceipt(stdout)).toBe('keyring')
    expect(stdout).not.toContain('secret-access')
    expect(stdout).not.toContain('secret-refresh')
    const writtenPath = agyOauthTokenPath(remoteHome)
    expect((await stat(writtenPath)).mode & 0o777).toBe(0o600)
    expect(await Bun.file(writtenPath).json()).toEqual(tokenObj)
    const settingsDoc = await Bun.file(join(remoteHome, '.gemini', 'antigravity-cli', 'settings.json')).json() as Record<string, unknown>
    expect(settingsDoc).toEqual({ model: 'Gemini 4 Neon (High)' })
  } finally {
    await rm(remoteHome, { recursive: true, force: true })
  }
})

test('AgyLoginManager extracts the Google OAuth URL and submits the authorization code on stdin', async () => {
  let receivedStdin = ''
  const child = new EventEmitter() as EventEmitter & { stdin: Writable; stdout: Readable; stderr: Readable; kill: () => void }
  child.stdout = new Readable({ read() {} })
  child.stderr = new Readable({ read() {} })
  child.stdin = new Writable({
    write(chunk, _enc, cb) {
      receivedStdin += String(chunk)
      cb()
      queueMicrotask(() => child.emit('close', 0))
    },
  })
  child.kill = () => {}

  const manager = new AgyLoginManager({
    findAgyPath: async () => '/usr/local/bin/agy',
    spawnLogin: () => {
      queueMicrotask(() => {
        child.stdout.push('Open this link in a browser:\n\n  https://accounts.google.com/o/oauth2/auth?client_id=test&state=abc\n\nPaste code:\n')
      })
      return child as never
    },
  })
  const started = await manager.start()
  expect(started.url).toBe('https://accounts.google.com/o/oauth2/auth?client_id=test&state=abc')
  await manager.complete(started.loginId, '4/0test-auth-code')
  expect(receivedStdin).toBe('4/0test-auth-code\n')
})

test('agy SSH writer sends credentials over stdin only and receiver writes mode 0600 files cleanly', async () => {
  const secret = 'AIzaSy-never-in-argv-secret'
  let argv = ''
  let stdin = ''
  const child = new EventEmitter() as EventEmitter & { stdin: Writable; stdout: Readable; kill: () => void }
  child.stdin = new Writable({
    write(chunk, _encoding, callback) {
      stdin += String(chunk)
      callback()
      queueMicrotask(() => {
        child.stdout.push(JSON.stringify({ ok: true, provider: 'agy', mode: 'gemini_api_key' }) + '\n')
        child.emit('close', 0)
      })
    },
  })
  child.stdout = new Readable({ read() {} })
  child.kill = () => {}

  const output = await spawnAgySecretWriter(
    target,
    { kind: 'agy', mode: 'gemini_api_key', apiKey: secret },
    new AbortController().signal,
    1000,
    (_binary, args) => { argv = args.join(' '); return child as never },
  )
  expect(remoteAgyReceipt(output)).toBe('gemini_api_key')
  expect(argv).not.toContain(secret)
  expect(stdin).toContain(secret)
  expect(remoteAgyReceiverSource()).not.toContain(secret)

  // Execute the real static Bun receiver for both ADC and Gemini API key modes.
  const remoteHome = await mkdtemp(join(tmpdir(), 'xerxes-agy-remote-'))
  try {
    // Seed existing user config to ensure unrelated settings are preserved.
    await mkdir(join(remoteHome, '.gemini', 'config'), { recursive: true })
    await writeFile(join(remoteHome, '.gemini', 'config', 'config.json'), JSON.stringify({
      theme: 'dark',
      userSettings: { existingFlag: true },
    }))
    await mkdir(join(remoteHome, '.gemini', 'antigravity-cli'), { recursive: true })
    await writeFile(join(remoteHome, '.gemini', 'antigravity-cli', 'settings.json'), JSON.stringify({
      customOption: 'keep-me',
    }))

    const runReceiver = async (payload: AgyRemoteSyncPayload) => {
      const proc = Bun.spawn([process.execPath, '-e', remoteAgyReceiverSource()], {
        stdin: new TextEncoder().encode(JSON.stringify(payload)),
        stdout: 'pipe',
        stderr: 'pipe',
        env: { ...process.env, HOME: remoteHome },
      })
      const [stdout, code] = await Promise.all([new Response(proc.stdout).text(), proc.exited])
      return { stdout, code }
    }

    const adcRes = await runReceiver({
      kind: 'agy',
      mode: 'adc',
      adc: { type: 'authorized_user', refresh_token: 'adc-secret-tok', quota_project_id: 'proj-1' },
      quotaProject: 'proj-1',
    })
    expect(adcRes.code).toBe(0)
    expect(remoteAgyReceipt(adcRes.stdout)).toBe('adc')
    expect(adcRes.stdout).not.toContain('adc-secret-tok')

    const adcFile = join(remoteHome, '.config', 'gcloud', 'application_default_credentials.json')
    expect((await stat(adcFile)).mode & 0o777).toBe(0o600)
    expect(await Bun.file(adcFile).json()).toMatchObject({ type: 'authorized_user', refresh_token: 'adc-secret-tok' })
    const geminiConfig = await Bun.file(join(remoteHome, '.gemini', 'config', 'config.json')).json() as Record<string, unknown>
    expect(geminiConfig).toEqual({ theme: 'dark', userSettings: { existingFlag: true, enableAdc: true } })

    const keyRes = await runReceiver({
      kind: 'agy',
      mode: 'gemini_api_key',
      apiKey: secret,
    })
    expect(keyRes.code).toBe(0)
    expect(remoteAgyReceipt(keyRes.stdout)).toBe('gemini_api_key')
    expect(keyRes.stdout).not.toContain(secret)

    const settingsDoc = await Bun.file(join(remoteHome, '.gemini', 'antigravity-cli', 'settings.json')).json() as Record<string, unknown>
    expect(settingsDoc).toEqual({ customOption: 'keep-me', modelProvider: 'gemini' })

    const loadedEnv = await loadAgyRuntimeEnv(remoteHome)
    expect(loadedEnv).toEqual({
      AGY_ADC_AUTH: 'true',
      GOOGLE_CLOUD_QUOTA_PROJECT: 'proj-1',
      GEMINI_API_KEY: secret,
    })
    expect((await stat(join(remoteHome, '.xerxes', 'provider-accounts', 'agy', 'env.json'))).mode & 0o777).toBe(0o600)
  } finally {
    await rm(remoteHome, { recursive: true, force: true })
  }
})

test('AgyCredentialTransfer enforces one-use unexpired reviews and refuses headless keyring failure clearly', async () => {
  const home = await mkdtemp(join(tmpdir(), 'xerxes-agy-manager-'))
  try {
    await mkdir(join(home, '.config', 'gcloud'), { recursive: true })
    await writeFile(adcCredentialsPath(home), JSON.stringify({ type: 'authorized_user', refresh_token: 'secret-tok' }), { mode: 0o600 })
    let now = 10_000
    const sent: AgyRemoteSyncPayload[] = []
    const manager = new AgyCredentialTransfer(
      target,
      { home, env: { PATH: '/nonexistent' }, hasKeyringEntry: async () => true, readKeyringSecret: async () => 'kr-secret' },
      () => now,
      async payload => {
        sent.push(payload)
        if (payload.mode === 'keyring') throw new Error('keyring_unavailable')
        return JSON.stringify({ ok: true, provider: 'agy', mode: payload.mode })
      },
    )

    const review = await manager.reviewSync('adc') as { review: string; expires_at: number }
    const confirmed = await manager.confirmSync(review.review, true)
    expect(confirmed).toEqual({ ok: true, provider: 'agy', mode: 'adc', destination: target.target })
    expect(sent).toHaveLength(1)
    // One-use review cannot be replayed.
    await expect(manager.confirmSync(review.review, true)).rejects.toThrow('expired')

    // Expired review is refused.
    const expiring = await manager.reviewSync('adc') as { review: string; expires_at: number }
    now = expiring.expires_at + 1
    await expect(manager.confirmSync(expiring.review, true)).rejects.toThrow('expired')

    // Keyring transfer onto a headless host returns a clear actionable error.
    const keyringReview = await manager.reviewSync('keyring') as { review: string }
    await expect(manager.confirmSync(keyringReview.review, true)).rejects.toThrow('Google Cloud ADC or a Gemini API key')
  } finally {
    await rm(home, { recursive: true, force: true })
  }
})

test('companion installer supports --agy and auto-installs agy once for concurrent callers', async () => {
  const dryRunLines: string[] = []
  const dryRun = await runInstallCommand(['--agy', '--dry-run'], {
    findExecutable: () => null,
    run: async () => { throw new Error('dry-run must not spawn') },
    write: line => dryRunLines.push(line),
  })
  expect(dryRun.status).toBe('dry-run')
  expect(dryRunLines[0]).toContain('https://antigravity.google/cli/install.sh')
  expect(dryRunLines[0]).not.toContain('--skip-aliases')
  expect(dryRunLines[0]).not.toContain('--skip-path')

  let installed: string | undefined
  const runs: string[][] = []
  const host = {
    autoInstall: true,
    find: () => installed,
    run: async (argv: readonly string[]) => {
      runs.push([...argv])
      await Bun.sleep(10)
      installed = '/home/me/.local/bin/agy'
      return { code: 0, output: '' }
    },
  }
  const [first, second] = await Promise.all([ensureAgyCli(host), ensureAgyCli(host)])
  expect(first).toEqual({ path: '/home/me/.local/bin/agy', status: 'installed' })
  expect(second).toEqual(first)
  expect(runs).toHaveLength(1)

  await expect(ensureAgyCli({
    environment: { XERXES_AUTO_INSTALL_AGY: '0' },
    find: () => undefined,
    run: async () => { throw new Error('must not run') },
  })).rejects.toThrow('automatic install is off')
})

test('ensureCodexCli and ensureDaemonCompanionClis auto-install missing claude, codex, and agy independently', async () => {
  // 1. ensureCodexCli installs @openai/codex once for concurrent callers.
  let codexBin: string | undefined
  const codexRuns: string[][] = []
  const codexHost = {
    autoInstall: true,
    bunExecutable: '/opt/bun',
    find: () => codexBin,
    run: async (argv: readonly string[]) => {
      codexRuns.push([...argv])
      await Bun.sleep(10)
      codexBin = '/home/me/.bun/bin/codex'
      return { code: 0, output: '' }
    },
  }
  const [c1, c2] = await Promise.all([ensureCodexCli(codexHost), ensureCodexCli(codexHost)])
  expect(c1).toEqual({ path: '/home/me/.bun/bin/codex', status: 'installed' })
  expect(c2).toEqual(c1)
  expect(codexRuns).toEqual([['/opt/bun', 'add', '--global', CODEX_PACKAGE]])

  // 2. ensureDaemonCompanionClis installs all missing binaries and isolates individual failures.
  const bins: Partial<Record<'claude' | 'codex' | 'agy', string>> = { claude: '/usr/local/bin/claude' }
  const seenCommands: string[] = []
  const result = await ensureDaemonCompanionClis({
    autoInstall: true,
    bunExecutable: '/opt/bun',
    find: name => bins[name],
    run: async argv => {
      const joined = argv.join(' ')
      seenCommands.push(joined)
      if (joined.includes(CODEX_PACKAGE)) {
        return { code: 1, output: 'registry temporarily unreachable' }
      }
      if (joined.includes('antigravity.google/cli/install.sh')) {
        bins.agy = '/home/me/.local/bin/agy'
        return { code: 0, output: 'installed' }
      }
      return { code: 0, output: '' }
    },
  })
  expect(result.claude).toEqual({ status: 'present', path: '/usr/local/bin/claude' })
  expect(result.codex.status).toBe('failed')
  expect(result.agy).toEqual({ status: 'installed', path: '/home/me/.local/bin/agy' })
  expect(seenCommands.some(cmd => cmd.includes(CODEX_PACKAGE))).toBe(true)
  expect(seenCommands.some(cmd => cmd.includes('antigravity.google/cli/install.sh') && !cmd.includes('--skip-aliases'))).toBe(true)

  // 3. XERXES_AUTO_INSTALL_COMPANIONS=0 disables all three cleanly.
  const skipped = await ensureDaemonCompanionClis({
    environment: { XERXES_AUTO_INSTALL_COMPANIONS: '0' },
    find: () => undefined,
    run: async () => { throw new Error('must not run when disabled') },
  })
  expect(skipped.claude.status).toBe('skipped')
  expect(skipped.codex.status).toBe('skipped')
  expect(skipped.agy.status).toBe('skipped')
})

test('desktop bridge forwards agy credential transfer, login, and provider-setup actions', async () => {
  const calls: unknown[][] = []
  const bridge = createBridge({
    invoke: async (...args) => { calls.push(args); return { ok: true } },
    on: () => () => {},
    setZoomFactor: () => {},
  })
  await bridge.accounts('agy-status', {})
  await bridge.accounts('agy-sync-review', { mode: 'keyring' })
  await bridge.accounts('agy-sync', { review: 'rev-1', confirm: true })
  await bridge.accounts('agy-login-start', {})
  await bridge.accounts('agy-login-complete', { loginId: 'id-1', code: 'code-1' })
  await bridge.accounts('agy-login-cancel', { loginId: 'id-1' })
  await bridge.remote('provider-setup', {})
  expect(calls.map(c => [c[0], c[1]])).toEqual([
    ['desktop:accounts', 'agy-status'],
    ['desktop:accounts', 'agy-sync-review'],
    ['desktop:accounts', 'agy-sync'],
    ['desktop:accounts', 'agy-login-start'],
    ['desktop:accounts', 'agy-login-complete'],
    ['desktop:accounts', 'agy-login-cancel'],
    ['desktop:remote', 'provider-setup'],
  ])
})
