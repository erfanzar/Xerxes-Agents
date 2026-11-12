// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { access, stat } from 'node:fs/promises'
import { dirname } from 'node:path'
import { openRemote, runCaptured, sshOptions, type RemoteRunner } from '../src/desktop/main/remote.js'
import { verifyRemoteHost } from '../src/desktop/main/remoteHostTrust.js'

const destination = 'v5p-inference'
const fingerprint = 'SHA256:' + 'A'.repeat(43)
const prompt = `The authenticity of host '${destination}' can't be established.\nED25519 key fingerprint is ${fingerprint}.\nAre you sure you want to continue connecting (yes/no/[fingerprint])?`
const signal = () => new AbortController().signal

function askpassRun(text: string, observe: (answer: string, path: string, args: string[]) => void): RemoteRunner {
  return async (binary, args, requestSignal, timeout, env) => {
    expect(binary).toBe('ssh')
    if (!env?.SSH_ASKPASS) throw new Error('missing askpass bridge')
    expect(env.SSH_ASKPASS_REQUIRE).toBe('force')
    expect(env.LC_ALL).toBe('C')
    expect((await stat(dirname(env.SSH_ASKPASS))).mode & 0o777).toBe(0o700)
    const answer = await runCaptured(env.SSH_ASKPASS, [text], requestSignal, timeout, env)
    observe(answer.trim(), env.SSH_ASKPASS, args)
    if (answer.trim() !== 'yes') throw new Error('Host key verification failed.')
    return ''
  }
}

test.skipIf(process.platform === 'win32')('trust confirmation is explicit, fail-closed and cleans up its private askpass bridge', async () => {
  let helper = '', dialogs = 0
  const run = askpassRun(prompt, (answer, path, args) => {
    helper = path
    expect(answer).toBe('no')
    expect(args).toContain('StrictHostKeyChecking=ask')
    expect(args).not.toContain('StrictHostKeyChecking=no')
    expect(args).not.toContain('StrictHostKeyChecking=accept-new')
    expect(args).toContain('UpdateHostKeys=no')
    expect(args).toContain('PasswordAuthentication=no')
    expect(args).toContain('ControlMaster=no')
  })
  await expect(openRemote({ alias: destination, target: destination, workspacePath: '/srv/project' }, signal(), () => {}, undefined, {
    confirmHostKey: async (target, text) => { dialogs++; expect(target).toBe(destination); expect(text).toBe(prompt); return false },
    run,
    spawnTunnel: () => { throw new Error('must not open a tunnel without trust') },
  })).rejects.toThrow('Host key verification failed')
  expect(dialogs).toBe(1)
  await expect(access(helper)).rejects.toThrow()
})

test.skipIf(process.platform === 'win32')('password prompts and malformed or control-character prompts are never offered as host trust', async () => {
  for (const text of ['Password:', 'Enter passphrase for key:', 'Are you sure you want to continue connecting?', prompt + '\u001b[0m']) {
    let dialogs = 0
    await expect(verifyRemoteHost(destination, signal(), async () => { dialogs++; return true },
      askpassRun(text, answer => expect(answer).toBe('no')), sshOptions)).rejects.toThrow('Host key verification failed')
    expect(dialogs).toBe(0)
  }
})

test.skipIf(process.platform === 'win32')('a known host needs no dialog and a changed key is never replaced', async () => {
  let dialogs = 0
  const confirm = async () => { dialogs++; return true }
  await verifyRemoteHost(destination, signal(), confirm, async () => '', sshOptions)
  await expect(verifyRemoteHost(destination, signal(), confirm, async () => {
    throw new Error('REMOTE HOST IDENTIFICATION HAS CHANGED!')
  }, sshOptions)).rejects.toThrow('HOST IDENTIFICATION HAS CHANGED')
  expect(dialogs).toBe(0)
})

test.skipIf(process.platform === 'win32')('cancelling while the host dialog is open aborts the dialog and removes helper files', async () => {
  const controller = new AbortController()
  let helper = '', dialogSignal: AbortSignal | undefined
  const pending = verifyRemoteHost(destination, controller.signal, async (_target, _text, requestSignal) => {
    dialogSignal = requestSignal
    controller.abort()
    return true
  }, async (_binary, _args, requestSignal, timeout, env) => {
    helper = env!.SSH_ASKPASS!
    return runCaptured(helper, [prompt], requestSignal, timeout, env)
  }, sshOptions)
  await expect(pending).rejects.toThrow(/cancel|abort/i)
  expect(dialogSignal?.aborted).toBe(true)
  await expect(access(helper)).rejects.toThrow()
})

test.skipIf(process.platform === 'win32')('a failed handshake closes a pending trust dialog', async () => {
  let dialogSignal: AbortSignal | undefined
  await expect(verifyRemoteHost(destination, signal(), async (_target, _text, requestSignal) => {
    dialogSignal = requestSignal
    return true
  }, askpassRun(prompt, () => { throw new Error('connection closed') }), sshOptions)).rejects.toThrow('connection closed')
  expect(dialogSignal?.aborted).toBe(true)
})
