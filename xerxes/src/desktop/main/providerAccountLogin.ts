// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { spawn } from 'node:child_process'
import { bunBinaryOf, daemonEntryOf } from './spawn.js'

export async function runBunProviderAccountLogin(
  accountId: string,
  options: { cwd?: string; env?: NodeJS.ProcessEnv; entry?: string; onAuthorizationUrl?: (url: string) => void } = {},
): Promise<void> {
  const cwd = options.cwd ?? process.cwd()
  const env = options.env ?? process.env
  const entry = options.entry ?? daemonEntryOf(cwd, env)
  const child = spawn(bunBinaryOf(env), [entry, 'provider-account-login', accountId], {
    cwd, env, stdio: ['ignore', 'pipe', 'pipe'],
  })
  let output = '', stdout = ''
  const acceptStdout = (line: string): void => {
    if (!line.trim()) return
    try {
      const event = JSON.parse(line) as { type?: unknown; url?: unknown }
      if (event.type === 'authorization_url' && typeof event.url === 'string' && new URL(event.url).protocol === 'https:') {
        options.onAuthorizationUrl?.(event.url)
        return
      }
    } catch { /* Ordinary helper output is retained for the error report. */ }
    output = `${output}${line}\n`.slice(-4000)
  }
  child.stdout.on('data', (chunk: Buffer) => {
    stdout += chunk.toString()
    const lines = stdout.split(/\r?\n/)
    stdout = lines.pop() ?? ''
    for (const line of lines) acceptStdout(line)
  })
  child.stderr.on('data', (chunk: Buffer) => { output = `${output}${chunk.toString()}`.slice(-4000) })
  const timeout = setTimeout(() => child.kill('SIGTERM'), 15 * 60 * 1000)
  timeout.unref()
  await new Promise<void>((resolve, reject) => {
    child.once('error', reject)
    child.once('close', (code, signal) => {
      clearTimeout(timeout)
      if (stdout) acceptStdout(stdout)
      if (code === 0) resolve()
      else reject(new Error(output.trim() || `OAuth login exited (${code ?? signal ?? 'unknown'})`))
    })
  })
}
