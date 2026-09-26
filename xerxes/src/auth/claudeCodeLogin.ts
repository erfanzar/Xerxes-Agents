// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Read-only access to the Claude Code CLI's own sign-in, so the `cc`
 * profile's plan windows can be shown without a separate Xerxes login.
 *
 * Claude Code keeps its OAuth credential in the macOS keychain (service
 * "Claude Code-credentials") or, elsewhere, in `~/.claude/.credentials.json`
 * (`CLAUDE_CONFIG_DIR` moves it). Only the access token is used, and only
 * while it is valid: refreshing it here would rotate Claude Code's refresh
 * token and sign the CLI out — Claude Code renews it itself on its next
 * request. (`claude auth status` does not: it only reads the saved state.)
 * `XERXES_CLAUDE_CODE_LOGIN=off` turns this off.
 */

import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

import { ConfigurationError } from '../core/errors.js'

export const CLAUDE_CODE_KEYCHAIN_SERVICE = 'Claude Code-credentials'

export interface ClaudeCodeLogin {
  readonly accessToken: string
  readonly expiresAt?: number
  /** "max", "pro" — Claude Code records the plan beside the token. */
  readonly subscriptionType?: string
}

export interface ClaudeCodeLoginOptions {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly platform?: NodeJS.Platform
  readonly now?: () => number
  /** Swap the keychain read (tests). Returns the stored JSON, or undefined. */
  readonly readKeychain?: () => Promise<string | undefined>
  readonly readFileText?: (path: string) => Promise<string>
}

/** Parse Claude Code's stored credential document; undefined when it carries no token. */
export function parseClaudeCodeCredential(text: string): ClaudeCodeLogin | undefined {
  let document: unknown
  try { document = JSON.parse(text) } catch { return undefined }
  const oauth = document && typeof document === 'object' ? (document as Record<string, unknown>).claudeAiOauth : undefined
  if (!oauth || typeof oauth !== 'object') return undefined
  const record = oauth as Record<string, unknown>
  const accessToken = typeof record.accessToken === 'string' ? record.accessToken.trim() : ''
  if (!accessToken) return undefined
  const expiresAt = typeof record.expiresAt === 'number' && Number.isFinite(record.expiresAt) ? record.expiresAt : undefined
  const subscriptionType = typeof record.subscriptionType === 'string' && record.subscriptionType.trim() ? record.subscriptionType.trim() : undefined
  return { accessToken, ...(expiresAt !== undefined ? { expiresAt } : {}), ...(subscriptionType ? { subscriptionType } : {}) }
}

interface SourceRead {
  readonly text?: string
  /** Why nothing usable came back, in words (keychain only). */
  readonly problem?: string
}

async function keychainCredential(): Promise<SourceRead> {
  const child = Bun.spawn(['/usr/bin/security', 'find-generic-password', '-s', CLAUDE_CODE_KEYCHAIN_SERVICE, '-w'], { stdout: 'pipe', stderr: 'ignore' })
  // The first read can raise macOS's "allow access" prompt; give a person time to answer it.
  let timedOut = false
  const timer = setTimeout(() => { timedOut = true; child.kill() }, 30_000)
  try {
    const [text, code] = await Promise.all([new Response(child.stdout).text(), child.exited])
    if (code === 0 && text.trim()) return { text: text.trim() }
    // 44: no such item. Anything else is a refused or unanswered access prompt.
    if (code === 44) return {}
    return { problem: timedOut ? 'the macOS keychain prompt was not answered' : 'macOS keychain access was denied' }
  } finally { clearTimeout(timer) }
}

/** The `claude` executable, from PATH or where its installers put it. */
export function claudeExecutable(environment: Readonly<Record<string, string | undefined>>): string | undefined {
  const home = environment.HOME?.trim() || homedir()
  const extra = [join(home, '.local', 'bin'), join(home, '.claude', 'local'), '/opt/homebrew/bin', '/usr/local/bin']
  const path = [environment.PATH ?? '', ...extra].filter(Boolean).join(':')
  return Bun.which('claude', { PATH: path }) ?? undefined
}

const when = (epoch: number) => new Date(epoch).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })

/** Claude Code's current sign-in, or a ConfigurationError saying what to do. */
export async function claudeCodeLogin(options: ClaudeCodeLoginOptions = {}): Promise<ClaudeCodeLogin> {
  const environment = options.environment ?? process.env
  // Opt-out for people who do not want Xerxes reading another app's login.
  if (/^(0|off|false|no)$/i.test(environment.XERXES_CLAUDE_CODE_LOGIN?.trim() ?? '')) {
    throw new ConfigurationError('claude_code', 'Reading the Claude Code sign-in is turned off (XERXES_CLAUDE_CODE_LOGIN).')
  }
  const platform = options.platform ?? process.platform
  const now = options.now ?? Date.now
  const sources: Array<() => Promise<SourceRead>> = []
  if (platform === 'darwin') sources.push(options.readKeychain ? async () => ({ text: (await options.readKeychain!()) ?? '' }) : keychainCredential)
  const directory = environment.CLAUDE_CONFIG_DIR?.trim() || join(environment.HOME?.trim() || homedir(), '.claude')
  const read = options.readFileText ?? ((path: string) => readFile(path, 'utf8'))
  sources.push(async () => ({ text: await read(join(directory, '.credentials.json')).catch(() => '') }))
  // Every source is read and the freshest token wins: the keychain can hold
  // a live token while an old credentials file lingers, or the reverse.
  const readAll = async () => {
    let newest: ClaudeCodeLogin | undefined
    const problems: string[] = []
    for (const source of sources) {
      const result = await source().catch((): SourceRead => ({}))
      if (result.problem) problems.push(result.problem)
      const login = parseClaudeCodeCredential(result.text ?? '')
      if (login && (!newest || (login.expiresAt ?? Infinity) > (newest.expiresAt ?? Infinity))) newest = login
    }
    return { newest, problems }
  }
  const valid = (login: ClaudeCodeLogin | undefined) => Boolean(login && (login.expiresAt === undefined || login.expiresAt > now()))
  const { newest, problems } = await readAll()
  if (newest && valid(newest)) return newest
  if (newest) {
    // Still signed in: only the short-lived access token lapsed while Claude
    // Code made no request (another Claude app signs in on its own). A plain
    // Error keeps the card visible with this advice.
    throw new Error(`You're signed in, but Claude Code's saved access token lapsed ${when(newest.expiresAt!)}. Claude Code renews it on its next request — use the Claude Code profile once or run 'claude' — and this card fills in.`)
  }
  if (problems.length) throw new Error(`Could not read Claude Code’s sign-in: ${problems.join('; ')}. Allow access when macOS asks, then refresh.`)
  throw new ConfigurationError('claude_code', "No Claude Code sign-in found. Run 'claude' and sign in with your Claude plan.")
}
