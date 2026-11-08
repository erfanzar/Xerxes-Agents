// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Whether THIS machine's own sign-in for a per-machine provider works. An SSH
 * window used to relay every new task through the desktop's login even when
 * the host had its own working sign-in; with this fact the desktop runs it
 * on the host instead. Only a state and a short reason leave this module:
 * never a token, an account or a path. Discovery never signs in, refreshes
 * tokens, or sends a provider request.
 */

import { homedir } from 'node:os'
import { delimiter, join } from 'node:path'

import { claudeCodeLogin, claudeExecutable } from '../auth/claudeCodeLogin.js'
import { CODEX_REFRESH_SKEW_SECONDS, CodexSession, importCodexCliTokens } from '../auth/codexAuth.js'
import { ConfigurationError } from '../core/errors.js'

export type SignInReadinessState = 'ready' | 'not_signed_in' | 'not_installed' | 'unknown'
export interface SignInReadiness {
  readonly state: SignInReadinessState
  readonly reason?: string
}
export type SignInReadinessProbe = (provider: string) => Promise<SignInReadiness>

export interface SignInReadinessDeps {
  readonly environment?: Readonly<Record<string, string | undefined>>
  readonly executable?: (environment: Readonly<Record<string, string | undefined>>) => string | undefined
  readonly login?: (environment: Readonly<Record<string, string | undefined>>) => Promise<unknown>
  readonly codexExecutable?: (environment: Readonly<Record<string, string | undefined>>) => string | undefined
  /** Read the host's saved OAuth session without resolving or refreshing it. */
  readonly codexSession?: Pick<CodexSession, 'stored'>
  readonly codexCliTokens?: typeof importCodexCliTokens
  readonly homeDirectory?: string
  readonly now?: () => number
  /** A keychain read can prompt; one answer serves this long. */
  readonly cacheMs?: number
}

/** Include common installer locations: an SSH daemon's PATH may be minimal. */
function codexExecutable(environment: Readonly<Record<string, string | undefined>>): string | undefined {
  const home = environment.HOME?.trim() || homedir()
  const bunBin = join(environment.BUN_INSTALL?.trim() || join(home, '.bun'), 'bin')
  const extra = [join(home, '.local', 'bin'), bunBin, '/opt/homebrew/bin', '/usr/local/bin']
  return Bun.which('codex', { PATH: [environment.PATH ?? '', ...extra].filter(Boolean).join(delimiter) }) ?? undefined
}

export function createSignInReadinessProbe(deps: SignInReadinessDeps = {}): SignInReadinessProbe {
  const environment = deps.environment ?? process.env
  const executable = deps.executable ?? claudeExecutable
  const login = deps.login ?? (env => claudeCodeLogin({ environment: env }))
  const codexInstalled = deps.codexExecutable ?? codexExecutable
  const homeDirectory = deps.homeDirectory ?? (environment.HOME?.trim() || homedir())
  const codexSession = deps.codexSession ?? new CodexSession({ environment, homeDirectory })
  const codexCliTokens = deps.codexCliTokens ?? importCodexCliTokens
  const now = deps.now ?? Date.now
  const cacheMs = deps.cacheMs ?? 30_000
  const cache = new Map<string, { readonly at: number; readonly value: Promise<SignInReadiness> }>()
  const probe = async (provider: string): Promise<SignInReadiness> => {
    if (provider === 'openai-codex') {
      try {
        const [stored, cli] = await Promise.all([codexSession.stored(), codexCliTokens(environment, homeDirectory)])
        // The native Responses transport uses either credential source directly;
        // it does not require the Codex executable. Expired access is still
        // usable with a refresh token, but listing never rotates it.
        if ([stored, cli].some(token => token && (token.refreshToken?.trim() || !token.isExpired(CODEX_REFRESH_SKEW_SECONDS, now() / 1000)))) {
          return { state: 'ready' }
        }
        return codexInstalled(environment)
          ? { state: 'not_signed_in', reason: 'Codex is not signed in on this machine.' }
          : { state: 'not_installed', reason: 'Codex CLI is not installed on this machine.' }
      } catch {
        return { state: 'unknown', reason: 'Could not read Codex\'s sign-in on this machine.' }
      }
    }
    if (provider !== 'claude-code') return { state: 'unknown', reason: 'This machine does not check that provider\'s sign-in.' }
    if (!executable(environment)) return { state: 'not_installed', reason: 'Claude Code is not installed on this machine.' }
    try {
      await login(environment)
      return { state: 'ready' }
    } catch (error) {
      if (error instanceof ConfigurationError) return { state: 'not_signed_in', reason: 'Claude Code is not signed in on this machine.' }
      // Still signed in: only the short-lived access token lapsed, and Claude Code renews it itself.
      if (error instanceof Error && /lapsed/i.test(error.message)) return { state: 'ready' }
      return { state: 'unknown', reason: 'Could not read Claude Code\'s sign-in on this machine.' }
    }
  }
  return async provider => {
    const key = provider.trim().toLowerCase()
    const hit = cache.get(key)
    if (hit && now() - hit.at < cacheMs) return hit.value
    const value = probe(key)
    cache.set(key, { at: now(), value })
    return value
  }
}
