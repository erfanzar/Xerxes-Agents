// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Whether THIS machine's own sign-in for a per-machine provider works. An SSH
 * window used to relay every new task through the desktop's login even when
 * the host had its own working Claude Code; with this fact the desktop runs it
 * on the host instead. Only a state and a short reason leave this module:
 * never a token, an account or a path.
 */

import { claudeCodeLogin, claudeExecutable } from '../auth/claudeCodeLogin.js'
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
  readonly now?: () => number
  /** A keychain read can prompt; one answer serves this long. */
  readonly cacheMs?: number
}

export function createSignInReadinessProbe(deps: SignInReadinessDeps = {}): SignInReadinessProbe {
  const environment = deps.environment ?? process.env
  const executable = deps.executable ?? claudeExecutable
  const login = deps.login ?? (env => claudeCodeLogin({ environment: env }))
  const now = deps.now ?? Date.now
  const cacheMs = deps.cacheMs ?? 30_000
  const cache = new Map<string, { readonly at: number; readonly value: Promise<SignInReadiness> }>()
  const probe = async (provider: string): Promise<SignInReadiness> => {
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
