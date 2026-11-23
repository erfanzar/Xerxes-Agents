// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile } from 'node:fs/promises'
import { homedir } from 'node:os'
import { join } from 'node:path'

/**
 * Allowed environment variables transferred for headless `agy acp` sessions.
 * Stored owner-only (mode 0600) in `~/.xerxes/provider-accounts/agy/env.json`
 * so remote SSH hosts never need credentials written into shell rc files.
 */
export const AGY_ALLOWED_ENV_KEYS = [
  'GEMINI_API_KEY',
  'GOOGLE_GEMINI_BASE_URL',
  'AGY_ADC_AUTH',
  'GOOGLE_CLOUD_QUOTA_PROJECT',
  'GOOGLE_CLOUD_LOCATION',
] as const

export type AgyEnvKey = (typeof AGY_ALLOWED_ENV_KEYS)[number]

/**
 * Outer Antigravity / host-sandbox variables that must never leak into a
 * Xerxes-owned `agy acp` subprocess (otherwise a nested `agy` instance can
 * collide with its parent host's app-data dir, CSRF token, or forced
 * file-based token storage and miss the user's real sign-in).
 */
export const AGY_OUTER_HOST_ENV_KEYS = [
  'AGY_ACP_FORCE_FILE_STORAGE',
  'ANTIGRAVITY_AGENT',
  'ANTIGRAVITY_AGENTAPI_EXE',
  'ANTIGRAVITY_APP_DATA_DIR',
  'ANTIGRAVITY_CONVERSATION_ID',
  'ANTIGRAVITY_CSRF_TOKEN',
  'ANTIGRAVITY_LS_ADDRESS',
  'ANTIGRAVITY_LS_VERSION',
  'ANTIGRAVITY_PROJECT_ID',
  'ANTIGRAVITY_SOURCE_METADATA',
  'ANTIGRAVITY_TRAJECTORY_ID',
] as const

const AGY_OUTER_SANDBOX_MARKER_KEYS = [
  'AGY_ACP_FORCE_FILE_STORAGE',
  'ANTIGRAVITY_AGENT',
  'ANTIGRAVITY_APP_DATA_DIR',
  'ANTIGRAVITY_LS_ADDRESS',
] as const

/**
 * SSH session indicator variables that cause `agy` to skip the OS keyring
 * (`Using file-based token storage because SSH session detected`).
 * `SSH_AUTH_SOCK` is intentionally excluded so SSH agent authentication keeps working.
 */
export const SSH_SESSION_ENV_KEYS = [
  'SSH_CONNECTION',
  'SSH_CLIENT',
  'SSH_TTY',
] as const

export function hasOuterAgyHostEnv(env: Readonly<Record<string, string | undefined>>): boolean {
  return AGY_OUTER_SANDBOX_MARKER_KEYS.some(key => {
    const value = env[key]
    return typeof value === 'string' && value.trim().length > 0
  })
}

export function hasSshSessionEnv(env: Readonly<Record<string, string | undefined>>): boolean {
  return SSH_SESSION_ENV_KEYS.some(key => {
    const value = env[key]
    return typeof value === 'string' && value.trim().length > 0
  })
}

/**
 * Build a sanitized environment for spawning a Xerxes-owned `agy acp` child.
 *
 * Explicitly assigns `undefined` for stripped keys so downstream merges with
 * `process.env` (such as `spawnBunAcpProcess`) omit them instead of inheriting
 * outer-host values.
 */
export function buildAgySubprocessEnv(
  baseEnv: Readonly<Record<string, string | undefined>>,
  options: { readonly stripSshSession?: boolean } = {},
): Record<string, string | undefined> {
  const sanitized: Record<string, string | undefined> = { ...baseEnv }
  const outerHostDetected = hasOuterAgyHostEnv(baseEnv)

  for (const key of AGY_OUTER_HOST_ENV_KEYS) {
    sanitized[key] = undefined
  }
  if (outerHostDetected) {
    sanitized.GEMINI_HOME = undefined
  }
  if (outerHostDetected || options.stripSshSession) {
    for (const key of SSH_SESSION_ENV_KEYS) {
      sanitized[key] = undefined
    }
  }
  return sanitized
}

const ALLOWED_SET: ReadonlySet<string> = new Set(AGY_ALLOWED_ENV_KEYS)

export function agyRuntimeEnvPath(home = homedir()): string {
  return join(home, '.xerxes', 'provider-accounts', 'agy', 'env.json')
}

/**
 * Load transferred `agy` credential env overrides for `agy acp`, if present.
 * Returns an empty object when no transferred env file exists or it is unreadable;
 * never logs or exposes secret values.
 */
export async function loadAgyRuntimeEnv(home = homedir()): Promise<Record<string, string>> {
  let raw: string
  try {
    raw = await readFile(agyRuntimeEnvPath(home), 'utf8')
  } catch {
    return {}
  }
  let parsed: unknown
  try {
    parsed = JSON.parse(raw)
  } catch {
    return {}
  }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return {}
  const env: Record<string, string> = {}
  for (const [key, value] of Object.entries(parsed as Record<string, unknown>)) {
    if (!ALLOWED_SET.has(key) || typeof value !== 'string') continue
    const trimmed = value.trim()
    if (trimmed === '' || trimmed.length > 16_384 || /[\0\r\n]/.test(trimmed)) continue
    env[key] = trimmed
  }
  return env
}
