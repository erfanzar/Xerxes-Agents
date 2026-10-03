// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * A saved SSH target: an `~/.ssh/config` alias or `user@host`, optionally
 * followed by `:port`. Nothing here reaches a shell; the target is passed to
 * `ssh` as a single argument after `--`.
 */
const SSH_TARGET = /^(?:[a-zA-Z0-9_][a-zA-Z0-9_.-]*@)?[a-zA-Z0-9_][a-zA-Z0-9_.-]*(?::([0-9]{1,5}))?$/

export function isSshTarget(value: unknown): value is string {
  if (typeof value !== 'string' || value.length > 255) return false
  const match = SSH_TARGET.exec(value)
  if (!match) return false
  const port = match[1]
  return port === undefined || (Number(port) >= 1 && Number(port) <= 65535)
}

/**
 * The destination argument for `ssh`. A target with a port becomes an
 * `ssh://` URI, which OpenSSH reads as host and port while still applying
 * the user's config for that host.
 */
export function sshDestination(target: string): string {
  return /:[0-9]+$/.test(target) ? `ssh://${target}` : target
}
