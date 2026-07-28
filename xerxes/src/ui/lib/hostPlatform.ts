// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
//
// TUI-side copy of the host-OS facts in `core/hostPlatform.ts`.
//
// The duplication is deliberate and matches the existing precedent documented in
// `core/processLiveness.ts`: the TUI bundle compiles standalone under
// `rootDir: src/ui` and must not reach into runtime-side modules, the same
// reason `ui/protocol` shadows `protocol`.
//
// `test/windowsSupport.test.ts` asserts this file and its runtime twin derive
// identical control-channel addresses. That test is not ceremony: the TUI uses
// this copy to find the daemon and the daemon uses the other one to bind, so a
// one-character disagreement makes the TUI conclude no daemon is running and
// silently start a second one.

/** Windows named-pipe prefix accepted by libuv, and therefore by Bun's `node:net`. */
const NAMED_PIPE_PREFIX = '\\\\.\\pipe\\'

export function isWindows(platform: NodeJS.Platform = process.platform): boolean {
  return platform === 'win32'
}

/**
 * True for a Windows named pipe such as `\\.\pipe\xerxes-abc`.
 *
 * A pipe is a kernel object rather than a filesystem entry, so its parent must
 * not be created and it must not be unlinked before use.
 */
export function isNamedPipePath(path: string): boolean {
  return /^\\\\[.?]\\pipe\\/i.test(path)
}

/**
 * Derive the per-project control-channel address from a project digest.
 *
 * Must stay byte-for-byte identical to `controlChannelPath` in
 * `core/hostPlatform.ts`.
 */
export function controlChannelPath(
  socketDirectory: string,
  digest: string,
  platform: NodeJS.Platform = process.platform
): string {
  if (isWindows(platform)) {
    return `${NAMED_PIPE_PREFIX}xerxes-${digest}`
  }

  return `${socketDirectory}/${digest}.sock`
}
