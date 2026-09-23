// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { text, type RpcRecord } from './desktopRpc.js'

export interface ShellRow {
  readonly id: string
  readonly running: boolean
}

/** User shells are PTYs opened with no command; agent PTYs always carry one. */
export function userShells(terminals: readonly RpcRecord[]): ShellRow[] {
  return terminals
    .filter(row => row.kind === 'pty' && text(row.command) === '' && row.running === true && typeof row.id === 'string')
    .map(row => ({ id: row.id as string, running: true }))
}
