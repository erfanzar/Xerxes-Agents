// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { text, type RpcRecord } from './desktopRpc.js'

export interface ShellRow {
  readonly id: string
  readonly running: boolean
}

/** Label the runtime gives shells opened from this tab (daemon `terminal.open`). */
export const USER_SHELL_LABEL = 'User shell'

/** Shells you opened here, by label — agent PTYs are labelled with their command. */
export function userShells(terminals: readonly RpcRecord[]): ShellRow[] {
  return terminals
    .filter(row => row.kind === 'pty' && text(row.label) === USER_SHELL_LABEL && row.running === true && typeof row.id === 'string')
    .map(row => ({ id: row.id as string, running: true }))
}
