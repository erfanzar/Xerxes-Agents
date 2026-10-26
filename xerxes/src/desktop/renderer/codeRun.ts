// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createContext } from 'react'

/**
 * Runs a code block's text in one of the person's own shells (Terminal tab).
 * Provided only around agent replies; elsewhere (release notes, plans) code
 * blocks keep just their Copy button.
 */
export type CodeRunner = (code: string) => Promise<void>
export const CodeRunContext = createContext<CodeRunner | undefined>(undefined)

/** Fence languages whose body is a shell command. An unlabelled block counts too. */
const RUNNABLE = new Set(['', 'text', 'sh', 'bash', 'zsh', 'shell', 'console', 'terminal', 'fish'])
export function isRunnableBlock(language: string, code: string): boolean {
  return RUNNABLE.has(language.trim().toLowerCase()) && code.trim() !== '' && code.length <= 16_384
}

/** `$ cmd` transcripts lose their prompt markers; plain commands are kept as they are. */
export function commandOf(code: string): string {
  const lines = code.replace(/\s+$/, '').split('\n')
  return lines.every(line => !line.trim() || /^\s*\$ /.test(line))
    ? lines.filter(line => line.trim()).map(line => line.replace(/^\s*\$ /, '')).join('\n')
    : lines.join('\n')
}
