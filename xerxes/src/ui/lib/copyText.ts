// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { Msg } from '../types.js'
import { writeClipboardText } from './clipboard.js'
import { writeOsc52Clipboard } from './osc52.js'
import { compactPreview } from './text.js'

export type CopyBackend = 'native' | 'osc52'

export interface CopyOutcome {
  backend: CopyBackend | null
  characters: number
}

export const COPY_USAGE = 'usage: /copy [n] | /copy user [n] | /copy last | /copy all  (bare /copy opens a picker)'

/**
 * XERXES_TUI_FORCE_OSC52=1 skips the native platform tools (pbcopy, wl-copy,
 * xclip, …) and goes straight to the terminal escape sequence. The /copy
 * error path advertised this escape hatch for a long time without anything
 * actually honoring it — this check is the implementation.
 */
export const forceOsc52Clipboard = (env: NodeJS.ProcessEnv = process.env): boolean => {
  const raw = env.XERXES_TUI_FORCE_OSC52?.trim().toLowerCase()

  return Boolean(raw) && raw !== '0' && raw !== 'false' && raw !== 'off'
}

export interface CopyTextDeps {
  env?: NodeJS.ProcessEnv
  native?: (text: string) => Promise<boolean>
  osc52?: (text: string) => boolean
}

/**
 * Write text to the user's clipboard, trying the native platform tools first
 * and falling back to an OSC 52 escape sequence. Reports honestly which
 * backend succeeded (or that both failed) instead of claiming success after
 * a dead fallback.
 */
export async function copyTextToClipboard(text: string, deps: CopyTextDeps = {}): Promise<CopyOutcome> {
  if (!text) {
    return { backend: null, characters: 0 }
  }

  const native = deps.native ?? writeClipboardText
  const osc52 = deps.osc52 ?? writeOsc52Clipboard

  if (!forceOsc52Clipboard(deps.env ?? process.env)) {
    let nativeOk = false

    try {
      nativeOk = await native(text)
    } catch {
      nativeOk = false
    }

    if (nativeOk) {
      return { backend: 'native', characters: text.length }
    }
  }

  let osc52Ok = false

  try {
    osc52Ok = osc52(text)
  } catch {
    osc52Ok = false
  }

  return { backend: osc52Ok ? 'osc52' : null, characters: text.length }
}

export function formatCopyOutcome(outcome: CopyOutcome): string {
  if (outcome.backend === 'native') {
    return `copied ${outcome.characters} characters`
  }

  if (outcome.backend === 'osc52') {
    return `sent OSC52 copy sequence (${outcome.characters} characters; terminal clipboard support required)`
  }

  return (
    'copy failed: no clipboard backend available (native tools and OSC52 both failed) — ' +
    'set XERXES_TUI_FORCE_OSC52=1 to skip native tools, or use /copy all with the pager'
  )
}

export interface CopyableMessage {
  /** 1-based position among messages of the same role ("/copy user 2"). */
  ordinal: number
  role: 'assistant' | 'user'
  text: string
}

/** Conversation messages that are meaningful to copy: non-empty user + assistant text. */
export function copyableMessages(items: readonly Msg[]): CopyableMessage[] {
  const out: CopyableMessage[] = []
  let assistantOrdinal = 0
  let userOrdinal = 0

  for (const item of items) {
    if ((item.role !== 'user' && item.role !== 'assistant') || !item.text.trim()) {
      continue
    }

    const ordinal = item.role === 'user' ? ++userOrdinal : ++assistantOrdinal

    out.push({ ordinal, role: item.role, text: item.text })
  }

  return out
}

export const copyRoleLabel = (message: CopyableMessage): string =>
  message.role === 'user' ? `you #${message.ordinal}` : `xerxes #${message.ordinal}`

/** Role-labeled plain-text rendering of the whole conversation for /copy all. */
export function formatTranscriptForCopy(items: readonly CopyableMessage[]): string {
  return items
    .map(message => {
      const tag = message.role === 'user' ? `You #${message.ordinal}` : `Xerxes #${message.ordinal}`

      return `[${tag}]\n${message.text.trim()}`
    })
    .join('\n\n')
}

/**
 * Single-line picker preview: collapse whitespace, clamp to max columns.
 * Kept as a named re-export so copy call sites read in their own vocabulary,
 * but there is exactly one implementation — see `compactPreview`.
 */
export const copyPreview = compactPreview

export type CopyResolution =
  | { kind: 'empty'; message: string }
  | { items: CopyableMessage[]; kind: 'picker' }
  | { kind: 'text'; text: string }
  | { kind: 'usage' }

const NOTHING_TO_COPY = 'nothing to copy — start a conversation first'

/**
 * Resolve a /copy argument against the copyable transcript. Kept pure so the
 * slash command, the picker, and tests all share the exact same semantics.
 */
export function resolveCopyArg(rawArg: string, items: readonly CopyableMessage[]): CopyResolution {
  const arg = rawArg.trim()

  if (!items.length) {
    return { kind: 'empty', message: NOTHING_TO_COPY }
  }

  if (!arg) {
    return { items: [...items], kind: 'picker' }
  }

  const parts = arg.toLowerCase().split(/\s+/)
  const head = parts[0]!

  if (head === 'all') {
    if (parts.length > 1) {
      return { kind: 'usage' }
    }

    return { kind: 'text', text: formatTranscriptForCopy(items) }
  }

  if (head === 'last') {
    if (parts.length > 1) {
      return { kind: 'usage' }
    }

    return { kind: 'text', text: items.at(-1)!.text }
  }

  if (head === 'user') {
    if (parts.length > 2) {
      return { kind: 'usage' }
    }

    const users = items.filter(message => message.role === 'user')

    if (!users.length) {
      return { kind: 'empty', message: 'no user messages to copy yet' }
    }

    if (parts.length === 2) {
      if (!/^\d+$/.test(parts[1]!) || parts[1] === '0') {
        return { kind: 'usage' }
      }

      const index = Math.min(parseInt(parts[1]!, 10), users.length) - 1

      return { kind: 'text', text: users[index]!.text }
    }

    return { kind: 'text', text: users.at(-1)!.text }
  }

  // Numeric-only form preserves the original semantics: nth assistant
  // message, clamped to the last one.
  if (parts.length > 1 || !/^\d+$/.test(head) || head === '0') {
    return { kind: 'usage' }
  }

  const assistants = items.filter(message => message.role === 'assistant')

  if (!assistants.length) {
    return { kind: 'empty', message: 'no assistant messages to copy yet' }
  }

  const index = Math.min(parseInt(head, 10), assistants.length) - 1

  return { kind: 'text', text: assistants[index]!.text }
}

/** Ctrl+O keybinding payload: copy the newest assistant message, returning the status line. */
export async function copyLatestAssistantMessage(
  items: readonly Msg[],
  copy: (text: string) => Promise<CopyOutcome> = copyTextToClipboard
): Promise<string> {
  const target = copyableMessages(items)
    .filter(message => message.role === 'assistant')
    .at(-1)

  if (!target) {
    return 'nothing to copy — no assistant message yet'
  }

  try {
    return formatCopyOutcome(await copy(target.text))
  } catch (error) {
    return `copy failed: ${String(error)}`
  }
}
