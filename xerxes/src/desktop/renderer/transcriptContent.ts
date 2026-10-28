// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The context block Xerxes puts in front of a prompt for the model. Only the
 * model reads it: a message that still carries it (an old transcript, or one
 * loaded before its display text was recorded) is shown without it.
 */
const TURN_CONTEXT = /^\s*<turn-context>[\s\S]*?<\/turn-context>\s*/

export function withoutTurnContext(text: string): string {
  return text.replace(TURN_CONTEXT, '')
}

/**
 * Harness prompts saved before messages carried their origin: a goal
 * round's prompt. Never shown.
 */
export function isLegacyHarnessPrompt(text: string): boolean {
  const head = text.trimStart()
  if (/^Goal round \d+\/(?:\d+|unlimited) — /.test(head)) return true
  // Context the runtime injects (finished subagents, newly loaded tools, the
  // todo list, outside file changes), bare or wrapped in <system-reminder>.
  const body = head.startsWith('<system-reminder>') ? head.slice('<system-reminder>'.length).trimStart() : head
  return /^\[(?:sub-agent events|restored context|tools now available|files changed outside this session|todo list|Workspace guard|Objective gate)\]/.test(body)
}

/** The retry button's prompt, saved before its short display text was kept. */
const LEGACY_RETRY = /^Continue\. Your previous reply was cut off by an error \(/

function shownUserText(text: string): string {
  const clean = withoutTurnContext(text)
  return LEGACY_RETRY.test(clean) ? 'Continue' : clean
}

// Same marker the daemon strips (server.ts): steers from the phone or another person carry their
// display name, e.g. `[steer from Erfan]`, not only `[steer from user]`.
const STEER_PREFIX = /^\[(?:mid-turn steer from user|steer from [^\]\r\n]+?(?: saved for next turn)?)\](?:\r?\n|$)/

function withoutSteerPrefix(text: string): string {
  return text.replace(STEER_PREFIX, '')
}

/** The daemon's user-facing text can differ from the expanded provider input. */
export function transcriptContent(message: Record<string, unknown>): unknown {
  if (message.role === 'user') {
    for (const shown of [message.text, message.displayText]) {
      if (typeof shown === 'string' && shown.trim()) return shownUserText(shown)
    }
  }
  if (message.role === 'user' && typeof message.content === 'string') {
    return withoutSteerPrefix(shownUserText(message.content))
  }
  if (message.role === 'user' && Array.isArray(message.content)) {
    // A steer that carried a screenshot is stored as parts; its text part
    // holds the same framing prefix as a plain-text steer.
    return message.content.map(part => part && typeof part === 'object' && typeof (part as { text?: unknown }).text === 'string'
      ? { ...part, text: withoutSteerPrefix(shownUserText((part as { text: string }).text)) } : part)
  }
  return message.content
}
