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
  return /^Goal round \d+\/(?:\d+|unlimited) — /.test(text.trimStart())
}

/** The retry button's prompt, saved before its short display text was kept. */
const LEGACY_RETRY = /^Continue\. Your previous reply was cut off by an error \(/

function shownUserText(text: string): string {
  const clean = withoutTurnContext(text)
  return LEGACY_RETRY.test(clean) ? 'Continue' : clean
}

/** The daemon's user-facing text can differ from the expanded provider input. */
export function transcriptContent(message: Record<string, unknown>): unknown {
  if (message.role === 'user') {
    for (const shown of [message.text, message.displayText]) {
      if (typeof shown === 'string' && shown.trim()) return shownUserText(shown)
    }
  }
  if (message.role === 'user' && typeof message.content === 'string') {
    return shownUserText(message.content).replace(/^\[(?:mid-turn steer from user|steer from user(?: saved for next turn)?)\]\r?\n/, '')
  }
  if (message.role === 'user' && Array.isArray(message.content)) {
    return message.content.map(part => part && typeof part === 'object' && typeof (part as { text?: unknown }).text === 'string'
      ? { ...part, text: shownUserText((part as { text: string }).text) } : part)
  }
  return message.content
}
