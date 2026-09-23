// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** The daemon's user-facing text can differ from the expanded provider input. */
export function transcriptContent(message: Record<string, unknown>): unknown {
  if (message.role === 'user' && typeof message.text === 'string' && message.text.trim()) {
    return message.text
  }
  if (message.role === 'user' && typeof message.content === 'string') {
    return message.content.replace(/^\[(?:mid-turn steer from user|steer from user(?: saved for next turn)?)\]\r?\n/, '')
  }
  return message.content
}
