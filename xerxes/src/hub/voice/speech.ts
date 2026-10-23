// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Turn supervisor-authored text into short speakable sentences. The full text
 * stays on screen; speech is deliberately concise: code blocks, bare URLs and
 * markdown markup are not read aloud, and speech stops at `maxChars`.
 */
export function prepareSpeech(text: string, maxChars: number): readonly string[] {
  const cleaned = text
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/https?:\/\/\S+/g, ' ')
    .replace(/^[#>\-*\s]+/gm, '')
    .replace(/[*_~]+/g, '')
    .replace(/\s+/g, ' ')
    .trim()
  if (cleaned === '') return []
  const sentences = cleaned.split(/(?<=[.!?])\s+/).filter(sentence => sentence.trim() !== '')
  const out: string[] = []
  let used = 0
  for (const sentence of sentences) {
    if (used + sentence.length > maxChars) {
      // Never drop everything because the first sentence is long: speak its start.
      if (out.length === 0) out.push(sentence.slice(0, maxChars))
      break
    }
    out.push(sentence)
    used += sentence.length + 1
  }
  return out
}
