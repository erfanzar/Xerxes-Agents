// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Browser-safe copy of hub/voice/speech.prepareSpeech; the Hub's existing import is intentionally unchanged. */
export function prepareCallSpeech(text: string, maxChars = 4000, sentenceChars = 240, firstChars = sentenceChars): readonly string[] {
  const cleaned = text
    .replace(/```[\s\S]*?```/g, ' ')
    .replace(/`([^`]*)`/g, '$1')
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    .replace(/https?:\/\/\S+/g, ' ')
    .replace(/^[#>\-*\s]+/gm, '')
    .replace(/[*_~]+/g, '')
    .replace(/\s+/g, ' ')
    .trim().slice(0, maxChars)
  if (cleaned === '') return []
  const out: string[] = []
  for (let sentence of cleaned.split(/(?<=[.!?])\s+/u)) {
    // Bound each inference so the phone can start speaking early and cancel between sentences.
    // The first chunk may be shorter, so the first audio is ready sooner; it prefers a comma.
    for (let limit = out.length === 0 ? firstChars : sentenceChars; sentence.length > limit; limit = sentenceChars) {
      const comma = out.length === 0 ? sentence.lastIndexOf(', ', limit) : -1
      const space = sentence.lastIndexOf(' ', limit)
      const end = comma > limit / 3 ? comma + 1 : space > limit / 2 ? space : limit
      out.push(sentence.slice(0, end))
      sentence = sentence.slice(end).trim()
    }
    if (sentence) out.push(sentence)
  }
  return out
}
