// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Browser-safe copy of hub/voice/speech.prepareSpeech; the Hub's existing import is intentionally unchanged. */
export function prepareCallSpeech(text: string, maxChars = 4000, sentenceChars = 240): readonly string[] {
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
    while (sentence.length > sentenceChars) {
      const space = sentence.lastIndexOf(' ', sentenceChars)
      const end = space > sentenceChars / 2 ? space : sentenceChars
      out.push(sentence.slice(0, end))
      sentence = sentence.slice(end).trim()
    }
    if (sentence) out.push(sentence)
  }
  return out
}
