// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type ThinkingPart = { readonly text: string; readonly type: 'text' | 'thinking' }

/** Minimal incremental parser contract shared by production and diagnostic loops. */
export interface ThinkingStreamParser {
  process(chunk: string): readonly ThinkingPart[]
}

/** Incrementally split `<think>` and `<thinking>` tags across arbitrary stream chunks. */
export class ThinkingParser implements ThinkingStreamParser {
  private buffer = ''
  private inThinking = false
  private thinkingBuffer = ''

  private static readonly closeTags = ['</think>', '</thinking>'] as const
  private static readonly openTags = ['<think>', '<thinking>'] as const

  process(chunk: string): ThinkingPart[] {
    const events: ThinkingPart[] = []
    this.buffer += chunk
    const finalFlush = !chunk

    if (finalFlush && this.inThinking) {
      this.thinkingBuffer += this.buffer
      this.buffer = ''
      if (this.thinkingBuffer) {
        events.push({ type: 'thinking', text: this.thinkingBuffer })
      }
      this.thinkingBuffer = ''
      this.inThinking = false
      return events
    }

    while (this.buffer) {
      if (!this.inThinking) {
        const [index, tag] = findAny(this.buffer, ThinkingParser.openTags)
        if (index < 0) {
          const hold = finalFlush ? 0 : partialTail(this.buffer, ThinkingParser.openTags)
          const visible = hold ? this.buffer.slice(0, -hold) : this.buffer
          if (visible) {
            events.push({ type: 'text', text: visible })
          }
          this.buffer = hold ? this.buffer.slice(-hold) : ''
          break
        }
        if (index > 0) {
          events.push({ type: 'text', text: this.buffer.slice(0, index) })
        }
        this.buffer = this.buffer.slice(index + tag.length)
        this.inThinking = true
        this.thinkingBuffer = ''
        continue
      }

      const [index, tag] = findAny(this.buffer, ThinkingParser.closeTags)
      if (index < 0) {
        const hold = finalFlush ? 0 : partialTail(this.buffer, ThinkingParser.closeTags)
        this.thinkingBuffer += hold ? this.buffer.slice(0, -hold) : this.buffer
        this.buffer = hold ? this.buffer.slice(-hold) : ''
        break
      }
      if (index > 0) {
        this.thinkingBuffer += this.buffer.slice(0, index)
      }
      this.buffer = this.buffer.slice(index + tag.length)
      this.inThinking = false
      if (this.thinkingBuffer) {
        events.push({ type: 'thinking', text: this.thinkingBuffer })
        this.thinkingBuffer = ''
      }
    }

    return events
  }
}

function findAny(value: string, tags: readonly string[]): readonly [number, string] {
  let earliest = -1
  let matched = ''
  for (const tag of tags) {
    const index = value.indexOf(tag)
    if (index >= 0 && (earliest < 0 || index < earliest)) {
      earliest = index
      matched = tag
    }
  }
  return [earliest, matched]
}

function partialTail(value: string, tags: readonly string[]): number {
  let longest = 0
  for (const tag of tags) {
    for (let size = Math.min(value.length, tag.length - 1); size > 0; size -= 1) {
      if (value.endsWith(tag.slice(0, size))) {
        longest = Math.max(longest, size)
        break
      }
    }
  }
  return longest
}

export function splitThinkingTags(value: string): { readonly thinking: string; readonly visible: string } {
  const parser = new ThinkingParser()
  const parts = [...parser.process(value), ...parser.process('')]
  return {
    visible: parts.filter(part => part.type === 'text').map(part => part.text).join(''),
    thinking: parts.filter(part => part.type === 'thinking').map(part => part.text).join('').trim(),
  }
}
