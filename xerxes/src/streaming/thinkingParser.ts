// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export type ThinkingPart = { readonly text: string; readonly type: 'text' | 'thinking' }

/** Minimal incremental parser contract shared by production and diagnostic loops. */
export interface ThinkingStreamParser {
  process(chunk: string): readonly ThinkingPart[]
}

/**
 * Incrementally split `<think>` and `<thinking>` tags across arbitrary stream chunks.
 *
 * Every provider's text runs through this, so a reply that merely mentions
 * the tag must stay text: one in inline code (next to a backtick) is never a
 * boundary, and a block left unclosed at the end of the stream is reasoning
 * only when its tag opened a line. Otherwise "wraps its reasoning in <think>
 * tags…" moved the rest of the answer, summary included, into thinking.
 */
export class ThinkingParser implements ThinkingStreamParser {
  private buffer = ''
  private inThinking = false
  private thinkingBuffer = ''
  private openTag = ''
  private openedAtLineStart = false
  /** The text emitted so far on the current line is only whitespace. */
  private lineBlank = true
  private lastChar = ''

  private static readonly closeTags = ['</think>', '</thinking>'] as const
  private static readonly openTags = ['<think>', '<thinking>'] as const

  process(chunk: string): ThinkingPart[] {
    const events: ThinkingPart[] = []
    this.buffer += chunk
    const finalFlush = !chunk

    while (this.buffer) {
      if (!this.inThinking) {
        const [index, tag] = findAny(this.buffer, ThinkingParser.openTags)
        if (index < 0) {
          const hold = finalFlush ? 0 : partialTail(this.buffer, ThinkingParser.openTags)
          this.text(events, hold ? this.buffer.slice(0, -hold) : this.buffer)
          this.buffer = hold ? this.buffer.slice(-hold) : ''
          break
        }
        const after = this.buffer[index + tag.length]
        if (after === undefined && !finalFlush) {
          // The next character decides whether the tag is inline code.
          this.text(events, this.buffer.slice(0, index))
          this.buffer = this.buffer.slice(index)
          break
        }
        const before = index > 0 ? this.buffer[index - 1] : this.lastChar
        if (before === '`' || after === '`') {
          this.text(events, this.buffer.slice(0, index + tag.length))
          this.buffer = this.buffer.slice(index + tag.length)
          continue
        }
        this.text(events, this.buffer.slice(0, index))
        this.buffer = this.buffer.slice(index + tag.length)
        this.inThinking = true
        this.openTag = tag
        this.openedAtLineStart = this.lineBlank
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

    if (finalFlush && this.inThinking) {
      // Unclosed at the end: reasoning cut short when its tag opened a line,
      // otherwise a mention whose text is handed back unchanged.
      if (this.openedAtLineStart) {
        if (this.thinkingBuffer) events.push({ type: 'thinking', text: this.thinkingBuffer })
      } else {
        this.text(events, this.openTag + this.thinkingBuffer)
      }
      this.thinkingBuffer = ''
      this.inThinking = false
    }
    return events
  }

  private text(events: ThinkingPart[], text: string): void {
    if (!text) return
    events.push({ type: 'text', text })
    const newline = text.lastIndexOf('\n')
    const line = newline < 0 ? text : text.slice(newline + 1)
    this.lineBlank = (newline >= 0 || this.lineBlank) && !line.trim()
    this.lastChar = text.at(-1)!
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
