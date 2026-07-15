// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

export const SUPPRESSED_OPEN_TAGS = [
  '<think>',
  '<thinking>',
  '<reasoning>',
  '<reasoning_scratchpad>',
  '<scratchpad>'
] as const

export const SUPPRESSED_CLOSE_TAGS = [
  '</think>',
  '</thinking>',
  '</reasoning>',
  '</reasoning_scratchpad>',
  '</scratchpad>'
] as const

const TAG_TAIL_LENGTH = 32

export interface ReasoningFilterOptions {
  caseInsensitive?: boolean
}

export interface ReasoningFilterOutput {
  thinking: string
  visible: string
}

interface TagMatch {
  index: number
  tag: string
}

/**
 * Incrementally suppresses provider reasoning blocks from visible streamed text.
 *
 * A short tail stays buffered so tags split across consecutive chunks never flash
 * into the live response. Suppressed content remains available through both the
 * returned `thinking` text and `thinkingLog`.
 */
export class ReasoningFilter {
  thinkingLog = ''

  private buffer = ''
  private inBlock = false
  private readonly caseInsensitive: boolean

  constructor({ caseInsensitive = true }: ReasoningFilterOptions = {}) {
    this.caseInsensitive = caseInsensitive
  }

  feed(chunk: string): ReasoningFilterOutput {
    this.buffer += chunk

    let thinking = ''
    let visible = ''

    while (this.buffer) {
      if (this.inBlock) {
        const match = this.findAny(this.buffer, SUPPRESSED_CLOSE_TAGS)

        if (!match) {
          if (this.buffer.length > TAG_TAIL_LENGTH) {
            thinking += this.buffer.slice(0, -TAG_TAIL_LENGTH)
            this.buffer = this.buffer.slice(-TAG_TAIL_LENGTH)
          }

          break
        }

        thinking += this.buffer.slice(0, match.index)
        this.buffer = this.buffer.slice(match.index + match.tag.length)
        this.inBlock = false

        continue
      }

      const match = this.findAny(this.buffer, SUPPRESSED_OPEN_TAGS)

      if (!match) {
        if (this.buffer.length > TAG_TAIL_LENGTH) {
          visible += this.buffer.slice(0, -TAG_TAIL_LENGTH)
          this.buffer = this.buffer.slice(-TAG_TAIL_LENGTH)
        }

        break
      }

      visible += this.buffer.slice(0, match.index)
      this.buffer = this.buffer.slice(match.index + match.tag.length)
      this.inBlock = true
    }

    this.thinkingLog += thinking

    return { thinking, visible }
  }

  flush(): ReasoningFilterOutput {
    const thinking = this.inBlock ? this.buffer : ''
    const visible = this.inBlock ? '' : this.buffer

    this.thinkingLog += thinking
    this.buffer = ''
    this.inBlock = false

    return { thinking, visible }
  }

  private findAny(text: string, tags: readonly string[]): null | TagMatch {
    const haystack = this.caseInsensitive ? text.toLowerCase() : text
    let earliest: null | TagMatch = null

    for (const tag of tags) {
      const index = haystack.indexOf(tag)

      if (index >= 0 && (earliest === null || index < earliest.index)) {
        earliest = { index, tag }
      }
    }

    return earliest
  }
}
