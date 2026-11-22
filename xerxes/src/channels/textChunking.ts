// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Slice `text` to at most `maxUnits` UTF-16 code units without leaving a
 * trailing unpaired high surrogate (`0xD800..0xDBFF`) when `text` had a valid
 * surrogate pair straddling `maxUnits`.
 */
export function sliceWithoutLoneSurrogate(text: string, maxUnits: number): string {
  if (maxUnits <= 0) return ''
  if (text.length <= maxUnits) return text
  let end = maxUnits
  if (isSurrogateBoundary(text, end)) {
    end -= 1
  }
  return text.slice(0, end)
}

/**
 * Split long text into provider-safe chunks without producing empty pieces,
 * splitting UTF-16 surrogate pairs, or leaving Markdown fenced code blocks
 * unclosed across chunk boundaries.
 *
 * Shared by channel adapters whose providers reject oversized messages
 * (Discord, Telegram). Newline boundaries below the limit are preferred so
 * Markdown and code blocks stay readable; every returned piece is at most
 * `limit` UTF-16 code units.
 */
export function chunkText(text: string, limit: number): string[] {
  if (!Number.isSafeInteger(limit) || limit < 1) {
    throw new RangeError('message chunk limit must be a positive safe integer')
  }
  if (text.length <= limit) {
    return [text]
  }
  const chunks: string[] = []
  let remaining = text
  let carryFence: ActiveFence | undefined
  while (remaining.length > limit) {
    const splitAt = findSplitIndex(remaining, limit)
    const openFence = activeCodeFence(remaining.slice(0, splitAt), carryFence)
    if (openFence !== undefined) {
      const openBody = openFence.bodyStart < splitAt ? remaining.slice(openFence.bodyStart, splitAt) : ''
      if (!openBody.trim() && openFence.fenceStart > 0) {
        const beforeFence = remaining.slice(0, openFence.fenceStart).replace(/[ \t\r\n]+$/, '')
        if (beforeFence.length > 0 && activeCodeFence(beforeFence, carryFence) === undefined) {
          chunks.push(beforeFence)
          remaining = remaining.slice(openFence.fenceStart)
          carryFence = undefined
          continue
        }
      }
      const provisionalMax = limit - (openFence.marker.length + 1)
      if (provisionalMax >= 2) {
        let fencedSplitAt = findSplitIndex(remaining, provisionalMax)
        let stillOpen = activeCodeFence(remaining.slice(0, fencedSplitAt), carryFence)
        // If stepping back landed inside an earlier fence with a longer marker, re-bound for that marker.
        if (stillOpen !== undefined && stillOpen.marker.length > openFence.marker.length) {
          const adjustedMax = limit - (stillOpen.marker.length + 1)
          if (adjustedMax >= 2) {
            fencedSplitAt = findSplitIndex(remaining, adjustedMax)
            stillOpen = activeCodeFence(remaining.slice(0, fencedSplitAt), carryFence)
          } else {
            stillOpen = undefined
          }
        }
        if (stillOpen !== undefined) {
          const closeSuffix = `\n${stillOpen.marker}`
          const head = remaining.slice(0, fencedSplitAt).replace(/[ \t\r\n]+$/, '')
          const body = stillOpen.bodyStart < head.length ? head.slice(stillOpen.bodyStart) : ''
          if (!body.trim() && stillOpen.fenceStart > 0) {
            const beforeStillOpen = remaining.slice(0, stillOpen.fenceStart).replace(/[ \t\r\n]+$/, '')
            if (beforeStillOpen.length > 0 && activeCodeFence(beforeStillOpen, carryFence) === undefined) {
              chunks.push(beforeStillOpen)
              remaining = remaining.slice(stillOpen.fenceStart)
              carryFence = undefined
              continue
            }
          }
          const minHeadLength = stillOpen.fenceStart === 0 && carryFence === undefined ? stillOpen.reopen.length : 0
          if (
            head.length > minHeadLength
            && body.trim().length > 0
            && head.length + closeSuffix.length <= limit
          ) {
            const rawTail = remaining.slice(fencedSplitAt).replace(/^\r?\n/, '')
            if (!rawTail.trim()) {
              chunks.push(`${head}${closeSuffix}`)
              remaining = ''
              carryFence = undefined
              break
            }
            const afterClosing = stripLeadingClosingFence(rawTail, stillOpen.marker)
            if (afterClosing !== undefined) {
              chunks.push(`${head}${closeSuffix}`)
              remaining = afterClosing
              carryFence = undefined
              if (!remaining) break
              continue
            }
            const nextRemaining = `${stillOpen.reopen}\n${rawTail}`
            if (nextRemaining.length < remaining.length) {
              chunks.push(`${head}${closeSuffix}`)
              remaining = nextRemaining
              carryFence = undefined
              continue
            }
          }
        }
      }
    }
    const chunk = remaining.slice(0, splitAt).trimEnd()
    if (chunk) {
      chunks.push(chunk)
    }
    remaining = remaining.slice(splitAt).trimStart()
    carryFence = openFence === undefined
      ? undefined
      : { marker: openFence.marker, reopen: openFence.reopen, fenceStart: 0, bodyStart: 0 }
  }
  if (remaining) {
    chunks.push(remaining)
  }
  return chunks.length ? chunks : ['']
}

interface ActiveFence {
  readonly marker: string
  readonly reopen: string
  readonly fenceStart: number
  readonly bodyStart: number
}

function findSplitIndex(text: string, limit: number): number {
  let splitAt = text.lastIndexOf('\n', limit)
  if (splitAt < Math.max(1, Math.floor(limit / 2))) {
    splitAt = limit
  }
  if (splitAt > 1 && isSurrogateBoundary(text, splitAt)) {
    splitAt -= 1
  }
  return splitAt
}

function isSurrogateBoundary(text: string, index: number): boolean {
  if (index <= 0 || index >= text.length) return false
  const high = text.charCodeAt(index - 1)
  const low = text.charCodeAt(index)
  return high >= 0xd800 && high <= 0xdbff && low >= 0xdc00 && low <= 0xdfff
}

function activeCodeFence(prefix: string, initial?: ActiveFence): ActiveFence | undefined {
  let open: ActiveFence | undefined = initial === undefined
    ? undefined
    : { ...initial, fenceStart: 0, bodyStart: 0 }
  let offset = 0
  while (offset <= prefix.length) {
    const nl = prefix.indexOf('\n', offset)
    const lineEnd = nl === -1 ? prefix.length : nl
    const rawLine = lineEnd > offset && prefix.charCodeAt(lineEnd - 1) === 0x0d
      ? prefix.slice(offset, lineEnd - 1)
      : prefix.slice(offset, lineEnd)
    const nextOffset = nl === -1 ? prefix.length + 1 : nl + 1

    if (open === undefined) {
      const match = /^[ ]{0,3}(`{3,}|~{3,})([^\r\n]*)$/.exec(rawLine)
      if (match) {
        const marker = match[1]!
        const info = match[2] ?? ''
        if (!(marker[0] === '`' && info.includes('`'))) {
          open = {
            marker,
            reopen: `${marker}${info.trimEnd()}`,
            fenceStart: offset,
            bodyStart: nl === -1 ? prefix.length : nl + 1,
          }
        }
      }
    } else {
      const match = /^[ ]{0,3}(`{3,}|~{3,})[ \t]*$/.exec(rawLine)
      if (match && match[1]![0] === open.marker[0] && match[1]!.length >= open.marker.length) {
        open = undefined
      }
    }

    if (nl === -1) break
    offset = nextOffset
  }
  return open
}

function stripLeadingClosingFence(text: string, marker: string): string | undefined {
  const nl = text.indexOf('\n')
  const firstLine = nl === -1 ? text : text.slice(0, nl)
  const trimmedCr = firstLine.endsWith('\r') ? firstLine.slice(0, -1) : firstLine
  const match = /^[ ]{0,3}(`{3,}|~{3,})[ \t]*$/.exec(trimmedCr)
  if (!match || match[1]![0] !== marker[0] || match[1]!.length < marker.length) {
    return undefined
  }
  return nl === -1 ? '' : text.slice(nl + 1).trimStart()
}
