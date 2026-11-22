// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

const COMMAND_OR_DIAGNOSTIC_TOOLS = new Set([
  'exec_command',
  'bash',
  'check_command',
  'lsp',
  'pty_open',
  'pty_write',
  'read_terminal_output',
])

const COMMAND_TOOL_HINTS = [
  'exec',
  'shell',
  'bash',
  'test',
  'pytest',
  'build',
  'npm',
  'cargo',
  'tsc',
  'lsp',
] as const

const PRIMARY_FAILURE_PATTERNS: readonly RegExp[] = [
  /Traceback \(most recent call last\):/,
  /={3,}\s*(?:FAILURES|ERRORS)\s*={3,}/,
  /\berror TS\d+:/,
  /\berror\[E\d+\]:/,
  /\b(?:AssertionError|SyntaxError|TypeError|ReferenceError|RangeError|URIError|EvalError|AggregateError)\b/,
  /\b[A-Z][A-Za-z0-9_]*(?:Error|Exception):/,
  /(?:^|\s)Error:\s+\S/,
  /\bpanic:\s/,
  /\bthread\s+'[^']*'\s+panicked at\b/,
  /^\s*(?:FAIL|FAILED)\s+\S/,
  /\sFAILED(?:\s+-|\s*\[|\s*$)/,
  /---\s*FAIL:\s+\S/,
  /^\s*(?:✕|✗|×|\(fail\))\s+\S/u,
  /^\s*ERROR\s+\S/,
  /^E\s{2,}\S/,
  /\bfatal(?:\s+error)?:/i,
  /\bsegmentation fault\b/i,
]

const SOURCE_CODE_ERROR_FALSE_POSITIVE_RE =
  /\b(?:throw\s+new|instanceof|extends|catch\s*\(|import\s+.*?\bfrom|export\s+(?:class|interface|type|function|const))\b/

const STACK_FRAME_PATTERNS: readonly RegExp[] = [
  // JS/TS/Node/Bun stack frames and path:line:col coordinates
  /^\s*at\s+(?:.+?\s+\()?[^\s()]+\.[A-Za-z0-9]{1,10}:\d+:\d+\)?/,
  /(?:^|[\s(])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,10}):(\d+):(\d+)\b/,
  // TypeScript paren coordinates: path/to/file.ts(184,12)
  /(?:^|[\s(])([A-Za-z0-9_./\\-]+\.[A-Za-z0-9]{1,10})\((\d+),(\d+)\)/,
  // Python traceback frame: File "path/to/file.py", line 184
  /^\s*File "([^"]+)", line (\d+)\b/,
  // Pytest frame: path/to/file.py:184: in func_name
  /^\s*[A-Za-z0-9_./\\-]+\.py:\d+:\s+in\s+\S+/,
  // Rust / Go diagnostic pointers
  /^\s*-->\s+\S+:\d+:\d+/,
]

const SUMMARY_LINE_RE =
  /(=+\s+.*?\s+=+|short test summary|\b\d+\s+(?:tests?\s+)?(?:passed|failed|pass|fail)\b|\bpassed in\b|\bfailed in\b|\bnpm ERR!|\bcargo test\b|\ball checks passed\b|\bRan \d+ tests\b|\bexit code\b|\bexited with code\b)/i

export interface SalientFailureHunk {
  /** 0-based inclusive start line index. */
  readonly start: number
  /** 0-based inclusive end line index. */
  readonly end: number
  /** Highest salience score among lines in this hunk (1..3). */
  readonly maxScore: number
  /** Total salience score across anchor lines in this hunk. */
  readonly totalScore: number
  /** Number of duplicate hunks collapsed into this one (0 if unique). */
  readonly duplicateCount: number
}

export interface FailureSalientCompressOptions {
  readonly maxChars: number
  readonly headLines?: number
  readonly tailLines?: number
  readonly contextLines?: number
  /** Banner line placed at the very top when lines are omitted (e.g. headroom preview header). */
  readonly headerBanner?: (stats: {
    readonly totalLines: number
    readonly keptLines: number
    readonly omittedLines: number
  }) => string
  /** Label used in `[... N <omissionLabel> ...]` gap markers. */
  readonly omissionLabel: string
  /** Include `[lines A-B]` headers on the head and tail blocks in addition to middle hunks. */
  readonly annotateHeadAndTailRanges?: boolean
  /** Guard against treating `throw new TypeError(...)` source code lines as runtime errors. */
  readonly strictSourceCodeGuard?: boolean
}

export function isHighSurrogateCode(code: number): boolean {
  return code >= 0xd800 && code <= 0xdbff
}

export function isLowSurrogateCode(code: number): boolean {
  return code >= 0xdc00 && code <= 0xdfff
}

/** Largest prefix end `<= maxEnd` that does not leave a trailing unpaired high surrogate. */
export function safePrefixEnd(text: string, maxEnd: number): number {
  const clamped = Math.max(0, Math.min(text.length, Math.floor(maxEnd)))
  if (clamped > 0 && clamped < text.length && isHighSurrogateCode(text.charCodeAt(clamped - 1))) {
    return clamped - 1
  }
  return clamped
}

/** Smallest suffix start `>= minStart` that does not begin on a leading unpaired low surrogate. */
export function safeSuffixStart(text: string, minStart: number): number {
  const clamped = Math.max(0, Math.min(text.length, Math.ceil(minStart)))
  if (clamped > 0 && clamped < text.length && isLowSurrogateCode(text.charCodeAt(clamped))) {
    return clamped + 1
  }
  return clamped
}

/** Slice at most `maxChars` UTF-16 code units from the start of `text` without splitting a surrogate pair. */
export function safeSlicePrefix(text: string, maxChars: number): string {
  if (maxChars <= 0 || text.length === 0) return ''
  if (text.length <= maxChars) return text
  return text.slice(0, safePrefixEnd(text, maxChars))
}

/** Slice at most `maxChars` UTF-16 code units from the end of `text` without splitting a surrogate pair. */
export function safeSliceSuffix(text: string, maxChars: number): string {
  if (maxChars <= 0 || text.length === 0) return ''
  if (text.length <= maxChars) return text
  return text.slice(safeSuffixStart(text, text.length - maxChars))
}

/** Slice `[start, end)` from `text` while adjusting both boundaries away from split surrogate pairs. */
export function safeSliceSubstring(text: string, start: number, end: number): string {
  if (end <= start || text.length === 0) return ''
  const safeStart = safeSuffixStart(text, Math.max(0, start))
  const safeEnd = safePrefixEnd(text, Math.min(text.length, end))
  return safeEnd > safeStart ? text.slice(safeStart, safeEnd) : ''
}

/** Fit `text` inside `maxChars` using a centered omission marker, guaranteed `<= maxChars` and surrogate-safe. */
export function fitSurrogateSafeText(
  text: string,
  maxChars: number,
  markerFactory: (omittedChars: number) => string,
): string {
  if (maxChars <= 0) return ''
  const trimmed = text.trimEnd()
  if (trimmed.length <= maxChars) return trimmed
  const roughMarker = markerFactory(Math.max(1, trimmed.length - maxChars))
  if (maxChars <= roughMarker.length + 6) {
    return safeSlicePrefix(trimmed, maxChars).trimEnd()
  }
  let available = Math.max(2, maxChars - roughMarker.length)
  for (let attempt = 0; attempt < 4; attempt += 1) {
    const headEnd = safePrefixEnd(trimmed, Math.ceil(available / 2))
    const tailStart = safeSuffixStart(trimmed, trimmed.length - Math.floor(available / 2))
    if (tailStart <= headEnd) return safeSlicePrefix(trimmed, maxChars).trimEnd()
    const headPart = trimmed.slice(0, headEnd).trimEnd()
    const tailPart = trimmed.slice(tailStart).trimStart()
    const omitted = Math.max(1, trimmed.length - headPart.length - tailPart.length)
    const marker = markerFactory(omitted)
    const candidate = headPart + marker + tailPart
    if (candidate.length <= maxChars) return candidate.trimEnd()
    available = Math.max(2, available - (candidate.length - maxChars) - 2)
  }
  return safeSlicePrefix(trimmed, maxChars).trimEnd()
}

export function isCommandOrDiagnosticTool(toolName: string | undefined): boolean {
  if (!toolName) return false
  const normalized = toolName.trim().toLowerCase()
  if (!normalized) return false
  if (COMMAND_OR_DIAGNOSTIC_TOOLS.has(normalized)) return true
  return COMMAND_TOOL_HINTS.some(hint => normalized.includes(hint))
}

/**
 * Score how strongly `line` signals a failure, exception, compiler diagnostic,
 * or stack frame:
 * - 3: primary error / assertion / traceback / failed test / compiler code
 * - 2: file:line:col stack frame or diagnostic location
 * - 1: test/build summary line
 * - 0: ordinary output
 */
export function lineSalienceScore(
  line: string,
  options: { readonly strictSourceCodeGuard?: boolean } = {},
): number {
  if (!line || line.length > 16_384) {
    const sample = line ? safeSlicePrefix(line, 4_096) : ''
    if (!sample) return 0
    return scoreSample(sample, options.strictSourceCodeGuard === true)
  }
  return scoreSample(line, options.strictSourceCodeGuard === true)
}

function scoreSample(line: string, strictSourceGuard: boolean): number {
  if (/^\s*(?:PASS\b|✓\s|ok\s+\S)/u.test(line)) return 0
  if (/\b0\s+(?:failed|fail|errors?)\b/i.test(line) && !/\b[1-9]\d*\s+(?:failed|fail|errors?)\b/i.test(line)) {
    return SUMMARY_LINE_RE.test(line) ? 1 : 0
  }
  const looksLikeCode = strictSourceGuard && SOURCE_CODE_ERROR_FALSE_POSITIVE_RE.test(line)
  if (!looksLikeCode && PRIMARY_FAILURE_PATTERNS.some(pattern => pattern.test(line))) {
    return 3
  }
  if (!looksLikeCode && STACK_FRAME_PATTERNS.some(pattern => pattern.test(line))) {
    return 2
  }
  if (SUMMARY_LINE_RE.test(line)) {
    return 1
  }
  return 0
}

export function hasSalientFailureLines(
  lines: readonly string[],
  options: { readonly strictSourceCodeGuard?: boolean } = {},
): boolean {
  let frameCount = 0
  for (const line of lines) {
    const score = lineSalienceScore(line, options)
    if (score >= 3) return true
    if (score === 2) {
      frameCount += 1
      if (frameCount >= 2) return true
    }
  }
  return false
}

/**
 * Extract merged, deduplicated failure hunks (`+-contextLines`) within
 * `[startIndex, endExclusive)`. Only hunks containing at least one primary
 * failure (score 3) or stack frame (score 2) are returned.
 */
export function extractSalientFailureHunks(
  lines: readonly string[],
  startIndex: number,
  endExclusive: number,
  options: {
    readonly contextLines?: number
    readonly strictSourceCodeGuard?: boolean
  } = {},
): SalientFailureHunk[] {
  const lower = Math.max(0, startIndex)
  const upper = Math.min(lines.length, endExclusive)
  if (upper <= lower) return []
  const context = Math.max(0, options.contextLines ?? 2)
  const strictOption = options.strictSourceCodeGuard === true ? { strictSourceCodeGuard: true } : {}

  interface RawSpan {
    start: number
    end: number
    maxScore: number
    totalScore: number
    anchorIndices: number[]
  }

  const rawSpans: RawSpan[] = []
  for (let index = lower; index < upper; index += 1) {
    const line = lines[index] ?? ''
    const score = lineSalienceScore(line, strictOption)
    if (score < 2) continue
    const spanStart = Math.max(lower, index - context)
    const spanEnd = Math.min(upper - 1, index + context)
    const previous = rawSpans.at(-1)
    if (previous !== undefined && spanStart <= previous.end + 2) {
      previous.end = Math.max(previous.end, spanEnd)
      previous.maxScore = Math.max(previous.maxScore, score)
      previous.totalScore += score
      previous.anchorIndices.push(index)
    } else {
      rawSpans.push({
        start: spanStart,
        end: spanEnd,
        maxScore: score,
        totalScore: score,
        anchorIndices: [index],
      })
    }
  }

  // Filter out isolated single-frame matches in non-command text unless backed by a primary error or multi-frame stack.
  const qualified = options.strictSourceCodeGuard === true
    ? rawSpans.filter(span => span.maxScore >= 3 || span.anchorIndices.length >= 2)
    : rawSpans

  const deduplicated: Array<{
    start: number
    end: number
    maxScore: number
    totalScore: number
    duplicateCount: number
  }> = []
  const seenBySignature = new Map<string, number>()

  for (const span of qualified) {
    const signature = hunkSignature(lines, span.anchorIndices)
    const existingIdx = seenBySignature.get(signature)
    if (existingIdx !== undefined) {
      const existing = deduplicated[existingIdx]
      if (existing !== undefined) {
        existing.duplicateCount += 1
      }
      continue
    }
    seenBySignature.set(signature, deduplicated.length)
    deduplicated.push({
      start: span.start,
      end: span.end,
      maxScore: span.maxScore,
      totalScore: span.totalScore,
      duplicateCount: 0,
    })
  }

  return deduplicated
}

function hunkSignature(lines: readonly string[], anchorIndices: readonly number[]): string {
  return anchorIndices
    .map(index => normalizeErrorLineForDedupe(lines[index] ?? ''))
    .filter(Boolean)
    .join('\n')
}

function normalizeErrorLineForDedupe(line: string): string {
  return safeSlicePrefix(line.trim(), 320)
    .replace(/\b0x[0-9a-fA-F]+\b/g, '0xADDR')
    .replace(/\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b/g, 'TIMESTAMP')
    .replace(/\[\d+(?:\.\d+)?\s*(?:ms|s)\]/gi, '[DURATION]')
    .replace(/\s+/g, ' ')
}

/** Clip an individual oversized line while keeping any salient error / file:line coordinate centered. */
function clipSingleLineAroundError(line: string, maxLineChars: number): string {
  if (line.length <= maxLineChars) return line
  if (maxLineChars <= 24) return safeSlicePrefix(line, maxLineChars)
  const ellipsis = ' ... '
  const available = Math.max(8, maxLineChars - ellipsis.length * 2)
  const matchOffset = findSalientOffset(line)
  if (matchOffset <= 0 || matchOffset < Math.floor(available * 0.6)) {
    const keepPrefix = Math.max(4, maxLineChars - ellipsis.length)
    return safeSlicePrefix(line, keepPrefix) + ellipsis
  }
  const half = Math.floor(available / 2)
  const start = Math.max(0, matchOffset - Math.floor(half * 0.4))
  const end = Math.min(line.length, start + available)
  const adjustedStart = Math.max(0, end - available)
  const snippet = safeSliceSubstring(line, adjustedStart, end)
  const prefix = adjustedStart > 0 ? ellipsis : ''
  const suffix = end < line.length ? ellipsis : ''
  const built = prefix + snippet + suffix
  return built.length <= maxLineChars ? built : safeSlicePrefix(built, maxLineChars)
}

function findSalientOffset(line: string): number {
  for (const pattern of [...PRIMARY_FAILURE_PATTERNS, ...STACK_FRAME_PATTERNS]) {
    const match = pattern.exec(line)
    if (match?.index !== undefined) return match.index
  }
  return -1
}

/**
 * Compress a potentially massive multi-line or single-line command/log/text
 * payload into `<= options.maxChars` UTF-16 code units, preserving:
 * 1. Invocation/header head,
 * 2. Deduplicated middle failure hunks annotated with exact 1-based `[lines X-Y]` coordinates,
 * 3. Summary tail.
 */
export function compressFailureSalientText(
  text: string,
  options: FailureSalientCompressOptions,
): string {
  const maxChars = Math.max(1, Math.trunc(options.maxChars))
  if (text.length <= maxChars && options.headerBanner === undefined) {
    return text.trimEnd()
  }

  const lines = text.split(/\r?\n/)
  const strictGuard = options.strictSourceCodeGuard === true
  const maxHeadLines = Math.max(0, options.headLines ?? 10)
  const maxTailLines = Math.max(0, options.tailLines ?? 14)

  // Very few giant lines (e.g. single-line 30k string): extract head + salient middle snippet + tail by char offset.
  if (lines.length <= Math.max(2, Math.min(4, maxHeadLines + maxTailLines))) {
    return compressFewGiantLines(text, maxChars, options.omissionLabel)
  }

  // Quick check whether the interior contains salient failures so we can size head/tail budgets appropriately.
  const roughHeadEnd = Math.min(maxHeadLines, Math.floor(lines.length / 3))
  const roughTailStart = Math.max(roughHeadEnd, lines.length - Math.min(maxTailLines, Math.floor(lines.length / 3)))
  const interiorHasFailures = hasSalientFailureLines(
    lines.slice(roughHeadEnd, roughTailStart),
    strictGuard ? { strictSourceCodeGuard: true } : {},
  )

  const annotateHeadTail = options.annotateHeadAndTailRanges === true
  const reserveBannerChars = options.headerBanner !== undefined ? 76 : 0
  const markerSample = `[... ${lines.length} ${options.omissionLabel} ...]`
  const baseFramingOverhead = reserveBannerChars + markerSample.length + 16
  const usableBudget = Math.max(16, maxChars - baseFramingOverhead)

  const headFraction = maxHeadLines === 0
    ? 0
    : maxTailLines === 0
      ? (interiorHasFailures ? 0.38 : 0.92)
      : (interiorHasFailures ? 0.22 : 0.5)
  const tailFraction = maxTailLines === 0
    ? 0
    : maxHeadLines === 0
      ? (interiorHasFailures ? 0.38 : 0.92)
      : (interiorHasFailures ? 0.26 : 0.46)

  const headCharCap = maxHeadLines === 0 ? 0 : Math.max(24, Math.floor(usableBudget * headFraction))
  const tailCharCap = maxTailLines === 0 ? 0 : Math.max(24, Math.floor(usableBudget * tailFraction))
  const perLineCap = Math.max(160, Math.floor(maxChars * 0.35))

  // Select head lines [0, headEnd)
  const headRendered: string[] = []
  let headCharsUsed = annotateHeadTail ? 18 : 0
  let headEnd = 0
  while (headEnd < Math.min(maxHeadLines, lines.length)) {
    const clipped = clipSingleLineAroundError(lines[headEnd] ?? '', Math.min(perLineCap, Math.max(24, headCharCap)))
    const added = (headRendered.length > 0 ? 1 : 0) + clipped.length
    if (headRendered.length > 0 && headCharsUsed + added > headCharCap) break
    if (headRendered.length === 0 && headCharsUsed + added > headCharCap) {
      const firstClipped = safeSlicePrefix(clipped, Math.max(8, headCharCap - headCharsUsed))
      if (firstClipped) {
        headRendered.push(firstClipped)
        headEnd = 1
      }
      break
    }
    headRendered.push(clipped)
    headCharsUsed += added
    headEnd += 1
  }

  // Select tail lines [tailStart, lines.length)
  let desiredTailLines = maxTailLines
  if (options.headerBanner !== undefined && maxTailLines > 0) {
    // Extend slightly if a test runner summary banner sits just above the default tail window.
    const scanFrom = Math.max(headEnd, lines.length - Math.max(maxTailLines, 26))
    for (let idx = scanFrom; idx < lines.length - maxTailLines; idx += 1) {
      if (SUMMARY_LINE_RE.test(lines[idx] ?? '')) {
        desiredTailLines = lines.length - idx
        break
      }
    }
  }

  const tailReversed: string[] = []
  let tailCharsUsed = annotateHeadTail ? 20 : 0
  let tailStart = lines.length
  const minTailIndex = headEnd
  while (tailStart > minTailIndex && lines.length - tailStart < desiredTailLines) {
    const candidateIndex = tailStart - 1
    const clipped = clipSingleLineAroundError(lines[candidateIndex] ?? '', Math.min(perLineCap, Math.max(24, tailCharCap)))
    const added = (tailReversed.length > 0 ? 1 : 0) + clipped.length
    if (tailReversed.length > 0 && tailCharsUsed + added > tailCharCap) break
    if (tailReversed.length === 0 && tailCharsUsed + added > tailCharCap) {
      const lastClipped = safeSliceSuffix(clipped, Math.max(8, tailCharCap - tailCharsUsed))
      if (lastClipped) {
        tailReversed.push(lastClipped)
        tailStart = candidateIndex
      }
      break
    }
    tailReversed.push(clipped)
    tailCharsUsed += added
    tailStart = candidateIndex
  }
  const tailRendered = tailReversed.reverse()

  // If everything fit without omitting any middle lines and fits maxChars, emit directly.
  if (headEnd >= tailStart) {
    const allLines = [...headRendered, ...tailRendered]
    const body = options.headerBanner !== undefined
      ? [
          options.headerBanner({ totalLines: lines.length, keptLines: allLines.length, omittedLines: 0 }),
          ...(annotateHeadTail && allLines.length > 0 ? [`[lines 1-${lines.length}]`] : []),
          ...allLines,
        ].join('\n')
      : allLines.join('\n')
    return fitSurrogateSafeText(body, maxChars, omitted => `\n\n[... ${omitted} chars omitted ...]\n\n`)
  }

  // Extract deduplicated salient failure hunks from the omitted middle [headEnd, tailStart).
  const candidateHunks = extractSalientFailureHunks(lines, headEnd, tailStart, {
    contextLines: options.contextLines ?? 2,
    ...(strictGuard ? { strictSourceCodeGuard: true } : {}),
  })

  // Rank hunks by maxScore DESC, then totalScore DESC, then earlier line order so primary root causes win budget first.
  const rankedHunks = [...candidateHunks].sort((left, right) =>
    right.maxScore - left.maxScore
    || right.totalScore - left.totalScore
    || left.start - right.start,
  )

  const usedFixedChars = reserveBannerChars + headCharsUsed + tailCharsUsed + markerSample.length + 12
  let remainingMiddleBudget = Math.max(0, maxChars - usedFixedChars)
  const hunkLineCap = Math.max(120, Math.min(perLineCap, Math.floor(Math.max(240, remainingMiddleBudget) * 0.55)))

  interface RenderedHunk {
    readonly start: number
    readonly end: number
    readonly text: string
    readonly lineCount: number
  }

  const chosenHunks: RenderedHunk[] = []
  for (const hunk of rankedHunks) {
    const rangeHeader = hunk.duplicateCount > 0
      ? `[lines ${hunk.start + 1}-${hunk.end + 1}] (+${hunk.duplicateCount} similar omitted)`
      : `[lines ${hunk.start + 1}-${hunk.end + 1}]`
    const hunkLines: string[] = [rangeHeader]
    for (let idx = hunk.start; idx <= hunk.end; idx += 1) {
      hunkLines.push(clipSingleLineAroundError(lines[idx] ?? '', hunkLineCap))
    }
    const renderedText = hunkLines.join('\n')
    const costWithGapMarker = renderedText.length + markerSample.length + 6
    if (costWithGapMarker <= remainingMiddleBudget) {
      chosenHunks.push({
        start: hunk.start,
        end: hunk.end,
        text: renderedText,
        lineCount: hunk.end - hunk.start + 1,
      })
      remainingMiddleBudget -= costWithGapMarker
    } else if (chosenHunks.length === 0 && remainingMiddleBudget >= rangeHeader.length + 40) {
      // At least fit the top failure's anchor lines within the remaining budget so the primary error is never lost.
      const compactText = fitSingleHunkToBudget(lines, hunk, rangeHeader, remainingMiddleBudget - markerSample.length - 4)
      if (compactText !== undefined) {
        chosenHunks.push(compactText)
        break
      }
    }
  }

  chosenHunks.sort((left, right) => left.start - right.start)

  const keptMiddleLines = chosenHunks.reduce((sum, item) => sum + item.lineCount, 0)
  const totalKeptLines = headEnd + (lines.length - tailStart) + keptMiddleLines
  const totalOmittedLines = Math.max(0, lines.length - totalKeptLines)

  // Assemble blocks using newline separators matched to caller style (single-newline for headroom log, double-newline for pruner when no middle hunks).
  const isHeadroomLog = options.headerBanner !== undefined
  const blocks: string[] = []
  if (options.headerBanner !== undefined) {
    blocks.push(options.headerBanner({
      totalLines: lines.length,
      keptLines: totalKeptLines,
      omittedLines: totalOmittedLines,
    }))
  }

  let cursor = 0
  if (headRendered.length > 0) {
    const headBlock = annotateHeadTail
      ? `[lines 1-${headEnd}]\n${headRendered.join('\n')}`
      : headRendered.join('\n')
    blocks.push(headBlock)
    cursor = headEnd
  }

  for (const hunk of chosenHunks) {
    const gap = hunk.start - cursor
    if (gap > 0) {
      blocks.push(`[... ${gap} ${options.omissionLabel} ...]`)
    }
    blocks.push(hunk.text)
    cursor = hunk.end + 1
  }

  const tailGap = tailStart - cursor
  if (tailGap > 0) {
    blocks.push(`[... ${tailGap} ${options.omissionLabel} ...]`)
  }

  if (tailRendered.length > 0) {
    const tailBlock = annotateHeadTail
      ? `[lines ${tailStart + 1}-${lines.length}]\n${tailRendered.join('\n')}`
      : tailRendered.join('\n')
    blocks.push(tailBlock)
  }

  const separator = isHeadroomLog ? '\n' : '\n\n'
  const assembled = blocks.join(separator).trimEnd()
  if (assembled.length <= maxChars) {
    return assembled
  }

  // Final strict `<= maxChars` guard: if chosenHunks exist, preserve middle if possible; otherwise surrogate-safe fit.
  if (chosenHunks.length > 0) {
    const primaryHunk = chosenHunks[0]!
    const hunkIdx = assembled.indexOf(primaryHunk.text)
    if (hunkIdx >= 0 && primaryHunk.text.length + 32 < maxChars) {
      const remaining = maxChars - primaryHunk.text.length - 4
      const headAllowance = Math.max(0, Math.floor(remaining / 2))
      const tailAllowance = Math.max(0, remaining - headAllowance)
      const before = safeSlicePrefix(assembled.slice(0, hunkIdx).trimEnd(), headAllowance)
      const after = safeSliceSuffix(assembled.slice(hunkIdx + primaryHunk.text.length).trimStart(), tailAllowance)
      const stitched = [before, primaryHunk.text, after].filter(Boolean).join('\n').trimEnd()
      if (stitched.length <= maxChars) return stitched
    }
  }

  return fitSurrogateSafeText(
    assembled,
    maxChars,
    omitted => `\n[... ${omitted} chars omitted ...]\n`,
  )
}

function fitSingleHunkToBudget(
  lines: readonly string[],
  hunk: SalientFailureHunk,
  rangeHeader: string,
  budgetChars: number,
): { readonly start: number; readonly end: number; readonly text: string; readonly lineCount: number } | undefined {
  if (budgetChars <= rangeHeader.length + 12) return undefined
  // Pick the highest-scoring line(s) in the hunk so the error headline + frame survive even in tight budgets.
  let bestIndex = hunk.start
  let bestScore = -1
  for (let idx = hunk.start; idx <= hunk.end; idx += 1) {
    const score = lineSalienceScore(lines[idx] ?? '')
    if (score > bestScore) {
      bestScore = score
      bestIndex = idx
    }
  }
  const subStart = Math.max(hunk.start, bestIndex - 1)
  const subEnd = Math.min(hunk.end, bestIndex + 1)
  const compactHeader = `[lines ${subStart + 1}-${subEnd + 1}]`
  const lineAllowance = Math.max(16, Math.floor((budgetChars - compactHeader.length - 4) / Math.max(1, subEnd - subStart + 1)))
  const selectedLines = [compactHeader]
  for (let idx = subStart; idx <= subEnd; idx += 1) {
    selectedLines.push(clipSingleLineAroundError(lines[idx] ?? '', lineAllowance))
  }
  const text = selectedLines.join('\n')
  if (text.length <= budgetChars) {
    return { start: subStart, end: subEnd, text, lineCount: subEnd - subStart + 1 }
  }
  const singleHeader = `[lines ${bestIndex + 1}-${bestIndex + 1}]`
  const singleBody = clipSingleLineAroundError(lines[bestIndex] ?? '', Math.max(12, budgetChars - singleHeader.length - 1))
  const singleText = `${singleHeader}\n${singleBody}`
  return singleText.length <= budgetChars
    ? { start: bestIndex, end: bestIndex, text: singleText, lineCount: 1 }
    : undefined
}

function compressFewGiantLines(text: string, maxChars: number, omissionLabel: string): string {
  const charLabel = omissionLabel.includes('pre-pruning') ? 'chars omitted by pre-pruning' : 'chars omitted'
  const matchOffset = findSalientOffset(text)
  const roughEdge = Math.floor(maxChars * 0.25)
  if (matchOffset > roughEdge && matchOffset < text.length - roughEdge && maxChars >= 160) {
    const markerA = `\n\n[... chars omitted ...]\n\n`
    const usable = Math.max(24, maxChars - markerA.length * 2 - 24)
    const headLen = Math.max(8, Math.floor(usable * 0.25))
    const tailLen = Math.max(8, Math.floor(usable * 0.25))
    const midLen = Math.max(16, usable - headLen - tailLen)
    const headEnd = safePrefixEnd(text, headLen)
    const tailStart = safeSuffixStart(text, text.length - tailLen)
    const midStart = Math.max(headEnd, Math.min(matchOffset - Math.floor(midLen * 0.25), tailStart - midLen))
    const midEnd = Math.min(tailStart, midStart + midLen)
    if (midEnd > midStart && midStart >= headEnd && tailStart >= midEnd) {
      const headPart = text.slice(0, headEnd)
      const midPart = safeSliceSubstring(text, midStart, midEnd)
      const tailPart = text.slice(tailStart)
      const gap1 = Math.max(1, midStart - headEnd)
      const gap2 = Math.max(1, tailStart - midEnd)
      const candidate =
        `${headPart}\n\n[... ${gap1} ${charLabel} ...]\n\n${midPart}\n\n[... ${gap2} ${charLabel} ...]\n\n${tailPart}`
      if (candidate.length <= maxChars) return candidate.trimEnd()
    }
  }
  return fitSurrogateSafeText(text, maxChars, omitted => `\n\n[... ${omitted} ${charLabel} ...]\n\n`)
}
