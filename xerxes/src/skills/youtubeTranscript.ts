// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { requireSkillText, skillFetchText, skillJsonArray, skillJsonObject, type SkillFetch, type SkillJsonObject } from './http.js'

/** A fetched caption segment in seconds, matching the original transcript skill's JSON shape. */
export interface YoutubeTranscriptSegment {
  readonly duration: number
  readonly start: number
  readonly text: string
}

export interface YoutubeTranscriptOptions {
  readonly languages?: readonly string[]
  readonly signal?: AbortSignal
}

export interface YoutubeTranscriptSummary {
  readonly duration: string
  readonly fullText: string
  readonly segmentCount: number
  readonly timestampedText?: string
  readonly videoId: string
}

export interface YoutubeTranscriptClientOptions {
  readonly fetchImplementation?: SkillFetch
  /** Endpoint used to discover caption tracks. Defaults to YouTube's public watch page. */
  readonly watchUrl?: string
}

/** Raised when YouTube has no caption track that can satisfy the request. */
export class YoutubeTranscriptUnavailableError extends Error {
  constructor(message: string, options: { readonly cause?: unknown } = {}) {
    super(message, options)
    this.name = 'YoutubeTranscriptUnavailableError'
  }
}

/** Extract a YouTube's eleven-character identifier from a URL or leave an unmatched input untouched. */
export function extractYoutubeVideoId(urlOrId: string): string {
  const candidate = requireSkillText(urlOrId, 'urlOrId')
  const match = /(?:[?&]v=|youtu\.be\/|\/shorts\/|\/embed\/|\/live\/)([A-Za-z0-9_-]{11})(?:[?&#/]|$)/.exec(candidate)
    ?? /^([A-Za-z0-9_-]{11})$/.exec(candidate)
  return match?.[1] ?? candidate
}

/** Format seconds as `H:MM:SS` or `M:SS`, exactly as the CLI's timestamp output did. */
export function formatYoutubeTimestamp(seconds: number): string {
  if (!Number.isFinite(seconds)) throw new RangeError('seconds must be finite')
  const total = Math.trunc(seconds)
  const hours = Math.trunc(total / 3_600)
  const afterHours = total % 3_600
  const minutes = Math.trunc(afterHours / 60)
  const remainingSeconds = afterHours % 60
  return hours > 0
    ? `${hours}:${String(minutes).padStart(2, '0')}:${String(remainingSeconds).padStart(2, '0')}`
    : `${minutes}:${String(remainingSeconds).padStart(2, '0')}`
}

/** Build the output payload used for JSON and text-only transcript views. */
export function summarizeYoutubeTranscript(
  videoId: string,
  segments: readonly YoutubeTranscriptSegment[],
  options: { readonly timestamps?: boolean } = {},
): YoutubeTranscriptSummary {
  const normalizedVideoId = requireSkillText(videoId, 'videoId')
  const fullText = segments.map(segment => segment.text).join(' ')
  const last = segments.at(-1)
  const duration = last === undefined ? '0:00' : formatYoutubeTimestamp(last.start + last.duration)
  const timestampedText = options.timestamps
    ? segments.map(segment => `${formatYoutubeTimestamp(segment.start)} ${segment.text}`).join('\n')
    : undefined
  return {
    duration,
    fullText,
    segmentCount: segments.length,
    ...(timestampedText === undefined ? {} : { timestampedText }),
    videoId: normalizedVideoId,
  }
}

/** Parse YouTube's `json3` caption response without a third-party Python dependency. */
export function parseYoutubeJson3Transcript(payload: string): readonly YoutubeTranscriptSegment[] {
  let parsed: unknown
  try {
    parsed = JSON.parse(payload) as unknown
  } catch (error) {
    throw new YoutubeTranscriptUnavailableError('caption response was not valid json3', { cause: error })
  }
  const root = skillJsonObject(parsed, 'YouTube json3 transcript')
  const events = skillJsonArray(root.events ?? [], 'YouTube json3 events')
  const segments: YoutubeTranscriptSegment[] = []
  for (const [index, event] of events.entries()) {
    const record = skillJsonObject(event, `YouTube json3 event ${index}`)
    const fragments = skillJsonArray(record.segs ?? [], `YouTube json3 event ${index} segments`)
      .map((fragment, fragmentIndex) => skillJsonObject(fragment, `YouTube json3 event ${index} segment ${fragmentIndex}`))
      .map(fragment => textValue(fragment.utf8))
      .join('')
    if (!fragments) continue
    const start = millisecondsToSeconds(record.tStartMs, `YouTube json3 event ${index} tStartMs`)
    const duration = millisecondsToSeconds(record.dDurationMs ?? 0, `YouTube json3 event ${index} dDurationMs`)
    segments.push({ duration, start, text: decodeHtml(fragments).replace(/\s+/g, ' ').trim() })
  }
  return segments
}

/** Parse the XML timed-text fallback returned by older caption tracks. */
export function parseYoutubeTimedText(payload: string): readonly YoutubeTranscriptSegment[] {
  const expression = /<text\b([^>]*)>([\s\S]*?)<\/text\s*>/gi
  const segments: YoutubeTranscriptSegment[] = []
  for (const match of payload.matchAll(expression)) {
    const attributes = match[1] ?? ''
    const start = numericAttribute(attributes, 'start')
    if (start === undefined) continue
    const duration = numericAttribute(attributes, 'dur') ?? 0
    segments.push({
      duration,
      start,
      text: decodeHtml(match[2] ?? '').replace(/\s+/g, ' ').trim(),
    })
  }
  return segments
}

/** Native client that discovers a public caption track and fetches it from YouTube. */
export class YoutubeTranscriptClient {
  private readonly fetchImplementation: SkillFetch
  private readonly watchUrl: string

  constructor(options: YoutubeTranscriptClientOptions = {}) {
    this.fetchImplementation = options.fetchImplementation ?? fetch
    this.watchUrl = (options.watchUrl ?? 'https://www.youtube.com/watch').replace(/\?+$/, '')
  }

  async fetchTranscript(videoOrId: string, options: YoutubeTranscriptOptions = {}): Promise<readonly YoutubeTranscriptSegment[]> {
    const videoId = extractYoutubeVideoId(videoOrId)
    const watch = new URL(this.watchUrl)
    watch.searchParams.set('v', videoId)
    const page = await skillFetchText(this.fetchImplementation, watch, {
      headers: {
        Accept: 'text/html,application/xhtml+xml',
        'Accept-Language': acceptedLanguages(options.languages),
        'User-Agent': 'Xerxes/1.0 (youtube-content skill)',
      },
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    })
    const tracks = captionTracks(extractYoutubePlayerResponse(page))
    const track = selectCaptionTrack(tracks, options.languages)
    if (track === undefined) {
      throw new YoutubeTranscriptUnavailableError(
        options.languages?.length
          ? `No transcript found for requested languages: ${options.languages.join(', ')}`
          : 'No transcript found for this video.',
      )
    }
    const captionsUrl = new URL(track.url.replaceAll('&amp;', '&'))
    captionsUrl.searchParams.set('fmt', 'json3')
    const transcript = await skillFetchText(this.fetchImplementation, captionsUrl, {
      headers: { Accept: 'application/json, text/xml;q=0.9', 'User-Agent': 'Xerxes/1.0 (youtube-content skill)' },
      ...(options.signal === undefined ? {} : { signal: options.signal }),
    })
    const segments = transcript.trimStart().startsWith('{')
      ? parseYoutubeJson3Transcript(transcript)
      : parseYoutubeTimedText(transcript)
    if (!segments.length) throw new YoutubeTranscriptUnavailableError('No transcript segments found for this video.')
    return segments
  }

  async summarize(
    videoOrId: string,
    options: YoutubeTranscriptOptions & { readonly timestamps?: boolean } = {},
  ): Promise<YoutubeTranscriptSummary> {
    const videoId = extractYoutubeVideoId(videoOrId)
    const segments = await this.fetchTranscript(videoId, options)
    return summarizeYoutubeTranscript(videoId, segments, {
      ...(options.timestamps === undefined ? {} : { timestamps: options.timestamps }),
    })
  }
}

interface CaptionTrack {
  readonly languageCode: string
  readonly url: string
}

function extractYoutubePlayerResponse(page: string): SkillJsonObject {
  const markers = ['ytInitialPlayerResponse =', 'var ytInitialPlayerResponse =', 'window["ytInitialPlayerResponse"] =']
  for (const marker of markers) {
    const start = page.indexOf(marker)
    if (start < 0) continue
    const objectStart = page.indexOf('{', start + marker.length)
    if (objectStart < 0) continue
    const json = balancedJsonObject(page, objectStart)
    if (json === undefined) continue
    try {
      return skillJsonObject(JSON.parse(json) as unknown, 'YouTube player response')
    } catch {
      // Try later known placement if a page script contains an unrelated malformed assignment.
    }
  }
  throw new YoutubeTranscriptUnavailableError('Unable to locate YouTube caption metadata.')
}

function balancedJsonObject(source: string, start: number): string | undefined {
  let depth = 0
  let escaped = false
  let quote: '"' | "'" | undefined
  for (let index = start; index < source.length; index += 1) {
    const character = source[index]
    if (character === undefined) break
    if (quote !== undefined) {
      if (escaped) {
        escaped = false
      } else if (character === '\\') {
        escaped = true
      } else if (character === quote) {
        quote = undefined
      }
      continue
    }
    if (character === '"' || character === "'") {
      quote = character
      continue
    }
    if (character === '{') depth += 1
    if (character === '}') {
      depth -= 1
      if (depth === 0) return source.slice(start, index + 1)
    }
  }
  return undefined
}

function captionTracks(response: SkillJsonObject): readonly CaptionTrack[] {
  const captions = optionalObject(response.captions)
  const renderer = captions === undefined ? undefined : optionalObject(captions.playerCaptionsTracklistRenderer)
  const tracks = renderer === undefined ? [] : skillJsonArray(renderer.captionTracks ?? [], 'YouTube caption tracks')
  return tracks.flatMap((value, index) => {
    const track = skillJsonObject(value, `YouTube caption track ${index}`)
    const url = textValue(track.baseUrl)
    return url ? [{ languageCode: textValue(track.languageCode), url }] : []
  })
}

function selectCaptionTrack(tracks: readonly CaptionTrack[], languages: readonly string[] | undefined): CaptionTrack | undefined {
  if (!languages?.length) return tracks[0]
  for (const requested of languages) {
    const normalized = requested.trim().toLowerCase()
    const exact = tracks.find(track => track.languageCode.toLowerCase() === normalized)
    if (exact !== undefined) return exact
  }
  return undefined
}

function optionalObject(value: unknown): SkillJsonObject | undefined {
  return value === null || typeof value !== 'object' || Array.isArray(value) ? undefined : value as SkillJsonObject
}

function textValue(value: unknown): string {
  return typeof value === 'string' ? value : ''
}

function millisecondsToSeconds(value: unknown, context: string): number {
  const numeric = typeof value === 'number' ? value : typeof value === 'string' ? Number(value) : Number.NaN
  if (!Number.isFinite(numeric)) throw new YoutubeTranscriptUnavailableError(`${context} must be numeric`)
  return numeric / 1_000
}

function numericAttribute(attributes: string, name: string): number | undefined {
  const expression = new RegExp(`\\b${name}\\s*=\\s*(["'])(.*?)\\1`, 'i')
  const raw = expression.exec(attributes)?.[2]
  if (raw === undefined) return undefined
  const value = Number(raw)
  return Number.isFinite(value) ? value : undefined
}

function acceptedLanguages(languages: readonly string[] | undefined): string {
  const normalized = languages?.map(language => language.trim()).filter(Boolean) ?? []
  return normalized.length ? normalized.join(',') : 'en-US,en;q=0.9'
}

function decodeHtml(value: string): string {
  return value
    .replaceAll('&amp;', '&')
    .replaceAll('&lt;', '<')
    .replaceAll('&gt;', '>')
    .replaceAll('&quot;', '"')
    .replaceAll('&#39;', "'")
    .replace(/&#(\d+);/g, (_match, decimal: string) => String.fromCodePoint(Number(decimal)))
    .replace(/&#x([\da-f]+);/gi, (_match, hexadecimal: string) => String.fromCodePoint(Number.parseInt(hexadecimal, 16)))
}
