// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// OpenTUI message renderer. One Msg becomes one flat transcript block.
// Assistant text renders through OpenTUI's native <markdown> (tables, code,
// emphasis). Tool-call trail lines stay compact like Grok's transcript: the
// call is always visible, while diagnostic and diff detail remains available
// when it carries information the one-line result cannot safely summarize.
import { useStore } from '@nanostores/react'
import { memo, useRef } from 'react'

import {
  $thinkingVisibility,
  thinkingRowExpanded,
  toggleThinkingRow
} from '../app/thinkingVisibilityStore.js'
import { $uiDetailVisibility } from '../app/uiStore.js'
import { VOICE } from '../domain/roles.js'
import { messageHasVisibleDetails, trailHasRenderableContent } from '../lib/liveProgress.js'
import { estimateTokensRough, fmtK, parseToolTrailResultLine, toolTrailParts } from '../lib/text.js'
import { splitStreamingRender, STREAMING_CHUNKS_EMPTY, type StreamingChunks } from '../lib/streamingMarkdown.js'
import type { Theme } from '../theme.js'
import type { Msg } from '../types.js'

import { Box, Span, Text } from './primitives.js'
import { getSyntaxStyle } from './syntax.js'

const ERROR_DETAIL_MAX_LINES = 16
const DIFF_DETAIL_MAX_LINES = 20
const TOOL_RESULT_MAX_LINES = 12
const TOOL_RESULT_COMPACT_CHARS = 180

const TABLE_OPTIONS = {
  borderStyle: 'rounded' as const,
  borders: true,
  cellPadding: 1,
  columnFitter: 'balanced' as const,
  outerBorder: true,
  widthMode: 'full' as const,
  wrapMode: 'word' as const
}

function Markdown({ content, t }: { content: string; t: Theme }) {
  return (
    <markdown
      conceal
      content={content}
      flexShrink={0}
      selectable
      syntaxStyle={getSyntaxStyle(t)}
      tableOptions={{ ...TABLE_OPTIONS, borderColor: t.color.border }}
    />
  )
}

const MarkdownChunk = memo(Markdown)

/**
 * Live-streaming assistant text. Re-parsing the whole growing buffer into one
 * native <markdown> element on every delta is O(total) per delta, so the
 * fence-aware chunker (lib/streamingMarkdown) freezes completed blocks into
 * memoized chunks that never re-parse; only the unstable tail does.
 *
 * Visual parity with a single whole-buffer <markdown>: a lone document
 * inserts one blank row between top-level blocks, so chunk elements are
 * interleaved with a one-row spacer and each chunk is stripped of its
 * trailing blank line by the chunker (OpenTUI preserves a trailing "\n\n"
 * after heading blocks, which would otherwise double the spacing).
 */
export function StreamingMarkdown({ text, t }: { text: string; t: Theme }) {
  const stateRef = useRef<StreamingChunks>(STREAMING_CHUNKS_EMPTY)
  const render = splitStreamingRender(text, stateRef.current)
  stateRef.current = render.state

  return (
    <>
      {render.chunks.map((chunk, index) => (
        <Box flexDirection="column" flexShrink={0} key={index}>
          <MarkdownChunk content={chunk} t={t} />
          <Box flexShrink={0} height={1} />
        </Box>
      ))}
      {render.tail ? <Markdown content={render.tail} t={t} /> : null}
    </>
  )
}

function UserMessage({ msg, t }: { msg: Msg; t: Theme }) {
  return (
    // One blank row above the band, none below: the next block adds its own
    // lead gap when the voice changes. This used to paint 2 rows above and 3
    // below every turn, and the estimator only ever predicted 2.
    <Box flexDirection="row" flexShrink={0} marginTop={1}>
      <Box backgroundColor={VOICE.user(t).bar} flexShrink={0} width={1} />
      <Box
        backgroundColor={t.color.completionBg}
        flexDirection="column"
        flexGrow={1}
        paddingLeft={2}
        paddingRight={1}
      >
        <Text color={VOICE.user(t).body} wrap="wrap">
          {msg.text}
        </Text>
      </Box>
    </Box>
  )
}

function AssistantMessage({ leadGap, msg, t }: { leadGap?: boolean; msg: Msg; t: Theme }) {
  return (
    <Box flexDirection="column" flexShrink={0} marginTop={leadGap ? 1 : 0} paddingLeft={3}>
      <Markdown content={msg.text} t={t} />
    </Box>
  )
}

function SystemMessage({ msg, t }: { msg: Msg; t: Theme }) {
  return (
    <Box flexShrink={0} paddingLeft={2}>
      <Text color={VOICE.system(t).body} wrap="wrap">
        <Span color={VOICE.system(t).glyphColor}>{VOICE.system(t).glyph} </Span>
        {msg.text}
      </Text>
    </Box>
  )
}

function ToolResultMessage({ msg, t }: { msg: Msg; t: Theme }) {
  const text = msg.text.trim()
  const lines = text.split('\n')
  const first = lines.find(line => line.trim())?.trim() ?? ''
  const diagnostic = /^(?:error|exception|failed|failure|denied|fatal)(?:\b|:)/i.test(first)
  const diff = looksLikeDiff(lines)

  // Successful tool return values are already represented by their compact
  // chronological tool row. Rendering the protocol payload as a second block
  // is what produced the screenfuls of Args/Result JSON in the old view.
  if (!diagnostic && !diff) {
    return null
  }

  const maxLines = diff ? DIFF_DETAIL_MAX_LINES : diagnostic ? TOOL_RESULT_MAX_LINES : 1
  const shown = lines.slice(0, maxLines)
  let preview = shown.join('\n')
  if (!diff && !diagnostic && preview.length > TOOL_RESULT_COMPACT_CHARS) {
    preview = `${preview.slice(0, TOOL_RESULT_COMPACT_CHARS)}…`
  } else if (lines.length > shown.length) {
    preview += `\n… +${lines.length - shown.length} more line${lines.length - shown.length === 1 ? '' : 's'}`
  }

  return (
    <Box flexDirection="column" flexShrink={0} paddingLeft={3}>
      {preview ? (
        preview.split('\n').map((line, i) => (
          <Text
            color={toolDetailColor(line, diagnostic, t)}
            dimColor={!diagnostic && !isDiffLine(line)}
            key={i}
            wrap="wrap"
          >
            {i === 0 ? '→ ' : '  '}
            {line || ' '}
          </Text>
        ))
      ) : (
        <Text color={t.color.muted} dimColor>
          → (empty tool result)
        </Text>
      )}
    </Box>
  )
}

function isDiffLine(line: string): boolean {
  return /^(?:diff --git |index |@@ |--- |\+\+\+ |[-+](?![-+]))/.test(line)
}

function looksLikeDiff(lines: readonly string[]): boolean {
  return lines.some(isDiffLine)
}

function usefulToolDetail(detail: string, failed: boolean): { diagnostic: boolean; lines: string[]; overflow: number } {
  if (!detail) {
    return { diagnostic: failed, lines: [], overflow: 0 }
  }

  const all = detail.split('\n')
  if (failed) {
    const errorStart = all.findIndex(line => /^Error:\s*$/.test(line))
    const relevant = errorStart >= 0 ? all.slice(errorStart) : all
    const lines = relevant.slice(0, ERROR_DETAIL_MAX_LINES)

    return { diagnostic: true, lines, overflow: relevant.length - lines.length }
  }

  const diffStart = all.findIndex(isDiffLine)
  if (diffStart >= 0) {
    const relevant = all.slice(diffStart)
    const lines = relevant.slice(0, DIFF_DETAIL_MAX_LINES)

    return { diagnostic: false, lines, overflow: relevant.length - lines.length }
  }

  return { diagnostic: false, lines: [], overflow: 0 }
}

function toolDetailColor(line: string, diagnostic: boolean, t: Theme): string {
  if (/^\+(?!\+\+)/.test(line)) {
    return t.color.diffAddedWord
  }
  if (/^-(?!---)/.test(line)) {
    return t.color.diffRemovedWord
  }
  if (diagnostic && !/^Args:\s*$/.test(line)) {
    return t.color.error
  }

  return t.color.muted
}

function ToolStep({ line, t }: { line: string; t: Theme }) {
  const parsed = parseToolTrailResultLine(line)

  const voice = VOICE.tool(t)

  if (!parsed) {
    // In-flight / transient call line ("drafting …", a bare tool name). No
    // mark and a muted glyph, so "still running" reads differently from a
    // settled row at a glance.
    return (
      <Text color={voice.body} wrap="truncate-end">
        <Span color={t.color.muted}>{voice.glyph} </Span>
        {line}
      </Text>
    )
  }

  const failed = parsed.mark === '✗'
  const markColor = failed ? t.color.error : t.color.ok
  const detail = usefulToolDetail(parsed.detail, failed)
  const { args, duration, name } = toolTrailParts(parsed.call)

  return (
    <Box flexDirection="column" flexShrink={0}>
      {/* One line, styled by part: the tool's name reads as a name, its
          arguments recede, and the duration is available at a glance instead
          of being stripped out. All three were previously flattened into a
          single muted string. */}
      <Text color={voice.body} wrap="truncate-end">
        <Span color={failed ? t.color.error : voice.glyphColor}>{voice.glyph} </Span>
        <Span bold color={failed ? t.color.error : voice.glyphColor}>
          {name}
        </Span>
        {args ? <Span color={t.color.muted}>{`  ${args}`}</Span> : null}
        {duration ? (
          <Span color={t.color.muted} dimColor>
            {`  · ${duration}`}
          </Span>
        ) : null}
        {/* Paint the tick too. Rendering only '✗' left success visually
            identical to a call that is still running. */}
        {parsed.mark ? (
          <>
            {' '}
            <Span color={markColor} dimColor={!failed}>
              {parsed.mark}
            </Span>
          </>
        ) : null}
      </Text>
      {detail.lines.map((d, i) => (
        <Text
          color={toolDetailColor(d, detail.diagnostic, t)}
          dimColor={!detail.diagnostic && !isDiffLine(d)}
          key={i}
          wrap="wrap"
        >
          {'  '}
          {d || ' '}
        </Text>
      ))}
      {detail.lines.length > 0 && detail.overflow > 0 ? (
        <Text color={t.color.muted} dimColor>
          {'  '}… +{detail.overflow} more line{detail.overflow === 1 ? '' : 's'}
        </Text>
      ) : null}
    </Box>
  )
}

interface DetailVisibility {
  subagents: boolean
  thinking: boolean
  tools: boolean
}

const detailVisibility = (snapshot: string): DetailVisibility => {
  const [thinking, tools, subagents] = snapshot.split(':')

  return { subagents: subagents === 'true', thinking: thinking === 'true', tools: tools === 'true' }
}

// Stable per-message identity for thinking toggles when the caller has no
// row key (direct MessageLine consumers). Settled transcript rows should
// pass their virtual row key so the toggle survives virtualization; live
// stream segments pass their segment key so it survives per-delta Msg
// object replacement.
const fallbackRowIds = new WeakMap<Msg, string>()
let fallbackRowIdSeq = 0

const thinkingRowId = (msg: Msg, msgKey?: string): string => {
  if (msgKey) {
    return msgKey
  }

  let id = fallbackRowIds.get(msg)

  if (!id) {
    id = `thinking:${++fallbackRowIdSeq}`
    fallbackRowIds.set(msg, id)
  }

  return id
}

function ThinkingBlock({ msg, rowId, t }: { msg: Msg; rowId: string; t: Theme }) {
  const visibility = useStore($thinkingVisibility)
  const expanded = thinkingRowExpanded(visibility, rowId)
  const thinking = msg.thinking?.trim() ?? ''
  const tokens = msg.thinkingTokens && msg.thinkingTokens > 0 ? msg.thinkingTokens : estimateTokensRough(thinking)
  const tokenLabel = tokens > 0 ? `  ~${fmtK(tokens)} tokens` : ''

  return (
    <Box flexDirection="column" flexShrink={0}>
      {/* Header carries the violet; no dimColor on top of it, or it drops
          below readable on terminals that dim aggressively. The expanded
          trace below stays muted — a long trace must not be violet. */}
      <Box flexShrink={0} onClick={() => toggleThinkingRow(rowId)}>
        <Text color={t.color.thinking}>
          {expanded ? '▾' : '▸'} thinking{tokenLabel}
        </Text>
      </Box>
      {expanded
        ? thinking.split('\n').map((line, i) => (
            <Text color={t.color.muted} dimColor key={i} wrap="wrap">
              {'  '}
              {line || ' '}
            </Text>
          ))
        : null}
    </Box>
  )
}

function ToolTrail({
  leadGap,
  msg,
  msgKey,
  t,
  visibility
}: {
  leadGap?: boolean
  msg: Msg
  msgKey?: string
  t: Theme
  visibility: DetailVisibility
}) {
  const thinking = msg.thinking?.trim()
  const tools = msg.tools ?? []

  return (
    <Box flexDirection="column" flexShrink={0} marginTop={leadGap ? 1 : 0} paddingLeft={3}>
      {thinking && visibility.thinking ? <ThinkingBlock msg={msg} rowId={thinkingRowId(msg, msgKey)} t={t} /> : null}

      {visibility.tools ? tools.map((line, i) => <ToolStep key={i} line={line} t={t} />) : null}
    </Box>
  )
}

function MessageLineView({
  leadGap,
  msg,
  msgKey,
  t
}: {
  leadGap?: boolean
  msg: Msg
  msgKey?: string
  t: Theme
}) {
  const visibility = detailVisibility(useStore($uiDetailVisibility))
  const hasVisibleDetails = messageHasVisibleDetails(msg, visibility)

  if (msg.kind === 'intro') {
    return null
  }

  if (msg.kind === 'trail') {
    if (!trailHasRenderableContent(msg) || !hasVisibleDetails) {
      return null
    }

    return <ToolTrail leadGap={leadGap} msg={msg} msgKey={msgKey} t={t} visibility={visibility} />
  }

  if (msg.role === 'user') {
    return <UserMessage msg={msg} t={t} />
  }

  if (msg.role === 'assistant') {
    return hasVisibleDetails ? (
      <Box flexDirection="column" flexShrink={0}>
        <ToolTrail leadGap={leadGap} msg={msg} msgKey={msgKey} t={t} visibility={visibility} />
        {/* The trail already opened the band, so the prose inside it never
            adds a second gap. */}
        {msg.text ? <AssistantMessage msg={msg} t={t} /> : null}
      </Box>
    ) : (
      <AssistantMessage leadGap={leadGap} msg={msg} t={t} />
    )
  }

  if (msg.role === 'tool') {
    return <ToolResultMessage msg={msg} t={t} />
  }

  return hasVisibleDetails ? (
    <Box flexDirection="column" flexShrink={0}>
      <ToolTrail msg={msg} msgKey={msgKey} t={t} visibility={visibility} />
      {msg.text ? <SystemMessage msg={msg} t={t} /> : null}
    </Box>
  ) : (
    <SystemMessage msg={msg} t={t} />
  )
}

export const MessageLine = memo(MessageLineView)
