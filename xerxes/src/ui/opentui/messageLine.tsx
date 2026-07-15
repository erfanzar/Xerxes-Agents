// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */
// OpenTUI message renderer. One Msg becomes one flat transcript block.
// Assistant text renders through OpenTUI's native <markdown> (tables, code,
// emphasis). Tool-call trail lines stay compact like Grok's transcript: the
// call is always visible, while diagnostic and diff detail remains available
// when it carries information the one-line result cannot safely summarize.
import { useStore } from '@nanostores/react'

import { $uiDetailVisibility } from '../app/uiStore.js'
import { messageHasVisibleDetails, trailHasRenderableContent } from '../lib/liveProgress.js'
import { fmtK, inlineToolDisplay, parseToolTrailResultLine } from '../lib/text.js'
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
      // @ts-expect-error MarkdownProps omits inherited Renderable.selectable.
      selectable
      syntaxStyle={getSyntaxStyle(t)}
      tableOptions={{ ...TABLE_OPTIONS, borderColor: t.color.border }}
    />
  )
}

function UserMessage({ msg, t }: { msg: Msg; t: Theme }) {
  return (
    <Box flexDirection="row" flexShrink={0} marginBottom={1} marginTop={1}>
      <Box backgroundColor={t.color.accent} flexShrink={0} width={1} />
      <Box
        backgroundColor={t.color.completionBg}
        flexDirection="column"
        flexGrow={1}
        paddingBottom={1}
        paddingLeft={2}
        paddingRight={1}
        paddingTop={1}
      >
        <Text color={t.color.label} wrap="wrap">
          {msg.text}
        </Text>
      </Box>
    </Box>
  )
}

function AssistantMessage({ msg, t }: { msg: Msg; t: Theme }) {
  return (
    <Box flexDirection="column" flexShrink={0} marginTop={1} paddingLeft={3}>
      <Markdown content={msg.text} t={t} />
    </Box>
  )
}

function SystemMessage({ msg, t }: { msg: Msg; t: Theme }) {
  return (
    <Box flexShrink={0} paddingLeft={2}>
      <Text color={t.color.muted} wrap="wrap">
        · {msg.text}
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

  if (!parsed) {
    // In-flight / transient call line ("drafting …", a bare tool name).
    return (
      <Text color={t.color.muted} wrap="truncate-end">
        <Span color={t.color.muted}>→ </Span>
        {line}
      </Text>
    )
  }

  const label = inlineToolDisplay(parsed.call)
  const markColor = parsed.mark === '✗' ? t.color.error : t.color.ok
  const detail = usefulToolDetail(parsed.detail, parsed.mark === '✗')

  return (
    <Box flexDirection="column" flexShrink={0}>
      <Text color={t.color.muted} wrap="truncate-end">
        <Span color={parsed.mark === '✗' ? t.color.error : t.color.muted}>→ </Span>
        {label}
        {parsed.mark === '✗' ? (
          <>
            {' '}
            <Span color={markColor}>{parsed.mark}</Span>
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

function ToolTrail({ msg, t, visibility }: { msg: Msg; t: Theme; visibility: DetailVisibility }) {
  const thinking = msg.thinking?.trim()
  const tools = msg.tools ?? []
  const tokenLabel = msg.thinkingTokens && msg.thinkingTokens > 0 ? `  ~${fmtK(msg.thinkingTokens)} tokens` : ''

  return (
    <Box flexDirection="column" flexShrink={0} marginTop={1} paddingLeft={3}>
      {thinking && visibility.thinking ? (
        <Box flexDirection="column" flexShrink={0}>
          <Text color={t.color.muted} dimColor>
            ◇ Thinking{tokenLabel}
          </Text>
          {thinking.split('\n').map((line, i) => (
            <Text color={t.color.muted} dimColor key={i} wrap="wrap">
              {'  '}
              {line || ' '}
            </Text>
          ))}
        </Box>
      ) : null}

      {visibility.tools ? tools.map((line, i) => <ToolStep key={i} line={line} t={t} />) : null}
    </Box>
  )
}

export function MessageLine({ msg, t }: { msg: Msg; t: Theme }) {
  const visibility = detailVisibility(useStore($uiDetailVisibility))
  const hasVisibleDetails = messageHasVisibleDetails(msg, visibility)

  if (msg.kind === 'intro') {
    return null
  }

  if (msg.kind === 'trail') {
    if (!trailHasRenderableContent(msg) || !hasVisibleDetails) {
      return null
    }

    return <ToolTrail msg={msg} t={t} visibility={visibility} />
  }

  if (msg.role === 'user') {
    return <UserMessage msg={msg} t={t} />
  }

  if (msg.role === 'assistant') {
    return hasVisibleDetails ? (
      <Box flexDirection="column" flexShrink={0}>
        <ToolTrail msg={msg} t={t} visibility={visibility} />
        {msg.text ? <AssistantMessage msg={msg} t={t} /> : null}
      </Box>
    ) : (
      <AssistantMessage msg={msg} t={t} />
    )
  }

  if (msg.role === 'tool') {
    return <ToolResultMessage msg={msg} t={t} />
  }

  return hasVisibleDetails ? (
    <Box flexDirection="column" flexShrink={0}>
      <ToolTrail msg={msg} t={t} visibility={visibility} />
      {msg.text ? <SystemMessage msg={msg} t={t} /> : null}
    </Box>
  ) : (
    <SystemMessage msg={msg} t={t} />
  )
}
