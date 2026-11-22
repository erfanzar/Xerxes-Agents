// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * An ArtifactTool call in the conversation: a card naming the document with
 * an Open button, instead of a generic tool row the reader has to decode and
 * then go hunting for in the Artifacts section.
 */

import { memo, useEffect, useRef, useState, type ReactElement } from 'react'
import { Icon, type IconName } from './Icon.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import { ArtifactWidgetEmbed } from './WidgetFrame.js'
import type { Block, ToolItem } from './types.js'

export interface ArtifactRef {
  readonly id: string
  readonly title: string
  readonly version: number
  readonly action: 'created' | 'updated'
  readonly kind: string
  readonly embed?: boolean
}

const RESULT = /^(Created|Updated) artifact "(.*)" \(id ([A-Za-z0-9_-]+), version (\d+)\)/s

const AGY_ARTIFACT_TOOLS = new Set([
  'write_file',
  'edit_file',
  'write_to_file',
  'replace_file_content',
  'multi_replace_file_content',
])

const AGY_EDIT_TOOLS = new Set([
  'edit_file',
  'replace_file_content',
  'multi_replace_file_content',
])

const AGY_BRAIN_ARTIFACT_PATH_RE =
  /(?:^|[\\/])\.gemini[\\/]antigravity(?:-cli)?[\\/]brain[\\/][^\\/]+[\\/]([^\\/]+)$/i

const AGY_BRAIN_ARTIFACT_EXT_RE = /\.(?:md|markdown|html|htm)$/i

function canonicalToolTail(name: string | undefined): string {
  const safe = typeof name === 'string' ? name : ''
  return (safe.split(/[.:/]/).pop() ?? safe)
    .trim()
    .replace(/([a-z0-9])([A-Z])/g, '$1_$2')
    .replace(/[\s-]+/g, '_')
    .toLowerCase()
}

function asRecord(value: unknown): Readonly<Record<string, unknown>> | null {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Readonly<Record<string, unknown>>)
    : null
}

function parseJsonObject(raw: string | undefined): Readonly<Record<string, unknown>> | null {
  if (typeof raw !== 'string') return null
  const trimmed = raw.trim()
  if (!trimmed.startsWith('{') || !trimmed.endsWith('}')) return null
  try {
    return asRecord(JSON.parse(trimmed))
  } catch {
    return null
  }
}

function cleanStr(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function kindOfArtifactFilename(filename: string): string {
  const lower = filename.toLowerCase()
  if (lower.endsWith('.html') || lower.endsWith('.htm')) return 'html'
  if (lower.endsWith('.md') || lower.endsWith('.markdown')) return 'markdown'
  if (lower.endsWith('.txt')) return 'text'
  if (lower.includes('.')) return 'code'
  return 'markdown'
}

function humanizeArtifactTitle(filename: string, content = ''): string {
  if (content) {
    const heading = /^\s*#[ \t]+([^\n]+?)[ \t]*#*[ \t]*(?:\n|$)/.exec(content)
    if (heading?.[1]?.trim()) return heading[1].trim()
  }
  const stem = filename.replace(/\.(?:md|markdown|html|htm|txt)$/i, '').trim()
  if (!stem) return filename || 'Artifact'
  if (/[_-]/.test(stem) && stem === stem.toLowerCase()) {
    return stem
      .split(/[_-]+/)
      .filter(Boolean)
      .map(part => part[0]!.toUpperCase() + part.slice(1))
      .join(' ')
  }
  return stem[0]!.toUpperCase() + stem.slice(1)
}

function brainArtifactFilenameOf(rawPath: string): string {
  const match = AGY_BRAIN_ARTIFACT_PATH_RE.exec(rawPath.trim())
  const filename = match?.[1]?.trim() ?? ''
  if (!filename || filename.startsWith('.') || /\.metadata\.json$/i.test(filename)) return ''
  return AGY_BRAIN_ARTIFACT_EXT_RE.test(filename) ? filename : ''
}

function firstLocationPathOf(locations: unknown): string {
  if (!Array.isArray(locations)) return ''
  for (const entry of locations) {
    if (typeof entry === 'string' && entry.trim()) return entry.trim()
    const rec = asRecord(entry)
    const p = cleanStr(rec?.path) || cleanStr(rec?.uri)
    if (p) return p
  }
  return ''
}

/** The artifact an ArtifactTool result text names; null for anything else. */
export function parseArtifactResult(result: string): Omit<ArtifactRef, 'kind'> | null {
  const match = RESULT.exec(result.trim())
  if (!match) return null
  return { action: match[1] === 'Created' ? 'created' : 'updated', title: match[2]!, id: match[3]!, version: Number(match[4]) }
}

/**
 * The artifact a finished ArtifactTool OR Antigravity `write_file`/`edit_file`
 * call saved (carrying `ArtifactMetadata`, targeting a brain artifact path, or
 * carrying daemon-emitted `artifact_id: 'agy:<filename>'`); null otherwise.
 */
export function artifactOfTool(
  item: Pick<ToolItem, 'name' | 'state' | 'output' | 'input'> & {
    readonly path?: string
    readonly artifactId?: string
    readonly artifact_id?: string
  },
): ArtifactRef | null {
  if (item.state !== 'done') return null
  if (item.name === 'ArtifactTool') {
    const outputText = typeof item.output === 'string' ? item.output.trim() : ''
    const match = RESULT.exec(outputText)
    if (!match) return null
    let kind = 'markdown'
    let embed = false
    const input = parseJsonObject(item.input)
    if (input) {
      if (typeof input.kind === 'string') kind = input.kind
      if (input.embed === true) embed = true
    }
    return {
      action: match[1] === 'Created' ? 'created' : 'updated',
      title: match[2]!,
      id: match[3]!,
      version: Number(match[4]),
      kind,
      ...(embed ? { embed: true } : {}),
    }
  }

  const tail = canonicalToolTail(item.name)
  const input = parseJsonObject(item.input)
  const output = parseJsonObject(item.output)
  const explicitArtifactId =
    cleanStr(item.artifact_id)
    || cleanStr(item.artifactId)
    || cleanStr(output?.artifact_id)
    || cleanStr(output?.artifactId)
  if (!AGY_ARTIFACT_TOOLS.has(tail) && !explicitArtifactId) return null

  const rawPath =
    cleanStr(input?.Path)
    || cleanStr(input?.path)
    || cleanStr(input?.file_path)
    || cleanStr(input?.TargetFile)
    || cleanStr(item.path)
  const meta =
    asRecord(input?.ArtifactMetadata)
    ?? asRecord(input?.artifact_metadata)
    ?? asRecord(input?.artifactMetadata)
  const brainFilename = rawPath ? brainArtifactFilenameOf(rawPath) : ''
  if (!explicitArtifactId && !meta && !brainFilename) return null

  const baseFromPath = rawPath ? rawPath.replace(/[\\/]+$/, '').split(/[\\/]/).pop() ?? '' : ''
  const filename = explicitArtifactId.startsWith('agy:')
    ? explicitArtifactId.slice(4)
    : brainFilename || baseFromPath
  const id = explicitArtifactId || (filename ? `agy:${filename}` : '')
  if (!id) return null

  const content =
    cleanStr(input?.Content)
    || cleanStr(input?.content)
    || cleanStr(input?.CodeContent)
  const explicitTitle =
    cleanStr(meta?.Title)
    || cleanStr(meta?.title)
    || cleanStr(input?.title)
  const title = explicitTitle || humanizeArtifactTitle(filename || id.replace(/^agy:/, ''), content)
  const isEdit =
    AGY_EDIT_TOOLS.has(tail)
    || input?.Append === true
    || input?.append === true
  const kind = cleanStr(input?.kind) || kindOfArtifactFilename(filename || id)
  return {
    id,
    title,
    version: 1,
    action: isEdit ? 'updated' : 'created',
    kind,
  }
}

/** Extract an ArtifactRef from a live daemon `tool_result` event payload (native or ACP). */
export function artifactFromToolResultPayload(
  payload: Readonly<Record<string, unknown>>,
  fallbackInput = '',
): ArtifactRef | null {
  if (payload.permitted === false) return null
  if (typeof payload.error === 'string' && payload.error.trim()) return null
  const rawResult = payload.return_value ?? payload.result ?? payload.output
  const output =
    typeof rawResult === 'string'
      ? rawResult
      : rawResult !== null && rawResult !== undefined && typeof rawResult === 'object'
        ? JSON.stringify(rawResult)
        : ''
  const input =
    typeof payload.arguments === 'string'
      ? payload.arguments
      : payload.arguments !== null && payload.arguments !== undefined && typeof payload.arguments === 'object'
        ? JSON.stringify(payload.arguments)
        : fallbackInput
  const explicitArtifactId = cleanStr(payload.artifact_id) || cleanStr(payload.artifactId)
  const diffObj = asRecord(payload.diff)
  const path =
    cleanStr(payload.path)
    || cleanStr(diffObj?.path)
    || firstLocationPathOf(payload.locations)
  return artifactOfTool({
    name: cleanStr(payload.name),
    state: 'done',
    input,
    output,
    ...(path ? { path } : {}),
    ...(explicitArtifactId ? { artifact_id: explicitArtifactId } : {}),
  })
}

/**
 * The artifact the Plan ready card should open, read from the transcript.
 * Within the current plan round (since the last EnterPlanModeTool call), a
 * "Plan…" artifact wins over a report or goal artifact updated later; then
 * any artifact created in that round; then the newest artifact at all.
 * Undefined when the loaded transcript names none (the card then opens the list).
 */
export function latestArtifactId(blocks: readonly Block[]): string | undefined {
  let newest: ArtifactRef | undefined, created: ArtifactRef | undefined, titled: ArtifactRef | undefined
  let inRound = true
  for (let index = blocks.length - 1; index >= 0; index -= 1) {
    const block = blocks[index]!
    if (block.kind !== 'tools') continue
    for (let item = block.items.length - 1; item >= 0; item -= 1) {
      const tool = block.items[item]!
      if (/^EnterPlanMode(Tool)?$/.test(tool.name)) { inRound = false; continue }
      const artifact = artifactOfTool(tool)
      if (!artifact) continue
      newest ??= artifact
      if (!inRound) continue
      if (/^plan\b/i.test(artifact.title)) titled ??= artifact
      if (artifact.action === 'created') created ??= artifact
    }
    if (!inRound && newest) break
  }
  return (titled ?? created ?? newest)?.id
}

/**
 * "New artifact: <title> — Open", shown for a few seconds when the current
 * task saves an artifact while the right panel is not already on Artifacts.
 */
export function ArtifactToast({ sessionId, showingArtifacts }: { sessionId: string; showingArtifacts: boolean }): ReactElement | null {
  const open = useDesktopNavigation()
  const [toast, setToast] = useState<Omit<ArtifactRef, 'kind'> | null>(null)
  const showing = useRef(showingArtifacts)
  showing.current = showingArtifacts
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | undefined
    const pendingInputs = new Map<string, string>()
    const off = window.xerxes.onEvent(({ type, payload }) => {
      if (typeof payload.session_id === 'string' && payload.session_id !== sessionId) return
      if (type === 'tool_call') {
        const callId = cleanStr(payload.tool_call_id) || cleanStr(payload.id)
        const rawArgs =
          typeof payload.arguments === 'string'
            ? payload.arguments
            : payload.arguments && typeof payload.arguments === 'object'
              ? JSON.stringify(payload.arguments)
              : ''
        if (callId && rawArgs) {
          if (pendingInputs.size >= 64) {
            const oldest = pendingInputs.keys().next().value
            if (oldest !== undefined) pendingInputs.delete(oldest)
          }
          pendingInputs.set(callId, rawArgs)
        }
        return
      }
      if (type !== 'tool_result') return
      const callId = cleanStr(payload.tool_call_id) || cleanStr(payload.id)
      const fallbackInput = callId ? (pendingInputs.get(callId) ?? '') : ''
      if (callId) pendingInputs.delete(callId)
      const saved = artifactFromToolResultPayload(payload, fallbackInput)
      if (!saved || showing.current) return
      setToast(saved)
      clearTimeout(timer)
      timer = setTimeout(() => setToast(null), 8000)
    })
    return () => { off(); clearTimeout(timer) }
  }, [sessionId])
  if (!toast) return null
  return <div className="artifact-toast" role="status" aria-live="polite">
    <span className="artifact-toast__icon" aria-hidden="true"><Icon name="file" size={14} /></span>
    <span className="artifact-toast__text">
      <span className="artifact-toast__eyebrow">{toast.action === 'created' ? 'New artifact' : `Updated · v${toast.version}`}</span>
      <strong className="artifact-toast__title" title={toast.title}>{toast.title}</strong>
    </span>
    <button type="button" className="artifact-card__open" onClick={() => { open('docs', toast.id); setToast(null) }}>
      <span>Open</span>
      <span className="artifact-card__arrow" aria-hidden="true">→</span>
    </button>
    <button type="button" className="artifact-toast__close" aria-label="Dismiss" onClick={() => setToast(null)}><Icon name="close" size={12} /></button>
  </div>
}

const kindLabel = (kind: string): string => kind === 'markdown' ? 'Document' : kind === 'html' ? 'Interactive' : kind === 'code' ? 'Code' : 'Text'

const kindIcon = (kind: string): IconName => kind === 'html' ? 'spark' : kind === 'code' ? 'code' : kind === 'markdown' ? 'file' : 'note'

export const ArtifactCard = memo(function ArtifactCard({ artifact }: { artifact: ArtifactRef }): ReactElement {
  const open = useDesktopNavigation()
  const isInteractive = artifact.kind === 'html'
  const [previewOpen, setPreviewOpen] = useState(isInteractive && artifact.embed === true)
  const versionBadge = artifact.version > 1 ? `v${artifact.version}` : artifact.action === 'created' ? 'new' : ''

  return <div
    className="artifact-card"
    data-kind={artifact.kind}
    data-interactive={isInteractive || undefined}
    data-expanded={previewOpen || undefined}
    role="group"
    aria-label={`Artifact: ${artifact.title}`}
  >
    <div className="artifact-card__row">
      <span className="artifact-card__icon" data-kind={artifact.kind} aria-hidden="true">
        <Icon name={kindIcon(artifact.kind)} size={15} />
      </span>
      <button
        type="button"
        className="artifact-card__main"
        onClick={() => open('docs', artifact.id)}
        title={`Open ${artifact.title}`}
      >
        <strong className="artifact-card__title">{artifact.title}</strong>
        <span className="artifact-card__meta">
          <span className="artifact-card__pill artifact-card__pill--kind" data-kind={artifact.kind}>{kindLabel(artifact.kind)}</span>
          {versionBadge && (
            <span className={`artifact-card__pill ${artifact.version > 1 ? 'artifact-card__pill--ver' : 'artifact-card__pill--new'}`}>
              {versionBadge}
            </span>
          )}
        </span>
      </button>
      <span className="artifact-card__actions">
        {isInteractive && (
          <button
            type="button"
            className="artifact-card__preview"
            aria-expanded={previewOpen}
            onClick={() => setPreviewOpen(value => !value)}
          >
            <Icon name="eye" size={12} />
            <span>{previewOpen ? 'Hide' : 'Preview'}</span>
          </button>
        )}
        <button
          type="button"
          className="artifact-card__open"
          data-interactive={isInteractive || undefined}
          onClick={() => open('docs', artifact.id)}
        >
          <span>Open</span>
          <span className="artifact-card__arrow" aria-hidden="true">→</span>
        </button>
      </span>
    </div>
    {isInteractive && previewOpen && (
      <div className="artifact-card__embed">
        <ArtifactWidgetEmbed
          key={`${artifact.id}:${artifact.version}`}
          artifactId={artifact.id}
          title={artifact.title}
        />
      </div>
    )}
  </div>
})
