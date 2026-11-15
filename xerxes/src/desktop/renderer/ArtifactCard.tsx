// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * An ArtifactTool call in the conversation: a card naming the document with
 * an Open button, instead of a generic tool row the reader has to decode and
 * then go hunting for in the Artifacts section.
 */

import { memo, useEffect, useRef, useState, type ReactElement } from 'react'
import { Icon } from './Icon.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import type { Block, ToolItem } from './types.js'

export interface ArtifactRef {
  readonly id: string
  readonly title: string
  readonly version: number
  readonly action: 'created' | 'updated'
  readonly kind: string
}

const RESULT = /^(Created|Updated) artifact "(.*)" \(id ([A-Za-z0-9_-]+), version (\d+)\)/s

/** The artifact an ArtifactTool result text names; null for anything else. */
export function parseArtifactResult(result: string): Omit<ArtifactRef, 'kind'> | null {
  const match = RESULT.exec(result.trim())
  if (!match) return null
  return { action: match[1] === 'Created' ? 'created' : 'updated', title: match[2]!, id: match[3]!, version: Number(match[4]) }
}

/** The artifact a finished ArtifactTool call saved, read from its result; null otherwise. */
export function artifactOfTool(item: Pick<ToolItem, 'name' | 'state' | 'output' | 'input'>): ArtifactRef | null {
  if (item.name !== 'ArtifactTool' || item.state !== 'done') return null
  const match = RESULT.exec(item.output.trim())
  if (!match) return null
  let kind = 'markdown'
  try {
    const input: unknown = JSON.parse(item.input)
    if (input && typeof input === 'object' && typeof (input as { kind?: unknown }).kind === 'string') kind = (input as { kind: string }).kind
  } catch { /* the result alone names the artifact */ }
  return { action: match[1] === 'Created' ? 'created' : 'updated', title: match[2]!, id: match[3]!, version: Number(match[4]), kind }
}

/** The id of the newest artifact this conversation saved, read from its transcript; undefined when none. */
export function latestArtifactId(blocks: readonly Block[]): string | undefined {
  for (let index = blocks.length - 1; index >= 0; index -= 1) {
    const block = blocks[index]!
    if (block.kind !== 'tools') continue
    for (let item = block.items.length - 1; item >= 0; item -= 1) {
      const artifact = artifactOfTool(block.items[item]!)
      if (artifact) return artifact.id
    }
  }
  return undefined
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
    const off = window.xerxes.onEvent(({ type, payload }) => {
      if (type !== 'tool_result' || payload.name !== 'ArtifactTool' || payload.permitted === false) return
      if (typeof payload.session_id === 'string' && payload.session_id !== sessionId) return
      const saved = parseArtifactResult(typeof payload.result === 'string' ? payload.result : '')
      if (!saved || showing.current) return
      setToast(saved)
      clearTimeout(timer)
      timer = setTimeout(() => setToast(null), 8000)
    })
    return () => { off(); clearTimeout(timer) }
  }, [sessionId])
  if (!toast) return null
  return <div className="artifact-toast" role="status" aria-live="polite">
    <span className="artifact-card__icon"><Icon name="file" size={14} /></span>
    <span className="artifact-toast__text"><strong>{toast.action === 'created' ? 'New artifact' : 'Artifact updated'}</strong><span>{toast.title}</span></span>
    <button className="artifact-card__open" onClick={() => { open('docs', toast.id); setToast(null) }}>Open</button>
    <button className="artifact-toast__close" aria-label="Dismiss" onClick={() => setToast(null)}><Icon name="close" size={12} /></button>
  </div>
}

const kindLabel = (kind: string): string => kind === 'markdown' ? 'Document' : kind === 'code' ? 'Code' : kind === 'html' ? 'HTML' : 'Text'

export const ArtifactCard = memo(function ArtifactCard({ artifact }: { artifact: ArtifactRef }): ReactElement {
  const open = useDesktopNavigation()
  const meta = `${kindLabel(artifact.kind)} · ${artifact.action === 'created' ? 'new' : `updated to v${artifact.version}`}`
  return <div className="artifact-card" role="group" aria-label={`Artifact: ${artifact.title}`}>
    <span className="artifact-card__icon"><Icon name="file" size={16} /></span>
    <button className="artifact-card__main" onClick={() => open('docs', artifact.id)} title={`Open ${artifact.title}`}>
      <strong className="artifact-card__title">{artifact.title}</strong>
      <span className="artifact-card__meta">{meta}</span>
    </button>
    <button className="artifact-card__open" onClick={() => open('docs', artifact.id)}>Open</button>
  </div>
})
