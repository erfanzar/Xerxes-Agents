// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * An ArtifactTool call in the conversation: a card naming the document with
 * an Open button, instead of a generic tool row the reader has to decode and
 * then go hunting for in the Artifacts section.
 */

import { memo, type ReactElement } from 'react'
import { Icon } from './Icon.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import type { ToolItem } from './types.js'

export interface ArtifactRef {
  readonly id: string
  readonly title: string
  readonly version: number
  readonly action: 'created' | 'updated'
  readonly kind: string
}

const RESULT = /^(Created|Updated) artifact "(.*)" \(id ([A-Za-z0-9_-]+), version (\d+)\)/s

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
