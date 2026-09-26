// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The rail's two live lists: the agents this task spawned, and the files
 * it has touched.
 *
 * Both are deliberately flat rows, not disclosures. `AgentRoster` renders
 * every agent as its own `<details>` and folds finished ones into a "Past
 * agents" wrapper, which is the right shape for the full-width inspector
 * and the wrong one for a 340px rail — it turned the rail back into a
 * stack of triangles, which is the thing the rail redesign set out to
 * remove. A row here is a status you can read at a glance and a click
 * through to the detail, which still lives in the inspector and the
 * session-edits tab.
 */

import { useState, type ReactElement } from 'react'

import { agentKindLabel, agentState } from './AgentRoster.js'
import type { DiffFile, SessionRow } from './types.js'
import { Icon } from './Icon.js'

/** Beyond this the rail stops being a glance; the rest is one click away. */
const VISIBLE = 6
/** States that must never be collapsed out of view. */
const NEEDS_EYES = new Set(['working', 'waiting', 'failed'])

/** Active and blocked agents first — a finished one is not news. */
function byUrgency(a: SessionRow, b: SessionRow): number {
  return agentState(a.status).priority - agentState(b.status).priority
}

export function RailAgents({ rows, onInspect }: {
  rows: readonly SessionRow[]
  onInspect: (id: string) => void
}): ReactElement | null {
  // "N more" expands the list in place. It used to navigate to the
  // Activity tab — the tab this card already sits on — so it did nothing.
  const [expanded, setExpanded] = useState(false)
  if (rows.length === 0) return null
  const ordered = [...rows].sort(byUrgency)
  // Only the finished tail is ever capped. An agent that is working, stuck
  // or failed is the whole reason to look at this list, so it is never the
  // thing hidden behind "N more".
  const live = ordered.filter(row => NEEDS_EYES.has(agentState(row.status).tone))
  const finished = ordered.filter(row => !NEEDS_EYES.has(agentState(row.status).tone))
  const shown = expanded ? ordered : [...live, ...finished.slice(0, Math.max(0, VISIBLE - live.length))]
  const failed = ordered.filter(row => agentState(row.status).tone === 'failed').length
  const working = ordered.filter(row => agentState(row.status).tone === 'working').length
  return (
    <section className="railcard" aria-label="Agents">
      <header className="railcard__head">
        <span className="railcard__title">Agents</span>
        <span className="railcard__meta">
          {[working ? `${working} working` : '', failed ? `${failed} failed` : '', rows.length].filter(Boolean).join(' · ')}
        </span>
      </header>
      {shown.map(row => {
        const state = agentState(row.status)
        const kind = agentKindLabel(row.agentDetails)
        return (
          <button className="railrow" key={row.id} data-state={state.tone} onClick={() => onInspect(row.id)} title={`Inspect ${row.title}${kind ? ` (${kind})` : ''}`}>
            <span className="railrow__name agentname agentname--stack"><span className="agentname__t">{row.title}</span>{kind && <span className="agentname__kind">({kind})</span>}</span>
            {/* Always present, coloured by tone. Omitting it for the happy
                path left a gap-toothed right edge; the colour is what
                separates "failed" from "completed", not the presence. */}
            <span className="railrow__meta">{state.label.toLowerCase()}</span>
          </button>
        )
      })}
      {ordered.length > shown.length && (
        <button className="raillist__more" aria-expanded={false} onClick={() => setExpanded(true)}>
          {ordered.length - shown.length} more<Icon name="chevron" size={12} />
        </button>
      )}
      {expanded && ordered.length > VISIBLE && (
        <button className="raillist__more" aria-expanded onClick={() => setExpanded(false)}>
          Show fewer
        </button>
      )}
    </section>
  )
}

export function RailFiles({ files, onOpen }: {
  files: readonly DiffFile[]
  onOpen: () => void
}): ReactElement | null {
  if (files.length === 0) return null
  // Biggest edits first: the file with 200 changed lines is the one worth
  // looking at, not whichever the agent happened to write last.
  const ordered = [...files].sort((a, b) => (b.adds + b.dels) - (a.adds + a.dels))
  const shown = ordered.slice(0, VISIBLE)
  const adds = files.reduce((sum, file) => sum + file.adds, 0)
  const dels = files.reduce((sum, file) => sum + file.dels, 0)
  return (
    <section className="railcard" aria-label="Files this task changed">
      <header className="railcard__head">
        <span className="railcard__title">Touched</span>
        <span className="railcard__meta">
          {files.length} file{files.length === 1 ? '' : 's'} · +{adds} −{dels}
        </span>
      </header>
      {shown.map(file => {
        const segments = file.path.split('/')
        const name = segments.at(-1) ?? file.path
        const parent = segments.length > 1 ? segments.at(-2) : undefined
        return (
          <button
            className="railrow"
            key={file.path}
            onClick={onOpen}
            title={`${file.path} — open the session edits`}
            // The visible text is a basename, and this repo is full of
            // colliding ones (three `types.ts`, four `index.ts`). The full
            // path used to live only in `title`, which a keyboard or screen
            // reader user never sees.
            aria-label={`${file.path}, ${file.adds} added, ${file.dels} removed — open the session edits`}
          >
            <span className="railrow__name railrow__name--path">
              {parent && <span className="railrow__dir">{parent}/</span>}
              {name}
            </span>
            <span className="railrow__adds">+{file.adds}</span>
            <span className="railrow__dels">−{file.dels}</span>
          </button>
        )
      })}
      {ordered.length > VISIBLE && (
        <button className="raillist__more" onClick={onOpen}>
          {ordered.length - VISIBLE} more<Icon name="chevron" size={12} />
        </button>
      )}
    </section>
  )
}
