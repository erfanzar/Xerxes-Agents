// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * T3 Code-style empty Right Panel launcher ("Open a surface") tailored to Xerxes.
 *
 * Shown when the user opens the Right Panel via the top-right PanelRight toggle
 * and no specific surface tab is currently open (or after closing the last tab).
 * Single-letter shortcuts jump straight to any Xerxes surface; once a surface
 * is open, `DesktopRail` renders its tabs and the `+` RailAddMenu switcher.
 */

import { useEffect, type ReactElement } from 'react'
import { T3Icon, railT3IconOf } from './T3Icons.js'
import { activityFleetRows, agentState } from './AgentRoster.js'
import { showsImmortalsForSession, type Snapshot } from './store.js'
import { swarmMemberCounts } from './swarmState.js'
import type { RailPanel } from './railTabs.js'

export interface SurfaceLauncherItem {
  readonly panel: RailPanel
  readonly label: string
  readonly shortcut: string
}

export const XERXES_SURFACES: readonly SurfaceLauncherItem[] = [
  { panel: 'browser', label: 'Browser', shortcut: 'B' },
  { panel: 'terminal', label: 'Terminal', shortcut: 'T' },
  { panel: 'files', label: 'Files', shortcut: 'F' },
  { panel: 'review', label: 'Changes', shortcut: 'D' },
  { panel: 'btw', label: 'Side question', shortcut: 'S' },
  { panel: 'subagents', label: 'Subagents', shortcut: 'A' },
  { panel: 'swarm', label: 'Immortals', shortcut: 'I' },
  { panel: 'docs', label: 'Artifacts', shortcut: 'R' },
  { panel: 'background', label: 'Background', shortcut: 'G' },
  { panel: 'outputs', label: 'Outputs', shortcut: 'O' },
  { panel: 'activity', label: 'Activity', shortcut: 'V' },
  { panel: 'usage', label: 'Usage', shortcut: 'U' },
]

export function RightPanelLauncher({
  snap,
  onPick,
  onClose,
  filesExpanded = false,
  toggleFilesExpanded,
}: {
  readonly snap: Snapshot
  readonly onPick: (panel: RailPanel) => void
  readonly onClose: () => void
  readonly filesExpanded?: boolean
  readonly toggleFilesExpanded?: () => void
}): ReactElement {
  const showSwarm = showsImmortalsForSession(snap)
  const items = showSwarm
    ? XERXES_SURFACES
    : XERXES_SURFACES.filter(item => item.panel !== 'swarm')

  const fleet = activityFleetRows(snap.fleet, snap.blocks)
  const workingAgents = fleet.filter(row => agentState(row.status).priority < 2).length
  const swarmCounts = snap.swarm ? swarmMemberCounts(snap.swarm.members) : null

  // Single-letter shortcuts while the launcher is open (never hijacking typing in inputs).
  useEffect(() => {
    const onKey = (event: KeyboardEvent): void => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey) return
      const target = event.target as HTMLElement | null
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return
      }
      const key = event.key.toUpperCase()
      const match = items.find(item => item.shortcut === key)
      if (!match) return
      event.preventDefault()
      onPick(match.panel)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [items, onPick])

  const badgeFor = (panel: RailPanel): string | null => {
    if (panel === 'review' && snap.changes.length > 0) return `${snap.changes.length}`
    if (panel === 'subagents' && fleet.length > 0) {
      return workingAgents > 0 ? `${workingAgents} working` : `${fleet.length}`
    }
    if (panel === 'swarm' && snap.swarm?.active) {
      return swarmCounts && swarmCounts.working > 0 ? `${swarmCounts.working} active` : 'active'
    }
    return null
  }

  return (
    <aside
      className="desktop-rail studio-sheet t3-surface-launcher"
      data-right-panel-launcher="true"
      aria-label="Open a surface"
    >
      <header className="t3-surface-launcher__head">
        <span className="composer__flex" />
        {toggleFilesExpanded && (
          <button
            type="button"
            aria-label={filesExpanded ? 'Restore conversation' : 'Expand right panel'}
            title={filesExpanded ? 'Restore conversation' : 'Expand right panel'}
            aria-pressed={filesExpanded}
            onClick={toggleFilesExpanded}
          >
            <T3Icon name={filesExpanded ? 'minimize' : 'maximize'} size={14} />
          </button>
        )}
        <button type="button" aria-label="Close right panel" title="Close right panel" onClick={onClose}>
          <T3Icon name="close" size={14} />
        </button>
      </header>

      <div className="t3-surface-launcher__body">
        <div className="t3-surface-launcher__inner">
          <h2 className="t3-surface-launcher__title">Open a surface</h2>
          <div className="t3-surface-launcher__list" role="menu" aria-label="Available surfaces">
            {items.map(item => {
              const meta = badgeFor(item.panel)
              return (
                <button
                  key={item.panel}
                  type="button"
                  role="menuitem"
                  className="t3-surface-launcher__row"
                  data-surface={item.panel}
                  onClick={() => onPick(item.panel)}
                >
                  <span className="t3-surface-launcher__ico" aria-hidden="true">
                    <T3Icon name={railT3IconOf(item.panel)} size={15} />
                  </span>
                  <span className="t3-surface-launcher__label">{item.label}</span>
                  {meta && <span className="t3-surface-launcher__meta">{meta}</span>}
                  <kbd className="t3-surface-launcher__kbd">{item.shortcut}</kbd>
                </button>
              )
            })}
          </div>
        </div>
      </div>
    </aside>
  )
}
