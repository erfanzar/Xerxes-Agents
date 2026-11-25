// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The Sections menu: a header button lists the task's sections
 * (Subagents, Artifacts, Changes, Files, Terminal) with counts, and hovering
 * one previews its contents beside the list — agents, artifacts, changed
 * files with M/A badges and a filter. Picking a row opens that section (or
 * that exact file, diff or agent) in the right panel.
 */

import { useEffect, useMemo, useRef, useState, type ReactElement } from 'react'
import { Icon, type IconName } from './Icon.js'
import { desktopCall } from './desktopRpc.js'
import { activityFleetRows, agentState } from './AgentRoster.js'
import { artifactRowsOf, compactAge, type ArtifactRow } from './RailSections.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import { showsImmortalsForSession, store, type Snapshot } from './store.js'
import { swarmMemberCounts, type SwarmStatusSnapshot } from './swarmState.js'
import type { DiffFile, SessionRow } from './types.js'

export type MenuSection = 'subagents' | 'swarm' | 'btw' | 'docs' | 'review' | 'files' | 'terminal'
export type PreviewSide = 'left' | 'right'

export const MENU_SECTIONS: readonly { readonly id: MenuSection; readonly label: string; readonly icon: IconName; readonly preview: boolean }[] = [
  { id: 'subagents', label: 'Subagents', icon: 'agents', preview: true },
  { id: 'btw', label: 'Side question', icon: 'chat', preview: false },
  { id: 'docs', label: 'Artifacts', icon: 'stack', preview: true },
  { id: 'review', label: 'Changes', icon: 'changes', preview: true },
  { id: 'files', label: 'Files', icon: 'folder', preview: true },
  { id: 'terminal', label: 'Terminal', icon: 'terminal', preview: false },
]

const SWARM_MENU_SECTION: { readonly id: MenuSection; readonly label: string; readonly icon: IconName; readonly preview: boolean } = {
  id: 'swarm',
  label: 'Immortals',
  icon: 'tree',
  preview: true,
}

const baseName = (path: string): string => path.split('/').filter(Boolean).at(-1) ?? path

/** Changed files whose path contains the filter text (case-insensitive); all of them for an empty filter. */
export function filterChanges(changes: readonly DiffFile[], needle: string): readonly DiffFile[] {
  const query = needle.trim().toLowerCase()
  return query ? changes.filter(change => change.path.toLowerCase().includes(query)) : changes
}

export interface SectionsMenuPanelProps {
  readonly hover: MenuSection | null
  readonly hoverTop: number
  /** Which side of the list the preview opens on, and its width; left at 300px by default. */
  readonly hoverSide?: PreviewSide
  readonly hoverWidth?: number
  readonly needle: string
  readonly setNeedle: (value: string) => void
  readonly agents: readonly SessionRow[]
  /** null while unknown (not loaded, or the runtime has no artifact methods). */
  readonly artifacts: readonly ArtifactRow[] | null
  readonly changes: readonly DiffFile[]
  readonly swarm?: SwarmStatusSnapshot | null
  readonly showSwarm?: boolean
  readonly onHover: (section: MenuSection | null, top: number, side?: PreviewSide, width?: number) => void
  readonly onPick: (section: MenuSection, path?: string) => void
}

function ChangeBadge({ change }: { change: DiffFile }): ReactElement {
  return <span className="sections-menu__badge" data-change={change.isNew ? 'added' : 'modified'} title={change.isNew ? 'Added' : 'Modified'}>{change.isNew ? 'A' : 'M'}</span>
}

function SectionPreview(props: SectionsMenuPanelProps & { section: MenuSection }): ReactElement {
  const { section, hoverTop, needle, setNeedle, agents, artifacts, changes, swarm, onPick } = props
  const label = section === 'swarm' ? SWARM_MENU_SECTION.label : (MENU_SECTIONS.find(entry => entry.id === section)?.label ?? '')
  let body: ReactElement
  if (section === 'swarm') {
    const members = swarm?.members ?? []
    body = !swarm?.active && members.length === 0
      ? <p className="sections-menu__empty">No active Immortals run. Run /immortals &lt;task&gt; or open the Immortals Board.</p>
      : <>
          <div className="sections-menu__group"><span>{swarm?.title || 'Immortals Board'}</span><span>{members.length}</span></div>
          <div className="sections-menu__items">{members.map(member => <button key={member.name} type="button" role="menuitem" className="sections-menu__item" title={`@${member.name} · ${member.state}`} onClick={() => onPick('swarm')}>
            <span className="sections-menu__dot" data-live={member.state === 'working' || undefined} aria-hidden="true" />
            <span className="sections-menu__name">@{member.name}</span><span className="sections-menu__meta">{member.state}</span>
          </button>)}</div>
        </>
  } else if (section === 'subagents') {
    body = agents.length === 0 ? <p className="sections-menu__empty">No subagents in this task.</p> : <>
      <div className="sections-menu__group"><span>Subagents</span><span>{agents.length}</span></div>
      <div className="sections-menu__items">{agents.map(row => <button key={row.id} type="button" role="menuitem" className="sections-menu__item" title={row.title} onClick={() => onPick('subagents', row.id)}>
        <span className="sections-menu__dot" data-live={agentState(row.status).priority < 2 || undefined} aria-hidden="true" />
        <span className="sections-menu__name">{row.title}</span><span className="sections-menu__meta">{row.age}</span>
      </button>)}</div>
    </>
  } else if (section === 'docs') {
    body = !artifacts ? <p className="sections-menu__empty">Artifacts load when the workspace runtime supports them.</p>
      : artifacts.length === 0 ? <p className="sections-menu__empty">No artifacts yet.</p> : <>
        <div className="sections-menu__group"><span>Artifacts</span><span>{artifacts.length}</span></div>
        <div className="sections-menu__items">{artifacts.map(row => <button key={row.id} type="button" role="menuitem" className="sections-menu__item" title={row.title} onClick={() => onPick('docs')}>
          <Icon name="file" size={13} /><span className="sections-menu__name">{row.title}</span><span className="sections-menu__meta">{compactAge(row.updatedAt)}</span>
        </button>)}</div>
      </>
  } else if (section === 'review') {
    body = changes.length === 0 ? <p className="sections-menu__empty">No changes in this task yet.</p> : <>
      <div className="sections-menu__group"><span>Changes</span><span>{changes.length}</span></div>
      <div className="sections-menu__items">{changes.map(change => <button key={change.path} type="button" role="menuitem" className="sections-menu__item" title={change.path} onClick={() => onPick('review', change.path)}>
        <Icon name="file" size={13} /><span className="sections-menu__name">{baseName(change.path)}</span>
        <span className="sections-menu__meta"><span className="add">+{change.adds}</span> <span className="del">−{change.dels}</span></span>
      </button>)}</div>
    </>
  } else {
    const shown = filterChanges(changes, needle)
    body = <>
      <label className="sections-menu__search"><Icon name="search" size={13} /><input type="search" aria-label="Filter changed files" placeholder="Search in files" value={needle} spellCheck={false} onChange={event => setNeedle(event.target.value)} /></label>
      {changes.length === 0 ? <p className="sections-menu__empty">No changed files. Open Files to browse the project.</p> : <>
        <div className="sections-menu__group"><span>Changed files</span><span>{shown.length}</span></div>
        <div className="sections-menu__items" role="menu" aria-label="Changed files">{shown.map(change => <button key={change.path} type="button" role="menuitem" className="sections-menu__item" title={change.path} onClick={() => onPick('files', change.path)}>
          <Icon name="file" size={13} /><span className="sections-menu__name">{baseName(change.path)}</span><ChangeBadge change={change} />
        </button>)}</div>
      </>}
    </>
  }
  const side = props.hoverSide ?? 'left'
  const place = side === 'left' ? { right: 'calc(100% + 4px)', left: 'auto' } : { left: 'calc(100% + 4px)', right: 'auto' }
  return <div className="sections-menu__preview" role={section === 'files' ? 'dialog' : 'menu'} aria-label={label} data-side={side}
    style={{ top: hoverTop, ...place, width: props.hoverWidth ?? 300, maxWidth: 'none' }}>{body}</div>
}

/** The list and, for the hovered section, its preview. Pure, so tests can render it open. */
export function SectionsMenuPanel(props: SectionsMenuPanelProps): ReactElement {
  const { hover, agents, artifacts, swarm, showSwarm, onHover, onPick } = props
  const sections = showSwarm || swarm !== undefined
    ? [MENU_SECTIONS[0]!, SWARM_MENU_SECTION, ...MENU_SECTIONS.slice(1)]
    : MENU_SECTIONS
  const swarmCounts = swarm ? swarmMemberCounts(swarm.members) : null
  const count = (id: MenuSection): number | null =>
    id === 'subagents'
      ? agents.length
      : id === 'swarm' && swarm?.active
        ? swarm.members.length
        : id === 'docs' && artifacts
          ? artifacts.length
          : null
  return <div className="sections-menu__list" role="menu" aria-label="Sections">
    {sections.map(section => {
      const n = count(section.id)
      const swarmLive = section.id === 'swarm' && Boolean(swarm?.active)
      // The chat pane clips overflow, so open the preview on whichever side has room inside it.
      const enter = (row: HTMLElement): void => {
        const list = row.parentElement?.getBoundingClientRect()
        const bound = (row.closest('.chat') ?? document.documentElement).getBoundingClientRect()
        const spaceLeft = list ? list.left - bound.left - 8 : 0
        const spaceRight = list ? bound.right - list.right - 8 : 0
        const side: PreviewSide = spaceLeft >= 300 || spaceLeft >= spaceRight ? 'left' : 'right'
        const width = Math.max(180, Math.min(300, side === 'left' ? spaceLeft : spaceRight))
        onHover(section.preview ? section.id : null, row.offsetTop, side, width)
      }
      return <button key={section.id} type="button" role="menuitem" className={`sections-menu__row${hover === section.id ? ' is-hover' : ''}`}
        aria-haspopup={section.preview ? 'menu' : undefined} aria-expanded={section.preview ? hover === section.id : undefined}
        onMouseEnter={event => enter(event.currentTarget)} onFocus={event => enter(event.currentTarget)} onClick={() => onPick(section.id)}>
        <Icon name={section.icon} size={14} /><span className="sections-menu__label">{section.label}</span>
        {swarmLive && (
          <span className="sections-menu__live" data-working={(swarmCounts?.working ?? 0) > 0 ? 'true' : undefined}>
            {(swarmCounts?.working ?? 0) > 0 ? `${swarmCounts!.working} working` : 'active'}
          </span>
        )}
        {n !== null && n > 0 && !swarmLive && <span className="sections-menu__count">{n}</span>}
        {section.preview && <Icon name="chevron" size={10} />}
      </button>
    })}
    {hover && <SectionPreview {...props} section={hover} />}
  </div>
}

export function SectionsMenu({ snap }: { snap: Snapshot }): ReactElement {
  const navigate = useDesktopNavigation()
  const [open, setOpen] = useState(false)
  const [hover, setHover] = useState<{ section: MenuSection; top: number; side: PreviewSide; width: number } | null>(null)
  const [needle, setNeedle] = useState('')
  const [artifacts, setArtifacts] = useState<readonly ArtifactRow[] | null>(null)
  const root = useRef<HTMLDivElement>(null)
  const agents = useMemo(() => activityFleetRows(snap.fleet, snap.blocks), [snap.fleet, snap.blocks])
  const showSwarm = showsImmortalsForSession(snap)
  useEffect(() => {
    if (!open) return
    let live = true
    // The count is optional: an older runtime without artifact methods just shows none.
    if (snap.connection === 'online') {
      void desktopCall(window.xerxes, snap.sessionKey, 'artifacts.list')
        .then(result => { if (live) setArtifacts(artifactRowsOf(result.artifacts)) })
        .catch(() => { if (live) setArtifacts(null) })
      void store.refreshSwarm({ silent: true })
    }
    const away = (event: MouseEvent): void => { if (event.target instanceof Node && !root.current?.contains(event.target)) setOpen(false) }
    const escape = (event: KeyboardEvent): void => {
      if (event.key !== 'Escape') return
      event.preventDefault()
      event.stopPropagation()
      setOpen(false)
    }
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', escape, true)
    return () => { live = false; document.removeEventListener('mousedown', away); document.removeEventListener('keydown', escape, true) }
  }, [open, snap.sessionKey, snap.connection])
  const pick = (section: MenuSection, path?: string): void => {
    setOpen(false); setHover(null); setNeedle('')
    if (section === 'subagents' && path) navigate('activity', path)
    else navigate(section, path)
  }
  return <div className="sections-menu" ref={root}>
    <button type="button" className="sections-menu__trigger" data-swarm-active={snap.swarm?.active ? 'true' : undefined} aria-label="Sections" title="Sections" aria-haspopup="menu" aria-expanded={open} onClick={() => { setOpen(value => !value); setHover(null) }}><Icon name="stack" size={15} /></button>
    {open && <SectionsMenuPanel hover={hover?.section ?? null} hoverTop={hover?.top ?? 0} hoverSide={hover?.side ?? 'left'} hoverWidth={hover?.width ?? 300} needle={needle} setNeedle={setNeedle} agents={agents} artifacts={artifacts} changes={snap.changes} swarm={snap.swarm ?? null} showSwarm={showSwarm}
      onHover={(section, top, side, width) => setHover(section ? { section, top, side: side ?? 'left', width: width ?? 300 } : null)} onPick={pick} />}
  </div>
}
