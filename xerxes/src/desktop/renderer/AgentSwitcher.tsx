// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { useEffect, useMemo, useRef, useState, type KeyboardEvent, type ReactElement, type ReactNode } from 'react'
import type { SessionRow } from './types.js'
import { agentKindLabel, agentState } from './AgentRoster.js'
import { Icon } from './Icon.js'

/** Rows the menu draws at once; a workflow's thousands of members are reached by filtering. */
const MENU_ROWS = 200

/** Working agents first, then those needing attention, then settled ones; the runtime's order within each. */
export function agentMenuRows(rows: readonly SessionRow[], filter = ''): readonly SessionRow[] {
  const needle = filter.trim().toLowerCase()
  return rows
    .map((row, index) => ({ row, index, priority: agentState(row.status).priority }))
    .filter(({ row }) => !needle || row.title.toLowerCase().includes(needle) || row.id.toLowerCase().includes(needle))
    .sort((a, b) => a.priority - b.priority || a.index - b.index)
    .map(({ row }) => row)
}

/**
 * The header's agent switcher (DeepSeek Harness's `N subagents ⌄`): a
 * dropdown of every sub-agent in the task with its state, model and tool
 * count. Picking one opens its conversation in place of the main one.
 */
export function AgentMenu({ rows, currentId, label, title, onPick, variant = 'crumb' }: {
  rows: readonly SessionRow[]
  currentId?: string
  label: ReactNode
  title: string
  onPick: (id: string) => void
  variant?: 'crumb' | 'count'
}): ReactElement {
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const root = useRef<HTMLDivElement>(null)
  const trigger = useRef<HTMLButtonElement>(null)
  const list = useRef<HTMLDivElement>(null)
  useEffect(() => {
    if (!open) return
    const away = (event: MouseEvent) => { if (event.target instanceof Node && !root.current?.contains(event.target)) { setOpen(false); setFilter('') } }
    document.addEventListener('mousedown', away)
    // Focus lands where the keyboard can pick at once: the filter, else the open agent, else the first row.
    const frame = requestAnimationFrame(() => {
      const node = root.current
      const target = node?.querySelector<HTMLElement>('.agent-menu__filter') ?? node?.querySelector<HTMLElement>('[aria-current="true"]') ?? node?.querySelector<HTMLElement>('.agent-menu__row')
      target?.focus()
    })
    return () => { document.removeEventListener('mousedown', away); cancelAnimationFrame(frame) }
  }, [open])
  const shown = useMemo(() => open ? agentMenuRows(rows, filter) : [], [open, rows, filter])
  const close = (): void => { setOpen(false); setFilter(''); trigger.current?.focus() }
  const keys = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (!open) return
    // The window's Escape stops a running turn; inside the menu it only closes the menu.
    if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); close(); return }
    if (event.key !== 'ArrowDown' && event.key !== 'ArrowUp') return
    const items = [...(list.current?.querySelectorAll<HTMLButtonElement>('.agent-menu__row') ?? [])]
    if (!items.length) return
    event.preventDefault()
    const at = items.findIndex(item => item === document.activeElement)
    const next = at < 0 ? 0 : (at + (event.key === 'ArrowDown' ? 1 : items.length - 1)) % items.length
    items[next]?.focus()
  }
  return <div className={`agent-menu agent-menu--${variant}`} ref={root} onKeyDown={keys}>
    <button ref={trigger} className="agent-menu__trigger" aria-haspopup="menu" aria-expanded={open} title={title} onClick={() => { if (open) close(); else setOpen(true) }}>
      <span className="agent-menu__label">{label}</span><Icon name="caretDown" size={11} />
    </button>
    {open && <div className="agent-menu__pop" role="menu" aria-label={title}>
      {rows.length > 8 && <input className="agent-menu__filter" aria-label="Filter agents" placeholder="Filter agents" value={filter} onChange={event => setFilter(event.target.value)} />}
      <div className="agent-menu__list" ref={list}>
        {shown.slice(0, MENU_ROWS).map(row => {
          const state = agentState(row.status)
          const tools = row.agentDetails?.toolCount
          const sub = [state.label, agentKindLabel(row.agentDetails), tools ? `${tools} tool${tools === 1 ? '' : 's'}` : ''].filter(Boolean).join(' · ')
          const current = row.id === currentId
          return <button key={row.id} role="menuitem" className="agent-menu__row" data-tone={state.tone} aria-current={current ? 'true' : undefined} onClick={() => { setOpen(false); setFilter(''); onPick(row.id) }}>
            <span className="agent-menu__dot" aria-hidden="true" />
            <span className="agent-menu__text"><span className="agent-menu__title">{row.title}</span><span className="agent-menu__sub">{sub}</span></span>
            <Icon name={current ? 'check' : 'chevron'} size={12} />
          </button>
        })}
        {!shown.length && <p className="agent-menu__more">No agent matches.</p>}
        {shown.length > MENU_ROWS && <p className="agent-menu__more">{shown.length - MENU_ROWS} more. Filter to find them.</p>}
      </div>
    </div>}
  </div>
}
