// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The agents a turn started, drawn in the conversation: one agent or a
 * workflow of thousands. A header with the run's name, live clock and a
 * segmented progress bar; then the agents grouped by workflow phase, each
 * with its model, what it is doing right now, and its clock. Past a few
 * dozen agents a phase becomes a dot grid (one dot per agent, click to
 * inspect) with rows kept for the ones still running or failed.
 */

import { useEffect, useMemo, useState, type ReactElement } from 'react'

import { AgentOrb } from './AgentOrb.js'
import { agentKindLabel } from './AgentRoster.js'
import { useDesktopNavigation } from './DesktopPanels.js'
import { elapsedOf } from './duration.js'
import { Icon } from './Icon.js'
import { pageIsBackground } from './pageVisibility.js'
import type { AgentMember } from './types.js'

type Tone = 'working' | 'done' | 'failed' | 'stopped'

/** Rows shown per phase before "Show all". */
const ROWS = 8
/** A phase this large draws as a dot grid, with rows only for live and failed agents. */
const DENSE_AT = 60
/** Rows rendered per page once "Show all" is open, so thousands never mount at once. */
const PAGE = 200

/** An agent's state as the card draws it. */
export function memberTone(status: string): Tone {
  if (['working', 'running', 'acting', 'queued', 'pending', 'starting'].includes(status)) return 'working'
  if (['failed', 'error', 'timeout'].includes(status)) return 'failed'
  if (['cancelled', 'canceled', 'stopped', 'interrupted'].includes(status)) return 'stopped'
  return 'done'
}

const ORDER: Record<Tone, number> = { working: 0, failed: 1, stopped: 2, done: 3 }
const WORD: Record<Tone, string> = { working: 'Working', done: 'Done', failed: 'Failed', stopped: 'Stopped' }

interface Row { readonly member: AgentMember; readonly tone: Tone }
interface Phase { readonly name: string; readonly rows: readonly Row[] }

function counts(rows: readonly Row[]): Record<Tone, number> {
  const out: Record<Tone, number> = { working: 0, done: 0, failed: 0, stopped: 0 }
  for (const row of rows) out[row.tone] += 1
  return out
}

/** Phases in the order the run first reached them; ungrouped agents form one unnamed phase. */
function phasesOf(rows: readonly Row[]): Phase[] {
  const order: string[] = []
  const buckets = new Map<string, Row[]>()
  for (const row of rows) {
    const name = row.member.group?.phase ?? ''
    let bucket = buckets.get(name)
    if (!bucket) { bucket = []; buckets.set(name, bucket); order.push(name) }
    bucket.push(row)
  }
  return order.map(name => ({ name, rows: buckets.get(name)!.slice().sort((a, b) => ORDER[a.tone] - ORDER[b.tone]) }))
}

/** A ticking clock, only while something is running and the window is visible. */
function useNow(active: boolean): number {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    if (!active) return
    const timer = setInterval(() => { if (!pageIsBackground()) setNow(Date.now()) }, 1_000)
    return () => clearInterval(timer)
  }, [active])
  return now
}

function clock(member: AgentMember, now: number): string {
  if (member.startedAt === undefined) return ''
  const end = member.finishedAt ?? now
  return elapsedOf(Math.max(0, (end - member.startedAt) / 1_000))
}

function tokensLabel(tokens: number | undefined): string {
  if (!tokens) return ''
  return tokens >= 1_000_000 ? `${(tokens / 1_000_000).toFixed(1)}M tok` : tokens >= 1_000 ? `${Math.round(tokens / 1_000)}K tok` : `${tokens} tok`
}

function Glyph({ tone }: { tone: Tone }): ReactElement {
  if (tone === 'working') return <Icon name="spinner" size={13} />
  if (tone === 'done') return <Icon name="check" size={12} />
  if (tone === 'failed') return <Icon name="close" size={11} />
  return <span className="acard__dash" />
}

function AgentRow({ row, now, open }: { row: Row; now: number; open: (id: string) => void }): ReactElement {
  const { member, tone } = row
  const kind = agentKindLabel(member)
  const detail = tone === 'working' ? member.activity : tone === 'failed' ? member.error?.split('\n')[0] : member.summary
  const time = clock(member, now)
  return <li>
    <button className="acard__row" data-tone={tone} onClick={() => open(member.runtimeId || member.key)} aria-label={`Inspect agent: ${member.title}`}>
      <span className="acard__glyph" aria-hidden="true"><Glyph tone={tone} /></span>
      <span className="acard__body">
        <span className="acard__t">{member.title}</span>
        {(kind || detail) && <span className="acard__meta">
          {kind && <span className="acard__kind">{kind}</span>}
          {detail && <span className={tone === 'failed' ? 'acard__error' : 'acard__act'} title={tone === 'failed' ? member.error : detail}>{detail}</span>}
        </span>}
      </span>
      <span className="acard__s">{time ? <span className="acard__time">{time}</span> : null}{tone === 'working' && member.tokens ? <span className="acard__tok">{tokensLabel(member.tokens)}</span> : null}{!time && WORD[tone]}</span>
      <Icon name="chevron" size={12} />
    </button>
  </li>
}

function DotGrid({ rows, open }: { rows: readonly Row[]; open: (id: string) => void }): ReactElement {
  return <div className="acard__grid" role="list" aria-label={`${rows.length} agents`}>
    {rows.map(({ member, tone }) => <button
      key={member.key}
      role="listitem"
      className="acard__dot"
      data-tone={tone}
      title={`${member.title} · ${WORD[tone]}${member.activity && tone === 'working' ? ` · ${member.activity}` : ''}`}
      aria-label={`${member.title}: ${WORD[tone]}`}
      onClick={() => open(member.runtimeId || member.key)}
    />)}
  </div>
}

function PhaseSection({ phase, index, total, now, open, expanded, onExpand }: {
  phase: Phase; index: number; total: number; now: number; open: (id: string) => void; expanded: boolean; onExpand: () => void
}): ReactElement {
  const tally = counts(phase.rows)
  const dense = phase.rows.length >= DENSE_AT
  const [page, setPage] = useState(1)
  const state: Tone = tally.working ? 'working' : tally.failed ? 'failed' : 'done'
  let listed: readonly Row[]
  if (expanded) listed = phase.rows.slice(0, page * PAGE)
  else if (dense) listed = [...phase.rows.filter(row => row.tone === 'working').slice(0, ROWS), ...phase.rows.filter(row => row.tone === 'failed').slice(0, ROWS)]
  else listed = phase.rows.slice(0, ROWS)
  const hidden = phase.rows.length - listed.length
  return <div className="acard__phase" data-state={state}>
    {(phase.name || total > 1) && <div className="acard__phasehead">
      <span className="acard__phaseicon" aria-hidden="true">{state === 'working' ? <Icon name="spinner" size={12} /> : state === 'failed' ? <Icon name="close" size={10} /> : <Icon name="check" size={11} />}</span>
      <span className="acard__phasename">{phase.name || `Batch ${index + 1}`}</span>
      <span className="acard__phasemeta">
        {phase.rows.length} agent{phase.rows.length === 1 ? '' : 's'}
        {tally.working > 0 && ` · ${tally.working} running`}
        {tally.done > 0 && ` · ${tally.done} done`}
        {tally.failed > 0 && ` · ${tally.failed} failed`}
      </span>
    </div>}
    {dense && !expanded && <DotGrid rows={phase.rows} open={open} />}
    {listed.length > 0 && <ul className="acard__list">{listed.map(row => <AgentRow key={row.member.key} row={row} now={now} open={open} />)}</ul>}
    {(hidden > 0 || expanded) && <div className="acard__more">
      {!expanded && hidden > 0 && <button onClick={onExpand}>Show all {phase.rows.length}</button>}
      {expanded && phase.rows.length > page * PAGE && <button onClick={() => setPage(value => value + 1)}>Show {Math.min(PAGE, phase.rows.length - page * PAGE)} more</button>}
      {expanded && <button onClick={onExpand}>Show fewer</button>}
    </div>}
  </div>
}

export function AgentsCard({ members }: { members: readonly AgentMember[] }): ReactElement {
  const navigate = useDesktopNavigation()
  const [open, setOpen] = useState(true)
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(() => new Set())
  const rows = useMemo(() => members.map(member => ({ member, tone: memberTone(member.status) })), [members])
  const tally = counts(rows)
  const phases = useMemo(() => phasesOf(rows), [rows])
  const now = useNow(tally.working > 0)
  const label = members.find(member => member.group?.label)?.group?.label
  const settled = members.length - tally.working
  const starts = members.map(member => member.startedAt).filter((value): value is number => value !== undefined)
  const first = starts.length ? Math.min(...starts) : undefined
  const last = tally.working ? now : Math.max(0, ...members.map(member => member.finishedAt ?? 0))
  const elapsed = first !== undefined && last > first ? elapsedOf((last - first) / 1_000) : ''
  const models = [...new Set(members.map(member => member.model).filter((model): model is string => Boolean(model)))]
  const inspect = (id: string) => navigate('activity', id)
  const title = label ?? `${members.length} subagent${members.length === 1 ? '' : 's'}`
  return (
    <section className="acard" data-state={tally.working ? 'working' : tally.failed ? 'failed' : 'done'} aria-label={label ? `Workflow: ${label}` : 'Subagents'}>
      <button className="acard__head" onClick={() => setOpen(value => !value)} aria-expanded={open}>
        <span className="acard__icon">{tally.working ? <AgentOrb size={20} state="connecting" /> : <Icon name="agent" size={15} />}</span>
        <span className="acard__heading">
          <span className="acard__title">{title}</span>
          <span className="acard__sub">
            {label && `${members.length} agent${members.length === 1 ? '' : 's'}`}
            {phases.length > 1 && `${label ? ' · ' : ''}${phases.length} phases`}
            {models.length > 0 && models.length <= 3 && `${label || phases.length > 1 ? ' · ' : ''}${models.map(shortModel).join(', ')}`}
            {models.length > 3 && `${label || phases.length > 1 ? ' · ' : ''}${models.length} models`}
          </span>
        </span>
        <span className="acard__counts">
          {tally.working > 0 && <span data-tone="working">{tally.working} working</span>}
          {tally.done > 0 && <span data-tone="done">{tally.done} done</span>}
          {tally.failed > 0 && <span data-tone="failed">{tally.failed} failed</span>}
          {tally.stopped > 0 && <span data-tone="stopped">{tally.stopped} stopped</span>}
          {elapsed && <span className="acard__clock">{elapsed}</span>}
        </span>
        <span className={`acard__chev${open ? ' is-open' : ''}`}><Icon name="caretDown" size={12} /></span>
      </button>
      <div className="acard__progress" role="progressbar" aria-label="Subagents finished" aria-valuemin={0} aria-valuemax={members.length} aria-valuenow={settled}>
        {tally.done > 0 && <span data-tone="done" style={{ flexGrow: tally.done }} />}
        {tally.failed > 0 && <span data-tone="failed" style={{ flexGrow: tally.failed }} />}
        {tally.stopped > 0 && <span data-tone="stopped" style={{ flexGrow: tally.stopped }} />}
        {tally.working > 0 && <span data-tone="working" style={{ flexGrow: tally.working }} />}
      </div>
      {open && phases.map((phase, index) => <PhaseSection
        key={phase.name || `batch-${index}`}
        phase={phase}
        index={index}
        total={phases.length}
        now={now}
        open={inspect}
        expanded={expanded.has(phase.name)}
        onExpand={() => setExpanded(current => {
          const next = new Set(current)
          if (next.has(phase.name)) next.delete(phase.name)
          else next.add(phase.name)
          return next
        })}
      />)}
      <footer className="acard__foot">
        <button onClick={() => navigate('activity')}>Open in Activity</button>
      </footer>
    </section>
  )
}

/** `claude-code/sonnet` → `sonnet`; vendor prefixes are noise at card size. */
function shortModel(model: string): string {
  return model.split('/').pop() || model
}
