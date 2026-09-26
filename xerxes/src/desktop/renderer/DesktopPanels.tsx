// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { WorkspaceFileTree } from './WorkspaceFileTree.js'
import { RemoteProviders } from './RemoteProviders.js'
import { DiffPreview } from './DiffPreview.js'
import { WorkspaceReview } from "./WorkspaceReview.js"
import { CommandActivity } from './CommandActivity.js'
import { Deliveries } from "./Deliveries.js"
import { MonitorsDisclosure } from "./Monitors.js"
import { UnifiedRuns } from "./UnifiedRuns.js"
import { ContextInspectorDisclosure } from "./ContextInspector.js"
import {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
  type ReactElement,
  type ReactNode,
} from 'react'
import {
  desktopCall,
  desktopError,
  scheduleTime,
  cronInEnglish,
  record,
  records,
  specialistsOf,
  text,
  type RpcRecord,
  type Specialist,
} from './desktopRpc.js'
import { store, type Snapshot } from './store.js'
import { Markdown } from './markdown.js'
import { TerminalsCard } from './TerminalsPanel.js'
import { Icon } from './Icon.js'
import { GitPanel } from './GitPanel.js'
import { TerminalPanel } from './TerminalPanel.js'
import { UsagePanel } from './UsagePanel.js'
import { useDialogFocus } from './dialogFocus.js'
import { RunHistory, type RunHistoryPage } from './RunHistory.js'

export type DesktopPanel =
  | 'artifacts'
  | 'agents'
  | 'extensions'
  | 'schedules'
  | 'activity'
  | 'files'
  | 'review'
  | 'terminal'
  | 'usage'
  | 'workspace'
  | 'snapshots'
  | null
/** Views that open in the task-context rail beside the conversation. */
export type RailPanel = 'files' | 'review' | 'terminal' | 'activity' | 'usage'
export const RAIL_PANELS: readonly RailPanel[] = ['files', 'review', 'terminal', 'activity', 'usage']
export const railTabName = (panel: RailPanel): string =>
  panel === 'review' ? 'Git' : panel === 'files' ? 'Files' : panel === 'terminal' ? 'Terminal' : panel === 'usage' ? 'Usage' : 'Activity'
export const DesktopNavigation = createContext<(panel: DesktopPanel, filePath?: string) => void>(() => {})
export const useDesktopNavigation = () => useContext(DesktopNavigation)

function useRequest(snap: Snapshot) {
  const [busy, setBusy] = useState(false),
    [error, setError] = useState('')
  const alive = useRef(true),
    tail = useRef(Promise.resolve()),
    pending = useRef(0)
  useEffect(() => {
    alive.current = true
    return () => {
      alive.current = false
    }
  }, [])
  const run = (work: () => Promise<void>): Promise<void> => {
    pending.current++
    setBusy(true)
    const next = tail.current.then(async () => {
      if (!alive.current) return
      setError('')
      try {
        await work()
      } catch (failure) {
        if (alive.current) setError(desktopError(failure))
      } finally {
        pending.current--
        if (alive.current) setBusy(pending.current > 0)
      }
    })
    tail.current = next
    return next
  }
  const call = (method: string, params: RpcRecord = {}) =>
    desktopCall(window.xerxes, snap.sessionKey, method, params)
  return { busy, error, run, call, alive }
}
function Feedback({ busy, error }: { busy: boolean; error: string }) {
  return (
    <>
      {busy && (
        <div className="studio-progress" role="status">
          <span className="studio-spinner" />
          Working…
        </div>
      )}
      {error && (
        <p className="studio-error" role="alert">
          {error}
        </p>
      )}
    </>
  )
}
function Empty({ children }: { children: ReactNode }) {
  return <div className="studio-empty">{children}</div>
}

/** Durable destinations share the shell and never replace the composer. */
export function DesktopPage({ panel, snap }: { panel: 'agents' | 'extensions' | 'artifacts'; snap: Snapshot }): ReactElement {
  const close = useDesktopNavigation()
  return <section className={"desktop-page studio-sheet" + (panel === "extensions" ? " catalog-page" : "")} aria-label={panel === 'agents' ? 'Agents' : panel === 'artifacts' ? 'Artifacts' : 'Skills & tools'}>
    {/* A full-page destination had no way out of itself: no close button,
        and the tab bar is hidden while it is open, so Escape was the only
        thing left to try — and Escape cancelled the running turn. */}
    <header><div><h2>{panel === 'agents' ? 'Agents' : panel === 'artifacts' ? 'Artifacts' : 'Skills & tools'}</h2><p className="studio-muted">{panel === 'agents' ? 'Specialists you can bring into a conversation.' : panel === 'artifacts' ? 'Files this task changed, and its exportable transcript.' : 'Instructions and tools available in this workspace.'}</p></div><button className="desktop-page__close" aria-label="Back to the conversation" title="Back to the conversation (esc)" onClick={() => close(null)}><Icon name="close" size={13} /></button></header>
    <div className="studio-sheet-content" key={`${panel}:${snap.cwd}:${snap.sessionKey}`}>{panel === 'agents' ? <SpecialistsPanel snap={snap} /> : panel === 'artifacts' ? <ArtifactsPanel snap={snap} /> : <ExtensionsPanel snap={snap} />}</div>
  </section>
}

/** Nonmodal task context: the conversation and draft remain interactive. */
export function DesktopRail({ panel, snap, close, activityDetails, filesExpanded = false, toggleFilesExpanded, setExpanded, reviewPath = '', activityFocused = false }: {
  panel: RailPanel; snap: Snapshot; close: () => void; activityDetails: ReactNode; filesExpanded?: boolean; toggleFilesExpanded?: () => void
  /** Present when the window is wide enough to choose between rail and full width. */
  setExpanded?: (value: boolean) => void
  reviewPath?: string
  activityFocused?: boolean
}): ReactElement {
  const open = useDesktopNavigation()
  // Without the toggle the window is too narrow for a side rail: it is already full width.
  const wide = filesExpanded || !toggleFilesExpanded
  return <aside className={`desktop-rail studio-sheet${panel === "review" ? ` desktop-rail--git${wide ? " desktop-rail--review" : ""}` : panel === "terminal" ? " desktop-rail--terminal" : ""}`} aria-label="Task context">
    <header><nav aria-label="Task context views">{RAIL_PANELS.map(value => <button key={value} aria-pressed={panel === value} onClick={() => open(value)}>{railTabName(value)}</button>)}</nav>{toggleFilesExpanded && <button aria-label={filesExpanded ? 'Restore conversation' : panel === 'files' ? 'Expand files workspace' : panel === 'terminal' ? 'Expand terminal' : panel === 'review' ? 'Show changes' : panel === 'usage' ? 'Expand usage' : 'Expand Activity workspace'} title={filesExpanded ? 'Restore conversation' : panel === 'files' ? 'Expand files workspace' : panel === 'terminal' ? 'Expand terminal' : panel === 'review' ? 'Show changes' : panel === 'usage' ? 'Expand usage' : 'Expand Activity workspace'} aria-pressed={filesExpanded} onClick={toggleFilesExpanded}><Icon name={filesExpanded ? 'collapse' : 'expand'} size={15} /></button>}<button aria-label="Close task context" onClick={close}><Icon name="close" size={13} /></button></header>
    <div className="studio-sheet-content" key={`${panel}:${snap.cwd}:${snap.sessionKey}`}>
      {panel === 'files' && <FilesPanel snap={snap} close={close} />}
      {panel === 'review' && <GitPanel snap={snap} initialPath={reviewPath} expanded={wide} {...(setExpanded ? { onExpand: () => setExpanded(true) } : {})} onSnapshots={() => open('snapshots')} onReviewSent={() => { setExpanded?.(false); open('activity') }} />}
      {panel === 'terminal' && <TerminalPanel snap={snap} />}
      {panel === 'usage' && <UsagePanel snap={snap} />}
      {panel === 'activity' && <>{activityDetails}{!activityFocused && <ActivityPanel snap={snap} />}</>}
    </div>
  </aside>
}

export function DesktopSheet({
  panel,
  snap,
  close,
  activityDetails,
}: {
  activityDetails?: ReactNode
  panel: Exclude<DesktopPanel, null>
  snap: Snapshot
  close: () => void
}): ReactElement {
  const ref = useRef<HTMLDivElement>(null)
  useDialogFocus(ref)
  useEffect(() => {
    const key = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopImmediatePropagation()
        close()
      }
    }
    document.addEventListener('keydown', key, true)
    return () => {
      document.removeEventListener('keydown', key, true)
    }
  }, [])
  const names = {
    artifacts: 'Artifacts',
    agents: 'Specialists',
    extensions: 'Skills & tools',
    schedules: 'Scheduled jobs',
    activity: 'Activity',
    files: 'Project files',
    review: 'Git',
    terminal: 'Terminal',
    usage: 'Usage',
    workspace: 'Workspace',
    snapshots: 'Snapshots',
  }
  return (
    <div
      className="studio-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) close()
      }}
      onKeyDown={(event) => {
        event.stopPropagation()
        if (event.key === 'Escape') close()
      }}
    >
      <div
        className={"studio-sheet" + (panel === "workspace" ? " workspace-sheet" : "")}
        ref={ref}
        role="dialog"
        aria-modal="true"
        aria-label={names[panel]}
      >
        <header>
          <h2>{names[panel]}</h2>
          <button aria-label="Close dialog" onClick={close}>
            <Icon name="close" size={13} />
          </button>
        </header>
        <div className="studio-sheet-content" key={`${panel}:${snap.cwd}:${snap.sessionKey}`}>
          {panel === 'artifacts' && <ArtifactsPanel snap={snap} />}
          {panel === 'agents' && <SpecialistsPanel snap={snap} />}
          {panel === 'extensions' && <ExtensionsPanel snap={snap} />}
          {panel === 'schedules' && <SchedulesPanel snap={snap} />}
          {panel === 'activity' && (
            <>
              {activityDetails}
              <ActivityPanel snap={snap} />
            </>
          )}
          {panel === 'snapshots' && <SnapshotsPanel snap={snap} />}
          {panel === 'review' && <GitPanel snap={snap} expanded />}
          {panel === 'terminal' && <TerminalPanel snap={snap} />}
          {panel === 'usage' && <UsagePanel snap={snap} />}
          {panel === 'files' && <FilesPanel snap={snap} close={close} />}
          {panel === 'workspace' && <WorkspacePanel snap={snap} />}
        </div>
      </div>
    </div>
  )
}

function ArtifactsPanel({ snap }: { snap: Snapshot }): ReactElement {
  const open = useDesktopNavigation()
  const request = useRequest(snap)
  return <div className="studio-form">
    <Feedback {...request} />
    <button className="btn btn--ghost" disabled={!snap.currentId || request.busy || snap.connection !== 'online'} onClick={() => void request.run(() => store.downloadSessionTranscript(snap.sessionKey))}>Export session transcript</button>
    {snap.changes.map(file => <div className="studio-item" key={file.path}><div><strong>{file.path}</strong><p>+{file.adds} −{file.dels}</p></div><button onClick={() => open('review', file.path)}>Review changes <Icon name="arrow" size={12} /></button></div>)}
    {!snap.changes.length && <p className="studio-muted">No files changed in this session yet.</p>}
  </div>
}

export function SpecialistsPanel({ snap }: { snap: Snapshot }): ReactElement {
  const open = useDesktopNavigation()
  const request = useRequest(snap)
  const [rows, setRows] = useState<Specialist[]>([]),
    [selected, setSelected] = useState('')
  const [loaded, setLoaded] = useState(false),
    [mode, setMode] = useState<'list' | 'generate' | 'edit'>('list')
  const [description, setDescription] = useState(''),
    [content, setContent] = useState(''),
    [revision, setRevision] = useState<string | null>(null)
  const load = async () => {
    const result = await request.call('agentPreset.projectList')
    const next = specialistsOf(result.agents)
    if (!request.alive.current) return
    setRows(next)
    setSelected((id) => (next.some((row) => row.id === id) ? id : (next[0]?.id ?? '')))
    setLoaded(true)
  }
  useEffect(() => {
    void request.run(load)
  }, [])
  const active = rows.find((row) => row.id === selected)
  const edit = () =>
    request.run(async () => {
      const result = await request.call('agentPreset.projectRead', { id: selected })
      if (typeof result.content !== 'string' || typeof result.revision !== 'string')
        throw new Error('Invalid specialist draft')
      if (request.alive.current) {
        setContent(result.content)
        setRevision(result.revision)
        setMode('edit')
      }
    })
  const generate = () =>
    request.run(async () => {
      if (!description.trim()) throw new Error('Describe the specialist first.')
      const result = await request.call('agentPreset.projectGenerate', { description })
      if (!text(result.id) || typeof result.content !== 'string')
        throw new Error('Invalid generated specialist')
      if (request.alive.current) {
        setSelected(text(result.id))
        setContent(result.content)
        setRevision(null)
        setMode('edit')
      }
    })
  const save = () =>
    request.run(async () => {
      await request.call('agentPreset.projectWrite', {
        id: revision === null ? '' : selected,
        content,
        revision,
      })
      if (!request.alive.current) return
      setMode('list')
      await load()
    })
  return (
    <div
      onKeyDown={(event) => {
        if (
          event.key.toLowerCase() === 'g' &&
          mode === 'list' &&
          !(event.target instanceof HTMLInputElement) &&
          !request.busy
        ) {
          event.preventDefault()
          setMode('generate')
        }
      }}
    >
      <Feedback {...request} />
      <div className="agents-workspace"><span title={snap.cwd}>{snap.cwd.replace(/^\/(?:Users|home)\/[^/]+(?=\/|$)/, '~')}</span><button disabled={request.busy} onClick={() => store.chooseWorkspace()}>Change workspace…</button></div>
      {mode === 'generate' ? (
        <div className="studio-form">
          <h1>Who would help?</h1>
          <p>Your current model drafts a specialist. Review the instructions before saving.</p>
          <label>
            Description
            <textarea
              rows={5}
              disabled={request.busy}
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder="A JAX reviewer who diagnoses sharding mistakes and verifies array shapes…"
            />
          </label>
          <div className="studio-actions">
            <button disabled={request.busy} onClick={() => setMode('list')}>
              Back
            </button>
            <button
              className="studio-primary"
              disabled={request.busy || !description.trim()}
              onClick={() => void generate()}
            >
              {request.busy ? 'Generating…' : <>Draft specialist <Icon name="arrow" size={12} /></>}
            </button>
          </div>
        </div>
      ) : mode === 'edit' ? (
        <div className="studio-form">
          <label>
            Instructions · {selected}
            <textarea
              className="studio-source"
              rows={18}
              disabled={request.busy}
              value={content}
              onChange={(event) => setContent(event.target.value)}
            />
          </label>
          <p className="studio-muted">
            The name and description are exposed to the model for delegation. Full instructions load
            when used.
          </p>
          <div className="studio-actions">
            <button disabled={request.busy} onClick={() => setMode('list')}>
              Cancel
            </button>
            <button className="studio-primary" disabled={request.busy} onClick={() => void save()}>
              Save specialist
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className={rows.length ? "studio-split" : "agents-empty-layout"}>
            <nav hidden={!rows.length}>
              {rows.map((row) => (
                <button
                  key={row.id}
                  className={row.id === selected ? 'is-selected' : ''}
                  disabled={request.busy}
                  onClick={() => setSelected(row.id)}
                >
                  {row.id}
                  {row.error ? <span className="agent-list__warn" title={row.error} aria-label="Needs fixing"><Icon name="warning" size={13} /></span> : null}
                </button>
              ))}
            </nav>
            <section className="agent-detail">
              {active ? (
                <>
                  <h3>{active.id}</h3>
                  <div className="studio-label">When to delegate</div>
                  <p>{active.description}</p>
                  {active.error && <p role="alert">{active.error}</p>}
                  <p className="studio-muted">Project agent · .xerxes/agents</p>
                  <button disabled={request.busy || !!active.error} onClick={() => {
                    window.dispatchEvent(new CustomEvent('xerxes:add-context', { detail: 'Ask the ' + active.id + ' agent to ' }))
                    open(null)
                  }}>Use in conversation <Icon name="arrow" size={12} /></button>
                  <button disabled={request.busy} onClick={() => void edit()}>
                    Edit instructions
                  </button>
                </>
              ) : loaded ? (
                <Empty>
                  <h3>Create your first agent</h3>
                  <p>Give an agent a role and instructions for this workspace.</p>
                </Empty>
              ) : null}
            </section>
          </div>
          <footer className={rows.length ? "studio-actions" : "studio-actions agents-empty-actions"}>
            <button
              disabled={request.busy}
              onClick={() => {
                setSelected('new-agent')
                setContent(
                  '---\nname: new-agent\ndescription: Describe when to delegate to this specialist.\n---\nYou are a specialist.\n',
                )
                setRevision(null)
                setMode('edit')
              }}
            >
              New manually
            </button>
            <button
              className="studio-primary"
              disabled={request.busy}
              onClick={() => setMode('generate')}
            >
              Generate · G
            </button>
          </footer>
        </>
      )}
    </div>
  )
}

function ExtensionsPanel({ snap }: { snap: Snapshot }): ReactElement {
  const open = useDesktopNavigation()
  const request = useRequest(snap),
    [catalog, setCatalog] = useState<RpcRecord | null>(null),
    [tab, setTab] = useState<'skills' | 'tools' | 'plugins'>('skills'),
    [detail, setDetail] = useState(''),
    [needle, setNeedle] = useState(''),
    [selectedSkill, setSelectedSkill] = useState(''),
    [path, setPath] = useState(''),
    [plugins, setPlugins] = useState<RpcRecord[]>([])
  useEffect(() => {
    void request.run(async () => {
      const result = await request.call('capabilities.list')
      if (request.alive.current) setCatalog(result)
    })
  }, [])
  const loadPlugins = async () => {
    const result = await request.call('slash', { command: '/plugins list' })
    if (request.alive.current) setPlugins(records(result.inventory))
  }
  useEffect(() => {
    if (tab === 'plugins') void request.run(loadPlugins)
  }, [tab])
  // These ran inside render, so an unexpected daemon shape threw during
  // the render pass and (with no boundary above) blanked the window. The
  // panel now degrades to its own error state instead.
  let list: RpcRecord[] = []
  let catalogError = ''
  try { list = catalog && tab !== 'plugins' ? records(catalog[tab] ?? []) : [] }
  catch (failure) { catalogError = desktopError(failure) }
  const skillCount = ((): number | null => {
    try { return catalog ? records(catalog.skills).length : null } catch { return null }
  })()
  return (
    <div className={"studio-form catalog-form" + (tab === "plugins" ? " catalog-form--plugins" : "")}>
      {catalogError && <p className="studio-error" role="alert">This runtime returned a {tab} catalog this app build does not understand. {catalogError}</p>}
      <div className="studio-tabs">
        {(['skills', 'tools', 'plugins'] as const).map((name) => (
          <button
            className={name === tab ? 'is-selected' : ''}
            key={name}
            onClick={() => {
              setTab(name)
              setDetail('')
              setSelectedSkill('')
              setNeedle('')
            }}
          >
            {name}{name === 'skills' && skillCount !== null ? ' · ' + skillCount : ''}
          </button>
        ))}
        <button
          onClick={() => {
            open(null)
            store.openSettings('mcp')
          }}
        >
          MCP servers
        </button>
      </div>
      <Feedback {...request} />
      {tab === 'plugins' ? (
        <>
          {plugins.map((plugin) => (
            <div className="studio-item" key={text(plugin.name)}>
              <div>
                <strong>{text(plugin.name)}</strong>
                <p>{text(plugin.description)}</p>
                <small>{plugin.enabled === false ? 'Disabled' : 'Registered'}</small>
              </div>
              {typeof plugin.enabled === 'boolean' && (
                <button
                  disabled={request.busy}
                  onClick={() =>
                    void request.run(async () => {
                      await request.call('slash', {
                        command:
                          '/plugins ' +
                          (plugin.enabled ? 'disable ' : 'enable ') +
                          pluginArgument(text(plugin.name)),
                      })
                      await loadPlugins()
                    })
                  }
                >
                  {plugin.enabled ? 'Disable' : 'Enable'}
                </button>
              )}
              <button onClick={() => setDetail(JSON.stringify(plugin, null, 2))}>Inspect</button>
            </div>
          ))}
          <p>Install a local plugin module or create a bundle in Creator mode.</p>
          <label>
            Module path
            <input
              value={path}
              onChange={(event) => setPath(event.target.value)}
              placeholder="/path/to/plugin.ts"
            />
          </label>
          <button
            disabled={request.busy || !path.trim()}
            onClick={() =>
              void request.run(async () => {
                const result = await request.call('slash', {
                  command: `/plugins install ${pluginArgument(path)}`,
                })
                if (request.alive.current) {
                  setDetail(text(result.output) || 'Plugin installed.')
                  await loadPlugins()
                }
              })
            }
          >
            Install plugin
          </button>
          <button
            disabled={snap.turnActive}
            onClick={() => {
              open(null)
              store.startCreatorMode()
            }}
          >
            Open Creator mode
          </button>
        </>
      ) : (
        <>
          <input className="catalog-search" aria-label="Search skills and tools" placeholder={'Search all ' + list.length + ' ' + tab + '…'} value={needle} onChange={event => setNeedle(event.target.value)} />
          {catalog && !list.length && <Empty>No {tab} are available in this session.</Empty>}
          <div className="studio-split catalog-browser"><nav aria-label="Installed catalog">
          {list.filter(row => [text(row.name), text(row.description), text(row.source)].join(' ').toLowerCase().includes(needle.toLowerCase())).map(row => (
            <button className={selectedSkill === text(row.name) ? 'is-selected' : ''} key={text(row.name)} disabled={request.busy} onClick={() => void request.run(async () => {
              setSelectedSkill(text(row.name)); setDetail('')
              if (tab === 'skills') {
                const result = await request.call('capabilities.inspect', { name: row.name })
                if (request.alive.current) setDetail(text(result.instructions) + (result.truncated === true ? '\n\nPreview truncated. Source: ' + text(result.source) : ''))
              } else setDetail(JSON.stringify(row, null, 2))
            })}><strong>{text(row.name)}</strong><small>{text(row.description)}</small></button>
          ))}
          {list.length > 0 && !list.some(row => [text(row.name), text(row.description), text(row.source)].join(' ').toLowerCase().includes(needle.toLowerCase())) && <Empty>No matches.</Empty>}
          </nav><section key={selectedSkill} aria-label="Selected capability details" tabIndex={0}>
            {selectedSkill ? <><h2>{selectedSkill}</h2><p className="studio-muted">{text(list.find(row => text(row.name) === selectedSkill)?.source)}</p>
              {tab === 'skills' && <button onClick={() => { window.dispatchEvent(new CustomEvent('xerxes:insert-command', { detail: '/skill ' + selectedSkill })); open(null) }}>Use in conversation <Icon name="arrow" size={12} /></button>}
              {detail && <Markdown text={detail} />}
            </> : <Empty>Select a {tab === 'skills' ? 'skill' : 'tool'} to read its details.</Empty>}
          </section></div>
        </>
      )}
      {detail && tab === 'plugins' && (
        <div className="studio-detail">
          <Markdown text={detail} />
        </div>
      )}
    </div>
  )
}

function SchedulesPanel({ snap }: { snap: Snapshot }): ReactElement {
  const [allRuns,setAllRuns]=useState(false)
  const [runNotice,setRunNotice]=useState('')
  const [deliverySchedule,setDeliverySchedule]=useState<string|null>(null)
  const request = useRequest(snap),
    [jobs, setJobs] = useState<RpcRecord[]>([]),
    [loaded, setLoaded] = useState(false),
    [creating, setCreating] = useState(false)
  const [prompt, setPrompt] = useState(''),
    [schedule, setSchedule] = useState('0 8 * * 1-5'),
    [timezone, setTimezone] = useState(Intl.DateTimeFormat().resolvedOptions().timeZone),
    [preview, setPreview] = useState('')
  const [history, setHistory] = useState<RunHistoryPage | null>(null),
    [runResult, setRunResult] = useState<RpcRecord | null>(null),
    [editing, setEditing] = useState<RpcRecord | null>(null)
  const [pendingRemoval, setPendingRemoval] = useState<string | null>(null)
  const load = async () => {
    const result = await request.call('schedule.list')
    const next = records(result.jobs)
    if (request.alive.current) {
      setJobs(next)
      setLoaded(true)
    }
  }
  useEffect(() => {
    void request.run(load)
  }, [])
  const timing = { schedule, timezone }
  const loadHistory = (job: RpcRecord, append = false) => request.run(async () => {
    const last = append ? history?.runs.at(-1) : undefined
    const result = await request.call('run.list', {
      scope: 'workspace', kind: 'schedule', source_id: job.id,
      ...(last ? { before_started_at: last.startedAt, before_id: last.id } : {}),
    })
    if (request.alive.current) {
      setHistory({ job, runs: [...(append ? history?.runs ?? [] : []), ...records(result.runs)], hasMore: result.has_more === true })
      if (!append) setRunResult(null)
    }
  })
  return (
    <div className="studio-form">
      <Feedback {...request} />
      {runNotice && <p role="status">{runNotice}</p>}
      {deliverySchedule ? <Deliveries key={snap.sessionKey+deliverySchedule} snap={snap} scheduleId={deliverySchedule} close={()=>setDeliverySchedule(null)} /> : allRuns ? <UnifiedRuns key={snap.sessionKey} snap={snap} close={()=>setAllRuns(false)} /> : history ? <RunHistory page={history} result={runResult} busy={request.busy}
        onClose={() => { setHistory(null); setRunResult(null) }}
        onMore={() => void loadHistory(history.job, true)}
        onInspect={run => void request.run(async () => {
          const result = await request.call('run.inspect', { scope: 'workspace', run_id: run.id })
          if (request.alive.current) setRunResult(record(result.run))
        })}
      /> : creating ? (
        <>
          <h1>{editing ? 'Edit scheduled work' : 'Schedule useful work'}</h1>
          <label>
            Task
            <textarea
              rows={4}
              value={prompt}
              onChange={(event) => {
                setPrompt(event.target.value)
                setPreview('')
              }}
              placeholder="Run regression tests and report confirmed failures."
            />
          </label>
          <label>
            Schedule
            <select
              value={schedule}
              onChange={(event) => {
                setSchedule(event.target.value)
                setPreview('')
              }}
            >
              {!['0 8 * * 1-5', '0 2 * * *', '0 * * * *'].includes(schedule) && <option value={schedule}>Custom expression</option>}
              <option value="0 8 * * 1-5">Weekdays at 08:00</option>
              <option value="0 2 * * *">Every night at 02:00</option>
              <option value="0 * * * *">Every hour</option>
            </select>
          </label>
          <label>
            Cron expression
            <input
              value={schedule}
              onChange={(event) => {
                setSchedule(event.target.value)
                setPreview('')
              }}
              placeholder="0 8 * * 1-5"
            />
          </label>
          <label>
            Timezone
            <input
              value={timezone}
              onChange={(event) => {
                setTimezone(event.target.value)
                setPreview('')
              }}
            />
          </label>
          <p>Results are retained in this workspace. No external messages are sent.</p>
          {preview && (
            <p role="status">
              Next run: {scheduleTime(preview, timezone)}
            </p>
          )}
          <div className="studio-actions">
            <button disabled={request.busy} onClick={() => setCreating(false)}>
              Cancel
            </button>
            <button
              disabled={request.busy || !prompt.trim()}
              className="studio-primary"
              onClick={() =>
                void request.run(async () => {
                  if (!preview) {
                    const result = await request.call('schedule.preview', timing)
                    if (request.alive.current) setPreview(text(result.next_run_at))
                    return
                  }
                  await request.call(
                    editing ? 'schedule.update' : 'schedule.create',
                    editing
                      ? {
                          ...timing,
                          prompt,
                          paused: editing.paused,
                          schedule_id: editing.id,
                          revision: editing.revision,
                        }
                      : {
                          ...timing,
                          prompt,
                          paused: false,
                          deliver: 'workspace',
                          target: 'independent',
                        },
                  )
                  if (request.alive.current) {
                    setCreating(false)
                    setPreview('')
                    await load()
                  }
                })
              }
            >
              {preview ? (editing ? 'Save schedule' : 'Create schedule') : 'Preview schedule'}
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="studio-actions">
            <button disabled={request.busy} onClick={() => void request.run(load)}>
              Refresh
            </button>
            <button
              className="studio-primary"
              onClick={() => {
                setEditing(null)
                setPrompt('')
                setPreview('')
                setCreating(true)
              }}
            >
              New schedule
            </button>
            <button onClick={()=>setAllRuns(true)}>All run history</button>
          </div>
          {loaded && !jobs.length && <Empty>No scheduled work in this workspace yet.</Empty>}
          {jobs.map((job) => (
            <div className="studio-item schedule-row" key={text(job.id)}>
              {/* The heading was the whole prompt, the schedule was a raw
                  cron expression, and an empty execution_state left a
                  dangling separator at the end of the line. */}
              <div>
                <strong>{text(job.name) || text(job.prompt)}</strong>
                {text(job.name) && text(job.prompt) ? <p className="schedule-row__prompt" title={text(job.prompt)}>{text(job.prompt)}</p> : null}
                <p>
                  {[cronInEnglish(text(job.schedule)), text(job.timezone), job.paused ? 'Paused' : text(job.execution_state)].filter(Boolean).join(' · ')}
                </p>
                <small title={text(job.schedule)}>
                  {job.paused ? 'Paused — it will not run until you resume it'
                    : text(job.next_run_at) ? `Next: ${scheduleTime(text(job.next_run_at), text(job.timezone))}`
                    : 'Next run not scheduled yet'}
                </small>
              </div>
              <div className="studio-actions">
                <button
                  disabled={request.busy || job.execution_state === 'running'}
                  onClick={() => {
                    setAllRuns(true)
                    setRunNotice('')
                    void request.run(async () => {
                      try {
                        await request.call('schedule.run', { schedule_id: job.id })
                      } catch (error) {
                        if (desktopError(error) !== `job ${text(job.id)} cancelled by operator`) throw error
                        if (request.alive.current) setRunNotice('Scheduled run cancelled.')
                      }
                      await load()
                    })
                  }}
                >
                  Run now
                </button>
                {text(job.schedule) && (
                  <button
                    disabled={request.busy}
                    onClick={() => {
                      setEditing(job)
                      setPrompt(text(job.prompt))
                      setSchedule(text(job.schedule))
                      setTimezone(text(job.timezone))
                      setPreview('')
                      setCreating(true)
                    }}
                  >
                    Edit
                  </button>
                )}
                <button
                  disabled={request.busy}
                  onClick={() =>
                    void request.run(async () => {
                      await request.call(job.paused ? 'schedule.resume' : 'schedule.pause', {
                        schedule_id: job.id,
                      })
                      await load()
                    })
                  }
                >
                  {job.paused ? 'Resume' : 'Pause'}
                </button>
                <button
                  disabled={request.busy}
                  onClick={() => void loadHistory(job)}
                >
                  History
                </button>
                <button disabled={request.busy} onClick={()=>setDeliverySchedule(text(job.id))}>Deliveries</button>
                <button
                  disabled={request.busy}
                  onClick={() => {
                    setPendingRemoval(text(job.id))
                  }}
                >
                  Remove
                </button>
                {pendingRemoval === text(job.id) && <div className="schedule-confirmation" role="group" aria-label="Confirm schedule removal">
                  <p>Remove this schedule? Running work and retained history are kept.</p>
                  <button disabled={request.busy} onClick={() => setPendingRemoval(null)}>Keep schedule</button>
                  <button disabled={request.busy} onClick={() => {
                      void request.run(async () => {
                        await request.call('schedule.remove', {
                          schedule_id: job.id,
                          revision: job.revision,
                        })
                        if (request.alive.current) setPendingRemoval(null)
                        await load()
                      })
                  }}>Confirm removal</button>
                </div>}
              </div>
            </div>
          ))}
        </>
      )}
    </div>
  )
}

const finishedActivityStates = new Set(['completed', 'succeeded', 'failed', 'interrupted', 'cancelled', 'canceled', 'stopped', 'expired', 'archived'])

/** How many finished rows stay visible once nothing is running. */
const RECENT_ACTIVITY = 3

/** Map the daemon's activity states onto the rail's four status colours. */
function activityTone(state: string): string {
  if (['failed', 'error'].includes(state)) return 'failed'
  if (['running', 'working', 'watching', 'cancelling'].includes(state)) return 'working'
  if (['waiting', 'waiting_for_input', 'blocked'].includes(state)) return 'waiting'
  return 'done'
}

export function BackgroundActivity({ rows, renderRow }: {
  rows: readonly RpcRecord[]
  renderRow: (row: RpcRecord) => ReactElement
}): ReactElement {
  // Unknown states remain visible: a new daemon state must not silently hide work.
  const current = rows.filter(row => !finishedActivityStates.has(text(row.state)))
  const finished = rows.filter(row => finishedActivityStates.has(text(row.state)))
  // Once everything has finished, folding ALL of it away left the rail
  // showing a single closed triangle — the accordion stack this redesign
  // set out to remove. Keep the most recent few on screen and fold only
  // the tail, which is what "past" should have meant all along.
  const recent = current.length === 0 ? finished.slice(0, RECENT_ACTIVITY) : []
  const history = finished.slice(recent.length)
  const failures = history.filter(row => row.state === 'failed').length
  return <>
    {current.map(renderRow)}
    {recent.map(renderRow)}
    {history.length > 0 && <details className="activity-history">
      <summary><Icon name="chevron" size={12} />Earlier background activity <span>{history.length}{failures ? ` · ${failures} failed` : ''}</span></summary>
      <div className="activity-history__rows" role="region" aria-label="Earlier background activity" tabIndex={0}>{history.map(renderRow)}</div>
    </details>}
  </>
}

function ActivityPanel({ snap }: { snap: Snapshot }): ReactElement {
  const request = useRequest(snap),
    [rows, setRows] = useState<RpcRecord[]>([]),
    [loaded, setLoaded] = useState(false),
    [terminals, setTerminals] = useState(false)
  useEffect(() => {
    let active = true,
      timer: ReturnType<typeof setTimeout> | undefined
    // The 3s refresh must not reuse request.run: that flips the shared busy
    // flag, so an unreserved "Working…" row entered layout every three
    // seconds and re-disabled the row buttons under the pointer.
    const load = async (silent = false) => {
      if (timer) clearTimeout(timer)
      const fetchRows = async () => {
        const result = await request.call('background.activity')
        const next = records(result.rows)
        if (active) {
          setRows(next)
          setLoaded(true)
        }
      }
      if (silent) await fetchRows().catch(() => {})
      else await request.run(fetchRows)
      if (active) {
        if (timer) clearTimeout(timer)
        timer = setTimeout(() => void load(true), 3000)
      }
    }
    const unsubscribe = window.xerxes.onEvent(({ type, payload }) => {
      if (type === 'background_changed' && (!payload.session_id || payload.session_id === snap.currentId)) void load()
    })
    void load()
    return () => {
      unsubscribe()
      active = false
      if (timer) clearTimeout(timer)
    }
  }, [snap.sessionKey, snap.currentId])
  // Same row grammar as the agents above: dot, name, state. These used to
  // be two-line blocks whose second line was a whole sentence from the
  // daemon, so a finished watcher carried more visual weight than a
  // running agent.
  const renderRow = (row: RpcRecord): ReactElement => (
        row.kind === 'shell' ? <CommandActivity key={text(row.id)} row={row} sessionKey={snap.sessionKey} online={snap.connection === 'online'} /> :
        <div className="railrow railrow--static" data-state={activityTone(text(row.state))} key={text(row.id)} title={[text(row.title), text(row.kind), text(row.detail)].filter(Boolean).join(' · ')}>
          <span className="railrow__dot" aria-hidden="true" />
          <span className="railrow__name">{text(row.title)}</span>
          <span className="railrow__meta">{text(row.state) || text(row.kind)}</span>
          {row.kind === 'watcher' && row.state === 'watching' && (
            <button
              disabled={request.busy}
              onClick={() =>
                void request.run(async () => {
                  await request.call('monitor.stop', { monitor_id: row.id })
                })
              }
            >
              Stop watcher
            </button>
          )}
        </div>
      )

  return (
    <div className="studio-form activity-panel">
      <Feedback {...request} />
      {rows.length > 0 && (
        <section className="railcard" aria-label="Background work">
          <header className="railcard__head">
            <span className="railcard__title">Background</span>
            <span className="railcard__meta">{rows.length}</span>
          </header>
          <BackgroundActivity rows={rows} renderRow={renderRow} />
        </section>
      )}
      {loaded && !rows.length && !snap.fleet.length && (
        <div className="activity-idle" role="status"><Icon name="activity" size={22} /><strong>{snap.turnActive ? "Working on your request" : "No active work"}</strong><p>{snap.turnActive ? "Tools and background jobs appear here as they start." : "Agents and background jobs appear here when they run."}</p></div>
      )}
      {/* Zone 3 — one drawer. These were six sibling sections, each
          collapsed, each competing for attention with the live status
          above. They are the things you reach for roughly twice a month;
          one closed row is the honest amount of space for that. */}
      <details className="railcard rail-drawer">
        <summary>
          <Icon name="chevron" size={13} />
          <span>Terminals, monitors and diagnostics</span>
        </summary>
        <div className="rail-drawer__body">
          <button className="activity-terminal"
            onClick={() => {
              store.loadTerminals()
              setTerminals((value) => !value)
            }}
          >
            <Icon name="terminal" size={16} /> Open terminals
          </button>
          {terminals && <TerminalsCard snap={snap} />}
          <details className="activity-context"><summary>Conversation usage</summary>
          <div className="studio-item">
            <div>
              <strong>Context</strong>
              <p>
                {snap.contextMax
                  ? `${new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(snap.contextTokens ?? 0)} of ${new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(snap.contextMax)} tokens`
                  : 'Usage unavailable'}
              </p>
            </div>
            <button
              disabled={snap.turnActive || !snap.contextTokens}
              onClick={() => {
                void store.submit('/compact')
              }}
            >
              Summarize
            </button>
          </div>
          <button className="context-breakdown-toggle" aria-expanded={snap.contextMenuOpen} onClick={() => store.toggleContextMenu()}>Token breakdown</button>
          {snap.contextMenuOpen && (snap.contextBreakdownLoading ? <p role="status">Estimating token usage…</p> : snap.contextBreakdown ? <dl className="context-breakdown">
            <dt>System prompt</dt><dd>{snap.contextBreakdown.systemPromptTokens.toLocaleString()}</dd>
            <dt>Tools</dt><dd>{snap.contextBreakdown.toolsTokens.toLocaleString()}</dd>
            <dt>Messages</dt><dd>{snap.contextBreakdown.messagesTokens.toLocaleString()}</dd>
          </dl> : <p role="status">Token breakdown unavailable.</p>)}
          </details>
          <ContextInspectorDisclosure snap={snap} />
          <MonitorsDisclosure snap={snap} />
          {snap.skillSuggestions.length > 0 && (
            <details className="activity-context"><summary>Skills this task used · {snap.skillSuggestions.length}</summary>
              <div className="rail__skills">
                {snap.skillSuggestions.slice(-3).reverse().map(suggestion => (
                  <div className="skillcard" key={suggestion.skillName}>
                    <div className="skillcard__head">
                      <span>{suggestion.skillName}</span>
                      {suggestion.version && <span>v{suggestion.version}</span>}
                    </div>
                    {suggestion.description && <div className="skillcard__desc">{suggestion.description}</div>}
                    <div className="skillcard__meta">
                      {suggestion.toolCount} tool call{suggestion.toolCount === 1 ? '' : 's'}
                      {suggestion.uniqueTools.length ? ` · ${suggestion.uniqueTools.join(', ')}` : ''}
                    </div>
                  </div>
                ))}
              </div>
            </details>
          )}
          {snap.creatorTrace.length > 0 && (
            <details className="activity-context"><summary>Template forge · legacy</summary>
              <div className="rail__creator">
                {snap.creatorTrace.slice(-4).reverse().map((trace, index) => (
                  <div className="creatorrow" key={`${trace.at}:${trace.action}:${index}`}>
                    <span className="creatorrow__state" data-state={trace.status}>{trace.status === 'ok' ? <Icon name="check" size={12} /> : <Icon name="warning" size={12} />}</span>
                    <span className="creatorrow__body">
                      <span>{trace.action} · {trace.name || 'forge'}{trace.version ? `@${trace.version}` : ''}</span>
                      {trace.detail && <span>{trace.detail}</span>}
                    </span>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      </details>
    </div>
  )
}

function FilesPanel({ snap, close }: { snap: Snapshot; close: () => void }): ReactElement {
  const [selected, setSelected] = useState('')
  const [preview, setPreview] = useState<RpcRecord | null>(null)
  const [previewError, setPreviewError] = useState('')
  useEffect(() => {
    let current = true
    setPreview(null)
    setPreviewError('')
    if (selected) void desktopCall(window.xerxes, snap.sessionKey, 'workspace.filePreview', { path: selected })
      .then(value => { if (current) setPreview(value) })
      .catch(error => { if (current) setPreviewError(desktopError(error)) })
    return () => { current = false }
  }, [selected, snap.sessionKey, snap.cwd])
  const [needle, setNeedle] = useState('')
  return (
    <div className={'file-browser' + (selected ? ' file-browser--preview' : '')}>
      <div className="file-browser__navigation">
      <div className="file-browser__location" title={snap.cwd}><Icon name="folder" size={16} /><strong>{needle.includes("/") ? needle : snap.cwd.split('/').filter(Boolean).at(-1) || 'Workspace'}</strong></div>
      <label className="file-browser__search"><Icon name="search" size={15} />
        <input aria-label="Filter workspace files by path" value={needle} onChange={(event) => { setSelected(''); setNeedle(event.target.value) }} placeholder="Find a file…" />
      </label>
      <div className="file-browser__entries">
        <div hidden={Boolean(needle)}><WorkspaceFileTree key={snap.cwd + snap.sessionKey} sessionKey={snap.sessionKey} selected={selected} select={setSelected} /></div>
        {needle && <WorkspaceFileTree key={snap.cwd + snap.sessionKey + needle} path={needle.startsWith('./') ? needle : './' + needle} sessionKey={snap.sessionKey} selected={selected} select={setSelected} />}
      </div>
      </div>
      {selected && <div className="file-browser__document"><div className="file-browser__selection"><p title={selected}>{selected}</p><button onClick={() => { window.dispatchEvent(new CustomEvent('xerxes:add-context', { detail: '@' + JSON.stringify(selected.replace(/^@/, '')) })); close() }}>Add to message</button><button className="file-browser__close" aria-label="Close file preview" title="Close preview" onClick={() => setSelected('')}><Icon name="close" size={13} /></button></div>
      <section className="file-browser__preview" aria-label="File preview">
        <Feedback busy={!preview && !previewError} error={previewError} />
        {preview && <>
          {preview.truncated === true && <p className="studio-muted">Preview limited to the first 128 KB.</p>}
          <pre className="file-preview" tabIndex={0} role="region" aria-label="File contents">{text(preview.content).split('\n').map((line, index) => <span className="file-preview__line" key={index}><span aria-hidden="true">{index + 1}</span><code>{line || ' '}</code></span>)}</pre>
        </>}
      </section></div>}
    </div>
  )
}

function WorkspacePanel({ snap }: { snap: Snapshot }): ReactElement {
  const [managed,setManaged]=useState(false)
  const open = useDesktopNavigation(),
    request = useRequest(snap)
  const [machines, setMachines] = useState<RpcRecord[]>([]),
    [hosts, setHosts] = useState<string[]>([]),
    [hostsLoading, setHostsLoading] = useState(false),
    [hostsError, setHostsError] = useState(''),
    [remoteStatus, setRemoteStatus] = useState<RpcRecord | null>(null)
  const [adding, setAdding] = useState(false),
    [alias, setAlias] = useState(''),
    [target, setTarget] = useState(''),
    [path, setPath] = useState(''),
    [folders, setFolders] = useState<string[]>([]),
    [browsing, setBrowsing] = useState(false),
    [connecting, setConnecting] = useState(false)
  const native = async (action: string, params: RpcRecord = {}) => {
    if (!window.xerxes.remote)
      throw new Error('Remote workspaces require the current desktop build.')
    const result = record(await window.xerxes.remote(action, params))
    if (result.ok === false) throw new Error(text(result.error))
    return result
  }
  const load = async () => {
    const result = await native('list'),
      status = await native('status')
    if (request.alive.current) {
      setMachines(records(result.machines))
      setRemoteStatus(status)
    }
  }
  useEffect(() => {
    void request.run(load)
  }, [])
  useEffect(() => {
    if (!adding) return
    let active = true
    setHostsLoading(true)
    setHostsError('')
    void native('hosts').then(result => {
      if (active) setHosts(Array.isArray(result.hosts) ? result.hosts.map(text) : [])
    }).catch(error => {
      if (active) setHostsError(desktopError(error))
    }).finally(() => {
      if (active) setHostsLoading(false)
    })
    return () => { active = false }
  }, [adding])
  const browse = async (folder: string) => {
    const result = await native('browse', { target, path: folder })
    if (request.alive.current) {
      setPath(text(result.path))
      setFolders(Array.isArray(result.directories) ? result.directories.map(text) : [])
      setBrowsing(true)
    }
  }
  const connect = (machine: RpcRecord) =>
    request.run(async () => {
      setConnecting(true)
      try {
        await native('connect', { machine, resume_session_id: snap.currentId })
      } finally {
        if (request.alive.current) setConnecting(false)
      }
    })
  return (
    <div className="studio-form workspace-panel">
      {managed ? <WorkspaceReview key={snap.sessionKey} snap={snap} close={()=>setManaged(false)} /> : <>
      <Feedback {...request} />
      <div className="workspace-current">
        <div>
          <span className="workspace-eyebrow">Current workspace</span><h3>{snap.cwd.split('/').at(-1) || 'Choose a workspace'}</h3>
          <p>{snap.cwd}</p>
          <small className="workspace-status" data-online={snap.connection === "online"}>
            {remoteStatus?.machine ? 'SSH · local rendering' : 'Local workspace'} ·{' '}
            {snap.connection}
          </small>
        </div>
        <button
          disabled={request.busy}
          onClick={() => {
            if (remoteStatus?.machine) void connect(record(remoteStatus.machine))
            else store.retryConnection()
          }}
        >
          Reconnect
        </button>
      </div>
      {remoteStatus?.error ? (
        <p role="alert" className="studio-error">
          {text(remoteStatus.error)}
        </p>
      ) : null}
      <div className="workspace-local-actions">
        <button onClick={()=>setManaged(true)}>Review isolated work</button>
        <button onClick={() => store.chooseWorkspace()}>Switch this window’s folder</button>
        <button disabled={snap.workspaceBusy} onClick={() => void store.openWorkspaceWindow()}>Open workspace in new window…</button>
        <button
          onClick={() => {
            open(null)
            store.toggleWorkspaceMenu()
          }}
        >
          Recent workspaces
        </button>
      </div>
      {remoteStatus?.machine && <RemoteProviders key={snap.sessionKey} remote={window.xerxes.remote} changed={()=>store.loadModels(true)}/>}
      <h3 className="workspace-section-title">Saved connections</h3>
      <p className="studio-muted">
        Connect to a project on another machine over SSH.
      </p>
      {machines.map((machine) => (
        <div className="workspace-remote" key={text(machine.alias)}>
          <div>
            <strong>{text(machine.alias)}</strong>
            <p>
              <span>{text(machine.target)}</span><span>{text(machine.workspacePath)}</span>
            </p>
          </div>
          <button disabled={request.busy} onClick={() => void connect(machine)}>
            Connect
          </button>
          <button
            disabled={request.busy}
            onClick={() => {
              if (window.confirm('Remove this saved workspace? Remote files are kept.'))
                void request.run(async () => {
                  await native('remove', { alias: machine.alias })
                  await load()
                })
            }}
          >
            Remove
          </button>
        </div>
      ))}
      {connecting && (
        <div role="status">
          <p>Preparing remote runtime and opening the tunnel…</p>
          <button onClick={() => void native('cancel').catch(() => {})}>Cancel connection</button>
        </div>
      )}
      {adding ? (
        <>
          <label>
            Name
            <input
              value={alias}
              onChange={(e) => setAlias(e.target.value)}
              placeholder="training-server"
            />
          </label>
          <label>
            SSH host
            <input
              list="ssh-hosts"
              value={target}
              onChange={(e) => {
                setTarget(e.target.value)
                setPath('')
                setFolders([])
                setBrowsing(false)
              }}
              placeholder="Choose an SSH alias or user@host"
            />
            <datalist id="ssh-hosts">
              {hosts.map((host) => (
                <option key={host} value={host} />
              ))}
            </datalist>
          </label>
          {hostsLoading && <p className="studio-muted" role="status">Loading SSH aliases. You can also enter a host directly.</p>}
          {hostsError && <p className="studio-error" role="alert">{hostsError} You can enter an SSH alias or user@host directly.</p>}
          <label>
            Project folder
            <input
              value={path}
              onChange={(e) => {
                setPath(e.target.value)
                setBrowsing(false)
              }}
              placeholder="Choose a folder on the host"
            />
          </label>
          <button
            disabled={request.busy || !target}
            onClick={() => void request.run(() => browse(path))}
          >
            Browse remote folders
          </button>
          {browsing && (
            <div className="studio-folder-list">
              <button
                disabled={request.busy}
                onClick={() =>
                  void request.run(() => browse(path.split('/').slice(0, -1).join('/') || '/'))
                }
              >
                <Icon name="arrowUp" size={12} /> Parent folder
              </button>
              <strong>{path}</strong>
              {folders.map((folder) => (
                <button
                  disabled={request.busy}
                  key={folder}
                  onClick={() =>
                    void request.run(() => browse(path.replace(/\/$/, '') + '/' + folder))
                  }
                >
                  <Icon name="chevron" size={12} /> {folder}
                </button>
              ))}
              <button onClick={() => setBrowsing(false)}>Use this folder</button>
            </div>
          )}
          <div className="studio-actions">
            <button disabled={request.busy} onClick={() => setAdding(false)}>
              Cancel
            </button>
            <button
              className="studio-primary"
              disabled={request.busy || !alias || !target || !path}
              onClick={() =>
                void request.run(async () => {
                  await native('save', { machine: { alias, target, workspacePath: path } })
                  if (request.alive.current) {
                    setAdding(false)
                    await load()
                  }
                })
              }
            >
              Save workspace
            </button>
          </div>
        </>
      ) : (
        <button
          disabled={request.busy}
          onClick={() => {
            setAdding(true)
          }}
        >
          Add remote workspace
        </button>
      )}
    </>}
    </div>
  )
}

function pluginArgument(value: string): string {
  if (!value.trim() || /['\r\n\0]/.test(value))
    throw new Error('Use a plugin path without quotes or control characters.')
  return "'" + value + "'"
}

export function BackgroundIndicator({ snap }: { snap: Snapshot }): ReactElement {
  const open = useDesktopNavigation()
  const [counts, setCounts] = useState<{ shells: number; watchers: number } | null>(null)
  const [compacting, setCompacting] = useState('')
  useEffect(() => {
    let active = true,
      pending = false
    const load = async () => {
      if (pending) return
      pending = true
      try {
        const result = await desktopCall(window.xerxes, snap.sessionKey, 'background.status')
        if (typeof result.shells !== 'number' || typeof result.watchers !== 'number')
          throw new Error('Invalid activity counts')
        if (active) setCounts({ shells: result.shells, watchers: result.watchers })
      } catch {
        if (active) setCounts(null)
      } finally {
        pending = false
      }
    }
    const unsubscribe = window.xerxes.onEvent(({ type, payload }) => {
      if (payload.session_id && payload.session_id !== snap.currentId) return
      if (type === 'background_changed') void load()
      if (type === 'status_update' && payload.kind === 'compressing')
        setCompacting(text(payload.text) || 'Compacting conversation…')
      if ((type === 'status_update' && payload.kind === 'compaction') || type === 'turn_end')
        setCompacting('')
    })
    void load()
    const timer = setInterval(() => void load(), 5000)
    return () => {
      active = false
      clearInterval(timer)
      unsubscribe()
    }
  }, [snap.sessionKey, snap.currentId])
  useEffect(() => {
    if (snap.connection !== 'online') setCompacting('')
  }, [snap.connection])
  return (
    <button title="Background activity" aria-label="Activity" className="studio-background" onClick={() => open('activity')} aria-live="polite">
      <Icon name="activity" /><span>
      {compacting ? (
        <>
          <span className="studio-spinner" />
          {compacting}
        </>
      ) : counts ? (
        [
          counts.shells
            ? counts.shells + (counts.shells === 1 ? ' shell running' : ' shells running')
            : '',
          counts.watchers
            ? counts.watchers + (counts.watchers === 1 ? ' watcher' : ' watchers')
            : '',
        ]
          .filter(Boolean)
          .join(' · ') || 'Activity'
      ) : (
        'Activity'
      )}</span>
    </button>
  )
}

function SnapshotsPanel({ snap }: { snap: Snapshot }): ReactElement {
  const request = useRequest(snap),
    [rows, setRows] = useState<RpcRecord[]>([]),
    [preview, setPreview] = useState<RpcRecord | null>(null),
    [selected, setSelected] = useState(''),
    [path, setPath] = useState(''),
    [reviewedPath, setReviewedPath] = useState(''),
    [loaded, setLoaded] = useState(false)
  const load = async () => {
    const result = await request.call('snapshot.list')
    if (request.alive.current) {
      setRows(records(result.snapshots))
      setLoaded(true)
    }
  }
  useEffect(() => {
    void request.run(load)
  }, [])
  return (
    <div className="studio-form">
      <Feedback {...request} />
      <div className="studio-actions">
        <button disabled={request.busy} onClick={() => void request.run(load)}>
          Refresh
        </button>
        <button
          disabled={request.busy || snap.turnActive}
          onClick={() =>
            void request.run(async () => {
              await request.call('slash', { command: '/snapshot' })
              await load()
            })
          }
        >
          Capture snapshot
        </button>
      </div>
      {loaded && !rows.length && <Empty>No saved file states yet.</Empty>}
      {rows.map((row) => (
        <div className="studio-item" key={text(row.id)}>
          <div>
            <strong>{text(row.label) || text(row.id)}</strong>
            <p>{text(row.created_at)}</p>
          </div>
          <button
            disabled={request.busy}
            onClick={() =>
              void request.run(async () => {
                const result = await request.call('snapshot.preview', { snapshot_id: row.id })
                if (request.alive.current) {
                  setSelected(text(row.id))
                  setPath('')
                  setPreview(result)
                }
              })
            }
          >
            Preview
          </button>
        </div>
      ))}
      {preview && (
        <>
          <label>
            File to restore
            <select
              disabled={request.busy}
              value={path}
              onChange={(e) => {
                const path = e.target.value
                setPath(path)
                setReviewedPath('')
                if (path)
                  void request.run(async () => {
                    const result = await request.call('snapshot.preview', {
                      snapshot_id: selected,
                      path,
                    })
                    if (request.alive.current) {
                      setPreview(result)
                      setReviewedPath(path)
                    }
                  })
              }}
            >
              <option value="">Select a file</option>
              {(Array.isArray(preview.files) ? preview.files : []).map((file) => (
                <option key={String(file)} value={String(file)}>
                  {String(file)}
                </option>
              ))}
            </select>
          </label>
          <DiffPreview diff={text(preview.diff)} label="Snapshot restore diff" />
          {preview.truncated === true && <p>Large preview truncated.</p>}
          <button
            disabled={request.busy || !path || reviewedPath !== path || snap.turnActive}
            onClick={() => {
              if (
                window.confirm(
                  'Restore ' + path + ' from this snapshot? The runtime saves a backup first.',
                )
              )
                void request.run(async () => {
                  await request.call('snapshot.restoreFile', {
                    snapshot_id: selected,
                    path,
                    revision: preview.revision,
                  })
                  if (request.alive.current) {
                    setPreview(null)
                    setPath('')
                    await load()
                  }
                })
            }}
          >
            Restore reviewed file
          </button>
        </>
      )}
    </div>
  )
}
