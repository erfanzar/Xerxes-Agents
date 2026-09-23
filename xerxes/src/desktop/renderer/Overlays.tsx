// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Overlay surfaces: the settings modal (General / Models & Providers /
 * Permissions, wired to what the daemon actually supports), the ⌘K command
 * palette, and the model picker popover anchored to the composer chip.
 *
 * Every control here either flips real daemon state or is honestly marked
 * read-only — no dead switches. What the daemon does not expose on the wire
 * (per-file undo, per-model pricing) is not faked.
 */

import { createPortal } from 'react-dom'
import { createContext, useContext, useId, type ReactNode, useLayoutEffect, useEffect, useMemo, useRef, useState, type ReactElement, type RefObject } from 'react'

import { saveAppearance, FONT_SIZES, FONT_LABELS } from './appearance.js'
import { desktopError } from './desktopRpc.js'
import { useDialogFocus } from './dialogFocus.js'
import { ChannelsCard } from './ChannelsPanel.js'
import { LspCard } from "./LspPanel.js"
import { TerminalsCard } from './TerminalsPanel.js'
import { store, type Snapshot } from './store.js'
import type { CachedModel, ModelChoice, PermissionMode, ProviderRow, SettingsTab } from './types.js'
import { RemoteProviders } from './RemoteProviders.js'
import { Icon, type IconName } from './Icon.js'

// ── Settings modal ──────────────────────────────────────────────────────

const SETTINGS_TABS: ReadonlyArray<{ id: SettingsTab; label: string }> = [
  { id: 'general', label: 'General' },
  { id: 'models', label: 'Models & Providers' },
  { id: 'agents', label: 'Agent presets' },
  { id: 'permissions', label: 'Permissions' },
  { id: 'channels', label: 'Channels' },
  { id: 'mcp', label: 'MCP Servers' },
  { id: 'terminals', label: 'Terminals' },
  { id: 'lsp', label: 'Language servers' },
]

export function SettingsModal({ snap }: { snap: Snapshot }): ReactElement | null {
  const ref=useRef<HTMLDivElement>(null)
  useDialogFocus(ref, snap.settingsOpen)
  if (!snap.settingsOpen) return null
  return (
    <div className="backdrop">
      <div className="modal" ref={ref} role="dialog" aria-modal="true" aria-label="Settings">
        <div className="modal__side">
          <div className="cap">Settings <button className="chipbtn" aria-label="Close settings" onClick={()=>store.closeSettings()}><Icon name="close" size={13} /></button></div>
          {SETTINGS_TABS.map(tab => (
            <button
              key={tab.id}
              className={`mtab${snap.settingsTab === tab.id ? ' is-on' : ''}`}
              onClick={() => store.setSettingsTab(tab.id)}
            >
              {tab.label}
            </button>
          ))}
          <div className="modal__sidefoot">
            <span className="appr-policy">
              {snap.connection === 'online' ? 'daemon connected' : 'offline — reconnect to edit'}
            </span>
          </div>
        </div>
        <div className="modal__main">
          {snap.settingsTab === 'general' && <GeneralCard snap={snap} />}
          {snap.settingsTab === 'models' && <ModelsCard snap={snap} />}
          {snap.settingsTab === 'agents' && <AgentPresetsCard snap={snap} />}
          {snap.settingsTab === 'permissions' && <PermissionsCard snap={snap} />}
          {snap.settingsTab === 'channels' && <ChannelsCard snap={snap} />}
          {snap.settingsTab === 'mcp' && <McpCard snap={snap} />}
          {snap.settingsTab === 'terminals' && <TerminalsCard snap={snap} />}
          {snap.settingsTab === 'lsp' && <LspCard snap={snap} />}
        </div>
      </div>
    </div>
  )
}

function useThemeChoice(): [string, (next: string) => void] {
  // data-user-theme records an explicit pick; without it the main.tsx media
  // listener owns the theme. 'System' must RESOLVE to a concrete theme —
  // removing the attribute would silently fall back to the dark token set.
  const readChoice = (): string =>
    typeof document === 'undefined'
      ? 'system'
      : document.documentElement.getAttribute('data-user-theme') ?? 'system'
  const [choice, setChoice] = useState(readChoice)
  const set = (next: string): void => {
    if (typeof document === 'undefined') return
    if (next === 'system') {
      document.documentElement.removeAttribute('data-user-theme')
      document.documentElement.setAttribute(
        'data-theme',
        window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark',
      )
    } else {
      document.documentElement.setAttribute('data-user-theme', next)
      document.documentElement.setAttribute('data-theme', next)
    }
    saveAppearance()
    setChoice(next)
  }
  return [choice, set]
}

const NOTIFICATIONS_KEY = 'xerxes.notifications'

/** GeneralCard's native switches: OS pings + launch-at-login (mockup 10). */
function useNativeSwitches(): { notifications: boolean; loginItem: boolean | null; toggleNotifications: () => void; toggleLoginItem: () => void } {
  const [notifications, setNotifications] = useState(() => {
    try {
      return (typeof localStorage === 'undefined' ? null : localStorage.getItem(NOTIFICATIONS_KEY)) !== '0'
    } catch {
      return true
    }
  })
  const [loginItem, setLoginItem] = useState<boolean | null>(null)
  // Push the persisted preference once the shell is live; effects do not run
  // in the SSR/test render, so this stays headless-safe.
  useEffect(() => {
    void window.xerxes.setNotifications?.(notifications)
    void window.xerxes.getLoginItem?.().then(state => setLoginItem(state)).catch(() => setLoginItem(null))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])
  const toggleNotifications = (): void => {
    const next = !notifications
    setNotifications(next)
    try {
      localStorage?.setItem(NOTIFICATIONS_KEY, next ? '1' : '0')
    } catch {
      // A blocked storage area leaves the choice session-scoped.
    }
    void window.xerxes.setNotifications?.(next)
  }
  const toggleLoginItem = (): void => {
    void window.xerxes.setLoginItem?.(!loginItem).then(state => setLoginItem(state)).catch(() => setLoginItem(null))
  }
  return { notifications, loginItem, toggleNotifications, toggleLoginItem }
}

/** MCP server statuses (mockup 19) — the daemon is the source of truth. */
function McpCard({ snap }: { snap: Snapshot }): ReactElement {
  // Statuses are point-in-time daemon state, not pushed events.
  const [busy, setBusy] = useState<'status' | 'reload' | null>('status')
  const [error, setError] = useState('')
  const refresh = async (reload = false): Promise<void> => {
    setBusy(reload ? 'reload' : 'status')
    setError('')
    try {
      if (reload) await store.reloadMcp()
      else await store.refreshMcpStatus()
    } catch (error) { setError(desktopError(error)) }
    finally { setBusy(null) }
  }
  useEffect(() => { void refresh() }, [])
  const entries = Object.entries(snap.mcpStatus)
  return (
    <>
      <h2 className="modal__title">MCP Servers</h2>
      {/* The error used to lead, so a first-time visitor read a failure
          before ever learning what the panel is for. */}
      <p className="modal__sub">External tools the agents may call. Tool calls still pass the permission policy — enabling a server is not an auto-approve.</p>
      {error && <p className="studio-error" role="alert">{error}</p>}
      {busy && <p role="status">{busy === 'reload' ? 'Reconnecting servers…' : 'Checking server status…'}</p>}
      <div className="rowlist">
      {entries.length === 0 ? (!busy && !error && (
        <div className="row">
          <div className="row__main">
            <div className="row__t">No MCP servers configured</div>
            <div className="row__s">Add servers to <code>~/.xerxes/mcp.json</code>; the runtime connects them at startup.</div>
          </div>
        </div>
      )) : entries.map(([name, status]) => (
        <div className="row" key={name}>
          <span className={`dot ${status.connected ? 'dot--done' : 'dot--fail'}`} />
          <div className="row__main">
            <div className="row__t">{name}</div>
            <div className="row__s">
              {status.connected
                ? `connected · ${status.tools} tools · ${status.resources} resources · ${status.prompts} prompts`
                : `not connected${status.lastError ? ` — ${status.lastError}` : ''}`}
            </div>
          </div>
        </div>
      ))}
      </div>
      <div style={{ display: 'flex', gap: 8, paddingTop: 16 }}>
        <button className="btn btn--ghost" disabled={busy !== null || snap.connection !== 'online'} onClick={() => { void refresh() }}>Refresh status</button>
        <button className="btn btn--ghost" disabled={busy !== null || snap.connection !== 'online'} onClick={() => { void refresh(true) }}>Reload servers</button>
      </div>
      <p className="modal__sub" style={{ marginTop: 14 }}>
        Config: <code>~/.xerxes/mcp.json</code> · a reload reconnects every configured server.
      </p>
    </>
  )
}

function GeneralCard({ snap }: { snap: Snapshot }): ReactElement {
  const [theme, setTheme] = useThemeChoice()
  const [fontSize, setFontSize] = useState(() =>
    typeof document === 'undefined' ? '12' : document.documentElement.getAttribute('data-font') ?? '12')
  const { notifications, loginItem, toggleNotifications, toggleLoginItem } = useNativeSwitches()
  const applyFont = (size: string): void => {
    if (typeof document !== 'undefined') document.documentElement.setAttribute('data-font', size)
    saveAppearance()
    setFontSize(size)
  }
  return (
    <>
      <h2 className="modal__title">General</h2>
      <p className="modal__sub">Applies immediately; nothing here is per-task.</p>

      <div className="field">
        <label>Theme</label>
        <div className="seg">
          <button className={theme === 'system' ? 'is-on' : ''} onClick={() => setTheme('system')}>System</button>
          <button className={theme === 'dark' ? 'is-on' : ''} onClick={() => setTheme('dark')}>Dark</button>
          <button className={theme === 'light' ? 'is-on' : ''} onClick={() => setTheme('light')}>Light</button>
        </div>
      </div>
      <div className="field">
        <label id="ui-font-size">Interface text size</label>
        {/* Raw pixel numbers told you nothing about which way was bigger. */}
        <div className="seg" role="group" aria-labelledby="ui-font-size">
          {FONT_SIZES.map(size => (
            <button key={size} aria-pressed={fontSize === size} title={`${size}px`} className={fontSize === size ? 'is-on' : ''} onClick={() => applyFont(size)}>{FONT_LABELS[size]}</button>
          ))}
        </div>
      </div>

      <div className="row">
        <div className="row__main">
          <div className="row__t">Setup checklist</div>
          <div className="row__s">Provider, folder and runtime readiness in one place</div>
        </div>
        <button className="btn" onClick={() => { store.closeSettings(); window.dispatchEvent(new Event("xerxes:setup")) }}>Open…</button>
      </div>
      <div className="row">
        <div className="row__main">
          <div className="row__t">Session</div>
          <div className="row__s">
            {snap.currentId ? `${snap.currentId} · ${snap.model || 'model unset'}` : 'No session yet'}
          </div>
        </div>
      </div>
      <div className="row">
        <div className="row__main">
          <div className="row__t">Daemon</div>
          <div className="row__s">
            {snap.connection === 'online' ? 'Connected · shared across workspaces' : 'Offline — retrying with backoff'}
          </div>
        </div>
        <button className="btn" onClick={() => store.retryConnection()}>Reconnect</button>
      </div>
      <div className="row">
        <div className="row__main">
          <div className="row__t">Creator mode</div>
          <div className="row__s">Create and manage presets for future sessions</div>
        </div>
        <button className="btn" onClick={() => store.setSettingsTab('agents')}>Manage…</button>
      </div>
      <div className="row">
        <div className="row__main">
          <div className="row__t">Launch at login</div>
          <div className="row__s">Open Xerxes when you sign in</div>
        </div>
        <button
          className={`switch${loginItem ? ' is-on' : ''}`}
          role="switch"
          aria-checked={loginItem === true}
          aria-label="Launch at login"
          disabled={loginItem === null}
          onClick={toggleLoginItem}
        />
      </div>
      <div className="row">
        <div className="row__main">
          <div className="row__t">Notifications</div>
          <div className="row__s">When a task needs you or finishes, while Xerxes is in the background</div>
        </div>
        <button
          className={`switch${notifications ? ' is-on' : ''}`}
          role="switch"
          aria-checked={notifications}
          aria-label="Notifications"
          onClick={toggleNotifications}
        />
      </div>
      <div className="row">
        <div className="row__main">
          <div className="row__t">Stream thinking</div>
          <div className="row__s">Show reasoning while the agent works. Display only; the runtime always records it</div>
        </div>
        <button
          className={`switch${snap.streamThinking ? ' is-on' : ''}`}
          role="switch"
          aria-checked={snap.streamThinking}
          aria-label="Stream thinking"
          onClick={() => store.setStreamThinking(!snap.streamThinking)}
        />
      </div>
      {/* Was write-only: it hardcoded setPlanMode(true) and never read
          snap.planMode, so it always said "enable now" — including while
          plan mode was already on, where clicking it did nothing visible. */}
      <div className="row">
        <div className="row__main">
          <div className="row__t">Plan this task</div>
          <div className="row__s">{snap.planMode ? 'On — the agent proposes a checklist before it changes anything' : 'Review a plan before making changes'}</div>
        </div>
        <button
          className={`switch${snap.planMode ? ' is-on' : ''}`}
          role="switch"
          aria-checked={snap.planMode}
          aria-label="Plan this task"
          onClick={() => store.setPlanMode(!snap.planMode)}
        />
      </div>
    </>
  )
}

function AgentPresetsCard({ snap }: { snap: Snapshot }): ReactElement {
  const [readError, setReadError] = useState('')
  const [copyFrom, setCopyFrom] = useState<string | null>(null)
  const [copyId, setCopyId] = useState('')
  const [copyName, setCopyName] = useState('')
  const [viewing, setViewing] = useState<{ id: string; content: string } | null>(null)
  const [editing, setEditing] = useState<{ id: string; content: string; dirty: boolean } | null>(null)
  const [busy, setBusy] = useState(false)
  const presets = snap.agentPresets ?? []
  const unavailable = busy || snap.connection !== 'online'
  const clearError = (): void => { setReadError(''); store.clearAgentPresetError() }
  const runAction = async (action: () => Promise<unknown>): Promise<void> => {
    if (unavailable) return
    clearError()
    setBusy(true)
    try { await action() }
    catch (error) { setReadError(desktopError(error)) }
    finally { setBusy(false) }
  }
  const detail = useRef<HTMLDivElement>(null)
  const detailOpen = Boolean(copyFrom || viewing || editing)
  useEffect(() => {
    if (detailOpen) {
      detail.current?.focus()
      detail.current?.scrollIntoView({ block: 'start' })
    }
  }, [detailOpen])
  useEffect(() => { void store.loadAgentPresets() }, [])
  const duplicate = async (): Promise<void> => {
    if (!copyFrom || !copyId.trim()) return
    setBusy(true)
    const created = await store.copyAgentPreset(copyFrom, copyId.trim(), copyName)
    setBusy(false)
    if (created) { setCopyFrom(null); setCopyId(''); setCopyName('') }
  }
  const inspect = (id: string): void => {
    clearError()
    setBusy(true)
    void store.readAgentPreset(id)
      .then(content => setViewing({ id, content }))
      .catch(error => setReadError(desktopError(error)))
      .finally(() => setBusy(false))
  }
  const edit = (id: string): void => {
    clearError()
    setBusy(true)
    void store.readAgentPreset(id)
      .then(content => setEditing({ id, content, dirty: false }))
      .catch(error => setReadError(desktopError(error)))
      .finally(() => setBusy(false))
  }
  const saveEdit = async (): Promise<void> => {
    if (!editing) return
    setBusy(true)
    const saved = await store.writeAgentPreset(editing.id, editing.content)
    setBusy(false)
    if (saved) setEditing(null)
  }
  // A preset binds to the CURRENT session (agentPreset.select); there is
  // nothing to rebind before the first turn creates one, and the swap is
  // refused mid-turn like every other live-session mutation.
  const canSelect = snap.currentId !== '' && !snap.turnActive && snap.connection === 'online'
  return (
    <>
      <h2 className="modal__title">Agent presets</h2>
      {(readError || snap.agentPresetsError) && <p className="studio-error" role="alert">{readError || snap.agentPresetsError}</p>}
      <p className="modal__sub">
        A preset is the tools, system prompt, and subagents one session runs. Duplicate a known-good preset and edit its files, or let Creator mode draft one. Running sessions keep the preset they started with.
      </p>
      {!detailOpen && <><button className="pcard pcard--add" disabled={unavailable || !presets.some(row => row.id === 'creator' && !row.broken)} onClick={() => { void store.draftAgentPreset() }}>
        <span className="pcard__main"><span className="pcard__text"><span className="pcard__name"><Icon name="plus" size={12} /> Draft a custom preset with Creator mode</span><span className="pcard__meta">Starts a fresh Creator session that drafts it with you</span></span></span>
      </button>
      {(['system', 'user', 'project'] as const).map(trust => {
        const rows = presets.filter(row => row.trust === trust)
        if (!rows.length) return null
        return (
          <div key={trust} className="field">
            <label>{trust === 'system' ? 'Built-in' : trust === 'user' ? 'Custom' : 'Project'}</label>
            <div className="pcardlist">
              {rows.map(row => (
                <div key={row.id} className={`pcard${row.isDefault ? ' is-active' : ''}${row.broken ? ' is-broken' : ''}`}>
                  <div className="pcard__main">
                    <span className={`dot ${row.broken ? 'dot--fail' : row.isDefault ? 'dot--live' : 'dot--idle'}`} />
                    <span className="pcard__text">
                      <span className="pcard__name">{row.name}{row.id.toLowerCase() !== row.name.trim().toLowerCase() && <> <code>{row.id}</code></>}{row.isDefault ? <span className="chipbtn" style={{ marginLeft: 6 }}>default</span> : null}{snap.currentAgentPreset === row.id ? <span className="chipbtn" style={{ marginLeft: 6 }}>this session</span> : null}</span>
                      <span className="pcard__meta">{row.broken || row.description || 'No description.'}</span>
                    </span>
                  </div>
                  <div className="pcard__actions">
                    {!row.broken && !row.isDefault && <button className="chipbtn" disabled={unavailable} onClick={() => { void runAction(() => store.setDefaultAgentPreset(row.id)) }}>Set default</button>}
                    {!row.broken && !snap.turnActive && row.id !== snap.currentAgentPreset && (
                      <button
                        className="chipbtn"
                        disabled={unavailable || !canSelect}
                        title={canSelect ? `run the current session with ${row.id}` : 'start a task first — a preset binds to an existing session'}
                        onClick={() => { void runAction(() => store.selectAgentPreset(row.id)) }}
                      >
                        Use here
                      </button>
                    )}
                    <button className="chipbtn" disabled={unavailable || Boolean(row.broken)} onClick={() => inspect(row.id)}>View</button>
                    {!row.broken && row.manageable && <button className="chipbtn" disabled={unavailable} onClick={() => edit(row.id)}>Edit</button>}
                    {!row.broken && <button className="chipbtn" disabled={unavailable} onClick={() => { clearError(); setCopyFrom(row.id); setCopyId(''); setCopyName('') }}>Duplicate</button>}
                    {row.manageable && <button className="chipbtn" disabled={unavailable} onClick={() => { void runAction(() => store.openAgentPresetLocation(row.id)) }}>Open folder</button>}
                    {row.manageable && <button className="chipbtn chipbtn--danger" disabled={unavailable} onClick={() => { if (window.confirm(`Delete agent preset ${row.id}? Running sessions are unaffected.`)) void runAction(() => store.removeAgentPreset(row.id)) }}>Delete</button>}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )
      })}</>}
      {copyFrom && (
        <div className="provform" ref={detail} tabIndex={-1} aria-label="Duplicate preset">
          <div className="row__t">Duplicate {copyFrom}</div>
          <div className="field"><label htmlFor="preset-copy-id">Identifier</label><input id="preset-copy-id" disabled={unavailable} value={copyId} spellCheck={false} placeholder="my-agent" onChange={event => setCopyId(event.target.value)} /></div>
          <div className="field"><label htmlFor="preset-copy-name">Display name</label><input id="preset-copy-name" disabled={unavailable} value={copyName} placeholder="Optional" onChange={event => setCopyName(event.target.value)} /></div>
          <div className="preset-actions"><button className="btn btn--ghost" disabled={busy} onClick={() => { clearError(); setCopyFrom(null) }}>Cancel</button><button className="btn" disabled={unavailable || !/^[a-z0-9][a-z0-9-]*$/.test(copyId)} onClick={() => { void duplicate() }}>{busy ? 'Creating…' : 'Create'}</button></div>
        </div>
      )}
      {viewing && (
        <div className="provform" ref={detail} tabIndex={-1} aria-label="Preset composition">
          <button className="btn btn--ghost" onClick={() => setViewing(null)}>Back to presets</button>
          <div className="row__t">Composition · {viewing.id}</div>
          <pre className="preset-composition">{viewing.content}</pre>
        </div>
      )}
      {editing && (
        <div className="provform" ref={detail} tabIndex={-1} aria-label="Edit preset">
          <div className="row__t">Edit · {editing.id} — the daemon validates the spec on save</div>
          <textarea
            className="preset-editor"
            aria-label="Preset composition"
            disabled={unavailable}
            value={editing.content}
            rows={18}
            spellCheck={false}
            onChange={event => setEditing({ ...editing, content: event.target.value, dirty: true })}
          />
          <div className="preset-actions">
            <button className="btn btn--ghost" disabled={busy} onClick={() => { clearError(); setEditing(null) }}>Cancel</button>
            <button className="btn" disabled={unavailable || !editing.dirty} onClick={() => { void saveEdit() }}>{busy ? 'Saving…' : 'Save'}</button>
          </div>
        </div>
      )}
    </>
  )
}

function ModelsCard({ snap }: { snap: Snapshot }): ReactElement {
  const groups = useMemo(() => groupModels(snap.models), [snap.models])
  /** false = closed, true = new profile, a name = editing that profile. */
  const [form, setForm] = useState<boolean | string>(false)
  /** Name of the profile whose Delete is awaiting confirmation. */
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
  const editing = typeof form === 'string'
    ? snap.providers.find(p => p.name === form) ?? null
    : null
  return (
    <>
      <h2 className="modal__title">Models & Providers</h2>
      {snap.storageScope?.startsWith('ssh:') && <RemoteProviders key={snap.sessionKey} remote={window.xerxes.remote} changed={()=>store.loadModels(true)}/>}
      <p className="modal__sub">
        Change the session model from the composer. Select a provider to make it active. Profiles are stored in <code>~/.xerxes/profiles.json</code> on the workspace host.
      </p>

      <div className="field">
        <label>Current model</label>
        <button
          className="chipbtn settings-model"
          title="Change model"
          disabled={snap.turnActive}
          onClick={() => store.openPicker()}
        >
          <Icon name="spark" size={12} /> {snap.model || 'unset'} <Icon name="caretDown" size={11} />
        </button>
        {snap.turnActive && (
          <div className="row__s" style={{ marginTop: 4 }}>Locked while a turn runs. The model can't change mid-turn.</div>
        )}
      </div>

      <div className="row__t">Providers</div>
      {snap.providerSwitching && <p role="status">Switching to {snap.providerSwitching}…</p>}
      {snap.providerSwitchError && <p role="alert" className="studio-error">{snap.providerSwitchError}</p>}
      {snap.providerError && <div role="alert" className="studio-error">{snap.providerError}<button onClick={() => void store.loadProviders()}>Retry loading providers</button></div>}
      <div className="pcardlist">
      {/* "No saved provider profiles" is a claim, and it used to be made
          before provider_list had answered — so a configured user was told
          they had nothing, and Retry cleared the error first and showed the
          same false empty state again for the whole retry. */}
      {snap.providers.length === 0 && !snap.providerError && (
        <div className="row">
          <span className={snap.providersLoading ? 'dot dot--live' : 'dot dot--idle'} />
          <div className="row__main">
            <div className="row__t">{snap.providersLoading ? 'Loading provider profiles…' : 'No saved provider profiles'}</div>
            <div className="row__s">{snap.providersLoading ? 'Reading ~/.xerxes/profiles.json on the workspace host.' : 'Add a provider to configure a model.'}</div>
          </div>
        </div>
      )}
      {snap.providers.map(provider => (
        <div
          key={provider.name}
          className={`pcard${provider.active ? ' is-active' : ''}`}
        >
          <button
            className="pcard__main"
            disabled={provider.active || snap.turnActive || !!snap.providerSwitching}
            title={
              provider.active
                ? 'active profile — new tasks start here'
                : snap.turnActive
                  ? 'wait for the running turn to finish'
                  : `make ${provider.name} the active profile`
            }
            onClick={() => store.selectProvider(provider.name)}
          >
            <span className={`dot ${provider.active ? 'dot--live' : 'dot--idle'}`} />
            <span className="pcard__text">
              <span className="pcard__name">
                {provider.name}
                {provider.active ? <span className="chipbtn" style={{ marginLeft: 6 }}>active</span> : null}
              </span>
              <span className="pcard__sub">
                {provider.provider}{provider.model ? ` · ${provider.model}` : ''} · {provider.active ? 'in use' : 'saved'}
              </span>
            </span>
            {!provider.active && !snap.turnActive
              ? <span className="row__go">switch <Icon name="chevron" size={11} /></span>
              : null}
          </button>
          <span className="pcard__actions">
            <button
              className="chipbtn"
              disabled={snap.turnActive}
              title={`edit ${provider.name}`}
              onClick={() => setForm(provider.name)}
            >Edit</button>
            {/* This app confirms killing a terminal; deleting a stored
                credential profile was a single unguarded click. */}
            {!provider.active && (
              confirmDelete === provider.name ? (
                <>
                  <button className="pcard__del" onClick={() => { setConfirmDelete(null); store.deleteProvider(provider.name) }}>Delete for good</button>
                  <button onClick={() => setConfirmDelete(null)}>Keep</button>
                </>
              ) : (
                <button
                  className="pcard__del"
                  disabled={snap.turnActive}
                  title={`delete ${provider.name}`}
                  onClick={() => setConfirmDelete(provider.name)}
                >Delete</button>
              )
            )}
          </span>
        </div>
      ))}
      </div>

      {form !== false ? (
        <ProviderForm
          snap={snap}
          editing={editing}
          onCancel={() => setForm(false)}
        />
      ) : (
        <button className="btn" disabled={snap.turnActive} onClick={() => setForm(true)}>
          <Icon name="plus" size={12} /> Add provider
        </button>
      )}

      <div className="row__t settings-section">Discovered models{snap.models.length > 0 && <span className="settings-section__count">{snap.models.length}</span>}</div>
      <div className="rowlist">
      {groups.map(group => (
        <div key={group.provider} className="row">
          <span className="dot dot--done" />
          <div className="row__main">
            <div className="row__t">{group.provider} <span className="chipbtn" style={{ marginLeft: 6 }}>{group.choices.length} model{group.choices.length === 1 ? '' : 's'}</span></div>
            <div className="row__s">{group.choices.slice(0, 4).map(choice => choice.id).join(', ')}{group.choices.length > 4 ? ', …' : ''}</div>
          </div>
        </div>
      ))}
      {snap.models.length === 0 && (
        <div className="row">
          <div className="row__main">
            <div className="row__t">No models discovered yet</div>
            <div className="row__s">Fetching asks the active provider which models it serves.</div>
          </div>
        </div>
      )}
      </div>

      <div className="settings-actions">
        <button className="btn" onClick={() => store.loadModels(true)}><Icon name="retry" size={13} /> Fetch models</button>
      </div>
    </>
  )
}

/**
 * provider_save upsert by name — one sheet adds a new profile or edits an
 * existing one (Codex-style edit card). The Provider dropdown is the
 * daemon's real adapter registry (`provider_types`); a known type makes
 * Base URL optional ("Provider default") and names the env var a blank key
 * falls back to. Saving ACTIVATES the profile; the copy says so instead of
 * pretending it is a neutral write.
 */
export function ProviderForm({
  snap,
  editing,
  onCancel,
}: {
  snap: Snapshot
  editing: ProviderRow | null
  onCancel: () => void
}): ReactElement {
  const fieldId = useId()
  const [name, setName] = useState(editing?.name ?? '')
  const [provider, setProvider] = useState(editing?.provider ?? '')
  const [baseUrl, setBaseUrl] = useState(editing?.baseUrl ?? '')
  const [model, setModel] = useState(editing?.model ?? '')
  const [apiKey, setApiKey] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const formRef = useRef<HTMLDivElement>(null)
  useEffect(() => {
    formRef.current?.scrollIntoView({ block: 'start' })
    formRef.current?.querySelector<HTMLElement>('input:not(:disabled), select:not(:disabled)')?.focus({ preventScroll: true })
  }, [])
  useEffect(() => {
    if (error) formRef.current?.querySelector('[role="alert"]')?.scrollIntoView({ block: 'nearest' })
  }, [error])
  const types = snap.providerTypes
  const known = types.find(t => t.name === provider)
  const profileModels: readonly CachedModel[] = editing
    ? (snap.providerModels[editing.name] ?? []).map(entry => typeof entry === 'string'
        ? { id: entry, overridden: false }
        : entry)
    : []
  const modelsLoading = editing ? snap.providerModelLoading.includes(editing.name) : false
  const modelWarning = editing ? snap.providerModelWarnings[editing.name] ?? '' : ''
  useEffect(() => {
    if (editing) store.loadProviderModels(editing.name)
  }, [editing?.name])
  // Valid when the wire's required fields are covered; base URL may fall
  // back to the registry default for a known type.
  const valid = name.trim() !== '' && model.trim() !== '' && (baseUrl.trim() !== '' || (known?.baseUrl ?? '') !== '')
  const submit = async (): Promise<void> => {
    if (busy || !valid) return
    setBusy(true)
    setError('')
    try {
      const failure = await store.saveProvider({ name, baseUrl, model, provider, apiKey })
      if (failure) setError(failure)
      else onCancel()
    } finally { setBusy(false) }
  }
  return (
    <div className="provform" ref={formRef}>
      <div className="cap" style={{ paddingLeft: 0 }}>{editing ? `Edit ${editing.name}` : 'New provider profile'}</div>
      <div className="field">
        <label htmlFor={fieldId + '-name'}>Name</label>
        <input id={fieldId + '-name'} className="palette__in" value={name} spellCheck={false} placeholder="e.g. openrouter" disabled={editing !== null} onChange={e => setName(e.target.value)} />
      </div>
      <div className="field">
        <label htmlFor={fieldId + '-provider'}>Provider</label>
        {types.length > 0 ? (
          <select
            id={fieldId + '-provider'}
            className="palette__in provform__select"
            value={provider}
            onChange={e => setProvider(e.target.value)}
          >
            <option value="">choose a provider type…</option>
            {provider && !known && <option value={provider}>{provider} (saved type)</option>}
            {types.map(t => (
              <option key={t.name} value={t.name}>{t.name}</option>
            ))}
          </select>
        ) : (
          // Older daemons have no provider_types — keep the free-text path.
          <input id={fieldId + '-provider'} className="palette__in" value={provider} spellCheck={false} placeholder="e.g. openai, z-ai, moonshot (optional)" onChange={e => setProvider(e.target.value)} />
        )}
      </div>
      <div className="field">
        <label htmlFor={fieldId + '-api-key'}>API key</label>
        <input
          className="palette__in"
          id={fieldId + '-api-key'}
          type="password"
          value={apiKey}
          spellCheck={false}
          placeholder={
            editing
              ? 'leave blank to keep the stored key'
              : known?.apiKeyEnv
                ? `leave blank to use $${known.apiKeyEnv}`
                : 'stored in ~/.xerxes/profiles.json (optional)'
          }
          onChange={e => setApiKey(e.target.value)}
        />
      </div>
      <div className="field">
        <label htmlFor={fieldId + '-model'}>Model</label>
        <input id={fieldId + '-model'} className="palette__in" value={model} spellCheck={false} placeholder="e.g. glm-5.2" onChange={e => setModel(e.target.value)} />
      </div>
      {editing && (
        <div className="field provform__catalog">
          {/* The count claimed the whole catalog while the list rendered
              only the first 24, with nothing saying the rest existed. */}
          <label>Models · {modelsLoading ? 'fetching…' : profileModels.length > 24 ? `showing 24 of ${profileModels.length}` : `${profileModels.length} discovered`}</label>
          {profileModels.length > 0 ? (
            <div className="provform__models" role="list" aria-label={`${editing.name} models`}>
              {profileModels.slice(0, 24).map(cached => (
                <CachedModelEditor
                  key={cached.id}
                  profileName={editing.name}
                  model={cached}
                  selected={model === cached.id}
                  onSelect={() => setModel(cached.id)}
                />
              ))}
            </div>
          ) : (
            <div className="row__s">{modelsLoading ? 'asking this saved provider profile…' : 'no model catalog fetched yet'}</div>
          )}
          {modelWarning && <div className="row__s provform__warning">{modelWarning}</div>}
          <button
            type="button"
            className="chipbtn provform__fetch"
            disabled={modelsLoading}
            onClick={() => store.loadProviderModels(editing.name, true)}
          >
            {modelsLoading ? 'Fetching…' : <><Icon name="retry" size={13} /> Fetch this provider’s models</>}
          </button>
        </div>
      )}
      <details className="provform__custom">
        <summary>Customized settings</summary>
        <div className="field" style={{ marginTop: 8 }}>
          <label htmlFor={fieldId + '-base-url'}>Base URL</label>
          <input
            className="palette__in"
            id={fieldId + '-base-url'}
            value={baseUrl}
            spellCheck={false}
            placeholder={known?.baseUrl ? `Provider default — ${known.baseUrl}` : 'https://api.example.com/v1'}
            onChange={e => setBaseUrl(e.target.value)}
          />
          {known?.baseUrl && !baseUrl.trim() ? (
            <div className="row__s" style={{ marginTop: 4 }}>Leave blank to use the default for {known.name}.</div>
          ) : null}
        </div>
      </details>
      {error && <p role="alert" className="studio-error">{error}</p>}
      <div className="approval__row" style={{ paddingTop: 4 }}>
        <button className="btn btn--solid" disabled={busy || !valid} onClick={() => void submit()}>Save &amp; activate</button>
        <button className="btn" disabled={busy} onClick={onCancel}>Cancel</button>
      </div>
      <p className="row__s">Saving writes the profile, makes it the active provider, and points this session at its model — exactly what the TUI's <code>/provider</code> flow does.</p>
    </div>
  )
}

function CachedModelEditor({
  profileName,
  model,
  selected,
  onSelect,
}: {
  profileName: string
  model: CachedModel
  selected: boolean
  onSelect: () => void
}): ReactElement {
  const [open, setOpen] = useState(false)
  const [context, setContext] = useState(
    model.contextSource === 'override' && model.contextLimit !== undefined
      ? String(model.contextLimit)
      : '',
  )
  const [output, setOutput] = useState(
    model.outputSource === 'override' && model.maxOutputTokens !== undefined
      ? String(model.maxOutputTokens)
      : '',
  )
  useEffect(() => {
    setContext(model.contextSource === 'override' && model.contextLimit !== undefined
      ? String(model.contextLimit)
      : '')
    setOutput(model.outputSource === 'override' && model.maxOutputTokens !== undefined
      ? String(model.maxOutputTokens)
      : '')
  }, [model.contextLimit, model.contextSource, model.maxOutputTokens, model.outputSource])
  const parsedContext = capacityInput(context)
  const parsedOutput = capacityInput(output)
  const valid = parsedContext !== undefined && parsedOutput !== undefined
  const save = (): void => {
    if (!valid) return
    store.saveModelCapabilities(profileName, model.id, parsedContext, parsedOutput)
    setOpen(false)
  }
  const clear = (): void => {
    store.saveModelCapabilities(profileName, model.id, null, null)
    setContext('')
    setOutput('')
    setOpen(false)
  }
  return (
    <div className="provform__modelwrap" role="listitem">
      <div className="provform__modelrow">
        <button
          type="button"
          className={`provform__model${selected ? ' is-on' : ''}`}
          onClick={onSelect}
        >
          <span className="provform__modelname">{model.id}</span>
          <span className="provform__caps">
            in {capacityLabel(model.contextLimit)} · out {capacityLabel(model.maxOutputTokens)}
          </span>
          {selected ? <Icon name="check" size={13} /> : null}
        </button>
        <button
          type="button"
          className="chipbtn provform__editcap"
          aria-expanded={open}
          aria-label={`Edit token capacities for ${model.id}`}
          onClick={() => setOpen(value => !value)}
        >
          Edit
        </button>
      </div>
      {open ? (
        <div className="provform__capedit">
          <label>
            <span>Input / context tokens</span>
            <input
              className="palette__in"
              inputMode="numeric"
              value={context}
              placeholder={model.contextLimit === undefined ? 'unknown' : String(model.contextLimit)}
              onChange={event => setContext(event.target.value)}
            />
          </label>
          <label>
            <span>Maximum output tokens</span>
            <input
              className="palette__in"
              inputMode="numeric"
              value={output}
              placeholder={model.maxOutputTokens === undefined ? 'unknown' : String(model.maxOutputTokens)}
              onChange={event => setOutput(event.target.value)}
            />
          </label>
          {!valid ? <div className="row__s provform__warning">Use positive whole numbers, or leave blank to inherit.</div> : null}
          <div className="approval__row provform__capactions">
            <button className="btn btn--solid" type="button" disabled={!valid} onClick={save}>Save limits</button>
            {model.overridden ? <button className="btn" type="button" onClick={clear}>Use discovered defaults</button> : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}

function capacityInput(value: string): number | null | undefined {
  const trimmed = value.trim()
  if (!trimmed) return null
  if (!/^\d+$/.test(trimmed)) return undefined
  const parsed = Number(trimmed)
  return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : undefined
}

function capacityLabel(value: number | undefined): string {
  if (value === undefined) return 'unknown'
  return Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(value)
}

const PERMISSION_MODES: ReadonlyArray<{ id: PermissionMode; label: string; note: string }> = [
  { id: 'accept-all', label: 'Accept all', note: 'Tools run without asking. Use only in a workspace you trust.' },
  { id: 'auto', label: 'Auto', note: 'Safe reads run; writes and commands ask first.' },
  { id: 'manual', label: 'Manual', note: 'Every tool call asks first.' },
  { id: 'plan', label: 'Plan', note: 'Read-only. Changes are refused.' },
]

function PermissionsCard({ snap }: { snap: Snapshot }): ReactElement {
  const current = snap.permissionMode
  return (
    <>
      <h2 className="modal__title">Permissions</h2>
      <p className="modal__sub">
        The daemon evaluates every tool call against its policy; approvals in Activity map to allow-once / this-session / deny. The mode below is the daemon's live session policy.
      </p>

      <div className="field">
        <label>Permission mode</label>
        <div className="optlist">
          {PERMISSION_MODES.map(mode => (
            <button
              key={mode.id}
              className={`opt${current === mode.id ? ' is-approve' : ''}`}
              aria-pressed={current === mode.id}
              disabled={snap.connection !== 'online' || snap.permissionUpdating}
              onClick={() => store.setPermissionMode(mode.id)}
            >
              <span className="opt__label">{mode.label}</span>
              <span className="opt__desc">{mode.note}</span>
              {current === mode.id ? <span className="opt__kbd"><Icon name="check" size={12} /></span> : null}
            </button>
          ))}
        </div>
        {!current && <p className="row__s">The runtime has not reported a mode for this session yet.</p>}
        {snap.permissionUpdating && <p role="status">Updating permission mode…</p>}
        {snap.permissionError && <p role="alert" className="studio-error">{snap.permissionError}</p>}
      </div>

      <div className="row">
        <div className="row__main">
          <div className="row__t">Approval shortcuts</div>
          <div className="row__s keyhints"><span><kbd>1</kbd> Allow once</span><span><kbd>2</kbd> This session</span><span><kbd>3</kbd> Deny</span><span>while a request is showing</span></div>
        </div>
      </div>
      <div className="row">
        <div className="row__main">
          <div className="row__t">Scope</div>
          <div className="row__s">Current session on the shared daemon</div>
        </div>
      </div>
    </>
  )
}


const PickerAnchor = createContext<RefObject<HTMLSpanElement | null> | null>(null)

/** Escape compositor containing blocks while keeping the invoking chip as anchor. */
export function PickerLayer({ children }: { children: ReactNode }): ReactElement {
  const anchor = useRef<HTMLSpanElement>(null)
  return <><span hidden ref={anchor} /><PickerAnchor.Provider value={anchor}>
    {typeof document === 'undefined' ? children : createPortal(<div className="atelier picker-layer">{children}</div>, document.body)}
  </PickerAnchor.Provider></>
}

/** Anchor each picker to its invoking chip and clamp it inside the viewport. */
function useAnchoredPicker(ref: RefObject<HTMLElement | null>): void {
  const origin = useContext(PickerAnchor)
  useLayoutEffect(() => {
    const popup = ref.current?.closest<HTMLElement>('.modelpop')
    const anchor = origin?.current?.closest<HTMLElement>('.chipanchor') ?? popup?.parentElement?.closest<HTMLElement>('.chipanchor')
    if (!popup || !anchor) return
    const trigger = anchor.querySelector<HTMLElement>('button')
    const place = (): void => {
      const rect = anchor.getBoundingClientRect()
      popup.style.position = 'fixed'
      popup.style.bottom = 'auto'
      popup.style.right = 'auto'
      popup.style.maxWidth = 'calc(100vw - 24px)'
      popup.style.maxHeight = `${Math.max(80, window.innerHeight - 24)}px`
      const width = popup.getBoundingClientRect().width
      const height = popup.getBoundingClientRect().height
      popup.style.left = `${Math.max(12, Math.min(rect.right - width, window.innerWidth - width - 12))}px`
      popup.style.top = `${Math.max(12, Math.min(rect.top - height - 8, window.innerHeight - height - 12))}px`
    }
    place()
    const observer = new ResizeObserver(place)
    observer.observe(popup)
    observer.observe(anchor)
    window.addEventListener('resize', place)
    return () => { observer.disconnect(); window.removeEventListener('resize', place); trigger?.focus() }
  }, [ref, origin])
}

// ── Model picker (anchored popover) ─────────────────────────────────────

export interface ModelGroup {
  readonly provider: string
  readonly choices: readonly ModelChoice[]
}

export function groupModels(models: readonly ModelChoice[], needle = ''): ModelGroup[] {
  const lower = needle.trim().toLowerCase()
  const map = new Map<string, ModelChoice[]>()
  for (const choice of models) {
    if (lower && !choice.id.toLowerCase().includes(lower) && !choice.provider.toLowerCase().includes(lower)) continue
    const bucket = map.get(choice.provider) ?? []
    bucket.push(choice)
    map.set(choice.provider, bucket)
  }
  return [...map.entries()]
    .sort((a, b) => (a[0] === 'other' ? 1 : b[0] === 'other' ? -1 : a[0].localeCompare(b[0])))
    .map(([provider, choices]) => ({ provider, choices }))
}

export function ModelPicker({ snap, onClose }: { snap: Snapshot; onClose: () => void }): ReactElement {
  const [needle, setNeedle] = useState('')
  const groups = useMemo(() => groupModels(snap.models, needle), [snap.models, needle])
  const flat = useMemo(() => groups.flatMap(group => group.choices.map(choice => choice.id)), [groups])
  const [cursor, setCursor] = useState(0)
  const ref = useRef<HTMLInputElement>(null)
  useAnchoredPicker(ref)
  useEffect(() => {
    ref.current?.focus()
  }, [])
  useEffect(() => {
    setCursor(0)
  }, [needle])
  useEffect(() => { ref.current?.closest('.modelpop')?.querySelector('.is-hover')?.scrollIntoView({ block: 'nearest' }) }, [cursor])
  const pick = (id: string): void => {
    if (snap.turnActive) return // hot-swapping under a running turn is refused
    store.pickModel(id)
    onClose()
  }
  const onKey = (event: React.KeyboardEvent): void => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setCursor(value => Math.min(value + 1, Math.max(flat.length - 1, 0)))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setCursor(value => Math.max(value - 1, 0))
    } else if (event.key === 'Enter') {
      if (event.target instanceof HTMLButtonElement) return
      event.preventDefault()
      const id = flat[cursor]
      if (id) pick(id)
    } else if (event.key === 'Escape') {
      event.preventDefault()
      // Without stopping propagation the window-level GlobalKeys also sees
      // Escape — and its next rule cancels the running turn.
      event.stopPropagation()
      onClose()
    }
  }
  return (
    <>
      {/* Click-anywhere dismissal, matching the palette's contract. */}
      <div className="backdrop backdrop--clear" onClick={onClose} />
      <div className="modelpop" role="dialog" aria-label="Model picker" onKeyDown={onKey}>
        <input
          ref={ref}
          className="palette__in"
          placeholder="Search models…"
          spellCheck={false}
          value={needle}
          onChange={e => setNeedle(e.target.value)}
        />
        <div className="palette__list palette__list--models">
          {groups.map(group => (
            <div key={group.provider}>
              <div className="mgroup__cap">{group.provider}</div>
              {group.choices.map(choice => {
                const current = choice.id === snap.model
                const at = flat.indexOf(choice.id)
                return (
                  <button
                    key={choice.id}
                    className={`mrow${at === cursor ? ' is-hover' : ''}`}
                    disabled={snap.turnActive}
                    title={snap.turnActive ? 'wait for the running turn to finish' : undefined}
                    onClick={() => pick(choice.id)}
                    onMouseEnter={() => setCursor(at)}
                  >
                    <span className={`dot ${current ? 'dot--live' : 'dot--idle'}`} />
                    <span className="mrow__name">{choice.id}</span>
                    <span className="mrow__tags">
                      {current ? <span className="tag tag--fast"><Icon name="check" size={11} /> current</span> : null}
                    </span>
                  </button>
                )
              })}
            </div>
          ))}
          {/* "no models discovered yet" used to be asserted for the whole
              discovery window and after every failure alike, and the retry
              below could fail silently on every click. */}
          {groups.length === 0 && (
            <div className="modelpop__status" role="status">
              {snap.turnActive
                ? 'Locked while a turn is running'
                : snap.models.length ? 'No matching model'
                : snap.modelsLoading ? 'Asking your provider…'
                : snap.modelsError ? 'Discovery failed'
                : 'No models discovered yet'}
            </div>
          )}
          {snap.modelsError && snap.models.length === 0 && (
            <p className="modelpop__error" role="alert">{snap.modelsError}</p>
          )}
          {snap.models.length === 0 && (
            <button className="mrow" disabled={snap.modelsLoading} onClick={() => store.loadModels(true)}>
              <span className={snap.modelsLoading ? 'dot dot--live' : 'dot dot--idle'} />
              <span className="mrow__name">{snap.modelsLoading ? 'Fetching…' : snap.modelsError ? 'Try again' : 'Fetch from the runtime'}</span>
            </button>
          )}
        </div>
        <div className="modelpop__foot">
          <span className="keyhints"><span><kbd>↑↓</kbd> select</span><span><kbd>⏎</kbd> use</span><span><kbd>esc</kbd> close</span></span>
          <button className="lnk" onClick={() => { onClose(); store.openSettings('models') }}>Manage providers…</button>
        </div>
      </div>
    </>
  )
}

// ── Combined model/effort chip menu ─────────────────────────────────────

/**
 * The composer's single model chip opens this two-row menu; each row drills
 * into its own picker (models, reasoning levels). Mirrors the reference
 * shell: label left, current value right, chevron affording the drill-down.
 */
export function ModelMenu({ snap, onClose }: { snap: Snapshot; onClose: () => void }): ReactElement {
  const rows = [
    { key: 'model' as const, label: 'Model', value: snap.model ? bareModelName(snap.model) : '—' },
    { key: 'effort' as const, label: 'Effort', value: snap.reasoningEffort ? capitalize(snap.reasoningEffort) : '—' },
  ]
  const [cursor, setCursor] = useState(0)
  const ref = useRef<HTMLDivElement>(null)
  useAnchoredPicker(ref)
  useEffect(() => {
    ref.current?.focus()
  }, [])
  const drill = (key: 'model' | 'effort'): void => {
    onClose()
    if (key === 'model') store.openPicker()
    else store.openReasoningPicker()
  }
  const onKey = (event: React.KeyboardEvent): void => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setCursor(value => Math.min(value + 1, rows.length - 1))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setCursor(value => Math.max(value - 1, 0))
    } else if (event.key === 'Enter') {
      if (event.target instanceof HTMLButtonElement) return
      event.preventDefault()
      drill(rows[cursor]!.key)
    } else if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      onClose()
    }
  }
  return (
    <>
      <div className="backdrop backdrop--clear" onClick={onClose} />
      <div
        ref={ref}
        className="modelpop modelpop--menu"
        role="dialog"
        aria-label="Model and effort"
        tabIndex={-1}
        onKeyDown={onKey}
      >
        <div className="palette__list">
          {rows.map((row, at) => (
            <button
              key={row.key}
              className={`mrow mmenu__row${at === cursor ? ' is-hover' : ''}`}
              onClick={() => drill(row.key)}
              onMouseEnter={() => setCursor(at)}
            >
              <span className="mmenu__label">{row.label}</span>
              <span className="mmenu__value">{row.value}</span>
              <span className="mmenu__chev">›</span>
            </button>
          ))}
        </div>
      </div>
    </>
  )
}

// ── Context usage popover ───────────────────────────────────────────────

const compactTokens = (value: number): string =>
  value >= 1000 ? `~${(value / 1000).toFixed(1).replace(/\.0$/, '')}K` : `~${Math.round(value)}`

/**
 * Context window split: the daemon estimates system-prompt, tool-schema, and
 * transcript tokens with the same counter that drives auto-compaction. Every
 * number carries `~` because it is an estimate, not provider telemetry.
 */
export function ContextMenu({ snap, onClose }: { snap: Snapshot; onClose: () => void }): ReactElement {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    ref.current?.focus()
  }, [])
  const onKey = (event: React.KeyboardEvent): void => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      onClose()
    }
  }
  const breakdown = snap.contextBreakdown
  const total = breakdown?.totalTokens ?? snap.contextTokens ?? 0
  const limit = breakdown?.contextLimit ?? snap.contextMax ?? 0
  const percent = limit > 0 ? Math.round((total / limit) * 100) : null
  const rows: { label: string; tokens: number; tone: string }[] = breakdown
    ? [
        { label: 'System prompt', tokens: breakdown.systemPromptTokens, tone: 'var(--x-secondary)' },
        { label: 'Tools', tokens: breakdown.toolsTokens, tone: 'var(--x-needs)' },
        { label: 'Messages', tokens: breakdown.messagesTokens, tone: 'var(--x-accent)' },
      ]
    : []
  return (
    <>
      <div className="backdrop backdrop--clear" onClick={onClose} />
      <div
        ref={ref}
        className="ctxpop"
        role="dialog"
        aria-label="Context usage"
        tabIndex={-1}
        onKeyDown={onKey}
      >
        <div className="ctxpop__head">
          <span className="ctxpop__pct">
            {snap.contextBreakdownLoading
              ? 'estimating…'
              : percent === null
                ? 'context unknown'
                : `${percent}% of context used`}
          </span>
          {!snap.contextBreakdownLoading && limit > 0 && (
            <span className="ctxpop__frac">{compactTokens(total)} / {compactTokens(limit)}</span>
          )}
        </div>
        {limit > 0 && (
          <div className="meter__bar ctxpop__bar">
            <div className="meter__fill" style={{ width: `${Math.min(100, (total / limit) * 100)}%` }} />
          </div>
        )}
        {rows.map(row => (
          <div key={row.label} className="ctxpop__row">
            <span className="ctxpop__swatch" style={{ background: row.tone }} />
            <span className="ctxpop__name">{row.label}</span>
            <span className="ctxpop__tokens">{compactTokens(row.tokens)}</span>
          </div>
        ))}
      </div>
    </>
  )
}

/** Display name for the chip: the bare model id, not the provider prefix. */
export function bareModelName(model: string): string {
  const slash = model.lastIndexOf('/')
  return slash >= 0 ? model.slice(slash + 1) : model
}

function capitalize(word: string): string {
  return word ? word[0]!.toUpperCase() + word.slice(1) : word
}

// ── Reasoning effort picker ─────────────────────────────────────────────

/**
 * Effort selector for the active model. The rows are whatever the daemon
 * reports for the model in use (pi-ai's per-model ladder) — nothing here is
 * a fixed menu, and models with no reasoning control render the daemon's
 * note rather than a scale that cannot be honored.
 */
export function ReasoningPicker({ snap, onClose }: { snap: Snapshot; onClose: () => void }): ReactElement {
  const rows = snap.reasoningLevels
  const [cursor, setCursor] = useState(() => {
    const at = rows.findIndex(row => row.effort === snap.reasoningEffort)
    return at >= 0 ? at : 0
  })
  const ref = useRef<HTMLDivElement>(null)
  useAnchoredPicker(ref)
  useEffect(() => {
    ref.current?.focus()
  }, [])
  const pick = (effort: string): void => {
    if (snap.turnActive) return // hot-swapping under a running turn is refused
    store.pickReasoning(effort)
    onClose()
  }
  const onKey = (event: React.KeyboardEvent): void => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setCursor(value => Math.min(value + 1, Math.max(rows.length - 1, 0)))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setCursor(value => Math.max(value - 1, 0))
    } else if (event.key === 'Enter') {
      if (event.target instanceof HTMLButtonElement) return
      event.preventDefault()
      const row = rows[cursor]
      if (row) pick(row.effort)
    } else if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      onClose()
    }
  }
  return (
    <>
      <div className="backdrop backdrop--clear" onClick={onClose} />
      <div
        ref={ref}
        className="modelpop modelpop--effort"
        role="dialog"
        aria-label="Reasoning effort"
        tabIndex={-1}
        onKeyDown={onKey}
      >
        <div className="palette__list palette__list--models">
          <div className="mgroup__cap">
            {snap.reasoningLoading ? 'Asking the runtime…' : `Reasoning effort${snap.reasoningDefault ? ` · default ${snap.reasoningDefault}` : ''}`}
          </div>
          {rows.map((row, at) => {
            const current = row.effort === snap.reasoningEffort
            return (
              <button
                key={row.effort}
                className={`mrow${at === cursor ? ' is-hover' : ''}`}
                disabled={snap.turnActive}
                title={snap.turnActive ? 'wait for the running turn to finish' : row.description || undefined}
                onClick={() => pick(row.effort)}
                onMouseEnter={() => setCursor(at)}
              >
                <span className={`dot ${current ? 'dot--live' : 'dot--idle'}`} />
                <span className="mrow__name">{row.effort}</span>
                <span className="mrow__tags">
                  {row.description ? <span className="mrow__desc">{row.description}</span> : null}
                  {current ? <span className="tag tag--fast"><Icon name="check" size={11} /> current</span> : null}
                </span>
              </button>
            )
          })}
          {rows.length === 0 && !snap.reasoningLoading && (
            <div className="modelpop__status" role="status">
              {snap.reasoningNote || 'This model has no reasoning control.'}
            </div>
          )}
        </div>
        {rows.length > 0 && snap.reasoningNote ? (
          <div className="modelpop__foot">
            <span>{snap.reasoningNote}</span>
          </div>
        ) : (
          <div className="modelpop__foot">
            <span className="keyhints"><span><kbd>↑↓</kbd> select</span><span><kbd>⏎</kbd> use</span><span><kbd>esc</kbd> close</span></span>
          </div>
        )}
      </div>
    </>
  )
}

// ── Command palette (⌘K) ────────────────────────────────────────────────

interface PaletteAction {
  readonly id: string
  /** An icon from the app's set, or omitted when the label carries its own
   *  marker (the slash commands already read as `/undo`). These used to be
   *  loose Unicode — `⛨`, `⇄`, `⌕` — which rendered at whatever weight the
   *  system font happened to have and lined up with nothing else. */
  readonly icon?: IconName
  readonly label: string
  readonly hint?: string
  /** When set, run() prefills the input with this and keeps the palette open. */
  readonly prefill?: string
  readonly run: () => void
}

export function CommandPalette({ snap }: { snap: Snapshot }): ReactElement | null {
  if (!snap.paletteOpen) return null
  const [needle, setNeedle] = useState('')
  const [cursor, setCursor] = useState(0)
  const ref = useRef<HTMLInputElement>(null)
  const listRef = useRef<HTMLDivElement>(null)
  useEffect(() => { listRef.current?.children[cursor]?.scrollIntoView({ block: 'nearest' }) }, [cursor, needle])
  useDialogFocus(ref)

  const actions: PaletteAction[] = useMemo(() => {
    // 'new' and session rows join below, gated on turn/connection state.
    const list: PaletteAction[] = [
      {
        id: 'plan',
        icon: 'pause',
        label: snap.planMode ? 'Exit plan mode' : 'Switch to plan mode',
        run: () => store.togglePlanMode(),
      },
      ...(snap.turnActive
        ? [{ id: 'stop', icon: 'stop', label: 'Stop the running task', hint: 'esc', run: () => store.cancel() } satisfies PaletteAction]
        : []),
      {
        id: 'goal',
        label: '/goal — set the task objective',
        // Prefill, don't submit: the objective text still has to be typed.
        // Stays open by declaring keepOpen below.
        run: () => setNeedle('/goal '),
      },
      { id: 'compact', label: '/compact — compact this task now', run: () => void store.submit('/compact') },
      { id: 'settings', icon: 'settings', label: 'Settings…', run: () => store.openSettings() },
      { id: 'models-settings', icon: 'spark', label: 'Models & Providers settings…', run: () => store.openSettings('models') },
      { id: 'permissions', icon: 'shield', label: 'Permissions settings…', run: () => store.openSettings('permissions') },
      { id: 'channels', icon: 'cloud', label: 'Channels settings…', run: () => store.openSettings('channels') },
      { id: 'terminals', icon: 'terminal', label: 'Terminals…', run: () => store.openSettings('terminals') },
      { id: 'session-search', icon: 'search', label: 'Search sessions & messages…', run: () => store.openSessionSearch() },
    ]
    for (const provider of snap.providers) {
      if (provider.active || snap.turnActive) continue
      list.push({
        id: `provider:${provider.name}`,
        icon: 'spark',
        label: `Switch provider: ${provider.name}`,
        hint: provider.model || provider.provider,
        run: () => store.selectProvider(provider.name),
      })
    }
    // Same mid-turn refusal as the store's pickModel — the reload rebuilds
    // the turn runner under a running turn.
    if (!snap.turnActive) {
      for (const choice of snap.models) {
        list.push({
          id: `model:${choice.id}`,
          icon: 'spark',
          label: `Switch model: ${choice.id}`,
          hint: choice.id === snap.model ? 'current' : choice.provider,
          run: () => store.pickModel(choice.id),
        })
      }
    }
    // openSession silently no-ops mid-turn; newChat too — offering them as
    // runnable actions would just close the palette over a dead click.
    if (!snap.turnActive && snap.connection === 'online') {
      list.push({
        id: 'new',
        icon: 'plus',
        // ⌘N runs newChat() — a blank session. This row opens the wizard
        // (worktree, preset, plan ceiling, model), which is ⇧⌘N.
        label: 'New task…',
        hint: '⇧⌘N',
        run: () => store.openTaskModal(),
      })
      list.push({
        id: 'new-blank',
        icon: 'plus',
        label: 'New blank task',
        hint: '⌘N',
        run: () => store.newChat(),
      })
      const seenSessions = new Set<string>()
      for (const row of [...snap.live, ...snap.sessions]) {
        if (seenSessions.has(row.id)) continue
        seenSessions.add(row.id)
        list.push({
          id: `session:${row.id}`,
          icon: 'half',
          label: `Task: ${row.title}`,
          hint: row.age || row.status,
          run: () => void store.openSession(row.id),
        })
      }
    }
    // TUI parity: the daemon's own slash catalog, live from commands.catalog.
    // Prefill (not submit) — most commands take arguments the human still
    // has to type; ⏎ then sends the slash line through store.submit.
    for (const command of snap.commands) {
      if (command.name === 'goal' || command.name === 'compact' || command.name === 'plan') {
        continue // curated equivalents already sit above with better labels
      }
      list.push({
        id: `slash:${command.name}`,
        // the label already carries the slash — '/' + '/undo' reads as '//undo'
        label: `/${command.name}`,
        hint: command.description || 'daemon command',
        prefill: `/${command.name} `,
        run: () => setNeedle(`/${command.name} `),
      })
    }
    return list
  }, [snap.planMode, snap.turnActive, snap.connection, snap.models, snap.model, snap.live, snap.sessions, snap.providers, snap.commands])

  const lower = needle.trim().toLowerCase()
  const filtered = actions.filter(action =>
    !lower || action.label.toLowerCase().includes(lower) || action.id.toLowerCase().includes(lower) || action.hint?.toLowerCase().includes(lower))

  const run = (action: PaletteAction): void => {
    // Prefill actions keep the palette open — closing first would unmount
    // the component and silently discard the setNeedle.
    if (action.id === 'goal' || action.prefill) {
      action.run()
      return
    }
    store.closePalette()
    action.run()
  }
  const onKey = (event: React.KeyboardEvent): void => {
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setCursor(value => Math.min(value + 1, Math.max(filtered.length - 1, 0)))
    } else if (event.key === 'ArrowUp') {
      event.preventDefault()
      setCursor(value => Math.max(value - 1, 0))
    } else if (event.key === 'Enter') {
      if (event.target instanceof HTMLButtonElement) return
      event.preventDefault()
      const action = filtered[cursor]
      if (action) run(action)
    } else if (event.key === 'Escape') {
      event.preventDefault()
      store.closePalette()
    }
  }

  return (
    <>
      <div className="backdrop backdrop--clear" onClick={() => store.closePalette()} />
      <div className="palette" role="dialog" aria-label="Command palette">
        <input
          ref={ref}
          className="palette__in"
          placeholder="Type a command…"
          spellCheck={false}
          // The list is a listbox the input drives; without these a screen
          // reader is told about a plain text field and never hears the
          // highlighted row change under the arrow keys.
          role="combobox"
          aria-expanded={true}
          aria-controls="palette-options"
          aria-activedescendant={filtered[cursor] ? `palette-option-${filtered[cursor].id}` : undefined}
          aria-autocomplete="list"
          value={needle}
          onChange={e => { setNeedle(e.target.value); setCursor(0) }}
          onKeyDown={e => {
            if (e.key === 'Enter' && needle.startsWith('/')) {
              e.preventDefault()
              store.closePalette()
              void store.submit(needle)
            } else {
              onKey(e)
            }
          }}
        />
        <div className="palette__list" ref={listRef} id="palette-options" role="listbox" aria-label="Commands">
          {filtered.map((action, index) => (
            <button
              key={action.id}
              id={`palette-option-${action.id}`}
              role="option"
              aria-selected={index === cursor}
              className={`prow${index === cursor ? ' is-sel' : ''}`}
              onMouseEnter={() => setCursor(index)}
              onClick={() => run(action)}
            >
              <span className="prow__ico">{action.icon && <Icon name={action.icon} size={14} />}</span>
              {action.label}
              {action.hint ? <span className="prow__kbd">{action.hint}</span> : null}
            </button>
          ))}
          {filtered.length === 0 && (
            <div className="prow is-sel">
              {needle.startsWith('/')
                ? <>no match — ⏎ sends “{needle}” to the daemon</>
                : 'no matching command'}
            </div>
          )}
        </div>
        <div className="palette__foot">
          <span><kbd>↑↓</kbd> select</span>
          <span><kbd>⏎</kbd> run</span>
          <span><kbd>esc</kbd> close</span>
          <span className="palette__hint">/ prefix sends a daemon slash command</span>
        </div>
      </div>
    </>
  )
}
