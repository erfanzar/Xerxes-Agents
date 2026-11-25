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
import { Fragment, createContext, useContext, useId, type ReactNode, useLayoutEffect, useEffect, useMemo, useRef, useState, type ReactElement, type RefObject } from 'react'

import { saveAppearance, loadAppearance, isStyleChoice, DEFAULT_FONT_SIZE, DEFAULT_STYLE, FONT_SIZES, FONT_LABELS, STYLES, STYLE_LABELS, styleFromSaved, type StyleChoice } from './appearance.js'
import { chooseStyle } from './appearanceSync.js'
import { applySessionDensity, applyTerminalFont, applyUiScale, loadSessionDensity, loadTerminalFont, loadUiScale, parseTerminalFont, SESSION_DENSITIES, UI_SCALES, type SessionDensity } from './displayPrefs.js'
import { setSidebarViewMode, useThreadInbox, type SidebarViewMode } from './threadInbox.js'
import { loadPalette, PALETTES, paletteVariant, savePalette, type Palette } from './palettes.js'
import { applyBackdrop, backdropFromImage, BACKDROP_PRESETS, BUNDLED_BACKGROUNDS, DEFAULT_BLUR, DEFAULT_CHAT_OPACITY, DEFAULT_PANEL_OPACITY, loadBackdrop, saveBackdrop, transparencySupported, type Backdrop } from './backdrop.js'
import { desktopError } from './desktopRpc.js'
import { handleDialogEscape, useDialogFocus } from './dialogFocus.js'
import { ChannelsCard } from './ChannelsPanel.js'
import { LspCard } from "./LspPanel.js"
import { TerminalsCard } from './TerminalsPanel.js'
import { providerRowState, store, type Snapshot } from './store.js'
import type { CachedModel, ModelChoice, PermissionMode, ProviderRow, SettingsTab } from './types.js'
import { RemoteProviders } from './RemoteProviders.js'
import { ProviderAccountsCenter, accountKindLabel } from './ProviderAccountsCenter.js'
import { USAGE_GROUP_LABEL, usageGroupOf, type UsageGroupKind } from './UsagePanel.js'
import { RunModeSection } from './RunModeSection.js'
import { isMacPlatform } from './runModeView.js'
import { ScrollableCombobox } from './ScrollableCombobox.js'
import { AgentIntelligenceCard } from './AgentIntelligenceCard.js'
import { MemoryGuardSetting } from './MemoryGuardSetting.js'
import { CompactionSetting } from './CompactionSetting.js'
import { Icon, type IconName } from './Icon.js'
import { isVscodeHost } from './host.js'
import { HubPhoneCard } from './HubPhonePanel.js'
import { MacPermissionsCard } from './MacPermissionsPanel.js'
import { T3Icon, type T3IconName } from './T3Icons.js'

// ── Settings modal ──────────────────────────────────────────────────────

const SETTINGS_TABS: ReadonlyArray<{ id: SettingsTab; label: string; icon: T3IconName; group: 'App' | 'Models & Agents' | 'Integrations' }> = [
  { id: 'general', label: 'General', icon: 'sliders', group: 'App' },
  { id: 'permissions', label: 'Permissions', icon: 'shieldCheck', group: 'App' },
  { id: 'terminals', label: 'Terminals', icon: 'terminalSquare', group: 'App' },
  { id: 'macos', label: 'macOS permissions', icon: 'monitor', group: 'App' },
  { id: 'models', label: 'Models & Providers', icon: 'sparkles', group: 'Models & Agents' },
  { id: 'intelligence', label: 'Agent intelligence', icon: 'workflow', group: 'Models & Agents' },
  { id: 'agents', label: 'Agent presets', icon: 'bot', group: 'Models & Agents' },
  { id: 'channels', label: 'Channels', icon: 'messagesSquare', group: 'Integrations' },
  { id: 'mcp', label: 'MCP Servers', icon: 'server', group: 'Integrations' },
  { id: 'lsp', label: 'Language servers', icon: 'codeEditor', group: 'Integrations' },
  { id: 'hub', label: 'Hub & phone', icon: 'globe', group: 'Integrations' },
]

const SETTINGS_NAV_GROUPS: ReadonlyArray<'App' | 'Models & Agents' | 'Integrations'> = [
  'App',
  'Models & Agents',
  'Integrations',
]

/** Key handler for SettingsModal: closes on Escape or ⌘,/Ctrl+, when Settings is the top modal. */
export function handleSettingsModalKey(
  event: Pick<KeyboardEvent, 'key' | 'metaKey' | 'ctrlKey' | 'altKey' | 'shiftKey' | 'defaultPrevented' | 'preventDefault' | 'stopPropagation' | 'stopImmediatePropagation'>,
  close: () => void,
  options: { readonly isTopDialog: boolean; readonly ownsFocus: boolean },
): boolean {
  if (event.defaultPrevented) return false
  if (options.isTopDialog && (event.metaKey || event.ctrlKey) && !event.altKey && !event.shiftKey && event.key === ',') {
    event.preventDefault()
    event.stopPropagation()
    event.stopImmediatePropagation()
    close()
    return true
  }
  if (event.key === 'Escape' && options.ownsFocus) {
    handleDialogEscape(event, close, true)
    return true
  }
  return false
}

export function SettingsModal({ snap }: { snap: Snapshot }): ReactElement | null {
  const ref = useRef<HTMLDivElement>(null)
  const mainRef = useRef<HTMLDivElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)
  useDialogFocus(ref, snap.settingsOpen)
  // Settings modal: quiet search bar (focused via / or ⌘F) + grouped left rail.
  const [tabFilter, setTabFilter] = useState('')
  const vscode = isVscodeHost()
  useEffect(() => {
    if (!snap.settingsOpen) setTabFilter('')
  }, [snap.settingsOpen])
  useEffect(() => {
    if (!snap.settingsOpen || typeof document === 'undefined') return
    const onKey = (event: KeyboardEvent): void => {
      if (event.defaultPrevented) return
      const root = ref.current
      if (!root) return
      // Leave nested modals (e.g. ActionDialog portaled to document.body) untouched.
      const dialogs = document.querySelectorAll<HTMLElement>('[role="dialog"][aria-modal="true"]')
      const isTopDialog = dialogs.length === 0 || dialogs[dialogs.length - 1] === root
      const active = document.activeElement
      const ownsFocus = isTopDialog && (root.contains(active) || active === document.body || active === null)
      if (isTopDialog && !vscode && !event.isComposing) {
        const isCmdF = (event.metaKey || event.ctrlKey) && !event.altKey && !event.shiftKey && event.key.toLowerCase() === 'f'
        const targetEl = event.target instanceof HTMLElement ? event.target : null
        const inEditable = targetEl !== null && (targetEl.tagName === 'INPUT' || targetEl.tagName === 'TEXTAREA' || targetEl.tagName === 'SELECT' || targetEl.isContentEditable)
        const isSlash = event.key === '/' && !event.metaKey && !event.ctrlKey && !event.altKey && !inEditable
        if (isCmdF || isSlash) {
          event.preventDefault()
          searchInputRef.current?.focus()
          searchInputRef.current?.select()
          return
        }
      }
      handleSettingsModalKey(event, () => store.closeSettings(), { isTopDialog, ownsFocus })
    }
    document.addEventListener('keydown', onKey, true)
    return () => document.removeEventListener('keydown', onKey, true)
  }, [snap.settingsOpen, vscode])
  useEffect(() => {
    if (!snap.settingsOpen) return
    const container = mainRef.current
    if (!container) return
    if (vscode) {
      const target = container.querySelector<HTMLElement>(`[data-settings-section="${snap.settingsTab}"]`)
      if (target) target.scrollIntoView?.({ block: 'start' })
      else container.scrollTo?.({ top: 0 })
      return
    }
    container.scrollTo?.({ top: 0 })
  }, [snap.settingsOpen, snap.settingsTab, vscode])
  if (!snap.settingsOpen) return null
  const closeOnBackdrop = (event: React.MouseEvent<HTMLDivElement>): void => {
    if (event.target === event.currentTarget) store.closeSettings()
  }
  // VS Code: one page with what a coding panel needs — models and providers,
  // how hard agents think, then approvals. No section tabs; the rest belongs
  // to the desktop app, while still honoring snap.settingsTab via section scroll.
  if (vscode) return (
    <div className="backdrop" onMouseDown={closeOnBackdrop}>
      <div className="modal modal--single" ref={ref} role="dialog" aria-modal="true" aria-label="Settings" data-active-tab={snap.settingsTab}>
        <header className="modal__head">
          <strong>Settings</strong>
          <button type="button" className="chipbtn" aria-label="Close settings" onClick={() => store.closeSettings()}><Icon name="close" size={13} /></button>
        </header>
        <div className="modal__main" ref={mainRef}>
          <div data-settings-section="general"><StyleField /></div>
          <div data-settings-section="models"><ModelsCard snap={snap} /></div>
          <div data-settings-section="intelligence"><AgentIntelligenceCard snap={snap} /></div>
          <div data-settings-section="permissions"><PermissionsCard snap={snap} /></div>
        </div>
      </div>
    </div>
  )
  const activeTab: SettingsTab = SETTINGS_TABS.some(tab => tab.id === snap.settingsTab) ? snap.settingsTab : 'general'
  const activeTabMeta = SETTINGS_TABS.find(tab => tab.id === activeTab) ?? SETTINGS_TABS[0]!
  const filterQuery = tabFilter.trim().toLowerCase()
  const filteredTabs = SETTINGS_TABS.filter(tab => !filterQuery || tab.label.toLowerCase().includes(filterQuery) || tab.group.toLowerCase().includes(filterQuery))
  const renderTabButton = (tab: (typeof SETTINGS_TABS)[number]): ReactElement => (
    <button
      key={tab.id}
      type="button"
      aria-current={activeTab === tab.id ? 'page' : undefined}
      className={`mtab${activeTab === tab.id ? ' is-on' : ''}`}
      onClick={() => store.setSettingsTab(tab.id)}
    >
      <span className="mtab__icon" aria-hidden="true"><T3Icon name={tab.icon} size={15} /></span>
      <span className="mtab__label">{tab.label}</span>
    </button>
  )
  return (
    <div className="backdrop" onMouseDown={closeOnBackdrop}>
      <div className="modal" ref={ref} role="dialog" aria-modal="true" aria-label="Settings">
        <nav className="modal__side" aria-label="Settings sections">
          <div className="settings-side__top">
            <button type="button" className="settings-back" aria-label="Close settings" onClick={() => store.closeSettings()}>
              <span className="settings-back__icon"><Icon name="arrow" size={13} /></span>
              <span>Back</span>
            </button>
            <label className="settings-search">
              <Icon name="search" size={14} />
              <input
                ref={searchInputRef}
                type="search"
                aria-label="Search settings"
                placeholder="Search"
                value={tabFilter}
                onChange={event => setTabFilter(event.target.value)}
              />
              {tabFilter ? (
                <button type="button" className="settings-search__clear" aria-label="Clear settings search" onClick={() => { setTabFilter(''); searchInputRef.current?.focus() }}>
                  <Icon name="close" size={11} />
                </button>
              ) : (
                <kbd className="settings-search__kbd" aria-hidden="true">/</kbd>
              )}
            </label>
          </div>
          {filterQuery ? (
            <div className="settings-nav-groups">
              {filteredTabs.map(renderTabButton)}
              {filteredTabs.length === 0 && <p className="settings-nav-empty" role="status">No settings found</p>}
            </div>
          ) : (
            <div className="settings-nav-groups">
              {SETTINGS_NAV_GROUPS.map(group => (
                <div key={group} className="settings-nav-group" role="group" aria-label={group}>
                  <h3 className="settings-nav-group__title">{group}</h3>
                  <div className="settings-nav-group__list">
                    {SETTINGS_TABS.filter(tab => tab.group === group).map(renderTabButton)}
                  </div>
                </div>
              ))}
            </div>
          )}
          <div className="modal__sidefoot">
            <span className={`appr-policy${snap.connection === 'online' ? ' is-online' : ''}`}>
              <span className="appr-policy__dot" aria-hidden="true" />
              {snap.connection === 'online' ? 'Runtime connected' : 'Offline · Reconnect to edit'}
            </span>
          </div>
        </nav>
        <div className="modal__main" ref={mainRef}>
          <header className="settings-topbar">
            <nav className="settings-topbar__crumb" aria-label="Settings breadcrumb">
              <span className="settings-topbar__root">Settings</span>
              <span className="settings-topbar__sep" aria-hidden="true">/</span>
              <span className="settings-topbar__current">{activeTabMeta.label}</span>
            </nav>
            <button type="button" className="settings-topbar__close" aria-label="Close settings dialog" onClick={() => store.closeSettings()}>
              <Icon name="close" size={15} />
            </button>
          </header>
          <div className="settings-page-body">
            {activeTab === 'general' && <GeneralCard snap={snap} />}
            {activeTab === 'models' && <ModelsCard snap={snap} />}
            {activeTab === 'intelligence' && <AgentIntelligenceCard snap={snap} />}
            {activeTab === 'agents' && <AgentPresetsCard snap={snap} />}
            {activeTab === 'permissions' && <PermissionsCard snap={snap} />}
            {activeTab === 'channels' && <ChannelsCard snap={snap} />}
            {activeTab === 'mcp' && <McpCard snap={snap} />}
            {activeTab === 'terminals' && <TerminalsCard snap={snap} />}
            {activeTab === 'lsp' && <LspCard snap={snap} />}
            {activeTab === 'hub' && <HubPhoneCard />}
            {activeTab === 'macos' && <MacPermissionsCard />}
          </div>
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
    // Picking light or dark by hand takes precedence over a background's
    // derived mode. With a palette the background stays: the mode only picks
    // the palette's variant.
    // Under Neo a background lends only its colours (data-backdrop-colors); a
    // hand-picked mode overrides it exactly as it does in Classic.
    const root = document.documentElement
    if ((root.hasAttribute('data-backdrop') || root.hasAttribute('data-backdrop-colors')) && !loadPalette()) { applyBackdrop({ kind: 'none' }); saveBackdrop({ kind: 'none' }) }
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
    applyBackdrop(loadBackdrop())
    saveAppearance({ theme: next === 'dark' || next === 'light' ? next : 'system' })
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
    if (typeof window === 'undefined') return
    void window.xerxes?.setNotifications?.(notifications)
    void window.xerxes?.getLoginItem?.().then(state => setLoginItem(state)).catch(() => setLoginItem(null))
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
    if (typeof window !== 'undefined') void window.xerxes?.setNotifications?.(next)
  }
  const toggleLoginItem = (): void => {
    if (typeof window === 'undefined') return
    void window.xerxes?.setLoginItem?.(!loginItem).then(state => setLoginItem(state)).catch(() => setLoginItem(null))
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
      <p className="modal__sub">External tools the agents may call. Tool calls still pass the permission policy; enabling a server is not an auto-approve.</p>
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
      <div className="settings-actions">
        <button type="button" className="btn btn--ghost" disabled={busy !== null || snap.connection !== 'online'} onClick={() => { void refresh() }}>Refresh status</button>
        <button type="button" className="btn btn--ghost" disabled={busy !== null || snap.connection !== 'online'} onClick={() => { void refresh(true) }}>Reload servers</button>
      </div>
      <p className="modal__sub settings-footnote">
        Config: <code>~/.xerxes/mcp.json</code> · A reload reconnects every configured server.
      </p>
    </>
  )
}

/** One background as a card: a small window laid over the picture. */
function BackdropCard({ label, description, paint, on, onPick }: { label: string; description: string; paint: string | undefined; on: boolean; onPick: () => void }): ReactElement {
  return (
    <button type="button" role="radio" aria-checked={on} className={`palette-card backdrop-card${on ? ' is-on' : ''}`} onClick={onPick}>
      <span className={`backdrop-preview${paint === undefined ? ' backdrop-preview--none' : paint === 'transparent' ? ' backdrop-preview--transparent' : ''}`} style={paint && paint !== 'transparent' ? { background: paint } : undefined} aria-hidden>
        <span className="backdrop-preview__side" />
        <span className="backdrop-preview__main"><span className="palette-preview__bar" /><span className="palette-preview__bar palette-preview__bar--muted" /></span>
      </span>
      <span className="palette-card__label">{label}</span>
      <span className="palette-card__desc">{description}</span>
    </button>
  )
}

/**
 * Background the whole theme follows: pick a gradient, two colours, an
 * image, or the see-through window, and the surfaces, accent and light/dark
 * are derived from it.
 */
/** Whether the page is in the Neo style, following a switch made anywhere (Settings, ⌘K, another window). */
function useNeoStyle(): boolean {
  const read = (): boolean => typeof document !== 'undefined' && document.documentElement.getAttribute('data-style') === 'neo'
  const [neo, setNeo] = useState(read)
  useEffect(() => {
    if (typeof MutationObserver === 'undefined') return
    const observer = new MutationObserver(() => setNeo(read()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-style'] })
    return () => observer.disconnect()
  }, [])
  return neo
}

function BackdropField(): ReactElement {
  // Vibes and Neo both show a chosen background; None (the default) keeps their own surfaces.
  const [current, setCurrent] = useState<Backdrop>(() => typeof document === 'undefined' ? { kind: 'none' } : loadBackdrop())
  const [error, setError] = useState('')
  const [custom, setCustom] = useState(() => current.kind === 'gradient' ? { from: current.from, to: current.to } : { from: '#1b2a6b', to: '#1e3a5f' })
  const fileInput = useRef<HTMLInputElement>(null)
  const choose = (picked: Backdrop): void => {
    const levels = current.kind === 'none' ? {} : { ...(current.chat === undefined ? {} : { chat: current.chat }), ...(current.panels === undefined ? {} : { panels: current.panels }), ...(current.blur === undefined ? {} : { blur: current.blur }) }
    const next: Backdrop = picked.kind === 'none' ? picked : { ...levels, ...picked }
    applyBackdrop(next)
    setCurrent(next)
    setError(saveBackdrop(next) ? '' : 'That image is too large to keep; it applies to this window only.')
  }
  // Sliders apply live; the levels survive switching to another background.
  const tune = (levels: { chat?: number; panels?: number; blur?: number }): void => {
    if (current.kind === 'none') return
    choose({ ...current, ...levels })
  }
  const same = (backdrop: Backdrop) => current.kind === 'gradient' && backdrop.kind === 'gradient' && current.from === backdrop.from && current.to === backdrop.to && current.angle === backdrop.angle
  const swatch = (backdrop: Backdrop) => backdrop.kind === 'gradient' ? `linear-gradient(${backdrop.angle}deg, ${backdrop.from}, ${backdrop.to})` : undefined
  const pickImage = async (file: File | undefined): Promise<void> => {
    if (!file) return
    try { choose(await backdropFromImage(file)) } catch (failure) { setError(failure instanceof Error ? failure.message : String(failure)) }
  }
  return (
    <div className="field backdrop-field">
      <label id="backdrop-label">Background</label>
      {/* Cards like the colour themes: each shows a small window over the
          picture, so you see how the panels will sit on it. */}
      <div className="palette-grid backdrop-grid" role="radiogroup" aria-labelledby="backdrop-label">
        <BackdropCard label="None" description="The style's own solid surfaces" paint={undefined} on={current.kind === 'none'} onPick={() => choose({ kind: 'none' })} />
        {transparencySupported() && <BackdropCard label="Transparent" description="Your desktop, blurred through the window" paint="transparent" on={current.kind === 'transparent'} onPick={() => choose({ kind: 'transparent' })} />}
        {BUNDLED_BACKGROUNDS.map(art => (
          <BackdropCard key={art.name} label={art.name} description="Artwork" paint={`center / cover no-repeat url("${art.backdrop.src}")`} on={current.kind === 'image' && current.src === art.backdrop.src} onPick={() => choose(art.backdrop)} />
        ))}
        {BACKDROP_PRESETS.map(preset => (
          <BackdropCard key={preset.name} label={preset.name} description="Gradient" paint={swatch(preset.backdrop)} on={same(preset.backdrop)} onPick={() => choose(preset.backdrop)} />
        ))}
        {/* Your own picture; once chosen it shows here as its own card. */}
        {current.kind === 'image' && current.src.startsWith('data:') && <BackdropCard label="Your image" description="Uploaded picture" paint={`center / cover no-repeat url("${current.src}")`} on onPick={() => undefined} />}
      </div>
      <input ref={fileInput} type="file" accept="image/png,image/jpeg,image/webp" hidden onChange={event => { void pickImage(event.currentTarget.files?.[0]); event.currentTarget.value = '' }} />
      <div className="backdrop-custom">
        <button type="button" className="btn" onClick={() => fileInput.current?.click()}><Icon name="plus" size={13} /> Upload image…</button>
        <span>Custom gradient</span>
        <label className="backdrop-custom__swatch">
          <span>Start</span>
          <input type="color" aria-label="Gradient start" value={custom.from} onChange={event => setCustom(value => ({ ...value, from: event.target.value }))} />
        </label>
        <label className="backdrop-custom__swatch">
          <span>End</span>
          <input type="color" aria-label="Gradient end" value={custom.to} onChange={event => setCustom(value => ({ ...value, to: event.target.value }))} />
        </label>
        <button type="button" className="btn" onClick={() => choose({ kind: 'gradient', from: custom.from, to: custom.to, angle: 150 })}>Apply gradient</button>
      </div>
      {/* Opacity and blur shape how the picture shows through. */}
      {current.kind !== 'none' && <div className="backdrop-levels">
        <label htmlFor="backdrop-chat">Conversation</label>
        <input id="backdrop-chat" type="range" min={0} max={100} value={current.chat ?? DEFAULT_CHAT_OPACITY} aria-valuetext={`${current.chat ?? DEFAULT_CHAT_OPACITY}% solid`}
          onChange={event => tune({ chat: Number(event.target.value) })} />
        <output htmlFor="backdrop-chat">{current.chat ?? DEFAULT_CHAT_OPACITY}%</output>
        <label htmlFor="backdrop-panels">Side panels</label>
        <input id="backdrop-panels" type="range" min={0} max={100} value={current.panels ?? DEFAULT_PANEL_OPACITY} aria-valuetext={`${current.panels ?? DEFAULT_PANEL_OPACITY}% solid`}
          onChange={event => tune({ panels: Number(event.target.value) })} />
        <output htmlFor="backdrop-panels">{current.panels ?? DEFAULT_PANEL_OPACITY}%</output>
        {/* The OS blurs a transparent window itself and has no strength
            setting: on (system blur) or off (clear glass). */}
        {current.kind === 'transparent' && <>
          <label id="backdrop-system-blur">Blur</label>
          <div className="seg" role="radiogroup" aria-labelledby="backdrop-system-blur">
            <button type="button" role="radio" aria-checked={current.systemBlur !== false} className={current.systemBlur !== false ? 'is-on' : ''} onClick={() => choose({ ...current, systemBlur: true })}>System blur</button>
            <button type="button" role="radio" aria-checked={current.systemBlur === false} className={current.systemBlur === false ? 'is-on' : ''} onClick={() => choose({ kind: 'transparent', ...(current.chat === undefined ? {} : { chat: current.chat }), ...(current.panels === undefined ? {} : { panels: current.panels }), systemBlur: false })}>Clear glass</button>
          </div>
          <span />
        </>}
        {current.kind !== 'transparent' && <>
          <label htmlFor="backdrop-blur">Blur</label>
          <input id="backdrop-blur" type="range" min={0} max={100} value={current.blur ?? DEFAULT_BLUR} aria-valuetext={`${current.blur ?? DEFAULT_BLUR}% blur`}
            onChange={event => tune({ blur: Number(event.target.value) })} />
          <output htmlFor="backdrop-blur">{current.blur ?? DEFAULT_BLUR}%</output>
        </>}
      </div>}
      {current.kind !== 'none' && <p className="backdrop-note">{current.kind === 'transparent'
        ? 'How solid each surface is over your desktop. 0% leaves only the system blur. Light or dark follows the Theme setting.'
        : 'How solid each surface is, and how blurred the picture under them is. 0% on all three shows the background as-is. Light or dark and the accent follow the background.'}</p>}
      {error && <p className="backdrop-note backdrop-note--error" role="alert">{error}</p>}
    </div>
  )
}

const STYLE_NOTES: Record<StyleChoice, string> = {
  vibes: 'Pure obsidian and paper surfaces, Inbox sidebar, frosted glass composer, and cobalt blue accents',
  neo: 'Terminal-console layout with monospace typography, crisp corners, and high-contrast cobalt accents',
}

/**
 * Vibes or Neo. Applies at once in every open window; Theme, Colour theme,
 * background and text size work in both.
 */
function StyleField(): ReactElement {
  const vscode = isVscodeHost()
  // VS Code keeps the choice in its own settings and stamps it on the page.
  const read = (): StyleChoice => typeof document === 'undefined' ? DEFAULT_STYLE
    : vscode ? (styleFromSaved(document.documentElement.dataset?.styleChoice) ?? DEFAULT_STYLE)
    : loadAppearance().style
  const [style, setStyle] = useState<StyleChoice>(read)
  useEffect(() => {
    if (typeof MutationObserver === 'undefined' || typeof document === 'undefined') return
    const observer = new MutationObserver(() => setStyle(read()))
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-style', 'data-style-choice'] })
    return () => observer.disconnect()
  }, [])
  const pick = (next: StyleChoice): void => { chooseStyle(next); setStyle(next) }
  const choices = STYLES
  const paletteKey = isMacPlatform() ? '⌘K' : 'Ctrl+K'
  return (
    <div className="row settings-row-control style-field">
      <div className="row__main">
        <label id="style-label" className="row__t">Style</label>
        <p className="row__s">{vscode
          ? `${STYLE_NOTES[style]}. Saved as Xerxes: Style in VS Code settings; high-contrast themes keep VS Code's colours.`
          : `${STYLE_NOTES[style]}. Colour themes and backgrounds work in both styles (${paletteKey}).`}</p>
      </div>
      <div className="seg" role="radiogroup" aria-labelledby="style-label">
        {choices.map(value => (
          <button type="button" key={value} role="radio" aria-checked={style === value} title={STYLE_NOTES[value]} className={style === value ? 'is-on' : ''} onClick={() => pick(value)}>{STYLE_LABELS[value]}</button>
        ))}
      </div>
    </div>
  )
}

/** A palette as a small window: sidebar strip, two text bars, an accent pill. */
function PalettePreview({ palette, mode }: { palette: Palette | undefined; mode: 'light' | 'dark' }): ReactElement {
  const colors = palette ? paletteVariant(palette, mode).colors : undefined
  const style = colors ? {
    '--pv-bg': colors.background, '--pv-side': colors.sidebarBackground ?? colors.card, '--pv-fg': colors.foreground,
    '--pv-muted': colors.mutedForeground, '--pv-primary': colors.primary, '--pv-border': colors.border,
  } as React.CSSProperties : undefined
  return <span className={`palette-preview${colors ? '' : ' palette-preview--xerxes'}`} style={style} aria-hidden><span className="palette-preview__side" /><span className="palette-preview__main"><span className="palette-preview__bar" /><span className="palette-preview__bar palette-preview__bar--muted" /><span className="palette-preview__pill" /></span></span>
}

/**
 * Colour themes (Hermes's palettes). With a background they tint the glass;
 * without one the surfaces are solid. Xerxes keeps its own colours, or the
 * background's, when none is chosen.
 */
function PaletteField(): ReactElement {
  const neo = useNeoStyle()
  const [current, setCurrent] = useState(() => typeof document === 'undefined' ? undefined : loadPalette()?.name)
  const mode: 'light' | 'dark' = typeof document !== 'undefined' && document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark'
  const choose = (name: string | undefined): void => {
    savePalette(name)
    applyBackdrop(loadBackdrop())
    setCurrent(name)
  }
  const options: ReadonlyArray<{ name: string | undefined; label: string; description: string; palette: Palette | undefined }> = [
    { name: undefined, label: 'Xerxes', description: 'Built-in colours, or the background\'s', palette: undefined },
    ...PALETTES.map(palette => ({ name: palette.name, label: palette.label, description: palette.description, palette })),
  ]
  return (
    <div className="field palette-field">
      <label id="palette-label">Colour theme</label>
      <p className="field__note">{neo
        ? 'In Neo the theme\'s colours replace Neo\'s own (and a background\'s); Light / Dark picks the theme\'s variant. Xerxes keeps Neo\'s default cobalt accents.'
        : 'With a background, the theme tints the glass over it; Light / Dark picks the theme\'s variant.'}</p>
      <div className="palette-grid" role="radiogroup" aria-labelledby="palette-label">
        {options.map(option => (
          <button type="button" key={option.label} role="radio" aria-checked={current === option.name} className={`palette-card${current === option.name ? ' is-on' : ''}`} onClick={() => choose(option.name)}>
            <PalettePreview palette={option.palette} mode={mode} />
            <span className="palette-card__label">{option.label}</span>
            <span className="palette-card__desc">{option.description}</span>
          </button>
        ))}
      </div>
    </div>
  )
}

/** UI scale, terminal font, sidebar view and sidebar density (per machine). */
function DisplayFields(): ReactElement {
  const [scale, setScale] = useState(() => typeof document === 'undefined' ? 100 : loadUiScale())
  const [density, setDensity] = useState<SessionDensity>(() => typeof document === 'undefined' ? 'comfortable' : loadSessionDensity())
  const { state: inboxState } = useThreadInbox()
  const [font, setFont] = useState(() => typeof document === 'undefined' ? '' : loadTerminalFont() ?? '')
  const [fontError, setFontError] = useState('')
  const saveFont = (value: string): void => {
    try {
      applyTerminalFont(parseTerminalFont(value))
      setFontError('')
    } catch (error) { setFontError(error instanceof Error ? error.message : String(error)) }
  }
  const modKey = isMacPlatform() ? '⌘' : 'Ctrl'
  return (
    <>
      <div className="row settings-row-control">
        <div className="row__main">
          <label id="sidebar-layout" className="row__t">Sidebar layout</label>
          <p className="row__s">Projects groups threads by machine and project folder. Inbox shows a flat chronological feed with Pinned, Active, Snoozed and Settled shelves.</p>
        </div>
        <div className="seg" role="radiogroup" aria-labelledby="sidebar-layout">
          {(['folders', 'inbox'] as const satisfies readonly SidebarViewMode[]).map(mode => (
            <button
              type="button"
              key={mode}
              role="radio"
              aria-checked={inboxState.viewMode === mode}
              className={inboxState.viewMode === mode ? 'is-on' : ''}
              onClick={() => setSidebarViewMode(mode)}
            >
              {mode === 'folders' ? 'Projects' : 'Inbox'}
            </button>
          ))}
        </div>
      </div>
      <div className="row settings-row-control">
        <div className="row__main">
          <label id="ui-scale" className="row__t">UI scale</label>
          <p className="row__s">Zooms the whole window. {modKey} with +, − and 0 also works. Current: {scale}%.</p>
        </div>
        <div className="seg" role="radiogroup" aria-labelledby="ui-scale">
          {UI_SCALES.map(value => <button type="button" key={value} role="radio" aria-checked={scale === value} className={scale === value ? 'is-on' : ''} onClick={() => { applyUiScale(value); setScale(value) }}>{value}%</button>)}
        </div>
      </div>
      <div className="row settings-row-control">
        <div className="row__main">
          <label id="session-density" className="row__t">Session list density</label>
          <p className="row__s">How much each chat row in the sidebar shows.</p>
        </div>
        <div className="seg" role="radiogroup" aria-labelledby="session-density">
          {SESSION_DENSITIES.map(value => <button type="button" key={value} role="radio" aria-checked={density === value} className={density === value ? 'is-on' : ''} onClick={() => { applySessionDensity(value); setDensity(value) }}>{value[0]!.toUpperCase() + value.slice(1)}</button>)}
        </div>
      </div>
      <div className="row settings-row-stack terminal-font-field">
        <div className="row__main">
          <label htmlFor="terminal-font" className="row__t">Terminal font</label>
          <p className="row__s">An installed font for the built-in terminals. Nerd Fonts render Powerlevel10k and shell icons; leave blank for the default.</p>
        </div>
        <form className="terminal-font-field__row" onSubmit={event => { event.preventDefault(); saveFont(font) }}>
          <input
            id="terminal-font"
            value={font}
            placeholder="MesloLGS NF, or a CSS font list"
            spellCheck={false}
            aria-invalid={fontError ? true : undefined}
            aria-describedby={fontError ? 'terminal-font-error' : undefined}
            onChange={event => setFont(event.target.value)}
            onBlur={event => saveFont(event.target.value)}
          />
          <button type="button" className="btn" disabled={!font} onClick={() => { setFont(''); saveFont('') }}>Use default</button>
        </form>
        {fontError ? <p id="terminal-font-error" className="backdrop-note backdrop-note--error" role="alert">{fontError}</p>
          : <p className="terminal-font-field__preview"><span>Glyph preview</span><code style={{ fontFamily: font ? `${font}, var(--mono)` : 'var(--mono)' }}>{'~/project  main ❯ \ue0b0 \uf113'}</code></p>}
      </div>
    </>
  )
}

function GeneralCard({ snap }: { snap: Snapshot }): ReactElement {
  const [theme, setTheme] = useThemeChoice()
  const [fontSize, setFontSize] = useState(() =>
    typeof document === 'undefined' ? DEFAULT_FONT_SIZE : document.documentElement.getAttribute('data-font') ?? DEFAULT_FONT_SIZE)
  const { notifications, loginItem, toggleNotifications, toggleLoginItem } = useNativeSwitches()
  const applyFont = (size: string): void => {
    if (typeof document !== 'undefined') document.documentElement.setAttribute('data-font', size)
    const font = FONT_SIZES.find(value => value === size)
    if (font) saveAppearance({ font })
    setFontSize(size)
  }
  return (
    <>
      <h2 className="modal__title">General</h2>
      <p className="modal__sub">Appearance, workspace, and runtime preferences apply immediately. Per-task controls are marked below.</p>

      <section className="settings-section-block" aria-labelledby="general-appearance-heading">
        <h3 id="general-appearance-heading" className="row__t settings-section">Appearance</h3>
        <div className="rowlist settings-group">
          <StyleField />
          <div className="row settings-row-control">
            <div className="row__main">
              <label id="theme-label" className="row__t">Theme</label>
              <p className="row__s">Follow the system appearance or lock light or dark mode.</p>
            </div>
            <div className="seg" role="radiogroup" aria-labelledby="theme-label">
              <button type="button" role="radio" aria-checked={theme === 'system'} className={theme === 'system' ? 'is-on' : ''} onClick={() => setTheme('system')}>System</button>
              <button type="button" role="radio" aria-checked={theme === 'dark'} className={theme === 'dark' ? 'is-on' : ''} onClick={() => setTheme('dark')}>Dark</button>
              <button type="button" role="radio" aria-checked={theme === 'light'} className={theme === 'light' ? 'is-on' : ''} onClick={() => setTheme('light')}>Light</button>
            </div>
          </div>
          <div className="row settings-row-control">
            <div className="row__main">
              <label id="ui-font-size" className="row__t">Interface text size</label>
              <p className="row__s">Scales chrome, messages, and code readability across the workspace.</p>
            </div>
            <div className="seg" role="radiogroup" aria-labelledby="ui-font-size">
              {FONT_SIZES.map(size => (
                <button type="button" key={size} role="radio" aria-checked={fontSize === size} title={`${size}px`} className={fontSize === size ? 'is-on' : ''} onClick={() => applyFont(size)}>{FONT_LABELS[size]}</button>
              ))}
            </div>
          </div>
          <DisplayFields />
        </div>
        <div className="rowlist settings-group settings-group--padded">
          <PaletteField />
        </div>
        <div className="rowlist settings-group settings-group--padded">
          <BackdropField />
        </div>
      </section>

      <section className="settings-section-block" aria-labelledby="general-workspace-heading">
        <h3 id="general-workspace-heading" className="row__t settings-section">Workspace &amp; Session</h3>
        <div className="rowlist settings-group">
          <div className="row">
            <div className="row__main">
              <div className="row__t">Setup checklist</div>
              <div className="row__s">Provider, folder and runtime readiness in one place</div>
            </div>
            <button type="button" className="btn" onClick={() => { store.closeSettings(); window.dispatchEvent(new Event("xerxes:setup")) }}>Open checklist</button>
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
                {snap.connection === 'online' ? 'Connected · shared across workspaces' : 'Offline · Retrying automatically'}
              </div>
            </div>
            <button type="button" className="btn" onClick={() => store.retryConnection()}>Reconnect</button>
          </div>
          <div className="row">
            <div className="row__main">
              <div className="row__t">Creator mode</div>
              <div className="row__s">Create and manage presets for future sessions</div>
            </div>
            <button type="button" className="btn" onClick={() => store.setSettingsTab('agents')}>Manage presets</button>
          </div>
        </div>
      </section>

      <section className="settings-section-block" aria-labelledby="general-behavior-heading">
        <h3 id="general-behavior-heading" className="row__t settings-section">Behavior &amp; Safety</h3>
        <div className="rowlist settings-group">
          <div className="row">
            <div className="row__main">
              <div className="row__t">Launch at login</div>
              <div className="row__s">Open Xerxes when you sign in</div>
            </div>
            <button
              type="button"
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
              <div className="row__s">Send system-wide OS notifications when an agent creates an artifact, needs input, stops, fails, or finishes. Allow alerts for Xerxes Agents in System Settings.</div>
            </div>
            <div className="row__controls">
              {typeof window !== 'undefined' && typeof window.xerxes?.openNotificationSettings === 'function' && (
                <button
                  type="button"
                  className="btn btn--ghost"
                  onClick={() => { void window.xerxes?.openNotificationSettings?.() }}
                >
                  System Settings
                </button>
              )}
              {typeof window !== 'undefined' && typeof window.xerxes?.testNotification === 'function' && (
                <button
                  type="button"
                  className="btn btn--ghost"
                  disabled={!notifications}
                  onClick={() => { void window.xerxes?.testNotification?.() }}
                >
                  Test
                </button>
              )}
              <button
                type="button"
                className={`switch${notifications ? ' is-on' : ''}`}
                role="switch"
                aria-checked={notifications}
                aria-label="Notifications"
                onClick={toggleNotifications}
              />
            </div>
          </div>
          <div className="row">
            <div className="row__main">
              <div className="row__t">Stream thinking</div>
              <div className="row__s">Show reasoning while the agent works. Display only; the runtime always records it</div>
            </div>
            <button
              type="button"
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
              <div className="row__t">Plan this task{' '}<span className="provcard__badge provcard__badge--quiet">This task</span></div>
              <div className="row__s">{snap.planMode ? 'On · The agent proposes a checklist before making changes' : 'Review a plan before making changes'}</div>
            </div>
            <button
              type="button"
              className={`switch${snap.planMode ? ' is-on' : ''}`}
              role="switch"
              aria-checked={snap.planMode}
              aria-label="Plan this task"
              onClick={() => store.setPlanMode(!snap.planMode)}
            />
          </div>
          <MemoryGuardSetting snap={snap} />
          <CompactionSetting snap={snap} />
        </div>
      </section>
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
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null)
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
      {!detailOpen && <><button type="button" className="pcard pcard--add" disabled={unavailable || !presets.some(row => row.id === 'creator' && !row.broken)} onClick={() => { void store.draftAgentPreset() }}>
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
                      <span className="pcard__name">{row.name}{row.id.toLowerCase() !== row.name.trim().toLowerCase() && <> <code>{row.id}</code></>}{row.isDefault ? <>{' '}<span className="provcard__badge provcard__badge--accent">Default</span></> : null}{snap.currentAgentPreset === row.id ? <>{' '}<span className="provcard__badge provcard__badge--quiet">This session</span></> : null}</span>
                      <span className="pcard__meta">{row.broken || row.description || 'No description.'}</span>
                    </span>
                  </div>
                  <div className="pcard__actions">
                    {!row.broken && !row.isDefault && <button type="button" className="chipbtn" disabled={unavailable} onClick={() => { void runAction(() => store.setDefaultAgentPreset(row.id)) }}>Set default</button>}
                    {!row.broken && !snap.turnActive && row.id !== snap.currentAgentPreset && (
                      <button
                        type="button"
                        className="chipbtn"
                        disabled={unavailable || !canSelect}
                        title={canSelect ? `Run the current session with ${row.id}` : 'Start a task first. A preset binds to an existing session.'}
                        onClick={() => { void runAction(() => store.selectAgentPreset(row.id)) }}
                      >
                        Use here
                      </button>
                    )}
                    <button type="button" className="chipbtn" disabled={unavailable || Boolean(row.broken)} onClick={() => inspect(row.id)}>View</button>
                    {!row.broken && row.manageable && <button type="button" className="chipbtn" disabled={unavailable} onClick={() => edit(row.id)}>Edit</button>}
                    {!row.broken && <button type="button" className="chipbtn" disabled={unavailable} onClick={() => { clearError(); setCopyFrom(row.id); setCopyId(''); setCopyName('') }}>Duplicate</button>}
                    {row.manageable && <button type="button" className="chipbtn" disabled={unavailable} onClick={() => { void runAction(() => store.openAgentPresetLocation(row.id)) }}>Open folder</button>}
                    {row.manageable && (confirmDelete === row.id ? (
                      <>
                        <button type="button" className="chipbtn chipbtn--danger" disabled={unavailable} onClick={() => { setConfirmDelete(null); void runAction(() => store.removeAgentPreset(row.id)) }}>Confirm delete</button>
                        <button type="button" className="chipbtn" disabled={unavailable} onClick={() => setConfirmDelete(null)}>Cancel</button>
                      </>
                    ) : (
                      <button type="button" className="chipbtn chipbtn--danger" disabled={unavailable} onClick={() => setConfirmDelete(row.id)}>Delete</button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )
      })}</>}
      {copyFrom && (
        <div className="provform" ref={detail} tabIndex={-1} aria-label="Duplicate preset">
          <form onSubmit={event => { event.preventDefault(); if (!unavailable && /^[a-z0-9][a-z0-9-]*$/.test(copyId)) void duplicate() }}>
            <div className="row__t">Duplicate {copyFrom}</div>
            <div className="field">
              <label htmlFor="preset-copy-id">Identifier</label>
              <input
                id="preset-copy-id"
                disabled={unavailable}
                value={copyId}
                spellCheck={false}
                placeholder="my-agent"
                aria-invalid={copyId.length > 0 && !/^[a-z0-9][a-z0-9-]*$/.test(copyId) ? true : undefined}
                aria-describedby="preset-copy-id-hint"
                onChange={event => setCopyId(event.target.value)}
              />
              <p id="preset-copy-id-hint" className="row__s">Lowercase letters, numbers, and hyphens.</p>
            </div>
            <div className="field"><label htmlFor="preset-copy-name">Display name</label><input id="preset-copy-name" disabled={unavailable} value={copyName} placeholder="Optional" onChange={event => setCopyName(event.target.value)} /></div>
            <div className="preset-actions">
              <button type="button" className="btn btn--ghost" disabled={busy} onClick={() => { clearError(); setCopyFrom(null) }}>Cancel</button>
              <button type="submit" className="btn btn--solid" disabled={unavailable || !/^[a-z0-9][a-z0-9-]*$/.test(copyId)}>{busy ? 'Creating…' : 'Create preset'}</button>
            </div>
          </form>
        </div>
      )}
      {viewing && (
        <div className="provform" ref={detail} tabIndex={-1} aria-label="Preset composition">
          <button type="button" className="btn btn--ghost" onClick={() => setViewing(null)}>Back to presets</button>
          <div className="row__t">Composition · {viewing.id}</div>
          <pre className="preset-composition">{viewing.content}</pre>
        </div>
      )}
      {editing && (
        <div className="provform" ref={detail} tabIndex={-1} aria-label="Edit preset">
          <form onSubmit={event => { event.preventDefault(); if (!unavailable && editing.dirty) void saveEdit() }}>
            <div className="row__t">Edit · {editing.id}</div>
            <p className="row__s">The runtime validates the spec on save.</p>
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
              <button type="button" className="btn btn--ghost" disabled={busy} onClick={() => { clearError(); setEditing(null) }}>Cancel</button>
              <button type="submit" className="btn btn--solid" disabled={unavailable || !editing.dirty}>{busy ? 'Saving…' : 'Save preset'}</button>
            </div>
          </form>
        </div>
      )}
    </>
  )
}

/** The same families as the Usage tab: Claude Code, Codex, Gemini, then everything else. */
const PROVIDER_GROUP_RANK: Record<UsageGroupKind, number> = { claude: 0, codex: 1, gemini: 2, other: 3 }
function providerGroup(provider: ProviderRow): UsageGroupKind {
  return usageGroupOf(provider.provider)
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
      {snap.storageScope?.startsWith('ssh:') && <RemoteProviders key={snap.sessionKey} remote={typeof window === 'undefined' ? undefined : window.xerxes?.remote} accounts={typeof window === 'undefined' ? undefined : window.xerxes?.accounts} relay={snap.providerRelay} changed={()=>store.loadModels(true)}/>}
      {/* Private and Hybrid run Bonsai on the workspace host, so the choice is offered for this Mac only. */}
      {!snap.storageScope?.startsWith('ssh:') && <RunModeSection snap={snap} />}
      <p className="modal__sub">
        Change the session model from the composer. Using a profile makes it the one that does the work. Profiles are stored in <code>~/.xerxes/profiles.json</code> on the workspace host.
      </p>

      <div className="rowlist settings-group">
        <div className="row settings-row-control">
          <div className="row__main">
            <label className="row__t">Current model</label>
            <p className="row__s">{snap.turnActive ? 'Locked while a turn runs. The model cannot change mid-turn.' : 'Active model for this session. Also switchable from the composer.'}</p>
          </div>
          <button
            type="button"
            className="chipbtn settings-model"
            title="Change model"
            disabled={snap.turnActive}
            onClick={() => store.openPicker()}
          >
            <Icon name="spark" size={12} /> {snap.model || 'unset'} <Icon name="caretDown" size={11} />
          </button>
        </div>
      </div>

      {/* One Providers list: sign-ins and API keys share rows; a signed-in account adds its usage, Sync and Remove. */}
      <ProviderAccountsCenter
        key={`${snap.storageScope || 'local'}:${snap.cwd || ''}:${snap.sessionKey}`}
        accounts={typeof window === 'undefined' ? undefined : window.xerxes?.accounts}
        onAccountLogin={typeof window === 'undefined' ? undefined : window.xerxes?.onAccountLogin}
        sessionKey={snap.sessionKey}
        turnActive={snap.turnActive}
        changed={()=>void store.loadModels(true)}
        shownProfiles={new Set(snap.providers.map(provider => provider.name))} rows={slots => <>
      {snap.providerSwitching && <p role="status">Switching to {snap.providerSwitching}…</p>}
      {snap.providerSwitchError && <p role="alert" className="studio-error">{snap.providerSwitchError}</p>}
      {snap.providerError && <div role="alert" className="studio-error studio-error--row"><span>{snap.providerError}</span><button type="button" className="btn" onClick={() => void store.loadProviders()}>Retry loading providers</button></div>}
      <div className="provgrid">
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
      {[...snap.providers].sort((a, b) => PROVIDER_GROUP_RANK[providerGroup(a)] - PROVIDER_GROUP_RANK[providerGroup(b)] || Number(b.active) - Number(a.active)).map((provider, index, sorted) => {
        const row = providerRowState(snap, provider)
        const slot = slots.get(provider.name)
        const group = USAGE_GROUP_LABEL[providerGroup(provider)]
        const heading = index === 0 || providerGroup(sorted[index - 1]!) !== providerGroup(provider)
        const name = provider.label || provider.name
        const tone = row.chip === 'Mac not connected' ? 'needs' : row.chip === 'host default' ? 'quiet' : 'accent'
        const useViaLabel = isMacPlatform() ? 'Use via Mac' : 'Use via this computer'
        return (
        <Fragment key={provider.name}>
        {heading && <div className="provgroup">{group}</div>}
        {/* The account-card look: name + kind badge, model, provider line, text buttons, usage below. */}
        <div className={`provcard provider-account${row.inUse ? ' is-current' : ''}`}>
          <div className="provcard__main">
            <span className="provcard__text">
              <span className="provcard__title">
                <span className="provcard__name">{slot && slot.account.kind !== 'api_key' ? slot.account.label : name}</span>
                <span className="provcard__badge provcard__badge--quiet">{accountKindLabel(slot ? slot.account.kind : provider.signsIn ? 'sign-in' : 'api_key')}</span>
                {row.chip ? <span className={`provcard__badge provcard__badge--${tone}`}>{row.chip}</span> : null}
                {provider.local === true && <span className="provcard__badge provcard__badge--quiet">on this machine</span>}
                {snap.runMode?.mode === 'hybrid' && snap.runMode.worker?.profile === provider.name && <span className="provcard__badge provcard__badge--quiet">Hybrid lead</span>}
              </span>
              <span className="provcard__model">{provider.name} · {provider.model || 'model chosen when used'}</span>
              <span className="provcard__meta">{provider.provider} · {row.status}</span>
            </span>
            <span className="provcard__tools">
              {confirmDelete === provider.name ? (
                <>
                  <button type="button" className="btn btn--danger" disabled={snap.turnActive} onClick={() => { setConfirmDelete(null); store.deleteProvider(provider.name) }}>Delete for good</button>
                  <button type="button" className="btn" onClick={() => setConfirmDelete(null)}>Keep</button>
                </>
              ) : (
                <>
                  {!row.inUse && (
                    <button type="button" className="btn" disabled={snap.turnActive || !!snap.providerSwitching}
                      title={snap.turnActive ? 'Wait for the running turn to finish' : row.viaMac ? `Run this task through ${name} on this computer` : `make ${name} the active profile`}
                      onClick={() => store.selectProvider(provider.name)}>{row.viaMac ? useViaLabel : 'Use'}</button>
                  )}
                  {/* The host's sign-in can be selected before any relay exists, too. */}
                  {snap.storageScope?.startsWith('ssh:') && provider.signsIn && provider.ready === true && (!row.inUse || row.viaMac) && !snap.turnActive && (
                    <button className="btn" disabled={!!snap.providerSwitching} type="button" title={`Run this task on the host's own ${name} sign-in`} onClick={() => store.useHostProvider(provider.name)}>Run on the server</button>
                  )}
                  {slot?.sync && <button type="button" className="btn" disabled={snap.turnActive || slot.busy} title={`Sync ${name} to this SSH host`} onClick={slot.sync}>Sync to SSH</button>}
                  {slot?.relogin && <button type="button" className="btn" disabled={slot.busy} title={`Sign in to ${name} again; the open SSH host's copy is refreshed too`} onClick={slot.relogin}>Sign in again</button>}
                  <button type="button" className="btn" disabled={snap.turnActive} title={`Edit ${name}`} onClick={() => setForm(provider.name)}>Edit</button>
                  {/* A signed-in account's Remove also clears its credential; deleting only the profile would be relinked. */}
                  {!provider.active && (
                    <button type="button" className="btn" disabled={snap.turnActive || slot?.busy === true} title={`Delete ${name}`} onClick={() => slot?.remove ? slot.remove() : setConfirmDelete(provider.name)}>Remove</button>
                  )}
                </>
              )}
            </span>
          </div>
          {slot?.usage}
        </div>
        </Fragment>
      )})}
      {form === false && (
        <button type="button" className="provcard provcard--add" disabled={snap.turnActive} onClick={() => setForm(true)}>
          <span className="provcard__avatar" aria-hidden="true"><Icon name="plus" size={15} /></span>
          <span className="provcard__text"><span className="provcard__name">Add provider</span><span className="provcard__meta">An API key, a local endpoint or a sign-in</span></span>
        </button>
      )}
      </div>
      </>} />

      {form !== false && (
        <ProviderForm
          key={typeof form === 'string' ? form : 'new'}
          snap={snap}
          editing={editing}
          onCancel={() => setForm(false)}
        />
      )}

      <div className="row__t settings-section">Discovered models{snap.models.length > 0 && <span className="settings-section__count">{snap.models.length}</span>}</div>
      <div className="rowlist">
      {groups.map(group => (
        <div key={group.provider} className="row">
          <span className="dot dot--done" />
          <div className="row__main">
            <div className="row__t">{group.provider}{' '}<span className="provcard__badge provcard__badge--quiet">{group.choices.length} model{group.choices.length === 1 ? '' : 's'}</span></div>
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
        <button type="button" className="btn" onClick={() => store.loadModels(true)}><Icon name="retry" size={13} /> Fetch models</button>
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
  const [modelFilter, setModelFilter] = useState('')
  const [customOpen, setCustomOpen] = useState(() => Boolean(editing?.baseUrl))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const formRef = useRef<HTMLFormElement>(null)
  useEffect(() => {
    formRef.current?.scrollIntoView({ block: 'start' })
    formRef.current?.querySelector<HTMLElement>('input:not(:disabled), select:not(:disabled)')?.focus({ preventScroll: true })
  }, [])
  useEffect(() => {
    if (error) formRef.current?.querySelector('[role="alert"]')?.scrollIntoView({ block: 'nearest' })
  }, [error])
  const types = snap.providerTypes
  const known = types.find(t => t.name === provider)
  const baseUrlRequired = !(known?.baseUrl)
  const profileModels: readonly CachedModel[] = editing
    ? (snap.providerModels[editing.name] ?? []).map(entry => typeof entry === 'string'
        ? { id: entry, overridden: false }
        : entry)
    : []
  const modelIds = useMemo(() => profileModels.map(entry => entry.id), [profileModels])
  const filteredModels = useMemo(() => {
    const query = modelFilter.trim().toLowerCase()
    if (!query) return profileModels
    return profileModels.filter(entry => entry.id.toLowerCase().includes(query))
  }, [profileModels, modelFilter])
  const modelsLoading = editing ? snap.providerModelLoading.includes(editing.name) : false
  const modelWarning = editing ? snap.providerModelWarnings[editing.name] ?? '' : ''
  useEffect(() => {
    if (editing) store.loadProviderModels(editing.name)
  }, [editing?.name])
  useEffect(() => {
    if (baseUrlRequired) setCustomOpen(true)
  }, [baseUrlRequired])
  // Valid when the wire's required fields are covered; base URL may fall
  // back to the registry default for a known type.
  const valid = name.trim() !== '' && (baseUrl.trim() !== '' || (known?.baseUrl ?? '') !== '')
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
    <form className="provform" ref={formRef} onSubmit={event => { event.preventDefault(); void submit() }}>
      <div className="cap provform__cap">{editing ? `Edit ${editing.name}` : 'New provider profile'}</div>
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
            <option value="">Choose a provider type…</option>
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
                : 'Stored in ~/.xerxes/profiles.json (optional)'
          }
          onChange={e => setApiKey(e.target.value)}
        />
      </div>
      <div className="field">
        <label htmlFor={fieldId + '-model'}>Model <span className="field__optional">optional</span></label>
        <ScrollableCombobox
          id={fieldId + '-model'}
          variant="field"
          value={model}
          options={modelIds}
          onChange={setModel}
          placeholder="Leave blank to use the first model this key offers"
          listLabel={editing ? `${editing.name} models` : 'Provider models'}
          toggleLabel="Show discovered models"
          spellCheck={false}
        />
      </div>
      {editing && (
        <div className="field provform__catalog">
          <label>Models · {modelsLoading ? 'fetching…' : modelFilter.trim() ? `${filteredModels.length} of ${profileModels.length} discovered` : `${profileModels.length} discovered`}</label>
          {profileModels.length > 8 && (
            <input
              className="palette__in"
              type="search"
              aria-label={`Filter ${editing.name} models`}
              placeholder="Filter discovered models…"
              spellCheck={false}
              value={modelFilter}
              onChange={event => setModelFilter(event.target.value)}
            />
          )}
          {profileModels.length > 0 ? (
            filteredModels.length > 0 ? (
              <div className="provform__models" role="list" aria-label={`${editing.name} models`}>
                {filteredModels.map(cached => (
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
              <div className="row__s">No discovered models match “{modelFilter.trim()}”.</div>
            )
          ) : (
            <div className="row__s">{modelsLoading ? 'Fetching model catalog…' : 'No model catalog fetched yet.'}</div>
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
      <details
        className="provform__custom"
        open={baseUrlRequired || customOpen}
        onToggle={event => setCustomOpen(event.currentTarget.open)}
      >
        <summary>Customized settings</summary>
        <div className="field provform__custom-field">
          <label htmlFor={fieldId + '-base-url'}>Base URL{!baseUrlRequired && <> <span className="field__optional">optional</span></>}</label>
          <input
            className="palette__in"
            id={fieldId + '-base-url'}
            value={baseUrl}
            required={baseUrlRequired}
            spellCheck={false}
            placeholder={known?.baseUrl ? `Provider default — ${known.baseUrl}` : 'https://api.example.com/v1'}
            onChange={e => setBaseUrl(e.target.value)}
          />
          {known?.baseUrl && !baseUrl.trim() ? (
            <div className="row__s provform__hint">Leave blank to use the default for {known.name}.</div>
          ) : baseUrlRequired && !baseUrl.trim() ? (
            <div className="row__s provform__hint">Required for custom or OpenAI-compatible endpoints.</div>
          ) : null}
        </div>
      </details>
      {error && <p role="alert" className="studio-error">{error}</p>}
      <div className="approval__row provform__actions">
        <button type="submit" className="btn btn--solid" disabled={busy || !valid}>{busy ? 'Saving…' : 'Save & activate'}</button>
        <button type="button" className="btn" disabled={busy} onClick={onCancel}>Cancel</button>
      </div>
      <p className="row__s">Saving writes the profile, makes it the active provider, and points this session at its model, matching <code>/provider</code>.</p>
    </form>
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
  const capId = useId()
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
          <label htmlFor={`${capId}-ctx`}>
            <span>Input / context tokens</span>
            <input
              id={`${capId}-ctx`}
              className="palette__in"
              inputMode="numeric"
              value={context}
              aria-invalid={parsedContext === undefined ? true : undefined}
              aria-describedby={!valid ? `${capId}-err` : undefined}
              placeholder={model.contextLimit === undefined ? 'Unknown' : String(model.contextLimit)}
              onChange={event => setContext(event.target.value)}
            />
          </label>
          <label htmlFor={`${capId}-out`}>
            <span>Maximum output tokens</span>
            <input
              id={`${capId}-out`}
              className="palette__in"
              inputMode="numeric"
              value={output}
              aria-invalid={parsedOutput === undefined ? true : undefined}
              aria-describedby={!valid ? `${capId}-err` : undefined}
              placeholder={model.maxOutputTokens === undefined ? 'Unknown' : String(model.maxOutputTokens)}
              onChange={event => setOutput(event.target.value)}
            />
          </label>
          {!valid ? <div id={`${capId}-err`} className="row__s provform__warning" role="alert">Use positive whole numbers, or leave blank to inherit.</div> : null}
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
  { id: 'accept-all', label: 'Accept all', note: 'Skip ordinary tool approvals. Plan first and required confirmations still apply. Use only in a workspace you trust.' },
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
        {snap.planMode
          ? <><strong>Effective policy: Plan first</strong>. Accept all does not override Plan first. Read-only tools can run; commands Xerxes cannot verify as read-only may still ask. The plan itself requires approval before implementation.</>
          : 'The daemon checks each tool call against the session policy. Some actions require approval in every mode.'}
      </p>

      <div className="field">
        <label id="perm-mode-label">{snap.planMode ? 'Saved permission mode' : 'Permission mode'}</label>
        <div className="optlist" role="group" aria-labelledby="perm-mode-label">
          {PERMISSION_MODES.map(mode => (
            <button
              type="button"
              key={mode.id}
              className={`opt${current === mode.id ? ' is-approve' : ''}`}
              disabled={snap.connection !== 'online' || snap.permissionUpdating}
              aria-pressed={current === mode.id}
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

      <div className="rowlist settings-group">
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
      </div>
    </>
  )
}


const PickerAnchor = createContext<RefObject<HTMLSpanElement | null> | null>(null)

/** Escape compositor containing blocks while keeping the invoking chip as anchor. */
export function PickerLayer({ children }: { children: ReactNode }): ReactElement {
  const anchor = useRef<HTMLSpanElement>(null)
  return <><span hidden ref={anchor} /><PickerAnchor.Provider value={anchor}>
    {typeof document === 'undefined' || !document.body ? children : createPortal(<div className="atelier picker-layer">{children}</div>, document.body)}
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

/** A row as the provider names it; the id itself when the provider gives no name. */
export function modelRowName(choice: ModelChoice): { name: string; hint?: string } {
  return { name: choice.label ?? choice.id, ...(choice.hint ? { hint: choice.hint } : {}) }
}

export function groupModels(models: readonly ModelChoice[], needle = ''): ModelGroup[] {
  const lower = needle.trim().toLowerCase()
  const map = new Map<string, ModelChoice[]>()
  for (const choice of models) {
    if (lower && !choice.id.toLowerCase().includes(lower) && !choice.provider.toLowerCase().includes(lower)
      && !modelRowName(choice).name.toLowerCase().includes(lower)) continue
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
          type="search"
          className="palette__in"
          aria-label="Search models"
          placeholder="Search models…"
          spellCheck={false}
          role="combobox"
          aria-expanded={true}
          aria-controls="model-picker-options"
          aria-activedescendant={flat[cursor] ? `model-picker-option-${cursor}` : undefined}
          aria-autocomplete="list"
          value={needle}
          onChange={e => setNeedle(e.target.value)}
        />
        <div className="palette__list palette__list--models" id="model-picker-options" role="listbox" aria-label="Models">
          {groups.map(group => (
            <div key={group.provider}>
              <div className="mgroup__cap">{group.provider}</div>
              {group.choices.map(choice => {
                const current = choice.id === snap.model
                const at = flat.indexOf(choice.id)
                return (
                  <button
                    type="button"
                    key={choice.id}
                    id={`model-picker-option-${at}`}
                    role="option"
                    aria-selected={at === cursor}
                    className={`mrow${at === cursor ? ' is-hover' : ''}`}
                    disabled={snap.turnActive}
                    title={snap.turnActive ? 'Wait for the running turn to finish' : choice.id}
                    onClick={() => pick(choice.id)}
                    onMouseEnter={() => setCursor(at)}
                  >
                    <span className={`dot ${current ? 'dot--live' : 'dot--idle'}`} />
                    <span className="mrow__name">{modelRowName(choice).name}</span>
                    <span className="mrow__tags">
                      {modelRowName(choice).hint ? <span className="mrow__desc">{modelRowName(choice).hint}</span> : null}
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
            <button type="button" className="mrow" disabled={snap.modelsLoading} onClick={() => store.loadModels(true)}>
              <span className={snap.modelsLoading ? 'dot dot--live' : 'dot dot--idle'} />
              <span className="mrow__name">{snap.modelsLoading ? 'Fetching…' : snap.modelsError ? 'Try again' : 'Fetch from the runtime'}</span>
            </button>
          )}
        </div>
        <div className="modelpop__foot">
          <span className="keyhints"><span><kbd>↑↓</kbd> select</span><span><kbd>⏎</kbd> use</span><span><kbd>Esc</kbd> close</span></span>
          <button type="button" className="lnk" onClick={() => { onClose(); store.openSettings('models') }}>Manage providers…</button>
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
/** The Agents chip's choices, in the order the menu lists them. */
export const DELEGATION_CHOICES = [
  { mode: 'off', label: 'Off', detail: 'Works alone. No agents or workflows unless you ask for them.' },
  { mode: 'auto', label: 'Auto', detail: 'Fans out when work splits: reviews, sweeps, migrations, verification. Asks before a run of more than about 10 agents: budget or thorough.' },
  { mode: 'eager', label: 'Eager', detail: 'Delegates whenever work can split, with a verify step. Fastest and most thorough; uses the most tokens and plan quota.' },
] as const

export function DelegationMenu({ snap, onClose }: { snap: Snapshot; onClose: () => void }): ReactElement {
  const current = DELEGATION_CHOICES.findIndex(choice => choice.mode === snap.delegationMode)
  const [cursor, setCursor] = useState(Math.max(0, current))
  const ref = useRef<HTMLDivElement>(null)
  useAnchoredPicker(ref)
  useEffect(() => { ref.current?.focus() }, [])
  const choose = (at: number): void => { void store.setDelegationMode(DELEGATION_CHOICES[at]!.mode) }
  const onKey = (event: React.KeyboardEvent): void => {
    if (event.key === 'ArrowDown') { event.preventDefault(); setCursor(value => Math.min(value + 1, DELEGATION_CHOICES.length - 1)) }
    else if (event.key === 'ArrowUp') { event.preventDefault(); setCursor(value => Math.max(value - 1, 0)) }
    else if (event.key === 'Enter') { if (event.target instanceof HTMLButtonElement) return; event.preventDefault(); choose(cursor) }
    else if (event.key === 'Escape') { event.preventDefault(); event.stopPropagation(); onClose() }
  }
  return (
    <>
      <div className="backdrop backdrop--clear" onClick={onClose} />
      <div ref={ref} className="modelpop modelpop--menu delegation-menu" role="dialog" aria-label="Agents" tabIndex={-1} onKeyDown={onKey}>
        <div className="palette__list" role="listbox" aria-label="Agents">
          {DELEGATION_CHOICES.map((choice, at) => (
            <button
              type="button"
              key={choice.mode}
              role="option"
              aria-selected={choice.mode === snap.delegationMode}
              className={`mrow delegation-menu__row${at === cursor ? ' is-hover' : ''}`}
              onClick={() => choose(at)}
              onMouseEnter={() => setCursor(at)}
            >
              <span className="delegation-menu__mark" aria-hidden="true">{choice.mode === snap.delegationMode ? <Icon name="check" size={13} /> : null}</span>
              <span className="delegation-menu__text">
                <span className="delegation-menu__label">{choice.label}{choice.mode === 'auto' && <span className="delegation-menu__default"> · default</span>}</span>
                <span className="delegation-menu__detail">{choice.detail}</span>
              </span>
            </button>
          ))}
        </div>
        <div className="delegation-menu__foot">This conversation only · also <code>/delegate off|auto|eager</code></div>
      </div>
    </>
  )
}

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
              type="button"
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
              ? 'Estimating…'
              : percent === null
                ? 'Context unknown'
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
                type="button"
                key={row.effort}
                className={`mrow${at === cursor ? ' is-hover' : ''}`}
                disabled={snap.turnActive}
                title={snap.turnActive ? 'Wait for the running turn to finish' : row.description || undefined}
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
            <span className="keyhints"><span><kbd>↑↓</kbd> select</span><span><kbd>⏎</kbd> use</span><span><kbd>Esc</kbd> close</span></span>
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

  const mac = isMacPlatform()
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
        ? [{ id: 'stop', icon: 'stop', label: 'Stop the running task', hint: 'Esc Esc', run: () => store.cancel() } satisfies PaletteAction]
        : []),
      {
        id: 'goal',
        label: '/goal · Set the task objective',
        // Prefill, don't submit: the objective text still has to be typed.
        // Stays open by declaring keepOpen below.
        run: () => setNeedle('/goal '),
      },
      { id: 'compact', label: '/compact · Compact this task now', run: () => void store.submit('/compact') },
      // One entry per style the page is not in now.
      ...(() => {
        const stamped = typeof document !== 'undefined' ? document.documentElement.getAttribute('data-style') : null
        const current: StyleChoice = styleFromSaved(stamped) ?? DEFAULT_STYLE
        const choices = STYLES
        return choices.filter(style => style !== current).map(next => (
          { id: `style:${next}`, icon: 'half', label: `Switch to ${STYLE_LABELS[next]} style`, run: () => chooseStyle(next) } satisfies PaletteAction
        ))
      })(),
      { id: 'settings', icon: 'settings', label: 'Settings…', hint: mac ? '⌘,' : 'Ctrl+,', run: () => store.openSettings() },
      { id: 'models-settings', icon: 'spark', label: 'Models & Providers settings…', run: () => store.openSettings('models') },
      { id: 'permissions', icon: 'shield', label: 'Permissions settings…', run: () => store.openSettings('permissions') },
      { id: 'channels', icon: 'cloud', label: 'Channels settings…', run: () => store.openSettings('channels') },
      { id: 'terminals', icon: 'terminal', label: 'Terminals…', run: () => store.openSettings('terminals') },
      { id: 't3-terminal-drawer', icon: 'terminal', label: 'Toggle bottom terminal drawer', hint: mac ? '⌘J' : 'Ctrl+J', run: () => window.dispatchEvent(new CustomEvent('xerxes:t3-action', { detail: 'toggle-terminal-drawer' })) },
      { id: 't3-thread-details', icon: 'info', label: 'Toggle thread details card', run: () => window.dispatchEvent(new CustomEvent('xerxes:t3-action', { detail: 'toggle-thread-details' })) },
      { id: 't3-right-panel', icon: 'sidebar', label: 'Toggle right panel', hint: mac ? '⌘\\' : 'Ctrl+\\', run: () => window.dispatchEvent(new CustomEvent('xerxes:t3-action', { detail: 'toggle-right-panel' })) },
      { id: 't3-sidebar-mode', icon: 'folder', label: 'Toggle sidebar view (Inbox / Folders)', run: () => window.dispatchEvent(new CustomEvent('xerxes:t3-action', { detail: 'toggle-sidebar-view' })) },
      { id: 'session-search', icon: 'search', label: 'Search sessions & messages…', hint: mac ? '⇧⌘F' : 'Ctrl+Shift+F', run: () => store.openSessionSearch() },
    ]
    for (const provider of snap.providers) {
      if (provider.active || snap.turnActive) continue
      list.push({
        id: `provider:${provider.name}`,
        icon: 'spark',
        label: `Switch provider: ${(provider.label || provider.name)}`,
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
        // (worktree, preset, plan ceiling, model), which is ⌥⌘N (⇧⌘N is New window).
        label: 'New task…',
        hint: mac ? '⌥⌘N' : 'Ctrl+Alt+N',
        run: () => store.openTaskModal(),
      })
      list.push({
        id: 'new-blank',
        icon: 'plus',
        label: 'New blank task',
        hint: mac ? '⌘N' : 'Ctrl+N',
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
      event.stopPropagation()
      store.closePalette()
    }
  }

  return (
    <>
      <div className="backdrop backdrop--clear" onClick={() => store.closePalette()} />
      <div className="palette" role="dialog" aria-label="Command palette">
        <input
          ref={ref}
          type="search"
          className="palette__in"
          aria-label="Type a command"
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
              type="button"
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
            <div className="modelpop__status" role="status">
              {needle.startsWith('/')
                ? <>No match. Press ⏎ to send “{needle}” to the runtime.</>
                : 'No matching command.'}
            </div>
          )}
        </div>
        <div className="palette__foot">
          <span><kbd>↑↓</kbd> select</span>
          <span><kbd>⏎</kbd> run</span>
          <span><kbd>Esc</kbd> close</span>
          <span className="palette__hint">/ prefix sends a runtime slash command</span>
        </div>
      </div>
    </>
  )
}
