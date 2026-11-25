// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Unified 3-pane Provider + Model + Thinking Level + Permissions picker.
 *
 * Trigger: `[ProviderIcon] 4 Neon  Extra` (single calm composer control).
 * Popover:
 * - Left vertical provider rail (44px): ★ Favorites + Xerxes / Claude Code / Antigravity / Custom ACP
 *   with cobalt-blue active indicator pill on the right edge.
 * - Middle column (252px): `Search models...` with cobalt-blue active underline, scrollable
 *   model rows with provider glyph, active checkmark, and `⌘1`–`⌘8` shortcut badges.
 * - Right column (172px): `Thinking Level` + short model name header, reasoning / thinking levels
 *   (and any extra ACP config options or Xerxes Plan/Delegation switches).
 * - Bottom footer bar: centered Permissions / Access Mode switcher (`Full access ⌃` / `Auto ⌃` / etc.).
 */

import {
  useEffect,
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactElement,
} from 'react'

import {
  AntigravityLoginDialog,
  activeAcpConfigOptionLabel,
  activeAcpModeLabel,
  activeAcpModelLabel,
  activeAcpThoughtLevelLabel,
  extraAcpConfigOptions,
  groupAcpConfigSelectOptions,
  isAcpConfigOptionEnabled,
  isExternalAcpBinding,
  parseCustomArgs,
  resolveRuntimeList,
  runtimeSubtitle,
} from './AcpRuntimePicker.js'
import { RuntimeBrandIcon } from './BrandMark.js'
import { PickerLayer, bareModelName, DELEGATION_CHOICES } from './Overlays.js'
import { T3Icon } from './T3Icons.js'
import type { AgentBrandProvider } from './XerxesSprites.js'
import { store, type AcpExternalAgentInfo, type Snapshot } from './store.js'
import type { PermissionMode } from './types.js'

export { RuntimeBrandIcon }

const FAVORITES_KEY = 'xerxes.desktop.favoriteModels.v1'

const PERMISSION_OPTIONS: ReadonlyArray<{
  readonly id: PermissionMode
  readonly label: string
  readonly short: string
  readonly description: string
}> = [
  { id: 'manual', label: 'Ask', short: 'Ask', description: 'Confirm edits and commands' },
  { id: 'auto', label: 'Auto', short: 'Auto', description: 'Auto-approve workspace edits' },
  { id: 'accept-all', label: 'Full access', short: 'Full access', description: 'Run without prompting' },
  { id: 'plan', label: 'Plan only', short: 'Plan only', description: 'Propose plan before changes' },
]

function loadFavoriteModels(): string[] {
  try {
    if (typeof localStorage === 'undefined') return []
    const raw = localStorage.getItem(FAVORITES_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw)
    return Array.isArray(parsed) ? parsed.filter((item): item is string => typeof item === 'string') : []
  } catch {
    return []
  }
}

function saveFavoriteModels(ids: readonly string[]): void {
  try {
    if (typeof localStorage !== 'undefined') localStorage.setItem(FAVORITES_KEY, JSON.stringify(ids.slice(0, 32)))
  } catch {
    // Ignore storage quota / sandbox issues.
  }
}

/** Shortens "Gemini 4 Neon" -> "4 Neon", "Claude Sonnet 4.6" -> "Sonnet 4.6" for the compact trigger & thinking header. */
export function compactModelTriggerName(raw: string): string {
  const clean = bareModelName(raw).trim()
  if (!clean) return 'Choose model'
  const stripped = clean
    .replace(/^Gemini\s+/i, '')
    .replace(/^Claude\s+/i, '')
    .replace(/^OpenAI\s+/i, '')
    .trim()
  return stripped || clean
}

function capitalize(word: string): string {
  return word ? word[0]!.toUpperCase() + word.slice(1) : word
}

/** Human label for a reasoning effort rung (`xhigh` → `Extra high`). */
export function reasoningLevelLabel(effort: string): string {
  const clean = effort.trim().toLowerCase()
  if (clean === 'xhigh') return 'Extra high'
  if (clean === 'off') return 'Off'
  if (clean === 'on') return 'On'
  if (clean === 'minimal') return 'Minimal'
  if (clean === 'low') return 'Low'
  if (clean === 'medium') return 'Medium'
  if (clean === 'high') return 'High'
  if (clean === 'max') return 'Max'
  if (clean === 'ultra') return 'Ultra'
  return capitalize(effort.trim())
}

function permissionDisplayLabel(mode: string): string {
  if (mode === 'accept-all') return 'Full access'
  if (mode === 'auto') return 'Auto'
  if (mode === 'manual') return 'Ask'
  if (mode === 'plan') return 'Plan only'
  return mode ? capitalize(mode.replace(/[-_]/g, ' ')) : 'Full access'
}

interface UnifiedModelEntry {
  readonly key: string
  readonly id: string
  readonly name: string
  readonly runtimeId: string
  readonly provider: AgentBrandProvider
  readonly group?: string
  readonly active: boolean
}

export function T3ModelPicker({
  snap,
  defaultOpen = false,
  defaultOpenExtraConfigId = null,
}: {
  readonly snap: Snapshot
  readonly defaultOpen?: boolean
  readonly defaultOpenExtraConfigId?: string | null
}): ReactElement {
  const [open, setOpen] = useState(defaultOpen || Boolean(snap.pickerOpen))
  const [search, setSearch] = useState('')
  const [cursor, setCursor] = useState(0)
  const [selectedRail, setSelectedRail] = useState<string>('active')
  const [favorites, setFavorites] = useState<readonly string[]>(loadFavoriteModels)
  const [permissionsExpanded, setPermissionsExpanded] = useState(false)
  const [openExtraConfigId, setOpenExtraConfigId] = useState<string | null>(defaultOpenExtraConfigId)
  const [customOpen, setCustomOpen] = useState(false)
  const [customCommand, setCustomCommand] = useState('')
  const [customArgs, setCustomArgs] = useState('')
  const [agyLoginOpen, setAgyLoginOpen] = useState(false)
  const prevPickerOpenRef = useRef(Boolean(snap.pickerOpen))
  const listId = useId()

  const triggerRef = useRef<HTMLButtonElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)
  const searchRef = useRef<HTMLInputElement>(null)

  const externalBinding = isExternalAcpBinding(snap.acpBinding) ? snap.acpBinding : null
  const activeRuntimeId = externalBinding ? externalBinding.agent_id : 'xerxes'
  const activeProvider: AgentBrandProvider = externalBinding ? externalBinding.provider : 'xerxes'
  const agents = useMemo(() => resolveRuntimeList(snap.acpAgents), [snap.acpAgents])
  const online = snap.connection === 'online'
  const connecting = Boolean(snap.acpConnecting)

  // Effective selected rail tab ('favorites' | agent.id)
  const effectiveRail = selectedRail === 'active' ? activeRuntimeId : selectedRail

  // Keep native Xerxes reasoning levels synced whenever the session or model changes.
  useEffect(() => {
    if (!online || externalBinding) return
    store.loadReasoningLevels({ silent: true })
  }, [online, Boolean(externalBinding), snap.currentId, snap.model])

  // Observe store.openPicker() (e.g. Settings -> Models & Providers -> Current model)
  // so opening via store syncs the T3 picker open and closing clears snap.pickerOpen.
  useEffect(() => {
    const wasOpen = prevPickerOpenRef.current
    const isStoreOpen = Boolean(snap.pickerOpen)
    prevPickerOpenRef.current = isStoreOpen
    if (isStoreOpen && !wasOpen) {
      setSelectedRail(activeRuntimeId)
      setSearch('')
      setPermissionsExpanded(false)
      setCustomOpen(false)
      setOpen(true)
      if (online) {
        void store.refreshAcpAgents({ silent: true })
        if (!externalBinding) {
          if (snap.models.length === 0) void store.loadModels()
          store.loadReasoningLevels({ silent: snap.reasoningLevels.length > 0 })
        }
      }
    } else if (!isStoreOpen && wasOpen) {
      setOpen(false)
      setPermissionsExpanded(false)
      setCustomOpen(false)
    }
  }, [snap.pickerOpen])

  const close = (): void => {
    setOpen(false)
    setPermissionsExpanded(false)
    setCustomOpen(false)
    if (snap.pickerOpen || store.getSnapshot().pickerOpen) store.closePicker()
    triggerRef.current?.focus()
  }

  const toggle = (): void => {
    if (open) {
      close()
      return
    }
    setSelectedRail(activeRuntimeId)
    setSearch('')
    setPermissionsExpanded(false)
    setCustomOpen(false)
    setOpen(true)
    if (online) {
      void store.refreshAcpAgents({ silent: true })
      if (!externalBinding) {
        if (snap.models.length === 0) void store.loadModels()
        store.loadReasoningLevels({ silent: snap.reasoningLevels.length > 0 })
      }
    }
  }

  // Anchor popup above the trigger button aligned to its right edge (clamped to viewport).
  useLayoutEffect(() => {
    if (!open || typeof window === 'undefined') return
    const popup = popupRef.current
    const trigger = triggerRef.current
    if (!popup || !trigger) return
    const place = (): void => {
      const rect = trigger.getBoundingClientRect()
      popup.style.position = 'fixed'
      popup.style.bottom = 'auto'
      popup.style.right = 'auto'
      const width = popup.getBoundingClientRect().width
      const height = popup.getBoundingClientRect().height
      const left = Math.max(12, Math.min(rect.right - width, window.innerWidth - width - 12))
      const aboveTop = rect.top - height - 10
      const top = aboveTop >= 12 ? aboveTop : Math.max(12, window.innerHeight - height - 12)
      popup.style.left = `${left}px`
      popup.style.top = `${top}px`
    }
    place()
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(place) : null
    observer?.observe(popup)
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
    }
  }, [open, permissionsExpanded, customOpen, effectiveRail])

  useEffect(() => {
    if (!open) return
    if (!customOpen) searchRef.current?.focus()
    const onKey = (event: globalThis.KeyboardEvent): void => {
      if (event.key === 'Escape') {
        event.preventDefault()
        event.stopPropagation()
        if (openExtraConfigId !== null) setOpenExtraConfigId(null)
        else if (customOpen) setCustomOpen(false)
        else if (permissionsExpanded) setPermissionsExpanded(false)
        else close()
      }
    }
    window.addEventListener('keydown', onKey, true)
    return () => window.removeEventListener('keydown', onKey, true)
  }, [open, openExtraConfigId, customOpen, permissionsExpanded])

  // Build model rows for the currently active runtime and native Xerxes.
  const nativeEntries = useMemo<UnifiedModelEntry[]>(() => {
    const seen = new Set<string>()
    const list: UnifiedModelEntry[] = []
    if (snap.model && !seen.has(snap.model)) {
      seen.add(snap.model)
      const matched = snap.models.find(m => m.id === snap.model)
      list.push({
        key: `xerxes:${snap.model}`,
        id: snap.model,
        name: matched?.label?.trim() || bareModelName(snap.model),
        runtimeId: 'xerxes',
        provider: 'xerxes',
        group: matched?.provider || 'Xerxes',
        active: !externalBinding,
      })
    }
    for (const choice of snap.models) {
      if (seen.has(choice.id)) continue
      seen.add(choice.id)
      list.push({
        key: `xerxes:${choice.id}`,
        id: choice.id,
        name: choice.label?.trim() || bareModelName(choice.id),
        runtimeId: 'xerxes',
        provider: 'xerxes',
        group: choice.provider || 'Xerxes',
        active: !externalBinding && choice.id === snap.model,
      })
    }
    return list
  }, [snap.model, snap.models, externalBinding])

  const acpEntries = useMemo<UnifiedModelEntry[]>(() => {
    if (!externalBinding) return []
    const models = externalBinding.models ?? []
    const currentId = externalBinding.current_model ?? ''
    if (models.length === 0) {
      const fallback = activeAcpModelLabel(externalBinding)
      return [
        {
          key: `${externalBinding.agent_id}:${currentId || fallback}`,
          id: currentId || fallback,
          name: fallback,
          runtimeId: externalBinding.agent_id,
          provider: externalBinding.provider,
          active: true,
        },
      ]
    }
    return models.map(option => ({
      key: `${externalBinding.agent_id}:${option.id}`,
      id: option.id,
      name: option.name || option.id,
      runtimeId: externalBinding.agent_id,
      provider: externalBinding.provider,
      active: option.id === currentId,
    }))
  }, [externalBinding])

  const selectedRailAgent: AcpExternalAgentInfo | undefined = agents.find(agent => agent.id === effectiveRail)
  const isBrowsingDisconnectedExternal =
    effectiveRail !== 'favorites' &&
    effectiveRail !== 'xerxes' &&
    effectiveRail !== activeRuntimeId

  const visibleModels = useMemo<UnifiedModelEntry[]>(() => {
    const query = search.trim().toLowerCase()
    const matches = (entry: UnifiedModelEntry): boolean =>
      !query ||
      entry.name.toLowerCase().includes(query) ||
      entry.id.toLowerCase().includes(query) ||
      (entry.group?.toLowerCase().includes(query) ?? false)

    if (effectiveRail === 'favorites') {
      const pool = [...acpEntries, ...nativeEntries]
      const favSet = new Set(favorites)
      const pinned = pool.filter(entry => favSet.has(entry.key) || entry.active)
      const base = pinned.length > 0 ? pinned : pool.slice(0, 8)
      return base.filter(matches)
    }
    if (effectiveRail === 'xerxes') {
      return nativeEntries.filter(matches)
    }
    if (externalBinding && effectiveRail === externalBinding.agent_id) {
      return acpEntries.filter(matches)
    }
    return []
  }, [effectiveRail, acpEntries, nativeEntries, favorites, search, externalBinding])

  const pickModelEntry = async (entry: UnifiedModelEntry): Promise<void> => {
    if (snap.turnActive || connecting) return
    if (entry.runtimeId === 'xerxes') {
      if (externalBinding) {
        const ok = await store.disconnectAcpAgent()
        if (!ok) return
      }
      store.pickModel(entry.id)
      return
    }
    if (externalBinding && entry.runtimeId === externalBinding.agent_id) {
      await store.setAcpModel(entry.id)
    }
  }

  // Browse-only left rail: switching tabs inspects a provider's pane without
  // mutating, disconnecting, or auto-installing the live session runtime until
  // the user selects a model or clicks the explicit Connect / Switch CTA.
  const selectRailProvider = (railId: string): void => {
    setCustomOpen(false)
    setSelectedRail(railId)
    setSearch('')
    if (railId === 'xerxes' && online) {
      if (snap.models.length === 0) void store.loadModels()
      if (!externalBinding) store.loadReasoningLevels({ silent: snap.reasoningLevels.length > 0 })
    }
  }

  const activeCursor = visibleModels.length > 0 ? Math.max(0, Math.min(cursor, visibleModels.length - 1)) : -1

  useEffect(() => {
    const activeIdx = visibleModels.findIndex(entry => entry.active)
    setCursor(activeIdx >= 0 ? activeIdx : 0)
  }, [search, effectiveRail, open])

  useEffect(() => {
    if (!open || activeCursor < 0) return
    const row = popupRef.current?.querySelector<HTMLElement>(`[data-option-index="${activeCursor}"]`)
    row?.scrollIntoView?.({ block: 'nearest' })
  }, [open, activeCursor])

  const toggleFavorite = (entryKey: string): void => {
    setFavorites(current => {
      const next = current.includes(entryKey)
        ? current.filter(item => item !== entryKey)
        : [entryKey, ...current]
      saveFavoriteModels(next)
      return next
    })
  }

  const submitCustomAcp = async (event: FormEvent): Promise<void> => {
    event.preventDefault()
    const raw = customCommand.trim()
    if (!raw || connecting) return
    const tokens = parseCustomArgs(raw)
    const command = tokens[0] ?? raw
    const args = [...tokens.slice(1), ...parseCustomArgs(customArgs)]
    const payload = args.length > 0 ? { command, args } : { command }
    const ok = await store.connectAcpAgent('custom-acp', payload)
    if (ok) {
      setCustomOpen(false)
      setSelectedRail('custom-acp')
    }
  }

  // Trigger display strings (Image 2 bottom-right: `[Icon] 4 Neon  Extra`)
  const matchedNativeModel = !externalBinding && snap.model
    ? snap.models.find(m => m.id === snap.model)
    : undefined
  const fullModelLabel = externalBinding
    ? activeAcpModelLabel(externalBinding)
    : matchedNativeModel?.label?.trim()
      ? matchedNativeModel.label.trim()
      : snap.model
        ? bareModelName(snap.model)
        : 'Choose model'
  const shortModelLabel = compactModelTriggerName(fullModelLabel)
  const activeNativeEffort = snap.reasoningEffort || snap.reasoningDefault
  const activeThoughtLabel = externalBinding
    ? activeAcpThoughtLevelLabel(externalBinding)
    : snap.reasoningLevels.length > 0 && activeNativeEffort
      ? reasoningLevelLabel(activeNativeEffort)
      : null

  // Right column levels (ACP thought levels OR Xerxes reasoning levels)
  const thinkingLevels: ReadonlyArray<{
    readonly id: string
    readonly label: string
    readonly active: boolean
    readonly description?: string
  }> = externalBinding
    ? (externalBinding.thought_levels ?? []).map(option => ({
        id: option.id,
        label: option.name || reasoningLevelLabel(option.id),
        active: option.id === externalBinding.thought_level,
        ...(option.description ? { description: option.description } : {}),
      }))
    : snap.reasoningLevels.map(row => ({
        id: row.effort,
        label: reasoningLevelLabel(row.effort),
        active: row.effort === activeNativeEffort,
        ...(row.description ? { description: row.description } : {}),
      }))

  const extraAcpOptions = externalBinding ? extraAcpConfigOptions(externalBinding) : []

  // Footer access/mode display (Image 4 bottom bar: `[Shield] Full access ⌃`)
  const acpModes = externalBinding?.modes ?? []
  const footerLabel = externalBinding
    ? acpModes.length > 0
      ? activeAcpModeLabel(externalBinding)
      : externalBinding.name
    : permissionDisplayLabel(snap.permissionMode)

  const handlePopupKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if ((event.metaKey || event.ctrlKey) && /^[1-8]$/.test(event.key)) {
      const idx = Number.parseInt(event.key, 10) - 1
      const entry = visibleModels[idx]
      if (entry) {
        event.preventDefault()
        void pickModelEntry(entry)
      }
      return
    }
    if (customOpen || visibleModels.length === 0) return
    if (event.key === 'ArrowDown') {
      event.preventDefault()
      setCursor(current => (current + 1) % visibleModels.length)
      return
    }
    if (event.key === 'ArrowUp') {
      event.preventDefault()
      setCursor(current => (current - 1 + visibleModels.length) % visibleModels.length)
      return
    }
    if (event.key === 'Enter') {
      const target = event.target as HTMLElement | null
      if (target && target.tagName === 'BUTTON') return
      const entry = visibleModels[activeCursor]
      if (entry) {
        event.preventDefault()
        void pickModelEntry(entry)
      }
    }
  }

  return (
    <div className="chipanchor t3-model-anchor composer__runtime">
      <button
        ref={triggerRef}
        type="button"
        className={`t3-model-trigger${open ? ' is-open' : ''}${connecting ? ' is-connecting' : ''}`}
        data-t3-model-trigger="true"
        data-provider={activeProvider}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`Model: ${fullModelLabel}${activeThoughtLabel ? `, ${activeThoughtLabel}` : ''}`}
        title={`${fullModelLabel}${activeThoughtLabel ? ` · ${activeThoughtLabel}` : ''} · ${footerLabel}`}
        onClick={toggle}
      >
        <RuntimeBrandIcon provider={activeProvider} size={15} className="t3-model-trigger__glyph" />
        <span className="t3-model-trigger__name">{connecting ? 'Connecting…' : shortModelLabel}</span>
        {activeThoughtLabel && !connecting && (
          <span className="t3-model-trigger__effort">{activeThoughtLabel}</span>
        )}
      </button>

      <PickerLayer>
        {open && (
          <>
            <div className="backdrop backdrop--clear" onClick={close} />
            <div
              ref={popupRef}
              className="t3-mpop"
              data-t3-model-picker="true"
              role="dialog"
              aria-label="Model and reasoning picker"
              onKeyDown={handlePopupKeyDown}
            >
              {/* Left Vertical Provider Rail (44px) */}
              <div className="t3-mpop__rail" role="toolbar" aria-orientation="vertical" aria-label="Providers">
                <button
                  type="button"
                  className={`t3-mpop__rail-btn${effectiveRail === 'favorites' ? ' is-selected' : ''}`}
                  aria-pressed={effectiveRail === 'favorites'}
                  title="Favorites"
                  aria-label="Favorites"
                  onClick={() => selectRailProvider('favorites')}
                >
                  <T3Icon name="star" size={16} />
                </button>
                <div className="t3-mpop__rail-sep" aria-hidden="true" />
                {agents.map(agent => {
                  const selected = effectiveRail === agent.id && !customOpen
                  const connected = agent.id === activeRuntimeId
                  return (
                    <button
                      key={agent.id}
                      type="button"
                      className={`t3-mpop__rail-btn${selected ? ' is-selected' : ''}${connected ? ' is-connected' : ''}${!agent.available && agent.id !== 'xerxes' ? ' is-unavailable' : ''}`}
                      aria-pressed={selected}
                      data-provider={agent.provider}
                      title={`${agent.name} · ${runtimeSubtitle(agent)}`}
                      aria-label={agent.name}
                      onClick={() => selectRailProvider(agent.id)}
                    >
                      <RuntimeBrandIcon provider={agent.provider} size={18} />
                    </button>
                  )
                })}
                <div className="t3-mpop__rail-spacer" />
                <button
                  type="button"
                  className={`t3-mpop__rail-btn t3-mpop__rail-btn--add${customOpen ? ' is-selected' : ''}`}
                  aria-pressed={customOpen}
                  title="Connect custom ACP agent…"
                  aria-label="Custom ACP agent"
                  onClick={() => setCustomOpen(value => !value)}
                >
                  <T3Icon name="plus" size={15} />
                </button>
              </div>

              {/* Right Main Area: Columns + Centered Permission Footer */}
              <div className="t3-mpop__main">
                {customOpen ? (
                  <form className="t3-mpop__custom" onSubmit={event => void submitCustomAcp(event)}>
                    <div className="t3-mpop__col-head">
                      <span>Custom ACP runtime</span>
                      <button type="button" className="t3-mpop__link" onClick={() => setCustomOpen(false)}>Back</button>
                    </div>
                    <label className="t3-mpop__field">
                      <span>Command</span>
                      <input
                        type="text"
                        autoFocus
                        spellCheck={false}
                        placeholder="e.g. agy acp or claude-agent-acp"
                        value={customCommand}
                        onChange={event => setCustomCommand(event.target.value)}
                      />
                    </label>
                    <label className="t3-mpop__field">
                      <span>Arguments (optional)</span>
                      <input
                        type="text"
                        spellCheck={false}
                        placeholder="Extra CLI flags"
                        value={customArgs}
                        onChange={event => setCustomArgs(event.target.value)}
                      />
                    </label>
                    {snap.acpError && <p className="studio-error" role="alert">{snap.acpError}</p>}
                    <div className="t3-mpop__custom-foot">
                      <button type="submit" className="t3-mpop__primary" disabled={!customCommand.trim() || connecting}>
                        {connecting ? 'Connecting…' : 'Connect'}
                      </button>
                    </div>
                  </form>
                ) : (
                  <div className="t3-mpop__cols">
                    {/* Middle Column: Search + Models List */}
                    <div className="t3-mpop__models">
                      <label className="t3-mpop__search">
                        <T3Icon name="search" size={13} className="t3-mpop__search-ico" />
                        <input
                          ref={searchRef}
                          className="t3-mpop__search-in"
                          type="search"
                          role="combobox"
                          aria-expanded={open}
                          aria-controls={`${listId}-list`}
                          aria-autocomplete="list"
                          aria-activedescendant={activeCursor >= 0 ? `${listId}-opt-${activeCursor}` : undefined}
                          spellCheck={false}
                          placeholder="Search models..."
                          aria-label="Search models"
                          value={search}
                          onChange={event => setSearch(event.target.value)}
                        />
                        {search && (
                          <button
                            type="button"
                            className="t3-mpop__search-clear"
                            aria-label="Clear model search"
                            onClick={() => setSearch('')}
                          >
                            <T3Icon name="close" size={11} />
                          </button>
                        )}
                      </label>

                      <div className="t3-mpop__list" id={`${listId}-list`} role="listbox" aria-label="Models">
                        {connecting ? (
                          <div className="t3-mpop__empty">
                            {selectedRailAgent && !selectedRailAgent.available
                              ? `Installing & connecting to ${selectedRailAgent.name}…`
                              : `Connecting to ${selectedRailAgent?.name ?? 'runtime'}…`}
                          </div>
                        ) : isBrowsingDisconnectedExternal && selectedRailAgent ? (
                          <div className="t3-mpop__connect-box">
                            <RuntimeBrandIcon provider={selectedRailAgent.provider} size={24} />
                            <strong>{selectedRailAgent.name}</strong>
                            <p>
                              {selectedRailAgent.available
                                ? runtimeSubtitle(selectedRailAgent)
                                : selectedRailAgent.id === 'antigravity' || selectedRailAgent.id === 'claude-code'
                                  ? `${selectedRailAgent.name} CLI is not installed yet on this host. Xerxes will install it automatically when you switch.`
                                  : selectedRailAgent.installHint || 'Not installed on PATH'}
                            </p>
                            {selectedRailAgent.available ? (
                              <>
                                <button
                                  type="button"
                                  className="t3-mpop__primary"
                                  disabled={snap.turnActive}
                                  onClick={() => void store.connectAcpAgent(selectedRailAgent.id)}
                                >
                                  Switch to {selectedRailAgent.name}
                                </button>
                                {selectedRailAgent.id === 'antigravity' && (
                                  <button
                                    type="button"
                                    className="t3-mpop__secondary"
                                    disabled={snap.turnActive}
                                    onClick={() => {
                                      setOpen(false)
                                      setAgyLoginOpen(true)
                                    }}
                                  >
                                    Login with Antigravity
                                  </button>
                                )}
                              </>
                            ) : selectedRailAgent.id === 'antigravity' || selectedRailAgent.id === 'claude-code' ? (
                              <>
                                <button
                                  type="button"
                                  className="t3-mpop__primary"
                                  disabled={snap.turnActive}
                                  onClick={() => void store.connectAcpAgent(selectedRailAgent.id)}
                                >
                                  Install &amp; switch to {selectedRailAgent.name}
                                </button>
                                <button
                                  type="button"
                                  className="t3-mpop__secondary"
                                  onClick={() => {
                                    setCustomCommand(selectedRailAgent.command ?? '')
                                    setCustomArgs(selectedRailAgent.args?.join(' ') ?? '')
                                    setCustomOpen(true)
                                  }}
                                >
                                  Configure command…
                                </button>
                              </>
                            ) : (
                              <button
                                type="button"
                                className="t3-mpop__secondary"
                                onClick={() => {
                                  setCustomCommand(selectedRailAgent.command ?? '')
                                  setCustomArgs(selectedRailAgent.args?.join(' ') ?? '')
                                  setCustomOpen(true)
                                }}
                              >
                                Configure command…
                              </button>
                            )}
                          </div>
                        ) : visibleModels.length === 0 ? (
                          <div className="t3-mpop__empty">
                            {snap.modelsLoading ? 'Loading models…' : search ? 'No matching models' : 'No models discovered'}
                            {effectiveRail === 'xerxes' && Boolean(externalBinding) && !search && (
                              <button
                                type="button"
                                className="t3-mpop__primary"
                                disabled={snap.turnActive || connecting}
                                onClick={() => void store.disconnectAcpAgent()}
                              >
                                Switch to Xerxes
                              </button>
                            )}
                            {effectiveRail === 'xerxes' && !search && (
                              <button
                                type="button"
                                className="t3-mpop__link"
                                onClick={() => void store.loadModels(true)}
                              >
                                Refresh
                              </button>
                            )}
                          </div>
                        ) : (
                          visibleModels.map((entry, idx) => {
                            const starred = favorites.includes(entry.key)
                            const focused = idx === activeCursor
                            return (
                              <div key={entry.key} className="t3-mpop__row-wrap" role="none">
                                <button
                                  id={`${listId}-opt-${idx}`}
                                  type="button"
                                  role="option"
                                  aria-selected={entry.active}
                                  data-option-index={idx}
                                  data-focused={focused ? 'true' : undefined}
                                  disabled={snap.turnActive || connecting}
                                  className={`t3-mpop__row${entry.active ? ' is-active' : ''}${focused ? ' is-focused' : ''}`}
                                  title={entry.id}
                                  onMouseEnter={() => setCursor(idx)}
                                  onClick={() => void pickModelEntry(entry)}
                                >
                                  <RuntimeBrandIcon provider={entry.provider} size={14} className="t3-mpop__row-ico" />
                                  <span className="t3-mpop__row-name">{entry.name}</span>
                                  {entry.active && <T3Icon name="check" size={13} className="t3-mpop__row-check" />}
                                  {idx < 8 && <kbd className="t3-mpop__kbd">⌘{idx + 1}</kbd>}
                                </button>
                                <button
                                  type="button"
                                  className={`t3-mpop__star${starred ? ' is-starred' : ''}`}
                                  title={starred ? 'Remove from favorites' : 'Pin to favorites'}
                                  aria-label={starred ? `Unfavorite ${entry.name}` : `Favorite ${entry.name}`}
                                  onClick={event => {
                                    event.stopPropagation()
                                    toggleFavorite(entry.key)
                                  }}
                                >
                                  <T3Icon name="star" size={11} />
                                </button>
                              </div>
                            )
                          })
                        )}
                      </div>
                    </div>

                    {/* Right Column: Thinking Level + Model Options */}
                    <div className="t3-mpop__effort" data-model-picker-effort="true">
                      <div className="t3-mpop__col-head">
                        <span>Thinking Level</span>
                        <span className="t3-mpop__col-model" title={fullModelLabel}>{shortModelLabel}</span>
                      </div>

                      <div className="t3-mpop__effort-list" role="radiogroup" aria-label="Thinking level">
                        {thinkingLevels.length > 0 ? (
                          thinkingLevels.map(level => (
                            <button
                              key={level.id}
                              type="button"
                              role="radio"
                              aria-checked={level.active}
                              disabled={snap.turnActive}
                              className={`t3-mpop__effort-row${level.active ? ' is-active' : ''}`}
                              title={level.description}
                              onClick={() => {
                                if (externalBinding) void store.setAcpThoughtLevel(level.id)
                                else store.pickReasoning(level.id)
                              }}
                            >
                              <span>{level.label}</span>
                              {level.active && <T3Icon name="check" size={13} />}
                            </button>
                          ))
                        ) : (
                          <div className="t3-mpop__effort-empty">
                            {snap.reasoningLoading ? 'Loading levels…' : snap.reasoningNote || 'Default reasoning'}
                          </div>
                        )}
                      </div>

                      {/* Extra ACP options or Xerxes Plan/Agents controls neatly docked below */}
                      {externalBinding && extraAcpOptions.length > 0 && (
                        <div className="t3-mpop__extras">
                          {extraAcpOptions.map(opt => {
                            if (opt.type === 'boolean') {
                              const enabled = isAcpConfigOptionEnabled(opt)
                              return (
                                <div key={opt.id} className="t3-mpop__extra-group t3-mpop__extra-group--bool" data-config-id={opt.id}>
                                  <button
                                    type="button"
                                    role="switch"
                                    aria-checked={enabled}
                                    aria-label={opt.name}
                                    className={`t3-mpop__extra-btn t3-mpop__extra-toggle${enabled ? ' is-on' : ''}`}
                                    disabled={snap.turnActive || !online}
                                    title={opt.description || `${opt.name}: ${activeAcpConfigOptionLabel(opt)}`}
                                    onClick={() => void store.setAcpConfigOption(opt.id, !enabled)}
                                  >
                                    <span>{opt.name}</span>
                                    <span className="t3-mpop__extra-val">{activeAcpConfigOptionLabel(opt)}</span>
                                  </button>
                                </div>
                              )
                            }
                            const isMenuOpen = openExtraConfigId === opt.id
                            const grouped = groupAcpConfigSelectOptions(opt.options)
                            const hasNamedGroups = grouped.some(bucket => bucket.group !== null)
                            const activeVal = String(opt.current_value ?? '')
                            return (
                              <div key={opt.id} className="t3-mpop__extra-group" data-config-id={opt.id}>
                                <span className="t3-mpop__extra-label">{opt.name}</span>
                                <button
                                  type="button"
                                  className={`t3-mpop__extra-btn${isMenuOpen ? ' is-on' : ''}`}
                                  aria-haspopup="listbox"
                                  aria-expanded={isMenuOpen}
                                  aria-label={`${opt.name}: ${activeAcpConfigOptionLabel(opt)}`}
                                  disabled={snap.turnActive || !online}
                                  onClick={() => setOpenExtraConfigId(current => (current === opt.id ? null : opt.id))}
                                >
                                  <span>{activeAcpConfigOptionLabel(opt)}</span>
                                  <T3Icon name={isMenuOpen ? 'chevronDown' : 'chevronRight'} size={11} />
                                </button>
                                {isMenuOpen && (
                                  <div className="t3-mpop__extra-menu" role="listbox" aria-label={opt.name}>
                                    {grouped.map(bucket => (
                                      <div
                                        key={bucket.group ?? '__default__'}
                                        className="t3-mpop__extra-section"
                                        role={hasNamedGroups && bucket.group ? 'group' : undefined}
                                        aria-label={hasNamedGroups && bucket.group ? bucket.group : undefined}
                                      >
                                        {hasNamedGroups && bucket.group && (
                                          <div className="t3-mpop__extra-group-head" role="presentation">{bucket.group}</div>
                                        )}
                                        {bucket.options.map(choice => {
                                          const selected = choice.value === activeVal
                                          return (
                                            <button
                                              key={choice.value}
                                              type="button"
                                              role="option"
                                              aria-selected={selected}
                                              className={`t3-mpop__effort-row${selected ? ' is-active' : ''}`}
                                              title={choice.description || choice.value}
                                              disabled={snap.turnActive || !online}
                                              onClick={() => {
                                                setOpenExtraConfigId(null)
                                                void store.setAcpConfigOption(opt.id, choice.value)
                                              }}
                                            >
                                              <span>{choice.name}</span>
                                              {selected && <T3Icon name="check" size={12} />}
                                            </button>
                                          )
                                        })}
                                      </div>
                                    ))}
                                  </div>
                                )}
                              </div>
                            )
                          })}
                        </div>
                      )}

                      {!externalBinding && (
                        <div className="t3-mpop__extras">
                          <button
                            type="button"
                            className={`t3-mpop__extra-btn${snap.planMode ? ' is-on' : ''}`}
                            aria-pressed={snap.planMode}
                            onClick={() => store.togglePlanMode()}
                          >
                            <span>Plan first</span>
                            {snap.planMode && <T3Icon name="check" size={12} />}
                          </button>
                          <button
                            type="button"
                            className="t3-mpop__extra-btn"
                            title="Cycle multi-agent delegation eagerness"
                            onClick={() => {
                              const idx = DELEGATION_CHOICES.findIndex(choice => choice.mode === snap.delegationMode)
                              const next = DELEGATION_CHOICES[(idx + 1) % DELEGATION_CHOICES.length]!
                              void store.setDelegationMode(next.mode)
                            }}
                          >
                            <span>Agents</span>
                            <span className="t3-mpop__extra-val">{capitalize(snap.delegationMode)}</span>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                )}

                {/* Expanded Permission / Mode Strip */}
                {permissionsExpanded && (
                  <div className="t3-mpop__perms" role="radiogroup" aria-label="Permissions">
                    {externalBinding && acpModes.length > 0
                      ? acpModes.map(mode => {
                          const active = mode.id === externalBinding.current_mode
                          return (
                            <button
                              key={mode.id}
                              type="button"
                              role="radio"
                              aria-checked={active}
                              className={`t3-mpop__perm-pill${active ? ' is-on' : ''}`}
                              onClick={() => {
                                void store.setAcpMode(mode.id)
                                setPermissionsExpanded(false)
                              }}
                            >
                              {mode.name}
                            </button>
                          )
                        })
                      : PERMISSION_OPTIONS.map(option => {
                          const active = snap.permissionMode === option.id
                          return (
                            <button
                              key={option.id}
                              type="button"
                              role="radio"
                              aria-checked={active}
                              title={option.description}
                              className={`t3-mpop__perm-pill${active ? ' is-on' : ''}`}
                              onClick={() => {
                                store.setPermissionMode(option.id)
                                setPermissionsExpanded(false)
                              }}
                            >
                              {option.short}
                            </button>
                          )
                        })}
                  </div>
                )}

                {/* Bottom Footer Bar: Centered Permissions / Access Mode Trigger (Image 4) */}
                <div className="t3-mpop__foot">
                  <button
                    type="button"
                    className={`t3-mpop__foot-btn${permissionsExpanded ? ' is-open' : ''}`}
                    aria-expanded={permissionsExpanded}
                    onClick={() => setPermissionsExpanded(value => !value)}
                  >
                    <T3Icon name="shieldCheck" size={13} />
                    <span>{footerLabel}</span>
                    <T3Icon name={permissionsExpanded ? 'chevronDown' : 'chevronUp'} size={12} />
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
      </PickerLayer>
      <AntigravityLoginDialog
        open={agyLoginOpen}
        onClose={() => setAgyLoginOpen(false)}
        onSuccess={() => store.connectAcpAgent('antigravity')}
      />
    </div>
  )
}
