// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * T3-Code-style External Agent Runtime Switcher for Xerxes Desktop & VS Code.
 *
 * Lives right beside the model picker in the composer toolbar so users can
 * switch any thread seamlessly between native Xerxes, Claude Code (ACP /
 * stream-json), Google Antigravity (ACP), or any custom stdio ACP agent.
 */

import {
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type FormEvent,
  type KeyboardEvent,
  type ReactElement,
  type RefObject,
} from 'react'

import { ActionDialog } from './ActionDialog.js'
import { desktopError } from './desktopRpc.js'
import { Icon } from './Icon.js'
import { PickerLayer } from './Overlays.js'
import type { AgyCredentialMode, AgyStatusView } from './ProviderAccountsCenter.js'
import {
  store,
  type AcpConfigOption,
  type AcpExternalAgentInfo,
  type AcpExternalModeOption,
  type AcpExternalModelOption,
  type AcpExternalThoughtLevelOption,
  type AcpSessionBinding,
  type Snapshot,
} from './store.js'
import { RuntimeBrandIcon } from './BrandMark.js'
import type { AgentBrandProvider } from './XerxesSprites.js'

export const DEFAULT_ACP_AGENTS: readonly AcpExternalAgentInfo[] = [
  {
    id: 'xerxes',
    name: 'Xerxes',
    provider: 'xerxes',
    transport: 'native',
    available: true,
    description: 'Native multi-agent runtime & Immortals',
  },
  {
    id: 'claude-code',
    name: 'Claude Code',
    provider: 'claude-code',
    transport: 'claude-stream-json',
    available: false,
    installHint: 'Install Claude Code (`claude`) or `@zed-industries/claude-agent-acp`',
    description: 'Anthropic Claude Code via ACP / stream-json',
  },
  {
    id: 'antigravity',
    name: 'Antigravity',
    provider: 'antigravity',
    transport: 'acp',
    available: false,
    installHint: 'Install Antigravity CLI (`agy`) so `agy acp` is on PATH, or set XERXES_ANTIGRAVITY_ACP_CMD',
    description: 'Google Antigravity via ACP stdio',
  },
]

/** Merge discovered agents from `acp.agents` with the core Xerxes / Claude Code / Antigravity catalog. */
export function resolveRuntimeList(
  discovered: readonly AcpExternalAgentInfo[] | undefined,
): readonly AcpExternalAgentInfo[] {
  if (!discovered || discovered.length === 0) return DEFAULT_ACP_AGENTS
  const byId = new Map<string, AcpExternalAgentInfo>()
  for (const agent of discovered) {
    byId.set(agent.id, agent)
  }
  const ordered: AcpExternalAgentInfo[] = []
  for (const core of DEFAULT_ACP_AGENTS) {
    ordered.push(byId.get(core.id) ?? core)
    byId.delete(core.id)
  }
  for (const extra of byId.values()) {
    ordered.push(extra)
  }
  return ordered
}

export function transportBadgeLabel(transport: AcpExternalAgentInfo['transport']): string {
  if (transport === 'native') return 'Native'
  if (transport === 'claude-stream-json') return 'stream-json'
  return 'ACP'
}

/** Calm 1-line subtitle for each runtime row (zero raw backticks or transport jargon). */
export function runtimeSubtitle(agent: AcpExternalAgentInfo): string {
  if (agent.id === 'xerxes') return 'Built-in runtime'
  if (!agent.available) return 'Not installed'
  if (agent.id === 'claude-code') return 'Claude Code CLI'
  if (agent.id === 'antigravity') return 'Antigravity CLI'
  if (agent.id === 'codex') return 'Codex CLI'
  const cmd = agent.command?.trim()
  if (cmd) {
    const base = cmd.split(/[\\/]/).pop() || cmd
    return `${base} CLI`
  }
  return 'ACP runtime'
}

/** Hover tooltip carrying binary path, version, or install hint without cluttering the menu. */
export function runtimeTooltip(agent: AcpExternalAgentInfo): string {
  if (agent.id === 'xerxes') return 'Xerxes · Built-in runtime'
  if (agent.available) {
    const pathText = agent.binaryPath || agent.command || ''
    return [agent.name, agent.version ? `v${agent.version}` : '', pathText].filter(Boolean).join(' · ')
  }
  const cleanHint = agent.installHint?.replace(/`/g, '').trim()
  return cleanHint ? `${agent.name} · Not installed — ${cleanHint}` : `${agent.name} · Not installed`
}

/** Split an optional space-separated argument string, preserving simple single/double quoted tokens. */
export function parseCustomArgs(raw: string): string[] {
  const trimmed = raw.trim()
  if (!trimmed) return []
  const args: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  for (let i = 0; i < trimmed.length; i += 1) {
    const ch = trimmed[i]!
    if (quote) {
      if (ch === quote) quote = null
      else current += ch
      continue
    }
    if (ch === '"' || ch === "'") {
      quote = ch
      continue
    }
    if (/\s/.test(ch)) {
      if (current) {
        args.push(current)
        current = ''
      }
      continue
    }
    current += ch
  }
  if (current) args.push(current)
  return args
}

export function isExternalAcpBinding(
  binding: AcpSessionBinding | null | undefined,
): binding is AcpSessionBinding {
  return Boolean(binding?.connected && binding.transport !== 'native')
}

export function activeAcpModelKnown(binding: AcpSessionBinding): string | null {
  const fromList = binding.current_model
    ? binding.models?.find(option => option.id === binding.current_model)?.name
    : undefined
  const label = (binding.current_model_name || fromList || binding.current_model || '').trim()
  return label || null
}

export function activeAcpModelLabel(binding: AcpSessionBinding): string {
  return activeAcpModelKnown(binding) ?? `${binding.name} default`
}

export function activeAcpModeLabel(binding: AcpSessionBinding): string {
  const fromList = binding.current_mode
    ? binding.modes?.find(option => option.id === binding.current_mode)?.name
    : undefined
  return (binding.current_mode_name || fromList || binding.current_mode || binding.modes?.[0]?.name || 'Mode').trim()
}

export function activeAcpThoughtLevelLabel(binding: AcpSessionBinding): string | null {
  const levels = binding.thought_levels ?? []
  if (levels.length === 0) return null
  const activeId = binding.thought_level
  const fromList = activeId ? levels.find(option => option.id === activeId)?.name : undefined
  const label = (fromList || activeId || levels[0]?.name || '').trim()
  return label || null
}

/** Extra select/boolean configOptions advertised by an ACP agent beyond Model, Thinking Level, and Mode. */
export function extraAcpConfigOptions(binding: AcpSessionBinding): readonly AcpConfigOption[] {
  const options = binding.config_options ?? []
  if (options.length === 0) return []
  const modelId = binding.model_config_id ?? 'model'
  const thoughtId = binding.thought_level_config_id ?? 'thought_level'
  const modeId = binding.mode_config_id ?? 'mode'
  const extras: AcpConfigOption[] = []
  for (const opt of options) {
    const id = opt.id.trim().toLowerCase()
    const category = opt.category?.trim().toLowerCase() ?? ''
    if (id === modelId.toLowerCase() || id === 'model' || category === 'model') continue
    if (id === thoughtId.toLowerCase() || id === 'thought_level' || category === 'thought_level') continue
    if (id === modeId.toLowerCase() || id === 'mode' || category === 'mode') continue
    if (opt.type === 'select' && opt.options.length > 0) {
      extras.push(opt)
    } else if (opt.type === 'boolean') {
      extras.push(
        opt.options.length > 0
          ? opt
          : {
              ...opt,
              options: [
                { value: 'true', name: 'On' },
                { value: 'false', name: 'Off' },
              ],
            },
      )
    }
  }
  return extras
}

export function activeAcpConfigOptionLabel(option: AcpConfigOption): string {
  const matched = option.current_value
    ? option.options.find(item => item.value === option.current_value)?.name
    : undefined
  return (matched || option.current_value || option.options[0]?.name || option.name).trim()
}

/** Ordered active ACP metadata parts (`model`, `thinking`, `mode`, extra config options) for tooltips and VS Code pills. */
export function activeAcpSummaryParts(binding: AcpSessionBinding): readonly string[] {
  const parts: string[] = []
  const model = activeAcpModelKnown(binding)
  const thinking = activeAcpThoughtLevelLabel(binding)
  const mode = binding.modes?.length ? activeAcpModeLabel(binding) : null
  if (model) parts.push(model)
  if (thinking) parts.push(thinking)
  if (mode) parts.push(mode)
  for (const extra of extraAcpConfigOptions(binding)) {
    const label = activeAcpConfigOptionLabel(extra)
    if (label) parts.push(label)
  }
  return parts
}

/** Only show a secondary raw ID when it conveys information beyond a slugified form of the label. */
function shouldShowSecondaryId(name: string, id: string): boolean {
  const cleanName = name.trim()
  const cleanId = id.trim()
  if (!cleanId || cleanId === cleanName) return false
  const norm = (value: string): string => value.toLowerCase().replace(/[^a-z0-9]+/g, '')
  return norm(cleanName) !== norm(cleanId)
}

export interface EscapeKeyTarget {
  addEventListener(type: 'keydown', listener: (event: globalThis.KeyboardEvent) => void, capture?: boolean): void
  removeEventListener(type: 'keydown', listener: (event: globalThis.KeyboardEvent) => void, capture?: boolean): void
}

/**
 * Attach a capture-phase Escape listener so open ACP popovers close reliably
 * even when focus moved outside the popup (and without arming global Escape stop).
 */
export function attachEscapeCloseListener(
  target: EscapeKeyTarget,
  onClose: () => void,
): () => void {
  const onKeyDown = (event: globalThis.KeyboardEvent): void => {
    if (event.key !== 'Escape') return
    event.preventDefault()
    event.stopPropagation()
    event.stopImmediatePropagation?.()
    onClose()
  }
  target.addEventListener('keydown', onKeyDown, true)
  return () => target.removeEventListener('keydown', onKeyDown, true)
}

function useEscapeClose(open: boolean, onClose: () => void): void {
  const latestClose = useRef(onClose)
  latestClose.current = onClose
  useEffect(() => {
    if (!open || typeof window === 'undefined') return
    return attachEscapeCloseListener(window, () => latestClose.current())
  }, [open])
}

function useAnchoredRuntimePopup(
  open: boolean,
  triggerRef: RefObject<HTMLButtonElement | null>,
  popupRef: RefObject<HTMLDivElement | null>,
  align: 'left' | 'right' = 'left',
): void {
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
      popup.style.maxWidth = 'calc(100vw - 24px)'
      popup.style.maxHeight = `${Math.max(120, window.innerHeight - 24)}px`
      const width = popup.getBoundingClientRect().width
      const height = popup.getBoundingClientRect().height
      const preferredLeft = align === 'right' ? rect.right - width : rect.left
      const left = Math.max(12, Math.min(preferredLeft, window.innerWidth - width - 12))
      const aboveTop = rect.top - height - 8
      const top =
        aboveTop >= 12
          ? aboveTop
          : Math.max(12, Math.min(rect.bottom + 8, window.innerHeight - height - 12))
      popup.style.left = `${left}px`
      popup.style.top = `${top}px`
    }
    place()
    const observer = typeof ResizeObserver !== 'undefined' ? new ResizeObserver(place) : null
    observer?.observe(popup)
    observer?.observe(trigger)
    window.addEventListener('resize', place)
    window.addEventListener('scroll', place, true)
    return () => {
      observer?.disconnect()
      window.removeEventListener('resize', place)
      window.removeEventListener('scroll', place, true)
    }
  }, [open, triggerRef, popupRef, align])
}

export interface AcpRuntimePickerProps {
  readonly snap: Snapshot
  readonly compact?: boolean
  readonly defaultOpen?: boolean
  readonly onConnect?: (
    agentId: string,
    custom?: { readonly command: string; readonly args?: readonly string[] },
  ) => Promise<boolean> | boolean
  readonly onDisconnect?: () => Promise<boolean> | boolean
  readonly onRefresh?: () => Promise<void> | void
}

export function AcpRuntimePicker({
  snap,
  compact = false,
  defaultOpen = false,
  onConnect,
  onDisconnect,
  onRefresh,
}: AcpRuntimePickerProps): ReactElement {
  const [open, setOpen] = useState(defaultOpen)
  const [customOpen, setCustomOpen] = useState(false)
  const [customAgentId, setCustomAgentId] = useState('custom-acp')
  const [customCommand, setCustomCommand] = useState('')
  const [customArgs, setCustomArgs] = useState('')
  const triggerRef = useRef<HTMLButtonElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)
  const commandInputRef = useRef<HTMLInputElement>(null)

  const binding: AcpSessionBinding | null = isExternalAcpBinding(snap.acpBinding) ? snap.acpBinding : null
  const activeProvider: AgentBrandProvider = binding ? binding.provider : 'xerxes'
  const activeId = binding ? binding.agent_id : 'xerxes'
  const activeLabel = binding ? binding.name : 'Xerxes'
  const connecting = Boolean(snap.acpConnecting)
  const online = snap.connection === 'online'
  const agents = useMemo(() => resolveRuntimeList(snap.acpAgents), [snap.acpAgents])

  useAnchoredRuntimePopup(open, triggerRef, popupRef)

  useEffect(() => {
    if (!open) return
    if (customOpen) commandInputRef.current?.focus()
    else popupRef.current?.focus()
  }, [open, customOpen])

  const close = (): void => {
    setOpen(false)
    setCustomOpen(false)
    triggerRef.current?.focus()
  }

  useEscapeClose(open, () => {
    if (customOpen) setCustomOpen(false)
    else close()
  })

  const toggle = (): void => {
    if (open) {
      close()
      return
    }
    setOpen(true)
    if (online) {
      void Promise.resolve(onRefresh ? onRefresh() : store.refreshAcpAgents({ silent: true }))
    }
  }

  const selectAgent = async (agent: AcpExternalAgentInfo): Promise<void> => {
    if (connecting) return
    if (agent.id === 'xerxes') {
      if (!binding) {
        close()
        return
      }
      const ok = await (onDisconnect ? onDisconnect() : store.disconnectAcpAgent())
      if (ok) close()
      return
    }
    if (!agent.available && agent.id !== 'antigravity' && agent.id !== 'claude-code') {
      openCustomFor(agent)
      return
    }
    if (binding?.agent_id === agent.id) {
      close()
      return
    }
    const ok = await (onConnect ? onConnect(agent.id) : store.connectAcpAgent(agent.id))
    if (ok) close()
  }

  const openCustomFor = (agent?: AcpExternalAgentInfo): void => {
    if (agent && agent.id !== 'xerxes') {
      setCustomAgentId(agent.id)
      setCustomCommand(agent.command ?? (agent.id === 'antigravity' ? 'agy' : agent.id === 'claude-code' ? 'claude-agent-acp' : ''))
      setCustomArgs(agent.args?.join(' ') ?? (agent.id === 'antigravity' ? 'acp' : ''))
    } else {
      setCustomAgentId('custom-acp')
      setCustomCommand('')
      setCustomArgs('')
    }
    setCustomOpen(true)
  }

  const submitCustom = async (event: FormEvent): Promise<void> => {
    event.preventDefault()
    const rawCommand = customCommand.trim()
    if (!rawCommand || connecting) return
    const tokens = parseCustomArgs(rawCommand)
    const command = tokens[0] ?? rawCommand
    const inlineArgs = tokens.slice(1)
    const extraArgs = parseCustomArgs(customArgs)
    const args = [...inlineArgs, ...extraArgs]
    const targetId = customAgentId.trim() || 'custom-acp'
    const payload = args.length > 0 ? { command, args } : { command }
    const ok = await (onConnect ? onConnect(targetId, payload) : store.connectAcpAgent(targetId, payload))
    if (ok) close()
  }

  const onPopupKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      if (customOpen) setCustomOpen(false)
      else close()
    }
  }

  return (
    <div className={`chipanchor composer__runtime${compact ? ' composer__runtime--compact' : ''}`}>
      <button
        ref={triggerRef}
        type="button"
        className={`cchip composer__text acp-chip${binding ? ' is-external' : ''}${connecting ? ' is-connecting' : ''}${open ? ' is-open' : ''}`}
        data-provider={activeProvider}
        data-connected={binding ? 'true' : undefined}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`Agent runtime: ${activeLabel}${binding ? ` (${transportBadgeLabel(binding.transport)})` : ' (Native)'}`}
        title={
          binding
            ? `Running via ${binding.name} (${transportBadgeLabel(binding.transport)}) — click to switch runtime`
            : 'Agent runtime: Xerxes (Native) — click to switch to Claude Code, Antigravity, or custom ACP'
        }
        onClick={toggle}
      >
        <RuntimeBrandIcon
          provider={activeProvider}
          size={14}
          className="acp-chip__glyph"
        />
        <span className="acp-chip__label">{connecting ? 'Connecting…' : activeLabel}</span>
        {binding && <span className="acp-chip__live" aria-hidden="true" />}
        <Icon name="caretDown" size={10} />
      </button>

      <PickerLayer>
        {open && (
          <>
            <div className="backdrop backdrop--clear" onClick={close} />
            <div
              ref={popupRef}
              className={`acp-pop${customOpen ? ' is-custom-open' : ''}`}
              role="dialog"
              aria-label="Agent runtime"
              tabIndex={-1}
              onKeyDown={onPopupKeyDown}
            >
              {snap.acpError && (
                <div className="acp-pop__error" role="alert">
                  <span>{snap.acpError}</span>
                  <button
                    type="button"
                    aria-label="Dismiss runtime error"
                    onClick={() => store.clearAcpError()}
                  >
                    <Icon name="close" size={11} />
                  </button>
                </div>
              )}

              <div className="acp-pop__list" role="listbox" aria-label="Available agent runtimes">
                {agents.map(agent => {
                  const isActive = activeId === agent.id
                  return (
                    <div
                      key={agent.id}
                      className={`acp-row${isActive ? ' is-active' : ''}${!agent.available ? ' is-missing' : ''}`}
                    >
                      <button
                        type="button"
                        role="option"
                        aria-selected={isActive}
                        title={runtimeTooltip(agent)}
                        disabled={connecting || (!online && agent.id !== 'xerxes')}
                        className="acp-row__btn"
                        onClick={() => void selectAgent(agent)}
                      >
                        <span className="acp-row__glyph" data-provider={agent.provider}>
                          <RuntimeBrandIcon provider={agent.provider} size={16} />
                        </span>
                        <span className="acp-row__body">
                          <span className="acp-row__name">{agent.name}</span>
                          <span className="acp-row__sub">{runtimeSubtitle(agent)}</span>
                        </span>
                        <span className="acp-row__check" aria-hidden="true">
                          {isActive ? <Icon name="check" size={13} /> : null}
                        </span>
                      </button>
                    </div>
                  )
                })}
              </div>

              <div className="acp-pop__custom">
                {!customOpen ? (
                  <button
                    type="button"
                    className="acp-custom__toggle"
                    disabled={!online || connecting}
                    onClick={() => openCustomFor()}
                  >
                    <span className="acp-row__glyph" data-provider="acp">
                      <RuntimeBrandIcon provider="acp" size={16} />
                    </span>
                    <span>Custom ACP…</span>
                  </button>
                ) : (
                  <form className="acp-custom__form" onSubmit={event => void submitCustom(event)}>
                    <div className="acp-custom__head">
                      <RuntimeBrandIcon provider="acp" size={14} />
                      <strong>Connect custom ACP</strong>
                    </div>
                    <input
                      ref={commandInputRef}
                      type="text"
                      className="acp-custom__input"
                      aria-label="ACP command"
                      placeholder="Command (e.g. agy, hermes-acp, codex)"
                      spellCheck={false}
                      value={customCommand}
                      onChange={event => setCustomCommand(event.target.value)}
                    />
                    <input
                      type="text"
                      className="acp-custom__input"
                      aria-label="ACP arguments"
                      placeholder="Args (optional, e.g. acp --stdio)"
                      spellCheck={false}
                      value={customArgs}
                      onChange={event => setCustomArgs(event.target.value)}
                    />
                    <div className="acp-custom__actions">
                      <button
                        type="button"
                        className="acp-custom__cancel"
                        onClick={() => setCustomOpen(false)}
                      >
                        Cancel
                      </button>
                      <button
                        type="submit"
                        className="acp-custom__submit"
                        disabled={!online || connecting || !customCommand.trim()}
                      >
                        {connecting ? 'Connecting…' : 'Connect'}
                      </button>
                    </div>
                  </form>
                )}
              </div>
            </div>
          </>
        )}
      </PickerLayer>
    </div>
  )
}

export interface AcpModelPickerProps {
  readonly snap: Snapshot
  readonly compact?: boolean
  readonly defaultOpen?: boolean
  readonly onSelectModel?: (modelId: string) => Promise<boolean> | boolean
}

export function AcpModelPicker({
  snap,
  compact = false,
  defaultOpen = false,
  onSelectModel,
}: AcpModelPickerProps): ReactElement | null {
  const binding = isExternalAcpBinding(snap.acpBinding) ? snap.acpBinding : null
  const [open, setOpen] = useState(defaultOpen)
  const [query, setQuery] = useState('')
  const [pending, setPending] = useState(false)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)
  const filterRef = useRef<HTMLInputElement>(null)

  const models: readonly AcpExternalModelOption[] = binding?.models ?? []
  const hasModels = models.length > 0
  const showFilter = models.length > 6
  const online = snap.connection === 'online'
  const busy = pending || Boolean(snap.acpConnecting)

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return models
    return models.filter(
      option =>
        option.name.toLowerCase().includes(needle) ||
        option.id.toLowerCase().includes(needle) ||
        (option.description?.toLowerCase().includes(needle) ?? false),
    )
  }, [models, query])

  useAnchoredRuntimePopup(open && hasModels, triggerRef, popupRef, 'right')

  useEffect(() => {
    if (!open || !hasModels) return
    if (showFilter) filterRef.current?.focus()
    else popupRef.current?.focus()
  }, [open, hasModels, showFilter])

  useEffect(() => {
    if (!hasModels && open) setOpen(false)
  }, [hasModels, open])

  const close = (): void => {
    setOpen(false)
    setQuery('')
    triggerRef.current?.focus()
  }

  useEscapeClose(open && hasModels, close)

  if (!binding) return null

  const label = activeAcpModelLabel(binding)
  const activeId = binding.current_model ?? ''

  if (!hasModels) {
    return (
      <div className={`chipanchor composer__model composer__acp-model${compact ? ' composer__acp-model--compact' : ''}`}>
        <span
          className="cchip composer__text acp-model-chip acp-model-chip--static"
          data-provider={binding.provider}
          title={`${binding.name} manages model selection for this ACP session`}
          aria-label={`${binding.name} model: ${label}`}
        >
          <span className="acp-model-chip__label">{label}</span>
        </span>
      </div>
    )
  }

  const toggle = (): void => {
    if (open) close()
    else setOpen(true)
  }

  const pickModel = async (option: AcpExternalModelOption): Promise<void> => {
    if (busy || !online) return
    if (option.id === activeId) {
      close()
      return
    }
    setPending(true)
    try {
      const ok = await (onSelectModel ? onSelectModel(option.id) : store.setAcpModel(option.id))
      if (ok) close()
    } finally {
      setPending(false)
    }
  }

  const onPopupKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      close()
    }
  }

  return (
    <div className={`chipanchor composer__model composer__acp-model${compact ? ' composer__acp-model--compact' : ''}`}>
      <button
        ref={triggerRef}
        type="button"
        className={`cchip composer__text acp-model-chip${open ? ' is-open' : ''}`}
        data-provider={binding.provider}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`${binding.name} model: ${label}`}
        title={`${binding.name} model: ${label} — click to switch`}
        disabled={busy}
        onClick={toggle}
      >
        <span className="acp-model-chip__label">{pending ? 'Switching…' : label}</span>
        <Icon name="caretDown" size={10} />
      </button>

      <PickerLayer>
        {open && (
          <>
            <div className="backdrop backdrop--clear" onClick={close} />
            <div
              ref={popupRef}
              className="acp-optpop acp-modelpop"
              role="dialog"
              aria-label={`${binding.name} models`}
              tabIndex={-1}
              onKeyDown={onPopupKeyDown}
            >
              {snap.acpError && (
                <div className="acp-pop__error" role="alert">
                  <span>{snap.acpError}</span>
                  <button
                    type="button"
                    aria-label="Dismiss runtime error"
                    onClick={() => store.clearAcpError()}
                  >
                    <Icon name="close" size={11} />
                  </button>
                </div>
              )}

              {showFilter && (
                <label className="acp-optpop__search">
                  <Icon name="search" size={13} />
                  <input
                    ref={filterRef}
                    type="text"
                    className="acp-optpop__filter"
                    aria-label={`Filter ${binding.name} models`}
                    placeholder="Filter models…"
                    spellCheck={false}
                    value={query}
                    onChange={event => setQuery(event.target.value)}
                  />
                </label>
              )}

              <div className="acp-optpop__list" role="listbox" aria-label={`${binding.name} models`}>
                {filtered.length === 0 ? (
                  <div className="acp-optpop__empty">No matching models</div>
                ) : (
                  filtered.map(option => {
                    const isActive = option.id === activeId
                    const showId = !option.description && shouldShowSecondaryId(option.name, option.id)
                    return (
                      <button
                        key={option.id}
                        type="button"
                        role="option"
                        aria-selected={isActive}
                        title={option.id}
                        disabled={busy || !online}
                        className={`acp-opt-row${isActive ? ' is-active' : ''}`}
                        onClick={() => void pickModel(option)}
                      >
                        <span className="acp-opt-row__body">
                          <span className="acp-opt-row__name">{option.name}</span>
                          {option.description ? (
                            <span className="acp-opt-row__desc">{option.description}</span>
                          ) : showId ? (
                            <span className="acp-opt-row__id">{option.id}</span>
                          ) : null}
                        </span>
                        <span className="acp-opt-row__check" aria-hidden="true">
                          {isActive ? <Icon name="check" size={13} /> : null}
                        </span>
                      </button>
                    )
                  })
                )}
              </div>
            </div>
          </>
        )}
      </PickerLayer>
    </div>
  )
}

export interface AcpThinkingPickerProps {
  readonly snap: Snapshot
  readonly compact?: boolean
  readonly defaultOpen?: boolean
  readonly onSelectLevel?: (levelId: string, configId: string) => Promise<boolean> | boolean
}

export function AcpThinkingPicker({
  snap,
  compact = false,
  defaultOpen = false,
  onSelectLevel,
}: AcpThinkingPickerProps): ReactElement | null {
  const binding = isExternalAcpBinding(snap.acpBinding) ? snap.acpBinding : null
  const [open, setOpen] = useState(defaultOpen)
  const [pending, setPending] = useState(false)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)

  const levels: readonly AcpExternalThoughtLevelOption[] = binding?.thought_levels ?? []
  const hasLevels = levels.length > 0
  const configId = binding?.thought_level_config_id ?? 'thought_level'
  const online = snap.connection === 'online'
  const busy = pending || Boolean(snap.acpConnecting)

  useAnchoredRuntimePopup(open && hasLevels, triggerRef, popupRef, 'right')

  useEffect(() => {
    if (!open || !hasLevels) return
    popupRef.current?.focus()
  }, [open, hasLevels])

  useEffect(() => {
    if (!hasLevels && open) setOpen(false)
  }, [hasLevels, open])

  const close = (): void => {
    setOpen(false)
    triggerRef.current?.focus()
  }

  useEscapeClose(open && hasLevels, close)

  if (!binding || !hasLevels) return null

  const label = activeAcpThoughtLevelLabel(binding) ?? levels[0]?.name ?? 'Thinking'
  const activeId = binding.thought_level ?? ''

  const toggle = (): void => {
    if (open) close()
    else setOpen(true)
  }

  const pickLevel = async (option: AcpExternalThoughtLevelOption): Promise<void> => {
    if (busy || !online) return
    if (option.id === activeId) {
      close()
      return
    }
    setPending(true)
    try {
      const ok = await (
        onSelectLevel
          ? onSelectLevel(option.id, configId)
          : store.setAcpConfigOption(configId, option.id)
      )
      if (ok) close()
    } finally {
      setPending(false)
    }
  }

  const onPopupKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      close()
    }
  }

  return (
    <div className={`chipanchor composer__acp-thinking${compact ? ' composer__acp-thinking--compact' : ''}`}>
      <button
        ref={triggerRef}
        type="button"
        className={`cchip composer__text acp-thinking-chip${open ? ' is-open' : ''}`}
        data-provider={binding.provider}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`${binding.name} thinking level: ${label}`}
        title={`${binding.name} thinking level: ${label} — click to switch`}
        disabled={busy}
        onClick={toggle}
      >
        <span className="acp-thinking-chip__label">{pending ? 'Switching…' : label}</span>
        <Icon name="caretDown" size={10} />
      </button>

      <PickerLayer>
        {open && (
          <>
            <div className="backdrop backdrop--clear" onClick={close} />
            <div
              ref={popupRef}
              className="acp-optpop acp-thinkingpop"
              role="dialog"
              aria-label={`${binding.name} thinking level`}
              tabIndex={-1}
              onKeyDown={onPopupKeyDown}
            >
              {snap.acpError && (
                <div className="acp-pop__error" role="alert">
                  <span>{snap.acpError}</span>
                  <button
                    type="button"
                    aria-label="Dismiss runtime error"
                    onClick={() => store.clearAcpError()}
                  >
                    <Icon name="close" size={11} />
                  </button>
                </div>
              )}

              <div className="acp-optpop__list" role="listbox" aria-label={`${binding.name} thinking levels`}>
                {levels.map(option => {
                  const isActive = option.id === activeId
                  return (
                    <button
                      key={option.id}
                      type="button"
                      role="option"
                      aria-selected={isActive}
                      disabled={busy || !online}
                      className={`acp-opt-row${isActive ? ' is-active' : ''}`}
                      onClick={() => void pickLevel(option)}
                    >
                      <span className="acp-opt-row__body">
                        <span className="acp-opt-row__name">{option.name}</span>
                        {option.description ? (
                          <span className="acp-opt-row__desc">{option.description}</span>
                        ) : null}
                      </span>
                      <span className="acp-opt-row__check" aria-hidden="true">
                        {isActive ? <Icon name="check" size={13} /> : null}
                      </span>
                    </button>
                  )
                })}
              </div>
            </div>
          </>
        )}
      </PickerLayer>
    </div>
  )
}

export interface AcpModePickerProps {
  readonly snap: Snapshot
  readonly compact?: boolean
  readonly defaultOpen?: boolean
  readonly onSelectMode?: (modeId: string) => Promise<boolean> | boolean
}

export function AcpModePicker({
  snap,
  compact = false,
  defaultOpen = false,
  onSelectMode,
}: AcpModePickerProps): ReactElement | null {
  const binding = isExternalAcpBinding(snap.acpBinding) ? snap.acpBinding : null
  const [open, setOpen] = useState(defaultOpen)
  const [query, setQuery] = useState('')
  const [pending, setPending] = useState(false)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)
  const filterRef = useRef<HTMLInputElement>(null)

  const modes: readonly AcpExternalModeOption[] = binding?.modes ?? []
  const hasModes = modes.length > 0
  const showFilter = modes.length > 6
  const online = snap.connection === 'online'
  const busy = pending || Boolean(snap.acpConnecting)

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return modes
    return modes.filter(
      option =>
        option.name.toLowerCase().includes(needle) ||
        option.id.toLowerCase().includes(needle) ||
        (option.description?.toLowerCase().includes(needle) ?? false),
    )
  }, [modes, query])

  useAnchoredRuntimePopup(open && hasModes, triggerRef, popupRef, 'right')

  useEffect(() => {
    if (!open || !hasModes) return
    if (showFilter) filterRef.current?.focus()
    else popupRef.current?.focus()
  }, [open, hasModes, showFilter])

  useEffect(() => {
    if (!hasModes && open) setOpen(false)
  }, [hasModes, open])

  const close = (): void => {
    setOpen(false)
    setQuery('')
    triggerRef.current?.focus()
  }

  useEscapeClose(open && hasModes, close)

  if (!binding || !hasModes) return null

  const label = activeAcpModeLabel(binding)
  const activeId = binding.current_mode ?? ''

  const toggle = (): void => {
    if (open) close()
    else setOpen(true)
  }

  const pickMode = async (option: AcpExternalModeOption): Promise<void> => {
    if (busy || !online) return
    if (option.id === activeId) {
      close()
      return
    }
    setPending(true)
    try {
      const ok = await (onSelectMode ? onSelectMode(option.id) : store.setAcpMode(option.id))
      if (ok) close()
    } finally {
      setPending(false)
    }
  }

  const onPopupKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      close()
    }
  }

  return (
    <div className={`chipanchor composer__acp-mode${compact ? ' composer__acp-mode--compact' : ''}`}>
      <button
        ref={triggerRef}
        type="button"
        className={`cchip composer__text acp-mode-chip${open ? ' is-open' : ''}`}
        data-provider={binding.provider}
        data-mode={activeId || undefined}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`${binding.name} mode: ${label}`}
        title={`${binding.name} mode: ${label} — click to switch`}
        disabled={busy}
        onClick={toggle}
      >
        <span className="acp-mode-chip__label">{pending ? 'Switching…' : label}</span>
        <Icon name="caretDown" size={10} />
      </button>

      <PickerLayer>
        {open && (
          <>
            <div className="backdrop backdrop--clear" onClick={close} />
            <div
              ref={popupRef}
              className="acp-optpop acp-modepop"
              role="dialog"
              aria-label={`${binding.name} modes`}
              tabIndex={-1}
              onKeyDown={onPopupKeyDown}
            >
              {snap.acpError && (
                <div className="acp-pop__error" role="alert">
                  <span>{snap.acpError}</span>
                  <button
                    type="button"
                    aria-label="Dismiss runtime error"
                    onClick={() => store.clearAcpError()}
                  >
                    <Icon name="close" size={11} />
                  </button>
                </div>
              )}

              {showFilter && (
                <label className="acp-optpop__search">
                  <Icon name="search" size={13} />
                  <input
                    ref={filterRef}
                    type="text"
                    className="acp-optpop__filter"
                    aria-label={`Filter ${binding.name} modes`}
                    placeholder="Filter modes…"
                    spellCheck={false}
                    value={query}
                    onChange={event => setQuery(event.target.value)}
                  />
                </label>
              )}

              <div className="acp-optpop__list" role="listbox" aria-label={`${binding.name} modes`}>
                {filtered.length === 0 ? (
                  <div className="acp-optpop__empty">No matching modes</div>
                ) : (
                  filtered.map(option => {
                    const isActive = option.id === activeId
                    return (
                      <button
                        key={option.id}
                        type="button"
                        role="option"
                        aria-selected={isActive}
                        disabled={busy || !online}
                        className={`acp-opt-row${isActive ? ' is-active' : ''}`}
                        onClick={() => void pickMode(option)}
                      >
                        <span className="acp-opt-row__body">
                          <span className="acp-opt-row__name">{option.name}</span>
                          {option.description ? (
                            <span className="acp-opt-row__desc">{option.description}</span>
                          ) : null}
                        </span>
                        <span className="acp-opt-row__check" aria-hidden="true">
                          {isActive ? <Icon name="check" size={13} /> : null}
                        </span>
                      </button>
                    )
                  })
                )}
              </div>
            </div>
          </>
        )}
      </PickerLayer>
    </div>
  )
}

export interface AcpConfigOptionPickerProps {
  readonly snap: Snapshot
  readonly option: AcpConfigOption
  readonly compact?: boolean
  readonly defaultOpen?: boolean
  readonly onSelectValue?: (configId: string, value: string) => Promise<boolean> | boolean
}

/** Generic Claude Code-style footer picker for any additional select `configOptions` an ACP agent advertises. */
export function AcpConfigOptionPicker({
  snap,
  option,
  compact = false,
  defaultOpen = false,
  onSelectValue,
}: AcpConfigOptionPickerProps): ReactElement | null {
  const binding = isExternalAcpBinding(snap.acpBinding) ? snap.acpBinding : null
  const [open, setOpen] = useState(defaultOpen)
  const [query, setQuery] = useState('')
  const [pending, setPending] = useState(false)
  const triggerRef = useRef<HTMLButtonElement>(null)
  const popupRef = useRef<HTMLDivElement>(null)
  const filterRef = useRef<HTMLInputElement>(null)

  const choices = option.options
  const hasChoices = choices.length > 0
  const showFilter = choices.length > 6
  const online = snap.connection === 'online'
  const busy = pending || Boolean(snap.acpConnecting)

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    if (!needle) return choices
    return choices.filter(
      item =>
        item.name.toLowerCase().includes(needle)
        || item.value.toLowerCase().includes(needle)
        || (item.description?.toLowerCase().includes(needle) ?? false),
    )
  }, [choices, query])

  useAnchoredRuntimePopup(open && hasChoices, triggerRef, popupRef, 'right')

  useEffect(() => {
    if (!open || !hasChoices) return
    if (showFilter) filterRef.current?.focus()
    else popupRef.current?.focus()
  }, [open, hasChoices, showFilter])

  useEffect(() => {
    if (!hasChoices && open) setOpen(false)
  }, [hasChoices, open])

  const close = (): void => {
    setOpen(false)
    setQuery('')
    triggerRef.current?.focus()
  }

  useEscapeClose(open && hasChoices, close)

  if (!binding || !hasChoices) return null

  const label = activeAcpConfigOptionLabel(option)
  const activeValue = option.current_value

  const toggle = (): void => {
    if (open) close()
    else setOpen(true)
  }

  const pickValue = async (value: string): Promise<void> => {
    if (busy || !online) return
    if (value === activeValue) {
      close()
      return
    }
    setPending(true)
    try {
      const ok = await (
        onSelectValue
          ? onSelectValue(option.id, value)
          : store.setAcpConfigOption(option.id, value)
      )
      if (ok) close()
    } finally {
      setPending(false)
    }
  }

  const onPopupKeyDown = (event: KeyboardEvent<HTMLDivElement>): void => {
    if (event.key === 'Escape') {
      event.preventDefault()
      event.stopPropagation()
      close()
    }
  }

  return (
    <div
      className={`chipanchor composer__acp-config${compact ? ' composer__acp-config--compact' : ''}`}
      data-config-id={option.id}
    >
      <button
        ref={triggerRef}
        type="button"
        className={`cchip composer__text acp-config-chip${open ? ' is-open' : ''}`}
        data-provider={binding.provider}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-label={`${option.name}: ${label}`}
        title={`${option.name}: ${label}${option.description ? ` — ${option.description}` : ''} — click to switch`}
        disabled={busy}
        onClick={toggle}
      >
        <span className="acp-config-chip__label">{pending ? 'Switching…' : label}</span>
        <Icon name="caretDown" size={10} />
      </button>

      <PickerLayer>
        {open && (
          <>
            <div className="backdrop backdrop--clear" onClick={close} />
            <div
              ref={popupRef}
              className="acp-optpop acp-configpop"
              role="dialog"
              aria-label={`${binding.name} ${option.name}`}
              tabIndex={-1}
              onKeyDown={onPopupKeyDown}
            >
              {snap.acpError && (
                <div className="acp-pop__error" role="alert">
                  <span>{snap.acpError}</span>
                  <button
                    type="button"
                    aria-label="Dismiss runtime error"
                    onClick={() => store.clearAcpError()}
                  >
                    <Icon name="close" size={11} />
                  </button>
                </div>
              )}

              {showFilter ? (
                <label className="acp-optpop__search">
                  <Icon name="search" size={13} />
                  <input
                    ref={filterRef}
                    type="text"
                    className="acp-optpop__filter"
                    aria-label={`Filter ${option.name}`}
                    placeholder={`Filter ${option.name.toLowerCase()}…`}
                    spellCheck={false}
                    value={query}
                    onChange={event => setQuery(event.target.value)}
                  />
                </label>
              ) : (
                <div className="acp-optpop__caption">{option.name}</div>
              )}

              <div className="acp-optpop__list" role="listbox" aria-label={option.name}>
                {filtered.length === 0 ? (
                  <div className="acp-optpop__empty">No matching options</div>
                ) : (
                  filtered.map(choice => {
                    const isActive = choice.value === activeValue
                    const showId = !choice.description && shouldShowSecondaryId(choice.name, choice.value)
                    return (
                      <button
                        key={choice.value}
                        type="button"
                        role="option"
                        aria-selected={isActive}
                        title={choice.value}
                        disabled={busy || !online}
                        className={`acp-opt-row${isActive ? ' is-active' : ''}`}
                        onClick={() => void pickValue(choice.value)}
                      >
                        <span className="acp-opt-row__body">
                          <span className="acp-opt-row__name">{choice.name}</span>
                          {choice.description ? (
                            <span className="acp-opt-row__desc">{choice.description}</span>
                          ) : showId ? (
                            <span className="acp-opt-row__id">{choice.value}</span>
                          ) : null}
                        </span>
                        <span className="acp-opt-row__check" aria-hidden="true">
                          {isActive ? <Icon name="check" size={13} /> : null}
                        </span>
                      </button>
                    )
                  })
                )}
              </div>
            </div>
          </>
        )}
      </PickerLayer>
    </div>
  )
}

export function AcpExtraConfigPickers({
  snap,
  compact = false,
}: {
  readonly snap: Snapshot
  readonly compact?: boolean
}): ReactElement | null {
  const binding = isExternalAcpBinding(snap.acpBinding) ? snap.acpBinding : null
  if (!binding) return null
  const extras = extraAcpConfigOptions(binding)
  if (extras.length === 0) return null
  return (
    <>
      {extras.map(option => (
        <AcpConfigOptionPicker key={option.id} snap={snap} option={option} compact={compact} />
      ))}
    </>
  )
}

export interface AcpActiveBannerProps {
  readonly snap: Snapshot
  readonly onDisconnect?: () => Promise<boolean> | boolean
  readonly onRetry?: () => Promise<boolean> | boolean
  readonly onReconnect?: (binding: AcpSessionBinding) => Promise<boolean> | boolean
  readonly onDismissError?: () => void
}

/** Detect auth errors on Antigravity ACP so the inline banner can surface the exact login command. */
export function isAcpAuthError(message: string): boolean {
  return /\b(auth(?:entication|orization)?\s+required|unauthenticated|not signed in|sign\s*in|login\s+required|-32000)\b/i.test(message)
}

interface AgyQuickTransferState {
  readonly mode: AgyCredentialMode
  readonly label: string
  readonly destination: string
  readonly phase: 'reviewing' | 'ready' | 'working' | 'done'
  readonly review?: string
  readonly error: string
}

/**
 * Quiet Claude Code-style composer notice bar for ACP errors/auth recovery
 * and saved-session reconnects. Returns null during normal connected operation
 * so the borderless runtime selector and Environment rail stay calm and uncluttered.
 */
export function AcpActiveBanner({
  snap,
  onDisconnect,
  onRetry,
  onReconnect,
  onDismissError,
}: AcpActiveBannerProps): ReactElement | null {
  const binding = isExternalAcpBinding(snap.acpBinding) ? snap.acpBinding : null
  const savedBinding = !binding && snap.savedAcpBinding && snap.savedAcpBinding.transport !== 'native'
    ? snap.savedAcpBinding
    : null
  const acpError = snap.acpError?.trim() || ''
  const isAntigravityAuthError =
    Boolean(acpError) && isAcpAuthError(acpError) && /\b(antigravity|agy)\b/i.test(acpError)
  const [agyStatus, setAgyStatus] = useState<AgyStatusView | null>(null)
  const [transfer, setTransfer] = useState<AgyQuickTransferState | null>(null)
  const transferReq = useRef(0)

  useEffect(() => {
    if (!isAntigravityAuthError || typeof window === 'undefined' || !window.xerxes?.accounts) {
      setAgyStatus(null)
      return
    }
    let cancelled = false
    void window.xerxes.accounts('agy-status', {})
      .then(raw => {
        if (cancelled || !raw || typeof raw !== 'object') return
        const view = raw as AgyStatusView
        if (Array.isArray(view.sources)) setAgyStatus(view)
      })
      .catch(() => {
        if (!cancelled) setAgyStatus(null)
      })
    return () => { cancelled = true }
  }, [isAntigravityAuthError])

  if (!acpError && !savedBinding) return null

  const connecting = Boolean(snap.acpConnecting)
  const online = snap.connection === 'online'
  const showAuthHint = Boolean(acpError) && isAcpAuthError(acpError) && !/agy acp --login/i.test(acpError)
  const canRetry = Boolean(
    onRetry
    || binding
    || savedBinding
    || store.hasRetryableAcpTarget()
    || isAcpAuthError(acpError)
    || /\b(antigravity|agy|claude)\b/i.test(acpError),
  )
  const transferMode = agyStatus?.canSync ? agyStatus.recommendedMode : null
  const transferSource = transferMode ? agyStatus?.sources.find(s => s.mode === transferMode) : undefined

  const startTransferReview = async (): Promise<void> => {
    if (!transferMode || typeof window === 'undefined' || !window.xerxes?.accounts) return
    const req = ++transferReq.current
    const label = transferSource?.label ?? 'Gemini Subscription (Antigravity)'
    const destination = agyStatus?.destination ?? 'this SSH host'
    setTransfer({ mode: transferMode, label, destination, phase: 'reviewing', error: '' })
    try {
      const reviewed = await window.xerxes.accounts('agy-sync-review', {
        mode: transferMode,
        ...(transferSource?.accountId ? { accountId: transferSource.accountId } : {}),
      }) as { ok?: boolean; review?: string; label?: string; destination?: string; error?: string }
      if (transferReq.current !== req) return
      if (reviewed.ok === false || typeof reviewed.review !== 'string') {
        throw new Error(reviewed.error || 'Could not review Antigravity credential transfer')
      }
      setTransfer({
        mode: transferMode,
        label: reviewed.label || label,
        destination: reviewed.destination || destination,
        phase: 'ready',
        review: reviewed.review,
        error: '',
      })
    } catch (error) {
      if (transferReq.current === req) {
        setTransfer({ mode: transferMode, label, destination, phase: 'ready', error: desktopError(error) })
      }
    }
  }

  const confirmTransfer = async (): Promise<void> => {
    if (!transfer || typeof window === 'undefined' || !window.xerxes?.accounts) return
    if (transfer.phase === 'done') {
      setTransfer(null)
      return
    }
    if (transfer.phase !== 'ready') return
    if (!transfer.review) {
      void startTransferReview()
      return
    }
    const req = transferReq.current
    setTransfer({ ...transfer, phase: 'working', error: '' })
    try {
      const res = await window.xerxes.accounts('agy-sync', { review: transfer.review, confirm: true }) as { ok?: boolean; error?: string }
      if (transferReq.current !== req) return
      if (res.ok === false) throw new Error(res.error || 'Could not transfer Antigravity credentials')
      setTransfer(null)
      await (onRetry ? onRetry() : store.retryAcpConnect())
    } catch (error) {
      if (transferReq.current === req) {
        setTransfer({
          mode: transfer.mode,
          label: transfer.label,
          destination: transfer.destination,
          phase: 'ready',
          error: desktopError(error),
        })
      }
    }
  }

  if (acpError) {
    return (
      <>
        <div className="acp-banner acp-banner--error" role="alert">
          <span className="acp-banner__icon" aria-hidden="true">
            <Icon name="warning" size={14} />
          </span>
          <span className="acp-banner__text acp-banner__text--wrap">
            <span>{acpError}</span>
            {showAuthHint && (
              <span className="acp-banner__hint">
                {' '}Run <code>agy acp --login</code> in your terminal, then click Retry.
              </span>
            )}
          </span>
          <div className="acp-banner__actions">
            {transferMode && (
              <button
                type="button"
                className="acp-banner__switch acp-banner__switch--primary"
                disabled={connecting || !online || transfer?.phase === 'working'}
                onClick={() => void startTransferReview()}
              >
                Transfer from this Mac
              </button>
            )}
            {canRetry && (
              <button
                type="button"
                className={`acp-banner__switch${transferMode ? '' : ' acp-banner__switch--primary'}`}
                disabled={connecting || !online}
                onClick={() => void (onRetry ? onRetry() : store.retryAcpConnect())}
              >
                {connecting ? 'Retrying…' : 'Retry'}
              </button>
            )}
            {binding && (
              <button
                type="button"
                className="acp-banner__switch"
                disabled={connecting}
                onClick={() => void (onDisconnect ? onDisconnect() : store.disconnectAcpAgent())}
              >
                Switch to Xerxes
              </button>
            )}
            <button
              type="button"
              className="acp-banner__dismiss"
              aria-label="Dismiss runtime error"
              title="Dismiss error"
              onClick={() => (onDismissError ? onDismissError() : store.clearAcpError())}
            >
              <Icon name="close" size={12} />
            </button>
          </div>
        </div>
        {transfer && (
          <ActionDialog
            title={`Transfer ${transfer.label} to SSH`}
            confirmLabel={!transfer.review && transfer.phase === 'ready' ? 'Review transfer again' : 'Transfer & connect'}
            busyLabel="Transferring credentials…"
            busy={transfer.phase === 'working'}
            disabled={transfer.phase === 'reviewing'}
            done={transfer.phase === 'done'}
            error={transfer.error}
            onConfirm={() => void confirmTransfer()}
            onCancel={() => { transferReq.current += 1; setTransfer(null) }}
          >
            {transfer.phase === 'reviewing' ? (
              <p role="status">Preparing credential transfer review for <strong>{transfer.destination}</strong>…</p>
            ) : (
              <>
                <p>Transfer your local <strong>{transfer.label}</strong> credentials to <strong>{transfer.destination}</strong> and connect Antigravity?</p>
                <p className="studio-muted">
                  Credentials are sent over SSH stdin only and saved with owner-only permissions (mode 0600) in ~/.gemini/antigravity-cli/antigravity-oauth-token.
                </p>
              </>
            )}
          </ActionDialog>
        )}
      </>
    )
  }

  if (savedBinding) {
    return (
      <div className="acp-banner acp-banner--reconnect" role="status" data-provider={savedBinding.provider}>
        <RuntimeBrandIcon provider={savedBinding.provider} size={14} className="acp-banner__glyph" />
        <span className="acp-banner__text">
          Previous runtime: <strong>{savedBinding.name}</strong>
          {savedBinding.external_session_id ? (
            <span className="acp-banner__sid"> · {savedBinding.external_session_id.slice(0, 8)}</span>
          ) : null}
        </span>
        <div className="acp-banner__actions">
          <button
            type="button"
            className="acp-banner__switch acp-banner__switch--primary"
            disabled={connecting || !online}
            onClick={() => void (onReconnect ? onReconnect(savedBinding) : store.reconnectSavedAcp())}
          >
            {connecting ? 'Reconnecting…' : `Reconnect ${savedBinding.name}`}
          </button>
          <button
            type="button"
            className="acp-banner__dismiss"
            aria-label="Dismiss reconnect banner"
            title="Keep using Xerxes"
            onClick={() => store.dismissSavedAcpBinding()}
          >
            <Icon name="close" size={12} />
          </button>
        </div>
      </div>
    )
  }

  return null
}
