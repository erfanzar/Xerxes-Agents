// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import {
  AcpActiveBanner,
  AcpConfigOptionPicker,
  AcpExtraConfigPickers,
  AcpModePicker,
  AcpModelPicker,
  AcpRuntimePicker,
  AcpThinkingPicker,
  extraAcpConfigOptions,
} from '../src/desktop/renderer/AcpRuntimePicker.js'
import { Shell, formatQuestionAnswers, toggleQuestionOption } from '../src/desktop/renderer/App.js'
import { AcpCustomizePanel, DesktopPage, RailAddMenu } from '../src/desktop/renderer/DesktopPanels.js'
import { ArtifactsSection } from '../src/desktop/renderer/RailSections.js'
import {
  LearnedBadge,
  ReflectionCard,
} from '../src/desktop/renderer/LearnedPanel.js'
import {
  Store,
  parseAcpCatalogResult,
  parseAcpSessionBinding,
  resolveExternalAcpCatalog,
  shouldForwardSlashToAcp,
  type AcpExternalAgentInfo,
  type AcpSessionBinding,
  type DesktopBridge,
  type ReflectionNotice,
  type Snapshot,
} from '../src/desktop/renderer/store.js'
import { VscodeChat } from '../src/vscode/ui/VscodeApp.js'

const HEX_COLOR_RE = /#[0-9a-fA-F]{6}\b/

const sampleAgents: readonly AcpExternalAgentInfo[] = [
  {
    id: 'xerxes',
    name: 'Xerxes',
    provider: 'xerxes',
    transport: 'native',
    available: true,
    description: 'Native multi-agent runtime & Immortals',
  },
  {
    id: 'antigravity',
    name: 'Antigravity',
    provider: 'antigravity',
    transport: 'acp',
    available: true,
    command: '/Users/erfan/.local/bin/agy',
    description: 'Google Antigravity via ACP stdio',
  },
]

const fullAgyBinding: AcpSessionBinding = {
  session_id: 'sess-agy-full',
  agent_id: 'antigravity',
  name: 'Antigravity',
  provider: 'antigravity',
  transport: 'acp',
  cwd: '/workspace/xerxes',
  external_session_id: 'agy-ext-9f8e7d6c',
  connected: true,
  agent_title: 'Antigravity ACP Server',
  agent_version: '1.5.14',
  capabilities: {
    prompt: { image: true, audio: false, embedded_context: true },
    session: { load: true, resume: true, list: true, close: true, additional_directories: true },
    mcp: { http: true, sse: false },
    auth_methods: [
      { id: 'google-oauth', name: 'Google Sign-In', description: 'Authenticate with your Google account' },
    ],
    meta: { antigravity: { tier: true } },
  },
  current_model: 'gemini-4-carbon',
  current_model_name: 'Gemini 4 Carbon',
  models: [
    { id: 'gemini-4-carbon', name: 'Gemini 4 Carbon' },
    { id: 'gemini-pro-orchestrator', name: 'Orchestrator', description: 'High-depth coding model' },
    { id: 'gemini-3-flash', name: 'Gemini 3 Flash' },
    { id: 'claude-sonnet-4-6', name: 'Claude Sonnet 4.6' },
    { id: 'claude-opus-4-6', name: 'Claude Opus 4.6' },
    { id: 'gemini-4-neon', name: 'Gemini 4 Neon' },
    { id: 'gemini-4-ultra', name: 'Gemini 4 Ultra' },
  ],
  thought_level: 'high',
  thought_levels: [
    { id: 'low', name: 'Low' },
    { id: 'medium', name: 'Medium' },
    { id: 'high', name: 'High' },
  ],
  current_mode: 'planning',
  current_mode_name: 'Planning',
  modes: [
    { id: 'planning', name: 'Planning', description: 'Plan before editing files' },
    { id: 'fast', name: 'Fast', description: 'Execute directly' },
  ],
  config_options: [
    {
      id: 'model',
      name: 'Model',
      category: 'model',
      type: 'select',
      current_value: 'gemini-4-carbon',
      options: [
        { value: 'gemini-4-carbon', name: 'Gemini 4 Carbon' },
        { value: 'gemini-pro-orchestrator', name: 'Orchestrator', description: 'High-depth coding model' },
      ],
    },
    {
      id: 'thought_level',
      name: 'Thinking Level',
      category: 'thought_level',
      type: 'select',
      current_value: 'high',
      options: [
        { value: 'low', name: 'Low' },
        { value: 'high', name: 'High' },
      ],
    },
    {
      id: 'mode',
      name: 'Mode',
      category: 'mode',
      type: 'select',
      current_value: 'planning',
      options: [
        { value: 'planning', name: 'Planning' },
        { value: 'fast', name: 'Fast' },
      ],
    },
    {
      id: 'sandbox_profile',
      name: 'Sandbox',
      category: 'security',
      type: 'select',
      current_value: 'workspace-write',
      options: [
        { value: 'read-only', name: 'Read only', description: 'Inspect workspace without writes' },
        { value: 'workspace-write', name: 'Workspace write', description: 'Edit files inside project' },
      ],
    },
    {
      id: 'web_grounding',
      name: 'Web Search',
      type: 'boolean',
      current_value: 'true',
      options: [],
    },
  ],
  available_commands: [
    { name: 'compact', description: 'Compact the Antigravity conversation context', kind: 'command', source: 'Antigravity' },
    { name: 'goal', description: 'Run an overnight autonomous goal in Antigravity', kind: 'command', source: 'Antigravity' },
    { name: 'plan', description: 'Enter Antigravity step-by-step planning mode', kind: 'command', source: 'Antigravity' },
    { name: 'swarm', description: 'Coordinate an Antigravity multi-agent swarm', kind: 'command', source: 'Antigravity' },
    { name: 'boost', description: 'Boost reasoning depth for the next Antigravity step', kind: 'command', source: 'Antigravity' },
  ],
  discovered_skills: [
    {
      name: 'academic-cv-builder',
      description: 'Format CVs for academic positions with publications and grants',
      source: 'Workspace',
      path: '.agents/skills/academic-cv-builder/SKILL.md',
      slash_command: '/academic-cv-builder',
    },
    {
      name: 'run-workflow',
      description: 'Run a dynamic Python workflow script with parallel agents',
      source: 'Antigravity',
      path: '~/.gemini/antigravity/builtin/skills/run-workflow/SKILL.md',
      slash_command: '/run-workflow',
    },
    {
      // Guardrail: even if a broken server ever echoes a Xerxes bundled skill, renderer filters it out
      name: 'immortals-coordinator',
      description: 'Xerxes native coordinator skill',
      source: 'Xerxes bundled',
    },
  ],
}

interface MockBridge {
  readonly calls: Array<{ method: string; params: Record<string, unknown> }>
  readonly emit: (type: string, payload: Record<string, unknown>) => void
  readonly xerxes: DesktopBridge
}

function createMockBridge(
  overrides: Partial<Record<string, (params: Record<string, unknown>) => Record<string, unknown>>> = {},
): MockBridge {
  const calls: Array<{ method: string; params: Record<string, unknown> }> = []
  const listeners = new Set<(event: { type: string; payload: Record<string, unknown> }) => void>()
  const xerxes: DesktopBridge = {
    async call(method: string, params: Record<string, unknown> = {}) {
      calls.push({ method, params })
      const handler = overrides[method]
      if (handler) return handler(params)
      switch (method) {
        case 'initialize':
          return {
            ok: true,
            session_id: 'sess-agy-full',
            session_key: String(params.session_key ?? 'desktop-test'),
            cwd: '/workspace/xerxes',
            model: 'claude-opus-4-6',
            session: {
              id: 'sess-agy-full',
              title: 'Claude UI & ACP Discovery',
              cwd: '/workspace/xerxes',
              model: 'claude-opus-4-6',
              messages: [],
            },
          }
        case 'runtime.status':
          return { ok: true, model: 'claude-opus-4-6', permission_mode: 'accept-all' }
        case 'session.list':
          return { ok: true, sessions: [] }
        case 'acp.agents':
          return { ok: true, agents: sampleAgents, active: null }
        case 'acp.connect':
          return { ok: true, binding: fullAgyBinding }
        case 'acp.disconnect':
          return { ok: true, disconnected: true, binding: null }
        case 'acp.catalog':
          return {
            ok: true,
            connected: true,
            binding: fullAgyBinding,
            commands: fullAgyBinding.available_commands,
            skills: fullAgyBinding.discovered_skills,
            capabilities: fullAgyBinding.capabilities,
            config_options: fullAgyBinding.config_options,
          }
        case 'acp.set_config_option':
          return {
            ok: true,
            binding: {
              ...fullAgyBinding,
              config_options: fullAgyBinding.config_options?.map(opt =>
                opt.id === String(params.config_id) ? { ...opt, current_value: String(params.value) } : opt,
              ),
            },
          }
        case 'turn.submit':
          return { ok: true }
        default:
          return { ok: true }
      }
    },
    onEvent(handler) {
      listeners.add(handler)
      return () => {
        listeners.delete(handler)
      }
    },
  }
  return {
    calls,
    emit: (type, payload) => {
      for (const listener of listeners) listener({ type, payload })
    },
    xerxes,
  }
}

function createMemoryStorage(): Storage {
  const map = new Map<string, string>()
  return {
    get length() {
      return map.size
    },
    clear() {
      map.clear()
    },
    getItem(key: string) {
      return map.get(key) ?? null
    },
    key(index: number) {
      return Array.from(map.keys())[index] ?? null
    },
    removeItem(key: string) {
      map.delete(key)
    },
    setItem(key: string, value: string) {
      map.set(key, String(value))
    },
  }
}

const originalWindow = (globalThis as Record<string, unknown>).window
const originalLocalStorage = (globalThis as Record<string, unknown>).localStorage
let activeStore: Store | null = null

async function setupConnectedStore(
  overrides: Partial<Record<string, (params: Record<string, unknown>) => Record<string, unknown>>> = {},
): Promise<{ store: Store; bridge: MockBridge }> {
  const bridge = createMockBridge(overrides)
  ;(globalThis as Record<string, unknown>).localStorage = createMemoryStorage()
  ;(globalThis as Record<string, unknown>).window = { xerxes: bridge.xerxes }
  const store = new Store()
  activeStore = store
  store.start(bridge.xerxes)
  for (let i = 0; i < 40 && store.getSnapshot().connection !== 'online'; i += 1) {
    await Bun.sleep(5)
  }
  await Bun.sleep(10)
  return { store, bridge }
}

function sampleSnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  const base = new Store().getSnapshot()
  return {
    ...base,
    connection: 'online',
    noWorkspace: false,
    cwd: '/workspace/xerxes',
    branch: 'main',
    currentId: 'sess-agy-full',
    currentTitle: 'Claude UI & ACP Discovery',
    sessionKey: 'desktop-test',
    model: 'claude-opus-4-6',
    models: [{ id: 'claude-opus-4-6', provider: 'Anthropic', label: 'Claude Opus 4.6' }],
    acpAgents: sampleAgents,
    acpBinding: null,
    acpConnecting: false,
    acpError: null,
    ...overrides,
  }
}

afterEach(() => {
  activeStore = null
  if (originalWindow === undefined) delete (globalThis as Record<string, unknown>).window
  else (globalThis as Record<string, unknown>).window = originalWindow
  if (originalLocalStorage === undefined) delete (globalThis as Record<string, unknown>).localStorage
  else (globalThis as Record<string, unknown>).localStorage = originalLocalStorage
})

describe('Part 1 — Claude Code Visual Language & Popover Polish', () => {
  test('CSS eliminates double-bordered embedded inputs, thick blue focus rings, and noisy blue active pills', () => {
    const rendererDir = join(import.meta.dir, '../src/desktop/renderer')
    const vibesCss = readFileSync(join(rendererDir, 'vibes.css'), 'utf8')
    const layoutCss = readFileSync(join(rendererDir, 'layout.css'), 'utf8')
    const appCss = readFileSync(join(rendererDir, 'app.css'), 'utf8')
    const atelierCss = readFileSync(join(rendererDir, 'atelier.css'), 'utf8')

    // High-specificity embedded-input reset in vibes.css covers every composite search input
    expect(vibesCss).toContain(':not(#vibes):not(#embedded-input-reset)')
    for (const selector of [
      '.acp-optpop__filter',
      '.modelpop .palette__in',
      '.rsec__search input',
      '.settings-search input',
      '.swarm-model-picker__filter',
      '.btw-model-pop__filter',
      '.findbar__in',
      '.side-question__input',
      '.acp-catalog__search input',
    ]) {
      expect(vibesCss).toContain(selector)
    }
    expect(layoutCss).toContain('.acp-optpop__filter')

    // Borderless transparent external runtime chip (Claude Code "This computer ⌄" aesthetic)
    expect(appCss).toMatch(/\.acp-chip\.is-external\s*\{[^}]*border-color:\s*transparent;[^}]*background:\s*transparent/)
    expect(vibesCss).toContain('.acp-chip.is-external{border-color:transparent;background:transparent')

    // Flush borderless 36px top search row & neutral active selection (zero blue wash)
    expect(appCss).toMatch(/\.acp-optpop__search\s*\{[^}]*height:\s*36px;[^}]*border-bottom:\s*1px solid var\(--x-hairline\)/)
    expect(appCss).toMatch(/\.acp-opt-row\.is-active\s*\{[^}]*background:\s*color-mix\(in srgb,\s*var\(--x-selected\)/)
    expect(atelierCss).toContain('.swarm-model-picker__search{display:flex;align-items:center;gap:8px;padding:0 12px;height:36px;border:0;border-bottom:1px solid var(--x-hairline)')
    expect(atelierCss).toContain('.btw-model-pop__search{display:flex;align-items:center;gap:8px;padding:0 12px;height:36px;border:0;border-bottom:1px solid var(--x-hairline)')
  })

  test('ACP popovers drop competing .modelpop & clunky headers, hide redundant slug IDs, and stay calm when connected', () => {
    const agySnap = sampleSnapshot({ acpBinding: fullAgyBinding })

    // Normal connected state renders NO banner above the composer
    expect(renderToStaticMarkup(createElement(AcpActiveBanner, { snap: agySnap }))).toBe('')

    // Model picker: flush search bar, no standalone .modelpop class, no .acp-optpop__head header, omits redundant slug ID
    const modelHtml = renderToStaticMarkup(createElement(AcpModelPicker, { snap: agySnap, defaultOpen: true }))
    expect(modelHtml).not.toMatch(/["\s]modelpop["\s]/)
    expect(modelHtml).not.toContain('acp-optpop__head')
    expect(modelHtml).toContain('acp-optpop__search')
    expect(modelHtml).toContain('Filter models…')
    expect(modelHtml).toContain('Gemini 4 Carbon')
    // Redundant slugified ID omitted for "Gemini 4 Carbon" (gemini-4-carbon), kept as tooltip/title
    expect(modelHtml).not.toContain('<span class="acp-opt-row__id">gemini-4-carbon</span>')
    expect(modelHtml).toContain('title="gemini-4-carbon"')
    // Distinct human description shown cleanly for Orchestrator
    expect(modelHtml).toContain('High-depth coding model')
    expect(modelHtml).not.toMatch(HEX_COLOR_RE)

    // Thinking & Mode pickers also omit standalone .modelpop & clunky header bars
    const thinkingHtml = renderToStaticMarkup(createElement(AcpThinkingPicker, { snap: agySnap, defaultOpen: true }))
    expect(thinkingHtml).not.toMatch(/["\s]modelpop["\s]/)
    expect(thinkingHtml).not.toContain('acp-optpop__head')
    expect(thinkingHtml).toContain('High')

    const modeHtml = renderToStaticMarkup(createElement(AcpModePicker, { snap: agySnap, defaultOpen: true }))
    expect(modeHtml).not.toMatch(/["\s]modelpop["\s]/)
    expect(modeHtml).not.toContain('acp-optpop__head')
    expect(modeHtml).toContain('Planning')

    const runtimeHtml = renderToStaticMarkup(createElement(AcpRuntimePicker, { snap: agySnap, defaultOpen: true }))
    expect(runtimeHtml).not.toMatch(/["\s]modelpop["\s]/)
    expect(runtimeHtml).not.toContain('acp-pop__head')
    expect(runtimeHtml).not.toContain('acp-pop__foot')
    expect(runtimeHtml).not.toContain('acp-row__transport')
    expect(runtimeHtml).not.toContain('acp-row__dot')
    expect(runtimeHtml).not.toContain('stdio JSON-RPC')
    expect(runtimeHtml).not.toContain('xerxes-glyph-spin')
    expect(runtimeHtml).toContain('Antigravity')
    expect(runtimeHtml).toContain('Antigravity CLI')
    expect(runtimeHtml).toContain('Built-in runtime')
    expect(runtimeHtml).toContain('Custom ACP…')
  })

  test('Dynamic extra select & boolean configOptions render via AcpConfigOptionPicker in Desktop Shell and VS Code', async () => {
    const extras = extraAcpConfigOptions(fullAgyBinding)
    expect(extras.map(o => o.id)).toEqual(['sandbox_profile', 'web_grounding'])
    expect(extras[1]?.options.map(o => o.value)).toEqual(['true', 'false'])

    const agySnap = sampleSnapshot({ acpBinding: fullAgyBinding })
    const configHtml = renderToStaticMarkup(
      createElement(AcpConfigOptionPicker, { snap: agySnap, option: extras[0]!, defaultOpen: true }),
    )
    expect(configHtml).toContain('Workspace write')
    expect(configHtml).toContain('Read only')
    expect(configHtml).toContain('Inspect workspace without writes')
    expect(configHtml).not.toMatch(HEX_COLOR_RE)

    const extraStripHtml = renderToStaticMarkup(createElement(AcpExtraConfigPickers, { snap: agySnap }))
    expect(extraStripHtml).toContain('Workspace write')
    expect(extraStripHtml).toContain('On')

    ;(globalThis as Record<string, unknown>).window = { xerxes: createMockBridge().xerxes }
    const shellHtml = renderToStaticMarkup(createElement(Shell, { snap: agySnap }))
    expect(shellHtml).toContain('composer__acp-config')
    expect(shellHtml).toContain('Workspace write')

    const vscodeHtml = renderToStaticMarkup(createElement(VscodeChat, { snap: agySnap }))
    expect(vscodeHtml).toContain('xv-pill--acp')
    expect(vscodeHtml).toContain('Workspace write')
    expect(vscodeHtml).not.toContain('Connected via')

    const { store, bridge } = await setupConnectedStore()
    await store.connectAcpAgent('antigravity')
    bridge.calls.length = 0
    expect(await store.setAcpConfigOption('sandbox_profile', 'read-only')).toBe(true)
    expect(bridge.calls.find(c => c.method === 'acp.set_config_option')?.params).toMatchObject({
      config_id: 'sandbox_profile',
      value: 'read-only',
    })
  })
})

describe('Part 2 — Automatic ACP Skill & Runtime Feature Discovery + Strict Isolation', () => {
  test('resolveExternalAcpCatalog & parseAcpCatalogResult strip Xerxes native skills and preserve ACP capabilities', () => {
    const resolved = resolveExternalAcpCatalog(fullAgyBinding)
    expect(resolved.commands.map(c => c.name)).toEqual(['compact', 'goal', 'plan', 'swarm', 'boost'])
    expect(resolved.skills.map(s => s.name)).toEqual(['academic-cv-builder', 'run-workflow'])
    expect(resolved.skills.some(s => s.name.includes('immortals'))).toBe(false)

    const parsed = parseAcpCatalogResult({
      ok: true,
      connected: true,
      binding: fullAgyBinding,
      commands: fullAgyBinding.available_commands,
      skills: fullAgyBinding.discovered_skills,
      capabilities: fullAgyBinding.capabilities,
      config_options: fullAgyBinding.config_options,
    })
    expect(parsed?.connected).toBe(true)
    expect(parsed?.commands).toHaveLength(5)
    expect(parsed?.skills.map(s => s.name)).toEqual(['academic-cv-builder', 'run-workflow'])
    expect(parsed?.capabilities?.prompt.image).toBe(true)
    expect(parsed?.capabilities?.session.additional_directories).toBe(true)
  })

  test('store.loadAcpCatalog calls acp.catalog and falls back cleanly to snap.acpBinding on error', async () => {
    const { store, bridge } = await setupConnectedStore()
    await store.connectAcpAgent('antigravity')
    bridge.calls.length = 0

    const liveCatalog = await store.loadAcpCatalog()
    expect(bridge.calls.some(c => c.method === 'acp.catalog')).toBe(true)
    expect(liveCatalog.connected).toBe(true)
    expect(liveCatalog.skills.map(s => s.name)).toEqual(['academic-cv-builder', 'run-workflow'])
    expect(liveCatalog.commands.map(c => c.name)).toEqual(['compact', 'goal', 'plan', 'swarm', 'boost'])

    // Fallback when acp.catalog fails on an older daemon
    const fallbackSetup = await setupConnectedStore({
      'acp.catalog': () => {
        throw new Error('method not found')
      },
    })
    await fallbackSetup.store.connectAcpAgent('antigravity')
    const fallbackCatalog = await fallbackSetup.store.loadAcpCatalog()
    expect(fallbackCatalog.connected).toBe(true)
    expect(fallbackCatalog.skills.map(s => s.name)).toEqual(['academic-cv-builder', 'run-workflow'])
  })

  test('Strict `/` completion isolation & slash routing on external ACP — restores Xerxes immediately on disconnect', async () => {
    const nativeCompletions = [
      { value: '/goal', label: '/goal', meta: 'Manage Xerxes goal' },
      { value: '/compact', label: '/compact', meta: 'Compact Xerxes session' },
      { value: '/immortals', label: '/immortals', meta: 'Launch Xerxes Immortals swarm' },
      { value: '/swarm', label: '/swarm', meta: 'Native Xerxes swarm' },
      { value: '/provider', label: '/provider', meta: 'Switch Xerxes provider' },
      { value: '/memory', label: '/memory', meta: 'Xerxes tiered memory' },
      { value: '/skills', label: '/skills', meta: 'Xerxes native skills' },
      { value: '/acp', label: '/acp', meta: 'Manage external ACP runtime' },
      { value: '/new', label: '/new', meta: 'Start a fresh session' },
    ]
    const nativeSkillCompletions = [
      { value: '/skill immortals-coordinator', label: 'immortals-coordinator', meta: 'Native Xerxes skill' },
      { value: '/skill immortals-monitor', label: 'immortals-monitor', meta: 'Native Xerxes skill' },
    ]

    const { store, bridge } = await setupConnectedStore({
      complete: params => {
        const text = String(params.text ?? '')
        if (text.startsWith('/skill')) return { ok: true, completions: nativeSkillCompletions }
        if (/^\/acp\s/i.test(text)) {
          const sub = text.slice(5).trim().toLowerCase()
          const acpSubs = [
            { value: '/acp status', label: '/acp status', meta: 'Show connected ACP runtime status' },
            { value: '/acp catalog', label: '/acp catalog', meta: 'Inspect discovered ACP skills & commands' },
            { value: '/acp model', label: '/acp model', meta: 'Switch ACP model' },
            { value: '/acp mode', label: '/acp mode', meta: 'Switch ACP mode' },
            { value: '/acp thinking', label: '/acp thinking', meta: 'Switch ACP thinking level' },
            { value: '/acp disconnect', label: '/acp disconnect', meta: 'Return to Xerxes native runtime' },
          ].filter(item => !sub || item.value.toLowerCase().includes(sub))
          return { ok: true, completions: acpSubs }
        }
        return { ok: true, completions: nativeCompletions }
      },
    })

    // 1. Connect to Antigravity ACP
    await store.connectAcpAgent('antigravity')
    const acpHints = await store.completeText('/')
    const values = acpHints.map(h => h.value)
    expect(values).toContain('/compact')
    expect(values).toContain('/goal')
    expect(values).toContain('/plan')
    expect(values).toContain('/swarm')
    expect(values).toContain('/boost')
    expect(values).toContain('/academic-cv-builder')
    expect(values).toContain('/run-workflow')
    expect(values).toContain('/acp')
    expect(acpHints.find(h => h.value === '/acp')?.badge).toBe('Host')
    expect(acpHints.find(h => h.value === '/academic-cv-builder')?.badge).toBe('Workspace')
    expect(acpHints.find(h => h.value === '/run-workflow')?.badge).toBe('Antigravity')

    // ZERO Xerxes native commands or skills leak while connected to external ACP
    for (const forbidden of ['/immortals', '/provider', '/memory', '/skills', '/skill', '/reflect', '/hooks']) {
      expect(values).not.toContain(forbidden)
    }
    expect(values.some(v => v.includes('immortals'))).toBe(false)
    // Typing `/skill ` surfaces ONLY Antigravity's discovered skills (never Xerxes immortals-*)
    const acpSkillHints = await store.completeText('/skill ')
    expect(acpSkillHints.map(h => h.value)).toEqual(['/academic-cv-builder', '/run-workflow'])
    expect(acpSkillHints.some(h => h.label.includes('immortals'))).toBe(false)

    // Explicit host utility prefix (/new) stays reachable with Host badge
    const newHints = await store.completeText('/new')
    expect(newHints.find(h => h.value === '/new')?.badge).toBe('Host')

    // Multi-token `/acp <subcmd>` completions forward cleanly to host with Host badge while other Xerxes args stay blocked
    const acpSubHints = await store.completeText('/acp ')
    expect(acpSubHints.map(h => h.value)).toEqual([
      '/acp status',
      '/acp catalog',
      '/acp model',
      '/acp mode',
      '/acp thinking',
      '/acp disconnect',
    ])
    expect(acpSubHints.every(h => h.badge === 'Host')).toBe(true)
    const acpCatHints = await store.completeText('/acp cat')
    expect(acpCatHints.map(h => h.value)).toEqual(['/acp catalog'])
    expect(await store.completeText('/provider ')).toEqual([])

    // 2. Colliding commands & discovered skills forward straight to turn.submit; /acp stays host-owned
    expect(shouldForwardSlashToAcp('/compact', fullAgyBinding)).toBe(true)
    expect(shouldForwardSlashToAcp('/goal build app', fullAgyBinding)).toBe(true)
    expect(shouldForwardSlashToAcp('/swarm fix tests', fullAgyBinding)).toBe(true)
    expect(shouldForwardSlashToAcp('/plan refactor auth', fullAgyBinding)).toBe(true)
    expect(shouldForwardSlashToAcp('/academic-cv-builder update publications', fullAgyBinding)).toBe(true)
    expect(shouldForwardSlashToAcp('/run-workflow analyze', fullAgyBinding)).toBe(true)
    expect(shouldForwardSlashToAcp('/acp status', fullAgyBinding)).toBe(false)

    for (const cmd of ['/compact', '/swarm audit files', '/academic-cv-builder draft CV']) {
      bridge.calls.length = 0
      expect(await store.submit(cmd)).toBe(true)
      expect(bridge.calls.some(c => c.method === 'turn.submit')).toBe(true)
      expect(bridge.calls.some(c => c.method === 'slash')).toBe(false)
      expect(bridge.calls.some(c => c.method === 'swarm.activate')).toBe(false)
      bridge.emit('turn_end', { session_id: store.getSnapshot().currentId })
    }

    // /acp stays on host slash RPC
    bridge.calls.length = 0
    await store.submit('/acp status')
    expect(bridge.calls.some(c => c.method === 'slash')).toBe(true)
    expect(bridge.calls.some(c => c.method === 'turn.submit')).toBe(false)

    // 3. Disconnecting immediately restores full Xerxes native completions & skills
    expect(await store.disconnectAcpAgent()).toBe(true)
    const restoredHints = await store.completeText('/')
    expect(restoredHints.map(h => h.value)).toContain('/immortals')
    expect(restoredHints.map(h => h.value)).toContain('/provider')
    expect(restoredHints.map(h => h.value)).toContain('/skills')
    const restoredSkills = await store.completeText('/skill ')
    expect(restoredSkills.map(h => h.label)).toContain('immortals-coordinator')
  })

  test('Customize page swaps to runtime-aware AcpCustomizePanel with zero Xerxes native skills, and restores on disconnect', () => {
    ;(globalThis as Record<string, unknown>).window = { xerxes: createMockBridge().xerxes }
    const agySnap = sampleSnapshot({ acpBinding: fullAgyBinding })

    // Direct AcpCustomizePanel render (Skills tab default + Commands + Capabilities tabs)
    const acpCustomizeHtml = renderToStaticMarkup(
      createElement(AcpCustomizePanel, { snap: agySnap, binding: fullAgyBinding }),
    )
    expect(acpCustomizeHtml).toContain('Customize')
    expect(acpCustomizeHtml).toContain('Antigravity')
    expect(acpCustomizeHtml).toContain('1.5.14')
    expect(acpCustomizeHtml).toContain('Switch to Xerxes')
    expect(acpCustomizeHtml).toContain('Skills')
    expect(acpCustomizeHtml).toContain('Commands')
    expect(acpCustomizeHtml).toContain('Capabilities')
    expect(acpCustomizeHtml).toContain('academic-cv-builder')
    expect(acpCustomizeHtml).toContain('run-workflow')
    expect(acpCustomizeHtml).toContain('Use in chat')
    expect(acpCustomizeHtml).not.toContain('immortals-coordinator')
    expect(acpCustomizeHtml).not.toContain('Trust all')
    expect(acpCustomizeHtml).not.toContain('Learned')
    expect(acpCustomizeHtml).not.toMatch(HEX_COLOR_RE)

    const commandsHtml = renderToStaticMarkup(
      createElement(AcpCustomizePanel, { snap: agySnap, binding: fullAgyBinding, initialTab: 'commands' }),
    )
    expect(commandsHtml).toContain('/compact')
    expect(commandsHtml).toContain('/boost')
    expect(commandsHtml).toContain('Use in chat')
    expect(commandsHtml).not.toMatch(HEX_COLOR_RE)

    const capsHtml = renderToStaticMarkup(
      createElement(AcpCustomizePanel, { snap: agySnap, binding: fullAgyBinding, initialTab: 'capabilities' }),
    )
    expect(capsHtml).toContain('Prompt capabilities')
    expect(capsHtml).toContain('Images')
    expect(capsHtml).toContain('Embedded context')
    expect(capsHtml).toContain('Google Sign-In')
    expect(capsHtml).toContain('Sandbox')
    expect(capsHtml).not.toMatch(HEX_COLOR_RE)

    // DesktopPage panel="extensions" renders AcpCustomizePanel on ACP, native panel on disconnect
    const acpPageHtml = renderToStaticMarkup(
      createElement(DesktopPage, { snap: agySnap, panel: 'extensions', onClose: () => {} }),
    )
    expect(acpPageHtml).toContain('acp-customize-page')
    expect(acpPageHtml).not.toContain('catalog-page')
    expect(acpPageHtml).toContain('academic-cv-builder')
    expect(acpPageHtml).not.toContain('immortals-coordinator')
    expect(acpPageHtml).not.toContain('Trust all')

    const nativePageHtml = renderToStaticMarkup(
      createElement(DesktopPage, { snap: sampleSnapshot({ acpBinding: null }), panel: 'extensions', onClose: () => {} }),
    )
    expect(nativePageHtml).toContain('catalog-page')
    expect(nativePageHtml).not.toContain('acp-customize-page')
    expect(nativePageHtml).toContain('>skills<')
    expect(nativePageHtml).toContain('>plugins<')
    expect(nativePageHtml).toContain('>learned<')
  })

  test('Capability-gated attach, Right Rail Environment section, and Xerxes-native chrome isolation (Immortals & Learned)', () => {
    ;(globalThis as Record<string, unknown>).window = { xerxes: createMockBridge().xerxes }
    const noImageBinding: AcpSessionBinding = {
      ...fullAgyBinding,
      capabilities: {
        ...fullAgyBinding.capabilities!,
        prompt: { image: false, audio: false, embedded_context: true },
      },
    }

    // 1. Image attach gating on capabilities.prompt.image
    const noImgShellHtml = renderToStaticMarkup(
      createElement(Shell, { snap: sampleSnapshot({ acpBinding: noImageBinding }) }),
    )
    expect(noImgShellHtml).toContain('data-images-supported="false"')
    expect(noImgShellHtml).toContain('does not support image attachments')

    const imgShellHtml = renderToStaticMarkup(
      createElement(Shell, { snap: sampleSnapshot({ acpBinding: fullAgyBinding }) }),
    )
    expect(imgShellHtml).toContain('data-images-supported="true"')

    // 2. LearnedBadge & ReflectionCard hidden on external ACP, visible on native Xerxes
    const sampleReflection: ReflectionNotice = {
      id: 'refl-1',
      kind: 'skill',
      target: 'code-review',
      rationale: 'Drafted a new lesson',
      evidence_count: 2,
    }
    const agySnapWithLearned = sampleSnapshot({
      acpBinding: fullAgyBinding,
      learningCounts: { pending_proposals: 2, pending_lessons: 1 },
    })
    expect(renderToStaticMarkup(createElement(LearnedBadge, { snap: agySnapWithLearned }))).toBe('')
    expect(
      renderToStaticMarkup(
        createElement(ReflectionCard, {
          snap: agySnapWithLearned,
          reflection: sampleReflection,
        }),
      ),
    ).toBe('')

    const nativeSnapWithLearned = sampleSnapshot({
      acpBinding: null,
      learningCounts: { pending_proposals: 2, pending_lessons: 1 },
    })
    expect(renderToStaticMarkup(createElement(LearnedBadge, { snap: nativeSnapWithLearned }))).toContain('3')
    expect(
      renderToStaticMarkup(
        createElement(ReflectionCard, {
          snap: nativeSnapWithLearned,
          reflection: sampleReflection,
        }),
      ),
    ).toContain('code-review')

    // 3. Immortals hidden in FloatingRail & RailAddMenu on external ACP unless swarm is active
    expect(imgShellHtml).not.toContain('aria-label="Open Immortals"')
    const railMenuHidden = renderToStaticMarkup(
      createElement(RailAddMenu, {
        onPick: () => {},
        hideSwarm: true,
        defaultOpen: true,
      }),
    )
    expect(railMenuHidden).not.toContain('Immortals')
    const railMenuNative = renderToStaticMarkup(
      createElement(RailAddMenu, {
        onPick: () => {},
        hideSwarm: false,
        defaultOpen: true,
      }),
    )
    expect(railMenuNative).toContain('Immortals')

    // 4. Right Rail Environment section displays ACP runtime, version, skill/command counts & capability pills
    const railOpenSnap = sampleSnapshot({
      acpBinding: fullAgyBinding,
      railOpen: true,
      railTab: 'activity',
    })
    const railHtml = renderToStaticMarkup(createElement(Shell, { snap: railOpenSnap }))
    expect(railHtml).toContain('railcard--environment')
    expect(railHtml).toContain('Environment')
    expect(railHtml).toContain('Antigravity')
    expect(railHtml).toContain('1.5.14')
    expect(railHtml).toContain('2 skills · 5 commands')
    expect(railHtml).toContain('Images')
    expect(railHtml).toContain('Context')
    expect(railHtml).toContain('Resume')
    expect(railHtml).toContain('Load')
    expect(railHtml).toContain('MCP')
    expect(railHtml).not.toContain('rail-env__cap">Audio<')
    expect(railHtml).not.toMatch(HEX_COLOR_RE)
  })

  test('ACP elicitation/create multi_select & single-select questions parse in Store and render checkbox/radio controls in QuestionCard', async () => {
    const { store, bridge } = await setupConnectedStore()
    bridge.emit('question_request', {
      id: 'q-req-1',
      tool_call_id: 'tc-ask-1',
      questions: [
        {
          id: 'q1',
          question: 'Favorite color?',
          options: ['(Recommended) Blue', 'Green'],
          allow_free_form: true,
        },
        {
          id: 'q2',
          question: 'Which agents should get custom artwork?',
          options: ['Core roster: Xerxes, Coder, Reviewer', 'Immortals sprites', 'MCP helpers'],
          allow_free_form: true,
          multi_select: true,
        },
      ],
    })
    const q = store.getSnapshot().question
    expect(q?.requestId).toBe('q-req-1')
    expect(q?.items).toHaveLength(2)
    expect(q?.items[0]?.multiSelect).toBeUndefined()
    expect(q?.items[1]?.multiSelect).toBe(true)

    // Toggle logic for single-select vs multi-select
    expect(toggleQuestionOption(['(Recommended) Blue'], 'Green', false)).toEqual(['Green'])
    expect(toggleQuestionOption(['Core roster: Xerxes, Coder, Reviewer'], 'Immortals sprites', true)).toEqual([
      'Core roster: Xerxes, Coder, Reviewer',
      'Immortals sprites',
    ])
    expect(
      toggleQuestionOption(['Core roster: Xerxes, Coder, Reviewer', 'Immortals sprites'], 'Immortals sprites', true),
    ).toEqual(['Core roster: Xerxes, Coder, Reviewer'])

    // Multi-select formats newline-joined answers preserving commas inside option titles + custom Other text
    const formatted = formatQuestionAnswers(
      q!.items,
      {
        q1: ['(Recommended) Blue'],
        q2: ['Core roster: Xerxes, Coder, Reviewer', 'Immortals sprites'],
      },
      { q2: 'Custom mascot' },
    )
    expect(formatted).toEqual({
      q1: '(Recommended) Blue',
      q2: 'Core roster: Xerxes, Coder, Reviewer\nImmortals sprites\nCustom mascot',
    })

    // Shell renders role="radio" for single-select & role="checkbox" + "Select all that apply" for multi-select
    const qHtml = renderToStaticMarkup(createElement(Shell, { snap: sampleSnapshot({ question: q! }) }))
    expect(qHtml).toContain('role="radiogroup"')
    expect(qHtml).toContain('role="radio"')
    expect(qHtml).toContain('role="group"')
    expect(qHtml).toContain('role="checkbox"')
    expect(qHtml).toContain('Select all that apply')
    expect(qHtml).not.toMatch(HEX_COLOR_RE)
  })

  test('AcpCustomizePanel exposes official agy acp launch options, parses launch metadata, forwards flags on connect, and ArtifactsSection renders read-only brain images', async () => {
    const parsed = parseAcpSessionBinding({
      ...fullAgyBinding,
      additional_directories: ['/workspace/shared', '/workspace/docs'],
      sandbox: true,
      agent_override: 'architect',
      mcp_server_names: ['playwright', 'github'],
    })
    expect(parsed?.additional_directories).toEqual(['/workspace/shared', '/workspace/docs'])
    expect(parsed?.sandbox).toBe(true)
    expect(parsed?.agent_override).toBe('architect')
    expect(parsed?.mcp_server_names).toEqual(['playwright', 'github'])

    const capsHtml = renderToStaticMarkup(
      createElement(AcpCustomizePanel, {
        snap: sampleSnapshot({ acpBinding: parsed }),
        binding: parsed!,
        initialTab: 'capabilities',
      }),
    )
    expect(capsHtml).toContain('Launch options')
    expect(capsHtml).toContain('Terminal sandbox')
    expect(capsHtml).toContain('aria-checked="true"')
    expect(capsHtml).toContain('/workspace/shared')
    expect(capsHtml).toContain('/workspace/docs')
    expect(capsHtml).toContain('Agent override')
    expect(capsHtml).toContain('value="architect"')
    expect(capsHtml).toContain('Apply &amp; reconnect')
    expect(capsHtml).toContain('2 passed through (playwright, github)')
    expect(capsHtml).toContain('agy plugin')
    expect(capsHtml).not.toMatch(HEX_COLOR_RE)

    const { store, bridge } = await setupConnectedStore()
    await store.connectAcpAgent('antigravity', undefined, {
      modeId: 'turbo',
      resumeSessionId: 'agy-ext-42',
      additionalDirectories: ['/workspace/shared'],
      sandbox: true,
      agent: 'architect',
    })
    const connectCall = bridge.calls.find(c => c.method === 'acp.connect')
    expect(connectCall?.params).toMatchObject({
      agent_id: 'antigravity',
      mode_id: 'turbo',
      resume_session_id: 'agy-ext-42',
      additional_directories: ['/workspace/shared'],
      sandbox: true,
      agent: 'architect',
    })

    // Read-only Antigravity image artifact renders inside ArtifactsSection without Edit or Copy raw base64 buttons.
    const pngDataUrl =
      'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
    const artHtml = renderToStaticMarkup(
      createElement(ArtifactsSection, {
        sessionKey: 's1',
        online: true,
        initialDocument: {
          row: {
            id: 'agy:hero_render.png',
            title: 'Hero Render',
            kind: 'image',
            size: 68,
            createdAt: '2026-11-25T00:00:00Z',
            updatedAt: '2026-11-25T00:00:00Z',
            pinned: false,
            version: 1,
          },
          content: pngDataUrl,
        },
      }),
    )
    expect(artHtml).toContain('rsec__image-stage')
    expect(artHtml).toContain('Hero Render')
    expect(artHtml).toContain('Image')
    expect(artHtml).toContain(pngDataUrl)
    expect(artHtml).not.toContain('aria-label="Edit artifact"')
    expect(artHtml).not.toContain('aria-label="Copy artifact"')
  })
})
