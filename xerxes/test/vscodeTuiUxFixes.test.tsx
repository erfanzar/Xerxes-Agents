// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

import { parseColor, RGBA } from '@opentui/core'
import { afterEach, describe, expect, test } from 'bun:test'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import type { Snapshot } from '../src/desktop/renderer/store.js'
import type { OverlayState } from '../src/ui/app/interfaces.js'
import {
  getOverlayState,
  openOverlayIfUnblocked,
  OVERLAY_FLAG_KEYS,
  patchOverlayState,
  resetOverlayState
} from '../src/ui/app/overlayStore.js'
import { resetSideQuestionStore } from '../src/ui/app/sideQuestionStore.js'
import { resetUiState } from '../src/ui/app/uiStore.js'
import { dismissOverlayForCtrlC, shouldFallThroughForScroll } from '../src/ui/app/useInputHandlers.js'
import { shortenWorkspacePath } from '../src/ui/opentui/appChrome.js'
import { toRenderableColor } from '../src/ui/opentui/primitives.js'
import {
  collectSideQuestionModelOptions,
  filterSideQuestionModels
} from '../src/ui/opentui/sideQuestionOverlay.js'
import { DARK_THEME, LIGHT_THEME, normalizeThemeForAnsiLightTerminal } from '../src/ui/theme.js'
import { pickTargetHost, type TargetHostCandidate } from '../src/vscode/hostTarget.js'
import { SideQuestionCard } from '../src/vscode/ui/SideQuestionCard.js'
import { nextHistoryCursor, VscodeChat } from '../src/vscode/ui/VscodeApp.js'

const SRC_ROOT = join(import.meta.dir, '../src')

const makeSnapshot = (overrides: Partial<Snapshot> = {}): Snapshot => ({
  connection: 'online',
  cwd: '/repo',
  model: 'claude-opus-4-6',
  currentAgentPreset: 'default',
  agentPresets: [],
  models: [
    { id: 'claude-opus-4-6', label: 'Claude Opus 4.6', provider: 'Anthropic' },
    { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash', provider: 'Google' }
  ],
  reasoningLevels: [],
  contextTokens: null,
  contextMax: null,
  ttftMs: null,
  tokensPerSecond: null,
  llmDurationMs: 0,
  llmSteps: 0,
  toolDurationMs: 0,
  toolSteps: 0,
  inputTokens: 0,
  outputTokens: 0,
  metricPhase: null,
  metricPhaseStartedAt: null,
  cacheHitRate: null,
  sessions: [],
  live: [],
  fleet: [],
  skillSuggestions: [],
  creatorTrace: [],
  backgroundJobs: [],
  currentId: 's1',
  currentTitle: 'Refactor extension sidebar',
  sessionKey: 'sk-1',
  branch: 'main',
  daemonWarning: null,
  costUsd: null,
  mcpStatus: {},
  goal: '',
  approval: null,
  question: null,
  interactionHidden: false,
  planMode: false,
  turnActive: false,
  turnFailed: false,
  turnSeconds: 0,
  blocks: [],
  error: null,
  tab: 'activity',
  turnCount: 0,
  queue: [],
  changes: [],
  plan: null,
  log: [],
  failed: null,
  settingsOpen: false,
  settingsTab: 'general',
  paletteOpen: false,
  commands: [],
  pickerOpen: false,
  wsMenuOpen: false,
  sessionMenu: null,
  streamThinking: true,
  taskModalOpen: false,
  providers: [],
  providerModels: {},
  providerModelLoading: [],
  providerModelWarnings: {},
  providerTypes: [],
  permissionMode: '',
  snippets: {},
  ...overrides
})

afterEach(() => {
  resetOverlayState()
  resetUiState()
  resetSideQuestionStore()
})

// ── Part A: VS Code Extension Fixes ──────────────────────────────────────────

describe('VS Code extension UI/UX fixes', () => {
  test('extension.ts loads vscode.css LAST and vscode.css outranks desktop :not(#id) styles', () => {
    const extensionSrc = readFileSync(join(SRC_ROOT, 'vscode/extension.ts'), 'utf8')
    const layoutIdx = extensionSrc.indexOf("asset('renderer/layout.css')")
    const vibesIdx = extensionSrc.indexOf("asset('renderer/vibes.css')")
    const neoIdx = extensionSrc.indexOf("asset('renderer/neo.css')")
    const vscodeIdx = extensionSrc.indexOf("asset('vscode.css')")

    expect(layoutIdx).toBeGreaterThan(-1)
    expect(vibesIdx).toBeGreaterThan(layoutIdx)
    expect(neoIdx).toBeGreaterThan(vibesIdx)
    expect(vscodeIdx).toBeGreaterThan(neoIdx)

    const css = readFileSync(join(SRC_ROOT, 'vscode/ui/vscode.css'), 'utf8')
    expect(css).toContain(":root[data-host='vscode']:not(#vscode-host):not(#vscode-layer){")
    expect(css).toContain(":root[data-host='vscode'][data-high-contrast]:not(#vscode-host):not(#vscode-layer){")
    expect(css).toContain('width:100%;max-width:none;height:100%;max-height:none;flex-direction:column;border:0;border-radius:0')
    expect(css).toContain('padding:14px 14px 24px')
    expect(css).toContain('display:flex;flex-direction:column;gap:6px')
  })

  test('VscodeChat drives SideQuestionCard from store with no duplicate local btwBySession or ⌘; listener', () => {
    const vscodeAppSrc = readFileSync(join(SRC_ROOT, 'vscode/ui/VscodeApp.tsx'), 'utf8')
    expect(vscodeAppSrc).not.toContain('btwBySession')
    expect(vscodeAppSrc).not.toContain("event.key === ';'")

    const openHtml = renderToStaticMarkup(
      createElement(VscodeChat, {
        snap: makeSnapshot({
          sideQuestionOpen: true,
          sideQuestionTurns: [
            {
              requestId: 'sq-store-1',
              question: 'Where does CSS load?',
              answer: 'In `extension.ts`.',
              model: 'gemini-2.5-flash',
              status: 'done'
            }
          ]
        })
      })
    )
    expect(openHtml).toContain('class="xv-btw"')
    expect(openHtml).toContain('Where does CSS load?')
    expect(openHtml).toContain('extension.ts')

    const closedHtml = renderToStaticMarkup(
      createElement(VscodeChat, {
        snap: makeSnapshot({ sideQuestionOpen: false })
      })
    )
    expect(closedHtml).not.toContain('class="xv-btw"')
  })

  test('SideQuestionCard places model picker on dedicated .xv-btw__bar subrow so Copy/Clear/Close stay visible in 300px sidebars', () => {
    const html = renderToStaticMarkup(
      createElement(SideQuestionCard, {
        snap: makeSnapshot({
          sideQuestionOpen: true,
          sideQuestionTurns: [
            {
              requestId: 'sq-1',
              question: 'Check sidebar?',
              answer: 'Looks clean.',
              status: 'done'
            }
          ]
        })
      })
    )
    const headEnd = html.indexOf('</header>')
    const head = html.slice(0, headEnd)
    const barIdx = html.indexOf('class="xv-btw__bar"')

    expect(head).toContain('class="xv-btw__actions"')
    expect(head).toContain('Copy')
    expect(head).toContain('Clear')
    expect(head).toContain('aria-label="Close side question"')
    // Model picker moved out of .xv-btw__head into .xv-btw__bar so header never clips at 260-300px
    expect(head).not.toContain('xv-btw__model-btn')
    expect(barIdx).toBeGreaterThan(headEnd)
    expect(html.slice(barIdx)).toContain('xv-btw__model-btn')

    const css = readFileSync(join(SRC_ROOT, 'vscode/ui/vscode.css'), 'utf8')
    expect(css).toContain('@container (max-width:420px)')
    expect(css).toContain('.xv-btw__badge{display:none}')
  })

  test('pickTargetHost routes commands to active/visible host instead of broadcasting, and extension.ts uses it', () => {
    const makeHost = (
      id: string,
      opts: Partial<TargetHostCandidate>
    ): TargetHostCandidate & { id: string } => ({
      id,
      surface: 'panel',
      isVisible: () => false,
      isActive: () => false,
      lastActiveAt: 1,
      ...opts
    })

    const view = makeHost('sidebar', { surface: 'view', isVisible: () => true, isActive: () => false, lastActiveAt: 2 })
    const panelActive = makeHost('panel-1', { surface: 'panel', isVisible: () => true, isActive: () => true, lastActiveAt: 3 })
    const panelBg = makeHost('panel-2', { surface: 'panel', isVisible: () => false, isActive: () => false, lastActiveAt: 10 })

    expect(pickTargetHost([view, panelActive, panelBg])?.id).toBe('panel-1')
    expect(pickTargetHost([view, panelBg])?.id).toBe('sidebar')

    const hiddenView = makeHost('sidebar', { surface: 'view', isVisible: () => false, isActive: () => false, lastActiveAt: 1 })
    expect(pickTargetHost([hiddenView, panelBg])?.id).toBe('sidebar')
    expect(pickTargetHost([])).toBeUndefined()

    const extensionSrc = readFileSync(join(SRC_ROOT, 'vscode/extension.ts'), 'utf8')
    expect(extensionSrc).toContain('const target = pickTargetHost(hosts)')
    expect(extensionSrc).not.toContain("for (const entry of hosts) void entry.webview.postMessage({ kind: 'push', channel: 'desktop:menu'")
    expect(extensionSrc).not.toContain("for (const entry of hosts) void entry.webview.postMessage({ kind: 'push', channel: 'desktop:compose'")
  })

  test('History cursor never becomes -1 on empty list and 300px header drops duplicate Ask aside pill', () => {
    expect(nextHistoryCursor(0, 0, 1)).toBe(0)
    expect(nextHistoryCursor(0, 0, -1)).toBe(0)
    expect(nextHistoryCursor(-1, 0, 1)).toBe(0)
    expect(nextHistoryCursor(0, 3, 1)).toBe(1)
    expect(nextHistoryCursor(2, 3, 1)).toBe(2)
    expect(nextHistoryCursor(0, 3, -1)).toBe(0)

    const html = renderToStaticMarkup(createElement(VscodeChat, { snap: makeSnapshot() }))
    const headerStart = html.indexOf('<header class="xv-head">')
    const headerEnd = html.indexOf('</header>', headerStart)
    const headerHtml = html.slice(headerStart, headerEnd)
    expect(headerHtml).not.toContain('Ask aside')
    expect(headerHtml).toContain('class="xv-dot"')
  })
})

// ── Part B: Pure OpenTUI Unit Checks (render tests live in src/ui/__tests__/tuiUxFixes.test.tsx) ──

describe('OpenTUI terminal UI/UX pure invariants', () => {
  test('Apple Terminal light mode emits valid #rrggbb hex strings (never ansi256 / neon magenta) and primitives coerce defensively', () => {
    const normalized = normalizeThemeForAnsiLightTerminal(
      LIGHT_THEME,
      { TERM_PROGRAM: 'Apple_Terminal' } as NodeJS.ProcessEnv,
      true
    )

    for (const [role, value] of Object.entries(normalized.color)) {
      expect(value, `role ${role} must not emit ansi256(...)`).not.toMatch(/^ansi256\(/i)
      expect(value, `role ${role} must be valid hex`).toMatch(/^#[0-9a-f]{6}$/i)
      const parsed = parseColor(value)
      expect(parsed.r === 1 && parsed.g === 0 && parsed.b === 1, `role ${role} fell back to magenta`).toBe(false)
    }

    expect(toRenderableColor('ansi256(240)')).toBeInstanceOf(RGBA)
    expect(toRenderableColor('rgb(16, 32, 48)')).toBeInstanceOf(RGBA)
  })

  test('Ctrl+C declaratively dismisses ALL 36 OVERLAY_FLAG_KEYS without silently swallowing any', () => {
    expect(OVERLAY_FLAG_KEYS.length).toBeGreaterThanOrEqual(36)

    for (const key of OVERLAY_FLAG_KEYS) {
      resetOverlayState()
      const activeValue: unknown =
        key === 'approval'
          ? { command: 'rm -rf /tmp/x', description: 'test', requestId: 'req-a' }
          : key === 'clarify'
            ? { question: 'Choose?', requestId: 'req-c' }
            : key === 'confirm'
              ? { title: 'Confirm?', onConfirm: () => undefined }
              : key === 'copyPicker'
                ? { items: [] }
                : key === 'pager'
                  ? { lines: ['line'], offset: 0 }
                  : key === 'secret'
                    ? { envVar: 'API_KEY', prompt: 'Key', requestId: 'req-sec' }
                    : key === 'sudo'
                      ? { prompt: 'sudo', requestId: 'req-sudo' }
                      : true

      patchOverlayState({ [key]: activeValue } as Partial<OverlayState>)
      const calls: string[] = []
      const handled = dismissOverlayForCtrlC(getOverlayState(), {
        answerClarify: () => calls.push('clarify'),
        denyApproval: () => calls.push('approval'),
        cancelSudo: () => calls.push('sudo'),
        cancelSecret: () => calls.push('secret'),
        patchOverlay: patch => {
          calls.push(`patch:${Object.keys(patch).join(',')}`)
          patchOverlayState(patch)
        }
      })

      expect(handled, `Ctrl+C must handle overlay key "${key}"`).toBe(true)
      expect(calls.length, `Ctrl+C must invoke handler for "${key}"`).toBe(1)

      if (key !== 'clarify' && key !== 'approval' && key !== 'sudo' && key !== 'secret') {
        expect(Boolean(getOverlayState()[key]), `overlay "${key}" should be closed`).toBe(false)
      }
    }
  })

  test('approval PgUp/PgDn scroll isolation, openOverlayIfUnblocked, workspace path shortening, theme scrims, and side-question model filtering', () => {
    const baseKey = {
      downArrow: false,
      pageDown: false,
      pageUp: true,
      shift: false,
      upArrow: false,
      wheelDown: false,
      wheelUp: false
    }
    expect(shouldFallThroughForScroll(baseKey, { approvalActive: true })).toBe(false)
    expect(shouldFallThroughForScroll({ ...baseKey, pageUp: false, pageDown: true }, { approvalActive: true })).toBe(false)
    expect(shouldFallThroughForScroll({ ...baseKey, shift: true }, { approvalActive: true })).toBe(true)
    expect(shouldFallThroughForScroll(baseKey, { approvalActive: false })).toBe(true)
    expect(shouldFallThroughForScroll({ ...baseKey, pageUp: false, wheelUp: true }, { approvalActive: true })).toBe(true)

    resetOverlayState()
    expect(openOverlayIfUnblocked({ goal: true })).toBe(true)
    expect(openOverlayIfUnblocked({ modelPicker: true })).toBe(false)
    expect(getOverlayState().modelPicker).toBe(false)

    expect(shortenWorkspacePath('~/Documents/Projects/Xerxes-Agents', 22)).toBe('~/…/Xerxes-Agents')
    expect(shortenWorkspacePath('~/short', 22)).toBe('~/short')

    expect(DARK_THEME.ds.scrim).toBe('#000000cc')
    expect(LIGHT_THEME.ds.scrim).not.toBe('#000000cc')
    expect(DARK_THEME.color.overlayBg).not.toBe(DARK_THEME.color.statusBg)
    expect(LIGHT_THEME.color.overlayBg).not.toBe(LIGHT_THEME.color.statusBg)

    const opentuiDir = join(SRC_ROOT, 'ui/opentui')
    for (const file of readdirSync(opentuiDir).filter(name => name.endsWith('.tsx'))) {
      const src = readFileSync(join(opentuiDir, file), 'utf8')
      expect(src, `${file} must use t.ds.scrim instead of hardcoded #000000cc`).not.toContain('#000000cc')
    }

    const candidates = collectSideQuestionModelOptions(
      {
        model: 'claude-opus-4-6',
        providers: [
          { slug: 'anthropic', name: 'Anthropic', models: ['claude-opus-4-6', 'claude-haiku-4-5'] },
          { slug: 'google', name: 'Google', models: ['gemini-2.5-flash', 'gemini-2.5-pro'] }
        ]
      } as never,
      'claude-opus-4-6',
      ''
    )
    expect(filterSideQuestionModels(candidates, 'flash').map(item => item.id)).toEqual(['gemini-2.5-flash'])
  })
})
