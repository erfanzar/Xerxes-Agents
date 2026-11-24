// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from 'bun:test'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { createElement } from 'react'
import { renderToStaticMarkup } from 'react-dom/server'

import { initialHubState, initialSupervisorConfig, type HubState } from '../src/client/hubStore.js'
import { AgentIntelligenceCard } from '../src/desktop/renderer/AgentIntelligenceCard.js'
import { ChannelsCard } from '../src/desktop/renderer/ChannelsPanel.js'
import { EMPTY_STATUS, HubPhoneView, deviceStatusText } from '../src/desktop/renderer/HubPhonePanel.js'
import { LearnedPanel, ProposalCard, formatOutcomeEntries } from '../src/desktop/renderer/LearnedPanel.js'
import { isMemoryGuardDraftUnchanged } from '../src/desktop/renderer/MemoryGuardSetting.js'
import { CommandPalette, ProviderForm, SettingsModal, handleSettingsModalKey } from '../src/desktop/renderer/Overlays.js'
import { AgyCredentialCard, ProviderAccountsCenter, accountKindLabel, agyModeLabel } from '../src/desktop/renderer/ProviderAccountsCenter.js'
import { RelayBadge } from '../src/desktop/renderer/RelayBadge.js'
import { RunModeSection } from '../src/desktop/renderer/RunModeSection.js'
import { localMachineLabel, localMachineNoun, runModeChoices } from '../src/desktop/renderer/runModeView.js'
import { shouldShowSetup } from '../src/desktop/renderer/Setup.js'
import type { Snapshot } from '../src/desktop/renderer/store.js'
import { SupervisorView } from '../src/desktop/renderer/SupervisorPage.js'
import type { StoredReflectionProposal } from '../src/runtime/reflection.js'

const rendererDir = join(import.meta.dir, '../src/desktop/renderer')

const originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document')
const originalWindow = Object.getOwnPropertyDescriptor(globalThis, 'window')
const originalNavigator = Object.getOwnPropertyDescriptor(globalThis, 'navigator')

afterEach(() => {
  if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument)
  else Reflect.deleteProperty(globalThis, 'document')
  if (originalWindow) Object.defineProperty(globalThis, 'window', originalWindow)
  else Reflect.deleteProperty(globalThis, 'window')
  if (originalNavigator) Object.defineProperty(globalThis, 'navigator', originalNavigator)
  else Reflect.deleteProperty(globalThis, 'navigator')
})

function setNavigatorPlatform(platform: string, userAgent: string): void {
  Object.defineProperty(globalThis, 'navigator', {
    configurable: true,
    value: { platform, userAgent },
  })
}

function makeSnapshot(overrides: Partial<Snapshot> = {}): Snapshot {
  return {
    connection: 'online',
    noWorkspace: false,
    cwd: '/workspace/xerxes',
    storageScope: 'local',
    model: 'claude-opus-4-6',
    currentAgentPreset: 'default',
    agentPresets: [],
    agentPresetsError: null,
    channels: [],
    channelsAvailable: true,
    channelsConfigured: true,
    channelsLoading: false,
    channelsError: null,
    terminals: [],
    terminalsLoading: false,
    terminalsError: null,
    searchOpen: false,
    searchResults: [],
    searchStats: null,
    searchSearching: false,
    searchError: null,
    branch: 'main',
    daemonWarning: null,
    costUsd: null,
    costBasis: null,
    mcpStatus: {},
    models: [],
    contextTokens: null,
    computerUseWanted: false,
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
    cacheReadTokens: null,
    cacheWriteTokens: null,
    sessions: [],
    live: [],
    fleet: [],
    skillSuggestions: [],
    creatorTrace: [],
    backgroundJobs: [],
    currentId: 'sess-1',
    sessionOpenRevision: 0,
    currentTitle: 'Test task',
    sessionKey: 'sk-local',
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
    changesKept: false,
    todos: null,
    plan: null,
    log: [],
    failed: null,
    stopArmed: false,
    settingsOpen: false,
    settingsTab: 'general',
    paletteOpen: false,
    pickerOpen: false,
    modelMenuOpen: false,
    appUpdate: null,
    providerRelay: null,
    delegationMode: 'auto',
    delegationMenuOpen: false,
    contextMenuOpen: false,
    contextBreakdown: null,
    contextBreakdownLoading: false,
    reasoningPickerOpen: false,
    reasoningEffort: '',
    reasoningLevels: [],
    reasoningDefault: '',
    reasoningNote: '',
    reasoningLoading: false,
    wsMenuOpen: false,
    workspaceBusy: false,
    workspaceError: null,
    sessionMenu: null,
    streamThinking: true,
    taskModalOpen: false,
    providers: [],
    providerError: '',
    navigating: null,
    providersLoading: false,
    modelsLoading: false,
    modelsError: null,
    providerSwitching: null,
    providerSwitchError: null,
    runMode: null,
    runModeError: '',
    runModeSaving: false,
    localSetup: { status: null, job: null, error: '', loading: false },
    providerModels: {},
    providerModelLoading: [],
    providerModelWarnings: {},
    providerTypes: [],
    commands: [],
    permissionMode: 'auto',
    permissionUpdating: false,
    permissionError: null,
    snippets: {},
    ...overrides,
  }
}

describe('Desktop Settings, Dialogs & Copy UX fixes', () => {
  test('Fix 1: SettingsModal wires Escape, ⌘,/Ctrl+, toggle, backdrop click, tabFilter reset, honors snap.settingsTab, and tolerates missing window.xerxes', () => {
    const overlaysSrc = readFileSync(join(rendererDir, 'Overlays.tsx'), 'utf8')
    expect(overlaysSrc).toContain('handleSettingsModalKey(event, () => store.closeSettings(), { isTopDialog, ownsFocus })')
    expect(overlaysSrc).toContain('handleDialogEscape(event, close, true)')
    expect(overlaysSrc).toContain('onMouseDown={closeOnBackdrop}')
    expect(overlaysSrc).toContain("if (!snap.settingsOpen) setTabFilter('')")

    const makeKeyEvent = (patch: Partial<{
      key: string
      metaKey: boolean
      ctrlKey: boolean
      altKey: boolean
      shiftKey: boolean
      defaultPrevented: boolean
    }>) => {
      let prevented = false
      let stopped = false
      let immediate = false
      return {
        event: {
          key: 'Escape',
          metaKey: false,
          ctrlKey: false,
          altKey: false,
          shiftKey: false,
          defaultPrevented: false,
          ...patch,
          preventDefault: () => { prevented = true },
          stopPropagation: () => { stopped = true },
          stopImmediatePropagation: () => { immediate = true },
        },
        state: () => ({ prevented, stopped, immediate }),
      }
    }

    // ⌘, and Ctrl+, close Settings when it is the top dialog
    for (const mod of [{ metaKey: true }, { ctrlKey: true }]) {
      let closed = 0
      const { event, state } = makeKeyEvent({ key: ',', ...mod })
      expect(handleSettingsModalKey(event, () => { closed += 1 }, { isTopDialog: true, ownsFocus: true })).toBe(true)
      expect(closed).toBe(1)
      expect(state()).toEqual({ prevented: true, stopped: true, immediate: true })
    }

    // Nested ActionDialog on top: neither ⌘, nor Escape steals from it
    for (const patch of [{ key: ',', metaKey: true }, { key: 'Escape' }]) {
      let closed = 0
      const { event } = makeKeyEvent(patch)
      expect(handleSettingsModalKey(event, () => { closed += 1 }, { isTopDialog: false, ownsFocus: false })).toBe(false)
      expect(closed).toBe(0)
    }

    // Escape closes when Settings owns focus; modified ⌘⇧, / ⌥⌘, do not close
    let escClosed = 0
    const esc = makeKeyEvent({ key: 'Escape' })
    expect(handleSettingsModalKey(esc.event, () => { escClosed += 1 }, { isTopDialog: true, ownsFocus: true })).toBe(true)
    expect(escClosed).toBe(1)
    expect(esc.state().immediate).toBe(true)

    let modClosed = 0
    expect(handleSettingsModalKey(makeKeyEvent({ key: ',', metaKey: true, shiftKey: true }).event, () => { modClosed += 1 }, { isTopDialog: true, ownsFocus: true })).toBe(false)
    expect(handleSettingsModalKey(makeKeyEvent({ key: ',', metaKey: true, altKey: true }).event, () => { modClosed += 1 }, { isTopDialog: true, ownsFocus: true })).toBe(false)
    expect(modClosed).toBe(0)

    // Tolerates globalThis.window defined without window.xerxes (both local & SSH Models tab, and VS Code)
    Object.defineProperty(globalThis, 'window', { configurable: true, value: {} })
    const sshModelsHtml = renderToStaticMarkup(
      createElement(SettingsModal, { snap: makeSnapshot({ settingsOpen: true, settingsTab: 'models', storageScope: 'ssh:devbox' }) }),
    )
    expect(sshModelsHtml).toContain('Models &amp; Providers')
    expect(sshModelsHtml).toContain('Provider source for this conversation')

    // VS Code honors snap.settingsTab via section markers while remaining single-page
    Object.defineProperty(globalThis, 'document', {
      configurable: true,
      value: { documentElement: { dataset: { host: 'vscode' } } },
    })
    const vscodeHtml = renderToStaticMarkup(
      createElement(SettingsModal, { snap: makeSnapshot({ settingsOpen: true, settingsTab: 'permissions' }) }),
    )
    expect(vscodeHtml).toContain('data-active-tab="permissions"')
    expect(vscodeHtml).toContain('data-settings-section="models"')
    expect(vscodeHtml).toContain('data-settings-section="intelligence"')
    expect(vscodeHtml).toContain('data-settings-section="permissions"')
    expect(vscodeHtml).not.toContain('mtab')
  })

  test('Fix 2: ProviderForm is keyed per provider, keeps required Base URL open, and removes the 24-model cap', () => {
    const overlaysSrc = readFileSync(join(rendererDir, 'Overlays.tsx'), 'utf8')
    expect(overlaysSrc).toContain("key={typeof form === 'string' ? form : 'new'}")
    expect(overlaysSrc).not.toContain('profileModels.slice(0, 24)')

    // Custom/OpenAI-compatible provider without default baseUrl opens Customized settings automatically
    const customHtml = renderToStaticMarkup(
      createElement(ProviderForm, {
        snap: makeSnapshot({
          providerTypes: [
            { name: 'openai', baseUrl: 'https://api.openai.com/v1', apiKeyEnv: 'OPENAI_API_KEY' },
            { name: 'custom', baseUrl: '', apiKeyEnv: '' },
          ],
        }),
        editing: { name: 'local-vllm', provider: 'custom', model: '', active: false, baseUrl: '' },
        onCancel: () => {},
      }),
    )
    expect(customHtml).toMatch(/<details class="provform__custom" open="">/)
    expect(customHtml).toContain('Required for custom or OpenAI-compatible endpoints.')

    // Known provider with default baseUrl keeps Customized settings collapsed when no override is set
    const knownHtml = renderToStaticMarkup(
      createElement(ProviderForm, {
        snap: makeSnapshot({
          providerTypes: [{ name: 'openai', baseUrl: 'https://api.openai.com/v1', apiKeyEnv: 'OPENAI_API_KEY' }],
        }),
        editing: { name: 'my-openai', provider: 'openai', model: 'gpt-5', active: true, baseUrl: '' },
        onCancel: () => {},
      }),
    )
    expect(knownHtml).not.toMatch(/<details class="provform__custom" open="">/)
    expect(knownHtml).toContain('Leave blank to use the default for openai.')

    // All 35 models are rendered (no 24-model cap), with filter input and ScrollableCombobox on Model
    const models = Array.from({ length: 35 }, (_, index) => `model-${String(index + 1).padStart(2, '0')}`)
    const manyHtml = renderToStaticMarkup(
      createElement(ProviderForm, {
        snap: makeSnapshot({
          providerModels: { openrouter: models },
        }),
        editing: { name: 'openrouter', provider: 'openrouter', model: 'model-35', active: true, baseUrl: 'https://openrouter.ai/api/v1' },
        onCancel: () => {},
      }),
    )
    expect(manyHtml).toContain('Models · 35 discovered')
    expect(manyHtml).toContain('model-35')
    expect(manyHtml).toContain('Filter discovered models…')
    expect(manyHtml).toContain('role="combobox"')
    expect(manyHtml).not.toContain('showing 24 of')
  })

  test('Fix 3: ProviderAccountsCenter is keyed by workspace/session scope and reloads across local↔SSH switches', () => {
    const overlaysSrc = readFileSync(join(rendererDir, 'Overlays.tsx'), 'utf8')
    const accountsSrc = readFileSync(join(rendererDir, 'ProviderAccountsCenter.tsx'), 'utf8')
    expect(overlaysSrc).toContain("key={`${snap.storageScope || 'local'}:${snap.cwd || ''}:${snap.sessionKey}`}")
    expect(accountsSrc).toContain('[accounts,sessionKey]')
    expect(accountsSrc).toContain('loadEpoch.current!==epoch')
  })

  test('Fix 4: Setup checklist reopens via explicit Open… even when readiness is already satisfied', () => {
    // Auto-show stays suppressed once ready
    expect(shouldShowSetup({ visible: true, manualOpen: false, noWorkspace: false, settingsOpen: false, ready: true })).toBe(false)
    // Manual reopen works even when ready
    expect(shouldShowSetup({ visible: true, manualOpen: true, noWorkspace: false, settingsOpen: false, ready: true })).toBe(true)
    // Still respects settingsOpen and noWorkspace
    expect(shouldShowSetup({ visible: true, manualOpen: true, noWorkspace: false, settingsOpen: true, ready: true })).toBe(false)
    expect(shouldShowSetup({ visible: true, manualOpen: true, noWorkspace: true, settingsOpen: false, ready: true })).toBe(false)
  })

  test('Fix 5: MemoryGuardSetting handles float drift with epsilon equality and locks inputs while busy or offline', () => {
    // 11.7 GB * 1024 = 11980.8 MB vs saved 11981 MB (and 12000 MB rounded to 11.7 GB)
    expect(isMemoryGuardDraftUnchanged('11.7', 11981)).toBe(true)
    expect(isMemoryGuardDraftUnchanged('11.7', 12000)).toBe(true)
    expect(isMemoryGuardDraftUnchanged('11.75', 12032)).toBe(true)
    expect(isMemoryGuardDraftUnchanged('12', 11981)).toBe(false)
    expect(isMemoryGuardDraftUnchanged('11.8', 11981)).toBe(false)
    expect(isMemoryGuardDraftUnchanged('', 11981)).toBe(false)
    expect(isMemoryGuardDraftUnchanged('abc', 11981)).toBe(false)

    const memSrc = readFileSync(join(rendererDir, 'MemoryGuardSetting.tsx'), 'utf8')
    expect(memSrc).toContain('const locked = busy || !online')
    expect(memSrc).toContain('disabled={locked}')
    expect(memSrc).toContain('disabled={locked || !draft.trim() || unchanged}')
  })

  test('Fix 6: Monitors & Deliveries remove duplicate dialog headers and conflicting footer button bars inside ActionDialog', () => {
    const monitorsSrc = readFileSync(join(rendererDir, 'Monitors.tsx'), 'utf8')
    const deliveriesSrc = readFileSync(join(rendererDir, 'Deliveries.tsx'), 'utf8')

    expect(monitorsSrc).not.toContain('<h4>Create monitor</h4>')
    expect(monitorsSrc).not.toContain('<h4>Monitor details</h4>')
    expect(monitorsSrc).toContain('createFormRef')
    expect(monitorsSrc).toContain('done={dialogDone}')
    expect(monitorsSrc).toContain('confirmLabel={dialogConfirmLabel}')

    expect(deliveriesSrc).not.toContain('role="group" aria-label="Confirm delivery action"')
    expect(deliveriesSrc).toContain('done={confirm===null}')
    expect(deliveriesSrc).toContain('confirmLabel={dialogConfirmLabel}')
  })

  test('Fix 7: SupervisorPage unconfigured model chips focus the inline chooser instead of silently no-oping', () => {
    const supervisorSrc = readFileSync(join(rendererDir, 'SupervisorPage.tsx'), 'utf8')
    expect(supervisorSrc).toContain('if (tab === \'model\' && inlineChooser)')
    expect(supervisorSrc).toContain('focusInlineChooser()')

    const noop = (): void => {}
    const unconfiguredState: HubState = {
      ...initialHubState(),
      connection: { state: 'live' },
      supervisor: { phase: 'ready', busy: false, messages: [], confirmations: [] },
      supervisorConfig: { ...initialSupervisorConfig(), phase: 'ready', status: 'unconfigured', canEdit: true },
    }
    const html = renderToStaticMarkup(createElement(SupervisorView, {
      state: unconfiguredState,
      hubLabel: 'hub.test',
      now: Date.now(),
      onSend: async () => true,
      onRespond: noop,
      onReopen: noop,
      onOpenSettings: noop,
      chooserOpen: false,
      onChooserChange: noop,
      onChooserShown: noop,
      onOptions: noop,
      onSaveModel: async () => true,
    }))
    expect(html).toContain('jump to model setup')
    expect(html).toContain('Choose a model for XERXES in the setup panel above')

    // Unsupported hub disables model chips cleanly
    const unsupportedState: HubState = {
      ...unconfiguredState,
      supervisor: { phase: 'unsupported', busy: false, messages: [], confirmations: [] },
    }
    const unsupportedHtml = renderToStaticMarkup(createElement(SupervisorView, {
      state: unsupportedState,
      hubLabel: 'hub.test',
      now: Date.now(),
      onSend: async () => true,
      onRespond: noop,
      onReopen: noop,
      onOpenSettings: noop,
      chooserOpen: false,
      onChooserChange: noop,
      onChooserShown: noop,
      onOptions: noop,
      onSaveModel: async () => true,
    }))
    expect(unsupportedHtml).toMatch(/<button[^>]*class="xpill"[^>]*disabled=""[^>]*>[\s\S]*?Model/)
  })

  test('Fix 8: Command Palette displays accurate ⌥⌘N shortcut hint for New task wizard (not ⇧⌘N)', () => {
    const html = renderToStaticMarkup(createElement(CommandPalette, { snap: makeSnapshot({ paletteOpen: true }) }))
    expect(html).toContain('New task…<span class="prow__kbd">⌥⌘N</span>')
    expect(html).not.toContain('New task…<span class="prow__kbd">⇧⌘N</span>')
    expect(html).toContain('New blank task<span class="prow__kbd">⌘N</span>')
    expect(html).toContain('Settings…<span class="prow__kbd">⌘,</span>')
  })

  test('Fix 9: Destructive & bulk actions use inline two-step confirmation without blocking window.confirm', () => {
    const hubSrc = readFileSync(join(rendererDir, 'HubPhonePanel.tsx'), 'utf8')
    const remoteSrc = readFileSync(join(rendererDir, 'RemoteProviders.tsx'), 'utf8')
    const learnedSrc = readFileSync(join(rendererDir, 'LearnedPanel.tsx'), 'utf8')
    const overlaysSrc = readFileSync(join(rendererDir, 'Overlays.tsx'), 'utf8')

    expect(hubSrc).toContain('Confirm forget')
    expect(remoteSrc).toContain('Confirm revoke')
    expect(learnedSrc).toContain('Confirm approve ({lessons.length})')
    expect(learnedSrc).toContain('Confirm approve ({loadedPendingProposals})')
    expect(overlaysSrc).toContain('Confirm delete')
    expect(overlaysSrc).not.toContain('window.confirm(')
  })

  test('Fix 10: GeneralCard has clear sections & non-contradictory copy, HubPhonePanel uses h2.modal__title, and static tags avoid .chipbtn', () => {
    Object.defineProperty(globalThis, 'window', { configurable: true, value: { xerxes: {} } })

    const generalHtml = renderToStaticMarkup(
      createElement(SettingsModal, { snap: makeSnapshot({ settingsOpen: true, settingsTab: 'general' }) }),
    )
    expect(generalHtml).toContain('>Appearance</h3>')
    expect(generalHtml).toContain('>Workspace &amp; Session</h3>')
    expect(generalHtml).toContain('>Behavior &amp; Safety</h3>')
    expect(generalHtml).not.toContain('nothing here is per-task')
    expect(generalHtml).toContain('Plan this task <span class="provcard__badge provcard__badge--quiet">This task</span>')

    const hubHtml = renderToStaticMarkup(
      createElement(HubPhoneView, {
        available: true,
        status: EMPTY_STATUS,
        address: '',
        onAddress: () => {},
        saving: false,
        onSave: () => {},
        onForget: () => {},
        link: { phase: 'idle' },
        onLink: () => {},
        machines: { phase: 'idle', machines: [] },
        onRefreshMachines: () => {},
        pair: { phase: 'idle' },
        onPair: () => {},
        onConfirm: () => {},
      }),
    )
    expect(hubHtml).toContain('<h2 id="hubphone-title" class="modal__title">Hub &amp; phone</h2>')

    const presetsHtml = renderToStaticMarkup(
      createElement(SettingsModal, {
        snap: makeSnapshot({
          settingsOpen: true,
          settingsTab: 'agents',
          currentAgentPreset: 'default',
          agentPresets: [
            { id: 'default', name: 'Default', description: 'Main', trust: 'system', isDefault: true, manageable: false },
          ],
        }),
      }),
    )
    expect(presetsHtml).toContain('class="provcard__badge provcard__badge--accent">Default</span>')
    expect(presetsHtml).toContain('class="provcard__badge provcard__badge--quiet">This session</span>')
    expect(presetsHtml).not.toMatch(/<span class="chipbtn"/)

    const modelsHtml = renderToStaticMarkup(
      createElement(SettingsModal, {
        snap: makeSnapshot({
          settingsOpen: true,
          settingsTab: 'models',
          models: [{ id: 'gpt-5', provider: 'openai' }],
          providers: [{ name: 'openai', provider: 'openai', model: 'gpt-5', active: true, signsIn: false }],
        }),
      }),
    )
    expect(modelsHtml).toContain('class="provcard__badge provcard__badge--quiet">1 model</span>')
    expect(modelsHtml).not.toMatch(/<span class="chipbtn"/)

    const channelsHtml = renderToStaticMarkup(
      createElement(ChannelsCard, {
        snap: makeSnapshot({
          channels: [{ name: 'telegram', adapterName: 'telegram-bot', enabled: true }],
        }),
      }),
    )
    expect(channelsHtml).toContain('class="provcard__badge provcard__badge--quiet">telegram-bot</span>')
    expect(channelsHtml).not.toMatch(/<span class="chipbtn"/)
  })

  test('Fix 11: Humanizes developer tokens, structured Learned outcomes, LspPanel copy, RunMode copy, and cross-platform labels', () => {
    // Enum label humanization
    expect(accountKindLabel('api_key')).toBe('API key')
    expect(accountKindLabel('claude')).toBe('Claude sign-in')
    expect(agyModeLabel('gemini_api_key')).toBe('Gemini API key')
    expect(agyModeLabel('adc')).toBe('Google Cloud ADC')
    expect(deviceStatusText('active')).toBe('Active')
    expect(deviceStatusText('needs_review')).toBe('Needs review')

    const agyHtml = renderToStaticMarkup(
      createElement(AgyCredentialCard, {
        status: {
          agyInstalled: true,
          sources: [{ mode: 'gemini_api_key', label: 'Gemini API key', available: true, detail: 'Ready', headlessSafe: true }],
          recommendedMode: 'gemini_api_key',
          canSync: true,
          destination: 'devbox',
          syncedDestination: 'devbox',
          syncedMode: 'gemini_api_key',
        },
        selectedMode: 'gemini_api_key',
        onSelectMode: () => {},
        busy: false,
        onSync: () => {},
      }),
    )
    expect(agyHtml).toContain('Synced · Gemini API key')
    expect(agyHtml).not.toContain('gemini_api_key')

    // LspPanel uses "Stop server" instead of "Release host"
    const lspSrc = readFileSync(join(rendererDir, 'LspPanel.tsx'), 'utf8')
    expect(lspSrc).toContain('>Stop server<')
    expect(lspSrc).not.toContain('Release host')

    // Structured review outcome instead of JSON.stringify dump
    const entries = formatOutcomeEntries({ ok: false, error: 'Missing frontmatter', file_path: '.xerxes/skills/foo/SKILL.md' })
    expect(entries).toEqual([
      { label: 'Result', value: 'Failed' },
      { label: 'Error', value: 'Missing frontmatter' },
      { label: 'Path', value: '.xerxes/skills/foo/SKILL.md' },
    ])
    const storedProposal: StoredReflectionProposal = {
      id: 'p-out',
      workspace: '/repo',
      status: 'failed',
      createdAt: 1,
      updatedAt: 2,
      outcome: { ok: false, error: 'Missing frontmatter', path: '.xerxes/skills/foo/SKILL.md' },
      proposal: {
        kind: 'skill_create',
        target: 'foo',
        rationale: '[OBSERVATION] repeat check',
        evidence: ['[OBSERVATION] repeat check'],
        skill: { description: 'desc', instructions: 'inst' },
      },
    }
    const outcomeHtml = renderToStaticMarkup(
      createElement(ProposalCard, {
        stored: storedProposal,
        snap: makeSnapshot(),
        actions: {
          refreshLearning: async () => {},
          reviewLearning: async () => {},
          reviewAllLearning: async () => ({ approved: 0, failed: 0 }),
          openLearning: () => {},
        },
      }),
    )
    expect(outcomeHtml).toContain('class="learning-outcome"')
    expect(outcomeHtml).toContain('<dt>Result</dt><dd>Failed</dd>')
    expect(outcomeHtml).toContain('<dt>Error</dt><dd>Missing frontmatter</dd>')
    expect(outcomeHtml).not.toContain('&quot;ok&quot;')

    // RunModeSection & runModeView remove "local-model" and "Prism's Hugging Face"
    const runModeHtml = renderToStaticMarkup(
      createElement(RunModeSection, {
        snap: makeSnapshot({
          runMode: { mode: 'hybrid', worker: { profile: 'claude-code', model: 'opus', local: false }, restartPending: false, lockedBy: [] },
          localSetup: {
            status: {
              ready: true,
              engine: { name: 'llama-server', label: 'llama.cpp', found: true, version: '1.0', newEnough: true },
              model: { id: 'qwen-local', label: 'Qwen 3 14B', bytes: 8_000_000_000, downloaded: false },
              server: { running: false, port: null, contextLength: null },
              memoryBytes: 17_179_869_184,
              blocker: null,
            },
            job: null,
            error: '',
            loading: false,
          },
        }),
      }),
    )
    expect(runModeHtml).not.toContain('local-model')
    expect(runModeHtml).not.toContain('Prism')
    expect(runModeHtml).toContain('Download 8.0 GB and set up local model')
    expect(runModeHtml).toContain('Downloaded from Hugging Face and verified by SHA-256.')

    // Platform-aware labels: Linux/Windows must say "This computer" / "this computer" (never "Mac")
    setNavigatorPlatform('Linux x86_64', 'Mozilla/5.0 (X11; Linux x86_64)')
    expect(localMachineLabel()).toBe('this computer')
    expect(localMachineLabel(true)).toBe('This computer')
    expect(localMachineNoun(true)).toBe('Computer')
    expect(runModeChoices()[0]?.summary).toContain('this computer')
    expect(runModeChoices()[0]?.summary).not.toContain('Mac')

    const linuxRelayLive = renderToStaticMarkup(
      createElement(RelayBadge, {
        relay: { bound: true, live: true, destination: 'devbox', profile: 'codex', model: 'gpt-5', requests: 1, inFlight: 0 },
      }),
    )
    expect(linuxRelayLive).toContain('via this computer')
    expect(linuxRelayLive).not.toContain('Mac')

    const linuxRelayEnded = renderToStaticMarkup(
      createElement(RelayBadge, {
        relay: { bound: true, live: false, destination: 'devbox', profile: 'codex', model: 'gpt-5', requests: 1, inFlight: 0 },
      }),
    )
    expect(linuxRelayEnded).toContain('Computer not connected')
    expect(linuxRelayEnded).not.toContain('Mac')

    const linuxHub = renderToStaticMarkup(
      createElement(HubPhoneView, {
        available: true,
        status: {
          target: 'xerxes@203.0.113.10',
          origin: 'https://hub.example.ts.net',
          wsUrl: 'wss://hub.example.ts.net/api/hub/v1/ws',
          fingerprint: 'SHA256:abc',
          hubVersion: '1.0.0',
          linkedMachineId: null,
          hasSession: false,
        },
        address: 'xerxes@203.0.113.10',
        onAddress: () => {},
        saving: false,
        onSave: () => {},
        onForget: () => {},
        link: { phase: 'idle' },
        onLink: () => {},
        machines: { phase: 'idle', machines: [] },
        onRefreshMachines: () => {},
        pair: { phase: 'idle' },
        onPair: () => {},
        onConfirm: () => {},
        devices: {
          phase: 'ready',
          devices: [{
            id: 'd1', userId: 'u1', displayName: 'Alex', label: 'Phone', status: 'active',
            createdAt: '2026-10-01T00:00:00Z', lastSeenAt: null, revokedAt: null,
          }],
        },
      }),
    )
    expect(linuxHub).toContain('<strong>This computer</strong>')
    expect(linuxHub).toContain('>Link this computer<')
    expect(linuxHub).toContain('>Active<')
    expect(linuxHub).not.toContain('this Mac')
    expect(linuxHub).not.toContain('This Mac')

    const linuxAccounts = renderToStaticMarkup(
      createElement(ProviderAccountsCenter, {
        accounts: async () => ({ accounts: [] }),
        sessionKey: 'sk-1',
        turnActive: false,
        changed: () => {},
      }),
    )
    expect(linuxAccounts).toContain('One account center for this computer and every SSH workspace')
    expect(linuxAccounts).not.toContain('this Mac')

    // LearnedPanel renders clean approve-all button initially
    const learnedHtml = renderToStaticMarkup(
      createElement(LearnedPanel, {
        snap: makeSnapshot({
          learningCounts: { pending_proposals: 1, pending_lessons: 0 },
          learning: {
            workspace: '/repo',
            proposals: [{ ...storedProposal, status: 'pending', outcome: undefined }],
            lessons: [],
            counts: { pending_proposals: 1, pending_lessons: 0 },
          },
        }),
        actions: {
          refreshLearning: async () => {},
          reviewLearning: async () => {},
          reviewAllLearning: async () => ({ approved: 0, failed: 0 }),
          openLearning: () => {},
        },
      }),
    )
    expect(learnedHtml).toContain('Approve all (1)')

    // AgentIntelligenceCard hybrid copy on Linux uses "this computer" and no "local-model"
    const intelSrc = readFileSync(join(rendererDir, 'AgentIntelligenceCard.tsx'), 'utf8')
    expect(intelSrc).toContain('localMachineLabel()')
    expect(intelSrc).not.toContain('local-model')
    expect(intelSrc).not.toContain('this Mac')
    expect(AgentIntelligenceCard).toBeDefined()

    // On macOS, platform helper returns Mac labels
    setNavigatorPlatform('MacIntel', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)')
    expect(localMachineLabel()).toBe('this Mac')
    expect(localMachineLabel(true)).toBe('This Mac')
    expect(localMachineNoun(true)).toBe('Mac')
  })
})
