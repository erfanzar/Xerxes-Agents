// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  CATEGORIES,
  COMMAND_REGISTRY,
  listCommands,
  telegramBotCommands,
} from '../src/bridge/commands.js'
import {
  CLAUDE_CODE_DEFAULT_MODEL,
  CLAUDE_CODE_PROFILE_NAME,
  ProfileStore,
  type ProviderProfile,
  type SaveProfileInput,
} from '../src/bridge/profiles.js'
import {
  BridgeServer,
  type BridgeProfileStore,
  type BridgeRuntimePort,
  type BridgeRuntimeTurnInput,
  type BridgeServerOutput,
} from '../src/bridge/server.js'
import { BridgeSession, type BridgeSessionStore } from '../src/bridge/session.js'
import {
  toDaemonWireEventName,
  toKimiWireEventName,
  wireEventFrame,
  type BridgeWireFrame,
} from '../src/bridge/wireEvents.js'
import type { StreamEvent } from '../src/streaming/events.js'

const KIMI_WIRE_EVENT_NAMES: Readonly<Record<string, string>> = Object.freeze({
  init_done: 'InitDone',
  turn_begin: 'TurnBegin',
  turn_end: 'TurnEnd',
  step_begin: 'StepBegin',
  step_end: 'StepEnd',
  step_interrupted: 'StepInterrupted',
  steer_input: 'SteerInput',
  compaction_begin: 'CompactionBegin',
  compaction_end: 'CompactionEnd',
  hook_triggered: 'HookTriggered',
  hook_resolved: 'HookResolved',
  mcp_loading_begin: 'MCPLoadingBegin',
  mcp_loading_end: 'MCPLoadingEnd',
  btw_begin: 'BtwBegin',
  btw_end: 'BtwEnd',
  text_part: 'TextPart',
  think_part: 'ThinkPart',
  image_url_part: 'ImageURLPart',
  audio_url_part: 'AudioURLPart',
  video_url_part: 'VideoURLPart',
  tool_call: 'ToolCall',
  tool_call_part: 'ToolCallPart',
  tool_result: 'ToolResult',
  tool_call_request: 'ToolCallRequest',
  approval_request: 'ApprovalRequest',
  approval_response: 'ApprovalResponse',
  question_request: 'QuestionRequest',
  question_response: 'QuestionResponse',
  status_update: 'StatusUpdate',
  notification: 'Notification',
  plan_display: 'PlanDisplay',
  subagent_event: 'SubagentEvent',
})

class MemorySessionStore implements BridgeSessionStore {
  readonly records = new Map<string, unknown>()

  read(sessionId: string): unknown | undefined {
    return this.records.get(sessionId)
  }

  write(sessionId: string, record: Readonly<Record<string, unknown>>): void {
    this.records.set(sessionId, record)
  }
}

class RecordingOutput implements BridgeServerOutput {
  readonly legacy: Array<{ readonly data: Record<string, unknown>; readonly event: string }> = []
  readonly wire: BridgeWireFrame[] = []

  writeLegacy(event: { readonly data: Record<string, unknown>; readonly event: string }): void {
    this.legacy.push(event)
  }

  writeWire(frame: BridgeWireFrame): void {
    this.wire.push(frame)
  }
}

class InMemoryProfiles implements BridgeProfileStore {
  private activeName: string | undefined
  private readonly profiles = new Map<string, ProviderProfile>()

  constructor(profile?: ProviderProfile) {
    if (profile) {
      this.profiles.set(profile.name, copyProfile(profile))
      this.activeName = profile.name
    }
  }

  active(): ProviderProfile | undefined {
    const profile = this.activeName === undefined ? undefined : this.profiles.get(this.activeName)
    return profile === undefined ? undefined : copyProfile(profile)
  }

  delete(name: string): boolean {
    const deleted = this.profiles.delete(name)
    if (deleted && this.activeName === name) this.activeName = undefined
    return deleted
  }

  list(): Array<ProviderProfile & { readonly active?: boolean }> {
    return [...this.profiles.values()].map(profile => ({
      ...copyProfile(profile),
      active: profile.name === this.activeName,
    }))
  }

  save(input: SaveProfileInput): ProviderProfile {
    const profile: ProviderProfile = {
      name: input.name,
      api_key: input.apiKey,
      base_url: input.baseUrl,
      model: input.model,
      provider: input.provider ?? 'custom',
      sampling: { ...input.sampling },
    }
    this.profiles.set(profile.name, profile)
    if (input.setActive ?? true) this.activeName = profile.name
    return copyProfile(profile)
  }

  setActive(name: string): boolean {
    if (!this.profiles.has(name)) return false
    this.activeName = name
    return true
  }

  updateActiveModel(model: string): ProviderProfile | undefined {
    const profile = this.active()
    if (!profile) return undefined
    const updated = { ...profile, model }
    this.profiles.set(updated.name, updated)
    return copyProfile(updated)
  }
}

class RecordingRuntime implements BridgeRuntimePort {
  cancelAllCalls = 0
  cancelCalls = 0
  readonly configurations: Readonly<Record<string, unknown>>[] = []
  readonly turnInputs: Array<Readonly<Record<string, unknown>>> = []
  private resolveTurnStarted: (() => void) | undefined
  readonly turnStarted = new Promise<void>(resolve => { this.resolveTurnStarted = resolve })
  waitForCancellation = false

  cancel(): void {
    this.cancelCalls += 1
  }

  cancelAll(): number {
    this.cancelAllCalls += 1
    return 1
  }

  configure(config: Readonly<Record<string, unknown>>): void {
    this.configurations.push({ ...config })
  }

  async *streamTurn(input: BridgeRuntimeTurnInput, signal: AbortSignal): AsyncGenerator<StreamEvent> {
    this.turnInputs.push({ config: { ...input.config }, text: input.text })
    this.resolveTurnStarted?.()
    if (this.waitForCancellation) {
      if (!signal.aborted) {
        await new Promise<void>(resolve => signal.addEventListener('abort', () => resolve(), { once: true }))
      }
      return
    }
    yield {
      type: 'turn_done',
      model: String(input.config.model ?? ''),
      toolCallsCount: 0,
      usage: { inputTokens: 0, outputTokens: 0 },
    }
  }
}

function bridge(options: {
  readonly output?: RecordingOutput
  readonly profiles?: BridgeProfileStore
  readonly runtime?: RecordingRuntime
  readonly sessionStore?: MemorySessionStore
  readonly wireMode?: boolean
} = {}): {
  readonly output: RecordingOutput
  readonly profiles: BridgeProfileStore
  readonly runtime: RecordingRuntime
  readonly server: BridgeServer
  readonly sessionStore: MemorySessionStore
} {
  const output = options.output ?? new RecordingOutput()
  const profiles = options.profiles ?? new InMemoryProfiles()
  const runtime = options.runtime ?? new RecordingRuntime()
  const sessionStore = options.sessionStore ?? new MemorySessionStore()
  const server = new BridgeServer({
    output,
    profileStore: profiles,
    runtime,
    session: new BridgeSession({
      clock: () => new Date('2026-07-13T12:00:00.000Z'),
      cwd: '/initial-project',
      sessionId: 'feedface',
      store: sessionStore,
    }),
    wireMode: options.wireMode ?? false,
  })
  return { output, profiles, runtime, server, sessionStore }
}

test('native profiles retain the built-in Claude Code default, persistence, and provider detection', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bun-bridge-profiles-'))
  try {
    const store = new ProfileStore(join(directory, 'profiles.json'))

    expect(store.list()).toContainEqual({
      active: true,
      api_key: '',
      base_url: 'claude-code://local',
      model: CLAUDE_CODE_DEFAULT_MODEL,
      name: CLAUDE_CODE_PROFILE_NAME,
      provider: 'claude-code',
      sampling: {},
    })
    expect(store.active()).toEqual({
      api_key: '',
      base_url: 'claude-code://local',
      model: CLAUDE_CODE_DEFAULT_MODEL,
      name: CLAUDE_CODE_PROFILE_NAME,
      provider: 'claude-code',
      sampling: {},
    })

    expect(store.setActive(CLAUDE_CODE_PROFILE_NAME)).toBeTrue()
    expect(store.updateActiveModel('claude-code/opus')).toMatchObject({ model: 'claude-code/opus' })
    expect(store.active()).toMatchObject({ name: CLAUDE_CODE_PROFILE_NAME, model: 'claude-code/opus' })

    expect(store.save({
      apiKey: 'test-key',
      baseUrl: 'https://openrouter.ai/api/v1',
      model: 'anthropic/claude-sonnet-4.5',
      name: 'openrouter',
    })).toMatchObject({ provider: 'openrouter' })
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
})

test('bridge command metadata remains a complete, non-conflicting Telegram-safe registry', () => {
  expect(COMMAND_REGISTRY.length).toBeGreaterThanOrEqual(50)
  expect(listCommands()).toEqual([...COMMAND_REGISTRY])

  const canonicalNames = new Set<string>()
  for (const command of COMMAND_REGISTRY) {
    expect(CATEGORIES).toContain(command.category)
    expect(canonicalNames.has(command.name)).toBeFalse()
    canonicalNames.add(command.name)
  }
  for (const command of COMMAND_REGISTRY) {
    for (const alias of command.aliases) {
      expect(canonicalNames.has(alias)).toBeFalse()
    }
  }

  const telegram = telegramBotCommands()
  const telegramNames = telegram.map(command => command.command)
  expect(telegramNames).toEqual(expect.arrayContaining(['new', 'model', 'compact', 'skills', 'exit']))
  expect(telegramNames).not.toContain('skin')
  expect(telegramNames).not.toContain('verbose')
  expect(new Set(telegramNames).size).toBe(telegramNames.length)
  for (const command of telegram) {
    expect(command.command).toMatch(/^[a-z0-9_]{1,32}$/u)
    expect(command.description.length).toBeLessThanOrEqual(256)
  }
})

test('bridge modes reach the native runtime, plan mode wins, and init emits the complete client contract', async () => {
  const { output, runtime, server } = bridge({ wireMode: true })

  expect(await server.dispatch({
    method: 'init',
    params: { model: 'gpt-4.1', mode: 'researcher', project_dir: '/native-bridge-project' },
  })).toEqual({ accepted: true })
  expect(server.configuration).toMatchObject({ mode: 'researcher', plan_mode: false })
  expect(server.session).toMatchObject({ cwd: '/native-bridge-project', interactionMode: 'researcher', planMode: false })

  const init = output.wire.find(frame => frame.method === 'event' && frame.params.type === 'InitDone')
  if (!init || init.method !== 'event') throw new Error('Expected a wire InitDone event')
  expect(init.params.payload).toMatchObject({
    agent_name: 'default',
    context_limit: expect.any(Number),
    cwd: '/native-bridge-project',
    git_branch: '',
    head_hash: '',
    mode: 'researcher',
    model: 'gpt-4.1',
    session_id: 'feedface',
    skill_descriptions: {},
    skills: [],
    version: '0.3.0',
  })

  await submitAndWait(server, 'research')
  expect(runtime.turnInputs.at(-1)?.config).toMatchObject({ mode: 'researcher', plan_mode: false })

  expect(await server.dispatch({ method: 'set_plan_mode', params: { enabled: true, mode: 'code' } })).toEqual({ accepted: true })
  await submitAndWait(server, 'plan')
  expect(runtime.turnInputs.at(-1)?.config).toMatchObject({ mode: 'plan', plan_mode: true })

  expect(await server.dispatch({ method: 'set_mode', params: { mode: 'code' } })).toEqual({ accepted: true })
  await submitAndWait(server, 'implement')
  expect(runtime.turnInputs.at(-1)?.config).toMatchObject({ mode: 'code', plan_mode: false })

  expect(await server.dispatch({ method: 'set_mode', params: { mode: 'goal' } })).toEqual({ accepted: true })
  await submitAndWait(server, 'verify')
  expect(runtime.turnInputs.at(-1)?.config).toMatchObject({ mode: 'objective', plan_mode: false })
  expect(server.session).toMatchObject({ interactionMode: 'objective', planMode: false })
})

test('bridge model selection persists to the active profile and cancellation also reaches subagents', async () => {
  const profile: ProviderProfile = {
    api_key: '',
    base_url: 'claude-code://local',
    model: CLAUDE_CODE_DEFAULT_MODEL,
    name: CLAUDE_CODE_PROFILE_NAME,
    provider: 'claude-code',
    sampling: {},
  }
  const profiles = new InMemoryProfiles(profile)
  const runtime = new RecordingRuntime()
  const { server } = bridge({ profiles, runtime })

  expect(await server.dispatch({ method: 'init', params: {} })).toEqual({ accepted: true })
  expect(await server.dispatch({ method: 'slash', params: { command: '/model claude-code/opus' } })).toEqual({ accepted: true })
  expect(profiles.active()).toMatchObject({ model: 'claude-code/opus' })
  expect(server.configuration).toMatchObject({ model: 'claude-code/opus' })

  runtime.waitForCancellation = true
  expect(await server.dispatch({ method: 'query', params: { text: 'wait for cancellation' } })).toEqual({ accepted: true })
  await runtime.turnStarted
  expect(await server.dispatch({ method: 'cancel', params: {} })).toEqual({ accepted: true })
  await server.waitForIdle()
  expect(runtime.cancelCalls).toBe(1)
  expect(runtime.cancelAllCalls).toBe(1)
})

test('bridge initialization applies a project directory and restores a saved session directory', async () => {
  const sessionStore = new MemorySessionStore()
  sessionStore.records.set('saved-session', {
    cwd: '/persisted-project',
    messages: [{ role: 'user', content: 'resume me' }],
    model: 'gpt-4.1',
    session_id: 'saved-session',
  })
  const { server } = bridge({ sessionStore })

  expect(await server.dispatch({
    method: 'init',
    params: { model: 'gpt-4.1', project_dir: '/requested-project' },
  })).toEqual({ accepted: true })
  expect(server.session.cwd).toBe('/requested-project')

  expect(await server.dispatch({
    method: 'init',
    params: { model: 'gpt-4.1', resume_session_id: 'saved-session' },
  })).toEqual({ accepted: true })
  expect(server.session.cwd).toBe('/persisted-project')
})

test('the bridge preserves every documented legacy wire-name pair in both directions', () => {
  for (const [internal, kimi] of Object.entries(KIMI_WIRE_EVENT_NAMES)) {
    expect(toKimiWireEventName(internal)).toBe(kimi)
    expect(toDaemonWireEventName(kimi)).toBe(internal)
    expect(wireEventFrame(internal, {}, 'kimi').params.type).toBe(kimi)
    expect(wireEventFrame(kimi, {}, 'daemon').params.type).toBe(internal)
  }
})

async function submitAndWait(server: BridgeServer, text: string): Promise<void> {
  expect(await server.dispatch({ method: 'query', params: { text } })).toEqual({ accepted: true })
  await server.waitForIdle()
}

function copyProfile(profile: ProviderProfile): ProviderProfile {
  return { ...profile, sampling: { ...profile.sampling } }
}
