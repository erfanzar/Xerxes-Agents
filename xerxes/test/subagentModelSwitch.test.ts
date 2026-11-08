// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, spyOn, test } from 'bun:test'
import { createHash } from 'node:crypto'
import { mkdtemp, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { BUILTIN_AGENTS } from '../src/agents/definitions.js'
import { SubAgentManager, type SubagentTaskRunRequest } from '../src/agents/subagentManager.js'
import { ValidationError } from '../src/core/errors.js'
import { DaemonSubagentEventBus } from '../src/daemon/subagentEvents.js'
import { createNativeSubagentHost, type NativeSubagentHostOptions } from '../src/daemon/subagentHost.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { DaemonTranscriptStore } from '../src/session/daemonTranscript.js'
import { registerClaudeAgentTools } from '../src/tools/claudeTools/agentOps.js'
import type { JsonObject } from '../src/types/toolCalls.js'

class RecordingClient implements LlmClient {
  readonly requests: CompletionRequest[] = []
  beforeReply: (() => Promise<void>) | undefined

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(structuredClone(request))
    await this.beforeReply?.()
    yield { content: `Reply from ${request.model}` }
  }
}

async function fixture() {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-model-switch-'))
  const oldClient = new RecordingClient()
  const newClient = new RecordingClient()
  const registry = new ToolRegistry()
  const transcripts = new DaemonTranscriptStore({ currentProjectDirectory: directory, directory: join(directory, 'sessions') })
  let routeSuffix = ''
  let duringValidation: (() => Promise<void>) | undefined
  const validations: Array<{ profile: string; model: string; effort: string | undefined }> = []
  const route = (profile: string, model: string) => createHash('sha256').update(`${profile}:${model}:${routeSuffix}`).digest('hex')
  const definition = BUILTIN_AGENTS.get('coder')!
  const options: NativeSubagentHostOptions = {
    agentDefinitions: new Map([['coder', { ...definition, allowedTools: ['ReadFile'], tools: ['ReadFile'], excludeTools: ['WriteFile'] }]]),
    autoCompactThreshold: 0,
    cwd: directory,
    eventBus: new DaemonSubagentEventBus(),
    llm: oldClient,
    model: 'codex-old',
    reasoningEffort: 'medium',
    permissionMode: 'accept-all',
    tools: [
      { type: 'function', function: { name: 'ReadFile', description: 'Fixture read', parameters: { type: 'object', properties: {} } } },
      { type: 'function', function: { name: 'WriteFile', description: 'Fixture write', parameters: { type: 'object', properties: {} } } },
    ],
    toolExecutor: registry,
    transcriptStore: transcripts,
    inheritedProviderRoute: route('inherited', 'codex-old'),
    resolveProviderRoute: route,
    validateProviderSelection: async (profile, model, effort) => {
      validations.push({ profile, model, effort })
      if (!['old', 'new'].includes(profile)) throw new Error('unknown provider profile')
      if (!['codex-old', 'codex-new'].includes(model)) throw new Error('unknown model')
      if (effort && !['medium', 'high'].includes(effort)) throw new Error('invalid reasoning effort')
      await duringValidation?.()
    },
    resolveProviderProfile: (profile, model, expectedRoute) => {
      if (expectedRoute !== route(profile, model)) throw new Error('provider route changed')
      return { llm: profile === 'new' ? newClient : oldClient }
    },
  }
  const host = createNativeSubagentHost(options)
  registerClaudeAgentTools(registry, { manager: host.managerPort })
  const execute = (name: string, args: JsonObject, sessionId = 'parent') => registry.execute({
    id: crypto.randomUUID(), type: 'function', function: { name, arguments: args },
  }, { sessionId, metadata: {} })
  const spawn = async () => {
    const child = await host.managerPort.spawn({
      agent: { id: 'coder', model: 'codex-old', providerProfile: 'old', reasoningEffort: 'medium' },
      sourceAgentId: 'parent', nickname: 'worker', promptProfile: 'coder', message: 'Original brief', title: 'Saved worker',
    })
    await host.managerPort.wait([child.id], 5_000)
    return child
  }
  return {
    directory, options, host, oldClient, newClient, transcripts, validations, route, execute, spawn,
    setValidation(callback: (() => Promise<void>) | undefined) { duringValidation = callback },
    changeRoute() { routeSuffix += 'changed' },
    async close() { await host.manager.shutdown(); await rm(directory, { recursive: true, force: true }) },
  }
}

for (const tool of ['AgentTool', 'SendMessageTool']) {
  test(`${tool} changes terminal model/profile/effort under the same id and persisted conversation after a main-model switch`, async () => {
    const f = await fixture()
    try {
      const child = await f.spawn()
      f.host.reconfigure({ ...f.options, llm: f.newClient, model: 'codex-new', inheritedProviderRoute: f.route('inherited', 'codex-new') })
      const selectors = { model: 'codex-new', provider_profile: 'new', reasoning_effort: 'high' }
      const args = tool === 'AgentTool'
        ? { resume: child.id, prompt: 'Continue with the new model', wait: false, ...selectors }
        : { target: 'worker', message: 'Continue with the new model', ...selectors }
      const receipt = JSON.parse(await f.execute(tool, args))
      expect(receipt).toMatchObject({ id: child.id, model: 'codex-new', attempt: 1 })
      await f.host.managerPort.wait([child.id], 5_000)
      expect(f.oldClient.requests).toHaveLength(1)
      expect(f.newClient.requests).toHaveLength(1)
      const request = f.newClient.requests[0]!
      expect(request.model).toBe('codex-new')
      expect(request.thinking?.effort).toBe('high')
      expect(request.messages.filter(message => message.role === 'user').map(message => message.content)).toEqual(['Original brief', 'Continue with the new model'])
      expect(request.messages.some(message => message.role === 'assistant' && message.content === 'Reply from codex-old')).toBe(true)
      expect(request.tools?.map(tool => tool.function.name)).toEqual(['ReadFile'])
      const saved = f.host.managerPort.listHandles()[0]!
      expect(saved).toMatchObject({ id: child.id, historySessionId: child.historySessionId, name: 'worker', title: 'Saved worker', promptProfile: 'coder', sourceAgentId: 'parent', workspace: f.directory, providerRoute: f.route('new', 'codex-new'), providerProfile: 'new', reasoningEffort: 'high', status: 'completed', toolsets: ['ReadFile'] })
      expect(f.host.manager.listTasks()[0]?.rules).toEqual(child.rules)
      expect(f.host.manager.peekMailbox().filter(event => event.type === 'retry').at(-1)).toMatchObject({ taskId: child.id, model: 'codex-new', providerProfile: 'new', reasoningEffort: 'high', data: { attempt: 1 } })
      const transcript = await f.transcripts.load(child.historySessionId!, { currentProjectDirectory: f.directory })
      expect(transcript?.metadata).toMatchObject({ subagent_id: child.id, model: 'codex-new', status: 'completed' })
      expect(f.host.managerPort.listHandles()).toHaveLength(1)
    } finally { await f.close() }
  })
}

test('partial selectors preserve the saved model/profile/effort and unconfigured follow-ups retain the selected route', async () => {
  const f = await fixture()
  try {
    const child = await f.spawn()
    await f.execute('SendMessageTool', { target: child.id, message: 'Effort only', reasoning_effort: 'high' })
    await f.host.managerPort.wait([child.id], 5_000)
    expect(f.host.managerPort.listHandles()[0]).toMatchObject({ model: 'codex-old', providerProfile: 'old', reasoningEffort: 'high', attempt: 1 })
    await f.execute('AgentTool', { resume: child.id, prompt: 'Profile only', provider_profile: 'new', wait: false })
    await f.host.managerPort.wait([child.id], 5_000)
    expect(f.host.managerPort.listHandles()[0]).toMatchObject({ model: 'codex-old', providerProfile: 'new', reasoningEffort: 'high', attempt: 2 })
    await f.execute('AgentTool', { resume: child.id, prompt: 'Model only', model: 'codex-new', wait: false })
    await f.host.managerPort.wait([child.id], 5_000)
    await f.execute('SendMessageTool', { target: child.id, message: 'No selectors' })
    await f.host.managerPort.wait([child.id], 5_000)
    expect(f.host.managerPort.listHandles()[0]).toMatchObject({ id: child.id, model: 'codex-new', providerProfile: 'new', reasoningEffort: 'high', attempt: 4, providerRoute: f.route('new', 'codex-new') })
    expect(f.newClient.requests.map(request => request.model)).toEqual(['codex-old', 'codex-new', 'codex-new'])
    expect(f.newClient.requests.map(request => request.thinking?.effort)).toEqual(['high', 'high', 'high'])
  } finally { await f.close() }
})

test('running agents reject reconfiguration at the tool, host and manager boundaries without queuing new input', async () => {
  const f = await fixture()
  const started = Promise.withResolvers<void>()
  const release = Promise.withResolvers<void>()
  f.oldClient.beforeReply = async () => { started.resolve(); await release.promise }
  try {
    const child = await f.host.managerPort.spawn({ promptProfile: 'coder', sourceAgentId: 'parent', message: 'Running brief' })
    await started.promise
    for (const [tool, args] of [
      ['AgentTool', { resume: child.id, prompt: 'Do not queue', model: 'codex-new', wait: false }],
      ['SendMessageTool', { target: child.id, message: 'Do not queue', reasoning_effort: 'high' }],
    ] as const) await expect(f.execute(tool, args)).rejects.toThrow('cannot reconfigure a running agent')
    await expect(f.host.retry(child.id, { selection: { model: 'codex-new' } })).rejects.toThrow('cannot reconfigure a running agent')
    await expect(f.host.manager.retry(child.id, 'Do not queue', { model: 'codex-new', providerProfile: undefined, reasoningEffort: undefined, providerRoute: undefined, hostGeneration: 0 })).rejects.toThrow('cannot reconfigure a running agent')
    expect(f.host.managerPort.listHandles()[0]).toMatchObject({ id: child.id, attempt: 0, model: 'codex-old', queueSize: 0 })
    release.resolve()
    await f.host.managerPort.wait([child.id], 5_000)
    expect(f.oldClient.requests).toHaveLength(1)
    expect(f.newClient.requests).toHaveLength(0)
  } finally { release.resolve(); await f.close() }
})

test.each(['explicit', 'plain'] as const)('a competing %s retry cannot overwrite the admitted generation while cancellation settles across two host reconfigurations', async competing => {
  const f = await fixture()
  const started = Promise.withResolvers<void>()
  const release = Promise.withResolvers<void>()
  const cancelledCheckpoint = Promise.withResolvers<void>()
  const finishCancellation = Promise.withResolvers<void>()
  const validationStarted = Promise.withResolvers<void>()
  const finishValidation = Promise.withResolvers<void>()
  const resumedStarted = Promise.withResolvers<void>()
  const finishResumed = Promise.withResolvers<void>()
  const save = f.transcripts.save.bind(f.transcripts)
  const checkpoint = spyOn(f.transcripts, 'save').mockImplementation(async (transcript, options) => {
    if (transcript.metadata.status === 'cancelled') {
      cancelledCheckpoint.resolve()
      await finishCancellation.promise
    }
    return save(transcript, options)
  })
  f.oldClient.beforeReply = async () => { started.resolve(); await release.promise }
  try {
    // G0's runner remains unsettled until its cancellation checkpoint finishes.
    const child = await f.host.managerPort.spawn({
      agent: { id: 'coder', model: 'codex-old', providerProfile: 'old' },
      promptProfile: 'coder', sourceAgentId: 'parent', nickname: 'delayed-worker', message: 'Delayed original brief',
    })
    await started.promise
    expect(f.host.interruptSource('parent')).toBe(1)
    expect(f.host.manager.listTasks()[0]?.status).toBe('cancelled')
    await cancelledCheckpoint.promise
    const validationsBefore = f.validations.length
    f.setValidation(async () => { validationStarted.resolve(); await finishValidation.promise })
    f.newClient.beforeReply = async () => { resumedStarted.resolve(); await finishResumed.promise }
    f.host.reconfigure({ ...f.options, model: 'codex-new' }) // G1 admits the retry.
    const pending = f.host.retry(child.id, {
      message: 'Resume after delayed cancellation',
      selection: { model: 'codex-new', providerProfile: 'new', reasoningEffort: 'high' },
    })
    void pending.catch(() => undefined)
    await validationStarted.promise
    let competingSettled = false
    const other = f.host.retry('delayed-worker', {
      message: 'Competing retry must not replace the admitted input',
      ...(competing === 'explicit' ? { selection: { model: 'codex-old', providerProfile: 'old' } } : {}),
    })
    void other.then(() => { competingSettled = true }, () => { competingSettled = true })
    finishValidation.resolve()
    const admitted = await Promise.race([
      f.host.manager.waitFor(() => f.host.manager.listTasks().some(task => task.id === child.id && task.status === 'pending' && task.attempt === 1), { timeoutMs: 1_000 }),
      pending.then(() => { throw new Error('Retry settled before the cancelled runner was released') }),
    ])
    expect(admitted).toBe(true)
    expect(competingSettled).toBe(false)
    expect(f.validations).toHaveLength(validationsBefore + 1)
    expect(f.newClient.requests).toHaveLength(0)
    f.host.reconfigure({ ...f.options, model: 'codex-old' }) // G2 must not prune admitted G1.
    finishCancellation.resolve()
    release.resolve()
    const resumed = await pending
    if (competing === 'explicit') await expect(other).rejects.toThrow('cannot reconfigure a running agent')
    else expect(await other).toMatchObject({ id: child.id, attempt: 1, lastInput: 'Resume after delayed cancellation', model: 'codex-new', providerProfile: 'new' })
    await resumedStarted.promise
    finishResumed.resolve()
    await f.host.managerPort.wait([child.id], 5_000)
    expect(resumed).toMatchObject({ id: child.id, historySessionId: child.historySessionId, attempt: 1, model: 'codex-new', providerProfile: 'new', reasoningEffort: 'high' })
    expect(f.host.managerPort.listHandles()[0]).toMatchObject({ id: child.id, status: 'completed', lastOutput: 'Reply from codex-new' })
    expect(f.oldClient.requests).toHaveLength(1)
    expect(f.newClient.requests).toHaveLength(1)
    expect(f.newClient.requests[0]?.messages.filter(message => message.role === 'user').map(message => message.content)).toEqual(['Delayed original brief', 'Resume after delayed cancellation'])
  } finally { finishValidation.resolve(); finishCancellation.resolve(); finishResumed.resolve(); release.resolve(); await f.close(); checkpoint.mockRestore() }
})

test('retry queue survives rejection, preserves its newer owner and releases after the last operation', async () => {
  const f = await fixture()
  const validationStarted = Promise.withResolvers<void>()
  const finishValidation = Promise.withResolvers<void>()
  try {
    const child = await f.spawn()
    f.setValidation(async () => { validationStarted.resolve(); await finishValidation.promise })
    const rejected = f.host.retry(child.id, { message: 'Rejected selection', selection: { model: 'unknown-model' } })
    const accepted = f.host.retry('worker', { message: 'Accepted queued retry', selection: { model: 'codex-new', providerProfile: 'new' } })
    await expect(rejected).rejects.toThrow('unknown model')
    await validationStarted.promise
    let joinedSettled = false
    const joined = f.host.retry(child.id, { message: 'Must join rather than replace queued input' })
    void joined.then(() => { joinedSettled = true }, () => { joinedSettled = true })
    await Promise.resolve()
    expect(joinedSettled).toBe(false)
    expect(f.host.manager.listTasks()[0]?.attempt).toBe(0)
    finishValidation.resolve()
    const results = await Promise.all([accepted, joined])
    expect(results.map(result => result.attempt)).toEqual([1, 1])
    expect(results.every(result => result.id === child.id && result.lastInput === 'Accepted queued retry' && result.providerProfile === 'new')).toBe(true)
    await f.host.managerPort.wait([child.id], 5_000)
    const fresh = await f.host.retry(child.id, { message: 'Fresh retry after the queue was removed' })
    await f.host.managerPort.wait([child.id], 5_000)
    expect(fresh).toMatchObject({ id: child.id, attempt: 2, lastInput: 'Fresh retry after the queue was removed', providerProfile: 'new' })
    expect(f.oldClient.requests).toHaveLength(1)
    expect(f.newClient.requests).toHaveLength(2)
  } finally { finishValidation.resolve(); await f.close() }
})

test('explicit selection fails closed when saved selection metadata is unavailable', async () => {
  const f = await fixture()
  const savedSelection = spyOn(f.host.manager, 'retryModelSelection')
  const retry = spyOn(f.host.manager, 'retry')
  try {
    const child = await f.spawn()
    savedSelection.mockReturnValue(undefined)
    const before = f.host.managerPort.listHandles()[0]
    const validations = f.validations.length
    const error = await f.host.retry(child.id, { message: 'Must not silently retain old settings', selection: { model: 'codex-new', providerProfile: 'new' } }).then(() => undefined, (cause: unknown) => cause)
    expect(error).toBeInstanceOf(ValidationError)
    if (!(error instanceof ValidationError)) throw new Error('Expected selection validation failure')
    expect(error.field).toBe('selection')
    expect(error.message).toContain('saved model/provider selection metadata')
    expect(retry).not.toHaveBeenCalled()
    expect(f.host.managerPort.listHandles()[0]).toEqual(before)
    expect(f.validations).toHaveLength(validations)
    expect(f.oldClient.requests).toHaveLength(1)
    expect(f.newClient.requests).toHaveLength(0)
  } finally { savedSelection.mockRestore(); retry.mockRestore(); await f.close() }
})

test('invalid model/profile/effort and changed routes fail before mutating the saved attempt', async () => {
  const f = await fixture()
  try {
    const child = await f.spawn()
    for (const selection of [{ model: 'missing-model' }, { providerProfile: 'missing-profile' }, { reasoningEffort: 'invalid' }]) {
      await expect(f.host.retry(child.id, { message: 'Rejected', sourceAgentId: 'parent', selection })).rejects.toThrow(/unknown|invalid/)
    }
    f.setValidation(async () => { f.changeRoute() })
    await expect(f.host.retry(child.id, { selection: { model: 'codex-new', providerProfile: 'new' } })).rejects.toThrow('provider route changed')
    expect(f.host.managerPort.listHandles()[0]).toMatchObject({ id: child.id, model: 'codex-old', providerProfile: 'old', reasoningEffort: 'medium', attempt: 0 })
    expect(f.newClient.requests).toHaveLength(0)
    expect(f.oldClient.requests).toHaveLength(1)
    f.setValidation(undefined)
    await f.host.retry(child.id, { message: 'Accepted after failure', selection: { model: 'codex-new', providerProfile: 'new' } })
    await f.host.managerPort.wait([child.id], 5_000)
    expect(f.newClient.requests).toHaveLength(1)
    expect(f.host.managerPort.listHandles()[0]).toMatchObject({ attempt: 1, providerRoute: f.route('new', 'codex-new') })
  } finally { await f.close() }
})

test('reconfiguration cannot bypass ownership, closed-agent, agent-type, worktree or permission-generation safeguards', async () => {
  const f = await fixture()
  try {
    const child = await f.spawn()
    await expect(f.execute('AgentTool', { resume: child.id, prompt: 'Foreign', model: 'codex-new' }, 'other-parent')).rejects.toThrow('not found')
    for (const forbidden of [{ subagent_type: 'reviewer' }, { isolation: 'worktree' }, { worktree_ref: 'HEAD' }, { worktree_source: 'working-tree' }, { intelligence: 'smart' }]) {
      await expect(f.execute('AgentTool', { resume: child.id, prompt: 'Cannot change capabilities', model: 'codex-new', ...forbidden })).rejects.toThrow(/cannot|unknown parameter/)
    }
    f.host.managerPort.close(child.id)
    await expect(f.execute('SendMessageTool', { target: child.id, message: 'Closed', model: 'codex-new' })).rejects.toThrow('closed')
    // Explicit AgentTool resume may reopen a closed identity but still retains its history.
    await f.execute('AgentTool', { resume: child.id, prompt: 'Explicit reopen', model: 'codex-new', provider_profile: 'new', wait: false })
    await f.host.managerPort.wait([child.id], 5_000)
    expect(f.host.managerPort.listHandles()[0]?.historySessionId).toBe(child.historySessionId)
    f.host.reconfigure({ ...f.options, permissionMode: 'manual' })
    await expect(f.execute('AgentTool', { resume: child.id, prompt: 'Do not bypass tightened policy', model: 'codex-new', wait: false })).rejects.toThrow('invalidated')
    expect(f.newClient.requests).toHaveLength(1)
  } finally { await f.close() }
})

test('host-generation changes during route validation reject the switch before a new attempt starts', async () => {
  const f = await fixture()
  try {
    const child = await f.spawn()
    f.setValidation(async () => { f.host.reconfigure({ ...f.options, permissionMode: 'manual' }) })
    await expect(f.host.retry(child.id, { selection: { model: 'codex-new', providerProfile: 'new' } })).rejects.toThrow('host changed')
    expect(f.host.managerPort.listHandles()[0]?.attempt).toBe(0)
    expect(f.newClient.requests).toHaveLength(0)
  } finally { await f.close() }
})

test('inherited model switches recompute the current host route and keep the original workspace', async () => {
  const f = await fixture()
  try {
    const child = await f.host.managerPort.spawn({ promptProfile: 'coder', sourceAgentId: 'parent', message: 'Inherited original brief' })
    await f.host.managerPort.wait([child.id], 5_000)
    const newRoute = f.route('inherited', 'codex-new')
    f.host.reconfigure({ ...f.options, llm: f.newClient, model: 'codex-new', cwd: join(f.directory, 'different-main-workspace'), inheritedProviderRoute: newRoute })
    const resumed = await f.host.retry(child.id, { message: 'Inherited continuation', selection: { model: 'codex-new' } })
    await f.host.managerPort.wait([child.id], 5_000)
    expect(resumed).toMatchObject({ id: child.id, model: 'codex-new', reasoningEffort: 'medium', providerRoute: newRoute, workspace: f.directory })
    expect(resumed.providerProfile).toBeUndefined()
    expect(f.newClient.requests).toHaveLength(1)
    expect(f.newClient.requests[0]?.messages.filter(message => message.role === 'user').map(message => message.content)).toEqual(['Inherited original brief', 'Inherited continuation'])
  } finally { await f.close() }
})

test('cancellation during selection validation starts no new attempt and preserves the saved selection', async () => {
  const f = await fixture()
  try {
    const child = await f.spawn()
    const controller = new AbortController()
    f.setValidation(async () => { controller.abort(new Error('cancelled selection')) })
    await expect(f.host.retry(child.id, { signal: controller.signal, selection: { model: 'codex-new', providerProfile: 'new' } })).rejects.toThrow('cancelled selection')
    expect(f.host.managerPort.listHandles()[0]).toMatchObject({ id: child.id, model: 'codex-old', providerProfile: 'old', attempt: 0 })
    expect(f.newClient.requests).toHaveLength(0)
  } finally { await f.close() }
})

test('a recovered terminal agent can explicitly replace a stale route without changing its id, attempt lineage or persisted transcript', async () => {
  const f = await fixture()
  let restarted: ReturnType<typeof createNativeSubagentHost> | undefined
  try {
    const child = await f.spawn()
    const snapshots = f.host.managerPort.listHandles()
    await f.host.manager.shutdown()
    f.changeRoute()
    restarted = createNativeSubagentHost({ ...f.options, llm: f.newClient, model: 'codex-new' })
    restarted.turnCoordinator.restore?.('parent', snapshots)
    const result = await restarted.retry(child.id, { message: 'Recovered with a new provider', sourceAgentId: 'parent', selection: { model: 'codex-new', providerProfile: 'new', reasoningEffort: 'high' } })
    expect(result).toMatchObject({ id: child.id, historySessionId: child.historySessionId, attempt: 1, providerProfile: 'new', reasoningEffort: 'high', providerRoute: f.route('new', 'codex-new') })
    await restarted.managerPort.wait([child.id], 5_000)
    expect(f.newClient.requests[0]?.messages.filter(message => message.role === 'user').map(message => message.content)).toEqual(['Original brief', 'Recovered with a new provider'])
    expect(restarted.managerPort.listHandles()).toHaveLength(1)
  } finally { await restarted?.manager.shutdown(); await f.close() }
})

test('manager reconfiguration keeps archived policy, workspace, parentage and worktree isolation while updating attempt config', async () => {
  const requests: SubagentTaskRunRequest[] = []
  const allocations: Array<Readonly<Record<string, unknown>> | undefined> = []
  const manager = new SubAgentManager({
    maxRetainedTerminalTasks: 1,
    worktree: {
      async create(options) { allocations.push(options.config); return { path: process.cwd(), branch: 'fixture-worktree' } },
      async isClean() { return true },
      async remove() {},
    },
    runner: async request => { requests.push(request); return 'Fixture done' },
  })
  try {
    const child = await manager.spawn({
      prompt: 'Saved isolated brief', name: 'isolated-worker', isolation: 'worktree', sourceId: 'parent', creatorId: 'coder',
      model: 'codex-old', rules: ['permission:manual', 'isolation:worktree'], toolsets: ['ReadFile'],
      config: { model: 'codex-old', providerProfile: 'old', reasoningEffort: 'medium', _nativeSubagentWorkspace: process.cwd(), _nativeSubagentWorktreeRef: 'fixture-ref', permissionMode: 'manual', _toolsWhitelist: ['ReadFile'], _agentSkills: ['fixture-skill'] },
    })
    await manager.wait(child.id, 1_000)
    const other = await manager.spawn({ prompt: 'Evict the terminal worker' })
    await manager.wait(other.id, 1_000)
    expect(manager.findTask(child.id)?.archived).toBe(true)
    const resumed = await manager.retry(child.id, 'Continue isolated work', { model: 'codex-new', providerProfile: 'new', reasoningEffort: 'high', providerRoute: 'b'.repeat(64), hostGeneration: 1 })
    await manager.wait(child.id, 1_000)
    expect(resumed).toMatchObject({ id: child.id, model: 'codex-new', providerProfile: 'new', reasoningEffort: 'high', providerRoute: 'b'.repeat(64), attempt: 1, workspace: process.cwd(), sourceId: 'parent', creatorId: 'coder', rules: child.rules, toolsets: child.toolsets, worktreeBranch: 'fixture-worktree' })
    expect(requests.at(-1)?.config).toMatchObject({ model: 'codex-new', providerProfile: 'new', reasoningEffort: 'high', _nativeSubagentProviderRoute: 'b'.repeat(64), _nativeSubagentHostGeneration: 1, _nativeSubagentWorkspace: process.cwd(), _nativeSubagentWorktreeRef: 'fixture-ref', permissionMode: 'manual', _toolsWhitelist: ['ReadFile'], _agentSkills: ['fixture-skill'] })
    expect(allocations).toHaveLength(2)
    expect(allocations[1]?._nativeSubagentWorktreeRef).toBe('fixture-ref')
  } finally { await manager.shutdown() }
})
