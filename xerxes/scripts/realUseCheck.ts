// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { AcpServer } from '../src/acp/server.js'
import { CredentialStorage } from '../src/auth/storage.js'
import { COMMAND_REGISTRY, resolveCommand } from '../src/bridge/commands.js'
import { BridgeSlashRouter } from '../src/bridge/slashRouter.js'
import { createSessionResetPolicy, hashChat, hashUser, ResetTrigger, shouldReset, StickerCache } from '../src/channels/index.js'
import { MarkdownAgentWorkspace } from '../src/channels/workspace.js'
import { importWorkspace } from '../src/channels/workspaceImport.js'
import { ContextCompressor, naiveSummarizer } from '../src/context/compressor.js'
import { ToolResultStorage } from '../src/context/toolResultStorage.js'
import { CronJob, JobStore } from '../src/cron/jobs.js'
import { routeOutput } from '../src/cron/delivery.js'
import { CronScheduler } from '../src/cron/scheduler.js'
import { ToolRegistry } from '../src/executors/toolRegistry.js'
import { SlashPluginRegistry } from '../src/extensions/slashPlugins.js'
import { LocalSkillSource } from '../src/extensions/skillSources/local.js'
import { syncSkillManifest } from '../src/extensions/skillsSync.js'
import { calcCost, listAllModels } from '../src/llms/providerRegistry.js'
import { checkPackage, type OSVFetch } from '../src/mcp/osv.js'
import { buildAuthorizeUrl, generatePkcePair, OAuthToken } from '../src/mcp/oauth.js'
import { ReconnectPolicy, reconnectWithBackoff } from '../src/mcp/reconnect.js'
import { MCPToolServer } from '../src/mcp/server.js'
import { HolographicProvider, Mem0Provider, type ExternalMemoryUpstream } from '../src/memory/plugins/index.js'
import { createBrowserProviderRegistry, SUPPORTED_BROWSER_PROVIDERS } from '../src/operators/browserProviders.js'
import { AuxiliaryClient } from '../src/runtime/auxiliaryClient.js'
import { BackgroundSessionManager, BackgroundStatus } from '../src/runtime/backgroundSessions.js'
import { CostEvent, CostTracker } from '../src/runtime/costTracker.js'
import { hasDoctorFailures, runAllDoctorChecks } from '../src/runtime/doctor.js'
import { classifyError, ErrorKind } from '../src/runtime/errorClassifier.js'
import { buildInsightsReport } from '../src/runtime/insights.js'
import { InterruptToken } from '../src/runtime/interrupt.js'
import { BudgetExhausted, IterationBudget } from '../src/runtime/iterationBudget.js'
import { RateLimitTracker } from '../src/runtime/rateLimitTracker.js'
import { ProcessRegistry } from '../src/runtime/processRegistry.js'
import { runSetupWizard, writeSetupConfig } from '../src/runtime/setupWizard.js'
import { checkBunPackageUpdate, planBunUpdate } from '../src/runtime/update.js'
import { BUN_SHELL_INSTALL_SNIPPET, detectPlatform, filterTermuxDependencies, renderHomebrewFormula } from '../src/runtime/distribution.js'
import { NudgeContext, NudgeManager } from '../src/runtime/nudges.js'
import { ApprovalScope, ApprovalStore } from '../src/security/approvals.js'
import { syncPush } from '../src/security/fileSync.js'
import { resolveWithin } from '../src/security/pathSecurity.js'
import { ToolPolicy } from '../src/security/policy.js'
import { redactString } from '../src/security/redact.js'
import { SshSandboxAdapter } from '../src/security/sandboxBackends/ssh.js'
import { checkUrl } from '../src/security/urlSafety.js'
import { branchSession, sessionLineage } from '../src/session/branching.js'
import { SessionRecord, TurnRecord } from '../src/session/models.js'
import { SnapshotManager } from '../src/session/snapshots.js'
import { FileSessionStore, SQLiteSessionStore } from '../src/session/store.js'
import { getToolCallParser, TOOL_CALL_PARSER_REGISTRY } from '../src/streaming/parsers/index.js'
import { wrapSystemWithCache, wrapToolsWithCache, extractCacheTokens } from '../src/streaming/promptCaching.js'
import { ResponsesEventTranslator } from '../src/streaming/responsesApi.js'
import { parseSseStream } from '../src/streaming/sse.js'
import { createAgentState } from '../src/streaming/events.js'
import { deterministicToolCallId } from '../src/streaming/toolCallIds.js'
import { BatchRunner } from '../src/training/batchRunner.js'
import { builtinEnvironments, TinkerClient, TinkerRunConfig, RLRunStatus } from '../src/training/rl/index.js'
import { TrajectoryCompressor } from '../src/training/trajectoryCompressor.js'
import { clarify, StaticAsker } from '../src/tools/clarify.js'
import { ImageGenerationRegistry } from '../src/tools/imageGeneration.js'
import { OutboundMessageRegistry, sendMessage } from '../src/tools/sendMessage.js'
import { AgentAuthoredSkillStore } from '../src/tools/skillManage.js'
import { VisionRegistry, visionImageFromBase64 } from '../src/tools/vision.js'
import { VoiceModeController, type VoiceRecorder } from '../src/tools/voiceMode.js'
import { WorkspaceMemoryStore } from '../src/tools/workspaceMemory.js'
import { runSwarmIntegration } from './swarmIntegration.js'

export type RealUseStatus = 'failed' | 'passed' | 'skipped'

export interface RealUseResult {
  readonly detail: string
  readonly feature: string
  readonly plan: string
  readonly status: RealUseStatus
}

export interface RealUseReport {
  readonly failed: number
  readonly ok: boolean
  readonly passed: number
  readonly results: readonly RealUseResult[]
  readonly skipped: number
}

/** Explicit host boundaries for checks that can spend credentials, use hardware, or touch a public API. */
export interface RealUsePorts {
  readonly browserProbe?: () => Promise<string> | string
  /** Opt-in transport for the public OSV query; never inferred from global fetch. */
  readonly fetchImplementation?: OSVFetch
  readonly findExecutable?: (name: string) => string | null
  /** Opt-in host probe for an actual credentialed media operation. */
  readonly mediaProbe?: () => Promise<string> | string
  readonly networkProbe?: () => Promise<boolean>
  /** Opt-in host probe for an actual public package-registry request. */
  readonly packageRegistryProbe?: () => Promise<string> | string
  readonly providerProbe?: () => Promise<string> | string
  readonly temporaryDirectory?: <T>(label: string, run: (directory: string) => Promise<T>) => Promise<T>
  /** Opt-in host probe for an actual terminal/UI integration. */
  readonly tuiProbe?: () => Promise<string> | string
}

export interface RealUseContext {
  readonly allowNetwork: boolean
  readonly ports: RealUsePorts
  isNetworkAvailable(): Promise<boolean>
  withTemporaryDirectory<T>(label: string, run: (directory: string) => Promise<T>): Promise<T>
}

export interface RealUseCheck {
  readonly feature: string
  readonly plan: string
  readonly run: (context: RealUseContext) => Promise<string> | string
  readonly skip?: (context: RealUseContext) => Promise<string | undefined> | string | undefined
}

export interface RunRealUseOptions {
  /** Network calls are disabled by default, even when an internet connection is available. */
  readonly allowNetwork?: boolean
  readonly checks?: readonly RealUseCheck[]
  readonly ports?: RealUsePorts
}

/**
 * Build the Bun-native real-use suite.
 *
 * Every default pass exercises a local TypeScript implementation. Checks that
 * could use paid credentials, a browser/hardware backend, Git, or the public
 * internet are explicitly gated and report `skipped` when their port is absent.
 */
export function createRealUseChecks(): readonly RealUseCheck[] {
  return Object.freeze([
    {
      plan: '01',
      feature: 'Channel workspace seeds context, writes notes, and imports a compatible local workspace',
      run: checkWorkspace,
    },
    {
      plan: '02',
      feature: 'Prompt-cache wrappers preserve system/tool breakpoints and usage accounting',
      run: checkPromptCaching,
    },
    {
      plan: '03',
      feature: 'Context compaction preserves protected turns and writes a reference summary',
      run: checkContextCompaction,
    },
    {
      plan: '03',
      feature: 'Tool-result overflow storage round-trips through the native filesystem',
      run: checkToolResultStorage,
    },
    {
      plan: '04',
      feature: 'Memory and skill nudges fire through native deterministic heuristics',
      run: checkNudges,
    },
    {
      plan: '04',
      feature: 'Agent-authored skill store completes create, view, and delete locally',
      run: checkAgentAuthoredSkill,
    },
    {
      plan: '05',
      feature: 'External-memory plugin lifecycle uses an injected local upstream and preserves unavailable Mem0 state',
      run: checkMemoryPlugins,
    },
    {
      plan: '06',
      feature: 'MCP tool server lists and executes a registered native tool in-process',
      run: checkMcpToolServer,
    },
    {
      plan: '07',
      feature: 'MCP reconnect retries a local callback and redacts failed credential-like details',
      run: checkMcpReconnect,
    },
    {
      plan: '07',
      feature: 'PKCE generation and authorization URL construction',
      run: checkPkce,
    },
    {
      plan: '07',
      feature: 'OSV public advisory query through the injected fetch port',
      skip: skipWithoutNetwork,
      run: checkOsv,
    },
    {
      plan: '08',
      feature: 'ACP session, prompt, permission, cancellation, and close cycle',
      run: checkAcp,
    },
    {
      plan: '09',
      feature: 'Cron schedule fires and archives a real local result',
      run: checkCron,
    },
    {
      plan: '10',
      feature: 'Voice mode coordinates an injected local recorder and transcription port',
      run: checkVoiceMode,
    },
    {
      plan: '11',
      feature: 'Vision and image-generation registries dispatch local registered providers',
      run: checkMediaRegistries,
    },
    {
      plan: '11',
      feature: 'Credentialed or hardware media probe supplied by the embedding host',
      skip: skipWithoutMediaProbe,
      run: async context => await context.ports.mediaProbe?.() ?? fail('media probe disappeared after availability check'),
    },
    {
      plan: '12',
      feature: 'Browser provider registry exposes every native role without opening a browser',
      run: checkBrowserRegistry,
    },
    {
      plan: '13',
      feature: 'SSH sandbox serializes a direct argv request through an injected transport',
      run: checkSshSandbox,
    },
    {
      plan: '13',
      feature: 'File sync filters oversized files through explicit local transfer ports',
      run: checkFileSync,
    },
    {
      plan: '14',
      feature: 'Raw-text tool-call parsers cover every native provider family',
      run: checkParserMatrix,
    },
    {
      plan: '15',
      feature: 'RL Tinker client normalizes a local injected training lifecycle',
      run: checkRlTraining,
    },
    {
      plan: '16',
      feature: 'Trajectory compressor writes native batch metrics and JSONL output',
      run: checkTrajectoryCompression,
    },
    {
      plan: '17',
      feature: 'Batch runner records successful and failed local records in JSONL',
      run: checkBatchRunner,
    },
    {
      plan: '18',
      feature: 'SQLite session persistence, branching, and lineage',
      run: checkSessions,
    },
    {
      plan: '18',
      feature: 'Git shadow snapshot rollback through the native snapshot manager',
      skip: skipWithoutGit,
      run: checkSnapshot,
    },
    {
      plan: '19',
      feature: 'Bridge command registry resolves canonical and alias slash commands',
      run: checkBridgeCommands,
    },
    {
      plan: '20',
      feature: 'Bun doctor checks local runtime facts with an empty explicit credential environment',
      run: checkDoctor,
    },
    {
      plan: '20',
      feature: 'Setup wizard writes a native YAML-compatible configuration file',
      run: checkSetupWizard,
    },
    {
      plan: '20',
      feature: 'Bun update planning and injected package metadata parsing stay opt-in',
      run: checkUpdatePlanning,
    },
    {
      plan: '20',
      feature: 'Public package-registry update probe supplied by the embedding host',
      skip: skipWithoutPackageRegistryProbe,
      run: async context => await context.ports.packageRegistryProbe?.() ?? fail('package registry probe disappeared after availability check'),
    },
    {
      plan: '21',
      feature: 'Messaging identity hashing, sticker index persistence, and timeout reset policy',
      run: checkMessagingState,
    },
    {
      plan: '22',
      feature: 'Security redaction, policy gating, path containment, URL protection, and approval persistence',
      run: checkSecurity,
    },
    {
      plan: '23',
      feature: 'Local skill source search and manifest sync install a native SKILL.md bundle',
      run: checkSkillsHub,
    },
    {
      plan: '24',
      feature: 'Error classification, rate-limit throttling, and injected auxiliary summarization',
      run: checkRuntimeRouting,
    },
    {
      plan: '25',
      feature: 'Encrypted OAuth credential storage round-trip in a temporary directory',
      run: checkCredentialStorage,
    },
    {
      plan: '26',
      feature: 'Provider pricing, cost ledger, and insights aggregation calculate locally',
      run: checkPricingInsights,
    },
    {
      plan: '27/TUI',
      feature: 'Interactive runtime state supports bridge slash routing, background work, and plugin commands',
      run: checkInteractiveRuntime,
    },
    {
      plan: 'TUI',
      feature: 'Live terminal UI probe supplied by the embedding host',
      skip: skipWithoutTuiProbe,
      run: async context => await context.ports.tuiProbe?.() ?? fail('TUI probe disappeared after availability check'),
    },
    {
      plan: '28',
      feature: 'Iteration budget, process registry, interrupt token, and deterministic tool IDs run natively',
      run: checkRuntimePrimitives,
    },
    {
      plan: '28',
      feature: 'Native agent swarm exercises orchestration, execution, Cortex, and sandbox policy',
      run: checkNativeSwarm,
    },
    {
      plan: '29',
      feature: 'Current-schema file session persistence round-trips without temporary artifacts',
      run: checkFileSessionPersistence,
    },
    {
      plan: '30',
      feature: 'Bun distribution helpers render Homebrew guidance and filter Termux dependencies',
      run: checkDistribution,
    },
    {
      plan: '31',
      feature: 'SSE parsing and Responses API translation end-to-end',
      run: checkStreaming,
    },
    {
      plan: '32',
      feature: 'Message dispatch, clarification, and workspace memory CRUD run through explicit local adapters',
      run: checkMessagingTools,
    },
    {
      plan: 'LIVE',
      feature: 'Credentialed provider probe supplied by the embedding host',
      skip: skipWithoutProviderProbe,
      run: async context => await context.ports.providerProbe?.() ?? fail('provider probe disappeared after availability check'),
    },
    {
      plan: 'HARDWARE',
      feature: 'Live browser or hardware probe supplied by the embedding host',
      skip: context => context.ports.browserProbe ? undefined : 'no browserProbe supplied; no browser or hardware session was opened',
      run: async context => await context.ports.browserProbe?.() ?? fail('browser probe disappeared after availability check'),
    },
  ])
}

/** Execute checks in declaration order, retaining passes, safe skips, and failures separately. */
export async function runRealUseChecks(options: RunRealUseOptions = {}): Promise<RealUseReport> {
  const context = createContext(options)
  const results: RealUseResult[] = []
  for (const check of options.checks ?? createRealUseChecks()) {
    try {
      const reason = await check.skip?.(context)
      if (reason !== undefined) {
        results.push(resultFor(check, 'skipped', reason))
        continue
      }
      const detail = await check.run(context)
      results.push(resultFor(check, 'passed', detail))
    } catch (error) {
      results.push(resultFor(check, 'failed', `${errorName(error)}: ${errorMessage(error)}`))
    }
  }
  const passed = results.filter(result => result.status === 'passed').length
  const skipped = results.filter(result => result.status === 'skipped').length
  const failed = results.filter(result => result.status === 'failed').length
  return Object.freeze({
    results: Object.freeze(results),
    passed,
    skipped,
    failed,
    ok: failed === 0,
  })
}

/** Render a terminal-safe report. Details are redacted before the report is returned or printed. */
export function formatRealUseReport(report: RealUseReport): string {
  const groups: Record<RealUseStatus, RealUseResult[]> = { passed: [], skipped: [], failed: [] }
  for (const result of report.results) groups[result.status].push(result)
  const lines = [
    '='.repeat(80),
    'XERXES REAL-USE CHECK — NATIVE BUN/TYPESCRIPT SURFACES',
    '='.repeat(80),
    'Only checks that actually ran are marked passed. Credential, network, and hardware probes are opt-in.',
  ]
  for (const status of ['passed', 'skipped', 'failed'] as const) {
    lines.push('', `## ${status.toUpperCase()} (${groups[status].length})`)
    for (const result of groups[status]) {
      lines.push(`  ${statusMarker(status)} plan ${result.plan}: ${result.feature}`)
      if (result.detail) lines.push(...wrapDetail(result.detail))
    }
  }
  lines.push('', '='.repeat(80))
  lines.push(`summary: passed=${report.passed} | skipped=${report.skipped} | failed=${report.failed}`)
  lines.push('='.repeat(80))
  return lines.join('\n')
}

/** CLI entrypoint. `--online` is required before the suite makes its public OSV request. */
export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  if (args.includes('--help') || args.includes('-h')) {
    console.log('Usage: bun scripts/realUseCheck.ts [--online]')
    console.log('  --online  Opt in to a public OSV advisory query after a bounded connectivity probe.')
    return 0
  }
  const unknown = args.filter(argument => argument !== '--online')
  if (unknown.length) {
    console.error(`Unknown argument(s): ${unknown.join(', ')}`)
    return 2
  }
  const report = await runRealUseChecks({
    allowNetwork: args.includes('--online'),
    ports: args.includes('--online') ? (() => {
      const fetchImplementation: OSVFetch = (input, init) => fetch(input, init)
      return {
        fetchImplementation,
        networkProbe: () => defaultNetworkProbe(fetchImplementation),
      }
    })() : {},
  })
  console.log(formatRealUseReport(report))
  return report.ok ? 0 : 1
}

async function checkWorkspace(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('workspace', async directory => {
    const source = join(directory, 'compatible-source')
    const target = join(directory, 'channel-workspace')
    await mkdir(join(source, 'memory'), { recursive: true })
    await writeFile(join(source, 'AGENTS.md'), '# Imported agent instructions\n', 'utf8')
    await writeFile(join(source, 'SOUL.md'), '# Imported soul\nStay evidence-driven.\n', 'utf8')
    await writeFile(join(source, 'memory', '2026-07-13.md'), 'Imported daily note.\n', 'utf8')

    const workspace = new MarkdownAgentWorkspace(target)
    await workspace.ensure()
    for (const name of ['AGENTS.md', 'SOUL.md', 'USER.md', 'MEMORY.md', 'TOOLS.md']) {
      require(await Bun.file(join(target, name)).exists(), `workspace did not seed ${name}`)
    }
    const notePath = await workspace.appendDailyNote('native local workspace note', {
      when: new Date('2026-07-13T12:34:56.000Z'),
    })
    require((await readFile(notePath, 'utf8')).includes('native local workspace note'), 'workspace note did not round-trip')

    const imported = await importWorkspace(source, { targetWorkspace: workspace, overwrite: true })
    require(imported.copied.includes('AGENTS.md'), 'workspace import did not copy AGENTS.md')
    const loaded = await workspace.loadContext({ today: new Date('2026-07-13T13:00:00.000Z') })
    require(loaded.prompt.includes('Imported agent instructions'), 'workspace context omitted imported AGENTS.md')
    require(loaded.prompt.includes('Imported daily note.'), 'workspace context omitted imported daily note')
    return `seeded channel context, appended one note, and imported ${imported.copied.length} compatible local file(s)`
  })
}

async function checkPromptCaching(): Promise<string> {
  const system = wrapSystemWithCache('Be concise.')
  require(
    Array.isArray(system) && isRecord(system[0]?.cache_control) && system[0].cache_control.type === 'ephemeral',
    'system cache wrapper omitted breakpoint',
  )
  const tools = wrapToolsWithCache([
    { name: 'read_file', input_schema: { type: 'object' } },
    { name: 'write_file', input_schema: { type: 'object' } },
  ])
  require(
    tools.length === 2 && isRecord(tools[1]?.cache_control) && tools[1].cache_control.type === 'ephemeral',
    'tool cache wrapper omitted tail breakpoint',
  )
  require(tools[0]?.cache_control === undefined, 'tool cache wrapper marked the first schema')
  const [readTokens, createdTokens] = extractCacheTokens({
    cache_read_input_tokens: 12,
    cache_creation_input_tokens: 4,
  })
  require(readTokens === 12 && createdTokens === 4, 'cache usage extraction changed token counts')
  return 'system and tail-tool cache breakpoints preserved; usage read=12 creation=4'
}

async function checkContextCompaction(): Promise<string> {
  const messages = [
    { role: 'system', content: 'keep system' },
    { role: 'user', content: 'keep the first request' },
    { role: 'assistant', content: 'compress this reply' },
    { role: 'user', content: 'compress this detail' },
    { role: 'assistant', content: 'compress this conclusion' },
    { role: 'user', content: 'keep the latest request' },
  ]
  const result = new ContextCompressor({
    contextWindow: 5,
    threshold: 0.5,
    protectFirst: 2,
    protectLast: 1,
    summarizer: naiveSummarizer,
    summaryMinTokens: 1,
  }).compress(messages)
  require(result.compressed, 'context compressor did not compact the oversized transcript')
  require(result.messages[0] === messages[0], 'context compressor did not preserve the first protected turn')
  require(result.messages.at(-1) === messages.at(-1), 'context compressor did not preserve the latest protected turn')
  require(String(result.messages[2]?.content).includes('[CONTEXT COMPACTION'), 'reference summary was not inserted')
  return `${messages.length} messages -> ${result.messages.length}; compressed ${result.compressedCount} middle messages`
}

async function checkToolResultStorage(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('tool-results', async directory => {
    const store = new ToolResultStorage(directory, { inlineLimit: 16 })
    const payload = 'native overflow '.repeat(40)
    const reference = store.maybeStore('read_file', payload)
    require(ToolResultStorage.isRef(reference), 'oversized result was not stored as a reference')
    require(store.fetch(reference) === payload, 'overflow result did not round-trip from disk')
    return `stored ${payload.length} characters in ${store.listRefs().length} native overflow file`
  })
}

async function checkNudges(): Promise<string> {
  const manager = new NudgeManager()
  const memory = manager.check(new NudgeContext({
    turnIndex: 7,
    lastUserMessage: 'Please remember my preferred response format.',
  }))
  const skills = manager.check(new NudgeContext({ turnIndex: 0, successfulToolCallsThisTurn: 7 }))
  const fired = new Set([...memory, ...skills].map(([name]) => name))
  require(fired.has('memory') && fired.has('skill'), 'native nudge heuristics did not fire both expected rules')
  return `fired native heuristic rules: ${[...fired].sort().join(', ')}`
}

async function checkAgentAuthoredSkill(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('agent-authored-skill', async directory => {
    const store = new AgentAuthoredSkillStore({ authoredDirectory: join(directory, 'skills') })
    const created = await store.manage('create', { name: 'demo', body: '# Demo\n\nKeep this local.' })
    require(created.ok === true, 'agent-authored skill creation failed')
    const viewed = await store.manage('view', { name: 'demo' })
    require(viewed.ok === true && typeof viewed.body === 'string' && viewed.body.includes('Keep this local.'), 'agent-authored skill view failed')
    const deleted = await store.manage('delete', { name: 'demo' })
    require(deleted.ok === true, 'agent-authored skill deletion failed')
    return 'agent-authored skill create/view/delete completed in a temporary directory'
  })
}

async function checkMemoryPlugins(): Promise<string> {
  const entries: string[] = []
  const upstream: ExternalMemoryUpstream = {
    call(action, arguments_) {
      if (action === 'add') {
        const content = arguments_.content
        if (typeof content !== 'string') throw new Error('missing content')
        entries.push(content)
        return { id: `entry-${entries.length}`, content }
      }
      if (action === 'search') {
        const query = arguments_.query
        if (typeof query !== 'string') throw new Error('missing query')
        return entries.filter(content => content.includes(query)).map((content, index) => ({ id: `entry-${index + 1}`, content }))
      }
      return []
    },
  }
  const holographic = new HolographicProvider({ environment: { get: () => undefined }, upstream })
  const added = await holographic.handleToolCall({ name: 'holo_add', arguments: { content: 'user prefers linen tea cups' } })
  const found = await holographic.handleToolCall({ name: 'holo_search', arguments: { query: 'linen' } })
  require(added.ok === true, 'holographic provider did not add through its host upstream')
  require(found.ok === true && Array.isArray(found.result) && found.result.length === 1, 'holographic provider did not search its host upstream')

  const mem0 = new Mem0Provider({
    environment: { get: () => undefined },
    transport: { request: async () => ({ ok: true, status: 200, body: {} }) },
  })
  require(!(await mem0.isAvailable()), 'Mem0 reported available without explicit configuration')
  return 'holographic add/search used an injected local upstream; unconfigured Mem0 remained unavailable'
}

async function checkMcpToolServer(): Promise<string> {
  const registry = new ToolRegistry()
  registry.register(
    {
      type: 'function',
      function: {
        name: 'upper',
        description: 'Uppercase a string.',
        parameters: {
          type: 'object',
          properties: { text: { type: 'string' } },
          required: ['text'],
          additionalProperties: false,
        },
      },
    },
    inputs => ({ value: String(inputs.text).toUpperCase() }),
  )
  const server = new MCPToolServer(registry)
  const listed = resultOf(await server.handle({ jsonrpc: '2.0', id: 1, method: 'tools/list', params: {} }))
  const tools = listed.tools
  require(Array.isArray(tools) && tools.length === 1, 'MCP tool list did not expose the registered tool')
  const called = resultOf(await server.handle({
    jsonrpc: '2.0',
    id: 2,
    method: 'tools/call',
    params: { name: 'upper', arguments: { text: 'native' } },
  }))
  const content = called.content
  require(Array.isArray(content), 'MCP tool call returned no content')
  const first = content[0]
  require(isRecord(first) && first.text === '{"value":"NATIVE"}', 'MCP tool call produced an unexpected result')
  return 'registered tool listed and executed through MCP JSON-RPC'
}

async function checkMcpReconnect(): Promise<string> {
  let attempts = 0
  const sleeps: number[] = []
  const recovered = await reconnectWithBackoff(() => {
    attempts += 1
    if (attempts < 3) throw new Error('api_key=sk-secretsecretsecret-leak')
    return 'connected'
  }, {
    policy: new ReconnectPolicy({ baseSeconds: 0.25, maxAttempts: 3 }),
    sleep: seconds => { sleeps.push(seconds) },
  })
  require(recovered === 'connected' && attempts === 3, 'MCP reconnect did not retry to recovery')
  require(sleeps.join(',') === '0.25,0.5', 'MCP reconnect delay sequence changed')

  let terminal = ''
  try {
    await reconnectWithBackoff(
      () => { throw new Error('api_key=sk-secretsecretsecret-leak') },
      { policy: new ReconnectPolicy({ maxAttempts: 1 }) },
    )
  } catch (error) {
    terminal = errorMessage(error)
  }
  require(terminal.includes('[redacted]') && !terminal.includes('secretsecretsecret'), 'terminal reconnect failure leaked a credential-like value')
  return 'MCP reconnect recovered after three attempts; terminal diagnostics were redacted'
}

async function checkPkce(): Promise<string> {
  const first = generatePkcePair()
  const second = generatePkcePair()
  require(first.verifier !== second.verifier && first.challenge !== second.challenge, 'PKCE pairs were not unique')
  const authorizeUrl = buildAuthorizeUrl({
    clientId: 'real-use-client',
    authorizeUrl: 'https://oauth.example.test/authorize?prompt=consent',
    tokenUrl: 'https://oauth.example.test/token',
    scopes: ['profile'],
  }, { state: 'local-state', codeChallenge: first.challenge })
  const parsed = new URL(authorizeUrl)
  require(parsed.searchParams.get('code_challenge') === first.challenge, 'authorization URL omitted the PKCE challenge')
  require(parsed.searchParams.get('response_type') === 'code', 'authorization URL omitted response_type=code')
  return 'two unique PKCE pairs generated; authorization URL carries S256 challenge'
}

async function checkOsv(context: RealUseContext): Promise<string> {
  const fetchImplementation = context.ports.fetchImplementation
  require(fetchImplementation !== undefined, 'OSV check requires an injected fetchImplementation port')
  let responseStatus: number | undefined
  const observingFetch: OSVFetch = async (input, init) => {
    const response = await fetchImplementation(input, init)
    responseStatus = response.status
    return response
  }
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 5_000)
  try {
    const vulnerabilities = await checkPackage('npm', 'typescript', undefined, {
      fetchImplementation: observingFetch,
      signal: controller.signal,
    })
    require(responseStatus !== undefined, 'OSV fetch did not complete')
    require(responseStatus >= 200 && responseStatus < 300, `OSV returned HTTP ${responseStatus}`)
    return `OSV returned ${vulnerabilities.length} advisory record(s) for npm:typescript`
  } finally {
    clearTimeout(timeout)
  }
}

async function checkAcp(): Promise<string> {
  let seen: { sessionId: string; text: string } | undefined
  const server = new AcpServer({
    promptHandler: async ({ session, text }) => {
      seen = { sessionId: session.sessionId, text }
      return { ok: true, echo: text }
    },
    toolListProvider: () => [],
  })
  const initialized = server.initialize()
  const capabilities = initialized.capabilities
  require(isRecord(capabilities) && capabilities.streaming === true, 'ACP server did not advertise streaming')
  const opened = server.openSession('/tmp/xerxes-real-use', { model: 'gpt-4o' })
  const sessionId = stringField(opened, 'session_id')
  const prompt = await server.prompt(sessionId, 'native ACP prompt')
  require(isRecord(prompt) && prompt.ok === true, 'ACP prompt handler did not respond')
  require(seen !== undefined, 'ACP prompt handler did not observe the session')
  require(seen.sessionId === sessionId && seen.text === 'native ACP prompt', 'ACP prompt was not routed to the session handler')
  const permission = server.requestPermission({
    sessionId,
    toolName: 'write_file',
    description: 'write a temporary file',
    inputs: {},
  })
  require(server.respondPermission(stringField(permission, 'permission_id'), false).ok, 'ACP permission response was not accepted')
  require(server.cancel(sessionId).ok, 'ACP session was not cancelled')
  require(server.closeSession(sessionId).ok, 'ACP session was not closed')
  return 'session opened, prompted, permission-denied, cancelled, and closed'
}

async function checkCron(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('cron', async directory => {
    const store = new JobStore(join(directory, 'jobs.json'))
    const job = store.add(new CronJob({ id: 'real-use-cron', prompt: 'say hello', schedule: '* * * * *' }))
    let archivePath = ''
    const scheduler = new CronScheduler(store, current => `ran:${current.id}`, {
      onComplete: async (current, output) => {
        archivePath = await routeOutput({ platform: 'none' }, output, {
          archiveDirectory: join(directory, 'archive'),
          jobId: current.id,
        })
      },
    })
    require((await scheduler.tick(new Date('2026-07-13T12:00:00.000Z'))).length === 0, 'new cron job fired before scheduling')
    require((await scheduler.tick(new Date('2026-07-13T12:01:00.000Z'))).join(',') === job.id, 'cron job did not fire on schedule')
    require(await readFile(archivePath, 'utf8') === `ran:${job.id}`, 'cron output was not archived')
    return `scheduled ${job.id} and archived one local output`
  })
}

async function checkVoiceMode(): Promise<string> {
  let recording = false
  const recorder: VoiceRecorder = {
    get recording() { return recording },
    start() { recording = true },
    stop() {
      recording = false
      return { bytes: new Uint8Array([82, 73, 70, 70]), filename: 'local.wav', mediaType: 'audio/wav' }
    },
  }
  const controller = new VoiceModeController({
    recorder,
    transcription: {
      providerName: 'local-transcription',
      async transcribe(request) {
        return { backend: 'local-transcription', model: request.model ?? 'local-stt', text: 'captured local audio' }
      },
    },
  })
  await controller.start()
  require(controller.recording, 'voice controller did not start the injected recorder')
  const result = await controller.stopAndTranscribe({ model: 'local-stt' })
  require(result.text === 'captured local audio' && !controller.recording, 'voice controller did not stop and transcribe')
  return 'injected recorder captured 4 local bytes and returned a deterministic transcription'
}

async function checkMediaRegistries(): Promise<string> {
  const vision = new VisionRegistry()
  vision.register('local', {
    providerName: 'local-vision',
    async analyze(request) {
      return {
        provider: 'local-vision',
        model: request.model ?? 'local-vision-model',
        prompt: request.prompt ?? 'Describe this image in detail.',
        response: request.image.kind === 'base64' ? 'a local PNG' : 'a local URL image',
      }
    },
  })
  const analysis = await vision.analyze('local', { image: visionImageFromBase64('UE5H'), prompt: 'Describe it' })
  require(analysis.response === 'a local PNG', 'vision registry did not dispatch the registered local provider')

  const images = new ImageGenerationRegistry()
  images.register('local', {
    providerName: 'local-image',
    async generate(request) {
      return {
        count: 1,
        model: request.model ?? 'local-image-model',
        size: request.size ?? '64x64',
        images: [{ b64: 'UE5H', format: 'png', revisedPrompt: request.prompt }],
      }
    },
  })
  const generated = await images.generate('local', { prompt: 'a local moon' })
  require(generated.images[0]?.b64 === 'UE5H', 'image registry did not dispatch the registered local provider')
  return 'vision and image-generation registries dispatched deterministic local providers'
}

async function checkBrowserRegistry(): Promise<string> {
  const registry = createBrowserProviderRegistry()
  const names = registry.names()
  require(names.length === SUPPORTED_BROWSER_PROVIDERS.length, 'browser registry omitted a native provider role')
  require(names.every(name => SUPPORTED_BROWSER_PROVIDERS.includes(name)), 'browser registry included an unknown provider role')
  return `registered ${names.length} browser roles without opening a browser: ${names.join(', ')}`
}

async function checkSshSandbox(): Promise<string> {
  let received: { readonly argv: readonly string[]; readonly host: string; readonly toolName: string } | undefined
  const adapter = new SshSandboxAdapter({
    config: { host: 'build.example.test', port: 2222, user: 'runner' },
    host: {
      async executeRemote(request) {
        received = { argv: request.command.argv, host: request.connection.host, toolName: request.toolName }
        return { exitCode: 0, stdout: 'ok', stderr: '', timedOut: false, truncated: false }
      },
    },
  })
  const output = JSON.parse(await adapter.execute({
    arguments: { cmd: 'bun', args: ['--version'], workdir: 'project' },
    context: { metadata: {} },
    toolName: 'exec_command',
  })) as Record<string, unknown>
  const dispatch = received
  require(dispatch !== undefined, 'SSH adapter did not invoke the injected transport')
  require(dispatch.argv.join(' ') === 'bun --version', 'SSH adapter did not preserve direct argv')
  require(dispatch.host === 'build.example.test' && dispatch.toolName === 'exec_command', 'SSH adapter changed host dispatch metadata')
  require(output.exitCode === 0, 'SSH adapter did not serialize the host result')
  return 'SSH sandbox delivered a direct argv request to the injected host without spawning ssh'
}

async function checkFileSync(): Promise<string> {
  const copied: string[] = []
  const result = await syncPush([
    { localPath: 'small.txt', remotePath: 'inputs/small.txt' },
    { localPath: 'large.bin', remotePath: 'inputs/large.bin' },
  ], {
    async copy(request) { copied.push(`${request.source}->${request.destination}`) },
    async stat(request) {
      if (request.path.endsWith('small.txt')) return { size: 4 }
      if (request.path.endsWith('large.bin')) return { size: 20_000 }
      return undefined
    },
  }, {
    localRoot: '/host/workspace',
    remoteRoot: '/sandbox/workspace',
    maxBytes: 10_000,
  })
  require(result[0]?.status === 'copied', 'small file was not synced')
  const oversized = result[1]
  require(oversized !== undefined && oversized.status === 'skipped' && oversized.reason === 'max_bytes_exceeded', 'oversized file was not filtered')
  require(copied.length === 1 && copied[0]?.includes('/host/workspace/small.txt'), 'file sync copied an unexpected source')
  return 'file sync copied one 4-byte input and skipped one oversized input through injected ports'
}

async function checkParserMatrix(): Promise<string> {
  const fullwidthBar = '\uff5c'
  const lowerEighthBlock = '\u2581'
  const deepSeekOpen = '<' + fullwidthBar + 'tool' + lowerEighthBlock + 'call' + lowerEighthBlock + 'begin' + fullwidthBar + '>'
  const deepSeekClose = '<' + fullwidthBar + 'tool' + lowerEighthBlock + 'call' + lowerEighthBlock + 'end' + fullwidthBar + '>'
  const samples: Readonly<Record<string, string>> = {
    xml_tool_call: '<tool_call>{"name":"read","arguments":{}}</tool_call>',
    llama: '<|python_tag|>{"name":"read","parameters":{}}<|eom_id|>',
    mistral: '[TOOL_CALLS][{"name":"read","arguments":{}}]',
    qwen: '<tool_call>{"name":"read","arguments":{}}</tool_call>',
    qwen3_coder: '|<function_call_start|>{"name":"read","parameters":{}}|<function_call_end|>',
    deepseek_v3: `${deepSeekOpen}{"name":"read","arguments":{}}${deepSeekClose}`,
    deepseek_v3_1: '<tool>{"name":"read","arguments":{}}</tool>',
    glm45: '<tool_call>{"name":"read","arguments":{}}</tool_call>',
    glm47: '<function_call>{"name":"read","arguments":{}}</function_call>',
    kimi_k2: '<|tool_call|>{"name":"read","arguments":{}}<|/tool_call|>',
    longcat: '<longcat:tool>{"name":"read","arguments":{}}</longcat:tool>',
  }
  const formats = Object.keys(TOOL_CALL_PARSER_REGISTRY)
  require(formats.length === Object.keys(samples).length, 'parser registry and real-use fixture matrix diverged')
  for (const format of formats) {
    const parser = getToolCallParser(format)
    const sample = samples[format]
    require(parser !== undefined && sample !== undefined, `parser fixture missing for ${format}`)
    const calls = parser.parse(sample)
    require(calls.length === 1 && calls[0]?.name === 'read', `${format} did not parse its real wire form`)
  }
  return `parsed ${formats.length} provider wire formats through the native registry`
}

async function checkRlTraining(): Promise<string> {
  const states = [
    { status: 'running', iteration: 1, reward: 0.25 },
    { status: 'succeeded', iteration: 2, reward: 1 },
  ]
  const client = new TinkerClient({
    transport: {
      createRun: payload => {
        require(payload.env === 'xerxes-terminal-test', 'RL payload did not retain environment')
        return { id: 'local-run-1' }
      },
      getRun: () => states.shift() ?? { status: 'succeeded', iteration: 2, reward: 1 },
      cancelRun: () => true,
    },
  })
  const runId = await client.start(new TinkerRunConfig({ model: 'local/model', env: 'xerxes-terminal-test', steps: 2 }))
  const first = await client.status(runId)
  const final = await client.status(runId)
  require(first.status === RLRunStatus.RUNNING, 'RL client did not normalize a running state')
  require(final.status === RLRunStatus.SUCCEEDED && final.reward === 1, 'RL client did not normalize a successful state')
  require((await client.cancel(runId)) === true, 'RL client did not route cancellation through the host transport')
  const environments = builtinEnvironments().listEnvironments()
  require(environments.some(environment => environment.name === 'xerxes-terminal-test'), 'built-in RL environment catalog omitted terminal test')
  return `injected Tinker transport completed running→succeeded lifecycle and accepted cancellation; ${environments.length} environments registered`
}

async function checkTrajectoryCompression(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('trajectories', async directory => {
    const outputPath = join(directory, 'trajectories.jsonl')
    const metricsPath = join(directory, 'metrics.json')
    const compressor = new TrajectoryCompressor({
      compressor: new ContextCompressor({
        contextWindow: 4,
        threshold: 0.5,
        protectFirst: 1,
        protectLast: 1,
        summaryMinTokens: 1,
        summarizer: naiveSummarizer,
      }),
    })
    const run = await compressor.run([
      {
        id: 'trajectory-1',
        messages: [
          { role: 'user', content: 'start' },
          { role: 'assistant', content: 'middle one' },
          { role: 'user', content: 'middle two' },
          { role: 'assistant', content: 'middle three' },
          { role: 'user', content: 'finish' },
        ],
      },
      { id: 'trajectory-2', messages: [{ role: 'user', content: 'short' }] },
    ], { outPath: outputPath, metricsPath })
    require(run.processed === 2 && run.errors.length === 0, 'trajectory compressor did not process both local trajectories')
    const lines = (await readFile(outputPath, 'utf8')).trim().split(/\r?\n/)
    require(lines.length === 2, 'trajectory compressor did not write JSONL output')
    require((await Bun.file(metricsPath).exists()), 'trajectory compressor did not write metrics')
    return `compressed ${run.processed} trajectories and wrote ${lines.length} JSONL record(s)`
  })
}

async function checkBatchRunner(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('batch-runner', async directory => {
    const outputPath = join(directory, 'results.jsonl')
    const runner = new BatchRunner(async record => {
      if (record.id === 'bad') throw new Error('simulated local failure')
      return {
        id: record.id,
        response: `response:${record.prompt}`,
        inputTokens: 10,
        outputTokens: 5,
        costUsd: 0.001,
      }
    }, 2)
    const summary = await runner.run([
      { id: 'one', prompt: 'one' },
      { id: 'two', prompt: 'two' },
      { id: 'three', prompt: 'three' },
      { id: 'bad', prompt: 'bad' },
    ], { outPath: outputPath })
    require(summary.total === 4 && summary.succeeded === 3 && summary.failed === 1, 'batch runner summary changed')
    const lines = (await readFile(outputPath, 'utf8')).trim().split(/\r?\n/)
    require(lines.length === 4, 'batch runner did not write one JSONL line per record')
    return `processed ${summary.total} batch records (${summary.succeeded} succeeded, ${summary.failed} failed)`
  })
}

async function checkSessions(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('sessions', async directory => {
    const store = new SQLiteSessionStore({ dbPath: join(directory, 'sessions.db') })
    try {
      store.saveSession(new SessionRecord({
        sessionId: 'root',
        workspaceId: 'real-use',
        turns: [new TurnRecord({ turnId: 'turn-1', prompt: 'native session prompt', responseContent: 'native response' })],
      }))
      const child = branchSession(store, { sourceSessionId: 'root', newSessionId: 'branch' })
      require(child.turns.length === 1, 'branched session did not retain the source turn')
      require(sessionLineage(store, 'branch').join(',') === 'branch,root', 'session lineage did not retain parentage')
      require(store.search('native session').length > 0, 'SQLite session search did not index the persisted turn')
      return 'SQLite record persisted, indexed, branched, and traversed to its root'
    } finally {
      store.close()
    }
  })
}

async function checkSnapshot(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('snapshot', async directory => {
    const workspace = join(directory, 'workspace')
    const shadowRoot = join(directory, 'shadow')
    await mkdir(workspace, { recursive: true })
    await Bun.write(join(workspace, 'note.txt'), 'version one\n')
    const manager = new SnapshotManager(workspace, { shadowRoot })
    const snapshot = manager.snapshot('baseline')
    await Bun.write(join(workspace, 'note.txt'), 'broken version\n')
    manager.rollback(snapshot.id)
    require(await readFile(join(workspace, 'note.txt'), 'utf8') === 'version one\n', 'snapshot rollback did not restore the original file')
    return `shadow snapshot ${snapshot.id.slice(0, 8)} created and rolled back`
  })
}

async function checkBridgeCommands(): Promise<string> {
  const model = resolveCommand('/model gpt-4.1')
  const exit = resolveCommand('/q')
  require(model?.name === 'model', 'bridge command registry did not resolve /model')
  require(exit?.name === 'exit', 'bridge command registry did not resolve /q alias')
  require(COMMAND_REGISTRY.length > 20, 'bridge command registry is unexpectedly incomplete')
  return `resolved canonical model and exit alias from ${COMMAND_REGISTRY.length} native bridge commands`
}

async function checkDoctor(): Promise<string> {
  const report = runAllDoctorChecks({
    environment: {},
    home: '/tmp/xerxes-real-use-doctor-home',
    fileExists: () => false,
  })
  require(report.some(diagnosis => diagnosis.name === 'bun' && diagnosis.severity === 'ok'), 'Bun doctor did not confirm the active runtime')
  require(!hasDoctorFailures(report), 'Bun doctor reported a host failure')
  const warnings = report.filter(diagnosis => diagnosis.severity === 'warn').map(diagnosis => diagnosis.name)
  return warnings.length ? `doctor completed with warnings: ${warnings.join(', ')}` : `doctor completed ${report.length} checks without warnings`
}

async function checkSetupWizard(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('setup-wizard', async directory => {
    const result = runSetupWizard({ provider: 'openai', model: 'gpt-4o' })
    require(result.answers.provider === 'openai' && result.answers.model === 'gpt-4o', 'setup wizard did not preserve supplied answers')
    const path = join(directory, 'config.yaml')
    await writeSetupConfig(result.answers, path)
    const config = await readFile(path, 'utf8')
    require(config.includes('provider: "openai"') && config.includes('model: "gpt-4o"'), 'setup wizard did not write native YAML')
    return `setup wizard wrote ${config.length} bytes of YAML-compatible configuration`
  })
}

async function checkUpdatePlanning(): Promise<string> {
  const plan = planBunUpdate({ bunExecutable: 'bun', packageSpec: '@xerxes/runtime@1.2.3' })
  require(plan.argv.join(' ') === 'bun add --global @xerxes/runtime@1.2.3', 'Bun update plan did not preserve explicit package spec')
  const check = await checkBunPackageUpdate({
    packageName: 'xerxes-agent',
    currentVersion: '1.0.0',
    fetch: async () => Response.json({ version: '1.1.0' }),
  })
  require(check.latestVersion === '1.1.0' && check.updateAvailable === true, 'injected package metadata was not parsed')
  return 'Bun update plan stayed explicit; injected package metadata reported 1.0.0→1.1.0'
}

async function checkMessagingState(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('messaging-state', async directory => {
    const user = hashUser('telegram', 123, { salt: 'real-use-salt' })
    require(user === hashUser('telegram', 123, { salt: 'real-use-salt' }), 'identity hash was not stable')
    require(hashChat('slack', 'C1', { salt: 'real-use-salt' }).startsWith('slack:'), 'chat hash lost platform prefix')
    const cache = new StickerCache(directory, { clock: () => 123, lruSize: 2 })
    cache.put('telegram', 'sticker-1', join(directory, 'one.webp'))
    require(cache.get('telegram', 'sticker-1')?.localPath.endsWith('one.webp'), 'sticker cache did not retain a local record')
    const policy = createSessionResetPolicy({ trigger: ResetTrigger.TIMEOUT, timeoutMinutes: 10 })
    const now = new Date('2026-07-13T12:00:00.000Z')
    require(shouldReset(policy, {
      lastMessageAt: new Date('2026-07-13T11:40:00.000Z'),
      messageCount: 0,
      now,
    }), 'timeout reset policy did not trigger')
    return 'identity hash, persistent sticker index, and timeout reset policy completed locally'
  })
}

async function checkSecurity(context: RealUseContext): Promise<string> {
  const secret = 'api_key=sk-12345678901234567890'
  const redacted = redactString(secret)
  require(!redacted.includes('12345678901234567890'), 'credential redaction leaked its secret suffix')
  const policy = new ToolPolicy({ allow: ['read_file'] })
  require(policy.evaluate('read_file') === 'allow', 'allow-listed tool was denied')
  require(policy.evaluate('write_file') === 'deny', 'unlisted tool was allowed')
  require(!checkUrl('http://127.0.0.1/private').allowed, 'loopback URL was not rejected')
  return await context.withTemporaryDirectory('security', async directory => {
    const resolved = await resolveWithin(directory, '/escape.txt')
    require(resolved !== '/escape.txt' && resolved.endsWith('/escape.txt'), 'absolute/re-rooted path contract changed')
    const approvalPath = join(directory, 'approvals.json')
    const approvals = new ApprovalStore({ persistencePath: approvalPath })
    approvals.add({ toolName: 'rm', scope: ApprovalScope.ALWAYS, granted: true })
    const restored = new ApprovalStore({ persistencePath: approvalPath })
    require(restored.check('rm', 'any-session') === true, 'persistent approval did not round-trip')
    return 'credential redaction, policy, URL/path containment, and persisted approval passed'
  })
}

async function checkSkillsHub(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('skills-hub', async directory => {
    const sourceRoot = join(directory, 'source')
    const targetRoot = join(directory, 'installed')
    await mkdir(join(sourceRoot, 'alpha'), { recursive: true })
    await writeFile(join(sourceRoot, 'alpha', 'SKILL.md'), '---\nversion: 1.0\n---\n# Alpha\nNative local skill.\n', 'utf8')
    const source = new LocalSkillSource({ root: sourceRoot })
    const hits = await source.search('alpha')
    require(hits.some(hit => hit.name === 'alpha'), 'local skill source did not find alpha')
    const synced = await syncSkillManifest([{ source: 'local', identifier: 'alpha' }], { local: source }, {
      targetDirectory: targetRoot,
    })
    require(synced.installed.includes('alpha'), 'skill manifest did not install alpha')
    require(await Bun.file(join(targetRoot, 'alpha', 'SKILL.md')).exists(), 'installed skill bundle is missing')
    return 'local skill source found alpha and manifest sync installed one SKILL.md bundle'
  })
}

async function checkRuntimeRouting(): Promise<string> {
  require(classifyError(new Error('HTTP 429 too many requests')).kind === ErrorKind.RATE_LIMIT, 'error classifier did not identify 429')
  const rates = new RateLimitTracker({ throttleRatio: 0.1, now: () => 100 })
  rates.update('openai', 'gpt-4o', {
    'x-ratelimit-limit-requests': '100',
    'x-ratelimit-remaining-requests': '3',
  })
  require(rates.shouldThrottle('openai', 'gpt-4o'), 'rate-limit tracker did not throttle a low budget')
  const auxiliary = new AuxiliaryClient({
    model: 'local-auxiliary',
    backend: request => ({ text: `summary:${request.messages.length}`, requestTokens: 3, responseTokens: 2 }),
    monotonicNow: () => 1,
  })
  require(await auxiliary.summarize([{ role: 'user', content: 'hello' }]) === 'summary:2', 'auxiliary client did not use injected backend')
  return '429 classified, low rate budget throttled, and injected auxiliary summary completed'
}

async function checkPricingInsights(): Promise<string> {
  const providerCount = Object.keys(listAllModels()).length
  require(providerCount > 0, 'provider registry did not expose any model catalogs')
  const directCost = calcCost('gpt-4o', 1_000_000, 500_000)
  require(directCost > 0, 'provider registry did not price gpt-4o')

  const tracker = new CostTracker({
    now: () => new Date('2026-07-13T12:00:00.000Z'),
    sessionId: 'real-use-pricing',
  })
  tracker.recordTurn('gpt-4o', 100, 20, 'completion', { cacheReadTokens: 50 })
  tracker.record(new CostEvent({
    model: 'local-image',
    inputTokens: 0,
    outputTokens: 0,
    costUsd: 0.002,
    label: 'image',
    timestamp: '2026-07-13T12:00:00.000Z',
  }))
  const insights = buildInsightsReport(tracker.events, { now: new Date('2026-07-13T12:00:00.000Z') })
  require(tracker.eventCount === 2 && insights.totalEvents === 2, 'cost ledger did not preserve both local events')
  require(insights.byModel['gpt-4o']?.events === 1, 'insights did not group the completion by model')
  require(insights.totalCacheReadTokens === 50, 'insights did not preserve prompt-cache tokens')
  return `priced ${providerCount} provider catalog(s), recorded ${insights.totalEvents} ledger events, and aggregated cache usage`
}

async function checkInteractiveRuntime(): Promise<string> {
  const state = createAgentState([{ role: 'user', content: 'hello native runtime' }])
  const router = new BridgeSlashRouter({
    cwd: '/tmp/xerxes-real-use',
    config: { model: 'gpt-4o' },
    state,
  })
  const help = await router.dispatch('/h')
  require(help.status === 'handled' && help.command === 'help', 'bridge slash router did not handle /h')

  const sessions = new BackgroundSessionManager({
    idFactory: () => 'background-real-use',
    now: () => 1,
    runner: session => `completed:${session.prompt}`,
  })
  const submitted = sessions.submit('local background task')
  const settled = await sessions.wait(submitted.id, { settled: true })
  require(
    settled?.status === BackgroundStatus.SUCCEEDED && settled.result === 'completed:local background task',
    'background session did not reach its deterministic successful result',
  )
  await sessions.shutdown()

  const plugins = new SlashPluginRegistry()
  plugins.register('native-demo', () => 'ok', { aliases: ['nd'] })
  require(plugins.resolve('/nd')?.command.name === 'native-demo', 'slash plugin registry did not resolve its alias')
  return 'bridge /h routing, one background completion, and one plugin alias completed locally'
}

async function checkRuntimePrimitives(): Promise<string> {
  const budget = new IterationBudget(2)
  require(budget.consume(2) === 2 && budget.exhausted, 'iteration budget did not charge its bounded capacity')
  let exhausted = false
  try {
    budget.consume()
  } catch (error) {
    exhausted = error instanceof BudgetExhausted
  }
  require(exhausted, 'iteration budget did not raise BudgetExhausted')
  require(budget.refund() === 1 && budget.remaining === 1, 'iteration budget did not refund one charge')

  const registry = new ProcessRegistry({ idFactory: () => 'process-real-use', now: () => 1 })
  const processHandle = Bun.spawn([process.execPath, '-e', 'process.exit(0)'], { stdout: 'ignore', stderr: 'ignore' })
  const processId = registry.register(processHandle, { name: 'quick-native-bun', command: 'bun -e process.exit(0)' })
  require(await registry.wait(processId, 5) === 0, 'process registry did not report the local Bun child exit code')

  const interrupt = new InterruptToken()
  interrupt.set()
  require(interrupt.isSet() && interrupt.signal.aborted && await interrupt.wait(1), 'interrupt token did not abort and wake its waiter')
  const firstId = deterministicToolCallId('read_file', { path: 'README.md', recursive: false })
  const secondId = deterministicToolCallId('read_file', { recursive: false, path: 'README.md' })
  require(firstId === secondId, 'deterministic tool-call ID changed with object key ordering')
  return 'budget exhaustion/refund, settled process lookup, interruption, and stable tool-call ID completed locally'
}

async function checkFileSessionPersistence(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('file-session', async directory => {
    const store = new FileSessionStore({ baseDirectory: directory })
    const session = new SessionRecord({
      sessionId: 'file-session',
      workspaceId: 'file-workspace',
      turns: [new TurnRecord({ turnId: 'turn-1', prompt: 'persist this locally', responseContent: 'stored' })],
    })
    store.saveSession(session)
    const loaded = store.loadSession(session.sessionId)
    require(loaded?.turns[0]?.prompt === 'persist this locally', 'file session store did not round-trip the persisted turn')
    const entries = await readdir(join(directory, 'file-workspace'))
    require(entries.some(entry => entry === 'file-session.json'), 'file session store did not write its JSON record')
    require(!entries.some(entry => entry.endsWith('.tmp')), 'file session store left an atomic-write temporary artifact')
    return `file session ${loaded.sessionId} round-tripped with ${entries.length} durable workspace record(s)`
  })
}

async function checkDistribution(): Promise<string> {
  const platform = detectPlatform({
    platform: 'linux',
    release: 'Termux 14',
    bunVersion: '1.2.0',
    environment: { PREFIX: '/data/data/com.termux/files/usr' },
  })
  require(platform.isLinux && platform.isTermux, 'distribution helper did not recognize injected Termux platform facts')
  const dependencies = filterTermuxDependencies({ playwright: '^1.0.0', typescript: '^5.0.0' })
  require(dependencies.playwright === undefined && dependencies.typescript === '^5.0.0', 'Termux dependency filter changed')
  const formula = renderHomebrewFormula({
    tarballUrl: 'https://example.test/xerxes-1.0.0.tar.gz',
    version: '1.0.0',
    sha256: 'a'.repeat(64),
  })
  require(formula.includes('license "Apache-2.0"'), 'Homebrew formula omitted Apache-2.0 license metadata')
  require(BUN_SHELL_INSTALL_SNIPPET.includes('XERXES_PACKAGE'), 'Bun installer no longer requires an explicit package spec')
  return 'Termux facts, unsupported dependency filtering, Homebrew template, and explicit Bun installer requirement passed'
}

async function checkMessagingTools(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('messaging-tools', async directory => {
    const messages = new OutboundMessageRegistry()
    messages.register('local', async (_platform, recipient, payload) => ({
      ok: true,
      recipient,
      text: payload.text,
    }))
    const sent = await sendMessage({ platform: 'local', recipient: 'room-1', text: 'hello' }, messages)
    require(sent.ok === true && sent.recipient === 'room-1', 'outbound message registry did not dispatch the local adapter')
    const answer = await clarify({
      question: 'Choose a local answer',
      choices: ['first', 'second'],
      asker: new StaticAsker({ index: 1 }),
    })
    require(answer.answer === 'second' && answer.selected_index === 1, 'clarify did not return the selected local choice')

    const memory = new WorkspaceMemoryStore({ workspaceRoot: directory })
    const addedMemory = await memory.add('memory', 'keep this local memory')
    const addedUser = await memory.add('user', 'keep this local user preference')
    require(addedMemory.ok && addedUser.ok, 'workspace memory store did not add local entries')
    const listed = await memory.list('memory')
    require(listed.items[0]?.content === 'keep this local memory', 'workspace memory store did not list its saved entry')
    const removed = await memory.remove('user', 1)
    require(removed.ok, 'workspace memory store did not remove its saved user entry')
    return 'local outbound message, deterministic clarification, and workspace memory CRUD completed through injected adapters'
  })
}

async function checkCredentialStorage(context: RealUseContext): Promise<string> {
  return await context.withTemporaryDirectory('credentials', async directory => {
    const storage = new CredentialStorage(join(directory, 'credentials'), { credentialKey: 'real-use-test-key' })
    const token = new OAuthToken({ accessToken: 'local-access-token', refreshToken: 'local-refresh-token', scopes: ['profile'] })
    const path = await storage.save('local', token)
    const encrypted = await readFile(path, 'utf8')
    require(!encrypted.includes(token.accessToken), 'credential store wrote the access token in plaintext')
    const loaded = await storage.load('local')
    require(loaded?.accessToken === token.accessToken && loaded.refreshToken === token.refreshToken, 'encrypted credential did not round-trip')
    require(await storage.remove('local'), 'temporary credential was not removed')
    return 'AES-GCM credential round-trip succeeded without writing plaintext tokens'
  })
}

async function checkNativeSwarm(): Promise<string> {
  const report = await runSwarmIntegration()
  require(report.ok, `native swarm failed ${report.failed} category check(s)`)
  return `completed ${report.passed} native runtime categories (orchestration, executor, Cortex, security)`
}

async function checkStreaming(): Promise<string> {
  const sse = [...parseSseStream([
    'event: response.output_text.delta\n',
    'data: hello\n\n',
    'event: response.completed\n',
    'data: {}\n\n',
  ])]
  require(sse.length === 2 && sse[0]?.event === 'response.output_text.delta', 'SSE parser lost the text event')
  const translator = new ResponsesEventTranslator()
  const output = [...translator.translateAll([
    { type: 'response.output_text.delta', delta: 'Hello ' },
    { type: 'response.output_text.delta', delta: 'world' },
    { type: 'response.output_item.added', item: { type: 'function_call', id: 'call-1', name: 'read_file' } },
    { type: 'response.function_call_arguments.delta', item_id: 'call-1', delta: '{"path":"README.md"}' },
    { type: 'response.output_item.done', item: { type: 'function_call', id: 'call-1', name: 'read_file' } },
    { type: 'response.completed', response: { status: 'completed', usage: { input_tokens: 3, output_tokens: 2 } } },
  ])]
  require(output.filter(delta => delta.content !== undefined).map(delta => delta.content).join('') === 'Hello world', 'Responses translator lost text deltas')
  require(translator.usage.toolCalls.length === 1, 'Responses translator did not assemble the tool call')
  return `parsed ${sse.length} SSE events and assembled ${translator.usage.toolCalls.length} streamed tool call`
}

async function skipWithoutNetwork(context: RealUseContext): Promise<string | undefined> {
  if (!context.allowNetwork) return 'network checks disabled; rerun with --online or supply allowNetwork: true'
  if (!context.ports.fetchImplementation || !context.ports.networkProbe) {
    return 'no fetchImplementation and networkProbe ports supplied; public OSV check was not attempted'
  }
  return await context.isNetworkAvailable() ? undefined : 'network probe did not confirm public connectivity'
}

function skipWithoutGit(context: RealUseContext): string | undefined {
  const executable = (context.ports.findExecutable ?? Bun.which)('git')
  return executable ? undefined : 'git executable is unavailable; shadow snapshot was not attempted'
}

function skipWithoutProviderProbe(context: RealUseContext): string | undefined {
  return context.ports.providerProbe
    ? undefined
    : 'no providerProbe supplied; credentialed provider was not invoked'
}

function skipWithoutMediaProbe(context: RealUseContext): string | undefined {
  return context.ports.mediaProbe
    ? undefined
    : 'no mediaProbe supplied; credentialed or hardware media operation was not invoked'
}

function skipWithoutPackageRegistryProbe(context: RealUseContext): string | undefined {
  return context.ports.packageRegistryProbe
    ? undefined
    : 'no packageRegistryProbe supplied; public package registry was not queried'
}

function skipWithoutTuiProbe(context: RealUseContext): string | undefined {
  return context.ports.tuiProbe
    ? undefined
    : 'no tuiProbe supplied; live terminal UI was not opened'
}

function createContext(options: RunRealUseOptions): RealUseContext {
  const ports = options.ports ?? {}
  const allowNetwork = options.allowNetwork ?? false
  let networkAvailability: Promise<boolean> | undefined
  const withTemporaryDirectory = ports.temporaryDirectory ?? defaultTemporaryDirectory
  return {
    ports,
    allowNetwork,
    withTemporaryDirectory,
    isNetworkAvailable: async () => {
      if (!ports.networkProbe) return false
      networkAvailability ??= ports.networkProbe()
      try {
        return await networkAvailability
      } catch {
        return false
      }
    },
  }
}

async function defaultTemporaryDirectory<T>(label: string, run: (directory: string) => Promise<T>): Promise<T> {
  const directory = await mkdtemp(join(tmpdir(), `xerxes-bun-real-use-${safeLabel(label)}-`))
  try {
    return await run(directory)
  } finally {
    await rm(directory, { recursive: true, force: true })
  }
}

async function defaultNetworkProbe(fetchImplementation: OSVFetch): Promise<boolean> {
  const controller = new AbortController()
  const timeout = setTimeout(() => controller.abort(), 3_000)
  try {
    const response = await fetchImplementation('https://api.github.com/zen', {
      headers: { accept: 'text/plain' },
      signal: controller.signal,
    })
    return response.ok
  } catch {
    return false
  } finally {
    clearTimeout(timeout)
  }
}

function resultFor(check: RealUseCheck, status: RealUseStatus, detail: string | void): RealUseResult {
  return Object.freeze({
    plan: check.plan,
    feature: check.feature,
    status,
    detail: redactString(detail ?? ''),
  })
}

function resultOf(value: unknown): Record<string, unknown> {
  require(isRecord(value) && 'result' in value && isRecord(value.result), 'MCP request did not return a result object')
  return value.result
}

function stringField(value: Record<string, unknown>, name: string): string {
  const field = value[name]
  require(typeof field === 'string' && field.length > 0, `missing string field: ${name}`)
  return field
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function require(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}

function fail(message: string): never {
  throw new Error(message)
}

function errorName(error: unknown): string {
  return error instanceof Error && error.name ? error.name : 'Error'
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function safeLabel(value: string): string {
  const label = value.replaceAll(/[^a-z0-9-]+/gi, '-').replaceAll(/^-+|-+$/g, '')
  return label || 'check'
}

function statusMarker(status: RealUseStatus): string {
  if (status === 'passed') return '✓'
  if (status === 'skipped') return '·'
  return '✗'
}

function wrapDetail(detail: string): string[] {
  const width = 92
  const words = redactString(detail).split(/\s+/).filter(Boolean)
  if (!words.length) return []
  const lines: string[] = []
  let current = '      '
  for (const word of words) {
    if (current.length > 6 && current.length + word.length + 1 > width) {
      lines.push(current)
      current = `      ${word}`
    } else {
      current += current.length === 6 ? word : ` ${word}`
    }
  }
  lines.push(current)
  return lines
}

if (import.meta.main) {
  process.exitCode = await main()
}
