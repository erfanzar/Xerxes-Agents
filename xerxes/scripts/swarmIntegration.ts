// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  AgentOrchestrator,
  AgentSwitchTrigger,
  Cortex,
  CortexAgents,
  CortexCore,
  CortexPlanner,
  ExecutionContext,
  PolicyAction,
  PolicyEngine,
  SandboxMode,
  SandboxRouter,
  ShortTermMemory,
  SubAgentManager,
  TaskCreator,
  ToolPolicy,
  ToolRegistry,
  Xerxes,
  loadAgentDefinitions,
  type AgentDefinition,
  type CompletionRequest,
  type LlmClient,
  type LlmDelta,
  type ToolCall,
  type ToolDefinition,
} from '../src/index.js'

export interface SwarmReportEntry {
  readonly category: string
  readonly error: boolean
  readonly message: string
}

export interface SwarmIntegrationReport {
  readonly entries: readonly SwarmReportEntry[]
  readonly failed: number
  readonly ok: boolean
  readonly passed: number
}

type ReportLog = (message: string) => void
type TextResponse = (request: CompletionRequest) => string

const CALCULATOR_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'calculator',
    description: 'Evaluate one known deterministic integration expression.',
    parameters: {
      type: 'object',
      properties: { expression: { type: 'string' } },
      required: ['expression'],
      additionalProperties: false,
    },
  },
}

const MULTIPLIER_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'async_multiplier',
    description: 'Multiply two integration numbers.',
    parameters: {
      type: 'object',
      properties: { x: { type: 'number' }, y: { type: 'number' } },
      required: ['x', 'y'],
      additionalProperties: false,
    },
  },
}

const GREETER_TOOL: ToolDefinition = {
  type: 'function',
  function: {
    name: 'greeter',
    description: 'Greet an integration target or request an explicit handoff.',
    parameters: {
      type: 'object',
      properties: { name: { type: 'string' } },
      required: ['name'],
      additionalProperties: false,
    },
  },
}

/** A deterministic model port that first asks for one tool, then acknowledges its result. */
class ToolFirstLlm implements LlmClient {
  readonly requests: CompletionRequest[] = []

  constructor(
    private readonly requestedTool: ToolCall,
    private readonly completion: string,
  ) {}

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(snapshotRequest(request))
    if (request.messages.at(-1)?.role === 'tool') {
      yield { content: this.completion, usage: { inputTokens: 5, outputTokens: 4 } }
      return
    }
    yield {
      toolCalls: [this.requestedTool],
      usage: { inputTokens: 3, outputTokens: 1 },
    }
  }
}

/** Deterministic text-only model port used to exercise native Cortex agents. */
class TextLlm implements LlmClient {
  readonly requests: CompletionRequest[] = []

  constructor(private readonly response: TextResponse) {}

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(snapshotRequest(request))
    yield {
      content: this.response(request),
      usage: { inputTokens: 4, outputTokens: 3 },
    }
  }
}

/** Run the native Bun swarm integration exercise and return every verified observation. */
export async function runSwarmIntegration(): Promise<SwarmIntegrationReport> {
  const entries: SwarmReportEntry[] = []

  await runCategory(entries, 'ORCHESTRATOR', testAgentOrchestrator)
  await runCategory(entries, 'EXECUTOR', testToolRegistry)
  await runCategory(entries, 'XERXES', testXerxesFacade)
  await runCategory(entries, 'CORTEX', testCortexTopologies)
  await runCategory(entries, 'SUBAGENT', testSubAgentManager)
  await runCategory(entries, 'SECURITY', testPolicyAndSandbox)
  await runCategory(entries, 'DEFINITIONS', testDefinitions)

  const failed = entries.filter(entry => entry.error).length
  return Object.freeze({
    entries: Object.freeze([...entries]),
    passed: entries.length - failed,
    failed,
    ok: failed === 0,
  })
}

/** Format the integration result for direct `bun` invocation without hiding failures. */
export function formatSwarmReport(report: SwarmIntegrationReport): string {
  const lines = [
    '='.repeat(60),
    'XERXES AGENT SWARM — NATIVE BUN INTEGRATION',
    '='.repeat(60),
    '',
    'REPORT',
    '='.repeat(60),
  ]
  for (const entry of report.entries) {
    lines.push(`${entry.error ? '❌' : '✅'} ${entry.category}: ${entry.message}`)
  }
  lines.push('', `Passed: ${report.passed} | Failed: ${report.failed}`)
  lines.push(report.ok ? 'All native swarm checks passed.' : 'Some native swarm checks failed — see details above.')
  return lines.join('\n')
}

async function runCategory(
  entries: SwarmReportEntry[],
  category: string,
  operation: (log: ReportLog) => Promise<void>,
): Promise<void> {
  try {
    await operation(message => entries.push({ category, message, error: false }))
  } catch (error) {
    entries.push({ category, message: `failed: ${errorMessage(error)}`, error: true })
  }
}

async function testAgentOrchestrator(log: ReportLog): Promise<void> {
  const orchestrator = new AgentOrchestrator({ maxAgents: 2 })
  orchestrator.registerAgent({
    id: 'generalist',
    switchTriggers: [AgentSwitchTrigger.CAPABILITY_BASED],
  })
  orchestrator.registerAgent({ id: 'specialist' })

  expectThrows(
    () => orchestrator.registerAgent({ id: 'generalist' }),
    'Duplicate agent registration must be rejected',
  )
  log('Duplicate registration is rejected')

  expectThrows(
    () => orchestrator.registerAgent({ id: 'overflow' }),
    'Configured maximum number of agents must be enforced',
  )
  log('Configured maximum agent count is enforced')

  orchestrator.registerSwitchTrigger(
    AgentSwitchTrigger.CAPABILITY_BASED,
    context => context.needSpecialist === true ? 'specialist' : undefined,
  )
  requireCondition(
    orchestrator.shouldSwitchAgent({ needSpecialist: true }) === 'specialist',
    'Capability trigger did not select the specialist',
  )
  orchestrator.switchAgent('specialist', 'integration handoff')
  requireCondition(orchestrator.currentAgentId === 'specialist', 'Explicit handoff did not update the active agent')
  log('Capability trigger and explicit agent handoff work')
}

async function testToolRegistry(log: ReportLog): Promise<void> {
  const handoff = new AgentOrchestrator()
  handoff.registerAgent({ id: 'caller' })
  handoff.registerAgent({ id: 'specialist' })

  const registry = new ToolRegistry()
  registry.register(CALCULATOR_TOOL, inputs => ({ result: knownExpression(inputs.expression) }))
  registry.register(MULTIPLIER_TOOL, async inputs => {
    const x = inputs.x
    const y = inputs.y
    if (typeof x !== 'number' || typeof y !== 'number') throw new Error('x and y must be numbers')
    await Bun.sleep(1)
    return x * y
  })
  registry.register(GREETER_TOOL, inputs => {
    const name = inputs.name
    if (name === 'specialist') {
      handoff.switchAgent('specialist', 'greeter requested specialist')
      return 'Handing off to specialist.'
    }
    if (typeof name !== 'string') throw new Error('name must be text')
    return `Hello, ${name}!`
  })

  const context = { metadata: {} }
  const calculation = await registry.execute(toolCall('calculator', { expression: '2 + 3' }), context)
  requireCondition(calculation === '{"result":5}', 'Native calculator result did not serialize correctly')

  const multiplied = await Promise.all([
    registry.execute(toolCall('async_multiplier', { x: 2, y: 3 }, 'multiply-left'), context),
    registry.execute(toolCall('async_multiplier', { x: 5, y: 6 }, 'multiply-right'), context),
  ])
  requireCondition(multiplied.join(',') === '6,30', 'Parallel native tool dispatch returned unexpected values')
  log('Sequential and parallel native ToolRegistry dispatch work')

  const greeting = await registry.execute(toolCall('greeter', { name: 'specialist' }), context)
  requireCondition(greeting === 'Handing off to specialist.', 'Handoff tool returned an unexpected response')
  requireCondition(handoff.currentAgentId === 'specialist', 'Explicit tool-initiated handoff did not select specialist')
  log('Tool result can request an explicit orchestrator handoff')
}

async function testXerxesFacade(log: ReportLog): Promise<void> {
  const memory = new ShortTermMemory({ capacity: 8 })
  memory.save('The swarm retains native Bun memory for future turns.')
  const registry = new ToolRegistry()
  registry.register(CALCULATOR_TOOL, inputs => ({ result: knownExpression(inputs.expression) }))
  const client = new ToolFirstLlm(
    toolCall('calculator', { expression: '7 * 8' }, 'calculator-call'),
    'Native calculator result confirmed.',
  )
  const runtime = new Xerxes({
    agents: [agentDefinition('swarm-core', { tools: ['calculator'], allowedTools: ['calculator'] })],
    coreTools: false,
    llm: client,
    memory,
    memoryMinChars: 1,
    permissionMode: 'accept-all',
    policy: new PolicyEngine({ globalPolicy: new ToolPolicy({ allow: ['calculator'] }) }),
    systemPrompt: 'Use only the configured native tool.',
    toolRegistry: registry,
  })

  const result = await runtime.run('Calculate 7 * 8 and report the verified result.')
  requireCondition(result.output === 'Native calculator result confirmed.', 'Xerxes facade did not finish the tool turn')
  requireCondition(result.toolCalls.length === 1 && result.toolCalls[0] === 'calculator', 'Expected calculator tool call')
  requireCondition(memory.getRecent(2).length === 2, 'Completed facade turn was not retained in memory')
  const firstRequest = client.requests[0]
  requireCondition(firstRequest !== undefined, 'Facade did not issue an LLM request')
  requireCondition(
    firstRequest.messages.some(message => messageContent(message).includes('Relevant retained memory')),
    'Facade did not inject relevant native memory into the LLM request',
  )
  log('Facade drives injected LLM, tool registry, policy gate, and retained memory')

  const deniedClient = new ToolFirstLlm(
    toolCall('calculator', { expression: '7 * 8' }, 'denied-calculator-call'),
    'Denied tool call was handled safely.',
  )
  const denied = new Xerxes({
    agents: [agentDefinition('denied-core', { tools: ['calculator'], allowedTools: ['calculator'] })],
    coreTools: false,
    llm: deniedClient,
    permissionMode: 'accept-all',
    policy: new PolicyEngine({ globalPolicy: new ToolPolicy({ allow: ['echo'] }) }),
    toolRegistry: registry,
  })
  const deniedResult = await denied.run('Try the denied calculator tool.')
  requireCondition(deniedResult.output === 'Denied tool call was handled safely.', 'Denied facade turn did not complete')
  const denialRequest = deniedClient.requests[1]
  requireCondition(denialRequest !== undefined, 'Denied call did not return a tool result to the model')
  requireCondition(
    messageContent(denialRequest.messages.at(-1)).includes('Permission denied for calculator.'),
    'Policy denial was not preserved as an explicit tool result',
  )
  log('Policy denials are returned to the model without executing the tool')
}

async function testCortexTopologies(log: ReportLog): Promise<void> {
  const writerLlm = new TextLlm(() => 'native introduction')
  const editorLlm = new TextLlm(() => 'edited native introduction')
  const writer = cortexAgent('writer', 'Writer', writerLlm)
  const editor = cortexAgent('editor', 'Editor', editorLlm)
  const sequential = new Cortex({
    process: CortexCore.ProcessType.SEQUENTIAL,
    agents: [writer, editor],
    tasks: [
      { id: 'write', description: 'Write an introduction', expectedOutput: 'Introduction', agentId: 'writer' },
      {
        id: 'edit',
        description: 'Edit the introduction',
        expectedOutput: 'Edited introduction',
        agentId: 'editor',
        dependencies: ['write'],
        contextTaskIds: ['write'],
      },
    ],
  })
  const sequentialOutput = await sequential.kickoff()
  requireCondition(sequentialOutput.rawOutput === 'edited native introduction', 'Sequential Cortex output is incorrect')
  const editorRequest = editorLlm.requests[0]
  requireCondition(editorRequest !== undefined, 'Editor did not receive a Cortex task request')
  requireCondition(
    messageContent(editorRequest.messages.at(-1)).includes('Output from task write:\nnative introduction'),
    'Sequential Cortex did not pass dependency context to the editor',
  )
  log('Sequential topology drives native CortexAgent LLM ports with dependency context')

  const transitions: string[] = []
  const parallel = new Cortex({
    process: CortexCore.ProcessType.PARALLEL,
    maxParallel: 2,
    tasks: [
      { id: 'left', description: 'Left branch', expectedOutput: 'left' },
      { id: 'right', description: 'Right branch', expectedOutput: 'right' },
      { id: 'join', description: 'Join branches', expectedOutput: 'joined', dependencies: ['left', 'right'] },
    ],
    taskRunner: async context => {
      transitions.push(`start:${context.task.id}`)
      if (context.task.id === 'join') {
        requireCondition(transitions.includes('end:left') && transitions.includes('end:right'), 'Join began before both branches completed')
        return `${context.dependencyOutputs.get('left')?.output}/${context.dependencyOutputs.get('right')?.output}`
      }
      await Bun.sleep(1)
      transitions.push(`end:${context.task.id}`)
      return context.task.id
    },
  })
  const parallelOutput = await parallel.run()
  requireCondition(parallelOutput.rawOutput === 'left/right', 'Parallel Cortex join result is incorrect')
  log('Parallel topology keeps dependency barriers while using concurrent runner ports')

  const hierarchy = new Cortex({
    process: CortexCore.ProcessType.HIERARCHICAL,
    agents: [
      { id: 'researcher', role: 'Researcher', execute: () => 'facts' },
      {
        id: 'writer',
        role: 'Writer',
        execute: context => context.context.includes('Manager review feedback:') ? 'revised report' : 'first report',
      },
    ],
    tasks: [
      { id: 'research', description: 'Collect facts', expectedOutput: 'Facts', agentId: 'researcher' },
      { id: 'draft', description: 'Write report', expectedOutput: 'Report', agentId: 'writer', dependencies: ['research'] },
    ],
    hierarchy: {
      plan: () => ({
        assignments: [
          { taskId: 'research', agentId: 'researcher' },
          { taskId: 'draft', agentId: 'missing-manager-agent', dependencies: ['research'] },
        ],
      }),
      review: request => request.task.id === 'draft' && request.attempt === 1
        ? { approved: false, feedback: 'Cite the collected facts.', improvementsNeeded: ['Add citations'] }
        : { approved: true },
      summarize: request => `manager summary: ${request.taskOutputs.map(output => output.output).join(' | ')}`,
    },
  })
  const hierarchyOutput = await hierarchy.run()
  requireCondition(hierarchyOutput.rawOutput === 'manager summary: facts | revised report', 'Hierarchical Cortex result is incorrect')
  requireCondition(
    hierarchyOutput.diagnostics.some(diagnostic => diagnostic.code === 'hierarchy_plan_fallback'),
    'Unsafe hierarchy assignment was not reported as a fallback diagnostic',
  )
  log('Hierarchical topology uses typed manager planning, review, and safe fallback assignment')

  const consensusCalls: string[] = []
  const consensus = new Cortex({
    process: CortexCore.ProcessType.CONSENSUS,
    agents: [
      { id: 'analyst', role: 'Analyst', execute: () => { consensusCalls.push('analyst'); return 'analysis' } },
      { id: 'critic', role: 'Critic', execute: () => { consensusCalls.push('critic'); return 'critique' } },
    ],
    tasks: [{ id: 'decide', description: 'Compare options', expectedOutput: 'Decision' }],
    consensus: {
      maxCandidatesParallel: 2,
      synthesizer: request => ({ output: request.candidates.map(candidate => candidate.output).join(' + ') }),
    },
  })
  const consensusOutput = await consensus.kickoff()
  requireCondition(consensusOutput.rawOutput === 'analysis + critique', 'Consensus synthesis result is incorrect')
  requireCondition(consensusCalls.sort().join(',') === 'analyst,critic', 'Consensus did not execute every candidate')
  log('Consensus topology executes each candidate and uses an explicit synthesis port')

  let plannedWriterContext = ''
  const planned = new Cortex({
    process: CortexCore.ProcessType.PLANNED,
    agents: [
      { id: 'researcher', role: 'Researcher', execute: () => 'evidence' },
      {
        id: 'writer',
        role: 'Writer',
        execute: context => {
          plannedWriterContext = context.context
          return 'published report'
        },
      },
    ],
    planner: new CortexPlanner(async () => `<plan>
      <objective>Research and publish</objective>
      <complexity>medium</complexity>
      <estimated_time>2</estimated_time>
      <step id="research">
        <agent>researcher</agent>
        <action>research</action>
        <arguments><topic>TypeScript</topic></arguments>
        <dependencies></dependencies>
        <description>Collect evidence</description>
      </step>
      <step id="write">
        <agent>writer</agent>
        <action>write</action>
        <arguments><source>result_from_step_research</source></arguments>
        <dependencies></dependencies>
        <description>Publish a report</description>
      </step>
    </plan>`),
    tasks: [
      { id: 'research', description: 'Collect evidence', expectedOutput: 'Evidence', agentId: 'researcher' },
      { id: 'write', description: 'Publish report', expectedOutput: 'Report', agentId: 'writer', dependencies: ['research'] },
    ],
  })
  const plannedOutput = await planned.run()
  requireCondition(plannedOutput.rawOutput === 'published report', 'Planned Cortex output is incorrect')
  requireCondition(plannedWriterContext.includes('Output from planned step research:\nevidence'), 'Planned dependency context is missing')
  log('Planned topology binds XML plan steps to declared task dependency barriers')

  const created = await new TaskCreator({
    generator: async () => `<task_plan>
      <objective>Ship a report</objective>
      <approach>Research then write</approach>
      <complexity>medium</complexity>
      <sequential>true</sequential>
      <task id="research"><description>Research</description><expected_output>Evidence</expected_output><agent_role>Researcher</agent_role></task>
      <task id="write"><description>Write</description><expected_output>Report</expected_output><agent_role>Writer</agent_role><dependencies>research</dependencies><context_needed>true</context_needed></task>
    </task_plan>`,
  }).create({
    objective: 'Ship a report',
    agents: [{ id: 'researcher', role: 'Researcher' }, { id: 'writer', role: 'Writer' }],
  })
  requireCondition(!created.usedFallback && created.tasks.length === 2, 'TaskCreator did not produce the declared typed tasks')
  requireCondition(new Set(created.tasks.map(task => task.agentId)).size === 2, 'TaskCreator did not map tasks to distinct agents')
  log('TaskCreator produces typed multi-agent tasks without compatibility JSON shims')
}

async function testSubAgentManager(log: ReportLog): Promise<void> {
  const runCalls: string[] = []
  const ids = ['swarm-task-1', 'swarm-task-2']
  const manager = new SubAgentManager({
    idFactory: () => ids.shift() ?? `swarm-${crypto.randomUUID()}`,
    maxConcurrent: 2,
    maxDepth: 2,
    pathResolver: path => `/workspace/${path}`,
    runner: async request => {
      runCalls.push(request.prompt)
      request.report.text(`running:${request.prompt}`)
      request.report.toolStart({
        toolCallId: `read-${runCalls.length}`,
        name: 'ReadFile',
        inputs: { file_path: 'src/swarm.ts' },
      })
      request.report.toolEnd({
        toolCallId: `read-${runCalls.length}`,
        name: 'ReadFile',
        permitted: true,
        result: 'source',
      })
      await Bun.sleep(2)
      return { content: `done:${request.prompt}` }
    },
  })

  try {
    const task = await manager.spawn({
      prompt: 'Review the native swarm implementation',
      name: 'code-review',
      systemPrompt: 'You are a reviewer.',
      agentDefinition: agentDefinition('reviewer', { tools: ['ReadFile'], allowedTools: ['ReadFile'] }),
    })
    requireCondition(await manager.sendMessage(task.id, 'Review the native tests too.'), 'Follow-up inbox message was rejected')
    const settled = await manager.wait(task.id, 1_000)
    requireCondition(settled?.status === 'completed', `Subagent did not complete: ${settled?.error ?? 'missing task'}`)
    requireCondition(runCalls.length === 2, 'Subagent runner did not process the follow-up message')
    requireCondition(task.readFiles.has('/workspace/src/swarm.ts'), 'Subagent file observation was not tracked')
    log('Spawn, follow-up inbox, native runner reporting, and wait lifecycle work')

    const isolated = await manager.spawn({ prompt: 'Need isolated worktree', isolation: 'worktree' })
    requireCondition(
      isolated.status === 'failed' && isolated.error.includes("isolation='worktree' requires a configured worktree port"),
      'Missing worktree port did not fail explicitly',
    )
    log('Worktree isolation remains an explicit host-owned port')
  } finally {
    await manager.close()
  }
}

async function testPolicyAndSandbox(log: ReportLog): Promise<void> {
  const policy = new ToolPolicy({ allow: ['calculator', 'echo'] })
  requireCondition(policy.evaluate('calculator') === PolicyAction.ALLOW, 'ToolPolicy did not allow calculator')
  requireCondition(policy.evaluate('dangerous_tool') === PolicyAction.DENY, 'ToolPolicy did not deny omitted tool')
  const engine = new PolicyEngine({ globalPolicy: policy })
  requireCondition(engine.check('ECHO') === PolicyAction.ALLOW, 'PolicyEngine did not preserve case-insensitive allow-list')
  log('ToolPolicy and PolicyEngine enforce the native allow-list')

  let hostCalled = false
  const router = new SandboxRouter({
    config: { mode: SandboxMode.STRICT, sandboxedTools: ['calculator'] },
    backend: {
      execute: async request => {
        const expression = request.arguments.expression
        return `sandbox:${typeof expression === 'string' ? expression : 'invalid'}`
      },
    },
  })
  requireCondition(router.decide('calculator').context === ExecutionContext.SANDBOX, 'Sandbox router did not select sandbox')
  const output = await router.execute(
    toolCall('calculator', { expression: '7 * 8' }),
    { metadata: {} },
    async () => {
      hostCalled = true
      return 'host result'
    },
  )
  requireCondition(output === 'sandbox:7 * 8' && !hostCalled, 'Strict sandbox route executed the host handler')
  log('SandboxRouter routes serializable tool requests through the injected sandbox backend')
}

async function testDefinitions(log: ReportLog): Promise<void> {
  const definitions = loadAgentDefinitions()
  for (const name of ['coder', 'researcher', 'planner', 'objective']) {
    requireCondition(definitions.has(name), `Built-in ${name} agent definition was not loaded`)
  }
  const researcher = definitions.get('researcher')
  const objective = definitions.get('objective')
  requireCondition(researcher?.allowedTools?.includes('ReadFile') === true, 'Researcher tool allow-list is missing ReadFile')
  requireCondition(objective?.allowedTools?.includes('apply_patch') === true, 'Objective tool allow-list is missing apply_patch')
  log('Bundled native agent definitions and their tool boundaries load correctly')
}

function agentDefinition(name: string, overrides: Partial<AgentDefinition> = {}): AgentDefinition {
  return {
    name,
    description: `${name} integration agent`,
    systemPrompt: `You are the ${name} integration agent.`,
    model: 'mock',
    tools: [],
    allowedTools: null,
    excludeTools: [],
    source: 'swarm-integration',
    maxDepth: 2,
    isolation: 'shared',
    ...overrides,
  }
}

function cortexAgent(id: string, role: string, llm: LlmClient): InstanceType<typeof CortexAgents.CortexAgent> {
  return new CortexAgents.CortexAgent({
    id,
    role,
    goal: `Complete ${role} tasks with explicit evidence.`,
    backstory: `Deterministic ${role} used by the Bun swarm integration exercise.`,
    llm,
    model: 'mock',
    maxIterations: 1,
  })
}

function knownExpression(value: unknown): number {
  if (value === '2 + 3') return 5
  if (value === '7 * 8') return 56
  throw new Error(`Unsupported deterministic expression: ${String(value)}`)
}

function toolCall(name: string, argumentsValue: ToolCall['function']['arguments'], id = `${name}-call`): ToolCall {
  return { id, type: 'function', function: { name, arguments: argumentsValue } }
}

function snapshotRequest(request: CompletionRequest): CompletionRequest {
  return {
    ...request,
    messages: request.messages.map(message => ({ ...message })),
  }
}

function messageContent(message: CompletionRequest['messages'][number] | undefined): string {
  if (message === undefined) return ''
  return typeof message.content === 'string' ? message.content : JSON.stringify(message.content)
}

function expectThrows(operation: () => void, message: string): void {
  try {
    operation()
  } catch {
    return
  }
  throw new Error(message)
}

function requireCondition(condition: unknown, message: string): asserts condition {
  if (!condition) throw new Error(message)
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

if (import.meta.main) {
  const report = await runSwarmIntegration()
  console.log(formatSwarmReport(report))
  if (!report.ok) process.exitCode = 1
}
