// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { interpolateInputs } from './stringUtils.js'

export type TemplateContext = Readonly<Record<string, unknown>>

export interface PromptTemplateOptions {
  /** Disable the native template language and use simple `{name}` interpolation instead. */
  readonly renderingEnabled?: boolean
}

export interface AgentPromptOptions {
  readonly backstory: string
  readonly goal: string
  readonly instructions?: string | null
  readonly role: string
  readonly rules?: readonly unknown[] | null
  readonly tools?: readonly unknown[] | null
}

export interface TaskPromptOptions {
  readonly constraints?: readonly unknown[] | null
  readonly context?: string | null
  readonly description: string
  readonly expectedOutput: string
}

export interface ManagerReviewPromptOptions {
  readonly agentRole: string
  readonly output: string
  readonly taskDescription: string
}

export interface PlannerPromptOptions {
  readonly agents: readonly unknown[]
  readonly context?: string
  readonly objective: string
}

export interface StepExecutionPromptOptions {
  readonly action: string
  readonly arguments?: Readonly<Record<string, unknown>>
  readonly context?: string
  readonly description: string
}

/** Raised when a template uses syntax outside the small, safe Cortex grammar. */
export class TemplateSyntaxError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'TemplateSyntaxError'
  }
}

/**
 * Dependency-free renderer for the Cortex prompt grammar.
 *
 * It supports the documented variable, `if`/`else`, and `for` forms without
 * evaluating JavaScript. Property lookup is restricted to data values, and
 * only the `items()` iterator helper is callable. That keeps custom prompt
 * rendering portable across Bun deployments and avoids a Jinja/Python bridge.
 */
export class NativeTemplate {
  private readonly nodes: readonly TemplateNode[]

  constructor(readonly source: string) {
    if (typeof source !== 'string') throw new TypeError('template source must be a string')
    this.nodes = parseTemplate(source)
  }

  render(context: TemplateContext = {}): string {
    assertContext(context)
    return trimTrailingNewline(renderNodes(this.nodes, createScope(context)))
  }

  /** Return identifiers callers must provide to render this template. */
  variables(): Set<string> {
    const variables = new Set<string>()
    collectVariables(this.nodes, new Set(), variables)
    return variables
  }
}

/** Prompt text used to introduce one Cortex agent. */
export const AGENT_PROMPT_TEMPLATE = `
You are {{ role }}.
Goal: {{ goal }}
Backstory: {{ backstory }}

{% if instructions %}
Instructions:
{{ instructions }}
{% endif %}

{% if rules %}
Rules:
{% for rule in rules %}
- {{ rule }}
{% endfor %}
{% endif %}

{% if tools %}
Available Tools:
{% for tool in tools %}
- {{ tool.name }}: {{ tool.description }}
{% endfor %}
{% endif %}

You must work towards achieving your goal while following your role's responsibilities.
When using tools, always provide clear and detailed responses.
`

/** Prompt text used to execute one Cortex task. */
export const TASK_PROMPT_TEMPLATE = `
{% if context %}
Context from previous tasks:
{{ context }}

{% endif %}
Task: {{ description }}

Expected Output: {{ expected_output }}

{% if constraints %}
Constraints:
{% for constraint in constraints %}
- {{ constraint }}
{% endfor %}
{% endif %}

Please complete this task according to your role and capabilities.
`

/** Prompt text used by a hierarchical manager to plan task delegation. */
export const MANAGER_DELEGATION_PROMPT_TEMPLATE = `
You are managing a team with the following agents:
{% for agent in agents %}
- {{ agent.role }}: {{ agent.goal }}
{% endfor %}

Tasks to complete:
{% for task in tasks %}
{{ loop.index }}. {{ task.description }}
   Expected: {{ task.expected_output }}
{% endfor %}

Create an execution plan that:
1. Assigns each task to the most appropriate agent based on their expertise
2. Defines the order of execution considering dependencies
3. Identifies potential bottlenecks or challenges
4. Suggests optimizations for efficiency

Return your plan in the following JSON format:
{
  "execution_plan": [
    {
      "task_id": 1,
      "assigned_to": "agent_role",
      "reason": "why this agent is best suited",
      "dependencies": [],
      "estimated_complexity": "low|medium|high"
    }
  ],
  "optimizations": ["suggestion1", "suggestion2"],
  "risks": ["risk1", "risk2"]
}
`

/** Prompt text used by a hierarchical manager to review one task result. */
export const MANAGER_REVIEW_PROMPT_TEMPLATE = `
Review the following output from {{ agent_role }}:

Task: {{ task_description }}
Output: {{ output }}

Evaluate the output based on:
1. Completeness - Does it fully address the task?
2. Quality - Is the work of high standard?
3. Accuracy - Are there any errors or inconsistencies?
4. Alignment - Does it meet the expected output requirements?

Provide your assessment in the following format:
{
  "approved": true/false,
  "score": 0-100,
  "feedback": "detailed feedback",
  "improvements_needed": ["improvement1", "improvement2"],
  "strengths": ["strength1", "strength2"]
}
`

/** Prompt text used to synthesize multiple task results. */
export const CONSENSUS_PROMPT_TEMPLATE = `
Multiple agents have provided their perspectives on the following task:
{{ task_description }}

Agent Outputs:
{% for agent_role, output in agent_outputs.items() %}
{{ agent_role }}:
{{ output }}

{% endfor %}

Synthesize these outputs into a unified response that:
1. Incorporates the best insights from all agents
2. Resolves any contradictions or conflicts
3. Provides a comprehensive and balanced perspective
4. Maintains coherence and clarity

Create a consensus response that represents the collective intelligence of the team.
`

/** Prompt text used by the planned workflow strategy. */
export const PLANNER_PROMPT_TEMPLATE = `
You are a strategic planner. Create a detailed execution plan for the following objective.

OBJECTIVE: {{ objective }}

AVAILABLE AGENTS:
{% for agent in agents %}
- {{ agent.role }}: {{ agent.goal }}{% if agent.tools %} (Tools: {% for tool in agent.tools %}{{ tool.__class__.__name__ }}{% if not loop.last %}, {% endif %}{% endfor %}){% endif %}
{% endfor %}

{% if context %}
CONTEXT: {{ context }}
{% else %}
CONTEXT: No additional context provided
{% endif %}

Create a plan using the following XML format:

<plan>
    <objective>{{ objective }}</objective>
    <complexity>low|medium|high</complexity>
    <estimated_time>minutes</estimated_time>

    <step id="1">
        <agent>Agent Role Name</agent>
        <action>specific_action_to_take</action>
        <arguments>
            <key1>value1</key1>
            <key2>value2</key2>
        </arguments>
        <dependencies></dependencies>
        <description>Clear description of what this step accomplishes</description>
    </step>

    <step id="2">
        <agent>Another Agent Role Name</agent>
        <action>another_action</action>
        <arguments>
            <input>result_from_step_1</input>
        </arguments>
        <dependencies>1</dependencies>
        <description>This step depends on step 1 completion</description>
    </step>
</plan>

INSTRUCTIONS:
1. Break down the objective into logical, sequential steps
2. Assign each step to the most appropriate agent based on their role and capabilities
3. Specify clear dependencies between steps (use step IDs)
4. Include all necessary arguments for each action
5. Make sure the plan is executable and complete
6. Use specific action names like: research, write, analyze, review, create, etc.

Respond ONLY with the XML plan, no additional text.
`

/** Prompt text used to execute a step parsed from a planned workflow. */
export const STEP_EXECUTION_PROMPT_TEMPLATE = `
You are executing a planned step in a larger workflow.

STEP DETAILS:
- Action: {{ action }}
- Description: {{ description }}

{% if arguments %}
ARGUMENTS:
{% for key, value in arguments.items() %}
- {{ key }}: {{ value }}
{% endfor %}
{% endif %}

{% if context %}
CONTEXT FROM PREVIOUS STEPS:
{{ context }}
{% endif %}

Execute this step thoroughly and provide a clear result that can be used by subsequent steps in the workflow.
`

/**
 * Renders the built-in Cortex prompts and safe custom templates.
 *
 * The disable switch mirrors the source fallback behavior but is explicit in
 * the TypeScript constructor. No external template engine or subprocess is
 * created at runtime.
 */
export class PromptTemplate {
  renderingEnabled: boolean

  constructor(options: PromptTemplateOptions = {}) {
    this.renderingEnabled = options.renderingEnabled ?? true
  }

  render(template: string, context: TemplateContext = {}): string {
    if (!this.renderingEnabled) return interpolateInputs(template, context)
    try {
      return new NativeTemplate(template).render(context)
    } catch {
      return interpolateInputs(template, context)
    }
  }

  renderAgentPrompt(options: AgentPromptOptions): string {
    return this.render(AGENT_PROMPT_TEMPLATE, {
      role: options.role,
      goal: options.goal,
      backstory: options.backstory,
      instructions: options.instructions,
      rules: options.rules,
      tools: options.tools,
    })
  }

  renderTaskPrompt(options: TaskPromptOptions): string {
    return this.render(TASK_PROMPT_TEMPLATE, {
      description: options.description,
      expected_output: options.expectedOutput,
      context: options.context,
      constraints: options.constraints,
    })
  }

  renderManagerDelegation(agents: readonly unknown[], tasks: readonly unknown[]): string {
    return this.render(MANAGER_DELEGATION_PROMPT_TEMPLATE, {
      agents,
      tasks: tasks.map(task => templateTaskRecord(task)),
    })
  }

  renderManagerReview(options: ManagerReviewPromptOptions): string {
    return this.render(MANAGER_REVIEW_PROMPT_TEMPLATE, {
      agent_role: options.agentRole,
      task_description: options.taskDescription,
      output: options.output,
    })
  }

  renderConsensus(taskDescription: string, agentOutputs: Readonly<Record<string, string>>): string {
    return this.render(CONSENSUS_PROMPT_TEMPLATE, {
      task_description: taskDescription,
      agent_outputs: agentOutputs,
    })
  }

  renderPlanner(options: PlannerPromptOptions): string {
    return this.render(PLANNER_PROMPT_TEMPLATE, {
      objective: options.objective,
      agents: options.agents,
      context: options.context ?? '',
    })
  }

  renderStepExecution(options: StepExecutionPromptOptions): string {
    return this.render(STEP_EXECUTION_PROMPT_TEMPLATE, {
      action: options.action,
      description: options.description,
      arguments: options.arguments ?? {},
      context: options.context ?? '',
    })
  }

  /** Compile an application-owned template when native rendering is enabled. */
  createCustomTemplate(template: string): NativeTemplate | undefined {
    return this.renderingEnabled ? new NativeTemplate(template) : undefined
  }

  /** Return undeclared variables used by a custom template. */
  getTemplateVariables(template: string): Set<string> {
    return this.renderingEnabled ? new NativeTemplate(template).variables() : new Set()
  }
}

type TemplateNode = TextNode | VariableNode | IfNode | ForNode

interface TextNode {
  readonly kind: 'text'
  readonly value: string
}

interface VariableNode {
  readonly expression: string
  readonly kind: 'variable'
}

interface IfNode {
  readonly alternate: readonly TemplateNode[]
  readonly condition: string
  readonly consequent: readonly TemplateNode[]
  readonly kind: 'if'
}

interface ForNode {
  readonly iterable: string
  readonly kind: 'for'
  readonly targets: readonly string[]
  readonly body: readonly TemplateNode[]
}

interface TemplateToken {
  kind: 'tag' | 'text' | 'variable'
  value: string
}

interface ParseResult {
  readonly index: number
  readonly nodes: readonly TemplateNode[]
  readonly terminator?: 'else' | 'endif' | 'endfor'
}

interface TemplateScope {
  readonly parent?: TemplateScope
  readonly values: Readonly<Record<string, unknown>>
}

interface TemplateClassMarker {
  readonly kind: 'template-class'
  readonly name: string
}

const TOKEN_PATTERN = /(\{\{[\s\S]*?\}\}|\{%[\s\S]*?%\})/g
const IDENTIFIER = /^[a-zA-Z_][a-zA-Z0-9_]*$/
const PATH_SEGMENT = /^([a-zA-Z_][a-zA-Z0-9_]*)(\(\))?$/
const FOR_TAG = /^for\s+(.+?)\s+in\s+(.+)$/

function parseTemplate(source: string): readonly TemplateNode[] {
  const result = parseNodes(tokenize(source), 0, new Set())
  if (result.terminator !== undefined) throw new TemplateSyntaxError(`Unexpected ${result.terminator}`)
  return result.nodes
}

function tokenize(source: string): TemplateToken[] {
  const tokens: TemplateToken[] = []
  for (const part of source.split(TOKEN_PATTERN)) {
    if (!part) continue
    if (part.startsWith('{{') && part.endsWith('}}')) {
      tokens.push({ kind: 'variable', value: part.slice(2, -2).trim() })
      continue
    }
    if (part.startsWith('{%') && part.endsWith('%}')) {
      tokens.push({ kind: 'tag', value: part.slice(2, -2).trim() })
      continue
    }
    tokens.push({ kind: 'text', value: part })
  }
  trimBlockWhitespace(tokens)
  return tokens
}

/** Match Jinja's documented trim_blocks/lstrip_blocks behavior for prompt text. */
function trimBlockWhitespace(tokens: TemplateToken[]): void {
  for (let index = 0; index < tokens.length; index += 1) {
    if (tokens[index]?.kind !== 'tag') continue
    const previous = tokens[index - 1]
    if (previous?.kind === 'text') {
      previous.value = previous.value.replace(/(^|\r?\n)[ \t]+$/, '$1')
    }
    const next = tokens[index + 1]
    if (next?.kind === 'text') {
      next.value = next.value.replace(/^\r?\n/, '')
    }
  }
}

function parseNodes(tokens: readonly TemplateToken[], start: number, terminators: ReadonlySet<string>): ParseResult {
  const nodes: TemplateNode[] = []
  let index = start
  while (index < tokens.length) {
    const token = tokens[index]
    if (token === undefined) break
    if (token.kind === 'text') {
      nodes.push({ kind: 'text', value: token.value })
      index += 1
      continue
    }
    if (token.kind === 'variable') {
      if (!token.value) throw new TemplateSyntaxError('Template variables cannot be empty')
      nodes.push({ kind: 'variable', expression: token.value })
      index += 1
      continue
    }

    const tag = token.value
    if (terminators.has(tag)) {
      const terminator = templateTerminator(tag)
      if (terminator !== undefined) return { nodes, index, terminator }
    }
    if (tag.startsWith('if ')) {
      const condition = tag.slice(3).trim()
      if (!condition) throw new TemplateSyntaxError('if requires a condition')
      const consequent = parseNodes(tokens, index + 1, new Set(['else', 'endif']))
      if (consequent.terminator === undefined) throw new TemplateSyntaxError('if is missing endif')
      if (consequent.terminator === 'endif') {
        nodes.push({ kind: 'if', condition, consequent: consequent.nodes, alternate: [] })
        index = consequent.index + 1
        continue
      }
      const alternate = parseNodes(tokens, consequent.index + 1, new Set(['endif']))
      if (alternate.terminator !== 'endif') throw new TemplateSyntaxError('else is missing endif')
      nodes.push({ kind: 'if', condition, consequent: consequent.nodes, alternate: alternate.nodes })
      index = alternate.index + 1
      continue
    }
    if (tag.startsWith('for ')) {
      const match = FOR_TAG.exec(tag)
      if (!match) throw new TemplateSyntaxError(`Invalid for expression: ${tag}`)
      const targets = match[1]?.split(',').map(target => target.trim()).filter(Boolean) ?? []
      const iterable = match[2]?.trim() ?? ''
      if (!targets.length || !iterable || targets.some(target => !IDENTIFIER.test(target))) {
        throw new TemplateSyntaxError(`Invalid for expression: ${tag}`)
      }
      const body = parseNodes(tokens, index + 1, new Set(['endfor']))
      if (body.terminator !== 'endfor') throw new TemplateSyntaxError('for is missing endfor')
      nodes.push({ kind: 'for', targets, iterable, body: body.nodes })
      index = body.index + 1
      continue
    }
    if (tag === 'else' || tag === 'endif' || tag === 'endfor') {
      throw new TemplateSyntaxError(`Unexpected ${tag}`)
    }
    throw new TemplateSyntaxError(`Unsupported template tag: ${tag}`)
  }
  return { nodes, index }
}

function renderNodes(nodes: readonly TemplateNode[], scope: TemplateScope): string {
  let result = ''
  for (const node of nodes) {
    switch (node.kind) {
      case 'text':
        result += node.value
        break
      case 'variable':
        result += displayValue(resolveExpression(node.expression, scope))
        break
      case 'if':
        result += renderNodes(
          isTemplateTruthy(resolveExpression(node.condition, scope)) ? node.consequent : node.alternate,
          scope,
        )
        break
      case 'for':
        result += renderFor(node, scope)
        break
    }
  }
  return result
}

function renderFor(node: ForNode, scope: TemplateScope): string {
  const values = iterableValues(resolveExpression(node.iterable, scope))
  let result = ''
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index]
    const locals: Record<string, unknown> = Object.create(null) as Record<string, unknown>
    assignForTargets(node.targets, value, locals)
    locals.loop = {
      index: index + 1,
      index0: index,
      first: index === 0,
      last: index === values.length - 1,
      length: values.length,
    }
    result += renderNodes(node.body, { values: locals, parent: scope })
  }
  return result
}

function assignForTargets(targets: readonly string[], value: unknown, locals: Record<string, unknown>): void {
  if (targets.length === 1) {
    const target = targets[0]
    if (target !== undefined) locals[target] = value
    return
  }
  if (!Array.isArray(value) || value.length !== targets.length) {
    throw new TemplateSyntaxError('for loop value does not match the declared targets')
  }
  for (let index = 0; index < targets.length; index += 1) {
    const target = targets[index]
    if (target !== undefined) locals[target] = value[index]
  }
}

function resolveExpression(expression: string, scope: TemplateScope): unknown {
  const normalized = expression.trim()
  if (!normalized) throw new TemplateSyntaxError('Template expressions cannot be empty')

  const orExpressions = splitLogicalExpression(normalized, 'or')
  if (orExpressions.length > 1) return orExpressions.some(item => isTemplateTruthy(resolveExpression(item, scope)))
  const andExpressions = splitLogicalExpression(normalized, 'and')
  if (andExpressions.length > 1) return andExpressions.every(item => isTemplateTruthy(resolveExpression(item, scope)))
  if (normalized.startsWith('not ')) return !isTemplateTruthy(resolveExpression(normalized.slice(4), scope))
  if (normalized === 'true' || normalized === 'True') return true
  if (normalized === 'false' || normalized === 'False') return false
  if (normalized === 'none' || normalized === 'None' || normalized === 'null') return null
  if (/^-?(?:0|[1-9]\d*)(?:\.\d+)?$/.test(normalized)) return Number(normalized)
  return resolvePath(normalized, scope)
}

function splitLogicalExpression(expression: string, operator: 'and' | 'or'): string[] {
  const separator = new RegExp(`\\s+${operator}\\s+`)
  return expression.split(separator).map(part => part.trim()).filter(Boolean)
}

function resolvePath(expression: string, scope: TemplateScope): unknown {
  const segments = expression.split('.')
  const first = segments.shift()
  if (first === undefined || !IDENTIFIER.test(first)) {
    throw new TemplateSyntaxError(`Unsupported template expression: ${expression}`)
  }
  let value = lookup(scope, first)
  for (const segment of segments) {
    const match = PATH_SEGMENT.exec(segment)
    if (!match) throw new TemplateSyntaxError(`Unsupported template expression: ${expression}`)
    const name = match[1]
    if (name === undefined) throw new TemplateSyntaxError(`Unsupported template expression: ${expression}`)
    value = match[2] === '()' ? invokeBuiltin(value, name) : readProperty(value, name)
  }
  return value
}

function invokeBuiltin(value: unknown, name: string): unknown {
  if (name !== 'items') throw new TemplateSyntaxError(`Unsupported template helper: ${name}()`)
  if (value instanceof Map) return [...value.entries()]
  if (isDataRecord(value)) return Object.entries(value)
  return []
}

function readProperty(value: unknown, name: string): unknown {
  if (name === '__class__') return { kind: 'template-class', name: nativeClassName(value) } satisfies TemplateClassMarker
  if (isTemplateClassMarker(value)) return name === '__name__' ? value.name : undefined
  if (name === '__proto__' || name === 'prototype' || name === 'constructor') return undefined
  if (Array.isArray(value) && name === 'length') return value.length
  if (value instanceof Map) return value.get(name)
  if (!isDataRecord(value)) return undefined
  return Object.hasOwn(value, name) ? value[name] : undefined
}

function lookup(scope: TemplateScope, name: string): unknown {
  let current: TemplateScope | undefined = scope
  while (current !== undefined) {
    if (Object.hasOwn(current.values, name)) return current.values[name]
    current = current.parent
  }
  return undefined
}

function iterableValues(value: unknown): unknown[] {
  if (value === null || value === undefined) return []
  if (Array.isArray(value)) return [...value]
  if (typeof value === 'string') return [...value]
  if (value instanceof Map) return [...value.keys()]
  if (value instanceof Set) return [...value]
  if (isDataRecord(value)) return Object.keys(value)
  if (isIterable(value)) return [...value]
  return []
}

function displayValue(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'boolean') return value ? 'True' : 'False'
  if (typeof value === 'string' || typeof value === 'number' || typeof value === 'bigint') return String(value)
  if (value instanceof Date) return value.toString()
  if (isTemplateClassMarker(value)) return value.name
  try {
    const serialized = JSON.stringify(value)
    return serialized === undefined ? String(value) : serialized
  } catch {
    return String(value)
  }
}

function isTemplateTruthy(value: unknown): boolean {
  if (value === null || value === undefined || value === false || value === '') return false
  if (typeof value === 'number') return value !== 0 && !Number.isNaN(value)
  if (Array.isArray(value) || typeof value === 'string') return value.length > 0
  if (value instanceof Map || value instanceof Set) return value.size > 0
  if (isDataRecord(value)) return Object.keys(value).length > 0
  return true
}

function collectVariables(nodes: readonly TemplateNode[], defined: ReadonlySet<string>, variables: Set<string>): void {
  for (const node of nodes) {
    switch (node.kind) {
      case 'text':
        break
      case 'variable':
        collectExpressionVariables(node.expression, defined, variables)
        break
      case 'if':
        collectExpressionVariables(node.condition, defined, variables)
        collectVariables(node.consequent, defined, variables)
        collectVariables(node.alternate, defined, variables)
        break
      case 'for': {
        collectExpressionVariables(node.iterable, defined, variables)
        const loopDefined = new Set([...defined, ...node.targets, 'loop'])
        collectVariables(node.body, loopDefined, variables)
        break
      }
    }
  }
}

function collectExpressionVariables(expression: string, defined: ReadonlySet<string>, variables: Set<string>): void {
  const withoutQuotedStrings = expression.replace(/(['"]).*?\1/g, '')
  for (const match of withoutQuotedStrings.matchAll(/[a-zA-Z_][a-zA-Z0-9_]*/g)) {
    const identifier = match[0]
    const offset = match.index ?? 0
    const previous = withoutQuotedStrings[offset - 1]
    if (previous === '.' || TEMPLATE_KEYWORDS.has(identifier) || defined.has(identifier)) continue
    variables.add(identifier)
  }
}

function createScope(context: TemplateContext): TemplateScope {
  return { values: context }
}

/** Map the canonical TypeScript Cortex task casing into this prompt's data shape once. */
function templateTaskRecord(task: unknown): unknown {
  if (!isDataRecord(task) || Object.hasOwn(task, 'expected_output') || !Object.hasOwn(task, 'expectedOutput')) {
    return task
  }
  return { ...task, expected_output: task.expectedOutput }
}

function assertContext(value: unknown): asserts value is TemplateContext {
  if (!isDataRecord(value)) throw new TypeError('template context must be a record')
}

function isDataRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isIterable(value: unknown): value is Iterable<unknown> {
  return typeof value === 'object' && value !== null && Symbol.iterator in value
}

function isTemplateClassMarker(value: unknown): value is TemplateClassMarker {
  return isDataRecord(value) && value.kind === 'template-class' && typeof value.name === 'string'
}

function nativeClassName(value: unknown): string {
  if (value === null) return 'NoneType'
  if (Array.isArray(value)) return 'Array'
  if (typeof value === 'object' && value !== null) return value.constructor?.name || 'Object'
  return typeof value
}

function trimTrailingNewline(value: string): string {
  return value.replace(/\r?\n$/, '')
}

function templateTerminator(value: string): ParseResult['terminator'] {
  if (value === 'else' || value === 'endif' || value === 'endfor') return value
  return undefined
}

const TEMPLATE_KEYWORDS = new Set(['and', 'false', 'False', 'none', 'None', 'not', 'null', 'or', 'true', 'True'])
