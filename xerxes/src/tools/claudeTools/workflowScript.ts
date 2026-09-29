// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Workflow scripts: a model-written JavaScript body that orchestrates many
 * subagents deterministically with `agent()`, `parallel()`, `pipeline()` and
 * `phase()`, instead of spawning them one tool call at a time.
 *
 * The script runs in its own Bun process with a scrubbed environment (no
 * provider keys) and talks to this runtime over NDJSON on stdio: each
 * `agent()` call is a request the host answers with the child's result. The
 * host owns everything with authority — spawning, models, concurrency,
 * budgets, cancellation — so a script can only ask for agents, never reach
 * the daemon's state. It still runs as the user, which is why the tool is
 * gated like `exec_command`.
 */

import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { validateToolArguments } from '../../runtime/argumentValidation.js'
import type { JsonSchema } from '../../types/toolCalls.js'

/** What a script asks for in `agent(prompt, opts)`. */
export interface WorkflowAgentRequest {
  readonly prompt: string
  readonly label?: string
  readonly phase?: string
  readonly model?: string
  readonly intelligence?: string
  readonly profile?: string
  readonly effort?: string
  readonly type?: string
  readonly isolation?: 'worktree'
  readonly schema?: JsonSchema
  readonly timeoutMs?: number
}

/** One finished agent as the host reports it back to the script runner. */
export interface WorkflowAgentOutcome {
  readonly id: string
  readonly status: 'completed' | 'failed' | 'cancelled'
  readonly output: string
  readonly error?: string
  readonly tokens: number
  /** Published-price cost of this agent's usage; undefined when no price is published. */
  readonly costUsd?: number
}

/**
 * The host side of `agent()`: spawn one child, wait for it, and on a schema
 * miss ask the same child to correct its answer. Owned by the agent tools.
 */
export interface WorkflowAgentPort {
  run(request: WorkflowAgentRequest, signal: AbortSignal): Promise<WorkflowAgentOutcome>
  correct(id: string, message: string, request: WorkflowAgentRequest, signal: AbortSignal): Promise<WorkflowAgentOutcome>
  /** Stop agents still running when the script ends or is cancelled. */
  closeAll(): void
}

export interface WorkflowRunOptions {
  readonly script: string
  readonly name: string
  readonly args?: unknown
  /** Agents running at once. The total a script may start is not capped by this. */
  readonly concurrency: number
  /** Runaway backstop on agents started over the whole run. */
  readonly maxAgents: number
  /** Stop starting agents once children have spent this many tokens. */
  readonly tokenBudget?: number
  readonly timeoutMs?: number
  readonly port: WorkflowAgentPort
  readonly signal?: AbortSignal
  /** Bun executable; the daemon's own by default. */
  readonly executable?: string
}

export interface WorkflowRunResult {
  readonly name: string
  readonly status: 'completed' | 'failed' | 'cancelled'
  readonly result?: unknown
  readonly error?: string
  readonly agents: { readonly started: number; readonly completed: number; readonly failed: number; readonly cancelled: number }
  readonly tokens: number
  /** Summed published-price cost, and how many agents had no published price. */
  readonly cost: { readonly usd: number; readonly unpriced: number }
  readonly durationMs: number
  readonly phases: readonly string[]
  readonly logs: readonly string[]
  readonly failures: readonly { readonly label: string; readonly error: string }[]
}

export const DEFAULT_WORKFLOW_CONCURRENCY = 16
export const MAX_WORKFLOW_CONCURRENCY = 64
export const DEFAULT_WORKFLOW_MAX_AGENTS = 10_000
const MAX_LOG_LINES = 400
const MAX_FAILURES_REPORTED = 40
const MAX_RESULT_CHARS = 200_000

/**
 * Runs inside the script process. Kept as source text so the runtime ships as
 * one bundle: it is written next to the script at run time.
 */
const HARNESS = String.raw`
const out = line => process.stdout.write(JSON.stringify(line) + '\n')
const pending = new Map()
let nextId = 0
let spent = 0
let currentPhase
const phases = []
const format = values => values.map(v => typeof v === 'string' ? v : (() => { try { return JSON.stringify(v) } catch { return String(v) } })()).join(' ')
for (const level of ['log', 'info', 'warn', 'error', 'debug']) console[level] = (...values) => out({ t: 'log', message: format(values) })
const agent = (prompt, opts = {}) => {
  if (typeof prompt !== 'string' || !prompt.trim()) return Promise.reject(new Error('agent(prompt) needs a non-empty prompt string'))
  const id = ++nextId
  const request = { ...opts, prompt, phase: opts.phase ?? currentPhase }
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject })
    out({ t: 'agent', id, request })
  })
}
const parallel = async tasks => {
  if (!Array.isArray(tasks)) throw new Error('parallel() takes an array of functions or promises')
  return Promise.all(tasks.map(task => Promise.resolve().then(() => typeof task === 'function' ? task() : task).catch(error => { out({ t: 'log', message: 'parallel item failed: ' + (error && error.message || error) }); return null })))
}
const pipeline = async (items, ...stages) => {
  if (!Array.isArray(items)) throw new Error('pipeline() takes an array of items first')
  return Promise.all(items.map(async (item, index) => {
    let value = item
    for (const stage of stages) {
      try { value = await stage(value, item, index) }
      catch (error) { out({ t: 'log', message: 'pipeline item ' + index + ' dropped: ' + (error && error.message || error) }); return null }
      if (value === null || value === undefined) return value ?? null
    }
    return value
  }))
}
const phase = title => {
  currentPhase = String(title)
  phases.push(currentPhase)
  out({ t: 'phase', title: currentPhase })
}
const log = (...values) => out({ t: 'log', message: format(values) })
const budget = {
  total: null,
  spent: () => spent,
  remaining: () => budget.total === null ? Infinity : Math.max(0, budget.total - spent),
}
let buffer = ''
process.stdin.on('data', chunk => {
  buffer += chunk
  let newline
  while ((newline = buffer.indexOf('\n')) >= 0) {
    const line = buffer.slice(0, newline)
    buffer = buffer.slice(newline + 1)
    if (!line.trim()) continue
    const message = JSON.parse(line)
    if (message.t === 'start') {
      budget.total = message.budget ?? null
      run(message.script, message.args)
      continue
    }
    if (message.t === 'result') {
      spent = message.spent ?? spent
      const waiter = pending.get(message.id)
      if (!waiter) continue
      pending.delete(message.id)
      if (message.ok) waiter.resolve(message.value)
      else waiter.reject(new Error(message.error))
    }
  }
})
process.stdin.setEncoding('utf8')
async function run(script, args) {
  try {
    const AsyncFunction = (async () => {}).constructor
    let body = script.replace(/^\s*export\s+const\s+meta\s*=/m, 'const meta =')
    // Models sometimes hand over the whole script as a function; call it.
    if (/^\s*(async\s+)?(function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)/.test(body) && /[}\)]\s*;?\s*$/.test(body)) {
      body = 'return await (' + body.trim().replace(/;\s*$/, '') + ')()'
    }
    const value = await new AsyncFunction('agent', 'parallel', 'pipeline', 'phase', 'log', 'args', 'budget', body)(agent, parallel, pipeline, phase, log, args, budget)
    out({ t: 'done', value: value === undefined ? null : value })
  } catch (error) {
    out({ t: 'error', message: error && error.message ? error.message : String(error), stack: error && error.stack ? String(error.stack).split('\n').slice(0, 6).join('\n') : undefined })
  }
}
`

type ScriptMessage =
  | { readonly t: 'agent'; readonly id: number; readonly request: unknown }
  | { readonly t: 'phase'; readonly title: string }
  | { readonly t: 'log'; readonly message: string }
  | { readonly t: 'done'; readonly value: unknown }
  | { readonly t: 'error'; readonly message: string; readonly stack?: string }

/** Run one workflow script to completion, cancellation, or failure. */
export async function runWorkflowScript(options: WorkflowRunOptions): Promise<WorkflowRunResult> {
  const started = Date.now()
  const logs: string[] = []
  const phases: string[] = []
  const failures: { label: string; error: string }[] = []
  const counts = { started: 0, completed: 0, failed: 0, cancelled: 0 }
  let tokens = 0
  const cost = { usd: 0, unpriced: 0 }
  const charge = (outcome: WorkflowAgentOutcome) => {
    tokens += outcome.tokens
    if (outcome.costUsd === undefined) { if (outcome.tokens > 0) cost.unpriced += 1 }
    else cost.usd += outcome.costUsd
  }
  const controller = new AbortController()
  const cancel = () => controller.abort(options.signal?.reason ?? new Error('Workflow cancelled'))
  if (options.signal?.aborted) cancel()
  else options.signal?.addEventListener('abort', cancel, { once: true })
  const timer = options.timeoutMs ? setTimeout(() => controller.abort(new Error(`Workflow exceeded ${Math.round(options.timeoutMs! / 60_000)} minutes`)), options.timeoutMs) : undefined

  const directory = await mkdtemp(join(tmpdir(), 'xerxes-workflow-'))
  const harness = join(directory, 'harness.mjs')
  await writeFile(harness, HARNESS, 'utf8')
  const child = Bun.spawn([options.executable ?? process.execPath, harness], {
    cwd: directory,
    // No provider keys, tokens, or daemon sockets: the script only orchestrates.
    env: { PATH: process.env.PATH ?? '', HOME: directory, TMPDIR: directory, NO_COLOR: '1' },
    stdin: 'pipe',
    stdout: 'pipe',
    stderr: 'pipe',
  })
  const send = (message: unknown) => {
    try {
      child.stdin.write(`${JSON.stringify(message)}\n`)
      child.stdin.flush()
    } catch {
      // The script process is gone; its exit is reported below.
    }
  }
  const pushLog = (line: string) => {
    logs.push(line.length > 2_000 ? `${line.slice(0, 1_999)}…` : line)
    if (logs.length > MAX_LOG_LINES) logs.splice(0, logs.length - MAX_LOG_LINES)
  }

  // Concurrency gate: the script may ask for any number of agents; at most
  // `concurrency` run at once and the rest queue in request order.
  let running = 0
  const queue: Array<{ readonly grant: () => void; readonly deny: (reason: unknown) => void }> = []
  const acquire = () => new Promise<void>((resolve, reject) => {
    if (controller.signal.aborted) { reject(controller.signal.reason); return }
    if (running < options.concurrency) { running++; resolve(); return }
    queue.push({ grant: () => { running++; resolve() }, deny: reject })
  })
  const release = () => {
    running--
    queue.shift()?.grant()
  }

  const runAgent = async (id: number, raw: unknown) => {
    let request: WorkflowAgentRequest
    try {
      request = workflowAgentRequest(raw)
      if (counts.started >= options.maxAgents) throw new Error(`Workflow agent limit (${options.maxAgents}) reached; raise max_agents if this fan-out is intended`)
      if (options.tokenBudget !== undefined && tokens >= options.tokenBudget) throw new Error(`Workflow token budget (${options.tokenBudget}) spent`)
    } catch (error) {
      send({ t: 'result', id, ok: false, error: errorText(error), spent: tokens })
      return
    }
    counts.started++
    try {
      await acquire()
    } catch (error) {
      counts.cancelled++
      send({ t: 'result', id, ok: false, error: errorText(error), spent: tokens })
      return
    }
    try {
      let outcome = await options.port.run(request, controller.signal)
      charge(outcome)
      let value: unknown = outcome.output
      if (outcome.status === 'completed' && request.schema) {
        let parsed = structuredValue(outcome.output, request.schema)
        if (!parsed.ok) {
          const correction = await options.port.correct(outcome.id, schemaCorrection(parsed.error, request.schema), request, controller.signal)
          charge(correction)
          outcome = correction
          parsed = correction.status === 'completed' ? structuredValue(correction.output, request.schema) : parsed
        }
        if (!parsed.ok) throw new Error(`Agent result did not match the schema: ${parsed.error}`)
        value = parsed.value
      }
      if (outcome.status !== 'completed') throw new Error(outcome.error || `Agent ${outcome.status}`)
      counts.completed++
      send({ t: 'result', id, ok: true, value, spent: tokens })
    } catch (error) {
      if (controller.signal.aborted) counts.cancelled++
      else {
        counts.failed++
        if (failures.length < MAX_FAILURES_REPORTED) failures.push({ label: request.label ?? request.prompt.slice(0, 60), error: errorText(error) })
      }
      send({ t: 'result', id, ok: false, error: errorText(error), spent: tokens })
    } finally {
      release()
    }
  }

  const finished = new Promise<{ ok: true; value: unknown } | { ok: false; error: string }>(resolve => {
    void (async () => {
      const decoder = new TextDecoder()
      const reader = child.stdout.getReader()
      let buffer = ''
      try {
        for (;;) {
          const { done, value } = await reader.read()
          if (done) break
          buffer += decoder.decode(value, { stream: true })
          let newline: number
          while ((newline = buffer.indexOf('\n')) >= 0) {
            const line = buffer.slice(0, newline)
            buffer = buffer.slice(newline + 1)
            const message = scriptMessage(line)
            if (!message) { if (line.trim()) pushLog(line.trim()); continue }
            if (message.t === 'agent') void runAgent(message.id, message.request)
            else if (message.t === 'phase') { phases.push(message.title); pushLog(`phase: ${message.title}`) }
            else if (message.t === 'log') pushLog(message.message)
            else if (message.t === 'done') { resolve({ ok: true, value: message.value }); return }
            else { resolve({ ok: false, error: message.stack ? `${message.message}\n${message.stack}` : message.message }); return }
          }
        }
      } catch {
        // Stream torn down by a kill; the exit below explains it.
      }
      const code = await child.exited
      const stderr = (await new Response(child.stderr).text()).trim().slice(-2_000)
      resolve({ ok: false, error: `Workflow script exited (code ${code}) before returning${stderr ? `: ${stderr}` : ''}` })
    })()
  })
  send({ t: 'start', script: options.script, args: options.args ?? null, ...(options.tokenBudget === undefined ? {} : { budget: options.tokenBudget }) })

  const aborted = new Promise<'aborted'>(resolve => {
    if (controller.signal.aborted) resolve('aborted')
    else controller.signal.addEventListener('abort', () => resolve('aborted'), { once: true })
  })
  try {
    const outcome = await Promise.race([finished, aborted])
    const agents = { ...counts }
    const base = { name: options.name, agents, tokens, cost: { ...cost }, durationMs: Date.now() - started, phases, logs, failures }
    if (outcome === 'aborted') return { ...base, status: 'cancelled', error: errorText(controller.signal.reason) }
    if (!outcome.ok) return { ...base, status: 'failed', error: outcome.error }
    return { ...base, status: 'completed', result: boundedResult(outcome.value) }
  } finally {
    if (timer) clearTimeout(timer)
    options.signal?.removeEventListener('abort', cancel)
    if (!controller.signal.aborted) controller.abort(new Error('Workflow finished'))
    for (const waiting of queue.splice(0)) waiting.deny(controller.signal.reason)
    options.port.closeAll()
    child.kill('SIGKILL')
    await child.exited.catch(() => undefined)
    await rm(directory, { recursive: true, force: true })
  }
}

function scriptMessage(line: string): ScriptMessage | undefined {
  if (!line.startsWith('{')) return undefined
  let value: unknown
  try { value = JSON.parse(line) } catch { return undefined }
  if (!value || typeof value !== 'object') return undefined
  const record = value as Record<string, unknown>
  switch (record.t) {
    case 'agent': return typeof record.id === 'number' ? { t: 'agent', id: record.id, request: record.request } : undefined
    case 'phase': return { t: 'phase', title: String(record.title ?? '').slice(0, 200) }
    case 'log': return { t: 'log', message: String(record.message ?? '') }
    case 'done': return { t: 'done', value: record.value }
    case 'error': return { t: 'error', message: String(record.message ?? 'Workflow script failed'), ...(typeof record.stack === 'string' ? { stack: record.stack } : {}) }
    default: return undefined
  }
}

/** Validate what the script passed to `agent()`; it is untrusted input. */
export function workflowAgentRequest(raw: unknown): WorkflowAgentRequest {
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw new Error('agent() options must be an object')
  const value = raw as Record<string, unknown>
  const text = (key: string, limit = 200): string | undefined => {
    const field = value[key]
    if (field === undefined || field === null) return undefined
    if (typeof field !== 'string') throw new Error(`agent() option ${key} must be a string`)
    return field.trim() ? field.trim().slice(0, limit) : undefined
  }
  const prompt = typeof value.prompt === 'string' ? value.prompt.trim() : ''
  if (!prompt) throw new Error('agent(prompt) needs a non-empty prompt string')
  const isolation = text('isolation')
  if (isolation !== undefined && isolation !== 'worktree') throw new Error('agent() isolation must be "worktree"')
  const schema = value.schema
  if (schema !== undefined && schema !== null && (typeof schema !== 'object' || Array.isArray(schema))) throw new Error('agent() schema must be a JSON Schema object')
  const timeout = value.timeout_ms ?? value.timeoutMs
  if (timeout !== undefined && (typeof timeout !== 'number' || !Number.isFinite(timeout) || timeout <= 0)) throw new Error('agent() timeout_ms must be a positive number')
  const label = text('label', 120) ?? text('title', 120)
  const phase = text('phase')
  const model = text('model')
  const intelligence = text('intelligence')
  const profile = text('profile') ?? text('provider_profile')
  const effort = text('effort') ?? text('reasoning_effort')
  const type = text('type') ?? text('subagent_type')
  return {
    prompt,
    ...(label ? { label } : {}),
    ...(phase ? { phase } : {}),
    ...(model ? { model } : {}),
    ...(intelligence ? { intelligence } : {}),
    ...(profile ? { profile } : {}),
    ...(effort ? { effort } : {}),
    ...(type ? { type } : {}),
    ...(isolation ? { isolation: 'worktree' as const } : {}),
    ...(schema ? { schema: schema as JsonSchema } : {}),
    ...(typeof timeout === 'number' ? { timeoutMs: timeout } : {}),
  }
}

/** The instruction appended to an agent's prompt when the script wants structured data back. */
export function schemaInstruction(schema: JsonSchema): string {
  return `\n\nWhen you finish, reply with ONLY a JSON value matching this JSON Schema — no prose, no code fence:\n${JSON.stringify(schema)}`
}

function schemaCorrection(error: string, schema: JsonSchema): string {
  return `Your last reply could not be used: ${error}. Reply again with ONLY the JSON value matching this JSON Schema — no prose, no code fence:\n${JSON.stringify(schema)}`
}

/** Pull the JSON value out of an agent's final text and check it against the schema. */
export function structuredValue(output: string, schema: JsonSchema): { ok: true; value: unknown } | { ok: false; error: string } {
  const candidates: string[] = []
  const trimmed = output.trim()
  candidates.push(trimmed)
  const fenced = /```(?:json)?\s*\n([\s\S]*?)\n```/i.exec(trimmed)
  if (fenced?.[1]) candidates.push(fenced[1].trim())
  const object = trimmed.indexOf('{'), objectEnd = trimmed.lastIndexOf('}')
  if (object >= 0 && objectEnd > object) candidates.push(trimmed.slice(object, objectEnd + 1))
  const array = trimmed.indexOf('['), arrayEnd = trimmed.lastIndexOf(']')
  if (array >= 0 && arrayEnd > array) candidates.push(trimmed.slice(array, arrayEnd + 1))
  let parsed: unknown
  let found = false
  for (const candidate of candidates) {
    try { parsed = JSON.parse(candidate); found = true; break } catch { /* next candidate */ }
  }
  if (!found) return { ok: false, error: 'the reply held no JSON value' }
  const type = schema.type
  if (type === 'array') return Array.isArray(parsed) ? { ok: true, value: parsed } : { ok: false, error: 'expected a JSON array' }
  if (type !== undefined && type !== 'object') return { ok: true, value: parsed }
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return { ok: false, error: 'expected a JSON object' }
  const check = validateToolArguments('result', parsed, schema)
  return check.ok ? { ok: true, value: check.coerced ?? parsed } : { ok: false, error: check.error }
}

function boundedResult(value: unknown): unknown {
  let text: string
  try { text = JSON.stringify(value) ?? 'null' } catch { return String(value) }
  if (text.length <= MAX_RESULT_CHARS) return value
  return `${text.slice(0, MAX_RESULT_CHARS)}… [result truncated: ${text.length} characters; return a smaller summary from the script]`
}

function errorText(error: unknown): string {
  if (error instanceof Error) return error.message
  return typeof error === 'string' ? error : String(error)
}
