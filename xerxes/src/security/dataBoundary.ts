// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The local/cloud data boundary.
 *
 * With `data_boundary: off` (the default) every method here is a no-op: no
 * opinion on tool calls, subagent results pass through unchanged, tier
 * selection is never refused and outbound requests are not gated or logged.
 * Only an explicit mode turns enforcement on.
 *
 * When a mode is set, the boundary:
 * - keeps cloud-trust agents away from `private_paths` (reads, searches,
 *   shell commands that reference them);
 * - marks a local-trust agent "tainted" once it touches private data, and
 *   withholds that agent's result from a cloud-trust parent;
 * - refuses to move work from a local tier (or a tainted local parent) to a
 *   cloud-trust model;
 * - with `read_only_inputs`, refuses writes to existing files outside the
 *   output directory;
 * - gates outbound cloud model requests (`approve_outbound`) and records each
 *   decision in the disclosure log.
 *
 * Every refusal is observable: a `deny` disposition with a reason, or a typed
 * `DataBoundaryError`. Nothing here turns a refusal into a silent success.
 */

import { existsSync, readFileSync, realpathSync, statSync } from 'node:fs'
import { homedir } from 'node:os'
import { basename, dirname, isAbsolute, join, resolve, sep } from 'node:path'

import { ValidationError } from '../core/errors.js'
import type { PermissionDisposition } from '../streaming/permissions.js'
import type { JsonObject, ToolCall } from '../types/toolCalls.js'
import {
  DataBoundaryError,
  type DataBoundaryErrorCode,
  type DataBoundaryMode,
  type DataTrust,
  type OutboundApproval,
  type OutboundGate,
  type OutboundRequestSummary,
  type OutboundVerdict,
} from './boundaryTypes.js'
import type { DisclosureLog } from './disclosureLog.js'
import { analyzeShellCommand, isReadOnlyInvocation } from './shellAnalysis.js'

/** Project-owned policy file, relative to the project root. */
export const DATA_POLICY_FILE = '.xerxes/data-policy.yaml'
/** Taint key used when a tool call carries no agent id (the registry's default agent). */
export const DEFAULT_BOUNDARY_AGENT = 'default'

const MODES: readonly DataBoundaryMode[] = ['off', 'private', 'hybrid', 'cloud']
const APPROVALS: readonly OutboundApproval[] = ['off', 'first', 'every']
const BOUNDARY_KEYS = new Set(['approve_outbound', 'read_only_inputs', 'private_paths', 'output_dir'])
const PROJECT_POLICY_KEYS = new Set(['private_paths', 'read_only_inputs'])
const MAX_PRIVATE_PATHS = 256
const MAX_PATTERN_LENGTH = 1024
const MAX_POLICY_BYTES = 64 * 1024

export interface DataBoundaryConfig {
  readonly mode: DataBoundaryMode
  readonly approveOutbound: OutboundApproval
  readonly readOnlyInputs: boolean
  /** Globs relative to the project root (or absolute), merged from settings and the project policy. */
  readonly privatePaths: readonly string[]
  /** Project-relative directory where writes to existing files stay allowed under read_only_inputs. */
  readonly outputDir: string
}

export interface DataBoundaryProject {
  readonly root: string
  /** Reads one file; undefined when it does not exist. Defaults to the filesystem. */
  readonly readFile?: (path: string) => string | undefined
}

/**
 * Parse `runtime.data_boundary` and `runtime.boundary` (plus the project's
 * `.xerxes/data-policy.yaml` when a mode is on).
 *
 * `XERXES_DATA_BOUNDARY` wins over the file. The project policy can add
 * private paths and turn `read_only_inputs` on, never off: a cloned
 * repository must not be able to loosen the user's settings.
 */
export function parseDataBoundaryConfig(
  runtime: Record<string, unknown>,
  env: Record<string, string | undefined>,
  project?: DataBoundaryProject,
): DataBoundaryConfig {
  const envMode = env.XERXES_DATA_BOUNDARY?.trim().toLowerCase()
  const mode = envMode
    ? oneOf('XERXES_DATA_BOUNDARY', envMode, MODES)
    : runtime.data_boundary === undefined ? 'off' : oneOf('data_boundary', runtime.data_boundary, MODES)
  const raw = runtime.boundary
  if (raw !== undefined && !isRecord(raw)) {
    throw new ValidationError('boundary', 'must be an object with approve_outbound, read_only_inputs, private_paths or output_dir', raw)
  }
  const boundary = raw ?? {}
  for (const key of Object.keys(boundary)) {
    if (!BOUNDARY_KEYS.has(key)) throw new ValidationError(`boundary.${key}`, 'unknown setting', boundary[key])
  }
  const approveOutbound = boundary.approve_outbound === undefined
    ? mode === 'hybrid' ? 'first' : 'off'
    : oneOf('boundary.approve_outbound', boundary.approve_outbound, APPROVALS)
  let readOnlyInputs = boundary.read_only_inputs === undefined
    ? mode !== 'off'
    : booleanSetting('boundary.read_only_inputs', boundary.read_only_inputs)
  const privatePaths = patternList('boundary.private_paths', boundary.private_paths)
  const outputDir = boundary.output_dir === undefined ? 'out' : relativeDirectory('boundary.output_dir', boundary.output_dir)
  // With the boundary off the project file is not even read, so a broken
  // policy file cannot change today's behavior.
  if (mode !== 'off' && project) {
    const policy = readProjectPolicy(project)
    if (policy.readOnlyInputs === true) readOnlyInputs = true
    privatePaths.push(...policy.privatePaths)
  }
  const unique = [...new Set(privatePaths)]
  if (unique.length > MAX_PRIVATE_PATHS) throw new ValidationError('boundary.private_paths', `at most ${MAX_PRIVATE_PATHS} patterns`, unique.length)
  return Object.freeze({ mode, approveOutbound, readOnlyInputs, privatePaths: Object.freeze(unique), outputDir })
}

function readProjectPolicy(project: DataBoundaryProject): { readonly privatePaths: string[]; readonly readOnlyInputs?: boolean } {
  const path = join(project.root, DATA_POLICY_FILE)
  const text = (project.readFile ?? readPolicyFile)(path)
  if (text === undefined) return { privatePaths: [] }
  if (text.length > MAX_POLICY_BYTES) throw new ValidationError(DATA_POLICY_FILE, `must be at most ${MAX_POLICY_BYTES} bytes`, text.length)
  let parsed: unknown
  try {
    parsed = Bun.YAML.parse(text)
  } catch (error) {
    throw new ValidationError(DATA_POLICY_FILE, `is not valid YAML: ${error instanceof Error ? error.message : String(error)}`)
  }
  if (parsed === null || parsed === undefined) return { privatePaths: [] }
  if (!isRecord(parsed)) throw new ValidationError(DATA_POLICY_FILE, 'must be a mapping with private_paths and/or read_only_inputs', parsed)
  for (const key of Object.keys(parsed)) {
    if (!PROJECT_POLICY_KEYS.has(key)) throw new ValidationError(`${DATA_POLICY_FILE}: ${key}`, 'unknown setting', parsed[key])
  }
  const privatePaths = patternList(`${DATA_POLICY_FILE}: private_paths`, parsed.private_paths)
  return parsed.read_only_inputs === undefined
    ? { privatePaths }
    : { privatePaths, readOnlyInputs: booleanSetting(`${DATA_POLICY_FILE}: read_only_inputs`, parsed.read_only_inputs) }
}

function readPolicyFile(path: string): string | undefined {
  try {
    if (statSync(path).size > MAX_POLICY_BYTES) throw new ValidationError(DATA_POLICY_FILE, `must be at most ${MAX_POLICY_BYTES} bytes`)
    return readFileSync(path, 'utf8')
  } catch (error) {
    if (isNodeError(error) && error.code === 'ENOENT') return undefined
    throw error
  }
}

/**
 * Where a profile may send data. An explicit `data_trust` wins; otherwise a
 * loopback host or a unix socket is `local` and anything else (including an
 * empty base URL of a hosted provider) is `cloud`.
 */
export function deriveDataTrust(profile: { readonly base_url?: string; readonly data_trust?: DataTrust } | undefined): DataTrust {
  if (profile?.data_trust === 'local' || profile?.data_trust === 'cloud') return profile.data_trust
  const baseUrl = profile?.base_url?.trim() ?? ''
  if (!baseUrl) return 'cloud'
  if (baseUrl.startsWith('/') || /^unix:/i.test(baseUrl) || /^http\+unix:/i.test(baseUrl)) return 'local'
  let host: string
  try {
    host = new URL(baseUrl).hostname.toLowerCase()
  } catch {
    return 'cloud'
  }
  if (host.startsWith('[') && host.endsWith(']')) host = host.slice(1, -1)
  if (host === 'localhost' || host.endsWith('.localhost') || host === '::1') return 'local'
  const v4 = host.startsWith('::ffff:') ? host.slice('::ffff:'.length) : host
  return /^127\.\d{1,3}\.\d{1,3}\.\d{1,3}$/.test(v4) ? 'local' : 'cloud'
}

/** Records which agents have seen private data. Keys must match the agent ids passed to the boundary. */
export interface BoundaryTaintStore {
  mark(agentId: string, reason: string): void
  isTainted(agentId: string): boolean
}

export class MemoryTaintStore implements BoundaryTaintStore {
  private readonly marks = new Map<string, string>()

  mark(agentId: string, reason: string): void {
    if (!this.marks.has(agentId)) this.marks.set(agentId, reason)
  }

  isTainted(agentId: string): boolean {
    return this.marks.has(agentId)
  }

  reason(agentId: string): string | undefined {
    return this.marks.get(agentId)
  }
}

/** A boundary opinion on one tool call, with the reason to show the model and the user. */
export interface BoundaryToolVerdict {
  readonly disposition: Exclude<PermissionDisposition, 'allow'>
  readonly reason: string
  readonly code?: DataBoundaryErrorCode
}

export interface BoundaryToolContext {
  readonly agentId?: string
  readonly trust: DataTrust
}

export interface SubagentResultContext {
  readonly childAgentId: string
  readonly childTrust: DataTrust
  readonly parentTrust: DataTrust
  /** Parent taint key; a local parent receiving a tainted result becomes tainted itself. */
  readonly parentAgentId?: string
  /** Other keys the child may have been tainted under (for example its transcript id). */
  readonly childAliases?: readonly string[]
}

export interface TierSelectionContext {
  readonly requestedTrust: DataTrust
  /** Trust of the intelligence tier the request falls under, when one applies. */
  readonly tierTrust?: DataTrust
  readonly parentTrust: DataTrust
  /** Parent taint key: a tainted local parent may not hand work to a cloud model. */
  readonly parentAgentId?: string
}

export interface OutboundGatePorts {
  /** Ask the user whether this request may leave the machine. Never auto-approve here. */
  approve(summary: OutboundRequestSummary, signal?: AbortSignal): Promise<boolean>
  readonly log: DisclosureLog
}

/** The narrow port `PermissionGateOptions.boundary` expects. */
export interface BoundaryPermissionPort {
  toolDisposition(call: Pick<ToolCall, 'function'>, agentId?: string): PermissionDisposition | undefined
  /** The reason behind a non-allow verdict, shown to the model with a denial. */
  reason(call: Pick<ToolCall, 'function'>, agentId?: string): string | undefined
}

type Glob = InstanceType<typeof Bun.Glob>

interface CompiledPattern {
  readonly source: string
  readonly absolute: boolean
  readonly globs: readonly Glob[]
  /** Folded canonical absolute path of the pattern's static (glob-free) prefix. */
  readonly base: string
}

/** `stdin` is text the program reads; it is scanned like the command and is never read-only. */
type ShellInput =
  | { readonly kind: 'argv'; readonly command: string; readonly args: readonly string[]; readonly workdir?: string; readonly stdin?: string }
  | { readonly kind: 'shell'; readonly text: string; readonly workdir?: string; readonly stdin?: string }

interface ShellSegment {
  readonly executable: string | undefined
  readonly args: readonly string[]
}

const SHELL_TOOLS = new Set(['exec_command', 'ExecuteShell', 'Bash', 'shell', 'pty_open', 'pty_write'])
const WRITE_TOOLS = new Set(['Write', 'WriteFile', 'Edit', 'FileEditTool', 'MultiEdit', 'AppendFile', 'NotebookEdit'])
const CONTENT_SEARCH_TOOLS = new Set(['GrepTool', 'Grep'])
const NAME_GLOB_TOOLS = new Set(['GlobTool', 'Glob'])
const PATH_KEYS = ['file_path', 'path', 'directory_path', 'notebook_path'] as const
const SEARCH_EXECUTABLES = new Set(['rg', 'ag', 'ack', 'fd', 'find', 'tar', 'zip', 'rsync'])
const FOLD_CASE = process.platform === 'darwin' || process.platform === 'win32'

export class DataBoundary {
  readonly config: DataBoundaryConfig
  readonly root: string
  private readonly taints: BoundaryTaintStore
  private readonly rootKey: string
  private readonly patterns: readonly CompiledPattern[]
  private readonly outputKey: string

  constructor(config: DataBoundaryConfig, root: string, taints: BoundaryTaintStore = new MemoryTaintStore()) {
    this.config = config
    this.root = canonicalPath(resolve(root))
    this.taints = taints
    this.rootKey = pathKey(this.root)
    this.patterns = config.privatePaths.map(pattern => compilePattern(pattern, this.root))
    this.outputKey = pathKey(canonicalPath(resolve(this.root, config.outputDir)))
  }

  get mode(): DataBoundaryMode {
    return this.config.mode
  }

  get enabled(): boolean {
    return this.config.mode !== 'off'
  }

  isTainted(agentId: string | undefined): boolean {
    return this.taints.isTainted(agentId ?? DEFAULT_BOUNDARY_AGENT)
  }

  markTainted(agentId: string | undefined, reason: string): void {
    this.taints.mark(agentId ?? DEFAULT_BOUNDARY_AGENT, reason)
  }

  /** Whether a path (absolute, or relative to the project root) falls under `private_paths`. */
  isPrivatePath(path: string): boolean {
    if (!this.patterns.length || !path.trim()) return false
    const key = pathKey(canonicalPath(resolve(this.root, expandHome(path))))
    const relativeKey = insideKey(this.rootKey, key)
    for (const pattern of this.patterns) {
      const target = pattern.absolute ? key : relativeKey
      if (!target) continue
      if (ancestry(target).some(candidate => pattern.globs.some(glob => glob.match(candidate)))) return true
    }
    return false
  }

  /** Whether a directory scope contains (or lies inside) a private path pattern. */
  overlapsPrivate(scope: string): boolean {
    if (!this.patterns.length) return false
    const key = pathKey(canonicalPath(resolve(this.root, expandHome(scope))))
    return this.patterns.some(pattern => isAncestorOrEqual(key, pattern.base) || isAncestorOrEqual(pattern.base, key))
  }

  /** The boundary's opinion on one tool call, or undefined for "no opinion". */
  toolDisposition(call: Pick<ToolCall, 'function'>, ctx: BoundaryToolContext): PermissionDisposition | undefined {
    return this.toolVerdict(call, ctx)?.disposition
  }

  /** As {@link toolDisposition}, with the reason to report to the model and user. */
  toolVerdict(call: Pick<ToolCall, 'function'>, ctx: BoundaryToolContext): BoundaryToolVerdict | undefined {
    if (this.config.mode === 'off') return undefined
    const name = call.function.name
    const inputs = call.function.arguments
    const agent = ctx.agentId ?? DEFAULT_BOUNDARY_AGENT
    if (this.config.mode === 'private' && ctx.trust === 'cloud') {
      return deny('DATA_BOUNDARY_NO_LOCAL_ROUTE', `Data boundary is private: a cloud model may not run tools (${name}). Switch to a local provider profile.`)
    }
    const verdicts: BoundaryToolVerdict[] = []
    const shell = shellInput(name, inputs)
    if (shell) {
      const verdict = this.shellVerdict(name, shell, ctx.trust, agent)
      if (verdict) verdicts.push(verdict)
    } else {
      const touched = this.privateTargets(name, inputs)
      if (touched !== undefined) {
        if (ctx.trust === 'cloud') {
          verdicts.push(deny('DATA_BOUNDARY_DENIED', `Data boundary: ${name} would expose private data (${touched}) to a cloud model. Use a local agent for private files.`))
        } else {
          this.taints.mark(agent, `${name} touched ${touched}`)
        }
      }
    }
    if (this.config.readOnlyInputs && WRITE_TOOLS.has(name)) {
      const verdict = this.writeVerdict(name, inputs)
      if (verdict) verdicts.push(verdict)
    }
    return verdicts.find(verdict => verdict.disposition === 'deny') ?? verdicts[0]
  }

  /**
   * Gate a subagent result before it reaches its parent. A tainted child's
   * result never reaches a cloud-trust parent; a local parent that receives it
   * becomes tainted.
   */
  filterSubagentResult(text: string, ctx: SubagentResultContext): { readonly text: string; readonly withheld: boolean } {
    if (this.config.mode === 'off') return { text, withheld: false }
    const tainted = [ctx.childAgentId, ...(ctx.childAliases ?? [])].some(key => key && this.taints.isTainted(key))
    if (!tainted) return { text, withheld: false }
    if (ctx.parentTrust === 'cloud') {
      const bytes = Buffer.byteLength(text, 'utf8')
      return {
        text: `[Result withheld by the data boundary: subagent ${ctx.childAgentId} read private data and this conversation runs on a cloud model. ${bytes} bytes were not forwarded. Continue the follow-up on a local model, or ask the user to review the result.]`,
        withheld: true,
      }
    }
    if (ctx.parentAgentId) this.taints.mark(ctx.parentAgentId, `received a result from tainted subagent ${ctx.childAgentId}`)
    return { text, withheld: false }
  }

  /** Throw when a model/tier selection would move work across the boundary. */
  assertTierAllowed(ctx: TierSelectionContext): void {
    if (this.config.mode === 'off' || ctx.requestedTrust !== 'cloud') return
    if (this.config.mode === 'private') {
      throw new DataBoundaryError('DATA_BOUNDARY_NO_LOCAL_ROUTE', 'Data boundary is private: a cloud-trust model or profile cannot be selected. Choose a local profile.')
    }
    if (ctx.tierTrust === 'local') {
      throw new DataBoundaryError('DATA_BOUNDARY_TIER_LOCKED', 'Data boundary: this intelligence tier runs locally; an explicit cloud model or profile cannot override it. Omit model/provider_profile or pick a local one.')
    }
    if (ctx.parentTrust === 'local' && ctx.parentAgentId !== undefined && this.taints.isTainted(ctx.parentAgentId)) {
      throw new DataBoundaryError('DATA_BOUNDARY_TIER_LOCKED', 'Data boundary: this agent has read private data, so it cannot hand work to a cloud-trust model. Use a local model for the subagent.')
    }
  }

  /** An outbound gate implementing `approve_outbound`; every decision is logged. */
  outboundGate(ports: OutboundGatePorts): OutboundGate {
    const config = this.config
    const approved = new Set<string>()
    const pending = new Map<string, Promise<boolean>>()
    const sent: OutboundVerdict = { allow: true, decision: 'sent' }
    return {
      async check(request: OutboundRequestSummary, signal?: AbortSignal): Promise<OutboundVerdict> {
        if (config.mode === 'off') return sent
        if (request.trust === 'local') {
          ports.log.record(request, 'sent')
          return sent
        }
        if (config.mode === 'private') {
          const reason = `data boundary is private: refused to send ${request.purpose} request to cloud provider ${request.provider}`
          ports.log.record(request, 'denied', reason)
          return { allow: false, reason }
        }
        const key = `${request.sessionId ?? ''}\u0000${request.provider}`
        if (config.approveOutbound === 'off' || (config.approveOutbound === 'first' && approved.has(key))) {
          ports.log.record(request, 'sent')
          return sent
        }
        let ok: boolean
        try {
          if (config.approveOutbound === 'first') {
            let shared = pending.get(key)
            if (!shared) {
              shared = ports.approve(request, signal).finally(() => pending.delete(key))
              pending.set(key, shared)
            }
            ok = await shared
          } else {
            ok = await ports.approve(request, signal)
          }
        } catch (error) {
          ports.log.record(request, 'denied', 'approval was not completed')
          throw error
        }
        if (!ok) {
          const reason = `user declined sending ${request.bytes} bytes to ${request.provider}/${request.model}`
          ports.log.record(request, 'denied', reason)
          return { allow: false, reason }
        }
        if (config.approveOutbound === 'first') approved.add(key)
        ports.log.record(request, 'approved')
        return { allow: true, decision: 'approved' }
      },
    }
  }

  private privateTargets(name: string, inputs: JsonObject): string | undefined {
    for (const key of PATH_KEYS) {
      const value = inputs[key]
      if (typeof value === 'string' && value.trim() && this.isPrivatePath(value)) return value
    }
    if (CONTENT_SEARCH_TOOLS.has(name)) {
      const scope = typeof inputs.path === 'string' && inputs.path.trim() ? inputs.path : '.'
      if (this.overlapsPrivate(scope)) return `${scope} contains private paths`
    }
    if (NAME_GLOB_TOOLS.has(name) && typeof inputs.pattern === 'string') {
      const base = typeof inputs.path === 'string' && inputs.path.trim() ? inputs.path : '.'
      const prefix = literalPrefix(inputs.pattern)
      if (prefix && this.isPrivatePath(join(base, prefix))) return join(base, prefix)
    }
    return undefined
  }

  private shellVerdict(name: string, shell: ShellInput, trust: DataTrust, agent: string): BoundaryToolVerdict | undefined {
    const workdir = resolve(this.root, expandHome(shell.workdir ?? '.'))
    let readOnly: boolean
    let unresolved = false
    let segments: readonly ShellSegment[]
    let tokens: string[]
    if (shell.kind === 'argv') {
      readOnly = isReadOnlyInvocation(shell.command, shell.args)
      segments = [{ executable: basename(shell.command).toLowerCase(), args: shell.args }]
      tokens = [shell.command, ...shell.args].flatMap(shellTokens)
    } else {
      const analysis = analyzeShellCommand(shell.text)
      readOnly = analysis.verdict === 'safe'
      unresolved = analysis.verdict === 'unresolved'
      segments = analysis.segments.map(segment => ({ executable: segment.executable, args: segment.args }))
      tokens = shellTokens(shell.text)
    }
    if (shell.stdin !== undefined) {
      // A script on stdin is the same program text `python3 -c` would carry in args.
      readOnly = false
      tokens = [...tokens, ...shellTokens(shell.stdin)]
    }
    const referenced = this.shellPrivateReference(tokens, segments, workdir)
    if (referenced !== undefined) {
      if (trust === 'cloud') {
        return deny('DATA_BOUNDARY_DENIED', `Data boundary: ${name} references a private path (${referenced}) and this agent runs on a cloud model.`)
      }
      this.taints.mark(agent, `${name} referenced ${referenced}`)
      return prompt(`Data boundary: ${name} references a private path (${referenced}); confirm before it runs.`)
    }
    if (unresolved) {
      // An unresolved command may read anything; a local agent that runs it is
      // assumed to have seen private data.
      if (trust === 'local' && this.patterns.length) this.taints.mark(agent, `${name} ran a command the boundary cannot resolve`)
      return prompt(`Data boundary: ${name} uses shell constructs that cannot be checked statically (substitution, backticks, process substitution); confirm before it runs.`)
    }
    if (!readOnly) return prompt(`Data boundary: ${name} is not a read-only command; confirm before it runs.`)
    return undefined
  }

  private shellPrivateReference(tokens: readonly string[], segments: readonly ShellSegment[], workdir: string): string | undefined {
    if (!this.patterns.length) return undefined
    if (this.isPrivatePath(workdir)) return workdir
    for (const token of tokens) {
      if (!token || token.startsWith('-') || token.includes('://')) continue
      if (/[*?[{]/.test(token)) {
        const absolute = pathKey(resolve(workdir, expandHome(token)))
        const prefix = literalPrefix(token)
        if (prefix && this.isPrivatePath(resolve(workdir, expandHome(prefix)))) return token
        let glob: Glob
        try {
          glob = new Bun.Glob(absolute)
        } catch {
          continue
        }
        if (this.patterns.some(pattern => glob.match(pattern.base))) return token
      } else if (this.isPrivatePath(resolve(workdir, expandHome(token)))) {
        return token
      }
    }
    // Recursive readers with no explicit scope search the working directory.
    const searches = segments.length
      ? segments.filter(segment => isRecursiveSearch(segment))
      : tokens.some(token => SEARCH_EXECUTABLES.has(token.toLowerCase())) ? [{ executable: undefined, args: [] }] : []
    for (const segment of searches) {
      const scopes = segment.args
        .filter(argument => !argument.startsWith('-'))
        .map(argument => resolve(workdir, expandHome(argument)))
        .filter(path => isDirectory(path))
      for (const scope of scopes.length ? scopes : [workdir]) {
        if (this.overlapsPrivate(scope)) return `${scope} (recursive search over private paths)`
      }
    }
    return undefined
  }

  private writeVerdict(name: string, inputs: JsonObject): BoundaryToolVerdict | undefined {
    const raw = typeof inputs.file_path === 'string' ? inputs.file_path : typeof inputs.notebook_path === 'string' ? inputs.notebook_path : undefined
    if (!raw?.trim()) return undefined
    const absolute = resolve(this.root, expandHome(raw))
    if (this.isPrivatePath(absolute)) {
      return deny('DATA_BOUNDARY_READ_ONLY', `Data boundary: ${name} may not modify private input ${raw}.`)
    }
    if (!existsSync(absolute)) return undefined
    if (isAncestorOrEqual(this.outputKey, pathKey(canonicalPath(absolute)))) return undefined
    return deny('DATA_BOUNDARY_READ_ONLY', `Data boundary: inputs are read-only; ${name} may not modify existing file ${raw}. Write new files, or write into ${this.config.outputDir}/.`)
  }
}

/** Adapt a boundary to the permission gate, given how to find an agent's trust. */
export function boundaryPermissionPort(boundary: DataBoundary, trustOf: (agentId?: string) => DataTrust): BoundaryPermissionPort {
  const context = (agentId?: string): BoundaryToolContext => ({ ...(agentId === undefined ? {} : { agentId }), trust: trustOf(agentId) })
  return {
    toolDisposition: (call, agentId) => boundary.toolDisposition(call, context(agentId)),
    reason: (call, agentId) => boundary.toolVerdict(call, context(agentId))?.reason,
  }
}

function deny(code: DataBoundaryErrorCode, reason: string): BoundaryToolVerdict {
  return { disposition: 'deny', reason, code }
}

function prompt(reason: string): BoundaryToolVerdict {
  return { disposition: 'prompt', reason }
}

function shellInput(name: string, inputs: JsonObject): ShellInput | undefined {
  if (!SHELL_TOOLS.has(name)) return undefined
  const workdir = stringValue(inputs.workdir) ?? stringValue(inputs.cwd)
  const where = workdir === undefined ? {} : { workdir }
  if (name === 'exec_command') {
    const command = (stringValue(inputs.cmd) ?? stringValue(inputs.command))?.trim()
    if (!command) return undefined
    const stdinText = typeof inputs.stdin === 'string' ? inputs.stdin : undefined
    const extra = { ...where, ...(stdinText === undefined ? {} : { stdin: stdinText }) }
    const args = inputs.args
    if (Array.isArray(args) && args.every(value => typeof value === 'string')) {
      return { kind: 'argv', command, args: args as string[], ...extra }
    }
    if (args !== undefined || /\s|[;&|`$()<>]/.test(command)) return { kind: 'shell', text: command, ...extra }
    return { kind: 'argv', command, args: [], ...extra }
  }
  const text = name === 'pty_write' ? stringValue(inputs.chars) : stringValue(inputs.command) ?? stringValue(inputs.cmd)
  return text?.trim() ? { kind: 'shell', text, ...where } : undefined
}

function isRecursiveSearch(segment: ShellSegment): boolean {
  const executable = segment.executable?.toLowerCase()
  if (!executable) return false
  if (SEARCH_EXECUTABLES.has(executable)) return true
  const args = segment.args
  if (executable === 'grep' || executable === 'egrep' || executable === 'fgrep') {
    return args.some(argument => argument === '--recursive' || argument.startsWith('--dereference-recursive') || /^-[A-Za-z]*[rR]/.test(argument) || argument === '-d' || argument.startsWith('--directories=recurse'))
  }
  if (executable === 'git') return args[0] === 'grep' || args[0] === 'archive'
  if (executable === 'cp') return args.some(argument => /^-[A-Za-z]*[rRa]/.test(argument) || argument === '--recursive' || argument === '--archive')
  return false
}

function shellTokens(text: string): string[] {
  return text.split(/[\s;&|<>()`'"=,${}]+/).filter(Boolean)
}

/** The glob-free directory prefix of a pattern ('secret/cu*' -> 'secret'). */
function literalPrefix(pattern: string): string {
  const index = pattern.search(/[*?[{]/)
  if (index < 0) return pattern
  const head = pattern.slice(0, index)
  if (!head) return ''
  return head.endsWith('/') ? head.slice(0, -1) : dirname(head) === '.' ? '' : dirname(head)
}

function compilePattern(pattern: string, root: string): CompiledPattern {
  const trimmed = pattern.trim().replace(/^\.\//, '').replace(/\/+$/, '')
  const absolute = isAbsolute(trimmed) || trimmed.startsWith('~/')
  const segments = toPosix(absolute ? expandHome(trimmed) : trimmed).split('/')
  const staticCount = segments.findIndex(segment => /[*?[{]/.test(segment))
  const prefixSegments = staticCount < 0 ? segments : segments.slice(0, staticCount)
  const rest = staticCount < 0 ? [] : segments.slice(staticCount)
  let text: string
  let base: string
  if (absolute) {
    const prefix = canonicalPath(prefixSegments.join('/') || '/')
    base = pathKey(prefix)
    text = [base === '/' ? '' : base, ...rest].join('/') || '/'
  } else {
    base = pathKey(resolve(root, prefixSegments.join('/') || '.'))
    text = fold(segments.join('/'))
  }
  const sources = [text]
  if (text.endsWith('/**')) sources.push(text.slice(0, -3) || '/')
  return { source: pattern, absolute, globs: sources.map(source => new Bun.Glob(source)), base }
}

/** Resolve symlinks of the longest existing prefix; missing tails are appended unchanged. */
function canonicalPath(path: string): string {
  const missing: string[] = []
  let current = resolve(path)
  for (;;) {
    try {
      return join(realpathSync.native(current), ...[...missing].reverse())
    } catch {
      const parent = dirname(current)
      if (parent === current) return resolve(path)
      missing.push(basename(current))
      current = parent
    }
  }
}

function pathKey(path: string): string {
  return fold(toPosix(path))
}

function fold(value: string): string {
  return FOLD_CASE ? value.toLowerCase() : value
}

function toPosix(value: string): string {
  return sep === '/' ? value : value.split(sep).join('/')
}

function expandHome(path: string): string {
  if (path === '~') return homedir()
  return path.startsWith('~/') ? join(homedir(), path.slice(2)) : path
}

/** `key` relative to `root` when inside it ('' for the root itself), else undefined. */
function insideKey(root: string, key: string): string | undefined {
  if (key === root) return ''
  const prefix = root.endsWith('/') ? root : `${root}/`
  return key.startsWith(prefix) ? key.slice(prefix.length) : undefined
}

function isAncestorOrEqual(ancestor: string, path: string): boolean {
  return insideKey(ancestor, path) !== undefined
}

/** A path and each of its ancestors, deepest first; the filesystem root is excluded. */
function ancestry(path: string): string[] {
  const result: string[] = []
  let current = path
  while (current && current !== '/' && current !== '.') {
    result.push(current)
    const index = current.lastIndexOf('/')
    if (index <= 0) break
    current = current.slice(0, index)
  }
  return result
}

function isDirectory(path: string): boolean {
  try {
    return statSync(path).isDirectory()
  } catch {
    return false
  }
}

function oneOf<Value extends string>(key: string, value: unknown, allowed: readonly Value[]): Value {
  const match = allowed.find(candidate => candidate === value)
  if (match === undefined) throw new ValidationError(key, `must be one of ${allowed.join(', ')}`, value)
  return match
}

function booleanSetting(key: string, value: unknown): boolean {
  if (typeof value !== 'boolean') throw new ValidationError(key, 'must be true or false', value)
  return value
}

function patternList(key: string, value: unknown): string[] {
  if (value === undefined || value === null) return []
  if (!Array.isArray(value)) throw new ValidationError(key, 'must be a list of glob patterns', value)
  if (value.length > MAX_PRIVATE_PATHS) throw new ValidationError(key, `at most ${MAX_PRIVATE_PATHS} patterns`, value.length)
  return value.map((entry, index) => {
    if (typeof entry !== 'string' || !entry.trim()) throw new ValidationError(`${key}[${index}]`, 'must be a non-empty glob pattern', entry)
    const pattern = entry.trim()
    if (pattern.length > MAX_PATTERN_LENGTH) throw new ValidationError(`${key}[${index}]`, `must be at most ${MAX_PATTERN_LENGTH} characters`, pattern.length)
    if (/[\u0000-\u001f\u007f]/.test(pattern)) throw new ValidationError(`${key}[${index}]`, 'must not contain control characters', pattern)
    if (toPosix(pattern).split('/').includes('..')) throw new ValidationError(`${key}[${index}]`, "must not contain '..' segments", pattern)
    try {
      new Bun.Glob(pattern)
    } catch (error) {
      throw new ValidationError(`${key}[${index}]`, `is not a valid glob: ${error instanceof Error ? error.message : String(error)}`, pattern)
    }
    return pattern
  })
}

function relativeDirectory(key: string, value: unknown): string {
  if (typeof value !== 'string' || !value.trim()) throw new ValidationError(key, 'must be a non-empty project-relative directory', value)
  const directory = value.trim().replace(/\/+$/, '')
  if (isAbsolute(directory) || directory.startsWith('~') || toPosix(directory).split('/').includes('..')) {
    throw new ValidationError(key, "must be relative to the project and must not contain '..'", value)
  }
  return directory || '.'
}

function stringValue(value: unknown): string | undefined {
  return typeof value === 'string' ? value : undefined
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
  return error instanceof Error && 'code' in error
}
