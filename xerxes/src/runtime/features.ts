// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { statSync } from 'node:fs'
import { isAbsolute, join, resolve } from 'node:path'

import { HOOK_POINTS, HookRunner, type HookPoint } from '../extensions/hooks.js'
import { PluginRegistry } from '../extensions/plugins.js'
import { SkillRegistry, type Skill, type SkillDependencyLookup } from '../extensions/skills.js'
import {
  HIGH_POWER_OPERATOR_TOOLS,
  OperatorState,
  type OperatorRuntimeConfig,
} from '../operators/index.js'
import { PolicyEngine, ToolPolicy } from '../security/policy.js'
import { SandboxRouter, type SandboxBackend, type SandboxConfig } from '../security/sandbox.js'

/** Per-agent values that replace the corresponding runtime-wide setting. */
export interface AgentRuntimeOverrides {
  readonly enabledSkills?: readonly string[]
  readonly guardrails?: readonly string[]
  readonly policy?: ToolPolicy
  readonly sandbox?: SandboxConfig
}

/** Declarative, host-owned feature composition input. */
export interface RuntimeFeaturesConfig {
  /** Master feature switch for callers that own the turn loop. Composition remains inspectable when false. */
  readonly enabled?: boolean
  /** Absolute workspace root used to resolve relative extension directories and conventional roots. */
  readonly workspaceRoot?: string
  readonly pluginDirectories?: readonly string[]
  readonly skillDirectories?: readonly string[]
  /** Also probe `<workspaceRoot>/plugins` and `<workspaceRoot>/skills` when they exist. Defaults to true. */
  readonly discoverConventionalExtensions?: boolean
  readonly guardrails?: readonly string[]
  readonly policy?: ToolPolicy
  readonly sandbox?: SandboxConfig
  readonly enabledSkills?: readonly string[]
  readonly operator?: OperatorRuntimeConfig
  readonly agentOverrides?: ReadonlyMap<string, AgentRuntimeOverrides> | Readonly<Record<string, AgentRuntimeOverrides>>
}

/** Explicit filesystem boundary used only for lexical extension-root discovery. */
export interface RuntimeFeatureFilesystem {
  isDirectory(path: string): boolean | Promise<boolean>
  join(...segments: readonly string[]): string
  resolve(path: string): string
}

/** Native loader boundary; callers can replace it for a different extension transport. */
export interface RuntimeExtensionLoader {
  discoverPlugins(registry: PluginRegistry, directory: string): Promise<readonly string[]>
  discoverSkills(registry: SkillRegistry, directories: readonly string[]): Promise<readonly string[]>
}

/** Dependencies supplied by a host which already owns the concrete runtime objects. */
export interface RuntimeFeatureCompositionOptions {
  readonly extensionLoader?: RuntimeExtensionLoader
  readonly filesystem?: RuntimeFeatureFilesystem
  readonly hookRunner?: HookRunner
  /** Host-composed operator state, including any explicitly injected operator ports. */
  readonly operatorState?: OperatorState
  readonly pluginRegistry?: PluginRegistry
  readonly sandboxBackend?: SandboxBackend
  readonly skillRegistry?: SkillRegistry
  /** Existing tool lookup used to validate skill required_tools without fabricating handlers. */
  readonly toolLookup?: SkillDependencyLookup
}

export interface RuntimeExtensionDirectories {
  readonly pluginDirectories: readonly string[]
  readonly skillDirectories: readonly string[]
}

export interface RuntimeExtensionDiscovery extends RuntimeExtensionDirectories {
  readonly pluginNames: readonly string[]
  readonly skillNames: readonly string[]
}

/** Raised only after all discovered plugin and skill dependency errors have been collected. */
export class RuntimeExtensionDependencyError extends Error {
  readonly errors: readonly string[]

  constructor(errors: readonly string[]) {
    super(`Runtime extension dependency validation failed:\n${errors.join('\n')}`)
    this.name = 'RuntimeExtensionDependencyError'
    this.errors = Object.freeze([...errors])
  }
}

/** Create the native filesystem port without consulting process.cwd() or a Python runtime. */
export function createRuntimeFeatureFilesystem(): RuntimeFeatureFilesystem {
  const filesystem: RuntimeFeatureFilesystem = {
    isDirectory: (path: string) => {
      try {
        return statSync(path).isDirectory()
      } catch {
        return false
      }
    },
    join: (...segments: readonly string[]) => join(...segments),
    resolve: (path: string) => resolve(path),
  }
  return Object.freeze(filesystem)
}

/** Create the normal local-registry loader. The registries still own parsing and plugin execution. */
export function createNativeRuntimeExtensionLoader(): RuntimeExtensionLoader {
  const loader: RuntimeExtensionLoader = {
    discoverPlugins: (registry: PluginRegistry, directory: string) => registry.discover(directory),
    discoverSkills: (registry: SkillRegistry, directories: readonly string[]) => registry.discover(...directories),
  }
  return Object.freeze(loader)
}

/**
 * Resolve configured and conventional extension directories in deterministic precedence order.
 *
 * Relative configured directories are meaningful only with an explicit absolute workspace root;
 * this keeps composition independent of whichever directory started the Bun process.
 */
export async function resolveRuntimeExtensionDirectories(
  config: RuntimeFeaturesConfig,
  filesystem: RuntimeFeatureFilesystem = createRuntimeFeatureFilesystem(),
): Promise<RuntimeExtensionDirectories> {
  const workspaceRoot = normalizeWorkspaceRoot(config.workspaceRoot, filesystem)
  const pluginDirectories = resolveConfiguredDirectories(config.pluginDirectories ?? [], workspaceRoot, filesystem)
  const skillDirectories = resolveConfiguredDirectories(config.skillDirectories ?? [], workspaceRoot, filesystem)
  if ((config.discoverConventionalExtensions ?? true) && workspaceRoot !== undefined) {
    await appendConventionalDirectory(pluginDirectories, workspaceRoot, 'plugins', filesystem)
    await appendConventionalDirectory(skillDirectories, workspaceRoot, 'skills', filesystem)
  }
  return Object.freeze({
    pluginDirectories: Object.freeze(pluginDirectories),
    skillDirectories: Object.freeze(skillDirectories),
  })
}

/**
 * Compose native feature registries once at session setup.
 *
 * It discovers plugins and skills, validates their declared dependencies, installs known plugin
 * hooks, and derives the per-agent policy, sandbox, loop, skill, and operator accessors below.
 * It never constructs a provider client, shell handler, or placeholder tool implementation.
 */
export async function composeRuntimeFeatures(
  config: RuntimeFeaturesConfig,
  options: RuntimeFeatureCompositionOptions = {},
): Promise<RuntimeFeaturesState> {
  const pluginRegistry = options.pluginRegistry ?? new PluginRegistry()
  const skillRegistry = options.skillRegistry ?? new SkillRegistry()
  const hookRunner = options.hookRunner ?? new HookRunner()
  const loader = options.extensionLoader ?? createNativeRuntimeExtensionLoader()
  const directories = await resolveRuntimeExtensionDirectories(config, options.filesystem)
  const pluginNames: string[] = []
  for (const directory of directories.pluginDirectories) {
    pluginNames.push(...await loader.discoverPlugins(pluginRegistry, directory))
  }
  const skillNames = await loader.discoverSkills(skillRegistry, directories.skillDirectories)
  validateDependencies(pluginRegistry, skillRegistry, options.toolLookup)
  const registeredHooks = registerPluginHooks(pluginRegistry, hookRunner)
  const effectivePolicy = policyWithOperatorAccess(config.policy, config.operator)
  const policyEngine = new PolicyEngine({
    globalPolicy: effectivePolicy,
    agentPolicies: agentPolicies(config.agentOverrides),
  })
  const operatorState = config.operator?.enabled === true
    ? options.operatorState ?? new OperatorState({ config: config.operator })
    : undefined
  const stateOptions: RuntimeFeaturesStateOptions = {
    config,
    discovery: {
      ...directories,
      pluginNames: Object.freeze([...pluginNames]),
      skillNames: Object.freeze([...skillNames]),
    },
    effectivePolicy,
    hookRunner,
    pluginRegistry,
    policyEngine,
    registeredHooks,
    skillRegistry,
    ...(operatorState === undefined ? {} : { operatorState }),
    ...(options.sandboxBackend === undefined ? {} : { sandboxBackend: options.sandboxBackend }),
  }
  return new RuntimeFeaturesState(stateOptions)
}

interface RuntimeFeaturesStateOptions {
  readonly config: RuntimeFeaturesConfig
  readonly discovery: RuntimeExtensionDiscovery
  readonly effectivePolicy: ToolPolicy
  readonly hookRunner: HookRunner
  readonly operatorState?: OperatorState
  readonly pluginRegistry: PluginRegistry
  readonly policyEngine: PolicyEngine
  readonly registeredHooks: Readonly<Record<HookPoint, number>>
  readonly sandboxBackend?: SandboxBackend
  readonly skillRegistry: SkillRegistry
}

/** Fully composed native runtime feature state, with immutable configuration-derived accessors. */
export class RuntimeFeaturesState {
  readonly config: RuntimeFeaturesConfig
  readonly discovery: RuntimeExtensionDiscovery
  readonly effectivePolicy: ToolPolicy
  readonly hookRunner: HookRunner
  readonly operatorState: OperatorState | undefined
  readonly pluginRegistry: PluginRegistry
  readonly policyEngine: PolicyEngine
  readonly registeredHooks: Readonly<Record<HookPoint, number>>
  readonly skillRegistry: SkillRegistry
  private readonly overrides: ReadonlyMap<string, AgentRuntimeOverrides>
  private readonly sandboxBackend: SandboxBackend | undefined
  private readonly sandboxRouters = new Map<string, SandboxRouter>()

  constructor(options: RuntimeFeaturesStateOptions) {
    this.config = options.config
    this.discovery = freezeDiscovery(options.discovery)
    this.effectivePolicy = options.effectivePolicy
    this.hookRunner = options.hookRunner
    this.operatorState = options.operatorState
    this.pluginRegistry = options.pluginRegistry
    this.policyEngine = options.policyEngine
    this.registeredHooks = Object.freeze({ ...options.registeredHooks })
    this.sandboxBackend = options.sandboxBackend
    this.skillRegistry = options.skillRegistry
    this.overrides = normalizeOverrides(options.config.agentOverrides)
  }

  get enabled(): boolean {
    return this.config.enabled ?? false
  }

  getAgentOverrides(agentId: string | undefined): AgentRuntimeOverrides {
    return agentId === undefined ? EMPTY_OVERRIDES : this.overrides.get(agentId) ?? EMPTY_OVERRIDES
  }

  getGuardrails(agentId?: string): readonly string[] {
    const override = this.getAgentOverrides(agentId)
    return Object.freeze([...(override.guardrails ?? this.config.guardrails ?? [])])
  }

  getEnabledSkillNames(agentId?: string): readonly string[] {
    const override = this.getAgentOverrides(agentId)
    return Object.freeze([...(override.enabledSkills ?? this.config.enabledSkills ?? [])])
  }

  /** Resolve enabled skill names to discovered skills; missing names are observable through the paired helper. */
  getEnabledSkills(agentId?: string): Skill[] {
    return this.getEnabledSkillNames(agentId)
      .map(name => this.skillRegistry.get(name))
      .filter((skill): skill is Skill => skill !== undefined)
  }

  getMissingEnabledSkillNames(agentId?: string): string[] {
    return this.getEnabledSkillNames(agentId).filter(name => this.skillRegistry.get(name) === undefined)
  }

  getSandboxConfig(agentId?: string): SandboxConfig | undefined {
    return this.getAgentOverrides(agentId).sandbox ?? this.config.sandbox
  }

  /** Return one real sandbox router per agent, or undefined when no sandbox is configured. */
  getSandboxRouter(agentId?: string): SandboxRouter | undefined {
    const config = this.getSandboxConfig(agentId)
    if (config === undefined) return undefined
    const key = agentId ?? '__default__'
    const cached = this.sandboxRouters.get(key)
    if (cached !== undefined) return cached
    const router = new SandboxRouter({
      config,
      ...(this.sandboxBackend === undefined ? {} : { backend: this.sandboxBackend }),
    })
    this.sandboxRouters.set(key, router)
    return router
  }

  getOperatorConfig(): OperatorRuntimeConfig | undefined {
    return this.config.operator
  }

  /** Release the only session-attached resource this composition can create. */
  async close(): Promise<void> {
    await this.operatorState?.close()
  }
}

const EMPTY_OVERRIDES: AgentRuntimeOverrides = Object.freeze({})

function normalizeWorkspaceRoot(
  workspaceRoot: string | undefined,
  filesystem: RuntimeFeatureFilesystem,
): string | undefined {
  if (workspaceRoot === undefined) return undefined
  const value = requireDirectoryText(workspaceRoot, 'workspaceRoot')
  if (!isAbsolute(value)) {
    throw new TypeError('workspaceRoot must be an absolute path')
  }
  return filesystem.resolve(value)
}

function resolveConfiguredDirectories(
  configured: readonly string[],
  workspaceRoot: string | undefined,
  filesystem: RuntimeFeatureFilesystem,
): string[] {
  const directories: string[] = []
  for (const configuredDirectory of configured) {
    const value = requireDirectoryText(configuredDirectory, 'extension directory')
    if (!isAbsolute(value) && workspaceRoot === undefined) {
      throw new TypeError(`Relative extension directory '${value}' requires an explicit workspaceRoot`)
    }
    const candidate = isAbsolute(value)
      ? filesystem.resolve(value)
      : filesystem.resolve(filesystem.join(workspaceRoot!, value))
    if (!directories.includes(candidate)) directories.push(candidate)
  }
  return directories
}

async function appendConventionalDirectory(
  directories: string[],
  workspaceRoot: string,
  name: 'plugins' | 'skills',
  filesystem: RuntimeFeatureFilesystem,
): Promise<void> {
  const candidate = filesystem.resolve(filesystem.join(workspaceRoot, name))
  if (directories.includes(candidate) || !(await filesystem.isDirectory(candidate))) return
  directories.push(candidate)
}

function validateDependencies(
  pluginRegistry: PluginRegistry,
  skillRegistry: SkillRegistry,
  suppliedToolLookup: SkillDependencyLookup | undefined,
): void {
  const toolLookup: SkillDependencyLookup = {
    hasTool: name => pluginRegistry.getTool(name) !== undefined || suppliedToolLookup?.hasTool(name) === true,
  }
  const errors = [
    ...pluginRegistry.validateDependencies().map(error => `Plugin dependency issue: ${error}`),
    ...skillRegistry.validateDependencies(toolLookup).map(error => `Skill dependency issue: ${error}`),
  ]
  if (errors.length) throw new RuntimeExtensionDependencyError(errors)
}

function registerPluginHooks(
  pluginRegistry: PluginRegistry,
  hookRunner: HookRunner,
): Readonly<Record<HookPoint, number>> {
  const registered = {} as Record<HookPoint, number>
  for (const hookPoint of HOOK_POINTS) {
    const callbacks = pluginRegistry.getHooks(hookPoint)
    for (const callback of callbacks) hookRunner.register(hookPoint, callback)
    registered[hookPoint] = callbacks.length
  }
  return registered
}

function policyWithOperatorAccess(
  configuredPolicy: ToolPolicy | undefined,
  operator: OperatorRuntimeConfig | undefined,
): ToolPolicy {
  const policy = configuredPolicy ?? new ToolPolicy()
  const optionalTools = new Set(policy.optionalTools)
  if (operator?.enabled === true) {
    for (const toolName of HIGH_POWER_OPERATOR_TOOLS) {
      if (operator.powerToolsEnabled) optionalTools.delete(toolName.toLowerCase())
      else optionalTools.add(toolName.toLowerCase())
    }
  }
  return new ToolPolicy({
    allow: policy.allow,
    deny: policy.deny,
    optionalTools,
  })
}

function agentPolicies(
  overrides: RuntimeFeaturesConfig['agentOverrides'],
): ReadonlyMap<string, ToolPolicy> {
  const policies = new Map<string, ToolPolicy>()
  for (const [agentId, override] of normalizeOverrides(overrides)) {
    if (override.policy !== undefined) policies.set(agentId, override.policy)
  }
  return policies
}

function normalizeOverrides(
  overrides: RuntimeFeaturesConfig['agentOverrides'],
): ReadonlyMap<string, AgentRuntimeOverrides> {
  if (overrides === undefined) return new Map()
  if (typeof (overrides as ReadonlyMap<string, AgentRuntimeOverrides>).entries === 'function') {
    return new Map(overrides as ReadonlyMap<string, AgentRuntimeOverrides>)
  }
  return new Map(Object.entries(overrides))
}

function freezeDiscovery(discovery: RuntimeExtensionDiscovery): RuntimeExtensionDiscovery {
  return Object.freeze({
    pluginDirectories: Object.freeze([...discovery.pluginDirectories]),
    skillDirectories: Object.freeze([...discovery.skillDirectories]),
    pluginNames: Object.freeze([...discovery.pluginNames]),
    skillNames: Object.freeze([...discovery.skillNames]),
  })
}

function requireDirectoryText(value: string, name: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be a non-empty path`)
  if (value.includes('\0')) throw new TypeError(`${name} must not contain a null byte`)
  return value.trim()
}
