// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { basename, resolve } from 'node:path'

import { HookRunner } from '../extensions/hooks.js'
import {
  SkillRegistry,
  skillPromptSection,
  type Skill,
} from '../extensions/skills.js'
import { RepoMapper } from '../context/repoMap.js'
import { SandboxMode, type SandboxConfig } from '../security/sandbox.js'
import {
  PromptProfile,
  getPromptProfileConfig,
  resolvePromptProfileConfig,
  type PromptProfileConfig,
} from './promptProfiles.js'

export interface RuntimeInfo {
  readonly platform: string
  readonly runtimeVersion: string
  readonly timestamp: string
  readonly timezone: string
  readonly workingDirectory: string
  readonly workspaceName: string
  readonly xerxesVersion: string
}

export interface GitPromptContext {
  readonly branch: string
  readonly dirtyCount: number
  readonly recentCommits: readonly string[]
}

/** Effects needed to collect prompt context, kept separate from rendering. */
export interface PromptContextHost {
  captureRuntimeInfo(
    workspaceRoot: string | undefined,
  ): Promise<RuntimeInfo> | RuntimeInfo
  buildRepoMap?(workspaceRoot: string): Promise<string>
  gitContext?(workspaceRoot: string): Promise<GitPromptContext | undefined>
}

export type PromptMemoryProvider = (
  agentId: string | undefined,
  maximum: number,
) => Promise<readonly string[]> | readonly string[]

export type UserProfileProvider = (
  agentId: string | undefined,
) => Promise<string> | string

export interface PromptContextBuilderOptions {
  readonly defaultProfile?: PromptProfile | PromptProfileConfig
  readonly guardrails?: readonly string[]
  readonly hookRunner?: HookRunner
  readonly host: PromptContextHost
  readonly memoryProvider?: PromptMemoryProvider
  readonly sandboxConfig?: SandboxConfig
  readonly skillRegistry?: SkillRegistry
  readonly userProfileProvider?: UserProfileProvider
  readonly workspaceRoot?: string
}

export interface PromptContextBuildOptions {
  readonly agentId?: string
  readonly enabledSkills?: readonly Skill[]
  readonly guardrails?: readonly string[]
  readonly profile?: PromptProfile | PromptProfileConfig | string
  readonly sandboxConfig?: SandboxConfig
  readonly toolNames?: readonly string[]
  readonly workspaceRoot?: string
}

/** Separately rendered prompt fragments for inspection and profile gating. */
export interface PromptContext {
  readonly bootstrapSection: string
  readonly datetimeSection: string
  readonly enabledSkillsSection: string
  readonly gitSection: string
  readonly guardrailsSection: string
  readonly memorySection: string
  readonly reasoningSection: string
  readonly repoMapSection: string
  readonly runtimeSection: string
  readonly sandboxSection: string
  readonly skillsSection: string
  readonly toolsSection: string
  readonly userProfileSection: string
  readonly workspaceSection: string
}

export interface SystemPromptContextHostOptions {
  readonly now?: () => Date
  readonly repoMapper?: RepoMapper
  readonly xerxesVersion?: string
}

/**
 * Build a deterministic, profile-aware system-prompt prefix from native ports.
 *
 * External probes are intentionally optional: a missing git binary, repo map,
 * memory provider, or extension hook removes only that section, never the turn.
 */
export class PromptContextBuilder {
  private readonly defaultProfile: PromptProfileConfig
  private readonly guardrails: readonly string[]
  private readonly hookRunner: HookRunner | undefined
  private readonly host: PromptContextHost
  private readonly memoryProvider: PromptMemoryProvider | undefined
  private readonly sandboxConfig: SandboxConfig | undefined
  private readonly skillRegistry: SkillRegistry | undefined
  private readonly userProfileProvider: UserProfileProvider | undefined
  private readonly workspaceRoot: string | undefined

  constructor(options: PromptContextBuilderOptions) {
    this.defaultProfile =
      options.defaultProfile === undefined
        ? getPromptProfileConfig(PromptProfile.FULL)
        : resolvePromptProfileConfig(options.defaultProfile)
    this.guardrails = Object.freeze([...(options.guardrails ?? [])])
    this.hookRunner = options.hookRunner
    this.host = options.host
    this.memoryProvider = options.memoryProvider
    this.sandboxConfig = options.sandboxConfig
    this.skillRegistry = options.skillRegistry
    this.userProfileProvider = options.userProfileProvider
    this.workspaceRoot = options.workspaceRoot
  }

  /** Build every enabled context section without flattening it into prompt prose. */
  async build(options: PromptContextBuildOptions = {}): Promise<PromptContext> {
    const profile =
      options.profile === undefined
        ? this.defaultProfile
        : resolvePromptProfileConfig(options.profile)
    const workspaceRoot = options.workspaceRoot ?? this.workspaceRoot
    const runtime = await this.host.captureRuntimeInfo(workspaceRoot)
    const enabledSkills = options.enabledSkills ?? []
    const guardrails = options.guardrails ?? this.guardrails
    const sandbox = options.sandboxConfig ?? this.sandboxConfig

    const [
      bootstrapSection,
      gitSection,
      memorySection,
      repoMapSection,
      userProfileSection,
    ] = await Promise.all([
      profile.includeBootstrap
        ? this.bootstrapSection(options.agentId)
        : Promise.resolve(''),
      profile.includeGitInfo
        ? this.gitSection(workspaceRoot)
        : Promise.resolve(''),
      profile.includeRelevantMemories
        ? this.memorySection(options.agentId, profile)
        : Promise.resolve(''),
      profile.includeRepoMap
        ? this.repoMapSection(workspaceRoot)
        : Promise.resolve(''),
      profile.includeUserProfile
        ? this.userProfileSection(options.agentId)
        : Promise.resolve(''),
    ])

    return Object.freeze({
      runtimeSection: profile.includeRuntimeInfo
        ? renderRuntimeSection(runtime)
        : '',
      workspaceSection: profile.includeWorkspaceInfo
        ? renderWorkspaceSection(runtime)
        : '',
      datetimeSection: profile.includeRuntimeInfo
        ? renderDatetimeSection(runtime)
        : '',
      reasoningSection: profile.includeRuntimeInfo
        ? renderReasoningSection(profile)
        : '',
      sandboxSection: profile.includeSandboxInfo
        ? renderSandboxSection(sandbox)
        : '',
      skillsSection: profile.includeSkillsIndex
        ? renderSkillsSection(this.skillRegistry)
        : '',
      enabledSkillsSection: profile.includeEnabledSkills
        ? renderEnabledSkills(enabledSkills, profile)
        : '',
      toolsSection: profile.includeToolsList
        ? renderToolsSection(options.toolNames ?? [], profile)
        : '',
      guardrailsSection: profile.includeGuardrails
        ? renderGuardrailsSection(guardrails)
        : '',
      bootstrapSection,
      memorySection,
      userProfileSection,
      repoMapSection,
      gitSection,
    })
  }

  /** Render the full native prefix for one turn. */
  async assembleSystemPromptPrefix(
    options: PromptContextBuildOptions = {},
  ): Promise<string> {
    const profile =
      options.profile === undefined
        ? this.defaultProfile
        : resolvePromptProfileConfig(options.profile)
    if (profile.profile === PromptProfile.NONE) {
      return 'You are Xerxes, a runtime-managed AI agent operating inside a controlled tool environment.'
    }
    const context = await this.build({ ...options, profile })
    return [
      identityBlock(profile),
      toolingBlock(context),
      safetyBlock(context),
      skillsBlock(context),
      trimSection(context.userProfileSection),
      trimSection(context.memorySection),
      trimSection(context.repoMapSection),
      trimSection(context.gitSection),
      workspaceBlock(context),
      sandboxBlock(context),
      runtimeBlock(context),
      executionPolicyBlock(profile),
      outputStyleBlock(profile),
    ]
      .filter(Boolean)
      .join('\n\n')
  }

  /** Render the COMPACT profile without making callers duplicate its selection. */
  async buildCompactPrefix(
    options: Omit<PromptContextBuildOptions, 'profile'> = {},
  ): Promise<string> {
    return this.assembleSystemPromptPrefix({
      ...options,
      profile: PromptProfile.COMPACT,
    })
  }

  /** Render the MINIMAL profile without making callers duplicate its selection. */
  async buildMinimalPrefix(
    options: Omit<PromptContextBuildOptions, 'profile'> = {},
  ): Promise<string> {
    return this.assembleSystemPromptPrefix({
      ...options,
      profile: PromptProfile.MINIMAL,
    })
  }

  /** Render the bare identity-only NONE profile. */
  async buildNonePrefix(): Promise<string> {
    return this.assembleSystemPromptPrefix({ profile: PromptProfile.NONE })
  }

  private async bootstrapSection(agentId: string | undefined): Promise<string> {
    if (!this.hookRunner?.hasHooks('bootstrap_files')) return ''
    const values = await this.hookRunner.runAsync('bootstrap_files', {
      ...(agentId ? { agentId } : {}),
    })
    return flattenHookStrings(values).join('\n')
  }

  private async gitSection(workspaceRoot: string | undefined): Promise<string> {
    if (!workspaceRoot || !this.host.gitContext) return ''
    try {
      const context = await this.host.gitContext(workspaceRoot)
      if (!context?.branch) return ''
      const lines = ['[Git Context]', '  Branch: ' + context.branch]
      lines.push(
        context.dirtyCount
          ? `  Uncommitted changes: ${context.dirtyCount} file(s)`
          : '  Working tree: clean',
      )
      if (context.recentCommits.length) {
        lines.push(
          '  Recent commits:',
          ...context.recentCommits.map((commit) => '    ' + commit),
        )
      }
      return lines.join('\n') + '\n'
    } catch {
      return ''
    }
  }

  private async memorySection(
    agentId: string | undefined,
    profile: PromptProfileConfig,
  ): Promise<string> {
    if (!this.memoryProvider || profile.maxMemoriesInjected === 0) return ''
    try {
      const snippets = await this.memoryProvider(
        agentId,
        profile.maxMemoriesInjected,
      )
      const lines = snippets
        .slice(0, profile.maxMemoriesInjected)
        .map((snippet) => snippet.trim().replaceAll(/\s*\n\s*/g, ' '))
        .filter(Boolean)
        .map((snippet) => '  - ' + snippet)
      return lines.length
        ? '[Relevant Memories]\n' + lines.join('\n') + '\n'
        : ''
    } catch {
      return ''
    }
  }

  private async repoMapSection(
    workspaceRoot: string | undefined,
  ): Promise<string> {
    if (!workspaceRoot || !this.host.buildRepoMap) return ''
    try {
      const text = (await this.host.buildRepoMap(workspaceRoot)).trim()
      return text
        ? '[Repo Map — ranked by reference frequency]\n' + text + '\n'
        : ''
    } catch {
      return ''
    }
  }

  private async userProfileSection(
    agentId: string | undefined,
  ): Promise<string> {
    if (!this.userProfileProvider) return ''
    try {
      const text = (await this.userProfileProvider(agentId)).trim()
      return text ? '[User Profile]\n' + text + '\n' : ''
    } catch {
      return ''
    }
  }
}

/** Create the Bun-backed host explicitly used by local CLI/runtime composition. */
export function createSystemPromptContextHost(
  options: SystemPromptContextHostOptions = {},
): PromptContextHost {
  const now = options.now ?? (() => new Date())
  const repoMapper = options.repoMapper ?? new RepoMapper()
  const xerxesVersion = options.xerxesVersion ?? '0.2.0'
  return {
    captureRuntimeInfo(workspaceRoot) {
      const date = now()
      const workingDirectory = resolve(workspaceRoot ?? process.cwd())
      return {
        timestamp: date.toISOString(),
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'local',
        platform: process.platform,
        runtimeVersion: 'Bun ' + Bun.version,
        xerxesVersion,
        workingDirectory,
        workspaceName: basename(workingDirectory) || workingDirectory,
      }
    },
    async buildRepoMap(workspaceRoot) {
      return (await repoMapper.build(workspaceRoot)).text
    },
    async gitContext(workspaceRoot) {
      const branch = await runGit(
        ['rev-parse', '--abbrev-ref', 'HEAD'],
        workspaceRoot,
      )
      if (!branch) return undefined
      const status = await runGit(['status', '--porcelain'], workspaceRoot)
      const recent = await runGit(['log', '--oneline', '-3'], workspaceRoot)
      return {
        branch,
        dirtyCount: status ? status.split(/\r?\n/).filter(Boolean).length : 0,
        recentCommits: recent ? recent.split(/\r?\n/).filter(Boolean) : [],
      }
    },
  }
}

function renderRuntimeSection(info: RuntimeInfo): string {
  return (
    [
      '[Runtime Context]',
      '  Platform: ' + info.platform,
      '  Runtime: ' + info.runtimeVersion,
      '  Xerxes: v' + info.xerxesVersion,
    ].join('\n') + '\n'
  )
}

function renderWorkspaceSection(info: RuntimeInfo): string {
  return (
    [
      '[Workspace]',
      '  Directory: ' + info.workingDirectory,
      '  Project: ' + info.workspaceName,
    ].join('\n') + '\n'
  )
}

function renderDatetimeSection(info: RuntimeInfo): string {
  return (
    [
      '[Current Date & Time]',
      '  Local time: ' + info.timestamp,
      '  Time zone: ' + info.timezone,
    ].join('\n') + '\n'
  )
}

function renderReasoningSection(profile: PromptProfileConfig): string {
  return (
    [
      '[Response Guidance]',
      '  Profile: ' + profile.profile,
      '  Guidance: answer from actual tool and workspace results; avoid speculative claims;',
      '  keep internal reasoning private;',
      '  and put the final answer in normal assistant content, not a scratchpad or reasoning field.',
    ].join('\n') + '\n'
  )
}

function renderSandboxSection(config: SandboxConfig | undefined): string {
  if (!config) return ''
  const mode = config.mode ?? SandboxMode.OFF
  if (mode === SandboxMode.OFF)
    return '[Sandbox] Mode: off (all execution on host)\n'
  const sandboxed =
    [...(config.sandboxedTools ?? [])].sort().join(', ') || 'none'
  const elevated = [...(config.elevatedTools ?? [])].sort().join(', ') || 'none'
  return (
    [
      '[Sandbox]',
      '  Mode: ' + mode,
      '  Sandboxed tools: ' + sandboxed,
      '  Elevated tools: ' + elevated,
    ].join('\n') + '\n'
  )
}

function renderSkillsSection(registry: SkillRegistry | undefined): string {
  const index = registry?.markdownIndex() ?? ''
  return index ? '[Skills]\n' + index + '\n' : ''
}

function renderEnabledSkills(
  skills: readonly Skill[],
  profile: PromptProfileConfig,
): string {
  if (!skills.length) return ''
  const sections = skills.map((skill) =>
    truncate(skillPromptSection(skill), profile.maxSkillInstructionsLength),
  )
  return '[Enabled Skill Instructions]\n' + sections.join('\n\n') + '\n'
}

function renderToolsSection(
  tools: readonly string[],
  profile: PromptProfileConfig,
): string {
  if (!tools.length) return ''
  const maximum = profile.maxToolsListed
  const shown = maximum === undefined ? tools : tools.slice(0, maximum)
  const lines = shown.map((tool) => '  - ' + tool)
  if (maximum !== undefined && tools.length > maximum)
    lines.push(`  ... and ${tools.length - maximum} more`)
  return '[Available Tools]\n' + lines.join('\n') + '\n'
}

function renderGuardrailsSection(guardrails: readonly string[]): string {
  if (!guardrails.length) return ''
  return (
    '[Guardrails]\n' +
    guardrails.map((guardrail) => '  - ' + guardrail).join('\n') +
    '\n'
  )
}

function identityBlock(profile: PromptProfileConfig): string {
  const lines =
    profile.profile === PromptProfile.FULL
      ? [
          '[Identity]',
          '- You are Xerxes, a runtime-managed AI agent operating inside a controlled tool environment.',
          '- Complete the user task accurately, efficiently, and safely using available tools, skills, and workspace',
          '  context.',
          '- Follow runtime policy, sandbox limits, and tool restrictions.',
        ]
      : [
          '[Identity]',
          '- You are Xerxes, a delegated sub-agent running inside a controlled runtime.',
          '- Stay within the assigned subtask and return integration-friendly output.',
          '- Follow runtime policy, sandbox limits, and tool restrictions.',
        ]
  return lines.join('\n')
}

function toolingBlock(context: PromptContext): string {
  const tools =
    trimSection(context.toolsSection) || '[Available Tools]\n  - none'
  const lower = tools.toLowerCase()
  const lines = [
    '[Tooling]',
    'Available tools in this run:',
    tools,
    'Tool rules:',
    '- Use tools only when you need live external state, workspace contents, shell execution, or another real action.',
    '- Do not use or simulate tools for greetings, simple arithmetic, direct explanations, or summaries.',
    '- Do not repeat the same tool call with identical arguments when it did not make progress.',
    '- If a tool result answers the task, use it directly; if it fails, adjust strategy instead of blindly retrying.',
  ]
  const guidance: string[] = []
  if (lower.includes('web.search_query')) {
    guidance.push(
      '- `web.search_query`: use for explicit web requests and claims that depend on current information.',
    )
    guidance.push(
      '- Generic web-search follow-ups such as `search the web`, `look it up`, or `find it` rely on prior context; do not force an unrelated search.',
    )
    guidance.push(
      '- When web tools are available, do not claim that you cannot browse or access current information.',
    )
  }
  if (lower.includes('web.open')) {
    guidance.push(
      '- `web.open`: use after search when source contents or direct verification are needed.',
    )
  }
  if (lower.includes('readfile') || lower.includes('read_file')) {
    guidance.push(
      '- File-reading tools: use them for project-specific facts and exact code behavior.',
    )
  }
  if (lower.includes('exec_command') || lower.includes('write_stdin')) {
    guidance.push(
      '- Shell tools: use terminal sessions for short and long commands, and report actual results.',
    )
  }
  if (guidance.length) lines.push('Tool selection guidance:', ...guidance)
  lines.push(
    'Search grounding rules:',
    '- Search snippets and result titles are leads, not verification unless the source was opened and confirmed.',
  )
  return lines.join('\n')
}

function safetyBlock(context: PromptContext): string {
  return [
    '[Safety]',
    'Safety guidance:',
    trimSection(context.guardrailsSection) ||
      '- No additional runtime guardrails are configured for this run.',
    'Safety rules:',
    '- Do not try to bypass oversight, sandboxing, or tool restrictions.',
    '- Do not invent tool results, file contents, or execution outcomes.',
    '- If blocked by runtime policy or sandbox limits, say so plainly.',
  ].join('\n')
}

function skillsBlock(context: PromptContext): string {
  if (!context.skillsSection && !context.enabledSkillsSection) return ''
  const lines = ['[Skills & Instructions]']
  if (context.skillsSection)
    lines.push('Available skills:', trimSection(context.skillsSection))
  if (context.enabledSkillsSection)
    lines.push(
      'Enabled skill instructions:',
      trimSection(context.enabledSkillsSection),
    )
  lines.push(
    'Skill rules:',
    '- Use enabled skills as task-specific operating instructions.',
    '- If a skill is listed but not fully injected, load or apply it only when relevant.',
    '- Do not assume a skill exists unless it is present in runtime context.',
  )
  return lines.join('\n')
}

function workspaceBlock(context: PromptContext): string {
  if (!context.workspaceSection && !context.bootstrapSection) return ''
  const lines = ['[Workspace Context]']
  if (context.workspaceSection)
    lines.push(trimSection(context.workspaceSection))
  if (context.bootstrapSection)
    lines.push(
      'Project/bootstrap context:',
      trimSection(context.bootstrapSection),
    )
  lines.push(
    'Workspace rules:',
    '- Treat workspace files as the source of truth for project-specific behavior.',
    '- Prefer minimal, targeted changes over broad rewrites.',
  )
  return lines.join('\n')
}

function sandboxBlock(context: PromptContext): string {
  if (!context.sandboxSection) return ''
  return [
    '[Sandbox Runtime]',
    trimSection(context.sandboxSection),
    'Sandbox rules:',
    '- Treat sandboxed tools as sandboxed.',
    '- Elevated execution is exceptional and must be explicit.',
    '- Never describe host execution as sandboxed if it was not.',
  ].join('\n')
}

function runtimeBlock(context: PromptContext): string {
  const sections = [
    context.runtimeSection,
    context.datetimeSection,
    context.reasoningSection,
  ].filter(Boolean)
  return sections.length
    ? '[Runtime]\n' + sections.map(trimSection).join('\n')
    : ''
}

function executionPolicyBlock(profile: PromptProfileConfig): string {
  const lines =
    profile.profile === PromptProfile.FULL
      ? [
          '[Execution Policy]',
          '1. Understand the request and use workspace context first.',
          '2. Choose the smallest correct action that moves the task forward.',
          '3. Use tools only for missing live information, file contents, execution, or verification.',
          '4. Do not simulate tool calls or wrap normal answers in tool/XML markup.',
          '5. If one successful result is enough, stop and give the final answer.',
          '6. After tool use, answer from actual results and surface blockers, assumptions, and risks.',
          '7. Do not loop.',
        ]
      : [
          '[Execution Policy]',
          '- Stay within the assigned subtask.',
          '- Use tools only when needed for missing live information or real actions.',
          '- Do not simulate tool calls or emit tool/XML wrappers in normal answers.',
          '- Answer from actual tool and workspace results; keep output compact and integration-friendly.',
        ]
  return lines.join('\n')
}

function outputStyleBlock(profile: PromptProfileConfig): string {
  if (profile.profile !== PromptProfile.FULL) return ''
  return [
    '[Output Style]',
    '- Be precise, technical, and pragmatic.',
    '- Prefer concrete outcomes over general advice.',
    '- Keep internal reasoning out of the visible answer unless explicitly asked.',
    '- If code changed or tests ran, mention the real result and scope.',
  ].join('\n')
}

function flattenHookStrings(value: unknown): string[] {
  if (typeof value === 'string') return value.trim() ? [value.trim()] : []
  if (Array.isArray(value)) return value.flatMap(flattenHookStrings)
  return []
}

function trimSection(value: string): string {
  return value.trim()
}

function truncate(value: string, maximum: number | undefined): string {
  return maximum !== undefined && value.length > maximum
    ? value.slice(0, maximum) + '...'
    : value
}

async function runGit(
  arguments_: readonly string[],
  cwd: string,
): Promise<string> {
  try {
    const child = Bun.spawn(['git', ...arguments_], {
      cwd,
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'pipe',
    })
    const [exitCode, stdout] = await Promise.all([
      child.exited,
      new Response(child.stdout).text(),
    ])
    return exitCode === 0 ? stdout.trim() : ''
  } catch {
    return ''
  }
}
