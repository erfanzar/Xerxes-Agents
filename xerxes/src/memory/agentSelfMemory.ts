// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, mkdir, open, readFile, readdir, rename, rm, stat } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'

import { ValidationError } from '../core/errors.js'
import { xerxesHome } from '../daemon/paths.js'
import { withFileLock } from '../session/daemonTranscript.js'

export const AGENT_SELF_MEMORY_KEYS = [
  'user_taste',
  'project_context',
  'skill_journal',
  'self_reflection',
  'tool_usage_patterns',
] as const

export type AgentSelfMemoryKey = (typeof AGENT_SELF_MEMORY_KEYS)[number]
export type AgentSelfMemoryLearningCategory = 'self_reflection' | 'skill_proposal' | 'tool_pattern' | 'user_taste'

const KEY_FILES: Readonly<Record<AgentSelfMemoryKey, string>> = Object.freeze({
  user_taste: 'user_taste.md',
  project_context: 'project_context.md',
  skill_journal: 'skill_journal.md',
  self_reflection: 'self_reflection.md',
  tool_usage_patterns: 'tool_usage_patterns.md',
})

const DEFAULT_CONTENT: Readonly<Record<AgentSelfMemoryKey, string>> = Object.freeze({
  user_taste: '# User Taste Profile\n\n## Communication Style\n-\n\n## Preferred Tools\n-\n\n## Common Workflows\n-\n\n## Frustrations / Avoid\n-\n\n## Notes\n\n',
  project_context: '# Project Context\n\n## AGENTS.md Summary\n\n## XERXES.md Summary\n\n## Project Conventions\n-\n\n## Important Files\n-\n\n',
  skill_journal: '# Skill Journal\n\n## Observed Patterns\n\n## Proposed Skills\n\n## Implemented Skills\n\n',
  self_reflection: '# Self Reflection\n\n## What Worked\n\n## What Did Not\n\n## Improvements\n\n',
  tool_usage_patterns: '# Tool Usage Patterns\n\n## Frequently Used Tools\n\n## Tool Combinations\n\n## Success Patterns\n\n',
})

export interface AgentSelfMemoryOptions {
  readonly agentId: string
  readonly directory?: string
  readonly projectRoot?: string
}

/** Maximum age before an unverifiable self-memory mutation lock may be recovered. */
const MUTATION_LOCK_STALE_MS = 30_000
/** Maximum time a read-modify-write cycle waits for an active cross-process owner. */
const MUTATION_LOCK_WAIT_MS = 10_000

/**
 * Per-agent persistent self-knowledge, distinct from global/project durable
 * conversation memory. It backs the learn and sync-context tool family.
 */
export class AgentSelfMemory {
  readonly agentId: string
  readonly directory: string
  readonly projectRoot: string
  private readonly appendLocks: Map<AgentSelfMemoryKey, Promise<void>>

  constructor(options: AgentSelfMemoryOptions) {
    this.agentId = normalizeAgentId(options.agentId)
    this.directory = resolve(options.directory ?? join(xerxesHome(), 'agent_memory', this.agentId))
    this.projectRoot = resolve(options.projectRoot ?? process.cwd())
    this.appendLocks = selfMemoryLocks.get(this.directory) ?? new Map<AgentSelfMemoryKey, Promise<void>>()
    selfMemoryLocks.set(this.directory, this.appendLocks)
  }

  async ensure(): Promise<void> {
    await mkdir(this.directory, { recursive: true })
    for (const key of AGENT_SELF_MEMORY_KEYS) await this.ensureKey(key)
  }

  async read(key: AgentSelfMemoryKey | string): Promise<string> {
    const path = this.pathFor(normalizeKey(key))
    await this.ensure()
    try {
      return await readFile(path, 'utf8')
    } catch (error) {
      if (isMissing(error)) return ''
      throw error
    }
  }

  async readAll(): Promise<Readonly<Record<AgentSelfMemoryKey, string>>> {
    const values = {} as Record<AgentSelfMemoryKey, string>
    for (const key of AGENT_SELF_MEMORY_KEYS) values[key] = await this.read(key)
    return values
  }

  async write(key: AgentSelfMemoryKey | string, content: string): Promise<void> {
    const normalized = normalizeKey(key)
    if (typeof content !== 'string') throw new ValidationError('content', 'must be a string', content)
    // The whole-file replace publishes through atomic rename like every other
    // mutation, so it shares the per-file OS lock: without it a rename could
    // land between another process's read and publish (or vice versa) and
    // silently erase entries written in between.
    await this.withAppendLock(normalized, () => this.withMutationLock(normalized, () => this.writeUnlocked(normalized, content)))
  }

  async append(key: AgentSelfMemoryKey | string, content: string): Promise<void> {
    const normalized = normalizeKey(key)
    if (typeof content !== 'string') throw new ValidationError('content', 'must be a string', content)
    await this.withAppendLock(normalized, () => this.appendUnlocked(normalized, content))
  }

  async patch(key: AgentSelfMemoryKey | string, oldText: string, newText: string): Promise<boolean> {
    const normalized = normalizeKey(key)
    return this.withAppendLock(normalized, () => this.withMutationLock(normalized, async () => {
      const content = await this.read(normalized)
      if (!content.includes(oldText)) return false
      await this.writeUnlocked(normalized, content.replace(oldText, newText))
      return true
    }))
  }

  async syncProjectContext(projectRoot = this.projectRoot): Promise<void> {
    const root = resolve(projectRoot)
    const sections = ['# Project Context']
    for (const name of ['AGENTS.md', 'XERXES.md', 'USER.md', 'SOUL.md']) {
      const content = await readProjectFile(root, name)
      if (!content) continue
      const fence = String.fromCharCode(96).repeat(3)
      sections.push('## ' + name + '\n' + fence + '\n' + content.slice(0, 2_000) + '\n' + fence)
    }
    await this.write(AgentSelfMemoryKeyProjectContext, sections.join('\n\n') + '\n')
  }

  async learn(
    observation: string,
    category: AgentSelfMemoryLearningCategory | string,
    _importance: 'high' | 'low' | 'medium' | string = 'medium',
  ): Promise<string> {
    const value = observation.trim()
    if (!value) throw new ValidationError('observation', 'must be non-empty', observation)
    if (category === 'user_taste') {
      await this.updateUserTaste(value)
      return 'User taste updated: ' + value
    }
    if (category === 'tool_pattern') {
      await this.append('tool_usage_patterns', '- ' + value)
      return 'Tool pattern recorded: ' + value
    }
    if (category === 'skill_proposal') {
      const name = value.split('.')[0]?.slice(0, 40).trim() || 'Observed skill'
      await this.proposeSkill(name, value, 'observed')
      return 'Skill proposed: ' + name
    }
    if (category === 'self_reflection') {
      await this.append('self_reflection', '- ' + value)
      return 'Self-reflection recorded: ' + value
    }
    return 'Unknown category: ' + category
  }

  async learnFromInteraction(
    userMessage: string,
    agentResponse: string,
    toolsUsed: readonly string[],
    success: boolean,
  ): Promise<void> {
    await this.append('tool_usage_patterns', '- ' + toolsUsed.join(', ') + ' -> ' + (success ? 'success' : 'failure'))
    const reflection = [
      '## Turn at ' + new Date().toISOString(),
      'User: ' + userMessage.slice(0, 200),
      'Tools: ' + JSON.stringify(toolsUsed),
      'Result: ' + (success ? 'success' : 'failure'),
      'Response: ' + agentResponse.slice(0, 200),
    ].join('\n')
    await this.append('self_reflection', reflection)
  }

  async updateUserTaste(preference: string, category = 'notes'): Promise<void> {
    await this.withAppendLock('user_taste', () => this.withMutationLock('user_taste', async () => {
      const content = await this.read('user_taste')
      const header = '## ' + titleCase(category.replaceAll('_', ' '))
      const entry = '\n- ' + preference
      const index = content.indexOf(header)
      if (index < 0) {
        // Inline the write so the whole read-modify-write stays inside both locks.
        await this.writeUnlocked('user_taste', content + '\n' + entry)
        return
      }
      const afterHeader = index + header.length
      const nextSection = content.indexOf('\n## ', afterHeader)
      const updated = nextSection < 0
        ? content + entry + '\n'
        : content.slice(0, nextSection) + entry + '\n' + content.slice(nextSection)
      await this.writeUnlocked('user_taste', updated)
    }))
  }

  async proposeSkill(name: string, description: string, pattern: string): Promise<void> {
    await this.append(
      'skill_journal',
      '\n### ' + name + '\n- Description: ' + description + '\n- Pattern: ' + pattern + '\n- Status: proposed\n',
    )
  }

  async markSkillImplemented(name: string): Promise<void> {
    await this.withAppendLock('skill_journal', () => this.withMutationLock('skill_journal', async () => {
      const content = await this.read('skill_journal')
      const marker = '### ' + name
      const start = content.indexOf(marker)
      if (start < 0) return
      const next = content.indexOf('\n### ', start + marker.length)
      const section = next < 0 ? content.slice(start) : content.slice(start, next)
      if (!section.includes('Status: proposed')) return
      const updated = content.slice(0, start) + section.replace('Status: proposed', 'Status: implemented') + (next < 0 ? '' : content.slice(next))
      await this.writeUnlocked('skill_journal', updated)
    }))
  }

  async systemPromptAddendum(): Promise<string> {
    const parts: string[] = []
    for (const [label, key] of [
      ['User Taste Profile', 'user_taste'],
      ['Project Context', 'project_context'],
      ['Tool Usage Patterns', 'tool_usage_patterns'],
    ] as const) {
      const content = await this.read(key)
      if (content.trim()) parts.push('[' + label + ']\n' + content)
    }
    if (!parts.length) return ''
    return 'MEMORY INSTRUCTION: You have persistent memory. Read relevant memory at session start and write important observations.\n\n'
      + parts.join('\n\n')
  }

  private pathFor(key: AgentSelfMemoryKey): string {
    return join(this.directory, KEY_FILES[key])
  }

  /**
   * Create one memory file with its template through a single exclusive
   * `'wx'` handle: the template bytes go through the same handle before it
   * closes, so there is no exists-check/truncating-write window in which a
   * concurrent creator or appender could be clobbered, and no crashed
   * creator can leave anything worse than an empty file — which the next
   * pass heals. A foreign-created file is respected and never templated.
   */
  private async ensureKey(key: AgentSelfMemoryKey): Promise<void> {
    const path = this.pathFor(key)
    try {
      const handle = await open(path, 'wx')
      try {
        await handle.writeFile(DEFAULT_CONTENT[key], 'utf8')
      } finally {
        await handle.close()
      }
      return
    } catch (error) {
      if ((error as NodeJS.ErrnoException)?.code !== 'EEXIST') throw error
    }
    // Heal the empty leftover of a creator that crashed between exclusive
    // creation and its template write. The size re-check right before the
    // write bounds the race against a concurrent appender populating the
    // file between the stat and the heal.
    let info: Awaited<ReturnType<typeof stat>>
    try {
      info = await stat(path)
    } catch (error) {
      if (isMissing(error)) return
      throw error
    }
    if (!info.isFile() || info.size > 0) return
    const handle = await open(path, 'r+')
    try {
      if ((await handle.stat()).size > 0) return
      await handle.writeFile(DEFAULT_CONTENT[key], 'utf8')
    } finally {
      await handle.close()
    }
  }

  /**
   * Append-only write through O_APPEND instead of read-modify-write: kernel
   * append positioning keeps every entry durable without a rewrite.
   *
   * The publish still runs under the per-file OS mutation lock because the
   * RMW operations (write/patch/updateUserTaste/markSkillImplemented) replace
   * the whole file via atomic rename: an O_APPEND landing between one of
   * those reads and its rename was permanently erased by that rename. Under
   * the shared lock every append either fully precedes a cycle's read or
   * fully follows its publish, so nothing is lost in between.
   */
  private async appendUnlocked(key: AgentSelfMemoryKey, content: string): Promise<void> {
    return this.withMutationLock(key, async () => {
      await this.ensureKey(key)
      await appendFile(this.pathFor(key), `\n${content}`, 'utf8')
    })
  }

  /**
   * Guard one whole-file read-modify-write cycle with an OS-level lock file
   * keyed per memory file. The in-process {@link withAppendLock} chain cannot
   * see other processes, so without this lock their interleaved cycles clobber
   * each other's rewrite via atomic rename.
   */
  private withMutationLock<T>(key: AgentSelfMemoryKey, operation: () => Promise<T>): Promise<T> {
    return withFileLock(`${this.pathFor(key)}.lock`, operation, {
      staleMs: MUTATION_LOCK_STALE_MS,
      waitMs: MUTATION_LOCK_WAIT_MS,
    })
  }

  private async writeUnlocked(key: AgentSelfMemoryKey, content: string): Promise<void> {
    await this.ensureKey(key)
    await atomicWrite(this.pathFor(key), content)
  }

  private async withAppendLock<T>(key: AgentSelfMemoryKey, operation: () => Promise<T>): Promise<T> {
    const previous = this.appendLocks.get(key) ?? Promise.resolve()
    let release: (() => void) | undefined
    const current = new Promise<void>(resolveLock => {
      release = resolveLock
    })
    this.appendLocks.set(key, current)
    await previous
    try {
      return await operation()
    } finally {
      release?.()
      if (this.appendLocks.get(key) === current) this.appendLocks.delete(key)
    }
  }
}

const AgentSelfMemoryKeyProjectContext: AgentSelfMemoryKey = 'project_context'
const MAX_SELF_MEMORY_CACHE_ENTRIES = 256
const memories = new Map<string, AgentSelfMemory>()
const selfMemoryLocks = new Map<string, Map<AgentSelfMemoryKey, Promise<void>>>()

/** Return a process-local per-agent self-memory instance. */
export function getAgentSelfMemory(agentId = 'default'): AgentSelfMemory {
  const normalized = normalizeAgentId(agentId)
  const existing = memories.get(normalized)
  if (existing) {
    // Refresh recency so the bounded cache below behaves as a simple LRU.
    memories.delete(normalized)
    memories.set(normalized, existing)
    return existing
  }
  const memory = new AgentSelfMemory({ agentId: normalized })
  memories.set(normalized, memory)
  if (memories.size > MAX_SELF_MEMORY_CACHE_ENTRIES) {
    const oldest = memories.keys().next()
    if (!oldest.done) memories.delete(oldest.value)
  }
  return memory
}

/** List existing self-memory agent directories without creating new ones. */
export async function listAgentSelfMemories(directory = join(xerxesHome(), 'agent_memory')): Promise<string[]> {
  try {
    const entries = await readdir(directory, { withFileTypes: true })
    return entries.filter(entry => entry.isDirectory()).map(entry => entry.name).sort((left, right) => left.localeCompare(right))
  } catch (error) {
    if (isMissing(error)) return []
    throw error
  }
}

export function clearAgentSelfMemoryCache(): void {
  memories.clear()
}

function normalizeAgentId(agentId: string): string {
  const normalized = agentId.trim()
  if (!normalized || normalized.includes('/') || normalized.includes('\\') || normalized.includes('..') || normalized.includes('\0')) {
    throw new ValidationError('agent_id', 'must be a safe non-empty identifier', agentId)
  }
  return normalized
}

function normalizeKey(key: AgentSelfMemoryKey | string): AgentSelfMemoryKey {
  if ((AGENT_SELF_MEMORY_KEYS as readonly string[]).includes(key)) return key as AgentSelfMemoryKey
  throw new ValidationError('key', 'must be one of ' + AGENT_SELF_MEMORY_KEYS.join(', '), key)
}

async function readProjectFile(root: string, name: string): Promise<string> {
  let current = root
  while (true) {
    try {
      return await readFile(join(current, name), 'utf8')
    } catch (error) {
      if (!isMissing(error)) throw error
    }
    const parent = dirname(current)
    if (parent === current) return ''
    current = parent
  }
}

async function atomicWrite(path: string, content: string): Promise<void> {
  await mkdir(dirname(path), { recursive: true })
  const temporary = join(dirname(path), '.' + crypto.randomUUID() + '.tmp')
  try {
    await Bun.write(temporary, content)
    await rename(temporary, path)
  } finally {
    await rm(temporary, { force: true })
  }
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && error.code === 'ENOENT'
}

function titleCase(value: string): string {
  return value.split(/\s+/).filter(Boolean).map(word => word[0]?.toUpperCase() + word.slice(1)).join(' ')
}
