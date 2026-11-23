// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Read-only bridge exposing Xerxes workspace customizations (`XERXES.md`,
 * `XERXES.local.md`, `.xerxes/skills`, `.xerxes/agents`, `.xerxes/commands`,
 * and `.xerxes/` notes) to external ACP runtimes such as Antigravity (`agy acp`).
 *
 * Strictly skips `AGENTS.md`, `GEMINI.md`, and `.agents/` because Antigravity
 * already discovers those natively. Performs zero writes to the user's repository.
 */

import { createHash } from 'node:crypto'
import { lstat, readdir, readFile, realpath } from 'node:fs/promises'
import type { Dirent } from 'node:fs'
import { homedir } from 'node:os'
import { basename, dirname, join, relative, resolve, sep } from 'node:path'

import { clipBootstrapContext } from '../runtime/bootstrap.js'
import { scanContextContent } from '../security/promptScanner.js'

const MAX_INSTRUCTION_FILE_BYTES = 16_000
const MAX_NOTE_FILE_BYTES = 4_000
const MAX_TOTAL_BRIDGE_BYTES = 28_000
const MAX_INDEX_ENTRIES = 64
const MAX_PARENT_HOPS = 10

const EXCLUDED_XERXES_NOTE_DIRS = new Set([
  'agents',
  'skills',
  'commands',
  'plugins',
  'review',
  'reports',
  'lessons',
  'memory',
  'node_modules',
])

export interface XerxesBridgeEntry {
  readonly name: string
  readonly description: string
  readonly path: string
  readonly scope: 'project' | 'global'
}

export interface XerxesBridgeCommand extends XerxesBridgeEntry {
  readonly body: string
}

export interface XerxesAcpWorkspaceBridge {
  readonly cwd: string
  readonly instructionFiles: readonly { readonly label: string; readonly path: string; readonly content: string }[]
  readonly notes: readonly { readonly relativePath: string; readonly path: string; readonly content: string }[]
  readonly skills: readonly XerxesBridgeEntry[]
  readonly agents: readonly XerxesBridgeEntry[]
  readonly commands: readonly XerxesBridgeCommand[]
  readonly firstTurnContextBlock: string
  readonly contextHash: string
}

export interface LoadXerxesWorkspaceBridgeOptions {
  readonly homeDir?: string
  readonly includeGlobal?: boolean
}

function parseFrontmatter(raw: string): {
  readonly name?: string
  readonly description?: string
  readonly body: string
} {
  const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(raw)
  if (!match) {
    return { body: raw.trim() }
  }
  const frontmatter = match[1] ?? ''
  const body = (match[2] ?? '').trim()
  let name: string | undefined
  let description: string | undefined
  const lines = frontmatter.split(/\r?\n/)
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i] ?? ''
    const field = /^([A-Za-z0-9_-]+):\s*(.*)$/.exec(line)
    if (!field) continue
    const key = field[1]!.toLowerCase()
    let val = (field[2] ?? '').trim()
    if (val === '>' || val === '>-' || val === '|' || val === '|-') {
      const folded: string[] = []
      while (i + 1 < lines.length && /^\s+/.test(lines[i + 1] ?? '')) {
        i += 1
        folded.push((lines[i] ?? '').trim())
      }
      val = folded.join(' ').trim()
    } else if ((val.startsWith('"') && val.endsWith('"')) || (val.startsWith("'") && val.endsWith("'"))) {
      val = val.slice(1, -1).trim()
    }
    if (key === 'name' && val) name = val
    if (key === 'description' && val) description = val
  }
  return {
    ...(name ? { name } : {}),
    ...(description ? { description } : {}),
    body,
  }
}

function firstMeaningfulLine(body: string): string {
  for (const line of body.split(/\r?\n/)) {
    const clean = line.replace(/^#+\s*/, '').trim()
    if (clean) return clean.slice(0, 140)
  }
  return ''
}

async function readSafeText(filePath: string): Promise<string | undefined> {
  try {
    const info = await lstat(filePath)
    if (!info.isFile() || info.isSymbolicLink()) {
      // Allow regular files and valid symlinks resolving to regular files
      const resolved = await realpath(filePath)
      const targetInfo = await lstat(resolved)
      if (!targetInfo.isFile()) return undefined
    }
    return await readFile(filePath, 'utf8')
  } catch {
    return undefined
  }
}

async function findProjectRoot(cwd: string, homeDir: string): Promise<string> {
  const start = resolve(cwd)
  const resolvedHome = resolve(homeDir)
  let current = start
  for (let hop = 0; hop < MAX_PARENT_HOPS; hop += 1) {
    // Never walk up into `$HOME` and mistake `~/.xerxes` for a project root.
    if (current === resolvedHome && start !== resolvedHome) {
      break
    }
    try {
      const entries = await readdir(current)
      if (entries.includes('.git') || entries.includes('.xerxes') || entries.includes('XERXES.md')) {
        return current
      }
    } catch {
      break
    }
    const parent = dirname(current)
    if (parent === current) break
    current = parent
  }
  return start
}

async function collectInstructionFiles(
  cwd: string,
  xerxesHome: string | undefined,
  homeDir: string,
): Promise<Array<{ readonly label: string; readonly path: string; readonly content: string }>> {
  const results: Array<{ readonly label: string; readonly path: string; readonly content: string }> = []
  const seen = new Set<string>()
  const start = resolve(cwd)
  const resolvedHome = resolve(homeDir)

  if (xerxesHome) {
    const globalPath = join(xerxesHome, 'XERXES.md')
    const content = await readSafeText(globalPath)
    if (content?.trim()) {
      const canonical = await realpath(globalPath).catch(() => globalPath)
      seen.add(canonical)
      const scanned = clipBootstrapContext(
        scanContextContent(content, 'Global XERXES.md'),
        MAX_INSTRUCTION_FILE_BYTES,
        'Global XERXES.md',
      )
      results.push({ label: 'Global XERXES.md', path: globalPath, content: scanned.trim() })
    }
  }

  for (const filename of ['XERXES.md', 'XERXES.local.md'] as const) {
    let current = start
    for (let hop = 0; hop < MAX_PARENT_HOPS; hop += 1) {
      if (current === resolvedHome && start !== resolvedHome) {
        break
      }
      const candidate = join(current, filename)
      const content = await readSafeText(candidate)
      if (content?.trim()) {
        const canonical = await realpath(candidate).catch(() => candidate)
        if (!seen.has(canonical)) {
          seen.add(canonical)
          const label = `Project ${filename}`
          const scanned = clipBootstrapContext(
            scanContextContent(content, `${label}: ${candidate}`),
            MAX_INSTRUCTION_FILE_BYTES,
            label,
          )
          results.push({ label, path: candidate, content: scanned.trim() })
        }
        break
      }
      const parent = dirname(current)
      if (parent === current) break
      current = parent
    }
  }

  return results
}

async function collectXerxesSkills(
  roots: readonly { readonly dir: string; readonly scope: 'project' | 'global' }[],
  nativeAgentSkillNames: ReadonlySet<string>,
): Promise<XerxesBridgeEntry[]> {
  const out: XerxesBridgeEntry[] = []
  const seen = new Set<string>()

  for (const root of roots) {
    let entries: Dirent[]
    try {
      entries = await readdir(root.dir, { withFileTypes: true })
    } catch {
      continue
    }
    entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
    for (const entry of entries) {
      if (out.length >= MAX_INDEX_ENTRIES) break
      if (!entry.isDirectory() && !entry.isSymbolicLink()) continue
      if (entry.name.startsWith('.')) continue
      const skillFile = join(root.dir, entry.name, 'SKILL.md')
      const raw = await readSafeText(skillFile)
      if (!raw?.trim()) continue
      const parsed = parseFrontmatter(raw)
      const name = (parsed.name ?? entry.name).trim()
      const key = name.toLowerCase()
      if (!name || seen.has(key) || nativeAgentSkillNames.has(key)) continue
      // Never leak Xerxes native Immortals swarm skills into external ACP runtimes.
      if (key.startsWith('immortals-')) continue
      seen.add(key)
      const desc = (parsed.description ?? firstMeaningfulLine(parsed.body)).slice(0, 200)
      out.push({
        name,
        description: scanContextContent(desc, `Xerxes skill ${name}`).trim(),
        path: skillFile,
        scope: root.scope,
      })
    }
  }
  return out
}

async function collectMarkdownEntries(
  roots: readonly { readonly dir: string; readonly scope: 'project' | 'global' }[],
  kindLabel: string,
): Promise<XerxesBridgeCommand[]> {
  const out: XerxesBridgeCommand[] = []
  const seen = new Set<string>()

  for (const root of roots) {
    let entries: Dirent[]
    try {
      entries = await readdir(root.dir, { withFileTypes: true })
    } catch {
      continue
    }
    entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
    for (const entry of entries) {
      if (out.length >= MAX_INDEX_ENTRIES) break
      if ((!entry.isFile() && !entry.isSymbolicLink()) || !entry.name.endsWith('.md')) continue
      if (entry.name.startsWith('.')) continue
      const filePath = join(root.dir, entry.name)
      const raw = await readSafeText(filePath)
      if (!raw?.trim()) continue
      const parsed = parseFrontmatter(raw)
      const stem = basename(entry.name, '.md')
      const name = (parsed.name ?? stem).trim()
      const key = name.toLowerCase()
      if (!name || seen.has(key)) continue
      seen.add(key)
      const desc = (parsed.description ?? firstMeaningfulLine(parsed.body)).slice(0, 200)
      out.push({
        name,
        description: scanContextContent(desc, `Xerxes ${kindLabel} ${name}`).trim(),
        path: filePath,
        scope: root.scope,
        body: parsed.body,
      })
    }
  }
  return out
}

async function collectXerxesProjectNotes(
  xerxesDir: string,
): Promise<Array<{ readonly relativePath: string; readonly path: string; readonly content: string }>> {
  const notes: Array<{ readonly relativePath: string; readonly path: string; readonly content: string }> = []
  let entries: Dirent[]
  try {
    entries = await readdir(xerxesDir, { withFileTypes: true })
  } catch {
    return notes
  }
  entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0))
  for (const entry of entries) {
    if (notes.length >= 12) break
    if (entry.name.startsWith('.')) continue
    if (entry.isDirectory()) {
      if (EXCLUDED_XERXES_NOTE_DIRS.has(entry.name)) continue
      continue
    }
    if (!entry.isFile() || !entry.name.endsWith('.md')) continue
    const filePath = join(xerxesDir, entry.name)
    const raw = await readSafeText(filePath)
    if (!raw?.trim()) continue
    const rel = relative(xerxesDir, filePath).split(sep).join('/')
    const bounded = clipBootstrapContext(
      scanContextContent(raw, `.xerxes/${rel}`),
      MAX_NOTE_FILE_BYTES,
      `.xerxes/${rel}`,
    )
    notes.push({ relativePath: rel, path: filePath, content: bounded.trim() })
  }
  return notes
}

async function listNativeAgentSkillNames(
  projectRoot: string,
  resolvedHome: string,
  includeGlobal: boolean,
): Promise<Set<string>> {
  const names = new Set<string>()
  const dirs = [
    join(projectRoot, '.agents', 'skills'),
    join(projectRoot, '.agent', 'skills'),
    ...(includeGlobal
      ? [
          join(resolvedHome, '.agents', 'skills'),
          join(resolvedHome, '.gemini', 'antigravity', 'skills'),
          join(resolvedHome, '.gemini', 'antigravity', 'builtin', 'skills'),
        ]
      : []),
  ]
  for (const dir of dirs) {
    let entries: Dirent[]
    try {
      entries = await readdir(dir, { withFileTypes: true })
    } catch {
      continue
    }
    for (const entry of entries) {
      if ((entry.isDirectory() || entry.isSymbolicLink()) && !entry.name.startsWith('.')) {
        names.add(entry.name.toLowerCase())
      }
    }
  }
  return names
}

/**
 * Load Xerxes-specific project (and optionally global) customizations to bridge
 * into an external ACP runtime such as Antigravity (`agy acp`). Returns `null`
 * when the workspace has no Xerxes-specific files so plain sessions stay untouched.
 */
export async function loadXerxesWorkspaceBridgeForAcp(
  cwd: string,
  options: LoadXerxesWorkspaceBridgeOptions = {},
): Promise<XerxesAcpWorkspaceBridge | null> {
  const cleanCwd = resolve(cwd)
  const includeGlobal = options.includeGlobal === true
  const resolvedHome = resolve(options.homeDir ?? homedir())
  const projectRoot = await findProjectRoot(cleanCwd, resolvedHome)
  const xerxesHome = includeGlobal ? join(resolvedHome, '.xerxes') : undefined

  const projectXerxesDirs: string[] = []
  const pushUniqueDir = (dir: string): void => {
    const resolvedDir = resolve(dir)
    if (!projectXerxesDirs.includes(resolvedDir)) projectXerxesDirs.push(resolvedDir)
  }
  pushUniqueDir(join(cleanCwd, '.xerxes'))
  pushUniqueDir(join(projectRoot, '.xerxes'))

  const nativeAgentSkills = await listNativeAgentSkillNames(projectRoot, resolvedHome, includeGlobal)

  const skillRoots = [
    ...projectXerxesDirs.map(dir => ({ dir: join(dir, 'skills'), scope: 'project' as const })),
    ...(xerxesHome ? [{ dir: join(xerxesHome, 'skills'), scope: 'global' as const }] : []),
  ]
  const agentRoots = [
    ...projectXerxesDirs.map(dir => ({ dir: join(dir, 'agents'), scope: 'project' as const })),
    ...(xerxesHome ? [{ dir: join(xerxesHome, 'agents'), scope: 'global' as const }] : []),
  ]
  const commandRoots = [
    ...projectXerxesDirs.map(dir => ({ dir: join(dir, 'commands'), scope: 'project' as const })),
    ...(xerxesHome ? [{ dir: join(xerxesHome, 'commands'), scope: 'global' as const }] : []),
  ]

  const [instructionFiles, notes, skills, agentsRaw, commands] = await Promise.all([
    collectInstructionFiles(cleanCwd, xerxesHome, resolvedHome),
    collectXerxesProjectNotes(join(projectRoot, '.xerxes')),
    collectXerxesSkills(skillRoots, nativeAgentSkills),
    collectMarkdownEntries(agentRoots, 'agent'),
    collectMarkdownEntries(commandRoots, 'command'),
  ])
  const agents: XerxesBridgeEntry[] = agentsRaw.map(({ name, description, path, scope }) => ({
    name,
    description,
    path,
    scope,
  }))

  if (
    instructionFiles.length === 0 &&
    notes.length === 0 &&
    skills.length === 0 &&
    agents.length === 0 &&
    commands.length === 0
  ) {
    return null
  }

  const sections: string[] = [
    '<xerxes_workspace_context>',
    'This workspace contains Xerxes-specific instructions and customizations that Antigravity does not auto-discover by default. Follow `XERXES.md` instructions alongside `AGENTS.md`. Skills, specialist agents, and commands below use progressive disclosure — read the listed file with `read_file` when relevant to the user’s task.',
  ]

  for (const file of instructionFiles) {
    sections.push('', `## ${file.label} (${file.path})`, file.content)
  }
  for (const note of notes) {
    sections.push('', `## .xerxes/${note.relativePath} (${note.path})`, note.content)
  }
  if (skills.length > 0) {
    sections.push('', '## Xerxes Skills (`.xerxes/skills`)')
    for (const skill of skills) {
      sections.push(`- ${skill.name} (${skill.path})${skill.description ? `: ${skill.description}` : ''}`)
    }
  }
  if (agents.length > 0) {
    sections.push('', '## Xerxes Specialist Agents (`.xerxes/agents`)')
    for (const agent of agents) {
      sections.push(`- ${agent.name} (${agent.path})${agent.description ? `: ${agent.description}` : ''}`)
    }
  }
  if (commands.length > 0) {
    sections.push('', '## Xerxes Slash Commands (`.xerxes/commands`)')
    for (const cmd of commands) {
      sections.push(`- /${cmd.name} (${cmd.path})${cmd.description ? `: ${cmd.description}` : ''}`)
    }
  }
  sections.push('</xerxes_workspace_context>')

  const firstTurnContextBlock = clipBootstrapContext(
    sections.join('\n'),
    MAX_TOTAL_BRIDGE_BYTES,
    'xerxes_workspace_context',
  )
  const contextHash = createHash('sha256').update(firstTurnContextBlock).digest('hex').slice(0, 16)

  return {
    cwd: cleanCwd,
    instructionFiles,
    notes,
    skills,
    agents,
    commands,
    firstTurnContextBlock,
    contextHash,
  }
}

/**
 * Expand a user slash invocation (`/<name> [args]`) against discovered `.xerxes/commands`
 * or `.xerxes/skills` when not already owned by the ACP runtime's native slash commands.
 */
export async function expandXerxesAcpSlashCommand(
  rawInput: string,
  bridge: XerxesAcpWorkspaceBridge | null,
  acpOwnedSlashNames?: ReadonlySet<string>,
): Promise<{ readonly expandedPrompt: string; readonly displayText: string; readonly matchedName: string } | null> {
  if (!bridge) return null
  const trimmed = rawInput.trim()
  if (!trimmed.startsWith('/')) return null
  const match = /^\/([A-Za-z0-9][A-Za-z0-9._-]*)(?:\s+([\s\S]*))?$/.exec(trimmed)
  if (!match) return null
  const name = match[1]!.toLowerCase()
  const argsText = (match[2] ?? '').trim()
  if (acpOwnedSlashNames?.has(name)) return null

  const cmd = bridge.commands.find(entry => entry.name.toLowerCase() === name)
  if (cmd) {
    const rendered = cmd.body.includes('$ARGUMENTS')
      ? cmd.body.replaceAll('$ARGUMENTS', argsText)
      : argsText
        ? `${cmd.body}\n\nUser arguments:\n${argsText}`
        : cmd.body
    const safeBody = scanContextContent(rendered, `Xerxes command /${cmd.name}`).trim()
    return {
      matchedName: cmd.name,
      displayText: trimmed,
      expandedPrompt: `[Xerxes command /${cmd.name} from ${cmd.path}]\n${safeBody}`,
    }
  }

  const skill = bridge.skills.find(entry => entry.name.toLowerCase() === name)
  if (skill) {
    const raw = await readSafeText(skill.path)
    if (!raw?.trim()) return null
    const parsed = parseFrontmatter(raw)
    const body = scanContextContent(parsed.body || raw, `Xerxes skill ${skill.name}`).trim()
    return {
      matchedName: skill.name,
      displayText: trimmed,
      expandedPrompt: `[Xerxes skill "${skill.name}" from ${skill.path}]\n${body}${argsText ? `\n\nTask:\n${argsText}` : ''}`,
    }
  }

  return null
}
