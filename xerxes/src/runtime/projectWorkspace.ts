// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lstat, readdir, readFile, realpath } from 'node:fs/promises'
import type { Dirent } from 'node:fs'
import { homedir } from 'node:os'
import { isAbsolute, join, relative, resolve, sep } from 'node:path'

import { scanContextContent } from '../security/promptScanner.js'

export const PROJECT_AGENTS_DIR = '.agents'

/** Fixed project-owned files rendered first, in declaration order. */
export const PROJECT_AGENT_CONTEXT_FILES = [
  'AGENTS.md',
  'SKILL_MAP.md',
  'ops/OPS.md',
  'projects/README.md',
] as const

/** Every Markdown file under `.agents/` is injected; these subtrees stay out. */
const SKILLS_DIRECTORY = 'skills'
const MAX_DISCOVERED_CONTEXT_FILES = 200
const MAX_DISCOVERY_DEPTH = 8

export interface ProjectAgentWorkspaceContextOptions {
  readonly agentsDir: string
  readonly loadedFiles: readonly string[]
  readonly prompt: string
  readonly root: string
}

/** Rendered, prompt-safe project-local `.agents` operating context. */
export class ProjectAgentWorkspaceContext {
  readonly agentsDir: string
  readonly loadedFiles: readonly string[]
  readonly prompt: string
  readonly root: string

  constructor(options: ProjectAgentWorkspaceContextOptions) {
    this.root = options.root
    this.agentsDir = options.agentsDir
    this.prompt = options.prompt
    this.loadedFiles = Object.freeze([...options.loadedFiles])
    Object.freeze(this)
  }
}

export interface LoadProjectAgentWorkspaceOptions {
  /** UTF-8 byte ceiling for each injected file body. Defaults to 6,000. */
  readonly maxBytesPerFile?: number
}

/** Return the lexical `<projectRoot>/.agents` location without performing I/O. */
export function projectAgentsDir(projectRoot: string): string {
  return join(normalizeProjectRoot(projectRoot), PROJECT_AGENTS_DIR)
}

/** Return the lexical `<projectRoot>/.agents/skills` location without performing I/O. */
export function projectAgentSkillsDir(projectRoot: string): string {
  return join(projectAgentsDir(projectRoot), 'skills')
}

/**
 * Load compact, project-owned agent context in fixed declaration order.
 *
 * Missing, unreadable, scanned, or containment-escaping files are skipped.
 * Candidate files must resolve inside the project-owned `.agents` directory;
 * this prevents a symlinked runbook from injecting arbitrary host content.
 */
export async function loadProjectAgentWorkspace(
  projectRoot: string,
  options: LoadProjectAgentWorkspaceOptions = {},
): Promise<ProjectAgentWorkspaceContext> {
  const maximum = nonNegativeInteger(options.maxBytesPerFile ?? 6_000, 'maxBytesPerFile')
  const lexicalRoot = normalizeProjectRoot(projectRoot)
  const root = await canonicalDirectoryOrLexical(lexicalRoot)
  const agentsDir = join(root, PROJECT_AGENTS_DIR)
  const canonicalAgentsDir = await containedDirectory(root, agentsDir)
  if (canonicalAgentsDir === undefined) {
    return new ProjectAgentWorkspaceContext({ root, agentsDir, prompt: '', loadedFiles: [] })
  }

  const parts = [
    '# Project Agent Workspace',
    `Directory: ${agentsDir}`,
    '',
    'This is project-owned agent operating context. Every Markdown file under `.agents/` is included below.',
    'Repository-local skills under `.agents/skills/` are discovered separately and load on demand.',
    'Files clipped with a truncation note can be read directly with normal file tools.',
  ]
  const loadedFiles: string[] = []

  for (const candidate of await orderedContextFiles(agentsDir, canonicalAgentsDir)) {
    const relativePath = relative(agentsDir, candidate).split(sep).join('/')
    const content = await loadContextFile(root, canonicalAgentsDir, candidate, maximum)
    if (content === undefined) continue
    if (!content.trim()) continue
    const scanned = scanContextContent(content, `Project agent workspace: ${candidate}`)
    if (scanned.includes('[BLOCKED:')) continue
    loadedFiles.push(candidate)
    parts.push('', `## .agents/${relativePath}`, scanned.trim())
  }

  return new ProjectAgentWorkspaceContext({
    root,
    agentsDir,
    prompt: loadedFiles.length ? parts.join('\n').trim() : '',
    loadedFiles,
  })
}

async function canonicalDirectoryOrLexical(path: string): Promise<string> {
  try {
    const canonical = await realpath(path)
    const metadata = await lstat(canonical)
    return metadata.isDirectory() ? canonical : path
  } catch {
    return path
  }
}

/** Priority files in declaration order, then every other Markdown file under .agents sorted by path. */
async function orderedContextFiles(agentsDir: string, canonicalAgentsDir: string): Promise<string[]> {
  const priority = PROJECT_AGENT_CONTEXT_FILES.map(path => join(agentsDir, ...path.split('/')))
  const priorityPaths = new Set(priority.map(path => resolve(path)))
  const discovered = await discoverMarkdownFiles(canonicalAgentsDir)
  const rest = discovered
    .filter(path => !priorityPaths.has(resolve(path)))
    .sort((left, right) => left.localeCompare(right))
  return [...priority, ...rest]
}

/**
 * Recursively enumerate Markdown files under the canonical `.agents` directory.
 * The skills subtree stays on-demand via the skill registry, hidden and
 * dependency directories are skipped, and discovery is bounded so a hostile
 * or runaway tree cannot stall bootstrap. Symlinked entries are ignored here;
 * per-file containment is revalidated by loadContextFile.
 */
async function discoverMarkdownFiles(agentsDir: string): Promise<string[]> {
  const discovered: string[] = []

  async function walk(directory: string, depth: number): Promise<void> {
    if (depth > MAX_DISCOVERY_DEPTH || discovered.length >= MAX_DISCOVERED_CONTEXT_FILES) return
    let entries: Dirent[]
    try {
      entries = await readdir(directory, { withFileTypes: true })
    } catch {
      return
    }
    for (const entry of entries) {
      if (discovered.length >= MAX_DISCOVERED_CONTEXT_FILES) return
      const candidate = join(directory, entry.name)
      if (entry.isDirectory()) {
        if (entry.name === SKILLS_DIRECTORY && depth === 0) continue
        if (entry.name.startsWith('.') || entry.name === 'node_modules') continue
        await walk(candidate, depth + 1)
        continue
      }
      if (entry.isFile() && entry.name.endsWith('.md')) {
        discovered.push(candidate)
      }
    }
  }

  await walk(agentsDir, 0)
  return discovered
}

async function containedDirectory(root: string, candidate: string): Promise<string | undefined> {
  try {
    const canonical = await realpath(candidate)
    const metadata = await lstat(canonical)
    if (!metadata.isDirectory() || !isWithin(root, canonical)) return undefined
    return canonical
  } catch {
    return undefined
  }
}

async function loadContextFile(
  root: string,
  agentsDir: string,
  candidate: string,
  maximum: number,
): Promise<string | undefined> {
  if (!isWithin(agentsDir, candidate)) return undefined
  try {
    const metadata = await lstat(candidate)
    if (!metadata.isFile()) return undefined
    const canonical = await realpath(candidate)
    if (!isWithin(root, canonical) || !isWithin(agentsDir, canonical)) return undefined
    return clipUtf8(await readFile(canonical, 'utf8'), maximum)
  } catch {
    return undefined
  }
}

function clipUtf8(content: string, maximum: number): string {
  if (Buffer.byteLength(content, 'utf8') <= maximum) return content
  let used = 0
  let clipped = ''
  for (const character of content) {
    const size = Buffer.byteLength(character, 'utf8')
    if (used + size > maximum) break
    clipped += character
    used += size
  }
  return clipped + '\n\n[truncated: read this file directly for the rest]'
}

function normalizeProjectRoot(projectRoot: string): string {
  if (typeof projectRoot !== 'string' || !projectRoot.trim()) {
    throw new TypeError('projectRoot must be a non-empty string')
  }
  if (projectRoot.includes('\0')) {
    throw new TypeError('projectRoot must not contain a null byte')
  }
  return resolve(expandHome(projectRoot.trim()))
}

function expandHome(path: string): string {
  if (path === '~') return homedir()
  if (path.startsWith('~/') || path.startsWith('~\\')) return join(homedir(), path.slice(2))
  return path
}

function isWithin(root: string, candidate: string): boolean {
  const pathFromRoot = relative(root, candidate)
  return pathFromRoot === ''
    || (!pathFromRoot.startsWith(`..${sep}`) && pathFromRoot !== '..' && !isAbsolute(pathFromRoot))
}

function nonNegativeInteger(value: number, name: string): number {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new RangeError(name + ' must be a non-negative safe integer')
  }
  return value
}
