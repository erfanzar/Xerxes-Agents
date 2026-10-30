// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { createHash } from 'node:crypto'
import { constants } from 'node:fs'
import { lstat, mkdir, open, readdir, readFile, realpath, rename, rm } from 'node:fs/promises'
import { dirname, join, resolve } from 'node:path'
import { parseSkillMarkdown, skillInstructionsAreSafe } from '../extensions/skills.js'
import { containsCredential, scanContextContent } from '../security/promptScanner.js'
import type { LessonSharing } from './lessonSharing.js'
import { createRepoIdentityResolver, lessonGit, lessonGitRoot } from './repoIdentity.js'

/**
 * `/xerxes-sync`: carry what Xerxes learned in a Git project between machines through
 * `<git root>/.xerxes/`.
 *
 * - lessons: the existing repo-folder lesson sharing (`.xerxes/lessons/`);
 * - memory: project agent memory (`~/.xerxes/projects/<hash>/memory`, journals excluded)
 *   three-way merged with `.xerxes/memory/`;
 * - skills: skills learned here (applied reflection proposals) or named explicitly are
 *   published from `~/.xerxes/skills/<name>/` to `.xerxes/skills/<name>/`, which discovery
 *   already loads on every machine.
 *
 * The last synced digest of every file is kept outside the repository, so a change on one
 * side is told apart from a change on the other. Nothing is ever deleted, a deletion is not
 * undone, and a file changed on both sides is either block-merged (markdown without
 * frontmatter) or reported. Imports pass the prompt-injection scanner and the credential
 * check; exports never carry a credential pattern.
 */

export const PROJECT_SYNC_PARTS = ['lessons', 'memory', 'skills'] as const
export type ProjectSyncPart = (typeof PROJECT_SYNC_PARTS)[number]

const MAX_MEMORY_FILE_BYTES = 256 * 1024
const MAX_SKILL_FILE_BYTES = 512 * 1024
const MAX_SKILL_BUNDLE_FILES = 64
const MAX_FOLDER_FILES = 1000
const MAX_DEPTH = 6
const SKILL_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/u
const DIGEST = /^[0-9a-f]{64}$/u

export interface ProjectSyncSkipped { readonly path: string; readonly reason: string }
export interface ProjectMirrorResult {
  readonly exported: readonly string[]
  readonly imported: readonly string[]
  readonly merged: readonly string[]
  readonly unchanged: number
  readonly skipped: readonly ProjectSyncSkipped[]
}
export interface ProjectSkillSyncResult extends ProjectMirrorResult {
  /** Skills whose files were written to the project folder by this run. */
  readonly published: readonly string[]
  /** Every skill bundle present in `.xerxes/skills/`, from any machine. */
  readonly available: readonly string[]
}
export type ProjectLessonSync =
  | { readonly kind: 'synced'; readonly exported: number; readonly written: number; readonly applied: number; readonly retired: number; readonly skipped: readonly { readonly id: string; readonly reason: string }[] }
  | { readonly kind: 'preview'; readonly shareable: number }
  | { readonly kind: 'unavailable'; readonly error: string }
export interface ProjectSyncResult {
  readonly root: string
  readonly repo: string
  readonly dryRun: boolean
  readonly lessons?: ProjectLessonSync
  readonly memory?: ProjectMirrorResult
  readonly skills?: ProjectSkillSyncResult
  /** Project folders (`.xerxes/<part>/`) that Git ignores, so they would never be committed. */
  readonly ignored: readonly string[]
  /**
   * Synced folders with uncommitted changes according to `git status`, whether this run or
   * an earlier one made them. Absent when Git could not be asked.
   */
  readonly uncommitted?: readonly string[]
}
export interface ProjectSyncRequest {
  readonly parts?: readonly ProjectSyncPart[]
  /** Skill names in the user skills directory to publish into the project. */
  readonly publishSkills?: readonly string[]
  /** Report what would happen; write nothing, not even the sync state. */
  readonly dryRun?: boolean
}
export interface ProjectSyncOptions {
  /** The project agent-memory directory a session in this folder uses. */
  readonly memoryDirectory: (workspace: string) => string
  /** The user skill directory (`~/.xerxes/skills`). */
  readonly skillsDirectory: string
  /** Host-owned directory for the per-project sync state; never inside a repository. */
  readonly stateDirectory: string
  readonly sharing?: LessonSharing
  readonly resolveRoot?: (workspace: string) => Promise<string | undefined>
  readonly resolveRepo?: (workspace: string) => Promise<string | undefined>
  /** Skill names created or updated by approved reflection proposals in this workspace. */
  readonly learnedSkills?: (workspace: string) => Promise<readonly string[]>
  /** Record the exact published SKILL.md bytes as trusted on this machine. */
  readonly trustSkill?: (path: string, content: string) => Promise<void>
}

export class ProjectSyncError extends Error {
  constructor(readonly code: 'no_repository' | 'invalid_request' | 'unsafe_path', message: string) { super(message); this.name = 'ProjectSyncError' }
}

interface SyncState { version: 1; files: Record<string, string> }
interface MirrorBuilder { exported: string[]; imported: string[]; merged: string[]; unchanged: number; skipped: ProjectSyncSkipped[] }
type Decision =
  | { readonly kind: 'same' }
  | { readonly kind: 'export' }
  | { readonly kind: 'import' }
  | { readonly kind: 'merge'; readonly text: string }
  | { readonly kind: 'skip'; readonly reason: string }

export class ProjectSync {
  private queue: Promise<unknown> = Promise.resolve()
  private readonly defaultRepo = createRepoIdentityResolver()

  constructor(private readonly options: ProjectSyncOptions) {}

  /** One sync at a time per host: concurrent commands must not interleave writes or state. */
  sync(workspace: string, request: ProjectSyncRequest = {}): Promise<ProjectSyncResult> {
    const run = this.queue.then(() => this.run(workspace, request))
    this.queue = run.catch(() => {})
    return run
  }

  private async run(workspace: string, request: ProjectSyncRequest): Promise<ProjectSyncResult> {
    const parts = request.parts?.length ? [...new Set(request.parts)] : [...PROJECT_SYNC_PARTS]
    for (const part of parts) {
      if (!PROJECT_SYNC_PARTS.includes(part)) throw new ProjectSyncError('invalid_request', `Unknown sync part '${String(part)}'; use lessons, memory or skills.`)
    }
    const dryRun = request.dryRun === true
    const cwd = resolve(workspace)
    const canonical = await realpath(cwd)
    const root = await (this.options.resolveRoot ?? lessonGitRoot)(canonical)
    const repo = await (this.options.resolveRepo ?? this.defaultRepo)(canonical)
    if (!root || !repo) {
      throw new ProjectSyncError('no_repository', 'This folder is not in a Git repository with an identity (an origin URL or a first commit), so project sync is unavailable.')
    }
    const statePath = join(this.options.stateDirectory, `${digestText(`${root}\0${resolve(this.options.memoryDirectory(cwd))}`).slice(0, 24)}.json`)
    const state = await loadState(statePath)
    const lessons = parts.includes('lessons') ? await this.syncLessons(cwd, dryRun) : undefined
    const memory = parts.includes('memory') ? await this.syncMemory(root, cwd, state, dryRun) : undefined
    const skills = parts.includes('skills') ? await this.syncSkills(root, cwd, state, request.publishSkills ?? [], dryRun) : undefined
    if (!dryRun) await saveState(statePath, state)
    const ignored: string[] = []
    for (const part of parts) {
      const folder = `.xerxes/${part}/`
      if ((await lessonGit(root, ['check-ignore', '--quiet', folder]))?.code === 0) ignored.push(folder)
    }
    const uncommitted = await uncommittedFolders(root, parts)
    return {
      root, repo, dryRun, ignored,
      ...(uncommitted ? { uncommitted } : {}),
      ...(lessons ? { lessons } : {}),
      ...(memory ? { memory } : {}),
      ...(skills ? { skills } : {}),
    }
  }

  private async syncLessons(workspace: string, dryRun: boolean): Promise<ProjectLessonSync> {
    const sharing = this.options.sharing
    if (!sharing) return { kind: 'unavailable', error: 'Lesson sharing is not configured by this daemon host.' }
    try {
      if (dryRun) return { kind: 'preview', shareable: (await sharing.snapshot(workspace, { markShared: false })).lessons.length }
      const result = await sharing.share(workspace)
      return { kind: 'synced', exported: result.exported, written: result.written, applied: result.applied, retired: result.retired, skipped: result.skipped }
    } catch (error) { return { kind: 'unavailable', error: message(error) } }
  }

  private async syncMemory(root: string, cwd: string, state: SyncState, dryRun: boolean): Promise<ProjectMirrorResult> {
    const localDir = resolve(this.options.memoryDirectory(cwd))
    const repoDir = await repoFolder(root, 'memory')
    const out = mirror()
    const local = await listFiles(localDir, isMemoryFile, MAX_FOLDER_FILES, true)
    const remote = await listFiles(repoDir, isMemoryFile, MAX_FOLDER_FILES, false)
    if (local.truncated || remote.truncated) out.skipped.push({ path: 'memory', reason: `more than ${MAX_FOLDER_FILES} files; the rest were not synced` })
    for (const rel of [...new Set([...local.files, ...remote.files])].sort()) {
      const key = `memory/${rel}`
      try {
        const mine = await readBounded(join(localDir, ...rel.split('/')), MAX_MEMORY_FILE_BYTES)
        const theirs = await readBounded(join(repoDir, ...rel.split('/')), MAX_MEMORY_FILE_BYTES)
        const decision = decide(mine, theirs, state.files[key], true)
        switch (decision.kind) {
          case 'same':
            out.unchanged++
            if (!dryRun) state.files[key] = digest(mine!)
            break
          case 'skip':
            out.skipped.push({ path: key, reason: decision.reason })
            break
          case 'export': {
            if (containsCredential(mine!.toString('utf8'))) { out.skipped.push({ path: key, reason: 'contains a credential pattern; not exported' }); break }
            if (!dryRun) { await writeInside(repoDir, rel, mine!, 0o644, true); state.files[key] = digest(mine!) }
            out.exported.push(key)
            break
          }
          case 'import': {
            const refused = importRefusal(theirs!.toString('utf8'), rel)
            if (refused) { out.skipped.push({ path: key, reason: refused }); break }
            if (!dryRun) { await writeInside(localDir, rel, theirs!, 0o600, false); state.files[key] = digest(theirs!) }
            out.imported.push(key)
            break
          }
          case 'merge': {
            const refused = importRefusal(theirs!.toString('utf8'), rel) ?? (containsCredential(decision.text) ? 'contains a credential pattern; not merged' : undefined)
            if (refused) { out.skipped.push({ path: key, reason: refused }); break }
            const merged = Buffer.from(decision.text, 'utf8')
            if (!dryRun) {
              await writeInside(localDir, rel, merged, 0o600, false)
              await writeInside(repoDir, rel, merged, 0o644, true)
              state.files[key] = digest(merged)
            }
            out.merged.push(key)
            break
          }
        }
      } catch (error) { out.skipped.push({ path: key, reason: message(error) }) }
    }
    return out
  }

  private async syncSkills(root: string, cwd: string, state: SyncState, publish: readonly string[], dryRun: boolean): Promise<ProjectSkillSyncResult> {
    const repoSkills = await repoFolder(root, 'skills')
    const out = { ...mirror(), published: [] as string[] }
    const names = new Set<string>()
    for (const name of publish) {
      if (!SKILL_NAME.test(name)) throw new ProjectSyncError('invalid_request', `'${name}' is not a valid skill name.`)
      if (!await isRegularFile(join(this.options.skillsDirectory, name, 'SKILL.md'))) {
        throw new ProjectSyncError('invalid_request', `No skill named '${name}' in ${this.options.skillsDirectory}.`)
      }
      names.add(name)
    }
    for (const name of await this.options.learnedSkills?.(cwd) ?? []) {
      if (SKILL_NAME.test(name) && await isRegularFile(join(this.options.skillsDirectory, name, 'SKILL.md'))) names.add(name)
    }
    // Once published, a skill stays in step on later runs.
    for (const key of Object.keys(state.files)) {
      const match = /^skills\/([^/]+)\//u.exec(key)
      if (match) names.add(match[1]!)
    }
    for (const name of [...names].sort()) await this.syncSkill(name, repoSkills, state, out, dryRun)
    return { ...out, available: await projectSkillNames(repoSkills) }
  }

  /** A bundle is published whole or not at all; the project copy is never overwritten when it moved on elsewhere. */
  private async syncSkill(name: string, repoSkills: string, state: SyncState, out: MirrorBuilder & { published: string[] }, dryRun: boolean): Promise<void> {
    const label = `skills/${name}`
    const prefix = `${label}/`
    const localDir = join(this.options.skillsDirectory, name)
    const repoDir = join(repoSkills, name)
    try {
      const local = await listFiles(localDir, () => true, MAX_SKILL_BUNDLE_FILES, true)
      if (!local.files.includes('SKILL.md')) { out.skipped.push({ path: label, reason: 'no SKILL.md on this machine; the project copy is left as is' }); return }
      if (local.truncated) { out.skipped.push({ path: label, reason: `more than ${MAX_SKILL_BUNDLE_FILES} files; not published` }); return }
      const remote = await listFiles(repoDir, () => true, MAX_SKILL_BUNDLE_FILES, false)
      const exports: { rel: string; data: Buffer }[] = []
      const same: { rel: string; data: Buffer }[] = []
      for (const rel of [...new Set([...local.files, ...remote.files])].sort()) {
        const path = join(...rel.split('/'))
        const mine = await readBounded(join(localDir, path), MAX_SKILL_FILE_BYTES)
        const theirs = await readBounded(join(repoDir, path), MAX_SKILL_FILE_BYTES)
        const base = state.files[prefix + rel]
        const decision = decide(mine, theirs, base, false)
        if (decision.kind === 'same') same.push({ rel, data: mine! })
        else if (decision.kind === 'export') exports.push({ rel, data: mine! })
        else if (decision.kind === 'import' && mine === undefined && base === undefined) continue // added in the project by another machine
        else if (decision.kind === 'skip' && mine === undefined) continue // deleted here; never deleted from the project
        else {
          out.skipped.push({ path: label, reason: decision.kind === 'import'
            ? `the project copy of ${rel} changed on another machine and already takes precedence here; copy it into ${localDir} to keep both in step`
            : `${rel} changed on this machine and in the project; resolve by hand` })
          return
        }
      }
      const skillText = (exports.find(file => file.rel === 'SKILL.md') ?? same.find(file => file.rel === 'SKILL.md'))?.data.toString('utf8')
      if (skillText === undefined) { out.skipped.push({ path: label, reason: 'SKILL.md is unreadable' }); return }
      if (!skillInstructionsAreSafe(parseSkillMarkdown(skillText, join(localDir, 'SKILL.md')))) {
        out.skipped.push({ path: label, reason: 'instructions were flagged by the prompt-injection scanner; not published' }); return
      }
      const leaking = exports.find(file => containsCredential(file.data.toString('utf8')))
      if (leaking) { out.skipped.push({ path: label, reason: `${leaking.rel} contains a credential pattern; not published` }); return }
      if (!exports.length) {
        out.unchanged++
        if (!dryRun) for (const file of same) state.files[prefix + file.rel] = digest(file.data)
        return
      }
      // SKILL.md last: a half-written bundle never has instructions pointing at missing files.
      exports.sort((a, b) => Number(a.rel === 'SKILL.md') - Number(b.rel === 'SKILL.md'))
      if (!dryRun) {
        for (const file of exports) await writeInside(repoDir, file.rel, file.data, 0o644, true, repoSkills)
        for (const file of [...exports, ...same]) state.files[prefix + file.rel] = digest(file.data)
        // The project copy shadows the user copy in discovery; keep this machine's trust in the exact bytes it published.
        await this.options.trustSkill?.(join(repoDir, 'SKILL.md'), skillText)
      }
      out.exported.push(...exports.map(file => prefix + file.rel))
      out.published.push(name)
    } catch (error) { out.skipped.push({ path: label, reason: message(error) }) }
  }
}

export function renderProjectSync(result: ProjectSyncResult): string {
  const lines = [`${result.dryRun ? 'Project sync preview (nothing written)' : 'Project sync'} · ${result.root}`]
  const lessons = result.lessons
  if (lessons?.kind === 'synced') {
    lines.push(`Lessons: ${lessons.exported} in .xerxes/lessons (${lessons.written} new or changed), ${lessons.applied} imported, ${lessons.retired} retired, ${lessons.skipped.length} skipped.`)
    for (const [reason, ids] of groupSkippedLessons(lessons.skipped)) {
      lines.push(`  ${ids.length} skipped: ${describeLessonSkip(reason)} (${ids.slice(0, 3).join(', ')}${ids.length > 3 ? `, … ${ids.length - 3} more` : ''})`)
    }
  } else if (lessons?.kind === 'preview') lines.push(`Lessons: ${lessons.shareable} approved lesson(s) would be shared.`)
  else if (lessons) lines.push(`Lessons: unavailable — ${lessons.error}`)
  const verb = result.dryRun ? 'would be ' : ''
  if (result.memory) {
    const memory = result.memory
    lines.push(`Memory: ${memory.exported.length} ${verb}exported, ${memory.imported.length} ${verb}imported, ${memory.merged.length} ${verb}merged, ${memory.unchanged} unchanged.`)
    for (const path of [...memory.exported, ...memory.imported, ...memory.merged].slice(0, 20)) lines.push(`  ${path}`)
  }
  if (result.skills) {
    const skills = result.skills
    lines.push(`Skills: ${skills.published.length ? `${verb}published ${skills.published.join(', ')}` : 'nothing new to publish'}; ${skills.unchanged} unchanged.`)
    if (skills.available.length) lines.push(`  In this project: ${skills.available.join(', ')}. They load on every machine; /skills trust <name> lets one run commands there.`)
  }
  const skipped = [...result.memory?.skipped ?? [], ...result.skills?.skipped ?? []]
  if (skipped.length) {
    lines.push('Skipped:')
    for (const row of skipped.slice(0, 20)) lines.push(`  ${row.path}: ${row.reason}`)
    if (skipped.length > 20) lines.push(`  … ${skipped.length - 20} more`)
  }
  if (result.ignored.length) {
    lines.push(`Warning: Git ignores ${result.ignored.join(', ')}, so it would never be committed.`,
      'If .gitignore ignores all of .xerxes/, replace that line with `.xerxes/*` followed by `!.xerxes/lessons/`, `!.xerxes/memory/` and `!.xerxes/skills/`.')
  }
  if (!result.dryRun) {
    // Git is the authority: an earlier run can have left files uncommitted even when this one wrote nothing.
    const changed = result.uncommitted ?? [
      ...(lessons?.kind === 'synced' && lessons.written > 0 ? ['.xerxes/lessons'] : []),
      ...(result.memory && (result.memory.exported.length || result.memory.merged.length) ? ['.xerxes/memory'] : []),
      ...(result.skills && result.skills.exported.length ? ['.xerxes/skills'] : []),
    ]
    lines.push(changed.length
      ? `Uncommitted changes in ${changed.join(', ')}: commit them; on another machine, pull and Xerxes loads them on the next message.`
      : result.uncommitted ? 'Everything in the synced .xerxes folders is already committed.' : 'Nothing new was written to the project folder, so there is nothing to commit.')
  }
  return lines.join('\n')
}

/** `.xerxes/<part>` folders with staged, unstaged or untracked changes; undefined when Git cannot answer. */
async function uncommittedFolders(root: string, parts: readonly ProjectSyncPart[]): Promise<string[] | undefined> {
  const folders = parts.map(part => `.xerxes/${part}`)
  const status = await lessonGit(root, ['status', '--porcelain', '--untracked-files=normal', '--', ...folders])
  if (!status || status.code !== 0) return undefined
  const lines = status.text.split('\n').filter(Boolean)
  return folders.filter(folder => lines.some(line => line.slice(3).replace(/^"/u, '').startsWith(`${folder}/`) || line.includes(` -> ${folder}/`)))
}

function groupSkippedLessons(skipped: readonly { readonly id: string; readonly reason: string }[]): Map<string, string[]> {
  const groups = new Map<string, string[]>()
  for (const row of skipped) groups.set(row.reason, [...groups.get(row.reason) ?? [], row.id])
  return groups
}

function describeLessonSkip(reason: string): string {
  switch (reason) {
    case 'conflict': return 'same id as a lesson on this machine but different content; neither copy was changed'
    case 'repo_mismatch': return 'belongs to a different repository'
    case 'scanner_modified': return 'the prompt-injection scanner flagged its text'
    default: return reason
  }
}

function mirror(): MirrorBuilder { return { exported: [], imported: [], merged: [], unchanged: 0, skipped: [] } }

/** Three-way decision against the digest both sides had at the last sync. */
function decide(local: Buffer | undefined, repo: Buffer | undefined, base: string | undefined, mergeable: boolean): Decision {
  const mine = local === undefined ? undefined : digest(local)
  const theirs = repo === undefined ? undefined : digest(repo)
  if (mine === theirs) return { kind: 'same' }
  if (mine === undefined) return theirs === base ? { kind: 'skip', reason: 'deleted on this machine; not restored' } : { kind: 'import' }
  if (theirs === undefined) return mine === base ? { kind: 'skip', reason: 'deleted from the project folder; not restored' } : { kind: 'export' }
  if (mine === base) return { kind: 'import' }
  if (theirs === base) return { kind: 'export' }
  if (mergeable) {
    const text = unionBlocks(local!.toString('utf8'), repo!.toString('utf8'))
    if (text !== undefined) return { kind: 'merge', text }
  }
  return { kind: 'skip', reason: 'changed on this machine and in the project folder; resolve by hand' }
}

/**
 * Memory notes are appended as blank-line separated entries, so two machines that each
 * appended converge on the union: local entries in order, then entries only the project had.
 * Files with frontmatter are documents, not logs, and are never merged automatically.
 */
export function unionBlocks(local: string, repo: string): string | undefined {
  if (/^---\r?\n/u.test(local) || /^---\r?\n/u.test(repo)) return undefined
  const blocks = (text: string) => text.split(/\r?\n[ \t]*\r?\n/u).map(block => block.trim()).filter(Boolean)
  const mine = blocks(local)
  const seen = new Set(mine)
  return [...mine, ...blocks(repo).filter(block => !seen.has(block))].join('\n\n') + '\n'
}

function importRefusal(text: string, rel: string): string | undefined {
  if (text.includes('\0')) return 'not a text file; not imported'
  if (scanContextContent(text, rel) !== text) return 'flagged by the prompt-injection scanner; not imported'
  if (containsCredential(text)) return 'contains a credential pattern; not imported'
  return undefined
}

function isMemoryFile(rel: string): boolean { return rel.endsWith('.md') && !rel.startsWith('journal/') }

/** `<root>/.xerxes/<part>` must not lead out of the repository through a symlink. */
async function repoFolder(root: string, part: string): Promise<string> {
  const path = join(root, '.xerxes', part)
  for (const candidate of [join(root, '.xerxes'), path]) {
    try {
      if (await realpath(candidate) !== candidate) throw new ProjectSyncError('unsafe_path', `${candidate} must be a real directory inside the repository, not a symlink.`)
    } catch (error) { if (!missing(error)) throw error }
  }
  return path
}

async function listFiles(directory: string, accept: (rel: string) => boolean, limit: number, allowLinkedRoot: boolean): Promise<{ files: string[]; truncated: boolean }> {
  const files: string[] = []
  let truncated = false
  try {
    const info = await lstat(directory)
    if (info.isSymbolicLink() && !allowLinkedRoot) throw new ProjectSyncError('unsafe_path', `${directory} is a symlink; refusing to follow it.`)
  } catch (error) { if (missing(error)) return { files, truncated }; throw error }
  const walk = async (current: string, prefix: string, depth: number): Promise<void> => {
    const entries = (await readdir(current, { withFileTypes: true })).sort((a, b) => a.name.localeCompare(b.name))
    for (const entry of entries) {
      if (truncated) return
      if (entry.name.startsWith('.')) continue
      const rel = prefix ? `${prefix}/${entry.name}` : entry.name
      if (entry.isDirectory()) { if (depth < MAX_DEPTH) await walk(join(current, entry.name), rel, depth + 1); continue }
      // Symlinks report neither directory nor file here, so they are never followed.
      if (!entry.isFile() || !accept(rel)) continue
      if (files.length >= limit) { truncated = true; return }
      files.push(rel)
    }
  }
  await walk(directory, '', 0)
  return { files, truncated }
}

async function readBounded(path: string, max: number): Promise<Buffer | undefined> {
  let handle
  try { handle = await open(path, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK) }
  catch (error) { if (missing(error)) return undefined; throw error }
  try {
    const info = await handle.stat()
    if (!info.isFile()) throw new Error('not a regular file')
    if (info.size > max) throw new Error(`larger than ${max} bytes`)
    const data = await handle.readFile()
    if (data.byteLength > max) throw new Error(`larger than ${max} bytes`)
    return data
  } finally { await handle.close() }
}

/**
 * Atomic replacement below `base`. With `strict`, `anchor` (default `base`) must already be its
 * own real path, so a symlinked project folder can never redirect a write out of the repository.
 */
async function writeInside(base: string, rel: string, data: Uint8Array, mode: number, strict: boolean, anchor = base): Promise<void> {
  const segments = rel.split('/')
  if (segments.some(segment => !segment || segment === '.' || segment === '..')) throw new ProjectSyncError('unsafe_path', `Invalid relative path ${rel}.`)
  const target = join(base, ...segments)
  const directory = dirname(target)
  await mkdir(directory, { recursive: true, mode: strict ? 0o755 : 0o700 })
  if (strict && await realpath(anchor) !== anchor) throw new ProjectSyncError('unsafe_path', `${anchor} must be a real directory inside the repository.`)
  if (await realpath(directory) !== join(await realpath(base), ...segments.slice(0, -1))) {
    throw new ProjectSyncError('unsafe_path', `Refusing to write through a symlinked directory: ${directory}`)
  }
  try {
    if (!(await lstat(target)).isFile()) throw new ProjectSyncError('unsafe_path', `${target} is not a regular file; not replaced.`)
  } catch (error) { if (!missing(error)) throw error }
  const temporary = join(directory, `.xerxes-sync-${crypto.randomUUID()}.tmp`)
  const handle = await open(temporary, 'wx', mode)
  try {
    try { await handle.writeFile(data); await handle.sync() } finally { await handle.close() }
    await rename(temporary, target)
  } catch (error) { await rm(temporary, { force: true }); throw error }
}

async function projectSkillNames(directory: string): Promise<string[]> {
  let entries
  try { entries = await readdir(directory, { withFileTypes: true }) } catch (error) { if (missing(error)) return []; throw error }
  const names: string[] = []
  for (const entry of entries) {
    if (entry.isDirectory() && !entry.name.startsWith('.') && await isRegularFile(join(directory, entry.name, 'SKILL.md'))) names.push(entry.name)
  }
  return names.sort()
}

async function isRegularFile(path: string): Promise<boolean> {
  try { return (await lstat(path)).isFile() } catch (error) { if (missing(error)) return false; throw error }
}

async function loadState(path: string): Promise<SyncState> {
  let text: string
  try { text = await readFile(path, 'utf8') } catch (error) { if (missing(error)) return { version: 1, files: {} }; throw error }
  let value: unknown
  try { value = JSON.parse(text) } catch { value = undefined }
  if (!record(value) || value.version !== 1 || !record(value.files) || Object.values(value.files).some(hash => typeof hash !== 'string' || !DIGEST.test(hash))) {
    throw new ProjectSyncError('invalid_request', `Project sync state ${path} is invalid; move it aside to start over.`)
  }
  return { version: 1, files: { ...value.files as Record<string, string> } }
}

async function saveState(path: string, state: SyncState): Promise<void> {
  const sorted = Object.fromEntries(Object.entries(state.files).sort(([a], [b]) => a.localeCompare(b)))
  await mkdir(dirname(path), { recursive: true, mode: 0o700 })
  const temporary = `${path}.${crypto.randomUUID()}.tmp`
  const handle = await open(temporary, 'wx', 0o600)
  try {
    try { await handle.writeFile(JSON.stringify({ version: 1, files: sorted }, null, 2) + '\n'); await handle.sync() } finally { await handle.close() }
    await rename(temporary, path)
  } catch (error) { await rm(temporary, { force: true }); throw error }
}

function digest(data: Uint8Array): string { return createHash('sha256').update(data).digest('hex') }
function digestText(text: string): string { return createHash('sha256').update(text, 'utf8').digest('hex') }
function record(value: unknown): value is Record<string, unknown> { return typeof value === 'object' && value !== null && !Array.isArray(value) }
function missing(error: unknown): boolean { return error instanceof Error && 'code' in error && error.code === 'ENOENT' }
function message(error: unknown): string { return error instanceof Error ? error.message : String(error) }
