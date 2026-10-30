// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { cp, mkdir, mkdtemp, readFile, readdir, realpath, rm, writeFile } from 'node:fs/promises'
import { connect } from 'node:net'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { DaemonServer } from '../src/daemon/server.js'
import { recordTrustedSkillContent } from '../src/extensions/skillsGuard.js'
import { GovernedMemoryStore } from '../src/memory/governedMemoryStore.js'
import { LessonSharing } from '../src/memory/lessonSharing.js'
import { LessonStore, renderApprovedLessonsForTurn } from '../src/memory/lessons.js'
import { ProjectSync } from '../src/memory/projectSync.js'
import { createRepoIdentityResolver } from '../src/memory/repoIdentity.js'
import { ReflectionProposalStore } from '../src/runtime/reflection.js'

const SKILL = 'sync-e2e-probe'

async function git(cwd: string, ...args: string[]): Promise<void> {
  const process = Bun.spawn(['git', '-C', cwd, ...args], { stdout: 'ignore', stderr: 'pipe' })
  expect(await process.exited).toBe(0)
}

interface Machine {
  readonly home: string; readonly clone: string; readonly memory: string; readonly skills: string
  readonly lessons: LessonStore; readonly sharing: LessonSharing; readonly server: DaemonServer
  readonly request: (method: string, params: Record<string, unknown>) => Promise<Record<string, unknown>>
  close(): Promise<void>
}

/** One "machine": its own stores, its own clone, a real daemon over a real socket. */
async function machine(base: string, name: string, origin: string): Promise<Machine> {
  const home = join(base, name)
  const clone = join(home, 'work', 'project')
  await mkdir(clone, { recursive: true })
  await git(clone, 'init', '--quiet')
  await git(clone, 'remote', 'add', 'origin', origin)
  const memory = join(home, 'project-memory'); const skills = join(home, 'skills')
  const governed = new GovernedMemoryStore({ directory: join(home, 'governed-memory') })
  const resolveRepo = createRepoIdentityResolver()
  const lessons = new LessonStore({ memory: governed, resolveRepo })
  const sharing = new LessonSharing({ memory: governed, lessons, report: () => {} })
  const store = new ReflectionProposalStore({ directory: join(home, 'reflection') })
  const projectSync = new ProjectSync({
    memoryDirectory: () => memory, skillsDirectory: skills, stateDirectory: join(home, 'project-sync'), sharing, resolveRepo,
    trustSkill: (path, content) => recordTrustedSkillContent(path, content, { skillsDirectory: skills }),
  })
  const runtime = new InMemoryDaemonRuntime(undefined, { model: 'fixture', currentProjectDirectory: clone, sessionDirectory: join(home, 'sessions') })
  const server = new DaemonServer({ socketPath: join(home, 'daemon.sock'), exitLogPath: join(home, 'exits.jsonl'), autoTitle: false,
    skillDirectory: skills, runtime,
    learning: { store, lessons, sharing, projectSync, applier: { applySkill: () => ({ ok: false }), applyAgent: () => ({ ok: false }) } } })
  await server.start()
  const socket = connect({ path: join(home, 'daemon.sock') })
  await new Promise<void>((resolve, reject) => { socket.once('connect', resolve); socket.once('error', reject) })
  socket.setEncoding('utf8')
  let buffer = ''; let id = 0
  const pending = new Map<number, (value: Record<string, unknown>) => void>()
  socket.on('data', chunk => {
    buffer += String(chunk)
    let newline: number
    while ((newline = buffer.indexOf('\n')) >= 0) {
      const frame = JSON.parse(buffer.slice(0, newline)) as { id?: number; result?: Record<string, unknown>; error?: unknown }
      buffer = buffer.slice(newline + 1)
      if (frame.id !== undefined) { pending.get(frame.id)?.(frame.result ?? { rpc_error: frame.error }); pending.delete(frame.id) }
    }
  })
  const request = (method: string, params: Record<string, unknown>) => new Promise<Record<string, unknown>>((resolve, reject) => {
    const current = ++id
    const timer = setTimeout(() => { pending.delete(current); reject(new Error(`Timed out: ${method}`)) }, 10_000)
    pending.set(current, value => { clearTimeout(timer); resolve(value) })
    socket.write(JSON.stringify({ jsonrpc: '2.0', id: current, method, params }) + '\n')
  })
  await request('initialize', { session_key: name, project_dir: clone })
  return { home, clone, memory, skills, lessons, sharing, server, request,
    async close() { socket.destroy(); await server.stop(); sharing.close() } }
}

function skillsNamed(result: Record<string, unknown>, name: string): { name: string; source: string }[] {
  return (result.skills as { name: string; source: string }[]).filter(skill => skill.name === name)
}

test('/xerxes-sync carries lessons, memory and skills between two machines without duplicates', async () => {
  const base = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-daemon-project-sync-')))
  // The same repository spelled two ways, as SSH and HTTPS clones would be.
  const a = await machine(base, 'a', 'git@github.com:fixture/project-sync.git')
  const b = await machine(base, 'b', 'https://github.com/fixture/project-sync')
  try {
    // What machine A learned.
    const lesson = await a.lessons.record({ workspace: a.clone, cue: 'tpu lock busy', lesson: 'Check /tmp/libtpu_lockfile owners before retrying.', agentId: 'agent-a', confidence: 0.9,
      evidence: { kind: 'test', command: 'bun test tpu.test.ts', exitCode: 0, summary: '3 pass' } })
    await a.lessons.review(lesson.id, 'approved')
    await mkdir(join(a.memory, 'topics'), { recursive: true })
    await writeFile(join(a.memory, 'topics', 'tpu.md'), '---\nname: tpu\ndescription: TPU host notes\n---\nUse the v5p queue.\n')
    await mkdir(join(a.memory, 'journal'), { recursive: true })
    await writeFile(join(a.memory, 'journal', '2026-10-30.md'), '- private log\n')
    await mkdir(join(a.skills, SKILL), { recursive: true })
    await writeFile(join(a.skills, SKILL, 'SKILL.md'), `---\nname: ${SKILL}\ndescription: Probe skill for project sync\n---\n\nRun the focused test first.\n`)

    const pushed = await a.request('slash', { command: `/xerxes-sync skill ${SKILL}` })
    expect(pushed).toMatchObject({ ok: true, ignored: [], lessons: { kind: 'synced', exported: 1 }, memory: { exported: ['memory/topics/tpu.md'] }, skills: { published: [SKILL] } })
    expect(await readdir(join(a.clone, '.xerxes', 'lessons'))).toEqual([`${lesson.id}.json`])
    expect(await readdir(join(a.clone, '.xerxes', 'memory'))).toEqual(['topics'])
    // Machine A still loads the skill exactly once, now from the project, with command trust kept.
    const listedA = skillsNamed(await a.request('slash', { command: '/skills' }), SKILL)
    expect(listedA).toEqual([expect.objectContaining({ source: join(a.clone, '.xerxes', 'skills', SKILL, 'SKILL.md') })])
    const lintA = await a.request('slash', { command: '/skills lint' })
    const lintAboutSkill = (lintA.findings as { code: string; message: string }[]).filter(finding => finding.message.includes(`'${SKILL}'`))
    expect(lintAboutSkill.map(finding => finding.code)).toEqual(['skill-published-copy'])

    // "Commit and pull": copy the project folder to machine B's clone.
    await cp(join(a.clone, '.xerxes'), join(b.clone, '.xerxes'), { recursive: true })
    const pulled = await b.request('slash', { command: '/xerxes-sync' })
    expect(pulled).toMatchObject({ ok: true, repo: pushed.repo, lessons: { kind: 'synced', applied: 1 }, memory: { imported: ['memory/topics/tpu.md'] }, skills: { available: [SKILL] } })
    expect(await readFile(join(b.memory, 'topics', 'tpu.md'), 'utf8')).toContain('v5p queue')
    const recalled = await b.lessons.recall('tpu lock busy', b.clone)
    expect(recalled).toEqual([expect.objectContaining({ id: lesson.id, reviewStatus: 'approved' })])
    expect(renderApprovedLessonsForTurn(recalled).match(/libtpu_lockfile/gu)).toHaveLength(1)
    expect(skillsNamed(await b.request('slash', { command: '/skills' }), SKILL)).toHaveLength(1)

    // Running it again on either machine is a no-op: nothing new, nothing doubled.
    const again = await b.request('slash', { command: '/xerxes-sync' })
    expect(again).toMatchObject({ ok: true, lessons: { applied: 0, retired: 0 }, memory: { exported: [], imported: [], merged: [] }, skills: { published: [] } })
    const againA = await a.request('slash', { command: '/xerxes-sync' })
    expect(againA).toMatchObject({ ok: true, lessons: { applied: 0 }, memory: { exported: [], imported: [], merged: [] }, skills: { published: [], unchanged: 1 } })
    expect(await readdir(join(a.clone, '.xerxes', 'lessons'))).toEqual([`${lesson.id}.json`])
    expect((await a.lessons.list(a.clone)).map(item => item.id)).toEqual([lesson.id])
    expect((await b.lessons.list(b.clone)).map(item => item.id)).toEqual([lesson.id])

    // The singular spelling from the bug report reaches the lesson subcommand.
    expect(await b.request('slash', { command: '/xerxes lesson sync' })).toMatchObject({ ok: true, report: expect.stringContaining('Lesson sharing:') })
  } finally {
    await a.close(); await b.close()
    await rm(base, { recursive: true, force: true })
  }
}, 30_000)
