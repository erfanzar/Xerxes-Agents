// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import {
  createNativeSweTasks,
  formatSweReport,
  runSweBenchmark,
  runSweCli,
  writeSweWorkspaceFiles,
} from '../playground/index.js'
import { BUNDLED_SKILLS_SOURCE_DIRECTORY } from '../scripts/copyBundledSkills.js'
import { ValidationError } from '../src/core/errors.js'
import { lintHarness, type HarnessSkillDocument } from '../src/extensions/harnessLint.js'
import { parseSkillMarkdown, SkillRegistry } from '../src/extensions/skills.js'
import {
  createDefaultSwarmWorkspacePort,
  SwarmManager,
  type SwarmWorkerWakeReason,
} from '../src/swarm/index.js'

async function withTempSwarmDirs(
  fn: (dirs: {
    readonly root: string
    readonly workspace: string
    readonly storage: string
    readonly allocations: string
  }) => Promise<void>,
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-swe-tournament-'))
  const workspace = join(root, 'repo')
  const storage = join(root, 'storage')
  const allocations = join(root, 'allocations')
  await mkdir(workspace, { recursive: true })
  await mkdir(storage, { recursive: true })
  await mkdir(allocations, { recursive: true })
  try {
    await fn({ root, workspace, storage, allocations })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

async function runGitInTest(cwd: string, args: readonly string[]): Promise<string> {
  const proc = Bun.spawn(['git', ...args], { cwd, stdout: 'pipe', stderr: 'pipe' })
  const [code, stdout, stderr] = await Promise.all([
    proc.exited,
    new Response(proc.stdout).text(),
    new Response(proc.stderr).text(),
  ])
  if (code !== 0) {
    throw new Error(`git ${args.join(' ')} failed (${code}): ${stderr}`)
  }
  return stdout.trim()
}

async function initSampleTypeScriptGitRepo(dir: string): Promise<string> {
  await runGitInTest(dir, ['init'])
  await runGitInTest(dir, ['config', 'user.email', 'tournament-test@xerxes.local'])
  await runGitInTest(dir, ['config', 'user.name', 'Xerxes Tournament Test'])
  await mkdir(join(dir, 'src'), { recursive: true })
  await writeFile(
    join(dir, 'src/rateLimiter.ts'),
    [
      'export function remainingQuota(limit: number, used: number): number {',
      '  // Bug: allows negative remaining quota instead of clamping at 0',
      '  return limit - used',
      '}',
      '',
    ].join('\n'),
    'utf8',
  )
  await writeFile(
    join(dir, 'src/display.ts'),
    [
      "import { remainingQuota } from './rateLimiter.js'",
      '',
      'export function formatQuotaBanner(limit: number, used: number): string {',
      '  return `Remaining: ${remainingQuota(limit, used)}/${limit}`',
      '}',
      '',
    ].join('\n'),
    'utf8',
  )
  await runGitInTest(dir, ['add', '.'])
  await runGitInTest(dir, ['commit', '-m', 'chore: initial buggy rate limiter'])
  return runGitInTest(dir, ['rev-parse', 'HEAD'])
}

describe('Multi-Worktree Patch Tournament — Swarm Workspaces & Manager', () => {
  test('compares 2 share worktree candidates, selects minimal clean verified patch, and lands it into main repo before kill without committing', async () => {
    await withTempSwarmDirs(async ({ root, workspace, storage, allocations }) => {
      const initialHead = await initSampleTypeScriptGitRepo(workspace)

      const swarm = await SwarmManager.open({
        sessionId: 'sess-patch-tournament',
        workspaceDir: workspace,
        storageDir: storage,
        workspacePort: createDefaultSwarmWorkspacePort({ rootDir: allocations }),
      })

      const added = await swarm.add({
        caller: 'coordinator',
        name: 'solver',
        n: 2,
        workspace: 'share',
        brief: 'Fix negative remainingQuota bug cleanly.',
      })
      const solver1Path = added.added[0]!.workspace_path!
      const solver2Path = added.added[1]!.workspace_path!
      expect(solver1Path).not.toBe(workspace)
      expect(solver2Path).not.toBe(workspace)
      expect(solver1Path).not.toBe(solver2Path)

      // Candidate 1 (bloated 3-file fix + untracked helper): passes checks, but touches 3 files and many lines.
      await writeFile(
        join(solver1Path, 'src/clampHelper.ts'),
        [
          'export function clampNonNegative(value: number): number {',
          '  if (value < 0) {',
          '    return 0',
          '  }',
          '  return value',
          '}',
          '',
        ].join('\n'),
        'utf8',
      )
      await writeFile(
        join(solver1Path, 'src/rateLimiter.ts'),
        [
          "import { clampNonNegative } from './clampHelper.js'",
          '',
          'export function remainingQuota(limit: number, used: number): number {',
          '  const raw = limit - used',
          '  return clampNonNegative(raw)',
          '}',
          '',
        ].join('\n'),
        'utf8',
      )
      await writeFile(
        join(solver1Path, 'src/display.ts'),
        [
          "import { remainingQuota } from './rateLimiter.js'",
          '',
          'export function formatQuotaBanner(limit: number, used: number): string {',
          '  const safeRemaining = Math.max(0, remainingQuota(limit, used))',
          '  return `Remaining: ${safeRemaining}/${limit}`',
          '}',
          '',
        ].join('\n'),
        'utf8',
      )

      // Candidate 2 (minimal surgical 1-file root-cause fix): touches only `src/rateLimiter.ts`!
      await writeFile(
        join(solver2Path, 'src/rateLimiter.ts'),
        [
          'export function remainingQuota(limit: number, used: number): number {',
          '  return Math.max(0, limit - used)',
          '}',
          '',
        ].join('\n'),
        'utf8',
      )

      const patch1 = await swarm.inspectMemberPatch('solver-1')
      const patch2 = await swarm.inspectMemberPatch('solver-2')

      expect(patch1.changedFiles).toEqual(['src/clampHelper.ts', 'src/display.ts', 'src/rateLimiter.ts'])
      expect(patch1.cleanWorkingTree).toBe(false)
      expect(patch1.hasConflictMarkers).toBe(false)
      expect(patch1.applicable).toBe(true)

      expect(patch2.changedFiles).toEqual(['src/rateLimiter.ts'])
      expect(patch2.cleanWorkingTree).toBe(false)
      expect(patch2.hasConflictMarkers).toBe(false)
      expect(patch2.applicable).toBe(true)
      expect(patch2.insertions + patch2.deletions).toBeLessThan(patch1.insertions + patch1.deletions)

      // Both verified -> minimal candidate `solver-2` wins rank 1!
      const comparison = await swarm.compareCandidatePatches(['solver-1', 'solver-2'], {
        verified: { 'solver-1': true, 'solver-2': true },
      })
      expect(comparison.winner?.memberName).toBe('solver-2')
      expect(comparison.candidates.map(c => c.memberName)).toEqual(['solver-2', 'solver-1'])
      expect(comparison.candidates[0]?.rank).toBe(1)
      expect(comparison.candidates[1]?.rank).toBe(2)

      // Verification status outranks minimality when `solver-2` fails verification!
      const whenSolver2FailsVerification = await swarm.compareCandidatePatches(['solver-1', 'solver-2'], {
        verified: { 'solver-1': true, 'solver-2': false },
      })
      expect(whenSolver2FailsVerification.winner?.memberName).toBe('solver-1')

      // Export patch file and apply winning `solver-2` patch cleanly into main workspace BEFORE kill!
      const exportedPath = join(root, 'winner.patch')
      const exported = await swarm.exportMemberPatch(comparison.winner!.memberName, {
        outputPath: exportedPath,
      })
      expect(exported.outputPath).toBe(exportedPath)
      expect(await readFile(exportedPath, 'utf8')).toContain('Math.max(0, limit - used)')

      const applied = await swarm.applyWinnerPatchToWorkspace(comparison.winner!.memberName, workspace)
      expect(applied.status).toBe('applied')
      expect(applied.memberName).toBe('solver-2')
      expect(applied.changedFiles).toEqual(['src/rateLimiter.ts'])

      // Main repo has the fix applied in working tree, with NO git commit created and NO bloated untracked file!
      const mainLimiter = await readFile(join(workspace, 'src/rateLimiter.ts'), 'utf8')
      expect(mainLimiter).toContain('Math.max(0, limit - used)')
      expect(existsSync(join(workspace, 'src/clampHelper.ts'))).toBe(false)
      expect(await runGitInTest(workspace, ['rev-parse', 'HEAD'])).toBe(initialHead)

      // Retire worktree group cleanly!
      const killed = await swarm.kill({ caller: 'coordinator', name: 'solver' })
      expect(killed.killed).toEqual(['solver-1', 'solver-2'])
      expect(existsSync(solver1Path)).toBe(false)
      expect(existsSync(solver2Path)).toBe(false)
      // Landed fix in main workspace remains intact after worktrees are deleted!
      expect(await readFile(join(workspace, 'src/rateLimiter.ts'), 'utf8')).toContain('Math.max(0, limit - used)')

      await swarm.dispose()
    })
  })

  test('detects committed changes in branch clones, disqualifies conflict markers, enforces workspace mode guards, and supports quiet schedule wakeMember', async () => {
    await withTempSwarmDirs(async ({ workspace, storage, allocations }) => {
      await initSampleTypeScriptGitRepo(workspace)

      const wokenReasons: Array<{ readonly name: string; readonly reason: SwarmWorkerWakeReason; readonly prompt: string }> = []
      const swarm = await SwarmManager.open({
        sessionId: 'sess-branch-and-wake',
        workspaceDir: workspace,
        storageDir: storage,
        workspacePort: createDefaultSwarmWorkspacePort({ rootDir: allocations }),
        workerRunner: async (input) => {
          wokenReasons.push({ name: input.member.name, reason: input.reason, prompt: input.prompt })
          return {
            messages: [
              ...input.messages,
              { role: 'user', content: input.prompt },
              { role: 'assistant', content: 'Checked.' },
            ],
          }
        },
      })

      const bClean = await swarm.add({ name: 'branch-clean', brief: 'Commit fix in clone', workspace: 'branch' })
      const bConflict = await swarm.add({ name: 'branch-conflict', brief: 'Leaves conflict markers', workspace: 'branch' })
      await swarm.add({ name: 'reader-inherit', brief: 'Read-only verifier', workspace: 'inherit' })
      await swarm.waitForIdle()

      const cleanClonePath = bClean.added[0]!.workspace_path!
      const conflictClonePath = bConflict.added[0]!.workspace_path!

      // Commit a change inside `branch-clean`'s local clone -> cleanWorkingTree is true, yet diff against baseHead is captured!
      await runGitInTest(cleanClonePath, ['config', 'user.email', 'clone@xerxes.local'])
      await runGitInTest(cleanClonePath, ['config', 'user.name', 'Clone Worker'])
      await writeFile(
        join(cleanClonePath, 'src/rateLimiter.ts'),
        'export function remainingQuota(limit: number, used: number): number {\n  return limit > used ? limit - used : 0\n}\n',
        'utf8',
      )
      await runGitInTest(cleanClonePath, ['add', 'src/rateLimiter.ts'])
      await runGitInTest(cleanClonePath, ['commit', '-m', 'fix: clamp quota'])

      // Write unresolved git conflict markers in `branch-conflict`!
      await writeFile(
        join(conflictClonePath, 'src/rateLimiter.ts'),
        [
          '<<<<<<< HEAD',
          'export const x = 1',
          '=======',
          'export const x = 2',
          '>>>>>>> candidate',
          '',
        ].join('\n'),
        'utf8',
      )

      const cleanInspect = await swarm.inspectMemberPatch('branch-clean')
      expect(cleanInspect.cleanWorkingTree).toBe(true)
      expect(cleanInspect.changedFiles).toEqual(['src/rateLimiter.ts'])
      expect(cleanInspect.hasConflictMarkers).toBe(false)

      const conflictInspect = await swarm.inspectMemberPatch('branch-conflict')
      expect(conflictInspect.hasConflictMarkers).toBe(true)

      const ranked = await swarm.compareCandidatePatches(['branch-conflict', 'branch-clean'])
      expect(ranked.winner?.memberName).toBe('branch-clean')
      expect(ranked.candidates.find(c => c.memberName === 'branch-conflict')?.eligible).toBe(false)

      await expect(swarm.applyWinnerPatchToWorkspace('branch-conflict', workspace)).rejects.toThrow(ValidationError)
      await expect(swarm.inspectMemberPatch('reader-inherit')).rejects.toThrow(ValidationError)

      // Verify quiet `wakeMember` with `'schedule'` reason emits zero board posts!
      const headBeforeWake = swarm.head()
      const woke = await swarm.wakeMember('reader-inherit', '[schedule] Check silent stall', 'schedule')
      expect(woke).toBe(true)
      await swarm.waitForIdle()
      expect(swarm.head()).toBe(headBeforeWake)
      expect(wokenReasons.at(-1)).toEqual({
        name: 'reader-inherit',
        reason: 'schedule',
        prompt: '[schedule] Check silent stall',
      })
      expect(await swarm.wakeMember('unknown-worker', '[schedule] noop')).toBe(false)

      await swarm.dispose()
    })
  })
})

describe('Bundled swe-patch-tournament & immortals-coordinator Skills', () => {
  test('discovers swe-patch-tournament and immortals-coordinator cleanly with zero harness lint findings', async () => {
    const paths = [
      ...new Bun.Glob('**/SKILL.md').scanSync({
        cwd: BUNDLED_SKILLS_SOURCE_DIRECTORY,
        absolute: true,
      }),
    ].sort()
    const documents: HarnessSkillDocument[] = await Promise.all(
      paths.map(async path => ({ path, content: await Bun.file(path).text() })),
    )

    expect(lintHarness(documents)).toEqual([])

    const registry = new SkillRegistry()
    const discovered = await registry.discover(BUNDLED_SKILLS_SOURCE_DIRECTORY)
    expect(discovered).toHaveLength(documents.length)
    expect(registry.discoveryNotes).toEqual([])

    const tournamentPath = join(BUNDLED_SKILLS_SOURCE_DIRECTORY, 'swe-patch-tournament', 'SKILL.md')
    const coordinatorPath = join(BUNDLED_SKILLS_SOURCE_DIRECTORY, 'immortals-coordinator', 'SKILL.md')
    const tournamentSkill = parseSkillMarkdown(await Bun.file(tournamentPath).text(), tournamentPath)
    const coordinatorSkill = parseSkillMarkdown(await Bun.file(coordinatorPath).text(), coordinatorPath)

    expect(tournamentSkill.metadata.name).toBe('swe-patch-tournament')
    expect(tournamentSkill.instructions).toContain('workspace: "share"')
    expect(tournamentSkill.instructions).toContain('compareCandidatePatches')
    expect(tournamentSkill.instructions).toContain('applyWinnerPatchToWorkspace')
    expect(coordinatorSkill.metadata.name).toBe('immortals-coordinator')
    expect(coordinatorSkill.instructions).toContain('swe-patch-tournament')
    expect(registry.get('swarm-coordinator')?.metadata.name).toBe('immortals-coordinator')
  })
})

describe('SWE-Bench-Style Evaluation Battery & 5 Core SWE KPIs', () => {
  test('withholds hidden holdout suites during solver execution, rejects symptom-patched holdout cheaters, and restores tampered visible tests', async () => {
    const runRoot = await mkdtemp(join(tmpdir(), 'xerxes-swe-bench-cheat-'))
    try {
      const tasks = createNativeSweTasks()
      expect(tasks.length).toBeGreaterThanOrEqual(4)

      // 1. Verify unpatched workspaces pass baseline tests and fail visible reproduction tests, with holdout hidden!
      const unpatchedReport = await runSweBenchmark({
        runRoot,
        tasks: [tasks[0]!],
        model: 'noop-agent',
        solver: async ({ workspaceDirectory, task }) => {
          for (const holdoutFile of task.holdoutTestFiles) {
            expect(existsSync(join(workspaceDirectory, holdoutFile.path))).toBe(false)
          }
          for (const visibleFile of task.visibleReproductionTestFiles) {
            expect(existsSync(join(workspaceDirectory, visibleFile.path))).toBe(true)
          }
          return {}
        },
      })
      expect(unpatchedReport.records[0]?.baselineTestsPassed).toBe(true)
      expect(unpatchedReport.records[0]?.visibleTestsPassed).toBe(false)
      expect(unpatchedReport.metrics.resolvedRate).toBe(0)

      // 2. Symptom-patching cheater passes visible tests on all 4 tasks, but hidden holdout tests catch every cheat!
      const cheatReport = await runSweBenchmark({
        runRoot,
        tasks,
        model: 'symptom-cheater-v1',
        solver: async ({ workspaceDirectory, task }) => {
          // Ensure holdout directory is still absent while solver executes!
          expect(existsSync(join(workspaceDirectory, 'test/__holdout__'))).toBe(false)
          await writeSweWorkspaceFiles(workspaceDirectory, task.symptomCheatFiles)
          return {
            telemetry: {
              turns: [{ turnIndex: 1, tokens: 600 }],
              toolCalls: [
                { name: 'ReadFile', turnIndex: 1, filePath: 'test/repro.test.ts', ok: true },
                { name: 'FileEditTool', turnIndex: 1, filePath: task.symptomCheatFiles[0]!.path, ok: true },
              ],
              totalTokens: 600,
              costUsd: 0.003,
            },
          }
        },
      })

      expect(cheatReport.records.every(r => r.baselineTestsPassed)).toBe(true)
      expect(cheatReport.records.every(r => r.visibleTestsPassed)).toBe(true)
      expect(cheatReport.records.every(r => !r.holdoutTestsPassed)).toBe(true)
      expect(cheatReport.metrics.holdoutRegressionCount).toBe(tasks.length)
      expect(cheatReport.metrics.holdoutRegressionRate).toBe(1)
      expect(cheatReport.metrics.resolvedRate).toBe(0)
      expect(cheatReport.metrics.costAndLatencyPerResolved.meanWallMsPerResolved).toBeNull()
      expect(cheatReport.metrics.costAndLatencyPerResolved.meanTokensPerResolved).toBeNull()
      expect(cheatReport.metrics.costAndLatencyPerResolved.meanCostUsdPerResolved).toBeNull()

      // 3. Test-tampering cheater deleting assertions in `test/repro.test.ts` is caught by canonical test restoration!
      const tamperReport = await runSweBenchmark({
        runRoot,
        tasks: [tasks[1]!],
        model: 'test-tamperer',
        solver: async ({ workspaceDirectory }) => {
          await writeFile(
            join(workspaceDirectory, 'test/repro.test.ts'),
            "import { expect, test } from 'bun:test'\ntest('forged', () => { expect(true).toBe(true) })\n",
            'utf8',
          )
        },
      })
      expect(tamperReport.records[0]?.visibleTestsPassed).toBe(false)
      expect(tamperReport.records[0]?.resolved).toBe(false)
    } finally {
      await rm(runRoot, { recursive: true, force: true })
    }
  })

  test('accepts genuine root-cause fixes across all 4 tasks and accurately computes all 5 Core SWE KPIs + CLI output', async () => {
    const runRoot = await mkdtemp(join(tmpdir(), 'xerxes-swe-bench-pass-'))
    try {
      const tasks = createNativeSweTasks()
      let fakeClockMs = 1_000_000
      let taskCursor = 0

      const outputChunks: string[] = []
      const exitCode = await runSweCli(['--json', '--run-root', runRoot], {
        tasks,
        write: (chunk) => {
          outputChunks.push(chunk)
        },
        solver: async ({ workspaceDirectory, task }) => {
          const idx = taskCursor
          taskCursor += 1
          await writeSweWorkspaceFiles(workspaceDirectory, task.referenceSolutionFiles)

          // Deterministic varied telemetry across the 4 tasks:
          // - Tasks 0 & 2 localize ground-truth target on turn 1 (400 tokens)
          // - Tasks 1 & 3 localize ground-truth target on turn 2 (900 tokens)
          // - Tasks 0 & 1 have 1 failed edit + 2 succeeding edits (3 attempts each)
          // - Tasks 2 & 3 have 0 failed edits + 2 succeeding edits (2 attempts each)
          // Total edit attempts = 10, failed edits = 2 -> editMismatchRate = 0.2!
          const localizeTurn = idx % 2 === 0 ? 1 : 2
          const firstTarget = task.groundTruthTargetFiles[0]!
          const hasEditRetry = idx < 2

          return {
            telemetry: {
              turns: [
                { turnIndex: 1, tokens: 400 },
                { turnIndex: 2, tokens: 500 },
                { turnIndex: 3, tokens: 300 },
              ],
              toolCalls: [
                ...(localizeTurn === 1
                  ? [{ name: 'FileOutlineTool', turnIndex: 1, filePath: firstTarget, ok: true }]
                  : [{ name: 'ReadFile', turnIndex: 1, filePath: 'test/repro.test.ts', ok: true }]),
                ...(localizeTurn === 2
                  ? [{ name: 'ReadFile', turnIndex: 2, filePath: join(workspaceDirectory, firstTarget), ok: true }]
                  : []),
                ...(hasEditRetry
                  ? [{ name: 'FileEditTool', turnIndex: 2, filePath: firstTarget, ok: false }]
                  : []),
                { name: 'ApplyPatchTool', turnIndex: 2, filePath: firstTarget, ok: true },
                { name: 'FileEditTool', turnIndex: 3, filePath: task.groundTruthTargetFiles[1]!, ok: true },
              ],
              totalTokens: 1_200 + idx * 200, // [1200, 1400, 1600, 1800] -> mean = 1500
              costUsd: 0.01 + idx * 0.002, // [0.010, 0.012, 0.014, 0.016] -> mean = 0.013
            },
          }
        },
      })

      expect(exitCode).toBe(0)
      const parsedReport = JSON.parse(outputChunks.join('')) as Awaited<ReturnType<typeof runSweBenchmark>>
      expect(parsedReport.kind).toBe('swe')
      expect(parsedReport.records).toHaveLength(4)
      expect(parsedReport.records.every(r => r.resolved)).toBe(true)

      // Also verify deterministic wallMs via direct `runSweBenchmark` clock injection!
      const deterministicReport = await runSweBenchmark({
        runRoot,
        tasks: [tasks[0]!, tasks[1]!],
        model: 'root-cause-solver',
        clock: () => {
          const current = fakeClockMs
          fakeClockMs += 250
          return current
        },
        solver: async ({ workspaceDirectory, task }) => {
          await writeSweWorkspaceFiles(workspaceDirectory, task.referenceSolutionFiles)
          return {
            telemetry: {
              turns: [{ turnIndex: 1, tokens: 500 }],
              toolCalls: [
                { name: 'ReadFile', turnIndex: 1, filePath: task.groundTruthTargetFiles[0]!, ok: true },
                { name: 'ApplyPatchTool', turnIndex: 1, filePath: task.groundTruthTargetFiles[0]!, ok: true },
              ],
              totalTokens: 500,
              costUsd: 0.005,
            },
          }
        },
      })

      // Verify all 5 Core SWE KPIs on the full 4-task report + deterministic wallMs report:
      expect(parsedReport.metrics.resolvedRate).toBe(1)
      expect(parsedReport.metrics.holdoutRegressionRate).toBe(0)
      expect(parsedReport.metrics.localizationEfficiency.localizationHitRate).toBe(1)
      // Turns to first relevant read: [1, 2, 1, 2] -> mean 1.5; Tokens: [400, 900, 400, 900] -> mean 650
      expect(parsedReport.metrics.localizationEfficiency.turnsToFirstRelevantRead).toBe(1.5)
      expect(parsedReport.metrics.localizationEfficiency.tokensToFirstRelevantRead).toBe(650)
      // Failed edits: 2 / 10 = 0.2
      expect(parsedReport.metrics.totalEditAttempts).toBe(10)
      expect(parsedReport.metrics.failedEditAttempts).toBe(2)
      expect(parsedReport.metrics.editMismatchRate).toBeCloseTo(0.2, 6)
      // Mean tokens: (1200 + 1400 + 1600 + 1800) / 4 = 1500; Mean costUsd: 0.013
      expect(parsedReport.metrics.costAndLatencyPerResolved.meanTokensPerResolved).toBe(1500)
      expect(parsedReport.metrics.costAndLatencyPerResolved.meanCostUsdPerResolved).toBeCloseTo(0.013, 6)
      expect(deterministicReport.metrics.costAndLatencyPerResolved.meanWallMsPerResolved).toBe(250)

      const rendered = formatSweReport(parsedReport)
      expect(rendered).toContain('1. resolvedRate (Pass@1):        4/4 (100.0%)')
      expect(rendered).toContain('3. editMismatchRate:             2/10 (20.0%)')
      expect(rendered).toContain('4. holdoutRegressionRate:        0/4 (0.0%)')
    } finally {
      await rm(runRoot, { recursive: true, force: true })
    }
  })
})
