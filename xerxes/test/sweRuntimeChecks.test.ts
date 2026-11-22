// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { describe, expect, test } from 'bun:test'

import { loadBuiltinAgentDefinitions } from '../src/agents/definitions.js'
import type { ToolExecutor } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import {
  extractMutatedToolPaths,
  isReadOnlyInspectionTool,
  MUTATING_TOOL_NAMES,
  READ_ONLY_INSPECTION_TOOL_NAMES,
  runCascadingTurn,
  turnResultFindings,
  WorkspaceCheckpointManager,
  type CascadeStep,
} from '../src/runtime/cascade.js'
import { EscalationController, parseEscalationConfig } from '../src/runtime/escalationPolicy.js'
import {
  buildLeanSystemPrompt,
  LEAN_CORE_TOOL_NAMES,
  LEAN_SWE_TOOL_NAMES,
  selectLeanTools,
} from '../src/runtime/leanPrompt.js'
import { inspectObjectiveResponse, isReadOnlyObjectiveTool } from '../src/runtime/objectiveGuard.js'
import {
  captureCodeCheckBaseline,
  extractPatchTargetPaths,
  findingFingerprint,
  runResultChecks,
} from '../src/runtime/resultChecks.js'
import { createAgentState } from '../src/streaming/events.js'
import { fileStateTracker } from '../src/tools/fileState.js'
import type { ToolDefinition } from '../src/types/toolCalls.js'

async function withTempWorkspace(run: (workspace: string) => Promise<void>): Promise<void> {
  const workspace = await mkdtemp(join(tmpdir(), 'xerxes-swe-checks-'))
  try {
    await run(workspace)
  } finally {
    await rm(workspace, { force: true, recursive: true })
  }
}

async function runGit(args: readonly string[], cwd: string): Promise<{ exitCode: number; stdout: string }> {
  const proc = Bun.spawn(['git', ...args], { cwd, stdin: 'ignore', stdout: 'pipe', stderr: 'pipe' })
  const [stdout] = await Promise.all([new Response(proc.stdout).text(), new Response(proc.stderr).text()])
  const exitCode = await proc.exited
  return { exitCode, stdout }
}

describe('code-aware resultChecks: conflict_markers and syntax_error', () => {
  test('detects unresolved Git merge conflict markers in code/config files while ignoring Markdown fenced examples and setext headings', async () => {
    await withTempWorkspace(async workspace => {
      await writeFile(
        join(workspace, 'conflict.ts'),
        [
          'export function compute(x: number): number {',
          '<<<<<<< HEAD',
          '  return x + 1',
          '=======',
          '  return x * 2',
          '>>>>>>> feature/branch',
          '}',
        ].join('\n'),
      )
      await writeFile(
        join(workspace, 'guide.md'),
        [
          'Merge Guide',
          '=======',
          '',
          'Example conflict block inside fenced code:',
          '```diff',
          '<<<<<<< HEAD',
          'foo',
          '=======',
          'bar',
          '>>>>>>> main',
          '```',
          '',
          '~~~ts',
          '<<<<<<< stash',
          '=======',
          '>>>>>>> theirs',
          '~~~',
        ].join('\n'),
      )
      await writeFile(
        join(workspace, 'conflicted-doc.md'),
        [
          '# Notes',
          '<<<<<<< HEAD',
          'Ours prose',
          '=======',
          'Theirs prose',
          '>>>>>>> origin/main',
        ].join('\n'),
      )

      const findings = await runResultChecks({
        workspaceRoot: workspace,
        changedFiles: ['conflict.ts', 'guide.md', 'conflicted-doc.md'],
      })

      expect(findings.map(finding => [finding.path, finding.code, finding.severity])).toEqual([
        ['conflict.ts', 'conflict_markers', 'error'],
        ['conflicted-doc.md', 'conflict_markers', 'error'],
      ])
      expect(findings[0]?.message).toContain('line 2')
      expect(findings[0]?.message).toContain('line 4')
      expect(findings[0]?.message).toContain('line 6')
    })
  })

  test('validates TS/TSX/MTS/CTS/JS/JSX/MJS/CJS via Bun.Transpiler and JSON/JSONC with exact line and column coordinates', async () => {
    await withTempWorkspace(async workspace => {
      // Valid files across extensions
      await writeFile(
        join(workspace, 'valid.ts'),
        'export interface Opts { readonly count?: number }\nexport const run = async <const T>(v: T): Promise<T> => v satisfies T;\n',
      )
      await writeFile(
        join(workspace, 'valid.tsx'),
        'export function Badge({ label }: { label: string }) { return <span className="tag">{label}</span> }\n',
      )
      await writeFile(join(workspace, 'valid.mts'), 'export const value = await Promise.resolve(42);\n')
      await writeFile(join(workspace, 'valid.cjs'), 'module.exports = { ok: true };\n')
      await writeFile(join(workspace, 'valid.json'), '{\n  "name": "xerxes",\n  "version": 1\n}\n')
      await writeFile(join(workspace, 'tsconfig.json'), '{\n  // JSONC comment allowed in tsconfig\n  "compilerOptions": { "strict": true }\n}\n')

      expect(await runResultChecks({
        workspaceRoot: workspace,
        changedFiles: ['valid.ts', 'valid.tsx', 'valid.mts', 'valid.cjs', 'valid.json', 'tsconfig.json'],
      })).toEqual([])

      // Invalid syntax files across extensions
      await writeFile(join(workspace, 'broken.ts'), 'const ok = 1;\nexport const bad: number = ;\n')
      await writeFile(join(workspace, 'broken.tsx'), 'export const View = () => (\n  <div>\n    <span>\n  </div>\n);\n')
      await writeFile(join(workspace, 'broken.mjs'), 'export function f(a) {\n  return a +\n}\n')
      await writeFile(join(workspace, 'broken.json'), '{\n  "alpha": 1,\n  "beta": \n}\n')

      const findings = await runResultChecks({
        workspaceRoot: workspace,
        changedFiles: ['broken.ts', 'broken.tsx', 'broken.mjs', 'broken.json'],
      })

      expect(findings.map(finding => [finding.path, finding.code, finding.severity])).toEqual([
        ['broken.ts', 'syntax_error', 'error'],
        ['broken.tsx', 'syntax_error', 'error'],
        ['broken.mjs', 'syntax_error', 'error'],
        ['broken.json', 'syntax_error', 'error'],
      ])
      expect(findings[0]?.message).toMatch(/line 2, column \d+/u)
      expect(findings[1]?.message).toMatch(/line 4, column \d+/u)
      expect(findings[2]?.message).toMatch(/line 3, column \d+/u)
      expect(findings[3]?.message).toMatch(/line 4, column 1/u)
    })
  })

  test('filters pre-existing baseline code findings (including line-shifted ones) while reporting newly introduced errors', async () => {
    await withTempWorkspace(async workspace => {
      const legacyPath = join(workspace, 'legacy.ts')
      const cleanPath = join(workspace, 'clean.ts')
      await writeFile(legacyPath, 'export const existingBroken = ;\n')
      await writeFile(cleanPath, 'export const clean = 1;\n')

      const baseline = await captureCodeCheckBaseline(workspace, ['legacy.ts', 'clean.ts'])
      expect(baseline.size).toBe(1)

      // Inserting valid lines above the pre-existing syntax error shifts its line number but keeps the same error identity.
      await writeFile(legacyPath, 'import type { Foo } from "./foo.js";\n\nexport const existingBroken = ;\n')
      expect(await runResultChecks({
        workspaceRoot: workspace,
        changedFiles: ['legacy.ts', 'clean.ts'],
        baselineFindings: baseline,
      })).toEqual([])

      // Introducing a brand-new syntax error produces a distinct fingerprint and IS reported.
      await writeFile(legacyPath, 'export function newlyBroken( {\nexport const existingBroken = ;\n')
      const newFindings = await runResultChecks({
        workspaceRoot: workspace,
        changedFiles: ['legacy.ts'],
        baselineFindings: baseline,
      })
      expect(newFindings).toHaveLength(1)
      expect(newFindings[0]?.code).toBe('syntax_error')
      expect(baseline.has(findingFingerprint(workspace, newFindings[0]!))).toBe(false)
    })
  })
})

describe('ApplyPatchTool and FileOutlineTool tracking in cascade and objectiveGuard', () => {
  test('extracts V4A and unified diff patch targets for ApplyPatchTool and treats FileOutlineTool as read-only', async () => {
    const v4aPatch = [
      '*** Begin Patch',
      '*** Update File: src/alpha.ts',
      '@@',
      '-export const a = 1',
      '+export const a = 2',
      '*** Add File: src/beta.json',
      '+{"bad": }',
      '*** Update File: src/old.ts',
      '*** Move to: src/renamed.ts',
      '*** End Patch',
    ].join('\n')

    expect(extractPatchTargetPaths(v4aPatch)).toEqual([
      'src/alpha.ts',
      'src/beta.json',
      'src/old.ts',
      'src/renamed.ts',
    ])

    const unifiedPatch = [
      'diff --git a/pkg/one.ts b/pkg/one.ts',
      '--- a/pkg/one.ts',
      '+++ b/pkg/one.ts',
      '@@ -1 +1 @@',
      '-export const one = 1',
      '+export const one = ;',
    ].join('\n')
    expect(extractPatchTargetPaths(unifiedPatch)).toEqual(['pkg/one.ts'])

    expect(MUTATING_TOOL_NAMES.has('ApplyPatchTool')).toBe(true)
    expect(READ_ONLY_INSPECTION_TOOL_NAMES.has('FileOutlineTool')).toBe(true)
    expect(isReadOnlyInspectionTool('FileOutlineTool')).toBe(true)
    expect(extractMutatedToolPaths('FileOutlineTool', { file_path: 'src/alpha.ts' })).toEqual([])

    await withTempWorkspace(async workspace => {
      await mkdir(join(workspace, 'src'), { recursive: true })
      await writeFile(join(workspace, 'src/alpha.ts'), 'export const a = 2;\n')
      await writeFile(join(workspace, 'src/beta.json'), '{"bad": }\n')
      await writeFile(join(workspace, 'src/untouchedBroken.ts'), 'export const untouched = ;\n')

      const state = createAgentState()
      state.toolExecutions.push(
        {
          toolCallId: 'outline-1',
          durationMs: 1,
          name: 'FileOutlineTool',
          inputs: { file_path: 'src/untouchedBroken.ts' },
          permitted: true,
          result: 'symbols...',
        },
        {
          toolCallId: 'patch-1',
          durationMs: 2,
          name: 'ApplyPatchTool',
          inputs: { patch: v4aPatch },
          permitted: true,
          result: 'Applied patch cleanly.',
        },
      )
      state.messages.push({ role: 'assistant', content: 'Updated files.' })

      const findings = await turnResultFindings(state, 0, workspace)
      expect(findings.map(finding => [finding.path, finding.code])).toEqual([
        ['src/beta.json', 'syntax_error'],
      ])
    })
  })

  test('objectiveGuard requires post-edit verification after ApplyPatchTool and treats FileOutlineTool as read-only', () => {
    expect(isReadOnlyObjectiveTool('FileOutlineTool')).toBe(true)
    expect(isReadOnlyObjectiveTool('ApplyPatchTool')).toBe(false)

    // ApplyPatchTool without verification -> must continue
    const unverifiedPatch = inspectObjectiveResponse('All tests pass.', {
      mode: 'objective',
      evidence: {
        toolExecutions: [
          { name: 'ApplyPatchTool', inputs: { patch: '*** Begin Patch\n*** End Patch' }, permitted: true, result: 'Applied' },
        ],
      },
    })
    expect(unverifiedPatch.shouldContinue).toBe(true)

    // ApplyPatchTool followed only by FileOutlineTool (no test run) -> still unverified
    const outlineOnlyAfterPatch = inspectObjectiveResponse('All tests pass.', {
      mode: 'objective',
      evidence: {
        toolExecutions: [
          { name: 'ApplyPatchTool', inputs: { patch: '*** Begin Patch\n*** End Patch' }, permitted: true, result: 'Applied' },
          { name: 'FileOutlineTool', inputs: { file_path: 'src/a.ts' }, permitted: true, result: 'outline' },
        ],
      },
    })
    expect(outlineOnlyAfterPatch.shouldContinue).toBe(true)

    // ApplyPatchTool -> bun test (exit 0) -> FileOutlineTool -> allowed to stop because FileOutlineTool is read-only
    const verifiedThenOutlined = inspectObjectiveResponse('All tests pass.', {
      mode: 'objective',
      evidence: {
        toolExecutions: [
          { name: 'ApplyPatchTool', inputs: { patch: '*** Begin Patch\n*** End Patch' }, permitted: true, result: 'Applied' },
          { name: 'exec_command', inputs: { cmd: 'bun', args: ['test'] }, permitted: true, result: JSON.stringify({ exitCode: 0 }) },
          { name: 'FileOutlineTool', inputs: { file_path: 'src/a.ts' }, permitted: true, result: 'outline' },
        ],
      },
    })
    expect(verifiedThenOutlined.shouldContinue).toBe(false)
  })
})

describe('cascade workspace checkpoint & best-attempt rollback', () => {
  const PATCH_TOOL: ToolDefinition = {
    type: 'function',
    function: {
      name: 'ApplyPatchTool',
      description: 'Apply an atomic patch.',
      parameters: { type: 'object', properties: { patch: { type: 'string' } }, required: ['patch'] },
    },
  }
  const WRITE_TOOL: ToolDefinition = {
    type: 'function',
    function: {
      name: 'WriteFile',
      description: 'Write a file.',
      parameters: { type: 'object', properties: { file_path: { type: 'string' }, content: { type: 'string' } }, required: ['file_path', 'content'] },
    },
  }

  test('restores touched files to best prior attempt on regression without touching .git/index or pre-existing dirty user files', async () => {
    await withTempWorkspace(async workspace => {
      await runGit(['init'], workspace)
      await runGit(['config', 'user.email', 'test@example.com'], workspace)
      await runGit(['config', 'user.name', 'Test'], workspace)

      await mkdir(join(workspace, 'src'), { recursive: true })
      await writeFile(join(workspace, 'src/math.ts'), 'export const add = (a: number, b: number): number => a + b;\n')
      await writeFile(join(workspace, 'src/helper.ts'), 'export const factor = 2;\n')
      await writeFile(join(workspace, 'src/user-wip.ts'), '// uncommitted user WIP with pre-existing syntax issue\nexport const wip = ;\n')
      await runGit(['add', 'src/math.ts', 'src/helper.ts'], workspace)
      await runGit(['commit', '-m', 'initial'], workspace)

      // Modify a tracked file BEFORE the turn starts to simulate dirty user edits outside the turn's touched set.
      await writeFile(join(workspace, 'src/untouched-dirty.ts'), 'export const dirtyBeforeTurn = "keep-me-exact";\n')
      const indexStatBefore = await stat(join(workspace, '.git/index'))
      const indexBytesBefore = new Uint8Array(await readFile(join(workspace, '.git/index')))

      const sessionId = 'cascade-rollback-session'
      const mathAbs = join(workspace, 'src/math.ts')
      const mathInitial = await readFile(mathAbs, 'utf8')
      const mathStats = await stat(mathAbs)
      fileStateTracker.record(sessionId, mathAbs, mathInitial, {
        mtimeMs: mathStats.mtimeMs,
        partialView: false,
        size: mathStats.size,
      })

      const requests: CompletionRequest[] = []
      let modelStep = 0
      const llm: LlmClient = {
        async *stream(request): AsyncGenerator<LlmDelta> {
          requests.push(request)
          modelStep += 1
          if (modelStep === 1) {
            // Round 0 (Attempt 1): introduces 1 syntax error in src/math.ts
            yield {
              toolCalls: [{
                id: 'c1',
                type: 'function',
                function: {
                  name: 'WriteFile',
                  arguments: { file_path: 'src/math.ts', content: 'export const add = (a: number, b: number): number => a + ;\n' },
                },
              }],
            }
            return
          }
          if (modelStep === 2) {
            yield { content: 'Attempt 1 complete.' }
            return
          }
          if (modelStep === 3) {
            // Round 1 (Attempt 2): REGRESSES — corrupts src/math.ts, corrupts src/helper.ts, and adds broken src/extra.json (3 errors!)
            yield {
              toolCalls: [
                {
                  id: 'c2',
                  type: 'function',
                  function: { name: 'WriteFile', arguments: { file_path: 'src/math.ts', content: '<<<<<<< HEAD\nbad\n=======\nworse\n>>>>>>> main\n' } },
                },
                {
                  id: 'c3',
                  type: 'function',
                  function: { name: 'WriteFile', arguments: { file_path: 'src/helper.ts', content: 'export const factor = ;\n' } },
                },
                {
                  id: 'c4',
                  type: 'function',
                  function: { name: 'WriteFile', arguments: { file_path: 'src/extra.json', content: '{"broken": }\n' } },
                },
              ],
            }
            return
          }
          if (modelStep === 4) {
            yield { content: 'Attempt 2 complete.' }
            return
          }
          if (modelStep === 5) {
            // Verify disk WAS ALREADY RESTORED to Attempt 1 before Round 2 executes!
            const restoredMath = await readFile(join(workspace, 'src/math.ts'), 'utf8')
            expect(restoredMath).toBe('export const add = (a: number, b: number): number => a + ;\n')
            expect(await readFile(join(workspace, 'src/helper.ts'), 'utf8')).toBe('export const factor = 2;\n')
            expect(await Bun.file(join(workspace, 'src/extra.json')).exists()).toBe(false)

            // Round 2 (Attempt 3): applies clean fix!
            yield {
              toolCalls: [{
                id: 'c5',
                type: 'function',
                function: {
                  name: 'ApplyPatchTool',
                  arguments: { patch: '*** Begin Patch\n*** Update File: src/math.ts\n@@\n-export const add = (a: number, b: number): number => a + ;\n+export const add = (a: number, b: number): number => a + b + 0;\n*** End Patch' },
                },
              }],
            }
            return
          }
          yield { content: 'Fixed src/math.ts cleanly.' }
        },
      }

      const toolExecutor: ToolExecutor = {
        async execute(call) {
          if (call.function.name === 'WriteFile') {
            const target = join(workspace, String(call.function.arguments['file_path']))
            const content = String(call.function.arguments['content'])
            await writeFile(target, content)
            const written = await stat(target)
            fileStateTracker.record(sessionId, target, content, {
              mtimeMs: written.mtimeMs,
              partialView: false,
              size: written.size,
            })
            return 'written'
          }
          if (call.function.name === 'ApplyPatchTool') {
            const target = join(workspace, 'src/math.ts')
            const content = 'export const add = (a: number, b: number): number => a + b + 0;\n'
            await writeFile(target, content)
            return 'patched'
          }
          return 'ok'
        },
      }

      const controller = new EscalationController({
        config: parseEscalationConfig({ escalation: { mode: 'on', max_local_retries: 3, max_advisor_calls: 1 } }, {}),
        boundaryMode: 'off',
        touchesPrivate: false,
        advisorAvailable: false,
        alternateModelAvailable: false,
      })

      const steps: CascadeStep[] = []
      const state = createAgentState()
      for await (const _event of runCascadingTurn(
        {
          sessionId,
          model: 'bonsai-local',
          permissionMode: 'accept-all',
          state,
          tools: [WRITE_TOOL, PATCH_TOOL],
          userMessage: 'Fix add in src/math.ts',
        },
        { llm, toolExecutor },
        {
          projectRoot: workspace,
          rollbackOnRegression: true,
          onStep: step => steps.push(step),
          hooks: {
            checks: () => 'enforce',
            escalate: findings => {
              const step = controller.next({ findings })
              if (step.action === 'accept') return undefined
              if (step.action === 'retry_local') {
                return { action: 'retry_local', level: 'info', message: 'retry', feedback: step.feedback }
              }
              return { action: 'stop', level: 'warning', message: 'stop' }
            },
          },
        },
      )) {
        // consume events
      }

      expect(steps).toHaveLength(3)
      // Step 0: Attempt 1 had 1 error, no rollback
      expect(steps[0]?.findings).toHaveLength(1)
      expect(steps[0]?.rollback).toBeUndefined()
      // Step 1: Attempt 2 regressed to 3 errors -> rolled back to Attempt 1!
      expect(steps[1]?.findings).toHaveLength(3)
      expect(steps[1]?.rollback).toMatchObject({
        restoredAttempt: 1,
        bestErrors: 1,
        lastErrors: 3,
        revertedPaths: ['src/extra.json', 'src/helper.ts', 'src/math.ts'],
      })
      expect(steps[1]?.feedback).toContain(
        '[harness] Workspace restored to best prior attempt (had 1 error(s) vs 3 in last attempt; reverted src/extra.json, src/helper.ts, src/math.ts)',
      )
      // Step 2: Attempt 3 fixed everything -> 0 findings!
      expect(steps[2]?.findings).toEqual([])

      // Pre-existing dirty user files and .git/index were never clobbered or touched!
      expect(await readFile(join(workspace, 'src/user-wip.ts'), 'utf8')).toBe(
        '// uncommitted user WIP with pre-existing syntax issue\nexport const wip = ;\n',
      )
      expect(await readFile(join(workspace, 'src/untouched-dirty.ts'), 'utf8')).toBe(
        'export const dirtyBeforeTurn = "keep-me-exact";\n',
      )
      const indexStatAfter = await stat(join(workspace, '.git/index'))
      const indexBytesAfter = new Uint8Array(await readFile(join(workspace, '.git/index')))
      expect(indexStatAfter.mtimeMs).toBe(indexStatBefore.mtimeMs)
      expect(indexBytesAfter).toEqual(indexBytesBefore)
    })
  })

  test('restores workspace and forwards rollback notice when escalating to advisor from a worse attempt, and respects rollbackOnRegression: false', async () => {
    await withTempWorkspace(async workspace => {
      const target = join(workspace, 'app.ts')
      await writeFile(target, 'export const v = 1;\n')

      const checkpoints = new WorkspaceCheckpointManager(workspace)
      await checkpoints.noteBeforeToolExecution('WriteFile', { file_path: 'app.ts' })
      await writeFile(target, 'export const v = ;\n')
      const attempt1Findings = await runResultChecks({ workspaceRoot: workspace, changedFiles: ['app.ts'] })
      expect(await checkpoints.recordAttemptAndMaybeRollback(attempt1Findings, { escalationAction: 'retry_local' })).toBeUndefined()

      // Regress to 2 files with errors when escalating to ask_advisor
      const second = join(workspace, 'second.json')
      await checkpoints.noteBeforeToolExecution('WriteFile', { file_path: 'second.json' })
      await writeFile(second, '{"bad": }\n')
      const attempt2Findings = await runResultChecks({ workspaceRoot: workspace, changedFiles: ['app.ts', 'second.json'] })
      expect(attempt2Findings).toHaveLength(2)

      const consultedContexts: string[] = []
      const state = createAgentState()
      state.toolExecutions.push(
        { toolCallId: 'w1', durationMs: 1, name: 'WriteFile', inputs: { file_path: 'app.ts' }, permitted: true, result: 'ok' },
        { toolCallId: 'w2', durationMs: 1, name: 'WriteFile', inputs: { file_path: 'second.json' }, permitted: true, result: 'ok' },
      )
      const step = await (await import('../src/runtime/cascade.js')).cascadeStep(
        state,
        0,
        workspace,
        {
          checks: () => 'enforce',
          checkpoints,
          rollbackOnRegression: true,
          escalate: () => ({
            action: 'ask_advisor',
            level: 'info',
            message: 'Consulting advisor',
            advisorRequest: { question: 'How to fix?', context: '2 check failures.' },
          }),
          consultAdvisor: async request => {
            consultedContexts.push(request.context)
            return 'Restore valid initializer.'
          },
        },
        new AbortController().signal,
      )

      expect(step.rollback?.revertedPaths).toEqual(['second.json'])
      expect(await Bun.file(second).exists()).toBe(false)
      expect(consultedContexts[0]).toContain('[harness] Workspace restored to best prior attempt (had 1 error(s) vs 2 in last attempt; reverted second.json)')
      expect(step.feedback).toContain('[harness] Workspace restored to best prior attempt (had 1 error(s) vs 2 in last attempt; reverted second.json)')

      // Opt-out check: rollbackOnRegression: false leaves regressed files in place
      await checkpoints.noteBeforeToolExecution('WriteFile', { file_path: 'second.json' })
      await writeFile(second, '{"bad": }\n')
      const disabledRollback = await checkpoints.recordAttemptAndMaybeRollback(attempt2Findings, {
        escalationAction: 'retry_local',
        rollbackEnabled: false,
      })
      expect(disabledRollback).toBeUndefined()
      expect(await Bun.file(second).exists()).toBe(true)
    })
  })
})

describe('SWE specialist agent YAMLs, system.md, and leanPrompt compatibility', () => {
  test('built-in agent profiles wire ApplyPatchTool and FileOutlineTool with SWE execution discipline', () => {
    const builtins = loadBuiltinAgentDefinitions()
    const defaultAgent = builtins.get('default')!
    const coder = builtins.get('coder')!
    const researcher = builtins.get('researcher')!
    const reviewer = builtins.get('reviewer')!
    const tester = builtins.get('tester')!

    expect(defaultAgent.tools).toEqual(expect.arrayContaining(['ApplyPatchTool', 'FileOutlineTool']))
    expect(coder.allowedTools).toEqual(expect.arrayContaining([
      'ApplyPatchTool',
      'FileOutlineTool',
      'XlsxReader',
      'XlsxWriter',
      'XlsxConvert',
    ]))
    expect(researcher.allowedTools).toContain('FileOutlineTool')
    expect(researcher.excludeTools).toContain('ApplyPatchTool')
    expect(reviewer.allowedTools).toContain('FileOutlineTool')
    expect(reviewer.excludeTools).toContain('ApplyPatchTool')
    expect(tester.allowedTools).toContain('FileOutlineTool')

    // Verify all 4 SWE execution discipline pillars in coder and default system prompts
    for (const prompt of [coder.systemPrompt, defaultAgent.systemPrompt]) {
      expect(prompt).toContain('FileOutlineTool')
      expect(prompt).toContain('GrepTool')
      expect(prompt).toContain('LSP')
      expect(prompt).toContain('ApplyPatchTool')
      expect(prompt).toContain('FileEditTool')
      expect(prompt).toContain('git diff')
      expect(prompt.toLowerCase()).toContain('reproduce')
    }
  })

  test('selectLeanTools supports includeSweTools while keeping default LEAN_CORE_TOOL_NAMES untouched', () => {
    expect(LEAN_SWE_TOOL_NAMES).toEqual(['ApplyPatchTool', 'FileOutlineTool'])
    expect(LEAN_CORE_TOOL_NAMES).not.toContain('ApplyPatchTool')
    expect(LEAN_CORE_TOOL_NAMES).not.toContain('FileOutlineTool')

    const definitions: ToolDefinition[] = [
      {
        type: 'function',
        function: { name: 'ReadFile', description: 'Read a file.', parameters: { type: 'object', properties: {} } },
      },
      {
        type: 'function',
        function: { name: 'ApplyPatchTool', description: 'Apply an atomic multi-file patch.', parameters: { type: 'object', properties: { patch: { type: 'string' } } } },
      },
      {
        type: 'function',
        function: { name: 'FileOutlineTool', description: 'Outline symbols in a source file.', parameters: { type: 'object', properties: { file_path: { type: 'string' } } } },
      },
    ]

    expect(selectLeanTools(definitions).map(tool => tool.function.name)).toEqual(['ReadFile'])
    expect(selectLeanTools(definitions, { includeSweTools: true }).map(tool => tool.function.name)).toEqual([
      'ReadFile',
      'ApplyPatchTool',
      'FileOutlineTool',
    ])

    const sweLeanPrompt = buildLeanSystemPrompt({
      cwd: '/work/repo',
      model: 'bonsai-8b',
      date: '2026-11-21',
      toolNames: ['ReadFile', 'FileOutlineTool', 'ApplyPatchTool'],
    })
    expect(sweLeanPrompt).toContain('FileOutlineTool')
    expect(sweLeanPrompt).toContain('ApplyPatchTool')
  })
})
