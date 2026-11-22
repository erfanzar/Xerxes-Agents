// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from 'bun:test'
import { existsSync } from 'node:fs'
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

import { ValidationError } from '../src/core/errors.js'
import { ToolRegistry, type ToolExecutionContext } from '../src/executors/toolRegistry.js'
import type { CompletionRequest, LlmClient, LlmDelta } from '../src/llms/client.js'
import { DaemonSwarmHost } from '../src/daemon/swarmHost.js'
import { InMemoryDaemonRuntime } from '../src/daemon/runtime.js'
import { AgentTurnRunner } from '../src/daemon/turnRunner.js'
import { BUNDLED_SKILLS_DIRECTORY, SkillRegistry } from '../src/extensions/skills.js'
import type { StreamEvent } from '../src/streaming/events.js'
import {
  createDefaultSwarmWorkspacePort,
  createRunTurnWorkerRunner,
  isQuotaOrCapacityError,
  SWARM_OPEN_BOARD_MARKDOWN,
  SwarmManager,
  type SwarmCoordinatorWakeEvent,
  type SwarmLifecycleEvent,
  type SwarmModelCatalogPort,
  type SwarmModelInfo,
} from '../src/swarm/index.js'
import {
  POST_MESSAGE_TOOL_NAME,
  READ_BOARD_TOOL_NAME,
  registerSwarmTools,
  SWARM_OVERVIEW_TOOL_NAME,
  SWARM_TOOL_DEFINITION,
  SWARM_TOOL_NAME,
  SWARM_WAIT_TOOL_NAME,
} from '../src/tools/swarmTools.js'
import type { ChatMessage } from '../src/types/messages.js'
import type { ToolCall } from '../src/types/toolCalls.js'

async function withTempDirs(
  fn: (dirs: { readonly root: string; readonly workspace: string; readonly storage: string }) => Promise<void>,
): Promise<void> {
  const root = await mkdtemp(join(tmpdir(), 'xerxes-swarm-core-'))
  const workspace = join(root, 'workspace')
  const storage = join(root, 'storage')
  await mkdir(workspace, { recursive: true })
  await mkdir(storage, { recursive: true })
  try {
    await fn({ root, workspace, storage })
  } finally {
    await rm(root, { recursive: true, force: true })
  }
}

async function initGitRepo(dir: string): Promise<void> {
  const run = async (args: readonly string[]) => {
    const proc = Bun.spawn(['git', ...args], { cwd: dir, stdout: 'pipe', stderr: 'pipe' })
    const [code, stderr] = await Promise.all([proc.exited, new Response(proc.stderr).text()])
    if (code !== 0) {
      throw new Error(`git ${args.join(' ')} failed: ${stderr}`)
    }
  }
  await run(['init'])
  await run(['config', 'user.email', 'swarm-test@xerxes.local'])
  await run(['config', 'user.name', 'Xerxes Swarm Test'])
  await writeFile(join(dir, 'README.md'), '# Test Repo\n', 'utf8')
  await run(['add', 'README.md'])
  await run(['commit', '-m', 'chore: initial commit'])
}

function makeToolContext(agentId?: string): ToolExecutionContext {
  return {
    ...(agentId === undefined ? {} : { agentId }),
    metadata: agentId ? { swarmMember: agentId } : {},
  }
}

function makeCall(id: string, name: string, args: Record<string, unknown>): ToolCall {
  return {
    id,
    type: 'function',
    function: {
      name,
      arguments: args as ToolCall['function']['arguments'],
    },
  }
}

describe('Native Swarm Core — Board, Channels, Roster & Ownership', () => {
  test('parses @mentions, @#channels, @all broadcast, refuses @#main, and supports read_board modes + feedback sanitization', async () => {
    await withTempDirs(async ({ workspace, storage }) => {
      const events: SwarmLifecycleEvent[] = []
      const swarm = await SwarmManager.open({
        sessionId: 'sess-board-1',
        workspaceDir: workspace,
        storageDir: storage,
        coordinatorModel: 'model-alpha',
        coordinatorModelDisplayName: 'Model Alpha',
        onEvent: event => events.push(event),
      })

      const titled = await swarm.setTitle({ caller: 'coordinator', title: 'Core Engine Verification' })
      expect(titled.title).toBe('Core Engine Verification')
      expect(titled.open_board).toBe(SWARM_OPEN_BOARD_MARKDOWN)

      await expect(swarm.setTitle({ caller: 'worker-a', title: 'Hijack' })).rejects.toThrow(ValidationError)

      await swarm.channelAction({ caller: 'coordinator', op: 'create', channel: 'backend', topic: 'Backend API slice' })
      await swarm.add({
        caller: 'coordinator',
        name: 'worker-a',
        brief: 'Own backend routes.',
        channels: ['backend'],
      })
      await swarm.add({
        caller: 'coordinator',
        name: 'narrator',
        brief: 'Curate the live swarm overview.',
      })

      // Refuses @#main since #main includes everyone (must use @all).
      await expect(
        swarm.postMessage({ caller: 'coordinator', text: 'Hey @#main check this out' }),
      ).rejects.toThrow(ValidationError)

      const posted = await swarm.postMessage({
        caller: 'coordinator',
        channel: 'backend',
        text: 'Update for @worker-a and @#backend',
      })
      expect(posted.mentions).toEqual(['worker-a'])
      expect(posted.channel_mentions).toEqual(['backend'])
      expect(posted.broadcast).toBe(false)

      const broadcast = await swarm.postMessage({
        caller: 'worker-a',
        text: '@all backend schema is ready (see #3)',
      })
      expect(broadcast.broadcast).toBe(true)

      // User feedback formatting strips @ before member names, @all, and @#channel.
      const feedback = await swarm.postFromUser({
        text: 'Please summarize @worker-a progress and @all items in @#backend',
        as_feedback: true,
      })
      expect(feedback.message.text).toBe(
        '@narrator user feedback: Please summarize worker-a progress and all items in #backend',
      )
      expect(feedback.mentions).toEqual(['narrator'])
      expect(feedback.broadcast).toBe(false)
      expect(feedback.channel_mentions).toEqual([])

      // read_board with seqs ignores since/limit and omits missing seqs.
      const bySeqs = await swarm.readBoard({ caller: 'worker-a', seqs: [1, 3, 999] })
      expect('messages' in bySeqs && bySeqs.messages.map(m => m.seq)).toEqual([1, 3])

      // read_board with status:true returns live roster snapshot.
      const rosterBoard = await swarm.readBoard({ caller: 'coordinator', status: true })
      expect('members' in rosterBoard).toBe(true)
      if ('members' in rosterBoard) {
        const workerEntry = rosterBoard.members.find(m => m.name === 'worker-a')
        expect(workerEntry?.last_post_seq).toBe(broadcast.seq)
        expect(workerEntry?.last_post_preview).toContain('backend schema is ready')
      }

      // Narrator updates overview; non-narrators are rejected.
      await expect(
        swarm.updateOverview({ caller: 'worker-a', markdown: '## Unauthorized' }),
      ).rejects.toThrow(ValidationError)

      const overviewRes = await swarm.updateOverview({
        caller: 'narrator',
        markdown: '## Progress\n- Backend schema landed (#4)',
        summary: 'Backend schema complete',
      })
      expect(overviewRes.overview.summary).toBe('Backend schema complete')
      expect(swarm.status().overview?.author).toBe('narrator')
      expect(events.some(e => e.kind === 'overview_updated')).toBe(true)

      await swarm.dispose()
    })
  })

  test('enforces channel visibility, subscription ownership, and archive read-only semantics', async () => {
    await withTempDirs(async ({ workspace, storage }) => {
      const swarm = await SwarmManager.open({
        sessionId: 'sess-channels',
        workspaceDir: workspace,
        storageDir: storage,
      })

      await swarm.add({ caller: 'coordinator', name: 'parent-w', brief: 'Lead author' })
      await swarm.add({ caller: 'coordinator', name: 'sibling-w', brief: 'Unrelated author' })
      await swarm.add({ caller: 'parent-w', name: 'child-w', brief: 'Helper owned by parent-w' })
      await swarm.add({ caller: 'coordinator', name: 'monitor', brief: 'Stall monitor' })

      await swarm.channelAction({ caller: 'parent-w', op: 'create', channel: 'design', topic: 'Design specs' })

      // parent-w can subscribe its own helper child-w, but NOT sibling-w!
      const subChild = await swarm.channelAction({
        caller: 'parent-w',
        op: 'subscribe',
        channel: 'design',
        name: 'child-w',
      })
      expect('members' in subChild && subChild.members).toEqual(['child-w'])

      await expect(
        swarm.channelAction({
          caller: 'parent-w',
          op: 'subscribe',
          channel: 'design',
          name: 'sibling-w',
        }),
      ).rejects.toThrow(ValidationError)

      // Post in #design: visible to parent-w, child-w, coordinator, monitor (*), invisible to sibling-w!
      const designPost = await swarm.postMessage({
        caller: 'parent-w',
        channel: 'design',
        text: '@child-w review RFC in #design',
      })

      const siblingRead = await swarm.readBoard({ caller: 'sibling-w', since: 0 })
      expect('messages' in siblingRead && siblingRead.messages.some(m => m.seq === designPost.seq)).toBe(false)
      await expect(
        swarm.readBoard({ caller: 'sibling-w', channel: 'design' }),
      ).rejects.toThrow(ValidationError)

      const monitorRead = await swarm.readBoard({ caller: 'monitor', channel: 'design' })
      expect('messages' in monitorRead && monitorRead.messages.map(m => m.seq)).toEqual([designPost.seq])

      // Archive #design: rejects new posts while keeping history readable!
      await expect(
        swarm.channelAction({ caller: 'sibling-w', op: 'archive', channel: 'design' }),
      ).rejects.toThrow(ValidationError)

      const archived = await swarm.channelAction({ caller: 'parent-w', op: 'archive', channel: 'design' })
      expect('status' in archived && archived.status).toBe('archived')

      await expect(
        swarm.postMessage({ caller: 'parent-w', channel: 'design', text: 'Late post' }),
      ).rejects.toThrow(ValidationError)

      const historyAfterArchive = await swarm.readBoard({ caller: 'child-w', channel: 'design' })
      expect('messages' in historyAfterArchive && historyAfterArchive.messages.map(m => m.seq)).toEqual([designPost.seq])

      await swarm.dispose()
    })
  })

  test('supports n homogeneous copies, tasks sharding, group wakeup/kill, helper lineage survival, and refuses killed name reuse', async () => {
    await withTempDirs(async ({ workspace, storage }) => {
      const swarm = await SwarmManager.open({
        sessionId: 'sess-sharding',
        workspaceDir: workspace,
        storageDir: storage,
      })

      // Reject reserved names and invalid characters.
      await expect(swarm.add({ name: 'coordinator', brief: 'bad' })).rejects.toThrow(ValidationError)
      await expect(swarm.add({ name: 'coordinator-1', brief: 'bad' })).rejects.toThrow(ValidationError)
      await expect(swarm.add({ name: 'all', brief: 'bad' })).rejects.toThrow(ValidationError)
      await expect(swarm.add({ name: 'UpperCase', brief: 'bad' })).rejects.toThrow(ValidationError)

      // Reject setting both n > 1 and tasks.
      await expect(
        swarm.add({ name: 'shard', brief: 'Audit files', n: 2, tasks: ['a.ts', 'b.ts'] }),
      ).rejects.toThrow(ValidationError)

      // Homogeneous pool n = 3
      const pool = await swarm.add({ name: 'reviewer', brief: 'Review PR chunks', n: 3 })
      expect(pool.added.map(m => m.name)).toEqual(['reviewer-1', 'reviewer-2', 'reviewer-3'])
      expect(pool.added.every(m => m.group === 'reviewer')).toBe(true)

      // @reviewer wakes all 3 copies in the group!
      const groupPost = await swarm.postMessage({
        caller: 'coordinator',
        text: '@reviewer check updated diff',
      })
      expect(groupPost.woken.sort()).toEqual(['reviewer-1', 'reviewer-2', 'reviewer-3'])

      // Tasks sharded pool
      const sharded = await swarm.add({
        name: 'auditor',
        brief: 'Audit module for edge cases.',
        tasks: ['src/auth.ts', 'src/db.ts'],
      })
      expect(sharded.added.map(m => m.name)).toEqual(['auditor-1', 'auditor-2'])
      const shardBriefs = await swarm.readBoard({ seqs: sharded.brief_seqs })
      if ('messages' in shardBriefs) {
        expect(shardBriefs.messages[0]?.text).toContain('Assigned slice (1/2): src/auth.ts')
        expect(shardBriefs.messages[1]?.text).toContain('Assigned slice (2/2): src/db.ts')
      }

      // auditor-1 spawns a helper; killing auditor-1 must NOT kill its helper!
      await swarm.add({ caller: 'auditor-1', name: 'sub-helper', brief: 'Check SQL queries' })
      await expect(swarm.kill({ caller: 'auditor-2', name: 'sub-helper' })).rejects.toThrow(ValidationError)

      // Group kill kills all living copies of reviewer!
      const killedGroup = await swarm.kill({ caller: 'coordinator', name: 'reviewer' })
      expect(killedGroup.killed).toEqual(['reviewer-1', 'reviewer-2', 'reviewer-3'])

      // Kill auditor-1: sub-helper remains alive!
      await swarm.kill({ caller: 'coordinator', name: 'auditor-1' })
      expect(swarm.getMember('auditor-1')?.killed).toBe(true)
      expect(swarm.getMember('sub-helper')?.killed).toBe(false)

      // Killed names cannot be reused!
      await expect(
        swarm.add({ caller: 'coordinator', name: 'auditor-1', brief: 'Try reusing killed name' }),
      ).rejects.toThrow(/cannot be reused/)

      // Numbered replacement succeeds!
      const narrator1 = await swarm.add({ caller: 'coordinator', name: 'narrator', brief: 'Overview v1' })
      expect(narrator1.added[0]?.role).toBe('narrator')
      await swarm.kill({ caller: 'coordinator', name: 'narrator' })
      await expect(
        swarm.add({ caller: 'coordinator', name: 'narrator', brief: 'Reuse killed narrator' }),
      ).rejects.toThrow(/narrator-2/)
      const narrator2 = await swarm.add({ caller: 'coordinator', name: 'narrator-2', brief: 'Overview v2' })
      expect(narrator2.added[0]?.role).toBe('narrator')

      await swarm.dispose()
    })
  })
})

describe('Native Swarm Core — Workspaces, Forking, Model Switching & Reload Revival', () => {
  test('allocates share git worktrees and branch clones, isolating edits and cleaning up on kill while preserving inherit', async () => {
    await withTempDirs(async ({ root, workspace, storage }) => {
      await initGitRepo(workspace)
      const wsRoot = join(root, 'allocations')
      const swarm = await SwarmManager.open({
        sessionId: 'sess-workspaces',
        workspaceDir: workspace,
        storageDir: storage,
        workspacePort: createDefaultSwarmWorkspacePort({ rootDir: wsRoot }),
      })

      const inherited = await swarm.add({ name: 'w-inherit', brief: 'Read-only check', workspace: 'inherit' })
      const shared = await swarm.add({ name: 'w-share', brief: 'Edit in git worktree', workspace: 'share' })
      const branched = await swarm.add({ name: 'w-branch', brief: 'Edit in local clone', workspace: 'branch' })

      const inheritPath = inherited.added[0]!.workspace_path!
      const sharePath = shared.added[0]!.workspace_path!
      const branchPath = branched.added[0]!.workspace_path!

      expect(inheritPath).toBe(workspace)
      expect(sharePath).not.toBe(workspace)
      expect(branchPath).not.toBe(workspace)
      expect(existsSync(join(sharePath, 'README.md'))).toBe(true)
      expect(existsSync(join(branchPath, 'README.md'))).toBe(true)

      // Isolated writes inside share & branch do not touch main workspace checkout.
      await writeFile(join(sharePath, 'README.md'), '# Share Edit\n', 'utf8')
      await writeFile(join(branchPath, 'README.md'), '# Branch Edit\n', 'utf8')
      expect(await readFile(join(workspace, 'README.md'), 'utf8')).toBe('# Test Repo\n')

      // Kill removes share worktree & branch clone, while inherit workspace stays intact!
      await swarm.kill({ caller: 'coordinator', name: 'w-share' })
      await swarm.kill({ caller: 'coordinator', name: 'w-branch' })
      await swarm.kill({ caller: 'coordinator', name: 'w-inherit' })

      expect(existsSync(sharePath)).toBe(false)
      expect(existsSync(branchPath)).toBe(false)
      expect(existsSync(join(workspace, 'README.md'))).toBe(true)

      await swarm.dispose()
    })
  })

  test('validates model quota, clones warm fork history (including mid-tool-call repair), and switches models preserving history', async () => {
    await withTempDirs(async ({ workspace, storage }) => {
      const models: SwarmModelInfo[] = [
        { id: 'pro-v1', display_name: 'Pro V1', suggested: true, quota_remaining: 80 },
        { id: 'exhausted-v1', display_name: 'Exhausted V1', suggested: true, quota_exhausted: true },
        { id: 'exhausted-credited', display_name: 'Credited V1', suggested: false, quota_exhausted: true, credits_available: true },
      ]
      const catalog: SwarmModelCatalogPort = {
        listModels: () => models,
      }

      const coordinatorHistory: ChatMessage[] = [
        { role: 'system', content: 'Coordinator system prompt' },
        { role: 'user', content: 'We need to migrate auth tokens to SHA-256.' },
        {
          role: 'assistant',
          content: 'Spawning warm worker with context.',
          tool_calls: [makeCall('call_in_flight', 'swarm', { action: 'add', name: 'warm-worker', fork: 'coordinator' })],
        },
      ]

      const observedRuns: Array<{ readonly member: string; readonly model: string; readonly reason: string; readonly messageCount: number }> = []

      const swarm = await SwarmManager.open({
        sessionId: 'sess-fork-model',
        workspaceDir: workspace,
        storageDir: storage,
        coordinatorModel: 'pro-v1',
        coordinatorModelDisplayName: 'Pro V1',
        modelCatalog: catalog,
        forkMessages: name => (name === 'coordinator' ? coordinatorHistory : undefined),
        workerRunner: async (input) => {
          observedRuns.push({
            member: input.member.name,
            model: input.member.model,
            reason: input.reason,
            messageCount: input.messages.length,
          })
          return {
            messages: [
              ...input.messages,
              { role: 'user', content: input.prompt },
              { role: 'assistant', content: `Completed step on ${input.member.model}` },
            ],
          }
        },
      })

      // Suggested vs all models
      expect((await swarm.listModels({ all: false })).models.map(m => m.id)).toEqual(['pro-v1', 'exhausted-v1'])
      expect((await swarm.listModels({ all: true })).models.map(m => m.id)).toEqual([
        'pro-v1',
        'exhausted-v1',
        'exhausted-credited',
      ])

      // Refuse quota-exhausted model without fallback credits, allow one covered by credits.
      await expect(
        swarm.add({ name: 'blocked-w', brief: 'Try exhausted model', model: 'Exhausted V1' }),
      ).rejects.toThrow(/exhausted its quota/)

      await swarm.add({
        name: 'warm-worker',
        brief: 'Implement SHA-256 token migration.',
        fork: 'coordinator',
      })
      await swarm.waitForIdle()

      // Fork stripped system prompt and repaired in-flight tool_call with synthetic tool_result (3 messages before startup turn).
      expect(observedRuns[0]).toEqual({
        member: 'warm-worker',
        model: 'pro-v1',
        reason: 'startup',
        messageCount: 3,
      })

      // Switch warm-worker to Credited V1 by display name: preserves full history (5 messages) and resumes!
      const switched = await swarm.switchModel({
        caller: 'coordinator',
        name: 'warm-worker',
        model: 'Credited V1',
      })
      expect(switched.model).toBe('exhausted-credited')
      expect(switched.resumed).toBe(true)
      await swarm.waitForIdle()

      expect(observedRuns[1]).toEqual({
        member: 'warm-worker',
        model: 'exhausted-credited',
        reason: 'model_switch',
        messageCount: 5,
      })

      await swarm.dispose()
    })
  })

  test('restores non-killed workers in stopped state after reload and revives them automatically when named on the board', async () => {
    await withTempDirs(async ({ workspace, storage }) => {
      const first = await SwarmManager.open({
        sessionId: 'sess-reload',
        workspaceDir: workspace,
        storageDir: storage,
        workerRunner: async input => ({
          messages: [
            ...input.messages,
            { role: 'user', content: input.prompt },
            { role: 'assistant', content: 'Initial work saved.' },
          ],
        }),
      })
      await first.setTitle({ caller: 'coordinator', title: 'Persistent Swarm' })
      await first.add({ name: 'survivor', brief: 'Stay ready across restarts' })
      await first.add({ name: 'retired', brief: 'Will be killed before restart' })
      await first.waitForIdle()
      await first.kill({ caller: 'coordinator', name: 'retired' })
      await first.dispose()

      // Reload from disk in a new SwarmManager instance!
      const revivedRuns: Array<{ readonly reason: string; readonly priorMessages: number }> = []
      const reloaded = await SwarmManager.open({
        sessionId: 'sess-reload',
        workspaceDir: workspace,
        storageDir: storage,
        workerRunner: async (input) => {
          revivedRuns.push({ reason: input.reason, priorMessages: input.messages.length })
          return {
            messages: [
              ...input.messages,
              { role: 'user', content: input.prompt },
              { role: 'assistant', content: 'Revived and finished!' },
            ],
          }
        },
      })

      expect(reloaded.status().title).toBe('Persistent Swarm')
      expect(reloaded.getMember('survivor')?.state).toBe('stopped')
      expect(reloaded.getMember('retired')?.state).toBe('killed')

      // Unaddressed chatter does NOT revive stopped workers.
      await reloaded.postMessage({ caller: 'coordinator', text: 'General status note without mentions' })
      expect(reloaded.getMember('survivor')?.state).toBe('stopped')
      expect(revivedRuns).toHaveLength(0)

      // Naming @survivor revives it automatically with its persisted history!
      await reloaded.postMessage({ caller: 'coordinator', text: '@survivor please run final check' })
      await reloaded.waitForIdle()

      expect(revivedRuns).toEqual([{ reason: 'revive', priorMessages: 2 }])
      expect(reloaded.getMember('survivor')?.state).toBe('waiting')

      await reloaded.dispose()
    })
  })
})

describe('Native Swarm Core — End-to-End Reactive runTurn + swarm_wait + Tool Unlocking', () => {
  test('drives real Xerxes runTurn workers with SwarmParkSignal zero-polling handoffs, dynamic coordinator tool unlocking, and onCoordinatorWake', async () => {
    await withTempDirs(async ({ workspace, storage }) => {
      let authorModelCalls = 0
      let verifierModelCalls = 0

      const scriptedLlm = (memberName: string): LlmClient => ({
        async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
          if (memberName === 'author') {
            authorModelCalls += 1
            if (authorModelCalls === 1) {
              // Author posts handoff to @verifier AND parks with swarm_wait in the SAME round!
              yield {
                toolCalls: [
                  makeCall('author_post_1', POST_MESSAGE_TOOL_NAME, {
                    text: '@verifier implementation ready for inspection',
                  }),
                  makeCall('author_wait_1', SWARM_WAIT_TOOL_NAME, {
                    since: 2,
                  }),
                ],
                usage: { inputTokens: 10, outputTokens: 12 },
              }
              return
            }
            // Should never reach round 2 during startup turn thanks to SwarmParkSignal!
            throw new Error(`Unexpected extra LLM call (${authorModelCalls}) on parked author`)
          }

          if (memberName === 'verifier') {
            verifierModelCalls += 1
            if (verifierModelCalls === 1) {
              // Verifier Startup: parks immediately waiting for @verifier handoff (or wakes if already posted!).
              const lastMsg = request.messages.at(-1)
              const text = typeof lastMsg?.content === 'string' ? lastMsg.content : ''
              if (text.includes('New board activity')) {
                yield {
                  toolCalls: [
                    makeCall('verifier_read_1', READ_BOARD_TOOL_NAME, { since: 2 }),
                  ],
                  usage: { inputTokens: 10, outputTokens: 8 },
                }
                return
              }
              yield {
                toolCalls: [
                  makeCall('verifier_wait_init', SWARM_WAIT_TOOL_NAME, { since: 1 }),
                ],
                usage: { inputTokens: 8, outputTokens: 6 },
              }
              return
            }
            if (verifierModelCalls === 2) {
              const lastRole = request.messages.at(-1)?.role
              if (lastRole === 'user') {
                yield {
                  toolCalls: [
                    makeCall('verifier_read_2', READ_BOARD_TOOL_NAME, { since: 2 }),
                  ],
                  usage: { inputTokens: 10, outputTokens: 8 },
                }
                return
              }
              yield {
                toolCalls: [
                  makeCall('verifier_done', POST_MESSAGE_TOOL_NAME, {
                    text: '@coordinator [done] Verified author changes cleanly (#3)',
                  }),
                  makeCall('verifier_park', SWARM_WAIT_TOOL_NAME, {
                    since: 3,
                  }),
                ],
                usage: { inputTokens: 12, outputTokens: 14 },
              }
              return
            }
            if (verifierModelCalls === 3) {
              yield {
                toolCalls: [
                  makeCall('verifier_done_3', POST_MESSAGE_TOOL_NAME, {
                    text: '@coordinator [done] Verified author changes cleanly (#3)',
                  }),
                  makeCall('verifier_park_3', SWARM_WAIT_TOOL_NAME, {
                    since: 3,
                  }),
                ],
                usage: { inputTokens: 12, outputTokens: 14 },
              }
              return
            }
          }

          yield { content: 'Done.', usage: { inputTokens: 4, outputTokens: 2 } }
        },
      })

      const idleCoordinatorWakes: SwarmCoordinatorWakeEvent[] = []
      const swarm = await SwarmManager.open({
        sessionId: 'sess-e2e-reactive',
        workspaceDir: workspace,
        storageDir: storage,
        onCoordinatorWake: ev => idleCoordinatorWakes.push(ev),
      })
      swarm.setWorkerRunner(
        createRunTurnWorkerRunner({
          llm: member => scriptedLlm(member.name),
        }),
      )

      // Verify dynamic coordinator tool visibility (one-turn-late unlocking).
      const coordRegistry = new ToolRegistry({ onDuplicateRegistration: () => undefined })
      const coordReg = registerSwarmTools(coordRegistry, { swarm, caller: 'coordinator' })
      expect(coordRegistry.definitions().map(d => d.function.name)).toEqual([SWARM_TOOL_NAME])
      expect(coordReg.visibleDefinitions({ turnStartedActive: false }).map(d => d.function.name)).toEqual([
        SWARM_TOOL_NAME,
      ])

      // Verify narrator worker gets all 5 tools including swarm_overview.
      const narratorRegistry = new ToolRegistry({ onDuplicateRegistration: () => undefined })
      const narratorReg = registerSwarmTools(narratorRegistry, { swarm, caller: 'narrator', agentId: 'narrator' })
      expect(narratorRegistry.definitions('narrator').map(d => d.function.name)).toEqual([
        SWARM_TOOL_NAME,
        POST_MESSAGE_TOOL_NAME,
        READ_BOARD_TOOL_NAME,
        SWARM_WAIT_TOOL_NAME,
        SWARM_OVERVIEW_TOOL_NAME,
      ])
      narratorReg.dispose()

      // Start coordinator reactive wait BEFORE spawning workers, or spawnverifier + author and await swarm_wait!
      await coordRegistry.execute(
        makeCall('c_add_verifier', SWARM_TOOL_NAME, {
          action: 'add',
          name: 'verifier',
          brief: 'Wait for @verifier handoff, inspect, and report [done] to @coordinator.',
        }),
        makeToolContext(),
      )

      // Once first `add` ran, registry dynamically unlocked board tools for Turn 2, while Turn 1 (`turnStartedActive: false`) stays `[swarm]`!
      expect(coordReg.visibleDefinitions({ turnStartedActive: false }).map(d => d.function.name)).toEqual([
        SWARM_TOOL_NAME,
      ])
      expect(coordRegistry.definitions().map(d => d.function.name)).toEqual([
        SWARM_TOOL_NAME,
        POST_MESSAGE_TOOL_NAME,
        READ_BOARD_TOOL_NAME,
        SWARM_WAIT_TOOL_NAME,
      ])

      const waitPromise = swarm.waitForBoard({ caller: 'coordinator', since: 2, mentions_only: true })

      await coordRegistry.execute(
        makeCall('c_add_author', SWARM_TOOL_NAME, {
          action: 'add',
          name: 'author',
          brief: 'Build feature and hand off to @verifier.',
        }),
        makeToolContext(),
      )

      const waitOutcome = await waitPromise
      expect(waitOutcome.status).toBe('woken')
      if (waitOutcome.status === 'woken') {
        expect(waitOutcome.message).toContain('[board] the latest message is #')
      }

      await swarm.waitForIdle()

      // Author made ONLY 1 LLM request and parked cleanly via SwarmParkSignal with valid paired tool messages!
      expect(authorModelCalls).toBe(1)
      expect(swarm.getMember('author')?.state).toBe('waiting')
      expect(swarm.getMember('verifier')?.state).toBe('waiting')
      const authorHistory = swarm.getMemberHistory('author')
      const toolResults = authorHistory.filter(m => m.role === 'tool')
      expect(toolResults).toHaveLength(2)

      // Now with coordinator NOT inside swarm_wait, an @coordinator post triggers onCoordinatorWake!
      expect(idleCoordinatorWakes).toHaveLength(0)
      await swarm.postMessage({
        caller: 'author',
        text: '@coordinator follow-up metrics attached',
      })
      expect(idleCoordinatorWakes).toHaveLength(1)
      expect(idleCoordinatorWakes[0]?.notice).toContain('[board] the latest message is #')

      coordReg.dispose()
      await swarm.dispose()
    })
  })

  test('covers coordinator swarm_wait AbortSignal & wakeSignal steer, mentions_only:false, @#channel wakeups, non-git branch fallback, and worker model ownership rules', async () => {
    await withTempDirs(async ({ root, workspace, storage }) => {
      // Non-git workspace verifies `branch` recursive-copy fallback and cleanup.
      await writeFile(join(workspace, 'hello.txt'), 'original\n', 'utf8')
      const models: SwarmModelInfo[] = [
        { id: 'base-m', display_name: 'Base M', suggested: true },
        { id: 'alt-m', display_name: 'Alt M', suggested: true },
      ]
      const swarm = await SwarmManager.open({
        sessionId: 'sess-edge-cases',
        workspaceDir: workspace,
        storageDir: storage,
        coordinatorModel: 'base-m',
        coordinatorModelDisplayName: 'Base M',
        modelCatalog: { listModels: () => models },
        workspacePort: createDefaultSwarmWorkspacePort({ rootDir: join(root, 'ws-copies') }),
      })

      // 1. Coordinator swarm_wait cancellation via AbortSignal
      const cancelCtrl = new AbortController()
      const cancelledWait = swarm.waitForBoard({
        caller: 'coordinator',
        since: 0,
        signal: cancelCtrl.signal,
      })
      cancelCtrl.abort()
      await expect(cancelledWait).rejects.toThrow(/swarm_wait cancelled/)

      // 2. Coordinator swarm_wait early return via wakeSignal (user steer)
      const steerCtrl = new AbortController()
      const steeredWait = swarm.waitForBoard({
        caller: 'coordinator',
        since: 0,
        wakeSignal: steerCtrl.signal,
      })
      steerCtrl.abort()
      const steerOutcome = await steeredWait
      expect(steerOutcome.status).toBe('interrupted_by_steer')

      // 3. Startup brief containing `@#main`, `@all`, and `@peer` prose does not reject or prematurely wake peers
      await swarm.add({ name: 'peer-idle', brief: 'Wait quietly until named.' })
      const idleAfterAdd = await swarm.waitForBoard({ caller: 'peer-idle', since: 1, mentions_only: true })
      expect(idleAfterAdd.status).toBe('parked')

      await swarm.add({
        name: 'branch-author',
        brief: 'Follow rules: never post @#main or @all casually; ping @peer-idle only when ready.',
        workspace: 'branch',
      })
      // peer-idle was NOT woken by branch-author's brief prose!
      const checkStillParked = await swarm.waitForBoard({ caller: 'peer-idle', since: 1, mentions_only: true })
      expect(checkStillParked.status).toBe('parked')

      const branchDir = swarm.getMember('branch-author')!.workspace_path!
      expect(await readFile(join(branchDir, 'hello.txt'), 'utf8')).toBe('original\n')
      await writeFile(join(branchDir, 'hello.txt'), 'isolated branch copy\n', 'utf8')
      expect(await readFile(join(workspace, 'hello.txt'), 'utf8')).toBe('original\n')

      // 4. mentions_only:false hunter wakes on unaddressed #main posts, ignoring private channels
      await swarm.channelAction({ caller: 'coordinator', op: 'create', channel: 'ops', topic: 'Ops' })
      await swarm.channelAction({ caller: 'coordinator', op: 'create', channel: 'secret', topic: 'Secret' })
      await swarm.add({ name: 'hunter', brief: 'Pick up unassigned work' })
      await swarm.add({ name: 'ops-watcher', brief: 'Watch ops channel', channels: ['ops'] })

      const headBefore = swarm.head()
      await swarm.waitForBoard({ caller: 'hunter', since: headBefore, mentions_only: false })
      await swarm.waitForBoard({ caller: 'ops-watcher', since: headBefore, mentions_only: true })

      // Post in #secret does NOT wake hunter (not subscribed to #secret).
      await swarm.postMessage({ caller: 'coordinator', channel: 'secret', text: 'Internal note' })
      const hunterStillParked = await swarm.waitForBoard({
        caller: 'hunter',
        since: headBefore,
        mentions_only: false,
      })
      expect(hunterStillParked.status).toBe('parked')

      // Unaddressed post in #main DOES wake hunter (mentions_only: false) AND @#ops wakes ops-watcher!
      const opsPing = await swarm.postMessage({
        caller: 'coordinator',
        text: 'Deployment queued for @#ops',
      })
      expect(opsPing.woken).toContain('hunter')
      expect(opsPing.woken).toContain('ops-watcher')

      // 5. Worker model switching ownership, working-state guard, self-switch, and helper default model
      await swarm.switchModel({ caller: 'branch-author', model: 'alt-m' })
      expect(swarm.getMember('branch-author')?.model).toBe('alt-m')
      await expect(swarm.switchModel({ caller: 'branch-author' })).rejects.toThrow(ValidationError)

      await swarm.add({ caller: 'branch-author', name: 'branch-helper', brief: 'Sub-slice' })
      // Helper inherits branch-author's model ('alt-m'); sibling 'hunter' cannot switch branch-helper!
      expect(swarm.getMember('branch-helper')?.model).toBe('alt-m')
      await expect(
        swarm.switchModel({ caller: 'hunter', name: 'branch-helper', model: 'base-m' }),
      ).rejects.toThrow(ValidationError)

      // Coordinator switches branch-helper omitting `model` -> defaults to coordinator's model ('base-m')!
      const coordDefaultSwitch = await swarm.switchModel({
        caller: 'coordinator',
        name: 'branch-helper',
      })
      expect(coordDefaultSwitch.model).toBe('base-m')

      // Refuses switching another member while its state is 'working'
      const blockingTurn = Promise.withResolvers<void>()
      const turnStarted = Promise.withResolvers<void>()
      swarm.setWorkerRunner(async (input) => {
        turnStarted.resolve()
        await blockingTurn.promise
        return { messages: input.messages }
      })
      await swarm.add({ name: 'busy-worker', brief: 'Long computation' })
      await turnStarted.promise
      expect(swarm.getMember('busy-worker')?.state).toBe('working')
      await expect(
        swarm.switchModel({ caller: 'coordinator', name: 'busy-worker', model: 'alt-m' }),
      ).rejects.toThrow(/currently working/)
      blockingTurn.resolve()
      await swarm.waitForIdle()

      // Cleanup non-git branch directory via killFromHost('all')
      const killedAll = await swarm.killFromHost('all')
      expect(killedAll.killed).toContain('branch-author')
      expect(killedAll.killed).not.toContain('coordinator')
      expect(existsSync(branchDir)).toBe(false)
      expect(existsSync(join(workspace, 'hello.txt'))).toBe(true)

      await swarm.dispose()
    })
  })
})

describe('Native Swarm Core — 1:1 Antigravity Parity Upgrades', () => {
  test('tracks coordinator live state in status() and readBoard({ status: true }), preserves worker-only invariants, and parks coordinator during swarm_wait', async () => {
    await withTempDirs(async ({ workspace, storage }) => {
      const events: SwarmLifecycleEvent[] = []
      const swarm = await SwarmManager.open({
        sessionId: 'sess-coord-roster',
        workspaceDir: workspace,
        storageDir: storage,
        coordinatorModel: 'model-coord',
        coordinatorModelDisplayName: 'Model Coord',
        onEvent: event => events.push(event),
      })

      // Before any worker is added: isActive() is false, getMember('coordinator') is undefined,
      // yet status().members includes coordinator and tail_from_seq is 1.
      expect(swarm.isActive()).toBe(false)
      expect(swarm.getMember('coordinator')).toBeUndefined()
      const coldStatus = swarm.status()
      expect(coldStatus.tail_from_seq).toBe(1)
      expect(coldStatus.members).toHaveLength(1)
      expect(coldStatus.members[0]).toMatchObject({
        name: 'coordinator',
        role: 'coordinator',
        model: 'model-coord',
        model_display_name: 'Model Coord',
        workspace: 'inherit',
        workspace_path: workspace,
        channels: ['*'],
        state: 'waiting',
        killed: false,
        conversation_id: 'sess-coord-roster:swarm:coordinator',
        last_read_seq: 0,
      })

      // setCoordinatorState before swarm is active updates state silently (no member_updated).
      swarm.setCoordinatorState('working')
      expect(swarm.status().members[0]?.state).toBe('working')
      expect(events.filter(e => e.kind === 'member_updated')).toHaveLength(0)

      // Add a worker: now swarm is active and add result includes conversation_id + open_board note.
      const added = await swarm.add({ caller: 'coordinator', name: 'worker-1', brief: 'Build parser.' })
      expect(swarm.isActive()).toBe(true)
      expect(added.added[0]?.conversation_id).toBe('sess-coord-roster:swarm:worker-1')
      expect(added.note).toContain(SWARM_OPEN_BOARD_MARKDOWN)

      const coordAfterAdd = swarm.status().members.find(m => m.name === 'coordinator')
      expect(coordAfterAdd?.last_post_seq).toBe(added.brief_seqs[0])
      expect(coordAfterAdd?.last_post_time).toBeDefined()

      // Posting as coordinator updates last_post_seq / last_post_preview in readBoard({ status: true }).
      const coordPost = await swarm.postMessage({ caller: 'coordinator', text: 'Architecture note for swarm' })
      const roster = await swarm.readBoard({ caller: 'worker-1', status: true })
      expect('members' in roster).toBe(true)
      if ('members' in roster) {
        const coordRow = roster.members.find(m => m.name === 'coordinator')
        expect(coordRow).toMatchObject({
          name: 'coordinator',
          role: 'coordinator',
          state: 'working',
          last_post_seq: coordPost.seq,
          last_post_preview: 'Architecture note for swarm',
        })
      }

      // Blocking coordinator swarm_wait sets state -> 'waiting' with parked_since/mentions_only,
      // emits member_updated, and restores 'working' in finally when woken!
      const waitPromise = swarm.waitForBoard({
        caller: 'coordinator',
        since: coordPost.seq,
        mentions_only: true,
      })
      const whileWaiting = swarm.status().members.find(m => m.name === 'coordinator')
      expect(whileWaiting?.state).toBe('waiting')
      expect(whileWaiting?.parked_since).toBe(coordPost.seq)
      expect(whileWaiting?.mentions_only).toBe(true)

      await swarm.postMessage({ caller: 'worker-1', text: '@coordinator [done] Parser complete' })
      const waitRes = await waitPromise
      expect(waitRes.status).toBe('woken')

      const afterWake = swarm.status().members.find(m => m.name === 'coordinator')
      expect(afterWake?.state).toBe('working')
      expect(afterWake?.parked_since).toBeUndefined()
      expect(afterWake?.mentions_only).toBeUndefined()

      // killFromHost('all') kills workers only, never coordinator.
      const killedAll = await swarm.killFromHost('all')
      expect(killedAll.killed).toEqual(['worker-1'])
      expect(swarm.status().members.find(m => m.name === 'coordinator')?.killed).toBe(false)

      await swarm.dispose()

      // Reload from disk and verify coordinator last_post_seq / last_post_time survive restart.
      const reloaded = await SwarmManager.open({
        sessionId: 'sess-coord-roster',
        workspaceDir: workspace,
        storageDir: storage,
      })
      const reloadedCoord = reloaded.status().members.find(m => m.name === 'coordinator')
      expect(reloadedCoord?.last_post_seq).toBe(coordPost.seq)
      expect(reloadedCoord?.last_post_time).toBeDefined()
      expect(reloadedCoord?.state).toBe('waiting')
      await reloaded.dispose()
    })
  })

  test('detects quota/capacity errors and wakes coordinator immediately on worker error (both idle and blocked in swarm_wait)', async () => {
    expect(isQuotaOrCapacityError('HTTP 429 Too Many Requests')).toBe(true)
    expect(isQuotaOrCapacityError('insufficient_quota for model')).toBe(true)
    expect(isQuotaOrCapacityError('Rate limit reached')).toBe(true)
    expect(isQuotaOrCapacityError('RESOURCE_EXHAUSTED: quota exceeded')).toBe(true)
    expect(isQuotaOrCapacityError('Server overloaded / at capacity')).toBe(true)
    expect(isQuotaOrCapacityError('ENOENT: no such file or directory')).toBe(false)
    expect(isQuotaOrCapacityError('SyntaxError: unexpected token on line 14290')).toBe(false)

    await withTempDirs(async ({ workspace, storage }) => {
      const wakes: SwarmCoordinatorWakeEvent[] = []
      let failMode: 'quota' | 'generic' = 'quota'

      const swarm = await SwarmManager.open({
        sessionId: 'sess-worker-error',
        workspaceDir: workspace,
        storageDir: storage,
        onCoordinatorWake: event => wakes.push(event),
        workerRunner: async () => {
          if (failMode === 'quota') {
            throw new Error('HTTP 429 insufficient_quota: model quota exhausted')
          }
          throw new Error('Unexpected tool sandbox crash')
        },
      })

      // Case 1: coordinator is IDLE -> fires onCoordinatorWake with reason: 'worker_error'
      await swarm.add({ name: 'quota-worker', brief: 'Run analysis' })
      await swarm.waitForIdle()

      expect(swarm.getMember('quota-worker')?.state).toBe('waiting')
      expect(swarm.getMember('quota-worker')?.error).toContain('429')
      expect(wakes).toHaveLength(1)
      expect(wakes[0]?.reason).toBe('worker_error')
      expect(wakes[0]?.message).toBeUndefined()
      expect(wakes[0]?.worker_error).toEqual({
        member_name: 'quota-worker',
        message: 'HTTP 429 insufficient_quota: model quota exhausted',
        quota_exhausted: true,
      })
      expect(wakes[0]?.notice).toContain('swarm{action:"models"}')
      expect(wakes[0]?.notice).toContain('swarm{action:"model"')

      // Case 2: coordinator is BLOCKED inside swarm_wait -> unblocks immediately with status: 'worker_error'
      // and does NOT fire idle onCoordinatorWake listener!
      failMode = 'generic'
      const waitPromise = swarm.waitForBoard({
        caller: 'coordinator',
        since: swarm.head(),
        mentions_only: true,
      })
      await swarm.add({ name: 'crash-worker', brief: 'Run failing step' })
      const waitOutcome = await waitPromise
      await swarm.waitForIdle()

      expect(waitOutcome.status).toBe('worker_error')
      if (waitOutcome.status === 'worker_error') {
        expect(waitOutcome.worker_error).toEqual({
          member_name: 'crash-worker',
          message: 'Unexpected tool sandbox crash',
          quota_exhausted: false,
        })
        expect(waitOutcome.message).toContain('crash-worker')
      }
      // Still only 1 idle wake from Case 1 (Case 2 was consumed by swarm_wait).
      expect(wakes).toHaveLength(1)

      // Case 3: real runTurn worker hitting terminal provider_retry (429) surfaces error and wakes coordinator!
      const failingLlm: LlmClient = {
        async *stream(): AsyncGenerator<LlmDelta> {
          throw new Error('429 rate limit exceeded: quota exhausted')
        },
      }
      swarm.setWorkerRunner(
        createRunTurnWorkerRunner({
          llm: failingLlm,
          turnDependencies: { retryDelays: [] },
        }),
      )
      await swarm.add({ name: 'runturn-429', brief: 'Call failing LLM' })
      await swarm.waitForIdle()

      expect(swarm.getMember('runturn-429')?.error).toContain('429')
      expect(wakes).toHaveLength(2)
      expect(wakes[1]?.reason).toBe('worker_error')
      expect(wakes[1]?.worker_error?.member_name).toBe('runturn-429')
      expect(wakes[1]?.worker_error?.quota_exhausted).toBe(true)

      await swarm.dispose()
    })
  })

  test('tracks live per-worker activity with 250ms delta throttling, caps recent_tools & preview, and serves bounded inspectMember snapshots', async () => {
    await withTempDirs(async ({ workspace, storage }) => {
      let currentMs = 1_700_000_000_000
      const activityEvents: SwarmLifecycleEvent[] = []
      const blockingTurn = Promise.withResolvers<void>()
      const turnStarted = Promise.withResolvers<void>()

      const swarm = await SwarmManager.open({
        sessionId: 'sess-activity-inspect',
        workspaceDir: workspace,
        storageDir: storage,
        now: () => new Date(currentMs),
        onEvent: (event) => {
          if (event.kind === 'member_activity') {
            activityEvents.push(event)
          }
        },
        workerRunner: async (input) => {
          turnStarted.resolve()
          await blockingTurn.promise
          return {
            messages: [
              ...input.messages,
              { role: 'user', content: input.prompt },
              {
                role: 'assistant',
                content: 'Checking server.ts now.',
                tool_calls: [makeCall('tc_read_1', 'ReadFile', { file_path: '/repo/xerxes/src/daemon/server.ts' })],
              },
              {
                role: 'tool',
                tool_call_id: 'tc_read_1',
                name: 'ReadFile',
                content: 'X'.repeat(2500),
              },
              { role: 'assistant', content: 'Finished inspecting server.ts.' },
            ],
          }
        },
      })

      await swarm.add({ name: 'inspector-w', brief: 'Inspect server.ts' })
      await turnStarted.promise
      expect(swarm.getMember('inspector-w')?.state).toBe('working')

      // Thinking delta emits immediately on first delta, then throttles within 250ms.
      swarm.recordWorkerStreamEvent('inspector-w', { type: 'thinking', text: 'Planning...' })
      expect(activityEvents).toHaveLength(1)
      expect(activityEvents[0]?.activity?.phase).toBe('thinking')
      expect(swarm.getMember('inspector-w')?.current_activity).toBe('Thinking…')

      currentMs += 100
      swarm.recordWorkerStreamEvent('inspector-w', { type: 'thinking', text: 'Still planning...' })
      expect(activityEvents).toHaveLength(1)

      // Advancing past 250ms lets text delta emit and caps live_text_preview at 400 chars.
      currentMs += 200
      swarm.recordWorkerStreamEvent('inspector-w', { type: 'text', text: 'A'.repeat(550) })
      expect(activityEvents).toHaveLength(2)
      expect(activityEvents[1]?.activity?.phase).toBe('responding')
      expect(activityEvents[1]?.activity?.live_text_preview?.length).toBe(400)

      // tool_start emits immediately regardless of 250ms throttle!
      currentMs += 10
      swarm.recordWorkerStreamEvent('inspector-w', {
        type: 'tool_start',
        call: makeCall('call_read', 'ReadFile', { file_path: '/repo/xerxes/src/daemon/server.ts' }),
      })
      expect(activityEvents).toHaveLength(3)
      expect(activityEvents[2]?.activity?.phase).toBe('tool')
      expect(activityEvents[2]?.member?.current_activity).toBe('Reading server.ts')
      expect(activityEvents[2]?.activity?.active_tools).toHaveLength(1)

      // tool_end emits immediately, moves tool to recent_tools, and caps recent_tools at 20.
      currentMs += 10
      swarm.recordWorkerStreamEvent('inspector-w', {
        type: 'tool_end',
        result: {
          toolCallId: 'call_read',
          name: 'ReadFile',
          permitted: true,
          durationMs: 12,
          result: 'line 1\nline 2',
        },
      })
      expect(activityEvents).toHaveLength(4)
      expect(activityEvents[3]?.activity?.active_tools).toHaveLength(0)
      expect(activityEvents[3]?.activity?.recent_tools).toHaveLength(1)
      expect(activityEvents[3]?.activity?.recent_tools[0]?.status).toBe('done')

      for (let i = 0; i < 25; i += 1) {
        currentMs += 5
        swarm.recordWorkerStreamEvent('inspector-w', {
          type: 'tool_start',
          call: makeCall(`c_${i}`, 'Bash', { command: `echo ${i}` }),
        })
        swarm.recordWorkerStreamEvent('inspector-w', {
          type: 'tool_end',
          result: {
            toolCallId: `c_${i}`,
            name: 'Bash',
            permitted: i !== 24,
            durationMs: 3,
            result: i === 24 ? 'Error: denied' : `ok ${i}`,
          },
        })
      }
      const midInspect = swarm.inspectMember('inspector-w')
      expect(midInspect.activity.recent_tools).toHaveLength(20)
      expect(midInspect.activity.recent_tools.at(-1)?.status).toBe('error')

      // Finish turn: phase resets to 'idle' and current_activity is omitted when waiting!
      blockingTurn.resolve()
      await swarm.waitForIdle()

      expect(swarm.getMember('inspector-w')?.state).toBe('waiting')
      expect(swarm.getMember('inspector-w')?.current_activity).toBeUndefined()

      // inspectMember returns sanitized bounded transcript with truncated tool output.
      const afterInspect = swarm.inspectMember('inspector-w', { limit: 2 })
      expect(afterInspect.conversation_id).toBe('sess-activity-inspect:swarm:inspector-w')
      expect(afterInspect.activity.phase).toBe('idle')
      expect(afterInspect.total_messages).toBe(4)
      expect(afterInspect.messages).toHaveLength(2)
      expect(afterInspect.messages[0]?.index).toBe(2)
      expect(afterInspect.messages[0]?.role).toBe('tool')
      expect(afterInspect.messages[0]?.tool_name).toBe('ReadFile')
      expect(afterInspect.messages[0]?.text).toContain('[truncated')
      expect(afterInspect.messages[0]?.text.length).toBeLessThan(1300)
      expect(afterInspect.messages[1]?.text).toBe('Finished inspecting server.ts.')

      const fullInspect = swarm.inspectMember('inspector-w')
      expect(fullInspect.messages[1]?.tool_calls?.[0]?.name).toBe('ReadFile')

      // Coordinator inspect succeeds cleanly; unknown member throws ValidationError.
      expect(swarm.inspectMember('coordinator').member.role).toBe('coordinator')
      expect(() => swarm.inspectMember('missing-worker')).toThrow(ValidationError)

      await swarm.dispose()
    })
  })

  test('passes swarm tool reason on action:"model" and wires DaemonSwarmHost turn lifecycle + worker stream activity', async () => {
    await withTempDirs(async ({ root, workspace, storage }) => {
      // 1. Verify SWARM_TOOL_DEFINITION exposes `reason` and passes it to switchModel + worker prompt.
      const props = SWARM_TOOL_DEFINITION.function.parameters.properties as Record<string, unknown>
      expect(props.reason).toBeDefined()

      const resumedPrompts: string[] = []
      const catalog: SwarmModelCatalogPort = {
        listModels: () => [
          { id: 'model-a', display_name: 'Model A', suggested: true },
          { id: 'model-b', display_name: 'Model B', suggested: true },
        ],
      }
      const swarm = await SwarmManager.open({
        sessionId: 'sess-reason',
        workspaceDir: workspace,
        storageDir: storage,
        coordinatorModel: 'model-a',
        coordinatorModelDisplayName: 'Model A',
        modelCatalog: catalog,
        workerRunner: async (input) => {
          if (input.reason === 'model_switch') {
            resumedPrompts.push(input.prompt)
          }
          return { messages: input.messages }
        },
      })
      const registry = new ToolRegistry({ onDuplicateRegistration: () => undefined })
      const reg = registerSwarmTools(registry, { swarm, caller: 'coordinator' })
      await swarm.add({ name: 'w-switch', brief: 'Do task' })
      await swarm.waitForIdle()

      const rawSwitch = await registry.execute(
        makeCall('sw_model_reason', SWARM_TOOL_NAME, {
          action: 'model',
          name: 'w-switch',
          model: 'model-b',
          reason: 'quota exhausted on primary model',
        }),
        makeToolContext(),
      )
      await swarm.waitForIdle()
      const parsedSwitch = JSON.parse(rawSwitch) as { readonly model: string; readonly reason?: string }
      expect(parsedSwitch.model).toBe('model-b')
      expect(parsedSwitch.reason).toBe('quota exhausted on primary model')
      expect(resumedPrompts[0]).toContain('quota exhausted on primary model')
      reg.dispose()
      await swarm.dispose()

      // 2. Verify DaemonSwarmHost hooks coordinator onTurnStart/onTurnEnd, injects coordinator skill when active,
      //    and feeds worker StreamEvents into SwarmManager.recordWorkerStreamEvent AND onWorkerStreamEvent.
      const skillRegistry = new SkillRegistry()
      await skillRegistry.discover(BUNDLED_SKILLS_DIRECTORY)

      let coordinatorStateDuringTurn: string | undefined
      const forwardedEvents: Array<{ readonly member: string; readonly type: StreamEvent['type'] }> = []
      const liveActivities: Array<string | undefined> = []
      let observedSystemPrompt = ''

      const llm: LlmClient = {
        async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
          const sys = request.messages.find(m => m.role === 'system')
          const sysText = typeof sys?.content === 'string' ? sys.content : ''
          if (sysText.includes('<swarm_worker>')) {
            yield { content: 'Worker reporting live text.' }
            return
          }
          observedSystemPrompt = sysText
          const mgr = host.peekManager(session.id)
          coordinatorStateDuringTurn = mgr?.status().members.find(m => m.name === 'coordinator')?.state
          yield { content: 'Coordinator done.' }
        },
      }

      const host = new DaemonSwarmHost({
        storageRootDir: join(root, 'host-storage'),
        fallbackLlm: llm,
        defaultModel: () => 'gpt-4.1',
        skillRegistryForWorkspace: () => skillRegistry,
        onEvent: (event) => {
          if (event.kind === 'member_activity') {
            liveActivities.push(event.activity?.phase)
          }
        },
        onWorkerStreamEvent: (_sid, memberName, event) => {
          forwardedEvents.push({ member: memberName, type: event.type })
        },
      })

      const runner = new AgentTurnRunner({
        llm,
        model: 'gpt-4.1',
        swarmSessionAdapter: sess => host.sessionTurnAdapter(sess),
      })
      const runtime = new InMemoryDaemonRuntime(runner, {
        currentProjectDirectory: workspace,
        model: 'gpt-4.1',
        sessionDirectory: join(root, 'runtime-sessions'),
      })
      const session = await runtime.openSession('coord-live-state', undefined, { cwd: workspace })

      // Activate swarm solely by adding a worker (without explicit activateCoordinator) to verify skill injection on isActive()!
      const manager = await host.ensureSession({ id: session.id, cwd: workspace, model: 'gpt-4.1' })
      await manager.add({ name: 'stream-worker', brief: 'Emit live text.' })
      await manager.waitForIdle()

      expect(forwardedEvents.some(item => item.member === 'stream-worker' && item.type === 'text')).toBe(true)
      expect(liveActivities.length).toBeGreaterThan(0)
      expect(manager.status().members.find(m => m.name === 'coordinator')?.state).toBe('waiting')

      // Run a coordinator turn through AgentTurnRunner and verify state transitions waiting -> working -> waiting!
      for await (const _ev of runner.run(session, 'Check swarm status', new AbortController().signal)) {
        // consume stream
      }
      expect(coordinatorStateDuringTurn).toBe('working')
      expect(manager.status().members.find(m => m.name === 'coordinator')?.state).toBe('waiting')
      expect(observedSystemPrompt).toContain('Xerxes Immortals Coordinator Guide')
      expect(observedSystemPrompt).toContain('@narrator user feedback (default):')

      await host.close()
    })
  })
})
