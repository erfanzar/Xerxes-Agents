// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The goal subsystem as production actually reaches it.
 *
 * The unit tests around `goalDomain`, `goalTools` and `goalRoundDriver` prove
 * the mechanism; every one of them calls the driver itself. This file never
 * does: it drives a real `DaemonServer` over its real socket, and the only way
 * a second round can appear is if the server's own idle path admitted it. That
 * is the distinction this repo has repeatedly gotten wrong — a subsystem whose
 * tests pass and whose production wiring is absent.
 */

import { expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { InMemoryDaemonRuntime } from "../src/daemon/runtime.js";
import { DaemonServer } from "../src/daemon/server.js";
import { GoalTokenLedger } from '../src/runtime/goalTokenLedger.js';
import { chargeModelCall } from '../src/llms/callBudget.js';
import { readGoalWake, queueGoalWake, claimGoalWake } from '../src/runtime/goalWake.js';
import {
  completeGoal,
  createGoal,
  getGoal,
  editGoal,
  pauseGoal,
  resetGoalActivations,
  admitGoalRound,
} from "../src/runtime/goalDomain.js";
import type {
  DaemonEvent,
  DaemonSession,
  TurnRunControls,
  TurnRunner,
} from "../src/daemon/runtime.js";

interface TurnRecord {
  readonly text: string;
  readonly displayText: string | undefined;
  readonly goalRound: number | undefined;
}

/**
 * A runner that plays the part of the model: it records how each turn was
 * opened, and mutates goal state exactly where the goal tools would.
 */
class GoalScriptRunner implements TurnRunner {
  readonly turns: TurnRecord[] = [];
  constructor(
    private readonly script: (
      session: DaemonSession,
      turn: TurnRecord,
      index: number,
    ) => "silent" | "failed" | void,
  ) {}

  async *run(
    session: DaemonSession,
    text: string,
    _signal: AbortSignal,
    controls?: TurnRunControls,
  ): AsyncGenerator<DaemonEvent> {
    const record: TurnRecord = {
      text,
      displayText: controls?.displayText,
      goalRound: controls?.goalRound,
    };
    const index = this.turns.length;
    this.turns.push(record);
    const outcome = this.script(session, record, index);
    if (outcome === "silent") return;
    if (outcome === "failed") {
      // Exactly how the runtime reports a provider failure: an error
      // notification, and then the failure rendered as assistant text.
      yield { type: "notification", payload: { level: "error", message: "stream request failed (403): quota" } };
      yield { type: "text_part", payload: { text: "[error] stream request failed (403): quota" } };
      return;
    }
    yield { type: "text_part", payload: { text: `turn ${index}` } };
  }
}

async function withServer(
  prefix: string,
  runner: TurnRunner,
  body: (client: SocketTestClient, runtime: InMemoryDaemonRuntime, directory: string) => Promise<void>,
): Promise<void> {
  resetGoalActivations();
  const directory = await mkdtemp(join(tmpdir(), prefix));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "goal-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const goalTokenLedger = new GoalTokenLedger(join(directory, 'goal-tokens.sqlite'));
  const server = new DaemonServer({ socketPath, runtime, goalTokenLedger, autoTitle: false });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "goal-session" },
    });
    await client.next((frame) => frame.id === 1);
    await body(client, runtime, directory);
  } finally {
    client.close();
    await server.stop();
    goalTokenLedger.close();
    await rm(directory, { recursive: true, force: true });
    resetGoalActivations();
  }
}

for (const previousState of ['queued', 'running'] as const) test(`a ${previousState} goal wake survives restart without replay until explicit resume`, async () => {
  resetGoalActivations()
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-goal-wake-restart-'))
  const sessionDirectory = join(directory, 'sessions')
  const seed = new InMemoryDaemonRuntime(undefined, { sessionDirectory, currentProjectDirectory: directory })
  const original = await seed.openSession('seed')
  const goal = createGoal(original.metadata, original.id, { objective: 'finish after review', maxGoalRounds: 5 }, Date.now())
  const wake = queueGoalWake(original.metadata, original.id, goal.id, goal.revision, Date.now())
  if (previousState === 'running') {
    const round = admitGoalRound(original.metadata, original.id, Date.now())!
    claimGoalWake(original.metadata, original.id, wake.id, 'old-process', round.round, Date.now())
  }
  await seed.flushSessions()
  const rounds: number[] = []
  let persistedClaim = false
  const runner: TurnRunner = { async *run(session, _text, _signal, controls) {
    rounds.push(controls!.goalRound!)
    const saved = await Bun.file(join(sessionDirectory, `${session.id}.json`)).json()
    const recorded = readGoalWake(saved.metadata, session.id)!
    persistedClaim = recorded.state === 'running' && recorded.round === controls?.goalRound
      && getGoal(saved.metadata, session.id)?.roundsStarted === controls?.goalRound
    completeGoal(session.metadata, session.id, getGoal(session.metadata, session.id)!, Date.now())
    yield { type: 'text_part', payload: { text: 'Verified and complete' } }
  } }
  const runtime = new InMemoryDaemonRuntime(runner, { sessionDirectory, currentProjectDirectory: directory })
  const socketPath = join(directory, 'restarted.sock')
  const server = new DaemonServer({ socketPath, runtime, autoTitle: false, goalTokenOwner: 'new-process' })
  await server.start()
  const client = await SocketTestClient.connect(socketPath)
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { resume_session_id: original.id, project_dir: directory } })
    await client.next(frame => frame.id === 1)
    const restored = runtime.sessionStatus(original.id)!
    expect(getGoal(restored.metadata, restored.id)?.activation).toBe('disarmed')
    expect(readGoalWake(restored.metadata, restored.id)?.state).toBe(previousState === 'running' ? 'interrupted' : 'queued')
    await Bun.sleep(30)
    expect(rounds).toEqual([])
    client.send({ jsonrpc: '2.0', id: 2, method: 'goal.inspect', params: {} })
    expect(JSON.stringify(await client.next(frame => frame.id === 2))).toContain('continuation')
    client.send({ jsonrpc: '2.0', id: 3, method: 'session.goal', params: { input: 'resume' } })
    client.send({ jsonrpc: '2.0', id: 4, method: 'session.goal', params: { input: 'resume' } })
    await client.next(frame => frame.id === 3)
    await client.next(frame => frame.id === 4)
    await waitFor(() => readGoalWake(restored.metadata, restored.id)?.state === 'settled')
    expect(rounds).toEqual([previousState === 'running' ? 2 : 1])
    expect(persistedClaim).toBe(true)
    expect(getGoal(restored.metadata, restored.id)?.phase).toBe('complete')
  } finally {
    client.close(); await server.stop()
    await rm(directory, { recursive: true, force: true })
    resetGoalActivations()
  }
})

for (const action of ['submit', 'cancel', 'disconnect'] as const) test(`goal claim persistence retains ownership during ${action}`, async () => {
  const runner = new GoalScriptRunner(session => {
    completeGoal(session.metadata, session.id, getGoal(session.metadata, session.id)!, Date.now())
  })
  await withServer(`xerxes-goal-claim-${action}-`, runner, async (client, runtime) => {
    const session = runtime.sessionStatus('goal-session')!
    const goal = createGoal(session.metadata, session.id, { objective: 'persist before running' }, Date.now())
    pauseGoal(session.metadata, session.id, goal, Date.now())
    const flush = runtime.flushSessions.bind(runtime)
    let release!: () => void
    const gate = new Promise<void>(resolve => { release = resolve })
    let savingClaim = false
    runtime.flushSessions = async () => {
      if (!savingClaim && readGoalWake(session.metadata, session.id)?.state === 'running') {
        savingClaim = true
        await gate
      }
      await flush()
    }
    try {
      client.send({ jsonrpc: '2.0', id: 2, method: 'session.goal', params: { input: 'resume' } })
      await client.next(frame => frame.id === 2)
      await waitFor(() => savingClaim)
      expect(runner.turns).toHaveLength(0)
      if (action === 'disconnect') {
        client.close()
        await waitFor(() => session.cancelRequested)
        expect(runtime.sessionStatus('goal-session')).toBe(session)
      } else {
        client.send({ jsonrpc: '2.0', id: 3, method: action === 'cancel' ? 'turn.cancel' : 'turn.submit',
          params: action === 'cancel' ? {} : { text: 'second submission' } })
        await client.next(frame => frame.id === 3)
      }
      if (action === 'submit') {
        await client.next(frame => JSON.stringify(frame).includes('a turn is already active'))
      }
      release()
      await waitFor(() => ['settled', 'interrupted'].includes(readGoalWake(session.metadata, session.id)?.state ?? ''))
      expect(runner.turns.map(turn => turn.goalRound)).toEqual(action === 'submit' ? [1] : [])
      expect(getGoal(session.metadata, session.id)?.phase).toBe(action === 'submit' ? 'complete' : 'paused')
      expect(readGoalWake(session.metadata, session.id)?.state).toBe(action === 'submit' ? 'settled' : 'interrupted')
    } finally { release(); runtime.flushSessions = flush }
  })
})

test('human work already queued at resume precedes the automatic goal wake', async () => {
  const runner = new GoalScriptRunner(session => {
    completeGoal(session.metadata, session.id, getGoal(session.metadata, session.id)!, Date.now())
  })
  await withServer('xerxes-goal-human-priority-', runner, async (client, runtime) => {
    const session = runtime.sessionStatus('goal-session')!
    const goal = createGoal(session.metadata, session.id, { objective: 'wait for the person' }, Date.now())
    queueGoalWake(session.metadata, session.id, goal.id, goal.revision, Date.now())
    pauseGoal(session.metadata, session.id, goal, Date.now())
    client.send({ jsonrpc: '2.0', id: 2, method: 'session.goal', params: { input: 'resume' } })
    client.send({ jsonrpc: '2.0', id: 3, method: 'turn.submit', params: { text: 'urgent human work' } })
    await client.next(frame => frame.id === 3)
    await waitFor(() => getGoal(session.metadata, session.id)?.phase === 'complete' && !session.activeTurnId)
    expect(runner.turns.map(turn => turn.text)).toEqual(['urgent human work'])
  })
})

test('milestone commands persist before acknowledgement and survive session reload', async () => {
  const runner = new GoalScriptRunner(() => {})
  await withServer('xerxes-goal-milestone-', runner, async (client, runtime, directory) => {
    const session = runtime.sessionStatus('goal-session')!
    createGoal(session.metadata, session.id, { objective: 'finish the full goal' }, Date.now())
    client.send({ jsonrpc: '2.0', id: 2, method: 'session.goal', params: { input: 'milestone Verify cancellation under load' } })
    expect(JSON.stringify(await client.next(frame => frame.id === 2))).toContain('Verify cancellation under load')
    const path = join(directory, 'sessions', `${session.id}.json`)
    expect(await Bun.file(path).text()).toContain('Verify cancellation under load')
    client.send({ jsonrpc: '2.0', id: 3, method: 'goal.inspect', params: {} })
    expect(JSON.stringify(await client.next(frame => frame.id === 3))).toContain('currentMilestone')
    const restoredRuntime = new InMemoryDaemonRuntime(undefined, { sessionDirectory: join(directory, 'sessions'), currentProjectDirectory: directory })
    const restored = await restoredRuntime.openSession(session.id, undefined, { resume: true, cwd: session.cwd })
    expect(getGoal(restored.metadata, restored.id)).toMatchObject({ objective: 'finish the full goal',
      currentMilestone: 'Verify cancellation under load', roundsStarted: 0, activation: 'disarmed' })
    client.send({ jsonrpc: '2.0', id: 4, method: 'session.goal', params: { input: 'milestone clear' } })
    await client.next(frame => frame.id === 4)
    expect(getGoal(session.metadata, session.id)?.currentMilestone).toBeUndefined()
    const stored = await Bun.file(path).json()
    expect(getGoal(stored.metadata, session.id)?.currentMilestone).toBeUndefined()
    expect(runner.turns).toHaveLength(0)
  })
})

test('human criterion decisions bind to the viewed session and revision and persist before acknowledgement', async () => {
  const runner = new GoalScriptRunner(() => {})
  await withServer('xerxes-goal-decision-', runner, async (client, runtime, directory) => {
    const session = runtime.sessionStatus('goal-session')!
    const initial = createGoal(session.metadata, session.id, { objective: 'Ship the reviewed layout',
      criteria: [{ id: 'visual', description: 'The user accepts the layout' }] }, Date.now())
    const params = { session_id: session.id, goal_id: initial.id, revision: initial.revision,
      criterion_id: 'visual', summary: 'I reviewed the wide terminal layout and accept it.' }
    client.send({ jsonrpc: '2.0', id: 2, method: 'goal.decision', params: { ...params, session_id: 'another-session' } })
    expect(JSON.stringify(await client.next(frame => frame.id === 2))).toContain('active session changed')
    client.send({ jsonrpc: '2.0', id: 3, method: 'goal.decision', params: { ...params, summary: '   ' } })
    expect(JSON.stringify(await client.next(frame => frame.id === 3))).toContain('nonempty acceptance note')
    client.send({ jsonrpc: '2.0', id: 11, method: 'goal.decision', params: { ...params, decisionId: 'caller-invented' } })
    expect(JSON.stringify(await client.next(frame => frame.id === 11))).toContain('nonempty acceptance note')
    expect(getGoal(session.metadata, session.id)?.revision).toBe(initial.revision)
    client.send({ jsonrpc: '2.0', id: 4, method: 'goal.decision', params })
    expect(JSON.stringify(await client.next(frame => frame.id === 4))).toContain('user-decision')
    const accepted = getGoal(session.metadata, session.id)!
    expect(accepted.criteria?.[0]?.evidence).toMatchObject({ kind: 'user-decision', summary: params.summary })
    expect(accepted.phase).toBe('active')
    const stored = await Bun.file(join(directory, 'sessions', `${session.id}.json`)).text()
    expect(stored).toContain('user-decision')
    expect(stored).toContain(params.summary)
    client.send({ jsonrpc: '2.0', id: 5, method: 'goal.decision', params: { ...params, summary: 'stale decision' } })
    expect(JSON.stringify(await client.next(frame => frame.id === 5))).toContain('revision')
    expect(getGoal(session.metadata, session.id)?.criteria).toEqual(accepted.criteria)
    client.send({ jsonrpc: '2.0', id: 6, method: 'goal.inspect', params: {} })
    expect(JSON.stringify(await client.next(frame => frame.id === 6))).toContain(params.summary)
    expect(runner.turns).toHaveLength(0)
  })
})

test('a decision during a live turn is rejected without altering evidence', async () => {
  let release!: () => void
  const gate = new Promise<void>(resolve => { release = resolve })
  const runner: TurnRunner = { async *run(session) {
    yield { type: 'text_part', payload: { text: 'Reviewing' } }
    await gate
    const goal = getGoal(session.metadata, session.id)!
    pauseGoal(session.metadata, session.id, goal, Date.now())
  } }
  await withServer('xerxes-goal-decision-busy-', runner, async (client, runtime) => {
    const session = runtime.sessionStatus('goal-session')!
    const goal = createGoal(session.metadata, session.id, { objective: 'review',
      criteria: [{ id: 'review', description: 'User accepts review' }] }, Date.now())
    try {
      client.send({ jsonrpc: '2.0', id: 2, method: 'turn.submit', params: { text: 'review' } })
      await client.next(frame => frame.id === 2)
      await waitFor(() => !!session.activeTurnId)
      client.send({ jsonrpc: '2.0', id: 3, method: 'goal.decision', params: {
        session_id: session.id, goal_id: goal.id, revision: goal.revision,
        criterion_id: 'review', summary: 'I accept it',
      } })
      expect(JSON.stringify(await client.next(frame => frame.id === 3))).toContain('Pause or wait')
      expect(getGoal(session.metadata, session.id)?.criteria?.[0]?.evidence).toBeUndefined()
    } finally { release() }
    await waitFor(() => !session.activeTurnId)
  })
})

test('goal token spend spans real daemon rounds and survives reopening its ledger', async () => {
  const runner = new GoalScriptRunner(() => {
    chargeModelCall()!({ inputTokens: 4, outputTokens: 2 })
  })
  await withServer('xerxes-goal-tokens-', runner, async (client, runtime, directory) => {
    client.send({ jsonrpc: '2.0', id: 2, method: 'session.goal', params: { input: 'bounded work' } })
    await client.next(frame => frame.id === 2)
    client.send({ jsonrpc: '2.0', id: 3, method: 'session.goal', params: { input: '--tokens 10' } })
    await client.next(frame => frame.id === 3)
    client.send({ jsonrpc: '2.0', id: 4, method: 'turn.submit', params: { text: 'start' } })
    await client.next(frame => frame.id === 4)
    await waitFor(() => {
      const session = runtime.sessionStatus('goal-session')!
      return getGoal(session.metadata, session.id)?.phase === 'blocked' && !session.activeTurnId
    })
    const session = runtime.sessionStatus('goal-session')!
    const goal = getGoal(session.metadata, session.id)!
    expect(runner.turns).toHaveLength(2)
    expect(goal.blockedReason?.code).toBe('token-budget')
    const reloaded = new GoalTokenLedger(join(directory, 'goal-tokens.sqlite'))
    try { expect(reloaded.inspect(session.id, goal.id)).toMatchObject({ inputTokens: 8, outputTokens: 4, settledCalls: 2, complete: true }) }
    finally { reloaded.close() }
    client.send({ jsonrpc: '2.0', id: 5, method: 'session.goal', params: { input: 'resume' } })
    expect(JSON.stringify(await client.next(frame => frame.id === 5))).toContain('exhausted')
    expect(getGoal(session.metadata, session.id)?.phase).toBe('blocked')
    client.send({ jsonrpc: '2.0', id: 6, method: 'goal.inspect', params: {} })
    expect(JSON.stringify(await client.next(frame => frame.id === 6))).toContain('"inputTokens":8')
    client.send({ jsonrpc: '2.0', id: 7, method: 'session.goal', params: { input: '--tokens 20' } })
    await client.next(frame => frame.id === 7)
    client.send({ jsonrpc: '2.0', id: 8, method: 'session.goal', params: { input: 'resume' } })
    expect(JSON.stringify(await client.next(frame => frame.id === 8))).toContain('Goal resumed')
    client.send({ jsonrpc: '2.0', id: 9, method: 'goal.inspect', params: {} })
    const afterRaise = JSON.stringify(await client.next(frame => frame.id === 9))
    expect(afterRaise).toContain('"inputTokens":8')
    expect(afterRaise).toContain('"maxTotalTokens":20')
  })
})

test('an expired goal acknowledges an unstarted turn without starting a provider', async () => {
  const runner = new GoalScriptRunner(() => {})
  await withServer('xerxes-goal-expired-', runner, async (client, runtime) => {
    const session = runtime.sessionStatus('goal-session')!
    createGoal(session.metadata, session.id, { objective: 'expired', maxDurationMs: 1 }, 1000)
    client.send({ jsonrpc: '2.0', id: 2, method: 'turn.submit', params: { text: 'continue' } })
    await client.next(frame => frame.id === 2)
    const ended = await client.next(frame => JSON.stringify(frame).includes('"unstarted":true'))
    expect(JSON.stringify(ended)).toContain('turn_end')
    expect(runner.turns).toHaveLength(0)
    expect(getGoal(session.metadata, session.id)?.blockedReason?.code).toBe('time-limit')
  })
})

test("a goal created mid-turn cancels live work at its wall-time deadline and persists the blocker", async () => {
  let aborted = false
  let calls = 0
  const runner: TurnRunner = {
    async *run(session, _text, signal) {
      calls++
      createGoal(session.metadata, session.id, { objective: 'bounded work', maxDurationMs: 50 }, Date.now())
      yield { type: 'text_part', payload: { text: 'Working until cancelled' } }
      await new Promise<void>(resolve => {
        if (signal.aborted) { resolve(); return }
        signal.addEventListener('abort', () => resolve(), { once: true })
      })
      aborted = signal.aborted
    },
  }
  await withServer('xerxes-goal-deadline-', runner, async (client, runtime, directory) => {
    client.send({ jsonrpc: '2.0', id: 2, method: 'turn.submit', params: { text: 'start' } })
    await client.next(frame => frame.id === 2)
    await waitFor(() => aborted && !runtime.sessionStatus('goal-session')?.activeTurnId)
    const session = runtime.sessionStatus('goal-session')!
    expect(calls).toBe(1)
    expect(getGoal(session.metadata, session.id)).toMatchObject({ phase: 'blocked', blockedReason: { code: 'time-limit' } })
    await runtime.flushSessions()
    const stored = await Bun.file(join(directory, 'sessions', `${session.id}.json`)).text()
    expect(stored).toContain('time-limit')
  })
})

test("an active goal drives further rounds as real, separately attributed turns", async () => {
  const runner = new GoalScriptRunner((session, _turn, index) => {
    if (index === 0) {
      createGoal(session.metadata, session.id, { objective: "ship it", maxGoalRounds: 8 }, 1_000);
      return;
    }
    if (index === 3) {
      const goal = getGoal(session.metadata, session.id)!;
      completeGoal(session.metadata, session.id, goal, 2_000);
    }
  });

  await withServer("xerxes-goal-rounds-", runner, async (client, runtime) => {
    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "start" } });
    await client.next((frame) => frame.id === 2);
    await waitFor(() => runner.turns.length === 4 && !runtime.sessionStatus("goal-session")?.activeTurnId);

    // Turn 0 is the human's. Rounds 1..3 were opened by the driver, numbered
    // consecutively, each carrying its round in the controls the tools
    // authorise against.
    expect(runner.turns.map((turn) => turn.goalRound)).toEqual([undefined, 1, 2, 3]);
    expect(runner.turns[0]?.text).toBe("start");
    for (const turn of runner.turns.slice(1)) {
      // The provider gets the whole brief every round...
      expect(turn.text).toContain("<goal_round>");
      expect(turn.text).toContain('"ship it"');
      expect(turn.text).toContain(`Round ${turn.goalRound} of 8`);
      // ...while the transcript gets one readable line, so a person can follow
      // a long run instead of scrolling past the same block N times.
      expect(turn.displayText).toBe(`Goal round ${turn.goalRound}/8 — ship it`);
      expect(turn.displayText).not.toContain("<goal_round>");
    }
    // Completing the goal is what stops it, not the round cap.
    const goal = getGoal(runtime.sessionStatus("goal-session")!.metadata, runtime.sessionStatus("goal-session")!.id);
    expect(goal?.phase).toBe("complete");
    expect(goal?.roundsStarted).toBe(3);
    await Bun.sleep(30);
    expect(runner.turns.length).toBe(4);
  });
});

test('goal edits and clear reach the socket while the turn is still running', async () => {
  let release!: () => void;
  const gate = new Promise<void>(resolve => { release = resolve; });
  const runner: TurnRunner = {
    async *run(session) {
      const goal = getGoal(session.metadata, session.id)!;
      editGoal(session.metadata, session.id, goal, { objective: 'Optimize TPU kernels' }, Date.now());
      yield { type: 'tool_result', payload: { name: 'update_goal', result: '{}' } };
      await gate;
      yield { type: 'text_part', payload: { text: 'Updated.' } };
    },
  };
  await withServer('xerxes-live-goal-', runner, async (client, runtime) => {
    const statusGoal = (objective: string | null) => (frame: Record<string, unknown>) => {
      const params = frame.params as { type?: string; payload?: { goal?: string | null } } | undefined;
      return params?.type === 'status_update' && params.payload?.goal === objective;
    };
    try {
      client.send({ jsonrpc: '2.0', id: 2, method: 'session.goal', params: { input: 'Old review goal' } });
      await client.next(frame => frame.id === 2);
      await client.next(statusGoal('Old review goal'));
      client.send({ jsonrpc: '2.0', id: 20, method: 'session.goal', params: { input: '--duration 30m' } });
      const limited = await client.next(frame => frame.id === 20);
      expect(JSON.stringify(limited)).toContain('Goal time limit updated');
      const limitedSession = runtime.sessionStatus('goal-session')!;
      expect(getGoal(limitedSession.metadata, limitedSession.id)?.maxDurationMs).toBe(1_800_000);
      client.send({ jsonrpc: '2.0', id: 3, method: 'turn.submit', params: { text: 'edit the goal' } });
      await client.next(frame => frame.id === 3);
      await client.next(statusGoal('Optimize TPU kernels'));
      expect(runtime.sessionStatus('goal-session')?.activeTurnId).toBeTruthy();
      client.send({ jsonrpc: '2.0', id: 4, method: 'session.goal', params: { input: 'clear' } });
      await client.next(frame => frame.id === 4);
      await client.next(statusGoal(null));
      expect(getGoal(runtime.sessionStatus('goal-session')!.metadata, runtime.sessionStatus('goal-session')!.id)).toBeUndefined();
    } finally { release(); }
    await waitFor(() => !runtime.sessionStatus('goal-session')?.activeTurnId);
  });
});

test("rounds stop at the goal's own cap without any completion claim", async () => {
  const runner = new GoalScriptRunner((session, _turn, index) => {
    if (index === 0) {
      createGoal(session.metadata, session.id, { objective: "never done", maxGoalRounds: 2 }, 1_000);
    }
  });

  await withServer("xerxes-goal-cap-", runner, async (client, runtime) => {
    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "start" } });
    await client.next((frame) => frame.id === 2);
    await waitFor(() => !runtime.sessionStatus("goal-session")?.activeTurnId && getGoal(runtime.sessionStatus("goal-session")!.metadata, runtime.sessionStatus("goal-session")!.id)?.phase === 'blocked');
    // One human turn plus exactly max_goal_rounds automatic rounds. A goal that
    // never completes is bounded by its own declared budget, not by a global
    // retry ceiling that the model cannot see.
    expect(runner.turns.map((turn) => turn.goalRound)).toEqual([undefined, 1, 2]);
    const session = runtime.sessionStatus("goal-session")!;
    const goal = getGoal(session.metadata, session.id);
    // Exhaustion is recorded, not silent: a person reading /goal sees why the
    // run stopped and what to change, instead of an "active" goal that never
    // moves again.
    expect(goal?.phase).toBe("blocked");
    expect(goal?.blockedReason?.code).toBe("round-limit");
    expect(goal?.blockedReason?.message).toContain("2 rounds");
  });
});

test("a session with no goal runs exactly one turn", async () => {
  const runner = new GoalScriptRunner(() => {});
  await withServer("xerxes-goal-absent-", runner, async (client, runtime) => {
    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "hello" } });
    await client.next((frame) => frame.id === 2);
    await waitFor(() => runner.turns.length === 1 && !runtime.sessionStatus("goal-session")?.activeTurnId);
    await Bun.sleep(40);
    expect(runner.turns.length).toBe(1);
  });
});

test("cancelling a turn withdraws continuation authority without erasing the goal", async () => {
  const runner = new GoalScriptRunner((session, _turn, index) => {
    if (index === 0) {
      createGoal(session.metadata, session.id, { objective: "keep going", maxGoalRounds: 9 }, 1_000);
    }
  });
  await withServer("xerxes-goal-cancel-", runner, async (client, runtime) => {
    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "start" } });
    await client.next((frame) => frame.id === 2);
    await waitFor(() => runner.turns.length >= 1);
    runtime.cancelTurn("goal-session");
    await Bun.sleep(60);
    const session = runtime.sessionStatus("goal-session")!;
    const goal = getGoal(session.metadata, session.id)!;
    // The objective survives verbatim; only the authority to advance it alone
    // is gone. An interrupt during the human's own turn disarms without a
    // phase change, so the goal is still active and simply not driving.
    expect(goal.objective).toBe("keep going");
    expect(goal.activation).toBe("disarmed");
    const before = runner.turns.length;
    await Bun.sleep(40);
    expect(runner.turns.length).toBe(before);
  });
});

test("interrupting an automatic round pauses the goal so a person can resume it", async () => {
  const started = { rounds: 0 };
  const runner: TurnRunner = { async *run(session, _text, signal, controls) {
    if (controls?.goalRound === undefined) {
      createGoal(session.metadata, session.id, { objective: "long haul", maxGoalRounds: 9 }, 1_000);
      yield { type: 'text_part', payload: { text: 'Starting the goal' } };
      return;
    }
    started.rounds += 1;
    yield { type: 'text_part', payload: { text: 'Running until interrupted' } };
    if (!signal.aborted) await new Promise<void>(resolve => signal.addEventListener('abort', () => resolve(), { once: true }));
  } };
  await withServer("xerxes-goal-pause-", runner, async (client, runtime) => {
    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "start" } });
    await client.next((frame) => frame.id === 2);
    await waitFor(() => started.rounds >= 1);
    client.send({ jsonrpc: '2.0', id: 3, method: 'turn.cancel', params: {} });
    await client.next(frame => frame.id === 3);
    await waitFor(() => readGoalWake(runtime.sessionStatus('goal-session')!.metadata, runtime.sessionStatus('goal-session')!.id)?.state === 'interrupted');
    const session = runtime.sessionStatus("goal-session")!;
    const goal = getGoal(session.metadata, session.id)!;
    // Durable and visible: /goal reports paused and offers resume. Merely
    // dropping authority would leave the goal reading "active" forever while
    // nothing advanced it.
    expect(goal.phase).toBe("paused");
    expect(goal.objective).toBe("long haul");
  });
});

test("a round that produces nothing stops the run instead of spending the budget", async () => {
  const runner = new GoalScriptRunner((session, _turn, index) => {
    if (index === 0) {
      createGoal(session.metadata, session.id, { objective: "unreachable", maxGoalRounds: 24 }, 1_000);
      return;
    }
    // Every automatic round fails before producing anything — a provider
    // outage, an auth failure, a context overflow. Without a stop this is a hot
    // loop: a real run burned all 24 rounds in nine seconds and wrote nothing
    // but its own prompts into the transcript.
    return "silent";
  });

  await withServer("xerxes-goal-silent-", runner, async (client, runtime) => {
    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "start" } });
    await client.next((frame) => frame.id === 2);
    await waitFor(() => runner.turns.length >= 2 && !runtime.sessionStatus("goal-session")?.activeTurnId);
    await Bun.sleep(60);

    // Exactly one automatic round was attempted, not twenty-four.
    expect(runner.turns.map((turn) => turn.goalRound)).toEqual([undefined, 1]);
    const session = runtime.sessionStatus("goal-session")!;
    const goal = getGoal(session.metadata, session.id);
    expect(goal?.phase).toBe("blocked");
    expect(goal?.blockedReason?.code).toBe("round-produced-nothing");
    expect(readGoalWake(session.metadata, session.id)?.state).toBe('interrupted');
  });
});

test("a failing provider stops the run on the first round, not after the budget", async () => {
  const runner = new GoalScriptRunner((session, _turn, index) => {
    if (index === 0) {
      createGoal(session.metadata, session.id, { objective: "out of quota", maxGoalRounds: 24 }, 1_000);
      return;
    }
    return "failed";
  });

  await withServer("xerxes-goal-failed-", runner, async (client, runtime) => {
    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "start" } });
    await client.next((frame) => frame.id === 2);
    await waitFor(() => runner.turns.length >= 2 && !runtime.sessionStatus("goal-session")?.activeTurnId);
    await Bun.sleep(60);

    // The failure is rendered as assistant text, so "did any text arrive" would
    // call this a productive round and keep going — which is what a live run
    // against an out-of-quota provider actually did, 24 times in nine seconds.
    expect(runner.turns.map((turn) => turn.goalRound)).toEqual([undefined, 1]);
    const session = runtime.sessionStatus("goal-session")!;
    const goal = getGoal(session.metadata, session.id);
    expect(goal?.phase).toBe("blocked");
    expect(goal?.blockedReason?.code).toBe("round-failed");
    expect(goal?.blockedReason?.message).toContain("403");
    expect(readGoalWake(session.metadata, session.id)?.state).toBe('interrupted');
  });
});

async function waitFor(predicate: () => boolean, timeoutMs = 3_000): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    if (predicate()) return;
    await Bun.sleep(5);
  }
  throw new Error("condition not met before timeout");
}

class SocketTestClient {
  private buffer = "";
  private readonly frames: Record<string, unknown>[] = [];
  private waiters: (() => void)[] = [];

  private constructor(private readonly socket: Socket) {
    socket.setEncoding("utf8");
    socket.on("data", (chunk: string) => {
      this.buffer += chunk;
      let index = this.buffer.indexOf("\n");
      while (index >= 0) {
        const line = this.buffer.slice(0, index).trim();
        this.buffer = this.buffer.slice(index + 1);
        if (line) this.frames.push(JSON.parse(line));
        index = this.buffer.indexOf("\n");
      }
      const waiters = this.waiters;
      this.waiters = [];
      for (const waiter of waiters) waiter();
    });
  }

  static connect(socketPath: string): Promise<SocketTestClient> {
    return new Promise((resolve, reject) => {
      const socket = connect(socketPath);
      socket.once("connect", () => resolve(new SocketTestClient(socket)));
      socket.once("error", reject);
    });
  }

  send(frame: Record<string, unknown>): void {
    this.socket.write(`${JSON.stringify(frame)}\n`);
  }

  async next(
    match: (frame: Record<string, unknown>) => boolean,
    timeoutMs = 3_000,
  ): Promise<Record<string, unknown>> {
    const deadline = Date.now() + timeoutMs;
    for (;;) {
      const index = this.frames.findIndex(match);
      if (index >= 0) return this.frames.splice(index, 1)[0]!;
      if (Date.now() > deadline) throw new Error("frame not received before timeout");
      await new Promise<void>((resolve) => {
        this.waiters.push(resolve);
        setTimeout(resolve, 25);
      });
    }
  }

  close(): void {
    this.socket.destroy();
  }
}

test('explicit goal resume clears a prior idle cancellation before scheduling the next round', async () => {
  const runner = new GoalScriptRunner(session => {
    completeGoal(session.metadata, session.id, getGoal(session.metadata, session.id)!, Date.now());
  });
  await withServer('xerxes-goal-resume-cancel-', runner, async (client, runtime) => {
    const session = runtime.sessionStatus('goal-session')!;
    const goal = createGoal(session.metadata, session.id, { objective: 'finish after interruption' }, Date.now());
    pauseGoal(session.metadata, session.id, goal, Date.now());
    session.cancelRequested = true;
    client.send({ jsonrpc: '2.0', id: 2, method: 'session.goal', params: { input: 'resume' } });
    const result = await client.next(frame => frame.id === 2);
    expect(result.result).toMatchObject({ ok: true });
    await waitFor(() => runner.turns.length === 1);
    expect(runner.turns[0]?.goalRound).toBe(1);
    await waitFor(() => getGoal(session.metadata, session.id)?.phase === 'complete');
  });
});
