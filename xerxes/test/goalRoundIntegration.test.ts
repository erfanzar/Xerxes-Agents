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
import {
  completeGoal,
  createGoal,
  getGoal,
  resetGoalActivations,
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
  body: (client: SocketTestClient, runtime: InMemoryDaemonRuntime) => Promise<void>,
): Promise<void> {
  resetGoalActivations();
  const directory = await mkdtemp(join(tmpdir(), prefix));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "goal-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ socketPath, runtime });
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
    await body(client, runtime);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
    resetGoalActivations();
  }
}

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

test("rounds stop at the goal's own cap without any completion claim", async () => {
  const runner = new GoalScriptRunner((session, _turn, index) => {
    if (index === 0) {
      createGoal(session.metadata, session.id, { objective: "never done", maxGoalRounds: 2 }, 1_000);
    }
  });

  await withServer("xerxes-goal-cap-", runner, async (client, runtime) => {
    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "start" } });
    await client.next((frame) => frame.id === 2);
    await waitFor(() => !runtime.sessionStatus("goal-session")?.activeTurnId && runner.turns.length > 1);
    await Bun.sleep(40);
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
  const runner = new GoalScriptRunner((session, turn, index) => {
    if (index === 0) {
      createGoal(session.metadata, session.id, { objective: "long haul", maxGoalRounds: 9 }, 1_000);
      return;
    }
    if (turn.goalRound !== undefined) started.rounds += 1;
  });
  await withServer("xerxes-goal-pause-", runner, async (client, runtime) => {
    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "start" } });
    await client.next((frame) => frame.id === 2);
    await waitFor(() => started.rounds >= 1);
    runtime.cancelTurn("goal-session");
    await Bun.sleep(80);
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
