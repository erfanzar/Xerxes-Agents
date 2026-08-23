// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp, realpath, rm } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  InMemoryDaemonRuntime,
  type DaemonEvent,
  type DaemonSession,
  type TurnRunner,
} from "../src/daemon/runtime.js";
import { DaemonServer } from "../src/daemon/server.js";

test("session.open mid-turn exposes the in-flight thinking and tool trail", async () => {
  const directory = await realpath(
    await mkdtemp(join(tmpdir(), "xerxes-daemon-inflight-trail-")),
  );
  const socketPath = join(directory, "daemon.sock");
  const runner = new GatedTurnRunner();
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(runner, {
      currentProjectDirectory: directory,
      model: "gpt-4o",
      sessionDirectory: join(directory, "sessions"),
      workspaceRoot: join(directory, "agents"),
    }),
  });
  await server.start();
  const client = await TrailTestClient.connect(socketPath);
  try {
    const init = await client.request(1, "initialize", {
      project_dir: directory,
      session_key: "main",
    });
    expect(init.ok).toBe(true);

    await client.request(2, "turn.submit", { text: "inspect the trail" });
    await runner.waitForPhase("called");

    // Mid-turn attach: the runner manages session state and only synchronizes
    // session.messages at turn end, so the persisted transcript is still empty
    // and the inflight trail is the only carrier of the work so far.
    const midTurn = await client.request(3, "session.open", { session_key: "main" });
    const session = midTurn.session as Record<string, unknown>;
    expect(session.messages).toBe(0);
    const inflight = session.inflight as Record<string, unknown>;
    expect(inflight.user).toBe("inspect the trail");
    expect(inflight.streaming).toBe(true);
    expect(typeof inflight.started_at).toBe("number");
    expect(inflight.started_at as number).toBeGreaterThan(0);
    expect(inflight.thinking).toBe("mid-turn reasoning");
    expect(inflight.tools).toEqual([
      { id: "call-1", name: "ReadFile", arguments: '{"path":"a.ts"}' },
    ]);

    runner.release("settle");
    await runner.waitForPhase("settled");

    const settled = await client.request(4, "session.open", { session_key: "main" });
    const settledInflight = (settled.session as Record<string, unknown>).inflight as Record<
      string,
      unknown
    >;
    expect(settledInflight.tools).toEqual([
      {
        id: "call-1",
        name: "ReadFile",
        arguments: '{"path":"a.ts"}',
        duration_ms: 42,
        ok: true,
      },
    ]);

    runner.release("finish");
    await client.nextEvent("turn_end");

    const afterTurn = await client.request(5, "session.open", { session_key: "main" });
    expect((afterTurn.session as Record<string, unknown>).inflight).toBeUndefined();
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

type Phase = "called" | "settled" | "finished";

/** Runner-managed turn that blocks on explicit gates between trail events. */
class GatedTurnRunner implements TurnRunner {
  readonly managesSessionState = true;

  private phase: Phase = "called";
  private readonly gates = new Map<"settle" | "finish", PromiseWithResolvers<void>>();
  private readonly waiters: Array<{ phase: Phase; resolve: () => void }> = [];

  async *run(
    _session: DaemonSession,
    _text: string,
  ): AsyncGenerator<DaemonEvent> {
    this.markPhase("called");
    yield { type: "think_part", payload: { think: "mid-turn reasoning" } };
    yield {
      type: "tool_call",
      payload: { id: "call-1", name: "ReadFile", arguments: '{"path":"a.ts"}' },
    };
    await this.gate("settle");
    this.markPhase("settled");
    yield {
      type: "tool_result",
      payload: {
        tool_call_id: "call-1",
        name: "ReadFile",
        return_value: "file body",
        duration_ms: 42,
        permitted: true,
      },
    };
    await this.gate("finish");
    this.markPhase("finished");
    yield { type: "text_part", payload: { text: "done" } };
  }

  release(phase: "settle" | "finish"): void {
    this.gates.get(phase)?.resolve();
  }

  async waitForPhase(phase: Phase): Promise<void> {
    const order: Phase[] = ["called", "settled", "finished"];
    if (order.indexOf(this.phase) >= order.indexOf(phase)) return;
    await new Promise<void>((resolve) => this.waiters.push({ phase, resolve }));
  }

  private markPhase(phase: Phase): void {
    this.phase = phase;
    const order: Phase[] = ["called", "settled", "finished"];
    for (const waiter of this.waiters.splice(0)) {
      if (order.indexOf(phase) >= order.indexOf(waiter.phase)) waiter.resolve();
    }
  }

  private gate(name: "settle" | "finish"): Promise<void> {
    let gate = this.gates.get(name);
    if (!gate) {
      gate = Promise.withResolvers<void>();
      this.gates.set(name, gate);
    }
    return gate.promise;
  }
}

class TrailTestClient {
  private buffer = "";
  private readonly frames: Frame[] = [];
  private readonly waiters: Array<{ predicate: (frame: Frame) => boolean; resolve: (frame: Frame) => void }> = [];

  private constructor(private readonly socket: Socket) {
    socket.setEncoding("utf8");
    socket.on("data", (chunk) =>
      this.receive(typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk)),
    );
  }

  static async connect(socketPath: string): Promise<TrailTestClient> {
    const socket = connect({ path: socketPath });
    await new Promise<void>((resolveConnection, rejectConnection) => {
      socket.once("connect", resolveConnection);
      socket.once("error", rejectConnection);
    });
    return new TrailTestClient(socket);
  }

  close(): void {
    this.socket.destroy();
  }

  request(id: number, method: string, params: Record<string, unknown>): Promise<Record<string, unknown>> {
    this.socket.write(`${JSON.stringify({ id, jsonrpc: "2.0", method, params })}\n`);
    return this.next((frame) => frame.id === id).then((frame) => {
      if (frame.error) {
        throw new Error(`RPC ${method} failed: ${JSON.stringify(frame.error)}`);
      }
      return frame.result as Record<string, unknown>;
    });
  }

  nextEvent(type: string): Promise<Frame> {
    return this.next(
      (frame) => frame.method === "event" && frame.params?.type === type,
    );
  }

  private next(predicate: (frame: Frame) => boolean): Promise<Frame> {
    const index = this.frames.findIndex(predicate);
    if (index >= 0) {
      const frame = this.frames.splice(index, 1)[0];
      if (frame) return Promise.resolve(frame);
    }
    return new Promise((resolveFrame) => this.waiters.push({ predicate, resolve: resolveFrame }));
  }

  private receive(chunk: string): void {
    this.buffer += chunk;
    let newline = this.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      if (line.trim()) this.handle(JSON.parse(line) as Frame);
      newline = this.buffer.indexOf("\n");
    }
  }

  private handle(frame: Frame): void {
    const index = this.waiters.findIndex((waiter) => waiter.predicate(frame));
    const waiter = index >= 0 ? this.waiters.splice(index, 1)[0] : undefined;
    if (waiter) {
      waiter.resolve(frame);
      return;
    }
    this.frames.push(frame);
  }
}

interface Frame {
  readonly id?: number | string;
  readonly method?: string;
  readonly params?: { readonly type?: string };
  readonly result?: Record<string, unknown>;
  readonly error?: Record<string, unknown>;
}
