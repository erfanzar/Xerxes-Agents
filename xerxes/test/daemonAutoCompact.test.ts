// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { createCompactionAgent } from "../src/agents/compactionAgent.js";
import { InMemoryDaemonRuntime } from "../src/daemon/runtime.js";
import { DaemonServer } from "../src/daemon/server.js";
import { ProfileStore } from "../src/bridge/profiles.js";
import type { FetchImplementation } from "../src/llms/client.js";
import type { DaemonEvent, DaemonSession, TurnRunner } from "../src/daemon/runtime.js";

interface Frame {
  readonly error?: {
    readonly code?: number;
    readonly message?: string;
  };
  readonly id?: number;
  readonly method?: string;
  readonly params?: {
    readonly payload?: Record<string, unknown>;
    readonly type?: string;
  };
  readonly result?: Record<string, unknown>;
}

function eventFrame(type: string): (frame: Frame) => boolean {
  return (frame) => frame.method === "event" && frame.params?.type === type;
}

function notificationWith(
  text: string,
): (frame: Frame) => boolean {
  return (frame) =>
    frame.method === "event" &&
    frame.params?.type === "notification" &&
    String(frame.params?.payload?.body ?? "").includes(text);
}

class SocketTestClient {
  private buffer = "";
  private readonly frames: Frame[] = [];
  private readonly waiters: Array<{
    predicate: (frame: Frame) => boolean;
    resolve: (frame: Frame) => void;
  }> = [];

  private constructor(private readonly socket: Socket) {
    socket.setEncoding("utf8");
    socket.on("data", (chunk) =>
      this.receive(
        typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk),
      ),
    );
  }

  static async connect(socketPath: string): Promise<SocketTestClient> {
    const socket = connect({ path: socketPath });
    await new Promise<void>((resolve, reject) => {
      socket.once("connect", resolve);
      socket.once("error", reject);
    });
    return new SocketTestClient(socket);
  }

  close(): void {
    this.socket.destroy();
  }

  /** Buffered frames matching `predicate`, for asserting that something happened only once. */
  matching(predicate: (frame: Frame) => boolean): Frame[] {
    return this.frames.filter(predicate);
  }

  next(predicate: (frame: Frame) => boolean): Promise<Frame> {
    const index = this.frames.findIndex(predicate);
    if (index >= 0) {
      const frame = this.frames.splice(index, 1)[0];
      if (frame) {
        return Promise.resolve(frame);
      }
    }
    return new Promise((resolve) => this.waiters.push({ predicate, resolve }));
  }

  send(frame: Record<string, unknown>): void {
    this.socket.write(`${JSON.stringify(frame)}\n`);
  }

  private receive(chunk: string): void {
    this.buffer += chunk;
    let newline = this.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      newline = this.buffer.indexOf("\n");
      if (!line.trim()) {
        continue;
      }
      let frame: Frame;
      try {
        frame = JSON.parse(line) as Frame;
      } catch {
        continue;
      }
      const waiterIndex = this.waiters.findIndex((waiter) =>
        waiter.predicate(frame)
      );
      if (waiterIndex >= 0) {
        const waiter = this.waiters.splice(waiterIndex, 1)[0];
        waiter?.resolve(frame);
      } else {
        this.frames.push(frame);
      }
    }
  }
}

function fakeOpenAiFetch(requests: unknown[]): FetchImplementation {
  return async (input, init) => {
    const url = typeof input === "string" ? input : input.toString();
    if (!url.includes("/chat/completions")) {
      return new Response(JSON.stringify({ error: "unexpected endpoint" }), {
        status: 404,
      });
    }
    const body = typeof init?.body === "string"
      ? JSON.parse(init.body)
      : undefined;
    requests.push(body);
    return new Response(
      JSON.stringify({
        choices: [{ message: { content: "durable auto-compact summary" } }],
      }),
    );
  };
}

/** An OpenAI-compatible fake driven by a per-call script of responses. */
function scriptedOpenAiFetch(
  requests: Array<Record<string, unknown>>,
  respond: (call: number) => Response,
): FetchImplementation {
  return async (input, init) => {
    const url = typeof input === "string" ? input : input.toString();
    if (!url.includes("/chat/completions")) {
      return new Response(JSON.stringify({ error: "unexpected endpoint" }), {
        status: 404,
      });
    }
    const body = typeof init?.body === "string"
      ? (JSON.parse(init.body) as Record<string, unknown>)
      : {};
    requests.push(body);
    return respond(requests.length);
  };
}

function contextOverflowResponse(): Response {
  return new Response(
    JSON.stringify({
      error: {
        message: "This model's maximum context length is 128000 tokens",
        code: "context_length_exceeded",
      },
    }),
    { status: 400 },
  );
}

function seedTranscript(
  runtime: InMemoryDaemonRuntime,
  sessionKey: string,
): void {
  const session = runtime.sessionStatus(sessionKey);
  if (!session) throw new Error("session must exist before seeding");
  const filler = "transcript filler ".repeat(300);
  session.messages.push(
    { role: "user", content: `first request ${filler}` },
    { role: "assistant", content: `first answer ${filler}` },
    { role: "user", content: "second request" },
    { role: "assistant", content: "second answer" },
    { role: "user", content: "third request" },
    { role: "assistant", content: "third answer" },
  );
}

test("forced compaction summarizes tool-heavy transcripts instead of silently no-oping", async () => {
  const bigTool = "x".repeat(20_000);
  const toolCall = (id: string) => ({
    id,
    type: "function",
    function: { name: "ReadFile", arguments: "{}" },
  });
  const messages = [
    { role: "system", content: "system prompt" },
    { role: "user", content: "please inspect the repo" },
    { role: "assistant", content: "reading files", tool_calls: [toolCall("c1")] },
    { role: "tool", tool_call_id: "c1", content: bigTool },
    { role: "assistant", content: "more reading", tool_calls: [toolCall("c2")] },
    { role: "tool", tool_call_id: "c2", content: bigTool },
    { role: "assistant", content: "still reading", tool_calls: [toolCall("c3")] },
    { role: "tool", tool_call_id: "c3", content: bigTool },
    { role: "user", content: "what did you find?" },
    { role: "assistant", content: "here is the summary so far" },
  ];

  let completionCalls = 0;
  const agent = createCompactionAgent({
    model: "gpt-4",
    completion: () => {
      completionCalls += 1;
      return "REAL LLM SUMMARY";
    },
  });
  const compacted = await agent.summarizeMessages(messages);

  // Regression: pruning alone dropped the transcript under the compressor's
  // internal threshold, so the summary path never ran and /compact no-oped.
  expect(completionCalls).toBe(1);
  const serialized = JSON.stringify(compacted);
  expect(serialized).toContain("REAL LLM SUMMARY");
  expect(serialized.length).toBeLessThan(JSON.stringify(messages).length);
});

test("daemon auto-compacts before submitting a turn once usage crosses the threshold", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-autocompact-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    name: "openai-test",
    apiKey: "fake-api-key",
    baseUrl: "https://api.openai.test",
    model: "gpt-4",
    provider: "openai",
    setActive: true,
  });
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "gpt-4",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    autoCompactThreshold: 0.01,
    autoTitle: false,
    socketPath,
    runtime,
    profileStore,
  });
  const nativeFetch = globalThis.fetch;
  const requests: unknown[] = [];
  globalThis.fetch = fakeOpenAiFetch(requests) as typeof globalThis.fetch;
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "auto-compact", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    seedTranscript(runtime, "auto-compact");

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { text: "hello" },
    });
    await client.next((frame) => frame.id === 2);
    const notice = await client.next(notificationWith("auto-compacting"));
    expect(String(notice.params?.payload?.body ?? "")).toContain("%");
    const completed = await client.next(notificationWith("Auto-compacted"));
    expect(completed.params?.payload).toMatchObject({
      category: "history",
      type: "compaction",
      payload: {
        automatic: true,
        tokens_after: expect.any(Number),
        tokens_before: expect.any(Number),
      },
    });
    await client.next(eventFrame("turn_begin"));
    await client.next(eventFrame("turn_end"));

    // The only provider call is the compaction summary; the echo runner
    // itself never talks to a model.
    expect(requests.length).toBe(1);
    const prompt = String(
      (requests[0] as { messages?: Array<{ content?: unknown }> })
        ?.messages?.[0]?.content ?? "",
    );
    expect(prompt).toContain("CONTEXT TO SUMMARIZE");

    const session = runtime.sessionStatus("auto-compact");
    expect(session?.metadata.last_compaction).toBeDefined();
    expect(JSON.stringify(session?.messages ?? [])).toContain(
      "durable auto-compact summary",
    );
  } finally {
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("cancel during pre-turn auto-compaction prevents the admitted turn from launching", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-autocompact-cancel-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    name: "openai-test",
    apiKey: "fake-api-key",
    baseUrl: "https://api.openai.test",
    model: "gpt-4",
    provider: "openai",
    setActive: true,
  });
  const compactionStarted = Promise.withResolvers<void>();
  const releaseCompaction = Promise.withResolvers<void>();
  let runnerCalls = 0;
  const runtime = new InMemoryDaemonRuntime({
    async *run(): AsyncGenerator<DaemonEvent> {
      runnerCalls += 1;
      yield { type: "text_part", payload: { text: "must not run" } };
    },
  }, {
    currentProjectDirectory: directory,
    model: "gpt-4",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    autoCompactThreshold: 0.01,
    autoTitle: false,
    socketPath,
    runtime,
    profileStore,
  });
  const nativeFetch = globalThis.fetch;
  globalThis.fetch = Object.assign(
    async () => {
      compactionStarted.resolve();
      await releaseCompaction.promise;
      return new Response(JSON.stringify({ choices: [{ message: { content: "summary" } }] }));
    },
    { preconnect: globalThis.fetch.preconnect },
  );
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "cancel-before-launch", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    seedTranscript(runtime, "cancel-before-launch");

    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "hello" } });
    await client.next((frame) => frame.id === 2);
    await compactionStarted.promise;
    client.send({ jsonrpc: "2.0", id: 3, method: "turn.cancel", params: {} });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({ ok: true });

    releaseCompaction.resolve();
    await Bun.sleep(50);

    expect(runnerCalls).toBe(0);
    expect(runtime.sessionStatus("cancel-before-launch")).toMatchObject({
      activeTurnId: "",
      cancelRequested: true,
      status: "idle",
    });
  } finally {
    releaseCompaction.resolve();
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("disconnect during pre-turn auto-compaction does not start an ownerless turn", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-autocompact-disconnect-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    name: "openai-test",
    apiKey: "fake-api-key",
    baseUrl: "https://api.openai.test",
    model: "gpt-4",
    provider: "openai",
    setActive: true,
  });
  const compactionStarted = Promise.withResolvers<void>();
  const releaseCompaction = Promise.withResolvers<void>();
  let runnerCalls = 0;
  const runner: TurnRunner = {
    async *run(_session: DaemonSession, _text: string, _signal: AbortSignal): AsyncGenerator<DaemonEvent> {
      runnerCalls += 1;
      yield { type: "text_part", payload: { text: "must not run" } };
    },
  };
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "gpt-4",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    autoCompactThreshold: 0.01,
    autoTitle: false,
    socketPath,
    runtime,
    profileStore,
  });
  const nativeFetch = globalThis.fetch;
  globalThis.fetch = Object.assign(
    async () => {
      compactionStarted.resolve();
      await releaseCompaction.promise;
      return new Response(JSON.stringify({ choices: [{ message: { content: "summary" } }] }));
    },
    { preconnect: globalThis.fetch.preconnect },
  );
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "disconnecting", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    seedTranscript(runtime, "disconnecting");

    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "hello" } });
    await client.next((frame) => frame.id === 2);
    await compactionStarted.promise;
    client.close();
    await Bun.sleep(25);
    releaseCompaction.resolve();
    await Bun.sleep(50);

    expect(runnerCalls).toBe(0);
    expect(runtime.sessionStatus("disconnecting")?.activeTurnId).toBe("");
  } finally {
    releaseCompaction.resolve();
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("a second submit cannot wait behind compaction and launch after its owner disconnects", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-autocompact-double-submit-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    name: "openai-test",
    apiKey: "fake-api-key",
    baseUrl: "https://api.openai.test",
    model: "gpt-4",
    provider: "openai",
    setActive: true,
  });
  const compactionStarted = Promise.withResolvers<void>();
  const releaseCompaction = Promise.withResolvers<void>();
  const submitted: string[] = [];
  const runtime = new InMemoryDaemonRuntime({
    async *run(_session, text): AsyncGenerator<DaemonEvent> {
      submitted.push(text);
      yield { type: "text_part", payload: { text } };
    },
  }, {
    currentProjectDirectory: directory,
    model: "gpt-4",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    autoCompactThreshold: 0.01,
    autoTitle: false,
    socketPath,
    runtime,
    profileStore,
  });
  const nativeFetch = globalThis.fetch;
  globalThis.fetch = Object.assign(
    async () => {
      compactionStarted.resolve();
      await releaseCompaction.promise;
      return new Response(JSON.stringify({ choices: [{ message: { content: "summary" } }] }));
    },
    { preconnect: globalThis.fetch.preconnect },
  );
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { session_key: "double", project_dir: directory } });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    seedTranscript(runtime, "double");

    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "first" } });
    await client.next((frame) => frame.id === 2);
    await compactionStarted.promise;
    client.send({ jsonrpc: "2.0", id: 3, method: "turn.submit", params: { text: "second" } });
    await client.next((frame) => frame.id === 3);
    client.close();
    await Bun.sleep(25);
    releaseCompaction.resolve();
    await Bun.sleep(50);

    expect(submitted).toEqual([]);
    expect(runtime.sessionStatus("double")?.activeTurnId).toBe("");
  } finally {
    releaseCompaction.resolve();
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("a submit serialized behind manual compaction keeps its newly appended prompt", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-compact-prompt-race-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    name: "openai-test",
    apiKey: "fake-api-key",
    baseUrl: "https://api.openai.test",
    model: "gpt-4",
    provider: "openai",
    setActive: true,
  });
  const compactionStarted = Promise.withResolvers<void>();
  const releaseCompaction = Promise.withResolvers<void>();
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "gpt-4",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    autoCompactThreshold: 0.01,
    autoTitle: false,
    socketPath,
    runtime,
    profileStore,
  });
  const nativeFetch = globalThis.fetch;
  globalThis.fetch = Object.assign(
    async () => {
      compactionStarted.resolve();
      await releaseCompaction.promise;
      return new Response(JSON.stringify({ choices: [{ message: { content: "summary" } }] }));
    },
    { preconnect: globalThis.fetch.preconnect },
  );
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  const submitter = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { session_key: "prompt-race", project_dir: directory } });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    submitter.send({ jsonrpc: "2.0", id: 10, method: "initialize", params: { session_key: "prompt-race", project_dir: directory } });
    await submitter.next((frame) => frame.id === 10);
    await submitter.next(eventFrame("init_done"));
    await submitter.next(eventFrame("status_update"));
    seedTranscript(runtime, "prompt-race");

    client.send({ jsonrpc: "2.0", id: 2, method: "session.compress", params: {} });
    await compactionStarted.promise;
    submitter.send({ jsonrpc: "2.0", id: 3, method: "turn.submit", params: { text: "new prompt must survive" } });
    await submitter.next((frame) => frame.id === 3);
    releaseCompaction.resolve();
    await client.next((frame) => frame.id === 2);
    await submitter.next(eventFrame("turn_end"));

    expect(JSON.stringify(runtime.sessionStatus("prompt-race")?.messages ?? []))
      .toContain("new prompt must survive");
  } finally {
    releaseCompaction.resolve();
    globalThis.fetch = nativeFetch;
    client.close();
    submitter.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("manual compaction does not restore stale idle status over a live turn", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-compact-status-race-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    name: "openai-test",
    apiKey: "fake-api-key",
    baseUrl: "https://api.openai.test",
    model: "gpt-4",
    provider: "openai",
    setActive: true,
  });
  const compactionStarted = Promise.withResolvers<void>();
  const releaseCompaction = Promise.withResolvers<void>();
  const releaseTurn = Promise.withResolvers<void>();
  const runtime = new InMemoryDaemonRuntime({
    async *run(): AsyncGenerator<DaemonEvent> {
      await releaseTurn.promise;
      yield { type: "text_part", payload: { text: "done" } };
    },
  }, {
    currentProjectDirectory: directory,
    model: "gpt-4",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ autoTitle: false, socketPath, runtime, profileStore });
  const nativeFetch = globalThis.fetch;
  globalThis.fetch = Object.assign(
    async () => {
      compactionStarted.resolve();
      await releaseCompaction.promise;
      return new Response(JSON.stringify({ choices: [{ message: { content: "summary" } }] }));
    },
    { preconnect: globalThis.fetch.preconnect },
  );
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { session_key: "status-race", project_dir: directory } });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    seedTranscript(runtime, "status-race");

    client.send({ jsonrpc: "2.0", id: 2, method: "session.compress", params: {} });
    await compactionStarted.promise;
    const liveTurn = runtime.submitTurn("status-race", "live", () => undefined);
    await Bun.sleep(10);
    expect(runtime.sessionStatus("status-race")?.status).toBe("working");
    releaseCompaction.resolve();
    await client.next((frame) => frame.id === 2);

    expect(runtime.sessionStatus("status-race")?.status).toBe("working");
    releaseTurn.resolve();
    await liveTurn;
  } finally {
    releaseCompaction.resolve();
    releaseTurn.resolve();
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("default auto-compaction threshold is 80% of the prompt budget", async () => {
  const { DEFAULT_AUTO_COMPACT_THRESHOLD } = await import("../src/daemon/compactionRunner.js");
  expect(DEFAULT_AUTO_COMPACT_THRESHOLD).toBe(0.8);
});

test("daemon leaves small transcripts alone under the default 80% threshold", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-autocompact-off-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    name: "openai-test",
    apiKey: "fake-api-key",
    baseUrl: "https://api.openai.test",
    model: "gpt-4",
    provider: "openai",
    setActive: true,
  });
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "gpt-4",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ autoTitle: false, socketPath, runtime, profileStore });
  const nativeFetch = globalThis.fetch;
  const requests: unknown[] = [];
  globalThis.fetch = fakeOpenAiFetch(requests) as typeof globalThis.fetch;
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "small-transcript", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    seedTranscript(runtime, "small-transcript");

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { text: "hello" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    await client.next(eventFrame("turn_end"));

    // ~1.7k tokens against the 128k window is nowhere near 90%: no provider
    // call, no compaction metadata.
    expect(requests.length).toBe(0);
    expect(
      runtime.sessionStatus("small-transcript")?.metadata.last_compaction,
    ).toBeUndefined();
  } finally {
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

interface CompactionHarnessOptions {
  readonly fetch: FetchImplementation;
  readonly model?: string;
  readonly runtimeSettings?: Record<string, unknown>;
  readonly sessionKey: string;
  readonly threshold?: number;
}

interface CompactionHarness {
  readonly client: SocketTestClient;
  readonly runtime: InMemoryDaemonRuntime;
  submit(id: number): Promise<void>;
}

/** Boot a daemon whose only provider traffic is compaction, on an isolated cron lease. */
async function withCompactionDaemon(
  label: string,
  options: CompactionHarnessOptions,
  body: (harness: CompactionHarness) => Promise<void>,
): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), `xerxes-bun-${label}-`));
  const socketPath = join(directory, "daemon.sock");
  const model = options.model ?? "gpt-4";
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    name: "openai-test",
    apiKey: "fake-api-key",
    baseUrl: "https://api.openai.test",
    model,
    provider: "openai",
    setActive: true,
  });
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model,
    ...(options.runtimeSettings ? { runtimeSettings: options.runtimeSettings } : {}),
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    ...(options.threshold === undefined ? {} : { autoCompactThreshold: options.threshold }),
    autoTitle: false,
    cronLeasePath: join(directory, "cron.lease"),
    socketPath,
    runtime,
    profileStore,
  });
  const nativeFetch = globalThis.fetch;
  globalThis.fetch = options.fetch as typeof globalThis.fetch;
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: options.sessionKey, project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    await body({
      client,
      runtime,
      submit: async (id) => {
        client.send({
          jsonrpc: "2.0",
          id,
          method: "turn.submit",
          params: { text: "hello" },
        });
        await client.next((frame) => frame.id === id);
        await client.next(eventFrame("turn_begin"));
        await client.next(eventFrame("turn_end"));
      },
    });
  } finally {
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
}

/** Seed roughly `tokens` worth of transcript, using the estimator's own accounting. */
function seedLargeTranscript(
  runtime: InMemoryDaemonRuntime,
  sessionKey: string,
  tokens: number,
): void {
  const session = runtime.sessionStatus(sessionKey);
  if (!session) throw new Error("session must exist before seeding");
  // "word " is one lexical token per repetition and five characters, so the
  // length/4 floor dominates at 1.25 tokens per repetition.
  const filler = "word ".repeat(Math.ceil(tokens / 1.25));
  session.messages.push(
    { role: "user", content: filler },
    { role: "assistant", content: "acknowledged" },
  );
}

test("context accounting prices tool_calls, so a tool-heavy session compacts before it overflows", async () => {
  const requests: Array<Record<string, unknown>> = [];
  await withCompactionDaemon(
    "autocompact-toolcalls",
    {
      fetch: scriptedOpenAiFetch(
        requests,
        () => new Response(JSON.stringify({ choices: [{ message: { content: "tool summary" } }] })),
      ),
      sessionKey: "tool-heavy",
      // 30% of the 96_000-token prompt budget for gpt-4.
      threshold: 0.3,
    },
    async ({ client, runtime, submit }) => {
      const session = runtime.sessionStatus("tool-heavy");
      if (!session) throw new Error("session must exist before seeding");
      // Every token lives in the tool calls' arguments. Mapping messages to
      // {role, content} dropped those entirely, so the window read as nearly
      // empty right up until the provider rejected the request.
      const writeCall = (id: string) => ({
        id,
        type: "function",
        function: {
          name: "WriteFile",
          arguments: JSON.stringify({ patch: "word ".repeat(15_000) }),
        },
      });
      session.messages.push(
        { role: "user", content: "rewrite the module" },
        { role: "assistant", content: "", tool_calls: [writeCall("call-1")] },
        { role: "tool", tool_call_id: "call-1", content: "written" },
        { role: "assistant", content: "", tool_calls: [writeCall("call-2")] },
        { role: "tool", tool_call_id: "call-2", content: "written" },
        { role: "assistant", content: "", tool_calls: [writeCall("call-3")] },
        { role: "tool", tool_call_id: "call-3", content: "written" },
        { role: "user", content: "and now the tests" },
        { role: "assistant", content: "on it" },
      );

      client.send({ jsonrpc: "2.0", id: 2, method: "session.usage", params: {} });
      const usage = await client.next((frame) => frame.id === 2);
      expect(Number(usage.result?.context_used ?? 0)).toBeGreaterThan(40_000);

      await submit(3);
      await client.next(notificationWith("Auto-compacted"));
      expect(requests).toHaveLength(1);
    },
  );
});

test("a compaction call that overflows the window retries with a smaller summary budget", async () => {
  const requests: Array<Record<string, unknown>> = [];
  await withCompactionDaemon(
    "autocompact-overflow",
    {
      fetch: scriptedOpenAiFetch(requests, (call) =>
        call === 1
          ? contextOverflowResponse()
          : new Response(
            JSON.stringify({ choices: [{ message: { content: "second-try summary" } }] }),
          )),
      sessionKey: "overflow",
      threshold: 0.01,
    },
    async ({ client, runtime, submit }) => {
      seedTranscript(runtime, "overflow");
      await submit(2);
      await client.next(notificationWith("Auto-compacted"));

      expect(requests).toHaveLength(2);
      // One lever, halved: the summary's token budget.
      expect(requests[0]?.max_tokens).toBe(8_192);
      expect(requests[1]?.max_tokens).toBe(4_096);
      expect(JSON.stringify(runtime.sessionStatus("overflow")?.messages ?? []))
        .toContain("second-try summary");
    },
  );
});

test("a response holding no text is permanent and is never retried", async () => {
  const requests: Array<Record<string, unknown>> = [];
  await withCompactionDaemon(
    "autocompact-shape",
    {
      // 200 OK with a choice that carries no content: the shape will never
      // parse, so a second full-window call only wastes tokens.
      fetch: scriptedOpenAiFetch(
        requests,
        () => new Response(JSON.stringify({ choices: [{ message: {} }] })),
      ),
      sessionKey: "bad-shape",
      threshold: 0.01,
    },
    async ({ client, runtime, submit }) => {
      seedTranscript(runtime, "bad-shape");
      await submit(2);
      await client.next(notificationWith("Auto-compaction skipped"));

      expect(requests).toHaveLength(1);
      expect(
        runtime.sessionStatus("bad-shape")?.metadata.last_compaction,
      ).toBeUndefined();
    },
  );
});

test("auto-compaction stops after three consecutive failures instead of retrying every turn", async () => {
  const requests: Array<Record<string, unknown>> = [];
  await withCompactionDaemon(
    "autocompact-loop",
    {
      fetch: scriptedOpenAiFetch(requests, () => contextOverflowResponse()),
      sessionKey: "looping",
      threshold: 0.01,
    },
    async ({ client, runtime, submit }) => {
      seedTranscript(runtime, "looping");
      await submit(2);
      await client.next(notificationWith("Auto-compaction skipped"));
      await submit(3);
      await client.next(notificationWith("Auto-compaction skipped"));
      await submit(4);
      const bail = await client.next(notificationWith("now off for this session"));
      expect(String(bail.params?.payload?.body ?? "")).toContain("/compact");

      // Three attempts, each exhausting the three summary budgets.
      const callsBeforeBail = requests.length;
      expect(callsBeforeBail).toBe(9);
      await submit(5);
      expect(requests).toHaveLength(callsBeforeBail);
    },
  );
});

test("a disabled threshold still warns once as the prompt budget fills", async () => {
  const requests: Array<Record<string, unknown>> = [];
  await withCompactionDaemon(
    "autocompact-warn",
    {
      fetch: scriptedOpenAiFetch(
        requests,
        () => new Response(JSON.stringify({ choices: [{ message: { content: "unused" } }] })),
      ),
      // 8_192-token window with a 2_048-token reply reserve: a 6_144-token
      // prompt budget, so the warning fires without a huge fixture.
      model: "moonshot-v1-8k",
      runtimeSettings: { auto_compact_threshold: 0 },
      sessionKey: "warned",
    },
    async ({ client, runtime, submit }) => {
      seedLargeTranscript(runtime, "warned", 6_000);
      await submit(2);
      const warning = await client.next(notificationWith("auto-compaction is disabled"));
      expect(String(warning.params?.payload?.body ?? "")).toContain("6,144");
      expect(requests).toHaveLength(0);

      // Once, not once per turn: the warning is emitted before the turn runs,
      // so a second one would already be buffered by the time turn_end lands.
      await submit(3);
      expect(client.matching(notificationWith("auto-compaction is disabled"))).toHaveLength(0);
    },
  );
});

test("runtime setting auto_compact_threshold = 0 disables auto-compaction", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-autocompact-zero-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    name: "openai-test",
    apiKey: "fake-api-key",
    baseUrl: "https://api.openai.test",
    model: "gpt-4",
    provider: "openai",
    setActive: true,
  });
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "gpt-4",
    runtimeSettings: { auto_compact_threshold: 0 },
    sessionDirectory: join(directory, "sessions"),
  });
  // A low constructor default must lose to the explicit runtime setting.
  const server = new DaemonServer({
    autoCompactThreshold: 0.01,
    autoTitle: false,
    socketPath,
    runtime,
    profileStore,
  });
  const nativeFetch = globalThis.fetch;
  const requests: unknown[] = [];
  globalThis.fetch = fakeOpenAiFetch(requests) as typeof globalThis.fetch;
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "disabled", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    seedTranscript(runtime, "disabled");

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { text: "hello" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    await client.next(eventFrame("turn_end"));

    expect(requests.length).toBe(0);
    expect(
      runtime.sessionStatus("disabled")?.metadata.last_compaction,
    ).toBeUndefined();
  } finally {
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});
