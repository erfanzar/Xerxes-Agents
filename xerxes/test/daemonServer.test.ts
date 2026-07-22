// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdir, mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { InMemoryDaemonRuntime } from "../src/daemon/runtime.js";
import { DaemonInteractionBoard } from "../src/daemon/interactions.js";
import { DaemonServer, MIGRATED_ERROR } from "../src/daemon/server.js";
import { ProfileStore } from "../src/bridge/profiles.js";
import {
  ChannelManager,
  type Channel,
  type ChannelMessage,
  type InboundHandler,
} from "../src/channels/index.js";
import { CronJob, JobStore } from "../src/cron/jobs.js";
import { DaemonTranscriptStore } from "../src/session/daemonTranscript.js";
import { SnapshotManager } from "../src/session/snapshots.js";
import type { FetchImplementation } from "../src/llms/client.js";
import type { PermissionRequest } from "../src/streaming/events.js";
import type {
  DaemonEvent,
  DaemonSession,
  SubmitTurnOptions,
  TurnRunControls,
  TurnRunner,
} from "../src/daemon/runtime.js";

test("daemon preserves JSON-RPC v35 NDJSON responses and stream event framing", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-daemon-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "protocol-model",
      sessionDirectory: join(directory, "sessions"),
      statusInventory: () => ({ activeSubagents: 2 }),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "runtime.status",
      params: {},
    });
    const status = await client.next((frame) => frame.id === 1);
    expect(status.result).toMatchObject({
      ok: true,
      runtime_ready: true,
      active_subagents: 2,
      daemon_protocol: 35,
      runtime: "bun-typescript",
    });

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "initialize",
      params: { session_key: "test-session" },
    });
    const initialized = await client.next((frame) => frame.id === 2);
    const initDone = await client.next(eventFrame("init_done"));
    const initialStatus = await client.next(eventFrame("status_update"));
    expect(initialized.result).toMatchObject({
      ok: true,
      session: { key: "test-session", status: "idle" },
    });
    expect(initDone.params?.payload).toMatchObject({
      session_id: expect.any(String),
      context_limit: 128_000,
      mode: "code",
    });
    expect(initialStatus.params?.payload).toMatchObject({
      max_context: 128_000,
      mode: "code",
    });

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.open",
      params: { session_key: "test-session" },
    });
    const opened = await client.next((frame) => frame.id === 3);
    expect(opened.result).toMatchObject({
      ok: true,
      session: { key: "test-session", messages: 0, status: "idle" },
    });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "turn.submit",
      params: { session_key: "test-session", text: "hello" },
    });
    expect((await client.next((frame) => frame.id === 4)).result).toMatchObject(
      { ok: true },
    );
    const turnBegin = await client.next(eventFrame("turn_begin"));
    const textPart = await client.next(eventFrame("text_part"));
    const turnEnd = await client.next(eventFrame("turn_end"));
    expect(turnBegin.params?.payload).toMatchObject({ text: "hello" });
    expect(textPart.params?.payload).toMatchObject({
      text: "Bun daemon foundation received: hello",
    });
    expect(turnEnd.params?.payload).toMatchObject({ cancelled: false });

    client.send({ jsonrpc: "2.0", id: 5, method: "task.submit", params: {} });
    expect((await client.next((frame) => frame.id === 5)).result).toEqual({
      ok: false,
      error: MIGRATED_ERROR,
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("unconfigured daemon status is neutral and turn submission rejects model inference", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-unconfigured-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "unconfigured" },
    });
    const initialized = await client.next((frame) => frame.id === 1);
    const initDone = await client.next(eventFrame("init_done"));
    const status = await client.next(eventFrame("status_update"));
    expect(initialized.result?.session).toMatchObject({
      model: "",
      context_limit: 0,
      max_context: 0,
    });
    expect(initDone.params?.payload).toMatchObject({ model: "", context_limit: 0 });
    expect(status.params?.payload).toMatchObject({ model: "", max_context: 0 });

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "unconfigured", text: "do not guess" },
    });
    expect((await client.next((frame) => frame.id === 2)).error).toEqual({
      code: -32000,
      message: expect.stringContaining(
        "Configuration model: is not configured; select a provider model",
      ),
    });

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.status",
      params: { session_key: "unconfigured" },
    });
    expect((await client.next((frame) => frame.id === 3)).result?.session).toMatchObject({
      model: "",
      messages: 0,
      turn_count: 0,
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon context limits follow the active provider and live model metadata", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-context-limit-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    apiKey: "",
    baseUrl: "https://api.kimi.com/coding/v1",
    model: "k3",
    name: "kimi-code",
    provider: "kimi-code",
  });
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    runtimeSettings: {
      base_url: "https://api.kimi.com/coding/v1",
      model: "k3",
      provider: "kimi-code",
    },
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ profileStore, runtime, socketPath });
  const nativeFetch = globalThis.fetch;
  let modelFetchCount = 0;
  const modelFetch: FetchImplementation = async () => {
    modelFetchCount += 1;
    return new Response(
      JSON.stringify({
        data: [
          modelFetchCount === 1
            ? { id: "k3", context_length: 400_000 }
            : { id: "k3" },
        ],
      }),
    );
  };
  globalThis.fetch = modelFetch as typeof globalThis.fetch;
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "provider-context" },
    });
    expect((await client.next((frame) => frame.id === 1)).result).toMatchObject({
      context_limit: 262_144,
      session: { context_limit: 262_144, max_context: 262_144 },
    });
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "fetch_models",
      params: { profile_name: "kimi-code" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({
      models: ["k3"],
      ok: true,
      source: "remote",
    });

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.status",
      params: { session_key: "provider-context" },
    });
    expect((await client.next((frame) => frame.id === 3)).result?.session).toMatchObject({
      context_limit: 400_000,
      max_context: 400_000,
    });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "fetch_models",
      params: { profile_name: "kimi-code" },
    });
    expect((await client.next((frame) => frame.id === 4)).result).toMatchObject({
      models: ["k3"],
      ok: true,
      source: "remote",
    });

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "session.status",
      params: { session_key: "provider-context" },
    });
    expect((await client.next((frame) => frame.id === 5)).result?.session).toMatchObject({
      context_limit: 262_144,
      max_context: 262_144,
    });
  } finally {
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("session.list scopes history to the active project and exposes additive subagent hierarchy fields", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-session-list-"));
  const projectDirectory = join(directory, "project-a");
  const otherProjectDirectory = join(directory, "project-b");
  const sessionDirectory = join(directory, "sessions");
  const socketPath = join(directory, "daemon.sock");
  await mkdir(sessionDirectory, { recursive: true });

  const writeTranscript = async (
    sessionId: string,
    projectRoot: string,
    updatedAt: string,
    metadata: Record<string, unknown>,
  ) => {
    await writeFile(
      join(sessionDirectory, `${sessionId}.json`),
      JSON.stringify({
        format: "xerxes-daemon-session",
        schema_version: 2,
        session_id: sessionId,
        key: sessionId,
        agent_id: "default",
        cwd: projectRoot,
        workspace: "",
        updated_at: updatedAt,
        messages: [
          { role: "user", content: `request ${sessionId}` },
          { role: "assistant", content: `response ${sessionId}` },
        ],
        turn_count: 1,
        interaction_mode: "code",
        plan_mode: false,
        total_input_tokens: 1,
        total_output_tokens: 1,
        metadata: { project_root: projectRoot, ...metadata },
        thinking_content: [],
        tool_executions: [],
      }),
      "utf8",
    );
  };

  await writeTranscript("aaaabbbb0001", projectDirectory, "2026-07-17T00:02:00.000Z", {
    model: "root-model",
    title: "Project root",
  });
  await writeTranscript("aaaabbbb0002", projectDirectory, "2026-07-17T00:01:00.000Z", {
    parent_session_id: "aaaabbbb0001",
    title: "Regular branch",
  });
  await writeTranscript("ccccdddd0001", projectDirectory, "2026-07-17T00:03:00.000Z", {
    model: "child-model",
    parent_session_id: "aaaabbbb0001",
    root_session_id: "aaaabbbb0001",
    session_kind: "subagent",
    status: "completed",
    subagent_id: "subagent_child_one",
    title: "Child history",
  });
  await writeTranscript("eeeeffff0001", otherProjectDirectory, "2026-07-17T00:04:00.000Z", {
    title: "Other project root",
  });

  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: projectDirectory,
      sessionDirectory,
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { project_dir: projectDirectory, session_key: "session-list" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "session.list",
      params: { kind: "main", limit: 10 },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
      sessions: [
        expect.objectContaining({
          id: "aaaabbbb0001",
          kind: "main",
          model: "root-model",
          session_kind: "main",
        }),
        expect.objectContaining({
          id: "aaaabbbb0002",
          kind: "main",
          session_kind: "main",
          title: "Regular branch",
        }),
      ],
    });

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.list",
      params: { kind: "subagent", limit: 10 },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
      sessions: [
        expect.objectContaining({
          id: "ccccdddd0001",
          kind: "subagent",
          model: "child-model",
          parent_session_id: "aaaabbbb0001",
          root_session_id: "aaaabbbb0001",
          session_kind: "subagent",
          status: "completed",
          subagent_id: "subagent_child_one",
        }),
      ],
    });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "session.list",
      params: { kind: "main", limit: 10, scope: "global" },
    });
    const global = (await client.next((frame) => frame.id === 4)).result?.sessions as Array<Record<string, unknown>>;
    expect(global.map((session) => session.id)).toEqual([
      "eeeeffff0001",
      "aaaabbbb0001",
      "aaaabbbb0002",
    ]);

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "session.list",
      params: { kind: "worker" },
    });
    expect((await client.next((frame) => frame.id === 5)).result).toEqual({
      ok: false,
      error: "session kind must be main, subagent, or all",
    });

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "session.most_recent",
      params: { project_dir: projectDirectory },
    });
    expect((await client.next((frame) => frame.id === 6)).result).toMatchObject({
      ok: true,
      session: { id: "aaaabbbb0001", kind: "main" },
    });

    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "initialize",
      params: {
        project_dir: projectDirectory,
        resume_session_id: "ccccdddd0001",
      },
    });
    expect((await client.next((frame) => frame.id === 7)).result?.session).toMatchObject({
      id: "ccccdddd0001",
      kind: "subagent",
      parent_session_id: "aaaabbbb0001",
      root_session_id: "aaaabbbb0001",
      session_kind: "subagent",
      subagent_id: "subagent_child_one",
      title: "Child history",
    });
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 8,
      method: "initialize",
      params: {
        project_dir: otherProjectDirectory,
        session_key: "project-b-connection",
      },
    });
    await client.next((frame) => frame.id === 8);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 9,
      method: "session.open",
      params: { session_key: "ccccdddd0001" },
    });
    expect((await client.next((frame) => frame.id === 9)).error?.message).toContain(
      "different project",
    );
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("shutdown RPC notifies the process host so its daemon lifetime can finish", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-shutdown-"));
  const socketPath = join(directory, "daemon.sock");
  let shutdowns = 0;
  const server = new DaemonServer({
    socketPath,
    onShutdown: () => {
      shutdowns += 1;
    },
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "shutdown", params: {} });
    expect((await client.next((frame) => frame.id === 1)).result).toEqual({
      ok: true,
    });
    await waitFor(() => shutdowns === 1);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon shutdown cancels active turns before flushing session state", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-stop-order-"));
  const socketPath = join(directory, "daemon.sock");
  let childHostShutdowns = 0;
  const runtime = new StopOrderRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, "sessions"),
    shutdown: () => {
      childHostShutdowns += 1;
    },
  });
  const server = new DaemonServer({ socketPath, runtime });
  await server.start();
  try {
    await server.stop();
    expect(runtime.shutdownOperations).toEqual(["cancel", "flush", "shutdown"]);
    expect(childHostShutdowns).toBe(1);
  } finally {
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon stop releases its runtime even when the transport was never started", async () => {
  let shutdowns = 0;
  const runtime = new InMemoryDaemonRuntime(undefined, {
    shutdown: () => {
      shutdowns += 1;
    },
  });
  const server = new DaemonServer({
    runtime,
    socketPath: join(tmpdir(), `xerxes-never-started-${crypto.randomUUID()}.sock`),
  });

  await server.stop();
  await server.stop();

  expect(shutdowns).toBe(1);
});

test("daemon derives slash discovery from implemented canonical commands and rejects unsupported definitions", async () => {
  const directory = await mkdtemp(
    join(tmpdir(), "xerxes-bun-command-registry-"),
  );
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "commands" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "commands.catalog",
      params: {},
    });
    const catalog = await client.next((frame) => frame.id === 2);
    expect(catalog.result).toMatchObject({
      ok: true,
      canon: {
        "/?": "/help",
        "/compress": "/compact",
        "/h": "/help",
      },
    });
    expect(catalog.result?.pairs).toEqual(
      expect.arrayContaining([
        ["/help", "Show help"],
        ["/compact", "Compress the conversation"],
        ["/cron", "Manage scheduled tasks"],
        ["/history", "Show or search conversation history"],
        ["/snapshot", "Take a filesystem snapshot"],
      ]),
    );
    expect(catalog.result?.pairs).toContainEqual([
      "/retry",
      "Re-run the last turn",
    ]);
    expect(catalog.result?.categories).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "session",
          pairs: expect.arrayContaining([
            ["/compact", "Compress the conversation"],
          ]),
        }),
      ]),
    );

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "complete",
      params: { text: "/?" },
    });
    expect(
      (await client.next((frame) => frame.id === 3)).result?.completions,
    ).toEqual([{ value: "/help", label: "help", meta: "Show help" }]);

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "complete",
      params: { text: "/not-a-command" },
    });
    expect(
      (await client.next((frame) => frame.id === 4)).result?.completions,
    ).toEqual([]);

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "slash",
      params: { command: "/not-a-command" },
    });
    expect((await client.next((frame) => frame.id === 5)).result).toEqual({
      ok: false,
      error: "Unknown slash command: /not-a-command",
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      severity: "warning",
      body: "Unknown command: /not-a-command (type /help).",
    });

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "complete",
      params: { text: "/his" },
    });
    expect(
      (await client.next((frame) => frame.id === 6)).result?.completions,
    ).toEqual([
      {
        value: "/history",
        label: "history",
        meta: "Show or search conversation history",
      },
    ]);

    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "slash",
      params: { command: "/?" },
    });
    expect((await client.next((frame) => frame.id === 7)).result).toEqual({
      ok: true,
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: expect.stringContaining("Available Bun daemon commands:"),
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon history reports active session counters over the socket", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-history-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(new UsageRunner(), {
      currentProjectDirectory: directory,
      model: "usage-model",
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "history" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "history", text: "track this turn" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
    });
    await client.next(eventFrame("turn_begin"));
    await client.next(eventFrame("status_update"));
    await client.next(eventFrame("text_part"));
    await client.next(eventFrame("turn_end"));

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: "/history" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
      history: {
        message_count: 2,
        turn_count: 1,
        input_tokens: 17,
        output_tokens: 9,
      },
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      severity: "info",
      body: "Messages: 2\nTurns: 1\nInput tokens: 17\nOutput tokens: 9",
    });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "session.status",
      params: { session_key: "history" },
    });
    const status = (await client.next((frame) => frame.id === 4)).result?.session as
      | Record<string, unknown>
      | undefined;
    expect(status).toMatchObject({
      calls: 1,
      context_limit: 128_000,
      input_tokens: 17,
      max_context: 128_000,
      output_tokens: 9,
      total_tokens: 26,
      usage_complete: true,
    });
    expect(Number(status?.context_tokens)).toBeGreaterThan(0);

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "session.usage",
      params: { session_key: "history" },
    });
    const usage = (await client.next((frame) => frame.id === 5)).result;
    expect(usage).toMatchObject({
      calls: 1,
      context_max: 128_000,
      input: 17,
      model: "usage-model",
      output: 9,
      total: 26,
      usage_complete: true,
    });
    expect(Number(usage?.context_used)).toBe(Number(status?.context_tokens));
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon usage marks imported counters unknown instead of fabricating cumulative API calls", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-imported-usage-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, "sessions"),
  });
  const imported = await runtime.openSession("legacy-slot");
  imported.turnCount = 2;
  imported.totalApiCalls = 1;
  imported.totalInputTokens = 30;
  imported.totalOutputTokens = 7;
  delete imported.apiCallsComplete;
  delete imported.usageComplete;
  const server = new DaemonServer({ socketPath, runtime });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "session.status",
      params: { session_key: "legacy-slot" },
    });
    const status = (await client.next((frame) => frame.id === 1)).result?.session as
      | Record<string, unknown>
      | undefined;
    expect(status).toMatchObject({
      calls_complete: false,
      input_tokens: 30,
      observed_calls: 1,
      output_tokens: 7,
      usage_complete: false,
    });
    expect(status?.calls).toBeUndefined();

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "session.usage",
      params: { session_key: "legacy-slot" },
    });
    const usage = (await client.next((frame) => frame.id === 2)).result;
    expect(usage).toMatchObject({
      calls_complete: false,
      input: 30,
      observed_calls: 1,
      output: 7,
      total: 37,
      usage_complete: false,
    });
    expect(usage?.calls).toBeUndefined();
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon lists and controls persistent cron jobs through slash commands", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-cron-command-"));
  const socketPath = join(directory, "daemon.sock");
  const store = new JobStore(join(directory, "cron", "jobs.json"));
  const server = new DaemonServer({
    socketPath,
    cronStoreFactory: () => store,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "slash",
      params: { command: "/cron" },
    });
    expect((await client.next((frame) => frame.id === 1)).result).toEqual({
      ok: true,
      jobs: [],
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: "No cron jobs scheduled.",
    });

    store.add(
      new CronJob({
        id: "active-job",
        prompt: "summarize changes",
        schedule: "0 9 * * 1",
        nextRunAt: "2026-07-20T09:00:00Z",
      }),
    );
    store.add(
      new CronJob({
        id: "paused-job",
        prompt: "send a report",
        schedule: "30 17 * * 5",
        paused: true,
      }),
    );

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/cron list" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject(
      {
        ok: true,
        jobs: [
          {
            id: "active-job",
            prompt: "summarize changes",
            schedule: "0 9 * * 1",
            paused: false,
            next_run_at: "2026-07-20T09:00:00Z",
          },
          {
            id: "paused-job",
            prompt: "send a report",
            schedule: "30 17 * * 5",
            paused: true,
          },
        ],
      },
    );
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: "Cron jobs (2):\n  `active-job` — `0 9 * * 1` (active)\n  `paused-job` — `30 17 * * 5` (paused)",
    });

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: "/cron add" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: false,
      error:
        "Provide exactly one of `--schedule <five-field-cron>` or `--at <ISO-8601-time>`.",
    });
    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "slash",
      params: {
        command:
          '/cron add --schedule "0 9 * * 1" --prompt "summarize native changes"',
      },
    });
    const added = await client.next((frame) => frame.id === 4);
    expect(added.result).toMatchObject({
      ok: true,
      job: {
        prompt: "summarize native changes",
        schedule: "0 9 * * 1",
        paused: false,
        oneshot: false,
      },
    });
    const job = added.result?.job as Record<string, unknown>;
    const jobId = String(job.id);

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "slash",
      params: { command: `/cron pause ${jobId}` },
    });
    expect((await client.next((frame) => frame.id === 5)).result).toMatchObject(
      {
        ok: true,
        job: { id: jobId, paused: true },
      },
    );

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "slash",
      params: { command: `/cron resume ${jobId}` },
    });
    expect((await client.next((frame) => frame.id === 6)).result).toMatchObject(
      {
        ok: true,
        job: { id: jobId, paused: false },
      },
    );

    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "slash",
      params: { command: `/cron run ${jobId}` },
    });
    expect((await client.next((frame) => frame.id === 7)).result).toMatchObject(
      {
        ok: true,
        job: { id: jobId },
        output: "Bun daemon foundation received: summarize native changes",
      },
    );

    client.send({
      jsonrpc: "2.0",
      id: 8,
      method: "slash",
      params: { command: `/cron remove ${jobId}` },
    });
    expect((await client.next((frame) => frame.id === 8)).result).toEqual({
      ok: true,
      id: jobId,
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon automatically runs due cron jobs, archives output, and delivers through a configured native channel", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-cron-lifecycle-"));
  const socketPath = join(directory, "daemon.sock");
  const store = new JobStore(join(directory, "cron", "jobs.json"));
  const channel = new DaemonRecordingChannel("recording");
  store.add(
    new CronJob({
      id: "due-once",
      prompt: "automatic native report",
      nextRunAt: new Date(Date.now() - 2_000).toISOString(),
      oneshot: true,
      deliver: "recording",
      recipient: "room-42",
    }),
  );
  const server = new DaemonServer({
    socketPath,
    channelManager: new ChannelManager({
      channels: [["recording", channel]],
    }),
    cronArchiveDirectory: join(directory, "cron", "archive"),
    cronPollInterval: 5,
    cronStoreFactory: () => store,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  try {
    await waitFor(() => channel.sent.length === 1);
    expect(channel.sent[0]).toMatchObject({
      channel: "recording",
      direction: "outbound",
      channelUserId: "room-42",
      roomId: "room-42",
      text: "Bun daemon foundation received: automatic native report",
    });
    expect(store.get("due-once")).toBeUndefined();
    const archives = await readdir(
      join(directory, "cron", "archive", "due-once"),
    );
    expect(archives).toHaveLength(1);
    expect(
      await Bun.file(
        join(directory, "cron", "archive", "due-once", archives[0] ?? ""),
      ).text(),
    ).toBe("Bun daemon foundation received: automatic native report");
  } finally {
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon exposes a read-only native update status contract", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-update-status-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "update-status", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "runtime.update_status",
      params: {},
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject(
      {
        ok: true,
        applied: false,
        command: "bun run xerxes update",
        git: { isGit: false },
        summary: "not a git checkout",
        next_steps: [
          "bun run xerxes update --dry-run --spec <package-or-source-spec>",
          "bun run xerxes update --apply --spec <package-or-source-spec>",
        ],
      },
    );
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon exposes real browser management state without fabricating a browser session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-browser-manage-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "browser.manage",
      params: {},
    });
    expect((await client.next((frame) => frame.id === 1)).result).toEqual({
      ok: true,
      status: { connected: false, kind: "none" },
      pages: [],
    });
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/browser" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
      status: { connected: false, kind: "none" },
      pages: [],
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: expect.stringContaining("Native browser: not connected"),
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon supplies real direct session controls used by the native TUI", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-direct-session-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "direct-session-model",
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "direct-session", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { text: "persist this native session" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    await client.next(eventFrame("text_part"));
    await client.next(eventFrame("turn_end"));

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.title",
      params: { title: "native title" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
      title: "native title",
    });

    client.send({ jsonrpc: "2.0", id: 4, method: "session.save", params: {} });
    expect((await client.next((frame) => frame.id === 4)).result).toMatchObject(
      {
        ok: true,
        session: { title: "native title" },
      },
    );

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "session.most_recent",
      params: {},
    });
    expect((await client.next((frame) => frame.id === 5)).result).toMatchObject(
      {
        ok: true,
        session: { title: "native title" },
      },
    );

    client.send({ jsonrpc: "2.0", id: 6, method: "session.undo", params: {} });
    expect((await client.next((frame) => frame.id === 6)).result).toMatchObject(
      {
        ok: true,
        dropped: 2,
      },
    );

    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "session.compress",
      params: {},
    });
    expect((await client.next((frame) => frame.id === 7)).result).toMatchObject(
      {
        ok: true,
        compacted: false,
      },
    );

    client.send({
      jsonrpc: "2.0",
      id: 8,
      method: "session.delete",
      params: {},
    });
    expect((await client.next((frame) => frame.id === 8)).result).toEqual({
      ok: true,
      deleted: true,
      session_id: expect.any(String),
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon saves named sessions and routes the advertised btw alias", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-save-command-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "save-command-model",
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "save-command", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { text: "capture this work" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    await client.next(eventFrame("text_part"));
    await client.next(eventFrame("turn_end"));

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: "/save release-notes" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toMatchObject(
      {
        ok: true,
        title: "release-notes",
        session: { title: "release-notes" },
      },
    );
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: expect.stringContaining("as `release-notes`"),
    });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "session.list",
      params: {},
    });
    expect((await client.next((frame) => frame.id === 4)).result).toMatchObject(
      {
        ok: true,
        sessions: [expect.objectContaining({ title: "release-notes" })],
      },
    );

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "slash",
      params: { command: "/btw keep the title" },
    });
    expect((await client.next((frame) => frame.id === 5)).result).toEqual({
      ok: true,
    });
    expect(
      (await client.next(eventFrame("steer_input"))).params?.payload,
    ).toEqual({ content: "keep the title" });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({ category: "slash", body: "Steer accepted." });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon compact uses the active provider to summarize instead of the naive dev summarizer", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-compact-llm-"));
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
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "gpt-4",
      sessionDirectory: join(directory, "sessions"),
    }),
    profileStore,
  });
  const nativeFetch = globalThis.fetch;
  const requests: unknown[] = [];
  const modelFetch: FetchImplementation = async (input, init) => {
    const url = typeof input === "string" ? input : input.toString();
    if (!url.includes("/chat/completions")) {
      return new Response(
        JSON.stringify({ error: "unexpected endpoint" }),
        { status: 404 },
      );
    }
    const body = typeof init?.body === "string" ? JSON.parse(init.body) : undefined;
    requests.push(body);
    return new Response(
      JSON.stringify({
        choices: [{ message: { content: "durable compact summary" } }],
      }),
    );
  };
  globalThis.fetch = modelFetch as typeof globalThis.fetch;
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "compact-llm", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    for (let i = 0; i < 4; i += 1) {
      client.send({
        jsonrpc: "2.0",
        id: 2 + i,
        method: "turn.submit",
        params: { text: `message ${i + 1}` },
      });
      await client.next((frame) => frame.id === 2 + i);
      await client.next(eventFrame("turn_begin"));
      await client.next(eventFrame("text_part"));
      await client.next(eventFrame("turn_end"));
    }

    client.send({
      jsonrpc: "2.0",
      id: 10,
      method: "session.compress",
      params: {},
    });
    const result = (await client.next((frame) => frame.id === 10)).result;
    expect(result).toMatchObject({ ok: true, compacted: true });
    expect(requests.length).toBeGreaterThan(0);
    const prompt = String(
      (requests[0] as { messages?: Array<{ content?: unknown }> })?.messages?.[0]
        ?.content ?? "",
    );
    expect(prompt).toContain("CONTEXT TO SUMMARIZE");
    expect(prompt).toContain("message 1");
  } finally {
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon refuses to compact while a turn is running", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-compact-running-"));
  const socketPath = join(directory, "daemon.sock");
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "gate-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ runtime, socketPath });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "compact-running", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { text: "hold the turn open" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    await waitFor(() => runner.runs === 1);

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.compress",
      params: {},
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: false,
      error: "turn is running",
    });
    // The in-flight transcript must be untouched.
    expect(
      runtime.sessionStatus("compact-running")?.metadata.last_compaction,
    ).toBeUndefined();

    client.send({ jsonrpc: "2.0", id: 4, method: "turn.cancel", params: {} });
    await client.next((frame) => frame.id === 4);
    await waitFor(
      () => runtime.sessionStatus("compact-running")?.activeTurnId === "",
    );
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon snapshots, lists, and rolls back the active session workspace", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-snapshots-"));
  const workspace = join(directory, "workspace");
  const socketPath = join(directory, "daemon.sock");
  const sourcePath = join(workspace, "state.txt");
  await mkdir(workspace);
  await writeFile(sourcePath, "first", "utf8");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: workspace,
      sessionDirectory: join(directory, "sessions"),
    }),
    snapshotManagerFactory: (workspaceDirectory) =>
      new SnapshotManager(workspaceDirectory, {
        shadowRoot: join(directory, "shadow"),
      }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "slash",
      params: { command: "/snapshot" },
    });
    expect((await client.next((frame) => frame.id === 1)).result).toEqual({
      ok: false,
      error: "no active session",
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      severity: "warning",
      body: "No active session yet.",
    });

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "initialize",
      params: { session_key: "snapshots", project_dir: workspace },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: "/snapshots" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
      snapshots: [],
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: "No snapshots yet. Take one with `/snapshot [label]`.",
    });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "slash",
      params: { command: "/snapshot first" },
    });
    const first = await client.next((frame) => frame.id === 4);
    const firstId = String(
      (first.result?.snapshot as Record<string, unknown>).id,
    );
    expect(first.result).toMatchObject({
      ok: true,
      snapshot: {
        id: expect.any(String),
        label: "first",
        workspace_dir: expect.stringMatching(/\/workspace$/),
      },
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: `Snapshot \`${firstId}\` saved.`,
    });

    await writeFile(sourcePath, "second", "utf8");
    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "slash",
      params: { command: "/snapshot" },
    });
    expect((await client.next((frame) => frame.id === 5)).result).toMatchObject(
      {
        ok: true,
        snapshot: { label: "manual" },
      },
    );
    await client.next(eventFrame("notification"));

    await writeFile(sourcePath, "third", "utf8");
    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "slash",
      params: { command: "/snapshots" },
    });
    expect((await client.next((frame) => frame.id === 6)).result).toMatchObject(
      {
        ok: true,
        snapshots: [{ id: firstId, label: "first" }, { label: "manual" }],
      },
    );
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: expect.stringContaining(
        `Snapshots (2):\n  \`${firstId}\` — \`first\``,
      ),
    });

    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "slash",
      params: { command: `/rollback ${firstId}` },
    });
    expect((await client.next((frame) => frame.id === 7)).result).toMatchObject(
      { ok: true, snapshot: { id: firstId, label: "first" } },
    );
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: `Rolled back to snapshot \`${firstId}\`.`,
    });
    expect(await Bun.file(sourcePath).text()).toBe("first");

    client.send({
      jsonrpc: "2.0",
      id: 8,
      method: "slash",
      params: { command: "/rollback missing" },
    });
    expect((await client.next((frame) => frame.id === 8)).result).toEqual({
      ok: false,
      error: "snapshot not found: missing",
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      severity: "error",
      body: "Rollback failed: `snapshot not found: missing`",
    });

    client.send({
      jsonrpc: "2.0",
      id: 9,
      method: "slash",
      params: { command: "/rollback" },
    });
    expect((await client.next((frame) => frame.id === 9)).result).toEqual({
      ok: false,
      error: "snapshot reference is required",
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      severity: "warning",
      body: "Usage: `/rollback <snapshot-id>` — list with `/snapshots`.",
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon resumes only initialize resume IDs and lists saved sessions separately from live sessions", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-resume-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "gpt-4o",
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
      params: { session_key: "tui:first", project_dir: directory },
    });
    const created = await client.next((frame) => frame.id === 1);
    const firstSession = created.result?.session as Record<string, unknown>;
    const firstSessionId = String(firstSession.id);
    expect(firstSession).toMatchObject({
      key: "tui:first",
      messages: 0,
      model: "gpt-4o",
    });
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "tui:first", text: "saved question" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
    });
    await client.next(eventFrame("turn_begin"));
    await client.next(eventFrame("text_part"));
    await client.next(eventFrame("turn_end"));

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.active_list",
      params: {},
    });
    const active = await client.next((frame) => frame.id === 3);
    expect(active.result).toMatchObject({
      ok: true,
      sessions: [{ id: firstSessionId, key: "tui:first", messages: 2 }],
    });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "session.list",
      params: { limit: 200 },
    });
    const saved = await client.next((frame) => frame.id === 4);
    expect(saved.result).toMatchObject({
      ok: true,
      sessions: [
        {
          session_id: firstSessionId,
          key: "tui:first",
          messages: 2,
          title: "saved question",
        },
      ],
    });

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "initialize",
      params: { session_key: "tui:first", project_dir: directory },
    });
    const fresh = await client.next((frame) => frame.id === 5);
    expect(fresh.result?.session).toMatchObject({
      key: "tui:first",
      messages: 0,
    });
    expect((fresh.result?.session as Record<string, unknown>).id).not.toBe(
      firstSessionId,
    );
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "initialize",
      params: {
        resume_session_id: firstSessionId,
        session_key: "ignored-slot",
        project_dir: directory,
      },
    });
    const resumed = await client.next((frame) => frame.id === 6);
    expect(resumed.result).toMatchObject({
      ok: true,
      daemon_protocol: 35,
      daemon_build_id: expect.any(String),
      session: { id: firstSessionId, key: firstSessionId, messages: 2 },
    });
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    const replayedUser = await client.next(
      (frame) =>
        frame.method === "event" &&
        frame.params?.type === "notification" &&
        frame.params.payload?.type === "replay_user",
    );
    expect(replayedUser.params?.payload?.body).toBe("✨ saved question");
    await client.next(
      (frame) =>
        frame.method === "event" &&
        frame.params?.type === "notification" &&
        frame.params.payload?.type === "resumed",
    );

    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "runtime.status",
      params: {},
    });
    expect((await client.next((frame) => frame.id === 7)).result).toMatchObject(
      {
        ok: true,
        runtime_ready: true,
        daemon_protocol: 35,
        daemon_build_id: expect.any(String),
        channels: [],
        channels_available: false,
        channels_configured: false,
        model: "gpt-4o",
      },
    );

    client.send({
      jsonrpc: "2.0",
      id: 8,
      method: "session.status",
      params: { session_key: "missing" },
    });
    expect((await client.next((frame) => frame.id === 8)).result).toEqual({
      ok: false,
      session: null,
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon implements native completion, slash, steering, mode, and provider controls", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-controls-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ socketPath, runtime, profileStore });
  await writeFile(join(directory, "alpha.txt"), "alpha", "utf8");
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  const nativeFetch = globalThis.fetch;
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "controls", project_dir: directory },
    });
    expect((await client.next((frame) => frame.id === 1)).result).toMatchObject(
      { ok: true, session: { key: "controls" } },
    );
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "complete",
      params: { text: "/mo" },
    });
    expect(
      (await client.next((frame) => frame.id === 2)).result?.completions,
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ value: "/model", meta: expect.any(String) }),
      ]),
    );

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "complete",
      params: { text: "./al" },
    });
    expect(
      (await client.next((frame) => frame.id === 3)).result?.completions,
    ).toEqual([{ value: "./alpha.txt", label: "alpha.txt", meta: "file" }]);

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "steer",
      params: { session_key: "controls", content: "keep it concise" },
    });
    expect((await client.next((frame) => frame.id === 4)).result).toEqual({
      ok: true,
    });
    expect(
      (await client.next(eventFrame("steer_input"))).params?.payload,
    ).toEqual({ content: "keep it concise" });
    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "session.status",
      params: { session_key: "controls" },
    });
    expect(
      (await client.next((frame) => frame.id === 5)).result?.session,
    ).toMatchObject({ messages: 1 });

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "set_plan_mode",
      params: { enabled: true },
    });
    expect((await client.next((frame) => frame.id === 6)).result).toMatchObject(
      { ok: true, mode: "plan", plan_mode: true },
    );
    expect(
      (await client.next(eventFrame("status_update"))).params?.payload,
    ).toMatchObject({ mode: "plan", plan_mode: true });

    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "slash",
      params: { command: "/title Bun control plane" },
    });
    expect((await client.next((frame) => frame.id === 7)).result).toEqual({
      ok: true,
      title: "Bun control plane",
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: expect.stringContaining("Bun control plane"),
    });

    client.send({
      jsonrpc: "2.0",
      id: 8,
      method: "provider_save",
      params: {
        name: "native",
        base_url: "https://provider.example/v1",
        api_key: "do-not-echo",
        model: "native-model",
        provider: "openai",
      },
    });
    const saved = await client.next((frame) => frame.id === 8);
    expect(saved.result).toMatchObject({
      ok: true,
      profile: { name: "native", model: "native-model", active: true },
    });
    expect(JSON.stringify(saved.result)).not.toContain("do-not-echo");
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 80,
      method: "session.status",
      params: { session_key: "controls" },
    });
    const matchedRuntimeStatus = (
      await client.next((frame) => frame.id === 80)
    ).result?.session;
    expect(matchedRuntimeStatus).toMatchObject({
      model: "native-model",
      profile_name: "native",
    });
    expect(matchedRuntimeStatus).not.toHaveProperty("api_key");
    expect(matchedRuntimeStatus).not.toHaveProperty("base_url");

    runtime.reload({
      base_url: "https://runtime-override.example/v1",
      provider: "anthropic",
    });
    client.send({
      jsonrpc: "2.0",
      id: 81,
      method: "session.status",
      params: { session_key: "controls" },
    });
    expect(
      (await client.next((frame) => frame.id === 81)).result?.session,
    ).toMatchObject({ profile_name: null });
    runtime.reload({
      base_url: "https://provider.example/v1",
      provider: "openai",
    });

    client.send({ jsonrpc: "2.0", id: 9, method: "provider_list", params: {} });
    expect(
      (await client.next((frame) => frame.id === 9)).result?.profiles,
    ).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "native", active: true }),
      ]),
    );

    profileStore.save({
      apiKey: "inactive-secret",
      baseUrl: "https://inactive.example/v1",
      model: "inactive-saved-model",
      name: "inactive",
      provider: "openai",
      setActive: false,
    });
    profileStore.save({
      apiKey: "fallback-secret",
      baseUrl: "https://failure.example/v1",
      model: "fallback-saved-model",
      name: "fallback",
      provider: "openai",
      setActive: false,
    });
    const modelRequests: Array<{ authorization: string | null; url: string }> = [];
    const mockFetch: FetchImplementation = async (input, init) => {
      const request = {
        authorization: new Headers(init?.headers).get("authorization"),
        url: String(input),
      };
      modelRequests.push(request);
      if (request.url.includes("failure.example")) {
        throw new Error(`upstream echoed ${request.authorization}`);
      }
      return new Response(
        JSON.stringify({
          data: [
            {
              id: String(input).includes("inactive.example")
                ? "inactive-remote-model"
                : "remote-model",
            },
          ],
        }),
        { status: 200 },
      );
    };
    globalThis.fetch = mockFetch as typeof globalThis.fetch;
    client.send({
      jsonrpc: "2.0",
      id: 10,
      method: "fetch_models",
      params: { base_url: "https://provider.example/v1", provider: "openai" },
    });
    expect((await client.next((frame) => frame.id === 10)).result).toEqual({
      ok: false,
      error:
        "model discovery only accepts a stored profile name; save the provider profile first",
      models: [],
    });
    expect(modelRequests).toEqual([]);

    client.send({
      jsonrpc: "2.0",
      id: 11,
      method: "fetch_models",
      params: { profile_name: "inactive" },
    });
    expect((await client.next((frame) => frame.id === 11)).result).toEqual({
      ok: true,
      models: ["inactive-remote-model"],
      profile: "inactive",
      source: "remote",
    });
    expect(profileStore.active()?.name).toBe("native");
    expect(modelRequests.at(-1)).toEqual({
      authorization: "Bearer inactive-secret",
      url: "https://inactive.example/v1/models",
    });

    client.send({
      jsonrpc: "2.0",
      id: 12,
      method: "fetch_models",
      params: { profile_name: "fallback" },
    });
    const fallback = (await client.next((frame) => frame.id === 12)).result;
    expect(fallback).toMatchObject({
      ok: true,
      models: ["fallback-saved-model"],
      profile: "fallback",
      source: "profile",
      warning: expect.stringContaining("[redacted]"),
    });
    expect(JSON.stringify(fallback)).not.toContain("fallback-secret");

    const requestCount = modelRequests.length;
    client.send({
      jsonrpc: "2.0",
      id: 13,
      method: "fetch_models",
      params: { profile_name: "missing" },
    });
    expect((await client.next((frame) => frame.id === 13)).result).toEqual({
      ok: false,
      error: "No provider profile named missing",
      models: [],
    });
    expect(modelRequests).toHaveLength(requestCount);

    client.send({
      jsonrpc: "2.0",
      id: 14,
      method: "fetch_models",
      params: {
        profile: "native",
        base_url: "https://attacker.example/v1",
      },
    });
    expect((await client.next((frame) => frame.id === 14)).result).toEqual({
      ok: false,
      error:
        "model discovery only accepts a stored profile name; save the provider profile first",
      models: [],
    });
    expect(modelRequests).toHaveLength(requestCount);

    client.send({
      jsonrpc: "2.0",
      id: 15,
      method: "fetch_models",
      params: { base_url: "http://127.0.0.1:11434/v1" },
    });
    expect((await client.next((frame) => frame.id === 15)).result).toMatchObject({
      ok: false,
      error:
        "model discovery only accepts a stored profile name; save the provider profile first",
      models: [],
    });
    expect(modelRequests).toHaveLength(requestCount);
    globalThis.fetch = nativeFetch;

    client.send({
      jsonrpc: "2.0",
      id: 16,
      method: "provider_delete",
      params: { name: "native" },
    });
    expect((await client.next((frame) => frame.id === 16)).result).toEqual({
      ok: true,
    });
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
  } finally {
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon yolo toggles the live permission mode in both directions", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-yolo-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "yolo", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    expect((await client.next(eventFrame("init_done"))).params?.payload).toMatchObject({
      permission_mode: "accept-all",
    });
    expect((await client.next(eventFrame("status_update"))).params?.payload).toMatchObject({
      permission_mode: "accept-all",
    });

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/yolo" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
      permission_mode: "auto",
    });
    expect((await client.next(eventFrame("status_update"))).params?.payload).toMatchObject({
      permission_mode: "auto",
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload?.body,
    ).toBe("YOLO mode OFF.");

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "runtime.status",
      params: {},
    });
    expect(
      (await client.next((frame) => frame.id === 3)).result?.permission_mode,
    ).toBe("auto");

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "slash",
      params: { command: "/yolo" },
    });
    expect((await client.next((frame) => frame.id === 4)).result).toEqual({
      ok: true,
      permission_mode: "accept-all",
    });
    expect((await client.next(eventFrame("status_update"))).params?.payload).toMatchObject({
      permission_mode: "accept-all",
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload?.body,
    ).toBe("YOLO mode ON.");

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "runtime.status",
      params: {},
    });
    expect(
      (await client.next((frame) => frame.id === 5)).result?.permission_mode,
    ).toBe("accept-all");
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon slash config, sampling, agents, and platforms use native backing state", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-slash-parity-"));
  const socketPath = join(directory, "daemon.sock");
  const rebuiltSettings: Array<Readonly<Record<string, unknown>>> = [];
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    runtimeSettings: {
      api_key: "must-not-leak",
      base_url: "https://native.example/v1",
      max_tokens: 512,
      model: "native-model",
      provider: "openai",
      reasoning_effort: "high",
      temperature: 0.2,
      thinking: true,
      thinking_budget: 1_024,
      top_p: 0.8,
    },
    sessionDirectory: join(directory, "sessions"),
    turnRunnerFactory: (settings) => {
      rebuiltSettings.push({ ...settings });
      return undefined;
    },
  });
  const channelManager = new ChannelManager({
    channels: [["telegram", new DaemonRecordingChannel("telegram")]],
    onInbound: async () => {},
  });
  const server = new DaemonServer({
    socketPath,
    channelManager,
    profileStore: new ProfileStore(join(directory, "profiles.json")),
    runtime,
    agentDefinitionLoader: () => [
      {
        allowedTools: ["ReadFile"],
        description: "Reviews native changes.",
        excludeTools: [],
        isolation: "",
        maxDepth: 3,
        model: "native-model",
        name: "reviewer",
        source: "project",
        systemPrompt: "This private prompt must not be returned by /agents.",
        tools: ["ReadFile"],
      },
    ],
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "slash-parity", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "commands.catalog",
      params: {},
    });
    const catalog = await client.next((frame) => frame.id === 2);
    expect(catalog.result?.pairs).toEqual(
      expect.arrayContaining([
        ["/agents", "List native agent definitions"],
        ["/config", "Show effective native runtime configuration"],
        ["/platforms", "List configured messaging platforms"],
        ["/sampling", "Show or set next-turn native sampling options"],
      ]),
    );
    expect(catalog.result?.pairs).toContainEqual([
      "/reasoning",
      "Show/set thinking effort (off|low|medium|high)",
    ]);

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: "/config" },
    });
    const config = await client.next((frame) => frame.id === 3);
    expect(config.result).toMatchObject({
      ok: true,
      config: {
        base_url: "https://native.example/v1",
        max_tokens: 512,
        model: "native-model",
        permission_mode: "accept-all",
        provider: "openai",
        temperature: 0.2,
        top_p: 0.8,
      },
    });
    expect(JSON.stringify(config.result)).not.toContain("must-not-leak");
    expect(config.result?.config).not.toHaveProperty("reasoning_effort");
    expect(config.result?.config).not.toHaveProperty("thinking");
    expect(config.result?.config).not.toHaveProperty("thinking_budget");
    expect(
      (await client.next(eventFrame("notification"))).params?.payload?.body,
    ).toContain("Effective native runtime config");

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "slash",
      params: { command: "/sampling temperature 0.35" },
    });
    expect((await client.next((frame) => frame.id === 4)).result).toMatchObject(
      {
        ok: true,
        sampling: { temperature: 0.35, top_p: 0.8, max_tokens: 512 },
      },
    );
    expect(rebuiltSettings.at(-1)).toMatchObject({ temperature: 0.35 });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload?.body,
    ).toContain("temperature");

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "slash",
      params: { command: "/sampling top_k 10" },
    });
    expect((await client.next((frame) => frame.id === 5)).result).toMatchObject(
      {
        ok: true,
        sampling: { top_k: 10 },
      },
    );
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      severity: "info",
      body: expect.stringContaining("top_k"),
    });

    client.send({
      jsonrpc: "2.0",
      id: 51,
      method: "slash",
      params: { command: "/sampling reset" },
    });
    expect((await client.next((frame) => frame.id === 51)).result).toMatchObject({
      ok: true,
      sampling: { temperature: 0.6, top_k: 64 },
    });
    expect(rebuiltSettings.at(-1)).toMatchObject({
      temperature: 0.6,
      top_k: 64,
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload?.body,
    ).toContain("temperature 0.6, top_k 64");

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "slash",
      params: { command: "/agents" },
    });
    const agents = await client.next((frame) => frame.id === 6);
    expect(agents.result).toMatchObject({
      ok: true,
      agents: [
        {
          name: "reviewer",
          source: "project",
          tools: ["ReadFile"],
        },
      ],
    });
    expect(JSON.stringify(agents.result)).not.toContain("private prompt");
    await client.next(eventFrame("notification"));

    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "slash",
      params: { command: "/platforms" },
    });
    expect((await client.next((frame) => frame.id === 7)).result).toMatchObject(
      {
        ok: true,
        platforms: [{ name: "telegram", enabled: false }],
        channels_available: true,
        channels_configured: true,
      },
    );
    await client.next(eventFrame("notification"));

    client.send({
      jsonrpc: "2.0",
      id: 8,
      method: "channel.enable",
      params: { name: "telegram" },
    });
    await client.next((frame) => frame.id === 8);
    await client.next(eventFrame("channel_status"));
    client.send({
      jsonrpc: "2.0",
      id: 9,
      method: "slash",
      params: { command: "/platforms" },
    });
    expect((await client.next((frame) => frame.id === 9)).result).toMatchObject(
      {
        ok: true,
        platforms: [{ name: "telegram", enabled: true }],
      },
    );
    await client.next(eventFrame("notification"));

    client.send({
      jsonrpc: "2.0",
      id: 10,
      method: "slash",
      params: { command: "/reasoning high" },
    });
    expect((await client.next((frame) => frame.id === 10)).result).toEqual({
      ok: true,
      reasoning_effort: "high",
    });
    await client.next(eventFrame("status_update"));
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({ severity: "info", body: "Thinking: `high`." });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon routes approval and question replies through the active connection", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-replies-"));
  const socketPath = join(directory, "daemon.sock");
  const interactions = new DaemonInteractionBoard();
  const runner = new ReplyRunner(interactions);
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(runner, {
      currentProjectDirectory: directory,
      interactions,
      model: "reply-model",
      sessionDirectory: join(directory, "sessions"),
    }),
    interactions,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "replies" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "replies", text: "run control flow" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
    });
    await client.next(eventFrame("turn_begin"));
    const approval = await client.next(eventFrame("approval_request"));
    expect(approval.params?.payload).toMatchObject({
      id: "approval-1",
      request_id: "approval-1",
    });

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "permission_response",
      params: { request_id: "approval-1", response: "approve" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
    });
    expect(
      (await client.next(eventFrame("approval_response"))).params?.payload,
    ).toEqual({ request_id: "approval-1", response: "approve" });
    expect(
      (await client.next(eventFrame("text_part"))).params?.payload,
    ).toEqual({ text: "approval:approve" });

    const question = await client.next(eventFrame("question_request"));
    const requestId = String(question.params?.payload?.id);
    expect(question.params?.payload).toMatchObject({
      questions: [
        { id: "answer", question: "Continue?", allow_free_form: false },
      ],
    });
    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "question_response",
      params: { request_id: requestId, answers: { answer: "yes" } },
    });
    expect((await client.next((frame) => frame.id === 4)).result).toEqual({
      ok: true,
    });
    expect(
      (await client.next(eventFrame("question_response"))).params?.payload,
    ).toEqual({ id: requestId, answers: { answer: "yes" } });
    expect(
      (await client.next(eventFrame("text_part"))).params?.payload,
    ).toEqual({ text: "answer:yes" });
    await client.next(eventFrame("turn_end"));
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("disconnecting an interaction owner cancels approval and question waits", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-disconnect-"));
  const socketPath = join(directory, "daemon.sock");
  const interactions = new DaemonInteractionBoard();
  const runtime = new InMemoryDaemonRuntime(new ReplyRunner(interactions), {
    currentProjectDirectory: directory,
    interactions,
    model: "reply-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ socketPath, runtime, interactions });
  await server.start();
  try {
    const approvalClient = await SocketTestClient.connect(socketPath);
    approvalClient.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "disconnect-approval" },
    });
    await approvalClient.next((frame) => frame.id === 1);
    await approvalClient.next(eventFrame("init_done"));
    await approvalClient.next(eventFrame("status_update"));
    approvalClient.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "disconnect-approval", text: "wait for approval" },
    });
    await approvalClient.next((frame) => frame.id === 2);
    await approvalClient.next(eventFrame("turn_begin"));
    await approvalClient.next(eventFrame("approval_request"));
    expect(interactions.pendingPermissionIds()).toEqual(["approval-1"]);
    approvalClient.close();

    await waitFor(
      () =>
        interactions.pendingPermissionIds().length === 0 &&
        runtime.sessionStatus("disconnect-approval")?.activeTurnId === "",
    );
    expect(runtime.sessionStatus("disconnect-approval")?.cancelRequested).toBeTrue();

    const questionClient = await SocketTestClient.connect(socketPath);
    questionClient.send({
      jsonrpc: "2.0",
      id: 3,
      method: "initialize",
      params: { session_key: "disconnect-question" },
    });
    await questionClient.next((frame) => frame.id === 3);
    await questionClient.next(eventFrame("init_done"));
    await questionClient.next(eventFrame("status_update"));
    questionClient.send({
      jsonrpc: "2.0",
      id: 4,
      method: "turn.submit",
      params: { session_key: "disconnect-question", text: "wait for question" },
    });
    await questionClient.next((frame) => frame.id === 4);
    await questionClient.next(eventFrame("turn_begin"));
    await questionClient.next(eventFrame("approval_request"));
    questionClient.send({
      jsonrpc: "2.0",
      id: 5,
      method: "permission_response",
      params: { request_id: "approval-1", response: "approve" },
    });
    await questionClient.next((frame) => frame.id === 5);
    await questionClient.next(eventFrame("approval_response"));
    await questionClient.next(eventFrame("text_part"));
    await questionClient.next(eventFrame("question_request"));
    expect(interactions.pendingQuestionIds()).toHaveLength(1);
    questionClient.close();

    await waitFor(
      () =>
        interactions.pendingQuestionIds().length === 0 &&
        runtime.sessionStatus("disconnect-question")?.activeTurnId === "",
    );
    expect(runtime.sessionStatus("disconnect-question")?.cancelRequested).toBeTrue();
  } finally {
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon applies queued steering at a native runner boundary", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-steer-"));
  const socketPath = join(directory, "daemon.sock");
  const runner = new SteerRunner();
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(runner, {
      currentProjectDirectory: directory,
      model: "steer-model",
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "steer" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "steer", text: "start" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    expect(
      (await client.next(eventFrame("text_part"))).params?.payload,
    ).toEqual({ text: "waiting for steer" });
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "steer",
      params: { session_key: "steer", content: "focus tests" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
    });
    expect(
      (await client.next(eventFrame("steer_input"))).params?.payload,
    ).toEqual({ content: "focus tests" });
    runner.release();
    expect(
      (await client.next(eventFrame("text_part"))).params?.payload,
    ).toEqual({ text: "steer:focus tests" });
    await client.next(eventFrame("turn_end"));
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon exposes only host-configured channel lifecycle controls", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-channels-"));
  const socketPath = join(directory, "daemon.sock");
  const channel = new DaemonRecordingChannel("telegram");
  const channelManager = new ChannelManager({
    channels: [["telegram", channel]],
    onInbound: async () => {},
  });
  const server = new DaemonServer({
    socketPath,
    channelManager,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "runtime.status",
      params: {},
    });
    expect((await client.next((frame) => frame.id === 1)).result).toMatchObject(
      {
        ok: true,
        channels_available: true,
        channels_configured: true,
        channels: [
          { name: "telegram", adapter_name: "telegram", enabled: false },
        ],
      },
    );

    client.send({ jsonrpc: "2.0", id: 2, method: "channel.list", params: {} });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
      channels_available: true,
      channels_configured: true,
      channels: [
        { name: "telegram", adapter_name: "telegram", enabled: false },
      ],
    });

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "channel.enable",
      params: { name: "telegram" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toMatchObject(
      {
        ok: true,
        channel: { name: "telegram", enabled: true },
      },
    );
    expect(
      (await client.next(eventFrame("channel_status"))).params?.payload,
    ).toMatchObject({
      channels: [{ name: "telegram", enabled: true }],
    });
    expect(channel.starts).toBe(1);

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "channel.enable",
      params: { name: "missing" },
    });
    expect((await client.next((frame) => frame.id === 4)).result).toEqual({
      ok: false,
      error: "channel 'missing' is not configured",
    });

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "channel.disable",
      params: { channel: "telegram" },
    });
    expect((await client.next((frame) => frame.id === 5)).result).toMatchObject(
      {
        ok: true,
        channel: { name: "telegram", enabled: false },
      },
    );
    expect(
      (await client.next(eventFrame("channel_status"))).params?.payload,
    ).toMatchObject({
      channels: [{ name: "telegram", enabled: false }],
    });
    expect(channel.stops).toBe(1);

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "channel.enable",
      params: { name: "telegram" },
    });
    expect((await client.next((frame) => frame.id === 6)).result).toMatchObject(
      {
        ok: true,
        channel: { name: "telegram", enabled: true },
      },
    );
    await client.next(eventFrame("channel_status"));
    await server.stop();
    expect(channel.stops).toBe(2);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon returns explicit channel-manager errors when no host adapters are configured", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-no-channels-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "channel.list", params: {} });
    expect((await client.next((frame) => frame.id === 1)).result).toEqual({
      ok: false,
      error: "channel manager is not configured",
      channels: [],
      channels_available: false,
      channels_configured: false,
    });
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "channel.enable",
      params: { name: "telegram" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: false,
      error: "channel manager is not configured",
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

class DaemonRecordingChannel implements Channel {
  readonly name: string;
  readonly sent: ChannelMessage[] = [];
  starts = 0;
  stops = 0;
  private handler: InboundHandler | undefined;

  constructor(name: string) {
    this.name = name;
  }

  async start(onInbound: InboundHandler): Promise<void> {
    this.starts += 1;
    this.handler = onInbound;
  }

  async stop(): Promise<void> {
    this.stops += 1;
    this.handler = undefined;
  }

  async send(message: ChannelMessage): Promise<void> {
    this.sent.push(message);
  }
}

class UsageRunner implements TurnRunner {
  async *run(): AsyncGenerator<DaemonEvent> {
    yield {
      type: "status_update",
      payload: {
        calls: 1,
        calls_complete: true,
        usage: { input_tokens: 17, output_tokens: 9 },
        usage_complete: true,
      },
    };
    yield { type: "text_part", payload: { text: "usage recorded" } };
  }
}

class StopOrderRuntime extends InMemoryDaemonRuntime {
  readonly shutdownOperations: string[] = [];

  override cancelAllTurns(): number {
    this.shutdownOperations.push("cancel");
    return super.cancelAllTurns();
  }

  override async flushSessions(): Promise<void> {
    this.shutdownOperations.push("flush");
    await super.flushSessions();
  }

  override async shutdown(): Promise<void> {
    this.shutdownOperations.push("shutdown");
    await super.shutdown();
  }
}

class ReplyRunner implements TurnRunner {
  constructor(private readonly interactions: DaemonInteractionBoard) {}

  async *run(
    session: DaemonSession,
    _text: string,
    signal: AbortSignal,
  ): AsyncGenerator<DaemonEvent> {
    const request: PermissionRequest = {
      requestId: "approval-1",
      description: "Run a native control-flow test.",
      inputs: {},
      toolCall: {
        id: "tool-1",
        type: "function",
        function: { name: "WriteFile", arguments: {} },
      },
    };
    yield {
      type: "approval_request",
      payload: {
        id: request.requestId,
        request_id: request.requestId,
        description: request.description,
      },
    };
    const decision = await this.interactions
      .permissionBroker(session.id)
      .request(request, signal);
    yield { type: "text_part", payload: { text: `approval:${decision}` } };
    const answer = await this.interactions.ask(
      session.id,
      { question: "Continue?", options: ["yes", "no"], allowFreeform: false },
      signal,
    );
    yield { type: "text_part", payload: { text: `answer:${answer}` } };
  }
}

class SteerRunner implements TurnRunner {
  private resolveGate: (() => void) | undefined;
  private readonly gate = new Promise<void>((resolve) => {
    this.resolveGate = resolve;
  });

  release(): void {
    this.resolveGate?.();
  }

  async *run(
    _session: DaemonSession,
    _text: string,
    _signal: AbortSignal,
    controls?: TurnRunControls,
  ): AsyncGenerator<DaemonEvent> {
    yield { type: "text_part", payload: { text: "waiting for steer" } };
    await this.gate;
    yield {
      type: "text_part",
      payload: { text: `steer:${controls?.drainSteer?.().join("|") ?? ""}` },
    };
  }
}

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

async function waitFor(
  predicate: () => boolean,
  timeout = 2_000,
): Promise<void> {
  const deadline = Date.now() + timeout;
  while (!predicate()) {
    if (Date.now() >= deadline) {
      throw new Error("Timed out waiting for native daemon state");
    }
    await Bun.sleep(10);
  }
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

  /** Write several frames in one chunk so the server parses them back-to-back. */
  sendBatch(frames: ReadonlyArray<Record<string, unknown>>): void {
    this.socket.write(
      frames.map((frame) => `${JSON.stringify(frame)}\n`).join(""),
    );
  }

  private receive(chunk: string): void {
    this.buffer += chunk;
    let newline = this.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      if (line.trim()) {
        this.handle(JSON.parse(line) as Frame);
      }
      newline = this.buffer.indexOf("\n");
    }
  }

  private handle(frame: Frame): void {
    const waiterIndex = this.waiters.findIndex((waiter) =>
      waiter.predicate(frame),
    );
    const waiter =
      waiterIndex >= 0 ? this.waiters.splice(waiterIndex, 1)[0] : undefined;
    if (waiter) {
      waiter.resolve(frame);
      return;
    }
    this.frames.push(frame);
  }
}

test("disconnect cancels only the turn submitted by the disconnecting connection", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-turn-owner-"));
  const socketPath = join(directory, "daemon.sock");
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "gate-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    runtime,
    socketPath,
  });
  await server.start();
  const owner = await SocketTestClient.connect(socketPath);
  const bystander = await SocketTestClient.connect(socketPath);
  try {
    owner.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "shared-turn" },
    });
    await owner.next((frame) => frame.id === 1);
    await owner.next(eventFrame("init_done"));
    await owner.next(eventFrame("status_update"));
    owner.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "shared-turn", text: "long shared work" },
    });
    await owner.next((frame) => frame.id === 2);
    await owner.next(eventFrame("turn_begin"));
    await waitFor(() => runner.runs === 1);

    bystander.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.open",
      params: { session_key: "shared-turn" },
    });
    await bystander.next((frame) => frame.id === 3);
    // A duplicate submit on the shared key fails and must not transfer
    // cancellation ownership away from the original submitter.
    bystander.send({
      jsonrpc: "2.0",
      id: 4,
      method: "turn.submit",
      params: { session_key: "shared-turn", text: "duplicate work" },
    });
    await bystander.next((frame) => frame.id === 4);
    await bystander.next(
      (frame) =>
        frame.method === "event" &&
        String(frame.params?.payload?.message ?? "").includes("already active"),
    );
    bystander.close();
    // A different connection's disconnect must leave the live turn alone.
    await Bun.sleep(100);
    expect(runtime.sessionStatus("shared-turn")?.activeTurnId).not.toBe("");
    expect(runtime.sessionStatus("shared-turn")?.cancelRequested).toBe(false);

    owner.close();
    await waitFor(
      () => runtime.sessionStatus("shared-turn")?.activeTurnId === "",
    );
    expect(runtime.sessionStatus("shared-turn")?.cancelRequested).toBe(true);
  } finally {
    owner.close();
    bystander.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon stop drains in-flight turns so their final state reaches disk", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-stop-drain-"));
  const socketPath = join(directory, "daemon.sock");
  const sessionDirectory = join(directory, "sessions");
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "drain-model",
    sessionDirectory,
  });
  const server = new DaemonServer({
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    runtime,
    socketPath,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "drain-session" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "drain-session", text: "persist my final state" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    await waitFor(() => runner.runs === 1);

    await server.stop();

    const files = await readdir(sessionDirectory);
    expect(files).toHaveLength(1);
    const saved = JSON.parse(
      await readFile(join(sessionDirectory, String(files[0])), "utf8"),
    ) as { messages: Array<{ content?: unknown; role?: string }> };
    expect(saved.messages).toContainEqual(
      expect.objectContaining({ role: "user", content: "persist my final state" }),
    );
    expect(saved.messages).toContainEqual(
      expect.objectContaining({
        role: "assistant",
        content: expect.stringContaining("turn drained"),
      }),
    );
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon drops a Unix client whose buffered request exceeds the frame limit", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-frame-limit-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    maxSocketFrameBytes: 1024,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
    socketPath,
  });
  await server.start();
  const previousError = console.error;
  const errors: unknown[][] = [];
  console.error = (...args: unknown[]) => {
    errors.push(args);
  };
  try {
    const offender = connect({ path: socketPath });
    await new Promise<void>((resolve, reject) => {
      offender.once("connect", resolve);
      offender.once("error", reject);
    });
    const closed = new Promise<boolean>((resolve) =>
      offender.once("close", (hadError) => resolve(hadError)),
    );
    offender.write("x".repeat(2_000));
    await closed;
    expect(
      errors.some((entry) =>
        String(entry[0]).includes("exceeds the socket frame limit"),
      ),
    ).toBe(true);

    const survivor = await SocketTestClient.connect(socketPath);
    try {
      survivor.send({
        jsonrpc: "2.0",
        id: 1,
        method: "runtime.status",
        params: {},
      });
      expect((await survivor.next((frame) => frame.id === 1)).result).toMatchObject({
        ok: true,
      });
    } finally {
      survivor.close();
    }
  } finally {
    console.error = previousError;
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("session.list without an active session refuses to silently scope to the daemon cwd", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-list-scope-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
    socketPath,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "session.list",
      params: { kind: "main" },
    });
    expect((await client.next((frame) => frame.id === 1)).result).toEqual({
      ok: false,
      error:
        "project-scoped session.list needs an active session or project_dir; pass scope \"global\" to list every project",
    });

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "session.list",
      params: { kind: "main", project_dir: directory },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
      sessions: [],
    });

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.list",
      params: { scope: "global" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
      sessions: [],
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("resuming a saved session evicts its live session registered under another key", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-resume-evict-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "resume-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    runtime,
    socketPath,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "picker-key" },
    });
    const initialized = await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    const initializedSession = initialized.result?.session as
      | { id?: string }
      | undefined;
    const sessionId = initializedSession?.id ?? "";
    expect(sessionId).not.toBe("");

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "picker-key", text: "remember this turn" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_end"));
    expect(runtime.sessionStatus("picker-key")?.id).toBe(sessionId);

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: `/resume ${sessionId}` },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toMatchObject({
      ok: true,
      session: { id: sessionId },
    });

    expect(runtime.sessionStatus("picker-key")).toBeUndefined();
    const live = runtime.listSessions().filter((session) => session.id === sessionId);
    expect(live).toHaveLength(1);
    expect(live[0]?.sessionKey).toBe(sessionId);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("requests on one connection dispatch serially so a queued turn lands in the newly opened session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-serial-dispatch-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "serial-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ runtime, socketPath });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "serial-start" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    // Parsed in one chunk: session.open must fully dispatch before
    // turn.submit reads the connection's active session key.
    client.sendBatch([
      {
        jsonrpc: "2.0",
        id: 2,
        method: "session.open",
        params: { session_key: "serial-target" },
      },
      {
        jsonrpc: "2.0",
        id: 3,
        method: "turn.submit",
        params: { text: "serialized dispatch" },
      },
    ]);
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({
      ok: true,
      session: { key: "serial-target" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
    });
    await client.next(eventFrame("turn_end"));

    expect(
      runtime.sessionStatus("serial-target")?.messages.map((message) => message.role),
    ).toEqual(["user", "assistant"]);
    expect(runtime.sessionStatus("serial-start")?.messages).toHaveLength(0);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("initialize adopts a live session with an active turn and reports ultra mode", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-init-adopt-"));
  const socketPath = join(directory, "daemon.sock");
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "adopt-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ runtime, socketPath });
  await server.start();
  const owner = await SocketTestClient.connect(socketPath);
  const adopter = await SocketTestClient.connect(socketPath);
  try {
    owner.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "adopt-shared" },
    });
    const initialized = await owner.next((frame) => frame.id === 1);
    await owner.next(eventFrame("init_done"));
    await owner.next(eventFrame("status_update"));
    expect(initialized.result).toMatchObject({ ok: true, ultra_mode: false });
    const sessionId = String(
      (initialized.result?.session as { id?: string } | undefined)?.id ?? "",
    );
    expect(sessionId).not.toBe("");

    owner.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/ultra" },
    });
    expect((await owner.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
      ultra_mode: true,
    });
    await owner.next(eventFrame("status_update"));

    owner.send({
      jsonrpc: "2.0",
      id: 3,
      method: "turn.submit",
      params: { session_key: "adopt-shared", text: "long adopt work" },
    });
    await owner.next((frame) => frame.id === 3);
    await owner.next(eventFrame("turn_begin"));
    await waitFor(() => runner.runs === 1);

    // A second initialize on the busy key must adopt, not evict, the live
    // session another connection is using.
    adopter.send({
      jsonrpc: "2.0",
      id: 4,
      method: "initialize",
      params: { session_key: "adopt-shared" },
    });
    const adopted = await adopter.next((frame) => frame.id === 4);
    expect(adopted.result).toMatchObject({
      ok: true,
      ultra_mode: true,
      session: { id: sessionId, status: "working" },
    });
    expect(runtime.sessionStatus("adopt-shared")?.activeTurnId).not.toBe("");
    expect(runtime.sessionStatus("adopt-shared")?.cancelRequested).toBe(false);

    owner.send({
      jsonrpc: "2.0",
      id: 5,
      method: "turn.cancel",
      params: { session_key: "adopt-shared" },
    });
    await owner.next((frame) => frame.id === 5);
    await owner.next(eventFrame("turn_end"));
    await waitFor(
      () => runtime.sessionStatus("adopt-shared")?.activeTurnId === "",
    );

    // Once the session is idle again, initialize resets the key to a fresh
    // session as before.
    adopter.send({
      jsonrpc: "2.0",
      id: 6,
      method: "initialize",
      params: { session_key: "adopt-shared" },
    });
    const reset = await adopter.next((frame) => frame.id === 6);
    expect(reset.result).toMatchObject({ ok: true, ultra_mode: false });
    const resetId = String(
      (reset.result?.session as { id?: string } | undefined)?.id ?? "",
    );
    expect(resetId).not.toBe(sessionId);
    expect(runtime.sessionStatus("adopt-shared")?.messages).toHaveLength(0);
  } finally {
    owner.close();
    adopter.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("slash-submitted image turns are tracked so disconnect cancels them", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-image-tracked-"));
  const socketPath = join(directory, "daemon.sock");
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "image-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    runtime,
    socketPath,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "image-tracked" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/image a tiny moon" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
      queued: true,
    });
    await client.next(eventFrame("turn_begin"));
    await waitFor(() => runner.runs === 1);
    expect(runtime.sessionStatus("image-tracked")?.activeTurnId).not.toBe("");

    client.close();
    await waitFor(
      () => runtime.sessionStatus("image-tracked")?.activeTurnId === "",
    );
    expect(runtime.sessionStatus("image-tracked")?.cancelRequested).toBe(true);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon stop drains slash-submitted turns so their final state reaches disk", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-slash-drain-"));
  const socketPath = join(directory, "daemon.sock");
  const sessionDirectory = join(directory, "sessions");
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "slash-drain-model",
    sessionDirectory,
  });
  const server = new DaemonServer({
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    runtime,
    socketPath,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "slash-drain" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/image drain my slash turn" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    await waitFor(() => runner.runs === 1);

    await server.stop();

    const files = await readdir(sessionDirectory);
    expect(files).toHaveLength(1);
    const saved = JSON.parse(
      await readFile(join(sessionDirectory, String(files[0])), "utf8"),
    ) as { messages: Array<{ content?: unknown; role?: string }> };
    expect(saved.messages).toContainEqual(
      expect.objectContaining({
        role: "assistant",
        content: expect.stringContaining("turn drained"),
      }),
    );
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon stop drains scheduled cron turns before flushing sessions", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-cron-drain-"));
  const socketPath = join(directory, "daemon.sock");
  const sessionDirectory = join(directory, "sessions");
  const store = new JobStore(join(directory, "cron", "jobs.json"));
  store.add(
    new CronJob({
      id: "gated-job",
      prompt: "gated cron work",
      nextRunAt: new Date(Date.now() - 1_000).toISOString(),
      oneshot: true,
    }),
  );
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "cron-drain-model",
    sessionDirectory,
  });
  const server = new DaemonServer({
    cronArchiveDirectory: join(directory, "cron", "archive"),
    cronPollInterval: 5,
    cronStoreFactory: () => store,
    runtime,
    socketPath,
  });
  await server.start();
  try {
    await waitFor(() => runner.runs === 1);
    await server.stop();

    const files = await readdir(sessionDirectory);
    expect(files).toHaveLength(1);
    const saved = JSON.parse(
      await readFile(join(sessionDirectory, String(files[0])), "utf8"),
    ) as { messages: Array<{ content?: unknown; role?: string }> };
    expect(saved.messages).toContainEqual(
      expect.objectContaining({ role: "user", content: "gated cron work" }),
    );
    expect(saved.messages).toContainEqual(
      expect.objectContaining({
        role: "assistant",
        content: expect.stringContaining("turn drained"),
      }),
    );
  } finally {
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("a failed resume leaves the connection on its current session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-resume-fail-"));
  const projectDirectory = join(directory, "project-a");
  const sessionDirectory = join(directory, "sessions");
  const socketPath = join(directory, "daemon.sock");
  await mkdir(sessionDirectory, { recursive: true });
  await writeFile(
    join(sessionDirectory, "cafebabe0001.json"),
    JSON.stringify({
      format: "xerxes-daemon-session",
      schema_version: 2,
      session_id: "cafebabe0001",
      key: "cafebabe0001",
      agent_id: "default",
      cwd: projectDirectory,
      workspace: "",
      updated_at: "2026-07-17T00:02:00.000Z",
      messages: [
        { role: "user", content: "saved request" },
        { role: "assistant", content: "saved response" },
      ],
      turn_count: 1,
      interaction_mode: "code",
      plan_mode: false,
      total_input_tokens: 1,
      total_output_tokens: 1,
      metadata: { project_root: projectDirectory },
      thinking_content: [],
      tool_executions: [],
    }),
    "utf8",
  );
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: projectDirectory,
    model: "resume-model",
    transcriptStore: new FailingLoadTranscriptStore({
      currentProjectDirectory: projectDirectory,
      directory: sessionDirectory,
    }),
  });
  const server = new DaemonServer({ runtime, socketPath });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { project_dir: projectDirectory, session_key: "resume-state" },
    });
    const initialized = await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    const sessionId = String(
      (initialized.result?.session as { id?: string } | undefined)?.id ?? "",
    );
    expect(sessionId).not.toBe("");

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "resume-state", text: "stay here" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_end"));

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: "/resume cafebabe0001" },
    });
    const failed = await client.next((frame) => frame.id === 3);
    expect(failed.error?.message ?? "").toContain("transcript store exploded");

    // The failed resume must not have evicted the live session or moved the
    // connection's active session key.
    expect(runtime.sessionStatus("resume-state")?.id).toBe(sessionId);
    expect(runtime.sessionStatus("resume-state")?.messages).toHaveLength(2);
    expect(runtime.sessionStatus("cafebabe0001")).toBeUndefined();

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "turn.submit",
      params: { text: "still home" },
    });
    await client.next((frame) => frame.id === 4);
    const turnBegin = await client.next(eventFrame("turn_begin"));
    expect(turnBegin.params?.payload).toMatchObject({ session_id: sessionId });
    await client.next(eventFrame("turn_end"));
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("a failed retry restores the discarded user turn", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-retry-restore-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new FlakySubmitRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "retry-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ runtime, socketPath });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "retry-restore" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "retry-restore", text: "keep this prompt" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_end"));
    const session = runtime.sessionStatus("retry-restore");
    expect(session?.messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
    expect(session?.turnCount).toBe(1);

    runtime.failSubmits = true;
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: "/retry" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
      retried: true,
    });
    const failure = await client.next(
      (frame) =>
        frame.method === "event" &&
        frame.params?.type === "notification" &&
        String(frame.params?.payload?.body ?? "").includes("Retry failed"),
    );
    expect(failure.params?.payload).toMatchObject({ severity: "error" });

    const restored = runtime.sessionStatus("retry-restore");
    expect(restored?.messages).toHaveLength(2);
    expect(restored?.messages[0]).toMatchObject({
      role: "user",
      content: "keep this prompt",
    });
    expect(restored?.messages[1]).toMatchObject({ role: "assistant" });
    expect(restored?.turnCount).toBe(1);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon advertises /ultra in catalog, completion, and slash handling", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-ultra-command-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "ultra-command" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({ jsonrpc: "2.0", id: 2, method: "commands.catalog", params: {} });
    const catalog = await client.next((frame) => frame.id === 2);
    expect(catalog.result?.pairs).toContainEqual(["/ultra", "Toggle ultra mode"]);
    expect(catalog.result?.canon).toMatchObject({ "/ultra": "/ultra" });
    expect(catalog.result?.categories).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "daemon",
          pairs: expect.arrayContaining([["/ultra", "Toggle ultra mode"]]),
        }),
      ]),
    );

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "complete",
      params: { text: "/ult" },
    });
    expect(
      (await client.next((frame) => frame.id === 3)).result?.completions,
    ).toEqual([
      { value: "/ultra", label: "ultra", meta: "Toggle ultra mode" },
    ]);

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "slash",
      params: { command: "/ultra" },
    });
    expect((await client.next((frame) => frame.id === 4)).result).toEqual({
      ok: true,
      ultra_mode: true,
    });
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "slash",
      params: { command: "/ultra off" },
    });
    expect((await client.next((frame) => frame.id === 5)).result).toEqual({
      ok: true,
      ultra_mode: false,
    });
    await client.next(eventFrame("status_update"));
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("turn completion releases approval ownership so late replies are not blocked", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-owner-cleanup-"));
  const socketPath = join(directory, "daemon.sock");
  const interactions = new DaemonInteractionBoard();
  const runtime = new InMemoryDaemonRuntime(new ReplyRunner(interactions), {
    currentProjectDirectory: directory,
    interactions,
    model: "cleanup-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ socketPath, runtime, interactions });
  await server.start();
  const owner = await SocketTestClient.connect(socketPath);
  const late = await SocketTestClient.connect(socketPath);
  try {
    owner.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "owner-cleanup" },
    });
    await owner.next((frame) => frame.id === 1);
    await owner.next(eventFrame("init_done"));
    await owner.next(eventFrame("status_update"));
    owner.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "owner-cleanup", text: "wait for approval" },
    });
    await owner.next((frame) => frame.id === 2);
    await owner.next(eventFrame("turn_begin"));
    await owner.next(eventFrame("approval_request"));

    // Cancel the turn without answering; once the turn settles, its approval
    // ownership entry must not block other connections.
    owner.send({
      jsonrpc: "2.0",
      id: 3,
      method: "turn.cancel",
      params: { session_key: "owner-cleanup" },
    });
    await owner.next((frame) => frame.id === 3);
    await owner.next(eventFrame("turn_end"));
    await waitFor(
      () => runtime.sessionStatus("owner-cleanup")?.activeTurnId === "",
    );

    late.send({
      jsonrpc: "2.0",
      id: 4,
      method: "permission_response",
      params: { request_id: "approval-1", response: "approve" },
    });
    expect((await late.next((frame) => frame.id === 4)).result).toEqual({
      ok: false,
    });
  } finally {
    owner.close();
    late.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

class FailingLoadTranscriptStore extends DaemonTranscriptStore {
  override load(): Promise<never> {
    throw new Error("transcript store exploded");
  }
}

class FlakySubmitRuntime extends InMemoryDaemonRuntime {
  failSubmits = false;

  override async submitTurn(
    sessionKey: string,
    text: string,
    emit: (event: DaemonEvent) => void,
    options: SubmitTurnOptions = {},
  ): Promise<void> {
    if (this.failSubmits) {
      throw new Error("provider submit exploded");
    }
    return super.submitTurn(sessionKey, text, emit, options);
  }
}

class AbortGateRunner implements TurnRunner {
  runs = 0;

  async *run(
    _session: DaemonSession,
    _text: string,
    signal: AbortSignal,
  ): AsyncGenerator<DaemonEvent> {
    this.runs += 1;
    yield { type: "text_part", payload: { text: "started" } };
    await new Promise<void>((resolve) => {
      if (signal.aborted) {
        resolve();
        return;
      }
      signal.addEventListener("abort", () => resolve(), { once: true });
    });
    yield { type: "text_part", payload: { text: "turn drained" } };
  }
}
