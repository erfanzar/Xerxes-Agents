// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { AgentSettingsStore } from '../src/agents/settingsStore.js';
import { DeclarativeToolForge } from '../src/extensions/declarativeForge.js';
import { createGoal, recordGoalEvidence } from '../src/runtime/goalDomain.js';
import { expect, test } from "bun:test";
import { existsSync } from "node:fs";
import { mkdir, mkdtemp, readdir, readFile, realpath, rm, writeFile } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { nativeSubagentWorktrees } from '../src/runtime/subagentWorktrees.js';
import { InMemoryDaemonRuntime } from "../src/daemon/runtime.js";
import { AgentPresetRoster } from "../src/agents/presets.js";
import { DaemonInteractionBoard } from "../src/daemon/interactions.js";
import { MCPManager } from "../src/mcp/manager.js";
import { McpSettingsStore } from "../src/mcp/settingsStore.js";
import { PluginRegistry } from "../src/extensions/plugins.js";
import {
  DaemonServer,
  MAX_ACCEPTED_SUBMISSION_IDS,
  MIGRATED_ERROR,
} from "../src/daemon/server.js";
import { ValidationError } from "../src/core/errors.js";
import { TerminalRegistry } from "../src/runtime/terminalRegistry.js";
import { PtySessionManager } from "../src/operators/pty.js";
import { ToolRegistry } from "../src/executors/toolRegistry.js";
import { registerMonitorTools } from "../src/tools/monitorTools.js";
import { Scheduler as LegacyScheduler } from "../src/runtime/scheduler.js";
import { DeliveryOutbox } from "../src/cron/outbox.js";
import { RunHistory } from "../src/runtime/runHistory.js";
import { ReactionMailbox } from "../src/runtime/reactionMailbox.js";
import { TerminalMonitors } from "../src/runtime/terminalMonitors.js";
import { ProfileStore } from "../src/bridge/profiles.js";
import {
  ChannelManager,
  type Channel,
  type ChannelMessage,
  type InboundHandler,
} from "../src/channels/index.js";
import { CronJob, JobStore } from "../src/cron/jobs.js";
import { readCronLease } from "../src/cron/lease.js";
import { processCommand } from "../src/core/processLiveness.js";
import { DaemonTranscriptStore } from "../src/session/daemonTranscript.js";
import { SnapshotManager } from "../src/session/snapshots.js";
import { resetTitleAttempts } from "../src/daemon/titleGenerator.js";
import type {
  CompletionRequest,
  FetchImplementation,
  LlmClient,
  LlmDelta,
} from "../src/llms/client.js";
import type { PermissionRequest } from "../src/streaming/events.js";
import type {
  DaemonEvent,
  DaemonSession,
  SubmitTurnOptions,
  TurnRunControls,
  TurnRunner,
} from "../src/daemon/runtime.js";

test("daemon slash RPC runs ! shell mode and # memory notes", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-shell-hash-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    projectDirectory: directory,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "protocol-model",
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
      params: { session_key: "shell-hash", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    // ! runs in the project shell and returns captured output.
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "!echo shell-mode-works" },
    });
    const shellResponse = await client.next((frame) => frame.id === 2);
    expect(shellResponse.result).toMatchObject({ code: 0, ok: true });
    expect(shellResponse.result?.stdout).toContain("shell-mode-works");
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({ category: "slash", body: "shell-mode-works" });

    // Non-zero exits surface the code instead of pretending success.
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: "!exit 3" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toMatchObject({
      code: 3,
      ok: false,
    });
    await client.next(eventFrame("notification"));

    // # appends one line to project MEMORY.md through the workspace store.
    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "slash",
      params: { command: "#this project uses Bun" },
    });
    const noteResponse = await client.next((frame) => frame.id === 4);
    expect(noteResponse.result).toMatchObject({ id: 1, ok: true });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({ category: "slash", severity: "info" });
    const memoryBody = await Bun.file(join(directory, "MEMORY.md")).text();
    expect(memoryBody).toContain("this project uses Bun");

    // Empty payloads stay usage errors, not silent no-ops.
    client.send({ jsonrpc: "2.0", id: 5, method: "slash", params: { command: "!" } });
    expect((await client.next((frame) => frame.id === 5)).result).toMatchObject({ ok: false });
    await client.next(eventFrame("notification"));
    client.send({ jsonrpc: "2.0", id: 6, method: "slash", params: { command: "#" } });
    expect((await client.next((frame) => frame.id === 6)).result).toMatchObject({ ok: false });
    await client.next(eventFrame("notification"));
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

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
      context_limit: 0,
      mode: "code",
    });
    expect(initialStatus.params?.payload).toMatchObject({
      max_context: 0,
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

test("workspace file preview RPC reads only active-session workspace text without starting a turn", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-preview-rpc-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, "sessions") });
  const server = new DaemonServer({ socketPath, runtime, projectDirectory: directory });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    await Bun.write(join(directory, "source.ts"), "first\n  second\n");
    client.send({ jsonrpc: "2.0", id: 1, method: "workspace.filePreview", params: { path: "source.ts" } });
    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({ ok: false });
    client.send({ jsonrpc: "2.0", id: 2, method: "initialize", params: { session_key: "preview", project_dir: directory } });
    await client.next(frame => frame.id === 2);
    client.send({ jsonrpc: "2.0", id: 3, method: "workspace.filePreview", params: { session_key: "preview", path: "source.ts" } });
    expect((await client.next(frame => frame.id === 3)).result).toEqual({ ok: true, path: "source.ts", content: "first\n  second\n", truncated: false });
    expect(runtime.sessionStatus("preview")?.turnCount).toBe(0);
    client.send({ jsonrpc: "2.0", id: 4, method: "workspace.filePreview", params: { session_key: "preview", path: "missing.ts" } });
    expect((await client.next(frame => frame.id === 4)).error).toBeDefined();
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test("project agent editor RPC scopes writes and preserves invalid drafts on disk", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-project-agent-rpc-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, "sessions") });
  const server = new DaemonServer({ socketPath, runtime, projectDirectory: directory });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    const content = '---\nname: "reviewer"\ndescription: Review code\n---\nFind defects.';
    client.send({ jsonrpc: "2.0", id: 1, method: "agentPreset.projectWrite", params: { content, revision: null } });
    const saved = (await client.next(frame => frame.id === 1)).result as Record<string, unknown>;
    expect(saved).toMatchObject({ ok: true, id: "reviewer", content });
    client.send({ jsonrpc: "2.0", id: 2, method: "agentPreset.projectList", params: {} });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, agents: [{ id: "reviewer" }] });
    client.send({ jsonrpc: "2.0", id: 3, method: "agentPreset.projectWrite", params: { id: "reviewer", content: "---\nname: wrong\n---\nBroken", revision: saved.revision } });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ ok: false });
    client.send({ jsonrpc: "2.0", id: 4, method: "agentPreset.projectRead", params: { id: "reviewer" } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: true, content, revision: saved.revision });
    expect(await readFile(join(directory, ".xerxes/agents/reviewer.md"), "utf8")).toBe(content);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test("Forge RPC and slash discovery preserve immutable definitions and explicit confirmation", async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-forge-rpc-'));
  const socketPath = join(directory, 'daemon.sock');
  const storage = join(directory, 'forge.json');
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath, runtime, projectDirectory: directory,
    declarativeForge: new DeclarativeToolForge(storage), cronLeasePath: join(directory, 'cron.lease'),
    cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  let id = 0;
  const rpc = async (method: string, params: Record<string, unknown> = {}) => {
    const requestId = ++id;
    client.send({ jsonrpc: '2.0', id: requestId, method, params });
    return (await client.next(frame => frame.id === requestId)).result as Record<string, unknown>;
  };
  try {
    await rpc('initialize', { session_key: 'forge-test' });
    expect(await rpc('forge.list')).toMatchObject({ ok: true, packages: [] });
    const definition = { name: 'greeting', version: '1.0.0', description: 'Readable greeting', template: 'Hello {{name}}\nReady.', parameters: [{ name: 'name', description: 'Recipient', required: true }] };
    expect(await rpc('forge.define', definition)).toMatchObject({ ok: false });
    expect(await rpc('forge.define', { ...definition, confirm: true })).toMatchObject({ ok: true });
    expect(new DeclarativeToolForge(storage).inspect('greeting', '1.0.0')?.template).toBe(definition.template);
    expect(await rpc('forge.define', { ...definition, template: 'overwrite', confirm: true })).toMatchObject({ ok: false });
    expect(await rpc('forge.run', { name: 'greeting', input: {} })).toMatchObject({ ok: false });
    expect(await rpc('forge.run', { name: 'greeting', input: { name: 'Ada' } })).toMatchObject({ ok: true, output: 'Hello Ada\nReady.' });
    const inspected = await rpc('slash', { command: '/forge inspect greeting' });
    expect(inspected.ok).toBe(true);
    expect(inspected.output).toContain('Readable greeting');
    expect(inspected.output).toContain('Hello {{name}}');
    expect(await rpc('forge.undefine', { name: 'greeting', version: '1.0.0' })).toMatchObject({ ok: false });
    expect(await rpc('forge.inspect', { name: 'greeting' })).toMatchObject({ ok: true });
    expect(await rpc('forge.stop')).toMatchObject({ ok: false });
    expect(await rpc('forge.undefine', { name: 'greeting', version: '1.0.0', confirm: true })).toMatchObject({ ok: true });
    expect(new DeclarativeToolForge(storage).list()).toEqual([]);
  } finally {
    client.close();
    await server.stop();
    await runtime.shutdown();
    await rm(directory, { recursive: true, force: true });
  }
});

test("agent preset RPC mirrors DSH roster, authoring, defaults, and blank-session locking", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-agent-presets-rpc-"));
  const socketPath = join(directory, "daemon.sock");
  const roster = new AgentPresetRoster({
    home: directory,
    projectDirectory: directory,
    userDirectory: join(directory, "agents"),
    settingsPath: join(directory, "agent-presets.json"),
  });
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ socketPath, runtime, projectDirectory: directory, agentPresetRoster: roster });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { session_key: "preset-session" } });
    const initialized = (await client.next(frame => frame.id === 1)).result as Record<string, unknown>;
    expect(initialized).toMatchObject({
      ok: true,
      session: { agent_id: "default" },
    });
    expect(initialized.session as Record<string, unknown>).not.toHaveProperty("cache_hit_rate");

    client.send({ jsonrpc: "2.0", id: 2, method: "agentPreset.list", params: {} });
    const listed = (await client.next(frame => frame.id === 2)).result as Record<string, unknown>;
    expect(listed).toMatchObject({ ok: true, default_id: "default", authorable: true });
    expect((listed.presets as Array<Record<string, unknown>>).some(row => row.id === "creator" && row.trust === "system")).toBe(true);

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "agentPreset.select",
      params: { session_key: "preset-session", agent_preset: "creator" },
    });
    expect((await client.next(frame => frame.id === 3)).result).toEqual({ ok: true, agent_preset: "creator" });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "agentPreset.copy",
      params: { from: "creator", agent_preset: "my-creator", name: "My Creator" },
    });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({
      ok: true,
      preset: { id: "my-creator", name: "My Creator", trust: "user", manageable: true },
    });
    client.send({ jsonrpc: "2.0", id: 40, method: "agentPreset.read", params: { agent_preset: "my-creator" } });
    const original = (await client.next(frame => frame.id === 40)).result!;
    expect(original.guarded_write).toBe(true);
    const nextContent = String(original.content) + "\n# User edit\n";
    client.send({ jsonrpc: "2.0", id: 41, method: "agentPreset.write", params: { agent_preset: "my-creator", content: nextContent, expected_content: original.content } });
    expect((await client.next(frame => frame.id === 41)).result?.ok).toBe(true);
    client.send({ jsonrpc: "2.0", id: 42, method: "agentPreset.write", params: { agent_preset: "my-creator", content: original.content, expected_content: original.content } });
    expect((await client.next(frame => frame.id === 42)).result).toMatchObject({ ok: false, code: "agent-preset-stale" });
    expect(roster.read("my-creator", directory).content).toBe(nextContent);
    client.send({ jsonrpc: "2.0", id: 43, method: "agentPreset.write", params: { agent_preset: "my-creator", content: "invalid: composition", expected_content: nextContent } });
    expect((await client.next(frame => frame.id === 43)).result?.ok).toBe(false);
    expect(roster.read("my-creator", directory).content).toBe(nextContent);
    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "agentPreset.setDefault",
      params: { agent_preset: "my-creator" },
    });
    expect((await client.next(frame => frame.id === 5)).result).toMatchObject({ ok: true, default_id: "my-creator" });

    runtime.sessionStatus("preset-session")!.extra.runtime_telemetry = {
      cacheHitRate: 0.8,
      cacheTelemetryKnown: true,
      llmDurationMs: 12_500,
      llmSteps: 3,
      toolDurationMs: 750,
      toolSteps: 2,
      tokensPerSecond: 42,
      ttftSamples: 2,
      ttftTotalMs: 1_200,
    };
    client.send({ jsonrpc: "2.0", id: 6, method: "session.status", params: { session_key: "preset-session" } });
    expect((await client.next(frame => frame.id === 6)).result).toMatchObject({
      ok: true,
      session: {
        cache_hit_rate: 0.8,
        llm_duration_ms: 12_500,
        llm_steps: 3,
        tool_duration_ms: 750,
        tool_steps: 2,
        tokens_per_second: 42,
        ttft_avg_ms: 600,
      },
    });

    runtime.sessionStatus("preset-session")!.extra.runtime_telemetry = {
      ...runtime.sessionStatus("preset-session")!.extra.runtime_telemetry as Record<string, unknown>,
      tokensPerSecond: 8_527_132,
    };
    client.send({ jsonrpc: "2.0", id: 61, method: "session.status", params: { session_key: "preset-session" } });
    expect((await client.next(frame => frame.id === 61)).result).toMatchObject({
      session: { tokens_per_second: 0 },
    });

    // The lock is transcript-based, not a client busy flag. Marking one durable
    // user message is enough to represent a session whose model-visible history
    // was produced under Creator mode.
    runtime.sessionStatus("preset-session")!.messages.push({ role: "user", content: "begin" });

    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "agentPreset.select",
      params: { session_key: "preset-session", agent_preset: "default" },
    });
    expect((await client.next(frame => frame.id === 7)).result).toMatchObject({
      ok: false,
      code: "agent-preset-locked",
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("embedded daemon defaults reject untrusted workspace skills", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-daemon-skill-trust-"));
  const workspaceSkill = join(directory, ".agents", "skills", "workspace-only");
  const socketPath = join(directory, "daemon.sock");
  await mkdir(workspaceSkill, { recursive: true });
  await writeFile(
    join(workspaceSkill, "SKILL.md"),
    "---\nname: workspace-only\ndescription: Untrusted workspace fixture\n---\nReview the current workspace.",
    "utf8",
  );
  const server = new DaemonServer({
    socketPath,
    skillDirectories: [join(directory, "user-skills")],
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
      params: { session_key: "skill-trust", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/skills" },
    });
    const response = await client.next((frame) => frame.id === 2);
    expect(response.result).toMatchObject({ ok: true, skills: expect.any(Array) });
    expect(response.result?.skills).not.toEqual(
      expect.arrayContaining([
        expect.objectContaining({ name: "workspace-only" }),
      ]),
    );
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

test("a provider switch from a second client keeps the first session's context limit", async () => {
  // Regression: contextLimit resolved the session's model against the
  // daemon-wide ACTIVE profile only. A second TUI running provider_select
  // flipped the active profile, and every session still on the old provider
  // reported context_limit 0 — the TUI's "ctx unknown".
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-cross-profile-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    apiKey: "",
    baseUrl: "https://api.kimi.com/coding/v1",
    model: "k3",
    name: "kimi-code",
    provider: "kimi-code",
  });
  profileStore.save({
    apiKey: "",
    baseUrl: "https://api.other.example/v1",
    model: "m2",
    name: "other-provider",
    provider: "other-provider",
  });
  // Only profile A knows k3's window; profile B never heard of it.
  profileStore.updateModelCapabilities("kimi-code", "k3", { contextLimit: 262_144 });
  profileStore.updateModelCapabilities("other-provider", "m2", { contextLimit: 100_000 });
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
  await server.start();
  const first = await SocketTestClient.connect(socketPath);
  const second = await SocketTestClient.connect(socketPath);
  try {
    first.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "stays-on-kimi" },
    });
    await first.next((frame) => frame.id === 1);
    // Pin session 1 to k3 explicitly, the way the user's /model pick does.
    first.send({ jsonrpc: "2.0", id: 2, method: "slash", params: { command: "/model k3" } });
    await first.next((frame) => frame.id === 2);

    second.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "switching-client" },
    });
    await second.next((frame) => frame.id === 1);
    second.send({ jsonrpc: "2.0", id: 2, method: "provider_select", params: { name: "other-provider" } });
    expect((await second.next((frame) => frame.id === 2)).result).toMatchObject({ ok: true });

    first.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.status",
      params: { session_key: "stays-on-kimi" },
    });
    const status = (await first.next((frame) => frame.id === 3)).result?.session as
      | { context_limit?: number; model?: string }
      | undefined;
    expect(status).toMatchObject({ model: "k3" });
    expect(status?.context_limit).toBe(262_144);
  } finally {
    first.close();
    second.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("session model remains selected after a session reasoning change", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-session-config-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    apiKey: "test-key",
    baseUrl: "https://api.openai.com/v1",
    model: "k3-256k",
    name: "openai-test",
    provider: "openai",
  });
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    runtimeSettings: { model: "k3-256k", provider: "openai" },
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ profileStore, runtime, socketPath });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "configured-session" },
    });
    await client.next((frame) => frame.id === 1);

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "set_model",
      params: { model: "gpt-5.6-luna", session_key: "configured-session" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({
      ok: true,
      model: "gpt-5.6-luna",
    });

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "set_reasoning",
      params: { reasoning_effort: "max", session_key: "configured-session" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toMatchObject({
      ok: true,
      reasoning_effort: "max",
    });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "session.status",
      params: { session_key: "configured-session" },
    });
    expect((await client.next((frame) => frame.id === 4)).result?.session).toMatchObject({
      model: "gpt-5.6-luna",
      reasoning_effort: "max",
    });
    // The picker changed this session, not the runtime default inherited by
    // unrelated live chats.
    expect(runtime.status().model).toBe("k3-256k");
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon context limits layer live metadata over the Pi model catalog", async () => {
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
  const server = new DaemonServer({
    autoDiscoverModelCapabilities: true,
    profileStore,
    runtime,
    socketPath,
  });
  const nativeFetch = globalThis.fetch;
  let modelFetchCount = 0;
  const modelFetch: FetchImplementation = async () => {
    modelFetchCount += 1;
    return new Response(
      JSON.stringify({
        data: [
          modelFetchCount === 1
            ? { id: "k3", context_length: 262_144 }
            : modelFetchCount === 2
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
      context_limit: 1_048_576,
      session: { context_limit: 1_048_576, max_context: 1_048_576 },
    });
    await client.next(eventFrame("init_done"));
    expect((await client.next(eventFrame("status_update"))).params?.payload).toMatchObject({
      max_context: 1_048_576,
    });
    expect((await client.next((frame) =>
      eventFrame("status_update")(frame)
      && frame.params?.payload?.max_context === 262_144
    )).params?.payload).toMatchObject({ max_context: 262_144 });
    expect(profileStore.get("kimi-code")?.model_capabilities).toEqual({
      k3: { context_limit: 262_144 },
    });

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
      context_limit: 1_048_576,
      max_context: 1_048_576,
    });
    expect(profileStore.get("kimi-code")?.model_capabilities).toEqual({ k3: {} });

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "provider_model_override",
      params: {
        profile_name: "kimi-code",
        model: "k3",
        context_limit: 500_000,
        max_output_tokens: 100_000,
      },
    });
    expect((await client.next((frame) => frame.id === 6)).result).toMatchObject({
      ok: true,
      model: { context_limit: 500_000, max_output_tokens: 100_000, overridden: true },
    });
    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "session.status",
      params: { session_key: "provider-context" },
    });
    expect((await client.next((frame) => frame.id === 7)).result?.session).toMatchObject({
      context_limit: 500_000,
      max_context: 500_000,
    });
    client.send({
      jsonrpc: "2.0",
      id: 8,
      method: "provider_model_override",
      params: {
        profile_name: "kimi-code",
        model: "k3",
        context_limit: null,
        max_output_tokens: null,
      },
    });
    expect((await client.next((frame) => frame.id === 8)).result).toMatchObject({ ok: true });
    client.send({
      jsonrpc: "2.0",
      id: 9,
      method: "session.status",
      params: { session_key: "provider-context" },
    });
    expect((await client.next((frame) => frame.id === 9)).result?.session).toMatchObject({
      context_limit: 1_048_576,
      max_context: 1_048_576,
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
    // Every row carries its project directory — the workspace grouping key
    // the desktop sidebar groups chats under.
    expect(global.map((session) => [session.id, session.cwd])).toEqual([
      ["eeeeffff0001", otherProjectDirectory],
      ["aaaabbbb0001", projectDirectory],
      ["aaaabbbb0002", projectDirectory],
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
    expect((await client.next((frame) => frame.id === 9)).error?.message).toMatch(/different project|another workspace/);
    client.send({ jsonrpc: "2.0", id: 10, method: "initialize", params: {
      project_dir: otherProjectDirectory, resume_session_id: "ccccdddd0001",
    } });
    expect((await client.next(frame => frame.id === 10)).error?.message).toMatch(/different project|another workspace/);
    client.send({ jsonrpc: "2.0", id: 11, method: "session.list", params: {} });
    const retained = (await client.next(frame => frame.id === 11)).result;
    expect(retained?.ok).toBe(true);
    expect(JSON.stringify(retained)).not.toContain('Child history');
    client.send({ jsonrpc: "2.0", id: 12, method: "session.open", params: {
      session_key: "project-b-connection", project_dir: projectDirectory,
    } });
    expect((await client.next(frame => frame.id === 12)).error?.message).toContain('another workspace');
    client.send({ jsonrpc: "2.0", id: 13, method: "session.open", params: { session_key: "project-b-connection" } });
    expect((await client.next(frame => frame.id === 13)).result?.session).toMatchObject({ cwd: otherProjectDirectory });
    client.send({ jsonrpc: "2.0", id: 14, method: "initialize", params: {
      session_key: "project-b-connection", project_dir: projectDirectory,
    } });
    expect((await client.next(frame => frame.id === 14)).error?.message).toContain('another workspace');
    client.send({ jsonrpc: "2.0", id: 15, method: "session.open", params: { session_key: "project-b-connection" } });
    expect((await client.next(frame => frame.id === 15)).result?.session).toMatchObject({ cwd: otherProjectDirectory });
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

test("disconnect reaps an exchange-less session but keeps one with history", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-shell-reap-"));
  const socketPath = join(directory, "daemon.sock");
  // The gate resolver is built eagerly (like GatedRunner): a lazily-assigned
  // closure loses the race against turn.submit, which returns before the
  // runner has been pulled to its gate.
  const runner = new GatedRunner();
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(runner, {
      currentProjectDirectory: directory,
      model: "claude-code/default",
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    // Bind 1: an empty shell (initialize only, never spoken to).
    client.send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { session_key: "shell-key" } });
    await client.next((frame) => frame.id === 1);
    // Bind 2: a session that completes a real exchange.
    client.send({ jsonrpc: "2.0", id: 2, method: "initialize", params: { session_key: "real-key" } });
    await client.next((frame) => frame.id === 2);
    client.send({ jsonrpc: "2.0", id: 3, method: "turn.submit", params: { session_key: "real-key", text: "speak" } });
    // The submit response settles at turn end; release the gate first, then
    // await the settle — the same order the evict-race test uses.
    runner.release();
    await client.next(eventFrame("turn_end"));
    await client.next((frame) => frame.id === 3);
    const realId = (
      (await (async () => {
        client.send({ jsonrpc: "2.0", id: 4, method: "session.active_list", params: {} });
        return (await client.next((frame) => frame.id === 4)).result;
      })())?.sessions as Array<{ id: string; key: string }> | undefined
    )?.find((s) => s.key === "real-key")?.id;
    expect(realId).toEqual(expect.any(String));

    // Hang up: the shell (no exchange) is reaped; the real session survives.
    client.close();
    await new Promise((resolve) => setTimeout(resolve, 120));

    const checker = await SocketTestClient.connect(socketPath);
    try {
      checker.send({ jsonrpc: "2.0", id: 5, method: "session.active_list", params: {} });
      const listed = (await checker.next((frame) => frame.id === 5)).result?.sessions as Array<{ id: string; key: string }>;
      expect(listed.some((s) => s.key === "real-key" && s.id === realId)).toBe(true);
      expect(listed.some((s) => s.key === "shell-key")).toBe(false);
    } finally {
      checker.close();
    }
  } finally {
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("duplicate submit from the same owner cannot release ownership of the live turn", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-duplicate-owner-"));
  const socketPath = join(directory, "daemon.sock");
  const turnStarted = Promise.withResolvers<void>();
  let aborted = false;
  const runner: TurnRunner = {
    async *run(_session, _text, signal): AsyncGenerator<DaemonEvent> {
      turnStarted.resolve();
      await new Promise<void>((resolve) => {
        if (signal.aborted) {
          aborted = true;
          resolve();
          return;
        }
        signal.addEventListener("abort", () => {
          aborted = true;
          resolve();
        }, { once: true });
      });
      yield { type: "text_part", payload: { text: "stopped" } };
    },
  };
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "test-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ socketPath, runtime });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { session_key: "duplicate-owner" } });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({ jsonrpc: "2.0", id: 2, method: "turn.submit", params: { text: "first" } });
    await client.next((frame) => frame.id === 2);
    await turnStarted.promise;
    client.send({ jsonrpc: "2.0", id: 3, method: "turn.submit", params: { text: "duplicate" } });
    await client.next((frame) => frame.id === 3);
    await Bun.sleep(25);
    client.close();

    await waitFor(() => aborted);
    expect(runtime.sessionStatus("duplicate-owner")?.activeTurnId).toBe("");
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon deduplicates a repeated submission id after the first turn settles", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-submit-dedup-"));
  const socketPath = join(directory, "daemon.sock");
  const prompts: string[] = [];
  const runner: TurnRunner = {
    async *run(_session, text): AsyncGenerator<DaemonEvent> {
      prompts.push(text);
      yield { type: "text_part", payload: { text: "done" } };
    },
  };
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "test-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ socketPath, runtime });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { session_key: "dedup" } });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { submission_id: "submit-1", text: "only once" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({ ok: true });
    await client.next(eventFrame("turn_end"));

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "turn.submit",
      params: { submission_id: "submit-1", text: "only once" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({ duplicate: true, ok: true });
    await Bun.sleep(25);

    expect(prompts).toEqual(["only once"]);
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
    // An empty skill library keeps the exact completion assertions below
    // hermetic — single-token completions now merge skill shorthands, and
    // the developer's real ~/.xerxes/skills must never leak into them.
    skillDirectories: [join(directory, "user-skills")],
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
    for (const [index, name] of ['runs', 'monitors', 'schedules', 'goal', 'loop'].entries()) {
      client.send({ jsonrpc: '2.0', id: 100 + index, method: 'complete', params: { text: '/' + name.slice(0, 3) } });
      expect((await client.next(frame => frame.id === 100 + index)).result?.completions).toEqual(
        expect.arrayContaining([expect.objectContaining({ value: '/' + name, category: 'activity' })]),
      );
    }
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
        ["/runs", "Inspect run history and unread results"],
        ["/monitors", "Create and inspect terminal, file, WebSocket and webhook watches"],
        ["/schedules", "Create and manage workspace schedules"],
        ["/goal", "Set or view the goal for a long-running task"],
        ["/history", "Show or search conversation history"],
        ["/remove-memory", "Wipe ALL Xerxes agent memory (global)"],
        ["/remove-history", "Wipe ALL saved chat history and snapshots (global)"],
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
    // `category` rides along so the TUI can order the bare-slash menu by
    // group instead of alphabetically.
    ).toEqual([
      { value: "/help", label: "help", meta: "Show help", category: "info" },
    ]);

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
        category: "session",
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

test("complete hints skill references for /skill arguments", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-skill-complete-"));
  const socketPath = join(directory, "daemon.sock");
  const skillDirectory = join(directory, "user-skills");
  const skillHome = join(skillDirectory, "review");
  await mkdir(skillHome, { recursive: true });
  await writeFile(
    join(skillHome, "SKILL.md"),
    "---\nname: review\ndescription: Review code with subcommands\nsubcommands:\n  - security\n---\nReview the workspace.",
    "utf8",
  );
  // A second skill whose NAME only contains the search term deep inside, and
  // whose description carries the words a user would actually type.
  const huntHome = join(skillDirectory, "read-project-and-hunt-bugs");
  await mkdir(huntHome, { recursive: true });
  await writeFile(
    join(huntHome, "SKILL.md"),
    "---\nname: read-project-and-hunt-bugs\ndescription: Bug bounty hunting across the workspace\n---\nHunt for bugs.",
    "utf8",
  );
  const server = new DaemonServer({
    socketPath,
    skillDirectories: [skillDirectory],
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
      params: { session_key: "skill-complete" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    // A bare `/skill ` completes every trusted skill name, sorted by label…
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "complete",
      params: { text: "/skill " },
    });
    expect(
      (await client.next((frame) => frame.id === 2)).result?.completions,
    ).toEqual([
      { value: "/skill read-project-and-hunt-bugs ", label: "read-project-and-hunt-bugs", meta: "Bug bounty hunting across the workspace" },
      { value: "/skill review ", label: "review", meta: "Review code with subcommands" },
      { value: "/skill review:security ", label: "review:security", meta: "Review code with subcommands" },
    ]);

    // …a prefix narrows it, and a plain `/rev` still completes as a command
    // prefix against the daemon slash registry, not skills.
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "complete",
      params: { text: "/skill rev" },
    });
    const narrowed = (
      (await client.next((frame) => frame.id === 3)).result?.completions as
        | Array<{ value: string; label: string }>
        | undefined
    )?.[0];
    expect(narrowed).toMatchObject({ value: "/skill review ", label: "review" });

    // A bare `/rev` completes the skill SHORTHAND — `/review` is the same
    // invocation as `/skill review`, subcommand references included.
    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "complete",
      params: { text: "/rev" },
    });
    expect(
      (await client.next((frame) => frame.id === 4)).result?.completions,
    ).toEqual([
      { value: "/review ", label: "review", meta: "Review code with subcommands" },
      { value: "/review:security ", label: "review:security", meta: "Review code with subcommands" },
    ]);

    // Substring and description tiers work for shorthands too.
    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "complete",
      params: { text: "/bounty" },
    });
    expect(
      (await client.next((frame) => frame.id === 5)).result?.completions,
    ).toEqual([
      { value: "/read-project-and-hunt-bugs ", label: "read-project-and-hunt-bugs", meta: "Bug bounty hunting across the workspace" },
    ]);

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "complete",
      params: { text: "/skill zz" },
    });
    expect(
      (await client.next((frame) => frame.id === 6)).result?.completions,
    ).toEqual([]);
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
      context_limit: 0,
      input_tokens: 17,
      max_context: 0,
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
      context_max: 0,
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

test("daemon /usage keeps the session report when subscription quota is unavailable", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-usage-slash-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    projectDirectory: directory,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "protocol-model",
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  // Point every subscription endpoint at an unroutable address so the test
  // never depends on the network or on stored credentials.
  const previousEnvironment = { ...process.env };
  process.env.XERXES_CLAUDE_USAGE_URL = "http://127.0.0.1:1/unreachable";
  process.env.XERXES_CODEX_USAGE_URL = "http://127.0.0.1:1/unreachable";
  process.env.XERXES_KIMI_USAGE_URL = "http://127.0.0.1:1/unreachable";
  process.env.XERXES_ZAI_USAGE_URL = "http://127.0.0.1:1/unreachable";
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "usage-slash", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/usage" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({ ok: true });
    const notification = (await client.next(eventFrame("notification"))).params?.payload as
      | Record<string, unknown>
      | undefined;
    expect(notification?.category).toBe("slash");
    expect(String(notification?.body)).toContain("Model: protocol-model");
    expect(String(notification?.body)).toContain("Input tokens:");
  } finally {
    process.env = previousEnvironment;
    await client.close();
    await server.stop();
    await rm(directory, { force: true, recursive: true });
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
    cronLeasePath: join(directory, "cron.lease"),
    cronArchiveDirectory: join(directory, "cron", "archive"),
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
      params: { command: "/schedules" },
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
    // Isolated lease: a real daemon holding the shared one would legitimately
    // refuse this server's scheduler and the job would never fire.
    cronLeasePath: join(directory, "cron.lease"),
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
    return new Response('data: ' + JSON.stringify({ choices: [{ delta: { content: "durable compact summary" }, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } });
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
    // A background session-title request may precede the compaction call; the
    // compaction request is the one carrying the summarization prompt.
    const compactionRequest = requests.find((request) =>
      String(
        (request as { messages?: Array<{ content?: unknown }> })?.messages?.[0]
          ?.content ?? "",
      ).includes("CONTEXT TO SUMMARIZE"),
    );
    const prompt = String(
      (compactionRequest as { messages?: Array<{ content?: unknown }> } | undefined)
        ?.messages?.[0]?.content ?? "",
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

test("compaction archives the transcript it replaces beside the session file", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-compact-archive-"));
  const sessions = join(directory, "sessions");
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
    sessionDirectory: sessions,
  });
  const server = new DaemonServer({
    socketPath,
    runtime,
    profileStore,
    sessionArchiveDirectory: sessions,
  });
  const nativeFetch = globalThis.fetch;
  const summaryFetch: FetchImplementation = async () =>
    new Response('data: ' + JSON.stringify({ choices: [{ delta: { content: "archived summary" }, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } });
  globalThis.fetch = summaryFetch as typeof globalThis.fetch;
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "compact-archive", project_dir: directory },
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
    const session = runtime.sessionStatus("compact-archive");
    const contextControls = { version: 1, revision: 2, pins: [{ scope: 'project', path: 'MEMORY.md', content: 'Keep this fact' }], excluded: [{ scope: 'global', path: 'USER.md' }] };
    session!.metadata.context_controls = contextControls;
    const before = JSON.stringify(session?.messages ?? []);

    client.send({ jsonrpc: "2.0", id: 10, method: "session.compress", params: {} });
    const result = (await client.next((frame) => frame.id === 10)).result;
    expect(result).toMatchObject({ ok: true, compacted: true });

    const stamp = session?.metadata.last_compaction as Record<string, unknown>;
    expect(stamp).toMatchObject({
      messages_summarized: expect.any(Number),
      reason: "compact",
      tokens_after: expect.any(Number),
      tokens_before: expect.any(Number),
    });
    expect(Date.parse(String(stamp.compacted_at))).toBeGreaterThan(0);
    expect(session?.metadata.compaction_history).toEqual([stamp]);
    const persisted = JSON.parse(await readFile(join(sessions, session!.id + '.json'), 'utf8'));
    expect(persisted.metadata.compaction_history).toEqual([stamp]);
    expect(persisted.metadata.context_controls).toEqual(contextControls);
    expect(session?.metadata.context_controls).toEqual(contextControls);
    client.send({ jsonrpc: '2.0', id: 11, method: 'context.inspect', params: { section: 'compaction' } });
    const history = (await client.next(frame => frame.id === 11)).result;
    expect(history).toMatchObject({ ok: true, section: 'compaction', entries: [{ estimated_tokens: 0, text: expect.stringContaining('compact') }] });


    // The pre-compaction transcript survives the swap that dropped it from the
    // session and from the single per-session JSON.
    const archivePath = String(stamp.archive_path);
    expect(archivePath).toBe(join(sessions, `${session?.id}.precompact.jsonl`));
    const record = JSON.parse((await readFile(archivePath, "utf8")).trim()) as {
      readonly messages: readonly Record<string, unknown>[];
    };
    expect(JSON.stringify(record.messages)).toBe(before);
    expect(JSON.stringify(session?.messages ?? [])).toContain("archived summary");
  } finally {
    globalThis.fetch = nativeFetch;
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("undo rejects while another connection is compacting the same session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-undo-compact-race-"));
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
    socketPath,
    runtime,
    profileStore,
    autoCompactThreshold: 0,
  });
  const nativeFetch = globalThis.fetch;
  const summaryStarted = Promise.withResolvers<void>();
  const releaseSummary = Promise.withResolvers<void>();
  const summaryFetch: FetchImplementation = async () => {
    summaryStarted.resolve();
    await releaseSummary.promise;
    return new Response('data: ' + JSON.stringify({ choices: [{ delta: { content: "gated summary" }, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } });
  };
  globalThis.fetch = summaryFetch as typeof globalThis.fetch;
  await server.start();
  const compacting = await SocketTestClient.connect(socketPath);
  const mutating = await SocketTestClient.connect(socketPath);
  try {
    for (const [client, id] of [[compacting, 1], [mutating, 2]] as const) {
      client.send({
        jsonrpc: "2.0",
        id,
        method: "initialize",
        params: { session_key: "undo-compact-race", project_dir: directory },
      });
      await client.next((frame) => frame.id === id);
      await client.next(eventFrame("init_done"));
      await client.next(eventFrame("status_update"));
    }
    for (let i = 0; i < 4; i += 1) {
      compacting.send({
        jsonrpc: "2.0",
        id: 10 + i,
        method: "turn.submit",
        params: { text: `message ${i + 1}` },
      });
      await compacting.next((frame) => frame.id === 10 + i);
      await compacting.next(eventFrame("turn_begin"));
      await compacting.next(eventFrame("text_part"));
      await compacting.next(eventFrame("turn_end"));
    }
    const session = runtime.sessionStatus("undo-compact-race")!;
    const before = structuredClone(session.messages);

    compacting.send({ jsonrpc: "2.0", id: 20, method: "session.compress", params: {} });
    await summaryStarted.promise;
    // These reads use the very same socket as the blocked compaction.
    compacting.send({ jsonrpc: '2.0', id: 30, method: 'runtime.status', params: {} });
    compacting.send({ jsonrpc: '2.0', id: 31, method: 'schedule.list', params: {} });
    expect((await compacting.next(frame => frame.id === 30)).result?.ok).toBe(true);
    expect((await compacting.next(frame => frame.id === 31)).result).toBeDefined();
    mutating.send({ jsonrpc: "2.0", id: 21, method: "session.undo", params: {} });
    expect((await mutating.next((frame) => frame.id === 21)).result).toEqual({
      ok: false,
      error: "session operation in progress",
    });
    expect(session.messages).toEqual(before);

    releaseSummary.resolve();
    expect((await compacting.next((frame) => frame.id === 20)).result).toMatchObject({
      ok: true,
      compacted: true,
    });
  } finally {
    releaseSummary.resolve();
    globalThis.fetch = nativeFetch;
    compacting.close();
    mutating.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("retry rejects while another connection has a turn pending behind compaction", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-retry-turn-race-"));
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
    socketPath,
    runtime,
    profileStore,
    autoCompactThreshold: 0,
  });
  const nativeFetch = globalThis.fetch;
  const summaryStarted = Promise.withResolvers<void>();
  const releaseSummary = Promise.withResolvers<void>();
  const summaryFetch: FetchImplementation = async () => {
    summaryStarted.resolve();
    await releaseSummary.promise;
    return new Response('data: ' + JSON.stringify({ choices: [{ delta: { content: "gated summary" }, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } });
  };
  globalThis.fetch = summaryFetch as typeof globalThis.fetch;
  await server.start();
  const compacting = await SocketTestClient.connect(socketPath);
  const turning = await SocketTestClient.connect(socketPath);
  const mutating = await SocketTestClient.connect(socketPath);
  try {
    for (const [client, id] of [[compacting, 1], [turning, 2], [mutating, 3]] as const) {
      client.send({
        jsonrpc: "2.0",
        id,
        method: "initialize",
        params: { session_key: "retry-turn-race", project_dir: directory },
      });
      await client.next((frame) => frame.id === id);
      await client.next(eventFrame("init_done"));
      await client.next(eventFrame("status_update"));
    }
    for (let i = 0; i < 4; i += 1) {
      compacting.send({
        jsonrpc: "2.0",
        id: 10 + i,
        method: "turn.submit",
        params: { text: `message ${i + 1}` },
      });
      await compacting.next((frame) => frame.id === 10 + i);
      await compacting.next(eventFrame("turn_begin"));
      await compacting.next(eventFrame("text_part"));
      await compacting.next(eventFrame("turn_end"));
    }
    const session = runtime.sessionStatus("retry-turn-race")!;
    const before = structuredClone(session.messages);

    compacting.send({ jsonrpc: "2.0", id: 20, method: "session.compress", params: {} });
    await summaryStarted.promise;
    turning.send({
      jsonrpc: "2.0",
      id: 21,
      method: "turn.submit",
      params: { text: "pending after compaction" },
    });
    mutating.send({
      jsonrpc: "2.0",
      id: 22,
      method: "slash",
      params: { command: "/retry" },
    });
    expect((await mutating.next((frame) => frame.id === 22)).result).toEqual({
      ok: false,
      error: "session operation in progress",
    });
    expect(session.messages).toEqual(before);

    releaseSummary.resolve();
    expect((await compacting.next((frame) => frame.id === 20)).result).toMatchObject({
      ok: true,
      compacted: true,
    });
    expect((await turning.next((frame) => frame.id === 21)).result).toMatchObject({ ok: true });
    await turning.next(eventFrame("turn_begin"));
    await turning.next(eventFrame("text_part"));
    await turning.next(eventFrame("turn_end"));
  } finally {
    releaseSummary.resolve();
    globalThis.fetch = nativeFetch;
    compacting.close();
    turning.close();
    mutating.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("compaction writes no archive when the daemon's transcripts are elsewhere", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-compact-noarchive-"));
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
  // No sessionArchiveDirectory: the server falls back to the daemon home, which
  // holds no transcript for this session, so it must not leave an orphan
  // archive next to nothing.
  const server = new DaemonServer({ socketPath, runtime, profileStore });
  const nativeFetch = globalThis.fetch;
  const summaryFetch: FetchImplementation = async () =>
    new Response('data: ' + JSON.stringify({ choices: [{ delta: { content: "unarchived summary" }, finish_reason: 'stop' }] }) + '\n\ndata: [DONE]\n\n', { headers: { 'content-type': 'text/event-stream' } });
  globalThis.fetch = summaryFetch as typeof globalThis.fetch;
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "compact-noarchive", project_dir: directory },
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

    client.send({ jsonrpc: "2.0", id: 10, method: "session.compress", params: {} });
    const result = (await client.next((frame) => frame.id === 10)).result;
    expect(result).toMatchObject({ ok: true, compacted: true });
    expect(result?.archive_path).toBeUndefined();
    const stamp = runtime.sessionStatus("compact-noarchive")?.metadata
      .last_compaction as Record<string, unknown>;
    expect(stamp.archive_path).toBeUndefined();
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

test("daemon persists /model selection to the active provider profile", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-model-persist-"));
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  profileStore.save({
    name: "main",
    apiKey: "fake-api-key",
    baseUrl: "https://api.openai.test",
    model: "old-model",
    provider: "openai",
    setActive: true,
  });
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "old-model",
      sessionDirectory: join(directory, "sessions"),
    }),
    profileStore,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "model-persist", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/model kimi-k3" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject(
      { ok: true, model: "kimi-k3" },
    );
    // A TUI/daemon restart resolves the model from the profile store, so the
    // selection must be durable there, not only in runtime memory.
    expect(profileStore.active()?.model).toBe("kimi-k3");
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon replays persisted thinking traces on resume", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-replay-thinking-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "replay-model",
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
      params: { session_key: "think-origin", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    const session = runtime.sessionStatus("think-origin");
    if (!session) throw new Error("expected live session");
    session.messages.push(
      { role: "user", content: "question" },
      { role: "assistant", content: "answer", thinking: "reasoning trace" },
    );
    const sessionId = session.id;
    await runtime.flushSessions();

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "initialize",
      params: {
        resume_session_id: sessionId,
        session_key: "ignored-slot",
        project_dir: directory,
      },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    const replayedAssistant = await client.next(
      (frame) =>
        frame.method === "event" &&
        frame.params?.type === "notification" &&
        frame.params.payload?.type === "replay_assistant",
    );
    const notification = replayedAssistant.params?.payload as Record<string, unknown>;
    expect(notification.body).toBe("answer");
    expect((notification.payload as Record<string, unknown>).thinking).toBe(
      "reasoning trace",
    );
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon replays persisted tool executions interleaved on resume", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-replay-tools-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "replay-model",
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
      params: { session_key: "tool-origin", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    const session = runtime.sessionStatus("tool-origin");
    if (!session) throw new Error("expected live session");
    session.messages.push(
      { role: "user", content: "inspect auth" },
      {
        role: "assistant",
        content: "Let me read the file.",
        thinking: "reasoning trace",
        tool_calls: [
          {
            id: "call_1",
            type: "function",
            function: { name: "ReadFile", arguments: { path: "src/auth.ts" } },
          },
        ],
      },
      {
        role: "tool",
        content: "file body",
        name: "ReadFile",
        tool_call_id: "call_1",
      },
      { role: "assistant", content: "The flow starts in auth.ts." },
    );
    session.toolExecutions.push(
      {
        name: "ReadFile",
        result: "file body",
        return_value: "file body",
        permitted: true,
        tool_call_id: "call_1",
        duration_ms: 250,
        display_blocks: [],
      },
      // Unmatched execution (call evicted from retained messages): replays in
      // recorded order after the interleaved rows.
      {
        name: "GrepTool",
        result: "3 matches",
        return_value: "3 matches",
        permitted: true,
        tool_call_id: "call_orphaned",
        duration_ms: 40,
        display_blocks: [],
      },
    );
    const sessionId = session.id;
    await runtime.flushSessions();

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "initialize",
      params: {
        resume_session_id: sessionId,
        session_key: "ignored-slot",
        project_dir: directory,
      },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    const historyType = (frame: Frame): string | undefined => {
      if (frame.method !== "event" || frame.params?.type !== "notification") {
        return undefined;
      }
      const payload = frame.params.payload as
        | { category?: unknown; type?: unknown }
        | undefined;
      return payload?.category === "history" && typeof payload.type === "string"
        ? payload.type
        : undefined;
    };

    const replayUser = await client.next(
      (frame) => historyType(frame) === "replay_user",
    );
    expect(
      (replayUser.params?.payload as Record<string, unknown>).body,
    ).toBe("✨ inspect auth");

    const replayAssistant = await client.next(
      (frame) => historyType(frame) === "replay_assistant",
    );
    const assistantNotification = replayAssistant.params
      ?.payload as Record<string, unknown>;
    expect(assistantNotification.body).toBe("Let me read the file.");
    expect(
      (assistantNotification.payload as Record<string, unknown>).thinking,
    ).toBe("reasoning trace");

    // The tool row lands right after the assistant message that requested it.
    const replayTool = await client.next(
      (frame) => historyType(frame) === "replay_tool",
    );
    const toolNotification = replayTool.params?.payload as Record<
      string,
      unknown
    >;
    expect(toolNotification.body).toBe("✓ ReadFile");
    expect(toolNotification.payload).toMatchObject({
      name: "ReadFile",
      ok: true,
      duration_ms: 250,
    });
    expect(
      (toolNotification.payload as Record<string, unknown>).context,
    ).toBe('{"path":"src/auth.ts"}');

    const replayFinal = await client.next(
      (frame) => historyType(frame) === "replay_assistant",
    );
    expect(
      (replayFinal.params?.payload as Record<string, unknown>).body,
    ).toBe("The flow starts in auth.ts.");

    const replayOrphan = await client.next(
      (frame) => historyType(frame) === "replay_tool",
    );
    expect(
      (replayOrphan.params?.payload as Record<string, unknown>).payload,
    ).toMatchObject({ name: "GrepTool", ok: true, duration_ms: 40 });

    const resumed = await client.next(
      (frame) => historyType(frame) === "resumed",
    );
    expect(
      (resumed.params?.payload as Record<string, unknown>).body,
    ).toContain("resumed session");
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

    client.send({ jsonrpc: "2.0", id: 60, method: "slash", params: { command: `/rollback diff ${firstId}` } });
    const preview = (await client.next(frame => frame.id === 60)).result;
    expect(preview).toMatchObject({ ok: true, snapshot_id: firstId, truncated: false });
    expect(preview?.diff).toContain("-third");
    expect(preview?.diff).toContain("+first");
    expect(await Bun.file(sourcePath).text()).toBe("third");
    expect((await client.next(eventFrame("notification"))).params?.payload).toMatchObject({ category: "slash", body: expect.stringContaining("Restore preview") });

    client.send({ jsonrpc: "2.0", id: 70, method: "snapshot.list", params: {} });
    expect((await client.next(frame => frame.id === 70)).result).toMatchObject({ ok: true, snapshots: [{ id: firstId }, {}] });
    client.send({ jsonrpc: "2.0", id: 71, method: "snapshot.preview", params: { snapshot_id: firstId } });
    expect((await client.next(frame => frame.id === 71)).result).toMatchObject({ ok: true, snapshot_id: firstId, revision: preview?.revision, diff: preview?.diff });
    client.send({ jsonrpc: "2.0", id: 72, method: "snapshot.preview", params: { snapshot_id: 10 } });
    expect((await client.next(frame => frame.id === 72)).result).toMatchObject({ ok: false });
    client.send({ jsonrpc: "2.0", id: 73, method: "snapshot.preview", params: { snapshot_id: "missing" } });
    expect((await client.next(frame => frame.id === 73)).result).toMatchObject({ ok: false, error: expect.stringContaining("not found") });

    client.send({ jsonrpc: "2.0", id: 74, method: "snapshot.preview", params: { snapshot_id: firstId, path: "state.txt" } });
    const filePreview = (await client.next(frame => frame.id === 74)).result;
    expect(filePreview).toMatchObject({ ok: true, action: "restore", files: ["state.txt"] });
    client.send({ jsonrpc: "2.0", id: 75, method: "snapshot.restoreFile", params: { snapshot_id: firstId, path: "state.txt", revision: filePreview?.revision } });
    expect((await client.next(frame => frame.id === 75)).result).toMatchObject({ ok: true, path: "state.txt" });
    await client.next(eventFrame("notification"));
    expect(await Bun.file(sourcePath).text()).toBe("first");
    client.send({ jsonrpc: "2.0", id: 76, method: "snapshot.restoreFile", params: { snapshot_id: firstId, path: "state.txt" } });
    expect((await client.next(frame => frame.id === 76)).result).toMatchObject({ ok: false });

    await writeFile(sourcePath, "concurrent", "utf8");
    client.send({ jsonrpc: "2.0", id: 61, method: "slash", params: { command: `/rollback apply ${firstId} ${preview?.revision}` } });
    expect((await client.next(frame => frame.id === 61)).result).toMatchObject({ ok: false, error: expect.stringContaining("preview is stale") });
    await client.next(eventFrame("notification"));
    expect(await Bun.file(sourcePath).text()).toBe("concurrent");
    await writeFile(sourcePath, "third", "utf8");
    client.send({ jsonrpc: "2.0", id: 62, method: "slash", params: { command: `/rollback apply ${firstId} ${preview?.revision}` } });
    expect((await client.next(frame => frame.id === 62)).result).toMatchObject({ ok: true });
    await client.next(eventFrame("notification"));
    expect(await Bun.file(sourcePath).text()).toBe("first");

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
      body: "Usage: `/rollback <snapshot-id> [path]` — list with `/snapshots`.",
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
          // Provisional title seeded from the opening prompt: a saved chat is
          // never listed as a nameless dash while it waits on (or never gets)
          // a model-written title.
          title: "Saved question",
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
      daemon_version: "0.5.0",
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
    ).toMatchObject({ profile_name: 'native' });
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
    let inactiveUnavailable = false;
    const mockFetch: FetchImplementation = async (input, init) => {
      const request = {
        authorization: new Headers(init?.headers).get("authorization"),
        url: String(input),
      };
      modelRequests.push(request);
      if (request.url.includes("failure.example")
        || (inactiveUnavailable && request.url.includes("inactive.example"))) {
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
      catalog: [{ id: "inactive-remote-model" }],
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
      id: 110,
      method: "provider_model_override",
      params: {
        profile_name: "inactive",
        model: "inactive-remote-model",
        context_limit: 400_000,
        max_output_tokens: 80_000,
      },
    });
    expect((await client.next((frame) => frame.id === 110)).result).toEqual({
      ok: true,
      model: {
        id: "inactive-remote-model",
        context_limit: 400_000,
        context_source: "override",
        max_output_tokens: 80_000,
        output_source: "override",
        overridden: true,
      },
    });
    client.send({
      jsonrpc: "2.0",
      id: 111,
      method: "provider_models",
      params: { profile_name: "inactive" },
    });
    expect((await client.next((frame) => frame.id === 111)).result).toMatchObject({
      ok: true,
      catalog: [{
        id: "inactive-remote-model",
        context_limit: 400_000,
        max_output_tokens: 80_000,
        overridden: true,
      }],
    });
    expect(profileStore.get("inactive")?.model_overrides).toEqual({
      "inactive-remote-model": { context_limit: 400_000, max_output_tokens: 80_000 },
    });
    client.send({
      jsonrpc: "2.0",
      id: 112,
      method: "provider_model_override",
      params: { profile_name: "inactive", model: "inactive-remote-model", context_limit: 0 },
    });
    expect((await client.next((frame) => frame.id === 112)).result).toEqual({
      ok: false,
      error: "context_limit must be a positive safe integer or null",
    });
    client.send({
      jsonrpc: "2.0",
      id: 114,
      method: "provider_model_override",
      params: { profile_name: "inactive", model: "not-cached", context_limit: 100_000 },
    });
    expect((await client.next((frame) => frame.id === 114)).result).toEqual({
      ok: false,
      error: "No cached model named not-cached for profile inactive",
    });
    client.send({
      jsonrpc: "2.0",
      id: 113,
      method: "provider_model_override",
      params: {
        profile_name: "inactive",
        model: "inactive-remote-model",
        context_limit: null,
        max_output_tokens: null,
      },
    });
    expect((await client.next((frame) => frame.id === 113)).result).toEqual({
      ok: true,
      model: { id: "inactive-remote-model" },
    });
    expect(profileStore.get("inactive")?.model_overrides).toEqual({});
    inactiveUnavailable = true;
    client.send({
      jsonrpc: "2.0",
      id: 115,
      method: "provider_models",
      params: { profile_name: "inactive" },
    });
    expect((await client.next((frame) => frame.id === 115)).result).toMatchObject({
      ok: true,
      models: ["inactive-saved-model", "inactive-remote-model"],
      source: "profile",
      warning: expect.stringContaining("[redacted]"),
    });
    inactiveUnavailable = false;

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
    // `runtime.status` is the daemon-wide default for sessions opened later.
    // /yolo pins the session it was typed in, so the default is deliberately
    // unchanged — that separation is what lets two sessions run at different
    // trust levels. The session's own mode is reported on status_update above.
    expect(
      (await client.next((frame) => frame.id === 3)).result?.permission_mode,
    ).toBe("accept-all");

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
      "Pick thinking effort from the levels this model supports",
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
    // `levels` reports what this model actually accepts, so a caller that is
    // not the picker still learns the valid set instead of guessing.
    expect((await client.next((frame) => frame.id === 10)).result).toEqual({
      ok: true,
      reasoning_effort: "high",
      levels: ["off", "low", "medium", "high"],
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
    ).toMatchObject({ text: "approval:approve" });

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
    ).toMatchObject({ text: "answer:yes" });
    await client.next(eventFrame("turn_end"));
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test('connection lease preserves a waiting turn and replays only pending interactions to its owner', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-lease-'));
  const socketPath = join(directory, 'daemon.sock');
  const interactions = new DaemonInteractionBoard();
  const runtime = new InMemoryDaemonRuntime(new ReplyRunner(interactions), {
    currentProjectDirectory: directory, interactions, model: 'reply-model', sessionDirectory: join(directory, 'sessions'),
  });
  const server = new DaemonServer({ socketPath, runtime, interactions });
  await server.start();
  const clients: SocketTestClient[] = [];
  let id = 0;
  const rpc = async (client: SocketTestClient, method: string, params: Record<string, unknown> = {}) => {
    const requestId = ++id;
    client.send({ jsonrpc: '2.0', id: requestId, method, params });
    return client.next(frame => frame.id === requestId);
  };
  const connectClient = async () => { const client = await SocketTestClient.connect(socketPath); clients.push(client); return client; };
  try {
    const first = await connectClient();
    const initialized = await rpc(first, 'initialize', { session_key: 'leased', project_dir: directory });
    expect(initialized.result?.connection_lease_supported).toBe(true);
    const sessionId = initialized.result?.session_id;
    const leased = await rpc(first, 'connection.lease');
    const token = leased.result?.token;
    expect(typeof token).toBe('string');
    await rpc(first, 'turn.submit', { text: 'permission then question' });
    await first.next(eventFrame('approval_request'));
    first.close();
    await Bun.sleep(30);
    expect(interactions.pendingPermissionIds()).toEqual(['approval-1']);
    expect(runtime.sessionStatus('leased')?.cancelRequested).toBe(false);

    const stranger = await connectClient();
    await rpc(stranger, 'initialize', { session_key: 'unrelated' });
    expect((await rpc(stranger, 'connection.lease', { token, project_dir: join(directory, 'different') })).error).toBeDefined();
    expect((await rpc(stranger, 'permission_response', { request_id: 'approval-1', response: 'approve' })).result?.ok).toBe(false);

    const second = await connectClient();
    const resumedLease = await rpc(second, 'connection.lease', { token, project_dir: directory });
    expect(resumedLease.error).toBeUndefined();
    expect(resumedLease.result?.ok).toBe(true);
    const reopened = await rpc(second, 'initialize', { resume_session_id: sessionId });
    expect(reopened.error).toBeUndefined();
    expect(reopened.result?.pending_interactions).toMatchObject([{ type: 'approval_request', payload: { id: 'approval-1' } }]);
    expect((await rpc(second, 'permission_response', { request_id: 'approval-1', response: 'approve' })).result?.ok).toBe(true);
    const question = await second.next(eventFrame('question_request'));
    second.close();
    await Bun.sleep(30);

    const third = await connectClient();
    expect((await rpc(third, 'connection.lease', { token, project_dir: directory })).result?.ok).toBe(true);
    const restoredQuestion = await rpc(third, 'initialize', { resume_session_id: sessionId });
    expect(restoredQuestion.result?.pending_interactions).toMatchObject([{ type: 'question_request', payload: { id: question.params?.payload?.id } }]);
    expect(third.seen(eventFrame('approval_request'))).toBe(false);
    expect((await rpc(third, 'question_response', { request_id: question.params?.payload?.id, answers: { answer: 'yes' } })).result?.ok).toBe(true);
    await third.next(eventFrame('turn_end'));
    expect(runtime.sessionStatus('leased')?.cancelRequested).toBe(false);
  } finally {
    clients.forEach(client => client.close());
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
    ).toMatchObject({ text: "waiting for steer" });
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
    ).toMatchObject({ text: "steer:focus tests" });
    await client.next(eventFrame("turn_end"));
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("turn events keep their session identity after the connection switches sessions", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-turn-session-route-"));
  const socketPath = join(directory, "daemon.sock");
  const runner = new SteerRunner();
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(runner, {
      currentProjectDirectory: directory,
      model: "session-route-model",
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
      params: { session_key: "session-a" },
    });
    const initializedA = await client.next((frame) => frame.id === 1);
    const sessionA = String(
      (initializedA.result?.session as { id?: unknown } | undefined)?.id ?? "",
    );
    expect(sessionA).not.toBe("");
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "session-a", text: "keep streaming" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    expect(
      (await client.next(eventFrame("text_part"))).params?.payload,
    ).toMatchObject({ session_id: sessionA, text: "waiting for steer" });

    // The same TUI socket can activate another live session while session A
    // remains in flight. Late A events must not inherit the connection's new
    // active session or arrive without a routing identity.
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "initialize",
      params: { session_key: "session-b" },
    });
    const initializedB = await client.next((frame) => frame.id === 3);
    const sessionB = String(
      (initializedB.result?.session as { id?: unknown } | undefined)?.id ?? "",
    );
    expect(sessionB).not.toBe("");
    expect(sessionB).not.toBe(sessionA);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    runner.release();
    expect(
      (await client.next(eventFrame("text_part"))).params?.payload,
    ).toMatchObject({ session_id: sessionA, text: "steer:" });
    expect(
      (await client.next(eventFrame("turn_end"))).params?.payload,
    ).toMatchObject({ cancelled: false, session_id: sessionA });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("set_mode targets its explicit session without changing the active connection session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-mode-target-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ socketPath, runtime });
  await runtime.openSession("target-session");
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "active-session" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "set_mode",
      params: { mode: "researcher", session_key: "target-session" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({
      ok: true,
      mode: "researcher",
      plan_mode: false,
    });
    expect(runtime.sessionStatus("target-session")?.interactionMode).toBe("researcher");
    expect(runtime.sessionStatus("active-session")?.interactionMode).toBe("code");
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("mid-turn steer and mode changes never cancel the active turn", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-steer-mode-"));
  const socketPath = join(directory, "daemon.sock");
  const modeChanges: Array<{ id: string; mode: string }> = [];
  const runner = new SteerRunner();
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(runner, {
      currentProjectDirectory: directory,
      model: "steer-model",
      onSessionModeChange: (id, mode) => {
        modeChanges.push({ id, mode });
      },
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
      params: { session_key: "steer-mode" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "steer-mode", text: "start" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    await client.next(eventFrame("text_part"));

    // A mid-turn user message steers the live turn instead of cancelling it.
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "steer",
      params: { session_key: "steer-mode", content: "keep going" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
    });
    await client.next(eventFrame("steer_input"));

    // Interaction-mode changes mid-turn re-scope future turns only; they
    // must not cancel the running turn (or any subagents it owns).
    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "set_mode",
      params: { mode: "plan" },
    });
    expect(
      (await client.next((frame) => frame.id === 4)).result,
    ).toMatchObject({ ok: true, mode: "plan", plan_mode: true });
    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "set_plan_mode",
      params: { enabled: false },
    });
    expect(
      (await client.next((frame) => frame.id === 5)).result,
    ).toMatchObject({ ok: true, plan_mode: false });

    runner.release();
    expect(
      (await client.next(eventFrame("text_part"))).params?.payload,
    ).toMatchObject({ text: "steer:keep going" });
    const turnEnd = await client.next(eventFrame("turn_end"));
    expect(turnEnd.params?.payload?.cancelled).toBe(false);
    expect(modeChanges.map((change) => change.mode)).toEqual([
      "plan",
      "code",
    ]);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("an explicit cancel still stops the owning turn and only that turn", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-cancel-scope-"));
  const socketPath = join(directory, "daemon.sock");
  const first = new SteerRunner();
  const second = new SteerRunner();
  const runners = new Map([
    ["cancel-a", first],
    ["cancel-b", second],
  ]);
  const runner: TurnRunner = {
    async *run(session, text, signal, controls) {
      const delegated = runners.get(session.sessionKey) ?? first;
      yield* delegated.run(session, text, signal, controls);
    },
  };
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(runner, {
      currentProjectDirectory: directory,
      model: "cancel-model",
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const clientA = await SocketTestClient.connect(socketPath);
  const clientB = await SocketTestClient.connect(socketPath);
  try {
    for (const [key, client] of [
      ["cancel-a", clientA],
      ["cancel-b", clientB],
    ] as const) {
      client.send({
        jsonrpc: "2.0",
        id: 1,
        method: "initialize",
        params: { session_key: key },
      });
      await client.next((frame) => frame.id === 1);
      await client.next(eventFrame("init_done"));
      await client.next(eventFrame("status_update"));
      client.send({
        jsonrpc: "2.0",
        id: 2,
        method: "turn.submit",
        params: { session_key: key, text: "start" },
      });
      await client.next((frame) => frame.id === 2);
      await client.next(eventFrame("turn_begin"));
      await client.next(eventFrame("text_part"));
    }

    // Cancelling session A's turn must leave session B's turn running.
    clientA.send({
      jsonrpc: "2.0",
      id: 3,
      method: "cancel",
      params: { session_key: "cancel-a" },
    });
    expect((await clientA.next((frame) => frame.id === 3)).result).toEqual({
      ok: true,
    });
    first.release();
    const cancelledEnd = await clientA.next(eventFrame("turn_end"));
    expect(cancelledEnd.params?.payload?.cancelled).toBe(true);
    // A cancel that landed mid-turn is NOT unstarted — assistant content
    // existed (turn_begin was emitted), so the additive field stays absent.
    expect(cancelledEnd.params?.payload?.unstarted).toBeUndefined();

    second.release();
    expect(
      (await clientB.next(eventFrame("text_part"))).params?.payload,
    ).toMatchObject({ text: "steer:" });
    const survivingEnd = await clientB.next(eventFrame("turn_end"));
    expect(survivingEnd.params?.payload?.cancelled).toBe(false);
  } finally {
    clientA.close();
    clientB.close();
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

test("a turn aborted during admission setup still delivers exactly one terminal cancel event", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-setup-abort-"));
  const hexId = "deadbeef01";
  const releaseCleanup = Promise.withResolvers<void>();
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "setup-abort-model",
    sessionDirectory: join(directory, "sessions"),
    backgroundCommands: {
      disposeAll: async () => {},
      disposeOwner: async () => {
        // Park openSession mid-setup: initializeSession awaits this before
        // the session can be registered.
        await releaseCleanup.promise;
      },
    },
  });
  // Queue a pending owner cleanup for this id so submitTurn's openSession
  // blocks after admission but before the turn launches.
  runtime.evictSession(hexId);
  const events: DaemonEvent[] = [];
  try {
    const submitting = runtime.submitTurn(hexId, "hello", (event) =>
      events.push(event),
    );
    await Bun.sleep(10);
    expect(runtime.cancelTurn(hexId)).toBe(true);
    releaseCleanup.resolve();
    await submitting;

    const terminal = events.filter((event) => event.type === "turn_end");
    expect(terminal).toHaveLength(1);
    expect(terminal[0]?.payload).toMatchObject({
      cancelled: true,
      unstarted: true,
      session_id: hexId,
    });
    // Nothing ran: no turn_begin, no text, no transcript growth.
    expect(events.some((event) => event.type === "turn_begin")).toBe(false);
    expect(runtime.sessionStatus(hexId)?.messages ?? []).toHaveLength(0);

    // The controller was released with the terminal event, so the session is
    // immediately usable again.
    const followUp: DaemonEvent[] = [];
    await runtime.submitTurn(hexId, "again", (event) => followUp.push(event));
    expect(followUp.some((event) => event.type === "turn_end")).toBe(true);
  } finally {
    releaseCleanup.resolve();
    await rm(directory, { recursive: true, force: true });
  }
});

test("initialize re-checks admitted turns across its flush and leaves them running", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-init-race-"));
  const socketPath = join(directory, "daemon.sock");
  const store = new GatedSaveStore({
    currentProjectDirectory: directory,
    directory: join(directory, "sessions"),
  });
  const runner = new GatedRunner();
  const server = new DaemonServer({
    socketPath,
    autoTitle: false,
    runtime: new InMemoryDaemonRuntime(runner, {
      currentProjectDirectory: directory,
      model: "race-model",
      transcriptStore: store,
    }),
  });
  await server.start();
  const initializer = await SocketTestClient.connect(socketPath);
  const submitter = await SocketTestClient.connect(socketPath);
  try {
    initializer.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "evict-race" },
    });
    await initializer.next((frame) => frame.id === 1);
    await initializer.next(eventFrame("init_done"));
    await initializer.next(eventFrame("status_update"));

    // Seed a completed exchange first. Sessions with no completed exchange
    // never reach the transcript store anymore — they must not become
    // phantom session rows — and the race below needs a genuine save to park
    // the gate on.
    initializer.send({
      jsonrpc: "2.0",
      id: 90,
      method: "turn.submit",
      params: { session_key: "evict-race", text: "seed history" },
    });
    // The gated runner holds the turn open until released, and turn.submit
    // only settles at turn end — so release first, then await the settle.
    runner.release();
    await initializer.next((frame) => frame.id === 90);
    await initializer.next(eventFrame("turn_end"));

    // Block flushSessions inside the second initialize's eviction branch.
    store.armSaveGate();
    initializer.send({
      jsonrpc: "2.0",
      id: 2,
      method: "initialize",
      params: { session_key: "evict-race" },
    });
    await store.saveEntered.promise;

    // While that flush is parked, a turn is admitted for the same key.
    submitter.send({
      jsonrpc: "2.0",
      id: 3,
      method: "turn.submit",
      params: { session_key: "evict-race", text: "admitted mid-flush" },
    });
    await submitter.next(eventFrame("turn_begin"));

    // Release the flush: the post-await re-check must now see the live turn
    // and skip eviction instead of aborting it silently.
    store.releaseSave();
    expect(
      (await initializer.next((frame) => frame.id === 2)).result,
    ).toMatchObject({ ok: true });

    runner.release();
    const end = await submitter.next(eventFrame("turn_end"));
    expect(end.params?.payload).toMatchObject({
      cancelled: false,
      session_id: expect.any(String),
    });
  } finally {
    initializer.close();
    submitter.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("/new after /resume starts an empty session instead of re-adopting the transcript", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-new-after-resume-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "resume-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({ socketPath, runtime, autoTitle: false });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "origin" },
    });
    const firstInit = await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    const originalId = String(
      (firstInit.result?.session as { id?: unknown } | undefined)?.id ?? "",
    );
    expect(originalId).not.toBe("");

    const originSession = runtime.sessionStatus("origin");
    if (!originSession) throw new Error("expected live session");
    originSession.messages.push(
      { role: "user", content: "older question" },
      { role: "assistant", content: "older answer" },
    );
    await runtime.flushSessions();

    // Resume by id: the connection's active session key becomes the hex id.
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "initialize",
      params: { resume_session_id: originalId },
    });
    const resumed = await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    expect(resumed.result?.session).toMatchObject({
      id: originalId,
      messages: 2,
    });

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: "/new" },
    });
    const fresh = await client.next((frame) => frame.id === 3);
    await client.next(eventFrame("init_done"));
    expect(fresh.result?.ok).toBe(true);
    const freshSession = fresh.result?.session as Record<string, unknown>;
    const freshKey = String(freshSession.key ?? "");
    expect(freshSession.id).not.toBe(originalId);
    expect(Number(freshSession.message_count)).toBe(0);
    expect(freshKey.startsWith("tui:")).toBe(true);

    // The old live copy is gone and the prompt after /new lands in the new
    // session, not in the resumed history.
    expect(runtime.sessionStatus(originalId)).toBeUndefined();
    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "turn.submit",
      params: { text: "fresh hello" },
    });
    await client.next((frame) => frame.id === 4);
    await client.next(eventFrame("turn_begin"));
    await client.next(eventFrame("turn_end"));
    const active = runtime.sessionStatus(freshKey);
    expect(active?.id).not.toBe(originalId);

    // The persisted transcript of the resumed session is untouched.
    const persisted = await new DaemonTranscriptStore({
      currentProjectDirectory: directory,
      directory: join(directory, "sessions"),
    }).load(originalId);
    expect(persisted?.messages.map((message) => message.role)).toEqual([
      "user",
      "assistant",
    ]);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("concurrent opens of one persisted id fold into exactly one live session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-open-claim-"));
  const sessionId = "feedface01";
  const seedDirectory = join(directory, "seed-sessions");
  const seedRuntime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "claim-model",
    sessionDirectory: seedDirectory,
  });
  const seeded = await seedRuntime.openSession(sessionId);
  seeded.messages.push(
    { role: "user", content: "history one" },
    { role: "assistant", content: "history two" },
  );
  await seedRuntime.flushSessions();

  const releaseCleanup = Promise.withResolvers<void>();
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "claim-model",
    // Both slot keys resolve to the same persisted transcript.
    transcriptStore: new SameIdStore(sessionId, {
      currentProjectDirectory: directory,
      directory: seedDirectory,
    }),
    backgroundCommands: {
      disposeAll: async () => {},
      disposeOwner: async () => {
        await releaseCleanup.promise;
      },
    },
  });
  // Park the first opener between its synchronous claim and registration.
  runtime.evictSession(sessionId);
  try {
    const first = runtime.openSession("slot-a", undefined, { resume: true });
    await Bun.sleep(10);
    const second = runtime.openSession("slot-b", undefined, { resume: true });
    let secondError: unknown;
    let secondSettled = false;
    try {
      await second;
      secondSettled = true;
    } catch (error) {
      secondError = error;
    }
    expect(secondSettled).toBe(false);
    expect(secondError).toBeInstanceOf(ValidationError);
    expect(String(secondError instanceof Error ? secondError.message : secondError)).toMatch(
      /already being opened/,
    );

    releaseCleanup.resolve();
    const opened = await first;
    expect(opened.id).toBe(sessionId);
    expect(
      runtime.listSessions().filter((session) => session.id === sessionId),
    ).toHaveLength(1);

    // After the claim releases, a sequential reopen folds into the live
    // session instead of duplicating it.
    const third = await runtime.openSession("slot-b", undefined, {
      resume: true,
    });
    expect(third.id).toBe(sessionId);
    expect(
      runtime.listSessions().filter((session) => session.id === sessionId),
    ).toHaveLength(1);
  } finally {
    releaseCleanup.resolve();
    await rm(directory, { recursive: true, force: true });
  }
});

test("accepted submission ids stay bounded and are dropped when their session is evicted", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-submission-cap-"));
  const server = new DaemonServer({
    socketPath: join(directory, "daemon.sock"),
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  type Remember = (submissionKey: string) => void;
  type Forget = (sessionKeys: readonly string[]) => void;
  const internals = server as unknown as {
    acceptedSubmissionIds: Set<string>;
    forgetAcceptedSubmissions: Forget;
    rememberAcceptedSubmission: Remember;
  };
  try {
    for (let index = 0; index < MAX_ACCEPTED_SUBMISSION_IDS + 50; index += 1) {
      internals.rememberAcceptedSubmission(`key-${index}\u0000sub-${index}`);
    }
    expect(internals.acceptedSubmissionIds.size).toBe(MAX_ACCEPTED_SUBMISSION_IDS);
    // Oldest entries were evicted FIFO; the newest survive.
    expect(internals.acceptedSubmissionIds.has("key-0\u0000sub-0")).toBe(false);
    expect(
      internals.acceptedSubmissionIds.has(
        `key-${MAX_ACCEPTED_SUBMISSION_IDS + 49}\u0000sub-${MAX_ACCEPTED_SUBMISSION_IDS + 49}`,
      ),
    ).toBe(true);

    // Eviction drops only the evicted session's entries.
    internals.forgetAcceptedSubmissions(["key-100"]);
    expect(internals.acceptedSubmissionIds.has("key-100\u0000sub-100")).toBe(
      false,
    );
    expect(internals.acceptedSubmissionIds.has("key-101\u0000sub-101")).toBe(true);
    expect(internals.acceptedSubmissionIds.size).toBe(
      MAX_ACCEPTED_SUBMISSION_IDS - 1,
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("duplicate submission ids still short-circuit to an idempotent duplicate result", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-submission-dedupe-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    autoTitle: false,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "dedupe-model",
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "turn.submit",
      params: {
        session_key: "dedupe",
        text: "exactly once",
        submission_id: "client-submit-1",
      },
    });
    expect((await client.next((frame) => frame.id === 1)).result).toEqual({
      ok: true,
    });
    await client.next(eventFrame("turn_end"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: {
        session_key: "dedupe",
        text: "exactly once",
        submission_id: "client-submit-1",
      },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
      duplicate: true,
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

/** Title client that records the abort signal each completion received. */
class SignalCapturingTitleClient implements LlmClient {
  readonly signals: Array<AbortSignal | undefined> = [];

  async *stream(
    _request: CompletionRequest,
    signal?: AbortSignal,
  ): AsyncGenerator<LlmDelta> {
    this.signals.push(signal);
    yield { content: "Quiet Session Notes", usage: { inputTokens: 2, outputTokens: 2 } };
  }
}

test("title generation rides a session-scoped signal aborted when the session closes", async () => {
  resetTitleAttempts();
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-title-signal-"));
  const socketPath = join(directory, "daemon.sock");
  const titleClient = new SignalCapturingTitleClient();
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "title-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    socketPath,
    runtime,
    titleClientFactory: () => titleClient,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "titled" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "titled", text: "hello there" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_end"));

    // The background title call carries the session-lifetime signal.
    for (let waited = 0; waited < 4_000; waited += 25) {
      if (titleClient.signals.length > 0) break;
      await Bun.sleep(25);
    }
    expect(titleClient.signals).toHaveLength(1);
    expect(titleClient.signals[0]?.aborted).toBe(false);

    // The generated title lands on the session.
    for (let waited = 0; waited < 4_000; waited += 25) {
      if (runtime.sessionStatus("titled")?.metadata.title) break;
      await Bun.sleep(25);
    }
    expect(runtime.sessionStatus("titled")?.metadata.title).toBe(
      "Quiet Session Notes",
    );

    // Closing the session (a fresh initialize without resume evicts it)
    // aborts the signal so any still-open title work stops with it.
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "initialize",
      params: { session_key: "titled" },
    });
    await client.next((frame) => frame.id === 3);
    await client.next(eventFrame("init_done"));
    expect(titleClient.signals[0]?.aborted).toBe(true);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

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

/** Turn runner that parks mid-turn until released. */
class GatedRunner implements TurnRunner {
  private resolveGate: (() => void) | undefined;
  private readonly gate = new Promise<void>((resolve) => {
    this.resolveGate = resolve;
  });

  release(): void {
    this.resolveGate?.();
  }

  async *run(): AsyncGenerator<DaemonEvent> {
    yield { type: "text_part", payload: { text: "waiting" } };
    await this.gate;
    yield { type: "text_part", payload: { text: "done" } };
  }
}

/** Transcript store whose save() can be parked to hold flushSessions open. */
class GatedSaveStore extends DaemonTranscriptStore {
  private gate: PromiseWithResolvers<void> | undefined;
  /** Resolver kept out-of-band so releaseSave works after save consumes `gate`. */
  private parkedGate: PromiseWithResolvers<void> | undefined;
  saveEntered: PromiseWithResolvers<void> = Promise.withResolvers();

  armSaveGate(): void {
    this.parkedGate = Promise.withResolvers();
    this.gate = this.parkedGate;
    this.saveEntered = Promise.withResolvers();
  }

  releaseSave(): void {
    this.parkedGate?.resolve();
    this.parkedGate = undefined;
    this.gate = undefined;
  }

  override async save(
    transcript: Parameters<DaemonTranscriptStore["save"]>[0],
    options?: Parameters<DaemonTranscriptStore["save"]>[1],
  ): Promise<void> {
    if (this.gate) {
      const gate = this.gate;
      this.gate = undefined;
      this.saveEntered.resolve();
      await gate.promise;
    }
    return super.save(transcript, options);
  }
}

/** Transcript store that resolves every load against one persisted id. */
class SameIdStore extends DaemonTranscriptStore {
  constructor(
    private readonly targetSessionId: string,
    options: ConstructorParameters<typeof DaemonTranscriptStore>[0],
  ) {
    super(options);
  }

  override loadResult(
    _sessionKey: string,
    options?: Parameters<DaemonTranscriptStore["loadResult"]>[1],
  ): ReturnType<DaemonTranscriptStore["loadResult"]> {
    return super.loadResult(this.targetSessionId, options);
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
  predicate: () => boolean | Promise<boolean>,
  timeout = 2_000,
): Promise<void> {
  const deadline = Date.now() + timeout;
  while (!(await predicate())) {
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

  /** Whether any buffered frame matched, for asserting that nothing was emitted. */
  seen(predicate: (frame: Frame) => boolean): boolean {
    return this.frames.some(predicate);
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

    const files = (await readdir(sessionDirectory)).filter(file => file.endsWith(".json"));
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

test("daemon validates the size of each complete NDJSON frame instead of the aggregate receive chunk", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-frame-batch-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    maxSocketFrameBytes: 128,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
    socketPath,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.sendBatch([
      { jsonrpc: "2.0", id: 1, method: "runtime.status", params: {} },
      { jsonrpc: "2.0", id: 2, method: "runtime.status", params: {} },
    ]);
    expect((await client.next((frame) => frame.id === 1)).result).toMatchObject({ ok: true });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({ ok: true });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon drops a Unix client that exceeds the pending parsed-request limit", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-pending-limit-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    maxPendingSocketRequests: 1,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
    socketPath,
  });
  await server.start();
  const offender = connect({ path: socketPath });
  try {
    await new Promise<void>((resolve, reject) => {
      offender.once("connect", resolve);
      offender.once("error", reject);
    });
    const closed = new Promise<void>((resolve) => offender.once("close", () => resolve()));
    offender.write(
      Array.from({ length: 3 }, (_, index) =>
        `${JSON.stringify({ jsonrpc: "2.0", id: index + 1, method: "runtime.status", params: {} })}\n`,
      ).join(""),
    );
    await closed;
  } finally {
    offender.destroy();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon bounds buffered Unix socket output and disconnects a slow client", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-output-limit-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    maxSocketOutputBytes: 32,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
    socketPath,
  });
  await server.start();
  const offender = connect({ path: socketPath });
  try {
    await new Promise<void>((resolve, reject) => {
      offender.once("connect", resolve);
      offender.once("error", reject);
    });
    const closed = new Promise<void>((resolve) => offender.once("close", () => resolve()));
    offender.write(`${JSON.stringify({ jsonrpc: "2.0", id: 1, method: "runtime.status", params: {} })}\n`);
    await closed;
  } finally {
    offender.destroy();
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
  // Canonicalize the temp dir and bind the session to it: the daemon rejects
  // resuming a transcript whose stored project dir differs from the
  // (realpath-resolved) project dir, and the default project is the process
  // cwd — not this fixture's temp project.
  const directory = await realpath(await mkdtemp(join(tmpdir(), "xerxes-bun-resume-evict-")));
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
      params: { project_dir: directory, session_key: "picker-key" },
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

    const files = (await readdir(sessionDirectory)).filter(file => file.endsWith(".json"));
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
    // Isolated lease: a real daemon holding the shared one would legitimately
    // refuse this server's scheduler and the job would never fire.
    cronLeasePath: join(directory, "cron.lease"),
    cronPollInterval: 5,
    cronStoreFactory: () => store,
    runtime,
    socketPath,
  });
  await server.start();
  try {
    await waitFor(() => runner.runs === 1);
    await server.stop();

    const files = (await readdir(sessionDirectory)).filter(file => file.endsWith(".json"));
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
    // Empty skill library: single-token completions merge skill shorthands,
    // and the exact assertion below must not see the developer's real library.
    skillDirectories: [join(directory, "user-skills")],
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
      { value: "/ultra", label: "ultra", meta: "Toggle ultra mode", category: "daemon" },
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

/** A turn that ignores its abort signal, as a wedged provider stream does. */
class NeverSettlingRunner implements TurnRunner {
  runs = 0;

  async *run(): AsyncGenerator<DaemonEvent> {
    this.runs += 1;
    yield { type: "text_part", payload: { text: "started" } };
    await new Promise<void>(() => undefined);
  }
}

test("daemon stop persists sessions even when an in-flight turn never settles", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-stop-wedged-"));
  const socketPath = join(directory, "daemon.sock");
  const sessionDirectory = join(directory, "sessions");
  const runner = new NeverSettlingRunner();
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "wedged-model",
    sessionDirectory,
  });
  const server = new DaemonServer({
    cronLeasePath: join(directory, "cron.lease"),
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
      params: { session_key: "wedged-session" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "wedged-session", text: "wedge the drain" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_begin"));
    await waitFor(() => runner.runs === 1, 10_000);

    // The prompt is durable before the turn ends: the transcript itself is
    // only written in the turn's `finally`, which a crash never reaches.
    // Generous budgets: these waits separate "lands" from "never lands" and
    // must stay robust while the rest of the suite saturates the machine.
    await waitFor(async () => {
      try {
        return (await readdir(sessionDirectory)).some((file) => file.endsWith(".jsonl"));
      } catch (error) {
        if ((error as NodeJS.ErrnoException).code === "ENOENT") return false;
        throw error;
      }
    }, 10_000);
    const journal = (await readdir(sessionDirectory)).find((file) =>
      file.endsWith(".jsonl"),
    );
    expect(
      await readFile(join(sessionDirectory, String(journal)), "utf8"),
    ).toContain("wedge the drain");

    // Regression: the only session flush sat behind an unbounded await on the
    // drain, so one turn that never settles parked the daemon with the
    // transcript unwritten. The bound only has to separate "resolves" from
    // "hangs forever", so keep it far above scheduler jitter on loaded
    // machines instead of asserting interactive latency.
    const startedAt = Date.now();
    await server.stop();
    expect(Date.now() - startedAt).toBeLessThan(20_000);

    const files = (await readdir(sessionDirectory)).filter(file => file.endsWith(".json"));
    expect(files).toHaveLength(1);
    const saved = JSON.parse(
      await readFile(join(sessionDirectory, String(files[0])), "utf8"),
    ) as { messages: Array<{ content?: unknown; role?: string }> };
    expect(saved.messages).toContainEqual(
      expect.objectContaining({ role: "user", content: "wedge the drain" }),
    );
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("session listing renders rows from transcript headers and surfaces unreadable files", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-list-tiers-"));
  const sessionDirectory = join(directory, "sessions");
  const store = new DaemonTranscriptStore({
    directory: sessionDirectory,
    currentProjectDirectory: directory,
  });
  await mkdir(sessionDirectory, { recursive: true });
  // A record whose header is intact but whose message array is unparseable: it
  // can only be listed by a reader that stops at the head of the file.
  await writeFile(
    join(sessionDirectory, "aaaabbbbccc1.json"),
    [
      "{",
      '  "session_id": "aaaabbbbccc1",',
      '  "key": "aaaabbbbccc1",',
      '  "agent_id": "default",',
      `  "cwd": ${JSON.stringify(directory)},`,
      '  "updated_at": "2026-05-05T00:00:00.000Z",',
      '  "turn_count": 4,',
      '  "message_count": 8,',
      `  "metadata": {"title": "head-only row", "project_root": ${JSON.stringify(directory)}},`,
      '  "messages": [ not json at all',
    ].join("\n"),
    "utf8",
  );
  await writeFile(join(sessionDirectory, "ddddeeeefff2.json"), "totally corrupt", "utf8");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "list-model",
    transcriptStore: store,
  });
  try {
    const listed = await runtime.listSavedSessions(10, {
      projectDirectory: directory,
    });
    expect(listed.map((session) => session.title)).toContain("head-only row");
    expect(listed.find((session) => session.id === "aaaabbbbccc1")).toMatchObject({
      turnCount: 4,
      messageCount: 8,
    });
    // A corrupt file used to vanish from every listing, so it could never be
    // seen or deleted.
    expect(listed.find((session) => session.id === "ddddeeeefff2")).toMatchObject({
      resumable: false,
      status: "unreadable",
    });
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("a daemon refused the cron lease never fires the shared job store", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-cron-lease-"));
  const socketPath = join(directory, "daemon.sock");
  const leasePath = join(directory, "cron.lease");
  const jobsPath = join(directory, "cron", "jobs.json");
  // A live holder running this very command, recorded against a different
  // project: exactly the shape of a second project's daemon owning cron.
  await writeFile(
    leasePath,
    `${JSON.stringify({
      acquired_at: new Date().toISOString(),
      command: processCommand(process.pid),
      owner_key: join(directory, "some-other-project"),
      pid: process.pid,
    })}\n`,
    "utf8",
  );
  const store = new JobStore(jobsPath);
  store.add(
    new CronJob({
      id: "leased-job",
      prompt: "should never run",
      schedule: "* * * * *",
      nextRunAt: new Date(Date.now() - 60_000).toISOString(),
    }),
  );
  const server = new DaemonServer({
    cronLeasePath: leasePath,
    cronPollInterval: 5,
    cronStoreFactory: () => new JobStore(jobsPath),
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "cron-model",
      sessionDirectory: join(directory, "sessions"),
    }),
    socketPath,
  });
  await server.start();
  try {
    await new Promise((resolve) => setTimeout(resolve, 120));
    expect(new JobStore(jobsPath).get("leased-job")?.lastRunAt).toBeUndefined();
    // The foreign lease is left exactly as it was found.
    expect(readCronLease(leasePath)?.ownerKey).toBe(
      join(directory, "some-other-project"),
    );
  } finally {
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

/** A runtime whose session flush can be held open, to observe shutdown mid-flight. */
class BlockedFlushRuntime extends InMemoryDaemonRuntime {
  private gate: Promise<void> | undefined;
  private open: (() => void) | undefined;

  blockFlush(): void {
    this.gate = new Promise<void>((resolve) => {
      this.open = resolve;
    });
  }

  releaseFlush(): void {
    this.open?.();
    this.gate = undefined;
  }

  override async flushSessions(): Promise<void> {
    await this.gate;
    await super.flushSessions();
  }
}

test("shutdown arms a hard-exit signal handler and hands back its process handlers", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-stop-signals-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new BlockedFlushRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "signal-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const baselineSigterm = process.listenerCount("SIGTERM");
  const baselineCrash = process.listenerCount("uncaughtException");
  const server = new DaemonServer({
    crashHandlers: true,
    cronLeasePath: join(directory, "cron.lease"),
    cronStoreFactory: () => new JobStore(join(directory, "cron", "jobs.json")),
    runtime,
    socketPath,
  });
  try {
    await server.start();
    expect(process.listenerCount("uncaughtException")).toBe(baselineCrash + 1);
    expect(process.listenerCount("unhandledRejection")).toBeGreaterThan(0);

    runtime.blockFlush();
    const stopping = server.stop();
    // A second SIGTERM has to reach something: the host consumed the first one
    // with `process.once`, so without this handler it is silently dropped and
    // only SIGKILL is left.
    await waitFor(() => process.listenerCount("SIGTERM") === baselineSigterm + 1);
    runtime.releaseFlush();
    await stopping;

    expect(process.listenerCount("SIGTERM")).toBe(baselineSigterm);
    expect(process.listenerCount("uncaughtException")).toBe(baselineCrash);
  } finally {
    runtime.releaseFlush();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("an opted-in daemon snapshots the workspace before every turn and links it to the turn", async () => {
  if (!Bun.which("git")) return;
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-turn-snapshot-"));
  const workspace = join(directory, "workspace");
  const socketPath = join(directory, "daemon.sock");
  await mkdir(workspace);
  await writeFile(join(workspace, "state.txt"), "before turn zero", "utf8");
  const snapshots = new SnapshotManager(workspace, {
    shadowRoot: join(directory, "shadow"),
  });
  const server = new DaemonServer({
    socketPath,
    autoSnapshotTurns: true,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: workspace,
      model: "snapshot-model",
      sessionDirectory: join(directory, "sessions"),
    }),
    snapshotManagerFactory: () => snapshots,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "snap-turns", project_dir: workspace },
    });
    const initialized = await client.next((frame) => frame.id === 1);
    const sessionId = String(
      (initialized.result?.session as Record<string, unknown>).id,
    );
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({ jsonrpc: "2.0", id: 90, method: "slash", params: { command: "!echo after-shell > state.txt" } });
    expect((await client.next(frame => frame.id === 90)).result).toMatchObject({ ok: true });
    const shellSnapshot = snapshots.list()[0];
    expect(shellSnapshot).toBeDefined();
    expect(await Bun.file(join(workspace, "state.txt")).text()).toBe("after-shell\n");
    await snapshots.restoreFile(shellSnapshot!.id, "state.txt");
    expect(await Bun.file(join(workspace, "state.txt")).text()).toBe("before turn zero");

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "snap-turns", text: "first" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_end"));
    await waitFor(() => snapshots.getForTurn(sessionId, 0) !== undefined);

    await writeFile(join(workspace, "state.txt"), "before turn one", "utf8");
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "turn.submit",
      params: { session_key: "snap-turns", text: "second" },
    });
    await client.next((frame) => frame.id === 3);
    await client.next(eventFrame("turn_end"));
    await waitFor(() => snapshots.getForTurn(sessionId, 1) !== undefined);

    const first = snapshots.getForTurn(sessionId, 0);
    expect(first).toMatchObject({ sessionId, turnIndex: 0, label: "turn-0" });
    // "Take me back to before turn 1" is now expressible from the record alone.
    await writeFile(join(workspace, "state.txt"), "agent damage", "utf8");
    const target = snapshots.getForTurn(sessionId, 1);
    await snapshots.rollback(String(target?.id));
    expect(await readFile(join(workspace, "state.txt"), "utf8")).toBe(
      "before turn one",
    );
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("a daemon that was not opted in never snapshots a turn", async () => {
  if (!Bun.which("git")) return;
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-no-turn-snapshot-"));
  const workspace = join(directory, "workspace");
  const socketPath = join(directory, "daemon.sock");
  await mkdir(workspace);
  const snapshots = new SnapshotManager(workspace, {
    shadowRoot: join(directory, "shadow"),
  });
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: workspace,
      model: "snapshot-model",
      sessionDirectory: join(directory, "sessions"),
    }),
    snapshotManagerFactory: () => snapshots,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "no-snap", project_dir: workspace },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "no-snap", text: "first" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_end"));

    expect(snapshots.list()).toEqual([]);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("a turn snapshot failure never fails the turn", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-snapshot-fail-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    autoSnapshotTurns: true,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "snapshot-model",
      sessionDirectory: join(directory, "sessions"),
    }),
    snapshotManagerFactory: () => {
      throw new Error("shadow repository unavailable");
    },
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "snap-fail", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "snap-fail", text: "still runs" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
    });
    const end = await client.next(eventFrame("turn_end"));
    expect(end.params?.payload).toMatchObject({ cancelled: false });
    const warning = await client.next(frame => frame.params?.type === "notification" && String(frame.params?.payload?.body ?? "").includes("Could not snapshot"));
    expect(warning.params?.payload?.body).toContain("shadow repository unavailable");
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("a single-file rollback restores one path and leaves the rest of the tree", async () => {
  if (!Bun.which("git")) return;
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-restore-file-"));
  const workspace = join(directory, "workspace");
  const socketPath = join(directory, "daemon.sock");
  await mkdir(workspace);
  await writeFile(join(workspace, "damaged.txt"), "good version", "utf8");
  await writeFile(join(workspace, "keep.txt"), "original", "utf8");
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
      method: "initialize",
      params: { session_key: "restore", project_dir: workspace },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/snapshot base" },
    });
    const taken = await client.next((frame) => frame.id === 2);
    const snapshotId = String(
      (taken.result?.snapshot as Record<string, unknown>).id,
    );
    await client.next(eventFrame("notification"));

    await writeFile(join(workspace, "damaged.txt"), "agent damage", "utf8");
    await writeFile(join(workspace, "keep.txt"), "later edit", "utf8");

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "slash",
      params: { command: `/rollback ${snapshotId} damaged.txt` },
    });
    const restored = await client.next((frame) => frame.id === 3);
    expect(restored.result).toMatchObject({ ok: true, path: "damaged.txt" });
    await client.next(eventFrame("notification"));

    expect(await readFile(join(workspace, "damaged.txt"), "utf8")).toBe(
      "good version",
    );
    // A single-file restore is not a rollback of everything else.
    expect(await readFile(join(workspace, "keep.txt"), "utf8")).toBe("later edit");

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "slash",
      params: { command: `/rollback ${snapshotId} ../escape.txt` },
    });
    expect((await client.next((frame) => frame.id === 4)).result).toMatchObject({
      ok: false,
      error: expect.stringContaining("escapes the snapshot workspace"),
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("resume reports what loading the transcript changed instead of losing messages quietly", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-resume-repair-"));
  const sessionDirectory = join(directory, "sessions");
  const socketPath = join(directory, "daemon.sock");
  await mkdir(sessionDirectory, { recursive: true });
  await writeFile(
    join(sessionDirectory, "aaaabbbbccc1.json"),
    JSON.stringify({
      session_id: "aaaabbbbccc1",
      key: "aaaabbbbccc1",
      agent_id: "default",
      cwd: directory,
      updated_at: "2026-05-05T00:00:00.000Z",
      turn_count: 2,
      message_count: 5,
      metadata: { title: "repaired", project_root: directory },
      messages: [
        { role: "user", content: "hello" },
        "this entry is not a message",
        {
          role: "assistant",
          content: "working",
          tool_calls: [{ id: "call-a", name: "ReadFile", input: {} }],
        },
        { role: "tool", tool_call_id: "orphan", content: "no matching call" },
      ],
    }),
    "utf8",
  );
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "resume-model",
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
      params: { session_key: "tui:live", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/resume aaaabbbbccc1" },
    });
    await client.next((frame) => frame.id === 2);
    const notice = await client.next(
      (frame) =>
        frame.method === "event" &&
        frame.params?.type === "notification" &&
        String(
          (frame.params.payload as Record<string, unknown> | undefined)?.body ??
            "",
        ).startsWith("Resume repaired"),
    );
    const body = String(
      (notice.params?.payload as Record<string, unknown>).body,
    );
    expect(body).toContain("dropped 1 malformed message");
    expect(body).toContain("dropped 1 orphaned tool reply");
    expect(body).toContain("inserted 1 interrupted-call placeholder");
    expect((notice.params?.payload as Record<string, unknown>).severity).toBe(
      "warning",
    );
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("a clean resume says nothing about repairs", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-resume-clean-"));
  const sessionDirectory = join(directory, "sessions");
  const socketPath = join(directory, "daemon.sock");
  await mkdir(sessionDirectory, { recursive: true });
  await writeFile(
    join(sessionDirectory, "aaaabbbbccc2.json"),
    JSON.stringify({
      session_id: "aaaabbbbccc2",
      key: "aaaabbbbccc2",
      agent_id: "default",
      cwd: directory,
      updated_at: "2026-05-05T00:00:00.000Z",
      turn_count: 1,
      message_count: 2,
      metadata: { title: "clean", project_root: directory },
      messages: [
        { role: "user", content: "hello" },
        { role: "assistant", content: "hi" },
      ],
    }),
    "utf8",
  );
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "resume-model",
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
      params: { session_key: "tui:live", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/resume aaaabbbbccc2" },
    });
    await client.next((frame) => frame.id === 2);
    const resumed = await client.next(
      (frame) =>
        frame.method === "event" &&
        frame.params?.type === "notification" &&
        String(
          (frame.params.payload as Record<string, unknown> | undefined)?.body ??
            "",
        ).startsWith("Resumed session"),
    );
    expect(resumed.params?.payload).toBeDefined();
    // Nothing further: a clean load must not manufacture a warning.
    await Bun.sleep(50);
    expect(
      client.seen((frame) =>
        String(
          (frame.params?.payload as Record<string, unknown> | undefined)?.body ??
            "",
        ).startsWith("Resume repaired"),
      ),
    ).toBe(false);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("transcript search spans saved sessions and reports what it could not index", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-search-"));
  const sessionDirectory = join(directory, "sessions");
  const socketPath = join(directory, "daemon.sock");
  await mkdir(sessionDirectory, { recursive: true });
  await writeFile(
    join(sessionDirectory, "aaaabbbbccc3.json"),
    JSON.stringify({
      session_id: "aaaabbbbccc3",
      key: "aaaabbbbccc3",
      agent_id: "default",
      cwd: directory,
      updated_at: "2026-05-05T00:00:00.000Z",
      turn_count: 1,
      message_count: 3,
      metadata: { title: "older", project_root: directory },
      messages: [
        { role: "user", content: "why does the retry backoff regress" },
        { role: "assistant", content: "because the timer is reset" },
        { role: "tool", content: 42 },
      ],
    }),
    "utf8",
  );
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "search-model",
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
      params: { session_key: "tui:live", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    // A live turn feeds the index incrementally, next to the cold read.
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "tui:live", text: "retry backoff in the daemon" },
    });
    await client.next((frame) => frame.id === 2);
    await client.next(eventFrame("turn_end"));

    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "session.search",
      params: { query: "retry backoff" },
    });
    const found = await client.next((frame) => frame.id === 3);
    const results = found.result?.results as Array<Record<string, unknown>>;
    expect(found.result).toMatchObject({ ok: true });
    expect(results.map((row) => row.session_id)).toContain("aaaabbbbccc3");
    expect(results.length).toBeGreaterThanOrEqual(2);
    expect(String(results[0]?.excerpt)).toContain("retry backoff");
    // The row the indexer does not model is counted, not serialized.
    expect(found.result?.stats).toMatchObject({ unrecognized_messages: 1 });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "session.search",
      params: { query: "retry backoff", session_id: "aaaabbbbccc3" },
    });
    const scoped = await client.next((frame) => frame.id === 4);
    expect(
      (scoped.result?.results as Array<Record<string, unknown>>).every(
        (row) => row.session_id === "aaaabbbbccc3",
      ),
    ).toBe(true);

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "slash",
      params: { command: "/search nothingmatchesthisatall" },
    });
    expect((await client.next((frame) => frame.id === 5)).result).toMatchObject({
      ok: true,
      results: [],
    });
    expect(
      (await client.next(eventFrame("notification"))).params?.payload,
    ).toMatchObject({
      category: "slash",
      body: expect.stringContaining("No transcript matches"),
    });

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "session.search",
      params: { query: "   " },
    });
    expect((await client.next((frame) => frame.id === 6)).result).toEqual({
      ok: false,
      error: "search query is required",
    });
    client.send({ jsonrpc: '2.0', id: 7, method: 'slash', params: { command: '/search --session aaaabbbbccc3 --limit 1 retry backoff' } });
    const filtered = (await client.next(frame => frame.id === 7)).result;
    expect(filtered).toMatchObject({ ok: true, results: [{ session_id: 'aaaabbbbccc3' }] });
    client.send({ jsonrpc: '2.0', id: 8, method: 'slash', params: { command: '/search --limit -1 retry' } });
    expect((await client.next(frame => frame.id === 8)).result).toMatchObject({ ok: false });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("terminal RPCs list, inspect, and control the shells the agent is running", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-terminals-"));
  const socketPath = join(directory, "daemon.sock");
  const terminals = new TerminalRegistry();
  const server = new DaemonServer({
    socketPath,
    terminalRegistry: terminals,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  client.send({ jsonrpc: "2.0", id: 0, method: "initialize", params: { session_key: "terminal-owner" } });
  const initialized = await client.next((frame) => frame.id === 0);
  const ownerSessionId = String((initialized.result?.session as { id?: unknown } | undefined)?.id ?? "");
  await client.next(eventFrame("init_done"));
  await client.next(eventFrame("status_update"));
  const written: string[] = [];
  let killed: string | undefined;
  const handle = terminals.open({
    id: "pty_live",
    kind: "pty",
    ownerSessionId,
    command: "bash -i",
    cwd: directory,
    control: {
      write: async (chars) => void written.push(chars),
      kill: async (signal) => {
        killed = signal;
      },
    },
  });
  handle.append("$ echo hi\nhi\n");

  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "terminal.list", params: {} });
    const listed = await client.next((frame) => frame.id === 1);
    expect(listed.result?.ok).toBe(true);
    expect(
      (listed.result?.terminals as Array<Record<string, unknown>>)[0],
    ).not.toHaveProperty("ownerSessionId");
    expect(
      (listed.result?.terminals as Array<Record<string, unknown>>)[0],
    ).toMatchObject({
      id: "pty_live",
      kind: "pty",
      running: true,
      canWrite: true,
      canKill: true,
      canInterrupt: false,
    });

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "terminal.inspect",
      params: { terminal_id: "pty_live" },
    });
    const inspected = await client.next((frame) => frame.id === 2);
    expect(inspected.result?.terminal).toMatchObject({ output: "$ echo hi\nhi\n" });

    // Inspecting twice returns the same tail: the viewer mirrors output rather
    // than draining the buffer the model reads from.
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "terminal.inspect",
      params: { terminal_id: "pty_live" },
    });
    expect(
      (await client.next((frame) => frame.id === 3)).result?.terminal,
    ).toMatchObject({ output: "$ echo hi\nhi\n" });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "terminal.control",
      params: { terminal_id: "pty_live", action: "write", chars: "ls\n" },
    });
    expect((await client.next((frame) => frame.id === 4)).result?.ok).toBe(true);
    expect(written).toEqual(["ls\n"]);

    client.send({
      jsonrpc: "2.0",
      id: 5,
      method: "terminal.control",
      params: { terminal_id: "pty_live", action: "interrupt" },
    });
    expect((await client.next((frame) => frame.id === 5)).result).toMatchObject({
      ok: false,
      error: "this terminal cannot be interrupted",
    });

    client.send({
      jsonrpc: "2.0",
      id: 6,
      method: "terminal.control",
      params: { terminal_id: "pty_live", action: "kill", signal: "SIGKILL" },
    });
    expect((await client.next((frame) => frame.id === 6)).result?.ok).toBe(true);
    expect(killed).toBe("SIGKILL");

    client.send({
      jsonrpc: "2.0",
      id: 7,
      method: "terminal.inspect",
      params: { terminal_id: "nope" },
    });
    expect((await client.next((frame) => frame.id === 7)).result).toEqual({
      ok: false,
      error: "unknown terminal",
    });

    const other = await SocketTestClient.connect(socketPath);
    try {
      other.send({ jsonrpc: "2.0", id: 8, method: "terminal.list", params: {} });
      expect((await other.next((frame) => frame.id === 8)).result).toEqual({ ok: true, terminals: [] });
      other.send({ jsonrpc: "2.0", id: 9, method: "terminal.inspect", params: { terminal_id: "pty_live" } });
      expect((await other.next((frame) => frame.id === 9)).result).toEqual({ ok: false, error: "unknown terminal" });
      other.send({ jsonrpc: "2.0", id: 10, method: "terminal.control", params: { terminal_id: "pty_live", action: "kill" } });
      expect((await other.next((frame) => frame.id === 10)).result).toMatchObject({ ok: false, error: expect.stringContaining("unknown terminal") });
    } finally {
      other.close();
    }
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("a daemon with no terminal registry reports an empty list rather than pretending", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-no-terminals-"));
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
    client.send({ jsonrpc: "2.0", id: 1, method: "terminal.list", params: {} });
    expect((await client.next((frame) => frame.id === 1)).result).toEqual({
      ok: true,
      terminals: [],
    });

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "terminal.control",
      params: { terminal_id: "anything", action: "kill" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: false,
      error: "this daemon tracks no terminals",
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("background turn events are routed to their own live session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-background-route-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "test-model",
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { session_key: "foreground" } });
    const initialized = await client.next((frame) => frame.id === 1);
    const foregroundId = String((initialized.result?.session as { id?: unknown } | undefined)?.id ?? "");
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.background",
      params: { session_key: "foreground", text: "work elsewhere" },
    });
    const started = await client.next((frame) => frame.id === 2);
    const taskId = String((started.result as { task_id?: unknown } | undefined)?.task_id ?? "");
    expect(taskId).not.toBe("");
    expect(taskId).not.toBe(foregroundId);

    const begin = await client.next(eventFrame("turn_begin"));
    const delta = await client.next(eventFrame("text_part"));
    expect(begin.params?.payload).toMatchObject({ background_task_id: taskId, session_id: taskId });
    expect(delta.params?.payload).toMatchObject({ background_task_id: taskId, session_id: taskId });
    const complete = await client.next(eventFrame("background.complete"));
    expect(complete.params?.payload).toEqual({ task_id: taskId, text: "finished" });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("a completed first exchange generates a model-written session title and broadcasts it", async () => {
  resetTitleAttempts();
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-autotitle-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "test-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    socketPath,
    runtime,
    titleClientFactory: () =>
      ({
        async *stream() {
          yield { type: "text_part", payload: { text: "" } };
        },
        async complete() {
          return { content: '"Greeting exchange"' } as never;
        },
      }) as unknown as LlmClient,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "titled" },
    });
    const initialized = await client.next((frame) => frame.id === 1);
    const sessionId = String(
      (initialized.result?.session as { id?: unknown } | undefined)?.id ?? "",
    );
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "titled", text: "hello there" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({ ok: true });

    // A provisional title lands with the submit, before a single token
    // streams, so the chat is never anonymous while it works.
    const provisional = await client.next(eventFrame("session_title"));
    expect(provisional.params?.payload).toMatchObject({
      session_id: sessionId,
      title: "Hello there",
    });

    await client.next(eventFrame("turn_end"));

    // The model-written title lands in the background after the turn and
    // replaces the placeholder; wait for its event.
    const titled = await client.next(eventFrame("session_title"));
    expect(titled.params?.payload).toMatchObject({
      session_id: sessionId,
      title: "Greeting exchange",
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
    resetTitleAttempts();
  }
});

test("auto_title off leaves the session untitled with no provider call", async () => {
  resetTitleAttempts();
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-autotitle-off-"));
  const socketPath = join(directory, "daemon.sock");
  let titleCalls = 0;
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "test-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    socketPath,
    runtime,
    autoTitle: false,
    titleClientFactory: () => {
      titleCalls += 1;
      throw new Error("title client must not be built when auto_title is off");
    },
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "untitled" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "untitled", text: "hello there" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({ ok: true });
    await client.next(eventFrame("turn_end"));

    expect(titleCalls).toBe(0);
    expect(runtime.sessionStatus("untitled")?.metadata.title).toBeUndefined();
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
    resetTitleAttempts();
  }
});

test("daemon.wipe_history removes the transcript store and reports counts while sessions stay live", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-wipe-history-"));
  const home = join(directory, "home");
  const socketPath = join(directory, "daemon.sock");
  const sessionDirectory = join(directory, "sessions");
  const priorHome = process.env.XERXES_HOME;
  process.env.XERXES_HOME = home;
  await mkdir(sessionDirectory, { recursive: true });
  await mkdir(join(home, "snapshots"), { recursive: true });
  await writeFile(join(sessionDirectory, "a.json"), "{}");
  await writeFile(join(sessionDirectory, "b.json"), "{}");
  await writeFile(join(home, "snapshots", "snap.json"), "{}");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "test-model",
    sessionDirectory,
  });
  const server = new DaemonServer({
    sessionArchiveDirectory: sessionDirectory,
    socketPath,
    runtime,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "wipe-target" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({ jsonrpc: "2.0", id: 2, method: "daemon.wipe_history", params: {} });
    const reply = await client.next((frame) => frame.id === 2);
    expect(reply.result).toMatchObject({ ok: true });
    const removed = (reply.result as { removed: { bytes: number; files: number } }).removed;
    expect(removed.files).toBe(3);
    expect(removed.bytes).toBeGreaterThan(0);
    expect(existsSync(sessionDirectory)).toBe(false);
    expect(existsSync(join(home, "snapshots"))).toBe(false);

    // The live session survives the wipe and its next turn recreates history
    // from a zero persistence generation instead of conflicting with the
    // deleted transcript.
    client.send({ jsonrpc: "2.0", id: 3, method: "session.status", params: {} });
    const status = await client.next((frame) => frame.id === 3);
    expect(status.result).toMatchObject({ ok: true });

    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "turn.submit",
      params: { session_key: "wipe-target", text: "recreate history" },
    });
    expect((await client.next((frame) => frame.id === 4)).result).toMatchObject({ ok: true });
    await client.next(eventFrame("turn_end"));
    const recreated = JSON.parse(await readFile(join(sessionDirectory, `${String((status.result as { session?: { id?: unknown } }).session?.id ?? "")}.json`), "utf8")) as { messages?: unknown[] };
    expect(recreated.messages?.length).toBeGreaterThanOrEqual(2);
  } finally {
    client.close();
    await server.stop();
    if (priorHome === undefined) {
      delete process.env.XERXES_HOME;
    } else {
      process.env.XERXES_HOME = priorHome;
    }
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon.wipe_history refuses while a turn is mid-write", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-wipe-busy-"));
  const socketPath = join(directory, "daemon.sock");
  const sessionDirectory = join(directory, "sessions");
  let release: (() => void) | undefined;
  const hanging = new Promise<void>((resolve) => {
    release = resolve;
  });
  const runner: TurnRunner = {
    async *run(): AsyncGenerator<DaemonEvent> {
      yield { type: "text_part", payload: { text: "hang" } } as DaemonEvent;
      await hanging;
    },
  };
  const runtime = new InMemoryDaemonRuntime(runner, {
    currentProjectDirectory: directory,
    model: "test-model",
    sessionDirectory,
  });
  const server = new DaemonServer({
    sessionArchiveDirectory: sessionDirectory,
    socketPath,
    runtime,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "busy-target" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "busy-target", text: "hang" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({ ok: true });
    await client.next(eventFrame("text_part"));

    client.send({ jsonrpc: "2.0", id: 3, method: "daemon.wipe_history", params: {} });
    const reply = await client.next((frame) => frame.id === 3);
    expect(reply.result).toMatchObject({ ok: false });
    expect(String((reply.result as { error: string }).error)).toContain("mid-write");
  } finally {
    release?.();
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("daemon.wipe_memory removes global and project memory stores with counts", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-wipe-memory-"));
  const home = join(directory, "home");
  const socketPath = join(directory, "daemon.sock");
  const priorHome = process.env.XERXES_HOME;
  process.env.XERXES_HOME = home;
  await mkdir(join(home, "memory"), { recursive: true });
  await mkdir(join(home, "agent_memory"), { recursive: true });
  await mkdir(join(home, "projects", "p1"), { recursive: true });
  await mkdir(join(directory, ".xerxes_memory"), { recursive: true });
  await writeFile(join(home, "memory", "notes.md"), "remember me");
  await writeFile(join(home, "agent_memory", "self.md"), "self");
  await writeFile(join(home, "projects", "p1", "facts.md"), "facts");
  await writeFile(join(directory, ".xerxes_memory", "memory.db"), "sqlite");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "test-model",
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
      params: { session_key: "memory-target" },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({ jsonrpc: "2.0", id: 2, method: "daemon.wipe_memory", params: {} });
    const reply = await client.next((frame) => frame.id === 2);
    expect(reply.result).toMatchObject({ ok: true });
    const removed = (reply.result as { removed: { bytes: number; files: number } }).removed;
    // Three seeded memory files are always removed; the project SQLite store joins
    // them when the active session's cwd resolves to this directory.
    expect(removed.files).toBeGreaterThanOrEqual(3);
    expect(removed.bytes).toBeGreaterThan(0);
    expect(existsSync(join(home, "memory"))).toBe(false);
    expect(existsSync(join(home, "agent_memory"))).toBe(false);
    expect(existsSync(join(home, "projects"))).toBe(false);
  } finally {
    client.close();
    await server.stop();
    if (priorHome === undefined) {
      delete process.env.XERXES_HOME;
    } else {
      process.env.XERXES_HOME = priorHome;
    }
    await rm(directory, { recursive: true, force: true });
  }
});

test("a slow model discovery does not stall the rest of the connection", async () => {
  // A provider endpoint that accepts the request and takes its time. Model
  // discovery used to run inline on the per-connection serialization queue, so
  // one of these held up every later request from the same client — measured
  // at 8.4s for a single unreachable endpoint, with the provider the user was
  // actually looking at stuck behind it in the picker.
  const slowProvider = Bun.serve({
    port: 0,
    fetch: async () => {
      await Bun.sleep(400);
      return Response.json({ data: [{ id: "slow-model" }] });
    },
  });
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-discovery-queue-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "test-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const profilesPath = join(directory, "profiles.json");
  const server = new DaemonServer({
    socketPath,
    runtime,
    profileStore: new ProfileStore(profilesPath),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "provider_save",
      params: {
        name: "slow",
        base_url: `http://127.0.0.1:${slowProvider.port}/v1`,
        model: "slow-model",
        provider: "openai",
        api_key: "k",
      },
    });
    expect((await client.next((frame) => frame.id === 1)).result).toMatchObject({ ok: true });

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "fetch_models",
      params: { profile_name: "slow" },
    });
    client.send({ jsonrpc: "2.0", id: 3, method: "runtime.status", params: {} });

    // The cheap request must come back while discovery is still in flight.
    const status = await client.next((frame) => frame.id === 3);
    expect(status.result).toBeDefined();
    expect(client.seen((frame) => frame.id === 2)).toBe(false);

    // …and the discovery still answers on its own schedule.
    const discovery = await client.next((frame) => frame.id === 2);
    expect(discovery.result).toMatchObject({ ok: true, models: ["slow-model"] });
  } finally {
    client.close();
    await server.stop();
    await slowProvider.stop(true);
    await rm(directory, { recursive: true, force: true });
  }
});

test("initialize reports the workspace git branch and session.status prices the run", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-branch-daemon-"));
  const socketPath = join(directory, "daemon.sock");
  // A real git repo, so the branch the daemon reports is the one git does.
  const repo = join(directory, "repo");
  await mkdir(repo, { recursive: true });
  const git = (args: string[]): void => {
    const proc = Bun.spawnSync(["git", "-C", repo, ...args], { stdin: "ignore" });
    if (proc.exitCode !== 0) throw new Error(`git ${args.join(" ")} failed`);
  };
  git(["init"]);
  git(["config", "user.email", "test@example.com"]);
  git(["config", "user.name", "Test"]);
  git(["commit", "--allow-empty", "-m", "seed"]);
  const expectedBranch = new TextDecoder().decode(
    Bun.spawnSync(["git", "-C", repo, "symbolic-ref", "--short", "HEAD"], { stdin: "ignore" }).stdout,
  ).trim();

  const server = new DaemonServer({
    socketPath,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: repo,
      model: "protocol-model",
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
      params: { session_key: "branch-session", project_dir: repo },
    });
    const initialized = await client.next((frame) => frame.id === 1);
    const initDone = await client.next(eventFrame("init_done"));
    // The branch rides init_done (the shell's initialize payload)…
    expect(initDone.params?.payload).toMatchObject({ branch: expectedBranch });
    // …and the session projection carries the cost estimate plus an MCP
    // status record (empty — no manager is injected on this server).
    expect(initialized.result?.session).toMatchObject({
      model: "protocol-model",
      cost_usd: expect.any(Number),
      mcp_status: {},
    });

    // Outside a git repo the branch is absent, not fabricated.
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "initialize",
      params: { session_key: "plain-session", project_dir: directory },
    });
    const plain = await client.next((frame) => frame.id === 2);
    const plainInitDone = await client.next(eventFrame("init_done"));
    expect(plainInitDone.params?.payload).toMatchObject({ branch: "" });
    // The daemon resolves the project directory through realpath, so on
    // macOS /tmp paths arrive as /private/tmp.
    expect((plain.result?.session as Record<string, unknown> | undefined)?.cwd).toBe(await realpath(directory));
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("session.status surfaces a connected MCP manager's redacted statuses", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-mcp-daemon-"));
  const socketPath = join(directory, "daemon.sock");
  const stubManager = {
    listStatus: () => [
      {
        name: "filesystem",
        connected: true,
        tools: 11,
        resources: 2,
        prompts: 1,
      },
      {
        name: "sqlite",
        connected: false,
        tools: 0,
        resources: 0,
        prompts: 0,
        lastError: "connect ECONNREFUSED",
      },
    ],
  } as unknown as MCPManager;
  const server = new DaemonServer({
    socketPath,
    mcpManager: stubManager,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "protocol-model",
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
      params: { session_key: "mcp-session" },
    });
    const initialized = await client.next((frame) => frame.id === 1);
    const mcp = ((initialized.result?.session as Record<string, unknown> | undefined)?.mcp_status ?? {}) as Record<string, Record<string, unknown>>;
    expect(mcp).toMatchObject({
      filesystem: { connected: true, tools: 11, resources: 2, prompts: 1 },
      sqlite: { connected: false, lastError: "connect ECONNREFUSED" },
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("changes.undo reverse-applies recorded edits and refuses drifted files", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-undo-daemon-"));
  const socketPath = join(directory, "daemon.sock");
  const file = join(directory, "draft.ts");
  const original = "const a = 1;\nconst b = 2;\n";
  await writeFile(file, original);
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "protocol-model",
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    socketPath,
    runtime,
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "undo-session", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    const session = runtime.sessionStatus("undo-session");
    if (!session) throw new Error("expected live session");
    const edit = (id: string, oldString: string, newString: string) => ({
      name: "FileEditTool",
      inputs: { file_path: file, old_string: oldString, new_string: newString },
      permitted: true,
      tool_call_id: id,
      duration_ms: 5,
      display_blocks: [],
    });
    session.messages.push(
      { role: "user", content: "edit the draft" },
      { role: "tool", content: "ok", name: "FileEditTool", tool_call_id: "e1" },
      { role: "tool", content: "ok", name: "FileEditTool", tool_call_id: "e2" },
    );
    session.toolExecutions.push(edit("e1", "const a = 1;", "const a = 2;"), edit("e2", "const b = 2;", "const b = 3;"));
    // The recorded edits were actually applied before the undo.
    await writeFile(file, "const a = 2;\nconst b = 3;\n");

    client.send({ jsonrpc: "2.0", id: 2, method: "changes.undo", params: { session_key: "undo-session" } });
    const undone = await client.next((frame) => frame.id === 2);
    expect(undone.result).toMatchObject({ ok: true, reverted: 2 });
    expect(await readFile(file, "utf8")).toBe(original);

    // A file that drifted after the recorded edit refuses the undo.
    await writeFile(file, "someone else was here;\n");
    session.toolExecutions.push(edit("e3", "const a = 1;", "const a = 99;"));
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "changes.undo",
      params: { session_key: "undo-session", path: file },
    });
    const refused = await client.next((frame) => frame.id === 3);
    expect(refused.result?.ok).toBe(false);
    const refusal = (refused.result?.results as Array<Record<string, unknown>> | undefined)?.[0];
    expect(String(refusal?.error)).toContain("refusing");
    expect(await readFile(file, "utf8")).toBe("someone else was here;\n");

    // A path with no recorded edits says so instead of guessing.
    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "changes.undo",
      params: { session_key: "undo-session", path: join(directory, "never-edited.ts") },
    });
    const missing = await client.next((frame) => frame.id === 4);
    expect(missing.result?.ok).toBe(false);
    expect(String(missing.result?.error)).toContain("no reversible recorded edits");
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("workspace.worktree creates a real git worktree and refuses non-repos", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-wt-daemon-"));
  const socketPath = join(directory, "daemon.sock");
  const repo = join(directory, "repo");
  await mkdir(repo, { recursive: true });
  const git = (args: string[]): void => {
    const proc = Bun.spawnSync(["git", "-C", repo, ...args], { stdin: "ignore" });
    if (proc.exitCode !== 0) throw new Error(`git ${args.join(" ")} failed`);
  };
  git(["init"]);
  git(["config", "user.email", "test@example.com"]);
  git(["config", "user.name", "Test"]);
  git(["commit", "--allow-empty", "-m", "seed"]);

  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: repo,
    model: "protocol-model",
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
      params: { session_key: "wt-session", project_dir: repo },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "workspace.worktree",
      params: { session_key: "wt-session", action: "create", name: "feat x" },
    });
    const created = await client.next((frame) => frame.id === 2);
    const path = String(created.result?.path ?? "");
    expect(created.result?.ok).toBe(true);
    expect(created.result?.branch).toBe("feat-x");
    expect(path.endsWith(`repo-feat-x`)).toBe(true);
    expect(existsSync(path)).toBe(true);

    const listed = new TextDecoder().decode(
      Bun.spawnSync(["git", "-C", repo, "worktree", "list"], { stdin: "ignore", stdout: "pipe" }).stdout,
    );
    expect(listed).toContain("repo-feat-x");

    // Creating it again refuses on the existing path instead of clobbering.
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "workspace.worktree",
      params: { session_key: "wt-session", action: "create", name: "feat-x" },
    });
    const again = await client.next((frame) => frame.id === 3);
    expect(again.result?.ok).toBe(false);
    expect(String(again.result?.error)).toContain("already exists");
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("scheduled timeout cancels the daemon provider turn without archiving a successful result", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-cron-timeout-wiring-"));
  const store = new JobStore(join(directory, "jobs.json"));
  store.add(new CronJob({ id: "timeout", prompt: "wait", oneshot: true, nextRunAt: new Date(Date.now() - 1000).toISOString() }));
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, "sessions") });
  const archive = join(directory, "archive");
  const history = new RunHistory(join(directory, "runs.sqlite"));
  const server = new DaemonServer({ socketPath: join(directory, "daemon.sock"), cronStoreFactory: () => store, cronLeasePath: join(directory, "cron.lease"), cronArchiveDirectory: archive, cronJobTimeout: 30, cronPollInterval: 5, runtime, runHistory: history });
  try {
    await server.start();
    await waitFor(() => runner.runs === 1);
    await waitFor(() => typeof store.get("timeout")?.metadata.last_error === "string");
    await waitFor(() => runtime.sessionStatus("cron:timeout")?.status === "idle");
    expect(store.get("timeout")?.metadata.last_error).toContain("timed out");
    expect(runtime.sessionStatus("cron:timeout")?.cancelRequested).toBe(true);
    expect(existsSync(archive)).toBe(false);
    expect(runner.runs).toBe(1);
    const owner = runtime.sessionStatus("cron:timeout")!.id;
    await waitFor(() => history.list(owner)[0]?.state === "cancelled");
    expect(history.list(owner, { unreadOnly: true })[0]).toMatchObject({ state: "cancelled", sourceId: "timeout" });
  } finally { await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test("shutdown keeps cron ownership until cancelled provider cleanup has settled", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-cron-lease-drain-"));
  const leasePath = join(directory, "cron.lease");
  const store = new JobStore(join(directory, "jobs.json"));
  store.add(new CronJob({ id: "cleanup", prompt: "work", oneshot: true, nextRunAt: new Date(Date.now() - 1000).toISOString() }));
  const started = Promise.withResolvers<void>();
  const cancelled = Promise.withResolvers<void>();
  const cleanup = Promise.withResolvers<void>();
  const runtime = new InMemoryDaemonRuntime({ async *run(_session, _text, signal) {
    signal.addEventListener("abort", () => cancelled.resolve(), { once: true });
    started.resolve();
    await cleanup.promise;
    yield { type: "text_part", payload: { text: "cleanup finished" } };
  } }, { currentProjectDirectory: directory, sessionDirectory: join(directory, "sessions") });
  const server = new DaemonServer({ socketPath: join(directory, "daemon.sock"), cronStoreFactory: () => store, cronLeasePath: leasePath, runtime });
  try {
    await server.start();
    await started.promise;
    const stopping = server.stop();
    await cancelled.promise;
    expect(readCronLease(leasePath)).toBeDefined();
    cleanup.resolve();
    await stopping;
    await waitFor(() => !existsSync(leasePath));
  } finally { cleanup.resolve(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test("run history RPC and slash listing expose scoped unread results", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-run-history-rpc-"));
  const history = new RunHistory(join(directory, "runs.sqlite"));
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, "sessions") });
  const session = await runtime.openSession("runs-owner");
  const run = history.start({ ownerSessionId: session.id, workspace: directory, kind: "schedule", sourceId: "one", title: "Check build" });
  const finished = history.finish(session.id, run.id, "succeeded", { output: "build passed" });
  const socketPath = join(directory, "daemon.sock");
  const interactions = new DaemonInteractionBoard();
  const server = new DaemonServer({ socketPath, runtime, interactions, runHistory: history, cronLeasePath: join(directory, "cron.lease"), cronStoreFactory: () => new JobStore(join(directory, "jobs.json")) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "run.list", params: { session_key: "runs-owner", unread_only: true } });
    const list = await client.next(frame => frame.id === 1);
    expect(list.result?.runs).toMatchObject([{ id: run.id, unread: true, state: "succeeded" }]);
    expect(JSON.stringify(list.result)).not.toContain("build passed");
    const permission = interactions.permissionBroker(session.id).request({ requestId: 'pending-report', description: 'Write report', inputs: {}, toolCall: { id: 'write', type: 'function', function: { name: 'WriteFile', arguments: {} } } });
    const foreign = interactions.permissionBroker('another-session').request({ requestId: 'foreign-report', description: 'Private report', inputs: {}, toolCall: { id: 'foreign', type: 'function', function: { name: 'WriteFile', arguments: {} } } });
    client.send({ jsonrpc: '2.0', id: 99, method: 'run.list', params: { session_key: 'runs-owner', scope: 'workspace' } });
    const attention = (await client.next(frame => frame.id === 99)).result;
    expect(attention).toMatchObject({ attention_total: 1, attention: [{ id: 'pending-report', kind: 'approval', title: 'Write report' }] });
    expect(JSON.stringify(attention)).not.toContain('Private report');
    interactions.respondPermission('pending-report', 'reject'); interactions.respondPermission('foreign-report', 'reject');
    await Promise.all([permission, foreign]);

    client.send({ jsonrpc: "2.0", id: 2, method: "run.inspect", params: { session_key: "other", run_id: run.id } });
    expect((await client.next(frame => frame.id === 2)).result?.ok).toBe(false);
    client.send({ jsonrpc: "2.0", id: 3, method: "run.acknowledge", params: { session_key: "runs-owner", run_id: run.id, revision: finished.revision } });
    expect((await client.next(frame => frame.id === 3)).result?.ok).toBe(true);
    client.send({ jsonrpc: "2.0", id: 4, method: "slash", params: { session_key: "runs-owner", command: "/runs unread" } });
    expect((await client.next(frame => frame.id === 4)).result?.runs).toEqual([]);
    const sibling = history.start({ ownerSessionId: "sibling-session", workspace: directory, kind: "agent", sourceId: "child", title: "Sibling result" });
    const unrelated = history.start({ ownerSessionId: "unrelated", workspace: join(directory, "different-project"), kind: "agent", sourceId: "other", title: "Private other project" });
    client.send({ jsonrpc: "2.0", id: 5, method: "run.list", params: { session_key: "runs-owner", scope: "workspace" } });
    const workspaceList = await client.next(frame => frame.id === 5);
    expect(workspaceList.result?.runs).toHaveLength(2);
    expect(JSON.stringify(workspaceList.result)).toContain(sibling.id);
    expect(JSON.stringify(workspaceList.result)).not.toContain(unrelated.id);
    client.send({ jsonrpc: "2.0", id: 6, method: "run.inspect", params: { session_key: "runs-owner", scope: "workspace", run_id: unrelated.id, workspace: join(directory, "different-project") } });
    expect((await client.next(frame => frame.id === 6)).result?.ok).toBe(false);
    client.send({ jsonrpc: "2.0", id: 8, method: "run.list", params: { session_key: "runs-owner", scope: "workspace", kind: "schedule", source_id: "one" } });
    expect((await client.next(frame => frame.id === 8)).result?.runs).toMatchObject([{ id: run.id }]);
    client.send({ jsonrpc: "2.0", id: 9, method: "run.list", params: { session_key: "runs-owner", kind: "unsupported" } });
    expect((await client.next(frame => frame.id === 9)).result?.ok).toBe(false);
    client.send({ jsonrpc: "2.0", id: 10, method: "run.list", params: { session_key: "runs-owner", scope: "workspace", state: "running" } });
    expect((await client.next(frame => frame.id === 10)).result?.runs).toMatchObject([{ id: sibling.id }]);
    client.send({ jsonrpc: "2.0", id: 11, method: "run.list", params: { session_key: "runs-owner", state: "unknown" } });
    expect((await client.next(frame => frame.id === 11)).result).toMatchObject({ ok: false, error: "Unknown run state" });
    const siblingDone = history.finish("sibling-session", sibling.id, "succeeded");
    client.send({ jsonrpc: "2.0", id: 7, method: "run.acknowledge", params: { session_key: "runs-owner", scope: "workspace", run_id: sibling.id, revision: siblingDone.revision } });
    expect((await client.next(frame => frame.id === 7)).result?.ok).toBe(true);
  } finally { client.close(); await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test("terminal completion reaches its attached session and archived output stays inspectable", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-terminal-completion-"));
  const history = new RunHistory(join(directory, "runs.sqlite"));
  const terminals = new TerminalRegistry({ runHistory: history });
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, "sessions") });
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({ socketPath, runtime, runHistory: history, terminalRegistry: terminals, cronLeasePath: join(directory, "cron.lease"), cronStoreFactory: () => new JobStore(join(directory, "jobs.json")) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "session.open", params: { session_key: "terminal-owner" } });
    await client.next(frame => frame.id === 1);
    const owner = runtime.sessionStatus("terminal-owner")!.id;
    const handle = terminals.open({ ownerSessionId: owner, cwd: directory, command: "bun test", id: "test-process", kind: "background" });
    handle.append("checks complete");
    handle.close(0);
    const event = await client.next(frame => frame.method === "event" && String(frame.params?.payload?.id ?? "").startsWith("run:"));
    expect(event.params?.payload?.body).toContain("terminal succeeded");
    const run = history.list(owner)[0]!;
    expect(run.unread).toBe(true);
    terminals.clear();
    client.send({ jsonrpc: "2.0", id: 2, method: "terminal.inspect", params: { terminal_id: `run:${run.id}` } });
    expect((await client.next(frame => frame.id === 2)).result?.terminal).toMatchObject({ output: "checks complete", canKill: false, exitCode: 0 });
    client.send({ jsonrpc: "2.0", id: 3, method: "terminal.output", params: { terminal_id: 'run:' + run.id, max_output_chars: 6 } });
    const firstPage = (await client.next(frame => frame.id === 3)).result?.page as { cursor: { streamId: string; offset: number }; text: string };
    expect(firstPage.text).toBe("checks");
    client.send({ jsonrpc: "2.0", id: 4, method: "terminal.output", params: { terminal_id: 'run:' + run.id, cursor: firstPage.cursor } });
    expect((await client.next(frame => frame.id === 4)).result?.page).toMatchObject({ text: " complete", hasMore: false });
    client.send({ jsonrpc: "2.0", id: 5, method: "terminal.output", params: { terminal_id: 'run:' + run.id, cursor: { streamId: run.id, offset: -1 } } });
    expect((await client.next(frame => frame.id === 5)).result?.ok).toBe(false);

  } finally { client.close(); await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test("terminal monitors announce matching events and native stop leaves the source alive", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-monitor-daemon-"));
  const history = new RunHistory(join(directory, "runs.sqlite"));
  const terminals = new TerminalRegistry();
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, "sessions") });
  const monitors = new TerminalMonitors(terminals, history, (watch, event) => server.notifyMonitorEvent(watch, event));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({ socketPath, runtime, monitors, runHistory: history, terminalRegistry: terminals, cronLeasePath: join(directory, "cron.lease"), cronStoreFactory: () => new JobStore(join(directory, "jobs.json")) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "session.open", params: { session_key: "monitor-owner" } });
    await client.next(frame => frame.id === 1);
    const owner = runtime.sessionStatus("monitor-owner")!.id;
    const terminal = terminals.open({ id: "build", ownerSessionId: owner, command: "build", cwd: directory, kind: "background" });
    const watch = monitors.start(owner, { terminalId: "build", match: "error" });
    client.send({ jsonrpc: "2.0", id: 30, method: "monitor.create", params: { terminal_id: "build", match: "warning", duration_seconds: 60 } });
    expect((await client.next(frame => frame.id === 30)).result?.monitor).toMatchObject({ state: "watching", match: "warning" });
    client.send({ jsonrpc: "2.0", id: 31, method: "monitor.create", params: { terminal_id: "build", match: "bad", duration_seconds: -1 } });
    expect((await client.next(frame => frame.id === 31)).result).toMatchObject({ ok: false, error: "Invalid monitor settings" });
    client.send({ jsonrpc: "2.0", id: 32, method: "monitor.create", params: { terminal_id: "someone-elses-process", match: "error" } });
    expect((await client.next(frame => frame.id === 32)).error).toBeDefined();

    const finished = terminals.open({ id: "finished-build", ownerSessionId: owner, command: "tests", cwd: directory, kind: "background" });
    finished.append("all checks passed"); finished.close(0);
    client.send({ jsonrpc: "2.0", id: 33, method: "monitor.create", params: { terminal_id: "finished-build", trigger: "completion" } });
    expect((await client.next(frame => frame.id === 33)).result?.monitor).toMatchObject({ trigger: "completion", state: "source-ended" });
    client.send({ jsonrpc: "2.0", id: 34, method: "monitor.create", params: { terminal_id: "build", trigger: "invalid" } });
    expect((await client.next(frame => frame.id === 34)).result?.ok).toBe(false);
    const completion = await client.next(frame => frame.method === "event" && frame.params?.payload?.title === "Command finished");
    expect(completion.params?.payload?.body).toContain("all checks passed");

    terminal.append("error: compile failed\n");
    const notification = await client.next(frame => frame.method === "event" && String(frame.params?.payload?.id ?? "").startsWith("monitor:"));
    expect(notification.params?.payload?.body).toContain("compile failed");
    client.send({ jsonrpc: "2.0", id: 21, method: "monitor.list", params: {} });
    const inventory = (await client.next(frame => frame.id === 21)).result?.monitors as Record<string, unknown>[];
    expect(inventory[0]?.id).toBe(watch.id);
    expect(inventory[0]?.stopAction).toBe("stop-watch");
    expect(inventory[0]?.events).toBeUndefined();
    client.send({ jsonrpc: "2.0", id: 22, method: "monitor.inspect", params: { monitor_id: watch.id } });
    expect((await client.next(frame => frame.id === 22)).result?.monitor).toMatchObject({ id: watch.id, events: [{ text: "error: compile failed" }] });

    expect(history.inspect(owner, watch.id)?.unread).toBe(true);
    client.send({ jsonrpc: "2.0", id: 10, method: "run.events", params: { run_id: watch.id, after_sequence: 0, limit: 1 } });
    expect((await client.next(frame => frame.id === 10)).result).toMatchObject({ ok: true, next_cursor: 1, has_more: false, events: [{ sequence: 1, text: "error: compile failed" }] });
    const reconnected = await SocketTestClient.connect(socketPath);
    try {
      reconnected.send({ jsonrpc: "2.0", id: 11, method: "session.open", params: { session_key: "monitor-owner" } });
      await reconnected.next(frame => frame.id === 11);
      reconnected.send({ jsonrpc: "2.0", id: 12, method: "run.events", params: { run_id: watch.id } });
      expect((await reconnected.next(frame => frame.id === 12)).result?.events).toMatchObject([{ sequence: 1 }]);
      reconnected.send({ jsonrpc: "2.0", id: 13, method: "run.events", params: { run_id: watch.id, after_sequence: 1 } });
      expect((await reconnected.next(frame => frame.id === 13)).result?.events).toEqual([]);
      reconnected.send({ jsonrpc: "2.0", id: 14, method: "session.open", params: { session_key: "other-monitor-owner" } });
      await reconnected.next(frame => frame.id === 14);
      reconnected.send({ jsonrpc: "2.0", id: 15, method: "run.events", params: { run_id: watch.id } });
      expect((await reconnected.next(frame => frame.id === 15)).result).toMatchObject({ ok: false, error: "Unknown run" });
      reconnected.send({ jsonrpc: "2.0", id: 23, method: "monitor.stop", params: { monitor_id: watch.id } });
      expect((await reconnected.next(frame => frame.id === 23)).error).toBeDefined();
      expect(monitors.inspect(owner, watch.id).state).toBe("watching");

    } finally { reconnected.close(); }

    client.send({ jsonrpc: "2.0", id: 2, method: "slash", params: { command: `/monitors stop ${watch.id}` } });
    expect((await client.next(frame => frame.id === 2)).result?.ok).toBe(true);
    expect(terminals.inspect(owner, "build")?.running).toBe(true);
    client.send({ jsonrpc: "2.0", id: 3, method: "slash", params: { command: "/monitors list" } });
    expect((await client.next(frame => frame.id === 3)).result?.monitors).toMatchObject([{ state: "stopped", stopAction: null }, { state: "watching", stopAction: "stop-watch" }, { state: "source-ended", stopAction: null }]);
  } finally { client.close(); monitors.close(); await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test("manual cron execution preserves background origin through daemon admission", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-cron-origin-"));
  const origins: unknown[] = [];
  const runtime = new InMemoryDaemonRuntime({
    async *run(_session, _text, _signal, controls) {
      origins.push(controls?.origin);
      yield { type: "text_part", payload: { text: "scheduled result" } };
    },
  }, { currentProjectDirectory: directory, sessionDirectory: join(directory, "sessions") });
  const store = new JobStore(join(directory, "jobs.json"));
  const server = new DaemonServer({ socketPath: join(directory, "daemon.sock"), runtime, cronStoreFactory: () => store,
    cronLeasePath: join(directory, "cron.lease"), cronArchiveDirectory: join(directory, "archive") });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, "daemon.sock"));
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "session.open", params: { session_key: "cron-origin" } });
    await client.next(frame => frame.id === 1);
    client.send({ jsonrpc: "2.0", id: 2, method: "slash", params: { command: '/cron add --schedule "0 9 * * 1" --prompt "report"' } });
    const added = await client.next(frame => frame.id === 2);
    const job = added.result?.job as { id: string };
    client.send({ jsonrpc: "2.0", id: 3, method: "slash", params: { command: `/cron run ${job.id}` } });
    expect((await client.next(frame => frame.id === 3)).result?.ok).toBe(true);
    expect(origins).toEqual(['schedule']);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test.each([[false, 'completed'], [true, 'completed'], [false, 'output_limit'], [true, 'tool_budget_exhausted']] as const)("configured monitor reactions preserve outcome and usage (children: %s, reason: %s)", async (children, reason) => {
  const { ReactionMailbox } = await import('../src/runtime/reactionMailbox.js');
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-monitor-reaction-'));
  const history = new RunHistory(join(directory, 'runs.sqlite'));
  const mailbox = new ReactionMailbox(join(directory, 'reactions.sqlite'));
  const terminals = new TerminalRegistry();
  const turns: { origin: unknown; text: string }[] = [];
  const runtime = new InMemoryDaemonRuntime({ async *run(_session, text, _signal, controls) {
    turns.push({ origin: controls?.origin, text });
    if (children) {
      yield { type: 'subagent_event', payload: { agent_id: 'child', event: { type: 'turn_begin' } } };
      yield { type: 'subagent_event', payload: { agent_id: 'child', input_tokens: 40, output_tokens: 10 } };
      yield { type: 'subagent_event', payload: { agent_id: 'child', input_tokens: 40, output_tokens: 10 } };
    }
    yield { type: 'text_part', payload: { text: 'Investigated monitor evidence.' } };
    yield { type: 'status_update', payload: { usage: { inputTokens: 120, outputTokens: 30 }, usage_complete: true, stop_reason: reason } };
  } }, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const monitors = new TerminalMonitors(terminals, history, (watch, event) => server.notifyMonitorEvent(watch, event), undefined, mailbox);
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), projectDirectory: directory, runtime, runHistory: history, reactionMailbox: mailbox,
    monitors, terminalRegistry: terminals, cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')), cronLeasePath: join(directory, 'cron.lease') });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'reaction-owner' } });
    await client.next(frame => frame.id === 1);
    const owner = runtime.sessionStatus('reaction-owner')!.id;
    const buildDirectory = join(directory, 'build'); await mkdir(buildDirectory);
    const terminal = terminals.open({ ownerSessionId: owner, id: 'build', cwd: buildDirectory, command: 'build', kind: 'background' });
    const registry = new ToolRegistry();
    registerMonitorTools(registry, monitors);
    const request = { id: 'watch', type: 'function' as const, function: { name: 'monitor_terminal', arguments: { terminal_id: 'build', match: 'error', react: true, max_reactions: 1, reaction_timeout_seconds: 5 } } };
    await expect(registry.execute(request, { sessionId: owner, metadata: { goal_turn_human: false } })).rejects.toThrow('direct user');
    const watch = JSON.parse(await registry.execute(request, { sessionId: owner, metadata: { goal_turn_human: true } }));
    expect(watch.reaction).toEqual({ maxReactions: 1, maxDurationMs: 5000 });
    expect(watch.reactionHealth.state).toBe('waiting');
    terminal.append('error: build failed\n');
    await client.next(frame => frame.method === 'event' && frame.params?.type === 'turn_end');
    expect(turns).toHaveLength(1);
    expect(turns[0]?.origin).toBe('monitor');
    expect(turns[0]?.text).toContain('error: build failed');
    terminal.append('error: another failure\n');
    await Bun.sleep(20);
    expect(turns).toHaveLength(1);
    expect(runtime.sessionStatus('reaction-owner')?.messages.some(message => String(message.content).includes('Investigated'))).toBe(true);
    client.send({ jsonrpc: '2.0', id: 2, method: 'run.inspect', params: { run_id: watch.id } });
    expect((await client.next(frame => frame.id === 2)).result?.run).toMatchObject({ reaction_health: { state: 'exhausted', attempts: 1, maxReactions: 1, lastOutcome: reason === 'completed' ? 'completed' : 'failed', usage: { inputTokens: children ? 160 : 120, outputTokens: children ? 40 : 30, complete: false } } });
    const reaction = history.list(owner).find(run => run.sourceId !== 'build' && run.title.startsWith('Monitor reaction:'));
    expect(reaction).toMatchObject({ state: reason === 'completed' ? 'succeeded' : 'failed', output: 'Investigated monitor evidence.' });
    if (reason !== 'completed') expect(reaction?.error).toContain(reason);
  } finally { client.close(); monitors.close(); await server.stop(); mailbox.close(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test('schedule RPC manages workspace jobs without exposing other projects', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-schedule-rpc-'));
  const socketPath = join(directory, 'daemon.sock');
  const store = new JobStore(join(directory, 'jobs.json'));
  store.add(new CronJob({ id: 'local', prompt: 'review', projectRoot: directory, schedule: '0 9 * * *', paused: true }));
  store.add(new CronJob({ id: 'foreign', prompt: 'private', projectRoot: join(directory, 'other'), schedule: '0 9 * * *', paused: true }));
  const server = new DaemonServer({ socketPath, legacyScheduleDirectory: join(directory, 'legacy'), projectDirectory: directory, cronArchiveDirectory: join(directory, 'archive'), cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') }) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'schedule.list', params: {} });
    const listed = (await client.next(frame => frame.id === 1)).result as { jobs: { id: string }[] };
    expect(listed.jobs.map(job => job.id)).toEqual(['local']);
    const recordsBeforePreview = store.listJobs().map(job => job.toRecord());
    client.send({ jsonrpc: '2.0', id: 101, method: 'schedule.preview', params: { at: '2099-01-01T09:00:00+03:00' } });
    expect((await client.next(frame => frame.id === 101)).result).toEqual({ ok: true, next_run_at: '2099-01-01T06:00:00.000Z', timezone: 'UTC' });
    client.send({ jsonrpc: '2.0', id: 102, method: 'schedule.preview', params: { schedule: '0 9 * * *', timezone: 'Asia/Tokyo' } });
    const preview = (await client.next(frame => frame.id === 102)).result;
    expect(preview?.ok).toBe(true);
    expect(String(preview?.next_run_at)).toContain('T00:00:00.000Z');
    for (const [offset, params] of [{ schedule: 'bad' }, { at: '2000-01-01T00:00:00Z' }, { interval_seconds: 0 }, { schedule: '0 9 * * *', timezone: 'Invalid/Zone' }, { interval_seconds: 30, at: '2099-01-01T00:00:00Z' }].entries()) {
      client.send({ jsonrpc: '2.0', id: 110 + offset, method: 'schedule.preview', params });
      expect((await client.next(frame => frame.id === 110 + offset)).error).toBeDefined();
    }
    expect(store.listJobs().map(job => job.toRecord())).toEqual(recordsBeforePreview);
    let id = 2;
    for (const action of ['inspect', 'pause', 'resume', 'cancel', 'run']) {
      client.send({ jsonrpc: '2.0', id, method: `schedule.${action}`, params: { schedule_id: 'foreign' } });
      expect((await client.next(frame => frame.id === id)).result).toMatchObject({ ok: false, error: 'Schedule not found in this workspace' });
      id++;
    }
    client.send({ jsonrpc: '2.0', id: 10, method: 'schedule.resume', params: { schedule_id: 'local' } });
    expect((await client.next(frame => frame.id === 10)).result).toMatchObject({ ok: true, job: { paused: false, execution_state: 'idle' } });
    expect(store.get('local')?.nextRunAt).toBeDefined();
    client.send({ jsonrpc: '2.0', id: 11, method: 'schedule.pause', params: { schedule_id: 'local' } });
    expect((await client.next(frame => frame.id === 11)).result).toMatchObject({ ok: true, job: { paused: true } });
    client.send({ jsonrpc: '2.0', id: 12, method: 'schedule.cancel', params: { schedule_id: 'local' } });
    expect((await client.next(frame => frame.id === 12)).result).toMatchObject({ ok: true, requested: false });
    client.send({ jsonrpc: "2.0", id: 13, method: "schedule.create", params: { prompt: "new review", max_runs: 3, expires_at: "2099-01-01T00:00:00Z", schedule: "0 8 * * *", timezone: "America/New_York", paused: true, timeout_seconds: 45, max_retries: 0, missed_run_policy: 'skip', misfire_grace_seconds: 45 } });
    const created = (await client.next(frame => frame.id === 13)).result as { ok: boolean; job: { id: string; revision: string } };
    expect(created.ok).toBe(true);
    expect(created.job).toMatchObject({ missed_run_policy: 'skip', misfire_grace_seconds: 45, overlap_policy: 'forbid' });
    expect(store.get(created.job.id)?.timeoutMs).toBe(45000);
    expect(store.get(created.job.id)?.maxRetries).toBe(0);
    expect(store.get(created.job.id)?.timezone).toBe('America/New_York');
    expect(store.get(created.job.id)?.projectRoot).toBe(await realpath(directory));
    expect(created.job).toMatchObject({ max_runs: 3, runs_started: 0, expires_at: "2099-01-01T00:00:00.000Z" });
    client.send({ jsonrpc: "2.0", id: 14, method: "schedule.update", params: { schedule_id: created.job.id, revision: created.job.revision, prompt: "edited review", schedule: "0 10 * * *", paused: true } });
    expect((await client.next(frame => frame.id === 14)).result).toMatchObject({ ok: true, job: { prompt: "edited review", timezone: "America/New_York", expires_at: "2099-01-01T00:00:00.000Z" } });
    client.send({ jsonrpc: "2.0", id: 15, method: "schedule.update", params: { schedule_id: created.job.id, revision: created.job.revision, prompt: "stale edit", schedule: "0 11 * * *", paused: true } });
    expect(store.get(created.job.id)?.missedRunPolicy).toBe('skip');
    expect(store.get(created.job.id)?.misfireGraceSeconds).toBe(45);
    expect((await client.next(frame => frame.id === 15)).result).toMatchObject({ ok: false, error: "Schedule changed; refresh before editing" });
    client.send({ jsonrpc: "2.0", id: 16, method: "schedule.create", params: { prompt: "invalid", schedule: "bad", paused: true } });
    expect((await client.next(frame => frame.id === 16)).error).toBeDefined();
    const outbox = new DeliveryOutbox(join(directory, "archive", "deliveries.sqlite"));
    const deliveryId = outbox.enqueue("local", { platform: "test" }, "saved output", "archive.md");
    client.send({ jsonrpc: "2.0", id: 30, method: "schedule.deliveries", params: { schedule_id: "local" } });
    const deliveries = (await client.next(frame => frame.id === 30)).result;
    expect(deliveries?.deliveries).toMatchObject([{ id: deliveryId, state: "pending" }]);
    expect(JSON.stringify(deliveries)).not.toContain("saved output");
    client.send({ jsonrpc: "2.0", id: 31, method: "schedule.delivery.inspect", params: { schedule_id: "foreign", delivery_id: deliveryId } });
    expect((await client.next(frame => frame.id === 31)).result?.ok).toBe(false);
    client.send({ jsonrpc: "2.0", id: 32, method: "schedule.delivery.send", params: { schedule_id: "local", delivery_id: deliveryId } });
    expect((await client.next(frame => frame.id === 32)).result?.ok).toBe(false);
    expect(outbox.inspect("local", deliveryId)?.state).toBe("pending");
    client.send({ jsonrpc: "2.0", id: 40, method: "schedule.create", params: { prompt: "interval work", interval_seconds: 30, paused: true } });
    const intervalResult = (await client.next(frame => frame.id === 40)).result as { job: { id: string } };
    expect(store.get(intervalResult.job.id)?.intervalSeconds).toBe(30);
    expect(store.get(intervalResult.job.id)?.oneshot).toBe(false);
    client.send({ jsonrpc: "2.0", id: 41, method: "schedule.resume", params: { schedule_id: intervalResult.job.id } });
    expect((await client.next(frame => frame.id === 41)).result?.ok).toBe(true);
    expect(new Date(store.get(intervalResult.job.id)!.nextRunAt!).getTime()).toBeGreaterThan(Date.now());
    const legacy = new LegacyScheduler({ directory: join(directory, "legacy") });
    await legacy.createTrigger({ id: "old-interval", owner: "user", schedule: { kind: "interval", intervalSeconds: 60 }, payload: { id: "task", objective: "Review legacy", creatorId: "user", dependencies: [] } });
    client.send({ jsonrpc: "2.0", id: 50, method: "slash", params: { command: "/schedules legacy" } });
    expect((await client.next(frame => frame.id === 50)).result?.triggers).toMatchObject([{ id: "old-interval", supported: true }]);
    client.send({ jsonrpc: "2.0", id: 51, method: "slash", params: { command: "/schedules migrate old-interval" } });
    expect((await client.next(frame => frame.id === 51)).result?.job).toMatchObject({ paused: true, interval_seconds: 60, prompt: "Review legacy" });
    expect((await legacy.load()).triggers.get("old-interval")?.enabled).toBe(false);
    const countBefore = store.listJobs().length;
    client.send({ jsonrpc: "2.0", id: 55, method: "schedule.create", params: { prompt: "invalid zone", schedule: "0 9 * * *", timezone: "Invalid/Zone", paused: true } });
    expect((await client.next(frame => frame.id === 55)).error).toBeDefined();
    client.send({ jsonrpc: "2.0", id: 52, method: "schedule.create", params: { prompt: "impossible date", at: "2099-02-29T12:00:00Z", paused: true } });
    expect((await client.next(frame => frame.id === 52)).error).toBeDefined();
    client.send({ jsonrpc: "2.0", id: 53, method: "slash", params: { command: '/cron add --at "2099-02-29T12:00:00Z" --prompt "Impossible date"' } });
    expect((await client.next(frame => frame.id === 53)).result?.ok).toBe(false);
    client.send({ jsonrpc: "2.0", id: 54, method: "slash", params: { command: '/cron add --at "2000-01-01T12:00:00Z" --prompt "Past date"' } });
    expect((await client.next(frame => frame.id === 54)).result?.ok).toBe(false);
    expect(store.listJobs().length).toBe(countBefore);
    client.send({ jsonrpc: "2.0", id: 17, method: "schedule.create", params: { prompt: "invalid date", at: "2099-01-01", paused: true } });
    expect((await client.next(frame => frame.id === 17)).error).toBeDefined();
    expect(store.listJobs().length).toBe(countBefore);
    client.send({ jsonrpc: "2.0", id: 18, method: "schedule.create", params: { prompt: "one shot", at: "2099-01-01T09:00:00Z", paused: true } });
    expect((await client.next(frame => frame.id === 18)).result).toMatchObject({ ok: true, job: { oneshot: true, next_run_at: "2099-01-01T09:00:00.000Z" } });
    client.send({ jsonrpc: "2.0", id: 56, method: "slash", params: { command: '/cron add --schedule "0 9 * * *" --timezone "Asia/Tokyo" --prompt "Local review"' } });
    expect((await client.next(frame => frame.id === 56)).result).toMatchObject({ ok: true, job: { timezone: "Asia/Tokyo" } });
    store.update('local', { paused: true, metadata: { execution_recovery_required: true, execution_receipt: { state: 'completed', occurrence: '2026-09-01T09:00:00Z' } } });
    client.send({ jsonrpc: '2.0', id: 57, method: 'schedule.resume', params: { schedule_id: 'local' } });
    expect((await client.next(frame => frame.id === 57)).result?.ok).toBe(true);
    expect(store.get('local')?.metadata.execution_receipt).toBeUndefined();
    expect(store.get('local')?.metadata.previous_execution_receipt).toMatchObject({ state: 'completed' });
    expect(store.get(created.job.id)?.prompt).toBe("edited review");
    expect(store.get("foreign")?.paused).toBe(true);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('model schedule requests share workspace scope and optimistic edits with RPC jobs', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-model-schedule-'));
  const store = new JobStore(join(directory, 'jobs.json'));
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  try {
    const session = await runtime.openSession('model');
    store.add(new CronJob({ id: 'other', prompt: 'private', projectRoot: join(directory, 'other'), paused: true, schedule: '0 9 * * *' }));
    await expect(server.scheduleToolRequest('unknown', 'list', {})).rejects.toThrow('active workspace');
    const created = await server.scheduleToolRequest(session.id, 'create', { prompt: 'Review changes', paused: true, interval_seconds: 60 });
    const job = created.job as { id: string; revision: string };
    expect(store.get(job.id)?.projectRoot).toBe(await realpath(directory));
    expect((await server.scheduleToolRequest(session.id, 'list', {})).jobs).toMatchObject([{ id: job.id }]);
    expect(await server.scheduleToolRequest(session.id, 'resume', { schedule_id: 'other' })).toMatchObject({ ok: false, error: 'Schedule not found in this workspace' });
    store.update(job.id, { prompt: 'An intervening UI edit' });
    expect(await server.scheduleToolRequest(session.id, 'update', { schedule_id: job.id, revision: job.revision, prompt: 'stale', paused: true, interval_seconds: 60 })).toMatchObject({ ok: false, error: 'Schedule changed; refresh before editing' });
    expect(store.get(job.id)?.prompt).toBe('An intervening UI edit');
    const abort = new AbortController(); abort.abort(new Error('cancelled'));
    await expect(server.scheduleToolRequest(session.id, 'resume', { schedule_id: job.id }, abort.signal)).rejects.toThrow('cancelled');
    expect(store.get(job.id)?.paused).toBe(true);
  } finally { await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('model schedule run-now is isolated and propagates caller cancellation without successful delivery', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-model-run-'));
  const store = new JobStore(join(directory, 'jobs.json'));
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  await server.start();
  const abort = new AbortController();
  try {
    const session = await runtime.openSession('caller');
    store.add(new CronJob({ id: 'manual', prompt: 'Run checks', projectRoot: directory, paused: true, schedule: '0 9 * * *' }));
    const result = server.scheduleToolRequest(session.id, 'run', { schedule_id: 'manual' }, abort.signal).then(value => ({ value }), error => ({ error }));
    await waitFor(() => runner.runs === 1);
    expect(session.activeTurnId).toBe('');
    expect(runtime.sessionStatus('cron:manual')?.activeTurnId).not.toBe('');
    await expect(server.scheduleToolRequest(session.id, 'run', { schedule_id: 'manual' })).rejects.toThrow('already running');
    abort.abort();
    expect(await result).toHaveProperty('error');
    expect(store.get('manual')?.lastRunAt).toBeUndefined();
    expect(existsSync(join(directory, 'archive'))).toBe(false);
    expect(await server.scheduleToolRequest(session.id, 'inspect', { schedule_id: 'manual' })).toMatchObject({ ok: true, job: { execution_state: 'idle' } });
  } finally { abort.abort(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('scheduled model runs use their stored project instead of the daemon default', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-schedule-project-'));
  const other = join(directory, 'other'); await mkdir(other);
  const seen: string[] = [];
  const runner: TurnRunner = { async *run(session) { seen.push(session.cwd); yield { type: 'text_part', payload: { text: session.cwd } }; } };
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const store = new JobStore(join(directory, 'jobs.json'));
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  await server.start();
  try {
    const caller = await runtime.openSession('caller', undefined, { cwd: other });
    store.add(new CronJob({ id: 'project-job', prompt: 'Check project', projectRoot: other, paused: true, schedule: '0 9 * * *' }));
    const result = await server.scheduleToolRequest(caller.id, 'run', { schedule_id: 'project-job' });
    expect(result.ok).toBe(true);
    expect(seen).toEqual([await realpath(other)]);
    expect(runtime.sessionStatus('cron:project-job')?.cwd).toBe(await realpath(other));
    expect(caller.messages).toHaveLength(0);
  } finally { await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('scheduled runs reject a conflicting existing session without moving it or calling the provider', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-schedule-conflict-'));
  const other = join(directory, 'other'); await mkdir(other);
  let runs = 0;
  const runner: TurnRunner = { async *run() { runs++; yield { type: 'text_part', payload: { text: 'unexpected' } }; } };
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const store = new JobStore(join(directory, 'jobs.json'));
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  await server.start();
  try {
    const caller = await runtime.openSession('caller', undefined, { cwd: other });
    const existing = await runtime.openSession('occupied');
    const original = existing.cwd;
    store.add(new CronJob({ id: 'conflicting', workspaceId: 'occupied', prompt: 'Check project', projectRoot: other, paused: true, schedule: '0 9 * * *' }));
    await expect(server.scheduleToolRequest(caller.id, 'run', { schedule_id: 'conflicting' })).rejects.toThrow('another workspace');
    expect(existing.cwd).toBe(original);
    expect(existing.messages).toHaveLength(0);
    expect(runs).toBe(0);
    expect(store.get('conflicting')?.lastRunAt).toBeUndefined();
  } finally { await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('background session open preserves project even when another opener is queued first', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-session-project-lock-'));
  const other = join(directory, 'other'); await mkdir(other);
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  try {
    const first = runtime.openSession('shared', undefined, { cwd: directory });
    const second = runtime.openSession('shared', undefined, { cwd: other, preserveProject: true });
    await expect(second).rejects.toThrow('another workspace');
    expect((await first).cwd).toBe(directory);
    expect(runtime.sessionStatus('shared')?.cwd).toBe(directory);
  } finally { await runtime.shutdown(); await rm(directory, { recursive: true, force: true }); }
});

test('reopening a saved session recovers durable monitor evidence not offered before restart', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-monitor-restart-'));
  const sessionDirectory = join(directory, 'sessions');
  const original = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory });
  const owner = await original.openSession('original');
  await original.submitTurn(owner.sessionKey, 'Create durable monitor owner history', () => {});
  await original.flushSessions();
  await original.shutdown();
  const historyPath = join(directory, 'runs.sqlite'), mailboxPath = join(directory, 'mailbox.sqlite');
  const firstHistory = new RunHistory(historyPath);
  const run = firstHistory.start({ ownerSessionId: owner.id, workspace: directory, kind: 'monitor', sourceId: 'terminal', title: 'watch' });
  firstHistory.appendEvent(owner.id, run.id, { sequence: 1, text: 'build failed', at: Date.now() }, 'build failed');
  firstHistory.close();
  const firstMailbox = new ReactionMailbox(mailboxPath);
  firstMailbox.configure({ owner: owner.id, runId: run.id, expiresAt: Date.now() + 60000, maxReactions: 2, maxDurationMs: 1000 });
  firstMailbox.close();
  const history = new RunHistory(historyPath), mailbox = new ReactionMailbox(mailboxPath);
  let calls = 0;
  const runner: TurnRunner = { async *run(_session, prompt) { calls++; expect(prompt).toContain('build failed'); yield { type: 'text_part', payload: { text: 'Recovered reaction' } }; } };
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory });
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), projectDirectory: directory, runtime, runHistory: history, reactionMailbox: mailbox, cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')), cronLeasePath: join(directory, 'lease') });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { resume_session_id: owner.id } });
    await client.next(frame => frame.id === 1);
    await waitFor(() => mailbox.inspect(owner.id, run.id)?.lastOutcome === 'completed');
    expect(calls).toBe(1);
    client.send({ jsonrpc: '2.0', id: 2, method: 'initialize', params: { resume_session_id: owner.id } });
    await client.next(frame => frame.id === 2);
    expect(mailbox.inspect(owner.id, run.id)).toMatchObject({ attempts: 1, pendingEvents: 0 });
    expect(calls).toBe(1);
  } finally { client.close(); await server.stop(); history.close(); mailbox.close(); await rm(directory, { recursive: true, force: true }); }
});

test('run cancellation routes to the live terminal and rejects stale or foreign rows', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-run-cancel-'));
  const history = new RunHistory(join(directory, 'runs.sqlite'));
  const terminals = new TerminalRegistry({ runHistory: history });
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const session = await runtime.openSession('owner');
  let stopped = 0;
  const terminal = terminals.open({ id: 'process', ownerSessionId: session.id, cwd: directory, command: 'build', kind: 'background', control: { kill: async () => { stopped++ } } });
  const run = history.list(session.id)[0]!;
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, runHistory: history, terminalRegistry: terminals, cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')), cronLeasePath: join(directory, 'lease') });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'run.inspect', params: { session_key: 'owner', run_id: run.id } });
    expect((await client.next(frame => frame.id === 1)).result?.run).toMatchObject({ cancel_label: 'Stop process' });
    client.send({ jsonrpc: '2.0', id: 2, method: 'run.cancel', params: { session_key: 'other', run_id: run.id, revision: run.revision } });
    expect((await client.next(frame => frame.id === 2)).result?.ok).toBe(false);
    client.send({ jsonrpc: '2.0', id: 3, method: 'run.cancel', params: { session_key: 'owner', run_id: run.id, revision: run.revision + 1 } });
    expect((await client.next(frame => frame.id === 3)).result?.error).toBe('Run changed; refresh before cancelling');
    expect(stopped).toBe(0);
    client.send({ jsonrpc: '2.0', id: 4, method: 'run.cancel', params: { session_key: 'owner', run_id: run.id, revision: run.revision } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: true, requested: true });
    expect(stopped).toBe(1);
    expect(history.inspect(session.id, run.id)?.state).toBe('running');
    terminal.close(0);
    client.send({ jsonrpc: '2.0', id: 5, method: 'run.inspect', params: { session_key: 'owner', run_id: run.id } });
    expect((await client.next(frame => frame.id === 5)).result?.run).toMatchObject({ cancel_label: null, state: 'cancelled' });
  } finally { client.close(); await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test('Runs cancels only the exact active schedule execution', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-schedule-run-control-')));
  const history = new RunHistory(join(directory, 'runs.sqlite'));
  const store = new JobStore(join(directory, 'jobs.json'));
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const caller = await runtime.openSession('caller');
  store.add(new CronJob({ id: 'job', prompt: 'Check', projectRoot: directory, paused: true, schedule: '0 9 * * *' }));
  const stale = history.start({ ownerSessionId: caller.id, workspace: directory, kind: 'schedule', sourceId: 'job', title: 'Older incomplete record' });
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), projectDirectory: directory, runtime, runHistory: history, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    const running = server.scheduleToolRequest(caller.id, 'run', { schedule_id: 'job' }).catch(error => error);
    await waitFor(() => runner.runs === 1);
    const current = history.listWorkspace(directory).find(run => run.id !== stale.id)!;
    client.send({ jsonrpc: '2.0', id: 1, method: 'run.cancel', params: { session_key: 'caller', scope: 'workspace', run_id: stale.id, revision: stale.revision } });
    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({ ok: false, error: 'This run has no active cancellation control' });
    client.send({ jsonrpc: '2.0', id: 2, method: 'run.cancel', params: { session_key: 'caller', scope: 'workspace', run_id: current.id, revision: current.revision } });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, requested: true });
    expect(await running).toBeInstanceOf(Error);
    expect(history.inspect(current.ownerSessionId, current.id)?.state).toBe('cancelled');
    expect(store.get('job')?.paused).toBe(true);
  } finally { client.close(); await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test('one connection can inspect and cancel its pending schedule.run', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-schedule-live-control-')));
  const history = new RunHistory(join(directory, 'runs.sqlite'));
  const store = new JobStore(join(directory, 'jobs.json'));
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  store.add(new CronJob({ id: 'job', prompt: 'Check', projectRoot: directory, paused: true, schedule: '0 9 * * *' }));
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), projectDirectory: directory, runtime, runHistory: history, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'caller' } });
    await client.next(frame => frame.id === 1);
    client.send({ jsonrpc: '2.0', id: 2, method: 'schedule.run', params: { schedule_id: 'job' } });
    await waitFor(() => runner.runs === 1);
    client.send({ jsonrpc: '2.0', id: 3, method: 'run.list', params: { scope: 'workspace' } });
    const listed = (await client.next(frame => frame.id === 3)).result;
    expect(listed?.runs).toMatchObject([{ kind: 'schedule', state: 'running' }]);
    const run = history.listWorkspace(directory)[0]!;
    expect(run.ownerSessionId).not.toBe(runtime.sessionStatus('caller')?.id);
    client.send({ jsonrpc: '2.0', id: 4, method: 'run.inspect', params: { scope: 'workspace', run_id: run.id } });
    expect((await client.next(frame => frame.id === 4)).result?.run).toMatchObject({ cancel_label: 'Cancel run' });
    client.send({ jsonrpc: '2.0', id: 5, method: 'run.cancel', params: { scope: 'workspace', run_id: run.id, revision: run.revision + 1 } });
    expect((await client.next(frame => frame.id === 5)).result).toMatchObject({ ok: false, error: 'Run changed; refresh before cancelling' });
    client.send({ jsonrpc: '2.0', id: 6, method: 'run.cancel', params: { scope: 'workspace', run_id: run.id, revision: run.revision } });
    expect((await client.next(frame => frame.id === 6)).result).toMatchObject({ ok: true, requested: true });
    expect((await client.next(frame => frame.id === 2)).error).toBeDefined();
    expect(history.inspect(run.ownerSessionId, run.id)?.state).toBe('cancelled');
    expect(runtime.sessionStatus('caller')?.activeTurnId).toBe('');
    expect(client.seen(eventFrame('text_part'))).toBe(false);
    expect(client.seen(eventFrame('cron_event'))).toBe(true);
  } finally { client.close(); await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test('hooks slash reports the active workspace configuration without running commands', async () => {
  const { workspaceShellHooks } = await import('../src/extensions/workspaceHooks.js');
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-hook-slash-'));
  await Bun.write(join(directory, 'config.json'), JSON.stringify({ hooks: { PreToolUse: [{ command: 'touch forbidden' }] } }));
  const factory = workspaceShellHooks({ home: directory, allowWorkspace: false, reportError: () => {} });
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'), hookRunnerForSession: session => factory(session.cwd) });
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')), cronLeasePath: join(directory, 'lease') });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'hooks' } });
    await client.next(frame => frame.id === 1);
    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/hooks' } });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, inspection: { workspaceTrusted: false, hooks: [{ command: 'touch forbidden', blocking: true }], recent: [] } });
    expect(await Bun.file(join(directory, 'forbidden')).exists()).toBe(false);
    client.send({ jsonrpc: '2.0', id: 3, method: 'slash', params: { command: '/hooks run' } });
    expect((await client.next(frame => frame.id === 3)).result?.ok).toBe(false);
    client.send({ jsonrpc: '2.0', id: 4, method: 'slash', params: { command: '/hooks preview PreToolUse ReadFile' } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: true, preview: { event: 'tool_permission_check', executed: false, matched: 1, hooks: [{ blocking: true, matches: true }] } });
    expect(await Bun.file(join(directory, 'forbidden')).exists()).toBe(false);
    client.send({ jsonrpc: '2.0', id: 5, method: 'slash', params: { command: '/hooks preview unknown_event' } });
    expect((await client.next(frame => frame.id === 5)).result).toMatchObject({ ok: false });
    client.send({ jsonrpc: '2.0', id: 6, method: 'slash', params: { command: '/hooks failures PreToolUse' } });
    expect((await client.next(frame => frame.id === 6)).result).toMatchObject({ ok: true, failures: { event: 'tool_permission_check', failed: 0, denied: 0, results: [] } });
    expect(await Bun.file(join(directory, 'forbidden')).exists()).toBe(false);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('agent settings RPC persists profile mappings without exposing credentials and rejects stale edits', async () => {
  const { AgentSettingsStore } = await import('../src/agents/settingsStore.js');
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-config-'));
  const profiles = new ProfileStore(join(directory, 'profiles.json'));
  profiles.save({ name: 'work', provider: 'openai', apiKey: 'private-test-key', baseUrl: 'https://example.invalid/v1', model: 'gpt-4o' });
  const store = new AgentSettingsStore(join(directory, 'settings.sqlite'));
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, profileStore: profiles, agentSettingsStore: store, cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')), cronLeasePath: join(directory, 'lease') });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'agent.settings.get' });
    const result = (await client.next(frame => frame.id === 1)).result;
    expect(JSON.stringify(result)).not.toContain('private-test-key');
    expect(result).toMatchObject({ revision: 0, profiles: expect.arrayContaining([{ name: 'work', provider: 'openai', model: 'gpt-4o' }]) });
    client.send({ jsonrpc: '2.0', id: 2, method: 'agent.settings.save', params: { revision: 0, settings: { light: { model: 'gpt-4o', provider_profile: 'work' } } } });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, revision: 1 });
    client.send({ jsonrpc: '2.0', id: 3, method: 'agent.settings.save', params: { revision: 0, settings: { smart: 'other' } } });
    expect((await client.next(frame => frame.id === 3)).error).toBeDefined();
    client.send({ jsonrpc: '2.0', id: 4, method: 'agent.settings.save', params: { revision: 1, settings: { smart: { model: 'gpt-4o', provider_profile: 'missing' } } } });
    expect((await client.next(frame => frame.id === 4)).error).toBeDefined();
    expect(store.read().revision).toBe(1);
    client.send({ jsonrpc: '2.0', id: 5, method: 'agent.settings.options', params: { provider_profile: 'work', model: 'gpt-5' } });
    const options = (await client.next(frame => frame.id === 5)).result;
    expect(options).toMatchObject({ ok: true, model: 'gpt-5', reasoning_efforts: expect.arrayContaining(['high']) });
    expect(JSON.stringify(options)).not.toContain('private-test-key');
    client.send({ jsonrpc: '2.0', id: 6, method: 'agent.settings.options', params: { provider_profile: 'missing' } });
    expect((await client.next(frame => frame.id === 6)).error).toBeDefined();
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('agent settings use the selected Codex profile live reasoning ladder for choices and saves', async () => {
  const { AgentSettingsStore } = await import('../src/agents/settingsStore.js');
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-tier-reasoning-'));
  const profiles = new ProfileStore(join(directory, 'profiles.json'));
  for (const name of ['deep', 'fast']) profiles.save({ name, provider: 'openai-codex', apiKey: '', baseUrl: `https://${name}.invalid`, model: 'future-model' });
  const requested: string[] = [];
  let releaseCatalog!: () => void;
  const catalogGate = new Promise<void>(resolve => { releaseCatalog = resolve; });
  const store = new AgentSettingsStore(join(directory, 'settings.sqlite'));
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, profileStore: profiles, agentSettingsStore: store,
    cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')), cronLeasePath: join(directory, 'lease'),
    codexModelCatalog: async profile => {
      requested.push(profile.name);
      if (profile.name === 'deep') await catalogGate;
      return [{ id: 'future-model', displayName: undefined, contextLimit: undefined, harnessCoupled: false,
        defaultReasoningLevel: 'high', reasoningLevels: [{ effort: profile.name === 'deep' ? 'ultra' : 'low', description: undefined }] }];
    },
  });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    for (const [id, profile, effort] of [[1, 'deep', 'ultra'], [2, 'fast', 'low']] as const) {
      client.send({ jsonrpc: '2.0', id, method: 'agent.settings.options', params: { provider_profile: profile, model: 'future-model' } });
      if (id === 1) {
        client.send({ jsonrpc: '2.0', id: 10, method: 'agent.settings.get' });
        expect((await client.next(frame => frame.id === 10)).result).toMatchObject({ ok: true, revision: 0 });
        releaseCatalog();
      }
      expect((await client.next(frame => frame.id === id)).result).toMatchObject({ reasoning_efforts: ["off", effort] });
    }
    client.send({ jsonrpc: '2.0', id: 3, method: 'agent.settings.save', params: { revision: 0, settings: { smart: { model: 'future-model', provider_profile: 'deep', reasoning_effort: 'ultra' } } } });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ ok: true, revision: 1 });
    client.send({ jsonrpc: '2.0', id: 4, method: 'agent.settings.save', params: { revision: 1, settings: { smart: { model: 'future-model', provider_profile: 'fast', reasoning_effort: 'ultra' } } } });
    expect((await client.next(frame => frame.id === 4)).error).toBeDefined();
    expect(store.read().revision).toBe(1);
    expect(requested).toEqual(['deep', 'fast']);
  } finally { releaseCatalog(); client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test.each(['output_limit', 'tool_budget_exhausted', 'provider_failed'])('scheduled %s outcomes are failed and never delivered', async reason => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-schedule-outcome-'));
  const history = new RunHistory(':memory:');
  const runner: TurnRunner = { async *run() {
    yield { type: 'text_part', payload: { text: 'Partial scheduled analysis' } };
    yield { type: 'status_update', payload: { stop_reason: reason } };
  } };
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const store = new JobStore(join(directory, 'jobs.json'));
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, runHistory: history, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  await server.start();
  try {
    const caller = await runtime.openSession('caller');
    store.add(new CronJob({ id: 'limited-job', prompt: 'Check project', projectRoot: directory, paused: true, schedule: '0 9 * * *' }));
    await expect(server.scheduleToolRequest(caller.id, 'run', { schedule_id: 'limited-job' })).rejects.toThrow(reason);
    const scheduled = runtime.sessionStatus('cron:limited-job')!;
    expect(history.list(scheduled.id)[0]).toMatchObject({ state: 'failed', output: 'Partial scheduled analysis' });
    expect(existsSync(join(directory, 'archive'))).toBe(false);
    expect(store.get('limited-job')?.lastRunAt).toBeUndefined();
  } finally { await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test('schedule destinations validate before persistence and deliver through the configured adapter', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-schedule-destination-'));
  const socketPath = join(directory, 'daemon.sock');
  const store = new JobStore(join(directory, 'jobs.json'));
  const channel = new DaemonRecordingChannel('recording');
  const server = new DaemonServer({ socketPath, projectDirectory: directory, channelManager: new ChannelManager({ channels: [['recording', channel]] }), cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive'), runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') }) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'schedule.options', params: {} });
    const options = (await client.next(frame => frame.id === 1)).result;
    expect(options?.destinations).toEqual([{ name: 'none', enabled: true }, { name: 'recording', enabled: false }]);
    const base = { prompt: 'scheduled report', schedule: '0 9 * * *', paused: true };
    for (const [index, destination] of [{ deliver: 'unknown', recipient: 'room' }, { deliver: 'recording', recipient: '' }, { deliver: 'none', recipient: 'room' }, { deliver: 'recording', recipient: 'room\nother' }].entries()) {
      client.send({ jsonrpc: '2.0', id: 10 + index, method: 'schedule.create', params: { ...base, ...destination } });
      expect((await client.next(frame => frame.id === 10 + index)).error).toBeDefined();
    }
    expect(store.listJobs()).toHaveLength(0);
    client.send({ jsonrpc: '2.0', id: 20, method: 'schedule.create', params: { ...base, deliver: 'recording', recipient: 'room-42' } });
    const created = (await client.next(frame => frame.id === 20)).result as { job: { id: string; revision: string } };
    expect(created.job).toMatchObject({ deliver: 'recording', recipient: 'room-42', paused: true });
    expect(channel.sent).toHaveLength(0);
    client.send({ jsonrpc: '2.0', id: 21, method: 'schedule.update', params: { ...base, schedule_id: created.job.id, revision: created.job.revision } });
    const updated = (await client.next(frame => frame.id === 21)).result as { job: { revision: string } };
    expect(updated.job).toMatchObject({ deliver: 'recording', recipient: 'room-42' });
    client.send({ jsonrpc: '2.0', id: 22, method: 'schedule.run', params: { schedule_id: created.job.id } });
    expect((await client.next(frame => frame.id === 22)).result?.ok).toBe(true);
    expect(channel.sent).toHaveLength(1);
    expect(channel.sent[0]).toMatchObject({ channel: 'recording', roomId: 'room-42', text: 'Bun daemon foundation received: scheduled report' });
    client.send({ jsonrpc: '2.0', id: 23, method: 'schedule.inspect', params: { schedule_id: created.job.id } });
    const inspected = (await client.next(frame => frame.id === 23)).result as { job: { revision: string } };
    client.send({ jsonrpc: '2.0', id: 24, method: 'schedule.update', params: { ...base, schedule_id: created.job.id, revision: inspected.job.revision, deliver: 'none', recipient: '' } });
    expect((await client.next(frame => frame.id === 24)).result?.job).toMatchObject({ deliver: 'none', recipient: '' });
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('session reattach restores persisted and live todo progress without mixing chats', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-todo-reattach-'));
  const gate = Promise.withResolvers<void>();
  const emitted = Promise.withResolvers<void>();
  const result = '1. [x] One\n2. [x] Two\n3. [x] Three\n4. [~] Four\n5. [ ] Five\n\nProgress: 3/5';
  const runner: TurnRunner = { async *run() {
    yield { type: 'tool_call', payload: { id: 'todo', name: 'TodoWriteTool', arguments: '{}' } };
    yield { type: 'tool_result', payload: { tool_call_id: 'todo', name: 'TodoWriteTool', permitted: true, return_value: result } };
    emitted.resolve(); await gate.promise;
  } };
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const original = await runtime.openSession('original');
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), projectDirectory: directory, runtime });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  const running = runtime.submitTurn('original', 'work', () => {});
  try {
    await emitted.promise;
    for (const [index, key] of ['other', 'original'].entries()) {
      client.send({ jsonrpc: '2.0', id: index + 1, method: 'session.open', params: { session_key: key, project_dir: directory } });
      const payload = (await client.next(frame => frame.id === index + 1)).result as { session: { todos: { status: string }[] } };
      expect(payload.session.todos).toHaveLength(key === 'original' ? 5 : 0);
      if (key === 'original') expect(payload.session.todos.filter(item => item.status === 'completed')).toHaveLength(3);
    }
    gate.resolve(); await running;
    // Completed and legacy records use the same canonical result parser.
    expect(original.inflightTodoResult).toBeUndefined();
    client.send({ jsonrpc: '2.0', id: 3, method: 'session.status', params: { session_key: 'original' } });
    expect(((await client.next(frame => frame.id === 3)).result?.session as { todos: unknown[] }).todos).toHaveLength(5);
    original.toolExecutions.push({ name: 'TodoWriteTool', permitted: true, result: 'Progress: 0/0' });
    client.send({ jsonrpc: '2.0', id: 4, method: 'session.status', params: { session_key: 'original' } });
    expect(((await client.next(frame => frame.id === 4)).result?.session as { todos: unknown[] }).todos).toEqual([]);
  } finally { gate.resolve(); await running; client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('workspaces slash lists and inspects persisted checkouts without applying changes', async () => {
  const directory = await realpath(await mkdtemp(join(tmpdir(), 'xerxes-workspace-slash-')));
  const git = async (...args: string[]) => {
    const process = Bun.spawn(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false', ...args], { cwd: directory, stdout: 'pipe', stderr: 'pipe' });
    const [code, , error] = await Promise.all([process.exited, new Response(process.stdout).text(), new Response(process.stderr).text()]);
    if (code) throw new Error(error);
  };
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')), cronLeasePath: join(directory, 'lease') });
  try {
    await git('init');
    await Bun.write(join(directory, 'file.txt'), 'parent');
    await git('add', '.'); await git('commit', '-m', 'workspace fixture');
    const tree = await nativeSubagentWorktrees(directory).create({ taskId: 'task', taskName: 'Task' });
    const id = tree.branch.replace('xerxes/agent-', '');
    await Bun.write(join(tree.path, 'file.txt'), 'agent result');
    await server.start();
    const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
    try {
      client.send({ jsonrpc: '2.0', id: 0, method: 'workspace.list', params: {} });
      expect((await client.next(frame => frame.id === 0)).result).toMatchObject({ ok: false });
      client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'workspaces' } });
      await client.next(frame => frame.id === 1);
      client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/workspaces' } });
      expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, inventory: { records: [{ id, taskId: 'task' }] } });
      client.send({ jsonrpc: '2.0', id: 3, method: 'slash', params: { command: '/workspaces inspect ' + id } });
      expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ ok: true, review: { id, path: tree.path, diff: expect.stringContaining('+agent result') } });
      client.send({ jsonrpc: '2.0', id: 4, method: 'slash', params: { command: '/workspaces inspect ../foreign' } });
      expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: false });
      client.send({ jsonrpc: '2.0', id: 5, method: 'workspace.list', params: {} });
      expect((await client.next(frame => frame.id === 5)).result).toMatchObject({ ok: true, inventory: { records: [{ id }] } });
      client.send({ jsonrpc: '2.0', id: 6, method: 'workspace.inspect', params: { workspace_id: id } });
      expect((await client.next(frame => frame.id === 6)).result).toMatchObject({ ok: true, review: { id, diff: expect.stringContaining('+agent result') } });
      const reviewed = await nativeSubagentWorktrees(directory).inspect(id);
      client.send({ jsonrpc: '2.0', id: 7, method: 'workspace.checkApply', params: { workspace_id: id, review_id: reviewed.reviewId } });
      const checked = (await client.next(frame => frame.id === 7)).result;
      expect(checked).toMatchObject({ ok: true, check: { reviewId: reviewed.reviewId, canApply: true, destination: directory } });
      expect(await Bun.file(join(directory, 'file.txt')).text()).toBe('parent');
      expect(await Bun.file(join(tree.path, 'file.txt')).text()).toBe('agent result');
      const applyParams = { workspace_id: id, review_id: reviewed.reviewId, destination_state: (checked?.check as { destinationState: string }).destinationState };
      client.send({ jsonrpc: '2.0', id: 8, method: 'workspace.apply', params: applyParams });
      expect((await client.next(frame => frame.id === 8)).result).toMatchObject({ ok: false });
      expect(await Bun.file(join(directory, 'file.txt')).text()).toBe('parent');
      client.send({ jsonrpc: '2.0', id: 9, method: 'workspace.apply', params: { ...applyParams, confirm: true, destination_state: '0'.repeat(64) } });
      expect((await client.next(frame => frame.id === 9)).result).toMatchObject({ ok: false, error: expect.stringContaining('Destination changed') });
      client.send({ jsonrpc: '2.0', id: 10, method: 'workspace.apply', params: { ...applyParams, confirm: true } });
      const applied = (await client.next(frame => frame.id === 10)).result;
      expect(applied).toMatchObject({ ok: true, integration: { status: 'applied', destination: directory } });
      expect(await Bun.file(join(directory, 'file.txt')).text()).toBe('agent result');
      expect(await Bun.file(join(tree.path, 'file.txt')).text()).toBe('agent result');
      const integration = applied?.integration as { id: string; backupPath: string };
      client.send({ jsonrpc: '2.0', id: 11, method: 'workspace.integrations', params: {} });
      expect((await client.next(frame => frame.id === 11)).result).toMatchObject({ ok: true, inventory: { records: expect.arrayContaining([expect.objectContaining({ id: integration.id, status: 'applied' })]) } });
      client.send({ jsonrpc: '2.0', id: 12, method: 'workspace.recover', params: { integration_id: integration.id, confirm: true } });
      expect((await client.next(frame => frame.id === 12)).result).toMatchObject({ ok: true, recovery: { status: 'applied', conflicts: [] } });
      expect(await Bun.file(join(directory, 'file.txt')).text()).toBe('agent result');
      const recordPath = join(integration.backupPath, 'record.json');
      await Bun.write(recordPath, JSON.stringify({ ...await Bun.file(recordPath).json(), status: 'prepared' }));
      client.send({ jsonrpc: '2.0', id: 15, method: 'workspace.integration.inspect', params: { integration_id: integration.id } });
      expect((await client.next(frame => frame.id === 15)).result).toMatchObject({ ok: true, inspection: { id: integration.id, files: [{ path: 'file.txt', action: 'restore' }] } });
      client.send({ jsonrpc: '2.0', id: 13, method: 'workspace.recover', params: { integration_id: integration.id } });
      expect((await client.next(frame => frame.id === 13)).result).toMatchObject({ ok: false });
      expect(await Bun.file(join(directory, 'file.txt')).text()).toBe('agent result');
      client.send({ jsonrpc: '2.0', id: 14, method: 'workspace.recover', params: { integration_id: integration.id, confirm: true } });
      expect((await client.next(frame => frame.id === 14)).result).toMatchObject({ ok: true, recovery: { id: integration.id, status: 'rolled-back', conflicts: [] } });
      expect(await Bun.file(join(directory, 'file.txt')).text()).toBe('parent');
    } finally { client.close(); }
  } finally { await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('workspace Runs includes ordered upcoming jobs without leaking another project or paused schedules', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-upcoming-runs-'))
  const history = new RunHistory(join(directory, 'runs.sqlite'))
  const store = new JobStore(join(directory, 'jobs.json'))
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') })
  await runtime.openSession('owner')
  for (const [id, projectRoot, paused, nextRunAt] of [
    ['later', directory, false, '2099-01-02T09:00:00Z'],
    ['first', directory, false, '2099-01-01T09:00:00Z'],
    ['paused', directory, true, '2099-01-01T08:00:00Z'],
    ['foreign', join(directory, 'other'), false, '2099-01-01T07:00:00Z'],
  ] as const) store.add(new CronJob({ id, projectRoot, paused, nextRunAt, prompt: id, schedule: '0 9 * * *' }))
  const socketPath = join(directory, 'daemon.sock')
  const server = new DaemonServer({ socketPath, runtime, runHistory: history, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease') })
  await server.start()
  const client = await SocketTestClient.connect(socketPath)
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'run.list', params: { session_key: 'owner', scope: 'workspace' } })
    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({ upcoming_total: 2, upcoming: [{ id: 'first' }, { id: 'later' }], runs: [] })
    client.send({ jsonrpc: '2.0', id: 2, method: 'run.list', params: { session_key: 'owner', scope: 'session' } })
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ upcoming_total: 0, upcoming: [] })
  } finally { client.close(); await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }) }
})

test('daemon tool inventory uses the runtime surface when no catalog port is injected', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-tool-inventory-daemon-'));
  const socketPath = join(directory, 'daemon.sock');
  const runtime = new InMemoryDaemonRuntime({
    toolInventory: () => [{ name: 'ReadFile', exposure: 'loaded', reason: 'Schema exposed to the current session' }],
    async *run() { yield { type: 'text_part', payload: { text: 'unused' } }; },
  }, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath, runtime });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'tool.inventory', params: {} });
    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({ source: 'unavailable', execution_readiness: 'unknown', tools: [] });
    client.send({ jsonrpc: '2.0', id: 2, method: 'initialize', params: { session_key: 'inventory', project_dir: directory } });
    await client.next(frame => frame.id === 2);
    await client.next(eventFrame('init_done'));
    await client.next(eventFrame('status_update'));
    client.send({ jsonrpc: '2.0', id: 3, method: 'tool.inventory', params: {} });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ source: 'runtime-registry', execution_readiness: 'not_checked', tools: [{ name: 'ReadFile', exposure: 'loaded' }] });
    client.send({ jsonrpc: '2.0', id: 4, method: 'slash', params: { command: '/tools' } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ source: 'runtime-registry' });
    expect((await client.next(eventFrame('notification'))).params?.payload).toMatchObject({ body: expect.stringContaining('[loaded]') });
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('daemon MCP health exposes failed configurations and reload retries enabled servers', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-mcp-health-'));
  const socketPath = join(directory, 'daemon.sock');
  let fail = true;
  const manager = new MCPManager({ clientFactory: config => ({
    config, tools: [], resources: [], prompts: [], connected: true,
    async connect() { if (fail) throw new Error('fixture connection refused'); },
    async disconnect() {},
    async callTool() { return { content: [] }; },
    async readResource() { return { contents: [] }; },
    async getPrompt() { return { messages: [] }; },
  }), reconnect: { policy: { maxAttempts: 1 } } });
  await manager.addServer({ name: 'broken' });
  await manager.addServer({ name: 'off', enabled: false });
  const server = new DaemonServer({ socketPath, mcpManager: manager, runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') }) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'mcp.status', params: {} });
    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({ configured: true, servers: { broken: { state: 'failed', connected: false }, off: { state: 'disabled' } } });
    fail = false;
    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/reload-mcp' } });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, servers: [{ name: 'broken', reconnected: true }] });
    await client.next(eventFrame('notification'));
    client.send({ jsonrpc: '2.0', id: 3, method: 'slash', params: { command: '/mcp status' } });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ servers: { broken: { connected: true }, off: { state: 'disabled' } } });
    await client.next(eventFrame('notification'));
    client.send({ jsonrpc: '2.0', id: 4, method: 'slash', params: { command: '/mcp reconnect off' } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: false, error: expect.stringContaining('disabled') });
  } finally { client.close(); await server.stop(); await manager.disconnectAll(); await rm(directory, { recursive: true, force: true }); }
});

test('skill inspection exposes source and literal instructions without activating or expanding them', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-skill-inspect-'));
  const skillDirectory = join(directory, 'skills');
  const source = join(skillDirectory, 'inspect-fixture', 'SKILL.md');
  const marker = join(directory, 'must-not-exist');
  await mkdir(join(skillDirectory, 'inspect-fixture'), { recursive: true });
  await writeFile(source, `---\nname: inspect-fixture\ndescription: Inspect fixture\n---\nLiteral $ARGUMENTS and !\`touch ${marker}\``);
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const socketPath = join(directory, 'daemon.sock');
  const server = new DaemonServer({ socketPath, runtime, skillDirectories: [skillDirectory] });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { session_key: 'inspect' } });
    await client.next(frame => frame.id === 1);
    await client.next(eventFrame('init_done'));
    await client.next(eventFrame('status_update'));
    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/skills inspect inspect-fixture' } });
    const result = await client.next(frame => frame.id === 2);
    expect(result.result).toMatchObject({ ok: true, skill: { source, name: 'inspect-fixture', execution_readiness: 'not_checked', truncated: false, instructions: expect.stringContaining('$ARGUMENTS') } });
    expect(await Bun.file(marker).exists()).toBe(false);
    expect(runtime.sessionStatus('inspect')?.messages).toEqual([]);
    client.send({ jsonrpc: '2.0', id: 3, method: 'complete', params: { text: '/skills inspect inspect-' } });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ completions: [{ value: '/skills inspect inspect-fixture ', label: 'inspect-fixture' }] });
    client.send({ jsonrpc: '2.0', id: 4, method: 'slash', params: { command: '/skills inspect missing' } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: false, error: expect.stringContaining('not discovered') });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test('skill diagnostics report shadowed sources and refresh after the conflict is removed', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-skill-diagnostics-'));
  const first = join(directory, 'first');
  const second = join(directory, 'second');
  for (const root of [first, second]) {
    await mkdir(root, { recursive: true });
    await writeFile(join(root, 'SKILL.md'), '---\nname: duplicated\ndescription: fixture\n---\nReview files.');
  }
  const socketPath = join(directory, 'daemon.sock');
  const server = new DaemonServer({ socketPath, skillDirectories: [first, second], runtime: new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions'),
  }) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { session_key: 'diagnostics' } });
    await client.next(frame => frame.id === 1);
    await client.next(eventFrame('init_done'));
    await client.next(eventFrame('status_update'));
    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/skills diagnostics' } });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, total: 1, truncated: false, diagnostics: [
      { kind: 'shadowed', name: 'duplicated', path: join(second, 'SKILL.md'), detail: expect.stringContaining(join(first, 'SKILL.md')) },
    ] });
    await rm(join(second, 'SKILL.md'));
    client.send({ jsonrpc: '2.0', id: 3, method: 'slash', params: { command: '/skills diagnostics' } });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ ok: true, total: 0, diagnostics: [] });
    client.send({ jsonrpc: '2.0', id: 4, method: 'complete', params: { text: '/skills dia' } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ completions: [{ value: '/skills diagnostics ', label: 'diagnostics' }] });
    client.send({ jsonrpc: '2.0', id: 5, method: 'slash', params: { command: '/skills diagnostics unexpected' } });
    expect((await client.next(frame => frame.id === 5)).result).toMatchObject({ ok: false });
  } finally {
    client.close(); await server.stop(); await rm(directory, { recursive: true, force: true });
  }
});

test.each([false, true])('plugin discovery reports actual host configuration and read-only inspection (configured=%s)', async configured => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-plugin-inspect-'));
  const pluginRegistry = new PluginRegistry();
  pluginRegistry.registerPlugin({ name: 'fixture' });
  pluginRegistry.registerTool('fixture-tool', () => { throw new Error('must not execute'); }, undefined, 'fixture');
  const socketPath = join(directory, 'daemon.sock');
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath, runtime, ...(configured ? { pluginRegistry } : {}) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { session_key: 'plugins' } });
    await client.next(frame => frame.id === 1);
    await client.next(eventFrame('init_done')); await client.next(eventFrame('status_update'));
    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/plugins' } });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, source: configured ? 'host-registry' : 'unconfigured', plugins: configured ? ['fixture'] : [], execution_readiness: 'not_checked' });
    client.send({ jsonrpc: '2.0', id: 3, method: 'slash', params: { command: '/plugins inspect fixture' } });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject(configured
      ? { ok: true, plugin: { name: 'fixture', source: { kind: 'host-registration' }, tools: ['fixture-tool'] } }
      : { ok: false });
    client.send({ jsonrpc: '2.0', id: 4, method: 'complete', params: { text: '/plugins inspect fix' } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ completions: configured ? [{ value: '/plugins inspect fixture ', label: 'fixture' }] : [] });
    expect(runtime.sessionStatus('plugins')?.messages).toEqual([]);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('MCP settings RPC masks launch secrets and preserves omitted credentials during guarded saves', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-mcp-settings-rpc-'));
  const path = join(directory, 'mcp.json'), socketPath = join(directory, 'daemon.sock');
  const store = new McpSettingsStore(path);
  const manager = new MCPManager({ clientFactory: config => ({
    config, tools: [], resources: [], prompts: [],
    async connect() { if (config.command === 'bad') throw new Error('rpc-private-token'); },
    async disconnect() {}, async callTool() { return { content: [] }; },
    async readResource() { return { contents: [] }; }, async getPrompt() { return { messages: [] }; },
  }) });
  await writeFile(path, JSON.stringify({ alpha: { command: 'bun', args: ['rpc-private-token'], env: { TOKEN: 'rpc-private-token' } } }));
  await manager.addServer(store.read().servers[0]!);
  const server = new DaemonServer({ socketPath, mcpManager: manager, mcpSettingsStore: store,
    runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') }) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'mcp.settings.get' });
    const first = (await client.next(frame => frame.id === 1)).result!;
    expect(first).toMatchObject({ ok: true, servers: [{ name: 'alpha', enabled: true, transport: 'stdio' }] });
    expect(JSON.stringify(first)).not.toContain('rpc-private-token');
    client.send({ jsonrpc: '2.0', id: 2, method: 'mcp.settings.save', params: { name: 'alpha', revision: first.revision, changes: { command: 'bad' } } });
    const failed = (await client.next(frame => frame.id === 2)).result;
    expect(failed).toMatchObject({ ok: false });
    expect(JSON.stringify(failed)).not.toContain('rpc-private-token');
    expect(manager.getServer('alpha')?.config.command).toBe('bun');
    expect(store.read().revision).toBe(String(first.revision));
    client.send({ jsonrpc: '2.0', id: 3, method: 'mcp.settings.save', params: { name: 'alpha', revision: first.revision, changes: { enabled: false } } });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ ok: true, server: { state: 'disabled' } });
    expect(store.read().servers[0]?.env).toEqual({ TOKEN: 'rpc-private-token' });
    expect(store.read().servers[0]?.args).toEqual(['rpc-private-token']);
    client.send({ jsonrpc: '2.0', id: 4, method: 'mcp.settings.save', params: { name: 'alpha', revision: first.revision, changes: { enabled: true } } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: false });
    expect(manager.status('alpha')?.state).toBe('disabled');
    client.send({ jsonrpc: '2.0', id: 5, method: 'slash', params: { command: '/config mcp' } });
    const config = (await client.next(frame => frame.id === 5)).result;
    expect(config).toMatchObject({ ok: true, servers: [{ name: 'alpha', enabled: false }] });
    expect(JSON.stringify(config)).not.toContain('rpc-private-token');
    client.send({ jsonrpc: '2.0', id: 6, method: 'complete', params: { text: '/config m' } });
    expect((await client.next(frame => frame.id === 6)).result).toMatchObject({ completions: [{ value: '/config mcp ' }] });
    client.send({ jsonrpc: '2.0', id: 7, method: 'mcp.settings.save', params: { name: 'beta', revision: store.read().revision, create: true, changes: { command: 'bun', enabled: false } } });
    expect((await client.next(frame => frame.id === 7)).result).toMatchObject({ ok: true, server: { name: 'beta', state: 'disabled' } });
    expect(store.read().servers.map(server => server.name)).toEqual(['alpha', 'beta']);
  } finally { client.close(); await server.stop(); await manager.disconnectAll(); await rm(directory, { recursive: true, force: true }); }
});

test('disconnecting an MCP settings editor cancels its candidate before persistence', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-mcp-settings-cancel-'));
  const path = join(directory, 'mcp.json'), socketPath = join(directory, 'daemon.sock');
  const store = new McpSettingsStore(path);
  const entered = Promise.withResolvers<void>(), release = Promise.withResolvers<void>(), cleaned = Promise.withResolvers<void>();
  const manager = new MCPManager({ clientFactory: config => ({
    config, tools: [], resources: [], prompts: [],
    async connect() { if (config.command === 'new') { entered.resolve(); await release.promise; } },
    async disconnect() { if (config.command === 'new') cleaned.resolve(); },
    async callTool() { return { content: [] }; }, async readResource() { return { contents: [] }; }, async getPrompt() { return { messages: [] }; },
  }) });
  await writeFile(path, JSON.stringify({ alpha: { command: 'old' } }));
  const original = store.read();
  await manager.addServer(original.servers[0]!);
  const server = new DaemonServer({ socketPath, mcpManager: manager, mcpSettingsStore: store,
    runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') }) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'mcp.settings.save', params: { name: 'alpha', revision: original.revision, changes: { command: 'new' } } });
    await entered.promise;
    client.close();
    // Wait for the server's transport to observe EOF before releasing connect.
    for (let i = 0; i < 100 && (server as unknown as { mcpSettingsUpdates: Map<unknown, AbortController> }).mcpSettingsUpdates.values().next().value?.signal.aborted !== true; i++) await Bun.sleep(5);
    expect([...((server as unknown as { mcpSettingsUpdates: Map<unknown, AbortController> }).mcpSettingsUpdates.values())][0]?.signal.aborted).toBe(true);
    release.resolve();
    await cleaned.promise;
    expect(store.read()).toEqual(original);
    expect(manager.getServer('alpha')?.config.command).toBe('old');
  } finally { release.resolve(); client.close(); await server.stop(); await manager.disconnectAll(); await rm(directory, { recursive: true, force: true }); }
});

test('goal inspection exposes criteria and evidence only for the active session without mutating history', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-goal-inspection-'));
  const socketPath = join(directory, 'daemon.sock');
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath, runtime, projectDirectory: directory, cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')), cronLeasePath: join(directory, 'cron.lease') });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'goal-owner' } });
    await client.next(frame => frame.id === 1);
    const session = runtime.listSessions().find(row => row.sessionKey === 'goal-owner')!;
    const goal = createGoal(session.metadata, session.id, { objective: 'Match artifacts', criteria: [{ id: 'match', description: 'Files match' }] }, 100);
    recordGoalEvidence(session.metadata, session.id, goal, 'match', { toolCallId: 'cmp-result', summary: 'The comparison passed', recordedAt: 101 }, 101);
    const before = JSON.stringify(session.metadata);
    client.send({ jsonrpc: '2.0', id: 2, method: 'goal.inspect', params: {} });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, session_id: session.id, goal: { revision: 2, criteria: [{ id: 'match', evidence: { toolCallId: 'cmp-result' } }] } });
    expect(JSON.stringify(session.metadata)).toBe(before);
    expect(session.messages).toHaveLength(0);
    client.send({ jsonrpc: '2.0', id: 3, method: 'session.open', params: { session_key: 'goal-other' } });
    await client.next(frame => frame.id === 3);
    client.send({ jsonrpc: '2.0', id: 4, method: 'goal.inspect', params: { session_id: session.id } });
    expect((await client.next(frame => frame.id === 4)).result?.goal).toBeNull();
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('context inspection is session-scoped, read-only and rejects stale pages', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-context-inspector-'));
  const socketPath = join(directory, 'daemon.sock');
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath, runtime, projectDirectory: directory, cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')), cronLeasePath: join(directory, 'cron.lease') });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'context-owner' } });
    await client.next(frame => frame.id === 1);
    const session = runtime.listSessions().find(row => row.sessionKey === 'context-owner')!;
    session.messages.push({ role: 'user', content: 'private source evidence' });
    session.requestScaffold = { capturedAt: 1, systemSegments: [{ name: 'memory', text: 'saved preference' }] };
    client.send({ jsonrpc: '2.0', id: 2, method: 'context.inspect', params: { section: 'memory' } });
    const inspected = (await client.next(frame => frame.id === 2)).result!;
    expect(inspected.entries).toMatchObject([{ title: 'memory', text: 'saved preference' }]);
    client.send({ jsonrpc: '2.0', id: 3, method: 'slash', params: { command: '/context' } });
    expect((await client.next(frame => frame.id === 3)).result?.ok).toBe(true);
    expect(session.messages).toHaveLength(1);
    session.messages[0] = { role: 'user', content: 'changed source' };
    client.send({ jsonrpc: '2.0', id: 4, method: 'context.inspect', params: { section: 'conversation', generation: inspected.generation } });
    expect((await client.next(frame => frame.id === 4)).error).toBeDefined();
    client.send({ jsonrpc: '2.0', id: 5, method: 'session.open', params: { session_key: 'context-other' } });
    await client.next(frame => frame.id === 5);
    client.send({ jsonrpc: '2.0', id: 6, method: 'context.inspect', params: { section: 'memory', session_id: session.id } });
    expect((await client.next(frame => frame.id === 6)).result?.entries).toEqual([]);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('context controls validate source and revision, survive reload and reject active turns', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-context-controls-'));
  const sessionDirectory = join(directory, 'sessions');
  const socketPath = join(directory, 'daemon.sock');
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory });
  const server = new DaemonServer({ socketPath, runtime, projectDirectory: directory, cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')), cronLeasePath: join(directory, 'cron.lease') });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  let id = 0;
  const rpc = async (method: string, params: Record<string, unknown>) => {
    const callId = ++id;
    client.send({ jsonrpc: '2.0', id: callId, method, params });
    return (await client.next(frame => frame.id === callId)).result!;
  };
  try {
    await rpc('session.open', { session_key: 'controls-owner' });
    const session = runtime.listSessions().find(row => row.sessionKey === 'controls-owner')!;
    session.messages.push({ role: 'user', content: 'hello' }, { role: 'assistant', content: 'done' });
    session.turnCount = 1;
    session.requestScaffold = { memorySources: [{ scope: 'project', path: 'MEMORY.md', content: 'Captured fact' }] };
    const page = await rpc('context.inspect', { section: 'memory' });
    const patch = { action: 'pin', scope: 'project', path: 'MEMORY.md', revision: 0, generation: page.generation, content: 'untrusted replacement' };
    expect((await rpc('context.control', { ...patch, path: 'bootstrap' })).ok).toBe(false);
    expect((await rpc('context.control', patch)).ok).toBe(true);
    expect(session.metadata.context_controls).toMatchObject({ revision: 1, pins: [{ content: 'Captured fact' }] });
    expect((await rpc('context.control', patch)).ok).toBe(false);
    const updated = await rpc('context.inspect', { section: 'memory' });
    session.activeTurnId = 'running';
    expect((await rpc('context.control', { ...patch, action: 'exclude', generation: updated.generation, revision: 1 })).ok).toBe(false);
    session.activeTurnId = '';
    const restoredRuntime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory });
    const restored = await restoredRuntime.openSession(session.id, session.agentId, { resume: true, cwd: session.cwd });
    expect(restored.metadata.context_controls).toEqual(session.metadata.context_controls);
    expect(restored.requestScaffold).toBeUndefined();
    const transcriptPath = join(sessionDirectory, session.id + '.json');
    const original = await readFile(transcriptPath, 'utf8');
    const divergent = JSON.parse(original);
    divergent.generation = (divergent.generation ?? 0) + 1;
    divergent.messages[0].content = 'External transcript replacement';
    await writeFile(transcriptPath, JSON.stringify(divergent));
    try {
      const failed = await rpc('context.control', { ...patch, action: 'exclude', generation: updated.generation, revision: 1 });
      expect(failed.ok).toBe(false);
      expect(session.metadata.context_controls).toMatchObject({ revision: 1, pins: [{ content: 'Captured fact' }] });
      expect(runtime.listSessions()).toContain(session);
    } finally { await writeFile(transcriptPath, original); }
    await rpc('session.open', { session_key: 'controls-other' });
    expect((await rpc('context.control', { ...patch, session_id: session.id })).ok).toBe(false);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('session schedules bind identity, serialize follow-ups and cancel queued work', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-session-followup-'));
  const started = Promise.withResolvers<void>(), release = Promise.withResolvers<void>();
  const seen: string[] = [];
  const runner: TurnRunner = { async *run(session, prompt) {
    seen.push(session.id + ':' + prompt);
    started.resolve(); await release.promise;
    yield { type: 'text_part', payload: { text: 'Follow-up finished' } };
  } };
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const store = new JobStore(join(directory, 'jobs.json'));
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  await server.start();
  try {
    const caller = await runtime.openSession('caller');
    const options = { prompt: 'Check back', paused: true, interval_seconds: 60, target: 'session', max_runs: 3, expires_at: '2099-01-01T00:00:00Z' };
    await expect(server.scheduleToolRequest(caller.id, 'create', { ...options, max_runs: null })).rejects.toThrow('require');
    const first = await server.scheduleToolRequest(caller.id, 'create', options) as { job: { id: string; target_session_id: string } };
    const second = await server.scheduleToolRequest(caller.id, 'create', options) as { job: { id: string } };
    expect(first.job.target_session_id).toBe(caller.id);
    const active = server.scheduleToolRequest(caller.id, 'run', { schedule_id: first.job.id });
    await started.promise;
    const controller = new AbortController();
    const queued = server.scheduleToolRequest(caller.id, 'run', { schedule_id: second.job.id }, controller.signal).catch(error => error);
    await Bun.sleep(20);
    expect(seen).toEqual([caller.id + ':Check back']);
    controller.abort(); release.resolve();
    expect((await active).ok).toBe(true);
    expect(await queued).toBeInstanceOf(Error);
    expect(seen).toHaveLength(1);
    expect(caller.messages.some(message => String(message.content).includes('Follow-up finished'))).toBe(true);
    expect(runtime.listSessions()).toHaveLength(1);
  } finally { release.resolve(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('bound session resume rejects a missing transcript instead of creating a replacement', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-bound-resume-'));
  const runtime = new InMemoryDaemonRuntime({ async *run() {} }, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  try {
    await expect(runtime.openSession('missing', undefined, { resume: true, expectedSessionId: 'missing' })).rejects.toThrow('missing');
    expect(runtime.listSessions()).toHaveLength(0);
    const original = await runtime.openSession('slot');
    await expect(runtime.openSession('slot', undefined, { expectedSessionId: 'different' })).rejects.toThrow('identity changed');
    expect(runtime.sessionStatus('slot')?.id).toBe(original.id);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('session follow-up reloads its persisted conversation after daemon restart', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-followup-restart-'));
  const sessions = join(directory, 'sessions');
  const seen: string[] = [];
  const runner: TurnRunner = { async *run(session) { seen.push(session.id); expect(JSON.stringify(session.messages)).toContain('remember this context'); yield { type: 'text_part', payload: { text: 'Resumed follow-up' } }; } };
  const store = new JobStore(join(directory, 'jobs.json'));
  let runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: sessions });
  const makeServer = () => new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  let server = makeServer();
  try {
    await server.start();
    const owner = await runtime.openSession('old-tab');
    owner.messages.push({ role: 'user', content: 'remember this context' }, { role: 'assistant', content: 'I will remember it.' });
    await runtime.flushSessions('rewrite');
    const result = await server.scheduleToolRequest(owner.id, 'create', { prompt: 'Follow up', target: 'session', paused: true, interval_seconds: 60, max_runs: 2, expires_at: '2099-01-01T00:00:00Z' }) as { job: { id: string } };
    await server.stop();
    runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: sessions });
    server = makeServer(); await server.start();
    const caller = await runtime.openSession('new-tab');
    expect((await server.scheduleToolRequest(caller.id, 'run', { schedule_id: result.job.id })).ok).toBe(true);
    expect(seen).toEqual([owner.id]);
    expect(caller.messages).toHaveLength(0);
    expect(runtime.listSessions().find(session => session.id === owner.id)?.messages.some(message => String(message.content).includes('Resumed follow-up'))).toBe(true);
  } finally { await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('an active model tool cannot await its own scheduled follow-up', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-followup-self-wait-'));
  let server: DaemonServer;
  let followupId = '';
  let observed = false;
  const runner: TurnRunner = { async *run(session) {
    expect(session.activeTurnId).toBeTruthy();
    await expect(server.scheduleToolRequest(session.id, 'run', { schedule_id: followupId }, undefined, true)).rejects.toThrow('inside its own active conversation');
    observed = true;
    yield { type: 'text_part', payload: { text: 'Parent turn remains responsive' } };
  } };
  const runtime = new InMemoryDaemonRuntime(runner, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const store = new JobStore(join(directory, 'jobs.json'));
  server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  await server.start();
  try {
    const caller = await runtime.openSession('caller');
    const created = await server.scheduleToolRequest(caller.id, 'create', { prompt: 'Check later', target: 'session', paused: true, interval_seconds: 60, max_runs: 3, expires_at: '2099-01-01T00:00:00Z' }) as { job: { id: string } };
    followupId = created.job.id;
    await runtime.submitTurn(caller.sessionKey, 'Run my follow-up', () => {});
    expect(observed).toBe(true);
    expect(store.get(followupId)?.runsStarted).toBe(0);
  } finally { await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('/loop only lists and controls the current conversation follow-ups', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-loop-scope-'));
  const runtime = new InMemoryDaemonRuntime({ async *run() {} }, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const store = new JobStore(join(directory, 'jobs.json'));
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease') });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'loop-owner' } });
    await client.next(frame => frame.id === 1);
    const owner = runtime.sessionStatus('loop-owner')!;
    const other = await runtime.openSession('other-owner');
    const options = { prompt: 'Check', target: 'session', paused: true, interval_seconds: 600, max_runs: 10, expires_at: '2099-01-01T00:00:00Z' };
    const mine = await server.scheduleToolRequest(owner.id, 'create', options) as { job: { id: string } };
    const theirs = await server.scheduleToolRequest(other.id, 'create', options) as { job: { id: string } };
    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/loop' } });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true, jobs: [{ id: mine.job.id }] });
    client.send({ jsonrpc: '2.0', id: 3, method: 'schedule.resume', params: { scope: 'session', schedule_id: theirs.job.id } });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ ok: false });
    expect(store.get(theirs.job.id)?.paused).toBe(true);
    client.send({ jsonrpc: '2.0', id: 4, method: 'slash', params: { command: '/loop resume ' + mine.job.id } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: true });
    expect(store.get(mine.job.id)?.paused).toBe(false);
    client.send({ jsonrpc: '2.0', id: 5, method: 'schedule.list', params: { scope: 'session', owner_session_id: owner.id } });
    expect((await client.next(frame => frame.id === 5)).result).toMatchObject({ ok: true, owner_session_id: owner.id, jobs: [{ id: mine.job.id }] });
    client.send({ jsonrpc: '2.0', id: 6, method: 'schedule.pause', params: { scope: 'session', owner_session_id: other.id, schedule_id: mine.job.id } });
    expect((await client.next(frame => frame.id === 6)).error?.message).toContain('active conversation changed');
    expect(store.get(mine.job.id)?.paused).toBe(false);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('follow-ups can report only their own configured condition and require explicit rearm after completion', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-followup-complete-'));
  let daemon!: DaemonServer;
  let id = '';
  let releaseLate!: () => void;
  let lateResult: Promise<unknown> | undefined;
  let runs = 0;
  const runtime = new InMemoryDaemonRuntime({ async *run(session, prompt) {
    runs++;
    expect(prompt).toContain('Follow-up stop condition: "Deployment healthy"');
    expect(prompt).toContain('action "complete"');
    const args = { schedule_id: id, evidence: 'Health endpoint returned 200 and all replicas were ready.' };
    await expect(daemon.scheduleToolRequest('other-session', 'complete', args, undefined, true)).rejects.toThrow('currently executing');
    await expect(daemon.scheduleToolRequest(session.id, 'complete', { ...args, schedule_id: 'other-job' }, undefined, true)).rejects.toThrow('currently executing');
    await expect(daemon.scheduleToolRequest(session.id, 'complete', { ...args, evidence: '' }, undefined, true)).rejects.toThrow('evidence');
    const abort = new AbortController(); abort.abort(new Error('cancelled attempt'));
    await expect(daemon.scheduleToolRequest(session.id, 'complete', args, abort.signal, true)).rejects.toThrow('cancelled attempt');
    const update = store.update.bind(store);
    store.update = (jobId, changes, revision) => {
      if (changes.metadata && typeof changes.metadata === 'object' && 'followup_completion' in changes.metadata) {
        store.update = update;
        update(jobId, { metadata: { ...store.get(jobId)!.metadata, concurrent_edit: runs } });
      }
      return update(jobId, changes, revision);
    };
    await expect(daemon.scheduleToolRequest(session.id, 'complete', args, undefined, true)).rejects.toThrow('Schedule changed');
    expect(store.get(id)?.metadata.followup_completion).toBeUndefined();
    const result = await daemon.scheduleToolRequest(session.id, 'complete', args, undefined, true);
    expect(result).toMatchObject({ ok: true, completion: { source: 'model_reported', evidence: args.evidence, condition: 'Deployment healthy' } });
    expect(await daemon.scheduleToolRequest(session.id, 'complete', args, undefined, true)).toMatchObject({ ok: true, completion: result.completion });
    lateResult = new Promise<void>(resolve => { releaseLate = resolve; }).then(() =>
      expect(daemon.scheduleToolRequest(session.id, 'complete', args, undefined, true)).rejects.toThrow('currently executing'));
    yield { type: 'text_part', payload: { text: 'Condition met.' } };
  } }, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const path = join(directory, 'jobs.json');
  const store = new JobStore(path);
  daemon = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), runtime, projectDirectory: directory, cronStoreFactory: () => store, cronLeasePath: join(directory, 'lease'), cronArchiveDirectory: join(directory, 'archive') });
  await daemon.start();
  try {
    const session = await runtime.openSession('followup-owner');
    const created = await daemon.scheduleToolRequest(session.id, 'create', { prompt: 'Check deployment', target: 'session', stop_condition: 'Deployment healthy', paused: true, interval_seconds: 600, max_runs: 5, expires_at: '2099-01-01T00:00:00Z' }) as { job: { id: string } };
    id = created.job.id;
    await expect(daemon.scheduleToolRequest(session.id, 'complete', { schedule_id: id, evidence: 'outside attempt' }, undefined, true)).rejects.toThrow('currently executing');
    await daemon.scheduleToolRequest(session.id, 'run', { schedule_id: id });
    releaseLate(); await lateResult;
    expect(new JobStore(path).get(id)).toMatchObject({ paused: true, runsStarted: 1, stopCondition: 'Deployment healthy', metadata: { followup_completion: { source: 'model_reported' } } });
    expect(store.get(id)?.metadata.followup_completions).toHaveLength(1);
    const summary = await daemon.scheduleToolRequest(session.id, 'list', { scope: 'session', summary: true }) as { jobs: { metadata: Record<string, unknown> }[] };
    expect(summary.jobs[0]?.metadata).toEqual({ execution_recovery_required: false, followup_completion: { source: 'model_reported' } });
    await expect(daemon.scheduleToolRequest(session.id, 'run', { schedule_id: id })).rejects.toThrow('explicitly resume');
    expect(runs).toBe(1);
    await daemon.scheduleToolRequest(session.id, 'resume', { schedule_id: id });
    expect(store.get(id)?.metadata.followup_completion).toBeUndefined();
    expect(store.get(id)?.metadata.followup_completions).toHaveLength(1);
    await daemon.scheduleToolRequest(session.id, 'run', { schedule_id: id });
    releaseLate(); await lateResult;
    expect(store.get(id)?.metadata.followup_completions).toHaveLength(2);
    expect(runs).toBe(2);
  } finally { releaseLate?.(); await lateResult; await daemon.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('monitor policy RPC edits are guarded and owner scoped', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'monitor-policy-rpc-'));
  const history = new RunHistory(join(directory, 'runs.sqlite'));
  const mailbox = new ReactionMailbox(join(directory, 'reactions.sqlite'));
  const terminals = new TerminalRegistry();
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const monitors = new TerminalMonitors(terminals, history, () => {}, undefined, mailbox);
  const socketPath = join(directory, 'daemon.sock');
  const server = new DaemonServer({ socketPath, runtime, monitors, reactionMailbox: mailbox, runHistory: history, terminalRegistry: terminals, cronLeasePath: join(directory, 'cron.lease'), cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')) });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'owner' } });
    await client.next(frame => frame.id === 1);
    const owner = runtime.sessionStatus('owner')!.id;
    terminals.open({ id: 'build', ownerSessionId: owner, command: 'fixture', cwd: directory, kind: 'background' });
    const watch = monitors.start(owner, { terminalId: 'build', match: 'error', reaction: { maxReactions: 3, maxDurationMs: 60000, maxTotalTokens: 100 } });
    const params = { monitor_id: watch.id, revision: watch.reactionHealth!.policy!.revision, max_reactions: 5, reaction_timeout_seconds: 30, max_total_tokens: 200 };
    client.send({ jsonrpc: '2.0', id: 2, method: 'monitor.update', params });
    expect((await client.next(frame => frame.id === 2)).result?.monitor).toMatchObject({ reactionHealth: { attempts: 0, policy: { maxReactions: 5, maxDurationMs: 30000, maxTotalTokens: 200 } } });
    client.send({ jsonrpc: '2.0', id: 3, method: 'monitor.update', params });
    expect((await client.next(frame => frame.id === 3)).error).toBeDefined();
    client.send({ jsonrpc: '2.0', id: 4, method: 'session.open', params: { session_key: 'other' } });
    await client.next(frame => frame.id === 4);
    client.send({ jsonrpc: '2.0', id: 5, method: 'monitor.update', params: { ...params, revision: mailbox.inspect(owner, watch.id)!.policy!.revision } });
    expect((await client.next(frame => frame.id === 5)).error).toBeDefined();
    expect(mailbox.inspect(owner, watch.id)?.policy?.maxTotalTokens).toBe(200);
  } finally { client.close(); monitors.close(); await server.stop(); mailbox.close(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test('branch keeps independent metadata and the source reasoning and permission settings', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'session-branch-copy-'));
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const source = await runtime.openSession('owner');
  source.messages = [{ role: 'user', content: 'Review' }, { role: 'assistant', content: 'Reviewed' }];
  source.metadata = { nested: { note: 'source' } };
  source.extra = { nested: { note: 'source-extra' } };
  source.reasoningEffort = 'high'; source.reasoningPinned = true;
  source.permissionMode = 'manual'; source.permissionPinned = true;
  source.turnCount = 1; source.totalApiCalls = 2;
  source.usageComplete = false;
  delete source.apiCallsComplete;
  const server = new DaemonServer({ runtime, socketPath: join(directory, 'daemon.sock'), agentSettingsStore: new AgentSettingsStore(join(directory, 'settings.sqlite')),
    cronLeasePath: join(directory, 'lease'), cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')) });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'owner' } });
    await client.next(frame => frame.id === 1);
    source.status = 'working';
    client.send({ jsonrpc: '2.0', id: 3, method: 'slash', params: { command: '/branch Partial' } });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ ok: false });
    expect(runtime.listSessions()).toHaveLength(1);
    source.status = 'idle';
    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/branch Fork' } });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true });
    const branch = runtime.listSessions().find(session => session.id !== source.id)!;
    expect(branch).toMatchObject({ reasoningEffort: 'high', reasoningPinned: true, permissionMode: 'manual', permissionPinned: true, totalApiCalls: 2, turnCount: 1, usageComplete: false, apiCallsComplete: false });
    const reopenedRuntime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
    const reopened = await reopenedRuntime.openSession(branch.sessionKey, branch.agentId, { resume: true, cwd: branch.cwd });
    expect(reopened.id).toBe(branch.id);
    expect(reopened).toMatchObject({ reasoningEffort: 'high', reasoningPinned: true, permissionMode: 'manual', permissionPinned: true, totalApiCalls: 2, turnCount: 1, usageComplete: false, apiCallsComplete: false });
    expect(reopened.messages).toEqual(source.messages);
    expect(branch.messages).toEqual(source.messages);
    (branch.metadata.nested as { note: string }).note = 'branch';
    (branch.extra.nested as { note: string }).note = 'branch-extra';
    branch.messages[0] = { role: 'user', content: 'Changed' };
    expect(source.metadata.nested).toEqual({ note: 'source' });
    expect(source.extra.nested).toEqual({ note: 'source-extra' });
    expect(source.messages[0]?.content).toBe('Review');
    expect(branch.metadata.parent_session_id).toBe(source.id);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('native branch selects a retained turn without copying later derived state', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'session-branch-turn-'));
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const source = await runtime.openSession('owner');
  source.messages = [{ role: 'user', content: 'First' }, { role: 'assistant', content: 'First answer' }, { role: 'user', content: 'Later' }, { role: 'assistant', content: 'Later answer' }];
  source.metadata.future_note = 'Only true after second turn';
  source.turnCount = 2; source.totalInputTokens = 100; source.totalOutputTokens = 50;
  const server = new DaemonServer({ runtime, socketPath: join(directory, 'daemon.sock'), agentSettingsStore: new AgentSettingsStore(join(directory, 'settings.sqlite')),
    cronLeasePath: join(directory, 'lease'), cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')) });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'owner' } });
    await client.next(frame => frame.id === 1);
    client.send({ jsonrpc: '2.0', id: 2, method: 'slash', params: { command: '/branch --through-turn 1 Earlier' } });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: true });
    const branch = runtime.listSessions().find(session => session.id !== source.id)!;
    expect(branch.messages).toEqual(source.messages.slice(0, 2));
    expect(branch.metadata).toMatchObject({ title: 'Earlier', branch_through_retained_turn: 1, branch_message_count: 2, parent_session_id: source.id });
    expect(branch.metadata.future_note).toBeUndefined();
    expect(branch).toMatchObject({ turnCount: 1, usageComplete: false, apiCallsComplete: false });
    expect(source.messages).toHaveLength(4);
    expect(source.metadata.future_note).toBe('Only true after second turn');
    for (const [id, value] of [[3, '0'], [4, '99'], [5, 'bad']] as const) {
      client.send({ jsonrpc: '2.0', id, method: 'slash', params: { command: `/branch --through-turn ${value}` } });
      expect((await client.next(frame => frame.id === id)).result).toMatchObject({ ok: false });
      expect(runtime.listSessions()).toHaveLength(2);
    }
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('Codex inventory uses one fresh catalog for capacities and reasoning and rejects stale pages', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'codex-inventory-snapshot-'));
  const profiles = new ProfileStore(join(directory, 'profiles.json'));
  const profile = { name: 'fixture', provider: 'openai-codex', apiKey: '', baseUrl: 'https://example.invalid', model: 'worker-a' };
  profiles.save(profile);
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const session = await runtime.openSession('owner');
  let calls = 0, effort = 'high', replace = false;
  const controller = new AbortController();
  const server = new DaemonServer({ runtime, profileStore: profiles, agentSettingsStore: new AgentSettingsStore(join(directory, 'notes.sqlite')),
    socketPath: join(directory, 'daemon.sock'), cronLeasePath: join(directory, 'lease'), cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')),
    codexModelCatalog: async (_profile, signal) => {
      expect(signal).toBe(controller.signal);
      calls++;
      if (replace) profiles.save({ ...profile, baseUrl: 'https://replacement.invalid' });
      return ['worker-a', 'worker-b'].map(id => ({ id, displayName: undefined, contextLimit: 64000, harnessCoupled: false,
        defaultReasoningLevel: effort, reasoningLevels: [{ effort, description: undefined }] }));
    },
  });
  await server.start();
  try {
    const first = await server.modelInventoryToolRequest(session.id, { provider_profile: 'fixture', limit: 1 }, controller.signal) as { revision: string; entries: unknown[] };
    expect(calls).toBe(1);
    expect(first.entries[0]).toMatchObject({ context_window: 64000, reasoning_efforts: ['off', 'high'], reasoning_source: 'provider_reported' });
    effort = 'low';
    await expect(server.modelInventoryToolRequest(session.id, { provider_profile: 'fixture', offset: 1, revision: first.revision }, controller.signal)).rejects.toThrow('changed');
    expect(calls).toBe(2);
    profiles.updateModelCapabilities('fixture', 'worker-a', { contextLimit: 32000, maxOutputTokens: 4096 });
    const overridden = await server.modelInventoryToolRequest(session.id, { provider_profile: 'fixture', limit: 1 }, controller.signal) as { entries: unknown[] };
    expect(overridden.entries[0]).toMatchObject({ context_window: 32000, context_source: 'override', max_output_tokens: 4096, output_source: 'override' });
    replace = true;
    await expect(server.modelInventoryToolRequest(session.id, { provider_profile: 'fixture' }, controller.signal)).rejects.toThrow('profile changed');
  } finally { await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('model inventory uses configured discovery without exposing credentials or switching the session', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'model-inventory-daemon-'));
  const endpoint = Bun.serve({ port: 0, hostname: '127.0.0.1', fetch: () => Response.json({ data: [{ id: 'fixture-model', context_length: 32000 }] }) });
  const profileStore = new ProfileStore(join(directory, 'profiles.json'));
  profileStore.save({ name: 'fixture', provider: 'openai', apiKey: 'fixture-secret', baseUrl: `http://127.0.0.1:${endpoint.port}/v1`, model: 'configured-model' });
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, model: 'parent-model', sessionDirectory: join(directory, 'sessions') });
  const session = await runtime.openSession('owner');
  const server = new DaemonServer({ runtime, profileStore, agentSettingsStore: new AgentSettingsStore(join(directory, 'notes.sqlite')), socketPath: join(directory, 'daemon.sock'), cronLeasePath: join(directory, 'lease'), cronStoreFactory: () => new JobStore(join(directory, 'jobs.json')) });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 41, method: 'model.routing_note.save', params: { provider_profile: 'fixture', model: 'fixture-model', note: 'Use for careful review', revision: 0 } });
    expect((await client.next(frame => frame.id === 41)).result).toMatchObject({ ok: true, routing_note: { revision: 1 } });
    client.send({ jsonrpc: '2.0', id: 42, method: 'model.routing_note.save', params: { provider_profile: 'fixture', model: 'fixture-model', note: 'stale', revision: 0 } });
    expect((await client.next(frame => frame.id === 42)).error).toBeDefined();
    const profiles = await server.modelInventoryToolRequest(session.id, {});
    expect(JSON.stringify(profiles)).not.toContain('fixture-secret');
    for (const provider_profile of ['', ' \t\n']) {
      expect(await server.modelInventoryToolRequest(session.id, { provider_profile, include_usage: false, query: '', offset: 0, limit: 1, revision: '' })).toMatchObject({ source: 'configured_profiles', mode: 'providers', entries: [expect.objectContaining({ provider_profile: expect.any(String) })] });
      await expect(server.modelInventoryToolRequest(session.id, { provider_profile, include_usage: true })).rejects.toThrow('requires provider_profile');
    }
    await expect(server.modelInventoryToolRequest(session.id, { provider_profile: '' }, AbortSignal.abort(new Error('cancelled')))).rejects.toThrow('cancelled');
    await expect(server.modelInventoryToolRequest(session.id, { provider_profile: 'missing' })).rejects.toThrow('Unknown provider profile');
    const models = await server.modelInventoryToolRequest(session.id, { provider_profile: 'fixture' }) as { entries: Array<Record<string, unknown>>; quota: unknown };
    expect(await server.modelInventoryToolRequest(session.id, { provider_profile: '  fixture\t' })).toMatchObject({ entries: expect.arrayContaining([expect.objectContaining({ model: 'fixture-model' })]) });
    expect(models.entries).toContainEqual(expect.objectContaining({ model: 'fixture-model', provider_profile: 'fixture', context_window: 32000, reasoning_efforts: expect.any(Array) }));
    expect(JSON.stringify(models.entries)).toContain('Use for careful review');
    expect(models.quota).toMatchObject({ status: 'unknown' });
    await server.validateAgentProviderSelection('fixture', 'fixture-model', 'high');
    await expect(server.validateAgentProviderSelection('wrong-host-name', 'configured-model')).rejects.toThrow('Configured profiles for this model on the execution host: "fixture"');
    await expect(server.validateAgentProviderSelection('fixture', 'unknown-model')).rejects.toThrow('not configured or discovered');
    await expect(server.validateAgentProviderSelection('fixture', 'fixture-model', 'impossible')).rejects.toThrow('Unsupported reasoning');
    await expect(server.validateAgentProviderSelection('fixture', 'fixture-model', undefined, AbortSignal.abort(new Error('cancelled')))).rejects.toThrow('cancelled');
    expect(session.model).toBe('parent-model');
    expect(profileStore.active()?.model).toBe('configured-model');
    await expect(server.modelInventoryToolRequest('other', {})).rejects.toThrow('session unavailable');
  } finally { client.close(); await server.stop(); endpoint.stop(true); await rm(directory, { recursive: true, force: true }); }
});

test('workspace.diff reads untracked content in the daemon project without trusting a client cwd', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-rpc-diff-'));
  await Bun.spawn(['git', 'init', '-q', directory]).exited;
  await Bun.write(join(directory, 'new.ts'), 'export const remote = true\n');
  const server = new DaemonServer({ socketPath: join(directory, 'rpc.sock'), projectDirectory: directory, runtime: new InMemoryDaemonRuntime() });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'rpc.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'workspace.diff', params: { cwd: '/definitely/not/the/project' } });
    const response = await client.next(frame => frame.id === 1);
    expect(response.result).toMatchObject({ kind: 'ok', diff: { untracked: expect.arrayContaining(['new.ts']), lines: expect.arrayContaining([{ kind: 'add', text: '+export const remote = true', newLine: 1 }]) } });
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('git.* methods drive source control in the daemon project and report non-repositories plainly', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-rpc-git-'));
  const plain = await mkdtemp(join(tmpdir(), 'xr-rpc-nogit-'));
  const gitRun = async (...args: string[]) => { await Bun.spawn(['git', ...args], { cwd: directory, stdout: 'ignore', stderr: 'ignore' }).exited; };
  await gitRun('init', '-q', '-b', 'main');
  await gitRun('config', 'user.name', 'Test');
  await gitRun('config', 'user.email', 'test@example.com');
  await gitRun('config', 'commit.gpgsign', 'false');
  await Bun.write(join(directory, 'a.ts'), 'export const a = 1\n');
  const server = new DaemonServer({ socketPath: join(directory, 'rpc.sock'), projectDirectory: directory, runtime: new InMemoryDaemonRuntime() });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'rpc.sock'));
  let id = 0;
  const call = async (method: string, params: Record<string, unknown> = {}) => {
    const request = ++id;
    client.send({ jsonrpc: '2.0', id: request, method, params });
    return (await client.next(frame => frame.id === request)).result as Record<string, any>;
  };
  try {
    const initial = await call('git.status');
    expect(initial.repository).toMatchObject({ branch: 'main', hasHead: false });
    expect(initial.repository.untracked.map((f: { path: string }) => f.path)).toContain('a.ts');
    expect(await call('git.stage', { paths: ['../escape'] })).toMatchObject({ ok: false, code: 'git-error' });
    const staged = await call('git.stage', { paths: ['a.ts'] });
    expect(staged.status.staged).toEqual([{ path: 'a.ts', status: 'A' }]);
    const diff = await call('git.diff', { path: 'a.ts', staged: true });
    expect(diff.lines).toEqual(expect.arrayContaining([expect.objectContaining({ kind: 'add', text: '+export const a = 1' })]));
    expect(await call('git.commit', { message: '' })).toMatchObject({ ok: false, error: expect.stringContaining('commit message') });
    const committed = await call('git.commit', { message: 'Add a' });
    expect(committed).toMatchObject({ ok: true, commit: { subject: 'Add a' }, status: { hasHead: true, counts: { staged: 0 } } });
    expect((await call('git.log')).commits.map((c: { subject: string }) => c.subject)).toEqual(['Add a']);
    expect((await call('git.branches')).branches).toEqual([expect.objectContaining({ name: 'main', current: true })]);
    expect(await call('git.bogus')).toMatchObject({ ok: false, code: 'git-unknown-method' });
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }

  const bare = new DaemonServer({ socketPath: join(plain, 'rpc.sock'), projectDirectory: plain, runtime: new InMemoryDaemonRuntime() });
  await bare.start();
  const other = await SocketTestClient.connect(join(plain, 'rpc.sock'));
  try {
    other.send({ jsonrpc: '2.0', id: 1, method: 'git.status', params: {} });
    expect((await other.next(frame => frame.id === 1)).result).toEqual({ ok: true, repository: null, reason: 'This folder is not a git repository.' });
  } finally { other.close(); await bare.stop(); await rm(plain, { recursive: true, force: true }); }
});

test('terminal.open gives the desktop a live shell: replayed history, pushed output, input, resize, kill', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-rpc-term-'));
  const terminals = new TerminalRegistry();
  const ptySessions = new PtySessionManager({ terminals, workspaceRoot: directory });
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const server = new DaemonServer({ socketPath: join(directory, 'rpc.sock'), runtime, terminalRegistry: terminals, ptySessions });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'rpc.sock'));
  let id = 0;
  const call = async (method: string, params: Record<string, unknown> = {}) => {
    const request = ++id;
    client.send({ jsonrpc: '2.0', id: request, method, params: { session_key: 'term-owner', ...params } });
    return (await client.next(frame => frame.id === request)).result as Record<string, any>;
  };
  try {
    await call('session.open');
    const opened = await call('terminal.open', { cols: 100, rows: 30 });
    expect(opened.ok).toBe(true);
    const terminalId = opened.terminal_id as string;
    const attached = await call('terminal.attach', { terminal_id: terminalId });
    expect(attached).toMatchObject({ ok: true, running: true });
    await call('terminal.control', { terminal_id: terminalId, action: 'write', chars: 'echo xr-$((20+22))\r' });
    let seen = attached.data as string;
    while (!seen.includes('xr-42')) {
      const frame = await client.next(frame => frame.method === 'event' && frame.params?.type === 'terminal_output');
      const payload = frame.params!.payload as Record<string, unknown>;
      expect(payload.terminal_id).toBe(terminalId);
      seen += String(payload.data);
    }
    expect(await call('terminal.resize', { terminal_id: terminalId, cols: 120, rows: 40 })).toEqual({ ok: true });
    expect(await call('terminal.resize', { terminal_id: 'pty_foreign', cols: 1, rows: 1 })).toMatchObject({ ok: false });
    // The desktop tab finds its shells again by kind + empty command, and a
    // re-attach (tab switch, window reload) replays what already happened.
    expect((await call('terminal.list')).terminals).toEqual(expect.arrayContaining([expect.objectContaining({ id: terminalId, kind: 'pty', command: '', running: true })]));
    const reattached = await call('terminal.attach', { terminal_id: terminalId });
    expect(reattached.data).toContain('xr-42');
    await call('terminal.control', { terminal_id: terminalId, action: 'kill' });
    const closed = await client.next(frame => frame.method === 'event' && frame.params?.type === 'terminal_output' && (frame.params.payload as Record<string, unknown> | undefined)?.closed === true);
    expect((closed.params!.payload as Record<string, unknown>).terminal_id).toBe(terminalId);
    expect(await call('terminal.detach', { terminal_id: terminalId })).toEqual({ ok: true });
  } finally { client.close(); await ptySessions.disposeAll(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('background.status counts live session-owned shells and watchers while idle', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-background-status-'));
  const history = new RunHistory(join(directory, 'runs.sqlite'));
  const terminals = new TerminalRegistry();
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const monitors = new TerminalMonitors(terminals, history, () => {});
  const server = new DaemonServer({ socketPath: join(directory, 'rpc.sock'), runtime, terminalRegistry: terminals, monitors });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'rpc.sock'));
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'session.open', params: { session_key: 'background-owner' } });
    await client.next(frame => frame.id === 1);
    const owner = runtime.sessionStatus('background-owner')!.id;
    const shell = terminals.open({ id: 'background-shell', ownerSessionId: owner, cwd: directory, command: 'build', kind: 'background' });
    terminals.open({ id: 'another-session', ownerSessionId: 'other', cwd: directory, command: 'build', kind: 'background' });
    const watch = monitors.start(owner, { terminalId: 'background-shell', match: 'done' });
    const status = async (id: number) => { client.send({ jsonrpc: '2.0', id, method: 'background.status', params: {} }); return (await client.next(frame => frame.id === id)).result; };
    expect(await status(2)).toEqual({ ok: true, shells: 1, watchers: 1 });
    shell.close(0);
    monitors.stop(owner, watch.id);
    expect(await status(3)).toEqual({ ok: true, shells: 0, watchers: 0 });
  } finally { client.close(); monitors.close(); await server.stop(); history.close(); await rm(directory, { recursive: true, force: true }); }
});

test('activity pushes lifecycle changes, lists schedules and scopes process controls', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-activity-events-'));
  const history = new RunHistory(join(directory, 'runs.sqlite'));
  const terminals = new TerminalRegistry({ runHistory: history });
  const runtime = new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const monitors = new TerminalMonitors(terminals, history, () => {});
  const store = new JobStore(join(directory, 'jobs.json'));
  const server = new DaemonServer({ socketPath: join(directory, 'rpc.sock'), runtime, projectDirectory: directory, terminalRegistry: terminals, runHistory: history, monitors, cronStoreFactory: () => store });
  await server.start();
  const client = await SocketTestClient.connect(join(directory, 'rpc.sock'));
  let requestId = 0;
  const request = async (method: string, params: Record<string, unknown> = {}) => { const id = ++requestId; client.send({ jsonrpc:'2.0', id, method, params }); return (await client.next(frame => frame.id === id)).result; };
  try {
    await request('session.open', { session_key:'activity-owner' });
    const owner = runtime.sessionStatus('activity-owner')!.id;
    const shell = terminals.open({id:'owned',ownerSessionId:owner,cwd:directory,command:'bun test',kind:'background',control:{kill: async () => { shell.close(0); }}});
    await client.next(eventFrame('background_changed'));
    const watch = monitors.start(owner, { terminalId:shell.id,match:'done' });
    store.add(new CronJob({id:'queued',prompt:'Check the build',schedule:'0 9 * * *',projectRoot:directory,nextRunAt:'2099-01-01T09:00:00Z'}));
    terminals.open({id:'other',ownerSessionId:'other-session',cwd:directory,command:'private command',kind:'background'});
    const activity = await request('background.activity');
    expect(activity).toMatchObject({ok:true,rows:expect.arrayContaining([
      expect.objectContaining({id:'owned',kind:'shell',state:'running',action:'stop'}),
      expect.objectContaining({id:watch.id,kind:'watcher',state:'watching'}),
      expect.objectContaining({id:'queued',kind:'schedule',state:'scheduled',action:'pause'}),
    ])});
    expect(JSON.stringify(activity)).not.toContain('private command');
    expect(await request('terminal.control',{terminal_id:'other',action:'kill'})).toMatchObject({ok:false});
    expect(await request('terminal.control',{terminal_id:'owned',action:'kill'})).toMatchObject({ok:true});
    expect(await request('background.activity')).toMatchObject({rows:expect.arrayContaining([expect.objectContaining({id:'owned',state:'cancelled',endedAt:expect.any(Number),action:null})])});
    const schedule = { revision: Bun.hash(JSON.stringify(store.get('queued')!.toRecord())).toString(16) };
    expect(await request('schedule.pause',{schedule_id:'queued',revision:schedule.revision})).toMatchObject({ok:true});
    expect(await request('background.activity')).toMatchObject({rows:expect.arrayContaining([expect.objectContaining({id:'queued',state:'paused'})])});
    expect(await request('slash',{command:'/activity'})).toMatchObject({ok:true,rows:expect.any(Array)});
  } finally { client.close(); monitors.close(); await server.stop(); history.close(); await rm(directory,{recursive:true,force:true}); }
});

test('model selection binds profile atomically without poisoning another profile', async () => {
  const directory = await mkdtemp(join(tmpdir(),'xr-atomic-model-'));
  const profiles = new ProfileStore(join(directory,'profiles.json'));
  profiles.save({name:'kimi',provider:'kimi-code',model:'kimi-for-coding',apiKey:'fixture',baseUrl:'https://example.invalid'});
  const runtime = new InMemoryDaemonRuntime(undefined,{model:'kimi-for-coding',currentProjectDirectory:directory});
  const server = new DaemonServer({socketPath:join(directory,'rpc.sock'),runtime,profileStore:profiles});
  await server.start();
  const client = await SocketTestClient.connect(join(directory,'rpc.sock'));
  let next=0;
  const call=async(method:string,params:Record<string,unknown>)=>{const id=++next;client.send({jsonrpc:'2.0',id,method,params});return (await client.next(frame=>frame.id===id)).result;};
  try {
    await call('session.open',{session_key:'picker'});
    expect(await call('set_model',{model:'gpt-6-astra',provider_profile:'codex'})).toMatchObject({ok:true,model:'gpt-6-astra'});
    expect(runtime.sessionStatus('picker')?.metadata.provider_profile).toBe('codex');
    expect(profiles.get('kimi')?.model).toBe('kimi-for-coding');
    expect(await call('set_model',{model:'gpt-6-astra',provider_profile:'kimi'})).toMatchObject({ok:false});
    expect(runtime.sessionStatus('picker')?.metadata.provider_profile).toBe('codex');
    expect(await call('session.status',{})).toMatchObject({session:{model:'gpt-6-astra',profile_name:'codex'}});
  } finally {client.close();await server.stop();await rm(directory,{recursive:true,force:true});}
});

test('project agent generation uses the selected model and returns a draft without writing', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-agent-generation-'));
  const socketPath = join(directory, 'daemon.sock');
  const runtime = new InMemoryDaemonRuntime(undefined, { model: 'selected-model', currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  let requestedModel = '';
  let prompt = '';
  let fail = false;
  let closed = 0;
  const server = new DaemonServer({ socketPath, runtime, projectDirectory: directory, projectAgentClientFactory: (model) => {
    requestedModel = model;
    return {
      async *stream() {},
      async complete(request: CompletionRequest) {
        prompt = String(request.messages[0]?.content);
        if (fail) throw new Error('Generation provider failed');
        return { content: '---\nname: reviewer\ndescription: Review code\n---\nFind defects.', toolCalls: [] };
      },
      close() { closed++; },
    } as unknown as LlmClient;
  } });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'agentPreset.projectGenerate', params: { description: 'Find concurrency bugs' } });
    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({ ok: true, id: 'reviewer', revision: null });
    expect(requestedModel).toBe('selected-model');
    expect(prompt).toContain('Find concurrency bugs');
    expect(existsSync(join(directory, '.xerxes/agents/reviewer.md'))).toBe(false);
    expect(closed).toBe(1);
    fail = true;
    client.send({ jsonrpc: '2.0', id: 2, method: 'agentPreset.projectGenerate', params: { description: 'Review code' } });
    expect((await client.next(frame => frame.id === 2)).result).toMatchObject({ ok: false, error: expect.stringContaining('Generation provider failed') });
    expect(closed).toBe(2);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});


test('capability catalog exposes admitted skills and retained usage without activating previews', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-capabilities-'));
  const skills = join(directory, 'user-skills');
  await mkdir(join(skills, 'review'), { recursive: true });
  await writeFile(join(skills, 'review', 'SKILL.md'), '---\nname: review\ndescription: Review changes\n---\nInspect code carefully.');
  const runtime = new InMemoryDaemonRuntime({
    toolInventory: () => [{ name: 'ReadFile', exposure: 'loaded', reason: 'Fixture tool' }],
    async *run() { throw new Error('Preview must not run a provider'); },
  }, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') });
  const socketPath = join(directory, 'daemon.sock');
  const server = new DaemonServer({ socketPath, runtime, skillDirectories: [skills] });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'capabilities.list', params: {} });
    expect((await client.next(frame => frame.id === 1)).result).toMatchObject({ ok: false });
    client.send({ jsonrpc: '2.0', id: 2, method: 'initialize', params: { session_key: 'catalog', project_dir: directory } });
    await client.next(frame => frame.id === 2);
    await client.next(eventFrame('init_done')); await client.next(eventFrame('status_update'));
    const session = runtime.sessionStatus('catalog')!;
    session.toolExecutions.push({ name: 'ReadFile' }, { name: 'ReadFile' }, null);
    session.messages.push({ role: 'user', content: '[Skill review activated]\nInspect code carefully.' });
    const before = session.messages.length;
    client.send({ jsonrpc: '2.0', id: 3, method: 'capabilities.list', params: {} });
    expect((await client.next(frame => frame.id === 3)).result).toMatchObject({ ok: true, usage_scope: 'retained session history', skills: expect.arrayContaining([expect.objectContaining({ name: 'review', uses: 1 })]), tools: [expect.objectContaining({ name: 'ReadFile', uses: 2 })] });
    client.send({ jsonrpc: '2.0', id: 4, method: 'capabilities.inspect', params: { name: 'review' } });
    expect((await client.next(frame => frame.id === 4)).result).toMatchObject({ ok: true, instructions: expect.stringContaining('Inspect code carefully.') });
    client.send({ jsonrpc: '2.0', id: 5, method: 'capabilities.inspect', params: { name: '../../missing' } });
    expect((await client.next(frame => frame.id === 5)).result).toMatchObject({ ok: false });
    expect(session.messages.length).toBe(before);
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }); }
});

test('skill completion returns the full library beyond 200 entries', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-large-skill-menu-'))
  const skills = join(directory, 'skills')
  await Promise.all(Array.from({ length: 215 }, async (_, i) => {
    const name = `skill-${String(i).padStart(3, '0')}`
    const folder = join(skills, name)
    await mkdir(folder, { recursive: true })
    await writeFile(join(folder, 'SKILL.md'), `---\nname: ${name}\ndescription: Example ${i}\n---\nInstructions.`)
  }))
  const server = new DaemonServer({ socketPath: join(directory, 'daemon.sock'), skillDirectories: [skills], runtime: new InMemoryDaemonRuntime(undefined, { currentProjectDirectory: directory, sessionDirectory: join(directory, 'sessions') }) })
  await server.start()
  const client = await SocketTestClient.connect(join(directory, 'daemon.sock'))
  try {
    client.send({ jsonrpc: '2.0', id: 1, method: 'initialize', params: { session_key: 'library' } })
    await client.next(frame => frame.id === 1)
    client.send({ jsonrpc: '2.0', id: 2, method: 'complete', params: { text: '/skill ' } })
    const reply = await client.next(frame => frame.id === 2)
    const rows = reply.result?.completions as { label: string }[]
    expect(rows).toHaveLength(215)
    expect(rows.at(-1)?.label).toBe('skill-214')
  } finally { client.close(); await server.stop(); await rm(directory, { recursive: true, force: true }) }
})

test('paged initialize omits full replay and metadata refresh never includes history', async()=>{
 const directory=await mkdtemp(join(tmpdir(),'xerxes-history-page-'))
 const runtime=new InMemoryDaemonRuntime(undefined,{currentProjectDirectory:directory,sessionDirectory:join(directory,'sessions')})
 const session=await runtime.openSession('history-source')
 session.messages=Array.from({length:250},(_,i)=>({role:i%2?'assistant':'user',content:`History entry ${i}`}))
 session.turnCount=125
 await runtime.flushSessions()
 const server=new DaemonServer({runtime,projectDirectory:directory,socketPath:join(directory,'daemon.sock')})
 await server.start()
 const client=await SocketTestClient.connect(join(directory,'daemon.sock'))
 try{
  client.send({jsonrpc:'2.0',id:1,method:'initialize',params:{resume_session_id:session.id,history_limit:100}})
  type History = { actions: Array<{ messages: Array<{ content: unknown }> }>; before: string | null }
  const init=(await client.next(f=>f.id===1)).result as {session:{transcript?:unknown;tool_executions?:unknown;history:History}}
  expect(init.session.transcript).toBeUndefined()
  expect(init.session.tool_executions).toBeUndefined()
  expect(init.session.history.actions).toHaveLength(100)
  expect(JSON.stringify(init.session.history)).not.toContain('History entry 0"')
  expect(client.seen(frame=>JSON.stringify(frame).includes('replay_user'))).toBe(false)
  expect(client.seen(frame=>JSON.stringify(frame).includes('replay_assistant'))).toBe(false)
  client.send({jsonrpc:'2.0',id:2,method:'session.history',params:{before:init.session.history.before,history_limit:100}})
  const older=(await client.next(f=>f.id===2)).result as {history:History}
  expect(older.history.actions).toHaveLength(100)
  expect(older.history.actions[0]?.messages[0]?.content).toBe('History entry 50')
  client.send({jsonrpc:'2.0',id:3,method:'session.active_list',params:{history_limit:0}})
  const active=(await client.next(f=>f.id===3)).result as {sessions:Array<Record<string,unknown>>}
  expect(active.sessions.every(s=>!('transcript' in s)&&!('tool_executions' in s)&&!('history' in s))).toBe(true)
  client.send({jsonrpc:'2.0',id:4,method:'session.history',params:{before:'invalid',history_limit:100}})
  expect((await client.next(f=>f.id===4)).error).toBeDefined()
 }finally{client.close();await server.stop();await rm(directory,{recursive:true,force:true})}
})

test('session-owned turns finish after the client closes and retain their saved output', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-detached-finish-'));
  const socketPath = join(directory, 'rpc.sock');
  const runner = new GatedRunner();
  const runtime = new InMemoryDaemonRuntime(runner, {model:'fixture',currentProjectDirectory:directory,sessionDirectory:join(directory,'sessions')});
  const server = new DaemonServer({runtime,socketPath,projectDirectory:directory,cronStoreFactory:()=>new JobStore(join(directory,'cron/jobs.json'))});
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  let id = 0;
  const rpc = async (method: string, params: Record<string, unknown> = {}) => { const request = ++id; client.send({jsonrpc:'2.0',id:request,method,params}); return client.next(frame=>frame.id===request) };
  try {
    const initialized = await rpc('initialize',{session_key:'durable',session_owned_turns:true});
    expect(initialized.result?.session_owned_turns_supported).toBe(true);
    await rpc('turn.submit',{text:'finish without this window'});
    await client.next(eventFrame('text_part'));
    const sessionId = runtime.sessionStatus('durable')!.id;
    client.close();await Bun.sleep(50);
    expect(runtime.sessionStatus('durable')?.cancelRequested).toBe(false);
    runner.release();
    await waitFor(()=>runtime.sessionStatus('durable')?.activeTurnId==='');
    await runtime.flushSessions();
    const saved = await new DaemonTranscriptStore({directory:join(directory,'sessions')}).load(sessionId);
    expect(JSON.stringify(saved)).toContain('waitingdone');
    expect(runtime.sessionStatus('durable')?.cancelRequested).toBe(false);
  } finally {runner.release();client.close();await server.stop();await rm(directory,{recursive:true,force:true})}
});

test('session-owned turns survive lease expiry, reattach to a new client and still accept explicit cancellation', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xr-detached-cancel-'));
  const socketPath = join(directory,'rpc.sock');
  const runner = new AbortGateRunner();
  const runtime = new InMemoryDaemonRuntime(runner,{model:'fixture',currentProjectDirectory:directory,sessionDirectory:join(directory,'sessions')});
  const server = new DaemonServer({runtime,socketPath,projectDirectory:directory,cronStoreFactory:()=>new JobStore(join(directory,'cron/jobs.json'))});await server.start();
  const first = await SocketTestClient.connect(socketPath), second = await SocketTestClient.connect(socketPath);
  let id=0;
  const rpc = async (client: SocketTestClient, method: string, params: Record<string,unknown>={}) => {const request=++id;client.send({jsonrpc:'2.0',id:request,method,params});return client.next(frame=>frame.id===request)};
  try {
    await rpc(first,'initialize',{session_key:'durable',session_owned_turns:true});
    await rpc(first,'connection.lease');
    await rpc(first,'turn.submit',{text:'keep working'});await first.next(eventFrame('text_part'));
    const sessionId=runtime.sessionStatus('durable')!.id;
    first.close();await Bun.sleep(30);
    // Expire the real lease through its production cleanup path, without a 30s test sleep.
    (server as unknown as {connectionLeases:{close():void}}).connectionLeases.close();
    expect(runtime.sessionStatus('durable')?.cancelRequested).toBe(false);
    expect(runtime.sessionStatus('durable')?.activeTurnId).not.toBe('');
    const resumed=await rpc(second,'initialize',{resume_session_id:sessionId,project_dir:directory,session_owned_turns:true});
    expect(resumed.result?.ok).toBe(true);
    expect(runner.runs).toBe(1);
    expect((await rpc(second,'turn.cancel')).result?.ok).toBe(true);
    await second.next(eventFrame('turn_end'));
    await waitFor(()=>runtime.sessionStatus('durable')?.activeTurnId==='');
    expect(runtime.sessionStatus('durable')?.cancelRequested).toBe(true);
  } finally {first.close();second.close();await server.stop();await rm(directory,{recursive:true,force:true})}
});

test('session-owned permission waits survive close and can only be answered by a client attached to that session', async () => {
  const directory=await mkdtemp(join(tmpdir(),'xr-detached-approval-'));
  const socketPath=join(directory,'rpc.sock');const interactions=new DaemonInteractionBoard();
  const runtime=new InMemoryDaemonRuntime(new ReplyRunner(interactions),{model:'fixture',currentProjectDirectory:directory,sessionDirectory:join(directory,'sessions'),interactions});
  const server=new DaemonServer({runtime,socketPath,interactions,projectDirectory:directory,cronStoreFactory:()=>new JobStore(join(directory,'cron/jobs.json'))});await server.start();
  const first=await SocketTestClient.connect(socketPath),second=await SocketTestClient.connect(socketPath),stranger=await SocketTestClient.connect(socketPath);
  let id=0;
  const rpc=async(client:SocketTestClient,method:string,params:Record<string,unknown>={})=>{const request=++id;client.send({jsonrpc:'2.0',id:request,method,params});return client.next(frame=>frame.id===request)};
  try{
    await rpc(first,'initialize',{session_key:'durable',session_owned_turns:true});
    await rpc(first,'turn.submit',{text:'wait for my return'});await first.next(eventFrame('approval_request'));
    const sessionId=runtime.sessionStatus('durable')!.id;
    first.close();await Bun.sleep(30);expect(interactions.pendingPermissionIds()).toEqual(['approval-1']);
    await rpc(stranger,'initialize',{session_key:'other',session_owned_turns:true});
    expect((await rpc(stranger,'permission_response',{request_id:'approval-1',response:'approve'})).result?.ok).toBe(false);
    const resumed=await rpc(second,'initialize',{resume_session_id:sessionId,project_dir:directory,session_owned_turns:true});
    expect(resumed.result?.pending_interactions).toMatchObject([{type:'approval_request',payload:{id:'approval-1'}}]);
    expect((await rpc(second,'permission_response',{request_id:'approval-1',response:'approve'})).result?.ok).toBe(true);
    await second.next(eventFrame('question_request'));
    expect(interactions.pendingQuestionIds().length).toBe(1);
    const question=interactions.pendingQuestionIds()[0]!;
    expect((await rpc(stranger,'question_response',{request_id:question,answers:{answer:'yes'}})).result?.ok).toBe(false);
    expect((await rpc(second,'question_response',{request_id:question,answers:{answer:'yes'}})).result?.ok).toBe(true);
    await second.next(eventFrame('turn_end'));expect(runtime.sessionStatus('durable')?.cancelRequested).toBe(false);
  }finally{first.close();second.close();stranger.close();await server.stop();await rm(directory,{recursive:true,force:true})}
});

test('session-owned background work keeps scoped progress and completes after its parent client closes', async () => {
  const directory=await mkdtemp(join(tmpdir(),'xr-detached-background-'));
  const runner=new GatedRunner();const socketPath=join(directory,'rpc.sock');
  const runtime=new InMemoryDaemonRuntime(runner,{model:'fixture',currentProjectDirectory:directory,sessionDirectory:join(directory,'sessions')});
  const server=new DaemonServer({runtime,socketPath,projectDirectory:directory,cronStoreFactory:()=>new JobStore(join(directory,'cron/jobs.json'))});await server.start();
  const client=await SocketTestClient.connect(socketPath);let id=0;
  const rpc=async(method:string,params:Record<string,unknown>={})=>{const request=++id;client.send({jsonrpc:'2.0',id:request,method,params});return client.next(frame=>frame.id===request)};
  try{
    await rpc('initialize',{session_key:'parent',session_owned_turns:true});
    const started=await rpc('turn.background',{text:'background work'});
    const frame=await client.next(eventFrame('text_part'));
    expect(frame.params?.payload).toMatchObject({background_task_id:started.result?.task_id,session_id:started.result?.task_id,text:'waiting'});
    const background=runtime.listSessions().find(session=>session.id===started.result?.task_id)!;
    client.close();await Bun.sleep(30);expect(background.cancelRequested).toBe(false);
    runner.release();await waitFor(()=>background.activeTurnId==='');
    expect(JSON.stringify(background.messages)).toContain('waitingdone');
  }finally{runner.release();client.close();await server.stop();await rm(directory,{recursive:true,force:true})}
});

test('subagent.inspect scopes retained evidence to the parent across reconnect and terminal states', async () => {
  const directory=await mkdtemp(join(tmpdir(),'xr-agent-inspect-'));
  const socketPath=join(directory,'rpc.sock');
  const runtime=new InMemoryDaemonRuntime(new UsageRunner(),{model:'fixture',currentProjectDirectory:directory,sessionDirectory:join(directory,'sessions')});
  const server=new DaemonServer({runtime,socketPath,projectDirectory:directory,cronStoreFactory:()=>new JobStore(join(directory,'cron/jobs.json'))});
  await server.start();let client=await SocketTestClient.connect(socketPath);let id=0;
  const rpc=async(method:string,params:Record<string,unknown>={})=>{const request=++id;client.send({jsonrpc:'2.0',id:request,method,params});return client.next(frame=>frame.id===request)};
  try {
    await rpc('initialize',{session_key:'parent'});
    await rpc('turn.submit',{text:'Inspect the child'});
    await client.next(eventFrame('turn_end'));
    const parent=runtime.sessionStatus('parent')!;
    parent.metadata.xerxes_subagent_snapshots_v1=[{id:'child',status:'running',agent_id:'reviewer',model:'fixture',provider_profile:'work',reasoning_effort:'high',last_input:'Review the change',last_output:'Reading files\n',private_config:'must not be returned'}];
    let result=await rpc('subagent.inspect',{task:'child'});
    expect(result.result?.agent).toMatchObject({id:'child',agent_id:'reviewer',provider_profile:'work',reasoning_effort:'high',prompt:'Review the change',output:'Reading files\n'});
    expect(JSON.stringify(result)).not.toContain('must not be returned');
    const status=await rpc('session.status');
    expect(JSON.stringify(status)).not.toContain('Review the change');
    expect(JSON.stringify(status)).not.toContain('Reading files');
    await rpc('initialize',{session_key:'foreign'});
    expect((await rpc('subagent.inspect',{task:'child'})).result?.ok).toBe(false);
    expect((await rpc('subagent.inspect',{})).result?.ok).toBe(false);
    client.close();client=await SocketTestClient.connect(socketPath);
    await rpc('initialize',{resume_session_id:parent.id});
    for(const state of ['failed','interrupted','completed']){
      parent.metadata.xerxes_subagent_snapshots_v1=[{id:'child',status:state,agent_id:'reviewer',last_output:'Retained result'}];
      result=await rpc('subagent.inspect',{task:'child'});
      expect(result.result?.agent).toMatchObject({id:'child',status:state,output:'Retained result'});
    }
  }finally{client.close();await server.stop();await rm(directory,{recursive:true,force:true})}
});
