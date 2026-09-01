// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ProfileStore } from "../src/bridge/profiles.js";
import { InMemoryDaemonRuntime } from "../src/daemon/runtime.js";
import { PROVIDER_FLOW_ADD_LABEL } from "../src/daemon/providerFlow.js";
import { DaemonServer } from "../src/daemon/server.js";
import { appendSkillSuggestion } from "../src/extensions/skillSuggestions.js";
import { DeclarativeToolForge } from "../src/extensions/declarativeForge.js";
import type { FetchImplementation } from "../src/llms/client.js";

interface Frame {
  readonly id?: number;
  readonly method?: string;
  readonly params?: {
    readonly payload?: Record<string, unknown>;
    readonly type?: string;
  };
  readonly result?: Record<string, unknown>;
}

interface ProviderPrompt {
  readonly questionId: string;
  readonly requestId: string;
}

test("daemon routes interactive provider setup through native question replies without credential echoing", async () => {
  const directory = await mkdtemp(
    join(tmpdir(), "xerxes-bun-provider-daemon-"),
  );
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  const nativeFetch = globalThis.fetch;
  const nativeOpenAiKey = process.env.OPENAI_API_KEY;
  const requests: Array<{ authorization: string | null; url: string }> = [];
  const modelFetch: FetchImplementation = async (input, init) => {
    requests.push({
      authorization: new Headers(init?.headers).get("authorization"),
      url: String(input),
    });
    return new Response(JSON.stringify({ data: [{ id: "daemon-remote-model" }] }));
  };
  globalThis.fetch = modelFetch as typeof globalThis.fetch;
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    sessionDirectory: join(directory, "sessions"),
  });
  const declarativeForge = new DeclarativeToolForge(join(directory, "forged-tools.json"));
  const server = new DaemonServer({ socketPath, profileStore, runtime, declarativeForge });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  process.env.OPENAI_API_KEY = "environment-must-not-leak";
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "provider-flow" },
    });
    expect((await client.next((frame) => frame.id === 1)).result).toMatchObject(
      { ok: true },
    );
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    const session = runtime.sessionStatus("provider-flow");
    if (!session) throw new Error("expected initialized provider-flow session");
    appendSkillSuggestion(session.metadata, {
      skillName: "release-checklist",
      description: "Repeat the verified release sequence.",
      version: "0.1.0",
      sourcePath: "/tmp/release-checklist/SKILL.md",
      toolCount: 4,
      uniqueTools: ["Read", "Bash"],
    });
    client.send({ jsonrpc: "2.0", id: 20, method: "skill_suggestions", params: {} });
    expect((await client.next((frame) => frame.id === 20)).result).toMatchObject({
      ok: true,
      suggestions: [{
        skill_name: "release-checklist",
        tool_count: 4,
        unique_tools: ["Read", "Bash"],
      }],
    });

    client.send({
      jsonrpc: "2.0",
      id: 21,
      method: "forge.define",
      params: {
        name: "briefing",
        version: "0.1.0",
        description: "Format a briefing.",
        parameters: [{ name: "topic", required: true }],
        template: "Briefing: {{topic}}",
      },
    });
    expect((await client.next((frame) => frame.id === 21)).result).toMatchObject({
      ok: false,
      error: expect.stringContaining("confirm: true"),
    });
    client.send({
      jsonrpc: "2.0",
      id: 22,
      method: "forge.define",
      params: {
        confirm: true,
        name: "briefing",
        version: "0.1.0",
        description: "Format a briefing.",
        parameters: [{ name: "topic", required: true }],
        template: "Briefing: {{topic}}",
      },
    });
    expect((await client.next((frame) => frame.id === 22)).result).toMatchObject({
      ok: true,
      package: { name: "briefing", version: "0.1.0" },
    });
    client.send({
      jsonrpc: "2.0",
      id: 23,
      method: "forge.run",
      params: { name: "briefing", input: { topic: "telemetry" } },
    });
    expect((await client.next((frame) => frame.id === 23)).result).toMatchObject({
      ok: true,
      output: "Briefing: telemetry",
    });
    client.send({ jsonrpc: "2.0", id: 24, method: "creator_trace", params: {} });
    expect((await client.next((frame) => frame.id === 24)).result).toMatchObject({
      ok: true,
      trace: [
        { action: "define", name: "briefing", status: "error" },
        { action: "define", name: "briefing", status: "ok" },
        { action: "run", name: "briefing", status: "ok" },
      ],
    });

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "slash",
      params: { command: "/provider" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toEqual({
      ok: true,
    });
    const firstPromptFrame = await client.next(eventFrame("question_request"));
    expect(firstPromptFrame.params?.payload?.flow).toBe("provider");
    let prompt = providerPrompt(firstPromptFrame);
    expect(prompt.questionId).toBe("action");

    prompt = await answerAndNext(client, 3, prompt, PROVIDER_FLOW_ADD_LABEL);
    expect(prompt.questionId).toBe("name");
    prompt = await answerAndNext(client, 4, prompt, "daemon-profile");
    expect(prompt.questionId).toBe("provider_type");
    prompt = await answerAndNext(client, 5, prompt, "openai");
    expect(prompt.questionId).toBe("base_url");
    prompt = await answerAndNext(client, 6, prompt, "");
    expect(prompt.questionId).toBe("api_key");
    prompt = await answerAndNext(client, 7, prompt, "");
    expect(prompt.questionId).toBe("model");

    client.send({
      jsonrpc: "2.0",
      id: 8,
      method: "question_response",
      params: {
        request_id: prompt.requestId,
        answers: { [prompt.questionId]: "daemon-remote-model" },
      },
    });
    expect((await client.next((frame) => frame.id === 8)).result).toEqual({
      ok: true,
      completed: true,
    });
    await client.next(eventFrame("question_response"));
    const notice = await client.next(eventFrame("notification"));
    expect(notice.params?.payload?.body).toContain("daemon-profile");
    expect(JSON.stringify(notice)).not.toContain("environment-must-not-leak");
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    client.send({ jsonrpc: "2.0", id: 9, method: "provider_list", params: {} });
    const listed = await client.next((frame) => frame.id === 9);
    expect(listed.result?.profiles).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "daemon-profile",
          model: "daemon-remote-model",
          active: true,
        }),
      ]),
    );
    expect(JSON.stringify(listed.result)).not.toContain("environment-must-not-leak");
    expect(profileStore.get("daemon-profile")?.api_key).toBe("");

    // Edit sheets can discover one exact saved profile without switching the
    // active runtime connection or receiving any credential material.
    client.send({
      jsonrpc: "2.0",
      id: 10,
      method: "provider_models",
      params: { profile_name: "daemon-profile" },
    });
    const catalog = await client.next((frame) => frame.id === 10);
    expect(catalog.result).toMatchObject({
      ok: true,
      profile: "daemon-profile",
      provider: "openai",
      configured_model: "daemon-remote-model",
      models: ["daemon-remote-model"],
    });
    expect(JSON.stringify(catalog.result)).not.toContain("environment-must-not-leak");
    expect(requests).toEqual([
      {
        authorization: null,
        url: "https://api.openai.com/v1/models",
      },
      {
        authorization: "Bearer environment-must-not-leak",
        url: "https://api.openai.com/v1/models",
      },
    ]);
  } finally {
    globalThis.fetch = nativeFetch;
    if (nativeOpenAiKey === undefined) {
      delete process.env.OPENAI_API_KEY;
    } else {
      process.env.OPENAI_API_KEY = nativeOpenAiKey;
    }
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("provider_types exposes the registry catalog; provider_save defaults endpoints and keeps stored keys", async () => {
  const directory = await mkdtemp(
    join(tmpdir(), "xerxes-bun-provider-types-"),
  );
  const socketPath = join(directory, "daemon.sock");
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  const server = new DaemonServer({
    socketPath,
    profileStore,
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    // The adapter catalog: registry names, default endpoints, env fallback
    // names — catalog facts only, never key material.
    client.send({ jsonrpc: "2.0", id: 1, method: "provider_types", params: {} });
    const types = (await client.next((frame) => frame.id === 1)).result?.types;
    expect(Array.isArray(types)).toBe(true);
    expect(types as unknown[]).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: "zhipu",
          transport: "openai",
          base_url: "https://api.z.ai/api/coding/paas/v4",
          api_key_env: "ZHIPU_API_KEY",
        }),
        // Types without a default endpoint say so explicitly.
        expect.objectContaining({ name: "custom", base_url: null }),
      ]),
    );

    // A known type with a blank base_url saves the registry default.
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "provider_save",
      params: { name: "zai", provider: "zhipu", model: "glm-5.2", api_key: "stored-key" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({
      ok: true,
      profile: { name: "zai", base_url: "https://api.z.ai/api/coding/paas/v4", active: true },
    });

    // An edit that omits api_key keeps the stored credential — the desktop
    // edit flow never re-sends the key.
    client.send({
      jsonrpc: "2.0",
      id: 3,
      method: "provider_save",
      params: { name: "zai", provider: "zhipu", model: "glm-5.3" },
    });
    expect((await client.next((frame) => frame.id === 3)).result).toMatchObject({ ok: true });
    const stored = profileStore.list().find(p => p.name === "zai");
    expect(stored?.api_key).toBe("stored-key");
    expect(stored?.model).toBe("glm-5.3");

    // A type without a registry default still must name its endpoint.
    client.send({
      jsonrpc: "2.0",
      id: 4,
      method: "provider_save",
      params: { name: "mine", provider: "custom", model: "qwen3.8-max" },
    });
    expect((await client.next((frame) => frame.id === 4)).result?.ok).toBe(false);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

async function answerAndNext(
  client: SocketTestClient,
  id: number,
  prompt: ProviderPrompt,
  answer: string,
): Promise<ProviderPrompt> {
  client.send({
    jsonrpc: "2.0",
    id,
    method: "question_response",
    params: {
      request_id: prompt.requestId,
      answers: { [prompt.questionId]: answer },
    },
  });
  expect((await client.next((frame) => frame.id === id)).result).toEqual({
    ok: true,
  });
  await client.next(eventFrame("question_response"));
  return providerPrompt(await client.next(eventFrame("question_request")));
}

function eventFrame(type: string): (frame: Frame) => boolean {
  return (frame) => frame.method === "event" && frame.params?.type === type;
}

function providerPrompt(frame: Frame): ProviderPrompt {
  const payload = frame.params?.payload;
  if (
    !payload ||
    typeof payload.id !== "string" ||
    !Array.isArray(payload.questions)
  ) {
    throw new Error("Expected a daemon question request");
  }
  const first = payload.questions[0];
  if (!isRecord(first) || typeof first.id !== "string") {
    throw new Error("Expected one provider-flow question");
  }
  return { questionId: first.id, requestId: payload.id };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
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
