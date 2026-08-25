// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  MAX_TRANSCRIPT_INLINE_IMAGE_BYTES,
  MAX_TRANSCRIPT_TOTAL_INLINE_IMAGE_BYTES,
  MAX_TURN_IMAGE_BYTES,
  MAX_TURN_IMAGES,
  MAX_TURN_IMAGES_TOTAL_BYTES,
  imageUrlContentParts,
  validateTurnImages,
} from "../src/daemon/images.js";
import { InMemoryDaemonRuntime } from "../src/daemon/runtime.js";
import { DaemonServer } from "../src/daemon/server.js";
import { AgentTurnRunner } from "../src/daemon/turnRunner.js";
import { ValidationError } from "../src/core/errors.js";
import type { CompletionRequest, LlmClient, LlmDelta } from "../src/llms/client.js";
import {
  DAEMON_SESSION_SCHEMA_VERSION,
  DaemonTranscriptStore,
  daemonTranscriptRecord,
} from "../src/session/daemonTranscript.js";

const PNG_BYTES = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 1, 2, 3, 4]);
const JPEG_BYTES = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 16, 1, 2]);
const PNG_B64 = Buffer.from(PNG_BYTES).toString("base64");
const JPEG_B64 = Buffer.from(JPEG_BYTES).toString("base64");

test("validateTurnImages accepts a well-formed attachment and canonicalizes it", () => {
  const images = validateTurnImages([{ media_type: "image/png", data: ` ${PNG_B64}\n` }]);
  expect(images).toEqual([{ data: PNG_B64, mediaType: "image/png" }]);
  expect(imageUrlContentParts(images)).toEqual([
    { type: "image_url", image_url: { url: `data:image/png;base64,${PNG_B64}` } },
  ]);
  expect(validateTurnImages(undefined)).toEqual([]);
});

test("validateTurnImages rejects malformed entries with typed errors", () => {
  expect(() => validateTurnImages("not-an-array")).toThrow(ValidationError);
  expect(() => validateTurnImages([{ media_type: "image/png", data: "!!!not-base64!!!" }])).toThrow(
    /valid base64/,
  );
  expect(() => validateTurnImages([{ media_type: "image/png", data: "" }])).toThrow(ValidationError);
  // Plain text is valid base64 but fails the magic-byte sniff.
  expect(() =>
    validateTurnImages([{ media_type: "image/png", data: Buffer.from("hello world").toString("base64") }]),
  ).toThrow(/magic-byte/);
  // Declared mime must match the sniffed payload.
  expect(() => validateTurnImages([{ media_type: "image/webp", data: PNG_B64 }])).toThrow(/sniffs as/);
  // Too many entries.
  expect(() =>
    validateTurnImages(
      Array.from({ length: MAX_TURN_IMAGES + 1 }, () => ({ media_type: "image/png", data: PNG_B64 })),
    ),
  ).toThrow(/at most/);
});

test("validateTurnImages enforces per-image and per-turn byte caps without truncation", () => {
  const oversized = new Uint8Array(MAX_TURN_IMAGE_BYTES + 1);
  oversized.set(PNG_BYTES);
  expect(() =>
    validateTurnImages([{ media_type: "image/png", data: Buffer.from(oversized).toString("base64") }]),
  ).toThrow(/per-image limit/);

  // Three images that individually fit the per-image cap but exceed the turn total.
  const each = new Uint8Array(Math.floor(MAX_TURN_IMAGES_TOTAL_BYTES / 3) + 1);
  each.set(PNG_BYTES);
  const encoded = Buffer.from(each).toString("base64");
  expect(each.byteLength).toBeLessThanOrEqual(MAX_TURN_IMAGE_BYTES);
  expect(() =>
    validateTurnImages([
      { media_type: "image/png", data: encoded },
      { media_type: "image/png", data: encoded },
      { media_type: "image/png", data: encoded },
    ]),
  ).toThrow(/combined turn limit/);
});

test("daemon turn.submit rejects invalid images at the RPC boundary", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-turn-images-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
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
      method: "turn.submit",
      params: {
        session_key: "images-session",
        text: "look at this",
        images: [{ media_type: "image/png", data: Buffer.from("not an image").toString("base64") }],
      },
    });
    const rejected = await client.next((frame) => frame.id === 1);
    expect(rejected.result?.ok).toBe(false);
    expect(String(rejected.result?.error)).toMatch(/magic-byte/);

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: "images-session", text: "look at this", images: "nope" },
    });
    const malformed = await client.next((frame) => frame.id === 2);
    expect(malformed.result?.ok).toBe(false);
    expect(String(malformed.result?.error)).toMatch(/must be an array/);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("turn.submit images reach the provider as image_url parts and round-trip the transcript", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-turn-images-e2e-"));
  const socketPath = join(directory, "daemon.sock");
  const llm = new CapturingClient();
  const transcriptStore = new DaemonTranscriptStore({
    currentProjectDirectory: directory,
    directory: join(directory, "sessions"),
  });
  const runtime = new InMemoryDaemonRuntime(
    new AgentTurnRunner({ llm, model: "vision-model" }),
    {
      currentProjectDirectory: directory,
      model: "vision-model",
      transcriptStore,
    },
  );
  const server = new DaemonServer({ socketPath, runtime });
  await server.start();
  const client = await SocketTestClient.connect(socketPath);
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "turn.submit",
      params: {
        session_key: "vision-session",
        text: "what is in these images?",
        images: [
          { media_type: "image/png", data: PNG_B64 },
          { media_type: "image/jpeg", data: JPEG_B64 },
        ],
      },
    });
    expect((await client.next((frame) => frame.id === 1)).result).toMatchObject({ ok: true });
    await client.next(eventFrame("turn_begin"));
    await client.next(eventFrame("turn_end"));

    // The fake provider saw one user message with text + image_url parts.
    expect(llm.requests).toHaveLength(1);
    const userMessage = llm.requests[0]!.messages.findLast((message) => message.role === "user");
    expect(userMessage?.content).toEqual([
      { type: "text", text: "what is in these images?" },
      { type: "image_url", image_url: { url: `data:image/png;base64,${PNG_B64}` } },
      { type: "image_url", image_url: { url: `data:image/jpeg;base64,${JPEG_B64}` } },
    ]);

    // The persisted transcript kept the structured parts and the display text.
    const session = runtime.sessionStatus("vision-session");
    expect(session).toBeDefined();
    const persisted = await transcriptStore.load(session!.id);
    expect(persisted).toBeDefined();
    const persistedUser = persisted!.messages.find((message) => message.role === "user");
    expect(persistedUser?.content).toEqual(userMessage?.content);

    // A resumed session (fresh session key bound to the persisted resume id)
    // feeds the same parts back to the provider.
    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "turn.submit",
      params: { session_key: session!.id, text: "and now?" },
    });
    expect((await client.next((frame) => frame.id === 2)).result).toMatchObject({ ok: true });
    await client.next(eventFrame("turn_end"));
    expect(llm.requests).toHaveLength(2);
    const historyUser = llm.requests[1]!.messages.find(
      (message) => message.role === "user" && Array.isArray(message.content),
    );
    expect(historyUser?.content).toEqual(userMessage?.content);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

test("transcript store round-trips structured content and still loads legacy plain-text transcripts", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-transcript-parts-"));
  try {
    const store = new DaemonTranscriptStore({
      currentProjectDirectory: directory,
      directory,
    });
    const parts = [
      { type: "text", text: "with image" },
      { type: "image_url", image_url: { url: `data:image/png;base64,${PNG_B64}` } },
    ];
    const record = daemonTranscriptRecord({
      agentId: "default",
      cwd: directory,
      extra: {},
      format: "bun-v2",
      interactionMode: "code",
      key: "aabbccdd",
      messages: [
        { role: "user", content: "legacy plain text" },
        { role: "assistant", content: "reply" },
        { role: "user", content: parts, text: "with image" },
      ],
      metadata: {},
      pendingResumeReplays: [],
      planMode: false,
      schemaVersion: DAEMON_SESSION_SCHEMA_VERSION,
      sessionId: "aabbccdd",
      thinkingContent: [],
      toolExecutions: [],
      totalInputTokens: 0,
      totalOutputTokens: 0,
      turnCount: 2,
      updatedAt: new Date().toISOString(),
      workspace: directory,
    });
    expect(record.schema_version).toBe(DAEMON_SESSION_SCHEMA_VERSION);
    await writeFile(store.pathFor("aabbccdd"), JSON.stringify(record));

    const loaded = await store.load("aabbccdd");
    expect(loaded).toBeDefined();
    expect(loaded!.messages[0]?.content).toBe("legacy plain text");
    expect(loaded!.messages[2]?.content).toEqual(parts);
    expect(loaded!.messages[2]?.text).toBe("with image");

    // An unversioned legacy record without the Bun format marker still loads.
    await writeFile(
      store.pathFor("bbccddee"),
      JSON.stringify({
        session_id: "bbccddee",
        cwd: directory,
        messages: [
          { role: "user", content: "old plain user message" },
          { role: "assistant", content: "old reply" },
        ],
        turn_count: 1,
      }),
    );
    const legacy = await store.load("bbccddee");
    expect(legacy).toBeDefined();
    expect(legacy!.format).toBe("legacy-v1");
    expect(legacy!.messages[0]?.content).toBe("old plain user message");
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test("session payloads omit oversized inline images but keep provider-facing parts intact", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-transcript-omit-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "vision-model",
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
      params: { session_key: "image-echo", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    // A payload above the per-message inline budget (a ~250KB decoded image
    // becomes a ~333KB base64 data URL) plus a tiny one that fits.
    const hugeBytes = new Uint8Array(250_000);
    hugeBytes.set(PNG_BYTES);
    const hugeB64 = Buffer.from(hugeBytes).toString("base64");
    const session = runtime.sessionStatus("image-echo");
    if (!session) throw new Error("expected live session");
    session.messages.push(
      {
        role: "user",
        content: [
          { type: "text", text: "look at this" },
          { type: "image_url", image_url: { url: `data:image/png;base64,${PNG_B64}` } },
          { type: "image_url", image_url: { url: `data:image/png;base64,${hugeB64}` } },
        ],
        text: "look at this",
      },
      { role: "assistant", content: "I see it." },
    );

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "session.open",
      params: { session_key: "image-echo" },
    });
    const opened = await client.next((frame) => frame.id === 2);
    expect(opened.result?.ok).toBe(true);
    const echoed = opened.result?.session as Record<string, unknown>;
    expect(echoed.message_count).toBe(2);

    // Part count and ordering survive; only the over-budget data URL is
    // replaced with a compact placeholder.
    const transcript = echoed.transcript as Array<Record<string, unknown>>;
    expect(transcript).toHaveLength(2);
    const userParts = transcript[0]?.content as Array<Record<string, unknown>>;
    expect(userParts).toHaveLength(3);
    expect(userParts[1]).toEqual({
      type: "image_url",
      image_url: { url: `data:image/png;base64,${PNG_B64}` },
    });
    expect(userParts[2]?.type).toBe("text");
    expect(String(userParts[2]?.text)).toMatch(/^\[image omitted: \d+ KB\]$/);
    expect(echoed.transcript_images_omitted).toBe(1);
    expect(JSON.stringify(opened.result)).not.toContain(hugeB64.slice(0, 64));

    // The live session keeps full images: turn submission and the provider
    // must never see the placeholder.
    const liveContent = session.messages[0]?.content as Array<Record<string, unknown>>;
    expect(liveContent[2]).toEqual({
      type: "image_url",
      image_url: { url: `data:image/png;base64,${hugeB64}` },
    });
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

class CapturingClient implements LlmClient {
  readonly requests: CompletionRequest[] = [];

  async *stream(request: CompletionRequest): AsyncGenerator<LlmDelta> {
    this.requests.push(request);
    yield { content: "I see the image.", usage: { inputTokens: 3, outputTokens: 2 } };
  }
}

interface Frame {
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

class SocketTestClient {
  private buffer = "";
  private readonly frames: Frame[] = [];
  private readonly waiters: Array<{
    predicate: (frame: Frame) => boolean;
    resolve: (frame: Frame) => void;
  }> = [];

  private constructor(private readonly socket: Socket) {
    socket.setEncoding("utf8");
    socket.on("data", (chunk) => this.receive(typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk)));
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
    for (;;) {
      const newline = this.buffer.indexOf("\n");
      if (newline < 0) {
        return;
      }
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (!line) {
        continue;
      }
      const frame = JSON.parse(line) as Frame;
      const waiterIndex = this.waiters.findIndex((waiter) => waiter.predicate(frame));
      if (waiterIndex >= 0) {
        this.waiters.splice(waiterIndex, 1)[0]?.resolve(frame);
      } else {
        this.frames.push(frame);
      }
    }
  }
}

test("a whole-projection ceiling keeps many legal per-turn images from wedging session.open", async () => {
  // Mirrors the round-2 audit reproduction: every turn carries an image that
  // is individually inside the per-message budget, but ~N such turns sum far
  // past the socket output cap. The whole-projection ceiling must omit the
  // oldest inline images first so initialize/open/status stay deliverable.
  const directory = await mkdtemp(join(tmpdir(), "xerxes-transcript-total-cap-"));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "vision-model",
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
      params: { session_key: "image-flood", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);
    await client.next(eventFrame("init_done"));
    await client.next(eventFrame("status_update"));

    // Each data URL: decoded 100KB → ~133,336 base64 chars + prefix — well
    // under MAX_TRANSCRIPT_INLINE_IMAGE_BYTES (256 KB) per message.
    const heavyBytes = new Uint8Array(100_000);
    heavyBytes.set(PNG_BYTES);
    const heavyUrl = `data:image/png;base64,${Buffer.from(heavyBytes).toString("base64")}`;
    const urlBytes = Buffer.byteLength(heavyUrl, "utf8");
    expect(urlBytes).toBeLessThanOrEqual(MAX_TRANSCRIPT_INLINE_IMAGE_BYTES);

    const turnCount = 20;
    const session = runtime.sessionStatus("image-flood");
    if (!session) throw new Error("expected live session");
    for (let index = 0; index < turnCount; index += 1) {
      session.messages.push({
        role: "user",
        content: [
          { type: "text", text: `turn ${index} screenshot` },
          { type: "image_url", image_url: { url: heavyUrl } },
        ],
        text: `turn ${index} screenshot`,
      });
    }

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "session.open",
      params: { session_key: "image-flood" },
    });
    const opened = await client.next((frame) => frame.id === 2);
    expect(opened.result?.ok).toBe(true);

    // The ceiling spends itself newest first: the oldest inline images drop
    // off first, recent turns keep real pixels.
    const expectedKept = Math.floor(MAX_TRANSCRIPT_TOTAL_INLINE_IMAGE_BYTES / urlBytes);
    expect(expectedKept).toBeGreaterThan(0);
    expect(expectedKept).toBeLessThan(turnCount);
    expect(echoedCount(opened)).toBe(turnCount);
    const echoed = opened.result?.session as Record<string, unknown>;
    expect(echoed.transcript_images_omitted).toBe(turnCount - expectedKept);

    const transcript = echoed.transcript as Array<Record<string, unknown>>;
    const oldestParts = transcript[0]?.content as Array<Record<string, unknown>>;
    expect(oldestParts[1]?.type).toBe("text");
    expect(String(oldestParts[1]?.text)).toMatch(/^\[image omitted: \d+ KB\]$/);
    const newestParts = transcript[turnCount - 1]?.content as Array<Record<string, unknown>>;
    expect(newestParts[1]).toEqual({ type: "image_url", image_url: { url: heavyUrl } });

    // The echoed payload stays far below any frame cap by construction.
    expect(JSON.stringify(opened.result).length).toBeLessThan(
      MAX_TRANSCRIPT_TOTAL_INLINE_IMAGE_BYTES * 2,
    );

    // The live session keeps every full image.
    for (const message of session.messages) {
      const parts = message.content as Array<{ image_url?: { url?: string } }>;
      expect(parts[1]?.image_url?.url).toBe(heavyUrl);
    }
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { recursive: true, force: true });
  }
});

function echoedCount(frame: { result?: Record<string, unknown> }): number {
  const session = frame.result?.session as Record<string, unknown> | undefined;
  return Number(session?.message_count ?? -1);
}
