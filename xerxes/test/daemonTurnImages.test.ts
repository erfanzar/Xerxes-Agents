// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

// Raw Unix-socket client tests: the v35 filesystem socket transport does not
// bind on native Windows, where the daemon runs the WebSocket transport.
const testUnixSocket = test.skipIf(process.platform === "win32");
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
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

testUnixSocket("daemon turn.submit rejects invalid images at the RPC boundary", async () => {
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

testUnixSocket("turn.submit images reach the provider as image_url parts and round-trip the transcript", async () => {
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
