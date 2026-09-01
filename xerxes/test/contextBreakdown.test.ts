// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ProfileStore } from "../src/bridge/profiles.js";
import { InMemoryDaemonRuntime } from "../src/daemon/runtime.js";
import { DaemonServer } from "../src/daemon/server.js";

interface Frame {
  readonly id?: number;
  readonly result?: Record<string, unknown>;
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
    socket.on("data", (chunk) => this.receive(String(chunk)));
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
      return Promise.resolve(this.frames.splice(index, 1)[0]!);
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
      if (!line.trim()) continue;
      try {
        const frame = JSON.parse(line) as Frame;
        const at = this.waiters.findIndex((waiter) => waiter.predicate(frame));
        if (at >= 0) this.waiters.splice(at, 1)[0]?.resolve(frame);
        else this.frames.push(frame);
      } catch {
        continue;
      }
    }
  }
}

const cleanups: Array<() => void | Promise<void>> = [];
afterEach(async () => {
  while (cleanups.length) await cleanups.pop()?.();
});

test("context_breakdown reports the estimated token split for the session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-ctx-breakdown-"));
  cleanups.push(() => rm(directory, { recursive: true, force: true }));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "gpt-4",
    runtimeSettings: {
      base_url: "https://api.openai.test",
      provider: "openai",
    },
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    autoTitle: false,
    profileStore: new ProfileStore(join(directory, "profiles.json")),
    runtime,
    socketPath,
  });
  await server.start();
  cleanups.push(() => server.stop());
  const client = await SocketTestClient.connect(socketPath);
  cleanups.push(() => client.close());
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "initialize",
      params: { session_key: "ctx", project_dir: directory },
    });
    await client.next((frame) => frame.id === 1);

    const session = runtime.sessionStatus("ctx");
    if (!session) throw new Error("session must exist");
    session.messages.push(
      { role: "user", content: "seed the transcript with some words to count" },
      { role: "assistant", content: "an answer with a few tokens in it" },
    );

    client.send({
      jsonrpc: "2.0",
      id: 2,
      method: "context_breakdown",
      params: { session_key: "ctx" },
    });
    const frame = await client.next((candidate) => candidate.id === 2);
    const result = frame.result!;
    expect(result.ok).toBe(true);
    const system = Number(result.system_prompt_tokens);
    const tools = Number(result.tools_tokens);
    const messages = Number(result.messages_tokens);
    const total = Number(result.total_tokens);
    expect(messages).toBeGreaterThan(0);
    expect(total).toBeGreaterThanOrEqual(messages);
    // The split is an estimate of the same counter that totals the window;
    // overhead buckets never turn negative and the total covers the parts.
    expect(system).toBeGreaterThanOrEqual(0);
    expect(tools).toBeGreaterThanOrEqual(0);
    expect(total).toBeGreaterThanOrEqual(system + tools);
    // JSON-RPC v35: zero means capacity metadata is unavailable for the
    // (fake) provider here — unknown, never fabricated.
    expect(Number(result.context_limit)).toBeGreaterThanOrEqual(0);
    expect(result.model).toBe("gpt-4");
  } finally {
    client.close();
    await server.stop();
  }
});

test("context_breakdown fails cleanly without a session", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-ctx-breakdown-"));
  cleanups.push(() => rm(directory, { recursive: true, force: true }));
  const socketPath = join(directory, "daemon.sock");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    model: "gpt-4",
    runtimeSettings: {
      base_url: "https://api.openai.test",
      provider: "openai",
    },
    sessionDirectory: join(directory, "sessions"),
  });
  const server = new DaemonServer({
    autoTitle: false,
    profileStore: new ProfileStore(join(directory, "profiles.json")),
    runtime,
    socketPath,
  });
  await server.start();
  cleanups.push(() => server.stop());
  const client = await SocketTestClient.connect(socketPath);
  cleanups.push(() => client.close());
  try {
    client.send({
      jsonrpc: "2.0",
      id: 1,
      method: "context_breakdown",
      params: { session_key: "missing" },
    });
    const frame = await client.next((candidate) => candidate.id === 1);
    expect(frame.result!.ok).toBe(false);
    expect(String(frame.result!.error)).toContain("no active session");
  } finally {
    client.close();
    await server.stop();
  }
});
