// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { DaemonServer } from "../src/daemon/server.js";

/** Just enough of a net.Socket for sendSocketFrame's write/end/destroy paths. */
class FakeSocket {
  destroyed = false;
  readonly written: string[] = [];
  writable = true;

  write(data: string): boolean {
    this.written.push(data);
    return this.writable;
  }

  end(data: string, callback?: () => void): void {
    // Mirrors the real semantics the guard relies on: the payload flushes
    // before FIN, then the callback closes the socket.
    this.written.push(data);
    callback?.();
    this.destroyed = true;
  }

  destroy(): void {
    this.destroyed = true;
  }
}

interface FakeConnection {
  activeSessionKey: string;
  buffer: string;
  pendingRequestBytes: number;
  pendingRequestCount: number;
  queuedOutputBytes: number;
  outputQueue: string[];
  outputBlocked: boolean;
  queue: Promise<void>;
  readonly socket: FakeSocket;
}

function fakeConnection(socket: FakeSocket): FakeConnection {
  return {
    activeSessionKey: "tui:test",
    buffer: "",
    pendingRequestBytes: 0,
    pendingRequestCount: 0,
    queuedOutputBytes: 0,
    outputQueue: [],
    outputBlocked: false,
    queue: Promise.resolve(),
    socket,
  };
}

function frameSender(
  server: DaemonServer,
): (connection: FakeConnection, frame: object) => void {
  const raw = (server as unknown as Record<string, unknown>).sendSocketFrame as
    | ((this: DaemonServer, connection: FakeConnection, frame: object) => void)
    | undefined;
  if (!raw) throw new Error("sendSocketFrame not found on DaemonServer");
  return raw.bind(server);
}

async function startedServer(maxSocketOutputBytes: number): Promise<{
  server: DaemonServer;
  dispose: () => Promise<void>;
}> {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-socket-guard-"));
  const server = new DaemonServer({
    socketPath: join(directory, "daemon.sock"),
    maxSocketOutputBytes,
  });
  return {
    server,
    dispose: async () => {
      await server.stop();
    },
  };
}

test("an oversized response frame delivers a correlated JSON-RPC error instead of a silent destroy", async () => {
  const { server, dispose } = await startedServer(256);
  try {
    const socket = new FakeSocket();
    const connection = fakeConnection(socket);
    frameSender(server)(connection, {
      jsonrpc: "2.0",
      id: 7,
      ok: true,
      result: { transcript: "x".repeat(1_024) },
    });
    expect(socket.written).toHaveLength(1);
    const failure = JSON.parse(socket.written[0]?.trim() ?? "{}") as {
      error?: { code?: number; message?: string };
      id?: unknown;
      jsonrpc?: string;
    };
    expect(failure.jsonrpc).toBe("2.0");
    expect(failure.id).toBe(7);
    expect(failure.error?.message).toBe("response exceeds socket output limit");
    expect(typeof failure.error?.code).toBe("number");
    expect(socket.destroyed).toBe(true);
  } finally {
    await dispose();
  }
});

test("an oversized event frame destroys the connection without inventing an error id", async () => {
  const { server, dispose } = await startedServer(256);
  try {
    const socket = new FakeSocket();
    const connection = fakeConnection(socket);
    frameSender(server)(connection, {
      jsonrpc: "2.0",
      method: "event",
      params: { type: "turn_end", payload: { text: "y".repeat(1_024) } },
    });
    expect(socket.written).toHaveLength(0);
    expect(socket.destroyed).toBe(true);
  } finally {
    await dispose();
  }
});

test("frames under the cap still flow normally through write and queue", async () => {
  const { server, dispose } = await startedServer(512);
  try {
    const socket = new FakeSocket();
    const connection = fakeConnection(socket);
    const send = frameSender(server);

    send(connection, { jsonrpc: "2.0", id: 1, ok: true });
    expect(socket.written).toHaveLength(1);
    expect(connection.outputQueue).toHaveLength(0);

    // A backpressured socket queues further small frames.
    socket.writable = false;
    socket.written.length = 0;
    send(connection, { jsonrpc: "2.0", id: 2, ok: true });
    expect(socket.written).toHaveLength(1); // attempted write still lands
    send(connection, { jsonrpc: "2.0", id: 3, ok: true });
    expect(connection.outputQueue).toHaveLength(1);
    expect(connection.queuedOutputBytes).toBeGreaterThan(0);

    // Queued bytes beyond the cap trigger the same belt-and-braces error.
    send(connection, {
      jsonrpc: "2.0",
      id: 4,
      ok: true,
      result: { blob: "z".repeat(600) },
    });
    expect(socket.destroyed).toBe(true);
    expect(socket.written.at(-1)).toContain("response exceeds socket output limit");
  } finally {
    await dispose();
  }
});
