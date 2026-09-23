// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * `/paste`, `/queue`, `/skin`, `/statusbar` and `/voice` are client-side
 * controls. The daemon re-emits them as a `ui_command` event and, when a host
 * installs one, calls a `DaemonUiControlPort`.
 *
 * No client in this repository listens for `ui_command`, and nothing injects
 * the port — yet the daemon answered "Sent native UI command `/skin` to the
 * connected client", which reads as an accomplished action. These tests pin the
 * honest reporting: the daemon says what it did, and `handled` distinguishes a
 * real host port from the bare event.
 */

import { expect, test } from "bun:test";
import { connect, type Socket } from "node:net";
import { mkdtemp, rm } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { DaemonServer, type DaemonUiControlInput } from "../src/daemon/server.js";
import { InMemoryDaemonRuntime } from "../src/daemon/runtime.js";

interface Frame {
  id?: number;
  method?: string;
  params?: { payload?: Record<string, unknown>; type?: string };
  result?: Record<string, unknown>;
}

class Client {
  private buffer = "";
  private readonly frames: Frame[] = [];
  private readonly waiters: { predicate: (frame: Frame) => boolean; resolve: (frame: Frame) => void }[] = [];

  private constructor(private readonly socket: Socket) {
    socket.setEncoding("utf8");
    socket.on("data", chunk => this.receive(typeof chunk === "string" ? chunk : new TextDecoder().decode(chunk)));
  }

  static async connect(socketPath: string): Promise<Client> {
    const socket = connect({ path: socketPath });
    await new Promise<void>((resolve, reject) => {
      socket.once("connect", resolve);
      socket.once("error", reject);
    });
    return new Client(socket);
  }

  close(): void {
    this.socket.destroy();
  }

  send(frame: Record<string, unknown>): void {
    this.socket.write(`${JSON.stringify(frame)}\n`);
  }

  next(predicate: (frame: Frame) => boolean): Promise<Frame> {
    const index = this.frames.findIndex(predicate);
    if (index >= 0) return Promise.resolve(this.frames.splice(index, 1)[0]!);
    return new Promise(resolve => this.waiters.push({ predicate, resolve }));
  }

  private receive(chunk: string): void {
    this.buffer += chunk;
    let newline = this.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline);
      this.buffer = this.buffer.slice(newline + 1);
      if (line.trim()) {
        const frame = JSON.parse(line) as Frame;
        const index = this.waiters.findIndex(waiter => waiter.predicate(frame));
        if (index >= 0) this.waiters.splice(index, 1)[0]!.resolve(frame);
        else this.frames.push(frame);
      }
      newline = this.buffer.indexOf("\n");
    }
  }
}

const event = (type: string) => (frame: Frame) => frame.method === "event" && frame.params?.type === type;

async function withServer(
  uiControl: ConstructorParameters<typeof DaemonServer>[0]["uiControl"],
  body: (client: Client) => Promise<void>,
): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-ui-control-"));
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    socketPath,
    projectDirectory: directory,
    ...(uiControl ? { uiControl } : {}),
    runtime: new InMemoryDaemonRuntime(undefined, {
      currentProjectDirectory: directory,
      model: "protocol-model",
      sessionDirectory: join(directory, "sessions"),
    }),
  });
  await server.start();
  const client = await Client.connect(socketPath);
  try {
    client.send({ jsonrpc: "2.0", id: 1, method: "initialize", params: { session_key: "ui-control", project_dir: directory } });
    await client.next(frame => frame.id === 1);
    await body(client);
  } finally {
    client.close();
    await server.stop();
    await rm(directory, { force: true, recursive: true });
  }
}

test("a UI control with no host port reports what the daemon did, not a delivered action", async () => {
  await withServer(undefined, async client => {
    client.send({ jsonrpc: "2.0", id: 2, method: "slash", params: { command: "/skin nocturne" } });

    const control = await client.next(event("ui_command"));
    expect(control.params?.payload).toMatchObject({ action: "skin", argument: "nocturne" });

    const response = await client.next(frame => frame.id === 2);
    expect(response.result).toMatchObject({ ok: true, action: "skin", handled: false });

    const notice = await client.next(event("notification"));
    const body = String(notice.params?.payload?.body ?? "");
    // The old wording asserted delivery to a client that had not handled it.
    expect(body).not.toContain("Sent native UI command");
    expect(body).toContain("client-side control");
    expect(body).toContain("ui_command");
  });
});

test("an installed host port owns both the outcome and the message", async () => {
  const seen: DaemonUiControlInput[] = [];
  await withServer(
    {
      execute(input) {
        seen.push(input);
        return { message: "Skin switched to nocturne.", payload: { skin: "nocturne" } };
      },
    },
    async client => {
      client.send({ jsonrpc: "2.0", id: 2, method: "slash", params: { command: "/skin nocturne" } });

      await client.next(event("ui_command"));
      const response = await client.next(frame => frame.id === 2);
      expect(response.result).toMatchObject({ ok: true, action: "skin", handled: true });
      expect(response.result?.result).toEqual({ skin: "nocturne" });

      const notice = await client.next(event("notification"));
      expect(String(notice.params?.payload?.body ?? "")).toBe("Skin switched to nocturne.");
    },
  );
  expect(seen).toHaveLength(1);
  expect(seen[0]).toMatchObject({ action: "skin", argument: "nocturne" });
});

test("an unsupported control is refused rather than forwarded", async () => {
  await withServer(undefined, async client => {
    client.send({ jsonrpc: "2.0", id: 2, method: "slash", params: { command: "/reboot-the-terminal" } });
    const response = await client.next(frame => frame.id === 2);
    expect(response.result?.ok).toBe(false);
  });
});
