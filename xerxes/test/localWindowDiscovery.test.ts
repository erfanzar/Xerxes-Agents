// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, expect, spyOn, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { connect, type Socket } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { ProfileStore } from "../src/bridge/profiles.js";
import { InMemoryDaemonRuntime } from "../src/daemon/runtime.js";
import { DaemonServer } from "../src/daemon/server.js";
import { saveLocalProfile } from "../src/local/cliEntry.js";
import {
  probeLoadedWindows,
  systemWindowFetch,
  type LoadedWindows,
  type ProbeLoadedWindowsInput,
} from "../src/local/loadedWindow.js";
import { seedModelsDev } from "./fixtures/modelsDev.js";

seedModelsDev();

const KEY = "ternary-bonsai-2-27b-mlx";

interface Frame {
  readonly id?: number;
  readonly method?: string;
  readonly params?: { readonly payload?: Record<string, unknown>; readonly type?: string };
  readonly result?: Record<string, unknown>;
}

class SocketTestClient {
  private buffer = "";
  private readonly frames: Frame[] = [];
  private readonly waiters: Array<{ predicate: (frame: Frame) => boolean; resolve: (frame: Frame) => void }> = [];

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
    if (index >= 0) return Promise.resolve(this.frames.splice(index, 1)[0] as Frame);
    return new Promise((resolve) => this.waiters.push({ predicate, resolve }));
  }

  async request(id: number, method: string, params: Record<string, unknown>): Promise<Record<string, unknown>> {
    this.socket.write(`${JSON.stringify({ jsonrpc: "2.0", id, method, params })}\n`);
    return (await this.next((frame) => frame.id === id)).result ?? {};
  }

  private receive(chunk: string): void {
    this.buffer += chunk;
    let newline = this.buffer.indexOf("\n");
    while (newline >= 0) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (line) {
        const frame = JSON.parse(line) as Frame;
        const waiter = this.waiters.findIndex((candidate) => candidate.predicate(frame));
        if (waiter >= 0) this.waiters.splice(waiter, 1)[0]?.resolve(frame);
        else this.frames.push(frame);
      }
      newline = this.buffer.indexOf("\n");
    }
  }
}

const statusWithContext = (window: number) => (frame: Frame): boolean =>
  frame.method === "event" && frame.params?.type === "status_update" && frame.params.payload?.max_context === window;

const cleanups: Array<() => Promise<void> | void> = [];
afterEach(async () => {
  for (const cleanup of cleanups.splice(0).reverse()) await cleanup();
});

async function daemonFor(baseUrl: string, probe?: (input: ProbeLoadedWindowsInput) => Promise<LoadedWindows>) {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-local-window-"));
  cleanups.push(() => rm(directory, { recursive: true, force: true }));
  const profileStore = new ProfileStore(join(directory, "profiles.json"));
  // What `xerxes local setup --engine lmstudio` saved earlier.
  saveLocalProfile(profileStore, {
    name: "bonsai-local",
    baseUrl,
    model: KEY,
    provider: "custom",
    dataTrust: "local",
    promptProfile: "lean",
    contextLimit: 16384,
  });
  profileStore.setActive("bonsai-local");
  const runtime = new InMemoryDaemonRuntime(undefined, {
    currentProjectDirectory: directory,
    runtimeSettings: { base_url: baseUrl, model: KEY, provider: "custom" },
    sessionDirectory: join(directory, "sessions"),
  });
  const socketPath = join(directory, "daemon.sock");
  const server = new DaemonServer({
    autoDiscoverModelCapabilities: true,
    profileStore,
    runtime,
    socketPath,
    ...(probe ? { localWindowProbe: probe } : {}),
  });
  await server.start();
  cleanups.push(async () => {
    await server.stop();
    await runtime.shutdown();
  });
  const client = await SocketTestClient.connect(socketPath);
  cleanups.push(() => client.close());
  return { client, profileStore };
}

/** A loopback LM Studio: `/v1/models` without windows, `/api/v0/models` with the loaded one. */
function fakeLmStudio(loadedWindow: () => number | undefined) {
  const http = Bun.serve({
    hostname: "127.0.0.1",
    port: 0,
    fetch(request) {
      const path = new URL(request.url).pathname;
      if (path === "/v1/models") return Response.json({ object: "list", data: [{ id: KEY }, { id: "qwen/qwen3-8b" }] });
      if (path === "/api/v0/models") {
        const window = loadedWindow();
        return Response.json({
          object: "list",
          data: [
            window === undefined
              ? { id: KEY, state: "not-loaded", max_context_length: 262144 }
              : { id: KEY, state: "loaded", max_context_length: 262144, loaded_context_length: window },
            { id: "qwen/qwen3-8b", state: "not-loaded", max_context_length: 40960 },
          ],
        });
      }
      return new Response("Unexpected endpoint", { status: 404 });
    },
  });
  cleanups.push(() => http.stop(true));
  return `${http.url.href}v1`;
}

test("the status bar shows the window LM Studio loaded, and a reload is picked up at the next discovery", async () => {
  let window: number | undefined = 42752;
  const baseUrl = fakeLmStudio(() => window);
  const { client, profileStore } = await daemonFor(baseUrl, input => probeLoadedWindows(input, systemWindowFetch));

  await client.request(1, "initialize", { session_key: "lmstudio" });
  // The first frame carries the window saved by setup; discovery then reports what is loaded.
  expect((await client.next(statusWithContext(42752))).params?.payload).toMatchObject({ max_context: 42752 });
  expect(profileStore.get("bonsai-local")?.model_capabilities?.[KEY]).toEqual({ context_limit: 42752 });

  // The user reloads the model in LM Studio with a smaller window.
  window = 8192;
  expect(await client.request(2, "fetch_models", { profile_name: "bonsai-local" })).toMatchObject({ ok: true, source: "remote" });
  expect((await client.request(3, "session.status", { session_key: "lmstudio" })).session).toMatchObject({
    context_limit: 8192,
    max_context: 8192,
  });
});

test("a failed probe keeps today's discovery result and logs the reason once", async () => {
  const baseUrl = fakeLmStudio(() => undefined);
  const warn = spyOn(console, "warn").mockImplementation(() => undefined);
  cleanups.push(() => warn.mockRestore());
  let probes = 0;
  const { client, profileStore } = await daemonFor(baseUrl, async () => {
    probes += 1;
    return { ok: false, reason: "no answer from the engine" };
  });

  await client.request(1, "initialize", { session_key: "lmstudio" });
  await client.request(2, "fetch_models", { profile_name: "bonsai-local" });
  await client.request(3, "fetch_models", { profile_name: "bonsai-local" });
  expect(probes).toBeGreaterThanOrEqual(2);
  // Unchanged behaviour: the /v1/models result (no window) replaces the cached metadata.
  expect(profileStore.get("bonsai-local")?.model_capabilities?.[KEY]).toEqual({});
  const logged = warn.mock.calls.map((call) => String(call[0])).filter((line) => line.includes("Local context window"));
  expect(logged).toEqual(["Local context window for profile bonsai-local is unknown: no answer from the engine"]);
});

test("a model LM Studio has not loaded stays unknown, with the reason logged", async () => {
  const baseUrl = fakeLmStudio(() => undefined);
  const warn = spyOn(console, "warn").mockImplementation(() => undefined);
  cleanups.push(() => warn.mockRestore());
  const { client, profileStore } = await daemonFor(baseUrl, input => probeLoadedWindows(input, systemWindowFetch));
  await client.request(1, "fetch_models", { profile_name: "bonsai-local" });
  expect(profileStore.get("bonsai-local")?.model_capabilities?.[KEY]).toEqual({});
  expect(warn.mock.calls.map((call) => String(call[0]))).toContain(
    `Local context window for profile bonsai-local is unknown: LM Studio has not loaded ${KEY}`,
  );
});

test("a cloud profile is never probed", async () => {
  const nativeFetch = globalThis.fetch;
  globalThis.fetch = (async () => Response.json({ data: [{ id: KEY }] })) as unknown as typeof globalThis.fetch;
  cleanups.push(() => {
    globalThis.fetch = nativeFetch;
  });
  let probes = 0;
  const { client, profileStore } = await daemonFor("https://models.example.test/v1", async () => {
    probes += 1;
    return { ok: true, engine: "llama-server", window: 1 };
  });
  expect(await client.request(1, "fetch_models", { profile_name: "bonsai-local" })).toMatchObject({ ok: true });
  expect(probes).toBe(0);
  expect(profileStore.get("bonsai-local")?.model_capabilities?.[KEY]).toEqual({});
});
