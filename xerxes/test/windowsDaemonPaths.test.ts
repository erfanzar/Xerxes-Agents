// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { daemonPaths, daemonTransport } from "../src/daemon/paths.js";

test("daemonTransport defaults to unix on POSIX and websocket on win32", () => {
  expect(daemonTransport({}, "linux")).toBe("unix");
  expect(daemonTransport({}, "darwin")).toBe("unix");
  expect(daemonTransport({}, "win32")).toBe("websocket");
});

test("XERXES_DAEMON_TRANSPORT overrides the platform default", () => {
  expect(daemonTransport({ XERXES_DAEMON_TRANSPORT: "unix" }, "win32")).toBe("unix");
  expect(daemonTransport({ XERXES_DAEMON_TRANSPORT: " Websocket " }, "linux")).toBe("websocket");
  // Garbage values fall back to the platform default rather than crashing.
  expect(daemonTransport({ XERXES_DAEMON_TRANSPORT: "pipes" }, "win32")).toBe("websocket");
});

test("daemon paths publish an endpoint file beside the socket and pid paths", async () => {
  const home = await mkdtemp(join(tmpdir(), "xerxes-paths-home-"));
  try {
    const project = await mkdtemp(join(tmpdir(), "xerxes-paths-project-"));
    const paths = daemonPaths(project, { XERXES_HOME: home });
    expect(paths.endpointPath.endsWith(".endpoint.json")).toBe(true);
    // All three coordinates share the per-project digest stem, so the TUI and
    // the daemon always agree on the endpoint-file location.
    const socketStem = paths.socketPath.replace(/\.sock$/, "");
    const pidStem = paths.pidPath.replace(/\.pid$/, "");
    const endpointStem = paths.endpointPath.replace(/\.endpoint\.json$/, "");
    expect(pidStem).toBe(socketStem);
    expect(endpointStem).toBe(socketStem);
    await rm(project, { force: true, recursive: true });
  } finally {
    await rm(home, { force: true, recursive: true });
  }
});

test("daemon paths honor XERXES_HOME including tilde expansion", () => {
  const custom = daemonPaths(".", { XERXES_HOME: "custom-home" });
  expect(custom.socketPath.replaceAll("\\", "/")).toContain("custom-home/");
  const tilde = daemonPaths(".", { XERXES_HOME: "~/xa" });
  expect(tilde.socketPath.replaceAll("\\", "/")).toContain("/xa/");
  const tildeWindows = daemonPaths(".", { XERXES_HOME: "~\\xa" });
  expect(tildeWindows.socketPath.replaceAll("\\", "/")).toContain("/xa/");
});
