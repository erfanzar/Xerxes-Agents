// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, readFile, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import { ToolRegistry } from "../src/executors/toolRegistry.js";
import {
  OperatorState,
  PtySessionManager,
  SpawnedAgentManager,
  createOperatorRuntimeConfig,
  type OperatorPatchRequest,
  type OperatorWebSearchRequest,
} from "../src/operators/index.js";
import { composeRuntimeFeatures } from "../src/runtime/features.js";
import type { JsonObject, ToolCall } from "../src/types/toolCalls.js";

function call(name: string, arguments_: JsonObject): ToolCall {
  return {
    id: crypto.randomUUID(),
    type: "function",
    function: { name, arguments: arguments_ },
  };
}

async function executeJson(
  registry: ToolRegistry,
  name: string,
  arguments_: JsonObject,
): Promise<JsonObject> {
  return JSON.parse(
    await registry.execute(call(name, arguments_), { metadata: {} }),
  ) as JsonObject;
}

const VALID_PATCH = [
  "--- a/demo.txt",
  "+++ b/demo.txt",
  "@@ -1 +1 @@",
  "-old",
  "+new",
  "",
].join("\n");

test("PTY manager preserves interactive stdin, shell expressions, and asynchronous polling", async () => {
  const shell = Bun.which("sh") ?? "/bin/sh";
  const manager = new PtySessionManager();
  const interactive = await manager.createSession(
    "printf 'ready\\n'; IFS= read line; printf 'echo:%s\\n' \"$line\"",
    { shell, login: false, yieldTimeMs: 500, maxOutputChars: 4_000 },
  );
  try {
    expect(interactive.stdout).toContain("ready");
    const echoed = await manager.write(interactive.sessionId, {
      chars: "hello\n",
      closeStdin: true,
      yieldTimeMs: 500,
      maxOutputChars: 4_000,
    });
    expect(echoed.stdout).toContain("echo:hello");
  } finally {
    if (
      manager
        .listSessions()
        .some((session) => session.sessionId === interactive.sessionId)
    ) {
      await manager.close(interactive.sessionId);
    }
  }

  const directory = await mkdtemp(join(tmpdir(), "xerxes-operator-pty-"));
  try {
    const physicalDirectory = await realpath(directory);
    const shellCommand =
      "mkdir -p tmp-files && { printf 'ready\\n'; printf 'Project: '; pwd; } > tmp-files/AGENT_NOTES.md && cat tmp-files/AGENT_NOTES.md";
    const created = await manager.createSession(shellCommand, {
      shell,
      login: false,
      workdir: directory,
      yieldTimeMs: 1_000,
      maxOutputChars: 4_000,
    });
    expect(created.exitCode).toBe(0);
    expect(created.stdout).toContain("ready");
    expect(created.stdout).toContain("Project: " + physicalDirectory);
    expect(
      await readFile(join(directory, "tmp-files", "AGENT_NOTES.md"), "utf8"),
    ).toContain("Project: " + physicalDirectory);
    expect(await Bun.file(join(directory, "{")).exists()).toBeFalse();
  } finally {
    await manager.closeAll();
    await rm(directory, { recursive: true, force: true });
  }

  const delayed = await manager.createSession(
    "printf 'start\\n'; sleep 0.4; printf 'done\\n'",
    {
      shell,
      login: false,
      yieldTimeMs: 10,
      maxOutputChars: 4_000,
    },
  );
  try {
    expect(delayed.running).toBeTrue();
    expect(delayed.note).toContain("poll with write_stdin");
    let polled = delayed;
    let stdout = delayed.stdout;
    for (
      let attempt = 0;
      attempt < 3 && polled.running && !stdout.includes("done");
      attempt += 1
    ) {
      polled = await manager.write(delayed.sessionId, {
        yieldTimeMs: 1_000,
        maxOutputChars: 4_000,
      });
      stdout += polled.stdout;
    }
    expect(stdout).toContain("done");
  } finally {
    if (
      manager
        .listSessions()
        .some((session) => session.sessionId === delayed.sessionId)
    ) {
      await manager.close(delayed.sessionId);
    }
  }
});

test("operator registrations dispatch host-injected patch, inspection, parallel, and web capabilities", async () => {
  const patches: OperatorPatchRequest[] = [];
  const readonlyCalls: Array<{ name: string; input: JsonObject }> = [];
  const searches: OperatorWebSearchRequest[] = [];
  const state = new OperatorState({
    config: createOperatorRuntimeConfig({
      enabled: true,
      powerToolsEnabled: true,
    }),
    clock: { now: () => new Date("2026-07-13T10:20:30.000Z") },
    patchApplier: {
      async applyPatch(request) {
        patches.push(request);
        return {
          applied: !request.check,
          checked: request.check,
          stdout: "ok",
          ...(request.workdir === undefined
            ? {}
            : { workdir: request.workdir }),
        };
      },
    },
    parallelReadonlyToolPort: {
      toolNames: new Set(["ReadFile"]),
      async execute(request) {
        readonlyCalls.push(request);
        return request.input.file_path === "notes.txt"
          ? "payload\n"
          : "unexpected";
      },
    },
    imageInspector: {
      async inspectImage(request) {
        expect(request).toEqual({
          path: "/workspace/diagram.png",
          detail: "high",
        });
        return {
          path: request.path,
          detail: request.detail,
          format: "PNG",
          width: 16,
          height: 12,
          mode: "RGBA",
          imageDataUrl: "data:image/png;base64,private-image-payload",
        };
      },
    },
    webPort: {
      async search(request) {
        searches.push(request);
        return {
          engine: "native-host",
          results: [
            { title: request.query, url: "https://example.test/result" },
          ],
        };
      },
      async weather(request) {
        return { location: request.location, current: { temperature: 19 } };
      },
      async finance(request) {
        return {
          ticker: request.ticker,
          market: request.market ?? null,
          kind: request.kind,
          price: 42,
        };
      },
      async sports(request) {
        return {
          league: request.league,
          fn: request.fn,
          team: request.team ?? null,
          data: [],
        };
      },
    },
  });
  const registry = new ToolRegistry();
  state.registerTools(registry);
  try {
    const names = new Set(
      registry.definitions().map((definition) => definition.function.name),
    );
    for (const name of [
      "apply_patch",
      "parallel_tools",
      "view_image",
      "web.time",
      "web.search_query",
      "web.image_query",
      "web.weather",
      "web.finance",
      "web.sports",
    ]) {
      expect(names).toContain(name);
    }
    const execDefinition = registry
      .definitions()
      .find((definition) => definition.function.name === "exec_command");
    expect(execDefinition?.function.description).toContain(
      "interactive terminal session",
    );
    expect(execDefinition?.function.parameters).toMatchObject({
      properties: {
        cmd: { type: "string" },
        yield_time_ms: { type: "integer" },
        login: { type: "boolean" },
      },
    });

    expect(
      await executeJson(registry, "apply_patch", {
        patch: VALID_PATCH,
        workdir: "/workspace",
        check: false,
      }),
    ).toEqual({
      applied: true,
      checked: false,
      workdir: "/workspace",
      stdout: "ok",
    });
    expect(patches).toEqual([
      { patch: VALID_PATCH, check: false, workdir: "/workspace" },
    ]);
    await expect(
      registry.execute(call("apply_patch", { patch: "not a patch" }), {
        metadata: {},
      }),
    ).rejects.toThrow("unified diff");
    expect(patches).toHaveLength(1);

    expect(
      await executeJson(registry, "parallel_tools", {
        calls: [
          { name: "ReadFile", input: { file_path: "notes.txt" } },
          {
            name: "WriteFile",
            input: { file_path: "notes.txt", content: "blocked" },
          },
        ],
        max_workers: 0,
      }),
    ).toEqual({
      max_workers: 1,
      results: [
        { index: 0, name: "ReadFile", ok: true, result: "payload\n" },
        {
          index: 1,
          name: "WriteFile",
          ok: false,
          error:
            'parallel_tools only allows read-only safe tools; rejected "WriteFile"',
        },
      ],
    });
    expect(readonlyCalls).toEqual([
      { name: "ReadFile", input: { file_path: "notes.txt" } },
    ]);

    const image = await executeJson(registry, "view_image", {
      path: "/workspace/diagram.png",
      detail: "high",
    });
    expect(image).toEqual({
      path: "/workspace/diagram.png",
      detail: "high",
      format: "PNG",
      width: 16,
      height: 12,
      mode: "RGBA",
    });
    expect(JSON.stringify(image)).not.toContain("private-image-payload");
    expect(
      state.createReinvokeMessage({
        path: "/workspace/diagram.png",
        detail: "original",
        format: "PNG",
        width: 16,
        height: 12,
        mode: "RGBA",
        imageDataUrl: "data:image/png;base64,private-image-payload",
      }),
    ).toEqual({
      role: "user",
      content: [
        {
          type: "text",
          text: "[TOOL IMAGE RESULT] PNG 16x12 RGBA image at /workspace/diagram.png",
        },
        {
          type: "image_url",
          image_url: {
            url: "data:image/png;base64,private-image-payload",
            detail: "high",
          },
        },
      ],
    });
    expect(
      state.createReinvokeMessage({
        path: "/workspace/diagram.png",
        detail: "auto",
        width: 1,
        height: 1,
        mode: "RGB",
      }),
    ).toBeUndefined();

    expect(
      await executeJson(registry, "web.search_query", {
        q: "latest native port news",
        search_type: "news",
        n_results: 3,
        domains: ["example.test"],
      }),
    ).toEqual({
      query: "latest native port news",
      search_type: "news",
      engine: "native-host",
      results: [
        {
          title: "latest native port news",
          url: "https://example.test/result",
        },
      ],
    });
    expect(
      await executeJson(registry, "web.image_query", {
        q: "agent diagram",
        n_results: 2,
      }),
    ).toMatchObject({
      query: "agent diagram",
      engine: "native-host",
    });
    expect(searches).toEqual([
      {
        kind: "text",
        query: "latest native port news",
        maxResults: 3,
        domains: ["example.test"],
        recency: "day",
      },
      {
        kind: "image",
        query: "agent diagram",
        maxResults: 2,
        domains: [],
        recency: undefined,
      },
    ]);
    expect(
      await executeJson(registry, "web.weather", { location: "Istanbul" }),
    ).toEqual({
      location: "Istanbul",
      current: { temperature: 19 },
    });
    expect(
      await executeJson(registry, "web.finance", {
        ticker: "AMD",
        market: "USA",
        kind: "equity",
      }),
    ).toEqual({
      ticker: "AMD",
      market: "USA",
      kind: "equity",
      price: 42,
    });
    expect(
      await executeJson(registry, "web.sports", {
        league: "nba",
        fn: "standings",
        team: "GSW",
      }),
    ).toEqual({
      league: "nba",
      fn: "standings",
      team: "GSW",
      data: [],
    });
    expect(
      await executeJson(registry, "web.time", { utc_offset: "+03:00" }),
    ).toEqual({
      utc_offset: "+03:00",
      iso: "2026-07-13T13:20:30+00:00",
      time: "13:20:30",
      date: "2026-07-13",
    });
  } finally {
    await state.close();
  }
});

test("unconfigured host-bound operator tools fail explicitly without fabricating results", async () => {
  const state = new OperatorState({
    config: createOperatorRuntimeConfig({
      enabled: true,
      powerToolsEnabled: true,
    }),
  });
  const registry = new ToolRegistry();
  state.registerTools(registry);
  try {
    await expect(
      registry.execute(call("apply_patch", { patch: VALID_PATCH }), {
        metadata: {},
      }),
    ).rejects.toThrow("operator.patchApplier");
    await expect(
      registry.execute(call("parallel_tools", { calls: [] }), { metadata: {} }),
    ).rejects.toThrow("operator.parallelReadonlyToolPort");
    await expect(
      registry.execute(call("view_image", { path: "/workspace/missing.png" }), {
        metadata: {},
      }),
    ).rejects.toThrow("operator.imageInspector");
    await expect(
      registry.execute(call("web.search_query", { q: "current facts" }), {
        metadata: {},
      }),
    ).rejects.toThrow("operator.webPort");
  } finally {
    await state.close();
  }
});

test("runtime composition retains a host-composed operator state and its injected ports", async () => {
  const operator = new OperatorState({
    config: createOperatorRuntimeConfig({
      enabled: true,
      powerToolsEnabled: true,
    }),
    patchApplier: {
      async applyPatch() {
        return { applied: false, checked: true };
      },
    },
  });
  const runtime = await composeRuntimeFeatures(
    {
      discoverConventionalExtensions: false,
      operator: createOperatorRuntimeConfig({
        enabled: true,
        powerToolsEnabled: true,
      }),
    },
    { operatorState: operator },
  );
  try {
    expect(runtime.operatorState).toBe(operator);
    const registry = new ToolRegistry();
    runtime.operatorState?.registerTools(registry);
    expect(
      await executeJson(registry, "apply_patch", {
        patch: VALID_PATCH,
        check: true,
      }),
    ).toEqual({ applied: false, checked: true });
  } finally {
    await runtime.close();
  }
});

test("operator subagent tools accept task-description and target aliases through the native manager", async () => {
  let tick = 0;
  const manager = new SpawnedAgentManager({
    now: () => new Date(Date.UTC(2026, 6, 13, 10, 0, tick++)),
    runner: async (request) => ({ content: "done:" + request.input }),
  });
  const state = new OperatorState({
    config: createOperatorRuntimeConfig({
      enabled: true,
      powerToolsEnabled: true,
    }),
    subagentManager: manager,
  });
  const registry = new ToolRegistry();
  state.registerTools(registry);
  try {
    const fromAlias = await executeJson(registry, "spawn_agent", {
      nickname: "alias",
      task_description: "hello from alias",
    });
    expect(fromAlias.id).toBe("alias");
    expect(
      await executeJson(registry, "wait_agent", {
        targets: ["alias"],
        timeout_ms: 1_000,
      }),
    ).toMatchObject({
      completed: [{ id: "alias", last_output: "done:hello from alias" }],
      pending: [],
    });

    await executeJson(registry, "spawn_agent", { nickname: "first" });
    await executeJson(registry, "spawn_agent", { nickname: "second" });
    expect(
      await executeJson(registry, "send_input", { message: "latest handle" }),
    ).toMatchObject({ id: "second", last_input: "latest handle" });
    expect(
      await executeJson(registry, "wait_agent", {
        targets: ["second"],
        timeout_ms: 1_000,
      }),
    ).toMatchObject({
      completed: [{ id: "second", last_output: "done:latest handle" }],
    });
    expect(
      await executeJson(registry, "send_input", {
        id: "first",
        task_description: "explicit alias",
      }),
    ).toMatchObject({ id: "first", last_input: "explicit alias" });
    expect(
      await executeJson(registry, "wait_agent", {
        targets: ["first"],
        timeout_ms: 1_000,
      }),
    ).toMatchObject({
      completed: [{ id: "first", last_output: "done:explicit alias" }],
    });
    expect(
      await executeJson(registry, "close_agent", { target: "first" }),
    ).toMatchObject({ id: "first", closed: true });
    expect(
      await executeJson(registry, "resume_agent", { agent_id: "first" }),
    ).toMatchObject({ id: "first", closed: false });
  } finally {
    await state.close();
  }
});

test("terminal operator tools keep a session visible through polling, interruption, and close", async () => {
  const state = new OperatorState({
    config: createOperatorRuntimeConfig({
      enabled: true,
      powerToolsEnabled: true,
    }),
  });
  const registry = new ToolRegistry();
  state.registerTools(registry);
  try {
    const started = await executeJson(registry, "exec_command", {
      cmd: "printf 'ready\\n'; sleep 3",
      yield_time_ms: 10,
    });
    const sessionId = started.session_id;
    expect(typeof sessionId).toBe("string");
    if (typeof sessionId !== "string")
      throw new Error("exec_command did not return a session id");
    const sessions = JSON.parse(
      await registry.execute(call("list_terminal_sessions", {}), {
        metadata: {},
      }),
    ) as JsonObject[];
    expect(
      sessions.some((session) => session.session_id === sessionId),
    ).toBeTrue();
    expect(
      await executeJson(registry, "write_stdin", {
        session_id: sessionId,
        chars: "",
        yield_time_ms: 10,
      }),
    ).toMatchObject({ session_id: sessionId, running: true });
    expect(
      await executeJson(registry, "write_stdin", {
        session_id: sessionId,
        interrupt: true,
        yield_time_ms: 200,
      }),
    ).toMatchObject({ session_id: sessionId });
    expect(
      await executeJson(registry, "close_terminal_session", {
        session_id: sessionId,
      }),
    ).toMatchObject({ session_id: sessionId, closed: true });
    const afterClose = JSON.parse(
      await registry.execute(call("list_terminal_sessions", {}), {
        metadata: {},
      }),
    ) as JsonObject[];
    expect(
      afterClose.some((session) => session.session_id === sessionId),
    ).toBeFalse();
  } finally {
    await state.close();
  }
});
