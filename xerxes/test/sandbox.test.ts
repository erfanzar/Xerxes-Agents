// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdtemp, realpath, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { expect, test } from "bun:test";

import {
  LocalSandboxBackend,
  LocalSandboxUnavailableError,
  SandboxExecutionUnavailableError,
  SandboxedToolExecutor,
  SandboxMode,
  SandboxRouter,
} from "../src/security/sandbox.js";
import {
  SubprocessSandboxBackend,
  SubprocessSandboxRequestError,
  SubprocessSandboxTimeoutError,
} from "../src/security/subprocessSandbox.js";
import type { ToolCall } from "../src/types/toolCalls.js";

const CALL: ToolCall = {
  id: "call",
  type: "function",
  function: { name: "exec_command", arguments: { cmd: "pwd" } },
};

test("sandbox router preserves off, warn, elevated, and strict routing decisions", () => {
  const warnings: string[] = [];
  const warn = new SandboxRouter({
    config: { mode: SandboxMode.WARN, sandboxedTools: ["exec_command"] },
    onWarning: (decision) => warnings.push(decision.reason),
  });
  expect(warn.decide("exec_command")).toMatchObject({
    context: "host",
    reason: expect.stringContaining("Warn mode"),
  });
  expect(warnings).toHaveLength(1);

  const strict = new SandboxRouter({
    config: {
      mode: SandboxMode.STRICT,
      sandboxedTools: ["exec_command"],
      elevatedTools: ["ReadFile"],
    },
  });
  expect(strict.decide("exec_command")).toMatchObject({ context: "sandbox" });
  expect(strict.decide("ReadFile")).toMatchObject({
    context: "host",
    reason: "Tool is marked as elevated",
  });
  expect(strict.decide("ListDir")).toMatchObject({
    context: "host",
    reason: "Tool not designated for sandbox",
  });
});

test("sandbox executor only invokes the host for host decisions and fails closed without a backend", async () => {
  const strict = new SandboxRouter({
    config: { mode: SandboxMode.STRICT, sandboxedTools: ["exec_command"] },
  });
  const executor = new SandboxedToolExecutor(
    {
      execute: async () => "host result",
    },
    strict,
  );
  await expect(executor.execute(CALL, { metadata: {} })).rejects.toBeInstanceOf(
    SandboxExecutionUnavailableError,
  );

  const calls: string[] = [];
  const routed = new SandboxedToolExecutor(
    {
      execute: async () => {
        calls.push("host");
        return "host result";
      },
    },
    new SandboxRouter({
      config: { mode: SandboxMode.STRICT, sandboxedTools: ["exec_command"] },
      backend: {
        execute: async (request) => {
          calls.push("sandbox:" + request.toolName);
          return "sandbox result";
        },
      },
    }),
  );
  expect(await routed.execute(CALL, { metadata: {} })).toBe("sandbox result");
  expect(calls).toEqual(["sandbox:exec_command"]);
});

test("local sandbox selects the first available platform runner and reports enforcement", async () => {
  const executed: string[] = [];
  const router = new SandboxRouter({
    config: {
      mode: SandboxMode.STRICT,
      sandboxedTools: ["exec_command"],
      backendType: "local",
      sandboxNetworkAccess: false,
      sandboxTimeout: 2,
      workingDirectory: "/workspace",
    },
    localSandboxHost: {
      platform: "linux",
      async probe(runner) {
        if (runner === "bubblewrap") throw new Error("bwrap probe failed");
        return runner === "landlock"
          ? {
              available: true,
              enforcement: "partial",
              filesystemIsolation: true,
              processIsolation: false,
              networkIsolation: true,
              limitations: ["kernel ABI limits filesystem coverage"],
            }
          : {
              available: false,
              enforcement: "none",
              filesystemIsolation: false,
              processIsolation: false,
              networkIsolation: false,
              limitations: [],
            };
      },
      async execute(runner, _request, policy) {
        executed.push(`${runner}:${policy.networkAccess}:${policy.timeoutMs}`);
        return "locally sandboxed";
      },
    },
  });

  expect(router.backend).toBeInstanceOf(LocalSandboxBackend);
  expect(await router.executeInSandbox({
    toolName: "exec_command",
    arguments: { cmd: "pwd" },
    context: { metadata: {} },
  })).toBe("locally sandboxed");
  expect(executed).toEqual(["landlock:false:2000"]);
  expect(router.backend?.getCapabilities?.()).toMatchObject({
    backend: "local",
    runner: "landlock",
    enforcement: "partial",
    filesystemIsolation: true,
    networkIsolation: true,
    processIsolation: false,
    failClosed: true,
  });
});

test("local sandbox fails closed when no platform runner is available", async () => {
  const router = new SandboxRouter({
    config: {
      mode: SandboxMode.STRICT,
      sandboxedTools: ["exec_command"],
      backendType: "local",
    },
    localSandboxHost: {
      platform: "darwin",
      async probe() {
        return {
          available: false,
          enforcement: "none",
          filesystemIsolation: false,
          processIsolation: false,
          networkIsolation: false,
          limitations: ["runner unavailable"],
        };
      },
      async execute() {
        throw new Error("must not execute without an available runner");
      },
    },
  });

  await expect(router.executeInSandbox({
    toolName: "exec_command",
    arguments: { cmd: "pwd" },
    context: { metadata: {} },
  })).rejects.toBeInstanceOf(LocalSandboxUnavailableError);
  expect(router.backend?.getCapabilities?.()).toMatchObject({
    backend: "local",
    available: false,
    enforcement: "none",
    failClosed: true,
  });
});

test("local sandbox rejects a conflicting explicit backend instead of weakening isolation", () => {
  expect(() => new SandboxRouter({
    config: {
      mode: SandboxMode.STRICT,
      sandboxedTools: ["exec_command"],
      backendType: "local",
    },
    backend: new SubprocessSandboxBackend({ allowedCommands: [process.execPath] }),
  })).toThrow("conflicts with explicitly supplied backend");
});

test("sandbox tool sets match case-insensitively so config casing typos cannot route to host", () => {
  // security/policy.ts compares tool names lowercased; a mixed-case sandbox or
  // elevated entry must behave identically instead of silently missing.
  const strict = new SandboxRouter({
    config: {
      mode: SandboxMode.STRICT,
      sandboxedTools: ["Exec_Command"],
      elevatedTools: ["ReadFile"],
    },
  });
  expect(strict.decide("exec_command")).toMatchObject({ context: "sandbox" });
  expect(strict.decide("EXEC_COMMAND")).toMatchObject({ context: "sandbox" });
  expect(strict.decide("Exec_Command")).toMatchObject({ context: "sandbox" });
  expect(strict.decide("readfile")).toMatchObject({
    context: "host",
    reason: "Tool is marked as elevated",
  });
  // Unrelated tools still fall through to host under strict mode.
  expect(strict.decide("ListDir")).toMatchObject({ context: "host" });

  // Warn mode keeps its advisory behavior for casing-mismatched entries too.
  const warnings: string[] = [];
  const warn = new SandboxRouter({
    config: { mode: SandboxMode.WARN, sandboxedTools: ["EXEC_COMMAND"] },
    onWarning: (decision) => warnings.push(decision.reason),
  });
  expect(warn.decide("exec_command")).toMatchObject({ context: "host" });
  expect(warnings).toHaveLength(1);
});

test("subprocess backend executes strict router requests with a bounded cwd and sanitized environment", async () => {
  await inTemporaryDirectory(async (workspace) => {
    const originalSecret = process.env.XERXES_SANDBOX_PARENT_SECRET;
    process.env.XERXES_SANDBOX_PARENT_SECRET = "must-not-reach-child";
    try {
      const router = new SandboxRouter({
        config: {
          mode: SandboxMode.STRICT,
          sandboxedTools: ["exec_command"],
          backendType: "subprocess",
          workingDirectory: workspace,
          sandboxTimeout: 1,
          backendConfig: {
            envVars: { XERXES_SANDBOX_VISIBLE: "configured-value" },
            extraArgs: {
              allowedCommands: [process.execPath],
              maxOutputChars: 512,
            },
          },
        },
      });
      expect(router.backend).toBeInstanceOf(SubprocessSandboxBackend);

      const childProgram = [
        "process.stdout.write(JSON.stringify({",
        "cwd: process.cwd(),",
        "configured: process.env.XERXES_SANDBOX_VISIBLE ?? null,",
        "parentSecret: process.env.XERXES_SANDBOX_PARENT_SECRET ?? null,",
        "}))",
      ].join("");
      const result = JSON.parse(
        await router.execute(
          {
            id: "subprocess-call",
            type: "function",
            function: {
              name: "exec_command",
              arguments: {
                cmd: process.execPath,
                args: ["-e", childProgram],
                workdir: ".",
              },
            },
          },
          { metadata: {} },
          async () => {
            throw new Error(
              "strict sandbox execution must not invoke the host",
            );
          },
        ),
      ) as {
        readonly cwd: string;
        readonly exitCode: number;
        readonly stdout: string;
        readonly truncated: boolean;
      };

      expect(result.cwd).toBe(".");
      expect(result.exitCode).toBe(0);
      expect(result.truncated).toBeFalse();
      expect(JSON.parse(result.stdout)).toEqual({
        cwd: await realpath(workspace),
        configured: "configured-value",
        parentSecret: null,
      });
      expect(router.backend?.getCapabilities?.()).toMatchObject({
        environmentSanitized: true,
        filesystemIsolation: false,
        networkIsolation: false,
        timeoutEnforced: true,
        workingDirectoryBounded: true,
      });
    } finally {
      if (originalSecret === undefined) {
        delete process.env.XERXES_SANDBOX_PARENT_SECRET;
      } else {
        process.env.XERXES_SANDBOX_PARENT_SECRET = originalSecret;
      }
    }
  });
});

test("subprocess backend rejects unsafe requests and enforces the child timeout", async () => {
  await inTemporaryDirectory(async (workspace) => {
    const backend = new SubprocessSandboxBackend({
      allowedCommands: [process.execPath],
      maxOutputChars: 256,
      maxTimeoutMs: 1_000,
      workingDirectory: workspace,
    });

    await expect(
      backend.execute(
        subprocessRequest("read_file", { cmd: process.execPath }),
      ),
    ).rejects.toBeInstanceOf(SubprocessSandboxRequestError);
    await expect(
      backend.execute(subprocessRequest("exec_command", { cmd: "sh" })),
    ).rejects.toBeInstanceOf(SubprocessSandboxRequestError);
    await expect(
      backend.execute(
        subprocessRequest("exec_command", {
          cmd: process.execPath,
          args: ["-e", 'process.stdout.write("ignored")'],
          workdir: "..",
        }),
      ),
    ).rejects.toBeInstanceOf(SubprocessSandboxRequestError);
    await expect(
      backend.execute(
        subprocessRequest("exec_command", {
          cmd: process.execPath,
          args: ["-e", "setTimeout(() => {}, 500)"],
          timeout_ms: 40,
        }),
      ),
    ).rejects.toBeInstanceOf(SubprocessSandboxTimeoutError);

    const capped = JSON.parse(
      await backend.execute(
        subprocessRequest("exec_command", {
          cmd: process.execPath,
          args: ["-e", 'process.stdout.write("abcdefghijklmnopqrst")'],
          max_output_chars: 8,
        }),
      ),
    ) as { readonly stdout: string; readonly truncated: boolean };
    expect(capped.truncated).toBeTrue();
    expect(capped.stdout.length).toBeLessThanOrEqual(8);
  });
});

async function inTemporaryDirectory(
  run: (directory: string) => Promise<void>,
): Promise<void> {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-bun-sandbox-"));
  try {
    await run(directory);
  } finally {
    await rm(directory, { force: true, recursive: true });
  }
}

function subprocessRequest(
  toolName: string,
  arguments_: ToolCall["function"]["arguments"],
) {
  return { toolName, arguments: arguments_, context: { metadata: {} } };
}
