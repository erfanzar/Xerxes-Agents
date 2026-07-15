// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

import {
  createNativeHardTasks,
  parseHardCliOptions,
  runHardCli,
  runHardSuite,
  type EvaluationAgent,
  type EvaluationSessionPort,
  type EvaluationTurnOptions,
  type EvaluationTurnResult,
  type NativeHardTask,
} from "../playground/index.js";

test("native hard catalog replaces all sixteen legacy scenarios with TypeScript fixtures and typed checkers", () => {
  const tasks = createNativeHardTasks();
  expect(tasks).toHaveLength(16);
  expect(tasks.map((task) => task.name)).toEqual([
    "multifile_feature_expr_evaluator",
    "multifile_feature_build_scheduler",
    "bug_hunt_booking_overlap",
    "bug_hunt_lru_and_mutable_default",
    "tool_exploration_tax_rate_target",
    "tool_exploration_handler_registry",
    "error_recovery_roman_roundtrip",
    "error_recovery_money_split",
    "hard_reasoning_leap_phase_schedule",
    "hard_reasoning_server_grid_logic",
    "memory_pressure_single_value_update",
    "memory_pressure_multifact_partial_update",
    "instruction_following_json_audit_strict",
    "instruction_following_terse_refuse_noedit",
    "robustness_injected_csv_summary",
    "robustness_corrupt_config_recover_and_recall",
  ]);
  for (const task of tasks) {
    expect(typeof task.check).toBe("function");
    expect(task.files.some((file) => file.path.endsWith(".py"))).toBeFalse();
    expect(
      task.files.some((file) =>
        file.content.includes("serialized grader source"),
      ),
    ).toBeFalse();
  }
});

test("native hard checkers execute TypeScript behavior, task-local file contracts, and fresh-session recall", async () => {
  const root = await temporaryDirectory("xerxes-hard-catalog-");
  const tasks = createNativeHardTasks();
  try {
    await expect(
      runOne(tasks, "bug_hunt_booking_overlap", root, new BookingAgent(root)),
    ).resolves.toMatchObject({
      rows: [{ name: "bug_hunt_booking_overlap", ok: true }],
    });
    await expect(
      runOne(
        tasks,
        "tool_exploration_handler_registry",
        root,
        new RegistryAgent(root),
      ),
    ).resolves.toMatchObject({
      rows: [{ name: "tool_exploration_handler_registry", ok: true }],
    });
    await expect(
      runOne(
        tasks,
        "memory_pressure_single_value_update",
        root,
        new PortMemoryAgent(root),
      ),
    ).resolves.toMatchObject({
      rows: [{ name: "memory_pressure_single_value_update", ok: true }],
    });
    await expect(
      runOne(
        tasks,
        "instruction_following_json_audit_strict",
        root,
        new StrictAuditAgent(),
      ),
    ).resolves.toMatchObject({
      rows: [{ name: "instruction_following_json_audit_strict", ok: true }],
    });
  } finally {
    await rm(root, { force: true, recursive: true });
  }
});

test("hard CLI preserves -v and -k syntax without sharing the warm-up entrypoint", () => {
  expect(
    parseHardCliOptions([
      "-v",
      "-k",
      "memory",
      "--transport",
      "./transport.ts",
      "--run-root",
      "./runs",
    ]),
  ).toMatchObject({
    help: false,
    keyword: "memory",
    transportModule: "./transport.ts",
    verbose: true,
  });
  expect(() => parseHardCliOptions(["--transport"])).toThrow(
    "--transport requires a non-empty value",
  );
  expect(() => parseHardCliOptions(["--unknown"])).toThrow("unknown argument");
});

test("hard CLI runs a typed native task through an injected Bun transport and cleans its isolated run", async () => {
  const root = await temporaryDirectory("xerxes-hard-cli-");
  const task = createNativeHardTasks().find(
    (candidate) => candidate.name === "hard_reasoning_leap_phase_schedule",
  );
  if (task === undefined) throw new Error("missing leap task");
  const output: string[] = [];
  try {
    const exitCode = await runHardCli(["--run-root", root, "--verbose"], {
      createSessionPort: async () =>
        answerTransport(
          "Final answer: 2028-08-10, Thursday, with one leap day.",
        ),
      tasks: [task],
      write: (text) => {
        output.push(text);
      },
      writeError: (text) => {
        throw new Error(text);
      },
    });
    expect(exitCode).toBe(0);
    expect(output.join("")).toContain("Xerxes HARD eval");
    expect(output.join("")).toContain("SCORE: 1/1 passed");
    const runs = await Array.fromAsync(
      new Bun.Glob("run-*").scan({ cwd: root }),
    );
    expect(runs).toHaveLength(0);
  } finally {
    await rm(root, { force: true, recursive: true });
  }
});

async function runOne(
  tasks: readonly NativeHardTask[],
  name: string,
  sandboxDirectory: string,
  agent: EvaluationAgent,
) {
  const task = tasks.find((candidate) => candidate.name === name);
  if (task === undefined) throw new Error(`missing task ${name}`);
  return runHardSuite({ agent, sandboxDirectory, tasks: [task] });
}

class BookingAgent implements EvaluationAgent {
  readonly model = "booking-fixture";

  constructor(private readonly root: string) {}

  async close(): Promise<void> {}

  async freshSession(): Promise<void> {}

  async start(): Promise<void> {}

  async turn(
    _prompt: string,
    _options?: EvaluationTurnOptions,
  ): Promise<EvaluationTurnResult> {
    await writeFile(
      join(this.root, "bug_hunt_booking_overlap", "scheduler.ts"),
      `export type Interval = readonly [number, number]

export function overlaps(left: Interval, right: Interval): boolean {
  return left[0] < right[1] && right[0] < left[1]
}

export function hasConflict(bookings: readonly Interval[]): boolean {
  for (let index = 0; index < bookings.length; index += 1) {
    for (let other = index + 1; other < bookings.length; other += 1) {
      const left = bookings[index]
      const right = bookings[other]
      if (left !== undefined && right !== undefined && overlaps(left, right)) return true
    }
  }
  return false
}
`,
      "utf8",
    );
    return result("fixed", ["ReadFile", "EditFile"]);
  }
}

class RegistryAgent implements EvaluationAgent {
  readonly model = "registry-fixture";
  private turns = 0;

  constructor(private readonly root: string) {}

  async close(): Promise<void> {}

  async freshSession(): Promise<void> {}

  async start(): Promise<void> {}

  async turn(
    _prompt: string,
    _options?: EvaluationTurnOptions,
  ): Promise<EvaluationTurnResult> {
    this.turns += 1;
    if (this.turns === 1) {
      await writeFile(
        join(
          this.root,
          "tool_exploration_handler_registry",
          "plugins",
          "b",
          "events.ts",
        ),
        `import { registerHandler } from "../../core/registry.js"

registerHandler("commit", event => \`commit:\${event}\`)
registerHandler("abort", event => \`abort:\${event}\`)

export function processEvent(event: string): string {
  return \`commit:\${event}\`
}
`,
        "utf8",
      );
      return result(
        "added abort in plugins/b/events.ts after tracing registry imports",
        ["GrepTool", "EditFile"],
      );
    }
    return result(
      "The handler was added in plugins/b/events.ts because that module registers the wired commit channel.",
      [],
    );
  }
}

class PortMemoryAgent implements EvaluationAgent {
  readonly model = "port-memory-fixture";
  private turns = 0;

  constructor(private readonly root: string) {}

  async close(): Promise<void> {}

  async freshSession(): Promise<void> {}

  async start(): Promise<void> {}

  async turn(
    _prompt: string,
    _options?: EvaluationTurnOptions,
  ): Promise<EvaluationTurnResult> {
    this.turns += 1;
    if (this.turns === 6) {
      await writeFile(
        join(
          this.root,
          "memory_pressure_single_value_update",
          "service",
          "config.ts",
        ),
        "export const PORT = 7474\n",
        "utf8",
      );
      return result("7474", ["EditFile", "ExecCommand"]);
    }
    return result("acknowledged");
  }
}

class StrictAuditAgent implements EvaluationAgent {
  readonly model = "strict-audit-fixture";

  async close(): Promise<void> {}

  async freshSession(): Promise<void> {}

  async start(): Promise<void> {}

  async turn(
    _prompt: string,
    _options?: EvaluationTurnOptions,
  ): Promise<EvaluationTurnResult> {
    return result(
      '{"total_files":5,"typescript_files":2,"largest_file":"archive.log","config_present":false,"git_remote_url":null}',
      ["ReadFile"],
    );
  }
}

function result(
  text: string,
  tools: readonly string[] = [],
): EvaluationTurnResult {
  return {
    contextTokens: 0,
    error: undefined,
    latencyMs: 1,
    retries: 0,
    text,
    tools,
  };
}

function answerTransport(answer: string): EvaluationSessionPort {
  return {
    approve: async () => undefined,
    close: async () => undefined,
    reset: async () => undefined,
    start: async () => ({ model: "hard-cli-fixture" }),
    submit: async function* () {
      yield { text: answer, type: "text" } as const;
      yield { type: "turn_end" } as const;
    },
  };
}

async function temporaryDirectory(prefix: string): Promise<string> {
  return mkdtemp(join(tmpdir(), prefix));
}
