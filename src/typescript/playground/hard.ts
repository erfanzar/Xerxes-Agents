// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir, rm, writeFile } from "node:fs/promises";
import { dirname, isAbsolute, relative, resolve } from "node:path";
import { diagnoseEvaluationFailure } from "./harness.js";
import type {
  EvaluationAgent,
  EvaluationCheck,
  EvaluationJudgePort,
  EvaluationReport,
  EvaluationScoreRow,
  EvaluationTaskFile,
  EvaluationTaskStep,
  EvaluationTurnResult,
} from "./types.js";

const DEFAULT_GRADER_TIMEOUT_MS = 90_000;

/** Context supplied to one native hard-task behavioral checker. */
export interface NativeHardCheckContext {
  readonly results: readonly EvaluationTurnResult[];
  readonly signal: AbortSignal;
  readonly workspaceDirectory: string;
}

/** A TypeScript-native hard task. Its checker may use an injected Bun evaluator or another host sandbox. */
export interface NativeHardTask {
  readonly category: string;
  check(
    context: NativeHardCheckContext,
  ): Promise<EvaluationCheck> | EvaluationCheck;
  readonly difficulty: number;
  readonly files: readonly EvaluationTaskFile[];
  readonly name: string;
  readonly steps: readonly EvaluationTaskStep[];
  readonly whyHard: string;
}

/** Options for a native hard-suite run. */
export interface HardSuiteOptions {
  readonly agent: EvaluationAgent;
  readonly graderTimeoutMs?: number;
  readonly judge?: EvaluationJudgePort;
  readonly keyword?: string;
  readonly onRow?: (row: EvaluationScoreRow) => void;
  /** Receives each completed turn when a CLI needs Python-compatible verbose progress. */
  readonly onStep?: (step: HardStepResult) => void | Promise<void>;
  readonly sandboxDirectory: string;
  readonly tasks: readonly NativeHardTask[];
}

/** One completed hard-suite prompt step, emitted only when the caller requests it. */
export interface HardStepResult {
  readonly prompt: string;
  readonly result: EvaluationTurnResult;
  readonly stepIndex: number;
  readonly task: Pick<NativeHardTask, "category" | "difficulty" | "name">;
}

/**
 * Run hard tasks with isolated task directories and watchdog-bounded graders.
 *
 * Callers provide native typed `NativeHardTask` checkers. The runner accepts
 * no serialized source-code grader format, so execution remains Bun-native.
 */
export async function runHardSuite(
  options: HardSuiteOptions,
): Promise<EvaluationReport> {
  const sandboxDirectory = requiredDirectory(
    options.sandboxDirectory,
    "sandboxDirectory",
  );
  const graderTimeoutMs = positiveInteger(
    options.graderTimeoutMs ?? DEFAULT_GRADER_TIMEOUT_MS,
    "graderTimeoutMs",
  );
  const tasks = options.tasks.filter(
    (task) =>
      options.keyword === undefined || task.name.includes(options.keyword),
  );
  await rm(sandboxDirectory, { force: true, recursive: true });
  await mkdir(sandboxDirectory, { recursive: true });
  const rows: EvaluationScoreRow[] = [];

  try {
    await options.agent.start();
    for (const task of tasks) {
      const workspaceDirectory = await setupTaskWorkspace(
        sandboxDirectory,
        task,
      );
      const results: EvaluationTurnResult[] = [];
      let lastPrompt = "";
      for (const [stepIndex, step] of task.steps.entries()) {
        if (step.freshSession) await options.agent.freshSession();
        lastPrompt = step.prompt.replaceAll("{dir}", task.name);
        const result = await options.agent.turn(lastPrompt);
        results.push(result);
        await options.onStep?.({
          prompt: lastPrompt,
          result,
          stepIndex,
          task: {
            category: task.category,
            difficulty: task.difficulty,
            name: task.name,
          },
        });
      }
      const last = lastResult(results);
      const check =
        last.error === undefined
          ? await checkBeforeDeadline(
              task,
              results,
              workspaceDirectory,
              graderTimeoutMs,
            )
          : { ok: false, detail: `ERROR: ${last.error}` };
      const tools = [
        ...new Set(results.flatMap((result) => result.tools)),
      ].sort();
      const diagnosis = check.ok
        ? undefined
        : await diagnoseEvaluationFailure(
            {
              graderDetail: check.detail,
              reply: last.text,
              taskPrompt: lastPrompt,
              tools,
              whyHard: task.whyHard,
            },
            last.error,
            options.judge,
          );
      const row: EvaluationScoreRow = {
        category: task.category,
        detail: check.detail,
        diagnosis,
        difficulty: task.difficulty,
        latencyMs: totalLatency(results),
        name: task.name,
        ok: check.ok,
      };
      rows.push(row);
      options.onRow?.(row);
    }
  } finally {
    await options.agent.close();
  }

  return {
    kind: "hard",
    model: options.agent.model,
    rows,
    sandboxDirectory,
    totalLatencyMs: totalLatency(rows),
  };
}

/** Build one hard-task directory from task-owned fixtures. */
export async function setupTaskWorkspace(
  sandboxDirectory: string,
  task: Pick<NativeHardTask, "files" | "name">,
): Promise<string> {
  const sandbox = requiredDirectory(sandboxDirectory, "sandboxDirectory");
  const name = requiredTaskName(task.name);
  const workspaceDirectory = safeChild(sandbox, name);
  await rm(workspaceDirectory, { force: true, recursive: true });
  await mkdir(workspaceDirectory, { recursive: true });
  await Promise.all(
    task.files.map(async (file) => {
      const target = safeChild(workspaceDirectory, file.path);
      await mkdir(dirname(target), { recursive: true });
      await writeFile(target, file.content, "utf8");
    }),
  );
  return workspaceDirectory;
}

/** Render the detailed hard-suite scorecard, including category and difficulty summaries. */
export function formatHardReport(report: EvaluationReport): string {
  if (report.kind !== "hard")
    throw new Error("formatHardReport requires a hard report");
  const rows = report.rows.flatMap((row) => [
    `  [${row.ok ? "PASS" : "FAIL"}] d${row.difficulty ?? "?"} ${pad(row.name, 28)} ${pad(row.category, 12)} ${seconds(row.latencyMs)}  ${row.detail}`,
    ...(row.ok || row.diagnosis === undefined
      ? []
      : [`        ↳ WHY: ${row.diagnosis}`]),
  ]);
  const passed = report.rows.filter((row) => row.ok).length;
  const failed = report.rows.filter((row) => !row.ok);
  return [
    "",
    `  Xerxes HARD eval — model: ${report.model}   tasks: ${report.rows.length}   sandbox: ${report.sandboxDirectory}`,
    "",
    ...rows,
    "",
    `  ${"-".repeat(70)}`,
    `  SCORE: ${passed}/${report.rows.length} passed  (${percentage(passed, report.rows.length)}%)   total ${seconds(report.totalLatencyMs)}   model=${report.model}`,
    `  by category:   ${summaryBy(report.rows, (row) => row.category)}`,
    `  by difficulty: ${summaryBy(report.rows, (row) => `d${row.difficulty ?? 0}`)}`,
    ...(failed.length === 0
      ? []
      : [
          `  FAILED: ${failed.map((row) => `${row.name} (${row.detail.slice(0, 44)})`).join(", ")}`,
        ]),
    "",
  ].join("\n");
}

async function checkBeforeDeadline(
  task: NativeHardTask,
  results: readonly EvaluationTurnResult[],
  workspaceDirectory: string,
  timeoutMs: number,
): Promise<EvaluationCheck> {
  const controller = new AbortController();
  let timeout: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      Promise.resolve(
        task.check({ results, signal: controller.signal, workspaceDirectory }),
      ),
      new Promise<EvaluationCheck>((resolve) => {
        timeout = setTimeout(() => {
          controller.abort();
          resolve({
            ok: false,
            detail: `grader timed out (> ${(timeoutMs / 1_000).toFixed(0)}s)`,
          });
        }, timeoutMs);
      }),
    ]);
  } catch (error) {
    return {
      ok: false,
      detail:
        `grader raised: ${errorName(error)}: ${errorMessage(error)}`.slice(
          0,
          120,
        ),
    };
  } finally {
    if (timeout !== undefined) clearTimeout(timeout);
  }
}

function lastResult(
  results: readonly EvaluationTurnResult[],
): EvaluationTurnResult {
  const result = results.at(-1);
  if (result === undefined)
    throw new Error("hard task must contain at least one step");
  return result;
}

function requiredDirectory(value: string, label: string): string {
  const directory = value.trim();
  if (!directory) throw new Error(`${label} must not be empty`);
  return resolve(directory);
}

function requiredTaskName(value: string): string {
  const name = value.trim();
  if (!/^[A-Za-z0-9][A-Za-z0-9._-]*$/.test(name))
    throw new Error("task name is not safe for a workspace path");
  return name;
}

function safeChild(parent: string, child: string): string {
  const candidate = resolve(parent, child);
  const fromParent = relative(parent, candidate);
  if (
    !fromParent ||
    fromParent === ".." ||
    fromParent.startsWith("../") ||
    isAbsolute(fromParent)
  ) {
    throw new Error(`task file must remain inside its workspace: ${child}`);
  }
  return candidate;
}

function totalLatency(
  items: readonly { readonly latencyMs: number }[],
): number {
  return items.reduce((total, item) => total + item.latencyMs, 0);
}

function pad(value: string, width: number): string {
  return value.padEnd(width);
}

function seconds(milliseconds: number): string {
  return `${(milliseconds / 1_000).toFixed(0)}s`;
}

function percentage(passed: number, total: number): number {
  return Math.floor((100 * passed) / Math.max(1, total));
}

function summaryBy(
  rows: readonly EvaluationScoreRow[],
  key: (row: EvaluationScoreRow) => string,
): string {
  const groups = new Map<string, boolean[]>();
  for (const row of rows) {
    const entries = groups.get(key(row)) ?? [];
    entries.push(row.ok);
    groups.set(key(row), entries);
  }
  return [...groups.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(
      ([name, values]) =>
        `${name}=${values.filter(Boolean).length}/${values.length}`,
    )
    .join("  ");
}

function positiveInteger(value: number, label: string): number {
  if (!Number.isSafeInteger(value) || value <= 0)
    throw new Error(`${label} must be a positive integer`);
  return value;
}

function errorName(error: unknown): string {
  return error instanceof Error && error.name ? error.name : "Error";
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
