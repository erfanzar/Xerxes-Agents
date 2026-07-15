// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { access, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

interface CommandResult {
  readonly exitCode: number;
  readonly stderr: string;
  readonly stdout: string;
}

/** Install a packed release into an empty project and exercise its CLI and OpenTUI module. */
export async function smokeReleasePackage(archivePath: string): Promise<void> {
  const archive = resolve(archivePath);
  await access(archive);
  const temporaryDirectory = await mkdtemp(
    join(tmpdir(), "xerxes-release-smoke-"),
  );

  try {
    await writeFile(
      join(temporaryDirectory, "package.json"),
      `${JSON.stringify({ name: "xerxes-release-smoke", private: true }, null, 2)}\n`,
      "utf8",
    );
    assertCommand(
      await run(
        [process.execPath, "add", "--exact", archive],
        temporaryDirectory,
      ),
      "install packed release",
    );

    const packageDirectory = join(
      temporaryDirectory,
      "node_modules",
      "xerxes-bun",
    );
    const version = assertCommand(
      await run(
        [process.execPath, join(packageDirectory, "bin/xerxes"), "--version"],
        temporaryDirectory,
      ),
      "run packed CLI",
    );
    if (
      !/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/u.test(version.stdout.trim())
    ) {
      throw new Error(
        `Packed CLI returned an invalid version: ${version.stdout.trim()}`,
      );
    }

    const openTuiRuntime = assertCommand(
      await run(
        [
          process.execPath,
          "--eval",
          [
            'const core = await import("@opentui/core");',
            'const react = await import("@opentui/react");',
            'if (typeof core.createCliRenderer !== "function" || typeof react.createRoot !== "function") throw new Error("OpenTUI runtime exports are unavailable");',
            'console.log("opentui-runtime-ok");',
          ].join(" "),
        ],
        temporaryDirectory,
      ),
      "load packed OpenTUI runtime dependencies",
    );
    if (openTuiRuntime.stdout.trim() !== "opentui-runtime-ok") {
      throw new Error(
        `Packed OpenTUI runtime probe returned unexpected output: ${openTuiRuntime.stdout.trim()}`,
      );
    }

    const tui = assertCommand(
      await run(
        [process.execPath, join(packageDirectory, "ui/entry.js")],
        temporaryDirectory,
      ),
      "load packed OpenTUI entry",
    );
    if (!tui.stdout.includes("xerxes-tui: no TTY")) {
      throw new Error(
        `Packed OpenTUI smoke returned unexpected output: ${tui.stdout.trim()}`,
      );
    }
  } finally {
    await rm(temporaryDirectory, { force: true, recursive: true });
  }
}

async function run(
  command: readonly string[],
  cwd: string,
): Promise<CommandResult> {
  const child = Bun.spawn([...command], {
    cwd,
    env: process.env,
    stdin: "ignore",
    stderr: "pipe",
    stdout: "pipe",
  });
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  return { exitCode, stderr, stdout };
}

function assertCommand(result: CommandResult, action: string): CommandResult {
  if (result.exitCode !== 0) {
    throw new Error(
      `Failed to ${action} (exit ${result.exitCode}): ${result.stderr.trim() || result.stdout.trim()}`,
    );
  }
  return result;
}

if (import.meta.main) {
  const archivePath = process.argv
    .slice(2)
    .find((argument) => argument !== "--");
  if (!archivePath) {
    console.error("Usage: bun smokeReleasePackage.ts <release.tgz>");
    process.exitCode = 2;
  } else {
    await smokeReleasePackage(archivePath);
    console.log(`Release smoke passed: ${resolve(archivePath)}`);
  }
}
