// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

const PROJECT_ROOT = resolve(import.meta.dir, "../..");

test("native installer validates as shell and identifies a local Bun checkout", async () => {
  const syntax = await execute(["sh", "-n", "scripts/install.sh"]);
  expect(syntax.exitCode).toBe(0);
  expect(syntax.stderr).toBe("");

  const sourceOnly = await execute(
    [
      "sh",
      "-c",
      ". ./scripts/install.sh; local_checkout_root ./scripts/install.sh",
    ],
    { XERXES_INSTALLER_SOURCE_ONLY: "1" },
  );
  expect(sourceOnly.exitCode).toBe(0);
  expect(sourceOnly.stdout.trim()).toBe(PROJECT_ROOT);
  expect(sourceOnly.stderr).toBe("");

  const source = await Bun.file(
    join(PROJECT_ROOT, "scripts", "install.sh"),
  ).text();
  expect(source).not.toContain("python");
  expect(source).not.toContain("uv ");
  expect(source).toContain("bun install --frozen-lockfile");
  expect(source).toContain("bun run build");
  expect(source).toContain("xerxes/dist/cli.js");
});

test("native installer writes Bun and ACP launchers against an explicit local source", async () => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "xerxes-bun-installer-"));
  try {
    const binDirectory = join(temporaryRoot, "bin");
    const installed = await execute(["sh", "scripts/install.sh"], {
      XERXES_BIN_DIRECTORY: binDirectory,
      XERXES_SOURCE_DIRECTORY: ".",
    });
    expect(installed.exitCode, installed.stderr).toBe(0);
    expect(installed.stdout).toContain("Xerxes Bun runtime is ready");

    const launcher = await execute([join(binDirectory, "xerxes"), "--version"]);
    expect(launcher.exitCode, launcher.stderr).toBe(0);
    expect(launcher.stdout.trim()).toBe("0.3.0");

    const acpLauncher = await Bun.file(join(binDirectory, "xerxes-acp")).text();
    expect(acpLauncher).toContain('cli.js" acp');
    expect(
      await Bun.file(join(binDirectory, "xerxes-acp")).exists(),
    ).toBeTrue();

    const registryDirectory = join(temporaryRoot, "registry");
    const registry = await execute(
      [join(binDirectory, "xerxes-acp"), "--write-registry"],
      {
        XDG_CONFIG_HOME: registryDirectory,
      },
    );
    expect(registry.exitCode, registry.stderr).toBe(0);
    expect(registry.stdout).toContain("Wrote ACP registry manifest:");
    const manifest = JSON.parse(
      await Bun.file(
        join(registryDirectory, "agent-registry", "xerxes", "agent.json"),
      ).text(),
    ) as Record<string, unknown>;
    expect(manifest.distribution).toEqual({
      type: "command",
      command: "xerxes-acp",
      args: [],
    });
  } finally {
    await rm(temporaryRoot, { force: true, recursive: true });
  }
}, 30_000);

test("native installer removes the retired Xerxes alias without changing other shell settings", async () => {
  const temporaryHome = await mkdtemp(join(tmpdir(), "xerxes-shell-home-"));
  const zshrc = join(temporaryHome, ".zshrc");
  try {
    await writeFile(
      zshrc,
      [
        "export KEEP_BEFORE=1",
        "# >>> xerxes installer >>>",
        `alias xerxes="${temporaryHome}/.xerxes-venv/bin/xerxes"`,
        "# <<< xerxes installer <<<",
        "export KEEP_AFTER=1",
        "",
      ].join("\n"),
      "utf8",
    );

    const migrated = await execute(
      ["sh", "-c", ". ./scripts/install.sh; remove_legacy_xerxes_aliases"],
      {
        HOME: temporaryHome,
        XERXES_INSTALLER_SOURCE_ONLY: "1",
      },
    );

    expect(migrated.exitCode, migrated.stderr).toBe(0);
    expect(migrated.stdout).toContain("removed retired Xerxes alias");
    expect(await readFile(zshrc, "utf8")).toBe(
      "export KEEP_BEFORE=1\nexport KEEP_AFTER=1\n",
    );
  } finally {
    await rm(temporaryHome, { force: true, recursive: true });
  }
});

async function execute(
  command: readonly string[],
  additions: Readonly<Record<string, string>> = {},
): Promise<{
  readonly exitCode: number;
  readonly stderr: string;
  readonly stdout: string;
}> {
  const child = Bun.spawn([...command], {
    cwd: PROJECT_ROOT,
    env: { ...process.env, ...additions },
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
