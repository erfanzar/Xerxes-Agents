// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { chmod, mkdir, mkdtemp, readFile, realpath, rm, writeFile } from "node:fs/promises";
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
      HOME: temporaryRoot,
      SHELL: "/bin/sh",
      XERXES_BIN_DIRECTORY: binDirectory,
      XERXES_SOURCE_DIRECTORY: ".",
    });
    expect(installed.exitCode, installed.stderr).toBe(0);
    expect(installed.stdout).toContain("Xerxes Bun runtime is ready");

    const canonicalBinDirectory = await realpath(binDirectory);
    const discovered = await execute(
      ["sh", "-c", '. "$HOME/.profile"; command -v xerxes'],
      {
        HOME: temporaryRoot,
        PATH: process.env.PATH ?? "/usr/bin:/bin",
        SHELL: "/bin/sh",
      },
    );
    expect(discovered.exitCode, discovered.stderr).toBe(0);
    expect(discovered.stdout.trim()).toBe(join(canonicalBinDirectory, "xerxes"));

    const launcher = await execute([join(binDirectory, "xerxes"), "--version"]);
    expect(launcher.exitCode, launcher.stderr).toBe(0);
    expect(launcher.stdout.trim()).toBe("0.3.0");

    const acpLauncher = await Bun.file(join(binDirectory, "xerxes-acp")).text();
    expect(acpLauncher).toContain("/xerxes/dist/cli.js' acp \"$@\"");
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

test("native installer persists an idempotent PATH entry for common user shells", async () => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "xerxes-shell-paths-"));
  try {
    const zshHome = join(temporaryRoot, "zsh-home");
    const zDotDirectory = join(zshHome, "z-dot");
    const firstZshBin = join(zshHome, "first bin");
    await mkdir(zshHome, { recursive: true });
    const firstZsh = await configureInstallerPath({
      HOME: zshHome,
      SHELL: "/bin/zsh",
      XERXES_BIN_DIRECTORY: firstZshBin,
      ZDOTDIR: zDotDirectory,
    });
    expect(firstZsh.exitCode, firstZsh.stderr).toBe(0);
    const zshrc = join(zDotDirectory, ".zshrc");
    const firstZshrc = await readFile(zshrc, "utf8");
    expect(markerCount(firstZshrc)).toBe(1);
    const canonicalFirstZshBin = await realpath(firstZshBin);
    const prioritized = await execute(
      ["sh", "-c", '. "$ZSHRC"; printf "%s" "$PATH"'],
      {
        PATH: `/tmp/older-xerxes:${canonicalFirstZshBin}:/usr/bin:/bin`,
        ZSHRC: zshrc,
      },
    );
    expect(prioritized.exitCode, prioritized.stderr).toBe(0);
    expect(prioritized.stdout.startsWith(`${canonicalFirstZshBin}:`)).toBeTrue();

    const repeatedZsh = await configureInstallerPath({
      HOME: zshHome,
      SHELL: "/bin/zsh",
      XERXES_BIN_DIRECTORY: firstZshBin,
      ZDOTDIR: zDotDirectory,
    });
    expect(repeatedZsh.exitCode, repeatedZsh.stderr).toBe(0);
    expect(await readFile(zshrc, "utf8")).toBe(firstZshrc);

    const secondZshBin = join(zshHome, "replacement bin");
    const changedZsh = await configureInstallerPath({
      HOME: zshHome,
      SHELL: "/bin/zsh",
      XERXES_BIN_DIRECTORY: secondZshBin,
      ZDOTDIR: zDotDirectory,
    });
    expect(changedZsh.exitCode, changedZsh.stderr).toBe(0);
    const changedZshrc = await readFile(zshrc, "utf8");
    expect(markerCount(changedZshrc)).toBe(1);
    expect(changedZshrc).not.toContain(await realpath(firstZshBin));
    expect(changedZshrc).toContain(await realpath(secondZshBin));

    const bashHome = join(temporaryRoot, "bash-home");
    const bashProfile = join(bashHome, ".bash_profile");
    await mkdir(bashHome, { recursive: true });
    await writeFile(bashProfile, "export KEEP_BASH_SETTING=1\n", "utf8");
    const bash = await configureInstallerPath({
      HOME: bashHome,
      SHELL: "/bin/bash",
      XERXES_BIN_DIRECTORY: join(bashHome, "bin"),
    });
    expect(bash.exitCode, bash.stderr).toBe(0);
    expect(markerCount(await readFile(join(bashHome, ".bashrc"), "utf8"))).toBe(1);
    const bashProfileText = await readFile(bashProfile, "utf8");
    expect(bashProfileText).toContain("export KEEP_BASH_SETTING=1");
    expect(markerCount(bashProfileText)).toBe(1);

    const fishHome = join(temporaryRoot, "fish-home");
    const fishConfigHome = join(fishHome, "config");
    await mkdir(fishHome, { recursive: true });
    const fish = await configureInstallerPath({
      HOME: fishHome,
      SHELL: "/usr/bin/fish",
      XDG_CONFIG_HOME: fishConfigHome,
      XERXES_BIN_DIRECTORY: join(fishHome, "bin ' slash\\"),
    });
    expect(fish.exitCode, fish.stderr).toBe(0);
    const fishConfig = await readFile(
      join(fishConfigHome, "fish", "conf.d", "xerxes.fish"),
      "utf8",
    );
    expect(markerCount(fishConfig)).toBe(1);
    expect(fishConfig).toContain("set -gx PATH");
    expect(fishConfig).not.toContain("export PATH");
    expect(fishConfig).toContain("\\'");
    expect(fishConfig).toContain("\\\\");
  } finally {
    await rm(temporaryRoot, { force: true, recursive: true });
  }
});

test("native installer shell-quotes custom launcher and PATH directories", async () => {
  const temporaryHome = await mkdtemp(join(tmpdir(), "xerxes-quoted-path-"));
  try {
    const binDirectory = join(temporaryHome, "bin ' quote $(touch SHOULD_NOT_EXIST)");
    const fakeSource = join(temporaryHome, "source ' quote $(touch SHOULD_NOT_EXIST)");
    const configured = await execute(
      [
        "sh",
        "-c",
        '. "$INSTALLER_PATH"; prepare_bin_directory; write_launcher "$XERXES_SOURCE_DIRECTORY" xerxes; persist_bin_path',
      ],
      {
        HOME: temporaryHome,
        INSTALLER_PATH: join(PROJECT_ROOT, "scripts", "install.sh"),
        SHELL: "/bin/sh",
        XERXES_BIN_DIRECTORY: binDirectory,
        XERXES_INSTALLER_SOURCE_ONLY: "1",
        XERXES_SOURCE_DIRECTORY: fakeSource,
      },
    );
    expect(configured.exitCode, configured.stderr).toBe(0);

    const canonicalBinDirectory = await realpath(binDirectory);
    const launcher = join(canonicalBinDirectory, "xerxes");
    const launcherSyntax = await execute(["sh", "-n", launcher]);
    expect(launcherSyntax.exitCode, launcherSyntax.stderr).toBe(0);
    await chmod(launcher, 0o755);

    const invoked = await execute(
      ["sh", "-c", 'cd "$HOME"; "$LAUNCHER" --version'],
      {
        HOME: temporaryHome,
        LAUNCHER: launcher,
        PATH: process.env.PATH ?? "/usr/bin:/bin",
      },
    );
    expect(invoked.exitCode).not.toBe(0);

    const discovered = await execute(
      ["sh", "-c", 'cd "$HOME"; . "$HOME/.profile"; command -v xerxes'],
      {
        HOME: temporaryHome,
        PATH: "/usr/bin:/bin",
        SHELL: "/bin/sh",
      },
    );
    expect(discovered.exitCode, discovered.stderr).toBe(0);
    expect(discovered.stdout.trim()).toBe(launcher);
    expect(await Bun.file(join(temporaryHome, "SHOULD_NOT_EXIST")).exists()).toBeFalse();
  } finally {
    await rm(temporaryHome, { force: true, recursive: true });
  }
});

test("native installer rejects launcher directories unsafe for PATH", async () => {
  const temporaryHome = await mkdtemp(join(tmpdir(), "xerxes-invalid-bin-"));
  try {
    const invalidDirectories = [
      "relative/bin",
      `${temporaryHome}/colon:bin`,
      `${temporaryHome}/line\nbreak`,
      `${temporaryHome}/carriage\rreturn`,
    ];
    for (const binDirectory of invalidDirectories) {
      const result = await execute(
        ["sh", "-c", '. "$INSTALLER_PATH"; prepare_bin_directory'],
        {
          HOME: temporaryHome,
          INSTALLER_PATH: join(PROJECT_ROOT, "scripts", "install.sh"),
          SHELL: "/bin/sh",
          XERXES_BIN_DIRECTORY: binDirectory,
          XERXES_INSTALLER_SOURCE_ONLY: "1",
        },
      );
      expect(result.exitCode).not.toBe(0);
      expect(result.stderr).toContain("XERXES_BIN_DIRECTORY");
    }
  } finally {
    await rm(temporaryHome, { force: true, recursive: true });
  }
});

test("remote installer clones and safely fast-forwards its managed checkout", async () => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "xerxes-managed-installer-"));
  try {
    const seed = join(temporaryRoot, "seed");
    const origin = join(temporaryRoot, "origin.git");
    const otherOrigin = join(temporaryRoot, "other.git");
    const installDirectory = join(temporaryRoot, "managed");
    await mkdir(join(seed, "xerxes"), { recursive: true });
    await writeFile(join(seed, "package.json"), '{}\n', "utf8");
    await writeFile(join(seed, "bun.lock"), "fixture lock\n", "utf8");
    await writeFile(join(seed, "xerxes", "revision.txt"), "first\n", "utf8");
    await git(["init", "-b", "main"], seed);
    await git(["config", "user.email", "installer-test@xerxes.invalid"], seed);
    await git(["config", "user.name", "Xerxes Installer Test"], seed);
    await git(["add", "."], seed);
    await git(["commit", "-m", "initial fixture"], seed);
    await git(["clone", "--bare", seed, origin], temporaryRoot);
    await git(["clone", "--bare", seed, otherOrigin], temporaryRoot);
    await git(["remote", "add", "origin", `file://${origin}`], seed);

    const additions = {
      XERXES_INSTALLER_SOURCE_ONLY: "1",
      XERXES_INSTALL_DIRECTORY: installDirectory,
      XERXES_REPOSITORY_URL: `file://${origin}`,
    };
    const cloned = await resolveRemoteSource(additions);
    const canonicalInstallDirectory = await realpath(installDirectory);
    expect(cloned.exitCode, cloned.stderr).toBe(0);
    expect(cloned.stdout).toBe(`${canonicalInstallDirectory}\n`);
    expect(cloned.stderr).toContain("cloning native Bun source");
    expect(await readFile(join(installDirectory, "xerxes", "revision.txt"), "utf8")).toBe(
      "first\n",
    );

    await git(["remote", "add", "other", `file://${otherOrigin}`], installDirectory);
    await git(["fetch", "other"], installDirectory);
    await git(["branch", "--set-upstream-to", "other/main", "main"], installDirectory);

    await writeFile(join(seed, "xerxes", "revision.txt"), "second\n", "utf8");
    await git(["add", "."], seed);
    await git(["commit", "-m", "advance fixture"], seed);
    await git(["push", "origin", "main"], seed);

    const updated = await resolveRemoteSource(additions);
    expect(updated.exitCode, updated.stderr).toBe(0);
    expect(updated.stdout).toBe(`${canonicalInstallDirectory}\n`);
    expect(updated.stderr).toContain("updating native Bun source");
    expect(await gitOutput(["rev-parse", "HEAD"], installDirectory)).toBe(
      await gitOutput(["rev-parse", "refs/remotes/origin/main"], installDirectory),
    );
    expect(await gitOutput(["rev-parse", "HEAD"], installDirectory)).not.toBe(
      await gitOutput(["rev-parse", "refs/remotes/other/main"], installDirectory),
    );
    expect(await readFile(join(installDirectory, "xerxes", "revision.txt"), "utf8")).toBe(
      "second\n",
    );

    await writeFile(join(installDirectory, "xerxes", "revision.txt"), "local edit\n", "utf8");
    const dirty = await resolveRemoteSource(additions);
    expect(dirty.exitCode).not.toBe(0);
    expect(dirty.stdout).toBe("");
    expect(dirty.stderr).toContain("has local changes; refusing to update");
  } finally {
    await rm(temporaryRoot, { force: true, recursive: true });
  }
}, 30_000);

test("remote installer refuses an unrelated existing directory", async () => {
  const temporaryRoot = await mkdtemp(join(tmpdir(), "xerxes-invalid-installer-"));
  try {
    const installDirectory = join(temporaryRoot, "managed");
    await mkdir(installDirectory);
    const result = await resolveRemoteSource({
      XERXES_INSTALLER_SOURCE_ONLY: "1",
      XERXES_INSTALL_DIRECTORY: installDirectory,
    });
    expect(result.exitCode).not.toBe(0);
    expect(result.stdout).toBe("");
    expect(result.stderr).toContain("is not a managed Git checkout");
  } finally {
    await rm(temporaryRoot, { force: true, recursive: true });
  }
});

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

test("native installer warns when an earlier build is still running", async () => {
  const cliEntry = join(PROJECT_ROOT, "xerxes", "dist", "cli.js");
  const uiEntry = join(PROJECT_ROOT, "xerxes", "dist", "ui", "entry.js");
  const running = await execute(
    [
      "sh",
      "-c",
      '. "$INSTALLER_PATH"; warn_running_xerxes_processes "$SOURCE_ROOT"',
    ],
    {
      INSTALLER_PATH: join(PROJECT_ROOT, "scripts", "install.sh"),
      SOURCE_ROOT: PROJECT_ROOT,
      XERXES_INSTALLER_PROCESS_LIST: [
        `101 bun ${cliEntry} daemon --project-dir /tmp/project`,
        `102 bun ${uiEntry}`,
        "103 bun /tmp/unrelated.js",
      ].join("\n"),
      XERXES_INSTALLER_SOURCE_ONLY: "1",
    },
  );

  expect(running.exitCode, running.stderr).toBe(0);
  expect(running.stdout).toBe("");
  expect(running.stderr).toContain(
    "2 running Xerxes process(es) still have the previous build loaded",
  );
  expect(running.stderr).toContain("leaves active sessions running");

  const idle = await execute(
    [
      "sh",
      "-c",
      '. "$INSTALLER_PATH"; warn_running_xerxes_processes "$SOURCE_ROOT"',
    ],
    {
      INSTALLER_PATH: join(PROJECT_ROOT, "scripts", "install.sh"),
      SOURCE_ROOT: PROJECT_ROOT,
      XERXES_INSTALLER_PROCESS_LIST: "103 bun /tmp/unrelated.js",
      XERXES_INSTALLER_SOURCE_ONLY: "1",
    },
  );
  expect(idle.exitCode, idle.stderr).toBe(0);
  expect(idle.stderr).toBe("");
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

async function resolveRemoteSource(
  additions: Readonly<Record<string, string>>,
): Promise<Awaited<ReturnType<typeof execute>>> {
  return execute(
    [
      "sh",
      "-c",
      '. "$INSTALLER_PATH"; resolve_source',
    ],
    {
      ...additions,
      INSTALLER_PATH: join(PROJECT_ROOT, "scripts", "install.sh"),
    },
  );
}

async function configureInstallerPath(
  additions: Readonly<Record<string, string>>,
): Promise<Awaited<ReturnType<typeof execute>>> {
  return execute(
    [
      "sh",
      "-c",
      '. "$INSTALLER_PATH"; prepare_bin_directory; persist_bin_path',
    ],
    {
      ...additions,
      INSTALLER_PATH: join(PROJECT_ROOT, "scripts", "install.sh"),
      XERXES_INSTALLER_SOURCE_ONLY: "1",
    },
  );
}

function markerCount(content: string): number {
  return content.split("# >>> xerxes PATH >>>").length - 1;
}

async function git(arguments_: readonly string[], cwd: string): Promise<void> {
  const result = await executeIn(["git", ...arguments_], cwd);
  expect(result.exitCode, result.stderr).toBe(0);
}

async function gitOutput(arguments_: readonly string[], cwd: string): Promise<string> {
  const result = await executeIn(["git", ...arguments_], cwd);
  expect(result.exitCode, result.stderr).toBe(0);
  return result.stdout.trim();
}

async function executeIn(
  command: readonly string[],
  cwd: string,
): Promise<Awaited<ReturnType<typeof execute>>> {
  const child = Bun.spawn([...command], {
    cwd,
    env: process.env,
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
