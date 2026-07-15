// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { afterEach, describe, expect, test } from "bun:test";
import {
  chmod,
  cp,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  stat,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  dryRunReleasePublish,
  parseReleasePackageCommand,
  prepareReleasePackage,
  RELEASE_PACKAGE_MINIMUM_BUN_VERSION,
  RELEASE_PACKAGE_NAME,
  RELEASE_REQUIRED_TUI_DEPENDENCIES,
  validateReleasePackage,
} from "../scripts/releasePackage.js";
import {
  resolveBundledSkillsDirectory,
  SkillRegistry,
} from "../src/extensions/skills.js";

const temporaryDirectories: string[] = [];

afterEach(async () => {
  await Promise.all(
    temporaryDirectories
      .splice(0)
      .map((directory) => rm(directory, { force: true, recursive: true })),
  );
});

describe("native release package staging", () => {
  test("stages a locked Bun package with an executable launcher and integrity manifest", async () => {
    const repositoryRoot = await releaseFixture();
    const packageDirectory = join(repositoryRoot, ".release");

    const prepared = await prepareReleasePackage({
      expectedVersion: "v1.2.3",
      outputDirectory: packageDirectory,
      repositoryRoot,
    });
    const validated = await validateReleasePackage({
      expectedVersion: "1.2.3",
      packageDirectory,
    });

    expect(prepared.version).toBe("1.2.3");
    expect(validated.files.map((file) => file.path)).toEqual([
      "bin/default/agent.yaml",
      "bin/default/coder.yaml",
      "bin/default/objective.yaml",
      "bin/default/planner.yaml",
      "bin/default/researcher.yaml",
      "bin/default/reviewer.yaml",
      "bin/default/system.md",
      "bin/default/tester.yaml",
      "bin/xerxes",
      "bin/xerxes-acp",
      "bin/xerxes-bun",
      "bin/xerxes.js",
      "LICENSE",
      "package.json",
      "README.md",
      "skills/fixture-skill/references/guide.md",
      "skills/fixture-skill/SKILL.md",
      "THIRD_PARTY_NOTICES.md",
      "ui/entry.js",
    ]);
    expect(
      (await stat(join(packageDirectory, "bin/xerxes"))).mode & 0o111,
    ).not.toBe(0);
    expect(
      (await stat(join(packageDirectory, "bin/xerxes-acp"))).mode & 0o111,
    ).not.toBe(0);
    expect(
      (await stat(join(packageDirectory, "bin/xerxes-bun"))).mode & 0o111,
    ).not.toBe(0);
    expect(
      await readFile(join(packageDirectory, "bin/xerxes-bun"), "utf8"),
    ).toBe(await readFile(join(packageDirectory, "bin/xerxes"), "utf8"));
    expect(
      await readFile(join(packageDirectory, "bin/xerxes-acp"), "utf8"),
    ).toContain("process.argv.splice(2, 0, 'acp')");
    const acp = Bun.spawn(
      [join(packageDirectory, "bin/xerxes-acp"), "--write-registry"],
      {
        stderr: "pipe",
        stdout: "pipe",
      },
    );
    expect(await new Response(acp.stderr).text()).toBe("");
    expect(await new Response(acp.stdout).text()).toBe(
      '["acp","--write-registry"]\n',
    );
    expect(await acp.exited).toBe(0);
    const metadata = JSON.parse(
      await readFile(join(packageDirectory, "package.json"), "utf8"),
    ) as Record<string, unknown>;
    expect(metadata).toMatchObject({
      bin: {
        xerxes: "./bin/xerxes",
        "xerxes-acp": "./bin/xerxes-acp",
        "xerxes-bun": "./bin/xerxes-bun",
      },
      bugs: {
        url: "https://github.com/erfanzar/Xerxes-Agents/issues",
      },
      dependencies: {
        "@opentui/core": "0.4.3",
        "@opentui/react": "0.4.3",
        react: "^19.2.4",
      },
      engines: { bun: `>=${RELEASE_PACKAGE_MINIMUM_BUN_VERSION}` },
      files: [
        "bin",
        "skills",
        "ui",
        "LICENSE",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "release-manifest.json",
      ],
      homepage: "https://github.com/erfanzar/Xerxes-Agents#readme",
      keywords: [
        "ai",
        "agent",
        "coding-agent",
        "multi-agent",
        "bun",
        "typescript",
        "terminal",
        "tui",
        "opentui",
        "mcp",
      ],
      name: RELEASE_PACKAGE_NAME,
      private: false,
      publishConfig: { access: "public" },
      repository: {
        type: "git",
        url: "git+https://github.com/erfanzar/Xerxes-Agents.git",
      },
      type: "module",
      version: "1.2.3",
    });
    expect(
      Object.keys(metadata.dependencies as Record<string, string>),
    ).toEqual([...RELEASE_REQUIRED_TUI_DEPENDENCIES]);
    expect(
      await readFile(
        join(packageDirectory, "skills/fixture-skill/references/guide.md"),
        "utf8",
      ),
    ).toBe("Bundled release skill asset.\n");

    const packagedSkillsDirectory = resolveBundledSkillsDirectory({
      moduleDirectory: join(packageDirectory, "bin"),
    });
    expect(packagedSkillsDirectory).toBe(join(packageDirectory, "skills"));
    const skills = new SkillRegistry();
    expect(await skills.discover(packagedSkillsDirectory)).toEqual([
      "fixture-skill",
    ]);
  });

  test("rejects mismatched versions, nonempty outputs, and tampered artifacts", async () => {
    const repositoryRoot = await releaseFixture();
    const packageDirectory = join(repositoryRoot, ".release");

    await expect(
      prepareReleasePackage({
        outputDirectory: join(repositoryRoot, "xerxes"),
        repositoryRoot,
      }),
    ).rejects.toThrow("cannot contain source or metadata");

    await expect(
      prepareReleasePackage({
        expectedVersion: "2.0.0",
        outputDirectory: packageDirectory,
        repositoryRoot,
      }),
    ).rejects.toThrow("does not match expected");

    await mkdir(packageDirectory, { recursive: true });
    await writeFile(join(packageDirectory, "occupied.txt"), "occupied", "utf8");
    await expect(
      prepareReleasePackage({
        outputDirectory: packageDirectory,
        repositoryRoot,
      }),
    ).rejects.toThrow("must be empty");

    await rm(packageDirectory, { force: true, recursive: true });
    await prepareReleasePackage({
      outputDirectory: packageDirectory,
      repositoryRoot,
    });
    await writeFile(
      join(packageDirectory, "skills/fixture-skill/references/guide.md"),
      "tampered",
      "utf8",
    );
    await expect(validateReleasePackage({ packageDirectory })).rejects.toThrow(
      "integrity mismatch",
    );
  });

  test("rejects an unmanifested bundled skill asset", async () => {
    const repositoryRoot = await releaseFixture();
    const packageDirectory = join(repositoryRoot, ".release");
    await prepareReleasePackage({
      outputDirectory: packageDirectory,
      repositoryRoot,
    });
    await writeFile(
      join(packageDirectory, "skills/fixture-skill/extra.md"),
      "unexpected",
      "utf8",
    );

    await expect(validateReleasePackage({ packageDirectory })).rejects.toThrow(
      "missing required file",
    );
  });

  test("verifies every packed archive file against the staged release", async () => {
    const repositoryRoot = await releaseFixture();
    const packageDirectory = join(repositoryRoot, ".release");
    const archivePath = join(repositoryRoot, "xerxes-valid.tgz");
    await prepareReleasePackage({
      outputDirectory: packageDirectory,
      repositoryRoot,
    });
    await packReleaseDirectory(packageDirectory, archivePath);

    await expect(
      validateReleasePackage({ archivePath, packageDirectory }),
    ).resolves.toMatchObject({ version: "1.2.3" });
  });

  test("safely dry-runs Bun publish for an exactly validated release archive", async () => {
    const repositoryRoot = await releaseFixture();
    const packageDirectory = join(repositoryRoot, ".release");
    const archivePath = join(repositoryRoot, "xerxes-publish-dry-run.tgz");
    await prepareReleasePackage({
      outputDirectory: packageDirectory,
      repositoryRoot,
    });
    await packReleaseDirectory(packageDirectory, archivePath);

    const report = await dryRunReleasePublish({
      archivePath,
      expectedVersion: "v1.2.3",
      packageDirectory,
    });

    expect(report).toMatchObject({
      packageDirectory,
      version: "1.2.3",
    });
    expect(report.output).toContain(`${RELEASE_PACKAGE_NAME}@1.2.3`);
  });

  test("rejects packed archives with tampered, missing, or unexpected files", async () => {
    const repositoryRoot = await releaseFixture();
    const packageDirectory = join(repositoryRoot, ".release");
    await prepareReleasePackage({
      outputDirectory: packageDirectory,
      repositoryRoot,
    });

    const tamperedDirectory = join(repositoryRoot, ".release-tampered");
    const tamperedArchive = join(repositoryRoot, "xerxes-tampered.tgz");
    await cp(packageDirectory, tamperedDirectory, { recursive: true });
    await writeFile(
      join(tamperedDirectory, "ui/entry.js"),
      "tampered\n",
      "utf8",
    );
    await packReleaseDirectory(tamperedDirectory, tamperedArchive);
    await expect(
      validateReleasePackage({
        archivePath: tamperedArchive,
        packageDirectory,
      }),
    ).rejects.toThrow(
      "Packed release archive file integrity mismatch: ui/entry.js",
    );

    const missingDirectory = join(repositoryRoot, ".release-missing");
    const missingArchive = join(repositoryRoot, "xerxes-missing.tgz");
    await cp(packageDirectory, missingDirectory, { recursive: true });
    await rm(join(missingDirectory, "README.md"));
    await packReleaseDirectory(missingDirectory, missingArchive);
    await expect(
      validateReleasePackage({
        archivePath: missingArchive,
        packageDirectory,
      }),
    ).rejects.toThrow(
      "Packed release archive is missing required file: README.md",
    );

    const unexpectedDirectory = join(repositoryRoot, ".release-unexpected");
    const unexpectedArchive = join(repositoryRoot, "xerxes-unexpected.tgz");
    await cp(packageDirectory, unexpectedDirectory, { recursive: true });
    await writeFile(
      join(unexpectedDirectory, "ui/extra.js"),
      "extra\n",
      "utf8",
    );
    await packReleaseDirectory(unexpectedDirectory, unexpectedArchive);
    await expect(
      validateReleasePackage({
        archivePath: unexpectedArchive,
        packageDirectory,
      }),
    ).rejects.toThrow(
      "Packed release archive contains unexpected file: ui/extra.js",
    );

    const unrelatedArchive = join(repositoryRoot, "xerxes-unrelated.tgz");
    await writeFile(unrelatedArchive, "not a gzip archive\n", "utf8");
    await expect(
      validateReleasePackage({
        archivePath: unrelatedArchive,
        packageDirectory,
      }),
    ).rejects.toThrow("Packed release archive is not a valid gzip stream");
  });

  test("rejects non-redistributable bundled skill assets", async () => {
    const forbiddenAssets = [
      "creative-ideation/SKILL.md",
      "powerpoint/LICENSE.txt",
      "research/research-paper-writing/templates/README.md",
      "training/axolotl/references/api.md",
      "training/pytorch-fsdp/references/index.md",
      "training/unsloth/references/llms-full.md",
    ];

    for (const forbiddenAsset of forbiddenAssets) {
      const repositoryRoot = await releaseFixture();
      const assetPath = join(
        repositoryRoot,
        "xerxes/dist/skills",
        forbiddenAsset,
      );
      await mkdir(join(assetPath, ".."), { recursive: true });
      await writeFile(assetPath, "not redistributable\n", "utf8");

      await expect(
        prepareReleasePackage({
          outputDirectory: join(repositoryRoot, ".release"),
          repositoryRoot,
        }),
      ).rejects.toThrow("non-redistributable bundled skill asset");
    }
  });

  test("accepts a clean-room single-file PowerPoint skill", async () => {
    const repositoryRoot = await releaseFixture();
    const skillPath = join(
      repositoryRoot,
      "xerxes/dist/skills/powerpoint/SKILL.md",
    );
    await mkdir(join(skillPath, ".."), { recursive: true });
    await writeFile(
      skillPath,
      "---\nname: powerpoint\ndescription: Native Bun PowerPoint tools\n---\nUse the native Xerxes PowerPoint commands.\n",
      "utf8",
    );

    const prepared = await prepareReleasePackage({
      outputDirectory: join(repositoryRoot, ".release"),
      repositoryRoot,
    });

    expect(prepared.files.map((file) => file.path)).toContain(
      "skills/powerpoint/SKILL.md",
    );
  });

  test("parses only explicit staging and checking commands", () => {
    expect(
      parseReleasePackageCommand([
        "prepare",
        "--output",
        "/tmp/release",
        "--expected-version",
        "v1.2.3",
      ]),
    ).toEqual({
      archivePath: undefined,
      expectedVersion: "v1.2.3",
      kind: "prepare",
      outputDirectory: "/tmp/release",
      packageDirectory: undefined,
    });
    expect(() =>
      parseReleasePackageCommand([
        "check",
        "--package",
        "/tmp/release",
        "--output",
        "/tmp/other",
      ]),
    ).toThrow("only accepts");
    expect(
      parseReleasePackageCommand([
        "publish-dry-run",
        "--package",
        "/tmp/release",
        "--archive",
        "/tmp/xerxes-bun.tgz",
      ]),
    ).toEqual({
      archivePath: "/tmp/xerxes-bun.tgz",
      expectedVersion: undefined,
      kind: "publish-dry-run",
      outputDirectory: undefined,
      packageDirectory: "/tmp/release",
    });
    expect(() =>
      parseReleasePackageCommand([
        "publish-dry-run",
        "--package",
        "/tmp/release",
      ]),
    ).toThrow("requires --archive");
  });
});

async function releaseFixture(): Promise<string> {
  const repositoryRoot = await mkdtemp(
    join(tmpdir(), "xerxes-release-package-"),
  );
  temporaryDirectories.push(repositoryRoot);
  await mkdir(join(repositoryRoot, "xerxes/dist"), { recursive: true });
  await mkdir(join(repositoryRoot, "xerxes/dist/default"), {
    recursive: true,
  });
  await mkdir(join(repositoryRoot, "xerxes/dist/ui"), {
    recursive: true,
  });
  await mkdir(
    join(repositoryRoot, "xerxes/dist/skills/fixture-skill/references"),
    { recursive: true },
  );

  const packageManifest = JSON.stringify(
    { name: "fixture", version: "1.2.3" },
    null,
    2,
  );
  const runtimePackageManifest = JSON.stringify(
    {
      dependencies: {
        "@opentui/core": "0.4.3",
        "@opentui/react": "0.4.3",
        react: "^19.2.4",
      },
      name: "fixture-runtime",
      version: "1.2.3",
    },
    null,
    2,
  );
  await Promise.all([
    writeFile(join(repositoryRoot, "package.json"), packageManifest, "utf8"),
    writeFile(
      join(repositoryRoot, "xerxes/package.json"),
      runtimePackageManifest,
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/cli.js"),
      "console.log(JSON.stringify(Bun.argv.slice(2)))\n",
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/default/agent.yaml"),
      "version: 1\nagent:\n  name: default\n  system_prompt_path: ./system.md\n",
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/default/coder.yaml"),
      "version: 1\nagent:\n  name: coder\n",
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/default/objective.yaml"),
      "version: 1\nagent:\n  name: objective\n",
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/default/planner.yaml"),
      "version: 1\nagent:\n  name: planner\n",
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/default/researcher.yaml"),
      "version: 1\nagent:\n  name: researcher\n",
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/default/reviewer.yaml"),
      "version: 1\nagent:\n  name: reviewer\n",
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/default/system.md"),
      "You are Xerxes.\n",
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/default/tester.yaml"),
      "version: 1\nagent:\n  name: tester\n",
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/skills/fixture-skill/SKILL.md"),
      "---\nname: fixture-skill\ndescription: Fixture release skill\n---\nUse the fixture.\n",
      "utf8",
    ),
    writeFile(
      join(
        repositoryRoot,
        "xerxes/dist/skills/fixture-skill/references/guide.md",
      ),
      "Bundled release skill asset.\n",
      "utf8",
    ),
    writeFile(
      join(repositoryRoot, "xerxes/dist/ui/entry.js"),
      'console.log("tui")\n',
      "utf8",
    ),
    writeFile(join(repositoryRoot, "LICENSE"), "Apache-2.0\n", "utf8"),
    writeFile(join(repositoryRoot, "README.md"), "# Xerxes\n", "utf8"),
    writeFile(
      join(repositoryRoot, "THIRD_PARTY_NOTICES.md"),
      "# Third-party notices\n",
      "utf8",
    ),
  ]);
  await chmod(join(repositoryRoot, "xerxes/dist/cli.js"), 0o644);
  return repositoryRoot;
}

async function packReleaseDirectory(
  packageDirectory: string,
  archivePath: string,
): Promise<void> {
  const child = Bun.spawn(
    [
      process.execPath,
      "pm",
      "pack",
      "--filename",
      archivePath,
      "--ignore-scripts",
    ],
    {
      cwd: packageDirectory,
      stderr: "pipe",
      stdout: "pipe",
    },
  );
  const [stdout, stderr, exitCode] = await Promise.all([
    new Response(child.stdout).text(),
    new Response(child.stderr).text(),
    child.exited,
  ]);
  if (exitCode !== 0) {
    throw new Error(
      `Failed to pack release fixture: ${stderr.trim() || stdout.trim()}`,
    );
  }
}
