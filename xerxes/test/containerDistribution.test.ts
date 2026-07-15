// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { join, resolve } from "node:path";

const REPOSITORY_ROOT = resolve(import.meta.dir, "../..");

test("Compose keeps the daemon authenticated and loopback-only", async () => {
  const compose = await Bun.file(
    join(REPOSITORY_ROOT, "docker-compose.yml"),
  ).text();

  expect(compose).toContain(
    'XERXES_DAEMON_TOKEN: "${XERXES_DAEMON_TOKEN:?',
  );
  expect(compose).toContain('"127.0.0.1:11996:11996"');
  expect(compose).not.toContain('- "11996:11996"');
});

test("the container account can match a Linux bind-mount owner without root", async () => {
  const dockerfile = await Bun.file(join(REPOSITORY_ROOT, "Dockerfile")).text();
  const compose = await Bun.file(
    join(REPOSITORY_ROOT, "docker-compose.yml"),
  ).text();

  expect(dockerfile).toContain("ARG XERXES_UID=1000");
  expect(dockerfile).toContain('usermod --login xerxes --uid "$XERXES_UID"');
  expect(dockerfile).toContain("USER xerxes");
  expect(dockerfile).toContain('/workspace');
  expect(compose).toContain('XERXES_UID: "${XERXES_UID:-1000}"');
});

test("the production image keeps the built-in agent profiles beside the bundled CLI", async () => {
  const dockerfile = await Bun.file(join(REPOSITORY_ROOT, "Dockerfile")).text();

  expect(dockerfile).toContain(
    "/app/xerxes/dist/default ./xerxes/dist/default",
  );
  expect(dockerfile).toContain(
    "/app/xerxes/dist/skills ./xerxes/dist/skills",
  );
});

test("the production image installs only the locked OpenTUI runtime graph", async () => {
  const dockerfile = await Bun.file(join(REPOSITORY_ROOT, "Dockerfile")).text();
  const runtimeManifest = await Bun.file(
    join(REPOSITORY_ROOT, "packaging/runtime/package.json"),
  ).json();
  const sourceManifest = await Bun.file(
    join(REPOSITORY_ROOT, "xerxes/package.json"),
  ).json();
  const runtimeDependencies = runtimeManifest.dependencies as Record<
    string,
    string
  >;
  const sourceDependencies = sourceManifest.dependencies as Record<
    string,
    string
  >;

  expect(dockerfile).toContain(
    "COPY packaging/runtime/package.json packaging/runtime/bun.lock ./",
  );
  expect(dockerfile).not.toContain(
    "COPY --from=build /app/xerxes/package.json",
  );
  expect(dockerfile).toContain(
    "COPY xerxes/package.json xerxes/package.json",
  );
  expect(Object.keys(runtimeDependencies)).toEqual([
    "@opentui/core",
    "@opentui/react",
    "react",
  ]);
  expect(runtimeDependencies["@opentui/core"]).toBe(
    sourceDependencies["@opentui/core"],
  );
  expect(runtimeDependencies["@opentui/react"]).toBe(
    sourceDependencies["@opentui/react"],
  );
  expect(runtimeDependencies.react).toBe(sourceDependencies.react);
  expect(runtimeManifest.version).toBe(sourceManifest.version);
  expect(runtimeManifest.devDependencies).toBeUndefined();
});

test("CI validates container workspace writes with authenticated Compose config", async () => {
  const workflow = await Bun.file(
    join(REPOSITORY_ROOT, ".github/workflows/bun-ci.yml"),
  ).text();

  expect(workflow).toContain(
    "XERXES_DAEMON_TOKEN=ci-compose-config-only docker compose config",
  );
  expect(workflow).toContain('--build-arg XERXES_UID="$(id -u)"');
  expect(workflow).toContain("test -w /workspace");
  expect(workflow).toContain(".xerxes-write-probe");
});
