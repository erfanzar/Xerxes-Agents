// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import {
  access,
  mkdir,
  mkdtemp,
  readFile,
  rm,
  writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  BunDocsBuildError,
  BUN_DOCS_BUILD_REPORT_FILENAME,
  BUN_DOCS_MANIFEST_FILENAME,
  buildBunDocsSite,
} from "../src/docs/siteBuilder.js";

test("native Bun docs builder renders Markdown and generated TypeScript API pages deterministically", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-bun-docs-"));
  const docsDirectory = join(root, "docs");
  const outputDirectory = join(docsDirectory, "output");
  try {
    await writeFixture(
      docsDirectory,
      "bun-docs.json",
      JSON.stringify(
        {
          outputDirectory: "output",
          sourceDirectory: ".",
          staticDirectory: "_static",
          title: "Fixture Xerxes",
          typescriptApi: {
            outputDirectory: "typescript-api",
            packageName: "@fixture/runtime",
            sourceDirectory: "../runtime",
          },
          version: 1,
        },
        null,
        2,
      ),
    );
    await writeFixture(
      docsDirectory,
      "index.md",
      [
        "# Fixture documentation",
        "",
        "[Guide](guide.md) · [Nested docs](nested/index.md)",
        "",
      ].join("\n"),
    );
    await writeFixture(
      docsDirectory,
      "guide.md",
      [
        "# Guide",
        "",
        "Read the [home](index.md) and keep **deterministic** output.",
        "",
        "- one",
        "- two",
        "",
      ].join("\n"),
    );
    await writeFixture(
      docsDirectory,
      "nested/index.md",
      "# Nested docs\n\n[Snippet](snippet.md)\n",
    );
    await writeFixture(
      docsDirectory,
      "nested/snippet.md",
      ["# Snippet", "", "```ts", "export const value = 1", "```", ""].join(
        "\n",
      ),
    );
    await writeFixture(
      docsDirectory,
      "obsolete.rst",
      ".. automodule:: obsolete.python.module\n",
    );
    await writeFixture(docsDirectory, "example.ipynb", "{}\n");
    await writeFixture(
      docsDirectory,
      "_static/custom.css",
      ".native { color: green; }\n",
    );
    await writeFixture(
      root,
      "runtime/index.ts",
      [
        "/** Root runtime contract. */",
        "export interface Runtime { readonly name: string }",
        'export const runtime: Runtime = { name: "fixture" }',
        "",
      ].join("\n"),
    );
    await writeFixture(
      root,
      "runtime/feature.ts",
      [
        "/** A feature exported by the fixture runtime. */",
        'export function feature(): string { return "ready" }',
        "",
      ].join("\n"),
    );

    const first = await buildBunDocsSite({
      configPath: join(docsDirectory, "bun-docs.json"),
    });
    expect(first.changed).toBeTrue();
    expect(first.typeScriptApiModules).toBe(2);
    expect(first.diagnostics).toEqual([
      {
        code: "unsupported-jupyter-notebook",
        level: "warning",
        message:
          "Jupyter notebooks are preserved as source but are not executed or rendered by the native Bun builder.",
        source: "example.ipynb",
      },
    ]);
    expect(
      first.documents.some(
        (document) => document.sourcePath === "obsolete.rst",
      ),
    ).toBeFalse();

    const index = await readFile(join(outputDirectory, "index.html"), "utf8");
    expect(index).toContain('href="guide.html"');
    expect(index).toContain('href="nested/index.html"');

    const guide = await readFile(join(outputDirectory, "guide.html"), "utf8");
    expect(guide).toContain('href="index.html"');
    expect(guide).toContain("<strong>deterministic</strong>");
    expect(guide).toContain("<li>one</li>");

    const nestedIndex = await readFile(
      join(outputDirectory, "nested/index.html"),
      "utf8",
    );
    expect(nestedIndex).toContain('href="snippet.html"');
    const snippet = await readFile(
      join(outputDirectory, "nested/snippet.html"),
      "utf8",
    );
    expect(snippet).toContain(
      '<code class="language-ts">export const value = 1</code>',
    );
    await expect(
      access(join(outputDirectory, "obsolete.html")),
    ).rejects.toThrow();

    const apiRoot = await readFile(
      join(outputDirectory, "typescript-api/index.html"),
      "utf8",
    );
    expect(apiRoot).toContain("@fixture/runtime API Reference");
    const apiModule = await readFile(
      join(outputDirectory, "typescript-api/index.api.html"),
      "utf8",
    );
    expect(apiModule).toContain("Root runtime contract.");
    expect(apiModule).toContain("<h3><code>runtime</code> (const)</h3>");

    expect(
      await readFile(join(outputDirectory, "assets/custom.css"), "utf8"),
    ).toBe(".native { color: green; }\n");
    const documentationReport = await readFile(
      join(outputDirectory, "documentation-report.html"),
      "utf8",
    );
    expect(documentationReport).toContain("Native Bun documentation report");
    expect(documentationReport).toContain("example.ipynb");

    const report = JSON.parse(
      await readFile(
        join(outputDirectory, BUN_DOCS_BUILD_REPORT_FILENAME),
        "utf8",
      ),
    ) as {
      legacy?: unknown;
      notices: typeof first.diagnostics;
      typescriptApi: { modules: number };
    };
    expect(report.legacy).toBeUndefined();
    expect(report.notices).toEqual(first.diagnostics);
    expect(report.typescriptApi.modules).toBe(2);
    await expect(
      access(join(outputDirectory, BUN_DOCS_MANIFEST_FILENAME)),
    ).resolves.toBeNull();

    const second = await buildBunDocsSite({
      configPath: join(docsDirectory, "bun-docs.json"),
    });
    expect(second.changed).toBeFalse();
    expect(
      second.changes.every((change) => change.action === "unchanged"),
    ).toBeTrue();

    await rm(join(docsDirectory, "nested/snippet.md"));
    const cleaned = await buildBunDocsSite({
      configPath: join(docsDirectory, "bun-docs.json"),
    });
    expect(cleaned.changes).toContainEqual({
      action: "deleted",
      path: join(outputDirectory, "nested/snippet.html"),
    });
    await expect(
      access(join(outputDirectory, "nested/snippet.html")),
    ).rejects.toThrow();
  } finally {
    await rm(root, { force: true, recursive: true });
  }
});

test("native builder refuses unowned output and does not process retired RST sources", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-bun-docs-owned-"));
  const docsDirectory = join(root, "docs");
  const outputDirectory = join(docsDirectory, "output");
  try {
    await writeFixture(
      docsDirectory,
      "bun-docs.json",
      JSON.stringify(
        {
          outputDirectory: "output",
          sourceDirectory: ".",
          title: "Fixture",
          typescriptApi: {
            outputDirectory: "typescript-api",
            packageName: "@fixture/runtime",
            sourceDirectory: "../runtime",
          },
          version: 1,
        },
        null,
        2,
      ),
    );
    await writeFixture(docsDirectory, "index.md", "# Fixture\n");
    await writeFixture(docsDirectory, "retired.rst", "Retired\n=======\n");
    await writeFixture(root, "runtime/index.ts", "export const value = true\n");
    await mkdir(outputDirectory, { recursive: true });
    await writeFile(
      join(outputDirectory, "manual.html"),
      "<p>keep</p>\n",
      "utf8",
    );

    await expect(
      buildBunDocsSite({ configPath: join(docsDirectory, "bun-docs.json") }),
    ).rejects.toBeInstanceOf(BunDocsBuildError);
    expect(await readFile(join(outputDirectory, "manual.html"), "utf8")).toBe(
      "<p>keep</p>\n",
    );
  } finally {
    await rm(root, { force: true, recursive: true });
  }
});

test("native builder preserves wrapped list items and surrounding paragraph boundaries", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-bun-docs-lists-"));
  const docsDirectory = join(root, "docs");
  const outputDirectory = join(docsDirectory, "output");
  try {
    await writeFixture(
      docsDirectory,
      "bun-docs.json",
      JSON.stringify(
        {
          outputDirectory: "output",
          sourceDirectory: ".",
          title: "Fixture",
          typescriptApi: {
            outputDirectory: "typescript-api",
            packageName: "@fixture/runtime",
            sourceDirectory: "../runtime",
          },
          version: 1,
        },
        null,
        2,
      ),
    );
    await writeFixture(
      docsDirectory,
      "index.md",
      [
        "# Wrapped lists",
        "",
        "Introduction.",
        "",
        "- first item starts here",
        "  and continues on the next source line",
        "- second item",
        "",
        "Between the lists.",
        "",
        "1. ordered item starts here",
        "   and continues with deeper indentation",
        "2. final ordered item",
        "",
        "After the lists.",
        "",
      ].join("\n"),
    );
    await writeFixture(root, "runtime/index.ts", "export const value = true\n");

    await buildBunDocsSite({
      configPath: join(docsDirectory, "bun-docs.json"),
    });

    const html = await readFile(join(outputDirectory, "index.html"), "utf8");
    const introduction = "<p>Introduction.</p>";
    const unordered =
      "<ul><li>first item starts here and continues on the next source line</li><li>second item</li></ul>";
    const between = "<p>Between the lists.</p>";
    const ordered =
      "<ol><li>ordered item starts here and continues with deeper indentation</li><li>final ordered item</li></ol>";
    const after = "<p>After the lists.</p>";

    expect(html).toContain(unordered);
    expect(html).toContain(ordered);
    expect(html.indexOf(introduction)).toBeLessThan(html.indexOf(unordered));
    expect(html.indexOf(unordered)).toBeLessThan(html.indexOf(between));
    expect(html.indexOf(between)).toBeLessThan(html.indexOf(ordered));
    expect(html.indexOf(ordered)).toBeLessThan(html.indexOf(after));
  } finally {
    await rm(root, { force: true, recursive: true });
  }
});

test("native configuration rejects the retired documentation compatibility field", async () => {
  const root = await mkdtemp(join(tmpdir(), "xerxes-bun-docs-config-"));
  const docsDirectory = join(root, "docs");
  try {
    await writeFixture(
      docsDirectory,
      "bun-docs.json",
      JSON.stringify(
        {
          legacy: {},
          outputDirectory: "output",
          sourceDirectory: ".",
          title: "Fixture",
          typescriptApi: {
            outputDirectory: "typescript-api",
            packageName: "@fixture/runtime",
            sourceDirectory: "../runtime",
          },
          version: 1,
        },
        null,
        2,
      ),
    );
    await writeFixture(docsDirectory, "index.md", "# Fixture\n");
    await writeFixture(root, "runtime/index.ts", "export const value = true\n");

    await expect(
      buildBunDocsSite({ configPath: join(docsDirectory, "bun-docs.json") }),
    ).rejects.toThrow('Unsupported Bun docs configuration key "legacy"');
  } finally {
    await rm(root, { force: true, recursive: true });
  }
});

async function writeFixture(
  root: string,
  relativePath: string,
  content: string,
): Promise<void> {
  const path = join(root, relativePath);
  await mkdir(join(path, ".."), { recursive: true });
  await writeFile(path, content, "utf8");
}
