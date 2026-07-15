// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

import {
  BUNDLED_SKILL_CLI_USAGE,
  runBundledSkillCli,
} from "../src/skills/cli.js";

test("bundled skill CLI exposes help and rejects unknown skills without a Python fallback", async () => {
  const output: string[] = [];
  expect(
    await runBundledSkillCli([], {
      writeLine: (line) => {
        output.push(line);
      },
    }),
  ).toBe(0);
  expect(output).toEqual([BUNDLED_SKILL_CLI_USAGE]);

  output.length = 0;
  expect(
    await runBundledSkillCli(["unknown-skill"], {
      writeLine: (line) => {
        output.push(line);
      },
    }),
  ).toBe(2);
  expect(output).toEqual(["Unknown native bundled skill: unknown-skill"]);
});

test("bundled skill CLI routes Google Workspace setup guidance without an ambient credential fallback", async () => {
  const output: string[] = [];
  expect(
    await runBundledSkillCli(["google-workspace", "setup", "guidance"], {
      writeLine: (line) => {
        output.push(line);
      },
    }),
  ).toBe(0);
  expect(JSON.parse(output[0] ?? "{}")).toMatchObject({
    security: expect.stringContaining("does not discover"),
  });
});

test("bundled skill CLI exposes Bun-native OCR document help without an engine fallback", async () => {
  const output: string[] = [];
  expect(
    await runBundledSkillCli(["ocr-and-documents", "--help"], {
      writeLine: (line) => {
        output.push(line);
      },
    }),
  ).toBe(0);
  expect(output[0]).toContain("pymupdf");
  expect(output[0]).toContain("never\nuses PyMuPDF");
});

test("bundled skill CLI routes GRPO planning through the native workflow without inventing an accelerator", async () => {
  const output: string[] = [];
  const code = await runBundledSkillCli(
    ["grpo-rl-training", "--dry-run", "--dataset", "gsm8k.jsonl"],
    {
      grpoTraining: {
        readTextFile: () =>
          '{"question":"What is 6 + 6?","answer":"Working. #### 12"}\n',
      },
      writeLine: (line) => {
        output.push(line);
      },
    },
  );

  expect(code).toBe(0);
  expect(JSON.parse(output[0] ?? "{}")).toMatchObject({
    kind: "xerxes.grpo-training-request.v1",
    datasetExamples: 1,
    hostBoundary: expect.objectContaining({
      accelerator: expect.stringContaining("caller-owned"),
    }),
  });

  output.length = 0;
  expect(
    await runBundledSkillCli(
      ["grpo-rl-training", "--run", "--dataset", "gsm8k.jsonl"],
      {
        grpoTraining: {
          readTextFile: () =>
            '{"question":"What is 6 + 6?","answer":"Working. #### 12"}\n',
        },
        writeLine: (line) => {
          output.push(line);
        },
      },
    ),
  ).toBe(2);
  expect(output).toEqual([
    expect.stringContaining(
      "inject explicit host accelerator and storage ports",
    ),
  ]);
});

test("bundled skill CLI runs the native Excalidraw uploader through injected Bun ports", async () => {
  const output: string[] = [];
  let requestUrl = "";
  const code = await runBundledSkillCli(
    ["excalidraw", "upload", "diagram.excalidraw"],
    {
      fetchImplementation: async (input) => {
        requestUrl = String(input);
        return Response.json({ id: "native-file" });
      },
      readTextFile: () => '{"elements":[]}',
      writeLine: (line) => {
        output.push(line);
      },
    },
  );

  expect(code).toBe(0);
  expect(requestUrl).toContain("excalidraw.com");
  expect(output).toEqual([expect.stringContaining("#json=native-file,")]);
});

test("bundled skill CLI dispatches nearby, YouTube, and arXiv queries through native clients", async () => {
  const nearbyOutput: string[] = [];
  const nearbyCode = await runBundledSkillCli(
    ["find-nearby", "--near", "Null Island", "--type", "cafe", "--json"],
    {
      fetchImplementation: async (input) => {
        const url = String(input);
        if (url.includes("nominatim.openstreetmap.org"))
          return Response.json([{ lat: "0", lon: "0" }]);
        return Response.json({
          elements: [
            { lat: 0, lon: 0, tags: { amenity: "cafe", name: "Zero Cafe" } },
          ],
        });
      },
      writeLine: (line) => {
        nearbyOutput.push(line);
      },
    },
  );
  expect(nearbyCode).toBe(0);
  expect(JSON.parse(nearbyOutput[0] ?? "{}")).toMatchObject({
    count: 1,
    origin: { lat: 0, lon: 0 },
    results: [{ distance_m: 0, name: "Zero Cafe" }],
  });

  const player = JSON.stringify({
    captions: {
      playerCaptionsTracklistRenderer: {
        captionTracks: [
          { baseUrl: "https://captions.test/api?lang=en", languageCode: "en" },
        ],
      },
    },
  });
  const youtubeOutput: string[] = [];
  const youtubeCode = await runBundledSkillCli(
    ["youtube-transcript", "abcdefghijk", "--timestamps", "--text-only"],
    {
      fetchImplementation: async (input) => {
        const url = String(input);
        if (url.includes("youtube.com/watch"))
          return new Response(
            `<script>var ytInitialPlayerResponse = ${player};</script>`,
          );
        return Response.json({
          events: [
            { dDurationMs: 1_000, segs: [{ utf8: "Hello" }], tStartMs: 0 },
          ],
        });
      },
      writeLine: (line) => {
        youtubeOutput.push(line);
      },
    },
  );
  expect(youtubeCode).toBe(0);
  expect(youtubeOutput).toEqual(["0:00 Hello"]);

  const arxivOutput: string[] = [];
  const arxivCode = await runBundledSkillCli(
    ["arxiv", "native bun", "--author", "Ada"],
    {
      fetchImplementation: async () =>
        new Response(
          `<?xml version="1.0"?><feed><entry><id>http://arxiv.org/abs/2501.12345v1</id><title>Native Bun</title><summary>Native paper.</summary><published>2026-01-01T00:00:00Z</published><updated>2026-01-02T00:00:00Z</updated><author><name>Ada</name></author><category term="cs.AI" /></entry></feed>`,
        ),
      writeLine: (line) => {
        arxivOutput.push(line);
      },
    },
  );
  expect(arxivCode).toBe(0);
  expect(arxivOutput[0]).toContain("Native Bun");
  expect(arxivOutput[0]).toContain("2501.12345v1");
});

test("bundled skill CLI formats native Polymarket search results", async () => {
  const output: string[] = [];
  const code = await runBundledSkillCli(["polymarket", "search", "native"], {
    fetchImplementation: async (input) => {
      expect(String(input)).toContain("public-search?q=native");
      return Response.json({
        events: [
          {
            markets: [
              {
                outcomePrices: '["0.75","0.25"]',
                outcomes: '["Yes","No"]',
                question: "Will Bun ship?",
                slug: "bun",
                volume: "1200",
              },
            ],
            slug: "native-event",
            title: "Native Event",
            volume: "1200",
          },
        ],
        pagination: { totalResults: 1 },
      });
    },
    writeLine: (line) => {
      output.push(line);
    },
  });

  expect(code).toBe(0);
  expect(output[0]).toContain("Found 1 results");
  expect(output[0]).toContain("Yes: 75.0%");
});
