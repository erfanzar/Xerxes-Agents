// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from "bun:test";

import {
  assembleContextLayers,
  layerDigests,
  MAX_ASSEMBLY_PROVENANCE_ENTRIES,
  readAssemblyProvenance,
  recordAssemblyProvenance,
} from "../src/context/assembly.js";
import { joinSystemSegments } from "../src/streaming/promptCaching.js";

const fullInput = {
  addendum: "operator note",
  agentPrompt: "You are a coding agent.",
  bootstrap: "Workspace prelude",
  contextDeltas: "reasoning: high",
  memoryRecall: "user prefers bun",
  modeHint: "[Mode: plan]",
  recoveredSubagents: "1 delegated task handle(s) were recovered",
  selfMemory: "I previously fixed the gateway",
  subagentJoin: "Background subagents are joined before the parent turn ends.",
  toolGuidance: "[Tool usage: WriteFile]\nread before write.",
};

describe("context assembly", () => {
  test("assembles every named layer in stable-then-volatile order", () => {
    const layers = assembleContextLayers(fullInput);
    expect(layers.map(layer => layer.name)).toEqual([
      "bootstrap",
      "agent",
      "tool_guidance",
      "mode_hint",
      "subagent_join",
      "recovered_subagents",
      "memory",
      "self_memory",
      "context_deltas",
      "addendum",
    ]);
    for (const layer of layers.slice(0, 5)) {
      expect(layer.volatile).toBeUndefined();
    }
    for (const layer of layers.slice(5)) {
      expect(layer.volatile).toBe(true);
    }
  });

  test("identical inputs assemble byte-identically (cache-parity contract)", () => {
    const first = joinSystemSegments(assembleContextLayers(fullInput));
    const second = joinSystemSegments(assembleContextLayers({ ...fullInput }));
    expect(second).toBe(first);
    // A one-byte drift changes the joined prompt — the property that makes
    // cache invalidation detectable rather than silent.
    expect(joinSystemSegments(assembleContextLayers({ ...fullInput, agentPrompt: "You are a coding agent. " }))).not.toBe(first);
  });

  test("drops empty contributions instead of emitting blank separators", () => {
    const layers = assembleContextLayers({
      addendum: "",
      agentPrompt: "",
      bootstrap: "prelude",
      contextDeltas: "",
      memoryRecall: "recall",
      modeHint: "",
      recoveredSubagents: "",
      selfMemory: "",
      subagentJoin: "",
      toolGuidance: "",
    });
    expect(layers.map(layer => layer.name)).toEqual(["bootstrap", "memory"]);
  });

  test("absent optional layers behave exactly like empty ones", () => {
    const absent = assembleContextLayers({
      agentPrompt: "",
      bootstrap: "b",
      modeHint: "",
      subagentJoin: "",
      toolGuidance: "",
    });
    const empty = assembleContextLayers({
      agentPrompt: "",
      bootstrap: "b",
      modeHint: "",
      subagentJoin: "",
      toolGuidance: "",
      memoryRecall: "",
    });
    expect(absent).toEqual(empty);
  });

  test("layer digests are per-layer, stable, and sensitive to content", () => {
    const layers = assembleContextLayers(fullInput);
    const digests = layerDigests(layers);
    expect(digests.map(digest => digest.name)).toEqual(layers.map(layer => layer.name));

    const again = layerDigests(assembleContextLayers({ ...fullInput }));
    expect(again).toEqual(digests);

    // Only the changed layer moves; its neighbors keep their digests.
    const moved = layerDigests(assembleContextLayers({ ...fullInput, memoryRecall: "different recall" }));
    const byName = new Map(moved.map(digest => [digest.name, digest.hash]));
    const before = new Map(digests.map(digest => [digest.name, digest.hash]));
    expect(byName.get("memory")).not.toBe(before.get("memory"));
    expect(byName.get("bootstrap")).toBe(before.get("bootstrap"));
    expect(byName.get("addendum")).toBe(before.get("addendum"));

    for (const digest of digests) {
      expect(digest.hash).toMatch(/^[0-9a-f]{16}$/);
    }
  });

  test("provenance records persist per turn in a bounded ring", () => {
    const metadata: Record<string, unknown> = {};
    for (let index = 0; index < MAX_ASSEMBLY_PROVENANCE_ENTRIES + 5; index += 1) {
      const layers = assembleContextLayers({ ...fullInput, memoryRecall: `recall-${index}` });
      recordAssemblyProvenance(metadata, {
        layers: layerDigests(layers),
        ...(index % 2 === 0 ? { turnId: `turn-${index}` } : {}),
        recordedAt: 1_000 + index,
      });
    }
    const records = readAssemblyProvenance(metadata);
    expect(records).toHaveLength(MAX_ASSEMBLY_PROVENANCE_ENTRIES);
    // Oldest entries drop; the newest survives intact with its optional id.
    expect(records[0]?.recordedAt).toBe(1_000 + 5);
    expect(records.at(-1)?.turnId).toBe(`turn-${MAX_ASSEMBLY_PROVENANCE_ENTRIES + 4}`);
    // The moved layer is identifiable across the ring.
    const newest = records.at(-1);
    expect(newest?.layers.find(layer => layer.name === "memory")?.hash).toBeDefined();
  });

  test("provenance reading ignores corrupt entries", () => {
    const metadata: Record<string, unknown> = {
      context_assembly: ["junk", { recordedAt: "nope" }, { layers: [], recordedAt: 7 }],
    };
    expect(readAssemblyProvenance(metadata)).toEqual([{ layers: [], recordedAt: 7 }]);
  });
});
