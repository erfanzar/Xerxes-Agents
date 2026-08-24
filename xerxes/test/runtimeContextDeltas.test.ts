// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from "bun:test";

import {
  appendContextDelta,
  contextDeltaFor,
  MAX_CONTEXT_DELTAS,
  readContextDeltas,
  renderContextDeltas,
  takeContextDeltas,
} from "../src/runtime/contextDeltas.js";

describe("context deltas", () => {
  test("a real change yields a delta; an unchanged value yields none", () => {
    expect(contextDeltaFor(undefined, "plan", 1, "interaction-mode")).toEqual({
      at: 1,
      layer: "interaction-mode",
      value: "plan",
    });
    expect(contextDeltaFor("plan", "plan", 2, "interaction-mode")).toBeUndefined();
    expect(contextDeltaFor("auto", "accept-all", 3, "permission")).toEqual({
      at: 3,
      layer: "permission",
      value: "accept-all",
    });
  });

  test("append, read, and take form an exactly-once consumption cycle", () => {
    const metadata: Record<string, unknown> = {};
    appendContextDelta(metadata, { at: 1, layer: "model", value: "gpt-x" });
    appendContextDelta(metadata, { at: 2, layer: "reasoning", value: "high" });

    expect(readContextDeltas(metadata)).toHaveLength(2);
    // Reading never consumes.
    expect(readContextDeltas(metadata)).toHaveLength(2);

    const drained = takeContextDeltas(metadata);
    expect(drained.map(delta => delta.layer)).toEqual(["model", "reasoning"]);
    // Taking consumes: the next turn assembles with no delta layer.
    expect(takeContextDeltas(metadata)).toEqual([]);
    expect(readContextDeltas(metadata)).toEqual([]);
  });

  test("the ring keeps only the newest entries", () => {
    const metadata: Record<string, unknown> = {};
    for (let index = 0; index < MAX_CONTEXT_DELTAS + 4; index += 1) {
      appendContextDelta(metadata, { at: index, layer: "model", value: `m${index}` });
    }
    const pending = readContextDeltas(metadata);
    expect(pending).toHaveLength(MAX_CONTEXT_DELTAS);
    expect(pending[0]?.value).toBe(`m${MAX_CONTEXT_DELTAS - MAX_CONTEXT_DELTAS + 4}`);
    expect(pending.at(-1)?.value).toBe(`m${MAX_CONTEXT_DELTAS + 3}`);
  });

  test("renders the [Context updated] block with one labeled line per change", () => {
    expect(renderContextDeltas([])).toBe("");
    expect(renderContextDeltas([{ at: 5, layer: "interaction-mode", value: "plan" }])).toBe(
      "[Context updated]\n- interaction mode: plan",
    );
    expect(
      renderContextDeltas([
        { at: 5, layer: "model", value: "kimi-k2" },
        { at: 6, layer: "permission", value: "manual" },
        { at: 7, layer: "reasoning", value: "ultra" },
      ]),
    ).toBe(
      "[Context updated]\n- model: kimi-k2\n- permission mode: manual\n- reasoning effort: ultra",
    );
  });

  test("corrupt metadata entries are ignored rather than crashing assembly", () => {
    const metadata: Record<string, unknown> = {
      context_deltas: [
        "junk",
        null,
        { at: "not-a-number", layer: "model", value: "x" },
        { at: 9, layer: "warp-drive", value: "y" },
        { at: 10, layer: "model", value: "real" },
      ],
    };
    expect(readContextDeltas(metadata)).toEqual([{ at: 10, layer: "model", value: "real" }]);
    expect(renderContextDeltas(takeContextDeltas(metadata))).toBe("[Context updated]\n- model: real");
  });
});
