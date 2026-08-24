// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from "bun:test";

import {
  clearGoalLedger,
  MAX_GOAL_CRITERIA,
  readGoalLedger,
  startGoalLedger,
  updateGoalLedger,
} from "../src/runtime/goalState.js";

describe("goal ledger", () => {
  test("start creates revision 1 exactly once; a second start reports the incumbent", () => {
    const metadata: Record<string, unknown> = {};
    const first = startGoalLedger(metadata, { criteria: ["tests pass"], now: 100, text: "beat the benchmark" });
    if (!("created" in first)) throw new Error("expected creation on empty metadata");
    const created = first.created;
    expect(created.revision).toBe(1);
    expect(created.phase).toBe("active");
    expect(created.roundsStarted).toBe(0);

    const second = startGoalLedger(metadata, { now: 200, text: "a different goal" });
    if (!("existing" in second)) throw new Error("expected conflict with the incumbent goal");
    expect(second.existing.text).toBe("beat the benchmark");
    // The incumbent was not clobbered.
    expect(readGoalLedger(metadata)?.text).toBe("beat the benchmark");
  });

  test("compare-and-set updates bump the revision and reject stale writers", () => {
    const metadata: Record<string, unknown> = {};
    startGoalLedger(metadata, { now: 1, text: "ship it" });
    const current = readGoalLedger(metadata)!;

    const stale = updateGoalLedger(metadata, current.revision + 5, { roundDelta: 1 }, 2);
    expect(stale.ok).toBe(false);
    if (!stale.ok && stale.reason === "stale") {
      expect(stale.conflictWith?.revision).toBe(current.revision);
    } else {
      throw new Error("expected a stale conflict");
    }
    // The rejected write changed nothing.
    expect(readGoalLedger(metadata)?.roundsStarted).toBe(0);

    const good = updateGoalLedger(metadata, current.revision, { roundDelta: 1 }, 3);
    expect(good.ok).toBe(true);
    if (good.ok) {
      expect(good.ledger.revision).toBe(2);
      expect(good.ledger.roundsStarted).toBe(1);
      expect(good.ledger.updatedAt).toBe(3);
    }
  });

  test("phase transitions and clears behave as terminal operations", () => {
    const metadata: Record<string, unknown> = {};
    startGoalLedger(metadata, { now: 1, text: "g" });
    let ledger = readGoalLedger(metadata)!;
    const verified = updateGoalLedger(metadata, ledger.revision, { phase: "verified" }, 2);
    expect(verified.ok).toBe(true);
    ledger = readGoalLedger(metadata)!;
    expect(ledger.phase).toBe("verified");

    expect(clearGoalLedger(metadata)).toBe(true);
    expect(readGoalLedger(metadata)).toBeUndefined();
    // Updating after a clear is a 'missing' conflict, not a silent recreate.
    const afterClear = updateGoalLedger(metadata, ledger.revision, { roundDelta: 1 }, 4);
    expect(afterClear.ok).toBe(false);
    if (!afterClear.ok) expect(afterClear.reason).toBe("missing");
    expect(clearGoalLedger(metadata)).toBe(false);
  });

  test("criteria lists are bounded and round floors stay at zero", () => {
    const metadata: Record<string, unknown> = {};
    const criteria = Array.from({ length: MAX_GOAL_CRITERIA + 10 }, (_, index) => `c${index}`);
    startGoalLedger(metadata, { criteria, now: 1, text: "g" });
    expect(readGoalLedger(metadata)?.criteria).toHaveLength(MAX_GOAL_CRITERIA);

    let ledger = readGoalLedger(metadata)!;
    updateGoalLedger(metadata, ledger.revision, { roundDelta: -10 }, 2);
    ledger = readGoalLedger(metadata)!;
    expect(ledger.roundsStarted).toBe(0);
  });

  test("corrupt or absent ledgers read as undefined", () => {
    expect(readGoalLedger({})).toBeUndefined();
    expect(readGoalLedger({ goal_ledger: "junk" })).toBeUndefined();
    expect(
      readGoalLedger({ goal_ledger: { revision: "x", roundsStarted: 1, updatedAt: 1, criteria: [], phase: "active" } }),
    ).toBeUndefined();
  });
});
