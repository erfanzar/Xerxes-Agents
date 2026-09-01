// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

import { parseGoalCommand, runGoalCommand } from "../src/daemon/goalCommand.js";
import { completeGoal, getGoal, resetGoalActivations } from "../src/runtime/goalDomain.js";

function fresh(): Record<string, unknown> {
  resetGoalActivations();
  return {};
}

test("only exact control words are subcommands; everything else is an objective", () => {
  expect(parseGoalCommand("")).toEqual({ kind: "show" });
  expect(parseGoalCommand("  ")).toEqual({ kind: "show" });
  expect(parseGoalCommand("pause")).toEqual({ kind: "pause" });
  expect(parseGoalCommand("RESUME")).toEqual({ kind: "resume" });
  expect(parseGoalCommand("edit")).toEqual({ kind: "invalid-edit" });
  expect(parseGoalCommand("edit ship the release")).toEqual({
    kind: "edit",
    objective: "ship the release",
  });
  // Prose that merely begins with a control word is an objective. A goal is
  // written in English, and "clear the backlog" is not a request to discard
  // the goal.
  expect(parseGoalCommand("clear the backlog")).toEqual({
    kind: "create",
    objective: "clear the backlog",
  });
  expect(parseGoalCommand("pause the ingestion job")).toEqual({
    kind: "create",
    objective: "pause the ingestion job",
  });
});

test("the full human lifecycle runs through the same domain the tools use", () => {
  const metadata = fresh();
  expect(runGoalCommand(metadata, "s1", "", 1).text).toContain("No goal is currently set");

  const created = runGoalCommand(metadata, "s1", "migrate the store", 2);
  expect(created.ok).toBe(true);
  expect(created.text).toContain("Goal created");
  expect(created.text).toContain("Objective: migrate the store");
  expect(created.text).toContain("Activation: armed");
  expect(getGoal(metadata, "s1")?.phase).toBe("active");

  // A second create must not silently discard work in flight.
  const second = runGoalCommand(metadata, "s1", "something else", 3);
  expect(second.ok).toBe(false);
  expect(second.text).toContain("already active");

  const edited = runGoalCommand(metadata, "s1", "edit migrate the store safely", 4);
  expect(edited.text).toContain("Objective: migrate the store safely");

  const paused = runGoalCommand(metadata, "s1", "pause", 5);
  expect(paused.text).toContain("Status: paused");
  expect(getGoal(metadata, "s1")?.activation).toBe("disarmed");

  const resumed = runGoalCommand(metadata, "s1", "resume", 6);
  expect(resumed.text).toContain("Status: active");
  expect(getGoal(metadata, "s1")?.activation).toBe("armed");

  expect(runGoalCommand(metadata, "s1", "clear", 7).text).toBe("Goal cleared.");
  expect(getGoal(metadata, "s1")).toBeUndefined();
});

test("operations that need a goal say so instead of failing opaquely", () => {
  const metadata = fresh();
  for (const action of ["pause", "resume", "edit new objective"]) {
    const result = runGoalCommand(metadata, "s1", action, 1);
    expect(result.ok).toBe(false);
    expect(result.text).toContain("No goal is currently set");
  }
  const invalid = runGoalCommand(metadata, "s1", "edit", 1);
  expect(invalid.ok).toBe(false);
  expect(invalid.text).toContain("replacement objective");
});

test("a refused transition reports the domain's own reason, not a generic error", () => {
  const metadata = fresh();
  runGoalCommand(metadata, "s1", "ship it", 1);
  // Already active and armed: resume has nothing to do, and says which goal.
  const resumed = runGoalCommand(metadata, "s1", "resume", 2);
  expect(resumed.ok).toBe(false);
  expect(resumed.text).toContain("already active and armed");
  expect(resumed.text).toContain("Run /goal");
});

test("editing a completed goal starts the next one rather than rewriting history", () => {
  const metadata = fresh();
  runGoalCommand(metadata, "s1", "first objective", 1);
  runGoalCommand(metadata, "s1", "pause", 2);
  const before = getGoal(metadata, "s1")!;
  // Complete it the way the tools would, then edit.
  completeGoal(metadata, "s1", { id: before.id, revision: before.revision }, 3);

  const edited = runGoalCommand(metadata, "s1", "edit second objective", 4);
  expect(edited.text).toContain("Goal created");
  expect(edited.text).toContain("Objective: second objective");
  expect(getGoal(metadata, "s1")?.phase).toBe("active");
});
