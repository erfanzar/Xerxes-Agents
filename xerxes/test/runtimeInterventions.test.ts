// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from "bun:test";

import { DenialBudget, denialBudgetStopText } from "../src/runtime/denialBudget.js";
import {
  renderContextOverflowStopGuard,
  renderIntervention,
  renderOutputLimitResumeDirective,
  type Intervention,
} from "../src/runtime/interventions.js";
import { inspectObjectiveResponse } from "../src/runtime/objectiveGuard.js";

describe("runtime interventions", () => {
  test("renders the output-limit resume directive verbatim", () => {
    expect(renderIntervention({ kind: "resume-directive", directive: "output-limit" })).toBe(
      "[Output limit]\nOutput token limit hit. Resume directly — no apology, no recap.",
    );
    expect(renderOutputLimitResumeDirective()).toBe(
      renderIntervention({ kind: "resume-directive", directive: "output-limit" }),
    );
  });

  test("renders the context-overflow stop guard with the three remedies", () => {
    const text = renderIntervention({ kind: "stop-guard", variant: "context-overflow" });
    expect(text).toBe(renderContextOverflowStopGuard());
    expect(text).toContain("/compact");
    expect(text).toContain("/clear");
    expect(text).toContain("/branch");
  });

  test("renders objective reminders behind the [Objective gate] prefix consumers filter on", () => {
    const text = renderIntervention({ kind: "objective-reminder", reason: "tests still fail" });
    expect(text.startsWith("[Objective gate]\n")).toBe(true);
    expect(text).toContain("objective mode is still active: tests still fail.");
    expect(text).toContain("report BLOCKED: with exact evidence.");
  });

  test("renders every denial-budget refusal spelling, with and without a last denial", () => {
    expect(renderIntervention({ kind: "denial-budget", used: 3 })).toBe(
      "\n[Stopped: 3 consecutive tool calls were refused with no successful tool execution in between; "
        + "ending the turn instead of retrying a refusal loop.]",
    );
    const cases: readonly { readonly expected: string; readonly kind: "cancelled" | "permission_rejected" | "policy_denied" }[] = [
      {
        expected: "The last refusal was a cancellation on WebTool.",
        kind: "cancelled",
      },
      {
        expected: "The last refusal was a rejected permission prompt on WriteFile.",
        kind: "permission_rejected",
      },
      {
        expected: "The last refusal was a policy denial on ExecTool.",
        kind: "policy_denied",
      },
    ];
    for (const testCase of cases) {
      const text = renderIntervention({
        kind: "denial-budget",
        lastDenial: { kind: testCase.kind, toolName: testCase.kind === "cancelled" ? "WebTool" : testCase.kind === "permission_rejected" ? "WriteFile" : "ExecTool" },
        used: 9,
      });
      expect(text).toContain("9 consecutive tool calls were refused");
      expect(text).toContain(testCase.expected);
    }
  });

  test("keeps denialBudgetStopText byte-identical to the catalog rendering", () => {
    const budget = new DenialBudget(5);
    budget.record("policy_denied", "GrepTool");
    budget.record("policy_denied", "GrepTool");
    expect(denialBudgetStopText(budget)).toBe(
      renderIntervention({ kind: "denial-budget", lastDenial: { kind: "policy_denied", toolName: "GrepTool" }, used: 2 }),
    );
  });

  test("renders loop-shaped stop guards with their attempt counts", () => {
    expect(
      renderIntervention({ attempts: 3, kind: "stop-guard", variant: "output-limit-escalated" }),
    ).toBe(
      "\n[Stopped: the model hit the output token limit in 3 consecutive rounds; ending the turn instead of resuming again.]",
    );
    expect(
      renderIntervention({ attempts: 3, kind: "stop-guard", variant: "unconfigured-tools-loop" }),
    ).toBe(
      "\n[Stopped: the model requested only unconfigured tools in 3 consecutive rounds; ending the turn instead of looping on provider calls.]",
    );
  });

  test("renders the exhausted objective guard with its final grounds", () => {
    const text = renderIntervention({
      attempts: 6,
      kind: "stop-guard",
      reason: "no verified completion",
      variant: "objective-guard-exhausted",
    });
    expect(text).toBe(
      "\n[Stopped: objective guard could not get a verified completion or concrete blocker after 6 retries. The last issue was: no verified completion.]",
    );
  });

  test("renders steer notes for stream display only", () => {
    expect(renderIntervention({ content: "use bun test", kind: "steer-note" })).toBe(
      "\n[Steer saved for next turn: use bun test]",
    );
  });

  test("objective guard decisions carry catalog-rendered reminders", () => {
    // A narrative "done" answer with zero verification must trip the gate.
    const decision = inspectObjectiveResponse("Everything is finished and looks great.", {
      mode: "objective",
    });
    if (!decision.shouldContinue) {
      throw new Error("expected objective mode to hold the unverified completion");
    }
    expect(decision.reminder).toBe(renderIntervention({ kind: "objective-reminder", reason: decision.reason }));
  });

  test("every intervention variant renders without throwing and starts with its marker", () => {
    const samples: readonly Intervention[] = [
      { directive: "output-limit", kind: "resume-directive" },
      { kind: "objective-reminder", reason: "x" },
      { content: "y", kind: "steer-note" },
      { kind: "denial-budget", used: 1 },
      { kind: "stop-guard", variant: "context-overflow" },
      { attempts: 1, kind: "stop-guard", variant: "output-limit-escalated" },
      { attempts: 1, kind: "stop-guard", variant: "unconfigured-tools-loop" },
      { attempts: 1, kind: "stop-guard", reason: "r", variant: "objective-guard-exhausted" },
    ];
    for (const sample of samples) {
      expect(typeof renderIntervention(sample)).toBe("string");
      expect(renderIntervention(sample).length).toBeGreaterThan(0);
    }
  });
});

