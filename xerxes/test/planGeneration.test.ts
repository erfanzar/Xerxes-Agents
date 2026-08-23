// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from "bun:test";

import type { LlmClient, LlmCompletion } from "../src/llms/client.js";
import {
  CLAUDE_WORKFLOW_TOOL_GUIDANCE,
  createLlmPlanGenerator,
  parsePlanXml,
} from "../src/tools/claudeTools/workflow.js";

function fakeClient(response: string): LlmClient {
  const completion: LlmCompletion = { content: response, toolCalls: [] };
  return {
    // Stream-only client on purpose: the generator must go through completeLlm's
    // collection path, not depend on an optional complete() fast path.
    async *stream() {
      yield { content: response };
      void completion;
    },
  };
}

describe("llm plan generator", () => {
  test("generates dependency-ordered steps from a stream-only client", async () => {
    const generator = createLlmPlanGenerator(
      fakeClient(
        '<step id="s1" agent="coder" depends=""><description>write the module</description></step>\n'
          + '<step id="s2" agent="tester" depends="s1"><description>test it</description></step>',
      ),
      { model: "test-model" },
    );
    const raw = await generator.generate({
      agents: [{ name: "coder", description: "writes code" }],
      objective: "ship a tested module",
    });
    const steps = parsePlanXml(typeof raw === "string" ? raw : "");
    expect(steps).toHaveLength(2);
    expect(steps[0]).toMatchObject({ agent: "coder", id: "s1" });
    expect(steps[1]?.depends).toEqual(["s1"]);
  });

  test("prose answers produce zero steps, which PlanTool rejects upstream", async () => {
    const generator = createLlmPlanGenerator(fakeClient("I cannot help with that."), {
      model: "test-model",
    });
    const raw = await generator.generate({ agents: [], objective: "anything" });
    expect(parsePlanXml(typeof raw === "string" ? raw : "")).toHaveLength(0);
  });

  test("the planner prompt pins the XML contract and step ceiling", async () => {
    let captured = "";
    const client: LlmClient = {
      async *stream(request) {
        captured = request.messages[0]?.content as string;
        yield { content: "" };
      },
    };
    const generator = createLlmPlanGenerator(client, { maxSteps: 3, model: "m" });
    await generator.generate({ agents: [{ name: "a", description: "d" }], objective: "obj" });
    expect(captured).toContain('at most 3 concrete steps');
    expect(captured).toContain('<step id=');
    expect(captured).toContain("- a: d");
  });
});

describe("workflow tool guidance", () => {
  test("PlanTool guidance restricts use to explicit requests", () => {
    expect(CLAUDE_WORKFLOW_TOOL_GUIDANCE.PlanTool).toContain("ONLY when the user explicitly asks");
    expect(CLAUDE_WORKFLOW_TOOL_GUIDANCE.PlanTool).toContain("joined before this turn ends");
  });

  test("parsePlanXml still accepts the documented planner shape", () => {
    const steps = parsePlanXml(
      '<step id="a" agent="x" depends=""><description>one</description></step>'
        + '<step id="b" agent="y" depends="a"><description>two</description></step>',
    );
    expect(steps.map(step => step.id)).toEqual(["a", "b"]);
    expect(steps[1]?.depends).toEqual(["a"]);
  });
});
