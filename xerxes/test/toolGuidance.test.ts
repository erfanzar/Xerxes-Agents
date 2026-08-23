// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from "bun:test";

import {
  renderToolGuidance,
  ToolRegistry,
} from "../src/executors/toolRegistry.js";
import type { ToolDefinition } from "../src/types/toolCalls.js";

function definition(name: string): ToolDefinition {
  return {
    function: { name, description: `${name} description`, parameters: { type: "object", properties: {} } },
    type: "function",
  };
}

const noop = () => "ok";

describe("tool guidance", () => {
  test("returns segments only for tools that declared them, in caller order", () => {
    const registry = new ToolRegistry();
    registry.register(definition("Alpha"), noop, "default", undefined, "alpha policy");
    registry.register(definition("Beta"), noop);
    registry.register(definition("Gamma"), noop, "default", undefined, "gamma policy");

    const segments = registry.guidanceForTools(["Beta", "Gamma", "Unknown", "Alpha"]);
    expect(segments).toEqual([
      { name: "Gamma", text: "gamma policy" },
      { name: "Alpha", text: "alpha policy" },
    ]);
  });

  test("agent-specific variants win through the same lookup as execution", () => {
    const registry = new ToolRegistry();
    registry.register(definition("Shared"), noop, "default", undefined, "shared default");
    registry.register(definition("Shared"), noop, "researcher", undefined, "researcher override");

    expect(registry.guidanceForTools(["Shared"], "researcher")).toEqual([
      { name: "Shared", text: "researcher override" },
    ]);
    expect(registry.guidanceForTools(["Shared"])).toEqual([
      { name: "Shared", text: "shared default" },
    ]);
    // An agent without its own variant falls back to the shared default —
    // exactly what execution and capabilities() do through pick().
    expect(registry.guidanceForTools(["Shared"], "coder")).toEqual([
      { name: "Shared", text: "shared default" },
    ]);
  });

  test("replace swaps the guidance with the definition", () => {
    const registry = new ToolRegistry();
    registry.register(definition("Tool"), noop, "default", undefined, "before");
    registry.replace(definition("Tool"), noop, "default", undefined, "after");
    expect(registry.guidanceForTools(["Tool"])).toEqual([{ name: "Tool", text: "after" }]);
    registry.replace(definition("Tool"), noop, "default");
    expect(registry.guidanceForTools(["Tool"])).toEqual([]);
  });

  test("rendering is empty for no segments and stable for an unchanged surface", () => {
    expect(renderToolGuidance([])).toBe("");
    const first = renderToolGuidance([
      { name: "WriteFile", text: "read before write." },
      { name: "exec_command", text: "argv only." },
    ]);
    expect(first).toBe("[Tool usage: WriteFile]\nread before write.\n\n[Tool usage: exec_command]\nargv only.");
    const second = renderToolGuidance([
      { name: "WriteFile", text: "read before write." },
      { name: "exec_command", text: "argv only." },
    ]);
    expect(second).toBe(first);
  });

  test("shipped file-write guidance travels with the registered surface", async () => {
    const { registerFileTools } = await import("../src/tools/fileTools.js");
    const registry = new ToolRegistry();
    registerFileTools(registry, { resolveInWorkspace: (value: string) => value } as never);
    const names = ["FileEditTool", "WriteFile"];
    const segments = registry.guidanceForTools(names);
    expect(segments.map(segment => segment.name)).toEqual(names);
    for (const segment of segments) {
      expect(segment.text).toContain("ReadFile the target immediately before writing");
    }
  });

  test("shipped exec_command guidance states the argv and batching contract", async () => {
    const { registerProcessTools } = await import("../src/tools/processTools.js");
    const registry = new ToolRegistry();
    registerProcessTools(
      registry,
      { resolveInWorkspace: (value: string) => value } as never,
    );
    const [segment] = registry.guidanceForTools(["exec_command"]);
    expect(segment?.text).toContain("never a single shell string");
    expect(segment?.text).toContain("runs alone");
  });
});
