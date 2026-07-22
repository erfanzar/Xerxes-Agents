// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtempSync, readdirSync, readFileSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  AgentSwitchEvent,
  AuditEmitter,
  AuditEvent,
  CompositeCollector,
  ErrorEvent,
  HookMutationEvent,
  InMemoryCollector,
  JSONLSinkCollector,
  OTelCollector,
  SandboxDecisionEvent,
  SkillAuthoredEvent,
  SkillFeedbackEvent,
  SkillUsedEvent,
  ToolCallAttemptEvent,
  ToolCallCompleteEvent,
  ToolCallFailureEvent,
  ToolLoopBlockEvent,
  ToolLoopWarningEvent,
  ToolPolicyDecisionEvent,
  TurnEndEvent,
  TurnStartEvent,
  cleanOtelAttributes,
  type OTelAttributes,
  type OTelSpan,
  type OTelTracer,
} from "../src/audit/index.js";

test("all audit event classes retain their stable discriminator and JSONL field names", () => {
  const events = [
    new AuditEvent({ metadata: { nested: { value: 1 } } }),
    new TurnStartEvent({ promptPreview: "start" }),
    new TurnEndEvent({ contentPreview: "end", functionCallsCount: 2 }),
    new ToolCallAttemptEvent({ toolName: "Read", argumentsPreview: "{}" }),
    new ToolCallCompleteEvent({
      toolName: "Read",
      durationMs: 1.5,
      resultPreview: "ok",
    }),
    new ToolCallFailureEvent({
      toolName: "Write",
      errorType: "Error",
      errorMessage: "nope",
    }),
    new ToolPolicyDecisionEvent({
      toolName: "Bash",
      action: "deny",
      policySource: "global",
    }),
    new SandboxDecisionEvent({
      toolName: "Bash",
      context: "shell",
      reason: "unsafe",
      backendType: "docker",
    }),
    new ToolLoopWarningEvent({
      toolName: "Read",
      pattern: "repeat",
      severityLevel: "high",
      callCount: 4,
    }),
    new ToolLoopBlockEvent({
      toolName: "Read",
      pattern: "repeat",
      callCount: 8,
    }),
    new HookMutationEvent({
      hookName: "before_tool_call",
      mutatedField: "arguments",
    }),
    new ErrorEvent({
      errorType: "RuntimeError",
      errorMessage: "broken",
      errorContext: "loop",
    }),
    new SkillUsedEvent({
      skillName: "review",
      version: "1",
      outcome: "success",
    }),
    new SkillAuthoredEvent({
      skillName: "review",
      uniqueTools: ["Read"],
      confirmedByUser: true,
    }),
    new SkillFeedbackEvent({ skillName: "review", rating: "positive" }),
    new AgentSwitchEvent({
      fromAgent: "planner",
      toAgent: "coder",
      reason: "capability",
    }),
  ];

  expect(new Set(events.map((event) => event.eventType)).size).toBe(
    events.length,
  );
  expect(new ToolCallFailureEvent().severity).toBe("error");
  expect(new ToolLoopWarningEvent().severity).toBe("warning");
  expect(new ToolLoopBlockEvent().severity).toBe("error");

  for (const event of events) {
    const record = JSON.parse(event.toJson()) as Record<string, unknown>;
    expect(record.event_type).toBe(event.eventType);
    expect(typeof record.timestamp).toBe("string");
    expect(record).toHaveProperty("agent_id");
    expect(record).toHaveProperty("turn_id");
    expect(record).toHaveProperty("session_id");
    expect(record).toHaveProperty("metadata");
  }
  expect(events[0]?.toRecord().metadata).toEqual({ nested: { value: 1 } });
});

test("audit records retain valid UTC timestamps, special text, and detached JSON-safe metadata", () => {
  const metadata: Record<string, unknown> = { nested: { values: [1, 2] } };
  const event = new AuditEvent({ metadata });
  const nested = metadata.nested as { values: number[] };
  nested.values.push(3);

  expect(event.timestamp.endsWith("Z")).toBeTrue();
  expect(Number.isNaN(new Date(event.timestamp).getTime())).toBeFalse();
  expect(event.toRecord().metadata).toEqual({ nested: { values: [1, 2] } });

  const circular: Record<string, unknown> = {};
  circular.self = circular;
  const serializable = new AuditEvent({ metadata: { circular } });
  expect(JSON.parse(serializable.toJson()).metadata.circular.self).toBe(
    "[Circular]",
  );
  expect(
    JSON.parse(
      new ErrorEvent({ errorMessage: 'He said "hello" & <goodbye>' }).toJson(),
    ).error_message,
  ).toBe('He said "hello" & <goodbye>');

  const attributes = cleanOtelAttributes({
    absent: null,
    date: new Date("2026-01-01T00:00:00.000Z"),
    nested: { value: true },
  });
  expect(attributes).toEqual({
    date: "2026-01-01T00:00:00.000Z",
    nested: '{"value":true}',
  });
});

test("emitter stamps session ids, truncates previews, covers skill/switch events, and respects hook replacement", () => {
  const memory = new InMemoryCollector();
  const emitter = new AuditEmitter({
    collector: memory,
    sessionId: "session-1",
  });
  const turnId = emitter.emitTurnStart({
    agentId: "coder",
    prompt: "x".repeat(250),
  });
  emitter.emitToolCallAttempt({
    toolName: "Read",
    args: { path: "README.md" },
    agentId: "coder",
    turnId,
  });
  emitter.emitToolCallComplete({
    toolName: "Read",
    durationMs: 4.5,
    result: "done",
    agentId: "coder",
    turnId,
  });
  emitter.emitSkillAuthored({
    skillName: "review",
    toolCount: 1,
    uniqueTools: ["Read"],
    confirmedByUser: true,
  });
  emitter.emitAgentSwitch({
    fromAgent: "planner",
    toAgent: "coder",
    reason: "capability",
    turnId,
  });
  emitter.emitTurnEnd({
    agentId: "coder",
    turnId,
    content: "complete",
    functionCallsCount: 1,
  });

  const events = memory.getEvents();
  expect(events).toHaveLength(6);
  expect(events.every((event) => event.sessionId === "session-1")).toBeTrue();
  expect((events[0] as TurnStartEvent).promptPreview).toHaveLength(200);
  expect((events[1] as ToolCallAttemptEvent).argumentsPreview).toBe(
    '{"path":"README.md"}',
  );
  expect((events[3] as SkillAuthoredEvent).uniqueTools).toEqual(["Read"]);
  expect((events[4] as AgentSwitchEvent).toAgent).toBe("coder");

  const hookCalls: string[] = [];
  const hooked = new AuditEmitter({
    collector: memory,
    hookRunner: {
      hasHooks: (hookPoint) => hookPoint === "on_loop_warning",
      run: (hookPoint, payload) =>
        hookCalls.push(`${hookPoint}:${payload.toolName}:${payload.count}`),
    },
  });
  hooked.emitLoopWarning({ toolName: "Read", count: 3 });
  expect(hookCalls).toEqual(["on_loop_warning:Read:3"]);
  expect(memory.getEventsByType("tool_loop_warning")).toHaveLength(0);

  hooked.emitToolLoopWarning({ toolName: "Read", count: 3 });
  expect(memory.getEventsByType("tool_loop_warning")).toHaveLength(1);
});

test("emitter covers failure, policy, loop, sandbox, hook, error, and skill audit actions", () => {
  const memory = new InMemoryCollector();
  const emitter = new AuditEmitter({
    collector: memory,
    sessionId: "session-methods",
  });
  const turnId = "turn-methods";

  emitter.emitToolCallFailure({
    agentId: "coder",
    errorMessage: "denied",
    errorType: "PermissionError",
    toolName: "Bash",
    turnId,
  });
  emitter.emitToolPolicyDecision({
    agentId: "coder",
    action: "deny",
    source: "global",
    toolName: "Bash",
    turnId,
  });
  emitter.emitToolLoopWarning({
    agentId: "coder",
    count: 3,
    pattern: "repeat",
    severity: "high",
    toolName: "Read",
    turnId,
  });
  emitter.emitToolLoopBlock({
    agentId: "coder",
    count: 4,
    pattern: "repeat",
    toolName: "Read",
    turnId,
  });
  emitter.emitSandboxDecision({
    agentId: "coder",
    backendType: "docker",
    context: "shell",
    reason: "policy",
    toolName: "Bash",
    turnId,
  });
  emitter.emitHookMutation({
    agentId: "coder",
    hookName: "before_tool_call",
    mutatedField: "arguments",
    toolName: "Bash",
    turnId,
  });
  emitter.emitError({
    agentId: "coder",
    context: "runtime",
    errorMessage: "broken",
    errorType: "RuntimeError",
    turnId,
  });
  emitter.emitSkillUsed({
    agentId: "coder",
    durationMs: 12.5,
    outcome: "success",
    skillName: "review",
    triggeredAutomatically: false,
    version: "1",
    turnId,
  });
  emitter.emitSkillFeedback({
    agentId: "coder",
    rating: "positive",
    reason: "useful",
    skillName: "review",
    source: "user",
    turnId,
  });

  const events = memory.getEvents();
  expect(events.map((event) => event.eventType)).toEqual([
    "tool_call_failure",
    "tool_policy_decision",
    "tool_loop_warning",
    "tool_loop_block",
    "sandbox_decision",
    "hook_mutation",
    "error",
    "skill_used",
    "skill_feedback",
  ]);
  expect(
    events.every(
      (event) =>
        event.agentId === "coder" &&
        event.sessionId === "session-methods" &&
        event.turnId === turnId,
    ),
  ).toBeTrue();
  expect((events[0] as ToolCallFailureEvent).errorMessage).toBe("denied");
  expect((events[1] as ToolPolicyDecisionEvent).policySource).toBe("global");
  expect((events[2] as ToolLoopWarningEvent).severityLevel).toBe("high");
  expect((events[4] as SandboxDecisionEvent).backendType).toBe("docker");
  expect((events[5] as HookMutationEvent).mutatedField).toBe("arguments");
  expect((events[7] as SkillUsedEvent).triggeredAutomatically).toBeFalse();
  expect((events[8] as SkillFeedbackEvent).reason).toBe("useful");
});

test("collectors preserve ordering, bound memory, fan out, and write valid JSON Lines", async () => {
  const directory = mkdtempSync(join(tmpdir(), "xerxes-audit-"));
  try {
    const path = join(directory, "nested", "audit.jsonl");
    const memory = new InMemoryCollector({ maxEvents: 2 });
    const jsonl = new JSONLSinkCollector(path);
    const collector = new CompositeCollector([memory, jsonl]);
    collector.emit(new TurnStartEvent({ turnId: "one" }));
    collector.emit(new TurnEndEvent({ turnId: "one" }));
    collector.emit(new ErrorEvent({ errorType: "E" }));
    collector.flush();
    await jsonl.close();

    expect(memory.size).toBe(2);
    expect(memory.getEvents().map((event) => event.eventType)).toEqual([
      "turn_end",
      "error",
    ]);
    const rows = readFileSync(path, "utf8")
      .trim()
      .split("\n")
      .map((line) => JSON.parse(line) as Record<string, unknown>);
    expect(rows).toHaveLength(3);
    expect(rows.map((row) => row.event_type)).toEqual([
      "turn_start",
      "turn_end",
      "error",
    ]);
    expect(() => jsonl.emit(new AuditEvent())).toThrow("closed");
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("collectors return detached snapshots and leave caller-owned text sinks usable", async () => {
  const memory = new InMemoryCollector();
  memory.emit(new TurnStartEvent({ turnId: "snapshot" }));
  const snapshot = memory.getEvents();
  snapshot.pop();
  expect(memory.size).toBe(1);
  memory.clear();
  expect(memory.size).toBe(0);

  const chunks: string[] = [];
  let flushCount = 0;
  const sink = {
    flush(): void {
      flushCount += 1;
    },
    write(value: string): void {
      chunks.push(value);
    },
  };
  const jsonl = new JSONLSinkCollector(sink);
  jsonl.emit(new ErrorEvent({ errorMessage: "streamed", errorType: "E" }));
  jsonl.flush();
  await jsonl.close();
  sink.write("caller-still-owns-this-sink\n");

  expect(flushCount).toBe(2);
  expect(JSON.parse(chunks[0] ?? "")).toMatchObject({
    error_message: "streamed",
    event_type: "error",
  });
  expect(chunks[1]).toBe("caller-still-owns-this-sink\n");
  expect(() => jsonl.emit(new AuditEvent())).toThrow("closed");
});

test("OTel collector records a no-op fallback without dependencies and attaches child events to injected turn spans", () => {
  const fallback = new OTelCollector();
  const fallbackEmitter = new AuditEmitter({ collector: fallback });
  fallbackEmitter.emitToolCallAttempt({ toolName: "Read", args: "{}" });
  fallback.emit(
    new HookMutationEvent({ hookName: "unknown", mutatedField: "field" }),
  );
  expect(fallback.hasOtel).toBeFalse();
  expect(fallback.fallbackLog.map((entry) => entry.name)).toContain(
    "tool.attempt:Read",
  );
  expect(fallback.fallbackLog.map((entry) => entry.name)).toContain(
    "hook_mutation",
  );

  const tracer = new RecordingTracer();
  const collector = new OTelCollector({ serviceName: "audit-test", tracer });
  const emitter = new AuditEmitter({ collector, sessionId: "session-2" });
  const turnId = emitter.emitTurnStart({ agentId: "agent", prompt: "hello" });
  emitter.emitToolCallAttempt({ toolName: "Read", args: "{}", turnId });
  emitter.emitTurnEnd({ turnId, functionCallsCount: 1 });

  expect(collector.hasOtel).toBeTrue();
  expect(tracer.spans).toHaveLength(1);
  const span = tracer.spans[0];
  expect(span?.name).toBe("xerxes.turn");
  expect(span?.attributes["service.name"]).toBe("audit-test");
  expect(span?.events.map((event) => event.name)).toEqual([
    "tool.attempt:Read",
  ]);
  expect(span?.attributes["xerxes.function_calls_count"]).toBe(1);
  expect(span?.ended).toBeTrue();
  expect(collector.openTurnCount).toBe(0);

  const unclosedTurnId = emitter.emitTurnStart({
    agentId: "agent",
    prompt: "will flush",
  });
  expect(unclosedTurnId).toHaveLength(12);
  expect(collector.openTurnCount).toBe(1);
  collector.flush();
  expect(collector.openTurnCount).toBe(0);
  expect(tracer.spans[1]?.ended).toBeTrue();

  collector.emit(
    new HookMutationEvent({ hookName: "unknown", mutatedField: "field" }),
  );
  expect(tracer.spans[2]?.name).toBe("hook_mutation");
  expect(tracer.spans[2]?.ended).toBeTrue();
});

test("JSONL file sink writes owner-only files inside a private directory", async () => {
  const directory = mkdtempSync(join(tmpdir(), "xerxes-audit-perms-"));
  try {
    const path = join(directory, "nested", "audit.jsonl");
    const collector = new JSONLSinkCollector(path);
    collector.emit(new ErrorEvent({ errorType: "E", errorMessage: "perm" }));
    await collector.close();

    if (process.platform !== "win32") {
      expect(statSync(path).mode & 0o777).toBe(0o600);
      expect(statSync(join(directory, "nested")).mode & 0o777).toBe(0o700);
    }
    expect(readFileSync(path, "utf8")).toContain('"error_type"');
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("JSONL file sink rotates at maxBytes and retains only the newest segments", async () => {
  const directory = mkdtempSync(join(tmpdir(), "xerxes-audit-rotate-"));
  try {
    const path = join(directory, "audit.jsonl");
    const collector = new JSONLSinkCollector(path, {
      maxBytes: 256,
      maxFiles: 2,
      flushIntervalMs: 60_000,
    });
    for (let index = 0; index < 12; index += 1) {
      collector.emit(
        new ErrorEvent({
          errorType: "E",
          errorMessage: `payload-${index}-${"x".repeat(80)}`,
        }),
      );
      // Drain each batch so rotation decisions are exercised deterministically.
      await collector.drain();
    }
    await collector.close();

    const active = readFileSync(path, "utf8");
    expect(active).toContain("payload-11");
    // The oldest events rotated out of the active file entirely.
    expect(active).not.toContain("payload-0");
    const segments = readdirSync(directory).filter((name) =>
      name.startsWith("audit.jsonl."),
    );
    expect(segments.sort()).toEqual(["audit.jsonl.1", "audit.jsonl.2"]);
    // The oldest rotations were discarded; every retained line is valid JSON.
    for (const segment of segments) {
      for (const line of readFileSync(join(directory, segment), "utf8").trim().split("\n")) {
        expect(() => JSON.parse(line)).not.toThrow();
      }
    }
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("JSONL file sink drops and counts events once the bounded queue is full", async () => {
  const directory = mkdtempSync(join(tmpdir(), "xerxes-audit-drop-"));
  try {
    const path = join(directory, "audit.jsonl");
    const collector = new JSONLSinkCollector(path, {
      maxQueueSize: 3,
      flushIntervalMs: 60_000,
    });
    for (let index = 0; index < 10; index += 1) {
      collector.emit(new ErrorEvent({ errorType: "E", errorMessage: `e${index}` }));
    }
    expect(collector.droppedEvents).toBe(7);
    expect(collector.pendingCount).toBe(3);
    await collector.close();

    const rows = readFileSync(path, "utf8").trim().split("\n");
    expect(rows).toHaveLength(3);
    expect(collector.failedWriteBatches).toBe(0);
    expect(() => collector.emit(new ErrorEvent({ errorType: "E" }))).toThrow(
      "closed",
    );
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
});

test("OTel collector evicts stale open turns and bounds the fallback log with counters", () => {
  const tracer = new RecordingTracer();
  const collector = new OTelCollector({
    maxFallbackEntries: 3,
    maxOpenTurns: 2,
    tracer,
  });
  for (let index = 0; index < 5; index += 1) {
    collector.emit(new TurnStartEvent({ turnId: `turn-${index}` }));
  }
  expect(collector.openTurnCount).toBe(2);
  expect(collector.evictedTurnCount).toBe(3);
  // Evicted spans are ended so exporters do not leak unfinished spans.
  expect(tracer.spans.filter((span) => span.ended)).toHaveLength(3);

  const fallback = new OTelCollector({ maxFallbackEntries: 4 });
  for (let index = 0; index < 10; index += 1) {
    fallback.emit(new ErrorEvent({ errorType: "E", errorMessage: `m${index}` }));
  }
  expect(fallback.fallbackLog).toHaveLength(4);
  expect(fallback.evictedFallbackCount).toBe(6);
  expect(fallback.fallbackLog[0]?.attributes.error_message).toBe("m6");

  expect(() => new OTelCollector({ maxOpenTurns: 0 })).toThrow(
    "positive integer",
  );
});

class RecordingSpan implements OTelSpan {
  readonly attributes: Record<string, boolean | number | string>;
  readonly events: Array<{
    readonly attributes: OTelAttributes | undefined;
    readonly name: string;
  }> = [];
  ended = false;

  constructor(
    readonly name: string,
    attributes: OTelAttributes | undefined,
  ) {
    this.attributes = { ...(attributes ?? {}) };
  }

  addEvent(name: string, attributes?: OTelAttributes): void {
    this.events.push({ name, attributes });
  }

  end(): void {
    this.ended = true;
  }

  setAttribute(key: string, value: boolean | number | string): void {
    this.attributes[key] = value;
  }
}

class RecordingTracer implements OTelTracer {
  readonly spans: RecordingSpan[] = [];

  startSpan(
    name: string,
    options?: Readonly<{ attributes?: OTelAttributes }>,
  ): OTelSpan {
    const span = new RecordingSpan(name, options?.attributes);
    this.spans.push(span);
    return span;
  }
}
