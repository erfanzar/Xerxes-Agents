// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from "bun:test";

import {
  DEFAULT_RETRY_POLICY,
  PROVIDERS,
  retryPolicyForModel,
} from "../src/llms/providerRegistry.js";
import { DEFAULT_RETRY_DELAYS } from "../src/streaming/loop.js";

describe("per-provider retry policies", () => {
  test("unrouted providers inherit the shared default policy", () => {
    expect(retryPolicyForModel("anthropic/claude-sonnet-4")).toEqual(DEFAULT_RETRY_POLICY);
    // Default schedule stays aligned with the loop's historical constants.
    expect(DEFAULT_RETRY_POLICY.delaysMs).toEqual([...DEFAULT_RETRY_DELAYS]);
    expect(DEFAULT_RETRY_POLICY.maxSuggestedDelayMs).toBe(60_000);
  });

  test("local routes retry fast and cap server hints low", () => {
    const ollama = retryPolicyForModel("ollama/llama3");
    expect(ollama.delaysMs).toEqual([250, 500]);
    expect(ollama.maxSuggestedDelayMs).toBeLessThan(DEFAULT_RETRY_POLICY.maxSuggestedDelayMs);
    expect(retryPolicyForModel("lmstudio/x")).toEqual(ollama);
  });

  test("the subscription codex route carries more patience than the default", () => {
    const codex = retryPolicyForModel("openai-codex/gpt-5.2");
    expect(codex.delaysMs.length).toBeGreaterThan(DEFAULT_RETRY_POLICY.delaysMs.length);
    expect(codex.delaysMs[0]).toBeGreaterThan(DEFAULT_RETRY_POLICY.delaysMs[0]!);
  });

  test("every declared policy is internally consistent", () => {
    for (const [name, config] of Object.entries(PROVIDERS)) {
      if (config.retry === undefined) continue;
      expect(config.retry.delaysMs.length, name).toBeGreaterThan(0);
      for (const delay of config.retry.delaysMs) {
        expect(Number.isFinite(delay) && delay >= 0, `${name} delay ${delay}`).toBe(true);
      }
      expect(config.retry.maxSuggestedDelayMs, name).toBeGreaterThanOrEqual(0);
    }
  });
});
