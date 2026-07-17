// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";

import type { ProviderProfile } from "../src/bridge/profiles.js";
import {
  discoverModelCatalog,
  discoverModelIds,
  modelCatalogFromResponse,
  modelDiscoveryEndpoint,
  modelsFromResponse,
  profileDiscoveryApiKey,
  sanitizeModelDiscoveryError,
  validateModelDiscoveryUrl,
} from "../src/daemon/modelDiscovery.js";
import type { FetchImplementation } from "../src/llms/client.js";

test("model discovery uses provider-specific endpoints and credentials", async () => {
  const requests: Array<{
    headers: Headers;
    redirect: RequestRedirect | undefined;
    url: string;
  }> = [];
  const fetchImplementation: FetchImplementation = async (input, init) => {
    requests.push({
      headers: new Headers(init?.headers),
      redirect: init?.redirect,
      url: String(input),
    });
    return new Response(
      JSON.stringify({
        data: [
          { id: " model-a " },
          { name: "model-b" },
          { model: "model-a" },
          { ignored: true },
        ],
      }),
    );
  };

  await expect(
    discoverModelIds(
      {
        apiKey: "openai-secret",
        baseUrl: "https://provider.example/v1/",
        provider: "openai",
      },
      { fetchImplementation },
    ),
  ).resolves.toEqual(["model-a", "model-b"]);

  await expect(
    discoverModelIds(
      {
        apiKey: "anthropic-secret",
        baseUrl: "https://api.anthropic.com",
        provider: "anthropic",
      },
      { fetchImplementation },
    ),
  ).resolves.toEqual(["model-a", "model-b"]);

  expect(requests[0]?.url).toBe("https://provider.example/v1/models");
  expect(requests[0]?.headers.get("authorization")).toBe(
    "Bearer openai-secret",
  );
  expect(requests[0]?.redirect).toBe("error");
  expect(requests[1]?.url).toBe("https://api.anthropic.com/v1/models");
  expect(requests[1]?.headers.get("authorization")).toBeNull();
  expect(requests[1]?.headers.get("x-api-key")).toBe("anthropic-secret");
  expect(requests[1]?.headers.get("anthropic-version")).toBe("2023-06-01");
});

test("model discovery resolves stored profile environment credentials but explicit probes can opt out", async () => {
  const profile: ProviderProfile = {
    api_key: "",
    base_url: "https://api.kimi.com/coding/v1",
    model: "saved-model",
    name: "kimi",
    provider: "kimi-code",
    sampling: {},
  };
  expect(
    profileDiscoveryApiKey(profile, {
      KIMI_CODE_API_KEY: "environment-secret",
    }),
  ).toBe("environment-secret");

  expect(
    profileDiscoveryApiKey(
      {
        ...profile,
        base_url: "https://untrusted.example/v1",
        provider: "openai",
      },
      { OPENAI_API_KEY: "must-stay-local" },
    ),
  ).toBe("");

  const authorizations: Array<string | null> = [];
  await discoverModelIds(
    {
      apiKey: "",
      baseUrl: "https://provider.example/v1",
      environment: { OPENAI_API_KEY: "must-not-be-used" },
      provider: "openai",
      resolveProviderCredential: false,
    },
    {
      fetchImplementation: async (_input, init) => {
        authorizations.push(new Headers(init?.headers).get("authorization"));
        return new Response(JSON.stringify({ data: [{ id: "remote" }] }));
      },
    },
  );
  await discoverModelIds(
    {
      apiKey: "",
      baseUrl: "https://untrusted.example/v1",
      environment: { OPENAI_API_KEY: "must-stay-local" },
      provider: "openai",
    },
    {
      fetchImplementation: async (_input, init) => {
        authorizations.push(new Headers(init?.headers).get("authorization"));
        return new Response(JSON.stringify({ data: [{ id: "remote" }] }));
      },
    },
  );
  expect(authorizations).toEqual([null, null]);
});

test("model discovery rejects unsafe legacy endpoints and does not double-append models", () => {
  expect(validateModelDiscoveryUrl("http://localhost:11434/v1")).toContain(
    "Private or local",
  );
  expect(validateModelDiscoveryUrl("http://127.0.0.1:11434/v1")).toContain(
    "Private or local",
  );
  expect(
    validateModelDiscoveryUrl("http://[::ffff:127.0.0.1]:11434/v1"),
  ).toContain("Private or local");
  expect(
    validateModelDiscoveryUrl("https://user:secret@provider.example/v1"),
  ).toContain("must not contain credentials");
  expect(
    validateModelDiscoveryUrl("http://localhost:11434/v1", true),
  ).toBeUndefined();
  expect(
    modelDiscoveryEndpoint("https://provider.example/v1/models", "openai"),
  ).toBe("https://provider.example/v1/models");
});

test("model discovery rejects DNS targets that resolve privately but trusted profile endpoints may stay local", async () => {
  let calls = 0;
  const fetchImplementation: FetchImplementation = async () => {
    calls += 1;
    return new Response(JSON.stringify({ data: [{ id: "remote" }] }));
  };

  await expect(
    discoverModelIds(
      {
        baseUrl: "https://rebinding.example/v1",
        provider: "custom",
        resolveProviderCredential: false,
      },
      {
        fetchImplementation,
        resolveHostname: async () => ["127.0.0.1"],
      },
    ),
  ).rejects.toThrow("Private or local");
  expect(calls).toBe(0);

  await expect(
    discoverModelIds(
      {
        allowPrivateEndpoint: true,
        baseUrl: "http://127.0.0.1:11434/v1",
        provider: "ollama",
      },
      {
        fetchImplementation,
        resolveHostname: async () => {
          throw new Error("trusted profile endpoints do not need public DNS");
        },
      },
    ),
  ).resolves.toEqual(["remote"]);
  expect(calls).toBe(1);
});

test("model discovery times out, bounds response bodies, and redacts exact credentials", async () => {
  const hangingFetch: FetchImplementation = async (_input, init) =>
    new Promise<Response>((_resolve, reject) => {
      const signal = init?.signal;
      signal?.addEventListener("abort", () => reject(signal.reason), {
        once: true,
      });
    });
  await expect(
    discoverModelIds(
      {
        apiKey: "tiny-secret",
        baseUrl: "https://provider.example/v1",
        provider: "openai",
      },
      { fetchImplementation: hangingFetch, timeoutMs: 5 },
    ),
  ).rejects.toThrow("timed out");

  await expect(
    discoverModelIds(
      {
        baseUrl: "https://slow-dns.example/v1",
        provider: "custom",
        resolveProviderCredential: false,
      },
      {
        fetchImplementation: async () => {
          throw new Error("fetch must not run before DNS validation");
        },
        resolveHostname: async () => new Promise<readonly string[]>(() => {}),
        timeoutMs: 5,
      },
    ),
  ).rejects.toThrow("timed out");

  await expect(
    discoverModelIds(
      {
        apiKey: "tiny-secret",
        baseUrl: "https://provider.example/v1",
        provider: "openai",
      },
      {
        fetchImplementation: async () => new Response("01234567890"),
        maxResponseBytes: 10,
      },
    ),
  ).rejects.toThrow("exceeded 10 bytes");

  expect(
    sanitizeModelDiscoveryError(new Error("upstream rejected tiny-secret"), {
      apiKey: "tiny-secret",
    }),
  ).toBe("upstream rejected [redacted]");
});

test("model response normalization accepts data or models arrays only", () => {
  expect(
    modelsFromResponse({
      models: [" first ", { name: "second" }, { id: "first" }],
    }),
  ).toEqual(["first", "second"]);
  expect(modelsFromResponse({ data: "not-an-array" })).toEqual([]);
  expect(modelsFromResponse(null)).toEqual([]);
});

test("model discovery preserves provider-reported context metadata", async () => {
  await expect(
    discoverModelCatalog(
      {
        baseUrl: "https://provider.example/v1",
        provider: "custom",
        resolveProviderCredential: false,
      },
      {
        fetchImplementation: async () =>
          new Response(
            JSON.stringify({
              data: [
                { id: "k3", context_length: 262_144 },
                { id: "k3" },
                { id: "wide", max_model_len: 1_000_000 },
                { id: "invalid", context_window: -1 },
              ],
            }),
          ),
      },
    ),
  ).resolves.toEqual([
    { id: "k3", contextLimit: 262_144 },
    { id: "wide", contextLimit: 1_000_000 },
    { id: "invalid" },
  ]);

  expect(
    modelCatalogFromResponse({
      models: [{ name: "camel", contextLength: 32_768 }],
    }),
  ).toEqual([{ id: "camel", contextLimit: 32_768 }]);
});
