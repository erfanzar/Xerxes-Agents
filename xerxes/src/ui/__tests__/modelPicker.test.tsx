// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
/** @jsxImportSource @opentui/react */

import { testRender } from "@opentui/react/test-utils";
import { act } from "react";
import { describe, expect, it, vi } from "vitest";

import { GatewayProvider } from "../app/gatewayContext.js";
import type { GatewayServices } from "../app/interfaces.js";
import type { GatewayClient } from "../gatewayClient.js";
import type {
  ModelModelsResponse,
  ModelOptionsResponse,
} from "../gatewayTypes.js";
import { ModelPicker } from "../opentui/modelPicker.js";
import { DEFAULT_THEME } from "../theme.js";

interface Deferred<T> {
  promise: Promise<T>;
  reject: (reason: unknown) => void;
  resolve: (value: T) => void;
}

const deferred = <T,>(): Deferred<T> => {
  let reject!: (reason: unknown) => void;
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((res, rej) => {
    reject = rej;
    resolve = res;
  });

  return { promise, reject, resolve };
};

const options: ModelOptionsResponse = {
  model: "k3",
  provider: "kimi-local",
  providers: [
    {
      configured_model: "gpt-4.1",
      is_current: false,
      name: "OpenAI work",
      provider_type: "openai",
      slug: "openai-work",
    },
    {
      configured_model: "kimi-for-coding",
      is_current: true,
      name: "Kimi work",
      provider_type: "kimi-code",
      slug: "kimi-local",
    },
  ],
};

const renderPicker = async (
  request: (
    method: string,
    params?: Record<string, unknown>,
  ) => Promise<unknown>,
  onSelect = vi.fn(),
) => {
  const services = {
    gw: { request: vi.fn(request) } as unknown as GatewayClient,
    rpc: vi.fn(),
  } as unknown as GatewayServices;
  const setup = await testRender(
    <GatewayProvider value={services}>
      <ModelPicker
        onSelect={onSelect}
        sessionId="live-session"
        t={DEFAULT_THEME}
      />
    </GatewayProvider>,
    { height: 24, width: 100 },
  );

  await act(async () => {
    await Bun.sleep(0);
  });
  await setup.flush();

  return {
    onSelect,
    request: services.gw.request as ReturnType<typeof vi.fn>,
    setup,
  };
};

const flushPromises = async (
  setup: Awaited<ReturnType<typeof testRender>>,
  delayMs = 0,
) => {
  await act(async () => {
    await Bun.sleep(delayMs);
  });
  await setup.flush();
};

describe("OpenTUI dynamic model picker", () => {
  it("discovers only the selected profile, caches the result, and reports a real count", async () => {
    const models = deferred<ModelModelsResponse>();
    const request = vi.fn(
      (method: string, params?: Record<string, unknown>) => {
        if (method === "model.options") {
          expect(params).toEqual({ session_id: "live-session" });
          return Promise.resolve(options);
        }
        if (method === "model.models") {
          expect(params).toEqual({ profile_name: "kimi-local" });
          return models.promise;
        }
        return Promise.reject(new Error(`unexpected request: ${method}`));
      },
    );
    const { setup } = await renderPicker(request);

    try {
      const providersFrame = setup.captureCharFrame();
      expect(providersFrame).toContain("Current: k3");
      expect(providersFrame).toContain("discover models");
      expect(providersFrame).not.toContain("1 models");
      expect(request).toHaveBeenCalledTimes(1);

      act(() => setup.mockInput.pressEnter());
      await setup.flush();
      expect(setup.captureCharFrame()).toContain(
        "discovering models from this profile",
      );
      expect(request).toHaveBeenCalledTimes(2);

      await act(async () => {
        models.resolve({ models: ["k3", "kimi-k2.5"], source: "remote" });
        await Bun.sleep(0);
      });
      await setup.flush();
      expect(setup.captureCharFrame()).toContain("k3");
      expect(setup.captureCharFrame()).toContain("kimi-k2.5");
      expect(setup.captureCharFrame()).toContain("source: remote");

      act(() => setup.mockInput.pressEscape());
      await flushPromises(setup, 50);
      expect(setup.captureCharFrame()).toContain("2 available");

      act(() => setup.mockInput.pressEnter());
      await setup.flush();
      expect(request).toHaveBeenCalledTimes(2);
    } finally {
      act(() => setup.renderer.destroy());
    }
  });

  it("keeps warning-backed fallbacks retryable when the profile is entered again", async () => {
    let attempts = 0;
    const request = vi.fn((method: string) => {
      if (method === "model.options") return Promise.resolve(options);
      if (method === "model.models") {
        attempts += 1;
        return Promise.resolve(
          attempts === 1
            ? ({
                models: ["kimi-for-coding"],
                source: "profile",
                warning: "provider catalogue unavailable",
              } satisfies ModelModelsResponse)
            : ({
                models: ["fresh-dynamic-model"],
                source: "remote",
              } satisfies ModelModelsResponse),
        );
      }
      return Promise.reject(new Error(`unexpected request: ${method}`));
    });
    const { setup } = await renderPicker(request);

    try {
      act(() => setup.mockInput.pressEnter());
      await flushPromises(setup);
      expect(setup.captureCharFrame()).toContain(
        "warning: provider catalogue unavailable",
      );
      expect(setup.captureCharFrame()).toContain("fallback available");

      act(() => setup.mockInput.pressEscape());
      await flushPromises(setup, 50);
      expect(setup.captureCharFrame()).toContain(
        "incomplete · provider catalogue unavailable",
      );

      act(() => setup.mockInput.pressEnter());
      await flushPromises(setup);
      expect(setup.captureCharFrame()).toContain("fresh-dynamic-model");
      expect(
        request.mock.calls.filter(([method]) => method === "model.options"),
      ).toHaveLength(1);
      expect(
        request.mock.calls.filter(([method]) => method === "model.models"),
      ).toHaveLength(2);
    } finally {
      act(() => setup.renderer.destroy());
    }
  });

  it("keeps live and typed fallbacks usable while discovery is still pending", async () => {
    const models = deferred<ModelModelsResponse>();
    const onSelect = vi.fn();
    const request = vi.fn((method: string) => {
      if (method === "model.options") return Promise.resolve(options);
      if (method === "model.models") return models.promise;
      return Promise.reject(new Error(`unexpected request: ${method}`));
    });
    const { setup } = await renderPicker(request, onSelect);

    try {
      act(() => setup.mockInput.pressEnter());
      await setup.flush();
      expect(setup.captureCharFrame()).toContain("Enter fallback");

      await act(async () => {
        await setup.mockInput.typeText("runtime/preview-model");
      });
      await setup.flush();
      expect(setup.captureCharFrame()).toContain('Use "runtime/preview-model"');

      act(() => setup.mockInput.pressEnter());
      await setup.flush();
      expect(onSelect).toHaveBeenCalledWith(
        "runtime/preview-model --provider kimi-local --tui-session",
      );
    } finally {
      act(() => setup.renderer.destroy());
      models.resolve({ models: ["late-model"], source: "remote" });
      await Bun.sleep(0);
    }
  });

  it("keeps discovery errors local and accepts a typed full model ID", async () => {
    const onSelect = vi.fn();
    const request = vi.fn((method: string) => {
      if (method === "model.options") {
        return Promise.resolve({
          model: "saved-default",
          providers: [
            {
              configured_model: "saved-default",
              is_current: true,
              name: "Custom profile",
              provider_type: "custom",
              slug: "custom-profile",
            },
          ],
        } satisfies ModelOptionsResponse);
      }
      if (method === "model.models") {
        return Promise.reject(new Error("catalogue unavailable"));
      }
      return Promise.reject(new Error(`unexpected request: ${method}`));
    });
    const { setup } = await renderPicker(request, onSelect);

    try {
      act(() => setup.mockInput.pressEnter());
      await flushPromises(setup);

      const errorFrame = setup.captureCharFrame();
      expect(errorFrame).toContain("discovery failed: catalogue unavailable");
      expect(errorFrame).toContain("saved-default");
      expect(errorFrame).toContain("type full ID");

      await act(async () => {
        await setup.mockInput.typeText("vendor/new-model");
      });
      await setup.flush();
      expect(setup.captureCharFrame()).toContain('Use "vendor/new-model"');

      act(() => setup.mockInput.pressEnter());
      await setup.flush();
      expect(onSelect).toHaveBeenCalledWith(
        "vendor/new-model --provider custom-profile --tui-session",
      );
    } finally {
      act(() => setup.renderer.destroy());
    }
  });

  it("treats q as filter input and lets Escape leave an in-flight discovery", async () => {
    const first = deferred<ModelModelsResponse>();
    const second = deferred<ModelModelsResponse>();
    const request = vi.fn(
      (method: string, params?: Record<string, unknown>) => {
        if (method === "model.options") {
          return Promise.resolve({
            model: "a-current",
            providers: [
              {
                configured_model: "a-current",
                is_current: true,
                name: "Alpha",
                slug: "alpha",
              },
              {
                configured_model: "qwen-default",
                is_current: false,
                name: "Qwen",
                slug: "qwen",
              },
            ],
          } satisfies ModelOptionsResponse);
        }
        if (method === "model.models") {
          return params?.profile_name === "alpha"
            ? first.promise
            : second.promise;
        }
        return Promise.reject(new Error(`unexpected request: ${method}`));
      },
    );
    const { setup } = await renderPicker(request);

    try {
      act(() => setup.mockInput.pressKey("q"));
      await setup.flush();
      expect(setup.captureCharFrame()).toContain("filter: q");
      expect(setup.captureCharFrame()).toContain("Qwen");

      act(() => setup.mockInput.pressEscape());
      await flushPromises(setup, 50);
      act(() => setup.mockInput.pressEnter());
      await setup.flush();
      expect(setup.captureCharFrame()).toContain("discovering models");

      act(() => setup.mockInput.pressEscape());
      await flushPromises(setup, 50);
      expect(setup.captureCharFrame()).toContain("Select provider");

      act(() => setup.mockInput.pressArrow("down"));
      act(() => setup.mockInput.pressEnter());
      await setup.flush();
      await act(async () => {
        second.resolve({ models: ["qwen-dynamic"], source: "remote" });
        await Bun.sleep(0);
      });
      await setup.flush();
      expect(setup.captureCharFrame()).toContain("qwen-dynamic");

      await act(async () => {
        first.resolve({ models: ["stale-alpha"], source: "remote" });
        await Bun.sleep(0);
      });
      await setup.flush();
      expect(setup.captureCharFrame()).toContain("qwen-dynamic");
      expect(setup.captureCharFrame()).not.toContain("stale-alpha");
    } finally {
      act(() => setup.renderer.destroy());
    }
  });

  it("keeps a failed options load inline and retryable without taking over the picker", async () => {
    let attempts = 0;
    const request = vi.fn((method: string) => {
      if (method === "model.options") {
        attempts += 1;
        return attempts === 1
          ? Promise.reject(new Error("catalog unreachable"))
          : Promise.resolve(options);
      }
      return Promise.reject(new Error(`unexpected request: ${method}`));
    });
    const { setup } = await renderPicker(request);

    try {
      // The failure renders inline inside the normal provider stage — no
      // full-screen error takeover.
      const errorFrame = setup.captureCharFrame();
      expect(errorFrame).toContain("Select provider · step 1/2");
      expect(errorFrame).toContain("error: catalog unreachable");
      expect(errorFrame).toContain("no providers available");

      // Browsing keys keep their normal meaning: filter input is accepted.
      await act(async () => {
        await setup.mockInput.typeText("kimi");
      });
      await setup.flush();
      expect(setup.captureCharFrame()).toContain("filter: kimi");
      expect(setup.captureCharFrame()).toContain("no providers match");

      act(() => setup.mockInput.pressKey("u", { ctrl: true }));
      await setup.flush();

      // Ctrl+R retries the load and restores full browsing.
      act(() => setup.mockInput.pressKey("r", { ctrl: true }));
      await flushPromises(setup);

      const retried = setup.captureCharFrame();
      expect(retried).toContain("Kimi work");
      expect(retried).not.toContain("error: catalog unreachable");
      expect(
        request.mock.calls.filter(([method]) => method === "model.options"),
      ).toHaveLength(2);
    } finally {
      act(() => setup.renderer.destroy());
    }
  });

  it("retries one failed profile without reloading every provider", async () => {
    let attempts = 0;
    const request = vi.fn((method: string) => {
      if (method === "model.options") return Promise.resolve(options);
      if (method === "model.models") {
        attempts += 1;
        return attempts === 1
          ? Promise.reject(new Error("temporary failure"))
          : Promise.resolve({
              models: ["retry-model"],
              source: "remote",
            } satisfies ModelModelsResponse);
      }
      return Promise.reject(new Error(`unexpected request: ${method}`));
    });
    const { setup } = await renderPicker(request);

    try {
      act(() => setup.mockInput.pressEnter());
      await flushPromises(setup);
      expect(setup.captureCharFrame()).toContain("temporary failure");

      act(() => setup.mockInput.pressKey("r", { ctrl: true }));
      await flushPromises(setup);
      expect(setup.captureCharFrame()).toContain("retry-model");
      expect(
        request.mock.calls.filter(([method]) => method === "model.options"),
      ).toHaveLength(1);
      expect(
        request.mock.calls.filter(([method]) => method === "model.models"),
      ).toHaveLength(2);
    } finally {
      act(() => setup.renderer.destroy());
    }
  });
});
