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
  pickerProps: { allowPersistGlobal?: boolean; width?: number } = {},
) => {
  const services = {
    gw: { request: vi.fn(request) } as unknown as GatewayClient,
    rpc: vi.fn(),
  } as unknown as GatewayServices;
  const setup = await testRender(
    <GatewayProvider value={services}>
      <ModelPicker
        allowPersistGlobal={pickerProps.allowPersistGlobal}
        onSelect={onSelect}
        sessionId="live-session"
        t={DEFAULT_THEME}
      />
    </GatewayProvider>,
    { height: 24, width: pickerProps.width ?? 76 },
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
      expect(setup.captureCharFrame()).toContain("Model");

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
      expect(errorFrame).toContain("Model");
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

describe("OpenTUI model picker — model pane scrolling", () => {
  it("follows the selection instead of showing the first rows forever", async () => {
    // Fourteen models on a tall terminal. The pane used to `.slice(0, N)`, so
    // arrowing down moved the selection while the list stood still — you saw
    // the same three or four models however far you scrolled.
    const many = Array.from({ length: 14 }, (_, i) => `vendor/model-${String(i).padStart(2, "0")}`);
    const request = vi.fn(async (method: string) =>
      method === "model.options" ? options : { models: many, source: "remote" },
    );
    const services = {
      gw: { request } as unknown as GatewayClient,
      rpc: vi.fn(),
    } as unknown as GatewayServices;
    const setup = await testRender(
      <GatewayProvider value={services}>
        <ModelPicker onSelect={vi.fn()} sessionId="live-session" t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 26, width: 120 },
    );

    try {
      await flushPromises(setup);
      await flushPromises(setup, 10);

      // Step into the model pane, then walk to the bottom of the list.
      act(() => setup.mockInput.pressArrow("right"));
      await flushPromises(setup, 2);
      for (let i = 0; i < 13; i += 1) {
        act(() => setup.mockInput.pressArrow("down"));
      }
      await flushPromises(setup, 2);

      const frame = setup.captureCharFrame();

      expect(frame).toContain("model-13");
      expect(frame).not.toContain("model-00");
    } finally {
      act(() => setup.renderer.destroy());
    }
  });
});

describe("OpenTUI model picker — mockup 09 two-pane layout", () => {
  it("shows the provider rail and the focused profile's models side by side", async () => {
    const models: ModelModelsResponse = {
      models: ["kimi-for-coding", "k3", "k3-256k"],
      source: "remote",
    };
    const onSelect = vi.fn();
    const request = vi.fn(async (method: string) =>
      method === "model.options"
        ? options
        : { models: models.models, source: "remote" },
    );
    const services = {
      gw: { request } as unknown as GatewayClient,
      rpc: vi.fn(),
    } as unknown as GatewayServices;
    const setup = await testRender(
      <GatewayProvider value={services}>
        <ModelPicker onSelect={onSelect} sessionId="live-session" t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 100 },
    );

    try {
      await flushPromises(setup);
      await flushPromises(setup, 10);
      const frame = setup.captureCharFrame();

      // Both stages visible at once: the rail caption and a family caption.
      expect(frame).toContain("STEP 1 provider  · 2");
      expect(frame).toContain("Kimi work");
      // `k3` and `k3-256k` are one family and earn a caption…
      expect(frame).toContain("k3 · 2");
      // …while `kimi-for-coding` is a family of one and must NOT get a
      // caption naming the single row beneath it. That cost a row per model
      // and is why a four-model provider only showed two.
      expect(frame).toContain("kimi-for-coding");
      expect(frame).not.toContain("kimi-for-coding ·");

      // Right arrow dives into the model pane; Enter there selects.
      setup.mockInput.pressArrow("right");
      await flushPromises(setup);
      setup.mockInput.pressArrow("down");
      await flushPromises(setup);
      setup.mockInput.pressEnter();
      await flushPromises(setup);

      expect(onSelect).toHaveBeenCalledTimes(1);
      const value = String(onSelect.mock.calls[0]?.[0]);

      expect(value).toContain("--provider kimi-local");
    } finally {
      act(() => setup.renderer.destroy());
    }
  });

  it("uses compact shared overlay chrome instead of stretching through tall terminals", async () => {
    const request = vi.fn(async (method: string) =>
      method === "model.options"
        ? options
        : { models: ["kimi-for-coding", "k3", "k3-256k"], source: "remote" },
    );
    const services = {
      gw: { request } as unknown as GatewayClient,
      rpc: vi.fn(),
    } as unknown as GatewayServices;
    const setup = await testRender(
      <GatewayProvider value={services}>
        <ModelPicker onSelect={vi.fn()} sessionId="live-session" t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 48, width: 120 },
    );

    try {
      await flushPromises(setup);
      await flushPromises(setup, 10);
      const frame = setup.captureCharFrame();
      const lines = frame.split("\n");
      const top = lines.findIndex((line) => line.includes("╭"));
      const bottom = lines.findIndex((line, index) => index > top && line.includes("╰"));

      expect(frame).toContain("Model  ›  Kimi work");
      expect(frame).toContain("profiles · type to filter");
      expect(frame).toContain("STEP 2 model  · 3");
      expect(frame).toContain("active");
      expect(frame).toContain("current k3 · session · Esc cancel");
      expect(top).toBeGreaterThanOrEqual(0);
      // 14, not 13: the panel now reserves its real chrome (frame, padding,
      // header, footer and the panes' margin — eight rows) instead of five,
      // so the frame is one row taller and, crucially, no longer three rows
      // shorter than the content it is supposed to contain.
      expect(bottom - top).toBeLessThanOrEqual(14);
    } finally {
      act(() => setup.renderer.destroy());
    }
  });

  it("keeps the sequential wizard below the wide threshold", async () => {
    const request = vi.fn(async (method: string) =>
      method === "model.options" ? options : { models: ["k3"], source: "remote" },
    );
    const services = {
      gw: { request } as unknown as GatewayClient,
      rpc: vi.fn(),
    } as unknown as GatewayServices;
    const setup = await testRender(
      <GatewayProvider value={services}>
        <ModelPicker onSelect={vi.fn()} sessionId="live-session" t={DEFAULT_THEME} />
      </GatewayProvider>,
      { height: 24, width: 76 },
    );

    try {
      await flushPromises(setup);
      const frame = setup.captureCharFrame();

      expect(frame).toContain("Model");
      expect(frame).not.toContain("PROFILES ·");
    } finally {
      act(() => setup.renderer.destroy());
    }
  });
});


describe("OpenTUI model picker — mockup 09 persistence alias", () => {
  it("toggles set-as-default with 'a' as well as ctrl+g", async () => {
    const request = vi.fn(async (method: string) =>
      method === "model.options"
        ? options
        : { models: ["k3"], source: "remote" },
    );
    const { setup } = await renderPicker(request, vi.fn(), {
      allowPersistGlobal: true,
    });

    try {
      await flushPromises(setup);
      let frame = setup.captureCharFrame();
      expect(frame).toContain("persist: live runtime");
      expect(frame).toContain("a set as default");

      // Mockup 09 footer: one key commits the picked model as the default.
      act(() => setup.mockInput.pressKey("a"));
      await setup.flush();
      frame = setup.captureCharFrame();
      expect(frame).toContain("persist: global");
      // The alias is consumed — it never leaks into the filter box.
      expect(frame).not.toContain("filter: a");

      // ctrl+g keeps working as the discoverable spelling of the same toggle.
      act(() => setup.mockInput.pressKey("g", { ctrl: true }));
      await setup.flush();
      expect(setup.captureCharFrame()).toContain("persist: live runtime");
    } finally {
      act(() => setup.renderer.destroy());
    }
  });

  it("shows the a-set-as-default hint in the two-pane header", async () => {
    const request = vi.fn(async (method: string) =>
      method === "model.options"
        ? options
        : { models: ["k3"], source: "remote" },
    );
    const { setup } = await renderPicker(request, vi.fn(), {
      allowPersistGlobal: true,
      width: 100,
    });

    try {
      await flushPromises(setup);
      expect(setup.captureCharFrame()).toContain("a set as default");
    } finally {
      act(() => setup.renderer.destroy());
    }
  });
});

describe("OpenTUI model picker — mockup 09 profile health", () => {
  it("marks a profile whose discovery failed offline before you select it", async () => {
    const request = vi.fn(
      async (method: string, params?: Record<string, unknown>) => {
        if (method === "model.options") return options;
        if (method === "model.models") {
          return params?.profile_name === "openai-work"
            ? Promise.reject(new Error("connection refused"))
            : Promise.resolve({
                models: ["kimi-for-coding"],
                source: "remote",
              });
        }
        return Promise.reject(new Error(`unexpected request: ${method}`));
      },
    );
    const { setup } = await renderPicker(request, vi.fn(), { width: 100 });

    try {
      await flushPromises(setup);
      await flushPromises(setup, 10);

      // Focus starts on the current profile; moving up lands on openai-work,
      // whose discovery fails. Only what discovery already knew is surfaced.
      // Cursor movement debounces discovery (DISCOVERY_DEBOUNCE_MS), so the
      // wait has to outlast the settle time the real picker uses.
      act(() => setup.mockInput.pressArrow("up"));
      await flushPromises(setup);
      await flushPromises(setup, 250);

      const frame = setup.captureCharFrame();
      // Red ✗ + right-side offline tag on the failed row, grayed name;
      // healthy rows keep their existing marks.
      // One dot shape, three colours: the glyph no longer changes with the
      // state, only its voice does — same rule as every other list.
      expect(frame).toContain("● OpenAI work");
      expect(frame).toContain("offline");
      expect(frame).toContain("● Kimi work");
      expect(
        request.mock.calls.some(
          ([method, params]) =>
            method === "model.models" &&
            params?.profile_name === "openai-work",
        ),
      ).toBe(true);
    } finally {
      act(() => setup.renderer.destroy());
    }
  });
});
