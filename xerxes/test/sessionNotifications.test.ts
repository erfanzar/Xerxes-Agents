// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { describe, expect, test } from "bun:test";

import {
  MAX_PENDING_SESSION_NOTIFICATIONS,
  queueSessionNotification,
  readSessionNotifications,
  takeSessionNotifications,
} from "../src/daemon/sessionNotifications.js";

describe("durable session notifications", () => {
  test("queued notices survive until a client attaches and drain exactly once", () => {
    const metadata: Record<string, unknown> = {};
    // Simulate a background task finishing while no client is attached.
    queueSessionNotification(metadata, {
      at: 1,
      level: "info",
      message: "Background task bg-1 finished.",
    });
    expect(readSessionNotifications(metadata)).toHaveLength(1);

    // The attach drains the backlog…
    const drained = takeSessionNotifications(metadata);
    expect(drained.map(notice => notice.level)).toEqual(["info"]);
    expect(drained[0]?.message).toBe("Background task bg-1 finished.");
    // …and never redelivers it.
    expect(takeSessionNotifications(metadata)).toEqual([]);
  });

  test("failures and successes share one bounded ring, newest wins at the cap", () => {
    const metadata: Record<string, unknown> = {};
    for (let index = 0; index < MAX_PENDING_SESSION_NOTIFICATIONS + 3; index += 1) {
      queueSessionNotification(metadata, {
        at: index,
        level: index % 2 === 0 ? "error" : "info",
        message: `task ${index}`,
      });
    }
    const pending = readSessionNotifications(metadata);
    expect(pending).toHaveLength(MAX_PENDING_SESSION_NOTIFICATIONS);
    expect(pending[0]?.message).toBe(`task ${MAX_PENDING_SESSION_NOTIFICATIONS - MAX_PENDING_SESSION_NOTIFICATIONS + 3}`);
    expect(pending.at(-1)?.message).toBe(`task ${MAX_PENDING_SESSION_NOTIFICATIONS + 2}`);
  });

  test("corrupt ledger entries are ignored instead of breaking session.open", () => {
    const metadata: Record<string, unknown> = {
      pending_notifications: ["junk", 42, { at: 3, level: "warning", message: "real" }],
    };
    expect(takeSessionNotifications(metadata)).toEqual([
      { at: 3, level: "warning", message: "real" },
    ]);
  });
});
