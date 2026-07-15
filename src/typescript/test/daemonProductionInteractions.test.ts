// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from "bun:test";
import { mkdtemp, readFile, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  createProductionInteractionBoard,
  daemonApprovalsPath,
} from "../src/daemon/productionInteractions.js";
import type { PermissionRequest } from "../src/streaming/events.js";

test("production interaction boards persist Always allow under XERXES_HOME across daemon restarts", async () => {
  const directory = await mkdtemp(join(tmpdir(), "xerxes-production-approvals-"));
  const environment = { ...process.env, XERXES_HOME: directory };
  const request: PermissionRequest = {
    requestId: "permission-first",
    description: "write an isolated fixture",
    inputs: {},
    toolCall: {
      id: "call-first",
      type: "function",
      function: {
        name: "WriteFile",
        arguments: { file_path: "/private/path", content: "do-not-persist" },
      },
    },
  };

  try {
    const firstBoard = createProductionInteractionBoard({ environment });
    const firstDecision = firstBoard
      .permissionBroker("session-first")
      .request(request);
    expect(firstBoard.respondPermission(request.requestId, "always")).toBeTrue();
    await expect(firstDecision).resolves.toBe("approve_for_session");

    const path = daemonApprovalsPath(environment);
    expect(path).toBe(join(directory, "approvals.json"));
    expect((await stat(path)).mode & 0o777).toBe(0o600);
    expect(await readFile(path, "utf8")).not.toContain("do-not-persist");
    expect(await readFile(path, "utf8")).not.toContain("/private/path");

    const restartedBoard = createProductionInteractionBoard({ environment });
    const remembered = await restartedBoard
      .permissionBroker("session-after-restart")
      .request({ ...request, requestId: "permission-after-restart" });
    expect(remembered).toBe("approve");
    expect(restartedBoard.pendingPermissionIds()).toEqual([]);
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});
