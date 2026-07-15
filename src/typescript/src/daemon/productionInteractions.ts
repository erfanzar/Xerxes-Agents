// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { join } from "node:path";

import { ApprovalStore } from "../security/approvals.js";
import { DaemonInteractionBoard } from "./interactions.js";
import { xerxesHome } from "./paths.js";

export interface ProductionInteractionBoardOptions {
  /** Explicit environment boundary, primarily for isolated daemon hosts and tests. */
  readonly environment?: NodeJS.ProcessEnv;
  /** Persistence failures stay observable without invalidating an already-made live decision. */
  readonly onApprovalStoreError?: (error: unknown) => void;
}

/** Location shared by daemon restarts for user-selected `Always allow` decisions. */
export function daemonApprovalsPath(
  environment: NodeJS.ProcessEnv = process.env,
): string {
  return join(xerxesHome(environment), "approvals.json");
}

/** Build the production interaction board with owner-only, atomic approval persistence. */
export function createProductionInteractionBoard(
  options: ProductionInteractionBoardOptions = {},
): DaemonInteractionBoard {
  const environment = options.environment ?? process.env;
  return new DaemonInteractionBoard({
    approvalStore: new ApprovalStore({
      persistencePath: daemonApprovalsPath(environment),
    }),
    ...(options.onApprovalStoreError === undefined
      ? {}
      : { onApprovalStoreError: options.onApprovalStoreError }),
  });
}
