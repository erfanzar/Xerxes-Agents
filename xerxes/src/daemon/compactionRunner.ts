// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";

import {
  CompactionResponseShapeError,
  createCompactionAgent,
  type CompactionCompletionPort,
} from "../agents/compactionAgent.js";
import { DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS } from "../context/compactionProvisioner.js";
import type { ContextMessage } from "../context/compressor.js";
import { estimateContextTokens } from "../context/windowUsage.js";
import { closeLlmClient, completeLlm, type LlmClient } from "../llms/client.js";
import { classifyError, ErrorKind } from "../runtime/errorClassifier.js";

/** Auto-compact once the estimated context usage reaches this fraction of the prompt budget. */
export const DEFAULT_AUTO_COMPACT_THRESHOLD = 0.8;

/**
 * Summary token budgets tried in order: the default, then a half, then a
 * quarter. A compaction call that overflows the window is the one failure a
 * smaller summary can fix, and there are only ever two retries — a third
 * attempt at a quarter budget is not going to fit either.
 */
export const COMPACTION_SUMMARY_BUDGETS = [
  DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS,
  Math.floor(DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS / 2),
  Math.floor(DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS / 4),
] as const;

export function normalizeCompactionThreshold(value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    return 0;
  }
  return Math.min(value, 1);
}

/**
 * Token count at which a context is due for compaction, or 0 when compaction
 * is disabled. Main sessions and delegated children both read the threshold
 * here so a child cannot silently run against a different — or missing — one.
 */
export function compactionThresholdTokens(
  promptBudget: number,
  threshold: number,
): number {
  const normalized = normalizeCompactionThreshold(threshold);
  if (normalized <= 0 || !Number.isFinite(promptBudget) || promptBudget <= 0) {
    return 0;
  }
  return Math.floor(promptBudget * normalized);
}

/**
 * A response shape that holds no text will never hold text: retrying it burns
 * a full-window provider call to reach the same conclusion. A transport
 * failure is the opposite, and a context overflow is precisely what a smaller
 * summary budget exists for.
 */
export function compactionAttemptIsRetryable(error: unknown): boolean {
  if (error instanceof CompactionResponseShapeError) {
    return false;
  }
  const classified = classifyError(error);
  return classified.kind === ErrorKind.CONTEXT_OVERFLOW || classified.retryable;
}

/** Adapt a streaming LLM client to the compaction agent's one-shot completion port. */
export function compactionCompletionPort(
  client: LlmClient,
  model: string,
): CompactionCompletionPort {
  return async (request) => {
    const result = await completeLlm(client, {
      model,
      messages: [{ role: "user", content: request.prompt }],
      maxTokens: request.maxTokens,
      temperature: request.temperature,
    });
    return result.content;
  };
}

/**
 * A completion port that builds its client on first use.
 *
 * Compaction frequently decides it has nothing to do without ever asking a
 * provider for a summary: the agent returns the transcript untouched for fewer
 * than two messages, and again when the provisioner finds no compactable window.
 * Constructing the client up front therefore made `/compact` and
 * `session.compress` require a usable provider in order to answer "nothing to
 * compact" — so on a fresh install, where the active profile is the built-in
 * `claude-code` entry that has no client adapter, both failed outright instead of
 * reporting a clean no-op.
 *
 * Deferring means a provider that cannot be constructed surfaces only on the path
 * that genuinely needs one, where it is reported as a compaction failure like any
 * other provider error. The factory runs at most once; its rejection is cached so
 * a retry loop cannot turn one misconfiguration into repeated construction.
 */
export function lazyCompactionCompletionPort(
  createClient: () => LlmClient,
  model: string,
): LazyCompactionPort {
  type Resolved =
    | { readonly error: unknown; readonly ok: false }
    | { readonly client: LlmClient; readonly ok: true };
  let resolved: Resolved | undefined;
  return {
    port: async (request) => {
      if (resolved === undefined) {
        try {
          resolved = { ok: true, client: createClient() };
        } catch (error) {
          resolved = { ok: false, error };
        }
      }
      if (!resolved.ok) throw resolved.error;
      return compactionCompletionPort(resolved.client, model)(request);
    },
    // Closing has to be the port's job rather than the caller's, because the
    // caller can no longer see whether a client was ever built. A port that was
    // never used, or whose construction failed, has nothing to release.
    close: async () => {
      if (resolved?.ok === true) await closeLlmClient(resolved.client);
    },
  };
}

/** A deferred completion port paired with the release of whatever it built. */
export interface LazyCompactionPort {
  close(): Promise<void>;
  readonly port: CompactionCompletionPort;
}

/** What a completed compaction records on the session it rewrote. */
export interface CompactionStamp {
  /** Why the pre-compaction transcript could not be archived, when it could not. */
  readonly archive_error?: string;
  readonly archive_path?: string;
  readonly compacted_at: string;
  readonly messages_summarized: number;
  readonly reason: string;
  readonly tokens_after: number;
  readonly tokens_before: number;
}

export interface CompactMessagesRequest {
  /**
   * Sidecar the pre-compaction transcript is appended to before the caller
   * swaps in the summary. Omitted only where no transcript path exists;
   * everywhere else it is what keeps compaction from destroying history.
   */
  readonly archivePath?: string;
  readonly completion: CompactionCompletionPort;
  readonly messages: readonly ContextMessage[];
  readonly model: string;
  /** Origin recorded in the archive and the stamp: `compact`, `auto`, `subagent`. */
  readonly reason: string;
  readonly summaryBudgets?: readonly number[];
  /** Leave the transcript alone below this many tokens; omit to always attempt. */
  readonly thresholdTokens?: number;
}

export type CompactionSkipReason = "below-threshold" | "failed" | "unchanged";

export type CompactionOutcome =
  | {
    readonly compacted: true;
    readonly messages: ContextMessage[];
    /** Messages the summary replaced; anything appended later is the caller's to preserve. */
    readonly originalCount: number;
    readonly stamp: CompactionStamp;
  }
  | {
    readonly compacted: false;
    readonly error?: string;
    readonly reason: CompactionSkipReason;
  };

/**
 * Summarize a transcript that has grown past its threshold, archiving what it
 * replaces.
 *
 * The single implementation shared by the main session and by delegated
 * children: children used to have no compaction at all, so a queued follow-up
 * or a retry continuation against a full conversation died as an opaque
 * provider error the parent had to retry blind.
 */
export async function compactMessagesIfNeeded(
  request: CompactMessagesRequest,
): Promise<CompactionOutcome> {
  const original = [...request.messages];
  const tokensBefore = estimateContextTokens(original, { model: request.model });
  if (
    request.thresholdTokens !== undefined &&
    tokensBefore < request.thresholdTokens
  ) {
    return { compacted: false, reason: "below-threshold" };
  }
  let lastError: unknown;
  // One lever, shrinking: the summary's token budget. Compaction is atomic —
  // the agent returns the original transcript on failure and the caller only
  // swaps on success — so each attempt starts from the same clean state.
  for (const summaryMaxTokens of request.summaryBudgets ?? COMPACTION_SUMMARY_BUDGETS) {
    let compacted: readonly ContextMessage[];
    try {
      compacted = await createCompactionAgent({
        completion: request.completion,
        model: request.model,
        summaryMaxTokens,
      }).summarizeMessages(original);
    } catch (error) {
      lastError = error;
      if (!compactionAttemptIsRetryable(error)) break;
      continue;
    }
    if (isUnchanged(original, compacted)) {
      return { compacted: false, reason: "unchanged" };
    }
    const tokensAfter = estimateContextTokens(compacted, { model: request.model });
    const archive = await archivePreCompaction(request.archivePath, {
      archived_at: new Date().toISOString(),
      messages: original,
      model: request.model,
      reason: request.reason,
      tokens_after: tokensAfter,
      tokens_before: tokensBefore,
    });
    // When the caller supplied an archive path, replacing the transcript is
    // safe only after the original was durably appended. Returning the summary
    // on an archive failure would make that history unrecoverable.
    if (archive.error !== undefined) {
      return { compacted: false, reason: "failed", error: archive.error };
    }
    return {
      compacted: true,
      messages: [...compacted],
      originalCount: original.length,
      stamp: {
        ...(archive.error === undefined ? {} : { archive_error: archive.error }),
        ...(archive.path === undefined ? {} : { archive_path: archive.path }),
        compacted_at: new Date().toISOString(),
        messages_summarized: Math.max(0, original.length - compacted.length),
        reason: request.reason,
        tokens_after: tokensAfter,
        tokens_before: tokensBefore,
      },
    };
  }
  return { compacted: false, reason: "failed", error: errorMessage(lastError) };
}

/**
 * Sidecar holding transcripts that compaction replaced, beside the session
 * file. It is not named `<id>.json`, so transcript listing and resume never
 * mistake an archive for a session.
 */
export function precompactArchivePath(transcriptPath: string): string {
  return `${transcriptPath.replace(/\.json$/u, "")}.precompact.jsonl`;
}

/** Archive path for a session id stored in `directory`, matching the transcript layout. */
export function precompactArchivePathFor(
  directory: string,
  sessionId: string,
): string {
  return precompactArchivePath(join(directory, `${sessionId}.json`));
}

interface PreCompactionRecord {
  readonly archived_at: string;
  readonly messages: readonly ContextMessage[];
  readonly model: string;
  readonly reason: string;
  readonly tokens_after: number;
  readonly tokens_before: number;
}

/**
 * Append the transcript about to be replaced.
 *
 * Append, never overwrite: a session compacts repeatedly and the second pass
 * would otherwise erase the only surviving copy of the first pass's history.
 * A failed write is reported rather than thrown so the caller can abort the
 * replacement and return a typed compaction failure while retaining history.
 */
async function archivePreCompaction(
  path: string | undefined,
  record: PreCompactionRecord,
): Promise<{ readonly error?: string; readonly path?: string }> {
  if (path === undefined) return {};
  try {
    await mkdir(dirname(path), { recursive: true });
    await appendFile(path, `${JSON.stringify(record)}\n`, "utf8");
    return { path };
  } catch (error) {
    return { error: errorMessage(error) };
  }
}

function isUnchanged(
  original: readonly ContextMessage[],
  compacted: readonly ContextMessage[],
): boolean {
  return (
    compacted.length === original.length &&
    compacted.every(
      (message, index) => JSON.stringify(message) === JSON.stringify(original[index]),
    )
  );
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
