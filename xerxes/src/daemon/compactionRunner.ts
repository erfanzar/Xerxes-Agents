// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { getTurnOutcome } from '../types/turnOutcome.js';
import { appendFile, mkdir } from "node:fs/promises";
import { dirname, join } from "node:path";

import {
  CompactionResponseShapeError,
  DEFAULT_COMPACTION_REQUEST_TOKENS,
  createCompactionAgent,
  type CompactionCompletionPort,
} from "../agents/compactionAgent.js";
import { DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS } from "../context/compactionProvisioner.js";
import {
  COMPACTION_ARCHIVE_NOTE,
  COMPACTION_REFERENCE_PREFIX,
  isCompactionSummaryMessage,
  type ContextMessage,
} from "../context/compressor.js";
import { shedToolResults } from "../context/toolResultPruner.js";
import { estimateContextTokens } from "../context/windowUsage.js";
import {
  closeLlmClient,
  completeLlm,
  CompletionDeadlineError,
  type LlmClient,
} from "../llms/client.js";
import { classifyError, ErrorKind } from "../runtime/errorClassifier.js";
import { catalogReasoningLevels, fallbackReasoningLevels } from "../llms/reasoningLevels.js";
import { detectProvider, isProviderName, type ProviderName } from "../llms/providerRegistry.js";
import { ConfigurationError } from "../core/errors.js";

/** Auto-compact once the estimated context usage reaches this fraction of the prompt budget. */
export const DEFAULT_AUTO_COMPACT_THRESHOLD = 0.8;

/**
 * Wall-clock deadline for one compaction completion call.
 *
 * Pre-turn auto-compaction runs ahead of every due turn, so a compaction call
 * that never settles would hang the user's turn indefinitely with no terminal
 * event and no way to retry. The deadline is enforced twice: as an
 * `AbortSignal.timeout` handed to the transport, which well-behaved providers
 * observe, and as a racing timer in `withCompletionDeadline`, which bounds
 * even a client that ignores its signal. A deadline miss classifies as a
 * timeout (retryable) exactly like a provider that fails slowly, but it can
 * no longer hang.
 */
export const COMPACTION_COMPLETION_TIMEOUT_MS = 180_000;

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

/** Largest summary any compaction asks for, whatever the window. */
export const MAX_COMPACTION_SUMMARY_TOKENS = 32_768;
/** Mid-turn rescues run with the window already full; their summary stays smaller. */
export const MIN_OVERFLOW_SUMMARY_TOKENS = 2_048;
export const MAX_OVERFLOW_SUMMARY_TOKENS = 8_192;

/**
 * Summary budgets that grow with the prompt budget instead of being the same
 * for a 128K and a 1M window: 2% of the budget before a turn (8K floor, 32K
 * ceiling), 0.8% mid-turn (2K floor, 8K ceiling). A known output ceiling
 * caps every attempt, since a summary larger than the model may write is cut
 * off. Fallback attempts halve, then quarter, as before.
 */
export function compactionSummaryBudgets(
  promptBudget: number | undefined,
  options: { readonly maxOutputTokens?: number; readonly midTurn?: boolean } = {},
): readonly number[] {
  const midTurn = options.midTurn === true;
  const floor = midTurn ? MIN_OVERFLOW_SUMMARY_TOKENS : DEFAULT_COMPACTION_SUMMARY_MAX_TOKENS;
  const ceiling = midTurn ? MAX_OVERFLOW_SUMMARY_TOKENS : MAX_COMPACTION_SUMMARY_TOKENS;
  const known = promptBudget !== undefined && Number.isFinite(promptBudget) && promptBudget > 0;
  let first = known ? Math.min(ceiling, Math.max(floor, Math.floor(promptBudget * (midTurn ? 0.008 : 0.02)))) : floor;
  const output = options.maxOutputTokens;
  if (output !== undefined && Number.isFinite(output) && output > 0) first = Math.min(first, Math.max(512, Math.floor(output * 0.9)));
  if (midTurn) return [first];
  return [first, Math.max(512, Math.floor(first / 2)), Math.max(512, Math.floor(first / 4))];
}

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
  /** Wall-clock deadline for the completion call; omit for the default. */
  timeoutMs: number = COMPACTION_COMPLETION_TIMEOUT_MS,
  /** External cancellation owned by the command or turn invoking compaction. */
  signal?: AbortSignal,
  /** The routed profile's provider, when the caller knows it; see {@link compactionHintProvider}. */
  provider?: string,
): CompactionCompletionPort {
  return async (request) => {
    throwIfCancelled(signal)
    const hintProvider = compactionHintProvider(model, provider);
    const levels = catalogReasoningLevels(model, hintProvider) ?? fallbackReasoningLevels(hintProvider);
    const thinking = levels.shape === 'effort' && levels.levels.some(level => level.effort === 'low')
      ? { effort: 'low' } : undefined;
    const completionSignal = combinedCompletionSignal(timeoutMs, signal)
    const completion = isFiniteDeadline(timeoutMs)
      ? completeLlm(
        client,
        {
          model,
          messages: [{ role: "user", content: request.prompt }],
          maxTokens: request.maxTokens,
          temperature: request.temperature,
          ...(thinking ? { thinking } : {}),
        },
        completionSignal,
        { timeoutMs, preferStream: true },
      )
      : completeLlm(
        client,
        {
          model,
          messages: [{ role: "user", content: request.prompt }],
          maxTokens: request.maxTokens,
          temperature: request.temperature,
          ...(thinking ? { thinking } : {}),
        },
        completionSignal,
        { preferStream: true },
      )
    let result: Awaited<typeof completion>;
    try {
      result = await withCompletionDeadline(
        completion,
        timeoutMs,
        signal,
      );
    } catch (error) {
      // Keep the compaction port's established timeout contract even when
      // completeLlm's internal deadline wins the race.
      if (error instanceof CompletionDeadlineError) {
        throw compactionTimeoutError(timeoutMs);
      }
      throw error;
    }
    throwIfCancelled(signal)
    if (result.finishReason && !['stop', 'completed', 'end_turn'].includes(result.finishReason)) {
      throw new CompactionResponseShapeError(`summary did not finish (${result.finishReason}); original history retained`);
    }
    return result.content;
  };
}

/** Cooperative-cancellation half of the deadline: `undefined` disables it. */
function deadlineSignal(timeoutMs: number): AbortSignal | undefined {
  if (!isFiniteDeadline(timeoutMs)) return undefined;
  return AbortSignal.timeout(timeoutMs);
}

/** Combine the compaction deadline with host cancellation before entering the provider. */
function combinedCompletionSignal(timeoutMs: number, signal?: AbortSignal): AbortSignal | undefined {
  const deadline = deadlineSignal(timeoutMs)
  if (deadline === undefined) return signal
  if (signal === undefined) return deadline
  return AbortSignal.any([deadline, signal])
}

/**
 * Bound a completion promise even when its client ignores the abort signal.
 *
 * The losing branch settles quietly: its rejection is routed into an
 * already-settled race, so neither path produces an unhandled rejection.
 */
async function withCompletionDeadline<T>(
  promise: Promise<T>,
  timeoutMs: number,
  signal?: AbortSignal,
): Promise<T> {
  if (!isFiniteDeadline(timeoutMs) && signal === undefined) return promise;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let onAbort: (() => void) | undefined;
  try {
    return await new Promise<T>((resolve, reject) => {
      if (isFiniteDeadline(timeoutMs)) {
        timer = setTimeout(() => reject(compactionTimeoutError(timeoutMs)), timeoutMs);
      }
      if (signal !== undefined) {
        onAbort = () => reject(compactionAbortError());
        if (signal.aborted) onAbort();
        else signal.addEventListener("abort", onAbort, { once: true });
      }
      promise.then(resolve, reject);
    });
  } finally {
    if (timer !== undefined) clearTimeout(timer);
    if (signal !== undefined && onAbort !== undefined) signal.removeEventListener("abort", onAbort);
  }
}

function throwIfCancelled(signal?: AbortSignal): void {
  if (signal?.aborted) throw compactionAbortError();
}

function compactionAbortError(): Error {
  const error = new Error("compaction completion cancelled");
  error.name = "AbortError";
  return error;
}

function isFiniteDeadline(timeoutMs: number): boolean {
  return Number.isFinite(timeoutMs) && timeoutMs > 0;
}

/** Named so error classification records the miss as a retryable timeout. */
function compactionTimeoutError(timeoutMs: number): Error {
  const error = new Error(
    `compaction completion exceeded its ${timeoutMs}ms deadline`,
  );
  error.name = "TimeoutError";
  return error;
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
/**
 * The provider whose reasoning levels shape the summary's low-effort hint. The
 * client is already routed — by the session's profile — so this only picks a
 * hint. A model id the bare registry cannot route (an OpenRouter vendor id like
 * `mistralai/mistral-large-4-0`) failed every compaction with "unknown provider
 * prefix"; it now goes without a hint, and the provider applies its default.
 */
function compactionHintProvider(model: string, provider: string | undefined): ProviderName | undefined {
  if (provider !== undefined) return isProviderName(provider) ? provider : undefined;
  try {
    return detectProvider(model);
  } catch (error) {
    if (error instanceof ConfigurationError) return undefined;
    throw error;
  }
}

export function lazyCompactionCompletionPort(
  createClient: () => LlmClient,
  model: string,
  /** Wall-clock deadline forwarded to every completion; omit for the default. */
  timeoutMs: number = COMPACTION_COMPLETION_TIMEOUT_MS,
  /** External cancellation owned by the command or turn invoking compaction. */
  signal?: AbortSignal,
): LazyCompactionPort {
  type Resolved =
    | { readonly error: unknown; readonly ok: false }
    | { readonly client: LlmClient; readonly ok: true };
  let resolved: Resolved | undefined;
  return {
    port: async (request) => {
      throwIfCancelled(signal)
      if (resolved === undefined) {
        try {
          resolved = { ok: true, client: createClient() };
        } catch (error) {
          resolved = { ok: false, error };
        }
      }
      if (!resolved.ok) throw resolved.error;
      throwIfCancelled(signal)
      return compactionCompletionPort(resolved.client, model, timeoutMs, signal)(request);
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
  /** Present when the model summary failed and older tool outputs were shed instead. */
  readonly fallback?: "shed-tool-results";
  /** Why the model summary failed, when the fallback ran. */
  readonly summary_error?: string;
  readonly messages_summarized: number;
  readonly reason: string;
  readonly tokens_after: number;
  readonly tokens_before: number;
}

export interface CompactMessagesRequest {
  readonly maxContextTokens?: number;
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
  /**
   * When every summary attempt fails, shed the oldest tool outputs (the
   * newest stay verbatim) down to half of `maxContextTokens` instead of
   * failing. Requires `archivePath` and `maxContextTokens`: the original is
   * archived first, so nothing is lost and `session_search` can read it back.
   * Automatic paths set this; an explicit `/compact` reports the failure.
   */
  readonly deterministicFallback?: boolean;
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
  // Context-overflow retries shrink the summary budget; timeout recovery stays
  // inside the agent's failed segment. Compaction is atomic —
  // the agent returns the original transcript on failure and the caller only
  // swaps on success — so each attempt starts from the same clean state.
  const summaryBudgets = request.summaryBudgets ?? COMPACTION_SUMMARY_BUDGETS;
  for (let attempt = 0; attempt < summaryBudgets.length; attempt++) {
    const summaryMaxTokens = summaryBudgets[attempt]!;
    let compacted: readonly ContextMessage[];
    try {
      compacted = await createCompactionAgent({
        completion: request.completion,
        model: request.model,
        summaryMaxTokens,
        maxRequestTokens: DEFAULT_COMPACTION_REQUEST_TOKENS,
        ...(request.maxContextTokens === undefined ? {} : { maxContextTokens: request.maxContextTokens }),
      }).summarizeMessages(original);
    } catch (error) {
      lastError = error;
      // The agent already retried the failed segment with smaller inputs.
      // Restarting here would replay every successfully summarized segment.
      if (classifyError(error).kind === ErrorKind.TIMEOUT) break;
      if (!compactionAttemptIsRetryable(error)) break;
      continue;
    }
    if (isUnchanged(original, compacted)) {
      return { compacted: false, reason: "unchanged" };
    }
    let tokensAfter = estimateContextTokens(compacted, { model: request.model });
    // Point the model at the archive, but only when there is one, and never
    // at the cost of a compaction that no longer shrinks the transcript.
    if (request.archivePath !== undefined) {
      const noted = withArchiveNote(compacted);
      if (noted !== compacted) {
        const notedTokens = estimateContextTokens(noted, { model: request.model });
        if (notedTokens < tokensBefore) {
          compacted = noted;
          tokensAfter = notedTokens;
        }
      }
    }
    const archive = await archivePreCompaction(request.archivePath, {
      archived_at: new Date().toISOString(),
      messages: original.map(message => {
        const outcome = getTurnOutcome(message);
        return outcome ? { ...message, turn_outcome: outcome } : message;
      }),
      model: request.model,
      reason: request.reason,
      ...retainedTail(compacted),
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
  const summaryError = errorMessage(lastError);
  const fallback = await shedFallback(request, original, tokensBefore, summaryError);
  return fallback ?? { compacted: false, reason: "failed", error: summaryError };
}

/** The summary message with the archive note after its prefix; the same array when nothing changes. */
function withArchiveNote(messages: readonly ContextMessage[]): readonly ContextMessage[] {
  let changed = false;
  const noted = messages.map(message => {
    if (!isCompactionSummaryMessage(message) || typeof message.content !== "string") return message;
    if (message.content.includes(COMPACTION_ARCHIVE_NOTE) || !message.content.startsWith(COMPACTION_REFERENCE_PREFIX)) return message;
    changed = true;
    return { ...message, content: `${COMPACTION_REFERENCE_PREFIX}\n${COMPACTION_ARCHIVE_NOTE}${message.content.slice(COMPACTION_REFERENCE_PREFIX.length)}` };
  });
  return changed ? noted : messages;
}

/** Newest tool results the deterministic fallback never sheds. */
const FALLBACK_KEEP_RECENT_TOOL_RESULTS = 8;

async function shedFallback(
  request: CompactMessagesRequest,
  original: readonly ContextMessage[],
  tokensBefore: number,
  summaryError: string,
): Promise<CompactionOutcome | undefined> {
  if (!request.deterministicFallback || request.archivePath === undefined || request.maxContextTokens === undefined) {
    return undefined;
  }
  const count = (messages: readonly ContextMessage[]) => estimateContextTokens(messages, { model: request.model });
  const shed = shedToolResults(original, {
    budgetTokens: Math.floor(request.maxContextTokens / 2),
    keepRecent: FALLBACK_KEEP_RECENT_TOOL_RESULTS,
    count,
  });
  if (shed.shedCount === 0) return undefined;
  const tokensAfter = count(shed.messages);
  if (tokensAfter >= tokensBefore) return undefined;
  const archive = await archivePreCompaction(request.archivePath, {
    archived_at: new Date().toISOString(),
    messages: original.map(message => {
      const outcome = getTurnOutcome(message);
      return outcome ? { ...message, turn_outcome: outcome } : message;
    }),
    model: request.model,
    reason: `${request.reason}:shed-fallback`,
    // No summary: every message is retained, only older outputs are emptied.
    retained_messages: shed.messages.length,
    tokens_after: tokensAfter,
    tokens_before: tokensBefore,
  });
  if (archive.error !== undefined) return { compacted: false, reason: "failed", error: archive.error };
  return {
    compacted: true,
    messages: shed.messages,
    originalCount: original.length,
    stamp: {
      ...(archive.path === undefined ? {} : { archive_path: archive.path }),
      compacted_at: new Date().toISOString(),
      fallback: "shed-tool-results",
      messages_summarized: 0,
      reason: request.reason,
      summary_error: summaryError,
      tokens_after: tokensAfter,
      tokens_before: tokensBefore,
    },
  };
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
  /**
   * Messages kept after the summary. History views align the next window on
   * exactly this tail: without it, an undo inside the tail followed by a new
   * turn could not be told apart from new messages, and the undone turn came
   * back (duplicated) in the stitched history.
   */
  readonly retained_messages?: number;
  readonly tokens_after: number;
  readonly tokens_before: number;
}

function retainedTail(compacted: readonly ContextMessage[]): { readonly retained_messages?: number } {
  const summary = compacted.findLastIndex(isCompactionSummaryMessage);
  return summary < 0 ? {} : { retained_messages: compacted.length - summary - 1 };
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
