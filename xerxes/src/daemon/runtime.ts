// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { join, resolve } from "node:path";

import { ValidationError } from "../core/errors.js";
import { disarmGoal } from "../runtime/goalDomain.js";
import { normalizeInteractionMode } from "../runtime/interactionModes.js";
import {
  appendContextDelta,
  contextDeltaFor,
} from "../runtime/contextDeltas.js";
import {
  DaemonTranscriptStore,
  looksLikeSessionId,
  transcriptHasHistory,
  type DaemonTranscript,
  type DaemonTranscriptEntry,
  type RawMessage,
  type TranscriptMessageJournalAppend,
} from "../session/daemonTranscript.js";
import type { JsonRpcPayload } from "../protocol/jsonRpc.js";
import {
  FILE_READS_METADATA_KEY,
  fileReadsForMetadata,
  hydrateFileReadsFromMetadata,
} from "../tools/fileState.js";
import { processAtMentions } from "./atMentions.js";
import { imageUrlContentParts, type TurnImage } from "./images.js";
import { xerxesHome } from "./paths.js";
import { displayTitle } from "./titleGenerator.js";
import type { DaemonInteractionBoard } from "./interactions.js";
import {
  claimDirectSubagentConversation,
  isSubagentConversationActive,
} from "./subagentConversations.js";

export const DAEMON_PROTOCOL_VERSION = 35;
export const XERXES_VERSION = "0.3.6";
export const BUN_DAEMON_BUILD_ID =
  process.env.XERXES_DAEMON_BUILD_ID?.trim() || `bun-runtime-v${XERXES_VERSION}`;
/** Maximum hints retained between active-turn provider/tool boundaries. */
export const MAX_ACTIVE_TURN_STEERS = 64;

export interface DaemonEvent {
  readonly payload: JsonRpcPayload;
  readonly type: string;
}

/**
 * A transcript record retained by the daemon. It intentionally accepts
 * provider-specific fields so legacy Python sessions survive a Bun resume.
 */
export type DaemonTranscriptMessage = RawMessage & {
  readonly content?: unknown;
  readonly role: string;
  readonly text?: string;
};

export interface DaemonSession {
  activeTurnId: string;
  agentId: string;
  /** False when imported history predates exact cumulative API-call accounting. */
  apiCallsComplete?: boolean;
  cancelRequested: boolean;
  cwd: string;
  extra: Record<string, unknown>;
  readonly id: string;
  interactionMode: string;
  lastActive: number;
  /** Transient visible turn state used when the TUI switches back mid-turn. */
  inflightUser?: string;
  inflightAssistant?: string;
  /** Epoch ms the in-flight turn began; lets a reattach keep elapsed continuity. */
  inflightStartedAt?: number;
  /** Thinking the in-flight turn produced so far (tail-bounded). */
  inflightThinking?: string;
  /**
   * Tool calls the in-flight turn produced so far, in call order. Runner-managed
   * sessions only synchronize session.messages at turn end, so without this a
   * mid-turn reattach would see a bare user line instead of the work so far.
   */
  inflightTools?: InflightToolSnapshot[];
  messages: DaemonTranscriptMessage[];
  metadata: Record<string, unknown>;
  /** Persisted revision and message boundary this in-memory copy was based on. */
  transcriptGeneration?: number;
  persistedMessageCount?: number;
  model: string;
  /**
   * True once this session owns its model explicitly — the user picked one
   * here, or it was restored from this session's own history.
   *
   * A global reload must not touch a pinned session: two open sessions are
   * allowed to run different models, and a resumed session must keep the model
   * its transcript was written with rather than adopting whatever the profile
   * last stored.
   */
  modelPinned?: boolean;
  /**
   * Reasoning effort this session runs at, when it owns one.
   *
   * Same reasoning as the model: two open sessions may legitimately want
   * different efforts, and a resumed conversation should continue at the
   * effort it was held at rather than adopting the daemon default.
   */
  reasoningEffort?: string;
  reasoningPinned?: boolean;
  /**
   * Permission mode this session runs under, when it owns one.
   *
   * Pinned like the model and effort so two sessions can run at different
   * trust levels — a throwaway one on accept-all while a session touching
   * something important stays on manual.
   */
  permissionMode?: string;
  permissionPinned?: boolean;
  planMode: boolean;
  /**
   * Provider-request scaffolding the turn runner assembles for this session:
   * the system prompt and tool schemas that ride every request but never
   * appear in the transcript. Cached here — never persisted — because pricing
   * the window from the messages alone under-reports it by the largest fixed
   * cost in the request, which is what drives auto-compaction too late.
   */
  requestScaffold?: {
    readonly systemPrompt?: string;
    readonly toolSchemas?: readonly Readonly<Record<string, unknown>>[];
  };
  readonly sessionKey: string;
  status: "idle" | "starting" | "waiting" | "working";
  /** Trusted dynamic system context. Never persisted in the transcript. */
  systemPromptAddendum?: string;
  thinkingContent: unknown[];
  toolExecutions: unknown[];
  /** Exact provider attempts, absent only for imported transcripts that predate this field. */
  totalApiCalls?: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  turnCount: number;
  /**
   * Session-scoped ultra mode: pins every turn to the maximum thinking
   * directive. Optional and deliberately NOT persisted: adding it to the
   * durable session record would force a wire-format migration for every
   * stored session, while an in-memory-only flag simply reads as
   * `undefined` (off) for old, resumed, and imported sessions.
   */
  ultraMode?: boolean;
  /** Whether cumulative token totals cover every provider attempt in this session. */
  usageComplete?: boolean;
  workspace: string;
}

/** Bounded snapshot of one in-flight tool call for mid-turn reattach payloads. */
export interface InflightToolSnapshot {
  /** Compact arguments preview, never the full payload. */
  arguments?: string;
  duration_ms?: number;
  /** Compact diagnostic when the call failed. */
  error?: string;
  id?: string;
  name: string;
  ok?: boolean;
}

export interface OpenSessionOptions {
  readonly cwd?: string;
  readonly model?: string;
  /** Only explicit resume requests may rehydrate a persisted transcript. */
  readonly resume?: boolean;
  /** Ephemeral trusted system context supplied by a host boundary. */
  readonly systemPromptAddendum?: string;
}

export interface SavedDaemonSession {
  readonly agentId: string;
  /** Canonical project directory the transcript ran in — the workspace key. */
  readonly cwd: string;
  readonly id: string;
  readonly key: string;
  /** Persisted session role; legacy transcripts default to main. */
  readonly kind: "main" | "subagent";
  readonly messageCount: number;
  readonly model?: string;
  readonly parentSessionId?: string;
  readonly path: string;
  /** False while a native child in this daemon still owns the transcript. */
  readonly resumable: boolean;
  readonly rootSessionId?: string;
  readonly status?: string;
  readonly subagentId?: string;
  readonly title: string;
  readonly turnCount: number;
  readonly updatedAt: string;
}

export interface SavedSessionListOptions {
  /** Include child transcripts alongside their selected root sessions. */
  readonly includeSubagents?: boolean;
  /** Select only root, only child, or both kinds of transcript. */
  readonly kind?: "all" | "main" | "subagent";
  /** Restrict history to the canonical project directory stored by each transcript. */
  readonly projectDirectory?: string;
}

export interface TurnRunner {
  /** True when the runner synchronizes complete agent state onto the session. */
  readonly managesSessionState?: boolean;
  /** Release cached per-session state when the daemon evicts the session. */
  dropSession?(sessionId: string): void;
  run(
    session: DaemonSession,
    text: string,
    signal: AbortSignal,
    controls?: TurnRunControls,
  ): AsyncIterable<DaemonEvent>;
}

/** Controls that the daemon supplies around a single active turn. */
export interface TurnRunControls {
  /** Drain user steering received since the last provider/tool boundary. */
  drainSteer?(): readonly string[];
  /** Text authored by the user before hidden attachment expansion. */
  readonly displayText?: string;
  /** Validated image attachments carried into the user message as content parts. */
  readonly images?: readonly TurnImage[];
  /**
   * Per-message crash journal for state-managing runners. When a runner owns
   * session.messages it must record each appended message so a crash between
   * tool calls does not lose the whole turn.
   */
  readonly journal?: TranscriptMessageJournalAppend;
  /**
   * Positive round number when this turn is an automatic goal continuation.
   *
   * Absent means a human opened the turn. The goal tools authorise against
   * this: only a human may create, edit, pause or resume a goal, while an
   * automatic round may additionally complete or block the one it belongs to.
   */
  readonly goalRound?: number;
}

export interface SubmitTurnOptions {
  /** Transcript text retained separately from the provider-facing prompt. */
  readonly displayText?: string;
  /** Validated image attachments for this turn's user message. */
  readonly images?: readonly TurnImage[];
  /**
   * Positive round number when the goal round driver opened this turn rather
   * than a person. Forwarded to {@link TurnRunControls.goalRound}.
   */
  readonly goalRound?: number;
}

export interface SubagentRetryRequest {
  /** Optional replacement instruction for the new attempt. */
  readonly message?: string;
  /** Connection session that requested the retry; informational. */
  readonly sessionKey?: string;
  /** Stable task id or name of the dead subagent to resume. */
  readonly task: string;
}

export interface SubagentRetryResult {
  readonly agent?: Readonly<Record<string, unknown>>;
  readonly error?: string;
  readonly ok: boolean;
  readonly [key: string]: unknown;
}

export interface DaemonRuntime {
  cancelAllTurns(): number;
  cancelTurn(sessionKey: string): boolean;
  /**
   * Optional first-class retry of a terminal subagent task under its stable
   * identity. Kept optional so test fakes and custom hosts stay
   * source-compatible; the server rejects `subagent.retry` when absent.
   */
  retrySubagent?(request: SubagentRetryRequest): Promise<SubagentRetryResult>;
  /** Optional persistent-session removal capability for hosts with native transcript storage. */
  deleteSavedSession?(sessionId: string): Promise<boolean>;
  /**
   * Optional persisted-transcript removal that leaves any live in-memory
   * session untouched. Explicit history-clearing flows (for example undoing
   * the last remaining turn) use it because routine saves never delete.
   */
  removeSavedTranscript?(sessionId: string): Promise<boolean>;
  evictSession(sessionKey: string): void;
  flushSessions(mode?: 'append' | 'rewrite'): Promise<void>;
  /** Reset optimistic persistence baselines after the saved store is wiped. */
  resetSavedTranscriptState?(): void;
  listSavedSessions(
    limit?: number,
    options?: SavedSessionListOptions,
  ): Promise<readonly SavedDaemonSession[]>;
  listSessions(): readonly DaemonSession[];
  /**
   * Optional per-message crash journal for a session. Kept optional so test
   * fakes and custom hosts stay source-compatible; a producer that has it can
   * record each persisted message as it is appended instead of relying on the
   * once-per-turn transcript write.
   */
  messageJournal?(sessionId: string): TranscriptMessageJournalAppend;
  openSession(
    sessionKey: string,
    agentId?: string,
    options?: OpenSessionOptions,
  ): Promise<DaemonSession>;
  reload(overrides?: JsonRpcPayload): JsonRpcPayload;
  setSessionMode(
    sessionKey: string,
    mode: string,
    planMode?: boolean,
  ): Promise<DaemonSession | undefined>;
  /** Select an agent preset only while the session is still blank. */
  selectSessionAgent?(
    sessionKey: string,
    agentId: string,
  ): Promise<DaemonSession | undefined>;
  /**
   * Pin one session to a model without disturbing any other session.
   *
   * Optional on the interface so existing runtimes stay source-compatible;
   * the server falls back to a global reload when a host does not implement it.
   */
  setSessionModel?(
    sessionKey: string,
    model: string,
  ): Promise<DaemonSession | undefined>;
  /** Pin one session to a reasoning effort without disturbing any other. */
  setSessionReasoning?(
    sessionKey: string,
    effort: string,
  ): Promise<DaemonSession | undefined>;
  /** Pin one session to a permission mode without disturbing any other. */
  setSessionPermissionMode?(
    sessionKey: string,
    mode: string,
  ): Promise<DaemonSession | undefined>;
  /**
   * Optional session-scoped ultra mode toggle. Kept optional on the
   * interface so existing DaemonRuntime implementations (test fakes, custom
   * hosts) stay source-compatible without implementing it; the server checks
   * for its absence and rejects the /ultra command with a typed error
   * instead of crashing.
   */
  setSessionUltra?(
    sessionKey: string,
    enabled: boolean,
  ): Promise<DaemonSession | undefined>;
  sessionStatus(sessionKey: string): DaemonSession | undefined;
  /** Release host-owned resources such as native delegated-agent managers. */
  shutdown?(): Promise<void>;
  steerTurn(sessionKey: string, content: string): boolean;
  /**
   * True when steering text is queued and not yet folded into the transcript.
   *
   * Automatic goal rounds yield to it: a person who has already typed should
   * not wait behind a round the machine queued for itself.
   */
  hasPendingSteer?(sessionKey: string): boolean;
  status(): JsonRpcPayload;
  submitTurn(
    sessionKey: string,
    text: string,
    emit: (event: DaemonEvent) => void,
    options?: SubmitTurnOptions,
  ): Promise<void>;
}

export interface DaemonBackgroundCommandLifecycle {
  disposeAll(): Promise<void>;
  disposeOwner(owner: string): Promise<void>;
}

export interface InMemoryDaemonRuntimeOptions {
  readonly backgroundCommands?: DaemonBackgroundCommandLifecycle;
  readonly baseUrl?: string;
  readonly buildId?: string;
  readonly currentProjectDirectory?: string;
  readonly model?: string;
  readonly permissionMode?: string;
  /** Coordinates approval and question replies for agent runners that opt in. */
  readonly interactions?: DaemonInteractionBoard;
  /** Live daemon settings used for status, reload, and runner reconstruction. */
  readonly runtimeSettings?: JsonRpcPayload;
  readonly sessionDirectory?: string;
  /** Cancel resources owned exclusively by a session before it is evicted. */
  readonly onSessionEvict?: (sessionId: string) => unknown;
  /**
   * Stop the delegated work a session started, because the user interrupted
   * its turn. Unlike eviction this is a pause, not a reclaim: implementations
   * must leave the cancelled handles inspectable and retryable.
   */
  readonly onTurnCancel?: (sessionId: string) => number | void;
  /** Host-owned subagent retry port wired to the daemon's subagent host. */
  readonly subagentRetry?: (
    request: SubagentRetryRequest,
  ) => Promise<SubagentRetryResult>;
  /** Reconcile session-owned resources after an interaction-policy change. */
  readonly onSessionModeChange?: (sessionId: string, mode: string) => void;
  /** Release resources captured by the host that constructed this runtime. */
  readonly shutdown?: () => Promise<void> | void;
  /** Live inventory owned by the embedding daemon host. */
  readonly statusInventory?: () => {
    readonly activeSubagents?: number;
    readonly skills?: number;
    readonly tools?: number;
  };
  readonly transcriptStore?: DaemonTranscriptStore;
  /** Rebuild a native runner after a profile/config mutation. */
  readonly turnRunnerFactory?: (
    settings: Readonly<JsonRpcPayload>,
  ) => TurnRunner | undefined;
  readonly workspaceRoot?: string;
}

/**
 * Stateful Bun daemon runtime with Python-readable transcript persistence.
 * A real turn runner can replace the echo runner without changing the session
 * lifecycle or v35 daemon contract.
 */
export class InMemoryDaemonRuntime implements DaemonRuntime {
  private readonly abortControllers = new Map<string, AbortController>();
  /** Serializes owner cleanup before a reused persisted id can own new commands. */
  private readonly backgroundOwnerCleanups = new Map<string, Promise<void>>();
  /** Children stopped by the last interrupt, reported once on the turn's settle edge. */
  private readonly cancelledSubagents = new Map<string, number>();
  private readonly directSubagentClaims = new Map<string, () => void>();
  private readonly currentProjectDirectory: string;
  private readonly options: InMemoryDaemonRuntimeOptions;
  private readonly runtimeSettings: JsonRpcPayload;
  private readonly sessions = new Map<string, DaemonSession>();
  /** Coalesces async transcript loads so one key cannot initialize twice. */
  private readonly sessionOpenPromises = new Map<string, Promise<DaemonSession>>();
  /**
   * In-progress initializations keyed by resolved persisted id (value is the
   * claiming session key). The duplicate-live-copy check inside
   * initializeSession is check-then-act across several awaits, so two
   * concurrent openers of one persisted id under different keys could both
   * observe "no live copy" and register two live sessions that race on every
   * save. The claim is taken synchronously before those awaits and released
   * when the initialization settles.
   */
  private readonly sessionIdClaims = new Map<string, string>();
  private shutdownPromise: Promise<void> | undefined;
  private readonly steerQueues = new Map<string, string[]>();
  private readonly transcriptStore: DaemonTranscriptStore;
  private turnRunner: TurnRunner;
  private readonly workspaceRoot: string;

  constructor(
    turnRunner: TurnRunner | undefined = undefined,
    options: InMemoryDaemonRuntimeOptions = {},
  ) {
    const home = xerxesHome();
    this.options = options;
    this.runtimeSettings = {
      ...(options.runtimeSettings ?? {}),
      ...(options.model ? { model: options.model } : {}),
      ...(options.baseUrl ? { base_url: options.baseUrl } : {}),
      ...(options.permissionMode
        ? { permission_mode: options.permissionMode }
        : {}),
    };
    this.currentProjectDirectory = resolve(
      options.currentProjectDirectory ?? process.cwd(),
    );
    this.workspaceRoot = resolve(options.workspaceRoot ?? join(home, "agents"));
    this.transcriptStore =
      options.transcriptStore ??
      new DaemonTranscriptStore({
        directory: options.sessionDirectory ?? join(home, "sessions"),
        currentProjectDirectory: this.currentProjectDirectory,
        workspaceRoot: this.workspaceRoot,
      });
    this.turnRunner =
      turnRunner ??
      options.turnRunnerFactory?.(this.runtimeSettings) ??
      new EchoTurnRunner();
  }

  cancelAllTurns(): number {
    let cancelled = 0;
    for (const sessionKey of this.sessions.keys()) {
      if (this.cancelTurn(sessionKey)) {
        cancelled += 1;
      }
    }
    return cancelled;
  }

  retrySubagent(request: SubagentRetryRequest): Promise<SubagentRetryResult> {
    const port = this.options.subagentRetry;
    if (!port) {
      return Promise.resolve({
        ok: false,
        error: "This daemon runtime does not expose subagent retry.",
      });
    }
    return port(request);
  }

  cancelTurn(sessionKey: string): boolean {
    const controller = this.abortControllers.get(sessionKey);
    if (!controller) {
      return false;
    }
    const session = this.sessions.get(sessionKey);
    if (session) {
      session.cancelRequested = true;
      // An interrupt is the clearest possible statement that unattended work
      // should stop. The goal keeps its phase and history; what it loses is
      // this process's authority to open another round on its own.
      disarmGoal(session.id);
    }
    // Delegated children run on their own execution boundary: aborting the
    // parent's signal alone leaves a detached subagent burning tokens and
    // writing files while every surface already reports the turn stopped.
    // Cancel them first so the turn's own teardown observes honest child
    // state, and never let a host-side failure block the parent abort.
    if (session) {
      try {
        const stopped = this.options.onTurnCancel?.(session.id);
        if (typeof stopped === "number" && stopped > 0) {
          this.cancelledSubagents.set(sessionKey, stopped);
        }
      } catch (error) {
        console.error(
          `Cancelling delegated children of session '${session.id}' failed: ${errorMessage(error)}`,
        );
      }
    }
    controller.abort(new Error("Turn cancelled"));
    return true;
  }

  async deleteSavedSession(sessionId: string): Promise<boolean> {
    const active = [...this.sessions.entries()].find(
      ([, session]) => session.id === sessionId,
    );
    if (active?.[1].activeTurnId) {
      throw new Error("Cannot delete a session with an active turn");
    }
    const deleted = await this.transcriptStore.remove(sessionId);
    if (active) {
      this.evictSession(active[0]);
    }
    return deleted || active !== undefined;
  }

  async removeSavedTranscript(sessionId: string): Promise<boolean> {
    return this.transcriptStore.remove(sessionId);
  }

  evictSession(sessionKey: string): void {
    const sessionId = this.sessions.get(sessionKey)?.id ?? sessionKey;
    // Abort any in-flight turn so its orphaned controller cannot block a
    // future submitTurn with "already active" or race the next saveSession.
    this.abortControllers.get(sessionKey)?.abort(new Error("Session evicted"));
    this.abortControllers.delete(sessionKey);
    this.options.interactions?.cancelSession(
      sessionId,
    );
    this.queueBackgroundOwnerCleanup(sessionId);
    this.observeSessionCleanup(
      sessionId,
      () => this.options.onSessionEvict?.(sessionId),
    );
    this.turnRunner.dropSession?.(sessionId);
    this.steerQueues.delete(sessionKey);
    this.cancelledSubagents.delete(sessionKey);
    this.directSubagentClaims.get(sessionKey)?.();
    this.directSubagentClaims.delete(sessionKey);
    this.sessions.delete(sessionKey);
  }

  private queueBackgroundOwnerCleanup(sessionId: string): void {
    const disposeOwner = this.options.backgroundCommands?.disposeOwner;
    if (!disposeOwner) return;
    const previous = this.backgroundOwnerCleanups.get(sessionId) ?? Promise.resolve();
    const pending = previous
      .catch(() => {})
      .then(() => disposeOwner.call(this.options.backgroundCommands, sessionId));
    this.backgroundOwnerCleanups.set(sessionId, pending);
    void pending.catch((error) => {
      console.error(
        `Cleaning up resources for evicted session '${sessionId}' failed: ${errorMessage(error)}`,
      );
    }).finally(() => {
      if (this.backgroundOwnerCleanups.get(sessionId) === pending) {
        this.backgroundOwnerCleanups.delete(sessionId);
      }
    });
  }

  private async awaitBackgroundOwnerCleanup(sessionId: string): Promise<void> {
    try {
      await this.backgroundOwnerCleanups.get(sessionId);
    } catch {
      // Eviction cleanup is best-effort and already reported. A failed cleanup
      // must not permanently prevent the owner id from being opened again.
    }
  }

  private observeSessionCleanup(
    sessionId: string,
    cleanup: () => unknown,
  ): void {
    try {
      const pending = cleanup();
      if (pending && typeof (pending as { catch?: unknown }).catch === "function") {
        void (pending as Promise<unknown>).catch((error) => {
          console.error(
            `Cleaning up resources for evicted session '${sessionId}' failed: ${errorMessage(error)}`,
          );
        });
      }
    } catch (error) {
      console.error(
        `Cleaning up resources for evicted session '${sessionId}' failed: ${errorMessage(error)}`,
      );
    }
  }

  async flushSessions(mode: 'append' | 'rewrite' = 'append'): Promise<void> {
    await Promise.all(
      [...this.sessions.values()].map(async (session) => {
        try {
          await this.saveSession(session, mode);
        } catch (error) {
          // A session whose in-memory state diverges from disk (transcripts
          // removed or restored under a live daemon) must never wedge the
          // connection: initialize flushes before every bind, so one stale
          // session would reject every new session the client tries to
          // create. Disk is the source of truth — drop the stale memory and
          // keep the flush moving. Sessions with a live turn are real
          // failures, not staleness; let them propagate.
          const message = error instanceof Error ? error.message : String(error);
          if (session.activeTurnId || !message.includes("conflicts with persisted history")) {
            throw error;
          }
          this.evictSession(session.sessionKey);
        }
      }),
    );
  }

  resetSavedTranscriptState(): void {
    for (const session of this.sessions.values()) {
      session.transcriptGeneration = 0;
      session.persistedMessageCount = 0;
    }
  }

  /**
   * List persisted sessions in three tiers: `stat` to enumerate, a bounded
   * head read to build and filter each row, and a full parse only for the
   * rows that genuinely require their full body. Reading every transcript in
   * full to render twenty rows cost a quarter of a second and half a gigabyte
   * of allocation on a directory this machine already has.
   */
  async listSavedSessions(
    limit = 0,
    options: SavedSessionListOptions = {},
  ): Promise<readonly SavedDaemonSession[]> {
    const projectDirectory = options.projectDirectory
      ? resolve(options.projectDirectory)
      : undefined;
    const summaries: SavedDaemonSession[] = [];
    for (const entry of await this.transcriptStore.listEntries()) {
      const summary = await this.savedSessionRow(entry, projectDirectory);
      if (summary) summaries.push(summary);
    }
    summaries.sort(
      (left, right) =>
        timestampMillis(right.updatedAt) - timestampMillis(left.updatedAt),
    );
    return this.titleSelectedSessions(
      selectSavedSessionSummaries(summaries, limit, options),
    );
  }

  private async savedSessionRow(
    entry: DaemonTranscriptEntry,
    projectDirectory: string | undefined,
  ): Promise<SavedDaemonSession | undefined> {
    const result = await this.transcriptStore.readHeader(entry.sessionId);
    const fastHeader =
      result.kind === "header" && result.header.turnCount > 0 ? result.header : undefined;
    if (fastHeader) {
      if (
        projectDirectory !== undefined &&
        sourceProjectDirectory(fastHeader.metadata, fastHeader.cwd) !== projectDirectory
      ) {
        return undefined;
      }
      return savedSessionSummary({ ...fastHeader, path: entry.path });
    }
    if (result.kind === "unreadable") return unreadableSavedSession(entry);
    // Both remaining cases go through the full parse: a truncated header
    // cannot answer at all, and a zero-turn header usually means a dangling
    // prompt (a turn that died before any reply) whose messages-only
    // transcript must not list as a session — transcriptHasHistory below
    // decides on roles, which the header cannot see.
    const transcript = await this.transcriptStore.loadForListing(
      entry.sessionId,
    );
    if (!transcript) return unreadableSavedSession(entry);
    if (!transcriptHasHistory(transcript)) return undefined;
    if (
      projectDirectory !== undefined &&
      transcriptProjectDirectory(transcript) !== projectDirectory
    ) {
      return undefined;
    }
    return savedSessionSummary({
      agentId: transcript.agentId,
      cwd: transcript.cwd,
      key: transcript.key,
      messageCount: transcript.messages.length,
      messages: transcript.messages,
      metadata: transcript.metadata,
      path: entry.path,
      sessionId: transcript.sessionId,
      turnCount: transcript.turnCount,
      updatedAt: transcript.updatedAt,
    });
  }

  /** Keep untitled rows untitled; chat content is never a title fallback. */
  private titleSelectedSessions(
    selected: readonly SavedDaemonSession[],
  ): readonly SavedDaemonSession[] {
    return selected;
  }

  /**
   * Journal callback for a session's message stream, handed to a producer as
   * a plain function so no message-producing code has to know about daemon
   * storage. A per-turn crash otherwise loses the entire turn: the transcript
   * is written once, in the turn's `finally`.
   */
  messageJournal(sessionId: string): TranscriptMessageJournalAppend {
    return this.transcriptStore.journalAppender(sessionId);
  }

  listSessions(): readonly DaemonSession[] {
    return [...this.sessions.values()].sort(
      (left, right) => right.lastActive - left.lastActive,
    );
  }

  async openSession(
    sessionKey: string,
    agentId?: string,
    options: OpenSessionOptions = {},
  ): Promise<DaemonSession> {
    const key = sessionKey || "default";
    const previous = this.sessionOpenPromises.get(key);
    const opening = (async () => {
      if (previous) {
        try {
          await previous;
        } catch {
          // A failed opener must not poison the key; this caller still gets
          // its own validation/load attempt.
        }
      }
      return this.initializeSession(key, agentId, options);
    })();
    this.sessionOpenPromises.set(key, opening);
    try {
      return await opening;
    } finally {
      if (this.sessionOpenPromises.get(key) === opening) {
        this.sessionOpenPromises.delete(key);
      }
    }
  }

  private async initializeSession(
    key: string,
    agentId?: string,
    options: OpenSessionOptions = {},
  ): Promise<DaemonSession> {
    const existing = this.sessions.get(key);
    if (existing) {
      const existingIsSubagent = metadataIsSubagent(existing.metadata);
      const requestedCwd = options.cwd ? resolve(options.cwd) : undefined;
      if (
        existingIsSubagent &&
        requestedCwd &&
        sessionProjectDirectory(existing) !== requestedCwd
      ) {
        throw new ValidationError(
          "session_id",
          "belongs to a subagent history from a different project",
          key,
        );
      }
      existing.lastActive = Date.now();
      if (agentId && agentId !== existing.agentId) {
        if (existing.activeTurnId || existing.turnCount > 0 || existing.messages.length > 0) {
          throw new ValidationError(
            "agent_preset",
            `session '${existing.id}' has already started; its agent preset is fixed`,
            agentId,
          );
        }
        existing.agentId = agentId;
        existing.workspace = workspaceFor(this.workspaceRoot, agentId);
      }
      if (requestedCwd && !existingIsSubagent) {
        existing.cwd = requestedCwd;
      }
      if (options.model) {
        // An explicit model on reopen is the caller choosing for this session,
        // so it pins: a later global reload must not undo it.
        existing.model = options.model;
        existing.modelPinned = true;
      }
      applySystemPromptAddendum(existing, options.systemPromptAddendum);
      return existing;
    }

    const cwd = resolve(options.cwd ?? this.currentProjectDirectory);
    const shouldResume = options.resume ?? looksLikeSessionId(key);
    if (shouldResume && isSubagentConversationActive(key)) {
      throw new ValidationError(
        "session_id",
        "is still owned by a running subagent; wait for it to finish before resuming its history",
        key,
      );
    }
    const loadResult = shouldResume
      ? await this.transcriptStore.loadResult(key, {
          currentProjectDirectory: cwd,
          workspaceRoot: this.workspaceRoot,
        })
      : { kind: "missing" } as const;
    if (options.resume === true && loadResult.kind === "corrupt") {
      throw new ValidationError(
        "session_id",
        "persisted transcript is corrupt or unreadable",
        key,
      );
    }
    const transcript = loadResult.kind === "loaded" ? loadResult.transcript : undefined;
    if (
      transcript &&
      transcriptProjectDirectory(transcript) !== cwd
    ) {
      const kind = transcriptIsSubagent(transcript)
        ? "subagent history"
        : "main session";
      throw new ValidationError(
        "session_id",
        `belongs to a ${kind} from a different project`,
        key,
      );
    }
    let effectiveTranscript = transcript;
    let session = effectiveTranscript
      ? sessionFromTranscript(
          effectiveTranscript,
          key,
          options.model ?? this.model(),
          this.workspaceRoot,
          stringValue(this.runtimeSettings.permission_mode),
        )
      : freshSession(
          key,
          agentId ?? "default",
          cwd,
          options.model ?? this.model(),
          this.workspaceRoot,
          // An explicit model at creation is the caller choosing for this
          // session — a background prompt inheriting its parent's model, for
          // instance — so it pins against later default changes.
          options.model !== undefined,
        );
    // A live copy of the same persisted id may still be registered under a
    // stale key (for example a `tui:` slot that predates a resume). Two
    // in-memory sessions sharing one id race on every save — flushSessions
    // persists both concurrently and the stale copy silently overwrites
    // newer history — so fold the live copy in before registering this one.
    //
    // The scan alone is check-then-act across the awaits below, so a second
    // concurrent opener of the same id would also observe "no duplicate" and
    // register a second live copy. Claim the resolved id synchronously first;
    // a concurrent opener of another key folds into a deterministic typed
    // error instead of racing.
    const claimant = this.sessionIdClaims.get(session.id);
    if (claimant !== undefined && claimant !== key) {
      throw new ValidationError(
        "session_id",
        "is already being opened for another session key; retry once that open settles",
        key,
      );
    }
    this.sessionIdClaims.set(session.id, key);
    // Held in a const: the duplicate-fold path may rebuild `session` from a
    // reloaded transcript, and the release must always target the claimed id.
    const claimedId = session.id;
    try {
      const duplicate = [...this.sessions.entries()].find(
        ([otherKey, other]) => otherKey !== key && other.id === session.id,
      );
      if (duplicate) {
        const [otherKey, other] = duplicate;
        if (other.activeTurnId) {
          throw new ValidationError(
            "session_id",
            "is still running a turn under another connection; wait for it to finish before resuming it here",
            key,
          );
        }
        // The live copy can hold state newer than the transcript loaded above
        // (idle steers, title or mode edits); persist it before dropping the
        // stale key, then re-read so the adopted session loses nothing.
        await this.saveSession(other);
        this.evictSession(otherKey);
        const reloaded = await this.transcriptStore.load(key, {
          currentProjectDirectory: cwd,
          workspaceRoot: this.workspaceRoot,
        });
        if (reloaded) {
          effectiveTranscript = reloaded;
          session = sessionFromTranscript(
            reloaded,
            key,
            options.model ?? this.model(),
            this.workspaceRoot,
          );
        }
      }
      // evictSession must remain synchronous for callers such as channel reset,
      // but its owner disposal is asynchronous. Do not expose a replacement
      // session with the same persisted id until that disposal has settled: a
      // late disposeOwner(id) could otherwise include commands the replacement
      // starts after openSession returns.
      await this.awaitBackgroundOwnerCleanup(session.id);
      const releaseSubagentClaim = effectiveTranscript && transcriptIsSubagent(effectiveTranscript)
        ? claimDirectSubagentConversation(effectiveTranscript.sessionId)
        : undefined;
      applySystemPromptAddendum(session, options.systemPromptAddendum);
      if (effectiveTranscript) {
        // A goal reloaded from disk keeps saying what it is, but this process
        // has not been told to keep driving it unattended. Continuation is
        // re-armed by an explicit goal call, never by the act of reopening a
        // session — otherwise merely listing history could restart an
        // objective that a person had walked away from.
        disarmGoal(session.id);
      }
      this.sessions.set(key, session);
      if (releaseSubagentClaim) this.directSubagentClaims.set(key, releaseSubagentClaim);
      return session;
    } finally {
      if (this.sessionIdClaims.get(claimedId) === key) {
        this.sessionIdClaims.delete(claimedId);
      }
    }
  }

  reload(overrides: JsonRpcPayload = {}): JsonRpcPayload {
    for (const [key, value] of Object.entries(overrides)) {
      if (value === undefined || value === "") {
        continue;
      }
      if (value === null) {
        delete this.runtimeSettings[key];
        continue;
      }
      this.runtimeSettings[key] = value;
    }
    if (this.options.turnRunnerFactory) {
      this.turnRunner =
        this.options.turnRunnerFactory(this.runtimeSettings) ??
        new EchoTurnRunner();
    }
    const model = this.model();
    for (const session of this.sessions.values()) {
      // Only sessions that never chose for themselves follow the global
      // default; a pinned one keeps what it was given.
      if (!session.modelPinned) {
        const modelDelta = contextDeltaFor(session.model, model, Date.now(), "model");
        if (modelDelta) appendContextDelta(session.metadata, modelDelta);
        session.model = model;
      }
      if (!session.reasoningPinned) {
        const effort = stringValue(this.runtimeSettings.reasoning_effort);
        if (effort) {
          const effortDelta = contextDeltaFor(session.reasoningEffort, effort, Date.now(), "reasoning");
          if (effortDelta) appendContextDelta(session.metadata, effortDelta);
          session.reasoningEffort = effort;
        }
      }
      if (!session.permissionPinned) {
        const permission = stringValue(this.runtimeSettings.permission_mode);
        if (permission) {
          const permissionDelta = contextDeltaFor(session.permissionMode, permission, Date.now(), "permission");
          if (permissionDelta) appendContextDelta(session.metadata, permissionDelta);
          session.permissionMode = permission;
        }
      }
    }
    return this.status();
  }

  async selectSessionAgent(
    sessionKey: string,
    agentId: string,
  ): Promise<DaemonSession | undefined> {
    const session = this.sessions.get(sessionKey);
    if (!session) return undefined;
    if (session.activeTurnId || session.turnCount > 0 || session.messages.length > 0) {
      throw new ValidationError(
        "agent_preset",
        `session '${session.id}' has already started; its agent preset is fixed`,
        agentId,
      );
    }
    const chosen = agentId.trim();
    if (!chosen) return session;
    session.agentId = chosen;
    session.workspace = workspaceFor(this.workspaceRoot, chosen);
    session.lastActive = Date.now();
    return session;
  }

  async setSessionMode(
    sessionKey: string,
    mode: string,
    planMode?: boolean,
  ): Promise<DaemonSession | undefined> {
    const session = this.sessions.get(sessionKey);
    if (!session) {
      return undefined;
    }
    const normalized = normalizeInteractionMode(mode, planMode ?? false);
    const modeDelta = contextDeltaFor(session.interactionMode, normalized, Date.now(), "interaction-mode");
    if (modeDelta) appendContextDelta(session.metadata, modeDelta);
    session.interactionMode = normalized;
    session.planMode = planMode ?? normalized === "plan";
    session.lastActive = Date.now();
    await this.saveSession(session);
    this.options.onSessionModeChange?.(session.id, normalized);
    return session;
  }

  async setSessionModel(
    sessionKey: string,
    model: string,
  ): Promise<DaemonSession | undefined> {
    const session = this.sessions.get(sessionKey);
    if (!session) {
      return undefined;
    }
    const chosen = model.trim();
    if (!chosen) {
      return session;
    }
    const modelDelta = contextDeltaFor(session.model, chosen, Date.now(), "model");
    if (modelDelta) appendContextDelta(session.metadata, modelDelta);
    session.model = chosen;
    // Pinned from here on, so a later global reload cannot silently move this
    // session onto another session's model.
    session.modelPinned = true;
    session.lastActive = Date.now();
    return session;
  }

  async setSessionReasoning(
    sessionKey: string,
    effort: string,
  ): Promise<DaemonSession | undefined> {
    const session = this.sessions.get(sessionKey);
    if (!session) {
      return undefined;
    }
    const chosen = effort.trim();
    if (!chosen) {
      return session;
    }
    const effortDelta = contextDeltaFor(session.reasoningEffort, chosen, Date.now(), "reasoning");
    if (effortDelta) appendContextDelta(session.metadata, effortDelta);
    session.reasoningEffort = chosen;
    session.reasoningPinned = true;
    session.lastActive = Date.now();
    return session;
  }

  async setSessionPermissionMode(
    sessionKey: string,
    mode: string,
  ): Promise<DaemonSession | undefined> {
    const session = this.sessions.get(sessionKey);
    if (!session) {
      return undefined;
    }
    const chosen = mode.trim();
    if (!chosen) {
      return session;
    }
    const permissionDelta = contextDeltaFor(session.permissionMode, chosen, Date.now(), "permission");
    if (permissionDelta) appendContextDelta(session.metadata, permissionDelta);
    session.permissionMode = chosen;
    session.permissionPinned = true;
    session.lastActive = Date.now();
    return session;
  }

  async setSessionUltra(
    sessionKey: string,
    enabled: boolean,
  ): Promise<DaemonSession | undefined> {
    const session = this.sessions.get(sessionKey);
    if (!session) {
      return undefined;
    }
    // In-memory flip only: no persistence call, matching the field's
    // wire-format contract (ultraMode never crosses the session store, so
    // a daemon restart always starts sessions with ultra mode off).
    session.ultraMode = enabled;
    session.lastActive = Date.now();
    return session;
  }

  sessionStatus(sessionKey: string): DaemonSession | undefined {
    return this.sessions.get(sessionKey);
  }

  async shutdown(): Promise<void> {
    this.shutdownPromise ??= Promise.allSettled([
      Promise.resolve().then(() => this.options.shutdown?.()),
      Promise.resolve().then(() => this.options.backgroundCommands?.disposeAll()),
    ]).then((results) => {
      const failures = results
        .filter((result): result is PromiseRejectedResult => result.status === "rejected")
        .map((result) => result.reason);
      if (failures.length === 1) throw failures[0];
      if (failures.length > 1) {
        throw new AggregateError(failures, "Daemon runtime shutdown cleanup failed");
      }
    });
    try {
      await this.shutdownPromise;
    } finally {
      for (const release of this.directSubagentClaims.values()) release();
      this.directSubagentClaims.clear();
    }
  }

  steerTurn(sessionKey: string, content: string): boolean {
    const session = this.sessions.get(sessionKey);
    const cleaned = content.trim();
    if (!session || !cleaned) {
      return false;
    }
    if (this.abortControllers.has(sessionKey)) {
      const queue = this.steerQueues.get(sessionKey) ?? [];
      if (queue.length >= MAX_ACTIVE_TURN_STEERS) {
        return false;
      }
      queue.push(cleaned);
      this.steerQueues.set(sessionKey, queue);
      return true;
    }
    session.messages.push({
      role: "user",
      content: `[steer from user]\n${cleaned}`,
    });
    session.lastActive = Date.now();
    return true;
  }

  status(): JsonRpcPayload {
    const inventory = this.options.statusInventory?.() ?? {};
    const sampling = Object.fromEntries(
      [
        "frequency_penalty",
        "max_tokens",
        "min_p",
        "presence_penalty",
        "repetition_penalty",
        "temperature",
        "thinking_budget",
        "top_k",
        "top_p",
      ].flatMap((key) => {
        const value = optionalFiniteNumber(this.runtimeSettings[key]);
        return value === undefined ? [] : [[key, value]];
      }),
    );
    return {
      ok: Boolean(this.model()),
      model: this.model(),
      base_url: stringValue(this.runtimeSettings.base_url),
      provider: stringValue(this.runtimeSettings.provider),
      permission_mode:
        stringValue(this.runtimeSettings.permission_mode) || "accept-all",
      ...sampling,
      ...(typeof this.runtimeSettings.thinking === "boolean"
        ? { thinking: this.runtimeSettings.thinking }
        : {}),
      ...(typeof this.runtimeSettings.responses_api === "boolean"
        ? { responses_api: this.runtimeSettings.responses_api }
        : {}),
      ...(typeof this.runtimeSettings.auto_title === "boolean"
        ? { auto_title: this.runtimeSettings.auto_title }
        : {}),
      ...(optionalFiniteNumber(this.runtimeSettings.auto_compact_threshold) !==
        undefined
        ? {
          auto_compact_threshold: optionalFiniteNumber(
            this.runtimeSettings.auto_compact_threshold,
          ),
        }
        : {}),
      ...Object.fromEntries(
        ["debug", "fast_mode", "nudge", "verbose"].flatMap((key) =>
          typeof this.runtimeSettings[key] === "boolean"
            ? [[key, this.runtimeSettings[key]]]
            : [],
        ),
      ),
      tools: inventoryCount(inventory.tools),
      skills: inventoryCount(inventory.skills),
      ...(inventory.activeSubagents === undefined
        ? {}
        : { active_subagents: inventoryCount(inventory.activeSubagents) }),
      reasoning_effort:
        stringValue(this.runtimeSettings.reasoning_effort) || "off",
      pid: process.pid,
      daemon_protocol: DAEMON_PROTOCOL_VERSION,
      daemon_build_id: this.options.buildId ?? BUN_DAEMON_BUILD_ID,
      channels: [],
      runtime: "bun-typescript",
      session_count: this.sessions.size,
    };
  }

  async submitTurn(
    sessionKey: string,
    text: string,
    emit: (event: DaemonEvent) => void,
    options: SubmitTurnOptions = {},
  ): Promise<void> {
    // Admission begins before session loading and other asynchronous setup.
    // Otherwise cancelTurn observes no controller in that window and the
    // supposedly stopped request proceeds to launch once setup resolves.
    if (this.abortControllers.has(sessionKey)) {
      throw new Error("A turn is already active for this session");
    }
    const controller = new AbortController();
    this.abortControllers.set(sessionKey, controller);
    let session: DaemonSession;
    try {
      session = await this.openSession(sessionKey);
    } catch (error) {
      if (this.abortControllers.get(sessionKey) === controller) {
        this.abortControllers.delete(sessionKey);
      }
      throw error;
    }
    if (controller.signal.aborted) {
      session.status = "idle";
      session.cancelRequested = true;
      session.lastActive = Date.now();
      if (this.abortControllers.get(sessionKey) === controller) {
        this.abortControllers.delete(sessionKey);
      }
      // A cancel landing during setup still owes the client a terminal event.
      // The initialize-eviction and disconnect races admit a turn and then
      // abort it before turn_begin; returning silently left the submitter
      // waiting for a turn_end that never came, so the TUI showed an
      // eternally in-flight prompt. Match the cancelled-turn vocabulary the
      // post-setup abort path below uses. `unstarted` is additive wire
      // vocabulary marking that no assistant content (indeed, no turn_begin)
      // ever existed for this submission, so clients must not synthesize an
      // empty assistant row from it.
      emit({
        type: "turn_end",
        payload: { cancelled: true, unstarted: true, session_id: session.id },
      });
      return;
    }

    session.status = "working";
    session.cancelRequested = false;
    // A cancel that lands during the previous turn's teardown must not have
    // its child count reported against this turn.
    this.cancelledSubagents.delete(sessionKey);
    session.activeTurnId = newSessionId();
    session.lastActive = Date.now();
    const runnerManagesState = this.turnRunner.managesSessionState === true;
    const assistantParts: string[] = [];
    const thinkingParts: string[] = [];
    const displayText = options.displayText?.trim() || text;
    session.inflightUser = displayText;
    session.inflightAssistant = "";
    session.inflightStartedAt = Date.now();
    delete session.inflightThinking;
    delete session.inflightTools;
    // Every event produced by this turn must retain its owning session. One
    // TUI connection can keep multiple native sessions alive and switch the
    // foreground tab while an earlier turn is still streaming; unscoped text,
    // tool, approval, or usage events would otherwise be applied to whichever
    // session happens to be visible when they arrive.
    const emitSessionEvent = (event: DaemonEvent): void => {
      // Recorded for BOTH state-management modes: a runner-managed session
      // only synchronizes session.messages at turn end, and this trail is
      // what a mid-turn session.open replays as the turn's work so far.
      recordInflightTrail(session, event);
      emit({
        ...event,
        payload: { ...event.payload, session_id: session.id },
      });
    };
    const processed = await processAtMentions(text, session.cwd);
    if (controller.signal.aborted) {
      session.status = "idle";
      session.activeTurnId = "";
      session.lastActive = Date.now();
      emitSessionEvent({
        type: "turn_end",
        payload: { cancelled: true },
      });
      if (this.abortControllers.get(sessionKey) === controller) {
        this.abortControllers.delete(sessionKey);
      }
      return;
    }
    const providerText = processed.enhancedMessage;
    const images = options.images ?? [];
    // Attachments ride the same ContentPart channel the provider mappings
    // already understand; plain-text turns keep their string content so
    // legacy transcripts and downstream string consumers are untouched.
    const providerContent =
      images.length > 0
        ? [{ type: "text" as const, text: providerText }, ...imageUrlContentParts(images)]
        : providerText;
    if (!runnerManagesState) {
      session.messages.push({
        role: "user",
        content: providerContent,
        ...(displayText === providerText && !images.length ? {} : { text: displayText }),
      });
      session.turnCount += 1;
      // The transcript is rewritten once, in this turn's `finally`. A crash
      // before that boundary would otherwise lose the prompt outright, so the
      // journal carries it until the next full save subsumes it.
      const index = session.messages.length - 1;
      const message = session.messages[index];
      if (message) this.messageJournal(session.id)(message, index);
    }
    emitSessionEvent({
      type: "turn_begin",
      payload: {
        turn_id: session.activeTurnId,
        text: displayText,
        ...(processed.mentionedFiles.length
          ? { mentioned_files: processed.mentionedFiles }
          : {}),
      },
    });

    const releaseInteractions = this.options.interactions?.bind(
      session.id,
      emitSessionEvent,
    );
    try {
      for await (const event of this.turnRunner.run(
        session,
        providerText,
        controller.signal,
        {
          drainSteer: () => this.drainSteers(sessionKey),
          displayText,
          journal: this.messageJournal(session.id),
          ...(images.length ? { images } : {}),
          ...(options.goalRound === undefined ? {} : { goalRound: options.goalRound }),
        },
      )) {
        emitSessionEvent(event);
        if (!runnerManagesState) {
          updateFallbackSession(session, event, assistantParts, thinkingParts);
        }
      }
    } catch (error) {
      emitSessionEvent({
        type: "notification",
        payload: { level: "error", message: errorMessage(error) },
      });
    } finally {
      if (!runnerManagesState && assistantParts.length) {
        session.messages.push({
          role: "assistant",
          content: assistantParts.join(""),
          ...(thinkingParts.length ? { thinking: thinkingParts.join("") } : {}),
        });
      }
      session.status = "idle";
      session.activeTurnId = "";
      session.lastActive = Date.now();
      const pendingSteers = this.drainSteers(sessionKey);
      for (const steer of pendingSteers) {
        session.messages.push({
          role: "user",
          content: `[steer from user saved for next turn]\n${steer}`,
        });
      }
      if (pendingSteers.length) {
        emitSessionEvent({
          type: "notification",
          payload: {
            level: "info",
            message: `Saved ${pendingSteers.length} steer${pendingSteers.length === 1 ? "" : "s"} for the next turn.`,
          },
        });
      }
      // Say out loud what the interrupt reached. Delegated work stops on a
      // boundary the transcript never shows, so without this the user only
      // sees the parent turn end and has to guess what happened to its
      // children.
      const stoppedChildren = this.cancelledSubagents.get(sessionKey) ?? 0;
      this.cancelledSubagents.delete(sessionKey);
      if (stoppedChildren > 0) {
        emitSessionEvent({
          type: "notification",
          payload: {
            level: "info",
            message: `Interrupt stopped ${stoppedChildren} delegated agent${stoppedChildren === 1 ? "" : "s"}.`,
          },
        });
      }
      // Conversation recency is distinct from file-save recency. Compaction,
      // title changes, and shutdown flushes rewrite the transcript without a
      // new message and must not make it look like a new chat in /resume.
      session.metadata.last_message_at = new Date(session.lastActive).toISOString();
      try {
        await this.saveSession(session);
      } catch (error) {
        emitSessionEvent({
          type: "notification",
          payload: {
            level: "error",
            message: `Could not save session: ${errorMessage(error)}`,
          },
        });
      }
      emitSessionEvent({
        type: "turn_end",
        payload: {
          cancelled: controller.signal.aborted,
        },
      });
      delete session.inflightUser;
      delete session.inflightAssistant;
      delete session.inflightStartedAt;
      delete session.inflightThinking;
      delete session.inflightTools;
      // An evicted session may already have a replacement turn registered;
      // only release the controller this turn actually owns.
      if (this.abortControllers.get(sessionKey) === controller) {
        this.abortControllers.delete(sessionKey);
      }
      releaseInteractions?.();
    }
  }

  hasPendingSteer(sessionKey: string): boolean {
    return (this.steerQueues.get(sessionKey)?.length ?? 0) > 0;
  }

  private drainSteers(sessionKey: string): readonly string[] {
    const queued = this.steerQueues.get(sessionKey) ?? [];
    this.steerQueues.delete(sessionKey);
    return queued;
  }

  private model(): string {
    return stringValue(this.runtimeSettings.model) || this.options.model || "";
  }

  private async saveSession(session: DaemonSession, mode: 'append' | 'rewrite' = 'append'): Promise<void> {
    // A session with no completed exchange must not be persisted: a fresh
    // GUI or daemon session that never sent a message — or a dangling prompt
    // whose turn died before any reply — would otherwise live forever in
    // every listing as a phantom session. In-flight turns still save: this
    // write plus the crash journal are what make a mid-turn crash recoverable.
    if (
      !session.activeTurnId &&
      !transcriptHasHistory({ messages: session.messages, turnCount: session.turnCount })
    ) {
      return;
    }
    await this.transcriptStore.save({
      agentId: session.agentId,
      ...(session.apiCallsComplete === undefined
        ? {}
        : { apiCallsComplete: session.apiCallsComplete }),
      cwd: session.cwd,
      extra: session.extra,
      format: "bun-v2",
      generation: session.transcriptGeneration ?? 0,
      interactionMode: session.interactionMode,
      key: session.sessionKey,
      messages: session.messages,
      // Stamped so a later resume can restore the model this history was
      // written with instead of silently adopting whatever the profile last
      // stored, which could be a different provider entirely.
      metadata: {
        ...session.metadata,
        model: session.model,
        // The read-before-edit guard's per-session state rides with the
        // transcript so a resumed session still knows which files the model
        // has current knowledge of. Refreshed on every save, so a daemon
        // restart loses at most the reads of the turn that was in flight.
        [FILE_READS_METADATA_KEY]: fileReadsForMetadata(session.id),
        ...(session.reasoningEffort ? { reasoning_effort: session.reasoningEffort } : {}),
        ...(session.permissionMode ? { permission_mode: session.permissionMode } : {}),
      },
      pendingResumeReplays: [],
      planMode: session.planMode,
      schemaVersion: undefined,
      sessionId: session.id,
      thinkingContent: session.thinkingContent,
      toolExecutions: session.toolExecutions,
      ...(session.totalApiCalls === undefined ? {} : { totalApiCalls: session.totalApiCalls }),
      totalInputTokens: session.totalInputTokens,
      totalOutputTokens: session.totalOutputTokens,
      turnCount: session.turnCount,
      ...(session.usageComplete === undefined ? {} : { usageComplete: session.usageComplete }),
      updatedAt: new Date().toISOString(),
      workspace: session.workspace,
    }, {
      mode,
      expectedGeneration: session.transcriptGeneration ?? 0,
      expectedMessageCount: session.persistedMessageCount ?? 0,
      onSavedGeneration: generation => {
        session.transcriptGeneration = generation;
        session.persistedMessageCount = session.messages.length;
      },
    });
  }
}

class EchoTurnRunner implements TurnRunner {
  async *run(
    _session: DaemonSession,
    text: string,
    _signal: AbortSignal,
  ): AsyncGenerator<DaemonEvent> {
    yield {
      type: "text_part",
      payload: { text: `Bun daemon foundation received: ${text}` },
    };
  }
}

function freshSession(
  sessionKey: string,
  agentId: string,
  cwd: string,
  model: string,
  workspaceRoot: string,
  modelPinned = false,
): DaemonSession {
  return {
    id: looksLikeSessionId(sessionKey) ? sessionKey : newSessionId(),
    sessionKey,
    agentId,
    apiCallsComplete: true,
    workspace: workspaceFor(workspaceRoot, agentId),
    cwd,
    extra: {},
    interactionMode: "code",
    lastActive: Date.now(),
    messages: [],
    metadata: {},
    transcriptGeneration: 0,
    persistedMessageCount: 0,
    model,
    modelPinned,
    planMode: false,
    status: "idle",
    thinkingContent: [],
    toolExecutions: [],
    totalApiCalls: 0,
    totalInputTokens: 0,
    totalOutputTokens: 0,
    turnCount: 0,
    usageComplete: true,
    activeTurnId: "",
    cancelRequested: false,
  };
}

function applySystemPromptAddendum(
  session: DaemonSession,
  addendum: string | undefined,
): void {
  if (addendum === undefined) return;
  const text = addendum.trim();
  if (text) {
    session.systemPromptAddendum = text;
    return;
  }
  delete session.systemPromptAddendum;
}

/**
 * Relative strictness of a permission mode, for the resume rule below.
 * Anything unrecognized ranks strictest so an unknown value never loosens.
 */
function permissionStrictness(mode: string): number {
  if (mode === "accept-all") return 0;
  if (mode === "auto") return 1;
  return 2;
}

/**
 * Permission mode to resume a transcript under.
 *
 * Unlike the model and the reasoning effort, a stored permission mode is only
 * honored when it is at least as strict as the current default. Continuity is
 * the goal for the other two, but silently re-granting a looser trust level
 * from a file on disk is not something a resume should be able to do.
 */
function resumedPermissionMode(
  stored: string,
  current: string,
): string | undefined {
  if (!stored) return undefined;
  return permissionStrictness(stored) >= permissionStrictness(current)
    ? stored
    : undefined;
}

function sessionFromTranscript(
  transcript: DaemonTranscript,
  sessionKey: string,
  model: string,
  workspaceRoot: string,
  currentPermissionMode = "",
): DaemonSession {
  const interactionMode = normalizeInteractionMode(
    transcript.interactionMode,
    transcript.planMode,
  );
  const resumedPermission = resumedPermissionMode(
    stringValue(transcript.metadata.permission_mode),
    currentPermissionMode,
  );
  const session: DaemonSession = {
    id: transcript.sessionId,
    sessionKey,
    agentId: transcript.agentId,
    ...(transcript.apiCallsComplete === undefined
      ? {}
      : { apiCallsComplete: transcript.apiCallsComplete }),
    workspace: workspaceFor(workspaceRoot, transcript.agentId),
    cwd: transcript.cwd,
    extra: { ...transcript.extra },
    interactionMode,
    // Resuming/activating a chat does not create a message. Preserve the
    // conversation clock so /resume age remains tied to its latest content.
    lastActive: timestampMillis(
      nonemptyMetadataString(transcript.metadata, "last_message_at") ?? transcript.updatedAt,
    ) || Date.now(),
    messages: transcript.messages.map((message) => ({
      ...message,
      role: stringValue(message.role),
    })),
    metadata: { ...transcript.metadata },
    transcriptGeneration: transcript.generation ?? 0,
    persistedMessageCount: transcript.messages.length,
    // The model this history was written with wins over the current global
    // default: resuming a transcript should continue the conversation on the
    // model that produced it, not silently move it to another provider.
    model: stringValue(transcript.metadata.model) || model,
    modelPinned: Boolean(stringValue(transcript.metadata.model)),
    ...(stringValue(transcript.metadata.reasoning_effort)
      ? {
          reasoningEffort: stringValue(transcript.metadata.reasoning_effort),
          reasoningPinned: true,
        }
      : {}),
    ...(resumedPermission ? { permissionMode: resumedPermission, permissionPinned: true } : {}),
    planMode: interactionMode === "plan",
    status: "idle",
    thinkingContent: [...transcript.thinkingContent],
    toolExecutions: [...transcript.toolExecutions],
    ...(transcript.totalApiCalls === undefined ? {} : { totalApiCalls: transcript.totalApiCalls }),
    totalInputTokens: transcript.totalInputTokens,
    totalOutputTokens: transcript.totalOutputTokens,
    turnCount: transcript.turnCount,
    ...(transcript.usageComplete === undefined ? {} : { usageComplete: transcript.usageComplete }),
    activeTurnId: "",
    cancelRequested: false,
  };
  // Restore the read-before-edit guard's saved state. This belongs to the
  // resume path — a session loaded from disk must know which files the model
  // already read in this conversation, or a restart would force needless
  // re-reads of files the model still holds current knowledge of. Mutates
  // the shared tracker deliberately; every transcript load funnels through
  // here, so both resume sites are covered by construction.
  hydrateFileReadsFromMetadata(transcript.sessionId, transcript.metadata);
  return session;
}

/** Everything a listing row needs, from either a header or a full transcript. */
interface SavedSessionSource {
  readonly agentId: string;
  /** Project directory the transcript ran in — surfaced on every row. */
  readonly cwd: string;
  readonly key: string;
  readonly messageCount: number;
  /** Present only when the row came from a full parse; titles can fall back to it. */
  readonly messages?: readonly RawMessage[];
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly path: string;
  readonly sessionId: string;
  readonly turnCount: number;
  readonly updatedAt: string;
}

/**
 * A transcript whose bytes are not a transcript at all.
 *
 * It is reported rather than dropped, and deliberately survives the project
 * filter: a corrupt file we cannot attribute to any project is one the user
 * should be able to see and delete, and an invisible one only accumulates.
 */
function unreadableSavedSession(entry: DaemonTranscriptEntry): SavedDaemonSession {
  return {
    id: entry.sessionId,
    key: entry.sessionId,
    // A corrupt file cannot be attributed to a project; it groups under the
    // fallback workspace bucket.
    cwd: "",
    kind: "main",
    resumable: false,
    title: `(unreadable transcript ${entry.sessionId})`,
    agentId: "default",
    status: "unreadable",
    updatedAt: new Date(entry.modifiedAtMillis).toISOString(),
    turnCount: 0,
    messageCount: 0,
    path: entry.path,
  };
}

function savedSessionSummary(
  transcript: SavedSessionSource,
): SavedDaemonSession {
  const metadata = transcript.metadata;
  const lastMessageAt = nonemptyMetadataString(metadata, "last_message_at");
  const parentSessionId = nonemptyMetadataString(
    metadata,
    "parent_session_id",
  );
  const subagentId = nonemptyMetadataString(metadata, "subagent_id");
  const declaredKind = nonemptyMetadataString(metadata, "session_kind");
  const kind =
    declaredKind?.toLowerCase() === "subagent" || subagentId !== undefined
      ? "subagent"
      : "main";
  const rootSessionId = nonemptyMetadataString(metadata, "root_session_id");
  const resolvedRootSessionId = rootSessionId ?? parentSessionId;
  const model = nonemptyMetadataString(metadata, "model");
  const persistedStatus = nonemptyMetadataString(metadata, "status");
  const activeChild = kind === "subagent" && isSubagentConversationActive(transcript.sessionId);
  const status = persistedStatus === "running" && !activeChild ? "interrupted" : persistedStatus;
  return {
    id: transcript.sessionId,
    key: transcript.key,
    cwd: transcriptProjectDirectory(transcript),
    kind,
    resumable: !activeChild,
    // A provisional title derived from the opening prompt counts: saved chats
    // whose title call never landed used to list as blank forever.
    title: displayTitle(stringValue(metadata.title)),
    agentId: transcript.agentId,
    ...(model ? { model } : {}),
    ...(parentSessionId ? { parentSessionId } : {}),
    ...(resolvedRootSessionId
      ? { rootSessionId: resolvedRootSessionId }
      : {}),
    ...(status ? { status } : {}),
    ...(subagentId ? { subagentId } : {}),
    updatedAt: lastMessageAt ?? transcript.updatedAt,
    turnCount: transcript.turnCount,
    messageCount: transcript.messageCount,
    path: transcript.path,
  };
}

function transcriptIsSubagent(transcript: DaemonTranscript): boolean {
  return metadataIsSubagent(transcript.metadata);
}

function transcriptProjectDirectory(transcript: {
  readonly metadata: Readonly<Record<string, unknown>>;
  readonly cwd: string;
}): string {
  return sourceProjectDirectory(transcript.metadata, transcript.cwd);
}

function sourceProjectDirectory(
  metadata: Readonly<Record<string, unknown>>,
  cwd: string,
): string {
  const persisted = nonemptyMetadataString(metadata, "project_root");
  return resolve(persisted ?? cwd);
}

/** Malformed timestamps sort as the epoch instead of producing NaN orderings. */
function timestampMillis(value: string): number {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function metadataIsSubagent(metadata: Readonly<Record<string, unknown>>): boolean {
  const declaredKind = nonemptyMetadataString(metadata, "session_kind");
  return declaredKind?.toLowerCase() === "subagent" ||
    nonemptyMetadataString(metadata, "subagent_id") !== undefined;
}

function sessionProjectDirectory(session: DaemonSession): string {
  const persisted = nonemptyMetadataString(session.metadata, "project_root");
  return resolve(persisted ?? session.cwd);
}

function nonemptyMetadataString(
  metadata: Readonly<Record<string, unknown>>,
  key: string,
): string | undefined {
  const value = metadata[key];
  return typeof value === "string" && value.trim() ? value.trim() : undefined;
}

/**
 * Apply hierarchy policy before the root limit. When child rows are requested,
 * `limit` selects root histories and then expands their descendants so a busy
 * swarm can never push every resumable parent out of the picker.
 */
function selectSavedSessionSummaries(
  summaries: readonly SavedDaemonSession[],
  limit: number,
  options: SavedSessionListOptions,
): SavedDaemonSession[] {
  const kind =
    options.kind ?? (options.includeSubagents === true ? "all" : "main");
  if (kind === "main") {
    return limitSavedSessions(
      summaries.filter((session) => session.kind === "main"),
      limit,
    );
  }
  if (kind === "subagent") {
    return limitSavedSessions(
      summaries.filter((session) => session.kind === "subagent"),
      limit,
    );
  }

  if (limit <= 0) return [...summaries];

  const roots = summaries.filter((session) => session.kind === "main");
  const selectedRoots = limitSavedSessions(roots, limit);
  if (!selectedRoots.length) {
    return limitSavedSessions(
      summaries.filter((session) => session.kind === "subagent"),
      limit,
    );
  }
  const selectedRootIds = new Set(selectedRoots.map((session) => session.id));
  const children = summaries.filter(
    (session) =>
      session.kind === "subagent" &&
      ((session.rootSessionId !== undefined &&
        selectedRootIds.has(session.rootSessionId)) ||
        (session.parentSessionId !== undefined &&
          selectedRootIds.has(session.parentSessionId))),
  );
  const selectedIds = new Set([
    ...selectedRootIds,
    ...children.map((session) => session.id),
  ]);
  return summaries.filter((session) => selectedIds.has(session.id));
}

function limitSavedSessions(
  sessions: readonly SavedDaemonSession[],
  limit: number,
): SavedDaemonSession[] {
  return limit > 0 ? sessions.slice(0, limit) : [...sessions];
}

/** Bounds for the mid-turn reattach trail: recognizable context, never raw dumps. */
const INFLIGHT_THINKING_TAIL_CHARS = 8000;
const INFLIGHT_ARGUMENTS_PREVIEW_CHARS = 200;
const INFLIGHT_ERROR_PREVIEW_CHARS = 160;
const INFLIGHT_TOOL_LIMIT = 100;

function inflightPreviewText(value: unknown, limit: number): string {
  const raw = typeof value === "string" ? value : "";
  const compact = raw.replace(/\s+/g, " ").trim();
  return compact.length > limit ? `${compact.slice(0, limit - 1)}…` : compact;
}

/**
 * Accumulate the in-flight turn's thinking and tool rows onto the session so
 * sessionPayload can show a mid-turn reattach the work so far. Reads the same
 * frozen wire vocabulary the TUI adapter consumes.
 */
function recordInflightTrail(session: DaemonSession, event: DaemonEvent): void {
  if (event.type === "think_part") {
    const think = stringValue(event.payload.think);
    if (think) {
      const next = (session.inflightThinking ?? "") + think;
      session.inflightThinking =
        next.length > INFLIGHT_THINKING_TAIL_CHARS
          ? next.slice(next.length - INFLIGHT_THINKING_TAIL_CHARS)
          : next;
    }
    return;
  }
  if (event.type === "tool_call") {
    const name = stringValue(event.payload.name) || "tool";
    const args = inflightPreviewText(
      event.payload.arguments,
      INFLIGHT_ARGUMENTS_PREVIEW_CHARS,
    );
    const id = stringValue(event.payload.id) || stringValue(event.payload.tool_call_id);
    const tools = [...(session.inflightTools ?? [])];
    tools.push({
      ...(args ? { arguments: args } : {}),
      ...(id ? { id } : {}),
      name,
    });
    session.inflightTools = tools.slice(-INFLIGHT_TOOL_LIMIT);
    return;
  }
  if (event.type === "tool_result") {
    const tools = session.inflightTools;
    if (!tools?.length) {
      return;
    }
    const callId = stringValue(event.payload.tool_call_id);
    const index = callId
      ? tools.findIndex((tool) => tool.id === callId)
      : -1;
    // Without a matching id, settle the most recent unsettled call: the row
    // belongs to the turn either way, and leaving it running forever would
    // misreport it on every later reattach.
    const fallbackIndex = tools.reduce(
      (found, tool, i) => (tool.ok === undefined ? i : found),
      -1,
    );
    const target = index >= 0 ? index : fallbackIndex;
    if (target < 0) {
      return;
    }
    const permitted = event.payload.permitted !== false;
    const explicitError = stringValue(event.payload.error);
    const durationMs = optionalFiniteNumber(event.payload.duration_ms);
    const error = permitted
      ? undefined
      : explicitError ||
        inflightPreviewText(
          event.payload.return_value,
          INFLIGHT_ERROR_PREVIEW_CHARS,
        ) ||
        "Tool execution failed.";
    tools[target] = {
      ...tools[target]!,
      ...(error ? { error } : {}),
      ...(durationMs === undefined ? {} : { duration_ms: durationMs }),
      ok: permitted && !explicitError,
    };
  }
}

function updateFallbackSession(
  session: DaemonSession,
  event: DaemonEvent,
  assistantParts: string[],
  thinkingParts: string[],
): void {
  if (event.type === "text_part") {
    const text = stringValue(event.payload.text);
    if (text) {
      assistantParts.push(text);
      session.inflightAssistant = assistantParts.join("");
    }
    return;
  }
  if (event.type === "think_part") {
    const thinking = stringValue(event.payload.think);
    if (thinking) {
      thinkingParts.push(thinking);
      session.thinkingContent.push(thinking);
    }
    return;
  }
  if (event.type === "tool_result") {
    session.toolExecutions.push({ ...event.payload });
    return;
  }
  if (event.type === "status_update") {
    session.totalInputTokens += numberValue(
      event.payload.usage,
      "inputTokens",
      "input_tokens",
    );
    session.totalOutputTokens += numberValue(
      event.payload.usage,
      "outputTokens",
      "output_tokens",
    );
    const calls = optionalFiniteNumber(
      event.payload.calls ?? event.payload.total_api_calls,
    );
    if (calls !== undefined) {
      session.totalApiCalls = Math.max(0, Math.trunc(calls));
    }
    if (typeof event.payload.calls_complete === "boolean") {
      session.apiCallsComplete = event.payload.calls_complete;
    }
    if (typeof event.payload.usage_complete === "boolean") {
      session.usageComplete = event.payload.usage_complete;
    }
  }
}

function messageText(message: RawMessage): string {
  if (typeof message.text === "string") {
    return message.text;
  }
  return contentText(message.content);
}

function contentText(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }
        return isRecord(part)
          ? stringValue(part.text) || stringValue(part.content)
          : "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return isRecord(content)
    ? stringValue(content.text) || stringValue(content.content)
    : "";
}

function workspaceFor(workspaceRoot: string, agentId: string): string {
  return resolve(workspaceRoot, agentId || "default");
}

function newSessionId(): string {
  return crypto.randomUUID().replaceAll("-", "").slice(0, 12);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function numberValue(value: unknown, ...keys: readonly string[]): number {
  if (!isRecord(value)) {
    return 0;
  }
  for (const key of keys) {
    const candidate = value[key];
    if (typeof candidate === "number" && Number.isFinite(candidate)) {
      return candidate;
    }
  }
  return 0;
}

function stringValue(value: unknown): string {
  return typeof value === "string" ? value.trim() : "";
}

function optionalFiniteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value)
    ? value
    : undefined;
}

function inventoryCount(value: unknown): number {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0
    ? value
    : 0;
}
