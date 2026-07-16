// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { join, resolve } from "node:path";

import { ValidationError } from "../core/errors.js";
import { normalizeInteractionMode } from "../runtime/interactionModes.js";
import {
  DaemonTranscriptStore,
  looksLikeSessionId,
  type DaemonTranscript,
  type RawMessage,
} from "../session/daemonTranscript.js";
import type { JsonRpcPayload } from "../protocol/jsonRpc.js";
import { processAtMentions } from "./atMentions.js";
import { xerxesHome } from "./paths.js";
import type { DaemonInteractionBoard } from "./interactions.js";
import {
  claimDirectSubagentConversation,
  isSubagentConversationActive,
} from "./subagentConversations.js";

export const DAEMON_PROTOCOL_VERSION = 35;
export const BUN_DAEMON_BUILD_ID =
  process.env.XERXES_DAEMON_BUILD_ID?.trim() || "bun-runtime-v0.3.0";

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
  messages: DaemonTranscriptMessage[];
  metadata: Record<string, unknown>;
  model: string;
  planMode: boolean;
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
  /** Whether cumulative token totals cover every provider attempt in this session. */
  usageComplete?: boolean;
  workspace: string;
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
}

export interface SubmitTurnOptions {
  /** Transcript text retained separately from the provider-facing prompt. */
  readonly displayText?: string;
}

export interface DaemonRuntime {
  cancelAllTurns(): number;
  cancelTurn(sessionKey: string): boolean;
  /** Optional persistent-session removal capability for hosts with native transcript storage. */
  deleteSavedSession?(sessionId: string): Promise<boolean>;
  evictSession(sessionKey: string): void;
  flushSessions(): Promise<void>;
  listSavedSessions(
    limit?: number,
    options?: SavedSessionListOptions,
  ): Promise<readonly SavedDaemonSession[]>;
  listSessions(): readonly DaemonSession[];
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
  sessionStatus(sessionKey: string): DaemonSession | undefined;
  /** Release host-owned resources such as native delegated-agent managers. */
  shutdown?(): Promise<void>;
  steerTurn(sessionKey: string, content: string): boolean;
  status(): JsonRpcPayload;
  submitTurn(
    sessionKey: string,
    text: string,
    emit: (event: DaemonEvent) => void,
    options?: SubmitTurnOptions,
  ): Promise<void>;
}

export interface InMemoryDaemonRuntimeOptions {
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
  readonly onSessionEvict?: (sessionId: string) => void;
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
  private readonly directSubagentClaims = new Map<string, () => void>();
  private readonly currentProjectDirectory: string;
  private readonly options: InMemoryDaemonRuntimeOptions;
  private readonly runtimeSettings: JsonRpcPayload;
  private readonly sessions = new Map<string, DaemonSession>();
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

  cancelTurn(sessionKey: string): boolean {
    const session = this.sessions.get(sessionKey);
    if (!session) {
      return false;
    }
    session.cancelRequested = true;
    this.abortControllers.get(sessionKey)?.abort(new Error("Turn cancelled"));
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

  evictSession(sessionKey: string): void {
    const sessionId = this.sessions.get(sessionKey)?.id ?? sessionKey;
    this.options.interactions?.cancelSession(
      sessionId,
    );
    this.options.onSessionEvict?.(sessionId);
    this.steerQueues.delete(sessionKey);
    this.directSubagentClaims.get(sessionKey)?.();
    this.directSubagentClaims.delete(sessionKey);
    this.sessions.delete(sessionKey);
  }

  async flushSessions(): Promise<void> {
    await Promise.all(
      [...this.sessions.values()].map((session) => this.saveSession(session)),
    );
  }

  async listSavedSessions(
    limit = 0,
    options: SavedSessionListOptions = {},
  ): Promise<readonly SavedDaemonSession[]> {
    const transcripts = await this.transcriptStore.list();
    const projectDirectory = options.projectDirectory
      ? resolve(options.projectDirectory)
      : undefined;
    const summaries = transcripts
      .filter(
        (transcript) =>
          projectDirectory === undefined ||
          transcriptProjectDirectory(transcript) === projectDirectory,
      )
      .map((transcript) =>
        savedSessionSummary(
          transcript,
          this.transcriptStore.pathFor(transcript.sessionId),
        ),
      );
    return selectSavedSessionSummaries(summaries, limit, options);
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
      if (agentId) {
        existing.agentId = agentId;
        existing.workspace = workspaceFor(this.workspaceRoot, agentId);
      }
      if (requestedCwd && !existingIsSubagent) {
        existing.cwd = requestedCwd;
      }
      if (options.model) {
        existing.model = options.model;
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
    const transcript = shouldResume
      ? await this.transcriptStore.load(key, {
          currentProjectDirectory: cwd,
          workspaceRoot: this.workspaceRoot,
        })
      : undefined;
    if (
      transcript &&
      transcriptIsSubagent(transcript) &&
      transcriptProjectDirectory(transcript) !== cwd
    ) {
      throw new ValidationError(
        "session_id",
        "belongs to a subagent history from a different project",
        key,
      );
    }
    const session = transcript
      ? sessionFromTranscript(
          transcript,
          key,
          options.model ?? this.model(),
          this.workspaceRoot,
        )
      : freshSession(
          key,
          agentId ?? "default",
          cwd,
          options.model ?? this.model(),
          this.workspaceRoot,
        );
    const releaseSubagentClaim = transcript && transcriptIsSubagent(transcript)
      ? claimDirectSubagentConversation(transcript.sessionId)
      : undefined;
    applySystemPromptAddendum(session, options.systemPromptAddendum);
    this.sessions.set(key, session);
    if (releaseSubagentClaim) this.directSubagentClaims.set(key, releaseSubagentClaim);
    return session;
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
      session.model = model;
    }
    return this.status();
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
    session.interactionMode = normalized;
    session.planMode = planMode ?? normalized === "plan";
    session.lastActive = Date.now();
    this.options.onSessionModeChange?.(session.id, normalized);
    return session;
  }

  sessionStatus(sessionKey: string): DaemonSession | undefined {
    return this.sessions.get(sessionKey);
  }

  async shutdown(): Promise<void> {
    this.shutdownPromise ??= Promise.resolve().then(() => this.options.shutdown?.());
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
    const session = await this.openSession(sessionKey);
    if (this.abortControllers.has(sessionKey)) {
      throw new Error("A turn is already active for this session");
    }

    const controller = new AbortController();
    this.abortControllers.set(sessionKey, controller);
    session.status = "working";
    session.cancelRequested = false;
    session.activeTurnId = newSessionId();
    session.lastActive = Date.now();
    const runnerManagesState = this.turnRunner.managesSessionState === true;
    const assistantParts: string[] = [];
    const thinkingParts: string[] = [];
    const displayText = options.displayText?.trim() || text;
    const processed = await processAtMentions(text, session.cwd);
    const providerText = processed.enhancedMessage;
    if (!runnerManagesState) {
      session.messages.push({
        role: "user",
        content: providerText,
        ...(displayText === providerText ? {} : { text: displayText }),
      });
      session.turnCount += 1;
    }
    emit({
      type: "turn_begin",
      payload: {
        session_id: session.id,
        turn_id: session.activeTurnId,
        text: displayText,
        ...(processed.mentionedFiles.length
          ? { mentioned_files: processed.mentionedFiles }
          : {}),
      },
    });

    const releaseInteractions = this.options.interactions?.bind(
      session.id,
      emit,
    );
    try {
      for await (const event of this.turnRunner.run(
        session,
        providerText,
        controller.signal,
        {
          drainSteer: () => this.drainSteers(sessionKey),
          displayText,
        },
      )) {
        emit(event);
        if (!runnerManagesState) {
          updateFallbackSession(session, event, assistantParts, thinkingParts);
        }
      }
    } catch (error) {
      emit({
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
        emit({
          type: "notification",
          payload: {
            level: "info",
            message: `Saved ${pendingSteers.length} steer${pendingSteers.length === 1 ? "" : "s"} for the next turn.`,
          },
        });
      }
      try {
        await this.saveSession(session);
      } catch (error) {
        emit({
          type: "notification",
          payload: {
            level: "error",
            message: `Could not save session: ${errorMessage(error)}`,
          },
        });
      }
      emit({
        type: "turn_end",
        payload: {
          session_id: session.id,
          cancelled: controller.signal.aborted,
        },
      });
      this.abortControllers.delete(sessionKey);
      releaseInteractions?.();
    }
  }

  private drainSteers(sessionKey: string): readonly string[] {
    const queued = this.steerQueues.get(sessionKey) ?? [];
    this.steerQueues.delete(sessionKey);
    return queued;
  }

  private model(): string {
    return stringValue(this.runtimeSettings.model) || this.options.model || "";
  }

  private async saveSession(session: DaemonSession): Promise<void> {
    const title = stringValue(session.metadata.title);
    if (!title) {
      const derivedTitle = titleFromMessages(session.messages);
      if (derivedTitle) {
        session.metadata = { ...session.metadata, title: derivedTitle };
      }
    }
    await this.transcriptStore.save({
      agentId: session.agentId,
      ...(session.apiCallsComplete === undefined
        ? {}
        : { apiCallsComplete: session.apiCallsComplete }),
      cwd: session.cwd,
      extra: session.extra,
      format: "bun-v2",
      interactionMode: session.interactionMode,
      key: session.sessionKey,
      messages: session.messages,
      metadata: session.metadata,
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
    model,
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

function sessionFromTranscript(
  transcript: DaemonTranscript,
  sessionKey: string,
  model: string,
  workspaceRoot: string,
): DaemonSession {
  const interactionMode = normalizeInteractionMode(
    transcript.interactionMode,
    transcript.planMode,
  );
  return {
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
    lastActive: Date.now(),
    messages: transcript.messages.map((message) => ({
      ...message,
      role: stringValue(message.role),
    })),
    metadata: { ...transcript.metadata },
    model,
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
}

function savedSessionSummary(
  transcript: DaemonTranscript,
  path: string,
): SavedDaemonSession {
  const metadata = transcript.metadata;
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
    kind,
    resumable: !activeChild,
    title:
      stringValue(metadata.title) ||
      titleFromMessages(transcript.messages),
    agentId: transcript.agentId,
    ...(model ? { model } : {}),
    ...(parentSessionId ? { parentSessionId } : {}),
    ...(resolvedRootSessionId
      ? { rootSessionId: resolvedRootSessionId }
      : {}),
    ...(status ? { status } : {}),
    ...(subagentId ? { subagentId } : {}),
    updatedAt: transcript.updatedAt,
    turnCount: transcript.turnCount,
    messageCount: transcript.messages.length,
    path,
  };
}

function transcriptIsSubagent(transcript: DaemonTranscript): boolean {
  return metadataIsSubagent(transcript.metadata);
}

function transcriptProjectDirectory(transcript: DaemonTranscript): string {
  const persisted = nonemptyMetadataString(transcript.metadata, "project_root");
  return resolve(persisted ?? transcript.cwd);
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

function titleFromMessages(messages: readonly RawMessage[]): string {
  for (const message of messages) {
    if (message.role !== "user") {
      continue;
    }
    const text = messageText(message).replaceAll(/\s+/g, " ").trim();
    if (text) {
      return text.length > 80 ? `${text.slice(0, 77)}...` : text;
    }
  }
  return "";
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
