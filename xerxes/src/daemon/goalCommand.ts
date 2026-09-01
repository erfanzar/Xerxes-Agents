// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The human half of the goal subsystem: `/goal`.
 *
 * The model drives a goal through typed tool calls; a person drives the same
 * durable state through this command. Both go through `goalDomain`, so there is
 * exactly one definition of what a goal is and what transitions are legal —
 * a second, parallel notion living in the UI is how the two-engine drift this
 * codebase already suffers from starts.
 *
 * Parsing and rendering live here rather than in the TUI so every surface
 * (terminal, bridge, channels) shows the same words for the same state.
 *
 * Modelled on DeepSeek Harness's `/goal`
 * (github.com/deepseek-ai/deepseek-harness, MIT); no source is reproduced.
 */

import {
  clearGoal,
  createGoal,
  editGoal,
  getGoal,
  GoalError,
  pauseGoal,
  resumeGoal,
  type GoalPhase,
  type GoalRef,
  type GoalView,
} from "../runtime/goalDomain.js";

export const GOAL_USAGE =
  "Usage: /goal [<objective>|clear|edit <objective>|pause|resume]";

export type GoalCommand =
  | { readonly kind: "show" }
  | { readonly kind: "create"; readonly objective: string }
  | { readonly kind: "edit"; readonly objective: string }
  | { readonly kind: "invalid-edit" }
  | { readonly kind: "pause" }
  | { readonly kind: "resume" }
  | { readonly kind: "clear" };

export interface GoalCommandResult {
  readonly ok: boolean;
  readonly text: string;
}

/**
 * Parse only the grammar `/goal` owns; anything else is an objective.
 *
 * Deliberately not a flag parser. A goal objective is prose, and prose starting
 * with a word this command happens to know is far more likely to be an
 * objective than a mistyped subcommand — so only the exact control words, and
 * only as the whole input, are treated as commands.
 */
export function parseGoalCommand(rawInput: string): GoalCommand {
  const input = rawInput.trim();
  if (!input) return { kind: "show" };
  const control = input.toLowerCase();
  if (control === "clear") return { kind: "clear" };
  if (control === "pause") return { kind: "pause" };
  if (control === "resume") return { kind: "resume" };
  if (control === "edit") return { kind: "invalid-edit" };
  if (/^edit\s/iu.test(input)) return { kind: "edit", objective: input.slice(4).trim() };
  return { kind: "create", objective: input };
}

/** Commands that mean something from this exact live state. */
function commandHint(goal: GoalView): string {
  if (goal.phase === "active") {
    return goal.activation === "armed"
      ? "/goal edit <objective>, /goal pause, /goal clear"
      : "/goal edit <objective>, /goal resume, /goal clear";
  }
  if (goal.phase === "complete") return "/goal <objective>, /goal clear";
  return "/goal edit <objective>, /goal resume, /goal clear";
}

/**
 * Render a goal for a person.
 *
 * Compare-and-set internals (the revision the tools echo back) are deliberately
 * absent: a person has no use for them, and printing them invites hand-editing
 * of state whose whole purpose is to detect concurrent edits.
 */
function renderGoal(title: string, goal: GoalView): GoalCommandResult {
  const blocker = goal.blockedReason
    ? [`Blocker: ${goal.blockedReason.code}: ${goal.blockedReason.message}`]
    : [];
  return {
    ok: true,
    text: [
      title,
      `Status: ${goal.phase satisfies GoalPhase}`,
      ...blocker,
      `Objective: ${goal.objective}`,
      `Rounds: ${goal.roundsStarted}/${goal.maxGoalRounds}`,
      `Activation: ${goal.activation}`,
      "",
      `Commands: ${commandHint(goal)}`,
    ].join("\n"),
  };
}

const refOf = (goal: GoalView): GoalRef => ({ id: goal.id, revision: goal.revision });

const missingGoal = (action: string): GoalCommandResult => ({
  ok: false,
  text: `No goal is currently set; /goal ${action} requires one. ${GOAL_USAGE}`,
});

/**
 * Execute one `/goal` invocation against a session's durable metadata.
 *
 * Returns text rather than printing or emitting: the caller owns how it reaches
 * the person, and a pure function is testable without a socket.
 */
export function runGoalCommand(
  metadata: Record<string, unknown>,
  sessionId: string,
  rawInput: string,
  now: number = Date.now(),
): GoalCommandResult {
  const command = parseGoalCommand(rawInput);
  try {
    const current = getGoal(metadata, sessionId);
    switch (command.kind) {
      case "show":
        return current
          ? renderGoal("Goal", current)
          : { ok: true, text: `No goal is currently set.\n${GOAL_USAGE}` };
      case "invalid-edit":
        return { ok: false, text: `Goal editing requires a replacement objective.\n${GOAL_USAGE}` };
      case "create": {
        if (current && current.phase !== "complete") {
          return {
            ok: false,
            text:
              `A goal is already ${current.phase}. Use /goal edit <objective> to change it, `
              + "or /goal clear before replacing it.",
          };
        }
        return renderGoal("Goal created", createGoal(metadata, sessionId, { objective: command.objective }, now));
      }
      case "edit": {
        if (!current) return missingGoal("edit");
        // A completed goal is history. Editing it would rewrite what was
        // achieved; the honest reading of "edit" here is "start the next one".
        if (current.phase === "complete") {
          return renderGoal("Goal created", createGoal(metadata, sessionId, { objective: command.objective }, now));
        }
        return renderGoal(
          "Goal updated",
          editGoal(metadata, sessionId, refOf(current), { objective: command.objective }, now),
        );
      }
      case "pause":
        if (!current) return missingGoal("pause");
        return renderGoal("Goal paused", pauseGoal(metadata, sessionId, refOf(current), now));
      case "resume":
        if (!current) return missingGoal("resume");
        return renderGoal("Goal resumed", resumeGoal(metadata, sessionId, refOf(current), now));
      case "clear":
        if (!current) return { ok: true, text: "No goal to clear." };
        clearGoal(metadata, sessionId, refOf(current), now);
        return { ok: true, text: "Goal cleared." };
    }
  } catch (error) {
    if (error instanceof GoalError) {
      // The domain's own message names the exact transition it refused, which
      // is more useful than a generic "invalid for the current state".
      return { ok: false, text: `${error.message}\nRun /goal to see the available commands.` };
    }
    throw error;
  }
}
