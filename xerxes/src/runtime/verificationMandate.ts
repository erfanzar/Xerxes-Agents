// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Claim-verification rules for every agent, main and sub-agent alike.
 *
 * Two independent assemblers render a system prompt: `PromptContextBuilder`
 * composes the profile-based prefix, and `buildBootstrapSystemPrompt` builds
 * the prompt the daemon actually sends on a live turn. These rules first
 * landed in the former only, so they shipped to tests and to nothing else —
 * a real session ran without them while the repo looked like it had them.
 * Both assemblers now read this constant so neither half can drift again.
 */
export const VERIFICATION_MANDATE_RULES: readonly string[] = Object.freeze([
  '- Never claim work is done, fixed, passing, or live without fresh evidence from this session: read the tool output, run the check, or say plainly that it is unverified.',
  '- Before reporting a test, build, or status claim, run the command and report what it actually printed; a claim without a tool result is a guess — label it as one.',
  '- Verify the environment before blaming it: confirm versions, paths, and process state with a real command before attributing failure to a stale install, the harness, or the user.',
  '- Distinguish observation from inference; never present an inference as a measured fact.',
  '- If a prior claim turns out to be wrong, correct it explicitly and immediately instead of hoping it goes unnoticed.',
])
