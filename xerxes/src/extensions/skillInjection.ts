// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Skill argument and command injection — Claude Code custom-command parity.
 *
 * Skill bodies support two expansions at activation time:
 *
 * - `$ARGUMENTS` / `${ARGUMENTS}` / `$0`…`$9` — the invocation arguments
 *   (whole string, then whitespace-split positionals), exactly like Claude
 *   Code custom commands.
 * - `` !`command` `` — execute the command in the project shell and splice
 *   its stdout into the instructions before the model sees them.
 *
 * Trust model: expansion runs only on skills that survived registry
 * admission (bundled, user-installed, or hash-trusted workspace skills —
 * untrusted workspace skills never reach the registry), and the expanded
 * text still passes through the prompt-injection scan in
 * skillPromptSection, so command output is neutralized like any other
 * untrusted content.
 */

const INJECTION_PATTERN = /!`([^`\n]+)`/g
const MAX_INJECTIONS_PER_SKILL = 5
const INJECTION_TIMEOUT_MS = 10_000
const INJECTION_OUTPUT_CAP = 4 * 1024

export interface SkillExpansionOptions {
  /** Raw argument string from the invocation (`/skill name arg1 arg2`). */
  readonly args?: string
  /** Working directory for `` !`cmd` `` execution. */
  readonly cwd: string
  /** Injectable executor for tests; defaults to the platform shell. */
  readonly run?: (command: string) => Promise<{ readonly code: number; readonly stdout: string; readonly stderr: string }>
}

/** Substitute $ARGUMENTS/$N and execute `` !`cmd` `` injections. */
export async function expandSkillInstructions(
  instructions: string,
  options: SkillExpansionOptions,
): Promise<string> {
  let expanded = substituteArguments(instructions, options.args)
  if (!INJECTION_PATTERN.test(expanded)) return expanded
  INJECTION_PATTERN.lastIndex = 0

  const run = options.run ?? defaultSkillCommandExecutor(options.cwd)
  const replacements: { readonly replacement: string; readonly span: string }[] = []
  let count = 0
  for (const match of expanded.matchAll(INJECTION_PATTERN)) {
    count += 1
    const span = match[0]
    const command = (match[1] ?? '').trim()
    if (count > MAX_INJECTIONS_PER_SKILL) {
      replacements.push({ replacement: `[skipped: more than ${MAX_INJECTIONS_PER_SKILL} command injections in one skill]`, span })
      continue
    }
    try {
      const { code, stdout, stderr } = await run(command)
      if (code === 0) {
        replacements.push({ replacement: stdout.trim() || '(no output)', span })
      } else {
        const detail = (stderr.trim() || `exit ${code}`).slice(0, 200)
        replacements.push({ replacement: `[injected command failed: ${detail}]`, span })
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error)
      replacements.push({ replacement: `[injected command failed: ${detail.slice(0, 200)}]`, span })
    }
  }
  for (const { replacement, span } of replacements) {
    expanded = expanded.replace(span, () => replacement)
  }
  return expanded
}

/** $ARGUMENTS, ${ARGUMENTS}, and positional $0…$9 (whitespace-split). */
export function substituteArguments(instructions: string, args?: string): string {
  if (!/\$(?:ARGUMENTS|\{ARGUMENTS\}|\d)/.test(instructions)) return instructions
  const words = (args ?? '').split(/\s+/).filter(Boolean)
  return instructions
    .replace(/\$\{ARGUMENTS\}|\$ARGUMENTS/g, () => args ?? '')
    .replace(/\$(\d)/g, (whole, digit: string) => {
      const index = Number(digit)
      return index < words.length ? (words[index] ?? whole) : whole
    })
}

function defaultSkillCommandExecutor(cwd: string): NonNullable<SkillExpansionOptions['run']> {
  return async command => {
    const shell = process.platform === 'win32' ? 'cmd.exe' : '/bin/sh'
    const args = process.platform === 'win32' ? ['/d', '/s', '/c', command] : ['-c', command]
    const proc = Bun.spawn([shell, ...args], {
      cwd,
      stdin: 'ignore',
      stdout: 'pipe',
      stderr: 'pipe',
    })
    const killer = setTimeout(() => proc.kill(), INJECTION_TIMEOUT_MS)
    try {
      const [stdout, stderr, code] = await Promise.all([
        new Response(proc.stdout).text(),
        new Response(proc.stderr).text(),
        proc.exited,
      ])
      return {
        code,
        stdout: stdout.length > INJECTION_OUTPUT_CAP
          ? `${stdout.slice(0, INJECTION_OUTPUT_CAP)}\n… (truncated)`
          : stdout,
        stderr,
      }
    } finally {
      clearTimeout(killer)
    }
  }
}
