// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { mkdir } from 'node:fs/promises'
import { join, resolve } from 'node:path'

export const SKILL_CREATE_AUTO = '<<auto>>'

const SKILL_CREATE_STEPS = [
  {
    key: 'what',
    question: 'What should this skill do? One or two sentences. Type `auto` to let me infer everything from this session, or `/cancel` to abort.',
    required: true,
  },
  {
    key: 'when',
    question: 'When should a future session activate this skill? Describe the trigger. Type `auto` to let me decide.',
    required: true,
  },
  {
    key: 'tools',
    question: 'Which tools or commands does the procedure use? List them, comma-separated. Type `auto` to let me decide.',
    required: true,
  },
  {
    key: 'pitfalls',
    question: 'Any pitfalls or things that went wrong worth recording? Press Enter to skip, or type `auto` to let me decide.',
    required: false,
  },
] as const

type SkillCreateAnswerKey = typeof SKILL_CREATE_STEPS[number]['key']

/** Which discovery root a generated skill is written to. */
export type SkillCreateScope = 'project' | 'user'

/**
 * The save-scope question, asked only when the host configured both roots.
 *
 * The two roots are not interchangeable: the project root is discovered first,
 * so a skill there shadows a same-named user skill, but it also travels with a
 * clone and therefore reaches the registry through the workspace-trust
 * predicate, which withholds any SKILL.md whose digest is not already trusted.
 */
const SKILL_CREATE_SCOPE_QUESTION = 'Where should this skill be saved? Type `project` to keep it with this '
  + 'repository (checked in, discovered before your personal skills, but workspace skills must be trusted '
  + 'before they load) or `user` for your Xerxes home skills root (available in every project, trusted '
  + 'already). Press Enter for `user`, or `/cancel` to abort.'

interface AwaitingNameState {
  readonly kind: 'awaiting_name'
  readonly sessionKey: string
}

interface InterviewState {
  readonly answers: Partial<Record<SkillCreateAnswerKey, string>>
  readonly kind: 'interview'
  readonly name: string
  readonly sessionKey: string
  /** Undefined until the save scope is settled; the directory is created at that moment. */
  scope: SkillCreateScope | undefined
  targetPath: string
}

type SkillCreateState = AwaitingNameState | InterviewState

export interface SkillCreateFlowOptions {
  /** Writable user-owned root for generated skill directories. */
  readonly skillsDirectory: string
  /** Explicit I/O seam for hosts and tests. Defaults to native recursive mkdir. */
  readonly ensureDirectory?: (path: string) => Promise<void>
  /**
   * Workspace-owned skills root. Supplying it enables the save-scope question;
   * without it there is only one place a skill can go, so the flow does not ask.
   */
  readonly projectSkillsDirectory?: string
}

export interface SkillCreateDraft {
  readonly announcement: string
  readonly name: string
  readonly prompt: string
  readonly scope: SkillCreateScope
  readonly targetPath: string
}

export type SkillCreateTransition =
  | { readonly kind: 'cancelled'; readonly message: string }
  | { readonly draft: SkillCreateDraft; readonly kind: 'draft' }
  | { readonly kind: 'prompt'; readonly message: string }

/**
 * Per-connection `/skill-create` interview that yields a safe synthetic turn.
 *
 * The flow deliberately does not write SKILL.md itself. It creates only the
 * bounded skill directory, then asks the configured native agent runner to
 * author the file through its normal tool and permission path.
 */
export class SkillCreateFlow {
  private readonly ensureDirectory: (path: string) => Promise<void>
  private readonly projectSkillsDirectory: string | undefined
  private readonly skillsDirectory: string
  private state: SkillCreateState | undefined

  constructor(options: SkillCreateFlowOptions) {
    this.skillsDirectory = resolve(options.skillsDirectory)
    this.projectSkillsDirectory = options.projectSkillsDirectory
      ? resolve(options.projectSkillsDirectory)
      : undefined
    this.ensureDirectory = options.ensureDirectory ?? (async path => {
      await mkdir(path, { recursive: true })
    })
  }

  get active(): boolean {
    return this.state !== undefined
  }

  ownsSession(sessionKey: string): boolean {
    return this.state?.sessionKey === sessionKey
  }

  /** Start an inline-name interview or request a name when none was supplied. */
  async start(rawName: string, sessionKey: string): Promise<SkillCreateTransition> {
    const name = sanitizeSkillSlug(rawName)
    if (!name) {
      this.state = { kind: 'awaiting_name', sessionKey }
      return prompt('What should this skill be called? Type a short kebab-case slug (e.g. `commit-helper`). `/cancel` to abort.')
    }
    return this.beginInterview(name, sessionKey)
  }

  /** Advance a pending interview, preserving blank optional answers verbatim. */
  async answer(sessionKey: string, rawText: string): Promise<SkillCreateTransition | undefined> {
    const state = this.state
    if (!state || state.sessionKey !== sessionKey) {
      return undefined
    }

    const answer = rawText.trim()
    if (isCancellation(answer)) {
      this.state = undefined
      return { kind: 'cancelled', message: 'Cancelled `/skill-create`.' }
    }

    if (state.kind === 'awaiting_name') {
      const name = sanitizeSkillSlug(answer)
      if (!name) {
        return prompt("That doesn't look like a valid slug. Use kebab-case letters/digits, e.g. `commit-helper`. `/cancel` to abort.")
      }
      return this.beginInterview(name, sessionKey)
    }

    if (state.scope === undefined) {
      const scope = parseSkillCreateScope(answer)
      if (!scope) {
        return prompt('Answer `project` or `user` (Enter for `user`), or `/cancel` to abort.')
      }
      return this.applyScope(state, scope)
    }

    if (answer.toLowerCase() === 'auto' || answer.toLowerCase() === '/auto') {
      for (const step of SKILL_CREATE_STEPS) {
        state.answers[step.key] ??= SKILL_CREATE_AUTO
      }
      return this.nextTransition(state)
    }

    const next = SKILL_CREATE_STEPS.find(step => state.answers[step.key] === undefined)
    if (!next) {
      return this.finishDraft(state)
    }
    if (next.required && !answer) {
      return prompt('That field is required. Type an answer, `auto` to let me decide, or `/cancel` to abort.')
    }
    state.answers[next.key] = answer
    return this.nextTransition(state)
  }

  /** Cancel this flow only when it belongs to the caller's active session. */
  cancel(sessionKey: string): boolean {
    if (!this.ownsSession(sessionKey)) {
      return false
    }
    this.state = undefined
    return true
  }

  private async beginInterview(name: string, sessionKey: string): Promise<SkillCreateTransition> {
    const state: InterviewState = {
      kind: 'interview',
      sessionKey,
      name,
      scope: undefined,
      targetPath: '',
      answers: {},
    }
    this.state = state
    // With one writable root there is no decision to put to the user, so the
    // directory is created up front exactly as it always was.
    return this.projectSkillsDirectory ? prompt(SKILL_CREATE_SCOPE_QUESTION) : this.applyScope(state, 'user')
  }

  /** Settle the save scope, create its bounded directory, and enter the questions. */
  private async applyScope(state: InterviewState, scope: SkillCreateScope): Promise<SkillCreateTransition> {
    const root = scope === 'project' ? this.projectSkillsDirectory ?? this.skillsDirectory : this.skillsDirectory
    const targetDirectory = resolve(root, state.name)
    await this.ensureDirectory(targetDirectory)
    state.scope = scope
    state.targetPath = join(targetDirectory, 'SKILL.md')
    return this.nextTransition(state)
  }

  private nextTransition(state: InterviewState): SkillCreateTransition {
    const next = SKILL_CREATE_STEPS.find(step => state.answers[step.key] === undefined)
    return next ? prompt(next.question) : this.finishDraft(state)
  }

  private finishDraft(state: InterviewState): SkillCreateTransition {
    this.state = undefined
    const answers = state.answers
    const autoKeys = SKILL_CREATE_STEPS
      .filter(step => answers[step.key] === SKILL_CREATE_AUTO)
      .map(step => step.key)
    const announcement = autoKeys.length
      ? `Drafting skill \`${state.name}\` — inferring ${autoKeys.join(', ')} from session context, saving to \`${state.targetPath}\`…`
      : `Drafting skill \`${state.name}\` from your answers — saving to \`${state.targetPath}\`…`
    return {
      kind: 'draft',
      draft: {
        announcement,
        name: state.name,
        scope: state.scope ?? 'user',
        targetPath: state.targetPath,
        prompt: draftPrompt(state.name, state.targetPath, answers),
      },
    }
  }
}

/** Map a free-text save-scope answer, treating an empty or deferred one as the trusted user root. */
export function parseSkillCreateScope(answer: string): SkillCreateScope | undefined {
  const normalized = answer.trim().toLowerCase()
  if (normalized === 'project' || normalized === 'p' || normalized === 'repo' || normalized === 'workspace') {
    return 'project'
  }
  // An empty or "you decide" answer picks the root that is trusted on load;
  // silently defaulting to the workspace root would produce a skill that
  // discovery then refuses, which reads as the flow having done nothing.
  if (normalized === '' || normalized === 'user' || normalized === 'u' || normalized === 'home'
    || normalized === 'auto' || normalized === '/auto') {
    return 'user'
  }
  return undefined
}

/** Match the daemon's historical slug rules while rejecting traversal input. */
export function sanitizeSkillSlug(rawName: string): string {
  return [...rawName.trim().toLowerCase()]
    .filter(character => /[a-z0-9_-]/.test(character))
    .join('')
    .replace(/^[-_]+|[-_]+$/g, '')
}

function prompt(message: string): SkillCreateTransition {
  return { kind: 'prompt', message }
}

function isCancellation(value: string): boolean {
  return value.toLowerCase() === '/cancel' || value.toLowerCase() === 'cancel'
}

function draftPrompt(name: string, targetPath: string, answers: Partial<Record<SkillCreateAnswerKey, string>>): string {
  const render = (label: string, key: SkillCreateAnswerKey, inferHint: string): string => {
    const value = answers[key]?.trim() ?? ''
    if (value === SKILL_CREATE_AUTO) {
      return `**${label}:** _auto — ${inferHint}_\n\n`
    }
    return value ? `**${label}:** ${value}\n\n` : ''
  }
  const pitfalls = answers.pitfalls?.trim() ?? ''
  const pitfallsBlock = pitfalls === SKILL_CREATE_AUTO
    ? '**Pitfalls:** _auto — list any real issues we hit this session; omit the `# Pitfalls` section if none occurred._\n\n'
    : pitfalls
      ? `**User-reported pitfalls:** ${pitfalls}\n\n`
      : 'User reported no pitfalls — omit the `# Pitfalls` section unless something in this session genuinely went wrong.\n\n'
  return [
    `Write a reusable agent skill called **\`${name}\`**. Do not ask follow-up questions — write the SKILL.md directly. Any field marked _auto_ below is yours to fill in based on what we did in this session so far.`,
    '## Inputs',
    render('What the skill should do', 'what', 'infer from what we worked on in this session.'),
    render('Activation trigger', 'when', 'pick a sensible trigger (e.g. when the user says the skill name, or when the task description matches the work we just did).'),
    render('Tools / commands it uses', 'tools', 'list the tools we actually invoked this session.'),
    pitfallsBlock,
    '## Output',
    `Write the file to **\`${targetPath}\`** using the Write tool. The file must be valid Markdown with this exact structure:`,
    `1. YAML frontmatter delimited by \`---\` lines, containing:\n   - \`name: ${name}\` (use this exact slug)\n   - \`description:\` (one short line — derived from "what the skill should do")\n   - \`version: 0.1.0\`\n   - \`tags: [...]\` (short list of topics / domain hints)\n   - \`required_tools: [...]\` (tool names from the tools field)`,
    [
      '2. `# When to use` — based on the activation trigger.',
      '3. `# Procedure` — numbered steps grounded in the tool list. Every step must name, inline and in the same'
      + ' step, the observable signal that it worked: the exact command output, the file that now exists, the'
      + ' status code, the test that now passes. A step whose result cannot be checked before the next step runs'
      + ' is not a step — split it or say what to look at. Prefix any step that is not trivially undoable'
      + ' (deletes data, force-pushes, publishes, migrates, mutates a remote system, spends money) with'
      + ' `**Irreversible.**` and state what to confirm before running it.',
      '4. `# Pitfalls` — only if there were real pitfalls.',
      '5. `# Verification` — the single end-to-end signal that the whole procedure succeeded. Do not restate the'
      + ' per-step signals here; a reader must not have to reach this section to know whether step 3 worked.',
    ].join('\n'),
    'After writing, confirm the final path in one short sentence. Do not output the SKILL.md body in chat; the Write tool is the only delivery channel.',
  ].filter(Boolean).join('\n\n')
}
