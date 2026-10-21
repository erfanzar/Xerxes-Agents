// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { realpathSync, statSync } from 'node:fs'
import { readFile } from 'node:fs/promises'
import { dirname, isAbsolute, resolve } from 'node:path'

import { resolveCommand } from '../bridge/commands.js'
import { lintSkillMarkdown } from './skillLint.js'
import { BUNDLED_SKILLS_DIRECTORY, isStrictDescendant, parseSkillMarkdown, type Skill } from './skills.js'

export interface HarnessLintFinding {
  readonly code: string
  /** `info` never blocks authoring and is not a problem to fix; it explains a deliberate state. */
  readonly severity: 'fail' | 'warn' | 'info'
  readonly path: string
  readonly message: string
  readonly fixHint: string
}

export interface HarnessSkillDocument {
  readonly path: string
  readonly content: string
  /** Files supplied with an unsaved bundle, relative to its SKILL.md. Disk files are not implicitly shipped. */
  readonly bundleFiles?: readonly string[]
  /** Updates may also reference regular, contained files already installed in this bundle. */
  readonly includeInstalledFiles?: boolean
}

/** Accept resolved AgentDefinitions as well as an author's tools/allowed_tools draft. */
export interface HarnessAgentDefinition {
  readonly name?: string
  readonly path?: string
  readonly source?: string
  readonly tools?: readonly string[]
  readonly allowedTools?: readonly string[] | null
  readonly allowed_tools?: readonly string[] | null
}

export interface HarnessLintOptions {
  readonly agents?: readonly HarnessAgentDefinition[]
  /** Omission means readiness is unknown, not that every tool is missing. */
  readonly knownToolNames?: ReadonlySet<string>
  /** Root of the skills shipped with Xerxes; defaults to the resolved bundled directory. */
  readonly bundledRoot?: string
}

export interface DiscoveredHarnessLintOptions extends HarnessLintOptions {
  /** Include shadowed/rejected discovery candidates so a registry's first-root-wins map hides no problems. */
  readonly additionalPaths?: readonly string[]
}

/** Validate every proposed document before returning; no finding short-circuits the rest. */
export function lintHarness(skills: readonly HarnessSkillDocument[], options: HarnessLintOptions = {}): HarnessLintFinding[] {
  const findings: HarnessLintFinding[] = []
  const names = new Map<string, Set<string>>()
  const visited = new Set<string>()
  for (const document of skills) {
    const identity = resolve(document.path)
    if (visited.has(identity)) continue
    visited.add(identity)
    for (const diagnosis of lintSkillMarkdown(document.content, document.path).diagnoses) {
      if (diagnosis.severity !== 'fail' && diagnosis.severity !== 'warn') continue
      findings.push({
        code: diagnosis.name,
        severity: diagnosis.name === 'description-missing' ? 'warn' : diagnosis.severity,
        path: document.path,
        message: diagnosis.name === 'description-missing'
          ? 'The skill has no description, so it is never discoverable by description-based activation.'
          : diagnosis.message,
        fixHint: diagnosis.fixHint,
      })
    }
    findings.push(...lintBundlePaths(document))
    let skill: Skill
    try { skill = parseSkillMarkdown(document.content, document.path) }
    catch { continue } // The strict linter already reported the parsing error, alongside frontmatter findings.
    if (!skill.instructions.trim()) findings.push({
      code: 'instructions-missing', severity: 'fail', path: document.path,
      message: 'The skill has no instructions to execute.',
      fixHint: 'Provide a non-empty instruction body for this skill.',
    })
    const paths = names.get(skill.metadata.name) ?? new Set<string>()
    paths.add(document.path)
    names.set(skill.metadata.name, paths)
    if (resolveCommand(skill.metadata.name) !== undefined) {
      findings.push({
        code: 'builtin-command-conflict', severity: 'fail', path: document.path,
        message: `Skill '${skill.metadata.name}' clashes with built-in slash command /${skill.metadata.name}.`,
        fixHint: 'Rename the skill and its directory to a name not reserved by a built-in command or alias.',
      })
    }
    for (const script of invokedScripts(skill.instructions)) {
      const problem = scriptProblem(document, script)
      if (problem) findings.push(problem)
    }
  }
  const bundledRoot = resolve(options.bundledRoot ?? BUNDLED_SKILLS_DIRECTORY)
  for (const [name, paths] of names) {
    if (paths.size < 2) continue
    const bundled = [...paths].filter(path => isStrictDescendant(bundledRoot, resolve(path)))
    const own = [...paths].filter(path => !bundled.includes(path))
    // One user/project copy over one bundled skill is a deliberate override:
    // discovery prefers the user's root by design. Report it once, on the
    // copy that wins, instead of warning about the bundled skill nobody edits.
    if (bundled.length === 1 && own.length === 1) {
      findings.push({
        code: 'skill-overrides-bundled', severity: 'info', path: own[0]!,
        message: `Skill '${name}' overrides the bundled skill at ${bundled[0]}. Discovery uses this copy.`,
        fixHint: 'Nothing to do if the override is intentional. To use the bundled version instead, remove or rename this copy.',
      })
      continue
    }
    for (const path of paths) findings.push({
      code: 'skill-name-duplicate', severity: 'warn', path,
      message: `Skill '${name}' occurs in multiple sources: ${[...paths].join(', ')}. Discovery only keeps the first source.`,
      fixHint: 'Give each skill a distinct name or explicitly remove the unintended duplicate from discovery roots.',
    })
  }
  if (options.knownToolNames !== undefined) {
    for (const agent of options.agents ?? []) {
      for (const tool of new Set([...(agent.tools ?? []), ...(agent.allowedTools ?? []), ...(agent.allowed_tools ?? [])])) {
        if (knownTool(tool, options.knownToolNames)) continue
        findings.push({
          code: 'agent-tool-unknown', severity: 'warn', path: agent.path ?? `${agent.source ?? 'agent'}:${agent.name ?? 'unnamed'}`,
          message: `Agent${agent.name ? ` '${agent.name}'` : ''} declares tool '${tool}', which is not registered in the supplied known-tool set.`,
          fixHint: 'Correct the tool name or register its host/plugin implementation before using this agent.',
        })
      }
    }
  }
  return findings
}

/** Read all discovered sources without executing any skill, script, or agent. Read errors are findings too. */
export async function lintDiscoveredHarness(skills: readonly Skill[], options: DiscoveredHarnessLintOptions = {}): Promise<HarnessLintFinding[]> {
  const documents: HarnessSkillDocument[] = []
  const unreadable: HarnessLintFinding[] = []
  const paths = new Set([...skills.map(skill => skill.sourcePath), ...(options.additionalPaths ?? [])])
  for (const path of paths) {
    try { documents.push({ path, content: await readFile(path, 'utf8') }) }
    catch (error) {
      unreadable.push({
        code: 'document-unreadable', severity: 'fail', path,
        message: error instanceof Error ? error.message : String(error),
        fixHint: 'Make this SKILL.md readable or remove its stale discovery entry.',
      })
    }
  }
  return [...lintHarness(documents, options), ...unreadable]
}

/** One complete retry message, grouped by source, with no preview limits or discarded findings. */
export function renderLintReport(findings: readonly HarnessLintFinding[]): string {
  if (!findings.length) return 'Harness lint: no problems found.'
  const paths = new Map<string, HarnessLintFinding[]>()
  for (const finding of findings) {
    const group = paths.get(finding.path) ?? []
    group.push(finding)
    paths.set(finding.path, group)
  }
  const failures = findings.filter(finding => finding.severity === 'fail').length
  const warnings = findings.filter(finding => finding.severity === 'warn').length
  const notes = findings.length - failures - warnings
  return [
    `Harness lint: ${failures} failure(s), ${warnings} warning(s)${notes ? `, ${notes} note(s)` : ''}.`,
    ...[...paths].map(([path, group]) => [
      path,
      ...group.map(finding => `  - [${finding.severity}] ${finding.code}: ${finding.message}\n    Fix: ${finding.fixHint}`),
    ].join('\n')),
  ].join('\n\n')
}

function lintBundlePaths(document: HarnessSkillDocument): HarnessLintFinding[] {
  const root = dirname(resolve(document.path))
  const findings: HarnessLintFinding[] = []
  for (const file of document.bundleFiles ?? []) {
    const target = resolve(root, file)
    if (file.includes('\\') || file.includes('\0') || isAbsolute(file) || !isStrictDescendant(root, target) || target === resolve(document.path)) {
      findings.push({
        code: 'bundle-file-invalid', severity: 'fail', path: document.path,
        message: `Supplied bundle file '${file}' escapes its skill root, has an invalid path, or replaces SKILL.md.`,
        fixHint: 'Use forward-slash relative file paths strictly inside the bundle, and supply SKILL.md through the skill fields.',
      })
    }
  }
  return findings
}

function scriptProblem(document: HarnessSkillDocument, script: string): HarnessLintFinding | undefined {
  const root = dirname(resolve(document.path))
  const target = resolve(root, script)
  const finding = (code: string, message: string): HarnessLintFinding => ({
    code, severity: 'fail', path: document.path, message,
    fixHint: 'Ship the referenced helper inside this skill bundle or change the instruction to an installed, contained script path.',
  })
  if (!isStrictDescendant(root, target)) return finding('script-escapes-root', `Invoked script '${script}' resolves outside the skill bundle.`)
  if (document.bundleFiles?.some(file => resolve(root, file) === target)) return undefined
  if (document.bundleFiles === undefined || document.includeInstalledFiles) {
    try {
      const canonicalRoot = realpathSync(root)
      const canonicalTarget = realpathSync(target)
      if (!isStrictDescendant(canonicalRoot, canonicalTarget)) return finding('script-escapes-root', `Invoked script '${script}' links outside the skill bundle.`)
      if (statSync(canonicalTarget).isFile()) return undefined
    } catch (error) {
      if (!isMissing(error)) return finding('script-unreadable', `Cannot inspect invoked script '${script}': ${error instanceof Error ? error.message : String(error)}`)
    }
  }
  // An installed skill with no scripts/ folder of its own is most likely
  // describing the user's project (a repo's scripts/export.py), not a helper it
  // forgot to ship. Authoring (bundleFiles supplied) stays strict: the model
  // must ship the file or reword the instruction.
  if (document.bundleFiles === undefined && !bundleHasScripts(root)) {
    return {
      code: 'script-missing', severity: 'warn', path: document.path,
      message: `Invoked script '${script}' is not in this skill bundle, which ships no scripts/ folder; if it is a path in the user's project, say so in the instructions.`,
      fixHint: 'Ship the helper under scripts/ in this bundle, or make clear the path belongs to the project being worked on.',
    }
  }
  return finding('script-missing', `Invoked script '${script}' is not shipped as a regular file in this skill bundle.`)
}

function bundleHasScripts(root: string): boolean {
  try { return statSync(resolve(root, 'scripts')).isDirectory() } catch { return false }
}

/** A path inside the skill bundle's own `scripts/` folder. */
const BUNDLE_SCRIPT = /^(?:\.\/)?scripts\/[^/\s]/u

/** Conservative command-token scan, not a shell parser: no expansion, execution, or content rewriting. */
function invokedScripts(instructions: string): Set<string> {
  const scripts = new Set<string>()
  const interpreter = /^(?:bun|deno|node|tsx|ts-node|bash|sh|zsh|fish|pwsh|powershell|python\d?(?:\.\d+)?|ruby|perl|lua)$/u
  const extension = /\.(?:[cm]?[jt]sx?|sh|bash|zsh|fish|ps1|py|rb|pl|lua|bat|cmd)$/iu
  for (const line of instructions.split(/\r?\n/)) {
    const tokens = [...line.matchAll(/"([^"\r\n]*)"|'([^'\r\n]*)'|([^\s`"'|;&()<>]+)/gu)]
      .map(match => ({ value: (match[1] ?? match[2] ?? match[3] ?? '').replace(/[.,:;]+$/u, ''), offset: match.index }))
    for (let index = 0; index < tokens.length; index += 1) {
      const { value: token, offset } = tokens[index]!
      const prefix = line.slice(0, offset)
      const commandPosition = /^\s*(?:[$>]\s*)?`?$/u.test(prefix) || /[;&|]\s*$/u.test(prefix) || /\b(?:run|execute|invoke|call)\s+`?$/iu.test(prefix)
      const relativePath = !isAbsolute(token) && !/^[a-z][a-z\d+.-]*:/iu.test(token)
      // Only a bundle-relative `scripts/...` path is something the skill must
      // ship. `python manage.py`, `bun run build.ts`, `node_modules/.bin/tsc`
      // and tools from other projects are the user's commands, not the bundle's.
      if (commandPosition && relativePath && BUNDLE_SCRIPT.test(token)) scripts.add(token)
      if (!interpreter.test(token) || !(commandPosition || /`$/u.test(prefix))) continue
      let argumentIndex = index + 1
      while (tokens[argumentIndex]?.value.startsWith('-') || ['run', 'exec', '--'].includes(tokens[argumentIndex]?.value ?? '')) argumentIndex += 1
      const argument = tokens[argumentIndex]?.value
      if (argument && !isAbsolute(argument) && !/^[a-z][a-z\d+.-]*:/iu.test(argument) && BUNDLE_SCRIPT.test(argument) && extension.test(argument)) scripts.add(argument)
    }
  }
  return scripts
}

function knownTool(pattern: string, names: ReadonlySet<string>): boolean {
  if (pattern === '*' || names.has(pattern)) return true
  if (!pattern.includes('*')) return false
  const expression = new RegExp(`^${pattern.split('*').map(part => part.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('.*')}$`)
  return [...names].some(name => expression.test(name))
}

function isMissing(error: unknown): boolean {
  return typeof error === 'object' && error !== null && 'code' in error && (error.code === 'ENOENT' || error.code === 'ENOTDIR')
}
