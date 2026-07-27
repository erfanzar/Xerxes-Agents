// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { realpathSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { basename, dirname, isAbsolute, resolve } from "node:path";

import type { Diagnosis, DiagnosisSeverity } from "../runtime/doctor.js";
import {
  FORBIDDEN_FRONTMATTER_KEYS,
  isStrictDescendant,
  parseSkillMarkdown,
  SKILL_FRONTMATTER_KEYS,
  skillInstructionsAreSafe,
  type Skill,
} from "./skills.js";

export interface SkillLintReport {
  readonly diagnoses: readonly Diagnosis[];
  /** True when the strict parser found nothing the tolerant parser would silently reshape. */
  readonly ok: boolean;
  readonly path: string;
}

const FRONTMATTER_PATTERN = /^---\s*\r?\n([\s\S]*?)\r?\n---\s*\r?\n?([\s\S]*)$/;
const BLOCK_SCALAR_PATTERN = /^[|>][+-]?\d*$/u;
/** Every finding is phrased against the tolerant parser's silent reinterpretation of the line. */
const MISREAD = "this line parses to something other than what you wrote";

/**
 * Hard-fail on everything `parseSkillMarkdown` forgives.
 *
 * The tolerant parser exists so one malformed third-party skill cannot break discovery, but it
 * turns an author's typo into a skill that quietly does not exist or exists under a name nobody
 * asked for. This second pass is the authoring-time counterpart: it refuses the same documents
 * discovery would accept-and-mangle, and reports in the doctor report's shape.
 */
export function lintSkillMarkdown(content: string, sourcePath: string): SkillLintReport {
  const diagnoses: Diagnosis[] = [];
  let skill: Skill | undefined;
  try {
    skill = parseSkillMarkdown(content, sourcePath);
  } catch (error) {
    diagnoses.push(
      fail(
        "document-unparsable",
        error instanceof Error ? error.message : String(error),
        "A skill document must be frontmatter followed by instructions, with no wrapper text.",
      ),
    );
  }

  const frontmatter = content.match(FRONTMATTER_PATTERN);
  if (!frontmatter) {
    diagnoses.push(
      fail(
        "frontmatter-missing",
        "the document has no '---' frontmatter block, so every field falls back to a default",
        "Open the file with '---', one 'key: value' per line, then a closing '---'.",
      ),
    );
  } else {
    diagnoses.push(...lintFrontmatterBlock(frontmatter[1] ?? ""));
  }

  if (skill !== undefined) {
    diagnoses.push(...lintParsedSkill(skill, sourcePath));
  }

  if (!diagnoses.length) {
    diagnoses.push({
      name: "skill-lint",
      severity: "ok",
      message: `${sourcePath} parses exactly as written`,
      fixHint: "",
    });
  }
  return Object.freeze({
    diagnoses: Object.freeze(diagnoses),
    ok: !diagnoses.some((item) => item.severity === "fail"),
    path: sourcePath,
  });
}

/** Read and lint one SKILL.md from disk. */
export async function lintSkillFile(path: string): Promise<SkillLintReport> {
  let content: string;
  try {
    content = await readFile(path, "utf8");
  } catch (error) {
    return Object.freeze({
      diagnoses: Object.freeze([
        fail(
          "document-unreadable",
          error instanceof Error ? error.message : String(error),
          "Point the linter at a readable SKILL.md file.",
        ),
      ]),
      ok: false,
      path,
    });
  }
  return lintSkillMarkdown(content, path);
}

function lintFrontmatterBlock(block: string): Diagnosis[] {
  const diagnoses: Diagnosis[] = [];
  const seenKeys = new Set<string>();
  let listKey: string | undefined;
  let lineNumber = 0;
  for (const rawLine of block.split(/\r?\n/)) {
    lineNumber += 1;
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const at = `frontmatter line ${lineNumber}`;

    if (line.startsWith("- ")) {
      diagnoses.push(...lintListItem(line.slice(2).trim(), listKey, at));
      continue;
    }

    const separator = line.indexOf(":");
    if (separator < 0) {
      diagnoses.push(
        fail(
          "frontmatter-unparsed-line",
          `${at} has no ':' and is not a '- ' list item, so it is dropped entirely: ${line}`,
          "Write 'key: value', or prefix the line with '- ' to extend the list above it.",
        ),
      );
      listKey = undefined;
      continue;
    }

    const key = line.slice(0, separator).trim();
    const value = line.slice(separator + 1).trim();
    if (rawLine.length !== rawLine.trimStart().length) {
      // The flat parser has no notion of depth: an indented mapping lands as a sibling top-level key.
      diagnoses.push(
        fail(
          "frontmatter-nested-mapping",
          `${at} is indented under another key, but ${MISREAD}: '${key}' becomes a top-level field`,
          "Flatten the mapping to top-level 'key: value' lines or a '- ' list.",
        ),
      );
      listKey = undefined;
      continue;
    }
    if (FORBIDDEN_FRONTMATTER_KEYS.has(key)) {
      diagnoses.push(
        fail(
          "frontmatter-forbidden-key",
          `${at} sets the reserved key '${key}', which is discarded without a word`,
          "Remove the key; prototype-shaped names can never become skill metadata.",
        ),
      );
      listKey = undefined;
      continue;
    }
    if (!SKILL_FRONTMATTER_KEYS.has(key)) {
      diagnoses.push(
        fail(
          "frontmatter-unknown-key",
          `${at} sets '${key}', which no skill field reads, so its value is lost`,
          `Use one of: ${[...SKILL_FRONTMATTER_KEYS].sort().join(", ")}.`,
        ),
      );
      listKey = undefined;
      continue;
    }
    if (seenKeys.has(key)) {
      diagnoses.push(
        fail(
          "frontmatter-duplicate-key",
          `${at} repeats '${key}'; the last occurrence silently wins`,
          "Keep one definition per key.",
        ),
      );
    }
    seenKeys.add(key);
    diagnoses.push(...lintScalarValue(key, value, at));
    listKey = value ? undefined : key;
  }
  return diagnoses;
}

function lintListItem(item: string, listKey: string | undefined, at: string): Diagnosis[] {
  if (listKey === undefined) {
    return [
      fail(
        "frontmatter-list-item-without-key",
        `${at} is a list item with no key above it, so it is dropped: - ${item}`,
        "Put 'key:' on its own line directly above the list.",
      ),
    ];
  }
  if (/:(?:\s|$)/u.test(item)) {
    return [
      fail(
        "frontmatter-nested-mapping",
        `${at} puts a mapping inside the '${listKey}' list, but ${MISREAD}: the whole line becomes one string`,
        "List entries must be plain scalars.",
      ),
    ];
  }
  return [];
}

function lintScalarValue(key: string, value: string, at: string): Diagnosis[] {
  if (BLOCK_SCALAR_PATTERN.test(value)) {
    return [
      fail(
        "frontmatter-block-scalar",
        `${at} opens a block scalar for '${key}', but ${MISREAD}: '${key}' becomes the literal '${value}'`,
        "Collapse the value onto one line, or move the prose into the instruction body.",
      ),
    ];
  }
  if (value.startsWith("[") && !value.endsWith("]")) {
    return [
      fail(
        "frontmatter-multiline-list",
        `${at} opens a '${key}' list that does not close on the same line, and ${MISREAD}`,
        "Keep '[a, b]' on one line, or use '- ' items on the following lines.",
      ),
    ];
  }
  return [];
}

function lintParsedSkill(skill: Skill, sourcePath: string): Diagnosis[] {
  const diagnoses: Diagnosis[] = [];
  const skillRoot = dirname(resolve(sourcePath));
  if (skill.nameFromDirectory) {
    diagnoses.push(
      fail(
        "name-missing",
        `no 'name:' was declared, so the skill is registered as '${skill.metadata.name}' after its directory`,
        "Declare 'name:' explicitly; a typo'd key otherwise yields a skill under an unexpected name.",
      ),
    );
  } else if (skill.metadata.name !== basename(skillRoot)) {
    diagnoses.push(
      fail(
        "name-directory-mismatch",
        `'name: ${skill.metadata.name}' disagrees with the directory '${basename(skillRoot)}'`,
        "Rename the directory or the skill so activation and installation agree.",
      ),
    );
  }
  if (!skill.metadata.description) {
    diagnoses.push(
      fail(
        "description-missing",
        "no 'description:' was declared, so the skill index lists it as 'No description'",
        "Describe in one line when this skill should be used.",
      ),
    );
  }
  for (const resource of skill.metadata.resources) {
    const escape = resourceEscape(skillRoot, resource);
    if (escape !== undefined) {
      diagnoses.push(
        fail(
          "resource-escapes-root",
          `declared resource '${resource}' ${escape}`,
          "Declare only paths below the skill's own directory.",
        ),
      );
    }
  }
  if (!skillInstructionsAreSafe(skill)) {
    diagnoses.push(
      fail(
        "instructions-injection",
        "the instruction body is flagged by the prompt-injection scan, so discovery will drop this skill",
        "Rewrite the flagged passage; instructions that read as overrides never reach a model.",
      ),
    );
  }
  return diagnoses;
}

/** Describe how a declared resource leaves the skill root, or `undefined` when it stays inside. */
function resourceEscape(skillRoot: string, resource: string): string | undefined {
  if (isAbsolute(resource)) return "is absolute and therefore outside the skill root";
  const target = resolve(skillRoot, resource);
  if (!isStrictDescendant(skillRoot, target)) return `resolves to ${target}, outside the skill root`;
  const canonicalRoot = canonicalPath(skillRoot);
  const canonicalTarget = canonicalPath(target);
  if (canonicalRoot === undefined || canonicalTarget === undefined) return undefined;
  // A symlinked resource passes the lexical check but reads a file the author does not control.
  if (!isStrictDescendant(canonicalRoot, canonicalTarget)) {
    return `links to ${canonicalTarget}, outside the skill root`;
  }
  return undefined;
}

function canonicalPath(path: string): string | undefined {
  try {
    return realpathSync(path);
  } catch {
    return undefined;
  }
}

function fail(name: string, message: string, fixHint: string): Diagnosis {
  const severity: DiagnosisSeverity = "fail";
  return Object.freeze({ name, severity, message, fixHint });
}
