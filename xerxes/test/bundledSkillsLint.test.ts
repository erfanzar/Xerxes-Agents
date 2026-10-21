// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { join } from 'node:path'

import { BUNDLED_SKILLS_SOURCE_DIRECTORY } from '../scripts/copyBundledSkills.js'
import { lintHarness, type HarnessSkillDocument } from '../src/extensions/harnessLint.js'
import { parseSkillMarkdown, SkillRegistry } from '../src/extensions/skills.js'

async function bundledDocuments(): Promise<HarnessSkillDocument[]> {
  const paths = [...new Bun.Glob('**/SKILL.md').scanSync({
    cwd: BUNDLED_SKILLS_SOURCE_DIRECTORY,
    absolute: true,
  })].sort()
  return Promise.all(paths.map(async path => ({ path, content: await Bun.file(path).text() })))
}

async function bundledSkill(directory: string) {
  const path = join(BUNDLED_SKILLS_SOURCE_DIRECTORY, directory, 'SKILL.md')
  return parseSkillMarkdown(await Bun.file(path).text(), path)
}

test('every bundled skill is discoverable and has zero lint findings without user discovery roots', async () => {
  // Enumerate files, not just registry entries: shadowed or rejected documents must not disappear.
  const documents = await bundledDocuments()
  expect(documents.length).toBeGreaterThan(0)
  expect(lintHarness(documents)).toEqual([])

  const registry = new SkillRegistry()
  const discovered = await registry.discover(BUNDLED_SKILLS_SOURCE_DIRECTORY)
  expect(discovered).toHaveLength(documents.length)
  expect(registry.discoveryNotes).toEqual([])
})

test('flattened bundled metadata preserves descriptions, tags, platform restrictions and configuration', async () => {
  const notes = await bundledSkill('apple-notes')
  expect(notes.metadata.platforms).toEqual(['macos'])
  expect(notes.metadata.tags).toEqual(['Notes', 'Apple', 'macOS', 'note-taking'])
  expect(notes.instructions).toContain('License: MIT')
  expect(notes.instructions).toContain('Related skills: obsidian')
  expect(notes.instructions).toContain('Prerequisite commands: memo')

  const review = await bundledSkill('software-development/requesting-code-review')
  expect(review.metadata.description).toBe('Pre-commit verification pipeline — static security scan, baseline-aware quality gates, independent reviewer subagent, and auto-fix loop. Use after code changes and before committing, pushing, or opening a PR.')
  const wiki = await bundledSkill('research/llm-wiki')
  expect(wiki.instructions).toContain('Key: `wiki.path`')
  expect(wiki.instructions).toContain('Default: `~/wiki`')
  expect(wiki.instructions).toContain('Prompt: Wiki directory path')
  const designs = await bundledSkill('popular-web-designs')
  expect(designs.instructions).toContain('make it look like stripe')
  expect(designs.metadata.description).toContain('layout rules, and ready-to-use CSS values.')
})

test('bundled skill names match their stable directories while retaining previous names in the body', async () => {
  const renamed = [
    ['training/trl-fine-tuning', 'trl-fine-tuning', 'fine-tuning-with-trl'],
    ['training/peft', 'peft', 'peft-fine-tuning'],
    ['models/audiocraft', 'audiocraft', 'audiocraft-audio-generation'],
    ['models/segment-anything', 'segment-anything', 'segment-anything-model'],
    ['models/stable-diffusion', 'stable-diffusion', 'stable-diffusion-image-generation'],
    ['cloud/modal', 'modal', 'modal-serverless-gpu'],
    ['inference/gguf', 'gguf', 'gguf-quantization'],
    ['inference/vllm', 'vllm', 'serving-llms-vllm'],
    ['evaluation/lm-evaluation-harness', 'lm-evaluation-harness', 'evaluating-llms-harness'],
  ] as const
  for (const [directory, name, previousName] of renamed) {
    const skill = await bundledSkill(directory)
    expect(skill.metadata.name).toBe(name)
    expect(skill.instructions).toContain(`Previous skill name: ${previousName}`)
  }
})

test('upstream and research example helpers are explicitly user-project paths, not missing bundled scripts', async () => {
  const sam = await bundledSkill('models/segment-anything')
  expect(sam.instructions).toContain('not this skill bundle')
  expect(sam.instructions).toContain('$SAM_REPOSITORY/scripts/export_onnx_model.py')
  const paper = await bundledSkill('research/research-paper-writing')
  expect(paper.instructions).toContain('example helpers in that user-owned project')
  expect(paper.instructions).toContain('$RESEARCH_PROJECT/scripts/reproduce_table1.sh')
  expect(paper.instructions).toContain('$RESEARCH_PROJECT/scripts/make_figure2.py')
})
