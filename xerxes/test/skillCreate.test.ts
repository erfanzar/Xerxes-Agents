// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SKILL_CREATE_AUTO, SkillCreateFlow, sanitizeSkillSlug } from '../src/daemon/skillCreate.js'

test('skill-create flow validates a deferred slug, creates only its bounded directory, and drafts after answers', async () => {
  const ensured: string[] = []
  const flow = new SkillCreateFlow({
    skillsDirectory: '/skills',
    ensureDirectory: async directory => { ensured.push(directory) },
  })

  let transition = await requiredTransition(flow.start('', 'session-a'))
  expect(transition).toEqual(expect.objectContaining({ kind: 'prompt', message: expect.stringContaining('What should this skill be called?') }))
  expect(flow.ownsSession('session-a')).toBeTrue()

  transition = await requiredTransition(flow.answer('session-a', '***'))
  expect(transition).toEqual(expect.objectContaining({ kind: 'prompt', message: expect.stringContaining("doesn't look like a valid slug") }))

  transition = await requiredTransition(flow.answer('session-a', 'Commit Helper!'))
  expect(ensured).toEqual(['/skills/commithelper'])
  expect(transition).toEqual(expect.objectContaining({ kind: 'prompt', message: expect.stringContaining('What should this skill do?') }))

  transition = await requiredTransition(flow.answer('session-a', 'Prepare clean commits.'))
  expect(transition).toEqual(expect.objectContaining({ kind: 'prompt', message: expect.stringContaining('When should') }))
  transition = await requiredTransition(flow.answer('session-a', 'When a user asks for a commit.'))
  transition = await requiredTransition(flow.answer('session-a', 'git status, git diff'))
  expect(transition).toEqual(expect.objectContaining({ kind: 'prompt', message: expect.stringContaining('pitfalls') }))

  transition = await requiredTransition(flow.answer('session-a', ''))
  if (transition.kind !== 'draft') throw new Error('Expected a final skill draft')
  expect(transition.draft).toMatchObject({
    name: 'commithelper',
    targetPath: '/skills/commithelper/SKILL.md',
    announcement: expect.stringContaining('Drafting skill `commithelper`'),
  })
  expect(transition.draft.prompt).toContain('Prepare clean commits.')
  expect(transition.draft.prompt).toContain('User reported no pitfalls')
  expect(flow.active).toBeFalse()
})

test('skill-create auto completes remaining fields and cancellation cannot cross session boundaries', async () => {
  const flow = new SkillCreateFlow({
    skillsDirectory: '/skills',
    ensureDirectory: async () => undefined,
  })
  await flow.start('release-notes', 'owner')
  expect(await flow.answer('other', 'auto')).toBeUndefined()
  const transition = await requiredTransition(flow.answer('owner', 'auto'))
  if (transition.kind !== 'draft') throw new Error('Expected an auto draft')
  expect(transition.draft.announcement).toContain('what, when, tools, pitfalls')
  expect(transition.draft.prompt).toContain('**What the skill should do:** _auto')
  expect(transition.draft.prompt).not.toContain(SKILL_CREATE_AUTO)

  await flow.start('another', 'owner')
  expect(flow.cancel('other')).toBeFalse()
  expect(flow.cancel('owner')).toBeTrue()
  expect(flow.active).toBeFalse()
})

test('skill-create slug normalization rejects traversal and trims separator edges', () => {
  expect(sanitizeSkillSlug(' ../Not a Skill/ ')).toBe('notaskill')
  expect(sanitizeSkillSlug('__release-notes__')).toBe('release-notes')
  expect(sanitizeSkillSlug('...')).toBe('')
})

async function requiredTransition(value: Promise<import('../src/daemon/skillCreate.js').SkillCreateTransition | undefined>) {
  const transition = await value
  if (!transition) throw new Error('Expected skill-create transition')
  return transition
}
