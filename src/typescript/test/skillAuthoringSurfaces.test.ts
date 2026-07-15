// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'

import { SkillDrafter } from '../src/extensions/skillAuthoring/drafter.js'
import {
  SkillImprover,
  bumpPatchVersion,
  extractSkillName,
  extractSkillVersion,
} from '../src/extensions/skillAuthoring/improver.js'
import { SkillTelemetry, attachSkillTelemetry } from '../src/extensions/skillAuthoring/telemetry.js'
import { ToolSequenceTracker } from '../src/extensions/skillAuthoring/tracker.js'

function observedCandidate() {
  const tracker = new ToolSequenceTracker({ now: () => 1 })
  tracker.beginTurn({ agentId: 'coder', turnId: 'turn-1', userPrompt: 'configure CI' })
  tracker.recordCall({ toolName: 'Read', arguments: { path: 'ci.yml' } })
  tracker.recordCall({ toolName: 'Edit', arguments: { path: 'ci.yml' } })
  return tracker.endTurn('CI configured')
}

test('SkillDrafter reuses canonical proposals and persists only through an injected store', async () => {
  const persisted: string[] = []
  let refinements = 0
  const drafter = new SkillDrafter({
    refiner: {
      refine: ({ proposal }) => {
        refinements += 1
        return proposal.markdown
      },
    },
    store: {
      persist: ({ proposal }) => {
        persisted.push(proposal.name)
        return { location: 'memory://skills/' + proposal.name }
      },
    },
  })

  const result = await drafter.draft(observedCandidate())
  expect(result).toMatchObject({
    persisted: true,
    persistence: { location: 'memory://skills/configure-ci' },
    proposal: { refinement: 'applied' },
  })
  expect(result.markdown).toContain('# Procedure')
  expect(refinements).toBe(1)
  expect(persisted).toEqual(['configure-ci'])

  const preview = await drafter.draft(observedCandidate(), { persist: false })
  expect(preview.persisted).toBeFalse()
  expect(persisted).toEqual(['configure-ci'])
})

test('SkillImprover preserves name, bumps versions, and delegates backup and writes to its document port', async () => {
  const location = 'memory://skills/ci/SKILL.md'
  const documents = new Map<string, string>([[location, [
    '---',
    'name: ci-helper',
    'version: 1.2.9',
    '---',
    '# Procedure',
  ].join('\n')]])
  const backups: string[] = []
  const writes: string[] = []
  const improver = new SkillImprover({
    documents: {
      read: path => documents.get(path),
      backup: ({ location: path, markdown, maxBackups, version }) => {
        expect(maxBackups).toBe(2)
        backups.push(path + ':' + version + ':' + markdown.length)
        return path + '.' + version + '.bak'
      },
      write: ({ location: path, markdown }) => {
        writes.push(path)
        documents.set(path, markdown)
      },
    },
  })

  const result = await improver.improve(location, observedCandidate(), { maxBackups: 2 })
  expect(result).toEqual({
    improved: true,
    oldVersion: '1.2.9',
    newVersion: '1.2.10',
    reason: '',
    skillLocation: location,
    backupLocation: location + '.1.2.9.bak',
  })
  expect(backups).toHaveLength(1)
  expect(writes).toEqual([location])
  expect(documents.get(location)).toContain('name: ci-helper')
  expect(documents.get(location)).toContain('version: 1.2.10')
  expect(await improver.improve('memory://missing/SKILL.md', observedCandidate())).toMatchObject({
    improved: false,
    reason: 'missing skill at memory://missing/SKILL.md',
  })
  expect(bumpPatchVersion('bad-version')).toBe('0.1.1')
  expect(extractSkillName(documents.get(location) ?? '')).toBe('ci-helper')
  expect(extractSkillVersion(documents.get(location) ?? '')).toBe('1.2.10')
})

test('telemetry consumes only a caller-owned source and can be detached', () => {
  let listener: ((event: { readonly kind: 'used'; readonly outcome: string; readonly skillName: string }) => void) | undefined
  let unsubscribed = 0
  const telemetry = new SkillTelemetry()
  const detach = attachSkillTelemetry(telemetry, {
    subscribe: eventListener => {
      listener = eventListener
      return () => {
        unsubscribed += 1
      }
    },
  })

  listener?.({ kind: 'used', skillName: 'ci-helper', outcome: 'success' })
  expect(telemetry.stats('ci-helper')).toMatchObject({ invocations: 1, successes: 1 })
  detach()
  listener?.({ kind: 'used', skillName: 'ci-helper', outcome: 'failure' })
  expect(telemetry.stats('ci-helper')).toMatchObject({ invocations: 1, failures: 0 })
  detach()
  expect(unsubscribed).toBe(1)
})
