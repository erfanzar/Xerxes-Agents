// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { mergeCompletionItems, slashCompletionsFromCatalog } from '../hooks/useCompletion.js'
import { rankCompletionItems } from '../lib/completion.js'
import type { SlashCatalog } from '../types.js'

const catalog: SlashCatalog = {
  canon: {
    '/deepscan': '/deepscan',
    '/eternal-army': '/eternal-army',
    '/help': '/help',
    '/provider': '/provider'
  },
  categories: [
    { name: 'info', pairs: [['/help', 'show help']] },
    { name: 'config', pairs: [['/provider', 'pick a model']] },
    {
      name: 'project skills',
      pairs: [
        ['/deepscan', 'deep codebase scan'],
        ['/eternal-army', 'swarm of subagents']
      ]
    }
  ],
  pairs: [
    ['/help', 'show help'],
    ['/provider', 'pick a model'],
    ['/deepscan', 'deep codebase scan'],
    ['/eternal-army', 'swarm of subagents']
  ],
  skillCount: 2,
  sub: {}
}

const ranked = (input: string) => rankCompletionItems(slashCompletionsFromCatalog(input, catalog), input.slice(1))

describe('slash catalog completions', () => {
  it('tags each completion with the category that owns it', () => {
    const byName = new Map(slashCompletionsFromCatalog('/', catalog).map(item => [item.display, item.group]))

    expect(byName.get('help')).toBe('info')
    expect(byName.get('provider')).toBe('config')
    expect(byName.get('deepscan')).toBe('skills')
  })

  // Previously this asserted the opposite — skills came first, alphabetically,
  // so a bare "/" opened onto a wall of project skills and none of the
  // commands anyone actually opens the menu for. Skills stay reachable by
  // prefix (see below) and through the skills hub.
  it('orders the bare slash menu by command group, not alphabetically', () => {
    expect(ranked('/').map(item => item.display)).toEqual(['provider', 'help', 'deepscan', 'eternal-army'])
  })

  it('filters skills by typed slash prefix', () => {
    expect(slashCompletionsFromCatalog('/dee', catalog)).toEqual([
      { display: 'deepscan', group: 'skills', meta: 'deep codebase scan', text: '/deepscan' }
    ])
  })

  it('puts a skill first when its name is what was typed', () => {
    expect(ranked('/deep')[0]?.display).toBe('deepscan')
    expect(ranked('/eter')[0]?.display).toBe('eternal-army')
  })

  it('prefers an exact name over a longer one that merely shares the prefix', () => {
    const items = [
      { display: 'helper-skill', group: 'skills', text: '/helper-skill' },
      { display: 'help', group: 'info', text: '/help' }
    ]

    expect(rankCompletionItems(items, 'help').map(item => item.display)).toEqual(['help', 'helper-skill'])
  })

  it('sorts a group alphabetically when nothing has been typed yet', () => {
    const items = [
      { display: 'undo', group: 'session', text: '/undo' },
      { display: 'btw', group: 'session', text: '/btw' },
      { display: 'clear', group: 'session', text: '/clear' }
    ]

    // Length is evidence of a better prefix match; with a bare '/' it is
    // evidence of nothing, so the group stays scannable instead of being
    // scrambled short-to-long.
    expect(rankCompletionItems(items, '').map(item => item.display)).toEqual(['btw', 'clear', 'undo'])
  })

  it('dedupes daemon completions, keeping the local entry', () => {
    const local = slashCompletionsFromCatalog('/', catalog)
    const remote = [
      { display: 'help', meta: 'Show help', text: 'help' },
      { display: 'tools', meta: 'List tools', text: 'tools' }
    ]
    const merged = mergeCompletionItems(local, remote)

    // Merge stays a position-preserving dedupe; ranking runs after it, so the
    // local entry (with its category) is the one that survives.
    expect(merged.map(item => item.display)).toEqual(['help', 'provider', 'deepscan', 'eternal-army', 'tools'])
    expect(merged.find(item => item.display === 'help')?.meta).toBe('show help')
  })

  // The mockup's lowest tier: "fuzzy: prefix → substring → skill body". The
  // catalog pairs carry each skill's description into the row's meta, and a
  // query hitting only that prose must rank below every name match.
  it('ranks a description-only hit below every name hit', () => {
    const local = slashCompletionsFromCatalog('/dee', catalog)
    expect(local.map(item => item.display)).toEqual(['deepscan'])

    const remote = [{ display: 'army', meta: 'deploys scouts deep into a repo', text: '/army' }]
    const ranked = rankCompletionItems(mergeCompletionItems(local, remote), 'dee')

    expect(ranked.map(item => item.display)).toEqual(['deepscan', 'army'])
  })

  it('surfaces a skill whose description matches but whose name does not', () => {
    // Local candidates are no longer name-prefix only: a query that hits a
    // skill's description reaches the ranker even when no name matches.
    const local = slashCompletionsFromCatalog('/swarm', catalog)
    expect(local.map(item => item.display)).toEqual(['eternal-army'])

    const remote = [
      { display: 'eternal-army', meta: 'swarm of subagents', text: '/eternal-army' },
      { display: 'model', meta: 'pick a model', text: '/model' }
    ]

    expect(rankCompletionItems(remote, 'swarm').map(item => item.display)).toEqual(['eternal-army', 'model'])
  })
})
