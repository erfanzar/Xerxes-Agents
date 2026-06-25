// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import { describe, expect, it } from 'vitest'

import { catalogFromSessionSkills, mergeSkillCatalog, skillInfoFromCatalog } from '../app/skillCatalog.js'
import type { SlashCatalog } from '../types.js'

describe('skillCatalog', () => {
  it('derives slash commands from session skill metadata', () => {
    const catalog = catalogFromSessionSkills(
      { local: ['deepscan', '/eternal-army', 'autoresearch:fix'] },
      {
        'autoresearch:fix': 'iterate on a fix',
        deepscan: 'deep scan',
        'eternal-army': 'spawn agents'
      }
    )

    expect(catalog?.skillCount).toBe(3)
    expect(catalog?.pairs).toEqual([
      ['/autoresearch:fix', 'iterate on a fix'],
      ['/deepscan', 'deep scan'],
      ['/eternal-army', 'spawn agents']
    ])
    expect(catalog?.sub).toEqual({ autoresearch: ['fix'] })
  })

  it('merges derived skills into a fallback core command catalog', () => {
    const base: SlashCatalog = {
      canon: { '/help': '/help' },
      categories: [{ name: 'core', pairs: [['/help', 'show help']] }],
      pairs: [['/help', 'show help']],
      skillCount: 0,
      sub: {}
    }
    const skills = catalogFromSessionSkills({ skills: ['deepscan'] }, { deepscan: 'deep scan' })

    expect(skills).not.toBeNull()

    const merged = mergeSkillCatalog(base, skills!)

    expect(merged.skillCount).toBe(1)
    expect(merged.pairs).toContainEqual(['/help', 'show help'])
    expect(merged.pairs).toContainEqual(['/deepscan', 'deep scan'])
    expect(merged.categories.map(category => category.name)).toEqual(['core', 'project skills'])
    expect(merged.canon['/deepscan']).toBe('/deepscan')
  })

  it('derives session skill metadata from daemon command catalog skills', () => {
    const skillInfo = skillInfoFromCatalog({
      categories: [
        { name: 'core', pairs: [['/help', 'show help']] },
        {
          name: 'project skills',
          pairs: [
            ['/deepscan', 'deep scan'],
            ['/eternal-army', 'spawn agents']
          ]
        }
      ]
    })

    expect(skillInfo).toEqual({
      skillDescriptions: {
        deepscan: 'deep scan',
        'eternal-army': 'spawn agents'
      },
      skills: { skills: ['deepscan', 'eternal-army'] }
    })
  })
})
