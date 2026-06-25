// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.
import type { SlashCatalog } from '../types.js'

const normalizeName = (name: string) => name.trim().replace(/^\/+/, '')

const uniqueSorted = (values: string[]) => [...new Set(values.filter(Boolean))].sort((a, b) => a.localeCompare(b))

export function catalogFromSessionSkills(
  skills?: Record<string, string[]>,
  descriptions?: Record<string, string>
): null | SlashCatalog {
  const names = uniqueSorted(
    Object.values(skills ?? {})
      .flat()
      .map(normalizeName)
  )

  if (!names.length) {
    return null
  }

  const pairs = names.map(name => [`/${name}`, descriptions?.[name] ?? 'skill'] as [string, string])
  const sub: Record<string, string[]> = {}

  for (const name of names) {
    const [root, ...rest] = name.split(':')
    const child = rest.join(':')

    if (!root || !child) {
      continue
    }

    sub[root] = uniqueSorted([...(sub[root] ?? []), child])
  }

  return {
    canon: Object.fromEntries(pairs.map(([name]) => [name, name])),
    categories: [{ name: 'project skills', pairs }],
    pairs,
    skillCount: names.length,
    sub
  }
}

export function skillInfoFromCatalog(catalog: Pick<SlashCatalog, 'categories'>): null | {
  skillDescriptions: Record<string, string>
  skills: Record<string, string[]>
} {
  const pairs = catalog.categories.find(category => category.name === 'project skills')?.pairs ?? []
  const names = uniqueSorted(pairs.map(([name]) => normalizeName(name)))

  if (!names.length) {
    return null
  }

  const skillDescriptions = Object.fromEntries(
    pairs
      .map(([name, description]) => [normalizeName(name), description] as const)
      .filter(([name]) => names.includes(name))
  )

  return {
    skillDescriptions,
    skills: { skills: names }
  }
}

export function mergeSkillCatalog(base: null | SlashCatalog, skills: SlashCatalog): SlashCatalog {
  const skillLabels = new Set(skills.pairs.map(([name]) => name))
  const baseCategories = (base?.categories ?? []).filter(category => category.name !== 'project skills')
  const basePairs = (base?.pairs ?? []).filter(([name]) => !skillLabels.has(name))

  return {
    canon: { ...(base?.canon ?? {}), ...skills.canon },
    categories: [...baseCategories, ...skills.categories],
    pairs: [...basePairs, ...skills.pairs],
    skillCount: skills.skillCount,
    sub: { ...(base?.sub ?? {}), ...skills.sub }
  }
}
