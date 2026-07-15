// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { OfficePackageError } from './errors.js'
import {
  joinOfficePartName,
  officePartBasename,
  relationshipTargetPartName,
  type OfficePackagePort,
} from './package.js'
import { parseOoxmlRelationships, parseXmlAttributes, removeOoxmlRelationships } from './xml.js'

const CONTENT_TYPES_PART = '[Content_Types].xml'
const PRESENTATION_PART = 'ppt/presentation.xml'
const PRESENTATION_RELATIONSHIPS_PART = 'ppt/_rels/presentation.xml.rels'
const RESOURCE_DIRECTORIES = ['media', 'embeddings', 'charts', 'diagrams', 'tags', 'drawings', 'ink'] as const
const RELATIONSHIP_RESOURCE_DIRECTORIES = ['charts', 'diagrams', 'drawings'] as const

/** Return slide filenames referenced from `presentation.xml` through its `sldIdLst` relationship IDs. */
export async function getSlidesInSlideIdList(packageDirectory: OfficePackagePort): Promise<ReadonlySet<string>> {
  if (!await packageDirectory.hasPart(PRESENTATION_PART) || !await packageDirectory.hasPart(PRESENTATION_RELATIONSHIPS_PART)) {
    return new Set()
  }
  const [presentation, relationships] = await Promise.all([
    packageDirectory.readText(PRESENTATION_PART),
    packageDirectory.readText(PRESENTATION_RELATIONSHIPS_PART),
  ])
  const slidesByRelationshipId = new Map<string, string>()
  for (const relationship of parseOoxmlRelationships(relationships)) {
    if (relationship.type.toLowerCase().includes('slide') && relationship.target.startsWith('slides/')) {
      slidesByRelationshipId.set(relationship.id, relationship.target.slice('slides/'.length))
    }
  }
  const referencedIds = [...presentation.matchAll(/<(?:[A-Za-z_][\w.-]*:)?sldId\b[^>]*\br:id\s*=\s*["']([^"']+)["']/gi)]
    .map(match => match[1])
    .filter((value): value is string => value !== undefined)
  return new Set(referencedIds.flatMap(id => {
    const slide = slidesByRelationshipId.get(id)
    return slide === undefined ? [] : [slide]
  }))
}

/** Delete slides not referenced by the presentation and their slide-level relationship parts. */
export async function removeOrphanedSlides(packageDirectory: OfficePackagePort): Promise<string[]> {
  const referencedSlides = await getSlidesInSlideIdList(packageDirectory)
  const slideParts = (await packageDirectory.listParts()).filter(part => /^ppt\/slides\/slide\d+\.xml$/.test(part))
  const removed: string[] = []
  for (const slidePart of slideParts) {
    const fileName = officePartBasename(slidePart)
    if (referencedSlides.has(fileName)) continue
    await packageDirectory.deletePart(slidePart)
    removed.push(slidePart)
    const relationshipPart = `ppt/slides/_rels/${fileName}.rels`
    if (await packageDirectory.hasPart(relationshipPart)) {
      await packageDirectory.deletePart(relationshipPart)
      removed.push(relationshipPart)
    }
  }
  if (removed.length && await packageDirectory.hasPart(PRESENTATION_RELATIONSHIPS_PART)) {
    const xml = await packageDirectory.readText(PRESENTATION_RELATIONSHIPS_PART)
    const cleaned = removeOoxmlRelationships(xml, relationship => (
      relationship.target.startsWith('slides/') && !referencedSlides.has(relationship.target.slice('slides/'.length))
    ))
    if (cleaned !== xml) await packageDirectory.writeText(PRESENTATION_RELATIONSHIPS_PART, cleaned)
  }
  return removed
}

/** Delete every file below the temporary `[trash]` package directory. */
export async function removeTrashDirectory(packageDirectory: OfficePackagePort): Promise<string[]> {
  const removed = (await packageDirectory.listParts()).filter(part => part.startsWith('[trash]/'))
  await Promise.all(removed.map(part => packageDirectory.deletePart(part)))
  return removed
}

/** Return internal target part names referenced directly from slide relationship files. */
export async function getSlideReferencedFiles(packageDirectory: OfficePackagePort): Promise<ReadonlySet<string>> {
  const references = new Set<string>()
  const relationshipParts = (await packageDirectory.listParts())
    .filter(part => /^ppt\/slides\/_rels\/[^/]+\.rels$/.test(part))
  for (const relationshipPart of relationshipParts) {
    const xml = await packageDirectory.readText(relationshipPart)
    for (const relationship of parseOoxmlRelationships(xml)) {
      if (relationship.targetMode?.toLowerCase() === 'external') continue
      const target = safelyResolveRelationshipTarget(relationshipPart, relationship.target)
      if (target !== undefined) references.add(target)
    }
  }
  return references
}

/** Delete child `.rels` parts for resources that have no slide-level reference. */
export async function removeOrphanedRelationshipFiles(packageDirectory: OfficePackagePort): Promise<string[]> {
  const slideReferenced = await getSlideReferencedFiles(packageDirectory)
  const removed: string[] = []
  for (const part of await packageDirectory.listParts()) {
    const match = /^ppt\/(charts|diagrams|drawings)\/_rels\/([^/]+)\.rels$/.exec(part)
    if (match === null || !RELATIONSHIP_RESOURCE_DIRECTORIES.includes(match[1] as typeof RELATIONSHIP_RESOURCE_DIRECTORIES[number])) {
      continue
    }
    const resourcePart = `ppt/${match[1]}/${match[2] ?? ''}`
    if (await packageDirectory.hasPart(resourcePart) && slideReferenced.has(resourcePart)) continue
    await packageDirectory.deletePart(part)
    removed.push(part)
  }
  return removed
}

/** Return every internally-addressable part target referenced by any package relationship file. */
export async function getReferencedFiles(packageDirectory: OfficePackagePort): Promise<ReadonlySet<string>> {
  const references = new Set<string>()
  for (const relationshipPart of (await packageDirectory.listParts()).filter(part => part.endsWith('.rels'))) {
    const xml = await packageDirectory.readText(relationshipPart)
    for (const relationship of parseOoxmlRelationships(xml)) {
      if (relationship.targetMode?.toLowerCase() === 'external') continue
      const target = safelyResolveRelationshipTarget(relationshipPart, relationship.target)
      if (target !== undefined) references.add(target)
    }
  }
  return references
}

/** Remove unreferenced PowerPoint resources, themes, and notes slides. */
export async function removeOrphanedFiles(
  packageDirectory: OfficePackagePort,
  referenced: ReadonlySet<string>,
): Promise<string[]> {
  const parts = await packageDirectory.listParts()
  const removed: string[] = []
  for (const directory of RESOURCE_DIRECTORIES) {
    const prefix = `ppt/${directory}/`
    for (const part of parts.filter(candidate => isDirectChild(candidate, prefix))) {
      if (referenced.has(part)) continue
      await packageDirectory.deletePart(part)
      removed.push(part)
    }
  }
  for (const themePart of parts.filter(part => /^ppt\/theme\/theme[^/]*\.xml$/.test(part))) {
    if (referenced.has(themePart)) continue
    await packageDirectory.deletePart(themePart)
    removed.push(themePart)
    const relationshipPart = `ppt/theme/_rels/${officePartBasename(themePart)}.rels`
    if (await packageDirectory.hasPart(relationshipPart)) {
      await packageDirectory.deletePart(relationshipPart)
      removed.push(relationshipPart)
    }
  }
  for (const notePart of parts.filter(part => /^ppt\/notesSlides\/[^/]+\.xml$/.test(part))) {
    if (referenced.has(notePart)) continue
    await packageDirectory.deletePart(notePart)
    removed.push(notePart)
  }
  for (const relationshipPart of parts.filter(part => /^ppt\/notesSlides\/_rels\/[^/]+\.rels$/.test(part))) {
    const notePart = `ppt/notesSlides/${officePartBasename(relationshipPart).slice(0, -'.rels'.length)}`
    if (await packageDirectory.hasPart(notePart)) continue
    await packageDirectory.deletePart(relationshipPart)
    removed.push(relationshipPart)
  }
  return removed
}

/** Remove content-type overrides whose parts were removed by a cleanup pass. */
export async function updateContentTypes(packageDirectory: OfficePackagePort, removedFiles: readonly string[]): Promise<void> {
  if (!removedFiles.length || !await packageDirectory.hasPart(CONTENT_TYPES_PART)) return
  const removed = new Set(removedFiles)
  const xml = await packageDirectory.readText(CONTENT_TYPES_PART)
  const expression = /\s*<Override\b([\s\S]*?)(?:\/\s*>|>\s*<\/Override\s*>)\s*/gi
  const cleaned = xml.replace(expression, (whole, rawAttributes: string) => {
    const partName = parseXmlAttributes(rawAttributes).PartName
    if (partName === undefined) return whole
    const normalized = partName.replace(/^\/+/, '')
    return removed.has(normalized) ? '\n' : whole
  })
  if (cleaned !== xml) await packageDirectory.writeText(CONTENT_TYPES_PART, cleaned)
}

/** Run the native cleanup passes until no newly orphaned part remains. */
export async function cleanUnusedFiles(packageDirectory: OfficePackagePort): Promise<string[]> {
  const removed: string[] = []
  removed.push(...await removeOrphanedSlides(packageDirectory))
  removed.push(...await removeTrashDirectory(packageDirectory))
  const maxPasses = (await packageDirectory.listParts()).length + 1
  for (let pass = 0; pass < maxPasses; pass += 1) {
    const removedRelationships = await removeOrphanedRelationshipFiles(packageDirectory)
    const referenced = await getReferencedFiles(packageDirectory)
    const removedResources = await removeOrphanedFiles(packageDirectory, referenced)
    if (!removedRelationships.length && !removedResources.length) break
    removed.push(...removedRelationships, ...removedResources)
    if (pass === maxPasses - 1) throw new OfficePackageError('Office cleanup did not reach a fixed point')
  }
  if (removed.length) await updateContentTypes(packageDirectory, removed)
  return removed
}

function isDirectChild(part: string, prefix: string): boolean {
  if (!part.startsWith(prefix)) return false
  return !part.slice(prefix.length).includes('/')
}

function safelyResolveRelationshipTarget(relationshipPart: string, target: string): string | undefined {
  try {
    return relationshipTargetPartName(relationshipPart, target)
  } catch (error) {
    if (error instanceof OfficePackageError) return undefined
    throw error
  }
}

/** Resolve an internal relationship target from a `.rels` part. */
export function resolveRelationshipTarget(relationshipPart: string, target: string): string {
  return joinOfficePartName(relationshipPart.slice(0, relationshipPart.lastIndexOf('/_rels/')), target)
}
