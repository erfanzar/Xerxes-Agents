// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { OfficePackageError, OfficePackagePartNotFoundError } from './errors.js'
import { officePartBasename, type OfficePackagePort } from './package.js'
import {
  appendXmlChild,
  appendXmlChildByLocalName,
  escapeXmlAttribute,
  expandEmptyXmlElement,
  removeOoxmlRelationships,
} from './xml.js'

const CONTENT_TYPES_PART = '[Content_Types].xml'
const PRESENTATION_PART = 'ppt/presentation.xml'
const PRESENTATION_RELATIONSHIPS_PART = 'ppt/_rels/presentation.xml.rels'
const SLIDE_CONTENT_TYPE = 'application/vnd.openxmlformats-officedocument.presentationml.slide+xml'
const SLIDE_LAYOUT_RELATIONSHIP_TYPE = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout'
const SLIDE_RELATIONSHIP_TYPE = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide'

/** The source category recognized by the native slide-adder. */
export type SlideSourceKind = 'layout' | 'slide'

/** A safely classified source passed to `addSlide`. */
export interface SlideSource {
  readonly fileName: string
  readonly kind: SlideSourceKind
}

/** Result of adding one slide and registering it in the presentation manifest. */
export interface SlideAddition {
  readonly presentationEntry: string
  readonly relationshipId: string
  readonly slideFileName: string
  readonly slideId: number
  readonly source: SlideSource
}

/** Classify `slideLayoutN.xml` as a layout; every other safe XML filename is an existing slide. */
export function parseSlideSource(source: string): SlideSource {
  const fileName = requiredXmlFileName(source, 'source')
  return fileName.startsWith('slideLayout') ? { fileName, kind: 'layout' } : { fileName, kind: 'slide' }
}

/** Return the next free `slideN.xml` number in an unpacked presentation. */
export async function getNextSlideNumber(packageDirectory: OfficePackagePort): Promise<number> {
  const numbers = (await packageDirectory.listParts())
    .map(part => /^ppt\/slides\/slide(\d+)\.xml$/.exec(part)?.[1])
    .filter((value): value is string => value !== undefined)
    .map(value => Number(value))
    .filter(value => Number.isSafeInteger(value) && value >= 0)
  return numbers.length ? Math.max(...numbers) + 1 : 1
}

/** Create a blank slide linked to a named layout, then register it in the presentation. */
export async function createSlideFromLayout(
  packageDirectory: OfficePackagePort,
  layoutFile: string,
): Promise<SlideAddition> {
  const fileName = requiredXmlFileName(layoutFile, 'layoutFile')
  const layoutPart = `ppt/slideLayouts/${fileName}`
  if (!await packageDirectory.hasPart(layoutPart)) throw new OfficePackagePartNotFoundError(layoutPart)
  return addSlideParts(packageDirectory, { fileName, kind: 'layout' })
}

/** Duplicate a slide, omit its notes-slide relation, then register the copy in the presentation. */
export async function duplicateSlide(packageDirectory: OfficePackagePort, source: string): Promise<SlideAddition> {
  const fileName = requiredXmlFileName(source, 'source')
  const sourcePart = `ppt/slides/${fileName}`
  if (!await packageDirectory.hasPart(sourcePart)) throw new OfficePackagePartNotFoundError(sourcePart)
  return addSlideParts(packageDirectory, { fileName, kind: 'slide' })
}

/** Add a layout-backed blank slide or a duplicate based on a classified source. */
export async function addSlide(packageDirectory: OfficePackagePort, source: string): Promise<SlideAddition> {
  const parsed = parseSlideSource(source)
  return parsed.kind === 'layout'
    ? createSlideFromLayout(packageDirectory, parsed.fileName)
    : duplicateSlide(packageDirectory, parsed.fileName)
}

async function addSlideParts(packageDirectory: OfficePackagePort, source: SlideSource): Promise<SlideAddition> {
  const nextNumber = await getNextSlideNumber(packageDirectory)
  const slideFileName = `slide${nextNumber}.xml`
  const slidePart = `ppt/slides/${slideFileName}`
  const slideRelationshipsPart = `ppt/slides/_rels/${slideFileName}.rels`
  const [contentTypes, presentation, presentationRelationships] = await Promise.all([
    requiredTextPart(packageDirectory, CONTENT_TYPES_PART),
    requiredTextPart(packageDirectory, PRESENTATION_PART),
    requiredTextPart(packageDirectory, PRESENTATION_RELATIONSHIPS_PART),
  ])
  const relationshipId = nextRelationshipId(presentationRelationships)
  const slideId = nextPresentationSlideId(presentation)
  const presentationEntry = `<p:sldId id="${slideId}" r:id="${relationshipId}"/>`
  const updatedContentTypes = addSlideContentType(contentTypes, slideFileName)
  const updatedPresentationRelationships = appendPresentationRelationship(
    presentationRelationships,
    relationshipId,
    slideFileName,
  )
  const updatedPresentation = appendPresentationSlide(presentation, presentationEntry)

  if (source.kind === 'layout') {
    await packageDirectory.writeText(slidePart, blankSlideXml())
    await packageDirectory.writeText(slideRelationshipsPart, layoutRelationshipsXml(source.fileName))
  } else {
    const sourcePart = `ppt/slides/${source.fileName}`
    await packageDirectory.writeBytes(slidePart, await packageDirectory.readBytes(sourcePart))
    const sourceRelationshipsPart = `ppt/slides/_rels/${source.fileName}.rels`
    if (await packageDirectory.hasPart(sourceRelationshipsPart)) {
      const copiedRelationships = removeOoxmlRelationships(
        await packageDirectory.readText(sourceRelationshipsPart),
        relationship => relationship.type.includes('notesSlide'),
      )
      await packageDirectory.writeText(slideRelationshipsPart, copiedRelationships)
    }
  }
  await Promise.all([
    packageDirectory.writeText(CONTENT_TYPES_PART, updatedContentTypes),
    packageDirectory.writeText(PRESENTATION_PART, updatedPresentation),
    packageDirectory.writeText(PRESENTATION_RELATIONSHIPS_PART, updatedPresentationRelationships),
  ])
  return {
    presentationEntry,
    relationshipId,
    slideFileName,
    slideId,
    source,
  }
}

function addSlideContentType(contentTypes: string, slideFileName: string): string {
  const partName = `/ppt/slides/${slideFileName}`
  if (new RegExp(`\\bPartName\\s*=\\s*(["'])${escapeRegExp(partName)}\\1`, 'i').test(contentTypes)) {
    return contentTypes
  }
  return appendXmlChild(
    contentTypes,
    'Types',
    `  <Override PartName="${partName}" ContentType="${SLIDE_CONTENT_TYPE}"/>\n`,
  )
}

function appendPresentationRelationship(xml: string, relationshipId: string, slideFileName: string): string {
  const target = `slides/${slideFileName}`
  if (new RegExp(`\\bTarget\\s*=\\s*(["'])${escapeRegExp(target)}\\1`, 'i').test(xml)) {
    throw new OfficePackageError(`Presentation relationship already targets ${target}`)
  }
  return appendXmlChild(
    xml,
    'Relationships',
    `  <Relationship Id="${relationshipId}" Type="${SLIDE_RELATIONSHIP_TYPE}" Target="${target}"/>\n`,
  )
}

function appendPresentationSlide(presentation: string, entry: string): string {
  const emptyList = expandEmptyXmlElement(presentation, 'p:sldIdLst', entry)
  if (emptyList !== undefined) return emptyList
  try {
    return appendXmlChild(presentation, 'p:sldIdLst', entry)
  } catch (error) {
    if (!(error instanceof OfficePackageError)) throw error
    return appendXmlChildByLocalName(presentation, 'presentation', `<p:sldIdLst>${entry}</p:sldIdLst>`)
  }
}

function nextRelationshipId(presentationRelationships: string): string {
  const ids = [...presentationRelationships.matchAll(/\bId\s*=\s*["']rId(\d+)["']/gi)]
    .map(match => Number(match[1]))
    .filter(value => Number.isSafeInteger(value) && value >= 0)
  return `rId${ids.length ? Math.max(...ids) + 1 : 1}`
}

function nextPresentationSlideId(presentation: string): number {
  const ids = [...presentation.matchAll(/<(?:[A-Za-z_][\w.-]*:)?sldId\b[^>]*\bid\s*=\s*["'](\d+)["']/gi)]
    .map(match => Number(match[1]))
    .filter(value => Number.isSafeInteger(value) && value >= 0)
  return ids.length ? Math.max(...ids) + 1 : 256
}

function blankSlideXml(): string {
  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>`
}

function layoutRelationshipsXml(layoutFileName: string): string {
  return `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="${SLIDE_LAYOUT_RELATIONSHIP_TYPE}" Target="../slideLayouts/${escapeXmlAttribute(layoutFileName)}"/></Relationships>`
}

async function requiredTextPart(packageDirectory: OfficePackagePort, partName: string): Promise<string> {
  if (!await packageDirectory.hasPart(partName)) throw new OfficePackagePartNotFoundError(partName)
  return packageDirectory.readText(partName)
}

function requiredXmlFileName(value: string, name: string): string {
  if (typeof value !== 'string' || !value.trim()) throw new TypeError(`${name} must be a non-empty XML filename`)
  const trimmed = value.trim()
  if (trimmed !== officePartBasename(trimmed) || !trimmed.endsWith('.xml')) {
    throw new OfficePackageError(`${name} must be a simple .xml filename`)
  }
  return trimmed
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}
