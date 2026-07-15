// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { expect, test } from 'bun:test'
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { tmpdir } from 'node:os'

import {
  BunOfficePackageDirectory,
  cleanUnusedFiles,
  createOfficeZip,
  createSlideFromLayout,
  duplicateSlide,
  getTrackedChangeAuthorsXml,
  inferTrackedChangeAuthor,
  mergeDocumentRuns,
  mergeDocumentRunsXml,
  packOfficePackage,
  packOfficeDirectory,
  readOfficeZip,
  runPowerPointCli,
  simplifyDocumentRedlines,
  simplifyDocumentRedlinesXml,
  type OfficePackagePort,
} from '../src/skills/powerpoint/index.js'
import { normalizeOfficePartName } from '../src/skills/powerpoint/package.js'
import { runBundledSkillCli } from '../src/skills/cli.js'

test('native PowerPoint slide insertion creates and registers layout-based and duplicated slides', async () => {
  const layoutDeck = presentationFixture()
  const created = await createSlideFromLayout(layoutDeck, 'slideLayout1.xml')
  expect(created).toEqual({
    presentationEntry: '<p:sldId id="257" r:id="rId2"/>',
    relationshipId: 'rId2',
    slideFileName: 'slide2.xml',
    slideId: 257,
    source: { fileName: 'slideLayout1.xml', kind: 'layout' },
  })
  expect(layoutDeck.text('ppt/slides/slide2.xml')).toContain('<p:sld')
  expect(layoutDeck.text('ppt/slides/_rels/slide2.xml.rels')).toContain('../slideLayouts/slideLayout1.xml')
  expect(layoutDeck.text('[Content_Types].xml')).toContain('/ppt/slides/slide2.xml')
  expect(layoutDeck.text('ppt/presentation.xml')).toContain('<p:sldId id="257" r:id="rId2"/>')
  expect(layoutDeck.text('ppt/_rels/presentation.xml.rels')).toContain('Target="slides/slide2.xml"')

  const duplicateDeck = presentationFixture()
  const duplicate = await duplicateSlide(duplicateDeck, 'slide1.xml')
  expect(duplicate.source).toEqual({ fileName: 'slide1.xml', kind: 'slide' })
  expect(duplicateDeck.text('ppt/slides/slide2.xml')).toBe(duplicateDeck.text('ppt/slides/slide1.xml'))
  expect(duplicateDeck.text('ppt/slides/_rels/slide2.xml.rels')).not.toContain('notesSlide')
  expect(duplicateDeck.text('ppt/slides/_rels/slide2.xml.rels')).toContain('slideLayout')
})

test('native PowerPoint cleanup follows OOXML relationships and removes orphan parts and content types', async () => {
  const deck = presentationFixture({
    '[Content_Types].xml': `<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
      <Override PartName="/ppt/slides/slide1.xml" ContentType="slide"/>
      <Override PartName="/ppt/slides/slide2.xml" ContentType="slide"/>
      <Override PartName="/ppt/media/image2.png" ContentType="image"/>
      <Override PartName="/ppt/charts/chart2.xml" ContentType="chart"/>
    </Types>`,
    '[trash]/nested/temp.txt': 'remove me',
    'ppt/charts/_rels/chart2.xml.rels': relationshipsXml('<Relationship Id="rId1" Type="chart" Target="../drawings/drawing2.xml"/>'),
    'ppt/charts/chart1.xml': '<c:chartSpace xmlns:c="chart"/>',
    'ppt/charts/chart2.xml': '<c:chartSpace xmlns:c="chart"/>',
    'ppt/media/image1.png': new Uint8Array([1]),
    'ppt/media/image2.png': new Uint8Array([2]),
    'ppt/slides/_rels/slide1.xml.rels': relationshipsXml(
      '<Relationship Id="rId1" Type="image" Target="../media/image1.png"/>',
      '<Relationship Id="rId2" Type="chart" Target="../charts/chart1.xml"/>',
    ),
    'ppt/slides/_rels/slide2.xml.rels': relationshipsXml('<Relationship Id="rId1" Type="image" Target="../media/image2.png"/>'),
    'ppt/slides/slide2.xml': '<p:sld xmlns:p="presentation"/>',
  })
  const removed = await cleanUnusedFiles(deck)
  expect(removed).toEqual(expect.arrayContaining([
    '[trash]/nested/temp.txt',
    'ppt/charts/_rels/chart2.xml.rels',
    'ppt/charts/chart2.xml',
    'ppt/media/image2.png',
    'ppt/slides/_rels/slide2.xml.rels',
    'ppt/slides/slide2.xml',
  ]))
  expect(await deck.hasPart('ppt/slides/slide1.xml')).toBeTrue()
  expect(await deck.hasPart('ppt/media/image1.png')).toBeTrue()
  expect(await deck.hasPart('ppt/charts/chart1.xml')).toBeTrue()
  expect(await deck.hasPart('ppt/slides/slide2.xml')).toBeFalse()
  expect(await deck.hasPart('ppt/media/image2.png')).toBeFalse()
  expect(deck.text('[Content_Types].xml')).not.toContain('slide2.xml')
  expect(deck.text('[Content_Types].xml')).not.toContain('image2.png')
  expect(deck.text('ppt/_rels/presentation.xml.rels')).not.toContain('slide2.xml')
})

test('native Office redline helpers merge runs, reduce revisions, and infer authors from a native ZIP baseline', async () => {
  const runXml = `<w:document xmlns:w="word"><w:body><w:p><w:proofErr w:type="spellStart"/><w:r w:rsidR="001"><w:t>Hello</w:t></w:r> <w:r><w:t> </w:t></w:r><w:r><w:t>world</w:t></w:r></w:p></w:body></w:document>`
  const merged = mergeDocumentRunsXml(runXml)
  expect(merged.count).toBe(2)
  expect(merged.xml).toContain('<w:t>Hello world</w:t>')
  expect(merged.xml).not.toContain('proofErr')
  expect(merged.xml).not.toContain('rsid')

  const revisionsXml = `<w:document xmlns:w="word"><w:body><w:p><w:ins w:author="Ada"><w:r/></w:ins> <w:ins w:author="Ada"><w:r/></w:ins><w:del w:author="Grace"><w:r/></w:del></w:p></w:body></w:document>`
  const simplified = simplifyDocumentRedlinesXml(revisionsXml)
  expect(simplified.count).toBe(1)
  expect((simplified.xml.match(/<w:ins\b/g) ?? []).length).toBe(1)
  expect(getTrackedChangeAuthorsXml(revisionsXml)).toEqual(new Map([['Ada', 2], ['Grace', 1]]))

  const packageDirectory = new MemoryOfficePackage({ 'word/document.xml': revisionsXml })
  const baseline = createOfficeZip([{
    data: new TextEncoder().encode(`<w:document xmlns:w="word"><w:body><w:p><w:ins w:author="Ada"><w:r/></w:ins><w:ins w:author="Ada"><w:r/></w:ins></w:p></w:body></w:document>`),
    name: 'word/document.xml',
  }])
  expect(await inferTrackedChangeAuthor(packageDirectory, baseline)).toBe('Grace')
  const simplifiedOnDisk = await simplifyDocumentRedlines(packageDirectory)
  expect(simplifiedOnDisk.count).toBe(1)

  const runPackage = new MemoryOfficePackage({ 'word/document.xml': runXml })
  expect(await mergeDocumentRuns(runPackage)).toEqual({ count: 2, message: 'Merged 2 runs' })
  expect(runPackage.text('word/document.xml')).toContain('Hello world')
})

test('native packer condenses OOXML text safely and round-trips it through its native ZIP implementation', async () => {
  const packageDirectory = new MemoryOfficePackage({
    '[Content_Types].xml': '<Types/>',
    '_rels/.rels': relationshipsXml('<Relationship Id="rId1" Type="officeDocument" Target="ppt/presentation.xml"/>'),
    'ppt/presentation.xml': `<?xml version="1.0"?>
      <p:presentation xmlns:p="presentation" xmlns:a="drawing">
        <!-- drop this comment -->
        <a:t> keep this whitespace </a:t>
      </p:presentation>`,
    'ppt/media/image.bin': new Uint8Array([0, 255, 1]),
  })
  const packed = await packOfficePackage(packageDirectory, { kind: 'pptx' })
  expect(packed.validation).toEqual({ message: 'Native OOXML structural validation passed', valid: true })
  const entries = readOfficeZip(packed.bytes)
  const presentation = new TextDecoder().decode(entries.get('ppt/presentation.xml') ?? fail('missing presentation'))
  expect(presentation).not.toContain('drop this comment')
  expect(presentation).toContain('<a:t> keep this whitespace </a:t>')
  expect(entries.get('ppt/media/image.bin')).toEqual(new Uint8Array([0, 255, 1]))

  const genericRoundTrip = readOfficeZip(createOfficeZip([{ name: 'word/document.xml', data: new TextEncoder().encode('<w:document/>') }]))
  expect(new TextDecoder().decode(genericRoundTrip.get('word/document.xml') ?? fail('missing ZIP entry'))).toBe('<w:document/>')
})

test('Bun Office directory adapter writes a real native PPTX without a subprocess boundary', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-powerpoint-'))
  try {
    const packageDirectory = new BunOfficePackageDirectory(directory)
    await packageDirectory.writeText('[Content_Types].xml', '<Types/>')
    await packageDirectory.writeText('_rels/.rels', relationshipsXml('<Relationship Id="rId1" Type="officeDocument" Target="ppt/presentation.xml"/>'))
    await packageDirectory.writeText('ppt/presentation.xml', '<p:presentation xmlns:p="presentation"/>')
    const output = join(directory, 'deck.pptx')
    await packOfficeDirectory(directory, output)
    const archive = readOfficeZip(new Uint8Array(await readFile(output)))
    expect(archive.has('ppt/presentation.xml')).toBeTrue()
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

test('PowerPoint skill CLI reaches every migrated script workflow through Bun-native package ports', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'xerxes-powerpoint-cli-'))
  try {
    const output: string[] = []
    const deck = new BunOfficePackageDirectory(directory)
    await Promise.all([
      deck.writeText('[Content_Types].xml', '<Types></Types>'),
      deck.writeText('_rels/.rels', relationshipsXml('<Relationship Id="rId1" Type="officeDocument" Target="ppt/presentation.xml"/>')),
      deck.writeText('ppt/_rels/presentation.xml.rels', relationshipsXml('<Relationship Id="rId1" Type="slide" Target="slides/slide1.xml"/>')),
      deck.writeText('ppt/presentation.xml', '<p:presentation xmlns:p="presentation" xmlns:r="relationships"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>'),
      deck.writeText('ppt/slideLayouts/slideLayout1.xml', '<p:sldLayout xmlns:p="presentation"/>'),
      deck.writeText('ppt/slides/slide1.xml', '<p:sld xmlns:p="presentation"/>'),
    ])

    expect(await runBundledSkillCli(['powerpoint', '--help'], {
      writeLine: line => { output.push(line) },
    })).toBe(0)
    expect(output[0]).toContain('add-slide')

    output.length = 0
    expect(await runPowerPointCli(['add-slide', directory, 'slideLayout1.xml'], {
      writeLine: line => { output.push(line) },
    })).toBe(0)
    expect(output).toEqual([
      'Created slide2.xml from slideLayout1.xml',
      'Add to presentation.xml <p:sldIdLst>: <p:sldId id="257" r:id="rId2"/>',
    ])

    output.length = 0
    expect(await runPowerPointCli(['clean', directory], {
      writeLine: line => { output.push(line) },
    })).toBe(0)
    expect(output).toEqual(['No unreferenced files found'])

    const baseline = join(directory, 'baseline.pptx')
    await writeFile(baseline, createOfficeZip([
      { data: new TextEncoder().encode('<Types/>'), name: '[Content_Types].xml' },
      { data: new TextEncoder().encode(relationshipsXml('<Relationship Id="rId1" Type="officeDocument" Target="ppt/presentation.xml"/>')), name: '_rels/.rels' },
      { data: new TextEncoder().encode('<p:presentation xmlns:p="presentation"/>'), name: 'ppt/presentation.xml' },
    ]))
    const packedPath = join(directory, 'native.pptx')
    output.length = 0
    expect(await runPowerPointCli(['pack', directory, packedPath, '--original', baseline], {
      writeLine: line => { output.push(line) },
    })).toBe(0)
    expect(readOfficeZip(new Uint8Array(await readFile(packedPath))).has('ppt/presentation.xml')).toBeTrue()
    expect(output.at(-1)).toContain('package and baseline')

    const documentDirectory = join(directory, 'document')
    const document = new BunOfficePackageDirectory(documentDirectory)
    await document.writeText('word/document.xml', '<w:document xmlns:w="word"><w:body><w:p><w:r><w:t>Hello</w:t></w:r><w:r><w:t> world</w:t></w:r><w:ins w:author="Ada"><w:r/></w:ins><w:ins w:author="Ada"><w:r/></w:ins></w:p></w:body></w:document>')
    output.length = 0
    expect(await runPowerPointCli(['merge-runs', documentDirectory], {
      writeLine: line => { output.push(line) },
    })).toBe(0)
    expect(output).toEqual(['Merged 1 runs'])
    output.length = 0
    expect(await runPowerPointCli(['simplify-redlines', documentDirectory], {
      writeLine: line => { output.push(line) },
    })).toBe(0)
    expect(output).toEqual(['Simplified 1 tracked changes'])
  } finally {
    await rm(directory, { force: true, recursive: true })
  }
})

class MemoryOfficePackage implements OfficePackagePort {
  private readonly parts = new Map<string, Uint8Array>()

  constructor(entries: Readonly<Record<string, string | Uint8Array>>) {
    for (const [name, contents] of Object.entries(entries)) {
      this.parts.set(normalizeOfficePartName(name), bytesFor(contents))
    }
  }

  async deletePart(partName: string): Promise<void> {
    this.parts.delete(normalizeOfficePartName(partName))
  }

  async hasPart(partName: string): Promise<boolean> {
    return this.parts.has(normalizeOfficePartName(partName))
  }

  async listParts(): Promise<readonly string[]> {
    return [...this.parts.keys()].sort()
  }

  async readBytes(partName: string): Promise<Uint8Array> {
    const value = this.parts.get(normalizeOfficePartName(partName))
    if (value === undefined) throw new Error(`Missing test package part: ${partName}`)
    return new Uint8Array(value)
  }

  async readText(partName: string): Promise<string> {
    return new TextDecoder().decode(await this.readBytes(partName))
  }

  async writeBytes(partName: string, contents: Uint8Array): Promise<void> {
    this.parts.set(normalizeOfficePartName(partName), new Uint8Array(contents))
  }

  async writeText(partName: string, contents: string): Promise<void> {
    await this.writeBytes(partName, new TextEncoder().encode(contents))
  }

  text(partName: string): string {
    const value = this.parts.get(normalizeOfficePartName(partName))
    if (value === undefined) throw new Error(`Missing test package part: ${partName}`)
    return new TextDecoder().decode(value)
  }
}

function presentationFixture(overrides: Readonly<Record<string, string | Uint8Array>> = {}): MemoryOfficePackage {
  return new MemoryOfficePackage({
    '[Content_Types].xml': `<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
      <Override PartName="/ppt/slides/slide1.xml" ContentType="slide"/>
    </Types>`,
    '_rels/.rels': relationshipsXml('<Relationship Id="rId1" Type="officeDocument" Target="ppt/presentation.xml"/>'),
    'ppt/_rels/presentation.xml.rels': relationshipsXml('<Relationship Id="rId1" Type="slide" Target="slides/slide1.xml"/>'),
    'ppt/presentation.xml': `<p:presentation xmlns:p="presentation" xmlns:r="relationships"><p:sldIdLst><p:sldId id="256" r:id="rId1"/></p:sldIdLst></p:presentation>`,
    'ppt/slideLayouts/slideLayout1.xml': '<p:sldLayout xmlns:p="presentation"/>',
    'ppt/slides/_rels/slide1.xml.rels': relationshipsXml(
      '<Relationship Id="rId1" Type="slideLayout" Target="../slideLayouts/slideLayout1.xml"/>',
      '<Relationship Id="rId2" Type="notesSlide" Target="../notesSlides/notesSlide1.xml"/>',
    ),
    'ppt/slides/slide1.xml': '<p:sld xmlns:p="presentation"><p:cSld/></p:sld>',
    ...overrides,
  })
}

function relationshipsXml(...relationships: readonly string[]): string {
  return `<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">${relationships.join('')}</Relationships>`
}

function bytesFor(contents: string | Uint8Array): Uint8Array {
  return typeof contents === 'string' ? new TextEncoder().encode(contents) : new Uint8Array(contents)
}

function fail(message: string): never {
  throw new Error(message)
}
