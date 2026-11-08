// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { extname } from 'node:path'

import { UnsupportedOfficeZipError } from '../skills/powerpoint/errors.js'
import { readOfficeZip } from '../skills/powerpoint/zip.js'

/** Office documents whose text ReadFile extracts natively. */
export type OfficeTextKind = 'docx' | 'pptx'

/** Git's rule: a NUL byte in the first 8,000 bytes marks a file as binary. */
export const BINARY_SNIFF_BYTES = 8000

/** The document kind `ReadFile` extracts for this path, if any. */
export function officeTextKind(path: string): OfficeTextKind | undefined {
  const extension = extname(path).toLowerCase()
  if (extension === '.docx' || extension === '.docm') return 'docx'
  if (extension === '.pptx' || extension === '.pptm') return 'pptx'
  return undefined
}

/** True when the sample (the start of a file) looks binary rather than text. */
export function looksBinary(sample: Uint8Array): boolean {
  const end = Math.min(sample.length, BINARY_SNIFF_BYTES)
  for (let index = 0; index < end; index += 1) if (sample[index] === 0) return true
  return false
}

/** What to tell the model about a binary file ReadFile will not return. */
export function binaryFileHint(path: string): string {
  switch (extname(path).toLowerCase()) {
    case '.xlsx':
    case '.xlsm':
      return 'is an Excel workbook; read it with XlsxReader'
    case '.pdf':
      return 'is a PDF; ReadFile has no PDF text extraction, so use a PDF tool or skill if one is listed, or say it cannot be read here'
    case '.doc':
    case '.ppt':
    case '.xls':
      return 'is a legacy Office binary; convert it to .docx, .pptx or .xlsx first'
    default:
      return 'is a binary file; ReadFile returns text only'
  }
}

/** Plain text of a .docx (body paragraphs and table cells) or .pptx (slides in order). */
export function extractOfficeText(bytes: Uint8Array, kind: OfficeTextKind): string {
  const files = readOfficeZip(bytes)
  const decoder = new TextDecoder()
  if (kind === 'docx') {
    const body = files.get('word/document.xml')
    if (body === undefined) throw new UnsupportedOfficeZipError('word/document.xml is missing')
    return tidy(wordXmlText(decoder.decode(body)))
  }
  const slides = [...files.keys()]
    .map(name => ({ name, number: /^ppt\/slides\/slide(\d+)\.xml$/u.exec(name)?.[1] }))
    .filter((entry): entry is { name: string; number: string } => entry.number !== undefined)
    .sort((left, right) => Number(left.number) - Number(right.number))
  if (slides.length === 0) throw new UnsupportedOfficeZipError('the presentation has no slides')
  return slides
    .map(slide => `--- Slide ${slide.number} ---\n${tidy(drawingXmlText(decoder.decode(files.get(slide.name)!)))}`)
    .join('\n\n')
}

function wordXmlText(xml: string): string {
  return stripTags(xml
    // Field codes and tracked deletions are not visible text.
    .replace(/<w:instrText\b[^>]*>[\s\S]*?<\/w:instrText>/gu, '')
    .replace(/<w:delText\b[^>]*>[\s\S]*?<\/w:delText>/gu, '')
    .replace(/<w:tab\/>/gu, '\t')
    .replace(/<w:(?:br|cr)\b[^>]*\/>/gu, '\n')
    .replace(/<\/w:p>/gu, '\n')
    .replace(/<\/w:tc>/gu, '\t')
    .replace(/<\/w:tr>/gu, '\n'))
}

function drawingXmlText(xml: string): string {
  return stripTags(xml
    .replace(/<a:br\b[^>]*\/>/gu, '\n')
    .replace(/<\/a:p>/gu, '\n'))
}

function stripTags(xml: string): string {
  return decodeXml(xml.replace(/<[^>]*>/gu, ''))
}

function decodeXml(value: string): string {
  return value.replace(/&(#x[0-9a-fA-F]+|#\d+|lt|gt|amp|quot|apos);/gu, (whole, entity: string) => {
    switch (entity) {
      case 'lt': return '<'
      case 'gt': return '>'
      case 'amp': return '&'
      case 'quot': return '"'
      case 'apos': return "'"
      default: {
        const code = entity[1] === 'x' ? Number.parseInt(entity.slice(2), 16) : Number.parseInt(entity.slice(1), 10)
        return Number.isInteger(code) && code >= 0 && code <= 0x10ffff ? String.fromCodePoint(code) : whole
      }
    }
  })
}

/** Trim trailing cell tabs and spaces, and collapse runs of blank lines. */
function tidy(text: string): string {
  return text
    .split('\n')
    .map(line => line.replace(/[\t ]+$/u, ''))
    .join('\n')
    .replace(/\n{3,}/gu, '\n\n')
    .trim()
}
