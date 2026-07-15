// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import {
  OfficePackageValidationError,
  UnsupportedOfficeOperationError,
} from './errors.js'
import {
  BunOfficePackageDirectory,
  bunOfficeFileOutput,
  type OfficeBinaryOutputPort,
  type OfficePackagePort,
} from './package.js'
import { condenseOoxmlXml } from './xml.js'
import { createOfficeZip, readOfficeZip, type OfficeZipEntry } from './zip.js'

/** Office archive extensions handled by the native packer. */
export type OfficePackageKind = 'docx' | 'pptx' | 'xlsx'

/** Result returned by either the default structural validator or a caller-provided validation port. */
export interface OfficeValidationReport {
  readonly message?: string
  readonly repairs?: number
  readonly valid: boolean
}

/**
 * Optional integration point for full Office schema/redline validation. The
 * native packer always validates package structure itself but does not emulate
 * an external Office renderer or perform undocumented auto-repairs.
 */
export interface OfficePackageValidator {
  validate(request: OfficeValidationRequest): Promise<OfficeValidationReport>
}

/** Complete information made available to an injected compatibility validator. */
export interface OfficeValidationRequest {
  readonly kind: OfficePackageKind
  readonly originalPackage?: Uint8Array
  readonly packageDirectory: OfficePackagePort
}

/** Options for transforming an unpacked Office directory into archive bytes. */
export interface PackOfficePackageOptions {
  readonly kind: OfficePackageKind
  readonly originalPackage?: Uint8Array
  readonly validate?: boolean
  readonly validator?: OfficePackageValidator
}

/** Options for packing an on-disk unpacked directory into an on-disk Office archive. */
export interface PackOfficeDirectoryOptions {
  readonly originalPackage?: Uint8Array
  readonly output?: OfficeBinaryOutputPort
  readonly validate?: boolean
  readonly validator?: OfficePackageValidator
}

/** Result emitted by a successful native pack. */
export interface PackedOfficePackage {
  readonly bytes: Uint8Array
  readonly kind: OfficePackageKind
  readonly message: string
  readonly validation?: OfficeValidationReport
}

/**
 * Pack all parts from an unpacked OOXML package into a deterministic ZIP.
 * XML and relationship parts are condensed in-memory; the source directory is
 * never copied, mutated, or delegated to Python.
 */
export async function packOfficePackage(
  packageDirectory: OfficePackagePort,
  options: PackOfficePackageOptions,
): Promise<PackedOfficePackage> {
  const kind = options.kind
  const validation = await validateForPack(packageDirectory, options)
  const entries: OfficeZipEntry[] = []
  for (const partName of await packageDirectory.listParts()) {
    const data = isXmlPart(partName)
      ? new TextEncoder().encode(condenseOoxmlXml(await packageDirectory.readText(partName)))
      : await packageDirectory.readBytes(partName)
    entries.push({ data, name: partName })
  }
  const bytes = createOfficeZip(entries)
  return {
    bytes,
    kind,
    message: `Successfully packed ${entries.length} Office parts as .${kind}`,
    ...(validation === undefined ? {} : { validation }),
  }
}

/** Pack an on-disk directory via the Bun-native package and binary-output adapters. */
export async function packOfficeDirectory(
  inputDirectory: string,
  outputFile: string,
  options: PackOfficeDirectoryOptions = {},
): Promise<PackedOfficePackage> {
  const kind = officePackageKindFromPath(outputFile)
  const packageDirectory = new BunOfficePackageDirectory(inputDirectory)
  const packed = await packOfficePackage(packageDirectory, {
    kind,
    ...(options.originalPackage === undefined ? {} : { originalPackage: options.originalPackage }),
    ...(options.validate === undefined ? {} : { validate: options.validate }),
    ...(options.validator === undefined ? {} : { validator: options.validator }),
  })
  await (options.output ?? bunOfficeFileOutput).writeBytes(outputFile, packed.bytes)
  return packed
}

/** Infer an Office kind from a `.docx`, `.pptx`, or `.xlsx` output filename. */
export function officePackageKindFromPath(outputFile: string): OfficePackageKind {
  if (typeof outputFile !== 'string' || !outputFile.trim()) throw new TypeError('outputFile must be a non-empty string')
  const extension = outputFile.trim().split('.').at(-1)?.toLowerCase()
  if (extension === 'docx' || extension === 'pptx' || extension === 'xlsx') return extension
  throw new UnsupportedOfficeOperationError(`${outputFile} must end in .docx, .pptx, or .xlsx`)
}

async function validateForPack(
  packageDirectory: OfficePackagePort,
  options: PackOfficePackageOptions,
): Promise<OfficeValidationReport | undefined> {
  if (options.validate === false) return undefined
  const request: OfficeValidationRequest = {
    kind: options.kind,
    packageDirectory,
    ...(options.originalPackage === undefined ? {} : { originalPackage: options.originalPackage }),
  }
  const report = options.validator === undefined
    ? await validateNativeOfficeStructure(request)
    : await options.validator.validate(request)
  if (!report.valid) throw new OfficePackageValidationError(report.message ?? `Office .${options.kind} validation failed`)
  return report
}

async function validateNativeOfficeStructure(request: OfficeValidationRequest): Promise<OfficeValidationReport> {
  const required = requiredOfficeParts(request.kind)
  const missing = (await Promise.all(required.map(async part => await request.packageDirectory.hasPart(part) ? undefined : part)))
    .filter((part): part is string => part !== undefined)
  if (missing.length) return { message: `Missing required Office part(s): ${missing.join(', ')}`, valid: false }
  if (request.originalPackage === undefined) return { message: 'Native OOXML structural validation passed', valid: true }
  let baselineParts: ReadonlyMap<string, Uint8Array>
  try {
    baselineParts = readOfficeZip(request.originalPackage)
  } catch (error) {
    return {
      message: `Could not read original Office archive for structural validation: ${errorMessage(error)}`,
      valid: false,
    }
  }
  const missingBaseline = required.filter(part => !baselineParts.has(part))
  if (missingBaseline.length) {
    return {
      message: `Original Office archive is missing required part(s): ${missingBaseline.join(', ')}`,
      valid: false,
    }
  }
  return {
    message: 'Native OOXML structural validation passed for package and baseline; schema, rendering, and auto-repair validation are not performed.',
    valid: true,
  }
}

function requiredOfficeParts(kind: OfficePackageKind): readonly string[] {
  const mainPart = kind === 'pptx'
    ? 'ppt/presentation.xml'
    : kind === 'docx'
      ? 'word/document.xml'
      : 'xl/workbook.xml'
  return ['[Content_Types].xml', '_rels/.rels', mainPart]
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}

function isXmlPart(partName: string): boolean {
  return partName.endsWith('.xml') || partName.endsWith('.rels')
}
