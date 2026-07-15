// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { readFile } from 'node:fs/promises'

import {
  addSlide,
  cleanUnusedFiles,
  mergeDocumentRuns,
  packOfficeDirectory,
  simplifyDocumentRedlines,
} from './index.js'
import { BunOfficePackageDirectory } from './package.js'

/** Help for the Bun-native PowerPoint and OOXML skill commands. */
export const POWERPOINT_CLI_USAGE = `Usage: xerxes skill powerpoint <command> [arguments]

Commands:
  add-slide <unpacked-dir> <slideN.xml|slideLayoutN.xml>
  clean <unpacked-dir>
  pack <unpacked-dir> <output.docx|output.pptx|output.xlsx> [--original <file>] [--validate true|false]
  merge-runs <unpacked-docx-dir>
  simplify-redlines <unpacked-docx-dir>

All commands use the Bun-native OOXML package editor. The optional --original
baseline receives structural validation; schema/rendering validation and
undocumented auto-repairs are intentionally not performed.`

/** Explicit host ports for the native PowerPoint command runner. */
export interface PowerPointCliDependencies {
  readonly readBytes?: (path: string) => Promise<Uint8Array>
  readonly writeLine?: (line: string) => void | Promise<void>
}

interface PackCommandOptions {
  readonly inputDirectory: string
  readonly originalPath?: string
  readonly outputFile: string
  readonly validate: boolean
}

class PowerPointCliUsageError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'PowerPointCliUsageError'
  }
}

/** Run a PowerPoint skill command without a Python subprocess or package dependency. */
export async function runPowerPointCli(
  args: readonly string[],
  dependencies: PowerPointCliDependencies = {},
): Promise<number> {
  const writeLine = dependencies.writeLine ?? ((line: string) => console.log(line))
  const [command, ...rest] = args
  try {
    if (command === undefined || isHelp(command)) {
      await writeLine(POWERPOINT_CLI_USAGE)
      return 0
    }
    switch (command) {
      case 'add-slide':
        await runAddSlide(rest, writeLine)
        return 0
      case 'clean':
        await runCleanup(rest, writeLine)
        return 0
      case 'pack':
        await runPack(rest, dependencies, writeLine)
        return 0
      case 'merge-runs':
        await runMergeRuns(rest, writeLine)
        return 0
      case 'simplify-redlines':
        await runSimplifyRedlines(rest, writeLine)
        return 0
      default:
        throw new PowerPointCliUsageError(`Unknown PowerPoint command: ${command}`)
    }
  } catch (error) {
    const message = errorMessage(error)
    if (error instanceof PowerPointCliUsageError) {
      await writeLine(`${message}\n\n${POWERPOINT_CLI_USAGE}`)
      return 2
    }
    await writeLine(`PowerPoint command failed: ${message}`)
    return 1
  }
}

/** Standalone Bun entry point for local PowerPoint skill development. */
export async function main(args: readonly string[] = process.argv.slice(2)): Promise<number> {
  return runPowerPointCli(args)
}

async function runAddSlide(
  args: readonly string[],
  writeLine: (line: string) => void | Promise<void>,
): Promise<void> {
  const [directory, source, ...extra] = args
  if (directory === undefined || source === undefined || extra.length) {
    throw new PowerPointCliUsageError(
      'Usage: powerpoint add-slide <unpacked-dir> <slideN.xml|slideLayoutN.xml>',
    )
  }
  const addition = await addSlide(new BunOfficePackageDirectory(directory), source)
  await writeLine(`Created ${addition.slideFileName} from ${addition.source.fileName}`)
  await writeLine(`Add to presentation.xml <p:sldIdLst>: ${addition.presentationEntry}`)
}

async function runCleanup(
  args: readonly string[],
  writeLine: (line: string) => void | Promise<void>,
): Promise<void> {
  const directory = exactlyOneArgument(args, 'Usage: powerpoint clean <unpacked-dir>')
  const removed = await cleanUnusedFiles(new BunOfficePackageDirectory(directory))
  if (!removed.length) {
    await writeLine('No unreferenced files found')
    return
  }
  await writeLine(`Removed ${removed.length} unreferenced files:`)
  for (const partName of removed) await writeLine(`  ${partName}`)
}

async function runPack(
  args: readonly string[],
  dependencies: PowerPointCliDependencies,
  writeLine: (line: string) => void | Promise<void>,
): Promise<void> {
  const options = parsePackCommand(args)
  const originalPackage = options.originalPath === undefined
    ? undefined
    : await (dependencies.readBytes ?? readOfficeBytes)(options.originalPath)
  const packed = await packOfficeDirectory(options.inputDirectory, options.outputFile, {
    ...(originalPackage === undefined ? {} : { originalPackage }),
    validate: options.validate,
  })
  await writeLine(packed.message)
  if (packed.validation?.message) await writeLine(packed.validation.message)
}

async function runMergeRuns(
  args: readonly string[],
  writeLine: (line: string) => void | Promise<void>,
): Promise<void> {
  const directory = exactlyOneArgument(args, 'Usage: powerpoint merge-runs <unpacked-docx-dir>')
  const result = await mergeDocumentRuns(new BunOfficePackageDirectory(directory))
  await writeLine(result.message)
}

async function runSimplifyRedlines(
  args: readonly string[],
  writeLine: (line: string) => void | Promise<void>,
): Promise<void> {
  const directory = exactlyOneArgument(args, 'Usage: powerpoint simplify-redlines <unpacked-docx-dir>')
  const result = await simplifyDocumentRedlines(new BunOfficePackageDirectory(directory))
  await writeLine(result.message)
}

function parsePackCommand(args: readonly string[]): PackCommandOptions {
  const positionals: string[] = []
  let originalPath: string | undefined
  let validate = true
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index]
    if (argument === undefined) continue
    if (argument === '--original') {
      originalPath = requiredValue(args, ++index, '--original')
      continue
    }
    if (argument === '--validate') {
      validate = parseBoolean(requiredValue(args, ++index, '--validate'), '--validate')
      continue
    }
    if (isHelp(argument)) throw new PowerPointCliUsageError('Usage: powerpoint pack <unpacked-dir> <output-file>')
    if (argument.startsWith('-')) throw new PowerPointCliUsageError(`Unknown powerpoint pack option: ${argument}`)
    positionals.push(argument)
  }
  if (positionals.length !== 2) {
    throw new PowerPointCliUsageError(
      'Usage: powerpoint pack <unpacked-dir> <output.docx|output.pptx|output.xlsx> [--original <file>] [--validate true|false]',
    )
  }
  const [inputDirectory, outputFile] = positionals
  if (inputDirectory === undefined || outputFile === undefined) throw new PowerPointCliUsageError('PowerPoint pack requires input and output paths')
  return {
    inputDirectory,
    ...(originalPath === undefined ? {} : { originalPath }),
    outputFile,
    validate,
  }
}

function exactlyOneArgument(args: readonly string[], usage: string): string {
  if (args.length !== 1 || args[0] === undefined || isHelp(args[0])) throw new PowerPointCliUsageError(usage)
  return args[0]
}

function requiredValue(args: readonly string[], index: number, flag: string): string {
  const value = args[index]
  if (value === undefined || !value.trim()) throw new PowerPointCliUsageError(`${flag} requires a value`)
  return value
}

function parseBoolean(value: string, flag: string): boolean {
  if (value === 'true') return true
  if (value === 'false') return false
  throw new PowerPointCliUsageError(`${flag} must be true or false`)
}

function isHelp(value: string): boolean {
  return value === '--help' || value === '-h' || value === 'help'
}

async function readOfficeBytes(path: string): Promise<Uint8Array> {
  return new Uint8Array(await readFile(path))
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error)
}
