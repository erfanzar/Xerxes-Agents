// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { resolve } from "node:path";
import { pathToFileURL } from "node:url";

import { bunOcrDocumentFilesystem } from "./files.js";
import {
  checkMarkerDiskSpace,
  extractMarkerDocument,
  formatMarkerDiskSpaceCheck,
  formatMarkerExtraction,
  type MarkerOutputFormat,
} from "./marker.js";
import {
  formatPdfMetadata,
  formatPdfTableExtraction,
  formatPdfTextExtraction,
  parsePdfPageSelection,
  PdfDocumentExtractor,
} from "./pymupdf.js";
import type {
  OcrConversionPort,
  OcrDocumentDiskSpacePort,
  OcrDocumentFilesystemPort,
  PdfTextConversionPort,
} from "./types.js";

/** Help for the Bun-native OCR and document extraction CLI. */
export const OCR_DOCUMENTS_CLI_USAGE = `Usage: xerxes skill ocr-and-documents [--adapter <module>] <pymupdf|marker> [arguments]

Commands:
  pymupdf <document.pdf> [--markdown | --tables | --images [directory] | --metadata] [--pages <zero-based-page|start-end>]
  marker --check [--path <directory>]
  marker <document.pdf> [--json] [--output-dir <directory>] [--use-llm]

The legacy spelling --output_dir is accepted for marker image output.

Every operational command needs an explicit adapter selected by the caller:
  xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts pymupdf document.pdf --markdown

The adapter must export a ready Bun-native PDF/OCR conversion port. Xerxes never
uses PyMuPDF, marker-pdf, Python, or a subprocess as a fallback.`;

/** Caller-owned native conversion ports accepted by the OCR CLI. */
export interface OcrDocumentsCliAdapter {
  readonly diskSpace?: OcrDocumentDiskSpacePort;
  readonly filesystem?: OcrDocumentFilesystemPort;
  readonly markerConverter?: OcrConversionPort;
  readonly pdfConverter?: PdfTextConversionPort;
}

export interface OcrDocumentsCliDependencies extends OcrDocumentsCliAdapter {
  /** Load an adapter only after the caller explicitly chose a local module. */
  readonly loadAdapter?: (
    path: string,
  ) => OcrDocumentsCliAdapter | Promise<OcrDocumentsCliAdapter>;
  /** Standard command output. */
  readonly writeLine?: (line: string) => void | Promise<void>;
  /** Non-data diagnostics, such as marker image save summaries. */
  readonly writeDiagnostic?: (line: string) => void | Promise<void>;
}

interface AdapterArguments {
  readonly adapterPath?: string;
  readonly command: readonly string[];
}

interface ParsedPymupdfCommand {
  readonly imageDirectory?: string;
  readonly mode: "images" | "markdown" | "metadata" | "tables" | "text";
  readonly pages?: readonly number[];
  readonly path: string;
}

interface ParsedMarkerCommand {
  readonly checkPath?: string;
  readonly mode: "check" | "extract";
  readonly outputDirectory?: string;
  readonly outputFormat: MarkerOutputFormat;
  readonly path?: string;
  readonly useLlm: boolean;
}

class OcrDocumentsCliUsageError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OcrDocumentsCliUsageError";
  }
}

class OcrDocumentsCliConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "OcrDocumentsCliConfigurationError";
  }
}

/**
 * Execute legacy OCR/document workflows using only caller-selected Bun-native
 * engines. No ambient package discovery or process invocation is performed.
 */
export async function runOcrDocumentsCli(
  args: readonly string[],
  dependencies: OcrDocumentsCliDependencies = {},
): Promise<number> {
  const writeLine = dependencies.writeLine ?? ((line) => console.log(line));
  const writeDiagnostic =
    dependencies.writeDiagnostic ?? ((line) => console.error(line));
  try {
    const adapterArguments = extractAdapterArguments(args);
    const activeDependencies = await resolveDependencies(
      adapterArguments,
      dependencies,
    );
    const [service, ...rest] = adapterArguments.command;
    if (service === undefined || isHelp(service)) {
      await writeLine(OCR_DOCUMENTS_CLI_USAGE);
      return 0;
    }
    switch (service) {
      case "pymupdf":
        return await runPymupdfCommand(rest, activeDependencies, writeLine);
      case "marker":
        return await runMarkerCommand(
          rest,
          activeDependencies,
          writeLine,
          writeDiagnostic,
        );
      default:
        throw new OcrDocumentsCliUsageError(
          `Unknown OCR/document command: ${service}`,
        );
    }
  } catch (error) {
    const message = errorMessage(error);
    if (error instanceof OcrDocumentsCliUsageError) {
      await writeLine(`${message}\n\n${OCR_DOCUMENTS_CLI_USAGE}`);
      return 2;
    }
    if (error instanceof OcrDocumentsCliConfigurationError) {
      await writeLine(
        `${message}\n\nSelect an explicit adapter with \`xerxes skill ocr-and-documents --adapter ./ocr-documents.adapter.ts …\`.`,
      );
      return 2;
    }
    await writeLine(`OCR/document command failed: ${message}`);
    return 1;
  }
}

/** Direct Bun entry point for the skill during development and embedding. */
export async function main(
  args: readonly string[] = process.argv.slice(2),
): Promise<number> {
  return runOcrDocumentsCli(args);
}

/**
 * Load a local adapter only when a caller supplied --adapter explicitly.
 *
 * The adapter owns the actual Bun/WASM/remote conversion implementation and
 * can never be inferred from a Python package, executable, or environment.
 */
export async function loadOcrDocumentsCliAdapter(
  path: string,
): Promise<OcrDocumentsCliAdapter> {
  const text = requiredText(path, "--adapter");
  const url = text.startsWith("file:")
    ? new URL(text)
    : pathToFileURL(resolve(text));
  if (url.protocol !== "file:") {
    throw new OcrDocumentsCliConfigurationError(
      "--adapter must be a local file path or file: URL",
    );
  }
  let loaded: Record<string, unknown>;
  try {
    loaded = (await import(url.href)) as Record<string, unknown>;
  } catch (error) {
    throw new OcrDocumentsCliConfigurationError(
      `could not load explicit OCR/document adapter: ${errorMessage(error)}`,
    );
  }
  const candidate = loaded.default ?? loaded.ocrDocumentsCliAdapter;
  if (!isRecord(candidate)) {
    throw new OcrDocumentsCliConfigurationError(
      "the explicit adapter must export an object as default or ocrDocumentsCliAdapter",
    );
  }
  validateAdapter(candidate);
  return candidate as OcrDocumentsCliAdapter;
}

async function runPymupdfCommand(
  args: readonly string[],
  dependencies: OcrDocumentsCliDependencies,
  writeLine: (line: string) => void | Promise<void>,
): Promise<number> {
  if (args.some(isHelp)) {
    await writeLine(
      "Usage: xerxes skill ocr-and-documents pymupdf <document.pdf> [--markdown | --tables | --images [directory] | --metadata] [--pages <zero-based-page|start-end>]",
    );
    return 0;
  }
  const command = parsePymupdfCommand(args);
  const extractor = new PdfDocumentExtractor(requirePdfConverter(dependencies));
  switch (command.mode) {
    case "text": {
      const extraction = await extractor.extractText(
        command.path,
        command.pages,
      );
      await writeLine(formatPdfTextExtraction(extraction));
      return 0;
    }
    case "markdown":
      await writeLine(
        await extractor.extractMarkdown(command.path, command.pages),
      );
      return 0;
    case "tables": {
      const extraction = await extractor.extractTables(command.path);
      await writeLine(formatPdfTableExtraction(extraction));
      return 0;
    }
    case "images": {
      const outputDirectory = command.imageDirectory ?? "./images";
      const extraction = await extractor.extractImages(
        command.path,
        outputDirectory,
        dependencies.filesystem ?? bunOcrDocumentFilesystem,
      );
      await writeLine(
        `Extracted ${extraction.images.length} images to ${displayDirectory(extraction.outputDirectory)}`,
      );
      return 0;
    }
    case "metadata":
      await writeLine(
        formatPdfMetadata(await extractor.metadata(command.path)),
      );
      return 0;
  }
}

async function runMarkerCommand(
  args: readonly string[],
  dependencies: OcrDocumentsCliDependencies,
  writeLine: (line: string) => void | Promise<void>,
  writeDiagnostic: (line: string) => void | Promise<void>,
): Promise<number> {
  if (args.some(isHelp)) {
    await writeLine(
      "Usage: xerxes skill ocr-and-documents marker --check [--path <directory>]\n       xerxes skill ocr-and-documents marker <document.pdf> [--json] [--output-dir <directory>] [--use-llm]",
    );
    return 0;
  }
  const command = parseMarkerCommand(args);
  if (command.mode === "check") {
    const check = await checkMarkerDiskSpace(
      requireDiskSpace(dependencies),
      command.checkPath ?? "/",
    );
    await writeLine(formatMarkerDiskSpaceCheck(check));
    return check.meetsRequirement ? 0 : 1;
  }

  const extraction = await extractMarkerDocument(
    command.path ?? "",
    requireMarkerConverter(dependencies),
    {
      ...(command.outputDirectory === undefined
        ? {}
        : { outputDirectory: command.outputDirectory }),
      outputFormat: command.outputFormat,
      useLlm: command.useLlm,
    },
    command.outputDirectory === undefined
      ? undefined
      : (dependencies.filesystem ?? bunOcrDocumentFilesystem),
  );
  await writeLine(formatMarkerExtraction(extraction, command.outputFormat));
  if (command.outputDirectory !== undefined && extraction.images.length) {
    await writeDiagnostic(
      `Saved ${extraction.images.length} image(s) to ${displayDirectory(command.outputDirectory)}`,
    );
  }
  return 0;
}

function parsePymupdfCommand(args: readonly string[]): ParsedPymupdfCommand {
  const [path, ...rest] = args;
  const normalizedPath = requiredText(path, "pymupdf requires a document path");
  let imageDirectory: string | undefined;
  let markdown = false;
  let metadata = false;
  let pages: readonly number[] | undefined;
  let tables = false;
  for (let index = 0; index < rest.length; index += 1) {
    const argument = rest[index];
    if (argument === undefined) continue;
    switch (argument) {
      case "--markdown":
        markdown = true;
        break;
      case "--tables":
        tables = true;
        break;
      case "--metadata":
        metadata = true;
        break;
      case "--images": {
        const following = rest[index + 1];
        if (following !== undefined && !following.startsWith("--")) {
          imageDirectory = requiredText(following, "--images");
          index += 1;
        } else {
          imageDirectory = "./images";
        }
        break;
      }
      case "--pages": {
        const selection = requiredText(rest[index + 1], "--pages");
        try {
          pages = parsePdfPageSelection(selection);
        } catch (error) {
          throw new OcrDocumentsCliUsageError(errorMessage(error));
        }
        index += 1;
        break;
      }
      default:
        if (argument.startsWith("-")) {
          throw new OcrDocumentsCliUsageError(
            `Unknown pymupdf option: ${argument}`,
          );
        }
        throw new OcrDocumentsCliUsageError(
          `Unexpected pymupdf argument: ${argument}`,
        );
    }
  }
  // Preserve the former script's precedence if callers supplied multiple modes.
  const mode = metadata
    ? "metadata"
    : tables
      ? "tables"
      : imageDirectory === undefined
        ? markdown
          ? "markdown"
          : "text"
        : "images";
  return {
    ...(imageDirectory === undefined ? {} : { imageDirectory }),
    mode,
    ...(pages === undefined ? {} : { pages }),
    path: normalizedPath,
  };
}

function parseMarkerCommand(args: readonly string[]): ParsedMarkerCommand {
  if (!args.length) {
    throw new OcrDocumentsCliUsageError(
      "marker requires a document path or --check",
    );
  }
  let check = false;
  let checkPath: string | undefined;
  let outputDirectory: string | undefined;
  let outputFormat: MarkerOutputFormat = "markdown";
  let path: string | undefined;
  let useLlm = false;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === undefined) continue;
    switch (argument) {
      case "--check":
        check = true;
        break;
      case "--path":
        checkPath = requiredText(args[index + 1], "--path");
        index += 1;
        break;
      case "--json":
        outputFormat = "json";
        break;
      case "--output-dir":
      case "--output_dir":
        outputDirectory = requiredText(args[index + 1], argument);
        index += 1;
        break;
      case "--use-llm":
      case "--use_llm":
        useLlm = true;
        break;
      default:
        if (argument.startsWith("-")) {
          throw new OcrDocumentsCliUsageError(
            `Unknown marker option: ${argument}`,
          );
        }
        if (path !== undefined) {
          throw new OcrDocumentsCliUsageError(
            "marker accepts exactly one document path",
          );
        }
        path = requiredText(argument, "marker document path");
    }
  }
  if (check) {
    if (
      path !== undefined ||
      outputDirectory !== undefined ||
      outputFormat !== "markdown" ||
      useLlm
    ) {
      throw new OcrDocumentsCliUsageError(
        "marker --check accepts only an optional --path",
      );
    }
    return {
      ...(checkPath === undefined ? {} : { checkPath }),
      mode: "check",
      outputFormat: "markdown",
      useLlm: false,
    };
  }
  if (checkPath !== undefined) {
    throw new OcrDocumentsCliUsageError(
      "--path is only valid with marker --check",
    );
  }
  return {
    mode: "extract",
    ...(outputDirectory === undefined ? {} : { outputDirectory }),
    outputFormat,
    path: requiredText(path, "marker requires a document path"),
    useLlm,
  };
}

function requirePdfConverter(
  dependencies: OcrDocumentsCliDependencies,
): PdfTextConversionPort {
  if (dependencies.pdfConverter === undefined) {
    throw new OcrDocumentsCliConfigurationError(
      "pymupdf commands require a caller-provided Bun-native pdfConverter",
    );
  }
  return dependencies.pdfConverter;
}

function requireMarkerConverter(
  dependencies: OcrDocumentsCliDependencies,
): OcrConversionPort {
  if (dependencies.markerConverter === undefined) {
    throw new OcrDocumentsCliConfigurationError(
      "marker commands require a caller-provided Bun-native markerConverter",
    );
  }
  return dependencies.markerConverter;
}

function requireDiskSpace(
  dependencies: OcrDocumentsCliDependencies,
): OcrDocumentDiskSpacePort {
  if (dependencies.diskSpace === undefined) {
    throw new OcrDocumentsCliConfigurationError(
      "marker --check requires a caller-provided diskSpace adapter",
    );
  }
  return dependencies.diskSpace;
}

async function resolveDependencies(
  argumentsWithAdapter: AdapterArguments,
  dependencies: OcrDocumentsCliDependencies,
): Promise<OcrDocumentsCliDependencies> {
  if (argumentsWithAdapter.adapterPath === undefined) return dependencies;
  const loadAdapter = dependencies.loadAdapter ?? loadOcrDocumentsCliAdapter;
  const adapter = await loadAdapter(argumentsWithAdapter.adapterPath);
  return {
    ...dependencies,
    ...adapter,
    ...(dependencies.loadAdapter === undefined
      ? {}
      : { loadAdapter: dependencies.loadAdapter }),
    ...(dependencies.writeLine === undefined
      ? {}
      : { writeLine: dependencies.writeLine }),
    ...(dependencies.writeDiagnostic === undefined
      ? {}
      : { writeDiagnostic: dependencies.writeDiagnostic }),
  };
}

function extractAdapterArguments(args: readonly string[]): AdapterArguments {
  const command: string[] = [];
  let adapterPath: string | undefined;
  for (let index = 0; index < args.length; index += 1) {
    const argument = args[index];
    if (argument === "--adapter") {
      if (adapterPath !== undefined) {
        throw new OcrDocumentsCliUsageError(
          "--adapter may only be provided once",
        );
      }
      adapterPath = requiredText(args[index + 1], "--adapter");
      index += 1;
      continue;
    }
    if (argument !== undefined) command.push(argument);
  }
  return { ...(adapterPath === undefined ? {} : { adapterPath }), command };
}

function validateAdapter(adapter: Readonly<Record<string, unknown>>): void {
  for (const [name, value] of Object.entries(adapter)) {
    if (
      name !== "pdfConverter" &&
      name !== "markerConverter" &&
      name !== "diskSpace" &&
      name !== "filesystem"
    ) {
      throw new OcrDocumentsCliConfigurationError(
        `the explicit OCR/document adapter contains an unsupported property: ${name}`,
      );
    }
    if (!isRecord(value)) {
      throw new OcrDocumentsCliConfigurationError(
        `adapter.${name} must be an object implementing its documented port`,
      );
    }
  }
}

function displayDirectory(path: string): string {
  return path.endsWith("/") ? path : `${path}/`;
}

function requiredText(value: string | undefined, name: string): string {
  if (typeof value !== "string" || !value.trim()) {
    throw new OcrDocumentsCliUsageError(`${name} requires a non-empty value`);
  }
  return value.trim();
}

function isHelp(value: string): boolean {
  return value === "--help" || value === "-h" || value === "help";
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

if (import.meta.main) {
  process.exitCode = await main();
}
