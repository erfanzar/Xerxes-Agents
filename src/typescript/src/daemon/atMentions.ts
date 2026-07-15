// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { realpath, stat } from "node:fs/promises";
import { isAbsolute, relative, resolve } from "node:path";

import { resolveProjectFileMentionRoot } from "./projectFileMentions.js";

export const MAX_ATTACHMENT_BYTES = 100 * 1024;
export const MAX_ATTACHMENT_FILES = 20;
const UNQUOTED_AT_RE = /(?:^|\s)@([\w\-./\\~][\w\-./\\~:#]*)/g;
const QUOTED_AT_RE = /(?:^|\s)@"([^"]+)"/g;

export interface AtMentionedFile {
  readonly endLine?: number;
  readonly filePath: string;
  readonly raw: string;
  readonly startLine?: number;
}

export interface ProcessedAtMentions {
  readonly enhancedMessage: string;
  readonly mentionedFiles: readonly string[];
}

/** Extract unique file mentions without treating email addresses as paths. */
export function extractAtMentionedFiles(text: string): AtMentionedFile[] {
  const mentions: AtMentionedFile[] = [];
  const seen = new Set<string>();

  for (const expression of [QUOTED_AT_RE, UNQUOTED_AT_RE]) {
    expression.lastIndex = 0;
    for (let match = expression.exec(text); match; match = expression.exec(text)) {
      const raw = match[1] ?? "";
      if (!raw || seen.has(raw)) {
        continue;
      }
      seen.add(raw);
      mentions.push({ raw, ...parseLineRange(raw) });
    }
  }

  return mentions;
}

/**
 * Attach referenced workspace files to the provider prompt while retaining the
 * user's original message below the attachment block. Missing, binary, and
 * out-of-workspace paths stay literal and never become hidden reads.
 */
export async function processAtMentions(
  text: string,
  cwd: string,
  workspaceRoot?: string,
): Promise<ProcessedAtMentions> {
  const mentions = extractAtMentionedFiles(text);
  if (!mentions.length) {
    return { enhancedMessage: text, mentionedFiles: [] };
  }

  const root = await canonicalDirectory(
    workspaceRoot ?? (await resolveProjectFileMentionRoot(cwd)),
  );
  if (!root) {
    return { enhancedMessage: text, mentionedFiles: [] };
  }

  const fileBlocks: string[] = [];
  const mentionedFiles: string[] = [];
  let remainingBytes =
    MAX_ATTACHMENT_BYTES - utf8ByteLength("<attached_files>\n\n</attached_files>");
  for (const mention of mentions.slice(0, MAX_ATTACHMENT_FILES)) {
    const separatorBytes = fileBlocks.length ? 2 : 0;
    if (remainingBytes <= separatorBytes) {
      break;
    }
    const file = await readMentionedFile(
      mention,
      cwd,
      root,
      remainingBytes - separatorBytes,
    );
    if (!file) {
      continue;
    }

    const escapedPath = escapeXmlAttribute(file.path);
    const prefix = `<file path="${escapedPath}">\n`;
    const suffix = "\n</file>";
    const truncationNote = `\n<!-- Note: ${escapedPath} was truncated (attachment limit is ${MAX_ATTACHMENT_BYTES / 1024}KB). Use read_file for the full content. -->`;
    const aggregateNote = `\n<!-- Note: ${escapedPath} was truncated by the ${MAX_ATTACHMENT_BYTES / 1024}KB aggregate attachment limit. Use read_file for the full content. -->`;
    const baseBudget =
      remainingBytes - separatorBytes - utf8ByteLength(`${prefix}${suffix}`);
    const note = file.truncated
      ? truncationNote
      : utf8ByteLength(file.content) > baseBudget
        ? aggregateNote
        : "";
    const fixedBytes = utf8ByteLength(`${prefix}${suffix}${note}`);
    const contentBudget = remainingBytes - separatorBytes - fixedBytes;
    if (contentBudget <= 0) {
      break;
    }
    const content = utf8Prefix(file.content, contentBudget);
    const block = `${prefix}${content}${suffix}${note}`;
    const blockBytes = utf8ByteLength(block) + separatorBytes;
    if (blockBytes > remainingBytes) {
      break;
    }
    mentionedFiles.push(file.path);
    fileBlocks.push(block);
    remainingBytes -= blockBytes;
  }

  if (!fileBlocks.length) {
    return { enhancedMessage: text, mentionedFiles: [] };
  }

  return {
    enhancedMessage: `<attached_files>\n${fileBlocks.join("\n\n")}\n</attached_files>\n\n${text}`,
    mentionedFiles,
  };
}

function parseLineRange(raw: string): Omit<AtMentionedFile, "raw"> {
  const hash = raw.lastIndexOf("#");
  if (hash < 0) {
    return { filePath: raw };
  }

  const range = raw.slice(hash + 1).match(/^L(\d+)(?:-L?(\d+))?$/);
  if (!range) {
    return { filePath: raw };
  }

  const startLine = Number.parseInt(range[1] ?? "1", 10);
  const parsedEnd = range[2]
    ? Number.parseInt(range[2], 10)
    : undefined;
  return {
    filePath: raw.slice(0, hash),
    startLine,
    ...(parsedEnd === undefined ? {} : { endLine: parsedEnd }),
  };
}

async function canonicalDirectory(path: string): Promise<string | undefined> {
  try {
    const canonical = await realpath(resolve(path));
    return (await stat(canonical)).isDirectory() ? canonical : undefined;
  } catch {
    return undefined;
  }
}

async function readMentionedFile(
  mention: AtMentionedFile,
  cwd: string,
  root: string,
  maxBytes: number,
): Promise<
  | {
      readonly content: string;
      readonly path: string;
      readonly truncated: boolean;
    }
  | undefined
> {
  const candidate = isAbsolute(mention.filePath)
    ? mention.filePath
    : resolve(cwd, mention.filePath);
  try {
    const canonical = await realpath(candidate);
    if (!containedBy(root, canonical) || !(await stat(canonical)).isFile()) {
      return undefined;
    }

    const blob = Bun.file(canonical);
    const sourceTruncated = blob.size > maxBytes;
    const bytes = new Uint8Array(
      await blob.slice(0, Math.min(blob.size, maxBytes + 1)).arrayBuffer(),
    );
    const raw = new TextDecoder("utf-8", { fatal: true }).decode(bytes, {
      stream: sourceTruncated,
    });
    if (raw.includes("\0")) {
      return undefined;
    }

    const lines = raw.split("\n");
    const start = Math.max(0, (mention.startLine ?? 1) - 1);
    const requestedEnd = Math.max(start + 1, mention.endLine ?? lines.length);
    const end = Math.min(lines.length, requestedEnd);
    const numbered = lines
      .slice(start, end)
      .map((line, index) => `${start + index + 1} | ${line}`)
      .join("\n");
    if (start >= lines.length) {
      return undefined;
    }
    const total = sourceTruncated ? `at least ${lines.length}` : String(lines.length);
    const header = `[${canonical}: lines ${start + 1}-${end} of ${total}]`;
    const content = `${header}\n${numbered}`;
    const truncated = sourceTruncated || utf8ByteLength(content) > maxBytes;
    return {
      content: truncated ? utf8Prefix(content, maxBytes) : content,
      path: canonical,
      truncated,
    };
  } catch {
    return undefined;
  }
}

function containedBy(root: string, candidate: string): boolean {
  const path = relative(root, candidate);
  return path === "" || (!path.startsWith("..") && !isAbsolute(path));
}

function utf8ByteLength(value: string): number {
  return new TextEncoder().encode(value).byteLength;
}

function utf8Prefix(value: string, maxBytes: number): string {
  const encoded = new TextEncoder().encode(value);
  if (encoded.byteLength <= maxBytes) {
    return value;
  }
  return new TextDecoder().decode(encoded.slice(0, maxBytes), {
    stream: true,
  });
}

function escapeXmlAttribute(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}
