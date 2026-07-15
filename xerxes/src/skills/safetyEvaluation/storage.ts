// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { appendFile, mkdir } from 'node:fs/promises'
import { dirname } from 'node:path'

import type {
  SafetyEvaluationReport,
  SafetyEvaluationReportStore,
  SafetyReportFilesystemPort,
} from './types.js'

/** Bun-compatible filesystem port for callers who explicitly choose local report persistence. */
export const bunSafetyReportFilesystem: SafetyReportFilesystemPort = {
  async appendText(path: string, text: string): Promise<void> {
    await appendFile(path, text, 'utf8')
  },
  dirname,
  async ensureDirectory(path: string): Promise<void> {
    await mkdir(path, { recursive: true })
  },
}

/** Serialize one complete report as an appendable JSONL record. */
export function serializeSafetyEvaluationReport(report: SafetyEvaluationReport): string {
  return `${JSON.stringify(report)}\n`
}

/**
 * Explicit JSONL report store. Evaluation does not instantiate this class by
 * default, so durable storage remains caller-controlled and opt-in.
 */
export class JsonlSafetyReportStore implements SafetyEvaluationReportStore {
  private readonly filesystem: SafetyReportFilesystemPort
  private readonly path: string

  constructor(path: string, filesystem: SafetyReportFilesystemPort = bunSafetyReportFilesystem) {
    if (path.trim() === '') throw new RangeError('report path must not be empty')
    this.path = path
    this.filesystem = filesystem
  }

  async save(report: SafetyEvaluationReport): Promise<void> {
    await this.filesystem.ensureDirectory(this.filesystem.dirname(this.path))
    await this.filesystem.appendText(this.path, serializeSafetyEvaluationReport(report))
  }
}
