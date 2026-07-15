// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/** Base failure for native Office Open XML package editing. */
export class OfficePackageError extends Error {
  constructor(message: string, options: { readonly cause?: unknown } = {}) {
    super(message, options)
    this.name = 'OfficePackageError'
  }
}

/** A requested package part or required presentation structure is absent. */
export class OfficePackagePartNotFoundError extends OfficePackageError {
  readonly partName: string

  constructor(partName: string) {
    super(`Office package part not found: ${partName}`)
    this.name = 'OfficePackagePartNotFoundError'
    this.partName = partName
  }
}

/** Raised for an OOXML structure that cannot be represented safely by this native editor. */
export class UnsupportedOfficeOperationError extends OfficePackageError {
  constructor(message: string) {
    super(message)
    this.name = 'UnsupportedOfficeOperationError'
  }
}

/** Raised when a package does not satisfy the small native validation contract. */
export class OfficePackageValidationError extends OfficePackageError {
  constructor(message: string) {
    super(message)
    this.name = 'OfficePackageValidationError'
  }
}

/** Raised when a tracked-change author cannot be inferred without guessing. */
export class AmbiguousTrackedChangeAuthorError extends OfficePackageError {
  constructor(message: string) {
    super(message)
    this.name = 'AmbiguousTrackedChangeAuthorError'
  }
}

/** Raised when a ZIP feature is outside the intentionally small OOXML archive reader. */
export class UnsupportedOfficeZipError extends OfficePackageError {
  constructor(message: string) {
    super(message)
    this.name = 'UnsupportedOfficeZipError'
  }
}
