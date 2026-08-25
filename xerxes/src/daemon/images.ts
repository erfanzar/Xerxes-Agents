// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from "../core/errors.js";
import { sniffImageMediaType } from "../core/multimodal.js";
import type { ImageUrlContentPart } from "../types/messages.js";

/** Per-image decoded-byte cap for daemon turn attachments. */
export const MAX_TURN_IMAGE_BYTES = 10 * 1024 * 1024;
/** Combined decoded-byte cap for every image attached to one turn. */
export const MAX_TURN_IMAGES_TOTAL_BYTES = 20 * 1024 * 1024;
/** Bound the number of attachments per turn so a frame cannot grow without limit. */
export const MAX_TURN_IMAGES = 16;
/**
 * Total data-URL bytes kept inline when one transcript message is echoed to
 * clients.
 *
 * Turn submissions accept up to 20MB of decoded images and embed the base64
 * (~27MB) in `session.messages`, but every socket response must fit the
 * daemon's output cap — a verbatim transcript echo permanently wedged any
 * session that ever received an image, because initialize could never deliver
 * its own payload. Session payloads therefore replace data URLs beyond this
 * per-message budget with compact text placeholders; turn submission and the
 * provider-facing messages always keep the full images.
 */
export const MAX_TRANSCRIPT_INLINE_IMAGE_BYTES = 256 * 1024;
/**
 * Whole-projection data-URL ceiling across an entire transcript echo.
 *
 * The per-message budget above alone still allowed a wedge: ~64 turns each
 * carrying a legal ~250 KB screenshot re-accumulated to ~16 MB of inline
 * base64 and blew every initialize/open/status response past the socket
 * output cap. This outer bound is spent newest first during projection —
 * the most recent context keeps real pixels, the oldest inline images are
 * omitted first once the ceiling hits — so images contribute at most ~2 MiB
 * (an eighth of both transports' 16 MiB default frame cap) plus placeholder
 * bytes to any echoed payload, provably leaving frame headroom no matter how
 * many image-bearing turns the session accumulates.
 */
export const MAX_TRANSCRIPT_TOTAL_INLINE_IMAGE_BYTES = 2 * 1024 * 1024;

/**
 * One validated turn attachment. `data` is canonical base64 of the decoded
 * bytes and `mediaType` is the magic-byte-sniffed mime, never the caller's
 * claim — the wire entry only carries a hint.
 */
export interface TurnImage {
  readonly data: string;
  readonly mediaType: string;
}

/**
 * Validate the optional `images` array on `turn.submit` at the RPC boundary.
 *
 * Every entry must be `{ media_type, data }` where `data` decodes as strict
 * base64 and the decoded bytes sniff as a real png/jpeg/gif/webp payload.
 * Oversize or malformed entries reject the whole submit with a typed
 * ValidationError — attachments are never silently truncated or dropped,
 * because a corrupted image reaching the provider would surface as an
 * unrelated model or API failure.
 */
export function validateTurnImages(raw: unknown): readonly TurnImage[] {
  if (raw === undefined || raw === null) {
    return [];
  }
  if (!Array.isArray(raw)) {
    throw new ValidationError("images", "must be an array of { media_type, data } entries");
  }
  if (raw.length > MAX_TURN_IMAGES) {
    throw new ValidationError("images", `must contain at most ${MAX_TURN_IMAGES} entries`, raw.length);
  }
  const images: TurnImage[] = [];
  let totalBytes = 0;
  for (const [index, entry] of raw.entries()) {
    const validated = validateTurnImage(entry, index);
    images.push(validated.image);
    totalBytes += validated.byteLength;
    if (totalBytes > MAX_TURN_IMAGES_TOTAL_BYTES) {
      throw new ValidationError(
        "images",
        `exceed the ${MAX_TURN_IMAGES_TOTAL_BYTES}-byte combined turn limit`,
      );
    }
  }
  return images;
}

/** Build provider-ready data-URL image parts for validated attachments. */
export function imageUrlContentParts(images: readonly TurnImage[]): ImageUrlContentPart[] {
  return images.map((image) => ({
    type: "image_url" as const,
    image_url: { url: `data:${image.mediaType};base64,${image.data}` },
  }));
}

function validateTurnImage(entry: unknown, index: number): { readonly byteLength: number; readonly image: TurnImage } {
  const field = `images[${index}]`;
  if (typeof entry !== "object" || entry === null || Array.isArray(entry)) {
    throw new ValidationError(field, "must be an object with media_type and data");
  }
  const record = entry as Record<string, unknown>;
  const data = record.data;
  if (typeof data !== "string" || !data.trim()) {
    throw new ValidationError(`${field}.data`, "must be a non-empty base64 string");
  }
  const normalized = data.replace(/\s/g, "");
  if (!/^[A-Za-z0-9+/]*={0,2}$/.test(normalized) || normalized.length % 4 !== 0) {
    throw new ValidationError(`${field}.data`, "must contain valid base64 image data");
  }
  const bytes = new Uint8Array(Buffer.from(normalized, "base64"));
  if (!bytes.byteLength) {
    throw new ValidationError(`${field}.data`, "must decode to non-empty image bytes");
  }
  if (bytes.byteLength > MAX_TURN_IMAGE_BYTES) {
    throw new ValidationError(
      `${field}.data`,
      `exceeds the ${MAX_TURN_IMAGE_BYTES}-byte per-image limit`,
      bytes.byteLength,
    );
  }
  const sniffed = sniffImageMediaType(bytes);
  if (!sniffed) {
    throw new ValidationError(
      `${field}.data`,
      "must decode to a png, jpeg, gif, or webp payload (magic-byte check failed)",
    );
  }
  const declared = typeof record.media_type === "string" ? record.media_type.trim().toLowerCase() : "";
  if (declared && declared !== sniffed) {
    throw new ValidationError(
      `${field}.media_type`,
      `declares ${declared} but the payload sniffs as ${sniffed}`,
      declared,
    );
  }
  return { byteLength: bytes.byteLength, image: { data: Buffer.from(bytes).toString("base64"), mediaType: sniffed } };
}
