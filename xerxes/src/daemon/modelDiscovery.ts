// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { lookup } from "node:dns/promises";
import { isIP } from "node:net";

import type { ProviderProfile } from "../bridge/profiles.js";
import type { FetchImplementation } from "../llms/client.js";
import {
  PROVIDERS,
  isProviderName,
  providerDefaultHeaders,
  type ProviderName,
} from "../llms/providerRegistry.js";
import { REDACTED, redactString } from "../security/redact.js";

export const MODEL_DISCOVERY_MAX_RESPONSE_BYTES = 1_048_576;
export const MODEL_DISCOVERY_TIMEOUT_MS = 8_000;

export interface ModelDiscoveryInput {
  readonly allowPrivateEndpoint?: boolean;
  readonly apiKey?: string;
  readonly baseUrl: string;
  readonly environment?: Readonly<Record<string, string | undefined>>;
  readonly provider: string;
  /** False for legacy caller-owned endpoint probes, which must never inherit daemon credentials. */
  readonly resolveProviderCredential?: boolean;
  readonly signal?: AbortSignal;
}

export interface ModelDiscoveryOptions {
  readonly fetchImplementation?: FetchImplementation;
  /** Trusted resolver injection for deterministic tests or host-owned transports. */
  readonly resolveHostname?: (hostname: string) => Promise<readonly string[]>;
  readonly maxResponseBytes?: number;
  readonly timeoutMs?: number;
}

export interface DiscoveredModel {
  readonly contextLimit?: number;
  readonly id: string;
}

const platformFetch = globalThis.fetch;

/** Resolve a stored profile credential without ever exposing it through the daemon protocol. */
export function profileDiscoveryApiKey(
  profile: ProviderProfile,
  environment: Readonly<Record<string, string | undefined>> = process.env,
): string {
  return discoveryApiKey(
    profile.provider,
    profile.api_key,
    environment,
    profile.base_url,
  );
}

/** Resolve an explicit credential, provider environment variable, or provider-local default. */
export function discoveryApiKey(
  provider: string,
  explicitApiKey: string | undefined,
  environment: Readonly<Record<string, string | undefined>> = process.env,
  baseUrl?: string,
): string {
  const explicit = explicitApiKey?.trim();
  if (explicit) {
    return explicit;
  }
  const name = canonicalProvider(provider);
  if (!isProviderName(name)) {
    return "";
  }
  const config = PROVIDERS[name];
  if (baseUrl !== undefined && !providerOwnsEndpoint(config.baseUrl, baseUrl)) {
    return "";
  }
  const fromEnvironment = config.apiKeyEnv
    ? environment[config.apiKeyEnv]?.trim()
    : "";
  return fromEnvironment || config.defaultApiKey || "";
}

/**
 * Discover the live model catalogue for one complete connection tuple.
 *
 * Callers must source `baseUrl` and `apiKey` atomically. This function never
 * consults a profile store, so an untrusted endpoint cannot inherit another
 * profile's credential by accident.
 */
export async function discoverModelIds(
  input: ModelDiscoveryInput,
  options: ModelDiscoveryOptions = {},
): Promise<string[]> {
  return (await discoverModelCatalog(input, options)).map((model) => model.id);
}

/** Discover model ids together with provider-reported context metadata. */
export async function discoverModelCatalog(
  input: ModelDiscoveryInput,
  options: ModelDiscoveryOptions = {},
): Promise<DiscoveredModel[]> {
  const validationError = validateModelDiscoveryUrl(
    input.baseUrl,
    input.allowPrivateEndpoint === true,
  );
  if (validationError) {
    throw new Error(validationError);
  }

  const fetchImplementation = options.fetchImplementation ?? globalThis.fetch;
  const provider = canonicalProvider(input.provider);
  const apiKey =
    input.resolveProviderCredential === false
      ? (input.apiKey?.trim() ?? "")
      : discoveryApiKey(
          provider,
          input.apiKey,
          input.environment,
          input.baseUrl,
        );
  const endpoint = modelDiscoveryEndpoint(input.baseUrl, provider);
  const timeoutMs = positiveInteger(
    options.timeoutMs,
    MODEL_DISCOVERY_TIMEOUT_MS,
  );
  const maxResponseBytes = positiveInteger(
    options.maxResponseBytes,
    MODEL_DISCOVERY_MAX_RESPONSE_BYTES,
  );
  const controller = new AbortController();
  const abortFromCaller = () => controller.abort(input.signal?.reason);
  input.signal?.addEventListener("abort", abortFromCaller, { once: true });
  const timer = setTimeout(
    () =>
      controller.abort(
        new Error(`model discovery timed out after ${timeoutMs}ms`),
      ),
    timeoutMs,
  );

  try {
    if (input.signal?.aborted) {
      controller.abort(input.signal.reason);
    }
    if (input.allowPrivateEndpoint !== true) {
      const resolveHostname =
        options.resolveHostname ??
        (fetchImplementation === platformFetch
          ? resolveModelDiscoveryHostname
          : undefined);
      if (resolveHostname) {
        await abortable(
          validateResolvedModelDiscoveryHost(input.baseUrl, resolveHostname),
          controller.signal,
        );
      }
    }
    const response = await fetchImplementation(endpoint, {
      headers: modelDiscoveryHeaders(provider, apiKey),
      redirect: "error",
      signal: controller.signal,
    });
    if (!response.ok) {
      throw new Error(`model endpoint returned HTTP ${response.status}`);
    }
    const body = await boundedResponseText(response, maxResponseBytes);
    let parsed: unknown;
    try {
      parsed = JSON.parse(body);
    } catch {
      throw new Error("model endpoint returned invalid JSON");
    }
    return modelCatalogFromResponse(parsed);
  } catch (error) {
    throw new Error(
      sanitizeModelDiscoveryError(error, { apiKey, baseUrl: input.baseUrl }),
    );
  } finally {
    clearTimeout(timer);
    input.signal?.removeEventListener("abort", abortFromCaller);
  }
}

/** Validate either a public legacy probe or an explicitly trusted stored profile endpoint. */
export function validateModelDiscoveryUrl(
  baseUrl: string,
  allowPrivateEndpoint = false,
): string | undefined {
  let parsed: URL;
  try {
    parsed = new URL(baseUrl);
  } catch {
    return "Invalid model discovery URL";
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    return "Model discovery URL scheme must be http or https";
  }
  if (!parsed.hostname) {
    return "Model discovery URL must have a host";
  }
  if (parsed.username || parsed.password) {
    return "Model discovery URL must not contain credentials";
  }
  if (allowPrivateEndpoint) {
    return undefined;
  }

  const hostname = normalizedHostname(parsed.hostname);
  if (
    hostname === "localhost" ||
    hostname.endsWith(".localhost") ||
    hostname.endsWith(".local") ||
    hostname === "metadata.google.internal"
  ) {
    return "Private or local model discovery URLs are not allowed";
  }
  if (isPrivateAddress(hostname)) {
    return "Private or local model discovery URLs are not allowed";
  }
  return undefined;
}

/** Build a provider-aware model-list endpoint without double-appending `/models`. */
export function modelDiscoveryEndpoint(
  baseUrl: string,
  provider: string,
): string {
  const url = new URL(baseUrl);
  const path = url.pathname.replace(/\/+$/u, "");
  if (!path.toLowerCase().endsWith("/models")) {
    const suffix =
      canonicalProvider(provider) === "anthropic" &&
      !/\/v\d+(?:beta\d*)?$/iu.test(path)
        ? "/v1/models"
        : "/models";
    url.pathname = `${path}${suffix}` || suffix;
  }
  url.hash = "";
  return url.toString();
}

/** Normalize OpenAI/Anthropic-compatible model-list payloads and remove duplicates. */
export function modelsFromResponse(value: unknown): string[] {
  return modelCatalogFromResponse(value).map((model) => model.id);
}

/** Normalize model-list payloads while preserving trustworthy positive context lengths. */
export function modelCatalogFromResponse(value: unknown): DiscoveredModel[] {
  if (!isRecord(value)) {
    return [];
  }
  const candidates = Array.isArray(value.data)
    ? value.data
    : Array.isArray(value.models)
      ? value.models
      : [];
  const models = new Map<string, DiscoveredModel>();
  for (const candidate of candidates) {
    const id =
      typeof candidate === "string"
        ? candidate.trim()
        : isRecord(candidate)
          ? firstString(candidate.id, candidate.name, candidate.model)
          : "";
    if (!id) {
      continue;
    }
    const contextLimit = isRecord(candidate)
      ? firstPositiveInteger(
          candidate.context_length,
          candidate.contextLength,
          candidate.context_window,
          candidate.max_context_length,
          candidate.max_model_len,
        )
      : undefined;
    const existing = models.get(id);
    const resolvedContextLimit = contextLimit ?? existing?.contextLimit;
    models.set(id, {
      id,
      ...(resolvedContextLimit === undefined
        ? {}
        : { contextLimit: resolvedContextLimit }),
    });
  }
  return [...models.values()];
}

export function sanitizeModelDiscoveryError(
  error: unknown,
  sensitive: { readonly apiKey?: string; readonly baseUrl?: string } = {},
): string {
  let message = error instanceof Error ? error.message : String(error);
  const apiKey = sensitive.apiKey?.trim();
  if (apiKey) {
    message = message.replaceAll(apiKey, REDACTED);
  }
  const baseUrl = sensitive.baseUrl?.trim();
  if (baseUrl) {
    try {
      const parsed = new URL(baseUrl);
      for (const value of parsed.searchParams.values()) {
        if (value) {
          message = message.replaceAll(value, REDACTED);
        }
      }
    } catch {
      // URL validation owns malformed endpoint errors; there is nothing else to redact here.
    }
  }
  return redactString(message);
}

function modelDiscoveryHeaders(
  provider: string,
  apiKey: string,
): Record<string, string> {
  const headers: Record<string, string> = { Accept: "application/json" };
  const name = canonicalProvider(provider);
  if (name === "anthropic") {
    headers["anthropic-version"] = "2023-06-01";
    if (apiKey) {
      headers["x-api-key"] = apiKey;
    }
    return headers;
  }
  if (isProviderName(name)) {
    Object.assign(headers, providerDefaultHeaders(name as ProviderName));
  }
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }
  return headers;
}

async function boundedResponseText(
  response: Response,
  maxBytes: number,
): Promise<string> {
  const contentLength = Number.parseInt(
    response.headers.get("content-length") ?? "",
    10,
  );
  if (Number.isFinite(contentLength) && contentLength > maxBytes) {
    throw new Error(`model endpoint response exceeded ${maxBytes} bytes`);
  }
  if (!response.body) {
    return "";
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let bytes = 0;
  let text = "";
  try {
    while (true) {
      const chunk = await reader.read();
      if (chunk.done) {
        break;
      }
      bytes += chunk.value.byteLength;
      if (bytes > maxBytes) {
        await reader.cancel();
        throw new Error(`model endpoint response exceeded ${maxBytes} bytes`);
      }
      text += decoder.decode(chunk.value, { stream: true });
    }
    return text + decoder.decode();
  } finally {
    reader.releaseLock();
  }
}

function canonicalProvider(provider: string): string {
  return provider.trim().toLowerCase().replaceAll("_", "-");
}

async function abortable<T>(
  promise: Promise<T>,
  signal: AbortSignal,
): Promise<T> {
  if (signal.aborted) {
    throw signal.reason instanceof Error
      ? signal.reason
      : new Error("Model discovery was aborted");
  }
  return new Promise<T>((resolve, reject) => {
    const abort = () =>
      reject(
        signal.reason instanceof Error
          ? signal.reason
          : new Error("Model discovery was aborted"),
      );
    signal.addEventListener("abort", abort, { once: true });
    promise.then(resolve, reject).finally(() => {
      signal.removeEventListener("abort", abort);
    });
  });
}

async function resolveModelDiscoveryHostname(
  hostname: string,
): Promise<readonly string[]> {
  const addresses = await lookup(hostname, { all: true, verbatim: true });
  return addresses.map((address) => address.address);
}

async function validateResolvedModelDiscoveryHost(
  baseUrl: string,
  resolveHostname: (hostname: string) => Promise<readonly string[]>,
): Promise<void> {
  const hostname = normalizedHostname(new URL(baseUrl).hostname);
  if (isIP(hostname)) {
    return;
  }
  let addresses: readonly string[];
  try {
    addresses = await resolveHostname(hostname);
  } catch {
    throw new Error("Model discovery host could not be resolved");
  }
  if (!addresses.length) {
    throw new Error("Model discovery host could not be resolved");
  }
  if (
    addresses.some((address) => isPrivateAddress(normalizedHostname(address)))
  ) {
    throw new Error("Private or local model discovery URLs are not allowed");
  }
}

function providerOwnsEndpoint(
  providerBaseUrl: string | undefined,
  candidateBaseUrl: string,
): boolean {
  if (!providerBaseUrl) {
    return false;
  }
  try {
    const provider = new URL(providerBaseUrl);
    const candidate = new URL(candidateBaseUrl);
    if (provider.origin !== candidate.origin) {
      return false;
    }
    const providerPath = provider.pathname.replace(/\/+$/u, "") || "/";
    const candidatePath = candidate.pathname.replace(/\/+$/u, "") || "/";
    return (
      providerPath === "/" ||
      candidatePath === providerPath ||
      candidatePath.startsWith(`${providerPath}/`)
    );
  } catch {
    return false;
  }
}

function firstString(...values: unknown[]): string {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function firstPositiveInteger(...values: unknown[]): number | undefined {
  for (const value of values) {
    if (typeof value === "number" && Number.isSafeInteger(value) && value > 0) {
      return value;
    }
  }
  return undefined;
}

function isPrivateAddress(hostname: string): boolean {
  if (!isIP(hostname)) {
    return false;
  }
  if (hostname === "::" || hostname === "::1") {
    return true;
  }
  if (hostname.includes(":")) {
    const normalized = hostname.toLowerCase();
    const mappedIpv4 = ipv4MappedAddress(normalized);
    if (mappedIpv4) {
      return isPrivateAddress(mappedIpv4);
    }
    return (
      normalized.startsWith("fc") ||
      normalized.startsWith("fd") ||
      normalized.startsWith("fe8") ||
      normalized.startsWith("fe9") ||
      normalized.startsWith("fea") ||
      normalized.startsWith("feb")
    );
  }
  const parts = hostname.split(".").map((part) => Number(part));
  const first = parts[0] ?? -1;
  const second = parts[1] ?? -1;
  return (
    first === 0 ||
    first === 10 ||
    first === 127 ||
    first >= 224 ||
    (first === 100 && second >= 64 && second <= 127) ||
    (first === 169 && second === 254) ||
    (first === 172 && second >= 16 && second <= 31) ||
    (first === 192 && second === 168)
  );
}

function ipv4MappedAddress(address: string): string | undefined {
  const dotted = /^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/iu.exec(address)?.[1];
  if (dotted && isIP(dotted) === 4) {
    return dotted;
  }
  const hexadecimal = /^::ffff:([\da-f]{1,4}):([\da-f]{1,4})$/iu.exec(address);
  if (!hexadecimal) {
    return undefined;
  }
  const high = Number.parseInt(hexadecimal[1] ?? "", 16);
  const low = Number.parseInt(hexadecimal[2] ?? "", 16);
  if (!Number.isFinite(high) || !Number.isFinite(low)) {
    return undefined;
  }
  return `${high >>> 8}.${high & 0xff}.${low >>> 8}.${low & 0xff}`;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function normalizedHostname(hostname: string): string {
  return hostname.toLowerCase().replace(/^\[/u, "").replace(/\]$/u, "");
}

function positiveInteger(value: number | undefined, fallback: number): number {
  return value !== undefined && Number.isFinite(value) && value > 0
    ? Math.trunc(value)
    : fallback;
}
