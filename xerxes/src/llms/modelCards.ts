// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * Model cards: every setting Xerxes tunes for a model, in one record. Prism's
 * Bonsai models each have a card; any other Bonsai id gets the family card,
 * and every other model gets {@link DEFAULT_CARD}, which changes nothing.
 *
 * This is the only file that matches Bonsai model names. Code that needs a
 * Bonsai-specific value asks {@link modelCard} for it.
 */
import type { SamplingValues } from './samplingDefaults.js'

const GIB = 1024 ** 3

/**
 * Sampling a model's maker publishes, for thinking and non-thinking requests.
 * Open-weight servers (MLX, llama.cpp, LM Studio) do not read the model's
 * `generation_config.json`, so a field a request leaves out is the engine's
 * own default, not the model's: a card with sampling gets it on every request.
 */
export interface PublishedSampling {
  /** Where the values come from, for status text. */
  readonly source: string
  readonly thinking: SamplingValues
  readonly instruct: SamplingValues
}

/** One file of a Hugging Face repo, pinned to a commit and checked by its SHA-256. */
export interface PinnedFile {
  /** Path in the repo; also the file name on disk. */
  readonly path: string
  readonly sha256: string
  readonly bytes: number
}

/** Where a build is published: a Hugging Face repo at one commit. */
export interface HuggingFaceRepo {
  readonly repo: string
  /** The commit the files are pinned to, so a later upload cannot change what is downloaded. */
  readonly revision: string
}

/** The download URL of a pinned file. */
export function huggingFaceUrl(source: HuggingFaceRepo, path: string): string {
  return `https://huggingface.co/${source.repo}/resolve/${source.revision}/${path}`
}

/** A llama-server (GGUF) build Xerxes can download and run. */
export interface LlamaServerBuild {
  /** File name under the local models directory; the repo path when the build is published. */
  readonly fileName: string
  readonly minMemoryBytes: number
  /** Launch window: what fits in this build's memory tier next to the weights, not the model's limit. */
  readonly ctxSize: number
  /** Where `xerxes local setup` downloads it from, when it is published. */
  readonly published?: HuggingFaceRepo & { readonly sha256: string; readonly bytes: number }
  /**
   * The oldest stock llama.cpp that runs it: a build number, or the release
   * version for builds that report one. Older builds may load the file and
   * write gibberish, so setup refuses them.
   */
  readonly minLlamaCpp?: { readonly build: number; readonly version: string }
}

/** An MLX build that the stock `mlx_lm.server` runs, downloaded file by file. */
export interface MlxBuild extends HuggingFaceRepo {
  /** Folder under the local models directory. */
  readonly dirName: string
  /** Every file the server reads; nothing else in the repo is downloaded. */
  readonly files: readonly PinnedFile[]
  readonly minMemoryBytes: number
  /** The oldest mlx-lm that loads this model type. */
  readonly minMlxLm: string
}

/**
 * How long one reply may reason before it acts (see `runtime/replyBudget.ts`):
 * a reply still reasoning past either limit, with no tool call and no answer
 * text yet, is stopped and the model is asked again with a note.
 */
export interface ReplyBudget {
  /** Reasoning tokens, estimated at 4 characters each. */
  readonly maxReasoningTokens: number
  /** Seconds from the reply's first streamed token, so reading the prompt does not count. */
  readonly maxSeconds: number
}

/** An LM Studio build (`lms ls` key). */
export interface LmStudioBuild {
  /** LM Studio model key; also the API identifier Xerxes loads it under. */
  readonly modelKey: string
  readonly minMemoryBytes: number
}

export interface ModelCard {
  /** Stable id; for a model with a local build, the id `xerxes local setup --model` takes. */
  readonly id: string
  readonly label: string
  /** True for Prism's Bonsai models, the ones this harness is tuned for. */
  readonly bonsai: boolean
  /** Model ids this card answers to, local or hosted. */
  readonly match?: RegExp
  /** Sent on every request unless a setting or the request says otherwise. */
  readonly sampling?: PublishedSampling
  /** Output ceiling when nothing else sets one (the maker's own agent setup). */
  readonly maxOutputTokens?: number
  /** The model's own context window. */
  readonly nativeContext?: number
  /** Prompt profile when none is configured; unset means the route's window decides. */
  readonly promptProfile?: 'lean' | 'full'
  /** The cascade is on by default for this model when no setting chooses. */
  readonly escalationByDefault?: boolean
  /** The per-reply budget when `runtime.reply_budget` is unset; none means no budget. */
  readonly replyBudget?: ReplyBudget
  readonly llamaServer?: LlamaServerBuild
  readonly mlx?: MlxBuild
  readonly lmstudio?: LmStudioBuild
}

/** Bonsai 2's non-thinking sampling, the same for every size. */
const BONSAI_2_INSTRUCT: SamplingValues = Object.freeze({
  temperature: 0.7,
  topP: 0.8,
  topK: 20,
  minP: 0,
  presencePenalty: 1.5,
  repetitionPenalty: 1,
})

/** Prism's small-model guide: the 27B's thinking values with presence penalty 1.0. */
const BONSAI_2_SMALL_SAMPLING: PublishedSampling = Object.freeze({
  source: "Prism's small-model guide",
  thinking: Object.freeze({ temperature: 1, topP: 0.95, topK: 20, minP: 0, presencePenalty: 1, repetitionPenalty: 1 }),
  instruct: BONSAI_2_INSTRUCT,
})

/**
 * What every Bonsai card shares: a small model picks better among a few tools,
 * a failed check is cheap to retry, and one reply that reasons for many
 * minutes without acting should not use up the whole task. The reply budget is
 * a quarter of the 27B's 32K output ceiling, and ten minutes so a slow machine
 * does not wait out the token limit.
 */
const BONSAI_DEFAULTS = {
  bonsai: true,
  promptProfile: 'lean',
  escalationByDefault: true,
  replyBudget: Object.freeze({ maxReasoningTokens: 8_192, maxSeconds: 600 }),
} as const

/**
 * Most specific first: the first card whose `match` accepts an id wins, and
 * the last Bonsai card catches any other Bonsai id.
 */
export const MODEL_CARDS: readonly ModelCard[] = Object.freeze([
  Object.freeze({
    ...BONSAI_DEFAULTS,
    id: 'ternary-bonsai-2-27b',
    label: 'Ternary local-model 2 27B',
    // Served as ternary-bonsai-2-27b(-mlx), prism-ml/Ternary-Bonsai-2-27B-mlx-2bit,
    // Bonsai-2-27B-Ternary-JANG and bonsai-2-27b on Ollama.
    match: /bonsai[-_. ]?2[-_. ]?27b/i,
    sampling: Object.freeze({
      source: "Prism's model card",
      thinking: Object.freeze({ temperature: 1, topP: 0.95, topK: 20, minP: 0, presencePenalty: 0, repetitionPenalty: 1 }),
      instruct: BONSAI_2_INSTRUCT,
    }),
    // The ceiling Prism's own agent demo runs with, and the hosted completion limit.
    maxOutputTokens: 32_768,
    nativeContext: 262_144,
    // Stock llama.cpp runs the Q2_0 packing from build b11555 (release v0.7.0);
    // Prism's smaller PQ2_0 and PTQ1_0 packings still need its fork. 20 GiB is a
    // 24 GB machine after the OS reserve. 16K is the window when free memory
    // cannot be measured; raise it with local.ctx_size.
    llamaServer: Object.freeze({
      fileName: 'Ternary-Bonsai-2-27B-Q2_0.gguf',
      minMemoryBytes: 20 * GIB,
      ctxSize: 16_384,
      published: Object.freeze({
        repo: 'prism-ml/Ternary-Bonsai-2-27B-gguf',
        revision: '839bd57023c1558945afea70a74fa83c054f35f9',
        sha256: '4f99aed01b8a877e153f9aa6569a4440fe17c59701cc0953e36d7f460549e70e',
        bytes: 7_626_008_928,
      }),
      minLlamaCpp: Object.freeze({ build: 11_555, version: '0.7.0' }),
    }),
    // The 2-bit MLX pack. mlx-lm 0.32.2 is the first release with its model type
    // (prism_hadamard_qwen35); the files are what mlx_lm.server reads.
    mlx: Object.freeze({
      repo: 'prism-ml/Ternary-Bonsai-2-27B-mlx-2bit',
      revision: '24b5937a4a104a44f790b6ccbd0c79b6fabd4299',
      dirName: 'Ternary-Bonsai-2-27B-mlx-2bit',
      minMemoryBytes: 16 * GIB,
      minMlxLm: '0.32.2',
      files: Object.freeze([
        { path: 'config.json', sha256: '238de7c512cc56a733421e3fd011d88f8260739e3d00e32c5d65b7943cc9f837', bytes: 58_145 },
        { path: 'model.safetensors', sha256: '130de5925082c168b7866b2e91b52e44abbafc99017e3ca352b77b5b55a269ed', bytes: 8_595_477_990 },
        { path: 'hadamard.json', sha256: '7132a3ec364f0bdac1f08f905f24f0ad2f14245060f592637a0396826d3b5fe6', bytes: 297_903 },
        { path: 'tokenizer.json', sha256: '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3', bytes: 12_809_320 },
        { path: 'tokenizer_config.json', sha256: 'b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27', bytes: 17_928 },
        { path: 'chat_template.jinja', sha256: 'c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041', bytes: 8_952 },
        { path: 'generation_config.json', sha256: 'e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e', bytes: 202 },
        { path: 'preprocessor_config.json', sha256: '27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516', bytes: 390 },
        { path: 'video_preprocessor_config.json', sha256: '7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13', bytes: 385 },
      ]),
    }),
    // The 2-bit MLX pack (8.6 GB on disk), measured on a 24 GB M4 Pro on 2026-11-05
    // with a 42,752-token window: a 16 GiB budget (20 after the OS reserve) is enough.
    lmstudio: Object.freeze({ modelKey: 'ternary-bonsai-2-27b-mlx', minMemoryBytes: 16 * GIB }),
  }),
  Object.freeze({
    ...BONSAI_DEFAULTS,
    // `bonsai-9b` is the local catalog's id for it, kept so existing configs still resolve.
    id: 'bonsai-9b',
    label: 'local-model 2 9B',
    match: /bonsai[-_. ]?2[-_. ]?9b|(?:^|[/:])bonsai-9b(?![0-9a-z])/i,
    sampling: BONSAI_2_SMALL_SAMPLING,
    llamaServer: Object.freeze({ fileName: 'bonsai-9b.gguf', minMemoryBytes: 12 * GIB, ctxSize: 32_768 }),
  }),
  Object.freeze({
    ...BONSAI_DEFAULTS,
    id: 'bonsai-2-4b',
    label: 'local-model 2 4B',
    match: /bonsai[-_. ]?2[-_. ]?4b/i,
    sampling: BONSAI_2_SMALL_SAMPLING,
  }),
  Object.freeze({
    ...BONSAI_DEFAULTS,
    id: 'bonsai-2-2b',
    label: 'local-model 2 2B',
    match: /bonsai[-_. ]?2[-_. ]?2b/i,
    sampling: BONSAI_2_SMALL_SAMPLING,
  }),
  // Any other Bonsai id (bonsai-8b, a fine-tune, a future size): the lean prompt
  // and the cascade, but no sampling, which Prism publishes per model.
  Object.freeze({ ...BONSAI_DEFAULTS, id: 'bonsai', label: 'local-model (no published card)', match: /bonsai/i }),
])

/** Every other model: no card values, so Xerxes's general defaults apply. */
export const DEFAULT_CARD: ModelCard = Object.freeze({ id: 'default', label: 'No model card', bonsai: false })

/** The card for a model id; {@link DEFAULT_CARD} when none matches or no model is set. */
export function modelCard(model: string | undefined): ModelCard {
  if (!model) return DEFAULT_CARD
  return MODEL_CARDS.find(card => card.match?.test(model)) ?? DEFAULT_CARD
}

/** Cards with a build `xerxes local setup` can run. */
export function localModelCards(): readonly ModelCard[] {
  return MODEL_CARDS.filter(card => card.llamaServer !== undefined)
}
