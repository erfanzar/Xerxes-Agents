// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

/**
 * The on-phone voice models for XERXES call mode, pinned to exact revisions
 * with the size and sha256 of every file. Pure and browser-safe.
 *
 * The Hub downloads these files once from Hugging Face, verifies each against
 * this manifest, and serves them same-origin under `/voice-models/` in the
 * Hugging Face layout (`<repo>/resolve/<revision>/<path>`). The phone never
 * contacts a third party. A file not listed here is never served.
 */

export interface VoiceModelFile {
  readonly path: string
  readonly size: number
  readonly sha256: string
}

export interface VoiceModel {
  readonly repo: string
  /** Full commit sha on Hugging Face. */
  readonly revision: string
  /** transformers.js `dtype` that selects the listed ONNX files. */
  readonly dtype: string
  readonly license: string
  readonly files: readonly VoiceModelFile[]
}

export type VoiceModelId = 'vad' | 'stt_base' | 'stt_tiny' | 'tts'

/** `standard`: Moonshine base + Kokoro. `light`: Moonshine tiny, and the phone's built-in voice for speech output. */
export type VoiceModelTier = 'standard' | 'light'

const MOONSHINE_SHARED = [
  { path: 'generation_config.json', size: 147, sha256: 'f9b3f711b57be7def2e50a8942f64f36ee0a55fad5b84ff93a687b6c5bcc1d44' },
  { path: 'preprocessor_config.json', size: 128, sha256: 'fa43a7017ef85cd1d0fba0d9aae77c8adb16990ae6f11115631f41ec5d8aa679' },
  { path: 'special_tokens_map.json', size: 3, sha256: 'ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356' },
  { path: 'tokenizer.json', size: 3761754, sha256: '7b913404bdd039af4756783218af4440bc07fb7d6d8258d677e34f95b3ec416f' },
  { path: 'tokenizer_config.json', size: 135735, sha256: 'edaee394565d428ea98a663ae7209cdcfeefc5585c42d7a570ff7c986df2cd15' },
] as const

export const VOICE_MODELS: Readonly<Record<VoiceModelId, VoiceModel>> = Object.freeze({
  // Silero VAD v5. The repo has no config.json: load it with `config: { model_type: 'custom' }`.
  vad: {
    repo: 'onnx-community/silero-vad',
    revision: 'e71cae966052b992a7eca6b17738916ce0eca4ec',
    dtype: 'int8',
    license: 'MIT',
    files: [
      { path: 'onnx/model_int8.onnx', size: 639383, sha256: '90b026c95f054d59d7bf79387b0ed93c8950f35a4d8b741cd78d4bb23a7d2776' },
    ],
  },
  stt_base: {
    repo: 'onnx-community/moonshine-base-ONNX',
    revision: 'b1e9b6aae3c3c7298f10c3798393fdf38e8fbbad',
    dtype: 'int8',
    license: 'MIT',
    files: [
      { path: 'config.json', size: 922, sha256: 'fab7241d1e9fc6c2370c4c6dfb5da79bb54d67ed9ab6b507ac51d29d2abe01d1' },
      ...MOONSHINE_SHARED,
      { path: 'onnx/encoder_model_int8.onnx', size: 20488801, sha256: '4edb38db96b52ddee5a3b25d2211bb6394f9a9a2d95f67bce0e8c861da018a4d' },
      { path: 'onnx/decoder_model_merged_int8.onnx', size: 42427261, sha256: '45febab0347ffbe1326459e092ae20a2cfda10ca33ab018b3c299247284de61b' },
    ],
  },
  stt_tiny: {
    repo: 'onnx-community/moonshine-tiny-ONNX',
    revision: 'a6da1241cd305dcd64eab1edbd615f2bb9aabb95',
    dtype: 'int8',
    license: 'MIT',
    files: [
      { path: 'config.json', size: 921, sha256: '558e1e02069137c796ace1e50c48d8fe451f04a295929138e6bea885517f0edb' },
      ...MOONSHINE_SHARED,
      { path: 'onnx/encoder_model_int8.onnx', size: 7919835, sha256: '2981b85cfd6a4647ec37fe139e268cbc2c8a24102bf1fbb70dedeb7964995be8' },
      { path: 'onnx/decoder_model_merged_int8.onnx', size: 20189512, sha256: 'bd126e9e95c62b2236116f676f80877ddde488599631014149f92196cd8ae3a7' },
    ],
  },
  // Kokoro-82M; transformers.js dtype `q8` selects onnx/model_quantized.onnx.
  tts: {
    repo: 'onnx-community/Kokoro-82M-v1.0-ONNX',
    revision: '1939ad2a8e416c0acfeecc08a694d14ef25f2231',
    dtype: 'q8',
    license: 'Apache-2.0',
    files: [
      { path: 'config.json', size: 44, sha256: 'df34b4f930b23447cd4dc410fabfb42eb3f24e803e6c3f97d618fb359380a36f' },
      { path: 'tokenizer.json', size: 3497, sha256: '77a02c8e164413299b4b4c403b14f8e0e1c1b727db4d46a09d6327b861060a34' },
      { path: 'tokenizer_config.json', size: 113, sha256: 'be1cb066d6ef6b074b3f15e6a6dd21ac88ff3cdaedf325f0aaed686c70f75d20' },
      { path: 'onnx/model_quantized.onnx', size: 92361116, sha256: 'fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478' },
      { path: 'voices/af_heart.bin', size: 522240, sha256: 'd583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b' },
    ],
  },
} satisfies Record<VoiceModelId, VoiceModel>)

/** The only Kokoro voice shipped. */
export const VOICE_CALL_TTS_VOICE = 'af_heart'

export const VOICE_MODEL_TIERS: Readonly<Record<VoiceModelTier, readonly VoiceModelId[]>> = Object.freeze({
  standard: ['vad', 'stt_base', 'tts'],
  light: ['vad', 'stt_tiny'],
})

/** Same-origin URL prefix the Hub serves models under. */
export const VOICE_MODELS_PATH_PREFIX = '/voice-models/'

/** `/voice-models/<repo>/resolve/<revision>/<path>` for one pinned file. */
export function voiceModelFileUrlPath(model: VoiceModel, path: string): string {
  return `${VOICE_MODELS_PATH_PREFIX}${model.repo}/resolve/${model.revision}/${path}`
}

export function tierDownloadBytes(tier: VoiceModelTier): number {
  return VOICE_MODEL_TIERS[tier].reduce((sum, id) => sum + VOICE_MODELS[id].files.reduce((s, f) => s + f.size, 0), 0)
}

export interface ResolvedVoiceModelFile {
  readonly id: VoiceModelId
  readonly model: VoiceModel
  readonly file: VoiceModelFile
}

/**
 * Look up a requested `<repo>/resolve/<revision>/<path>` (the part after the
 * prefix, already URL-decoded). `main` is accepted as an alias for the pinned
 * revision, because some loaders hard-code it; the pinned bytes are served
 * either way. Anything else, including any path not listed exactly, is undefined.
 */
export function resolveVoiceModelFile(relative: string): ResolvedVoiceModelFile | undefined {
  if (relative.length > 300 || /[\0\\]|(^|\/)\.\.?(\/|$)/u.test(relative)) return undefined
  const match = /^([A-Za-z0-9._-]+\/[A-Za-z0-9._-]+)\/resolve\/([0-9a-f]{40}|main)\/(.+)$/u.exec(relative)
  if (!match) return undefined
  const [, repo, revision, path] = match
  for (const id of Object.keys(VOICE_MODELS) as VoiceModelId[]) {
    const model = VOICE_MODELS[id]
    if (model.repo !== repo || (revision !== 'main' && revision !== model.revision)) continue
    const file = model.files.find(candidate => candidate.path === path)
    if (file) return { id, model, file }
  }
  return undefined
}
