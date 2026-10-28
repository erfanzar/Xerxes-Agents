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
  /**
   * transformers.js `dtype` that selects the listed ONNX files. Never `int8`: ONNX Runtime Web
   * has no ConvInteger kernel, so int8-quantized conv layers fail to create a session in the browser.
   */
  readonly dtype: 'fp32' | 'q8'
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
    dtype: 'fp32',
    license: 'MIT',
    files: [
      { path: 'onnx/model.onnx', size: 2243022, sha256: 'a4a068cd6cf1ea8355b84327595838ca748ec29a25bc91fc82e6c299ccdc5808' },
    ],
  },
  // Moonshine Streaming small (community ONNX export, MIT). Measured in Chrome/WASM on 13 real-speech clips:
  // 5.5% WER vs 6.8% for Moonshine base; loads with transformers.js 3.8.1.
  stt_base: {
    repo: 'Workmind/moonshine-streaming-small-ONNX',
    revision: 'bfd8beba2893ce694f67ce7c7424843ba3ae91eb',
    dtype: 'q8',
    license: 'MIT',
    files: [
      { path: 'config.json', size: 1745, sha256: '849ca79b5af5603b1e7b8eeebf67265f931c70117434446fca6f35a26fc03df1' },
      { path: 'generation_config.json', size: 163, sha256: 'fd54ad15ad0a14f68db3c58a8d8e5e8f3791ef95e1ea92ee46cd71b4a5d7528c' },
      { path: 'preprocessor_config.json', size: 284, sha256: '58d5920c0d2ada22a7254d26727dede50aad3d5692cbe755c564a639d76af70a' },
      { path: 'processor_config.json', size: 340, sha256: '41db69af4b4f77d88e3d9d403d3bda2d027547b8880cda6507f2ac23dc09f1c1' },
      { path: 'special_tokens_map.json', size: 96, sha256: '7755e8a96f77dea5458262fb120decdd6a654be977f3540bc781a0a3443f5974' },
      { path: 'tokenizer.json', size: 3761754, sha256: '7b913404bdd039af4756783218af4440bc07fb7d6d8258d677e34f95b3ec416f' },
      { path: 'tokenizer_config.json', size: 301, sha256: 'd84ef83190b3821f41de5cc1ee258925aff99d705d1019708c037d15829d85dd' },
      { path: 'onnx/encoder_model_quantized.onnx', size: 74923158, sha256: '69c786908794eb9e2e50d7137c4e755abdde7b4bdebffd68fe0c292eee5ffc55' },
      { path: 'onnx/decoder_model_merged_quantized.onnx', size: 90830447, sha256: '3bd2e7a7e94c608b1988fa035be1f1b1f2acdbe059f7fe5cc70d85201c2835f5' },
    ],
  },
  stt_tiny: {
    repo: 'onnx-community/moonshine-tiny-ONNX',
    revision: 'a6da1241cd305dcd64eab1edbd615f2bb9aabb95',
    dtype: 'q8',
    license: 'MIT',
    files: [
      { path: 'config.json', size: 921, sha256: '558e1e02069137c796ace1e50c48d8fe451f04a295929138e6bea885517f0edb' },
      ...MOONSHINE_SHARED,
      { path: 'onnx/encoder_model_quantized.onnx', size: 7937661, sha256: 'c6fc4b7bc5af75c0591fd157a1f3829b533d18e9769a888fd95a62e470dd4f4a' },
      { path: 'onnx/decoder_model_merged_quantized.onnx', size: 20243286, sha256: 'eed87831c3a6103534aae7d47a5d485025c659a1323901513961c39fe8a1a367' },
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
