// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { ValidationError } from '../core/errors.js'
import { ToolRegistry } from '../executors/toolRegistry.js'
import type { JsonObject, ToolDefinition } from '../types/toolCalls.js'
import {
  OpenAiCompatibleImageGenerationPort,
  serializableImageGenerationResult,
  type ImageGenerationPort,
} from './imageGeneration.js'
import { decodeMediaBase64, type HttpMediaClientOptions } from './mediaHttp.js'
import {
  OpenAiCompatibleTranscriptionPort,
  serializableTranscription,
  type TranscriptionPort,
} from './transcription.js'
import {
  normalizeAudioFormat,
  OpenAiCompatibleTextToSpeechPort,
  serializableSynthesizedAudio,
  type TextToSpeechPort,
} from './tts.js'
import { optionalInteger, optionalString, requireRange, requiredString } from './inputs.js'
import {
  AnthropicVisionPort,
  OpenAiCompatibleVisionPort,
  serializableVisionAnalysis,
  visionImageFromBase64,
  visionImageFromUrl,
  type VisionPort,
} from './vision.js'

const DEFAULT_IMAGE_SIZE = '1024x1024'
const DEFAULT_IMAGE_MEDIA_TYPE = 'image/png'
const DEFAULT_VISION_QUESTION = 'Describe this image in detail.'
const DEFAULT_AUDIO_MEDIA_TYPE = 'audio/wav'
const DEFAULT_AUDIO_FILENAME = 'audio.wav'

export const IMAGE_GENERATE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'image_generate',
    description: 'Generate one or more images through a configured image-generation provider.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        prompt: { type: 'string', description: 'Non-empty image description.' },
        size: { type: 'string', default: DEFAULT_IMAGE_SIZE, description: 'Provider-supported image dimensions.' },
        n: { type: 'integer', minimum: 1, maximum: 10, default: 1 },
        model: { type: 'string', description: 'Optional image-model override.' },
      },
      required: ['prompt'],
    },
  },
}

export const VISION_ANALYZE_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'vision_analyze',
    description: 'Analyze an HTTP(S) image URL or a base64 image through a configured vision provider.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        image_url: { type: 'string', description: 'Absolute HTTP(S) image URL. Mutually exclusive with image_b64.' },
        image_b64: { type: 'string', description: 'Base64 image payload. Mutually exclusive with image_url.' },
        image_media_type: { type: 'string', default: DEFAULT_IMAGE_MEDIA_TYPE },
        question: { type: 'string', default: DEFAULT_VISION_QUESTION },
        model: { type: 'string', description: 'Optional vision-model override.' },
        max_tokens: { type: 'integer', minimum: 1, maximum: 16384 },
      },
    },
  },
}

export const TEXT_TO_SPEECH_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'text_to_speech',
    description: 'Synthesize speech through a configured TTS provider and return base64 audio.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        text: { type: 'string', description: 'Non-empty text to speak.' },
        voice: { type: 'string', description: 'Provider voice identifier.' },
        audio_format: { type: 'string', enum: ['aac', 'flac', 'mp3', 'opus', 'pcm', 'wav'], default: 'mp3' },
        model: { type: 'string', description: 'Optional TTS-model override.' },
      },
      required: ['text'],
    },
  },
}

export const TRANSCRIBE_AUDIO_DEFINITION: ToolDefinition = {
  type: 'function',
  function: {
    name: 'transcribe_audio',
    description: 'Transcribe a base64 audio payload through a configured transcription provider.',
    parameters: {
      type: 'object',
      additionalProperties: false,
      properties: {
        audio_b64: { type: 'string', description: 'Non-empty base64 audio payload.' },
        media_type: { type: 'string', default: DEFAULT_AUDIO_MEDIA_TYPE },
        filename: { type: 'string', default: DEFAULT_AUDIO_FILENAME },
        model: { type: 'string', description: 'Optional transcription-model override.' },
        language: { type: 'string', description: 'Optional ISO language hint.' },
        prompt: { type: 'string', description: 'Optional provider transcription hint.' },
        temperature: { type: 'number', minimum: 0, maximum: 1 },
      },
      required: ['audio_b64'],
    },
  },
}

export const MEDIA_TOOL_DEFINITIONS: readonly ToolDefinition[] = [
  IMAGE_GENERATE_DEFINITION,
  VISION_ANALYZE_DEFINITION,
  TEXT_TO_SPEECH_DEFINITION,
  TRANSCRIBE_AUDIO_DEFINITION,
]

/** Ports are intentionally supplied by the application; this module never pulls credentials from process state. */
export interface MediaToolPorts {
  readonly imageGeneration?: ImageGenerationPort
  readonly textToSpeech?: TextToSpeechPort
  readonly transcription?: TranscriptionPort
  readonly vision?: VisionPort
}

export interface OpenAiCompatibleMediaToolsOptions extends HttpMediaClientOptions {
  readonly imageModel?: string
  readonly transcriptionModel?: string
  readonly ttsModel?: string
  readonly ttsVoice?: string
  readonly visionMaxTokens?: number
  readonly visionModel?: string
}

/** Create the OpenAI-compatible media family from one explicit HTTP configuration. */
export function createOpenAiCompatibleMediaTools(options: OpenAiCompatibleMediaToolsOptions): MediaToolPorts {
  return {
    imageGeneration: new OpenAiCompatibleImageGenerationPort({
      ...options,
      ...(options.imageModel === undefined ? {} : { defaultModel: options.imageModel }),
    }),
    textToSpeech: new OpenAiCompatibleTextToSpeechPort({
      ...options,
      ...(options.ttsModel === undefined ? {} : { defaultModel: options.ttsModel }),
      ...(options.ttsVoice === undefined ? {} : { defaultVoice: options.ttsVoice }),
    }),
    transcription: new OpenAiCompatibleTranscriptionPort({
      ...options,
      ...(options.transcriptionModel === undefined ? {} : { defaultModel: options.transcriptionModel }),
    }),
    vision: new OpenAiCompatibleVisionPort({
      ...options,
      ...(options.visionMaxTokens === undefined ? {} : { defaultMaxTokens: options.visionMaxTokens }),
      ...(options.visionModel === undefined ? {} : { defaultModel: options.visionModel }),
    }),
  }
}

/** Create a direct Anthropic vision port when that provider is configured separately. */
export function createAnthropicVisionTool(options: ConstructorParameters<typeof AnthropicVisionPort>[0]): VisionPort {
  return new AnthropicVisionPort(options)
}

/** Register only the media tools whose real ports have been configured. */
export function registerMediaTools(registry: ToolRegistry, ports: MediaToolPorts): void {
  const imageGeneration = ports.imageGeneration
  if (imageGeneration) {
    registry.register(IMAGE_GENERATE_DEFINITION, inputs => imageGenerateTool(inputs, imageGeneration))
  }
  const vision = ports.vision
  if (vision) {
    registry.register(VISION_ANALYZE_DEFINITION, inputs => visionAnalyzeTool(inputs, vision))
  }
  const textToSpeech = ports.textToSpeech
  if (textToSpeech) {
    registry.register(TEXT_TO_SPEECH_DEFINITION, inputs => textToSpeechTool(inputs, textToSpeech))
  }
  const transcription = ports.transcription
  if (transcription) {
    registry.register(TRANSCRIBE_AUDIO_DEFINITION, inputs => transcribeAudioTool(inputs, transcription))
  }
}

export async function imageGenerateTool(inputs: JsonObject, port: ImageGenerationPort): Promise<JsonObject> {
  const prompt = requiredString(inputs, 'prompt')
  const size = optionalString(inputs, 'size') ?? DEFAULT_IMAGE_SIZE
  const count = requireRange(optionalInteger(inputs, 'n', 1), 'n', 1, 10)
  const model = optionalString(inputs, 'model')
  return serializableImageGenerationResult(await port.generate({
    prompt,
    size,
    count,
    ...(model === undefined ? {} : { model }),
  }))
}

export async function visionAnalyzeTool(inputs: JsonObject, port: VisionPort): Promise<JsonObject> {
  const imageUrl = optionalString(inputs, 'image_url')
  const imageB64 = optionalString(inputs, 'image_b64')
  const hasUrl = Boolean(imageUrl?.trim())
  const hasB64 = Boolean(imageB64?.trim())
  if (hasUrl === hasB64) {
    throw new ValidationError('image', 'requires exactly one of image_url or image_b64')
  }
  const image = hasUrl && imageUrl
    ? visionImageFromUrl(imageUrl)
    : visionImageFromBase64(imageB64 ?? '', optionalString(inputs, 'image_media_type') ?? DEFAULT_IMAGE_MEDIA_TYPE)
  const question = optionalString(inputs, 'question') ?? DEFAULT_VISION_QUESTION
  const model = optionalString(inputs, 'model')
  const maxTokens = inputs.max_tokens === undefined
    ? undefined
    : requireRange(optionalInteger(inputs, 'max_tokens', 1), 'max_tokens', 1, 16_384)
  return serializableVisionAnalysis(await port.analyze({
    image,
    prompt: question,
    ...(model === undefined ? {} : { model }),
    ...(maxTokens === undefined ? {} : { maxTokens }),
  }))
}

export async function textToSpeechTool(inputs: JsonObject, port: TextToSpeechPort): Promise<JsonObject> {
  const text = requiredString(inputs, 'text')
  const voice = optionalString(inputs, 'voice')
  const model = optionalString(inputs, 'model')
  const rawFormat = optionalString(inputs, 'audio_format') ?? 'mp3'
  const format = normalizeAudioFormat(rawFormat)
  return serializableSynthesizedAudio(await port.synthesize({
    text,
    format,
    ...(voice === undefined ? {} : { voice }),
    ...(model === undefined ? {} : { model }),
  }))
}

export async function transcribeAudioTool(inputs: JsonObject, port: TranscriptionPort): Promise<JsonObject> {
  const audioB64 = requiredString(inputs, 'audio_b64')
  const model = optionalString(inputs, 'model')
  const language = optionalString(inputs, 'language')
  const prompt = optionalString(inputs, 'prompt')
  const mediaType = optionalString(inputs, 'media_type') ?? DEFAULT_AUDIO_MEDIA_TYPE
  const filename = optionalString(inputs, 'filename') ?? DEFAULT_AUDIO_FILENAME
  const temperature = optionalFiniteNumber(inputs, 'temperature')
  if (temperature !== undefined && (temperature < 0 || temperature > 1)) {
    throw new ValidationError('temperature', 'must be a number from 0 to 1', temperature)
  }
  return serializableTranscription(await port.transcribe({
    audio: { bytes: decodeMediaBase64(audioB64, 'audio_b64'), filename, mediaType },
    ...(model === undefined ? {} : { model }),
    ...(language === undefined ? {} : { language }),
    ...(prompt === undefined ? {} : { prompt }),
    ...(temperature === undefined ? {} : { temperature }),
  }))
}

function optionalFiniteNumber(inputs: JsonObject, field: string): number | undefined {
  const value = inputs[field]
  if (value === undefined) return undefined
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new ValidationError(field, 'must be a finite number', value)
  }
  return value
}
