// Copyright 2026 The Xerxes-Agents Author @erfanzar (Erfan Zare Chavoshi).
// Licensed under the Apache License, Version 2.0.

import { modelCard, type PublishedSampling } from './modelCards.js'

/** Provider-neutral temperature used when neither runtime nor profile overrides it. */
export const DEFAULT_TEMPERATURE = 0.6

/** Provider-neutral top-k used when neither runtime nor profile overrides it. */
export const DEFAULT_TOP_K = 64

/** One complete sampling setting, as a model's maker publishes it. */
export interface SamplingValues {
  readonly temperature: number
  readonly topP: number
  readonly topK: number
  readonly minP: number
  readonly presencePenalty: number
  readonly repetitionPenalty: number
}

/** Whether the model is asked to think before answering; makers tune sampling for each. */
export type SamplingMode = 'thinking' | 'instruct'

/** The maker's published sampling for this model id, from its model card. */
export function publishedSamplingProfile(model: string): PublishedSampling | undefined {
  return modelCard(model).sampling
}

/** The published values for one mode, if the model has a published profile. */
export function publishedSampling(model: string, mode: SamplingMode): SamplingValues | undefined {
  return publishedSamplingProfile(model)?.[mode]
}

/** `temperature 1, top_p 0.95, top_k 20, …` for status text. */
export function formatSamplingValues(values: SamplingValues): string {
  return `temperature ${values.temperature}, top_p ${values.topP}, top_k ${values.topK}, min_p ${values.minP}, `
    + `presence_penalty ${values.presencePenalty}, repetition_penalty ${values.repetitionPenalty}`
}

/**
 * The sampling Xerxes fills in when nothing was configured. A model with a
 * published profile gets none here: its profile is applied per request, where
 * the thinking or non-thinking values can be chosen, so an unset temperature
 * must stay unset until then.
 */
export function nativeSamplingDefaults(model: string | undefined): { readonly temperature?: number; readonly top_k?: number } {
  if (model !== undefined && publishedSamplingProfile(model)) return {}
  return { temperature: DEFAULT_TEMPERATURE, top_k: DEFAULT_TOP_K }
}
